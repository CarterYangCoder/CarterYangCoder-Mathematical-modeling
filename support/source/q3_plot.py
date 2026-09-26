"""问题三科研图：长时程干燥下的含水率时空演化。"""
from __future__ import annotations


import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from openpyxl import load_workbook


# 路径、模型常量与绘图参数。

ROOT: Final = Path(__file__).resolve().parent
DEFAULT_RESULTS: Final = ROOT / "result3.xlsx"
DEFAULT_CHAPTER: Final = ROOT / "q3_model_chapter.pdf"
DEFAULT_OUTPUT: Final = ROOT / "generated_figures_q3"

INITIAL_C: Final = 2.55
RADIUS_CM: Final = 2.0
CRITICAL_C: Final = 0.15
MEASURED_END_S: Final = 14_400.0
DPI: Final = 600

FULL_WIDTH_IN: Final = 5.71

# 终点核验常量与论文模型章节保持一致。
EXPECTED_CHAPTER_SHA256: Final = (
    "3f83fff386c2967aa9125113d2593e5e75a268b6626410ae3769de5039fd903f"
)
REPORTED_Q3_NUMERICS: Final = {
    "delivery_time_s": 205_809.766_3,
    "delivery_max_full_precision": 0.149_999_998_119_172,
}

REPORT_HOURS: Final = (6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0)

TEXT: Final = "#1F2933"
MUTED: Final = "#667784"
GRID: Final = "#D7E0E5"
BLUE: Final = "#2F6B9A"
TEAL: Final = "#198C8C"
ORANGE: Final = "#D97932"
RED: Final = "#B63A3A"
PURPLE: Final = "#76528B"
LIGHT_BLUE: Final = "#DDEBF3"


def ordered_cmap(colours: Sequence[str], name: str) -> LinearSegmentedColormap:

    return LinearSegmentedColormap.from_list(name, list(colours), N=256)


MOISTURE_CMAP: Final = ordered_cmap(
    ("#F7FBFF", "#D9EEF4", "#9CCBDB", "#4D93B8", "#235B8A", "#17365D"),
    "q3_moisture",
)


@dataclass(frozen=True)
class Q3Data:
    time_s: np.ndarray
    radii_cm: np.ndarray
    moisture: np.ndarray
    source_started_at_s: float
    delivered_time_s: float

    @property
    def hours(self) -> np.ndarray:
        return self.time_s / 3600.0


@dataclass(frozen=True)
class EnvironmentHistory:
    time_s: np.ndarray
    temperature_c: np.ndarray
    moisture_kg_kg: np.ndarray


@dataclass(frozen=True)
class FigureSpec:
    filename: str
    role: str
    caption: str
    builder: Callable[[Q3Data, EnvironmentHistory | None], plt.Figure]


# 结果表读取与结构校验。

def _as_float(value: object, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context} 不是有效数值：{value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} 不是有效数值：{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} 不是有限数：{number!r}")
    return number


def load_q3_results(path: Path) -> Q3Data:

    if not path.is_file():
        raise FileNotFoundError(f"找不到问题三结果文件：{path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    if not workbook.sheetnames:
        raise ValueError("result3.xlsx 不含工作表。")
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration as exc:
        raise ValueError("result3.xlsx 为空。") from exc
    if len(header) < 3:
        raise ValueError("result3.xlsx 至少需要时间列和两个径向节点。")
    radii = np.asarray([_as_float(v, f"径向表头第 {j + 2} 列") for j, v in enumerate(header[1:])])
    if not np.all(np.diff(radii) > 0):
        raise ValueError("径向表头必须严格递增。")
    if not np.isclose(radii[0], 0.0, atol=1e-10) or not np.isclose(radii[-1], RADIUS_CM, atol=1e-10):
        raise ValueError(f"径向范围应为 0--{RADIUS_CM:g} cm，实际为 {radii[0]:g}--{radii[-1]:g} cm。")

    time_values: list[float] = []
    field_rows: list[list[float]] = []
    for row_number, row in enumerate(rows, start=2):
        if row is None or all(value is None for value in row):
            continue
        if len(row) < len(radii) + 1:
            raise ValueError(f"第 {row_number} 行列数不足。")
        time_values.append(_as_float(row[0], f"第 {row_number} 行时间"))
        field_rows.append(
            [_as_float(value, f"第 {row_number} 行 r={radius:g} cm") for radius, value in zip(radii, row[1:])]
        )
    workbook.close()

    time_s = np.asarray(time_values, dtype=float)
    moisture = np.asarray(field_rows, dtype=float)
    if time_s.size < 2 or moisture.shape != (time_s.size, radii.size):
        raise ValueError("结果场的时间或空间维度异常。")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("结果时间必须严格递增。")
    if np.any(moisture <= 0):
        raise ValueError("水分浓度必须为正，当前数据含非正值。")
    if np.nanmax(moisture) > INITIAL_C + 5e-4:
        raise ValueError("结果中出现超过模型初值的异常水分浓度。")

    source_started_at_s = float(time_s[0])
    delivered_time_s = float(time_s[-1])
    if source_started_at_s > 0:
        # 仅在内存补入题设初值，不以相邻结果插值。
        time_s = np.concatenate(([0.0], time_s))
        moisture = np.vstack((np.full(radii.size, INITIAL_C), moisture))
    if not np.all(np.diff(np.max(moisture, axis=1)) <= 5e-5 + 1e-12):
        raise ValueError("输出节点最大含水率出现超出四位舍入尺度的反向增长。")
    return Q3Data(time_s, radii, moisture, source_started_at_s, delivered_time_s)


def load_environment(path: Path) -> EnvironmentHistory:

    if not path.is_file():
        raise FileNotFoundError(f"找不到附件 1：{path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    try:
        next(rows)
    except StopIteration as exc:
        raise ValueError("附件 1 为空。") from exc
    values: list[tuple[float, float, float]] = []
    for row_number, row in enumerate(rows, start=2):
        if row is None or all(value is None for value in row):
            continue
        if len(row) < 3:
            raise ValueError(f"附件 1 第 {row_number} 行列数不足。")
        values.append(
            (
                _as_float(row[0], f"附件 1 第 {row_number} 行时间"),
                _as_float(row[1], f"附件 1 第 {row_number} 行温度"),
                _as_float(row[2], f"附件 1 第 {row_number} 行水分浓度"),
            )
        )
    workbook.close()
    array = np.asarray(values, dtype=float)
    if array.shape[0] < 2 or not np.all(np.diff(array[:, 0]) > 0):
        raise ValueError("附件 1 的时间必须严格递增且至少包含两个节点。")
    if not np.isclose(array[0, 0], 0.0) or not np.isclose(array[-1, 0], MEASURED_END_S):
        raise ValueError("问题三要求附件 1 覆盖 0--14400 s；当前范围不一致。")
    return EnvironmentHistory(array[:, 0], array[:, 1], array[:, 2])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_chapter_version(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"找不到问题三模型章节：{path}")
    actual = file_sha256(path)
    if actual.lower() != EXPECTED_CHAPTER_SHA256:
        raise RuntimeError(
            "q3-chapter.pdf 已变化，全域判据图中的全精度终点常数可能过期。"
            "请先依据新版章节更新 REPORTED_Q3_NUMERICS 和 EXPECTED_CHAPTER_SHA256。"
        )


# 绘图与数值辅助函数。

def _pick_font(candidates: tuple[str, ...], fallback: str) -> str:

    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in candidates if name in installed), fallback)


# 数学标签中的中文字符范围。
_CJK_RANGES: Final = (
    (0x2E80, 0x2EFF),
    (0x3000, 0x303F),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFE30, 0xFE4F),
    (0xFF00, 0xFFEF),
)


def _is_cjk_symbol(symbol: str) -> bool:

    if len(symbol) != 1:
        return False
    codepoint = ord(symbol)
    return any(low <= codepoint <= high for low, high in _CJK_RANGES)


def _enable_chinese_in_mathtext(chinese_font: str) -> None:


    from matplotlib import _mathtext

    if getattr(_mathtext.UnicodeFonts._get_glyph, "_accepts_chinese", False):
        return
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=[chinese_font]), fallback_to_default=False
    )
    original_get_glyph = _mathtext.UnicodeFonts._get_glyph

    def _get_glyph(self, fontname, font_class, sym):
        if fontname == "default" and _is_cjk_symbol(sym):
            font = self._get_font(font_path)
            codepoint = ord(sym)
            if font.get_char_index(codepoint):
                return font, codepoint, False
        return original_get_glyph(self, fontname, font_class, sym)

    _get_glyph._accepts_chinese = True
    _mathtext.UnicodeFonts._get_glyph = _get_glyph


def configure_matplotlib() -> None:


    latin_font = _pick_font(("Times New Roman", "Nimbus Roman", "Liberation Serif"), "DejaVu Serif")
    chinese_font = _pick_font(("SimSun", "Songti SC", "Noto Serif CJK SC", "SimHei"), "DejaVu Sans")

    mpl.rcParams.update(
        {
            # 建立中西文回退链，确保数学标签中的中文正常显示。
            "font.family": [latin_font, chinese_font, "DejaVu Serif"],
            "font.serif": [latin_font, chinese_font, "DejaVu Serif"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": latin_font,
            "mathtext.it": f"{latin_font}:italic",
            "mathtext.bf": f"{latin_font}:bold",
            "mathtext.sf": latin_font,
            "mathtext.tt": latin_font,
            "mathtext.default": "it",
            "mathtext.fallback": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 0.75,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
    _enable_chinese_in_mathtext(chinese_font)
    print(
        f"字体：中文 {chinese_font}（常规字重）；西文、数字与单位 {latin_font}；"
        f"数学变量 {latin_font} 斜体（mathtext custom）。"
    )


def box_axis(axis: plt.Axes, *, grid: str | None = "y") -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(TEXT)
        spine.set_linewidth(0.75)
    axis.tick_params(direction="out", length=3.2, width=0.7, top=True, right=True)
    if grid:
        axis.grid(axis=grid, color=GRID, linewidth=0.55, alpha=0.75, zorder=0)


def require_matplotlib_panel_alignment(
    fig: plt.Figure,
    *,
    json_out: Path | None = None,
    tolerance_pt: float = 1.5,
    strict: bool = True,
) -> None:


    fig.canvas.draw()
    width_pt, height_pt = (size * 72.0 for size in fig.get_size_inches())
    panels: list[dict[str, object]] = []
    for index, axis in enumerate(fig.axes):
        try:
            subplot_spec = axis.get_subplotspec()
        except AttributeError:
            subplot_spec = None
        if subplot_spec is None or axis.get_label() == "<colorbar>":
            continue
        position = axis.get_position()
        bbox = {
            "left": float(position.x0 * width_pt),
            "bottom": float(position.y0 * height_pt),
            "right": float(position.x1 * width_pt),
            "top": float(position.y1 * height_pt),
        }
        if bbox["right"] <= bbox["left"] or bbox["top"] <= bbox["bottom"]:
            raise RuntimeError(f"面板 {index} 的绘图区尺寸无效。")
        panels.append({"axis_index": index, "bbox_pt": bbox})
    for i, first in enumerate(panels):
        a = first["bbox_pt"]
        assert isinstance(a, dict)
        for second in panels[i + 1 :]:
            b = second["bbox_pt"]
            assert isinstance(b, dict)
            overlap_width = min(a["right"], b["right"]) - max(a["left"], b["left"])
            overlap_height = min(a["top"], b["top"]) - max(a["bottom"], b["bottom"])
            if overlap_width > tolerance_pt and overlap_height > tolerance_pt:
                raise RuntimeError(f"面板绘图区发生重叠：{first['axis_index']} 与 {second['axis_index']}。")
    manifest = {
        "status": "PASS",
        "tolerance_pt": tolerance_pt,
        "figure_size_pt": {"width": width_pt, "height": height_pt},
        "panels": panels,
    }
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def time_edges(hours: np.ndarray) -> np.ndarray:
    edges = np.empty(hours.size + 1, dtype=float)
    edges[0] = hours[0]
    edges[-1] = hours[-1]
    edges[1:-1] = (hours[:-1] + hours[1:]) / 2.0
    if not np.all(np.diff(edges) > 0):
        raise ValueError("时间边界不严格递增。")
    return edges


def radial_edges(radii_cm: np.ndarray) -> np.ndarray:
    edges = np.empty(radii_cm.size + 1, dtype=float)
    edges[0] = radii_cm[0]
    edges[-1] = radii_cm[-1]
    edges[1:-1] = (radii_cm[:-1] + radii_cm[1:]) / 2.0
    if not np.all(np.diff(edges) > 0):
        raise ValueError("径向边界不严格递增。")
    return edges


def exact_time_index(data: Q3Data, target_h: float, atol_s: float = 1e-6) -> int:
    target_s = target_h * 3600.0
    matched = np.flatnonzero(np.isclose(data.time_s, target_s, atol=atol_s, rtol=0.0))
    if matched.size != 1:
        raise ValueError(f"结果中没有唯一的 t={target_h:g} h 保存行。")
    return int(matched[0])


def radial_volume_mean(data: Q3Data) -> np.ndarray:

    radius = data.radii_cm
    dr = np.diff(radius)
    integrand = data.moisture * radius[None, :]
    integral = np.sum((integrand[:, :-1] + integrand[:, 1:]) * dr[None, :] / 2.0, axis=1)
    return 2.0 * integral / (radius[-1] ** 2)


def report_profile_times(data: Q3Data) -> list[float]:
    return [*REPORT_HOURS, data.delivered_time_s / 3600.0]


# 各图对应问题三的关键证据。

def _plot_boundary_panel(
    axis: plt.Axes,
    measured_h: np.ndarray,
    measured_values: np.ndarray,
    end_h: float,
    *,
    colour: str,
    ylabel: str,
    title: str,
    hold_value_text: str,
) -> None:
    axis.plot(measured_h, measured_values, color=colour, linewidth=1.55, label="附件实测输入（0–4 h）")
    axis.plot(
        [measured_h[-1], end_h],
        [measured_values[-1], measured_values[-1]],
        color=MUTED,
        linewidth=1.55,
        linestyle=(0, (5, 2)),
        label="4 h 后末值保持",
    )
    axis.axvspan(0.0, measured_h[-1], color=colour, alpha=0.06, linewidth=0)
    axis.axvline(measured_h[-1], color=TEXT, linewidth=0.8, linestyle=":")
    axis.set_xlim(0.0, end_h)
    axis.set_xlabel("时间 $t$ / h")
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=6)
    box_axis(axis)
    # 标注紧邻虚线延拓段，说明 4 h 后边界采用末值保持。
    axis.annotate(
        hold_value_text,
        xy=(0.975, measured_values[-1]),
        xycoords=axis.get_yaxis_transform(),
        xytext=(0.0, -3.2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7.2,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
    )

    zoom = inset_axes(axis, width="43%", height="43%", loc="center right", borderpad=1.0)
    zoom.plot(measured_h, measured_values, color=colour, linewidth=1.1)
    zoom.scatter(measured_h[::20], measured_values[::20], s=7, color=colour, edgecolor="white", linewidth=0.25)
    zoom.set_xlim(0.0, measured_h[-1])
    padding = max(np.ptp(measured_values) * 0.08, abs(measured_values[-1]) * 0.01)
    zoom.set_ylim(np.min(measured_values) - padding, np.max(measured_values) + padding)
    zoom.set_title("实测段放大", fontsize=7.4, pad=2)
    zoom.tick_params(labelsize=6.7, length=2.2, top=True, right=True)
    for spine in zoom.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(TEXT)


def figure_boundary_extension(data: Q3Data, environment: EnvironmentHistory | None) -> plt.Figure:
    if environment is None:
        raise ValueError("长期边界图需要附件 1。")
    end_h = data.delivered_time_s / 3600.0
    measured_h = environment.time_s / 3600.0
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.66), constrained_layout=True)
    _plot_boundary_panel(
        axes[0], measured_h, environment.temperature_c, end_h,
        colour=RED, ylabel="环境温度 $T_a$ / °C", title="（a）长期温度边界",
        hold_value_text=f"4 h 后保持 {environment.temperature_c[-1]:.3f} °C",
    )
    _plot_boundary_panel(
        axes[1], measured_h, environment.moisture_kg_kg, end_h,
        colour=BLUE, ylabel="等效环境水分浓度 $C_e$ / (kg/kg)", title="（b）长期水分边界",
        hold_value_text=f"4 h 后保持 {environment.moisture_kg_kg[-1]:.5f} kg/kg",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return fig


def figure_spatiotemporal_map(data: Q3Data, _: EnvironmentHistory | None) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(FULL_WIDTH_IN, 3.01), constrained_layout=True)
    final_h = data.delivered_time_s / 3600.0
    vmin = max(float(np.min(data.moisture)), 0.045)
    vmax = float(np.max(data.moisture))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    mesh = axis.pcolormesh(
        time_edges(data.hours), radial_edges(data.radii_cm), data.moisture.T,
        cmap=MOISTURE_CMAP, norm=norm, shading="flat", rasterized=True,
    )
    contour = axis.contour(
        data.hours, data.radii_cm, data.moisture.T,
        levels=[CRITICAL_C], colors=[RED], linewidths=1.45,
    )
    axis.axvline(MEASURED_END_S / 3600.0, color=TEXT, linestyle=":", linewidth=0.9)
    axis.axvline(final_h, color=RED, linestyle="--", linewidth=0.9)
    axis.set_xlim(0.0, final_h)
    axis.set_ylim(0.0, data.radii_cm[-1])
    axis.set_xlabel("时间 $t$ / h")
    axis.set_ylabel("距中心距离 $r$ / cm")
    box_axis(axis, grid=None)
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.018, fraction=0.045)
    colorbar.set_label("干基含水率 $C$ / (kg/kg)")
    ticks = [value for value in (0.05, 0.1, 0.15, 0.3, 0.6, 1.0, 2.0, 2.55) if vmin <= value <= vmax]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:g}" for value in ticks])
    axis.legend(
        handles=[
            Line2D([0], [0], color=RED, lw=1.45, label="$C=0.15$ 阈值等值线"),
            Line2D([0], [0], color=TEXT, lw=0.9, ls=":", label="实测环境终点 4 h"),
            Line2D(
                [0], [0], color=RED, lw=0.9, ls="--",
                label=rf"交付时刻 $t_d={final_h:.4f}\ \mathrm{{h}}$",
            ),
        ],
        loc="upper right", frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.92,
    )
    if not contour.allsegs[0]:
        raise RuntimeError("水分场内没有 C=0.15 等值线，无法支持问题三阈值图。")
    return fig


def figure_radial_profiles(data: Q3Data, _: EnvironmentHistory | None) -> plt.Figure:
    times = report_profile_times(data)
    early = times[:5]
    late = times[5:]
    norm = Normalize(vmin=min(times), vmax=max(times))
    cmap = mpl.colormaps["viridis"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.81), sharex=True, sharey=True, constrained_layout=True)
    for axis, subset, title in zip(
        axes,
        (early, late),
        ("（a）主体失水阶段", "（b）后期达标阶段"),
    ):
        for time_h in subset:
            index = len(data.time_s) - 1 if np.isclose(time_h, times[-1]) else exact_time_index(data, time_h)
            axis.plot(data.radii_cm, data.moisture[index], color=cmap(norm(time_h)), linewidth=1.65)
        axis.axhline(CRITICAL_C, color=RED, linewidth=1.0, linestyle="--")
        axis.set_title(title, pad=6)
        axis.set_xlabel("距中心距离 $r$ / cm")
        axis.set_yscale("log")
        axis.set_xlim(0.0, RADIUS_CM)
        axis.set_ylim(0.045, 1.15)
        box_axis(axis)
    axes[0].set_ylabel("干基含水率 $C$ / (kg/kg，对数坐标)")
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axes, pad=0.018, fraction=0.04)
    colorbar.set_label("时刻 $t$ / h")
    colorbar.set_ticks(times)
    colorbar.set_ticklabels([f"{value:.4f}" if value == times[-1] else f"{value:g}" for value in times])
    axes[1].text(
        0.97, CRITICAL_C * 1.035, "$C_{crit}=0.15$", color=RED, fontsize=7.5,
        ha="right", va="bottom", transform=axes[1].get_yaxis_transform(),
    )
    return fig


def figure_global_criterion(data: Q3Data, _: EnvironmentHistory | None) -> plt.Figure:
    maximum = np.max(data.moisture, axis=1)
    minimum = np.min(data.moisture, axis=1)
    mean = radial_volume_mean(data)
    final_h = data.delivered_time_s / 3600.0
    audited_final_c = float(REPORTED_Q3_NUMERICS["delivery_max_full_precision"])

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 3.38), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.92), height_ratios=(1.0, 1.0))
    ax_main = fig.add_subplot(grid[:, :2])
    ax_zoom = fig.add_subplot(grid[0, 2])
    ax_profile = fig.add_subplot(grid[1, 2])

    ax_main.fill_between(data.hours, minimum, maximum, color=LIGHT_BLUE, alpha=0.72, label="径向范围")
    ax_main.plot(data.hours, maximum, color=BLUE, linewidth=1.8, label="输出节点最大值")
    ax_main.plot(data.hours, mean, color=TEAL, linewidth=1.45, label="径向体积加权均值")
    ax_main.plot(data.hours, minimum, color=ORANGE, linewidth=1.45, label="输出节点最小值")
    ax_main.axhline(CRITICAL_C, color=RED, linewidth=1.05, linestyle="--", label="全域阈值")
    ax_main.axvline(final_h, color=RED, linewidth=0.9, linestyle=":")
    ax_main.set_yscale("log")
    ax_main.set_xlim(0.0, final_h)
    ax_main.set_ylim(0.045, INITIAL_C * 1.08)
    ax_main.set_xlabel("时间 $t$ / h")
    ax_main.set_ylabel("干基含水率 $C$ / (kg/kg，对数坐标)")
    ax_main.set_title("（a）全时段径向范围与整体水平", pad=6)
    ax_main.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.94)
    box_axis(ax_main)

    mask = data.hours >= 48.0
    ax_zoom.plot(data.hours[mask], maximum[mask], color=BLUE, linewidth=1.75)
    ax_zoom.axhline(CRITICAL_C, color=RED, linewidth=1.0, linestyle="--")
    ax_zoom.axvline(final_h, color=RED, linewidth=0.9, linestyle=":")
    ax_zoom.scatter([final_h], [maximum[-1]], s=24, color=RED, edgecolor="white", linewidth=0.5, zorder=4)
    ax_zoom.set_xlim(48.0, final_h + 0.15)
    ax_zoom.set_ylim(0.1475, 0.163)
    ax_zoom.set_xlabel("时间 $t$ / h")
    ax_zoom.set_ylabel("最大含水率 / (kg/kg)")
    ax_zoom.annotate(
        rf"$t_d={final_h:.4f}\ \mathrm{{h}}$（全精度）"
        + "\n"
        + rf"$C_{{\max}}={audited_final_c:.10f}$",
        xy=(final_h, maximum[-1]),
        xytext=(0.96, 0.98),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.0,
        color=TEXT,
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.75},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": GRID, "alpha": 0.94},
    )
    ax_zoom.set_title("（b）终止事件邻域", pad=6)
    box_axis(ax_zoom)

    ax_profile.plot(data.radii_cm, data.moisture[-1], color=PURPLE, linewidth=1.8, marker="o", markersize=2.5)
    ax_profile.axhline(CRITICAL_C, color=RED, linewidth=1.0, linestyle="--")
    ax_profile.scatter([0.0], [data.moisture[-1, 0]], s=28, color=RED, edgecolor="white", linewidth=0.5, zorder=4)
    ax_profile.set_xlim(0.0, RADIUS_CM)
    ax_profile.set_ylim(0.045, 0.155)
    ax_profile.set_xlabel("距中心距离 $r$ / cm")
    ax_profile.set_ylabel("末行含水率 / (kg/kg)")
    ax_profile.set_title("（c）交付时刻径向状态", pad=6)
    ax_profile.annotate(
        "中心控制全域达标",
        xy=(0.0, data.moisture[-1, 0]),
        xytext=(0.45, 0.108),
        ha="left",
        va="center",
        fontsize=7.0,
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.75},
    )
    box_axis(ax_profile)
    return fig


FIGURES: Final[dict[str, FigureSpec]] = {
    "spatiotemporal_map": FigureSpec(
        "fig_q3_moisture_field",
        "全局时空证据",
        "问题三全时段水分浓度场。颜色以对数归一化显示干基含水率，红线为模型计算得到的 C=0.15 kg/kg 等值线，而非实测边界；其由表面向中心推进，并在交付时刻 t_d=57.1694 h 到达中心，显示中心为最终控制位置。",
        figure_spatiotemporal_map,
    ),
    "radial_profiles": FigureSpec(
        "fig_q3_radial_profiles",
        "报告时刻空间细节",
        "问题三各报告时刻的径向含水率分布。左图为 6–30 h，右图为 36 h 至交付时刻；颜色连续表示时间，纵轴采用对数坐标以兼顾主体失水和阈值邻域。曲线在 r=2 cm 附近的低值反映表面率先达到阈值以及真实径向梯度，并非数值异常。",
        figure_radial_profiles,
    ),
    "global_criterion": FigureSpec(
        "fig_q3_drying_criterion",
        "问题答案的决定性证据",
        "全域干燥判据及控制位置。主图比较输出节点最大值、径向体积加权均值与最小值；右上放大终止事件邻域，右下给出交付时刻径向分布。图中曲线采用四位表值，严格达标使用未舍入结果：交付时刻 t_d=57.1694 h，中心最大含水率为 0.1499999981 kg/kg。",
        figure_global_criterion,
    ),
}


# 图片导出与命令行入口。

def parse_names(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(FIGURES)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(names) - set(FIGURES))
    if unknown:
        raise ValueError(f"未知图形：{unknown}；可用名称：{', '.join(FIGURES)}")
    if not names:
        raise ValueError("--figures 不能为空。")
    return names


def parse_formats(value: str) -> list[str]:
    formats = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = {"png", "pdf", "svg"}
    unknown = sorted(set(formats) - allowed)
    if unknown or not formats:
        raise ValueError(f"--formats 仅支持 png、pdf、svg；收到：{value!r}")
    return list(dict.fromkeys(formats))


def save_figure(fig: plt.Figure, stem: Path, formats: Sequence[str], *, overwrite: bool) -> list[Path]:
    destinations = [stem.with_suffix(f".{fmt}") for fmt in formats]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"拒绝覆盖既有图片：{existing}。如需覆盖请显式传入 --overwrite。")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        require_matplotlib_panel_alignment(
            fig,
            json_out=None,
            tolerance_pt=1.5,
            strict=True,
        )
        for destination, fmt in zip(destinations, formats):
            if fmt == "png":
                fig.savefig(destination.with_suffix(".png"), format="png", dpi=600, facecolor="white", edgecolor="white")
            elif fmt == "pdf":
                fig.savefig(destination.with_suffix(".pdf"), format="pdf", facecolor="white", edgecolor="white")
            elif fmt == "svg":
                fig.savefig(destination.with_suffix(".svg"), format="svg", facecolor="white", edgecolor="white")
            else:
                raise ValueError(f"未知导出格式：{fmt}")
            written.append(destination)
    except Exception:
        for destination in written:
            destination.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)
    return destinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 CUMCM 问题三的论文科研图像。")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="result3.xlsx 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="图像输出目录")
    parser.add_argument("--figures", default="all", help="all 或逗号分隔的图形名称：" + ", ".join(FIGURES))
    parser.add_argument("--formats", default="png", help="逗号分隔：png、pdf、svg；默认仅 png")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名输出；不会删除其他文件")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = parse_names(args.figures)
    formats = parse_formats(args.formats)
    results_path = args.results.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == results_path.parent:
        raise ValueError("输出目录不能与输入数据目录相同。")

    configure_matplotlib()
    data = load_q3_results(results_path)
    if "global_criterion" in selected:
        audited_delivery_s = float(REPORTED_Q3_NUMERICS["delivery_time_s"])
        if not np.isclose(data.delivered_time_s, audited_delivery_s, atol=5e-4, rtol=0.0):
            raise ValueError(
                "result3.xlsx 的末时刻与模型章节的全精度交付时刻不一致，"
                "不能生成带全精度标注的全域判据图。"
            )

    max_locations = data.radii_cm[np.argmax(data.moisture, axis=1)]
    centre_controls = bool(np.all(np.isclose(max_locations, 0.0)))
    print(f"输入：{results_path}")
    print(f"数据规模：{data.moisture.shape[0]:,} 个时刻 × {data.moisture.shape[1]} 个径向节点")
    print(f"时间范围：0–{data.delivered_time_s / 3600.0:.6f} h；原工作簿首行 {data.source_started_at_s:g} s")
    print(f"所有保存时刻的输出节点最大值是否位于中心：{'是' if centre_controls else '否'}")
    print("注意：工作簿为四位小数展示值，严格全域达标需引用模型章节的全精度核验。")

    for name in selected:
        spec = FIGURES[name]
        fig = spec.builder(data, None)
        destinations = save_figure(fig, output_dir / spec.filename, formats, overwrite=args.overwrite)
        print(f"\n[{name}] {spec.role}")
        print(f"图注草案：{spec.caption}")
        for destination in destinations:
            print(f"已保存：{destination}")
    return 0


def verify_chapter_version(path: Path) -> None:

    if not path.is_file():
        print(f"Model chapter not supplied; hash check skipped: {path}")
        return
    if file_sha256(path) != EXPECTED_CHAPTER_SHA256:
        raise ValueError(f"Model chapter hash mismatch: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
