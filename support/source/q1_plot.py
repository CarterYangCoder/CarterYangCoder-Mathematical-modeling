"""问题一科研图：模型示意及温度、含水率径向分布。"""
from __future__ import annotations


import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, Polygon
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "result1.xlsx"
DEFAULT_OUTPUT = ROOT / "generated_figures_q1"

INITIAL_T_C = 28.0
INITIAL_C_KG_KG = 2.55
SELECTED_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

# 按论文单栏宽度设置画布。
FULL_WIDTH_IN = 5.71

# 温度与含水率曲线均按固定半径采用同一颜色和线型。
RADIUS_COLORS = ["#C7352B", "#C99700", "#2378B5", "#2E8B57", "#1A1A1A"]
RADIUS_STYLES = [
    (0, (1.2, 1.2)),
    (0, (4.0, 2.2)),
    (0, (5.0, 1.8, 1.2, 1.8)),
    "--",
    "-",
]
ACCENT = "#356E8E"
TEXT = "#1F2933"
MID_GRAY = "#667784"
LIGHT_GRAY = "#D9E1E6"


# 数学标签中的中文字符范围。
_CJK_RANGES = (
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


def _pick_font(candidates: tuple[str, ...], fallback: str) -> str:

    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in candidates if name in installed), fallback)


def configure_matplotlib() -> None:


    latin_font = _pick_font(("Times New Roman", "Nimbus Roman", "Liberation Serif"), "DejaVu Serif")
    chinese_font = _pick_font(("SimSun", "Songti SC", "Noto Serif CJK SC", "SimHei"), "DejaVu Sans")

    mpl.rcParams.update(
        {
            # 以字体列表建立中西文回退链，避免数学标签中的中文缺字。
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
            "font.size": 7.8,
            "axes.linewidth": 0.75,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    _enable_chinese_in_mathtext(chinese_font)
    print(
        f"字体：中文 {chinese_font}（常规字重）；西文、数字与单位 {latin_font}；"
        f"数学变量 {latin_font} 斜体（mathtext custom）。"
    )


def _read_sheet(workbook: Path, preferred_name: str, fallback_index: int) -> pd.DataFrame:
    sheets = pd.ExcelFile(workbook).sheet_names
    sheet = preferred_name if preferred_name in sheets else sheets[fallback_index]
    frame = pd.read_excel(workbook, sheet_name=sheet)
    if frame.shape[1] < 3:
        raise ValueError(f"工作表“{sheet}”的列数不足，无法构成时间—径向结果场。")
    return frame


def _normalise_field(
    frame: pd.DataFrame,
    initial_value: float,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:

    time_s = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=float)
    radius_cm = pd.to_numeric(pd.Index(frame.columns[1:]), errors="raise").to_numpy(dtype=float)
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    if values.shape != (time_s.size, radius_cm.size):
        raise ValueError(f"{field_name}结果场尺寸与时间或径向坐标不一致。")
    if not (np.all(np.isfinite(time_s)) and np.all(np.isfinite(radius_cm)) and np.all(np.isfinite(values))):
        raise ValueError(f"{field_name}结果场存在缺失值或非有限值，拒绝绘图。")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError(f"{field_name}时间轴必须严格递增。")
    if not np.all(np.diff(radius_cm) > 0):
        raise ValueError(f"{field_name}径向坐标必须严格递增。")
    if not np.isclose(radius_cm[0], 0.0) or not np.isclose(radius_cm[-1], 2.0):
        raise ValueError(f"{field_name}径向坐标应覆盖 0–2 cm，当前为 {radius_cm[0]}–{radius_cm[-1]} cm。")

    initial_added = False
    if np.isclose(time_s[0], 0.0):
        pass
    elif np.isclose(time_s[0], 1.0):
        # 仅在内存补入题设初始状态，不对结果表插值或回写。
        time_s = np.insert(time_s, 0, 0.0)
        values = np.vstack((np.full((1, radius_cm.size), initial_value), values))
        initial_added = True
    else:
        raise ValueError(
            f"{field_name}时间轴起点为 {time_s[0]:g} s；仅支持原始 t=0 或模型输出从 t=1 s 开始的情形。"
        )

    if not np.isclose(time_s[-1], 1800.0):
        raise ValueError(f"{field_name}时间轴终点应为 1800 s，当前为 {time_s[-1]:g} s。")
    return time_s, radius_cm, values, initial_added


def load_results(workbook: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:

    temperature = _read_sheet(workbook, "温度", 0)
    moisture = _read_sheet(workbook, "水分浓度", 1)
    time_t, radius_t, t_field, added_t = _normalise_field(temperature, INITIAL_T_C, "温度")
    time_c, radius_c, c_field, added_c = _normalise_field(moisture, INITIAL_C_KG_KG, "水分浓度")
    if not (np.array_equal(time_t, time_c) and np.array_equal(radius_t, radius_c)):
        raise ValueError("温度场与水分场的时间或径向网格不一致。")
    return time_t, radius_t, t_field, c_field, (added_t or added_c)


def selected_columns(radius_cm: np.ndarray, requested: Iterable[float]) -> np.ndarray:
    indices = []
    for radius in requested:
        matches = np.flatnonzero(np.isclose(radius_cm, radius, atol=1e-10))
        if matches.size != 1:
            raise ValueError(f"未在结果表中找到唯一的 r={radius:g} cm 节点。")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.085,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=TEXT,
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def make_model_schematic(output_dir: Path) -> None:

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 2.90), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.12, 0.88), wspace=0.07)
    ax_physical = fig.add_subplot(grid[0, 0])
    ax_radial = fig.add_subplot(grid[0, 1])
    for ax in (ax_physical, ax_radial):
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)

    heat_color = "#C9553D"
    moisture_color = "#2A7FA7"
    surface_color = "#8AAFC2"

    # 左图：在斜视三维坐标系中展示圆柱药材。
    axis_color = "#566875"
    # 以斜视地面投影增强圆柱体的空间感。
    origin = (0.88, 0.62)
    x_tip, y_tip, z_tip = (2.86, 2.06), (8.90, 0.62), (0.88, 6.25)
    far_corner = (9.62, 2.06)
    ax_physical.add_patch(
        Polygon(
            [origin, y_tip, far_corner, x_tip],
            closed=True,
            facecolor="#F7FAFB",
            edgecolor="none",
            zorder=-2,
        )
    )
    ax_physical.plot(
        [x_tip[0], far_corner[0]],
        [x_tip[1], far_corner[1]],
        color="#BBC7CE",
        linewidth=0.55,
        linestyle=(0, (3.0, 2.2)),
        zorder=-1,
    )
    ax_physical.plot(
        [y_tip[0], far_corner[0]],
        [y_tip[1], far_corner[1]],
        color="#BBC7CE",
        linewidth=0.55,
        linestyle=(0, (3.0, 2.2)),
        zorder=-1,
    )
    for fraction in (0.38, 0.70):
        near = (
            origin[0] + fraction * (y_tip[0] - origin[0]),
            origin[1] + fraction * (y_tip[1] - origin[1]),
        )
        far = (
            x_tip[0] + fraction * (far_corner[0] - x_tip[0]),
            x_tip[1] + fraction * (far_corner[1] - x_tip[1]),
        )
        ax_physical.plot(
            [near[0], far[0]],
            [near[1], far[1]],
            color="#DEE5E9",
            linewidth=0.42,
            zorder=-1,
        )
    for tip in (x_tip, y_tip, z_tip):
        ax_physical.add_patch(
            FancyArrowPatch(
                origin,
                tip,
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=0.80,
                color=axis_color,
                zorder=0,
            )
        )
    ax_physical.text(2.95, 2.15, "x", ha="left", va="bottom", fontsize=7.7, color=axis_color)
    ax_physical.text(9.05, 0.62, "y", ha="left", va="center", fontsize=7.7, color=axis_color)
    ax_physical.text(0.88, 6.37, "z", ha="center", va="bottom", fontsize=7.7, color=axis_color)

    cx, base_y, top_y = 5.22, 1.55, 5.18
    rx, ry = 1.22, 0.43
    left_x, right_x = cx - rx, cx + rx

    ax_physical.add_patch(
        Ellipse((cx, base_y), 2 * rx, 2 * ry, facecolor="#F8FBFC", edgecolor="none", zorder=1)
    )
    ax_physical.add_patch(
        Polygon(
            [(left_x, base_y), (right_x, base_y), (right_x, top_y), (left_x, top_y)],
            closed=True,
            facecolor="#EAF2F6",
            edgecolor="none",
            zorder=2,
        )
    )
    ax_physical.plot([left_x, left_x], [base_y, top_y], color=TEXT, linewidth=1.0, zorder=4)
    ax_physical.plot([right_x, right_x], [base_y, top_y], color=TEXT, linewidth=1.0, zorder=4)

    # 底面后半圆以虚线表示遮挡部分，前半圆保持实线。
    ax_physical.add_patch(
        Arc(
            (cx, base_y),
            2 * rx,
            2 * ry,
            theta1=0,
            theta2=180,
            color=MID_GRAY,
            linewidth=0.72,
            linestyle=(0, (3.0, 2.0)),
            zorder=3,
        )
    )
    ax_physical.add_patch(
        Arc((cx, base_y), 2 * rx, 2 * ry, theta1=180, theta2=360, color=TEXT, linewidth=1.05, zorder=4)
    )

    # 虚线椭圆标示代表性径向截面。
    slice_y = 3.30
    ax_physical.add_patch(
        Ellipse(
            (cx, slice_y),
            2 * rx,
            2 * ry,
            fill=False,
            edgecolor=surface_color,
            linewidth=0.95,
            linestyle=(0, (3.2, 2.0)),
            zorder=3,
        )
    )
    ax_physical.text(left_x - 0.30, slice_y, "代表截面", ha="right", va="center", fontsize=7.0, color=MID_GRAY)

    ax_physical.add_patch(
        Ellipse((cx, top_y), 2 * rx, 2 * ry, facecolor="#DDEAF0", edgecolor=TEXT, linewidth=1.05, zorder=5)
    )

    ax_physical.add_patch(
        FancyArrowPatch((cx, top_y), (right_x - 0.08, top_y), arrowstyle="-|>", mutation_scale=8.0, linewidth=0.78, color=MID_GRAY)
    )
    ax_physical.text(
        cx + 0.10,
        top_y,
        "R = 2 cm",
        ha="center",
        va="center",
        fontsize=6.8,
        color=TEXT,
        bbox=dict(facecolor="#DDEAF0", edgecolor="none", pad=0.9),
    )

    # 高度标注置于独立延长线上，避免与热流和水分箭头重叠。
    dim_x = 8.45
    for extension_y in (base_y, top_y):
        ax_physical.plot(
            [right_x + 0.06, dim_x + 0.10],
            [extension_y, extension_y],
            color=MID_GRAY,
            linewidth=0.55,
            linestyle=(0, (4.0, 2.0)),
        )
    ax_physical.annotate("", xy=(dim_x, base_y), xytext=(dim_x, top_y), arrowprops=dict(arrowstyle="<->", lw=0.78, color=MID_GRAY))
    ax_physical.text(
        dim_x,
        (base_y + top_y) / 2,
        "L = 25 cm",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.4,
        color=TEXT,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.4),
    )

    # 热量由外向内传递，水分由内向外迁移。
    for heat_y in (4.35, 3.75):
        ax_physical.add_patch(
            FancyArrowPatch((2.05, heat_y), (left_x - 0.08, heat_y), arrowstyle="-|>", mutation_scale=10, linewidth=1.15, color=heat_color)
        )
    ax_physical.text(2.95, 4.57, "热流向内", ha="center", va="bottom", fontsize=7.4, color=heat_color)
    moisture_start_x = right_x + 0.30
    for moisture_y in (4.35, 3.75):
        ax_physical.add_patch(
            FancyArrowPatch(
                (moisture_start_x, moisture_y),
                (7.92, moisture_y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.10,
                linestyle="--",
                color=moisture_color,
            )
        )
    ax_physical.text(
        moisture_start_x,
        4.57,
        "水分外流",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=moisture_color,
    )
    ax_physical.text(5.05, 6.80, "（a）圆柱物理模型", ha="center", va="top", fontsize=8.4, color=TEXT)

    # 右图：以截面同心圆表达仅考虑径向传递的理论简化。
    fig.canvas.draw()
    box = ax_radial.get_window_extent()
    y_limits, x_limits = ax_radial.get_ylim(), ax_radial.get_xlim()
    y_span, x_span = y_limits[1] - y_limits[0], x_limits[1] - x_limits[0]
    center = (4.35, 3.58)
    radius = 1.56
    radius_y = radius * (y_span / x_span) * (box.width / box.height)

    def add_section_ring(scale: float, **kwargs: object) -> None:
        ax_radial.add_patch(Ellipse(center, 2.0 * radius * scale, 2.0 * radius_y * scale, **kwargs))

    add_section_ring(1.0, facecolor="#F3F8FA", edgecolor=TEXT, linewidth=1.0)
    for fraction in (0.34, 0.68):
        add_section_ring(fraction, fill=False, edgecolor="#B7C9D2", linewidth=0.65, linestyle=":")
    ax_radial.plot(center[0], center[1], marker="o", markersize=2.8, color=TEXT)
    ax_radial.add_patch(FancyArrowPatch(center, (center[0] + radius, center[1]), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=TEXT))
    ax_radial.text(center[0] + radius / 2, center[1] + 0.18, "r", ha="center", va="bottom", fontsize=8.0, color=TEXT)
    ax_radial.text(center[0], 1.57, "中心对称", ha="center", va="top", fontsize=7.2, color=MID_GRAY)
    ax_radial.text(center[0], 0.94, "仅考虑径向传递", ha="center", va="top", fontsize=7.2, color=MID_GRAY)

    ax_radial.add_patch(FancyArrowPatch((5.46, 4.73), (4.96, 4.23), arrowstyle="-|>", mutation_scale=10, linewidth=1.15, color=heat_color))
    ax_radial.text(6.75, 5.04, "热流向内", ha="left", va="center", fontsize=7.3, color=heat_color)
    ax_radial.add_patch(FancyArrowPatch((5.45, 2.47), (6.32, 1.60), arrowstyle="-|>", mutation_scale=10, linewidth=1.10, linestyle="--", color=moisture_color))
    ax_radial.text(6.68, 1.40, "水分外流", ha="left", va="center", fontsize=7.3, color=moisture_color)
    ax_radial.text(4.85, 6.78, "（b）径向截面理论模型", ha="center", va="top", fontsize=8.4, color=TEXT)

    save_figure(fig, output_dir / "fig_q1_model_diagram")


def make_trend_figure(
    time_s: np.ndarray,
    radius_cm: np.ndarray,
    temperature_c: np.ndarray,
    moisture_kg_kg: np.ndarray,
    output_dir: Path,
) -> None:

    columns = selected_columns(radius_cm, SELECTED_RADII_CM)
    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH_IN, 3.85), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.965, bottom=0.12, top=0.875, hspace=0.23)
    ax_t, ax_c = axes

    line_handles = []
    for index, (radius, column) in enumerate(zip(SELECTED_RADII_CM, columns, strict=True)):
        common = dict(color=RADIUS_COLORS[index], ls=RADIUS_STYLES[index], lw=1.15 + 0.15 * (index == len(columns) - 1))
        line_t, = ax_t.plot(time_s, temperature_c[:, column], label=f"r = {radius:g} cm", **common)
        ax_c.plot(time_s, moisture_kg_kg[:, column], **common)
        line_handles.append(line_t)

    for column, color in ((columns[0], RADIUS_COLORS[0]), (columns[-1], RADIUS_COLORS[-1])):
        ax_t.plot(time_s[-1], temperature_c[-1, column], "o", ms=3.1, color=color, clip_on=False)
        ax_c.plot(time_s[-1], moisture_kg_kg[-1, column], "o", ms=3.1, color=color, clip_on=False)

    for ax in axes:
        ax.set_xlim(0, 1800)
        ax.set_xticks(np.arange(0, 1801, 300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.grid(axis="y", color="#E0E7EB", lw=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.65, labelsize=7.1)
        ax.tick_params(which="minor", direction="out", length=1.8, width=0.5)

    ax_t.set_ylim(27.8, 37.15)
    ax_t.yaxis.set_major_locator(MultipleLocator(2.0))
    ax_t.set_ylabel("温度 $T$ / °C", fontsize=8.0)

    ax_c.set_ylim(1.43, 2.61)
    ax_c.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_c.set_xlabel("时间 $t$ / s", fontsize=8.0)
    ax_c.set_ylabel("干基含水率 $C$ / (kg/kg)", fontsize=8.0)

    _add_panel_label(ax_t, "a")
    _add_panel_label(ax_c, "b")
    fig.legend(
        handles=line_handles,
        labels=[handle.get_label() for handle in line_handles],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.51, 0.992),
        fontsize=7.0,
        handlelength=2.25,
        handletextpad=0.45,
        columnspacing=1.05,
    )
    save_figure(fig, output_dir / "fig_q1_temperature_moisture_vs_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成问题一模型示意图和径向趋势图。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="result1.xlsx 的路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="图片输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    if not args.input.is_file():
        raise FileNotFoundError(f"未找到结果文件：{args.input}")

    time_s, radius_cm, temperature_c, moisture_kg_kg, initial_added = load_results(args.input)
    make_model_schematic(args.output)
    make_trend_figure(time_s, radius_cm, temperature_c, moisture_kg_kg, args.output)

    source_note = "已由模型初值补入 t=0 s" if initial_added else "结果表已包含 t=0 s"
    print(
        "已生成两张图："
        f"{args.output / 'fig_q1_model_diagram.png'}、"
        f"{args.output / 'fig_q1_temperature_moisture_vs_time.png'}"
    )
    print(f"时间轴：{time_s[0]:g}–{time_s[-1]:g} s；{source_note}。")
    print(f"1800 s：T中心={temperature_c[-1, 0]:.7g} °C，T表面={temperature_c[-1, -1]:.7g} °C；"
          f"C中心={moisture_kg_kg[-1, 0]:.7g} kg/kg，C表面={moisture_kg_kg[-1, -1]:.7g} kg/kg。")


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "result1.xlsx"
DEFAULT_OUTPUT = ROOT / "generated_figures_q1"

FULL_WIDTH_IN = 5.71
TIME_POINTS_S = (100, 300, 600, 900, 1200, 1500, 1800)
SERIES_STYLES = (
    ("#0072B2", "-"),
    ("#E69F00", "--"),
    ("#009E73", "-."),
    ("#D55E00", ":"),
    ("#CC79A7", (0, (5, 1.3, 1, 1.3))),
    ("#9A8200", (0, (3.2, 1.2))),
    ("#202020", (0, (1, 1))),
)


def configure_radial_matplotlib() -> None:

    configure_matplotlib()


def draw_profiles(ax: plt.Axes, radius_cm: np.ndarray, profiles: np.ndarray, *, moisture: bool) -> None:

    for time_s, values, (color, linestyle) in zip(TIME_POINTS_S, profiles, SERIES_STYLES, strict=True):
        ax.plot(
            radius_cm,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=1.45,
            solid_capstyle="round",
            label=f"{time_s} s",
        )
    ax.set_xlim(0.0, 2.0)
    ax.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2 if moisture else 2.0))
    ax.margins(y=0.08)
    ax.set_title("（二）含水率径向分布" if moisture else "（一）温度径向分布", fontsize=9.2, pad=7.0)
    ax.set_xlabel("距中心距离 $r$ / cm", fontsize=9.0)
    ax.set_ylabel("干基含水率 $C$ / (kg/kg)" if moisture else "温度 $T$ / °C", fontsize=9.0)
    ax.grid(False)
    ax.tick_params(direction="out", length=3.0, width=0.65, labelsize=8.0, top=True, right=True)
    ax.tick_params(axis="x", which="minor", direction="out", length=1.8, width=0.45)
    ax.legend(
        loc="lower left" if moisture else "upper left",
        ncol=2,
        fontsize=7.3,
        handlelength=1.70 if moisture else 2.25,
        handletextpad=0.80 if moisture else 0.45,
        columnspacing=0.95,
        borderaxespad=0.25,
    )


def create_figure(input_workbook: Path) -> plt.Figure:

    time_s, radius_cm, temperature_c, moisture_kg_kg, _ = load_results(input_workbook)
    rows = []
    for target_s in TIME_POINTS_S:
        matches = np.flatnonzero(np.isclose(time_s, target_s, rtol=0.0, atol=1e-10))
        if matches.size != 1:
            raise ValueError(f"未找到唯一的 t={target_s} s 结果行，拒绝进行插值或替代。")
        rows.append(int(matches[0]))
    selected_rows = np.asarray(rows, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.34), layout="constrained")
    draw_profiles(axes[0], radius_cm, temperature_c[selected_rows, :], moisture=False)
    draw_profiles(axes[1], radius_cm, moisture_kg_kg[selected_rows, :], moisture=True)
    return fig


def export_figure(fig: plt.Figure, output_dir: Path) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig_q1_temperature_moisture_profiles"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成问题一温度与含水率的横向径向分布组合图。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="result1.xlsx 的路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="图片输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    figure = create_figure(args.input)
    export_figure(figure, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all final Question 1 figures.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="result1.xlsx path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="figure output directory")
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing result workbook: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    time_s, radius_cm, temperature_c, moisture_kg_kg, _ = load_results(args.input)
    make_model_schematic(args.output)
    make_trend_figure(time_s, radius_cm, temperature_c, moisture_kg_kg, args.output)
    export_figure(create_figure(args.input), args.output)
    print(f"Question 1 figures written to {args.output}")


if __name__ == "__main__":
    main()
