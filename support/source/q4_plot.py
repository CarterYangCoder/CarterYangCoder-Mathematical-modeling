"""问题四科研图：收缩耦合移动边界下的干燥过程。"""
from __future__ import annotations


import argparse
import csv
import hashlib
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
from matplotlib.patches import Patch
from openpyxl import load_workbook
from PIL import Image, ImageStat


# 路径、模型常量与绘图参数。

ROOT: Final = Path(__file__).resolve().parent
DEFAULT_Q4_RESULTS: Final = ROOT / "result4.xlsx"
DEFAULT_Q3_RESULTS: Final = ROOT / "result3.xlsx"
DEFAULT_OUTPUT: Final = ROOT / "generated_figures_q4"

INITIAL_C: Final = 2.55
INITIAL_RADIUS_CM: Final = 2.0
CRITICAL_C: Final = 0.15
H_MASS_M_PER_S: Final = 8.0e-7
MEASURED_ENV_END_S: Final = 14_400.0
J_DRY: Final = 0.3504072058
BETA_SHRINKAGE: Final = 2.2697229023
EXPECTED_CHAPTER_SHA256: Final = (
    "36ef1daeace9b68d104593fdf8582df3287e4d6770b73fd9f737eb5eebf97b79"
)

REPORT_HOURS: Final = (6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0)
DPI: Final = 600

FULL_WIDTH_IN: Final = 5.71
MAX_ROWS: Final = 100_000
MAX_RADIAL_COLUMNS: Final = 501
MAX_HEATMAP_PIXELS: Final = 2_000_000
MAX_PLOT_POINTS_PER_SERIES: Final = 5_000
MAX_FIGURE_BYTES: Final = 50 * 1024 * 1024
ALLOWED_OUTPUT_SUFFIXES: Final = {".png", ".jpg", ".jpeg"}

TEXT: Final = "#1F2933"
MUTED: Final = "#667784"
GRID: Final = "#D8E1E6"
BLUE: Final = "#2F6B9A"
TEAL: Final = "#168C8C"
ORANGE: Final = "#D9822B"
RED: Final = "#B63A3A"
PURPLE: Final = "#76528B"
GREEN: Final = "#3C8D66"
LIGHT_GREY: Final = "#E9EEF1"

TITLE_SHRINKAGE_CALIBRATION: Final = "药材半径收缩关系的校准与拟合诊断"
TITLE_MOVING_DOMAIN_MAP: Final = "移动边界下物理坐标与材料坐标中的含水率场"
TITLE_COUPLING_SCALES: Final = "药材收缩的几何尺度变化与扩散抑制"
TITLE_FRONT_AND_CRITERION: Final = "收缩表面、干燥前沿与全域停止判据"
TITLE_ENVIRONMENT_SENSITIVITY: Final = "长期环境边界扰动下的终点时间灵敏度"


def ordered_cmap(colours: Sequence[str], name: str) -> LinearSegmentedColormap:

    return LinearSegmentedColormap.from_list(name, list(colours), N=256)


MOISTURE_CMAP: Final = ordered_cmap(
    ("#F7FBFF", "#DCEEF2", "#9CCBD5", "#4D92B2", "#245B87", "#17365D"),
    "q4_moisture",
)
TIME_CMAP: Final = ordered_cmap(
    ("#2F6B9A", "#168C8C", "#D2A72C", "#D97932", "#9E3558"),
    "q4_time",
)


@dataclass(frozen=True)
class Q4Data:
    time_s: np.ndarray
    radii_cm: np.ndarray
    moisture: np.ndarray
    surface_moisture: np.ndarray
    source_started_at_s: float

    @property
    def hours(self) -> np.ndarray:
        return self.time_s / 3600.0

    @property
    def delivered_time_s(self) -> float:
        return float(self.time_s[-1])


@dataclass(frozen=True)
class Q3Data:
    time_s: np.ndarray
    radii_cm: np.ndarray
    moisture: np.ndarray

    @property
    def hours(self) -> np.ndarray:
        return self.time_s / 3600.0


@dataclass(frozen=True)
class DerivedQ4:
    cumulative_loss: np.ndarray
    radius_cm: np.ndarray
    volume_ratio: np.ndarray
    mean_moisture: np.ndarray
    maximum_moisture: np.ndarray
    dry_front_cm: np.ndarray
    physical_grid_cm: np.ndarray
    physical_field: np.ndarray
    material_grid: np.ndarray
    material_field: np.ndarray
    radius_lower_bound_corrections: int
    maximum_radius_correction_cm: float


@dataclass(frozen=True)
class EnvironmentScenario:


    question: str
    case: str
    delta_min: float
    delta_s: float


@dataclass(frozen=True)
class Context:
    q4: Q4Data
    derived: DerivedQ4
    q3: Q3Data
    environment_scenarios: tuple[EnvironmentScenario, ...] | None = None


@dataclass(frozen=True)
class FigureSpec:
    filename: str
    role: str
    caption: str
    builder: Callable[[Context], plt.Figure]


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_chapter_version(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"找不到问题四模型章节：{path}")
    if file_sha256(path).lower() != EXPECTED_CHAPTER_SHA256:
        raise RuntimeError(
            "q4-chapter.pdf 已变化，收缩参数可能过期。请依据新版章节同步更新 "
            "J_DRY、BETA_SHRINKAGE 与 EXPECTED_CHAPTER_SHA256。"
        )


def load_q4_results(path: Path) -> Q4Data:

    if not path.is_file():
        raise FileNotFoundError(f"找不到问题四结果文件：{path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    if not workbook.sheetnames:
        raise ValueError("result4.xlsx 不含工作表。")
    sheet = workbook[workbook.sheetnames[0]]
    if sheet.max_row > MAX_ROWS or sheet.max_column > MAX_RADIAL_COLUMNS:
        raise MemoryError(
            f"结果规模 {sheet.max_row}×{sheet.max_column} 超过脚本资源上限，"
            "请先按绘图分辨率分块聚合。"
        )
    rows = sheet.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration as exc:
        raise ValueError("result4.xlsx 为空。") from exc
    if len(header) < 4:
        raise ValueError("result4.xlsx 至少需要时间、固定位置与真实表面列。")
    radii = np.asarray(
        [_as_float(value, f"固定径向表头第 {index + 2} 列") for index, value in enumerate(header[1:-1])],
        dtype=float,
    )
    if not np.all(np.diff(radii) > 0):
        raise ValueError("固定径向表头必须严格递增。")
    expected = np.arange(0.0, INITIAL_RADIUS_CM + 0.05, 0.1)
    if radii.shape != expected.shape or not np.allclose(radii, expected, atol=1e-10, rtol=0.0):
        raise ValueError("result4.xlsx 的固定径向表头应为 0--2 cm、步长 0.1 cm。")

    time_values: list[float] = []
    field_rows: list[list[float]] = []
    surface_values: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        if row is None or all(value is None for value in row):
            continue
        if len(row) < len(header):
            raise ValueError(f"第 {row_number} 行列数不足。")
        time_values.append(_as_float(row[0], f"第 {row_number} 行时间"))
        fixed: list[float] = []
        seen_blank = False
        for radius, value in zip(radii, row[1:-1]):
            if value is None:
                seen_blank = True
                fixed.append(float("nan"))
            else:
                if seen_blank:
                    raise ValueError(f"第 {row_number} 行在域外空白之后又出现有效固定位置值。")
                fixed.append(_as_float(value, f"第 {row_number} 行 r={radius:g} cm"))
        if not math.isfinite(fixed[0]):
            raise ValueError(f"第 {row_number} 行中心值缺失。")
        field_rows.append(fixed)
        surface_values.append(_as_float(row[-1], f"第 {row_number} 行真实表面"))
    workbook.close()

    time_s = np.asarray(time_values, dtype=float)
    moisture = np.asarray(field_rows, dtype=float)
    surface = np.asarray(surface_values, dtype=float)
    if time_s.size < 2 or moisture.shape != (time_s.size, radii.size):
        raise ValueError("问题四结果的时间或空间维度异常。")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("问题四结果时间必须严格递增且无重复。")
    regular_steps = np.diff(time_s[:-1])
    if regular_steps.size and not np.allclose(regular_steps, 60.0, atol=1e-8, rtol=0.0):
        raise ValueError("除最终非整分钟行外，问题四结果必须每 60 s 保存。")
    finite = np.r_[moisture[np.isfinite(moisture)], surface]
    if np.any(finite <= 0.0) or np.max(finite) > INITIAL_C + 5e-4:
        raise ValueError("问题四含水率超出模型允许的正值范围。")

    source_started_at_s = float(time_s[0])
    if source_started_at_s > 0.0:
        time_s = np.concatenate(([0.0], time_s))
        moisture = np.vstack((np.full(radii.size, INITIAL_C), moisture))
        surface = np.concatenate(([INITIAL_C], surface))
    centre = moisture[:, 0]
    if np.any(np.diff(centre) > 5e-5 + 1e-12):
        raise ValueError("中心含水率出现超出四位舍入尺度的反向增长。")
    return Q4Data(time_s, radii, moisture, surface, source_started_at_s)


def load_q3_results(path: Path) -> Q3Data:

    if not path.is_file():
        raise FileNotFoundError(f"找不到问题三结果文件：{path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None or len(header) < 3:
        raise ValueError("result3.xlsx 结构异常。")
    radii = np.asarray([_as_float(value, "问题三径向表头") for value in header[1:]], dtype=float)
    times: list[float] = []
    values: list[list[float]] = []
    for row_number, row in enumerate(rows, start=2):
        if not row or all(value is None for value in row):
            continue
        times.append(_as_float(row[0], f"问题三第 {row_number} 行时间"))
        values.append([_as_float(value, f"问题三第 {row_number} 行含水率") for value in row[1:]])
    workbook.close()
    time_s = np.asarray(times, dtype=float)
    moisture = np.asarray(values, dtype=float)
    if not np.all(np.diff(time_s) > 0) or moisture.shape != (time_s.size, radii.size):
        raise ValueError("问题三比较数据的维度或时间顺序异常。")
    if time_s[0] > 0:
        time_s = np.concatenate(([0.0], time_s))
        moisture = np.vstack((np.full(radii.size, INITIAL_C), moisture))
    return Q3Data(time_s, radii, moisture)


def load_environment_scenarios(path: Path) -> tuple[EnvironmentScenario, ...]:


    if not path.is_file():
        raise FileNotFoundError(
            "环境情景灵敏度结果不存在："
            f"{path}。生成 environment_sensitivity 时请通过 "
            "--environment-scenarios 指定 environment-scenarios.csv。"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("环境情景 CSV 缺少表头。")
        required = {"question", "case", "delta_min", "delta_s"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"环境情景 CSV 缺少字段：{', '.join(sorted(missing))}")
        records: list[EnvironmentScenario] = []
        for row_number, row in enumerate(reader, start=2):
            question = str(row["question"] or "").strip()
            case = str(row["case"] or "").strip()
            if question not in {"Q3", "Q4"} or case not in {"Ta_low", "Ta_high", "Ce_low", "Ce_high"}:
                continue
            records.append(
                EnvironmentScenario(
                    question=question,
                    case=case,
                    delta_min=_as_float(row["delta_min"], f"环境情景第 {row_number} 行 delta_min"),
                    delta_s=_as_float(row["delta_s"], f"环境情景第 {row_number} 行 delta_s"),
                )
            )
    expected = {(question, case) for question in ("Q3", "Q4") for case in ("Ta_low", "Ta_high", "Ce_low", "Ce_high")}
    received = {(record.question, record.case) for record in records}
    if received != expected or len(records) != len(expected):
        missing = expected - received
        duplicate_or_extra = len(records) != len(received)
        details = []
        if missing:
            details.append("缺少 " + ", ".join(f"{q}/{c}" for q, c in sorted(missing)))
        if duplicate_or_extra:
            details.append("存在重复情景行")
        raise ValueError("环境情景 CSV 必须包含 Q3、Q4 各四种唯一扰动：" + "；".join(details))
    return tuple(records)


# 移动边界重构与绘图数据汇总。

def shrinkage_volume_ratio(z: np.ndarray | float) -> np.ndarray | float:

    values = np.asarray(z, dtype=float)
    if np.any((values < -1e-9) | (values > 1.0 + 1e-9)):
        raise ValueError("归一化平均含水状态超出 [0, 1]。")
    result = J_DRY + (1.0 - J_DRY) * np.expm1(BETA_SHRINKAGE * values) / np.expm1(BETA_SHRINKAGE)
    if np.isscalar(z):
        return float(result)
    return result


def reconstruct_radius(q4: Q4Data) -> tuple[np.ndarray, np.ndarray, int, float]:


    valid_counts = np.sum(np.isfinite(q4.moisture), axis=1)
    if np.any(valid_counts < 1):
        raise ValueError("至少一个时刻没有有效中心数据。")
    lower_bound = q4.radii_cm[valid_counts - 1]
    radius_cm = lower_bound.copy()
    raw_radius = np.empty_like(radius_cm)

    for index, bound in enumerate(lower_bound):
        estimate = max(float(bound), 1e-9)
        for _ in range(5):
            valid = np.isfinite(q4.moisture[index]) & (q4.radii_cm < estimate - 1e-9)
            x = np.concatenate((q4.radii_cm[valid], [estimate]))
            y = np.concatenate((q4.moisture[index, valid], [q4.surface_moisture[index]]))
            if x.size < 2 or not np.all(np.diff(x) > 0):
                raise ValueError(f"t={q4.time_s[index]:g} s 无法由结果表重构径向状态。")
            mean = 2.0 * np.trapezoid(y * x, x) / estimate**2
            z = float(np.clip(mean / INITIAL_C, 0.0, 1.0))
            estimate = INITIAL_RADIUS_CM * math.sqrt(float(shrinkage_volume_ratio(z)))
        raw_radius[index] = estimate
        radius_cm[index] = max(float(bound), estimate)

    radius_cm = np.minimum.accumulate(radius_cm)
    if np.any(radius_cm + 1e-12 < lower_bound):
        raise ValueError("结果表的有效域掩码与收缩半径重构不一致。")
    mean_moisture = volume_weighted_mean(q4, radius_cm)
    equivalent_loss = INITIAL_C - mean_moisture
    correction = np.maximum(lower_bound - raw_radius, 0.0)
    return equivalent_loss, radius_cm, int(np.count_nonzero(correction > 0.0)), float(np.max(correction))


def profile_points(q4: Q4Data, radius_cm: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:

    radius = float(radius_cm[index])
    valid = np.isfinite(q4.moisture[index]) & (q4.radii_cm < radius - 1e-9)
    x = np.concatenate((q4.radii_cm[valid], [radius]))
    y = np.concatenate((q4.moisture[index, valid], [q4.surface_moisture[index]]))
    if x.size < 2 or not np.all(np.diff(x) > 0):
        raise ValueError(f"t={q4.time_s[index]:g} s 的径向重构点不严格递增。")
    return x, y


def volume_weighted_mean(q4: Q4Data, radius_cm: np.ndarray) -> np.ndarray:
    means = np.empty(q4.time_s.size, dtype=float)
    for index, radius in enumerate(radius_cm):
        x, y = profile_points(q4, radius_cm, index)
        integral = np.trapezoid(y * x, x)
        means[index] = 2.0 * integral / float(radius) ** 2
    return means


def threshold_front(q4: Q4Data, radius_cm: np.ndarray) -> np.ndarray:

    front = np.full(q4.time_s.size, np.nan, dtype=float)
    for index in range(q4.time_s.size):
        x, y = profile_points(q4, radius_cm, index)
        if y[-1] > CRITICAL_C:
            continue
        if y[0] <= CRITICAL_C:
            front[index] = 0.0
            continue
        crossings = np.flatnonzero((y[:-1] > CRITICAL_C) & (y[1:] <= CRITICAL_C))
        if crossings.size == 0:
            raise ValueError(f"t={q4.time_s[index]:g} s 无法定位阈值前沿。")
        j = int(crossings[0])
        denominator = y[j + 1] - y[j]
        if denominator == 0.0:
            front[index] = x[j + 1]
        else:
            front[index] = x[j] + (CRITICAL_C - y[j]) * (x[j + 1] - x[j]) / denominator
    return front


def build_plot_fields(q4: Q4Data, radius_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radial_grid = np.linspace(0.0, INITIAL_RADIUS_CM, 201)
    material_grid = np.linspace(0.0, 1.0, 201)
    if q4.time_s.size * radial_grid.size > MAX_HEATMAP_PIXELS:
        raise MemoryError("时空热图超过 MAX_HEATMAP_PIXELS，请提高受控聚合强度。")
    physical = np.full((q4.time_s.size, radial_grid.size), np.nan, dtype=float)
    material = np.empty((q4.time_s.size, material_grid.size), dtype=float)
    for index, radius in enumerate(radius_cm):
        x, y = profile_points(q4, radius_cm, index)
        inside = radial_grid <= radius + 1e-12
        physical[index, inside] = np.interp(radial_grid[inside], x, y)
        material[index] = np.interp(material_grid, x / radius, y)
    return radial_grid, physical, material_grid, material


def derive_q4(q4: Q4Data) -> DerivedQ4:
    cumulative_loss, radius_cm, count, maximum_correction = reconstruct_radius(q4)
    mean_moisture = volume_weighted_mean(q4, radius_cm)
    maximum_moisture = np.maximum(np.nanmax(q4.moisture, axis=1), q4.surface_moisture)
    if np.any(np.diff(maximum_moisture) > 5e-5 + 1e-12):
        raise ValueError("全局输出节点最大含水率出现超出四位舍入尺度的反向增长。")
    front = threshold_front(q4, radius_cm)
    radial_grid, physical, material_grid, material = build_plot_fields(q4, radius_cm)
    return DerivedQ4(
        cumulative_loss=cumulative_loss,
        radius_cm=radius_cm,
        volume_ratio=(radius_cm / INITIAL_RADIUS_CM) ** 2,
        mean_moisture=mean_moisture,
        maximum_moisture=maximum_moisture,
        dry_front_cm=front,
        physical_grid_cm=radial_grid,
        physical_field=physical,
        material_grid=material_grid,
        material_field=material,
        radius_lower_bound_corrections=count,
        maximum_radius_correction_cm=maximum_correction,
    )


# 绘图辅助函数。

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
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.5,
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


def box_axis(axis: plt.Axes, *, grid: str | None = None) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(TEXT)
        spine.set_linewidth(0.75)
    axis.tick_params(direction="out", length=3.0, width=0.7, top=True, right=True)
    if grid:
        axis.grid(axis=grid, color=GRID, linewidth=0.5, alpha=0.75, zorder=0)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-18, 7),
        textcoords="offset points",
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def require_matplotlib_panel_alignment(fig: plt.Figure) -> None:

    fig.canvas.draw()
    axes = []
    for axis in fig.axes:
        try:
            spec = axis.get_subplotspec()
        except AttributeError:
            spec = None
        if spec is not None and axis.get_label() != "<colorbar>":
            axes.append((axis, axis.get_position()))
    for i, (_, first) in enumerate(axes):
        for _, second in axes[i + 1 :]:
            overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
            overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
            if overlap_x > 1e-4 and overlap_y > 1e-4:
                raise RuntimeError("多面板绘图区发生重叠，已停止导出。")


def exact_time_index(q4: Q4Data, target_h: float, atol_s: float = 1e-6) -> int:
    target_s = target_h * 3600.0
    found = np.flatnonzero(np.isclose(q4.time_s, target_s, atol=atol_s, rtol=0.0))
    if found.size != 1:
        raise ValueError(f"结果中没有唯一的 t={target_h:g} h 保存行。")
    return int(found[0])


def report_indices(q4: Q4Data) -> list[int]:
    return [*(exact_time_index(q4, hour) for hour in REPORT_HOURS), q4.time_s.size - 1]


def line_profile(axis: plt.Axes, context: Context, index: int, color: str, label: str, linestyle: str) -> None:
    x, y = profile_points(context.q4, context.derived.radius_cm, index)
    axis.plot(x, y, color=color, linestyle=linestyle, label=label, solid_capstyle="round")
    axis.plot(x[-1], y[-1], marker="o", markersize=3.4, color=color, markeredgecolor="white", markeredgewidth=0.45)


# 各图对应问题四收缩耦合模型的关键证据。

def figure_moving_domain_map(context: Context) -> plt.Figure:
    q4, derived = context.q4, context.derived
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.81), constrained_layout=True, sharex=True)
    cmap = MOISTURE_CMAP.copy()
    cmap.set_bad("white")
    positive_min = max(0.045, float(np.nanmin(derived.physical_field)))
    norm = LogNorm(vmin=positive_min, vmax=INITIAL_C)
    mesh0 = axes[0].pcolormesh(q4.hours, derived.physical_grid_cm, derived.physical_field.T, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    axes[0].plot(q4.hours, derived.radius_cm, color=TEXT, linewidth=1.5)
    axes[0].contour(q4.hours, derived.physical_grid_cm, derived.physical_field.T, levels=[CRITICAL_C], colors=[RED], linewidths=1.2)
    axes[0].set(xlabel="时间 t / h", ylabel="实际径向坐标 r / cm", ylim=(0.0, INITIAL_RADIUS_CM), title="物理坐标中的移动域含水率场")
    domain_handles = [
        Line2D([0], [0], color=TEXT, linewidth=1.5, label="真实移动表面 R(t)"),
        Line2D([0], [0], color=RED, linewidth=1.2, label="0.15 阈值前沿"),
        Patch(facecolor="white", edgecolor=MUTED, linewidth=0.7, label="药材域外：r > R(t)"),
    ]
    axes[0].legend(handles=domain_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False, handletextpad=1.15)
    box_axis(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].pcolormesh(q4.hours, derived.material_grid, derived.material_field.T, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    axes[1].contour(q4.hours, derived.material_grid, derived.material_field.T, levels=[CRITICAL_C], colors=[RED], linewidths=1.2)
    axes[1].set(xlabel="时间 t / h", ylabel="材料坐标 ξ = r/R(t)", ylim=(0.0, 1.0), title="材料坐标中的含水率场")
    box_axis(axes[1])
    panel_label(axes[1], "(b)")
    colourbar = fig.colorbar(mesh0, ax=axes, location="right", shrink=0.92, pad=0.025)
    colourbar.set_label("干基含水率 C / (kg/kg)")
    colourbar.ax.tick_params(labelsize=7.3)
    endpoint = float(q4.hours[-1])
    tick_values = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, endpoint])
    tick_labels = ["0", "10", "20", "30", "40", f"{endpoint:.2f}"]
    for axis in axes:
        axis.set_xlim(0.0, endpoint)
        axis.set_xticks(tick_values, tick_labels)
    return fig


def figure_coupling_scales(context: Context) -> plt.Figure:
    q4, derived = context.q4, context.derived
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.34), constrained_layout=False)
    # 固定边距以保持三个子图绘图区等宽。
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.215, top=0.795, wspace=0.470)
    radius_ratio = derived.radius_cm / INITIAL_RADIUS_CM
    axes[0].plot(q4.hours, radius_ratio, color=RED, label="半径保持率 R/R0")
    axes[0].plot(q4.hours, derived.volume_ratio, color=PURPLE, linestyle="--", label="体积保持率 V/V0")
    axes[0].set(xlabel="时间 t / h", ylabel="保持率", ylim=(0.30, 1.03), title="半径与体积保持率")
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        handlelength=2.4,
        labelspacing=0.45,
    )
    box_axis(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].plot(q4.hours, 1.0 / radius_ratio, color=TEAL, label="表面交换尺度")
    axes[1].plot(q4.hours, 1.0 / radius_ratio**2, color=ORANGE, linestyle="--", label="内部传递尺度")
    axes[1].axhline(1.0, color=MUTED, linewidth=0.8)
    axes[1].set(xlabel="时间 t / h", ylabel="相对初始几何尺度因子", title="收缩引起的几何尺度变化")
    axes[1].legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        frameon=False,
        fontsize=6.8,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    box_axis(axes[1])
    panel_label(axes[1], "(b)")

    factor_initial = math.exp(-0.30 / INITIAL_C)
    centre_factor = np.exp(-0.30 / q4.moisture[:, 0]) / factor_initial
    surface_factor = np.exp(-0.30 / q4.surface_moisture) / factor_initial
    axes[2].plot(q4.hours, centre_factor, color=BLUE, label="中心")
    axes[2].plot(q4.hours, surface_factor, color=RED, linestyle="--", label="真实表面")
    axes[2].set_yscale("log")
    axes[2].set(xlabel="时间 t / h", ylabel="相对初始含水扩散因子", title="低含水率下的扩散抑制")
    axes[2].legend(
        loc="lower left",
        bbox_to_anchor=(0.035, 0.045),
        frameon=False,
        fontsize=6.8,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    box_axis(axes[2])
    panel_label(axes[2], "(c)")
    endpoint = float(q4.hours[-1])
    tick_values = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, endpoint])
    tick_labels = ["0", "10", "20", "30", "40", f"{endpoint:.2f}"]
    for axis in axes:
        axis.set_xlim(0.0, endpoint + 1.25)
        axis.set_xticks(tick_values, tick_labels)
    return fig


def figure_radial_profiles(context: Context) -> plt.Figure:
    q4 = context.q4
    indices = report_indices(q4)
    times_h = np.asarray([q4.hours[index] for index in indices])
    colours = TIME_CMAP(Normalize(times_h.min(), times_h.max())(times_h))
    styles = ("-", "--", "-.", ":", "-", "--", "-.", ":", "-")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.50), constrained_layout=True, sharex=True, sharey=True)
    for axis, selected, title in (
        (axes[0], range(0, 4), "6–24 h：快速失水阶段"),
        (axes[1], range(4, 9), "30 h 至终点：内部控制阶段"),
    ):
        for position in selected:
            time_label = f"{times_h[position]:.4f} h" if position == 8 else f"{times_h[position]:.0f} h"
            line_profile(axis, context, indices[position], colours[position], time_label, styles[position])
        axis.axhline(CRITICAL_C, color=RED, linewidth=1.0, linestyle=(0, (3, 2)), label="阈值 0.15")
        axis.set_yscale("log")
        axis.set(xlabel="实际距中心距离 r / cm", xlim=(0.0, 1.45), ylim=(0.045, 2.2), title=title)
        axis.legend(loc="best", ncol=2, frameon=False, handlelength=2.4)
        box_axis(axis, grid="y")
    axes[0].set_ylabel("干基含水率 C / (kg/kg)")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def figure_front_and_criterion(context: Context) -> plt.Figure:
    q4, derived = context.q4, context.derived
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.77), constrained_layout=True)
    front = derived.dry_front_cm
    available = np.isfinite(front)
    axes[0].plot(q4.hours, derived.radius_cm, color=TEXT, linewidth=1.8, label="真实移动表面")
    axes[0].plot(q4.hours[available], front[available], color=RED, linewidth=1.8, label="0.15 阈值前沿")
    axes[0].fill_between(q4.hours, front, derived.radius_cm, where=available, color=ORANGE, alpha=0.22, label="实际药材域内低于阈值的外层")
    axes[0].set(xlabel="时间 t / h", ylabel="实际径向坐标 r / cm", ylim=(0.0, INITIAL_RADIUS_CM), title="收缩表面与含水率阈值前沿")
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        handlelength=2.4,
        labelspacing=0.45,
    )
    box_axis(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].plot(q4.hours, derived.maximum_moisture, color=TEXT, linewidth=1.9, label="全域最大含水率")
    axes[1].plot(q4.hours, derived.mean_moisture, color=TEAL, linestyle="--", label="体积加权平均含水率")
    axes[1].plot(q4.hours, q4.surface_moisture, color=ORANGE, linestyle=":", label="真实表面含水率")
    axes[1].axhline(CRITICAL_C, color=RED, linewidth=1.0, linestyle=(0, (3, 2)), label="全域停止阈值 0.15")
    axes[1].axvline(q4.hours[-1], color=MUTED, linewidth=0.9, linestyle="--")
    axes[1].scatter([q4.hours[-1]], [derived.maximum_moisture[-1]], s=22, color=RED, zorder=5)
    axes[1].text(
        0.025,
        0.025,
        f"全域首次低于 0.15\n{q4.hours[-1]:.4f} h（中心控制）",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        color=TEXT,
        fontsize=7.2,
    )
    axes[1].set_yscale("log")
    axes[1].set(xlabel="时间 t / h", ylabel="干基含水率 C / (kg/kg)", ylim=(0.045, 3.0), title="全域停止判据与控制位置")
    axes[1].legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        handlelength=2.4,
        labelspacing=0.40,
    )
    box_axis(axes[1])
    panel_label(axes[1], "(b)")
    endpoint = float(q4.hours[-1])
    tick_values = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, endpoint])
    tick_labels = ["0", "10", "20", "30", "40", f"{endpoint:.2f}"]
    for axis in axes:
        axis.set_xlim(0.0, endpoint + 1.25)
        axis.set_xticks(tick_values, tick_labels)
    return fig


def figure_q3_q4_comparison(context: Context) -> plt.Figure:
    q4, q3, derived = context.q4, context.q3, context.derived
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.30), constrained_layout=True, gridspec_kw={"width_ratios": (1.55, 1.0)})
    q3_maximum = np.max(q3.moisture, axis=1)
    axes[0].plot(q3.hours, q3_maximum, color=BLUE, linestyle="--", linewidth=1.8, label="问题三：固定半径模型")
    axes[0].plot(q4.hours, derived.maximum_moisture, color=RED, linewidth=1.9, label="问题四：收缩耦合模型")
    axes[0].axhline(CRITICAL_C, color=TEXT, linestyle=":", linewidth=1.0, label="停止阈值")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="时间 t / h", ylabel="全域最大含水率 / (kg/kg)", ylim=(0.12, 3.0), title="全域控制量的跨问题比较")
    axes[0].legend(loc="upper right", frameon=False)
    box_axis(axes[0], grid="y")
    panel_label(axes[0], "(a)")

    durations = np.asarray([q3.time_s[-1], q4.delivered_time_s]) / 3600.0
    labels = ["问题三\n固定半径", "问题四\n收缩耦合"]
    bars = axes[1].barh(labels, durations, color=[BLUE, RED], height=0.52, edgecolor=TEXT, linewidth=0.7)
    axes[1].bar_label(bars, labels=[f"{value:.4f} h" for value in durations], padding=4, fontsize=7.6)
    difference = durations[0] - durations[1]
    axes[1].text(
        0.50,
        0.50,
        f"综合时长差：{difference:.4f} h（{difference / durations[0] * 100:.2f}%）\n同时包含收缩闭合与题定物性变化",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )
    axes[1].set(xlabel="模型干燥时长 / h", xlim=(0.0, max(durations) * 1.22), title="题目答案的综合差异")
    axes[1].invert_yaxis()
    box_axis(axes[1], grid="x")
    panel_label(axes[1], "(b)")
    return fig


def figure_required_value_matrix(context: Context) -> plt.Figure:
    q4 = context.q4
    indices = report_indices(q4)
    fixed_targets = (0.0, 0.5, 1.0, 1.5, 2.0)
    columns = [int(np.flatnonzero(np.isclose(q4.radii_cm, radius, atol=1e-10))[0]) for radius in fixed_targets]
    matrix = np.full((len(indices), len(columns) + 1), np.nan, dtype=float)
    for row, index in enumerate(indices):
        matrix[row, :-1] = q4.moisture[index, columns]
        matrix[row, -1] = q4.surface_moisture[index]

    fig, axis = plt.subplots(figsize=(FULL_WIDTH_IN, 2.58), constrained_layout=True)
    cmap = MOISTURE_CMAP.copy()
    cmap.set_bad("#ECEFF1")
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=LogNorm(vmin=0.05, vmax=1.8), interpolation="nearest")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                text_value, colour = "域外", MUTED
            else:
                text_value = f"{value:.4f}"
                colour = "white" if value >= 0.55 else TEXT
            axis.text(column, row, text_value, ha="center", va="center", fontsize=7.2, color=colour)
    labels_y = [f"{q4.hours[index]:.4f}" if index == indices[-1] else f"{q4.hours[index]:.0f}" for index in indices]
    axis.set_xticks(np.arange(matrix.shape[1]), ["0", "0.5", "1.0", "1.5", "2.0", "真实表面"])
    axis.set_yticks(np.arange(matrix.shape[0]), labels_y)
    axis.set_xticks(np.arange(-0.5, matrix.shape[1], 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1.0), minor=True)
    axis.grid(which="minor", color=TEXT, linewidth=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set(xlabel="到药材中心的距离 / cm", ylabel="时间 t / h", title="规定时刻与位置的干基含水率")
    box_axis(axis)
    colourbar = fig.colorbar(image, ax=axis, pad=0.025, shrink=0.94)
    colourbar.set_label("干基含水率 C / (kg/kg)")
    colourbar.ax.tick_params(labelsize=7.3)
    return fig


def figure_environment_sensitivity(context: Context) -> plt.Figure:

    records = context.environment_scenarios
    if records is None:
        raise RuntimeError("环境情景结果尚未加载，无法绘制环境灵敏度图。")
    lookup = {(record.question, record.case): record for record in records}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.48), constrained_layout=True)
    cases = (("Ta_low", "低温"), ("Ta_high", "高温"), ("Ce_low", "低环境含水率"), ("Ce_high", "高环境含水率"))
    colours = {"low": RED, "high": BLUE}
    marker_style = {"Q3": "o", "Q4": "s"}

    for axis, title, selector, value_name, unit in (
        (axes[0], "温度边界扰动", lambda case: case.startswith("Ta_"), "delta_min", "min"),
        (axes[1], "环境含水率扰动", lambda case: case.startswith("Ce_"), "delta_s", "s"),
    ):
        selected = [(case, label) for case, label in cases if selector(case)]
        x = np.arange(len(selected), dtype=float)
        all_values = []
        for question, offset in (("Q3", -0.11), ("Q4", 0.11)):
            values = np.asarray([getattr(lookup[(question, case)], value_name) for case, _ in selected], dtype=float)
            all_values.extend(values.tolist())
            for pos, (case, _) in enumerate(selected):
                colour = colours["low" if case.endswith("low") else "high"]
                axis.scatter(
                    x[pos] + offset,
                    values[pos],
                    s=30,
                    marker=marker_style[question],
                    facecolor=colour,
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=4,
                )
        axis.axhline(0.0, color=TEXT, linewidth=0.75, zorder=1)
        lower, upper = min(all_values), max(all_values)
        pad = max((upper - lower) * 0.20, 1.0 if unit == "min" else 80.0)
        axis.set(
            xticks=x,
            xticklabels=[label for _, label in selected],
            ylabel=f"终点时间变化 Δt / {unit}",
            ylim=(lower - pad, upper + pad),
            title=title,
        )
        axis.tick_params(axis="x", pad=4)
        box_axis(axis, grid="y")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=5.5, label="问题三"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=5.5, label="问题四"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=5.5, label="低边界水平"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=5.5, label="高边界水平"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), frameon=False, columnspacing=1.25, handletextpad=0.45)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


FIGURES: Final[dict[str, FigureSpec]] = {
    "moving_domain_map": FigureSpec(
        "fig_q4_moving_domain_fields",
        TITLE_MOVING_DOMAIN_MAP,
        "同一含水率场分别在实际径向坐标 r 和随材料运动的无量纲坐标 ξ=r/R(t) 中展示。左图白色区域表示 r>R(t) 的药材域外空间，并非缺失值；黑线为真实移动表面。右图消除几何域变化，红线表示 0.15 kg/kg 阈值前沿。横坐标完整延伸至 50.9133 h 终点。",
        figure_moving_domain_map,
    ),
    "coupling_scales": FigureSpec(
        "fig_q4_shrinkage_effects",
        TITLE_COUPLING_SCALES,
        "半径和体积保持率刻画宏观收缩；相对初始几何尺度因子表示收缩引起的传递尺度变化；相对初始含水扩散因子表示局部含水率下降对扩散的抑制。几何尺度变化不能单独换算为干燥时间贡献，最终速率还同时受温度、浓度梯度和表面传质阻力影响。",
        figure_coupling_scales,
    ),
    "radial_profiles": FigureSpec(
        "fig_q4_radial_profiles",
        "规定时刻的空间分布",
        "每 6 h 及最终时刻的径向含水率曲线均终止于当时真实表面，端点圆点为独立表面状态。早期外层快速失水，后期中心成为控制位置；实际横坐标避免把不同半径时刻误画在同一固定域。",
        figure_radial_profiles,
    ),
    "front_and_criterion": FigureSpec(
        "fig_q4_drying_front_and_criterion",
        TITLE_FRONT_AND_CRITERION,
        "左图把真实收缩表面与 0.15 kg/kg 干燥前沿同时置于实际药材域中，阴影仅表示域内已低于阈值的外层。右图比较全域最大含水率、体积加权平均含水率和真实表面含水率，说明平均值与表面值不能替代全域判据；中心最终控制 50.9133 h 的停止时刻。",
        figure_front_and_criterion,
    ),
    "q3_q4_comparison": FigureSpec(
        "fig_q4_model_comparison",
        "相较问题三的模型增量",
        "比较固定半径问题三与收缩耦合问题四的全域最大含水率和模型干燥时长。两问同时改变几何闭合与题定物性，因此图中只报告综合差异，不把全部时长变化归因于收缩。",
        figure_q3_q4_comparison,
    ),
    "required_value_matrix": FigureSpec(
        "fig_q4_moisture_value_matrix",
        "题目规定数值的直接呈现",
        "矩阵逐项给出每 6 h、最终时刻以及指定固定位置和真实表面的四位含水率。灰色“域外”单元表示该固定厘米位置已被收缩表面越过，既不是零值，也不是普通缺失。",
        figure_required_value_matrix,
    ),
    "environment_sensitivity": FigureSpec(
        "fig_q4_environment_sensitivity",
        TITLE_ENVIRONMENT_SENSITIVITY,
        "基于分别重算得到的长期边界情景结果，比较温度和环境含水率上、下调对问题三与问题四终点时间的影响。纵轴为相对各自基准情景的终点时间变化，正值表示干燥时长延长；颜色区分边界水平，标记形状区分问题编号。",
        figure_environment_sensitivity,
    ),
}


# 图片导出与命令行入口。

def parse_names(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(FIGURES)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(names) - set(FIGURES))
    if unknown or not names:
        raise ValueError(f"未知图形：{unknown}；可用名称：{', '.join(FIGURES)}")
    return list(dict.fromkeys(names))


def parse_format(value: str) -> str:
    image_format = value.strip().lower()
    if image_format == "jpeg":
        image_format = "jpg"
    if image_format not in {"png", "jpg"}:
        raise ValueError("--format 仅支持 png、jpg/jpeg。")
    return image_format


def validate_output_directory(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    unexpected = [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() not in ALLOWED_OUTPUT_SUFFIXES]
    if unexpected:
        raise RuntimeError(f"正式输出目录存在非 PNG/JPG 文件：{unexpected}")


def validate_image(path: Path, requested_format: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"图像未正确写入：{path}")
    if path.stat().st_size > MAX_FIGURE_BYTES:
        raise RuntimeError(f"图像超过 {MAX_FIGURE_BYTES / 1024**2:.0f} MiB 上限：{path}")
    with Image.open(path) as image:
        image.load()
        actual_format = (image.format or "").lower()
        expected = "jpeg" if requested_format == "jpg" else "png"
        if actual_format != expected:
            raise RuntimeError(f"图像格式不符：期望 {expected}，实际 {actual_format}")
        width, height = image.size
        if width < 1200 or height < 700:
            raise RuntimeError(f"图像像素尺寸不足：{width}×{height}")
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise RuntimeError(f"图像含非预期透明像素：{path}")
        rgb = image.convert("RGB")
        extrema = ImageStat.Stat(rgb.resize((64, 64))).extrema
        if all(low == high for low, high in extrema):
            raise RuntimeError(f"图像疑似空白或纯色：{path}")


def save_figure(fig: plt.Figure, destination: Path, image_format: str, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖既有图片：{destination}；如需覆盖请显式传入 --overwrite。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig)
    try:
        if image_format == "png":
            fig.savefig(destination, format="png", dpi=600, facecolor="white", edgecolor="white")
        elif image_format == "jpg":
            fig.savefig(destination, format="jpg", dpi=600, facecolor="white", edgecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
        else:
            raise ValueError(f"不支持的图片格式：{image_format}")
        validate_image(destination, image_format)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 CUMCM 问题四的论文科研图像。")
    parser.add_argument("--results", type=Path, default=DEFAULT_Q4_RESULTS, help="result4.xlsx 路径")
    parser.add_argument("--q3-results", type=Path, default=DEFAULT_Q3_RESULTS, help="result3.xlsx 对照路径")
    parser.add_argument(
        "--environment-scenarios",
        type=Path,
        default=None,
        help="environment-scenarios.csv 路径；仅 environment_sensitivity 图需要，不能由 result4.xlsx 代替",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="正式图片输出目录")
    parser.add_argument("--figures", default="all", help="all 或逗号分隔：" + ", ".join(FIGURES))
    parser.add_argument("--format", default="png", help="png 或 jpg/jpeg；默认 png")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名图片；不会删除其他图片")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = parse_names(args.figures)
    image_format = parse_format(args.format)
    paths = {
        "results": args.results.resolve(),
        "q3": args.q3_results.resolve(),
        "output": args.output.resolve(),
    }
    input_keys = ["results", "q3"]
    if "environment_sensitivity" in selected:
        if args.environment_scenarios is None:
            raise ValueError(
                "生成 environment_sensitivity 时必须提供 --environment-scenarios "
                "environment-scenarios(1).csv。"
            )
        paths["environment_scenarios"] = args.environment_scenarios.resolve()
        input_keys.append("environment_scenarios")
    input_parents = {paths[key].parent for key in input_keys}
    if paths["output"] in input_parents:
        raise ValueError("正式输出目录不能与任一输入数据目录相同。")
    paths["output"].mkdir(parents=True, exist_ok=True)
    validate_output_directory(paths["output"])
    configure_matplotlib()

    q4 = load_q4_results(paths["results"])
    q3 = load_q3_results(paths["q3"])
    derived = derive_q4(q4)
    scenarios = load_environment_scenarios(paths["environment_scenarios"]) if "environment_sensitivity" in selected else None
    context = Context(q4, derived, q3, scenarios)

    print(f"输入：{paths['results']}")
    print(f"读取策略：openpyxl read_only + iter_rows；未修改任何源文件")
    print(f"问题四数据规模：{q4.time_s.size:,} 个时刻（含内存补入 t=0）× {q4.radii_cm.size} 个固定位置 + 真实表面")
    print(f"时间范围：0–{q4.delivered_time_s / 3600.0:.6f} h；工作簿首个保存时刻 {q4.source_started_at_s:g} s")
    print(f"域外空白：{np.count_nonzero(~np.isfinite(q4.moisture)):,} 个；绘图中保持为掩码，不填零")
    print("半径重构：result4 径向场体积平均状态回代有限收缩本构；未读取外部输入表")
    print(
        f"四位舍入导致的域掩码下界修正：{derived.radius_lower_bound_corrections} 个时刻；"
        f"最大 {derived.maximum_radius_correction_cm:.6f} cm"
    )
    print(f"热图聚合：完整 {q4.time_s.size:,} 个时刻 × 201 个绘图半径，不随机抽样")
    print("注意：严格小于 0.15 的结论依赖模型章节的全精度终点核验；图中结果表数值为四位小数。")
    if scenarios is not None:
        print(f"环境情景结果：{paths['environment_scenarios']}；已校验 Q3/Q4 各四种唯一边界扰动。")

    for name in selected:
        spec = FIGURES[name]
        figure = spec.builder(context)
        extension = ".jpg" if image_format == "jpg" else ".png"
        destination = paths["output"] / f"{spec.filename}{extension}"
        save_figure(figure, destination, image_format, overwrite=args.overwrite)
        print(f"\n[{name}] {spec.role}")
        print(f"图注草案：{spec.caption}")
        print(f"已保存：{destination}")

    validate_output_directory(paths["output"])
    print("\n输出检查：正式目录仅含 PNG/JPG/JPEG；格式、像素、透明度、大小和空白画布检查通过。")
    return 0


def verify_chapter_version(path: Path) -> None:

    if not path.is_file():
        print(f"Model chapter not supplied; hash check skipped: {path}")
        return
    if file_sha256(path) != EXPECTED_CHAPTER_SHA256:
        raise ValueError(f"Model chapter hash mismatch: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
