"""Reusable Chinese scientific plots; require actual glyph coverage."""
from contextlib import contextmanager
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, ft2font


def chinese_font(text='中文坐标负数测量覆盖宽度距离米'):
    for name in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'SimSun']:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            glyphs = ft2font.FT2Font(path).get_charmap()
            if all(ord(c) in glyphs for c in text if not c.isspace()):
                return name
        except ValueError:
            continue
    raise RuntimeError('No installed font covers the required Chinese labels.')


@contextmanager
def style(text='中文坐标负数测量覆盖宽度距离米'):
    with matplotlib.rc_context({'font.family': chinese_font(text), 'axes.unicode_minus': False,
                                'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 300}):
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message=r'Glyph .* missing from font')
            yield


def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(stem.with_suffix('.png'), dpi=300)
