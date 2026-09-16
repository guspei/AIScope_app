"""Estilo AiScope para figuras (paleta y tipografía de GDD-app)."""
import matplotlib as mpl

from aiscope.data.classes import MASK_COLORS

BRAND = {
    "purple": "#7e65a8",
    "teal": "#65c2ca",
    "teal_dark": "#5cbeaf",
    "sky": "#9ed7ed",
    "navy": "#282e3e",
    "ink": "#25312f",
}
STAGE_COLORS = {stage: color for color, stage in MASK_COLORS.items()}


FONTS = ["Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"]


def apply():
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    mpl.rcParams.update({
        "font.family": [next(f for f in FONTS if f in installed)],
        "axes.prop_cycle": mpl.cycler(color=[BRAND["purple"], BRAND["teal"], BRAND["navy"], BRAND["sky"], BRAND["teal_dark"]]),
        "axes.edgecolor": BRAND["navy"],
        "axes.labelcolor": BRAND["navy"],
        "axes.titlecolor": BRAND["navy"],
        "xtick.color": BRAND["navy"],
        "ytick.color": BRAND["navy"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    })
