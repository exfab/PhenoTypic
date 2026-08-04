"""Public plotting contracts for operations, analyzers, and plot models.

The classes in this package describe plotting behavior and refresh timing. They
also define the single-page and multi-page values returned by custom plot
providers. Runtime paths, pipeline bindings, figure persistence, and GUI
adapters belong to the private :mod:`phenotypic.plotting._pipeline` package.
"""

from ._lifecycle import PlotAnalysis, PlotImage, PlotMeas, PlotQc
from ._output import FigureLike, PlotOutput, PlotPage, canonical_group_key
from ._pht_plot import BoundFigures, Control, FigureSpec, PhtPlot, figure

__all__ = [
    "BoundFigures",
    "Control",
    "FigureLike",
    "FigureSpec",
    "PhtPlot",
    "PlotAnalysis",
    "PlotImage",
    "PlotMeas",
    "PlotOutput",
    "PlotPage",
    "PlotQc",
    "canonical_group_key",
    "figure",
]
