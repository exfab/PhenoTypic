"""Feature extraction from detected fungal colonies.

Computes per-colony and grid-level metrics describing growth, morphology, and
appearance on agar plates. Measurements span bounds, size/integrated intensity,
shape descriptors, texture, color, and grid-level statistics such as spatial spread
and linear gradients. Results are returned as pandas DataFrames ready for analysis.
"""

from ._measure_bounds import MeasureBounds
from ._measure_color import MeasureColor
from ._measure_color_composition import MeasureColorComposition
from ._measure_intensity import MeasureIntensity
from ._measure_shape import MeasureShape
from ._measure_size import MeasureSize
from ._measure_texture import MeasureTexture
from ._measure_grid_spread import MeasureGridSpread
from ._measure_grid_linreg_stats import MeasureGridLinRegStats

__all__ = [
    "MeasureBounds",
    "MeasureColor",
    "MeasureColorComposition",
    "MeasureIntensity",
    "MeasureShape",
    "MeasureSize",
    "MeasureTexture",
    "MeasureGridSpread",
    "MeasureGridLinRegStats",
]
