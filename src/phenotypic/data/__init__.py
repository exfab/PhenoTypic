"""Sample and synthetic agar plate images for fungal colony testing.

Provides loaders and generators for representative plate scenes used in demos,
benchmarks, and tests: synthetic single colonies, synthetic full plates, time-course
captures at 12–72 hours, early low-contrast crops, and smear-plate examples. Utilities
return arrays or ready-to-use `Image`/`GridImage` objects for rapid pipeline trials.
"""

from ._sample_image_data import *

__all__ = [
    "load_synthetic_colony",
    "make_synthetic_colony",
    "make_synthetic_plate",
    "load_plate_12hr",
    "load_plate_72hr",
    "load_plate_series",
    "load_early_colony",
    "load_faint_early_colony",
    "load_colony",
    "load_smear_plate_12hr",
    "load_smear_plate_24hr",
]
