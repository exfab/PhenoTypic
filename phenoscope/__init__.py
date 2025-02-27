__version__ = "0.5.0"

import platform

# noinspection PyUnresolvedReferences
from .core import Image, imread

# noinspection PyUnresolvedReferences
from . import (
    data,
    detection,
    features,
    grid,
    interface,
    objects,
    morphology,
    pipeline,
    preprocessing,
    profiler,
    transform,
    util
)

# if platform.system() == 'Linux' or platform.system() == 'Darwin':
#     from . import cellprofiler_api
