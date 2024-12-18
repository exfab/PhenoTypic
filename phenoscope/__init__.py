__version__ = "0.4.2a"

import platform

from .core import Image, imread

from . import data, detection , feature_extraction, grid, interface, map_modification, morphology, pipeline, preprocessing, profiler, transform, util

if platform.system()=='Linux' or platform.system()=='Darwin':
    from . import cellprofiler_api
