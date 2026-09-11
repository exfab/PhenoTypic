"""Probe: import one target with `phenotypic` replaced by a PEP 562 lazy package (throwaway; mirrors the planned __init__)."""
import importlib, importlib.util, sys, time, traceback, types
from pathlib import Path
target = sys.argv[1]
spec = importlib.util.find_spec("phenotypic")
pkg = types.ModuleType("phenotypic")
pkg.__path__ = list(spec.submodule_search_locations)
pkg.__file__, pkg.__spec__, pkg.__version__ = spec.origin, spec, "0.19.0"
_CLASSES = {"Image": "phenotypic._core._image", "GridImage": "phenotypic._core._grid_image", "ImagePipeline": "phenotypic._core._image_pipeline"}
_SUBPACKAGES = {"abc_", "analysis", "correction", "data", "detect", "enhance", "grid", "measure", "refine", "schema", "settings", "sdk_", "tune", "util", "prefab"}
def __getattr__(name):
    if name in _CLASSES:
        value = getattr(importlib.import_module(_CLASSES[name]), name)
    elif name in _SUBPACKAGES:
        value = importlib.import_module(f"phenotypic.{name}")
    else:
        raise AttributeError(f"module 'phenotypic' has no attribute {name!r}")
    setattr(pkg, name, value)
    return value
pkg.__getattr__ = __getattr__
sys.modules["phenotypic"] = pkg
importlib.import_module("phenotypic._startup_perf")
t0 = time.perf_counter()
try:
    mod = importlib.import_module(target.split(":")[0])
    if ":" in target:
        getattr(mod, target.split(":")[1])
    ok = "ok"
except Exception as exc:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "/src/phenotypic/" in f.filename]
    last = frames[-1] if frames else None
    where = f"{Path(last.filename).relative_to(Path.cwd() / 'src')}:{last.lineno}" if last else "?"
    ok = f"FAIL {type(exc).__name__}: {str(exc)[:120]} @ {where}"
dt = time.perf_counter() - t0
heavy = [m for m in ("scipy", "skimage", "pandas", "colour", "numba", "h5py", "matplotlib.pyplot", "plotly", "mahotas", "cv2", "bm3d", "sklearn", "polars", "dash") if m in sys.modules]
print(f"{target:48s} {dt:5.2f} s  {ok}  heavy={','.join(heavy) or '-'}")
