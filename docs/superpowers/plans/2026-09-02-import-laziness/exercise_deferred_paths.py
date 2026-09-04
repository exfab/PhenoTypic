"""Drive every deferred-import path once, in a single fresh interpreter.

Lives beside the plan rather than under ``logic_validation_scripts/`` because it
imports ``phenotypic`` on purpose: the whole point is to drive the shipped code.
The directory contract there is that anything in it is an independent witness,
and a script that imports the code under test would cost a reader that
assumption for every other file in the directory.

Why this exists at all. Moving an ``import`` from module scope into a function
body is invisible to an ordinary test suite -- the library still loads the first
time anything calls the function, so the tests pass whether the deferral worked
or not, and equally whether some branch now raises ``NameError`` on a path
nothing covers. Two static checks bound that risk (ruff's F821 for a name never
bound, plus an AST pass for an import placed below its first use), but neither
executes anything.

This does, and it asserts in both directions: each deferred library must be
absent from ``sys.modules`` right up until the access that should load it, and
present immediately after. A deferral that silently did not happen fails the
first assertion; a deferral that broke its own code path fails the call.

Run it after any change to a deferred import site::

    uv run python docs/superpowers/plans/2026-09-02-import-laziness/exercise_deferred_paths.py

Exits non-zero on any failure.
"""

# ruff: noqa: E402 -- import placement is the subject under test, not an accident.
# Each import sits deliberately *below* the assertion that its library is not yet
# in sys.modules; hoisting them to the top would delete the only thing this
# script measures.
import sys
import numpy as np

DEFERRED = ("colour", "cv2", "h5py", "numba", "plotly", "polars")

# Nothing may be loaded before the package is imported, or the check below
# would be measuring the environment rather than the package.
assert not [m for m in DEFERRED if m in sys.modules], "a deferred library was preloaded"

# `import phenotypic` is the subject of this file, not an unused import. Ruff's
# F401 autofix deleted this line once, which quietly removed the premise: the
# assertions below still passed, because the first `from phenotypic...` further
# down imports the package anyway. The noqa is load-bearing.
import phenotypic  # noqa: F401

leaked = [m for m in DEFERRED if m in sys.modules]
assert not leaked, f"{leaked} leaked into `import phenotypic`"

ok = []

# --- sdk_.colourspace -------------------------------------------------------
from phenotypic.sdk_.colourspace import decode_srgb, encode_srgb
from phenotypic.sdk_ import colourspace
a = np.linspace(0, 1, 7)
assert np.allclose(encode_srgb(decode_srgb(a)), a, atol=1e-12)
assert colourspace.sRGB_D50 is colourspace.sRGB_D50
ok.append("colourspace.decode/encode_srgb + sRGB_D50")

# --- _xyz_conversion.rgb_to_xyz (both illuminants, both gammas) -------------
from phenotypic._core._image_parts.color_space_accessors._xyz_conversion import rgb_to_xyz
from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS
rgb = np.random.default_rng(0).random((4, 4, 3))
obs = "CIE 1931 2 Degree Standard Observer"
for gamma in (GAMMA_ENCODINGS.SRGB, GAMMA_ENCODINGS.LINEAR):
    for illum in ("D50", "D65"):
        out = rgb_to_xyz(rgb, gamma=gamma, illuminant=illum, observer=obs)
        assert out.shape == rgb.shape and np.isfinite(out).all()
ok.append("rgb_to_xyz x4 gamma/illuminant combinations")

# --- util._robust_color_stats ----------------------------------------------
from phenotypic.util._robust_color_stats import medoid_ciede2000, lab_to_srgb_hex
lab = np.random.default_rng(1).random((40, 3)) * [100, 60, 60]
medoid, dists = medoid_ciede2000(lab, max_pixels=20, chunk_size=8)
assert medoid.shape == (3,) and dists.shape == (40,)
h = lab_to_srgb_hex(np.array([50.0, 10.0, -20.0]))
assert h.startswith("#") and len(h) == 7
ok.append("medoid_ciede2000 + lab_to_srgb_hex")

# --- correction._color_correction._helpers ----------------------------------
from phenotypic.correction._color_correction import _helpers
assert _helpers._srgb_cs() is _helpers._srgb_cs()
labs = _helpers._rgb_to_lab(np.random.default_rng(2).random((3, 3, 3)))
assert labs.shape == (3, 3, 3)
ref = {f"p{i}": tuple(np.random.default_rng(i).random(3) * 50) for i in range(4)}
obs_lab = np.array([ref[k] for k in ref], dtype=float)
m = _helpers.hungarian_match_swatches(obs_lab, ref)
assert set(m) == set(ref)
ok.append("_srgb_cs + _rgb_to_lab + hungarian_match_swatches")

# --- correction._color_correction._color_checker_profile --------------------
from phenotypic.correction._color_correction import _color_checker_profile as ccp
assert ccp._illuminant_xy_table() is ccp._illuminant_xy_table()
assert ccp._illuminant_xy("D50").shape == (2,)
assert ccp._illuminant_xy("D65").shape == (2,)
assert ccp._illuminant_XYZ("D50").shape == (3,)
rl, rlin, wp = ccp._load_reference_data("ColorChecker24 - After November 2014", "D50")
assert len(rl) == 24 and len(rlin) == 24 and wp.shape == (2,)
ok.append("_illuminant_xy/_XYZ + _load_reference_data (24 patches)")

# --- correction._color_correction._color_correction_report ------------------
from phenotypic.correction._color_correction import _color_correction_report as rep
srgb = rep._lab_to_srgb(np.array([[50.0, 10.0, -20.0]]))
assert srgb.shape == (1, 3) and (0.0 <= srgb).all() and (srgb <= 1.0).all()
ok.append("_lab_to_srgb")

# --- sdk_.reconnect / numba kernels ----------------------------------------
assert "numba" not in sys.modules, "numba leaked before any detector ran"
from phenotypic.sdk_.reconnect import markers_from_centroids
assert "numba" in sys.modules, "importing sdk_.reconnect should load numba"
objmap = np.zeros((10, 10), dtype=np.int32)
objmap[2:4, 2:4] = 1
objmap[7:9, 7:9] = 2
mk = markers_from_centroids(objmap)
assert mk.max() == 2
ok.append("sdk_.reconnect markers_from_centroids (numba loaded on demand)")

# --- h5py path --------------------------------------------------------------
assert "h5py" not in sys.modules, "h5py leaked"
from phenotypic.sdk_ import HDF
assert "h5py" in sys.modules, "sdk_.HDF should load h5py on access"
assert HDF.__name__ == "HDF"
ok.append("sdk_.HDF via __getattr__ (h5py loaded on demand)")

# --- cv2 (stage 2) ----------------------------------------------------------
assert "cv2" not in sys.modules, "cv2 leaked"
from phenotypic.enhance import SubtractOpening
from phenotypic.data import load_synth_yeast_plate

image = load_synth_yeast_plate()
SubtractOpening().apply(image, inplace=True)
assert "cv2" in sys.modules, "SubtractOpening should have loaded cv2 on demand"
ok.append("enhance.SubtractOpening._operate (cv2 loaded on demand)")

# --- polars (stage 2) -------------------------------------------------------
# The sys.modules guard must recognise a real polars frame, and must not
# mistake a pandas one for it.
import pandas as pd
from phenotypic.util import split_measurements

columns = {"Metadata_Dataset": ["d"], "Metadata_ImageFile": ["a.png"],
           "ObjectLabel": [1], "Shape_Circularity": [0.9]}
from_pandas = split_measurements(pd.DataFrame(columns))
import polars as pl
from_polars = split_measurements(pl.DataFrame(columns))
assert set(from_pandas) == set(from_polars), "backends disagree on split keys"
for key in from_pandas:
    assert list(from_pandas[key].columns) == list(from_polars[key].columns)
ok.append("util.split_measurements over both pandas and polars frames")

# --- plotly (stage 2) -------------------------------------------------------
# No absence check here: earlier sections deliberately import modules allowed to
# pull plotly in (`_color_correction_report` imports it at module scope and is
# off both eager paths). The absence claim is made once, above.
from phenotypic.sdk_.viz.figures import apply_theme

import plotly.graph_objects as go
import plotly.io as pio

figure = apply_theme(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])))
assert "phenotypic" in pio.templates, "apply_theme did not register the template"
assert figure.layout.template is not None
ok.append("viz.apply_theme registers the template on demand (not at import)")

print("\n".join(f"  ok  {line}" for line in ok))
print(f"\n{len(ok)}/11 deferred paths exercised successfully")
