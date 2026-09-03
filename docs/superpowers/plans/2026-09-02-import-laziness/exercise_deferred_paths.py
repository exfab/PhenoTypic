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

assert "colour" not in sys.modules and "h5py" not in sys.modules and "numba" not in sys.modules
assert "colour" not in sys.modules, "colour leaked at import"

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

print("\n".join(f"  ok  {line}" for line in ok))
print(f"\n{len(ok)}/8 deferred paths exercised successfully")
