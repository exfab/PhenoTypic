"""Golden fixture: the float64 kernel reproduces Gwyddion 2.71 (design.md §7.3).

The fixture pins TRANSCRIPTION; behavioural correctness is pinned by the kernel and operation
tests. Tolerances are derived from drift D1, not guessed: Gwyddion solves monomial normal
equations by Cholesky, so its own forward error scales with cond(Gram_monomial) * eps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic.enhance._poly_surface_kernels import flatten_surface, term_powers

FIXTURE = Path(__file__).parents[2] / "fixtures" / "enhance" / "subtract_poly_surface_gwyddion"
MANIFEST = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
with np.load(FIXTURE.with_suffix(".npz")) as _npz:
    ARRAYS = dict(_npz)
EPS = np.finfo(np.float64).eps


def _monomial_gram_condition(case: dict, height: int, width: int) -> float:
    if case["method"] == "polynomial":
        v, u = np.meshgrid(2 * np.arange(height) / (height - 1.0) - 1, 2 * np.arange(width) / (width - 1.0) - 1,
                           indexing="ij")
        a = np.stack([u.ravel() ** p * v.ravel() ** q
                      for p, q in term_powers(case["order"], case["independent"])], axis=1)
    elif case["method"] == "line":
        n = width if case["line_axis"] == "row" else height
        x = np.arange(n) - (n - 1) / 2.0                     # Gwyddion's centred pixel coordinate
        a = np.vander(x, case["line_order"] + 1, increasing=True)
    else:
        return 1.0
    return float(np.linalg.cond(a.T @ a))


def _tolerance(case: dict, z: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(z))))
    # offset/plane take the 1e-12 floor: their error model is closed-form (condition 1).
    return max(1e-12, 10 * EPS * scale * _monomial_gram_condition(case, *z.shape))


@pytest.mark.parametrize("case", MANIFEST["cases"], ids=[c["name"] for c in MANIFEST["cases"]])
def test_kernel_reproduces_the_oracle(case):
    z = ARRAYS[f"input__{case['input']}"]
    expected = ARRAYS[f"output__{case['name']}"]
    out = flatten_surface(
        z, method=case["method"], order=case["order"] or 3,
        independent=True if case["independent"] is None else case["independent"],
        line_order=0 if case["line_order"] is None else case["line_order"],
        line_axis=case["line_axis"] or "row", fit="lstsq", clip_sigma=3.0, max_iter=10)
    np.testing.assert_allclose(out, expected, rtol=0, atol=_tolerance(case, z))


def test_fixture_provenance_is_recorded():
    assert MANIFEST["kind"] in {"gwyddion-oracle", "contract-rederivation"}
    assert MANIFEST["gwyddion_version"] == "2.71"
    assert len(MANIFEST["cases"]) == 30
