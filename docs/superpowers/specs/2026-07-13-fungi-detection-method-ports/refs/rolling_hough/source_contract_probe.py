"""Run exact Clark RHT and FilFinder source probes for the A09 G0 gate.

This harness imports the pinned source files directly. The Clark source predates NumPy 2,
so ``np.int`` and ``np.float`` are restored as compatibility aliases. Astropy is stubbed
because the probed numerical functions do not use FITS I/O. No algorithm statement is
reimplemented here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np


REFERENCE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def _assert_zero_ulp_float64(observed: float, expected: float, *, label: str) -> None:
    """Require exact float64 bits for a calibrated pinned-source probe."""

    observed_value = np.float64(observed)
    expected_value = np.float64(expected)
    observed_bits = int(observed_value.view(np.uint64))
    expected_bits = int(expected_value.view(np.uint64))
    if observed_bits != expected_bits:
        raise AssertionError(
            f"{label}: expected zero-ULP drift, observed bits "
            f"0x{observed_bits:016x} versus 0x{expected_bits:016x}"
        )


def _load_source_module(name: str, path: Path) -> Any:
    """Load a source module from an exact local path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_clark_source() -> Any:
    """Load the pinned Clark module with import-only compatibility shims."""
    np.int = int  # type: ignore[attr-defined]
    np.float = float  # type: ignore[attr-defined]

    astropy = types.ModuleType("astropy")
    astropy_io = types.ModuleType("astropy.io")
    astropy_fits = types.ModuleType("astropy.io.fits")
    astropy_io.fits = astropy_fits  # type: ignore[attr-defined]
    astropy.io = astropy_io  # type: ignore[attr-defined]
    sys.modules["astropy"] = astropy
    sys.modules["astropy.io"] = astropy_io
    sys.modules["astropy.io.fits"] = astropy_fits

    return _load_source_module(
        "pinned_clark_rht",
        REFERENCE_DIR / "source_clark" / "rht" / "rht.py",
    )


def _probe_clark(clark: Any) -> dict[str, Any]:
    """Probe Clark's exact discretization, axes, equality, and undefined angle."""
    window_diameter = 11
    ntheta = clark.ntheta_w(window_diameter)
    theta = np.linspace(0.0, np.pi, ntheta, endpoint=False)
    xyt = clark.all_thetas(window_diameter, theta, True)
    support = clark.fast_hough(clark.circ_kern(window_diameter), xyt)
    expected_support = np.array(
        [11, 9, 9, 9, 11, 13, 7, 11, 13, 9, 9, 11, 11, 9, 9, 13, 11, 7, 13, 11, 9, 9, 9]
    )
    np.testing.assert_array_equal(support, expected_support)

    radius = window_diameter // 2
    horizontal = np.zeros((window_diameter, window_diameter), dtype=int)
    horizontal[radius, :] = 1
    vertical = np.zeros_like(horizontal)
    vertical[:, radius] = 1
    diagonal = np.eye(window_diameter, dtype=int)

    summaries: dict[str, Any] = {}
    for name, template, expected_peaks, expected_angle in (
        ("horizontal", horizontal, [11, 12], np.pi / 2.0),
        ("vertical", vertical, [0], np.pi),
        ("diagonal", diagonal, [17, 18], 0.8053847952589468),
    ):
        counts = clark.fast_hough(template, xyt)
        peaks = np.flatnonzero(counts == counts.max())
        np.testing.assert_array_equal(peaks, expected_peaks)
        source_angle = clark.theta_rht(counts.astype(float), True)
        _assert_zero_ulp_float64(
            source_angle, expected_angle, label=f"Clark {name} source angle"
        )
        summaries[name] = {
            "peak_bins": peaks.tolist(),
            "source_angle": float(source_angle),
        }

    equality_residual = np.true_divide(support, support) - 1.0
    equality_residual *= np.greater_equal(equality_residual, 0.0)
    assert not np.any(equality_residual)

    zero_angle = clark.theta_rht(np.zeros(ntheta), True)
    np.testing.assert_allclose(zero_angle, np.pi, rtol=0.0, atol=0.0)

    constant_bitmask = clark.umask(np.ones((21, 21), dtype=float), radius=2)
    assert not np.any(constant_bitmask)

    return {
        "window_diameter": window_diameter,
        "ntheta": int(ntheta),
        "circle_pixel_count": int(clark.circ_kern(window_diameter).sum()),
        "orientation_dependent_line_support": support.astype(int).tolist(),
        "templates": summaries,
        "exact_threshold_has_positive_output": bool(np.any(equality_residual)),
        "zero_weight_source_angle": float(zero_angle),
        "constant_bitmask_on_pixels": int(constant_bitmask.sum()),
    }


def _probe_filfinder(filfinder: Any) -> dict[str, Any]:
    """Probe the stable FilFinder v1.8 modified-RHT output contract."""
    summaries: dict[str, Any] = {}
    for name in ("horizontal", "vertical", "diagonal"):
        mask = np.zeros((25, 25), dtype=bool)
        if name == "horizontal":
            mask[12, 5:20] = True
            expected_mean = -np.pi / 2.0
        elif name == "vertical":
            mask[5:20, 12] = True
            expected_mean = 0.0
        else:
            rows = np.arange(5, 20)
            mask[rows, rows] = True
            expected_mean = 0.8037521352916115

        theta, response, quantiles = filfinder.rht(mask, radius=5, ntheta=18)
        assert theta.shape == (17,)
        assert response.shape == (17, 1)
        _assert_zero_ulp_float64(
            quantiles[1], expected_mean, label=f"FilFinder {name} mean angle"
        )
        summaries[name] = {
            "theta_count_after_endpoint_drop": int(theta.size),
            "response_sum": float(response.sum()),
            "mean_angle": float(quantiles[1]),
        }
    return summaries


def probe_pinned_sources() -> None:
    """Run both source probes and print their verified summaries."""
    clark = _load_clark_source()
    filfinder = _load_source_module(
        "pinned_filfinder_rht",
        REFERENCE_DIR / "source_filfinder" / "fil_finder" / "rollinghough.py",
    )
    report = {
        "clark_rht_4d06f9f": _probe_clark(clark),
        "filfinder_v1.8_22539cf": _probe_filfinder(filfinder),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    probe_pinned_sources()
