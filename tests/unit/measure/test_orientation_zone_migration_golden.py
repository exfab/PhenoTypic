"""Pre-simplification golden for the complete orientation-zone contract.

This fixture is intentionally broader than the older legacy-only SymZones
parquets. It freezes the public measurement tables, selected center, solver
status, and operation serialization before the orientation-zone orchestration
is simplified. The source images are deterministic synthetic arrays generated
below, so the golden contains behavior rather than duplicated pixel data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from phenotypic import Image
from phenotypic.abc_ import BaseOperation
from phenotypic.detect import HysteresisDetector
from phenotypic.measure import MeasureOrientationZones, MeasureSymZones

_GOLDEN_PATH = (
    Path(__file__).parent / "_golden" / "orientation_zones_pre_simplification.json"
)
_RTOL = 1e-10
_ATOL = 1e-10


def _radial_spoke_image() -> Image:
    """Return a deterministic core with supported radial branches."""
    size = 181
    center = (90, 90)
    rows, cols = np.indices((size, size), dtype=float)
    radius = np.hypot(rows - center[0], cols - center[1])
    angle = np.arctan2(rows - center[0], cols - center[1])
    mask = radius <= 12.0
    for spoke in np.linspace(-np.pi, np.pi, 24, endpoint=False):
        axial_delta = 0.5 * np.arctan2(
            np.sin(2.0 * (angle - spoke)),
            np.cos(2.0 * (angle - spoke)),
        )
        mask |= (radius <= 75.0) & (np.abs(axial_delta) <= 0.018)
    signal = np.zeros((size, size), dtype=np.float32)
    signal[mask] = 1.0 + radius[mask] / 75.0
    image = Image(np.repeat((signal / signal.max())[..., None], 3, axis=2))
    image.detect_mat[:] = signal
    image.objmap[:] = mask.astype(np.int32)
    return image


def _unsupported_disk_image() -> Image:
    """Return a filled disk with no resolvable branch crossings."""
    rows, cols = np.indices((81, 81), dtype=float)
    mask = np.hypot(rows - 40.0, cols - 40.0) <= 30.0
    signal = mask.astype(np.float32)
    image = Image(np.repeat(signal[..., None], 3, axis=2))
    image.detect_mat[:] = signal
    image.objmap[:] = mask.astype(np.int32)
    return image


def _off_center_core_image() -> Image:
    """Return one colony with a separately thresholdable compact center."""
    rows, cols = np.indices((121, 121), dtype=float)
    object_mask = np.hypot(rows - 60.0, cols - 60.0) <= 45.0
    core_mask = np.hypot(rows - 45.0, cols - 35.0) <= 5.0
    signal = np.zeros(object_mask.shape, dtype=np.float32)
    signal[object_mask] = 0.2
    signal[core_mask] = 1.0
    image = Image(np.repeat(signal[..., None], 3, axis=2))
    image.detect_mat[:] = signal
    image.objmap[:] = object_mask.astype(np.int32)
    return image


def _tiny_image() -> Image:
    """Return an object below the canonical ten-pixel minimum."""
    signal = np.zeros((9, 9), dtype=np.float32)
    signal[4, 3:6] = 1.0
    image = Image(np.repeat(signal[..., None], 3, axis=2))
    image.detect_mat[:] = signal
    image.objmap[:] = (signal > 0).astype(np.int32)
    return image


def _center_detector() -> HysteresisDetector:
    return HysteresisDetector(
        low=0.8,
        high=0.8,
        ignore_borders=False,
    )


def _json_scalar(value: Any) -> int | float | str | bool | None:
    """Convert a dataframe scalar to strict JSON without NaN tokens."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    """Recursively convert cached NumPy values to strict JSON values."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return _json_scalar(value)


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": list(frame.columns),
        "rows": [
            [_json_scalar(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def _case_snapshot(
    image: Image,
    *,
    symmetric_kwargs: dict[str, Any],
    orientation_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Measure one case and retain its output plus orchestration state."""
    symmetric = MeasureSymZones(**symmetric_kwargs)
    orientation = MeasureOrientationZones(
        include_diagnostics=True,
        **orientation_kwargs,
    )
    symmetric_frame = symmetric.measure(image)
    orientation_frame = orientation.measure(image)
    symmetric_record = symmetric._MeasureSymZones__cache_intermediates[1]
    orientation_record = orientation._cache[1]
    return {
        "symmetric": _frame_payload(symmetric_frame),
        "orientation": _frame_payload(orientation_frame),
        "center_global": list(symmetric_record.centroid_global),
        "orientation_center_global": list(
            orientation_record["centroid_global"]
        ),
        "solver": _json_ready(orientation_record["zone_resolution"]),
    }


def _current_snapshot() -> dict[str, Any]:
    """Calculate every behavior that the simplification must preserve."""
    exact_params = {
        "radial_ring_width": 4.0,
        "zone_minimum_segment": 2,
        "zone_min_crossings": 1,
    }
    collapsed_params = {
        "radial_ring_width": 8.0,
        "zone_minimum_segment": 4,
        "zone_min_crossings": 1,
    }
    missing_params = {"zone_minimum_segment": 20}
    center_params = {"center_detector": _center_detector()}
    legacy_params = {"legacy_mode": True}

    old_symmetric = BaseOperation.from_json(
        json.dumps({"class": "MeasureSymZones", "params": {}})
    )
    old_orientation = BaseOperation.from_json(
        json.dumps({"class": "MeasureOrientationZones", "params": {}})
    )
    configured = MeasureOrientationZones(
        center_detector=_center_detector(),
        outer_zone_percentile=95.0,
    )

    return {
        "format_version": 1,
        "cases": {
            "canonical_exact": _case_snapshot(
                _radial_spoke_image(),
                symmetric_kwargs=exact_params,
                orientation_kwargs=exact_params,
            ),
            "canonical_collapsed": _case_snapshot(
                _radial_spoke_image(),
                symmetric_kwargs=collapsed_params,
                orientation_kwargs=collapsed_params,
            ),
            "canonical_missing": _case_snapshot(
                _unsupported_disk_image(),
                symmetric_kwargs=missing_params,
                orientation_kwargs=missing_params,
            ),
            "canonical_center_detector": _case_snapshot(
                _off_center_core_image(),
                symmetric_kwargs=center_params,
                orientation_kwargs=center_params,
            ),
            "canonical_tiny": _case_snapshot(
                _tiny_image(),
                symmetric_kwargs={},
                orientation_kwargs={},
            ),
            "legacy": _case_snapshot(
                _radial_spoke_image(),
                symmetric_kwargs=legacy_params,
                orientation_kwargs=legacy_params,
            ),
        },
        "serialization": {
            "old_symmetric_migrated": json.loads(old_symmetric.to_json()),
            "old_orientation_migrated": json.loads(old_orientation.to_json()),
            "configured_roundtrip": json.loads(
                MeasureOrientationZones.from_json(configured.to_json()).to_json()
            ),
        },
    }


def _assert_frame_matches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert actual["columns"] == expected["columns"]
    actual_values = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in actual["rows"]
        ],
        dtype=np.float64,
    )
    expected_values = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in expected["rows"]
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=_RTOL,
        atol=_ATOL,
        equal_nan=True,
    )


def test_orientation_zone_pre_simplification_golden() -> None:
    assert _GOLDEN_PATH.exists(), (
        f"missing golden {_GOLDEN_PATH}; regenerate with "
        "PHENOTYPIC_CAPTURE_GOLDEN=1 uv run pytest "
        "tests/unit/measure/test_orientation_zone_migration_golden.py"
    )
    expected = json.loads(_GOLDEN_PATH.read_text())
    actual = _current_snapshot()

    assert actual["format_version"] == expected["format_version"]
    assert actual["serialization"] == expected["serialization"]
    assert actual["cases"].keys() == expected["cases"].keys()
    for name, expected_case in expected["cases"].items():
        actual_case = actual["cases"][name]
        _assert_frame_matches(actual_case["symmetric"], expected_case["symmetric"])
        _assert_frame_matches(
            actual_case["orientation"], expected_case["orientation"]
        )
        np.testing.assert_allclose(
            actual_case["center_global"],
            expected_case["center_global"],
            rtol=0.0,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            actual_case["orientation_center_global"],
            expected_case["orientation_center_global"],
            rtol=0.0,
            atol=_ATOL,
        )
        assert actual_case["solver"] == expected_case["solver"]


@pytest.mark.skipif(
    os.environ.get("PHENOTYPIC_CAPTURE_GOLDEN") != "1",
    reason="golden capture only runs when PHENOTYPIC_CAPTURE_GOLDEN=1",
)
def test_capture_orientation_zone_pre_simplification_golden() -> None:
    _GOLDEN_PATH.parent.mkdir(exist_ok=True)
    snapshot = _current_snapshot()
    _GOLDEN_PATH.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
