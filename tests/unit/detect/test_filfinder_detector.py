"""Focused contract tests for the FilFinder 1.8 raster adapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Any
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic import Image
from phenotypic.detect._filfinder_detector import (
    EXPECTED_SUPPLIED_MASK_WARNING,
    FilFinderDetector,
    _WarningForwardingProcessPool,
    _copy_float32_source,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ORACLE_PATH = (
    _REPOSITORY_ROOT / "tests/fixtures/reconnect/filfinder/oracle.json"
)
_ORACLE = json.loads(_ORACLE_PATH.read_text(encoding="utf-8"))
_HAS_REAL_FILFINDER = importlib.util.find_spec("fil_finder") is not None


@dataclass(frozen=True)
class _FakeQuantity:
    """Minimal pixel quantity used to verify unit-bearing calls."""

    value: float
    unit: str = "pix"


class _FakePixelUnit:
    """Return a tagged quantity for ``value * u.pix``."""

    def __rmul__(self, value: float) -> _FakeQuantity:
        return _FakeQuantity(float(value))


class _RecordingPool:
    """Record lifecycle calls without starting a process."""

    def __init__(self) -> None:
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class _FakeFilFinder2D:
    """Source-shaped spy exposing every wrapper-visible attribute."""

    instances: list[_FakeFilFinder2D] = []
    fail_constructor = False
    fail_analysis = False
    emit_create_warnings = False

    def __init__(
        self,
        image: np.ndarray,
        *,
        beamwidth: _FakeQuantity,
        mask: np.ndarray,
        pool: _RecordingPool,
    ) -> None:
        if type(self).fail_constructor:
            raise RuntimeError("injected constructor failure")
        self.constructor_image = image
        self.constructor_mask = mask
        self.beamwidth = beamwidth
        self.pool = pool
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.mask = mask.copy()
        self.skeleton = np.zeros_like(mask, dtype=bool)
        self.skeleton[0, 0] = True
        self.skeleton[1, 1] = True
        self.skeleton[-1, -1] = True
        self.skeleton_longpath = np.zeros_like(mask, dtype=bool)
        self.skeleton_longpath[0, 0] = True
        self.skeleton_longpath[1, 1] = True
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.fail_constructor = False
        cls.fail_analysis = False
        cls.emit_create_warnings = False

    def create_mask(self, *, use_existing_mask: bool) -> None:
        self.calls.append(
            ("create_mask", {"use_existing_mask": use_existing_mask})
        )
        if type(self).emit_create_warnings:
            warnings.warn(EXPECTED_SUPPLIED_MASK_WARNING, UserWarning)
            warnings.warn(EXPECTED_SUPPLIED_MASK_WARNING, RuntimeWarning)
            warnings.warn(
                "A10 nonmatching warning remains visible", UserWarning
            )

    def medskel(self, *, rng: int) -> None:
        self.calls.append(("medskel", {"rng": rng}))

    def analyze_skeletons(self, **kwargs: object) -> None:
        self.calls.append(("analyze_skeletons", kwargs))
        if type(self).fail_analysis:
            raise RuntimeError("injected analysis failure")


@pytest.fixture
def fake_filfinder_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_RecordingPool]:
    """Install source-shaped optional modules and lifecycle-recording pools."""
    _FakeFilFinder2D.reset()
    fil_finder = types.ModuleType("fil_finder")
    fil_finder.FilFinder2D = _FakeFilFinder2D  # type: ignore[attr-defined]
    astropy = types.ModuleType("astropy")
    astropy.__path__ = []  # type: ignore[attr-defined]
    units = types.ModuleType("astropy.units")
    units.pix = _FakePixelUnit()  # type: ignore[attr-defined]
    astropy.units = units  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fil_finder", fil_finder)
    monkeypatch.setitem(sys.modules, "astropy", astropy)
    monkeypatch.setitem(sys.modules, "astropy.units", units)

    pools: list[_RecordingPool] = []

    def create_pool() -> _RecordingPool:
        pool = _RecordingPool()
        pools.append(pool)
        return pool

    target = importlib.import_module("phenotypic.detect._filfinder_detector")
    monkeypatch.setattr(target, "_create_warning_forwarding_pool", create_pool)
    return pools


def _threshold_image() -> Image:
    """Return float32 data covering threshold polarity and NaN."""
    array = np.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.0, 0.0, 0.0],
            [0.0, 0.0, np.nan, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.49, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    image = Image(np.zeros(array.shape + (3,), dtype=np.uint8))
    image._data.detect_mat = array.copy()
    return image


def _emit_worker_warning(value: int) -> int:
    """Picklable worker callable for keyed parent warning forwarding."""
    warnings.warn("A10 child warning", UserWarning)
    return value


def test_module_import_is_optional_dependency_free() -> None:
    """Importing and constructing the private operation imports no extra."""
    code = """
import sys
from phenotypic.detect._filfinder_detector import FilFinderDetector
FilFinderDetector().model_json_schema()
assert 'fil_finder' not in sys.modules
assert 'astropy' not in sys.modules
assert 'astropy.units' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_defaults_schema_and_serialized_fields() -> None:
    """The keyword-only contract remains machine-readable."""
    detector = FilFinderDetector()
    assert detector.model_dump() == {
        "threshold": 0.5,
        "output": "mask",
        "beamwidth_px": 1.0,
        "prune_criteria": "all",
        "relative_intensity_threshold": 0.2,
        "branch_threshold_px": None,
        "max_prune_iterations": 10,
        "rng_seed": 0,
    }
    assert set(detector.model_json_schema()["properties"]) == set(
        detector.model_dump()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", -0.1),
        ("threshold", 1.1),
        ("threshold", np.nan),
        ("threshold", np.inf),
        ("beamwidth_px", 0.0),
        ("beamwidth_px", np.inf),
        ("relative_intensity_threshold", 0.0),
        ("relative_intensity_threshold", 1.1),
        ("relative_intensity_threshold", np.nan),
        ("branch_threshold_px", 0.0),
        ("branch_threshold_px", np.inf),
        ("max_prune_iterations", 0),
        ("max_prune_iterations", True),
        ("rng_seed", -1),
        ("rng_seed", False),
        ("output", "graph"),
        ("prune_criteria", "unknown"),
    ],
)
def test_invalid_parameters_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        FilFinderDetector(**{field: value})


def test_empty_mask_short_circuits_before_optional_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fil_finder", None)
    monkeypatch.setitem(sys.modules, "astropy", None)
    image = Image(np.zeros((4, 6), dtype=np.float32))
    result = FilFinderDetector(threshold=0.5, output="longest_path").apply(
        image
    )
    np.testing.assert_array_equal(result.objmap[:], 0)
    np.testing.assert_array_equal(result.objmask[:], False)


def test_nonempty_mask_reports_missing_topology_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fil_finder", None)
    monkeypatch.setitem(sys.modules, "astropy", None)
    with pytest.raises(ImportError, match="topology"):
        FilFinderDetector().apply(_threshold_image())


@pytest.mark.parametrize(
    ("output", "expected_names"),
    [
        ("mask", ["create_mask"]),
        ("skeleton", ["create_mask", "medskel"]),
        ("longest_path", ["create_mask", "medskel", "analyze_skeletons"]),
    ],
)
def test_exact_stage_graph_and_pool_shutdown(
    fake_filfinder_runtime: list[_RecordingPool],
    output: str,
    expected_names: list[str],
) -> None:
    detector = FilFinderDetector(
        output=output,
        rng_seed=17,
        prune_criteria="length",
        relative_intensity_threshold=0.4,
        branch_threshold_px=3.5,
        max_prune_iterations=7,
    )
    detector.apply(_threshold_image())
    source = _FakeFilFinder2D.instances[-1]
    assert [name for name, _ in source.calls] == expected_names
    assert fake_filfinder_runtime[-1].shutdown_calls == [(True, False)]


def test_source_arguments_are_copied_float64_and_unit_bearing(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    image = _threshold_image()
    original_rgb = image.rgb[:].copy()
    original_gray = image.gray[:].copy()
    original_detect = image.detect_mat[:].copy()
    FilFinderDetector(
        output="longest_path",
        beamwidth_px=2.25,
        prune_criteria="intensity",
        relative_intensity_threshold=0.35,
        branch_threshold_px=4.5,
        max_prune_iterations=13,
        rng_seed=23,
    ).apply(image, inplace=True)
    source = _FakeFilFinder2D.instances[-1]
    np.testing.assert_array_equal(source.constructor_image, original_detect)
    assert source.constructor_image.dtype == np.float64
    assert not np.shares_memory(source.constructor_image, image.detect_mat[:])
    assert source.constructor_mask.dtype == np.bool_
    assert not np.shares_memory(source.constructor_mask, image.detect_mat[:])
    assert source.beamwidth == _FakeQuantity(2.25)
    assert source.calls == [
        ("create_mask", {"use_existing_mask": True}),
        ("medskel", {"rng": 23}),
        (
            "analyze_skeletons",
            {
                "prune_criteria": "intensity",
                "relintens_thresh": 0.35,
                "skel_thresh": _FakeQuantity(1.0),
                "branch_thresh": _FakeQuantity(4.5),
                "max_prune_iter": 13,
            },
        ),
    ]
    np.testing.assert_array_equal(image.rgb[:], original_rgb)
    np.testing.assert_array_equal(image.gray[:], original_gray)
    np.testing.assert_array_equal(image.detect_mat[:], original_detect)
    assert fake_filfinder_runtime[-1].shutdown_calls == [(True, False)]


def test_image_data_float32_quantization_precedes_threshold(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    native_predecessor = np.nextafter(np.float64(0.5), np.float64(0.0))
    image = Image(np.zeros((3, 3), dtype=np.uint8))
    values = np.zeros((3, 3), dtype=np.float64)
    values[1, 1] = native_predecessor
    image._data.detect_mat = values
    assert image.detect_mat.dtype == np.float32
    assert image.detect_mat[:][1, 1] == np.float32(0.5)
    result = FilFinderDetector(threshold=0.5, output="mask").apply(image)
    assert result.objmap[:][1, 1] == 1
    assert _FakeFilFinder2D.instances[-1].constructor_image.dtype == np.float64
    assert fake_filfinder_runtime[-1].shutdown_calls == [(True, False)]


def test_private_source_copy_kills_direct_float64_threshold_mutant() -> None:
    native_predecessor = np.nextafter(np.float64(0.5), np.float64(0.0))
    copied = _copy_float32_source(np.array([[native_predecessor]]))
    direct_float64 = np.array(
        [[native_predecessor]], dtype=np.float64, copy=True
    )
    assert copied.dtype == np.float64
    assert copied[0, 0] == 0.5
    assert bool((copied >= 0.5)[0, 0])
    assert not bool((direct_float64 >= 0.5)[0, 0])


def test_none_branch_threshold_is_forwarded_without_quantity(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    FilFinderDetector(output="longest_path", branch_threshold_px=None).apply(
        _threshold_image()
    )
    analysis = _FakeFilFinder2D.instances[-1].calls[-1][1]
    assert analysis["branch_thresh"] is None
    assert fake_filfinder_runtime[-1].shutdown_calls == [(True, False)]


def test_selected_rasters_use_eight_connectivity_and_row_major_labels(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    skeleton = FilFinderDetector(output="skeleton").apply(_threshold_image())
    expected_skeleton = np.zeros((5, 5), dtype=np.int32)
    expected_skeleton[0, 0] = 1
    expected_skeleton[1, 1] = 1
    expected_skeleton[4, 4] = 2
    np.testing.assert_array_equal(skeleton.objmap[:], expected_skeleton)
    np.testing.assert_array_equal(skeleton.objmask[:], expected_skeleton > 0)

    longest = FilFinderDetector(output="longest_path").apply(
        _threshold_image()
    )
    expected_longest = np.zeros((5, 5), dtype=np.int32)
    expected_longest[0, 0] = 1
    expected_longest[1, 1] = 1
    np.testing.assert_array_equal(longest.objmap[:], expected_longest)
    np.testing.assert_array_equal(longest.objmask[:], expected_longest > 0)


def test_inclusive_threshold_and_nan_polarity(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    detected = FilFinderDetector(threshold=0.5, output="mask").apply(
        _threshold_image()
    )
    expected = np.zeros((5, 5), dtype=np.int32)
    expected[0, 0] = 1
    expected[1, 1] = 1
    expected[4, 4] = 2
    np.testing.assert_array_equal(detected.objmap[:], expected)
    assert fake_filfinder_runtime[-1].shutdown_calls == [(True, False)]


def test_downstream_fields_are_inactive_for_earlier_outputs(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    image = _threshold_image()
    baseline_mask = (
        FilFinderDetector(output="mask").apply(image).objmap[:].copy()
    )
    changed_mask = (
        FilFinderDetector(
            output="mask",
            prune_criteria="length",
            relative_intensity_threshold=0.9,
            branch_threshold_px=99.0,
            max_prune_iterations=1,
            rng_seed=999,
        )
        .apply(image)
        .objmap[:]
    )
    np.testing.assert_array_equal(changed_mask, baseline_mask)
    baseline_skeleton = (
        FilFinderDetector(output="skeleton").apply(image).objmap[:]
    )
    changed_skeleton = (
        FilFinderDetector(
            output="skeleton",
            prune_criteria="intensity",
            relative_intensity_threshold=0.8,
            branch_threshold_px=88.0,
            max_prune_iterations=2,
        )
        .apply(image)
        .objmap[:]
    )
    np.testing.assert_array_equal(changed_skeleton, baseline_skeleton)


def test_only_exact_supplied_mask_warning_is_suppressed(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    _FakeFilFinder2D.emit_create_warnings = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FilFinderDetector(output="mask").apply(_threshold_image())
    assert [str(item.message) for item in caught] == [
        EXPECTED_SUPPLIED_MASK_WARNING,
        "A10 nonmatching warning remains visible",
    ]
    assert [item.category for item in caught] == [RuntimeWarning, UserWarning]
    assert fake_filfinder_runtime[-1].shutdown_calls == [(True, False)]


@pytest.mark.parametrize("failure_site", ["constructor", "analysis"])
def test_pool_shutdown_is_guaranteed_after_failure(
    fake_filfinder_runtime: list[_RecordingPool], failure_site: str
) -> None:
    if failure_site == "constructor":
        _FakeFilFinder2D.fail_constructor = True
        detector = FilFinderDetector(output="mask")
    else:
        _FakeFilFinder2D.fail_analysis = True
        detector = FilFinderDetector(output="longest_path")
    with pytest.raises(RuntimeError, match="injected"):
        detector.apply(_threshold_image())
    assert fake_filfinder_runtime[-1].shutdown_calls == [(True, False)]


def test_fresh_source_object_and_pool_per_apply(
    fake_filfinder_runtime: list[_RecordingPool],
) -> None:
    detector = FilFinderDetector(output="skeleton")
    detector.apply(_threshold_image())
    detector.apply(_threshold_image())
    assert len(_FakeFilFinder2D.instances) == 2
    assert _FakeFilFinder2D.instances[0] is not _FakeFilFinder2D.instances[1]
    assert len(fake_filfinder_runtime) == 2
    assert fake_filfinder_runtime[0] is not fake_filfinder_runtime[1]
    assert all(
        pool.shutdown_calls == [(True, False)]
        for pool in fake_filfinder_runtime
    )


def test_real_process_pool_forwards_keyed_child_warning_to_parent() -> None:
    with _WarningForwardingProcessPool(max_workers=1) as pool:
        with pytest.warns(UserWarning, match="A10 child warning"):
            assert pool.submit(_emit_worker_warning, 41).result() == 41
        record = pool.warning_records_by_task[0]
        assert record["task_index"] == 0
        assert record["function"] == "_emit_worker_warning"
        records = record["warnings"]
        assert isinstance(records, list)
        assert records[0][0] == "A10 child warning"  # type: ignore[index]


@pytest.mark.skipif(
    not _HAS_REAL_FILFINDER,
    reason="requires the pinned topology test runtime",
)
@pytest.mark.parametrize("output", ["mask", "skeleton", "longest_path"])
@pytest.mark.parametrize(
    "case",
    _ORACLE["cases"],
    ids=lambda case: case["name"],
)
def test_real_filfinder_matches_all_24_selected_oracle_outputs(
    case: dict[str, Any], output: str
) -> None:
    detector = FilFinderDetector(
        threshold=0.5,
        output=output,
        beamwidth_px=1.0,
        prune_criteria="all",
        relative_intensity_threshold=0.2,
        branch_threshold_px=None,
        max_prune_iterations=10,
        rng_seed=0,
    )
    source = np.asarray(case["image"], dtype=np.float64)
    image = Image(np.zeros(source.shape, dtype=np.uint8))
    image._data.detect_mat = source.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detected = detector.apply(image)

    expected_key = {
        "mask": "mask_labels_8_connected",
        "skeleton": "skeleton_labels_8_connected",
        "longest_path": "longest_path_labels_8_connected",
    }[output]
    expected_value = case[expected_key]
    expected = (
        np.zeros_like(np.asarray(case["threshold_mask"]), dtype=np.int32)
        if expected_value is None
        else np.asarray(expected_value, dtype=np.int32)
    )
    np.testing.assert_array_equal(detected.objmap[:], expected)
    np.testing.assert_array_equal(detected.objmask[:], expected > 0)

    messages = Counter(str(item.message) for item in caught)
    assert messages[EXPECTED_SUPPLIED_MASK_WARNING] == 0
    if output == "longest_path":
        expected_worker_messages: Counter[str] = Counter()
        for task in case["analyze_skeleton_worker_warning_records"]:
            for record in task["warnings"]:
                expected_worker_messages[record["message"]] += record["count"]
        for message, count in expected_worker_messages.items():
            assert messages[message] == count
