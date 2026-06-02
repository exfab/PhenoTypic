"""Smoke tests for JSON round-trip serialization of operations, pipelines, and prefab pipelines."""

import json
import tempfile
from pathlib import Path

import pytest

import phenotypic
from phenotypic import ImagePipeline
from phenotypic.abc_ import BaseOperation, ImageOperation
from phenotypic.correction import ImagePadder
from phenotypic.detect import CompositeDetector, OtsuDetector, TriangleDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.measure import MeasureShape
from phenotypic.prefab import HeavyOtsuPipeline
from phenotypic.refine import RemoveBorderObjects, SmallObjectRemover

from unit.test_fixtures import (
    walk_package_for_measurements,
    walk_package_for_operations,
)
from unit.resources.TestHelper import timeit

# ---------------------------------------------------------------------------
# Dynamic discovery of all concrete operations, measurements, and prefabs
# ---------------------------------------------------------------------------

# Operations that cannot be instantiated with an empty constructor
_SKIP_OPS = {"ColorCorrector", "ManualGridFinder", "GridApply"}

_all_operations = [
    (qualname, obj)
    for qualname, obj in walk_package_for_operations(phenotypic)
    if obj.__name__ not in _SKIP_OPS
]

_all_measurements = list(walk_package_for_measurements(phenotypic))

# Prefabs whose inner ops cannot be deserialized with empty constructors.
# Empty since the pydantic migration: ``GridApply`` is now a proper
# pydantic operation with an ``OperationField`` for its wrapped op, so
# ``GridSectionPipeline`` round-trips through JSON like every other prefab.
_SKIP_PREFABS: set[str] = set()

_all_prefabs = [
    pytest.param(
            f"phenotypic.prefab.{name}",
            getattr(phenotypic.prefab, name),
            marks=pytest.mark.xfail(
                    reason="GridApply requires mandatory image_op arg",
                    strict=True,
            ),
    )
    if name in _SKIP_PREFABS
    else (f"phenotypic.prefab.{name}", getattr(phenotypic.prefab, name))
    for name in phenotypic.prefab.__all__
]


# ---------------------------------------------------------------------------
# Parametrized: every operation survives round-trip
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", _all_operations)
@timeit
def test_operation_roundtrip(qualname, obj):
    """Every ImageOperation with default args survives JSON round-trip."""
    instance = obj()
    pipe = ImagePipeline(ops=[instance])

    loaded = ImagePipeline.from_json(pipe.to_json())

    loaded_op = list(loaded._ops.values())[0]
    assert type(loaded_op).__name__ == obj.__name__, (
        f"Class mismatch: expected {obj.__name__}, got {type(loaded_op).__name__}"
    )

    # Public attributes should survive the round-trip
    original_attrs = {
        k: v for k, v in instance.__dict__.items()
        if not k.startswith("_")
    }
    for key, expected in original_attrs.items():
        actual = getattr(loaded_op, key, _SENTINEL)
        assert actual is not _SENTINEL, (
            f"{obj.__name__}.{key} missing after round-trip"
        )
        try:
            json.dumps(expected)  # only check JSON-serializable attrs
        except (TypeError, ValueError):
            continue
        assert actual == expected, (
            f"{obj.__name__}.{key}: expected {expected!r}, got {actual!r}"
        )


_SENTINEL = object()


# ---------------------------------------------------------------------------
# Parametrized: every measurement survives round-trip
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", _all_measurements)
@timeit
def test_measurement_roundtrip(qualname, obj):
    """Every MeasureFeatures with default args survives JSON round-trip."""
    instance = obj()
    pipe = ImagePipeline(meas=[instance])

    loaded = ImagePipeline.from_json(pipe.to_json())

    loaded_meas = list(loaded._meas.values())[0]
    assert type(loaded_meas).__name__ == obj.__name__, (
        f"Class mismatch: expected {obj.__name__}, "
        f"got {type(loaded_meas).__name__}"
    )


# ---------------------------------------------------------------------------
# Parametrized: every prefab pipeline survives round-trip
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("qualname,cls", _all_prefabs)
@timeit
def test_prefab_roundtrip(qualname, cls):
    """Every PrefabPipeline with default args survives JSON round-trip."""
    original = cls()
    original_op_count = len(original._ops)
    original_meas_count = len(original._meas)

    loaded = cls.from_json(original.to_json())

    assert type(loaded).__name__ == cls.__name__
    assert len(loaded._ops) == original_op_count
    assert len(loaded._meas) == original_meas_count

    # Verify each inner op class name matches
    original_types = [type(o).__name__ for o in original._ops.values()]
    loaded_types = [type(o).__name__ for o in loaded._ops.values()]
    assert loaded_types == original_types


# ---------------------------------------------------------------------------
# Structural tests (hand-picked, covering nesting / file I/O / params)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@timeit
def test_operation_params_survive_roundtrip():
    """Operations from all ABC categories preserve params through JSON round-trip."""
    pipe = ImagePipeline(ops=[
        GaussianBlur(sigma=3.5, mode="reflect"),
        OtsuDetector(ignore_zeros=True, ignore_borders=False),
        RemoveBorderObjects(border_size=7),
        ImagePadder(left=10, right=20, top=5, bottom=15, mode="constant"),
    ])

    loaded = ImagePipeline.from_json(pipe.to_json())

    blur = loaded._ops["GaussianBlur"]
    assert blur.sigma == 3.5
    assert blur.mode == "reflect"

    det = loaded._ops["OtsuDetector"]
    assert det.ignore_zeros is True
    assert det.ignore_borders is False

    rem = loaded._ops["RemoveBorderObjects"]
    assert rem.border_size == 7

    pad = loaded._ops["ImagePadder"]
    assert pad.left == 10
    assert pad.right == 20
    assert pad.top == 5
    assert pad.bottom == 15
    assert pad.mode == "constant"


@pytest.mark.smoke
@timeit
def test_prefab_custom_params_roundtrip():
    """PrefabPipeline with non-default params preserves them through round-trip."""
    original = HeavyOtsuPipeline(gaussian_sigma=7, small_object_min_size=150)

    loaded = HeavyOtsuPipeline.from_json(original.to_json())

    first_op = list(loaded._ops.values())[0]
    assert isinstance(first_op, GaussianBlur)
    assert first_op.sigma == 7

    removers = [
        op for op in loaded._ops.values()
        if isinstance(op, SmallObjectRemover)
    ]
    assert all(r.min_size == 150 for r in removers)


@pytest.mark.smoke
@timeit
def test_pipeline_embedded_in_pipeline():
    """ImagePipeline nested inside another ImagePipeline survives round-trip."""
    inner = ImagePipeline(ops=[GaussianBlur(sigma=2), OtsuDetector()])
    outer = ImagePipeline(ops=[inner, SmallObjectRemover(min_size=30)])

    loaded = ImagePipeline.from_json(outer.to_json())
    ops = list(loaded._ops.values())

    assert len(ops) == 2
    assert isinstance(ops[0], ImagePipeline)
    assert isinstance(ops[1], SmallObjectRemover)

    inner_ops = list(ops[0]._ops.values())
    assert len(inner_ops) == 2
    assert isinstance(inner_ops[0], GaussianBlur)
    assert inner_ops[0].sigma == 2
    assert isinstance(inner_ops[1], OtsuDetector)


@pytest.mark.smoke
@timeit
def test_prefab_embedded_in_pipeline():
    """PrefabPipeline nested in a parent ImagePipeline preserves class and params."""
    prefab = HeavyOtsuPipeline(gaussian_sigma=5)
    outer = ImagePipeline(ops=[prefab, RemoveBorderObjects(border_size=12)])

    loaded = ImagePipeline.from_json(outer.to_json())
    ops = list(loaded._ops.values())

    assert len(ops) == 2
    assert type(ops[0]).__name__ == "HeavyOtsuPipeline"
    assert isinstance(ops[1], RemoveBorderObjects)
    assert ops[1].border_size == 12

    inner_first = list(ops[0]._ops.values())[0]
    assert isinstance(inner_first, GaussianBlur)
    assert inner_first.sigma == 5


@pytest.mark.smoke
@timeit
def test_file_roundtrip():
    """Pipeline serialization to file and back preserves structure."""
    pipe = ImagePipeline(
            ops=[GaussianBlur(sigma=4), OtsuDetector()],
            meas=[MeasureShape()],
            name="file_test",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "pipeline.json"
        pipe.to_json(filepath)

        assert filepath.exists()
        config = json.loads(filepath.read_text())
        assert "pipe_cfgs" in config
        assert "meas" in config

        loaded = ImagePipeline.from_json(filepath)
        assert loaded.name == "file_test"
        assert len(loaded._ops) == 2
        assert len(loaded._meas) == 1


# ---------------------------------------------------------------------------
# Operation-level to_json/from_json (BaseOperation)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@timeit
def test_operation_to_json_returns_string():
    """``to_json()`` with no filepath returns a ``{"class", "params"}`` envelope."""
    json_str = OtsuDetector(ignore_zeros=True).to_json()

    assert isinstance(json_str, str)
    envelope = json.loads(json_str)
    assert envelope["class"] == "OtsuDetector"
    assert isinstance(envelope["params"], dict)
    assert envelope["params"]["ignore_zeros"] is True


@pytest.mark.smoke
@timeit
def test_operation_to_json_from_json_file_roundtrip():
    """An operation written to file is recovered with its params intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "op.json"
        OtsuDetector(ignore_zeros=True).to_json(filepath)

        assert filepath.exists()
        loaded = OtsuDetector.from_json(filepath)
        assert isinstance(loaded, OtsuDetector)
        assert loaded.ignore_zeros is True


@pytest.mark.smoke
@timeit
def test_operation_from_json_accepts_string():
    """``from_json`` accepts a JSON string, not only a file path."""
    json_str = OtsuDetector(ignore_zeros=True).to_json()

    loaded = OtsuDetector.from_json(json_str)
    assert isinstance(loaded, OtsuDetector)
    assert loaded.ignore_zeros is True


@pytest.mark.smoke
@timeit
def test_operation_polymorphic_from_json_via_base():
    """``ImageOperation.from_json`` resolves the concrete subclass in the file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "op.json"
        OtsuDetector().to_json(filepath)

        loaded = ImageOperation.from_json(filepath)
        assert type(loaded).__name__ == "OtsuDetector"
        assert isinstance(loaded, ImageOperation)


@pytest.mark.smoke
@timeit
def test_operation_from_json_subclass_mismatch_raises():
    """Loading via a sibling subclass rejects a non-matching class."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "op.json"
        OtsuDetector().to_json(filepath)

        with pytest.raises(TypeError):
            TriangleDetector.from_json(filepath)


@pytest.mark.smoke
@timeit
def test_operation_from_json_measurement_via_image_op_raises():
    """A measurement file cannot be loaded through ``ImageOperation.from_json``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "meas.json"
        MeasureShape().to_json(filepath)

        # MeasureFeatures is a sibling of ImageOperation under BaseOperation.
        with pytest.raises(TypeError):
            ImageOperation.from_json(filepath)
        # ...but BaseOperation.from_json resolves it fine.
        assert type(BaseOperation.from_json(filepath)).__name__ == "MeasureShape"


@pytest.mark.smoke
@timeit
def test_nested_op_to_json_from_json_roundtrip():
    """A nested ``OperationField`` (CompositeDetector.detectors) round-trips."""
    composite = CompositeDetector(
            detectors=[OtsuDetector(ignore_zeros=True), TriangleDetector()],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "composite.json"
        composite.to_json(filepath)

        loaded = CompositeDetector.from_json(filepath)
        assert isinstance(loaded, CompositeDetector)
        assert len(loaded.detectors) == 2
        assert isinstance(loaded.detectors[0], OtsuDetector)
        assert loaded.detectors[0].ignore_zeros is True
        assert isinstance(loaded.detectors[1], TriangleDetector)
