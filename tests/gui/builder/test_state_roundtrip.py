"""Round-trip tests for :mod:`phenotypic.gui.builder._state`.

These tests exercise the pure-Python state model defined for the Dash
pipeline builder.  They convert :class:`BuilderScope` instances to
:class:`~phenotypic.ImagePipeline` and back, asserting that class names,
labels, scalar params, nested pipelines, and operation-typed parameters all
survive the trip.
"""

from __future__ import annotations

import json

import pytest

from phenotypic import ImagePipeline
from phenotypic.gui._operation_registry import get_registry
from phenotypic.gui.builder._state import (
    BuilderScope,
    BuilderState,
    StepNode,
    current_scope,
    from_pipeline,
    stage_of,
    state_from_json,
    state_to_json,
    to_pipeline,
)


def test_flat_pipeline_roundtrip() -> None:
    """A linear ops/meas chain survives ``to_pipeline`` -> ``from_pipeline``."""

    registry = get_registry()
    for required in ("GaussianBlur", "OtsuDetector", "MeasureSize"):
        assert registry.get(required) is not None, (
            f"Expected '{required}' in the operation registry"
        )

    scope = BuilderScope(
        nodes=[
            StepNode(
                node_id="aaaa1111",
                class_name="GaussianBlur",
                params={"sigma": 2.0},
                label="GaussianBlur",
            ),
            StepNode(
                node_id="bbbb2222",
                class_name="OtsuDetector",
                params={"ignore_zeros": False, "ignore_borders": True},
                label="OtsuDetector",
            ),
            StepNode(
                node_id="cccc3333",
                class_name="MeasureSize",
                params={},
                label="MeasureSize",
            ),
        ],
        name="flat_demo",
        desc="three-step demo pipeline",
    )

    pipeline = to_pipeline(scope)

    assert isinstance(pipeline, ImagePipeline)
    assert pipeline.name == "flat_demo"
    assert list(pipeline.get_ops().keys()) == ["GaussianBlur", "OtsuDetector"]
    assert list(pipeline.get_meas().keys()) == ["MeasureSize"]
    assert pipeline.get_ops()["GaussianBlur"].sigma == 2.0
    assert pipeline.get_ops()["OtsuDetector"].ignore_borders is True

    # to_json round-trip should be JSON-serializable.
    json_str = pipeline.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "pipe_cfgs" in parsed and "meas" in parsed

    # from_pipeline should mirror the structure (with fresh node_ids and
    # canonical labels coming from the minted dict keys).
    rebuilt_scope = from_pipeline(pipeline)
    assert [n.class_name for n in rebuilt_scope.nodes] == [
        "GaussianBlur",
        "OtsuDetector",
        "MeasureSize",
    ]
    assert [n.label for n in rebuilt_scope.nodes] == [
        "GaussianBlur",
        "OtsuDetector",
        "MeasureSize",
    ]
    assert rebuilt_scope.nodes[0].params["sigma"] == 2.0
    assert rebuilt_scope.nodes[1].params["ignore_borders"] is True

    # state-level JSON round-trip
    state = BuilderState(root=scope)
    state_json = state_to_json(state)
    json.dumps(state_json)  # must be JSON-serializable in stdlib
    rehydrated = state_from_json(state_json)
    assert [n.class_name for n in rehydrated.root.nodes] == [
        "GaussianBlur",
        "OtsuDetector",
        "MeasureSize",
    ]
    assert rehydrated.root.nodes[0].params["sigma"] == 2.0


def test_nested_pipeline_roundtrip() -> None:
    """A nested ``ImagePipeline`` step survives a full round-trip."""

    inner_scope = BuilderScope(
        nodes=[
            StepNode(
                node_id="inner001",
                class_name="GaussianBlur",
                params={"sigma": 1.25},
                label="GaussianBlur",
            ),
            StepNode(
                node_id="inner002",
                class_name="OtsuDetector",
                params={"ignore_zeros": True, "ignore_borders": False},
                label="OtsuDetector",
            ),
        ],
        name="inner_pipe",
        desc="nested",
    )

    outer_scope = BuilderScope(
        nodes=[
            StepNode(
                node_id="outer001",
                class_name="ImagePipeline",
                params={},
                label="inner_pipe",
                nested=inner_scope,
            ),
            StepNode(
                node_id="outer002",
                class_name="MeasureSize",
                params={},
                label="MeasureSize",
            ),
        ],
        name="outer_pipe",
    )

    pipeline = to_pipeline(outer_scope)
    ops = pipeline.get_ops()
    # Outer ops dict should contain exactly one ImagePipeline-typed entry.
    assert len(ops) == 1
    only_op = next(iter(ops.values()))
    assert isinstance(only_op, ImagePipeline)
    assert list(only_op.get_ops().keys()) == ["GaussianBlur", "OtsuDetector"]
    assert only_op.get_ops()["GaussianBlur"].sigma == 1.25
    assert only_op.get_ops()["OtsuDetector"].ignore_zeros is True
    assert list(pipeline.get_meas().keys()) == ["MeasureSize"]

    # from_pipeline should keep the nested structure intact.
    rebuilt_scope = from_pipeline(pipeline)
    assert len(rebuilt_scope.nodes) == 2
    nested_node = rebuilt_scope.nodes[0]
    assert nested_node.class_name == "ImagePipeline"
    assert nested_node.nested is not None
    assert [n.class_name for n in nested_node.nested.nodes] == [
        "GaussianBlur",
        "OtsuDetector",
    ]
    assert nested_node.nested.nodes[0].params["sigma"] == 1.25

    # current_scope should be able to drill into the nested pipeline.
    state = BuilderState(root=rebuilt_scope, breadcrumb=[nested_node.node_id])
    drilled = current_scope(state)
    assert drilled is nested_node.nested

    # stage_of for ImagePipeline is "ops".
    assert stage_of("ImagePipeline") == "ops"
    assert stage_of("MeasureSize") == "meas"
    assert stage_of("GaussianBlur") == "ops"


def test_op_typed_param_roundtrip() -> None:
    """An op carrying an op-typed param survives a full round-trip."""

    registry = get_registry()

    # Find an op with an operation-typed parameter that has a non-pipeline
    # acceptable concrete class we can plug in.
    candidate_name = None
    candidate_param = None
    for op_name, info in registry.get_all().items():
        for param in info.parameters.values():
            if param.is_operation and param.is_optional:
                candidate_name = op_name
                candidate_param = param.name
                break
        if candidate_name is not None:
            break

    if candidate_name is None or candidate_param is None:
        pytest.skip(
            "no registered op exposes an Optional operation-typed parameter"
        )

    # Use OtsuDetector as a generic ObjectDetector substitute since the only
    # current candidate (FilamentousFungiDetector) accepts an ObjectDetector.
    inner_marker = {
        "__type__": "operation",
        "class_name": "OtsuDetector",
        "params": {"ignore_zeros": True, "ignore_borders": False},
    }

    scope = BuilderScope(
        nodes=[
            StepNode(
                node_id="ffd00001",
                class_name=candidate_name,
                params={candidate_param: inner_marker},
                label=candidate_name,
            ),
        ],
        name="op_param_demo",
    )

    pipeline = to_pipeline(scope)
    ops = pipeline.get_ops()
    assert candidate_name in ops
    outer_op = ops[candidate_name]
    inner_op = getattr(outer_op, candidate_param)
    assert type(inner_op).__name__ == "OtsuDetector"
    assert inner_op.ignore_zeros is True
    assert inner_op.ignore_borders is False

    rebuilt_scope = from_pipeline(pipeline)
    assert len(rebuilt_scope.nodes) == 1
    rebuilt_node = rebuilt_scope.nodes[0]
    assert rebuilt_node.class_name == candidate_name
    rebuilt_marker = rebuilt_node.params[candidate_param]
    assert isinstance(rebuilt_marker, dict)
    assert rebuilt_marker["__type__"] == "operation"
    assert rebuilt_marker["class_name"] == "OtsuDetector"
    assert rebuilt_marker["params"]["ignore_zeros"] is True
    assert rebuilt_marker["params"]["ignore_borders"] is False

    # Re-converting the rebuilt scope must produce an equivalent pipeline.
    re_pipeline = to_pipeline(rebuilt_scope)
    re_inner = getattr(re_pipeline.get_ops()[candidate_name], candidate_param)
    assert type(re_inner).__name__ == "OtsuDetector"
    assert re_inner.ignore_zeros is True


@pytest.mark.slow
def test_nested_pipeline_apply_with_intermediates() -> None:
    """1-deep nested pipeline runs end-to-end on the synthetic plate."""

    pytest.importorskip("phenotypic.data._synthetic_data")
    from phenotypic.data._synthetic_data import load_synth_yeast_plate

    inner_scope = BuilderScope(
        nodes=[
            StepNode(
                node_id="inner_a",
                class_name="GaussianBlur",
                params={"sigma": 1.0},
                label="GaussianBlur",
            ),
            StepNode(
                node_id="inner_b",
                class_name="OtsuDetector",
                params={},
                label="OtsuDetector",
            ),
        ],
        name="enhance_then_detect",
    )

    outer_scope = BuilderScope(
        nodes=[
            StepNode(
                node_id="outer_a",
                class_name="ImagePipeline",
                params={},
                label="enhance_then_detect",
                nested=inner_scope,
            ),
        ],
        name="end_to_end",
    )

    pipeline = to_pipeline(outer_scope)
    image = load_synth_yeast_plate()
    result = pipeline.apply_with_intermediates(image)
    assert result.image is not None
    assert len(result.intermediates) >= 1
