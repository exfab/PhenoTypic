"""Round-trip tests for :mod:`phenotypic.gui.builder._state`.

These tests exercise the pure-Python state model defined for the Dash
pipeline builder.  They convert :class:`BuilderScope` instances to
:class:`~phenotypic.ImagePipeline` and back, asserting that class names,
labels, scalar params, nested pipelines, and operation-typed parameters
(now modelled as embedded aux :class:`StepNode` instances in
``StepNode.aux_ports``) all survive the trip.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from phenotypic import ImagePipeline
from phenotypic.gui._operation_registry import get_registry
from phenotypic.gui.builder._state import (
    _LegacyBuilderScope as BuilderScope,
    _LegacyBuilderState as BuilderState,
    StepNode,
    current_scope,
    from_pipeline,
    stage_of,
    state_from_json,
    state_to_json,
    to_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _otsu_step(
        *,
        node_id: str = "otsu_aux",
        ignore_zeros: bool = True,
        ignore_borders: bool = False,
) -> StepNode:
    """Build an ``OtsuDetector`` :class:`StepNode` suitable for embedding."""

    return StepNode(
            node_id=node_id,
            class_name="OtsuDetector",
            params={"ignore_zeros": ignore_zeros, "ignore_borders": ignore_borders},
            label="OtsuDetector",
    )


def _round_peaks_step(*, node_id: str = "rp_aux") -> StepNode:
    """Build a ``RoundPeaksDetector`` :class:`StepNode` suitable for embedding."""

    return StepNode(
            node_id=node_id,
            class_name="RoundPeaksDetector",
            params={},
            label="RoundPeaksDetector",
    )


def _structural_fingerprint(scope: BuilderScope) -> Dict[str, Any]:
    """Snapshot the structural shape of a :class:`BuilderScope`.

    Drops random ``node_id`` values (re-minted on every round-trip via
    ``_new_node_id``) so two scopes can be compared regardless of the
    transient identifiers.
    """

    def _node(node: StepNode) -> Dict[str, Any]:
        aux: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        for port_name, slots in node.aux_ports.items():
            aux[port_name] = [
                _node(s) if s is not None else None for s in slots
            ]
        nested = _scope(node.nested) if node.nested is not None else None
        return {
            "class_name": node.class_name,
            "params"    : dict(node.params),
            "label"     : node.label,
            "nested"    : nested,
            "aux_ports" : aux,
        }

    def _scope(s: BuilderScope) -> Dict[str, Any]:
        return {
            "nodes": [_node(n) for n in s.nodes],
            "name" : s.name,
            "desc" : s.desc,
            "nrows": s.nrows,
            "ncols": s.ncols,
        }

    return _scope(scope)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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
    """An op carrying an op-typed param survives a full round-trip.

    Under the embedded-aux model, ``from_pipeline`` extracts the op-typed
    marker out of ``params`` and stores the resulting :class:`StepNode`
    inline at ``consumer.aux_ports[<param>][0]`` (a length-1 list because
    the port is scalar).
    """

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
        "__type__"  : "operation",
        "class_name": "OtsuDetector",
        "params"    : {"ignore_zeros": True, "ignore_borders": False},
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
    # Embedded-aux representation: the op-typed marker is extracted into
    # an aux ``StepNode`` stored inline at ``aux_ports[<param>][0]``.  The
    # consumer's ``params`` no longer carries the marker.
    assert candidate_param not in rebuilt_node.params
    assert candidate_param in rebuilt_node.aux_ports
    slots = rebuilt_node.aux_ports[candidate_param]
    assert len(slots) == 1 and slots[0] is not None
    aux_node = slots[0]
    assert isinstance(aux_node, StepNode)
    assert aux_node.class_name == "OtsuDetector"
    assert aux_node.params["ignore_zeros"] is True
    assert aux_node.params["ignore_borders"] is False

    # Re-converting the rebuilt scope must produce an equivalent pipeline.
    re_pipeline = to_pipeline(rebuilt_scope)
    re_inner = getattr(re_pipeline.get_ops()[candidate_name], candidate_param)
    assert type(re_inner).__name__ == "OtsuDetector"
    assert re_inner.ignore_zeros is True


def test_filamentous_fungi_detector_extracts_to_embedded_aux() -> None:
    """``FilamentousFungiDetector`` + custom inoculum_detector embeds inline."""

    pytest.importorskip("phenotypic.detect._filamentous_fungi_detector")
    from phenotypic.detect import FilamentousFungiDetector, OtsuDetector

    inoculum = OtsuDetector(ignore_zeros=True, ignore_borders=False)
    detector = FilamentousFungiDetector(inoculum_detector=inoculum)
    # ``desc=""`` matches the canonical shape ``from_pipeline`` reconstructs
    # (it normalises ``None`` desc to the empty string).
    pipeline = ImagePipeline(ops=[detector], name="ffd_demo", desc="")

    rebuilt_scope = from_pipeline(pipeline)
    assert len(rebuilt_scope.nodes) == 1
    consumer = rebuilt_scope.nodes[0]
    assert consumer.class_name == "FilamentousFungiDetector"

    # Embedded-aux representation: the inoculum_detector marker is extracted
    # into a StepNode stored inline at aux_ports["inoculum_detector"][0].
    assert "inoculum_detector" not in consumer.params
    assert "inoculum_detector" in consumer.aux_ports
    slots = consumer.aux_ports["inoculum_detector"]
    assert len(slots) == 1
    aux_node = slots[0]
    assert aux_node is not None
    assert isinstance(aux_node, StepNode)
    assert aux_node.class_name == "OtsuDetector"
    assert aux_node.params["ignore_zeros"] is True
    assert aux_node.params["ignore_borders"] is False

    # Round-trip: re-constructing the runtime pipeline must produce a
    # byte-identical canonical JSON.
    original_json = pipeline.to_json()
    rebuilt_pipeline = to_pipeline(rebuilt_scope)
    assert rebuilt_pipeline.to_json() == original_json


def test_composite_detector_extracts_two_embedded_aux_nodes() -> None:
    """``CompositeDetector`` with two detectors materialises two embedded aux wires."""

    from phenotypic.detect import CompositeDetector, OtsuDetector, RoundPeaksDetector

    detectors = [
        OtsuDetector(ignore_zeros=True, ignore_borders=False),
        RoundPeaksDetector(),
    ]
    composite = CompositeDetector(ops=detectors, mode="union")
    pipeline = ImagePipeline(ops=[composite], name="composite_demo", desc="")

    rebuilt_scope = from_pipeline(pipeline)
    assert len(rebuilt_scope.nodes) == 1
    consumer = rebuilt_scope.nodes[0]
    assert consumer.class_name == "CompositeDetector"

    # Embedded-aux representation: detectors are extracted inline.
    assert "ops" not in consumer.params
    assert "ops" in consumer.aux_ports
    slots = consumer.aux_ports["ops"]
    assert len(slots) == 2
    assert all(s is not None for s in slots)
    assert all(isinstance(s, StepNode) for s in slots)

    # Order in slots must match the original detectors list.
    first_aux = slots[0]
    second_aux = slots[1]
    assert first_aux is not None and second_aux is not None
    assert first_aux.class_name == "OtsuDetector"
    assert first_aux.params["ignore_zeros"] is True
    assert first_aux.params["ignore_borders"] is False
    assert second_aux.class_name == "RoundPeaksDetector"

    # Round-trip: re-folded pipeline must canonicalise to the same JSON.
    original_json = pipeline.to_json()
    rebuilt_pipeline = to_pipeline(rebuilt_scope)
    assert rebuilt_pipeline.to_json() == original_json


def test_composite_detector_with_mixed_wired_and_empty_slots() -> None:
    """List-typed aux with one empty slot drops the empty on serialization.

    The runtime ``CompositeDetector.ops`` cannot represent an empty
    slot — it just gets a 2-element list when 2 of 3 slots are wired.  On
    ``from_pipeline``, the reconstructed scope has 2 slots (not 3): the
    original empty slot does not survive a round-trip through
    ``pipeline.json``.  This is expected behaviour — empty slots are an
    edit-time-only artefact.
    """

    otsu_aux = _otsu_step(node_id="otsu_a", ignore_zeros=True)
    round_peaks_aux = _round_peaks_step(node_id="rp_a")

    scope = BuilderScope(
            nodes=[
                StepNode(
                        node_id="comp_main",
                        class_name="CompositeDetector",
                        params={"mode": "union"},
                        label="CompositeDetector",
                        aux_ports={
                            # 3 slots: wired, empty, wired.
                            "ops": [otsu_aux, None, round_peaks_aux],
                        },
                ),
            ],
            name="mixed_slots_demo",
            desc="",
    )

    pipeline = to_pipeline(scope)
    composite = next(iter(pipeline.get_ops().values()))
    # The empty slot is dropped: runtime detectors is 2-element.
    assert hasattr(composite, "ops")
    assert len(composite.ops) == 2
    assert type(composite.ops[0]).__name__ == "OtsuDetector"
    assert type(composite.ops[1]).__name__ == "RoundPeaksDetector"

    # Round-trip back: the empty slot is gone.
    rebuilt_scope = from_pipeline(pipeline)
    assert len(rebuilt_scope.nodes) == 1
    consumer = rebuilt_scope.nodes[0]
    slots = consumer.aux_ports["ops"]
    assert len(slots) == 2  # ← the original 3 with one None becomes 2
    assert slots[0] is not None and slots[0].class_name == "OtsuDetector"
    assert slots[1] is not None and slots[1].class_name == "RoundPeaksDetector"


def test_embedded_aux_serialization_via_state_to_json() -> None:
    """An embedded aux ``StepNode`` survives ``state_to_json`` / ``state_from_json``.

    Builds a ``FilamentousFungiDetector`` wired to an ``OtsuDetector`` aux,
    converts to ``ImagePipeline`` via ``to_pipeline``, checks the
    resulting JSON, then converts back with ``from_pipeline`` and asserts
    the round-tripped state structure matches the original (modulo
    randomised ``node_id`` values).
    """

    otsu_aux = _otsu_step(node_id="otsu_aux", ignore_zeros=True)
    scope = BuilderScope(
            nodes=[
                StepNode(
                        node_id="ffd_main",
                        class_name="FilamentousFungiDetector",
                        params={},
                        label="FilamentousFungiDetector",
                        aux_ports={"inoculum_detector": [otsu_aux]},
                ),
            ],
            name="embedded_aux_demo",
            desc="",
    )

    pipeline = to_pipeline(scope)
    pipeline_json = pipeline.to_json()
    parsed = json.loads(pipeline_json)
    # The OtsuDetector should appear nested somewhere inside the FFD's
    # pipe_cfgs (it is the value of the FFD's inoculum_detector parameter).
    assert "OtsuDetector" in pipeline_json
    assert "FilamentousFungiDetector" in pipeline_json
    assert "pipe_cfgs" in parsed

    # Round-trip via from_pipeline.
    rebuilt_scope = from_pipeline(pipeline)
    assert len(rebuilt_scope.nodes) == 1
    rebuilt_ffd = rebuilt_scope.nodes[0]
    assert rebuilt_ffd.class_name == "FilamentousFungiDetector"
    assert "inoculum_detector" in rebuilt_ffd.aux_ports
    rebuilt_slots = rebuilt_ffd.aux_ports["inoculum_detector"]
    assert len(rebuilt_slots) == 1
    rebuilt_aux = rebuilt_slots[0]
    assert rebuilt_aux is not None
    assert rebuilt_aux.class_name == "OtsuDetector"
    assert rebuilt_aux.params["ignore_zeros"] is True
    assert rebuilt_aux.params["ignore_borders"] is False

    # The runtime pipeline → from_pipeline path materialises every
    # registry-known parameter (including ones the user left at default),
    # so the rebuilt FFD's ``params`` is a superset of the original.
    # Compare only the structural aspects we control: class_name, label,
    # aux_ports topology.  The aux subtree, where the user did set values,
    # must match exactly.
    assert rebuilt_ffd.label == "FilamentousFungiDetector"
    assert set(rebuilt_ffd.aux_ports.keys()) == {"inoculum_detector"}
    # Idempotence: re-converting the rebuilt scope is a fixed point.
    twice = from_pipeline(to_pipeline(rebuilt_scope))
    assert _structural_fingerprint(rebuilt_scope) == _structural_fingerprint(twice)


def test_recursive_aux_three_levels_deep() -> None:
    """Aux-of-aux-of-aux round-trips byte-identically through pipeline.json."""

    pytest.importorskip("phenotypic.detect._filamentous_fungi_detector")

    # Deepest level: a plain OtsuDetector.
    deepest_aux = _otsu_step(
            node_id="deep_aux", ignore_zeros=True, ignore_borders=False
    )

    # Middle level: a FilamentousFungiDetector whose inoculum_detector is
    # the deepest OtsuDetector aux.
    middle_aux = StepNode(
            node_id="mid_aux",
            class_name="FilamentousFungiDetector",
            params={},
            label="FilamentousFungiDetector",
            aux_ports={"inoculum_detector": [deepest_aux]},
    )

    # Outer level: another FilamentousFungiDetector whose inoculum_detector
    # is the middle FilamentousFungiDetector aux.
    outer_consumer = StepNode(
            node_id="outer_ffd",
            class_name="FilamentousFungiDetector",
            params={},
            label="FilamentousFungiDetector",
            aux_ports={"inoculum_detector": [middle_aux]},
    )

    scope = BuilderScope(
            nodes=[outer_consumer],
            name="recursive_aux_demo",
            desc="",
    )

    # 1) Build the runtime pipeline.
    p1 = to_pipeline(scope)
    json1 = p1.to_json()

    # 2) Round-trip through pipeline.json: ImagePipeline.from_json → from_pipeline.
    p2 = ImagePipeline.from_json(json1)
    json2 = p2.to_json()
    assert json2 == json1, (
        "ImagePipeline.from_json followed by to_json must be byte-identical."
    )

    rebuilt_scope = from_pipeline(p2)
    p3 = to_pipeline(rebuilt_scope)
    json3 = p3.to_json()
    assert json3 == json1, (
        "Round-trip through from_pipeline / to_pipeline must be byte-identical."
    )

    # Verify the recursive structure survived.
    assert len(rebuilt_scope.nodes) == 1
    outer = rebuilt_scope.nodes[0]
    assert outer.class_name == "FilamentousFungiDetector"
    outer_slots = outer.aux_ports["inoculum_detector"]
    assert len(outer_slots) == 1
    middle = outer_slots[0]
    assert middle is not None
    assert middle.class_name == "FilamentousFungiDetector"
    middle_slots = middle.aux_ports["inoculum_detector"]
    assert len(middle_slots) == 1
    deepest = middle_slots[0]
    assert deepest is not None
    assert deepest.class_name == "OtsuDetector"
    assert deepest.params["ignore_zeros"] is True
    assert deepest.params["ignore_borders"] is False


def test_inspector_focus_aux_roundtrip() -> None:
    """``BuilderState.inspector_focus_aux`` survives ``state_to_json`` / ``state_from_json``.

    The field's shape is ``{"target_node_id": str, "param": str, "slot": int}``
    or ``None`` (the unfocused default).  A round-trip through JSON must
    preserve either form.
    """

    # Case 1: explicitly None.
    state_a = BuilderState(
            root=BuilderScope(name="root"),
            inspector_focus_aux=None,
    )
    json_a = state_to_json(state_a)
    assert "inspector_focus_aux" in json_a
    assert json_a["inspector_focus_aux"] is None
    rebuilt_a = state_from_json(json_a)
    assert rebuilt_a.inspector_focus_aux is None

    # Case 2: a real focus dict.
    focus = {
        "target_node_id": "consumer_abc",
        "param"         : "inoculum_detector",
        "slot"          : 0,
    }
    state_b = BuilderState(
            root=BuilderScope(name="root"),
            inspector_focus_aux=dict(focus),
    )
    json_b = state_to_json(state_b)
    assert json_b["inspector_focus_aux"] == focus
    # Must survive a JSON encode/decode boundary (dcc.Store contract).
    json_b_str = json.dumps(json_b)
    json_b_round = json.loads(json_b_str)
    rebuilt_b = state_from_json(json_b_round)
    assert rebuilt_b.inspector_focus_aux == focus


def test_breadcrumb_aux_slot_segment_roundtrips() -> None:
    """The new ``{"target_node_id", "param", "slot"}`` segment survives JSON round-trip."""

    otsu_aux = _otsu_step(node_id="otsu_aux", ignore_zeros=True)
    main_node = StepNode(
            node_id="ffd_main",
            class_name="FilamentousFungiDetector",
            params={},
            label="FilamentousFungiDetector",
            aux_ports={"inoculum_detector": [otsu_aux]},
    )
    root = BuilderScope(nodes=[main_node], name="root")

    seg = {
        "target_node_id": "ffd_main",
        "param"         : "inoculum_detector",
        "slot"          : 0,
    }
    state = BuilderState(root=root, breadcrumb=[seg])

    # state_to_json must preserve the aux-slot segment verbatim.
    out = state_to_json(state)
    assert out["breadcrumb"] == [seg]

    # Round-trip via JSON encode/decode and state_from_json.
    serialised = json.dumps(out)
    deserialised = json.loads(serialised)
    rebuilt = state_from_json(deserialised)
    assert len(rebuilt.breadcrumb) == 1
    rebuilt_seg = rebuilt.breadcrumb[0]
    assert rebuilt_seg["target_node_id"] == "ffd_main"
    assert rebuilt_seg["param"] == "inoculum_detector"
    assert rebuilt_seg["slot"] == 0

    # current_scope should follow the aux-slot segment into a 1-step scope.
    drilled = current_scope(rebuilt)
    assert isinstance(drilled, BuilderScope)
    assert len(drilled.nodes) == 1
    assert drilled.nodes[0].class_name == "OtsuDetector"


def test_aux_breadcrumb_walks_into_pipeline_aux_scope() -> None:
    """An aux-slot breadcrumb segment drills into a pipeline aux's nested scope."""

    inner_scope = BuilderScope(
            nodes=[
                StepNode(
                        node_id="aux_inner",
                        class_name="GaussianBlur",
                        params={"sigma": 0.5},
                        label="GaussianBlur",
                ),
            ],
            name="aux_inner_pipeline",
    )

    aux_pipeline_node = StepNode(
            node_id="aux_pipe",
            class_name="ImagePipeline",
            params={},
            label="ImagePipeline",
            nested=inner_scope,
    )

    main_node = StepNode(
            node_id="main_a",
            class_name="FilamentousFungiDetector",
            params={},
            aux_ports={"inoculum_detector": [aux_pipeline_node]},
            label="FilamentousFungiDetector",
    )

    root_scope = BuilderScope(nodes=[main_node], name="root")

    state = BuilderState(
            root=root_scope,
            breadcrumb=[
                {
                    "target_node_id": "main_a",
                    "param"         : "inoculum_detector",
                    "slot"          : 0,
                }
            ],
    )

    drilled = current_scope(state)
    # An ImagePipeline aux drill descends into its inner scope directly.
    assert drilled is inner_scope
    assert [n.class_name for n in drilled.nodes] == ["GaussianBlur"]


def test_state_json_back_compat_without_aux_fields() -> None:
    """Older JSON payloads (no ``aux_ports`` / ``inspector_focus_aux``) rehydrate cleanly."""

    legacy_json = {
        "root"            : {
            "nodes": [
                {
                    "node_id"   : "a",
                    "class_name": "GaussianBlur",
                    "params"    : {"sigma": 1.0},
                    "label"     : "GaussianBlur",
                    "nested"    : None,
                    # NB: no ``aux_ports`` key
                },
            ],
            "name" : "Pipeline",
            "desc" : "",
            "nrows": None,
            "ncols": None,
            # NB: no ``aux_nodes`` key (was removed)
        },
        "breadcrumb"      : [],
        "selected_node_id": None,
        # NB: no ``inspector_focus_aux`` key
    }

    state = state_from_json(legacy_json)
    assert state.root.nodes[0].aux_ports == {}
    assert state.inspector_focus_aux is None
    # Re-emitting the state must produce the new fields.
    out = state_to_json(state)
    assert out["root"]["nodes"][0]["aux_ports"] == {}
    assert out["inspector_focus_aux"] is None
    # The deprecated ``aux_nodes`` key must not appear.
    assert "aux_nodes" not in out["root"]


class TestPrefabRoundTrip:
    """Smoke-test real prefab pipelines through ``from_pipeline`` / ``to_pipeline``.

    Backwards-compat sanity: a prefab built from real op classes (not
    test fixtures) must round-trip cleanly through the builder state
    representation.  Of the seven prefabs in :mod:`phenotypic.prefab`,
    only :class:`~phenotypic.prefab.FilamentousFungiPipeline` exercises
    op-typed parameters (its ``inoculum_detector`` slot accepts an
    ``ObjectDetector | ImagePipeline``); the other six (``HeavyOtsu``,
    ``HeavyWatershed``, ``HeavyRoundPeaks``, ``RoundPeaks``,
    ``GridSection``, ``SpImager``) are linear chains of scalar-param
    ops and trivially round-trip.

    A naive "byte-identical" assertion against the prefab's
    ``to_json()`` does NOT hold: ``from_pipeline`` normalizes
    ``ImagePipeline._desc=None`` to ``""`` (so the inspector never has
    to deal with two empty-string sentinels), and the prefab's
    docstring-derived ``.desc`` property differs from its stored
    ``_desc``.  Instead, these tests assert the meaningful invariant:
    **idempotence after one normalization pass.**  Specifically,
    ``to_pipeline(from_pipeline(p))`` produces a pipeline whose JSON is
    byte-identical to ``to_pipeline(from_pipeline(to_pipeline(from_pipeline(p))))``.
    The first pass normalizes; subsequent passes are fixed points.
    """

    def test_filamentous_fungi_prefab_is_idempotent_after_normalization(self) -> None:
        """``FilamentousFungiPipeline`` (the only prefab with op-typed params)."""

        from phenotypic.prefab import FilamentousFungiPipeline

        prefab = FilamentousFungiPipeline()

        # First pass: round-trip the prefab. This normalizes desc=None
        # to "" and assigns fresh aux node ids.
        once_pipeline = to_pipeline(from_pipeline(prefab))
        once_json = once_pipeline.to_json()

        # Second pass: round-trip again and assert byte-identical JSON.
        twice_pipeline = to_pipeline(from_pipeline(once_pipeline))
        twice_json = twice_pipeline.to_json()

        assert once_json == twice_json, (
            "Prefab round-trip is not idempotent — the second pass produced "
            "different JSON than the first."
        )

        # Sanity: the nested aux structure is preserved across the round-trip.
        # The aux node is now embedded inline under ``aux_ports`` instead of
        # living in a separate ``aux_nodes`` list.
        rebuilt_scope = from_pipeline(once_pipeline)
        consumer = next(
                n for n in rebuilt_scope.nodes
                if n.class_name == "FilamentousFungiDetector"
        )
        assert "inoculum_detector" in consumer.aux_ports
        slots = consumer.aux_ports["inoculum_detector"]
        assert len(slots) == 1
        aux = slots[0]
        assert aux is not None
        # The aux source for the inoculum_detector is an ImagePipeline
        # (the default constructed by FilamentousFungiPipeline).
        assert aux.class_name == "ImagePipeline"
        # That nested pipeline contains InoculumDetector + KeepSectionLargest.
        assert aux.nested is not None
        nested_classes = [n.class_name for n in aux.nested.nodes]
        assert "InoculumDetector" in nested_classes
        assert "KeepSectionLargest" in nested_classes

    @pytest.mark.parametrize(
            "prefab_name",
            [
                "HeavyOtsuPipeline",
                "HeavyWatershedPipeline",
                "RoundPeaksPipeline",
            ],
    )
    def test_scalar_param_prefab_round_trips_byte_identical(
            self, prefab_name: str
    ) -> None:
        """Prefabs without op-typed params must round-trip byte-identically.

        These prefabs have ``desc=None`` like FilamentousFungiPipeline,
        so the same normalization applies — but unlike FFD they don't
        have an ``inoculum_detector`` shaped aux to fold/unfold, so the
        first pass normalizes desc and the second pass is a fixed
        point.  Asserting byte-identity across two passes is the
        canonical way to confirm round-trip stability for the bulk of
        the prefab catalog.
        """

        import phenotypic.prefab as prefab_module

        prefab_cls = getattr(prefab_module, prefab_name)
        prefab = prefab_cls()

        once_pipeline = to_pipeline(from_pipeline(prefab))
        once_json = once_pipeline.to_json()

        twice_pipeline = to_pipeline(from_pipeline(once_pipeline))
        twice_json = twice_pipeline.to_json()

        assert once_json == twice_json, (
            f"{prefab_name}: round-trip is not idempotent — the second "
            f"pass produced different JSON than the first."
        )

        # Confirm every op in the original prefab survives — class names
        # and order must match.
        rebuilt_scope = from_pipeline(once_pipeline)
        original_op_classes = [type(o).__name__ for o in prefab.get_ops().values()]
        rebuilt_op_classes = [
            n.class_name for n in rebuilt_scope.nodes
            if n.class_name in original_op_classes
        ]
        assert rebuilt_op_classes == original_op_classes, (
            f"{prefab_name}: op classes drifted across round-trip."
        )


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
