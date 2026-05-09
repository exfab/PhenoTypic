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
    # New aux-port representation: the op-typed marker is extracted to an aux
    # node and the consumer's ``aux_ports`` records the slot id.  The
    # consumer's ``params`` no longer carries the marker inline.
    assert candidate_param not in rebuilt_node.params
    assert candidate_param in rebuilt_node.aux_ports
    slot_ids = rebuilt_node.aux_ports[candidate_param]
    assert len(slot_ids) == 1 and slot_ids[0] is not None
    aux_id = slot_ids[0]
    aux_node = next(n for n in rebuilt_scope.aux_nodes if n.node_id == aux_id)
    assert aux_node.class_name == "OtsuDetector"
    assert aux_node.params["ignore_zeros"] is True
    assert aux_node.params["ignore_borders"] is False

    # Re-converting the rebuilt scope must produce an equivalent pipeline.
    re_pipeline = to_pipeline(rebuilt_scope)
    re_inner = getattr(re_pipeline.get_ops()[candidate_name], candidate_param)
    assert type(re_inner).__name__ == "OtsuDetector"
    assert re_inner.ignore_zeros is True


def test_filamentous_fungi_detector_extracts_to_aux_node() -> None:
    """``FilamentousFungiDetector`` with a custom inoculum_detector materialises as an aux wire."""

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

    # Aux representation: the inoculum_detector marker is extracted.
    assert "inoculum_detector" not in consumer.params
    assert "inoculum_detector" in consumer.aux_ports
    slots = consumer.aux_ports["inoculum_detector"]
    assert len(slots) == 1
    aux_id = slots[0]
    assert aux_id is not None

    assert len(rebuilt_scope.aux_nodes) == 1
    aux_node = rebuilt_scope.aux_nodes[0]
    assert aux_node.node_id == aux_id
    assert aux_node.class_name == "OtsuDetector"
    assert aux_node.params["ignore_zeros"] is True
    assert aux_node.params["ignore_borders"] is False

    # Round-trip: re-constructing the runtime pipeline must produce a
    # byte-identical canonical JSON.
    original_json = pipeline.to_json()
    rebuilt_pipeline = to_pipeline(rebuilt_scope)
    assert rebuilt_pipeline.to_json() == original_json


def test_composite_detector_extracts_two_aux_nodes() -> None:
    """``CompositeDetector`` with two detectors materialises two ordered aux wires."""

    from phenotypic.detect import CompositeDetector, OtsuDetector, RoundPeaksDetector

    detectors = [
        OtsuDetector(ignore_zeros=True, ignore_borders=False),
        RoundPeaksDetector(),
    ]
    composite = CompositeDetector(detectors=detectors, mode="union")
    pipeline = ImagePipeline(ops=[composite], name="composite_demo", desc="")

    rebuilt_scope = from_pipeline(pipeline)
    assert len(rebuilt_scope.nodes) == 1
    consumer = rebuilt_scope.nodes[0]
    assert consumer.class_name == "CompositeDetector"

    assert "detectors" not in consumer.params
    assert "detectors" in consumer.aux_ports
    slots = consumer.aux_ports["detectors"]
    assert len(slots) == 2
    assert all(s is not None for s in slots)

    assert len(rebuilt_scope.aux_nodes) == 2
    aux_by_id = {n.node_id: n for n in rebuilt_scope.aux_nodes}
    # Order in slots must match the original detectors list.
    first_aux = aux_by_id[slots[0]]  # type: ignore[index]
    second_aux = aux_by_id[slots[1]]  # type: ignore[index]
    assert first_aux.class_name == "OtsuDetector"
    assert first_aux.params["ignore_zeros"] is True
    assert first_aux.params["ignore_borders"] is False
    assert second_aux.class_name == "RoundPeaksDetector"

    # Round-trip: re-folded pipeline must canonicalise to the same JSON.
    original_json = pipeline.to_json()
    rebuilt_pipeline = to_pipeline(rebuilt_scope)
    assert rebuilt_pipeline.to_json() == original_json


def test_orphan_aux_node_is_dropped_on_save() -> None:
    """Aux nodes not wired to any consumer must not surface in the runtime pipeline."""

    main_a = StepNode(
        node_id="main_a",
        class_name="GaussianBlur",
        params={"sigma": 1.5},
        label="GaussianBlur",
    )
    main_b = StepNode(
        node_id="main_b",
        class_name="OtsuDetector",
        params={"ignore_zeros": False, "ignore_borders": False},
        label="OtsuDetector",
    )
    orphan = StepNode(
        node_id="orphan_aux",
        class_name="OtsuDetector",
        params={"ignore_zeros": True, "ignore_borders": True},
        label="OtsuDetector",
    )

    scope = BuilderScope(
        nodes=[main_a, main_b],
        aux_nodes=[orphan],
        name="orphan_demo",
    )

    pipeline = to_pipeline(scope)
    # Only the two main-ribbon ops should be present.
    op_classes = [type(op).__name__ for op in pipeline.get_ops().values()]
    assert op_classes == ["GaussianBlur", "OtsuDetector"]
    # The orphan's distinctive "ignore_zeros=True" must NOT appear in any
    # of the runtime ops.
    for op in pipeline.get_ops().values():
        if isinstance(op, ImagePipeline):
            continue
        # The main OtsuDetector has ignore_zeros=False; the orphan would
        # have leaked True if it were included.
        if hasattr(op, "ignore_zeros"):
            assert op.ignore_zeros is False


def test_aux_breadcrumb_walks_to_aux_scope() -> None:
    """An ``aux_id`` breadcrumb segment drills into the aux node's nested scope."""

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
        aux_ports={"inoculum_detector": ["aux_pipe"]},
        label="FilamentousFungiDetector",
    )

    root_scope = BuilderScope(
        nodes=[main_node],
        aux_nodes=[aux_pipeline_node],
        name="root",
    )

    state = BuilderState(
        root=root_scope,
        breadcrumb=[{"aux_id": "aux_pipe", "param": None}],
    )

    drilled = current_scope(state)
    assert drilled is inner_scope
    assert [n.class_name for n in drilled.nodes] == ["GaussianBlur"]


def test_state_json_back_compat_without_aux_fields() -> None:
    """Older JSON payloads (no aux_nodes / aux_ports) must rehydrate cleanly."""

    legacy_json = {
        "root": {
            "nodes": [
                {
                    "node_id": "a",
                    "class_name": "GaussianBlur",
                    "params": {"sigma": 1.0},
                    "label": "GaussianBlur",
                    "nested": None,
                    # NB: no ``aux_ports`` key
                },
            ],
            "name": "Pipeline",
            "desc": "",
            "nrows": None,
            "ncols": None,
            # NB: no ``aux_nodes`` key
        },
        "breadcrumb": [],
        "selected_node_id": None,
    }

    state = state_from_json(legacy_json)
    assert state.root.aux_nodes == []
    assert state.root.nodes[0].aux_ports == {}
    # Re-emitting the state must produce the new fields.
    out = state_to_json(state)
    assert out["root"]["aux_nodes"] == []
    assert out["root"]["nodes"][0]["aux_ports"] == {}


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
        rebuilt_scope = from_pipeline(once_pipeline)
        # FFD lives at index 2 (after StableDenoise, HomomorphicFilter).
        consumer = next(
            n for n in rebuilt_scope.nodes
            if n.class_name == "FilamentousFungiDetector"
        )
        assert "inoculum_detector" in consumer.aux_ports
        slots = consumer.aux_ports["inoculum_detector"]
        assert len(slots) == 1
        assert slots[0] is not None
        # The aux source for the inoculum_detector is an ImagePipeline
        # (the default constructed by FilamentousFungiPipeline).
        aux = next(
            a for a in rebuilt_scope.aux_nodes if a.node_id == slots[0]
        )
        assert aux.class_name == "ImagePipeline"
        # That nested pipeline contains InoculumDetector + GridSectionLargest.
        assert aux.nested is not None
        nested_classes = [n.class_name for n in aux.nested.nodes]
        assert "InoculumDetector" in nested_classes
        assert "GridSectionLargest" in nested_classes

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
