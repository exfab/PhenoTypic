"""Unit tests for the DAG ↔ ImagePipeline conversion layer.

Covers spec §5.4 — both halves of the conversion:

* :func:`to_pipeline_dag` topologically walks image-flow edges, folds
  aux edges into ``params``, partitions by stage, and either returns
  an :class:`ImagePipeline` or raises :class:`ValueError` on any
  blocking validation issue.
* :func:`from_pipeline_dag` mints :class:`BlockNode`s and edges from an
  existing pipeline, deep-clones shared aux instances, and queues an
  info toast when cloning happens.

Tests use the spec's reference fixture set under
``tests/fixtures/builder_dag/`` AND hand-built dataclass instances for
the edge cases where authoring JSON is more awkward than authoring
Python (forks, cycles, shared instances).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, FilamentousFungiDetector, OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui._operation_registry import ParamInfo, get_registry
from phenotypic.gui.builder._conversion_dag import (
    from_pipeline_dag,
    to_pipeline_dag,
)
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    Edge,
    _DagBuilderScope as BuilderScope,
    _DagBuilderState as BuilderState,
    _new_block_id,
)
from phenotypic.measure import MeasureSize


FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "builder_dag"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    class_name: str,
    *,
    label: str | None = None,
    params: Dict[str, Any] | None = None,
    nested: BuilderScope | None = None,
    list_slot_counts: Dict[str, int] | None = None,
) -> BlockNode:
    """Build a :class:`BlockNode` with sane defaults for tests."""

    return BlockNode(
        block_id=_new_block_id(),
        class_name=class_name,
        params=dict(params or {}),
        label=label,
        nested=nested,
        list_slot_counts=dict(list_slot_counts or {}),
    )


def _make_image_edge(source: BlockNode, target: BlockNode) -> Edge:
    return Edge(
        edge_id=_new_block_id(),
        source_block_id=source.block_id,
        source_port="out",
        target_block_id=target.block_id,
        target_port="in",
        target_slot=None,
        kind="image",
    )


def _make_aux_edge(
    source: BlockNode,
    target: BlockNode,
    port: str,
    slot: int | None = None,
) -> Edge:
    return Edge(
        edge_id=_new_block_id(),
        source_block_id=source.block_id,
        source_port="out",
        target_block_id=target.block_id,
        target_port=port,
        target_slot=slot,
        kind="aux",
    )


def _scope_with_chain(*blocks: BlockNode) -> BuilderScope:
    """Build a scope whose ``InputImage`` flows through *blocks* in order."""

    scope = BuilderScope()
    input_block = scope.blocks[0]
    scope.blocks.extend(blocks)
    prev = input_block
    for block in blocks:
        scope.edges.append(_make_image_edge(prev, block))
        prev = block
    return scope


def _load_fixture(name: str) -> Dict[str, Any]:
    with (FIXTURES_DIR / name).open("r") as fh:
        return json.load(fh)


def _state_from_fixture(fixture: Dict[str, Any]) -> BuilderState:
    """Materialise a fixture dict into a DAG :class:`BuilderState`.

    The fixture schema mirrors ``state_to_json`` output for the DAG
    builder: ``{"root": <scope>, ...}``.
    """

    return _build_state(fixture)


def _build_state(data: Dict[str, Any]) -> BuilderState:
    root = _build_scope(data["root"])
    state = BuilderState(root=root)
    return state


def _build_scope(data: Dict[str, Any]) -> BuilderScope:
    # Construct without auto-seeding so blocks roundtrip unchanged.
    scope = BuilderScope.__new__(BuilderScope)
    scope.blocks = [_build_block(b) for b in data.get("blocks", [])]
    scope.edges = [_build_edge(e) for e in data.get("edges", [])]
    scope.name = data.get("name", "Pipeline")
    scope.desc = data.get("desc", "")
    scope.nrows = data.get("nrows")
    scope.ncols = data.get("ncols")
    return scope


def _build_block(data: Dict[str, Any]) -> BlockNode:
    nested_data = data.get("nested")
    nested = _build_scope(nested_data) if nested_data is not None else None
    return BlockNode(
        block_id=data["block_id"],
        class_name=data["class_name"],
        params=dict(data.get("params") or {}),
        label=data.get("label"),
        nested=nested,
        collapsed=bool(data.get("collapsed", False)),
        list_slot_counts=dict(data.get("list_slot_counts") or {}),
    )


def _build_edge(data: Dict[str, Any]) -> Edge:
    return Edge(
        edge_id=data["edge_id"],
        source_block_id=data["source_block_id"],
        source_port=data.get("source_port", "out"),
        target_block_id=data["target_block_id"],
        target_port=data["target_port"],
        target_slot=data.get("target_slot"),
        kind=data.get("kind", "image"),
    )


# ---------------------------------------------------------------------------
# to_pipeline_dag
# ---------------------------------------------------------------------------


class TestToPipelineDagBasic:
    """Empty + linear chain materialisation."""

    def test_to_pipeline_dag_empty_scope_returns_empty_pipeline(self):
        state = BuilderState()
        # Validation will report no issues; conversion yields a pipeline
        # whose ops/meas/post are empty.
        pipeline = to_pipeline_dag(state)
        assert isinstance(pipeline, ImagePipeline)
        assert pipeline.get_ops() == {}
        assert pipeline.get_meas() == {}
        assert pipeline.get_post() == {}

    def test_to_pipeline_dag_linear_chain_partitions_correctly(self):
        blur = _make_block("GaussianBlur", params={"sigma": 1.5})
        otsu = _make_block("OtsuDetector")
        meas = _make_block("MeasureSize")
        scope = _scope_with_chain(blur, otsu, meas)
        state = BuilderState(root=scope)

        pipeline = to_pipeline_dag(state)
        ops = list(pipeline.get_ops().values())
        meas_list = list(pipeline.get_meas().values())
        post_list = list(pipeline.get_post().values())

        assert len(ops) == 2
        assert isinstance(ops[0], GaussianBlur)
        assert isinstance(ops[1], OtsuDetector)
        assert len(meas_list) == 1
        assert isinstance(meas_list[0], MeasureSize)
        assert post_list == []

    def test_to_pipeline_dag_preserves_root_scope_metadata(self):
        blur = _make_block("GaussianBlur")
        scope = _scope_with_chain(blur)
        scope.name = "my-pipe"
        scope.desc = "test desc"
        scope.nrows = 8
        scope.ncols = 12
        state = BuilderState(root=scope)

        pipeline = to_pipeline_dag(state)
        assert pipeline.name == "my-pipe"
        assert pipeline._desc == "test desc"
        assert pipeline.nrows == 8
        assert pipeline.ncols == 12


class TestToPipelineDagAuxFolding:
    """Aux edge folding into consumer params."""

    def test_to_pipeline_dag_folds_scalar_aux_into_params(self):
        """Spec §5.4 step 4 — scalar aux at ``params[port]`` (from JSON fixture)."""

        state = _state_from_fixture(_load_fixture("scalar_aux.json"))

        pipeline = to_pipeline_dag(state)
        ops = list(pipeline.get_ops().values())
        assert len(ops) == 1
        consumer_inst = ops[0]
        assert isinstance(consumer_inst, FilamentousFungiDetector)
        assert isinstance(consumer_inst.inoculum_detector, OtsuDetector)

    def test_to_pipeline_dag_folds_list_aux_with_empty_slot(self):
        """Empty slot (slot 1) emits ``None`` (from JSON fixture)."""

        state = _state_from_fixture(_load_fixture("list_aux_with_empty_slot.json"))

        pipeline = to_pipeline_dag(state)
        ops = list(pipeline.get_ops().values())
        assert len(ops) == 1
        consumer_inst = ops[0]
        assert isinstance(consumer_inst, CompositeDetector)
        assert isinstance(consumer_inst.detectors, list)
        assert len(consumer_inst.detectors) == 3
        assert isinstance(consumer_inst.detectors[0], OtsuDetector)
        assert consumer_inst.detectors[1] is None
        assert isinstance(consumer_inst.detectors[2], OtsuDetector)


class TestToPipelineDagRaises:
    """Every blocking validation rule raises ``ValueError``."""

    def test_to_pipeline_dag_raises_on_fork(self):
        blur = _make_block("GaussianBlur")
        otsu_a = _make_block("OtsuDetector")
        otsu_b = _make_block("OtsuDetector")

        scope = BuilderScope()
        scope.blocks.extend([blur, otsu_a, otsu_b])
        scope.edges.append(_make_image_edge(scope.blocks[0], blur))
        # Fork: GaussianBlur drives both detectors via image-flow.
        scope.edges.append(_make_image_edge(blur, otsu_a))
        scope.edges.append(_make_image_edge(blur, otsu_b))
        state = BuilderState(root=scope)

        with pytest.raises(ValueError, match=r"fork"):
            to_pipeline_dag(state)

    def test_to_pipeline_dag_raises_on_stub(self):
        # Stub block: no edges reach it.
        stub = _make_block("GaussianBlur")
        scope = BuilderScope()
        scope.blocks.append(stub)
        # No edges from InputImage; stub is orphaned.
        state = BuilderState(root=scope)

        with pytest.raises(ValueError, match=r"stub"):
            to_pipeline_dag(state)

    def test_to_pipeline_dag_raises_on_required_aux_empty(self):
        """Patch registry so ``inoculum_detector`` is required."""

        registry = get_registry()
        info = registry.get("FilamentousFungiDetector")
        original = info.parameters["inoculum_detector"]
        patched = ParamInfo(
            name=original.name,
            type_hint=original.type_hint,
            default=None,
            has_default=False,  # mark as required
            is_operation=original.is_operation,
            is_pipeline=original.is_pipeline,
            is_optional=original.is_optional,
            is_list=original.is_list,
            description=original.description,
            column_ref=original.column_ref,
        )
        with patch.dict(info.parameters, {"inoculum_detector": patched}):
            consumer = _make_block("FilamentousFungiDetector")
            scope = _scope_with_chain(consumer)
            state = BuilderState(root=scope)
            with pytest.raises(ValueError, match=r"required_aux"):
                to_pipeline_dag(state)

    def test_to_pipeline_dag_raises_on_cycle(self):
        a = _make_block("GaussianBlur", label="A")
        b = _make_block("GaussianBlur", label="B")
        scope = BuilderScope()
        scope.blocks.extend([a, b])
        # Reach from InputImage so the cycle is the only issue.
        scope.edges.append(_make_image_edge(scope.blocks[0], a))
        # Cycle A→B→A via aux.
        scope.edges.append(_make_aux_edge(a, b, "sigma"))
        scope.edges.append(_make_aux_edge(b, a, "sigma"))
        state = BuilderState(root=scope)

        with pytest.raises(ValueError, match=r"cycle"):
            to_pipeline_dag(state)

    def test_to_pipeline_dag_raises_on_container_mode(self):
        inner = BuilderScope()
        container = _make_block(
            PIPELINE_CLASS_NAME,
            label="MyContainer",
            nested=inner,
        )
        downstream = _make_block("OtsuDetector")
        aux_consumer = _make_block("FilamentousFungiDetector")

        scope = BuilderScope()
        scope.blocks.extend([container, downstream, aux_consumer])
        # Left wired (image-in to container).
        scope.edges.append(_make_image_edge(scope.blocks[0], container))
        # Right wired to aux of another block — illegal combination.
        scope.edges.append(
            _make_aux_edge(container, aux_consumer, "inoculum_detector")
        )
        # Also include image-out so aux_consumer is reachable.
        scope.edges.append(_make_image_edge(container, downstream))
        scope.edges.append(_make_image_edge(downstream, aux_consumer))
        state = BuilderState(root=scope)

        with pytest.raises(ValueError, match=r"container_mode"):
            to_pipeline_dag(state)

    def test_to_pipeline_dag_raises_on_missing_input(self):
        """Bypass auto-seed so the scope is genuinely InputImage-less."""

        scope = BuilderScope.__new__(BuilderScope)
        scope.blocks = []
        scope.edges = []
        scope.name = "Pipeline"
        scope.desc = ""
        scope.nrows = None
        scope.ncols = None
        # Add a non-input block so the validator has a target.
        scope.blocks.append(_make_block("GaussianBlur"))
        state = BuilderState(root=scope)

        with pytest.raises(ValueError, match=r"missing_input"):
            to_pipeline_dag(state)

    def test_to_pipeline_dag_does_not_raise_on_advisory_only(self):
        """Rule 7 (stage_order_hint) is advisory and must NOT block."""

        # Misorder: MeasureSize precedes a MaskDilator op stage.
        meas = _make_block("MeasureSize")
        refiner = _make_block("MaskDilator")
        scope = _scope_with_chain(meas, refiner)
        state = BuilderState(root=scope)

        # Should succeed and return a valid pipeline.
        pipeline = to_pipeline_dag(state)
        assert isinstance(pipeline, ImagePipeline)


class TestToPipelineDagStructuralInvariants:
    """Aux-only invariants + topological-order behaviour."""

    def test_to_pipeline_dag_aux_only_block_not_in_topological_order(self):
        """Aux source blocks should NEVER surface as top-level ops."""

        consumer = _make_block("FilamentousFungiDetector")
        aux_source = _make_block("OtsuDetector")

        scope = BuilderScope()
        scope.blocks.extend([consumer, aux_source])
        scope.edges.append(_make_image_edge(scope.blocks[0], consumer))
        scope.edges.append(_make_aux_edge(aux_source, consumer, "inoculum_detector"))
        state = BuilderState(root=scope)

        pipeline = to_pipeline_dag(state)
        top_level_ops = list(pipeline.get_ops().values())
        # Aux-only OtsuDetector should be folded into the consumer, NOT
        # a top-level op of its own.
        assert len(top_level_ops) == 1
        # The aux is materialised on the consumer.
        assert isinstance(top_level_ops[0], FilamentousFungiDetector)
        assert isinstance(top_level_ops[0].inoculum_detector, OtsuDetector)


# ---------------------------------------------------------------------------
# from_pipeline_dag
# ---------------------------------------------------------------------------


class TestFromPipelineDagBasic:
    """Basic round-trip behaviour."""

    def test_from_pipeline_dag_round_trip_linear(self):
        pipe = ImagePipeline(
            ops=[GaussianBlur(sigma=1.5), OtsuDetector()],
            meas=[MeasureSize()],
            name="rt",
        )
        state = from_pipeline_dag(pipe)
        rt_pipe = to_pipeline_dag(state)
        # Same ops in same order.
        assert [type(o).__name__ for o in rt_pipe.get_ops().values()] == [
            "GaussianBlur",
            "OtsuDetector",
        ]
        assert [type(m).__name__ for m in rt_pipe.get_meas().values()] == [
            "MeasureSize"
        ]
        assert rt_pipe.name == "rt"

    def test_from_pipeline_dag_seeds_input_image_in_root(self):
        pipe = ImagePipeline(ops=[GaussianBlur()])
        state = from_pipeline_dag(pipe)
        root_classes = [b.class_name for b in state.root.blocks]
        assert root_classes[0] == INPUT_IMAGE_CLASS_NAME

    def test_from_pipeline_dag_copies_root_metadata(self):
        pipe = ImagePipeline(
            ops=[GaussianBlur()],
            name="mypipe",
            desc="d",
            nrows=4,
            ncols=6,
        )
        state = from_pipeline_dag(pipe)
        assert state.root.name == "mypipe"
        assert state.root.desc == "d"
        assert state.root.nrows == 4
        assert state.root.ncols == 6


class TestFromPipelineDagShared:
    """Shared-instance dedup path."""

    def test_from_pipeline_dag_clones_shared_aux(self):
        """Same Python instance in both ``_ops`` and aux must clone."""

        shared = OtsuDetector()
        consumer = FilamentousFungiDetector(inoculum_detector=shared)
        pipe = ImagePipeline(ops=[shared, consumer])

        state = from_pipeline_dag(pipe)

        # Two distinct BlockNodes for OtsuDetector should exist.
        otsu_blocks = [
            b for b in state.root.blocks if b.class_name == "OtsuDetector"
        ]
        assert len(otsu_blocks) == 2
        ids = {b.block_id for b in otsu_blocks}
        assert len(ids) == 2  # truly different block_ids

        # Toast queued with "shared" in the text.
        assert len(state.toast_queue) == 1
        toast = state.toast_queue[0]
        assert toast["kind"] == "info"
        assert "shared" in toast["text"]


class TestFromPipelineDagListAux:
    """List-aux empty slot preservation."""

    def test_from_pipeline_dag_preserves_list_aux_empty_slots(self):
        det_a = OtsuDetector()
        det_c = OtsuDetector()
        # slot 1 intentionally None (empty)
        consumer = CompositeDetector(detectors=[det_a, None, det_c])
        pipe = ImagePipeline(ops=[consumer])

        state = from_pipeline_dag(pipe)

        composite_blocks = [
            b for b in state.root.blocks if b.class_name == "CompositeDetector"
        ]
        assert len(composite_blocks) == 1
        consumer_block = composite_blocks[0]
        assert consumer_block.list_slot_counts.get("detectors") == 3

        # Two aux edges should exist, with target_slot 0 and 2.
        aux_edges = [
            e
            for e in state.root.edges
            if e.kind == "aux"
            and e.target_block_id == consumer_block.block_id
            and e.target_port == "detectors"
        ]
        assert len(aux_edges) == 2
        slots = sorted(e.target_slot for e in aux_edges)
        assert slots == [0, 2]


class TestFromPipelineDagContainers:
    """Container recursion."""

    def test_from_pipeline_dag_recurses_into_containers(self):
        inner_pipe = ImagePipeline(
            ops=[GaussianBlur(), OtsuDetector()], name="inner"
        )
        outer = ImagePipeline(ops=[inner_pipe], name="outer")
        state = from_pipeline_dag(outer)

        pipeline_blocks = [
            b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
        ]
        assert len(pipeline_blocks) == 1
        container_block = pipeline_blocks[0]
        assert container_block.nested is not None
        inner_scope = container_block.nested
        inner_classes = [
            b.class_name for b in inner_scope.blocks if b.class_name != INPUT_IMAGE_CLASS_NAME
        ]
        assert inner_classes == ["GaussianBlur", "OtsuDetector"]
        # Inner scope still has an auto-seeded InputImage.
        assert any(
            b.class_name == INPUT_IMAGE_CLASS_NAME for b in inner_scope.blocks
        )

    def test_from_pipeline_dag_container_scope_leaves_nrows_ncols_none(self):
        inner_pipe = ImagePipeline(
            ops=[GaussianBlur()], name="inner", nrows=2, ncols=3
        )
        outer = ImagePipeline(
            ops=[inner_pipe], name="outer", nrows=4, ncols=5
        )
        state = from_pipeline_dag(outer)

        # Root carries outer's grid.
        assert state.root.nrows == 4
        assert state.root.ncols == 5

        # Container scope MUST leave nrows/ncols as None per §4.5.
        container_block = next(
            b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
        )
        assert container_block.nested is not None
        assert container_block.nested.nrows is None
        assert container_block.nested.ncols is None


# ---------------------------------------------------------------------------
# Legacy fixture loading
# ---------------------------------------------------------------------------


class TestFromPipelineDagLegacyJson:
    """Round-trip a legacy ``pipeline.json`` saved by the popover builder."""

    def test_from_pipeline_dag_loads_legacy_popover_pipeline_json(self):
        """Spec §8.3.9 — legacy JSON load through DAG conversion."""

        # The legacy_popover_pipeline.json fixture is a real
        # ImagePipeline.to_json() payload; we feed it through
        # ImagePipeline.from_json then from_pipeline_dag.
        fixture_path = FIXTURES_DIR / "legacy_popover_pipeline.json"
        config_str = fixture_path.read_text()
        pipe = ImagePipeline.from_json(config_str)
        state = from_pipeline_dag(pipe)

        # Validation should pass on the loaded state (no errors).
        from phenotypic.gui.builder._validation import validate

        issues = validate(state)
        errors = [iss for iss in issues if iss.severity == "error"]
        assert errors == [], f"Legacy pipeline failed validation: {errors}"

        # Expect at least one non-InputImage block + InputImage seed.
        non_input = [
            b for b in state.root.blocks if b.class_name != INPUT_IMAGE_CLASS_NAME
        ]
        assert len(non_input) >= 1
