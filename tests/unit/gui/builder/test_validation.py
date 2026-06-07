"""Unit tests for the DAG builder's pure validation layer.

Mirrors spec ``2026-05-12-builder-dag-redesign-design.md`` §5.3: every
rule has a positive (issue emitted) and negative (issue absent) case,
plus a recursion test that exercises ``scope_path``.

The tests construct ``_DagBuilderState`` instances directly under the
stable underscore-prefixed name.  Phase 8 retired the
``PHENOTYPIC_GUI_DAG`` feature flag and the public ``BuilderState``
alias now resolves to the DAG class permanently; importing the
underscore name keeps these rule tests resilient to any future alias
re-binding.  All registry-dependent rules
(Rule 3, Rule 7) monkeypatch the validation module's
``get_registry`` symbol so the tests don't depend on the registry's
current operation inventory — that way new ops or signature changes
can never break this suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from phenotypic.gui._operation_registry import OperationInfo
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
)
from phenotypic.gui.builder._validation import Issue, validate

from .conftest import _make_op_info, _make_param


# ---------------------------------------------------------------------------
# Helpers for constructing test scopes/states.
# ---------------------------------------------------------------------------


def _new_block(class_name: str, **kwargs: Any) -> BlockNode:
    """Return a ``BlockNode`` with a fresh block_id."""

    return BlockNode(
        block_id=_new_block_id(),
        class_name=class_name,
        params=kwargs.pop("params", {}),
        **kwargs,
    )


def _image_edge(src: str, tgt: str) -> Edge:
    return Edge(
        edge_id=_new_block_id(),
        source_block_id=src,
        target_block_id=tgt,
        target_port="in",
        kind="image",
    )


def _aux_edge(
    src: str,
    tgt: str,
    port: str,
    slot: Optional[int] = None,
) -> Edge:
    return Edge(
        edge_id=_new_block_id(),
        source_block_id=src,
        target_block_id=tgt,
        target_port=port,
        target_slot=slot,
        kind="aux",
    )


def _wrap(scope: _DagBuilderScope) -> _DagBuilderState:
    """Wrap a scope in a ``_DagBuilderState`` for ``validate(state)``."""

    return _DagBuilderState(root=scope)


# ---------------------------------------------------------------------------
# Rule 1 — image-flow forks (output, input, mixed-kind).
# ---------------------------------------------------------------------------


class TestRule1Fork:
    def test_rule_1_image_fork_on_output(self):
        """A single block with 2 outgoing image edges → ``kind="fork"``."""

        scope = _DagBuilderScope()
        src = _new_block("GaussianBlur")
        t1 = _new_block("GaussianBlur")
        t2 = _new_block("GaussianBlur")
        scope.blocks.extend([src, t1, t2])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, src.block_id))
        scope.edges.append(_image_edge(src.block_id, t1.block_id))
        scope.edges.append(_image_edge(src.block_id, t2.block_id))

        issues = validate(_wrap(scope))
        forks = [i for i in issues if i.kind == "fork"]
        assert any(i.block_id == src.block_id for i in forks)

    def test_rule_1_image_fork_on_input(self):
        """Two image edges into one input port → ``kind="fork"``."""

        scope = _DagBuilderScope()
        a = _new_block("GaussianBlur")
        b = _new_block("GaussianBlur")
        sink = _new_block("GaussianBlur")
        scope.blocks.extend([a, b, sink])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
        scope.edges.append(_image_edge(scope.blocks[0].block_id, b.block_id))
        scope.edges.append(_image_edge(a.block_id, sink.block_id))
        scope.edges.append(_image_edge(b.block_id, sink.block_id))

        issues = validate(_wrap(scope))
        forks = [i for i in issues if i.kind == "fork"]
        # Input-port fork is reported on the target block.
        assert any(i.block_id == sink.block_id for i in forks)

    def test_rule_1_mixed_kind_fan_out_is_fork(self):
        """One image-out + one aux-out from the same source → fork.

        Per spec §4.2: "every output port has at most one outgoing
        wire, total."  Wiring the same source to one image-in AND one
        aux-in violates the rule.
        """

        scope = _DagBuilderScope()
        src = _new_block("GaussianBlur")
        downstream = _new_block("GaussianBlur")
        consumer = _new_block("GaussianBlur")
        scope.blocks.extend([src, downstream, consumer])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, src.block_id))
        scope.edges.append(_image_edge(src.block_id, downstream.block_id))
        scope.edges.append(
            _aux_edge(src.block_id, consumer.block_id, port="extra")
        )

        issues = validate(_wrap(scope))
        forks = [i for i in issues if i.kind == "fork"]
        assert any(i.block_id == src.block_id for i in forks)

    def test_rule_1_single_image_out_no_fork(self):
        """A clean linear chain emits no fork issue."""

        scope = _DagBuilderScope()
        a = _new_block("GaussianBlur")
        b = _new_block("GaussianBlur")
        scope.blocks.extend([a, b])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
        scope.edges.append(_image_edge(a.block_id, b.block_id))

        issues = validate(_wrap(scope))
        assert not [i for i in issues if i.kind == "fork"]


# ---------------------------------------------------------------------------
# Rule 2 — stubs (BFS-unreachable blocks).
# ---------------------------------------------------------------------------


class TestRule2Stub:
    def test_rule_2_stub_orphan_block(self):
        """A block with no edges is unreachable → ``kind="stub"``."""

        scope = _DagBuilderScope()
        orphan = _new_block("GaussianBlur")
        scope.blocks.append(orphan)

        issues = validate(_wrap(scope))
        stubs = [i for i in issues if i.kind == "stub"]
        assert any(i.block_id == orphan.block_id for i in stubs)

    def test_rule_2_stub_after_wire_delete(self):
        """Removing a middle wire orphans every downstream block."""

        scope = _DagBuilderScope()
        a = _new_block("GaussianBlur")
        b = _new_block("GaussianBlur")
        c = _new_block("GaussianBlur")
        scope.blocks.extend([a, b, c])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
        scope.edges.append(_image_edge(a.block_id, b.block_id))
        scope.edges.append(_image_edge(b.block_id, c.block_id))
        # "Delete the middle wire" — remove a → b.
        scope.edges = [e for e in scope.edges if not (
            e.source_block_id == a.block_id and e.target_block_id == b.block_id
        )]

        issues = validate(_wrap(scope))
        stubs = [i for i in issues if i.kind == "stub"]
        stub_ids = {i.block_id for i in stubs}
        assert b.block_id in stub_ids
        assert c.block_id in stub_ids
        # a is still reachable (it's wired from the InputImage).
        assert a.block_id not in stub_ids


# ---------------------------------------------------------------------------
# Rule 3 — required aux ports.
# ---------------------------------------------------------------------------


class TestRule3RequiredAux:
    def test_rule_3_required_scalar_empty(self, empty_registry):
        """A scalar required aux port with no wire → ``required_aux``."""

        empty_registry.ops["NeedsAux"] = _make_op_info(
            "NeedsAux",
            {
                "inoculum_detector": _make_param(
                    "inoculum_detector",
                    has_default=False,
                    is_operation=True,
                ),
            },
        )
        scope = _DagBuilderScope()
        consumer = _new_block("NeedsAux")
        scope.blocks.append(consumer)
        scope.edges.append(_image_edge(scope.blocks[0].block_id, consumer.block_id))

        issues = validate(_wrap(scope))
        req = [i for i in issues if i.kind == "required_aux"]
        assert any(
            i.block_id == consumer.block_id and "inoculum_detector" in i.detail
            for i in req
        )

    def test_rule_3_optional_aux_empty_is_clean(self, empty_registry):
        """``has_default=True`` aux param empty → no issue."""

        empty_registry.ops["OptionalAux"] = _make_op_info(
            "OptionalAux",
            {
                "inoculum_detector": _make_param(
                    "inoculum_detector",
                    has_default=True,
                    is_operation=True,
                ),
            },
        )
        scope = _DagBuilderScope()
        consumer = _new_block("OptionalAux")
        scope.blocks.append(consumer)
        scope.edges.append(_image_edge(scope.blocks[0].block_id, consumer.block_id))

        issues = validate(_wrap(scope))
        assert not [i for i in issues if i.kind == "required_aux"]

    def test_rule_3_required_uses_has_default_not_inspect_empty(
        self, empty_registry,
    ):
        """REGRESSION GUARD for the ``has_default`` vs ``inspect.Parameter.empty`` bug.

        The registry normalises missing defaults to ``None``, so testing
        ``p.default is inspect.Parameter.empty`` is always ``False`` and
        would silently disable Rule 3.  This test pins both axes:

        * ``default=None`` + ``has_default=False`` → required → issue.
        * ``default=None`` + ``has_default=True`` → optional → no issue.
        """

        # Case A: default=None, has_default=False → required.
        empty_registry.ops["RequiredNone"] = _make_op_info(
            "RequiredNone",
            {
                "needs_it": _make_param(
                    "needs_it",
                    has_default=False,
                    default=None,
                    is_operation=True,
                ),
            },
        )
        scope_a = _DagBuilderScope()
        a = _new_block("RequiredNone")
        scope_a.blocks.append(a)
        scope_a.edges.append(_image_edge(scope_a.blocks[0].block_id, a.block_id))
        issues_a = validate(_wrap(scope_a))
        assert any(
            i.kind == "required_aux" and i.block_id == a.block_id
            for i in issues_a
        )

        # Case B: default=None, has_default=True → optional.
        empty_registry.ops["OptionalNone"] = _make_op_info(
            "OptionalNone",
            {
                "may_have": _make_param(
                    "may_have",
                    has_default=True,
                    default=None,
                    is_operation=True,
                ),
            },
        )
        scope_b = _DagBuilderScope()
        b = _new_block("OptionalNone")
        scope_b.blocks.append(b)
        scope_b.edges.append(_image_edge(scope_b.blocks[0].block_id, b.block_id))
        issues_b = validate(_wrap(scope_b))
        assert not [
            i for i in issues_b
            if i.kind == "required_aux" and i.block_id == b.block_id
        ]

    def test_rule_3_required_list_empty(self, empty_registry):
        """A required list-aux with no wires → ``required_aux``."""

        empty_registry.ops["NeedsList"] = _make_op_info(
            "NeedsList",
            {
                "detectors": _make_param(
                    "detectors",
                    has_default=False,
                    is_operation=True,
                    is_list=True,
                ),
            },
        )
        scope = _DagBuilderScope()
        consumer = _new_block("NeedsList")
        scope.blocks.append(consumer)
        scope.edges.append(_image_edge(scope.blocks[0].block_id, consumer.block_id))

        issues = validate(_wrap(scope))
        req = [i for i in issues if i.kind == "required_aux"]
        assert any(
            i.block_id == consumer.block_id and "detectors" in i.detail
            for i in req
        )

    def test_rule_3_required_list_with_one_wire_clean(self, empty_registry):
        """A required list-aux with at least one wired slot → no issue."""

        empty_registry.ops["NeedsList"] = _make_op_info(
            "NeedsList",
            {
                "detectors": _make_param(
                    "detectors",
                    has_default=False,
                    is_operation=True,
                    is_list=True,
                ),
            },
        )
        empty_registry.ops["Producer"] = _make_op_info("Producer", {})
        scope = _DagBuilderScope()
        producer = _new_block("Producer")
        consumer = _new_block("NeedsList")
        scope.blocks.extend([producer, consumer])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, consumer.block_id))
        scope.edges.append(
            _aux_edge(producer.block_id, consumer.block_id, "detectors", slot=0)
        )

        issues = validate(_wrap(scope))
        assert not [i for i in issues if i.kind == "required_aux"]


# ---------------------------------------------------------------------------
# Rule 4 — cycles.
# ---------------------------------------------------------------------------


class TestRule4Cycle:
    def test_rule_4_image_cycle(self):
        """A → B → A on image edges → both members flagged."""

        scope = _DagBuilderScope()
        a = _new_block("GaussianBlur")
        b = _new_block("GaussianBlur")
        scope.blocks.extend([a, b])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
        scope.edges.append(_image_edge(a.block_id, b.block_id))
        scope.edges.append(_image_edge(b.block_id, a.block_id))

        issues = validate(_wrap(scope))
        cycles = [i for i in issues if i.kind == "cycle"]
        member_ids = {i.block_id for i in cycles}
        assert a.block_id in member_ids
        assert b.block_id in member_ids

    def test_rule_4_aux_cycle(self):
        """An aux-only A → B → A cycle is still detected."""

        scope = _DagBuilderScope()
        a = _new_block("GaussianBlur")
        b = _new_block("GaussianBlur")
        scope.blocks.extend([a, b])
        # Reach both via image flow + aux ring on top.
        scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
        scope.edges.append(_aux_edge(a.block_id, b.block_id, "p"))
        scope.edges.append(_aux_edge(b.block_id, a.block_id, "p"))

        issues = validate(_wrap(scope))
        cycles = [i for i in issues if i.kind == "cycle"]
        member_ids = {i.block_id for i in cycles}
        assert a.block_id in member_ids
        assert b.block_id in member_ids

    def test_rule_4_mixed_image_aux_cycle(self):
        """A mixed image+aux cycle still trips Rule 4."""

        scope = _DagBuilderScope()
        a = _new_block("GaussianBlur")
        b = _new_block("GaussianBlur")
        scope.blocks.extend([a, b])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
        scope.edges.append(_image_edge(a.block_id, b.block_id))
        scope.edges.append(_aux_edge(b.block_id, a.block_id, "p"))

        issues = validate(_wrap(scope))
        cycles = [i for i in issues if i.kind == "cycle"]
        member_ids = {i.block_id for i in cycles}
        assert a.block_id in member_ids
        assert b.block_id in member_ids


# ---------------------------------------------------------------------------
# Rule 5 — container left/right wiring consistency.
# ---------------------------------------------------------------------------


class TestRule5ContainerMode:
    def test_rule_5_container_left_wired_right_purple(self):
        """Container left wired AND right wires to aux → ``container_mode``."""

        scope = _DagBuilderScope()
        container = _new_block(PIPELINE_CLASS_NAME)
        consumer = _new_block("GaussianBlur")
        scope.blocks.extend([container, consumer])
        # Left wired (image-in to container).
        scope.edges.append(_image_edge(scope.blocks[0].block_id, container.block_id))
        # Right wired to aux.
        scope.edges.append(_aux_edge(container.block_id, consumer.block_id, "extra"))

        issues = validate(_wrap(scope))
        modes = [i for i in issues if i.kind == "container_mode"]
        assert any(i.block_id == container.block_id for i in modes)

    def test_rule_5_container_left_unwired_right_blue(self):
        """Container left unwired AND right wires to image → ``container_mode``."""

        scope = _DagBuilderScope()
        container = _new_block(PIPELINE_CLASS_NAME)
        downstream = _new_block("GaussianBlur")
        scope.blocks.extend([container, downstream])
        # Right wired to image-in, but left unwired.
        scope.edges.append(_image_edge(container.block_id, downstream.block_id))

        issues = validate(_wrap(scope))
        modes = [i for i in issues if i.kind == "container_mode"]
        assert any(i.block_id == container.block_id for i in modes)

    def test_rule_5_container_consistent_main_flow(self):
        """Left wired + right wires to image → no container_mode issue."""

        scope = _DagBuilderScope()
        container = _new_block(PIPELINE_CLASS_NAME)
        downstream = _new_block("GaussianBlur")
        scope.blocks.extend([container, downstream])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, container.block_id))
        scope.edges.append(_image_edge(container.block_id, downstream.block_id))

        issues = validate(_wrap(scope))
        assert not [i for i in issues if i.kind == "container_mode"]


# ---------------------------------------------------------------------------
# Rule 6 — exactly one Input Image per scope.
# ---------------------------------------------------------------------------


class TestRule6InputImage:
    def test_rule_6_zero_input_image(self):
        """A scope with no InputImage block → ``missing_input``."""

        # Bypass the auto-seed by building a bare _DagBuilderScope and then
        # stripping the seeded block.  This simulates corrupt JSON.
        scope = _DagBuilderScope()
        scope.blocks.clear()

        issues = validate(_wrap(scope))
        missing = [i for i in issues if i.kind == "missing_input"]
        assert len(missing) == 1
        assert missing[0].block_id is None

    def test_rule_6_two_input_images(self):
        """Two InputImage blocks in one scope → one ``duplicate_input``."""

        scope = _DagBuilderScope()
        extra = _new_block(INPUT_IMAGE_CLASS_NAME)
        scope.blocks.append(extra)

        issues = validate(_wrap(scope))
        dups = [i for i in issues if i.kind == "duplicate_input"]
        # Spec: "for each extra: Issue(...)" — the FIRST is kept.
        assert len(dups) == 1
        assert dups[0].block_id == extra.block_id

    def test_rule_6_one_input_image_no_issue(self):
        """The default scope (auto-seeded InputImage) emits no Rule 6 issue."""

        scope = _DagBuilderScope()
        issues = validate(_wrap(scope))
        assert not [
            i for i in issues
            if i.kind in ("missing_input", "duplicate_input")
        ]


# ---------------------------------------------------------------------------
# Rule 7 (advisory) — stage ordering hint.
# ---------------------------------------------------------------------------


class TestRule7StageOrder:
    def test_rule_7_stage_misorder_yields_advisory(self, empty_registry):
        """A meas → ops chain (visually) emits one advisory."""

        # MeasFake is a MeasureFeatures subclass; OpFake is an
        # ImageOperation subclass (resolves to stage "ops").  The
        # validator only consults ``OperationInfo.cls`` and uses
        # ``issubclass`` so we need real classes.
        from phenotypic.abc_ import MeasureFeatures
        from phenotypic.enhance import GaussianBlur

        # Build OperationInfo records pointing at real classes so
        # ``_safe_stage`` returns the right thing.  We replace .cls in
        # the OperationInfo wrapper.
        class _MeasFake(MeasureFeatures):  # type: ignore[misc]
            pass

        empty_registry.ops["MeasFake"] = OperationInfo(
            cls=_MeasFake,
            name="MeasFake",
            category="Measure",
            module="tests.fake",
            parameters={},
        )
        empty_registry.ops["OpFake"] = OperationInfo(
            cls=GaussianBlur,
            name="OpFake",
            category="Enhancer",
            module="tests.fake",
            parameters={},
        )

        scope = _DagBuilderScope()
        meas = _new_block("MeasFake")
        op = _new_block("OpFake")
        scope.blocks.extend([meas, op])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, meas.block_id))
        # Visually meas → op (stage 1 → stage 0); should fire advisory.
        scope.edges.append(_image_edge(meas.block_id, op.block_id))

        issues = validate(_wrap(scope))
        hints = [i for i in issues if i.kind == "stage_order_hint"]
        assert len(hints) == 1
        assert hints[0].severity == "advisory"
        assert hints[0].block_id == meas.block_id

    def test_rule_7_proper_order_no_advisory(self, empty_registry):
        """An ops → meas chain emits no advisory."""

        from phenotypic.abc_ import MeasureFeatures
        from phenotypic.enhance import GaussianBlur

        class _MeasFake(MeasureFeatures):  # type: ignore[misc]
            pass

        empty_registry.ops["OpFake"] = OperationInfo(
            cls=GaussianBlur,
            name="OpFake",
            category="Enhancer",
            module="tests.fake",
            parameters={},
        )
        empty_registry.ops["MeasFake"] = OperationInfo(
            cls=_MeasFake,
            name="MeasFake",
            category="Measure",
            module="tests.fake",
            parameters={},
        )

        scope = _DagBuilderScope()
        op = _new_block("OpFake")
        meas = _new_block("MeasFake")
        scope.blocks.extend([op, meas])
        scope.edges.append(_image_edge(scope.blocks[0].block_id, op.block_id))
        scope.edges.append(_image_edge(op.block_id, meas.block_id))

        issues = validate(_wrap(scope))
        assert not [i for i in issues if i.kind == "stage_order_hint"]


# ---------------------------------------------------------------------------
# Recursion into nested container scopes.
# ---------------------------------------------------------------------------


def test_recursion_into_container_aggregates_scope_path():
    """An issue in a nested scope carries its container's id in ``scope_path``."""

    outer = _DagBuilderScope()
    container = _new_block(PIPELINE_CLASS_NAME)
    container.nested = _DagBuilderScope()
    # Make the container's nested scope have a fork.
    inner = container.nested
    src = _new_block("GaussianBlur")
    t1 = _new_block("GaussianBlur")
    t2 = _new_block("GaussianBlur")
    inner.blocks.extend([src, t1, t2])
    inner.edges.append(_image_edge(inner.blocks[0].block_id, src.block_id))
    inner.edges.append(_image_edge(src.block_id, t1.block_id))
    inner.edges.append(_image_edge(src.block_id, t2.block_id))
    # Attach container in outer scope (left-wired + right wired image to
    # downstream — clean Rule 5).
    downstream = _new_block("GaussianBlur")
    outer.blocks.extend([container, downstream])
    outer.edges.append(_image_edge(outer.blocks[0].block_id, container.block_id))
    outer.edges.append(_image_edge(container.block_id, downstream.block_id))

    issues = validate(_wrap(outer))
    forks = [i for i in issues if i.kind == "fork"]
    # The fork lives in the inner scope; scope_path should be the
    # container's block_id.
    assert any(
        i.scope_path == [container.block_id] and i.block_id == src.block_id
        for i in forks
    )


# ---------------------------------------------------------------------------
# unknown_class blocking + non-crashing.
# ---------------------------------------------------------------------------


def test_unknown_class_blocks_save_preview_without_crashing(empty_registry):
    """A block whose ``class_name`` is not in the registry yields an error.

    Rule 3 skips the block (no required-aux check possible) and no
    other rule fires for that block.
    """

    scope = _DagBuilderScope()
    ghost = _new_block("DoesNotExist")
    scope.blocks.append(ghost)
    scope.edges.append(_image_edge(scope.blocks[0].block_id, ghost.block_id))

    issues = validate(_wrap(scope))
    unknown = [i for i in issues if i.kind == "unknown_class"]
    assert len(unknown) == 1
    assert unknown[0].block_id == ghost.block_id
    assert unknown[0].severity == "error"
    # No required_aux for the unknown class.
    assert not [
        i for i in issues
        if i.kind == "required_aux" and i.block_id == ghost.block_id
    ]


def test_empty_scope_with_only_input_image_is_valid():
    """A freshly constructed scope (InputImage only) is valid."""

    scope = _DagBuilderScope()
    issues = validate(_wrap(scope))
    assert issues == []


# ---------------------------------------------------------------------------
# Fixture-driven assertions.  Each invalid fixture pairs with a
# ``.expected_issues.json`` sibling describing the (kind, severity,
# scope_path) tuples we expect.
# ---------------------------------------------------------------------------


_FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "builder_dag"


def _normalise(
    issue: Issue, block_lookup: Dict[str, BlockNode]
) -> Tuple[str, str, str, Tuple[str, ...]]:
    """Project an issue into a comparable tuple.

    Returns ``(kind, severity, block_label, tuple(scope_path_labels))``.
    ``block_id`` and ``detail`` are excluded so the fixture authors
    don't have to know the auto-generated UUIDs.  ``scope_path`` is
    re-keyed against block labels for the same reason.
    """

    label = ""
    if issue.block_id is not None and issue.block_id in block_lookup:
        b = block_lookup[issue.block_id]
        label = b.label or b.class_name
    scope_labels: List[str] = []
    for bid in issue.scope_path:
        b = block_lookup.get(bid)
        if b is not None:
            scope_labels.append(b.label or b.class_name)
        else:
            scope_labels.append(bid)
    return (issue.kind, issue.severity, label, tuple(scope_labels))


def _build_lookup(scope: _DagBuilderScope) -> Dict[str, BlockNode]:
    """Recursively collect every block by id from a scope (and its nested)."""

    out: Dict[str, BlockNode] = {}
    for b in scope.blocks:
        out[b.block_id] = b
        if b.nested is not None:
            out.update(_build_lookup(b.nested))
    return out


def _scope_from_fixture(data: Dict[str, Any]) -> _DagBuilderScope:
    """Reconstruct a ``_DagBuilderScope`` from a fixture JSON dict.

    The fixture format is human-authored (not the canonical
    ``state_to_json`` shape) — see fixtures/builder_dag/README in the
    sibling directory for the schema.  Briefly: a ``blocks`` list whose
    entries carry ``block_id`` (the fixture's own stable id used in
    expected_issues), ``class_name``, ``label`` (optional), and
    optionally ``nested`` (a recursive scope dict).  ``edges`` carry
    ``source_block_id``, ``target_block_id``, ``target_port``,
    ``target_slot`` (optional), and ``kind``.
    """

    scope = _DagBuilderScope.__new__(_DagBuilderScope)
    # Bypass __post_init__ auto-seed so fixtures can decide to omit the
    # InputImage when they're exercising Rule 6.
    scope.blocks = []
    scope.edges = []
    scope.name = data.get("name", "Pipeline")
    scope.desc = data.get("desc", "")
    scope.nrows = data.get("nrows")
    scope.ncols = data.get("ncols")
    for b in data.get("blocks", []):
        block = BlockNode(
            block_id=b["block_id"],
            class_name=b["class_name"],
            params=b.get("params", {}),
            label=b.get("label"),
            collapsed=b.get("collapsed", False),
            list_slot_counts=b.get("list_slot_counts", {}) or {},
        )
        nested = b.get("nested")
        if nested is not None:
            block.nested = _scope_from_fixture(nested)
        scope.blocks.append(block)
    for e in data.get("edges", []):
        scope.edges.append(
            Edge(
                edge_id=e.get("edge_id", _new_block_id()),
                source_block_id=e["source_block_id"],
                source_port=e.get("source_port", "out"),
                target_block_id=e["target_block_id"],
                target_port=e.get("target_port", "in"),
                target_slot=e.get("target_slot"),
                kind=e.get("kind", "image"),
            )
        )
    return scope


_FIXTURE_NAMES = (
    "fork_offender",
    "mixed_kind_fan_out",
    "image_cycle",
    "aux_cycle",
    "unwired_required",
    "mixed_container_mode",
    "duplicate_input_image",
)


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_invalid_fixture_matches_expected_issues(name, empty_registry):
    """For each invalid fixture, ``validate`` matches the expected issues.

    Normalisation strips ``block_id`` and ``detail``; comparison uses
    ``(kind, severity, block_label, scope_path_labels)`` so fixtures
    don't have to track auto-generated UUIDs.
    """

    fixture_path = _FIXTURE_DIR / f"{name}.json"
    expected_path = _FIXTURE_DIR / f"{name}.expected_issues.json"
    if not fixture_path.exists() or not expected_path.exists():
        pytest.skip(
            f"fixture {name}.json or {name}.expected_issues.json not yet "
            "contributed (1A owns the directory; 1C contributes invalid "
            "fixtures)."
        )

    # Some fixtures rely on a registered op with a required aux.  Pre-
    # register a stub registry that covers the ``unwired_required``
    # case so the test is hermetic.
    empty_registry.ops["NeedsAux"] = _make_op_info(
        "NeedsAux",
        {
            "required_param": _make_param(
                "required_param", has_default=False, is_operation=True,
            ),
        },
    )
    empty_registry.ops["GaussianBlur"] = _make_op_info("GaussianBlur", {})

    with fixture_path.open(encoding="utf-8") as f:
        scope_data = json.load(f)
    with expected_path.open(encoding="utf-8") as f:
        expected_raw = json.load(f)
    # Expected files may use either a bare list or a wrapper dict like
    # ``{"issues": [...]}`` so the formatter can keep them readable.
    if isinstance(expected_raw, dict):
        expected = expected_raw.get("issues", [])
    else:
        expected = expected_raw

    scope = _scope_from_fixture(scope_data)
    state = _DagBuilderState(root=scope)

    actual = validate(state)
    lookup = _build_lookup(scope)
    actual_norm = {_normalise(i, lookup) for i in actual}

    expected_norm = set()
    for spec in expected:
        expected_norm.add(
            (
                spec["kind"],
                spec.get("severity", "error"),
                spec.get("block_label", ""),
                tuple(spec.get("scope_path", [])),
            )
        )

    assert actual_norm == expected_norm, (
        f"Fixture {name}: actual={actual_norm} expected={expected_norm}"
    )
