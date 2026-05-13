"""Auto-recovery + non-crash tests for the DAG validation layer.

Three guards are covered here:

1. Rule 6's auto-recovery — a corrupt scope missing its ``InputImage``
   self-heals on the next call to :func:`_seed_input_image` and
   subsequently validates clean.
2. ``unknown_class`` never crashes validation — a block referencing a
   class missing from the registry surfaces an advisory issue and
   continues; no other rule blows up on the same block.
3. Shared-instance clone behaviour delegated to
   :func:`from_pipeline_dag` (owned by Agent 1B) — when an ``ImagePipeline``
   carries the same operation instance in its ops list *and* embedded in
   another op's aux, the loader clones the inner usage into a fresh
   :class:`BlockNode` and queues a toast.  This test cross-covers the
   1B behaviour from the perspective of the recovery spec; the canonical
   test lives in ``test_state_dag.py::test_from_pipeline_clones_shared_aux``.
"""

from __future__ import annotations

import pytest

from phenotypic.gui._operation_registry import OperationInfo
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    BlockNode,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    _seed_input_image,
)
from phenotypic.gui.builder._validation import validate

from .conftest import _make_param


# ---------------------------------------------------------------------------
# Rule 6 — auto-recovery.
# ---------------------------------------------------------------------------


def test_missing_input_recovers_on_next_seed():
    """A scope missing its ``InputImage`` becomes valid after ``_seed_input_image``.

    Simulates the dispatcher's pre-mutation pass which calls
    ``_seed_input_image`` on every reachable scope.  Before seed, the
    validator flags ``missing_input`` and ``stub`` for every non-input
    block (since no root exists).  After seed, those vanish.
    """

    scope = _DagBuilderScope()  # auto-seeds an InputImage
    # Drop the seeded InputImage to simulate corrupt input.
    scope.blocks = [b for b in scope.blocks if b.class_name != INPUT_IMAGE_CLASS_NAME]

    issues_before = validate(_DagBuilderState(root=scope))
    assert any(i.kind == "missing_input" for i in issues_before)

    _seed_input_image(scope)
    issues_after = validate(_DagBuilderState(root=scope))
    assert not [i for i in issues_after if i.kind == "missing_input"]


def test_seed_is_idempotent_on_repeat_calls():
    """Calling ``_seed_input_image`` on a scope that already has one is a no-op."""

    scope = _DagBuilderScope()
    pre_count = sum(
        1 for b in scope.blocks if b.class_name == INPUT_IMAGE_CLASS_NAME
    )
    _seed_input_image(scope)
    _seed_input_image(scope)
    post_count = sum(
        1 for b in scope.blocks if b.class_name == INPUT_IMAGE_CLASS_NAME
    )
    assert pre_count == 1 == post_count


# ---------------------------------------------------------------------------
# unknown_class advisory + non-crash.
# ---------------------------------------------------------------------------


def test_unknown_class_does_not_crash_validation(empty_registry):
    """A block whose class is missing from the registry yields an advisory.

    Crucially, the call does NOT raise — the validator skips all
    other rules for that block and continues.
    """

    scope = _DagBuilderScope()
    ghost = BlockNode(
        block_id=_new_block_id(),
        class_name="MissingFromRegistry",
        params={},
    )
    scope.blocks.append(ghost)
    # Wire ghost so it's not also a stub (so we can focus the assert).
    from phenotypic.gui.builder._state import Edge

    scope.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=scope.blocks[0].block_id,
            target_block_id=ghost.block_id,
            target_port="in",
            kind="image",
        )
    )

    # Must not raise.
    issues = validate(_DagBuilderState(root=scope))

    unknowns = [i for i in issues if i.kind == "unknown_class"]
    assert len(unknowns) == 1
    assert unknowns[0].block_id == ghost.block_id
    assert unknowns[0].severity == "advisory"
    # Rule 3 does not fire for an unknown class.
    assert not [
        i for i in issues
        if i.kind == "required_aux" and i.block_id == ghost.block_id
    ]


def test_unknown_class_does_not_block_other_rules(empty_registry):
    """A scope with an unknown class still validates other blocks normally."""

    empty_registry.ops["NeedsAux"] = OperationInfo(
        cls=type("NeedsAuxStub", (), {}),
        name="NeedsAux",
        category="Detector",
        module="tests.fake",
        parameters={
            "required_param": _make_param(
                "required_param", has_default=False, is_operation=True,
            ),
        },
    )
    scope = _DagBuilderScope()
    ghost = BlockNode(
        block_id=_new_block_id(),
        class_name="MissingFromRegistry",
        params={},
    )
    real_consumer = BlockNode(
        block_id=_new_block_id(),
        class_name="NeedsAux",
        params={},
    )
    scope.blocks.extend([ghost, real_consumer])

    issues = validate(_DagBuilderState(root=scope))
    # Unknown class advisory present.
    assert any(
        i.kind == "unknown_class" and i.block_id == ghost.block_id
        for i in issues
    )
    # Required-aux still fires on the real consumer.
    assert any(
        i.kind == "required_aux" and i.block_id == real_consumer.block_id
        for i in issues
    )


# ---------------------------------------------------------------------------
# Shared-instance clone — delegates to ``from_pipeline_dag`` (Agent 1B).
# ---------------------------------------------------------------------------


def test_from_pipeline_dag_shared_instance_clone():
    """Loading an ImagePipeline with a shared op clones the inner usage.

    The DAG loader (``from_pipeline_dag``) detects when the same Python
    instance appears both as a top-level op and embedded as an aux
    param of another op; it clones the inner occurrence via
    ``copy.deepcopy`` and queues a toast describing the rewrite.

    This is the recovery-spec view of the same property tested
    canonically in ``test_state_dag.py``; we duplicate here so the
    recovery suite stands alone.
    """

    pytest.importorskip(
        "phenotypic.gui.builder._conversion_dag",
        reason=(
            "Agent 1B owns _conversion_dag; defer until that module is "
            "present.  Cross-coverage will run after orchestrator merges."
        ),
    )
    from phenotypic.gui.builder._conversion_dag import from_pipeline_dag

    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector, FilamentousFungiDetector

    # Shared OtsuDetector instance: appears as a top-level op AND as
    # FilamentousFungiDetector's inoculum_detector aux.
    shared = OtsuDetector()
    detector = FilamentousFungiDetector(inoculum_detector=shared)
    pipeline = ImagePipeline([shared, detector])

    state = from_pipeline_dag(pipeline)

    # Expect TWO distinct BlockNodes carrying the OtsuDetector class:
    # the original top-level op and a clone seeded into the aux slot.
    otsu_blocks = [
        b for b in state.root.blocks if b.class_name == "OtsuDetector"
    ]
    # The aux-embedded clone lives in the aux slot of FilamentousFungiDetector;
    # its representation depends on 1B's layout (it could be a sibling
    # block in the same scope or embedded elsewhere).  Either way, the
    # count of OtsuDetector BlockNodes across the state must be >= 2.
    nested_otsu = 0
    for b in state.root.blocks:
        if b.nested is not None:
            nested_otsu += sum(
                1 for nb in b.nested.blocks if nb.class_name == "OtsuDetector"
            )
    total_otsu = len(otsu_blocks) + nested_otsu
    assert total_otsu >= 2, (
        f"Expected >= 2 OtsuDetector blocks after de-share; got {total_otsu}"
    )

    # A toast was queued mentioning the share rewrite.  ``from_pipeline_dag``
    # stores the human-readable text under the ``text`` key (1B's contract);
    # we accept ``message`` as a fallback for forward-compatibility with any
    # future toast schema migration.
    toast_msgs = " ".join(
        (t.get("text", t.get("message", "")) if isinstance(t, dict) else str(t))
        for t in state.toast_queue
    ).lower()
    assert "shared" in toast_msgs, (
        f"Expected 'shared' toast; got toast_queue={state.toast_queue}"
    )
