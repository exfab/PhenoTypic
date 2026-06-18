"""Unit tests for the Run preview / Save pipeline validation gates.

Spec §5.6 specifies that ``request_run_preview`` and
``request_save_pipeline`` must pre-check :func:`validate` and abort
when any blocking-severity issue (``severity == "error"``) exists.
Advisory hints (``severity == "advisory"`` — currently
``stage_order_hint``) NEVER block these actions; they decorate the
canvas with yellow borders and surface in the issue badge tooltip
but the user can still preview / save through. Unknown classes and
unsupported linear DAG shapes block because the runtime cannot safely
materialize them.

The tests exercise the pure-Python gate helper
:func:`phenotypic.gui.builder._callbacks._filter_blocking_issues`
plus the toast builder
:func:`phenotypic.gui.builder._callbacks._gate_toast_for_issue` so
the gate's behaviour can be asserted without booting Dash.  Each
test constructs a JSON-shaped DAG state (the
:func:`~phenotypic.gui.builder._state.state_to_json` output) and
passes it through the gate, mirroring the way the Dash callbacks
invoke the helpers at runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.gui.builder._callbacks import (
    _filter_blocking_issues,
    _gate_toast_for_issue,
    _write_pipeline_config,
)
from phenotypic.sdk_ import CONFIG_SUFFIX_PIPELINE, ensure_typed_json_suffix
from phenotypic.gui.builder._state import (
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    state_to_json,
)
from phenotypic.gui.builder._validation import Issue, validate

from .conftest import _make_op_info


# ---------------------------------------------------------------------------
# Helpers (mirror tests/unit/gui/builder/test_validation.py shape)
# ---------------------------------------------------------------------------


def _new_block(class_name: str, **kwargs: Any) -> BlockNode:
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


def test_write_pipeline_config_normalizes_legacy_json_filename(tmp_path):
    """Builder save writes the typed pipeline suffix even for legacy filename input."""
    raw_target = tmp_path / "my_pipeline.json"
    typed_target = ensure_typed_json_suffix(raw_target, CONFIG_SUFFIX_PIPELINE)
    pipeline = ImagePipeline(ops=[OtsuDetector()])

    saved_path = _write_pipeline_config(pipeline, raw_target)

    assert saved_path == typed_target
    assert typed_target.exists()
    assert not raw_target.exists()


def _state_with_clean_scope() -> Dict[str, Any]:
    """Build a JSON state with a single image-edge linear chain.

    The InputImage auto-seeds when the scope is constructed, so this
    fixture results in ``InputImage → GaussianBlur`` — a clean chain
    with no validation issues.
    """

    scope = _DagBuilderScope()
    a = _new_block("GaussianBlur")
    scope.blocks.append(a)
    scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
    state = _DagBuilderState(root=scope)
    return state_to_json(state)


def _state_with_blocking_stub() -> Dict[str, Any]:
    """Build a state with a blocking ``stub`` issue (unreachable block).

    Adds an orphan ``GaussianBlur`` block with no edges; the validator
    emits ``Issue(kind="stub", severity="error")`` because the block
    is not reachable from the auto-seeded InputImage.
    """

    scope = _DagBuilderScope()
    orphan = _new_block("GaussianBlur")
    scope.blocks.append(orphan)
    state = _DagBuilderState(root=scope)
    return state_to_json(state)


def _state_with_unsupported_linear_fork() -> Dict[str, Any]:
    """Build a state the defensive linear map cannot safely edit."""

    scope = _DagBuilderScope()
    a = _new_block("GaussianBlur")
    b = _new_block("GaussianBlur")
    scope.blocks.extend([a, b])
    scope.edges.append(_image_edge(scope.blocks[0].block_id, a.block_id))
    scope.edges.append(_image_edge(scope.blocks[0].block_id, b.block_id))
    state = _DagBuilderState(root=scope)
    return state_to_json(state)


def _state_with_advisory_only(monkeypatch: Any) -> Dict[str, Any]:
    """Build a state whose only validation finding is advisory.

    Stage-order advisories are emitted when an image-flow edge points
    from a meas/post-stage block down to an ops-stage block (Rule 7).
    To produce a state with only-advisory issues, we monkeypatch the
    validation module's registry so the chain
    ``InputImage → meas → ops`` flips the advisory without any
    blocking error.
    """

    # Provide a fake registry whose entries classify the source as
    # a MeasureFeatures subclass so Rule 7 considers it stage='meas';
    # the downstream block stays 'ops' (default).
    from phenotypic.abc_ import MeasureFeatures

    class _StubMeas(MeasureFeatures):  # pragma: no cover - dataclass shell
        pass

    info_meas = _make_op_info(
        "StubMeas",
        parameters={},
        category="MeasureFeatures",
    )
    info_meas.cls = _StubMeas  # override the stub class so issubclass passes
    info_ops = _make_op_info("StubOp", parameters={})

    class _FakeReg:
        def get(self, name: str) -> Any:
            if name == "StubMeas":
                return info_meas
            if name == "StubOp":
                return info_ops
            return None

    monkeypatch.setattr(
        "phenotypic.gui.builder._validation.get_registry", lambda: _FakeReg()
    )

    scope = _DagBuilderScope()
    src = _new_block("StubMeas")
    tgt = _new_block("StubOp")
    scope.blocks.extend([src, tgt])
    scope.edges.append(_image_edge(scope.blocks[0].block_id, src.block_id))
    scope.edges.append(_image_edge(src.block_id, tgt.block_id))
    state = _DagBuilderState(root=scope)
    return state_to_json(state)


# ---------------------------------------------------------------------------
# Run preview gating
# ---------------------------------------------------------------------------


class TestRunPreviewGating:
    """The Run preview button must abort when any blocking issue exists."""

    def test_run_preview_filtered_to_severity_error(self) -> None:
        """``_filter_blocking_issues`` returns only severity=error issues."""

        state_data = _state_with_blocking_stub()
        errors = _filter_blocking_issues(state_data)
        assert errors, "Expected at least one blocking issue"
        assert all(i.severity == "error" for i in errors), (
            f"Expected only severity=error; got "
            f"{[(i.kind, i.severity) for i in errors]}"
        )
        # The seeded scope's only blocking issue should be the stub.
        assert any(i.kind == "stub" for i in errors)

    def test_run_preview_aborts_when_any_error_issue(self) -> None:
        """Run preview gate aborts when ``_filter_blocking_issues`` is non-empty.

        Simulates the callback's pre-check: load the state, ask for
        blocking issues, and assert that the toast message names the
        offence so the user knows what to fix.
        """

        state_data = _state_with_blocking_stub()
        errors = _filter_blocking_issues(state_data)
        assert errors
        toast = _gate_toast_for_issue("run preview", errors[0])
        # The toast tuple is (is_open, children, icon, header).
        is_open, message, icon, header = toast
        assert is_open is True
        assert "Cannot run preview" in message
        assert errors[0].kind in message
        # The toast must use the error / "danger" style so the gate
        # surfaces in red, not the success colour.
        assert icon == "danger"
        assert header == "Validation"

    def test_run_preview_unblocked_by_advisory_only(
        self, monkeypatch: Any
    ) -> None:
        """Advisory-only states return an empty blocking-issue list.

        Spec §5.6: "Advisory hints (severity=advisory) never block
        these."  So even though ``validate(state)`` returns advisory
        records, ``_filter_blocking_issues`` must filter them out and
        the gate stays green.
        """

        state_data = _state_with_advisory_only(monkeypatch)
        # Sanity check: the unfiltered validator does emit at least
        # one advisory (otherwise the test is a no-op).
        all_issues = _load_issues_from_state(state_data)
        assert any(
            i.severity == "advisory" for i in all_issues
        ), "Expected at least one advisory issue from the fixture"

        # The filtered list is empty so the Run preview gate would
        # not short-circuit.
        errors = _filter_blocking_issues(state_data)
        assert errors == []

    def test_run_preview_passes_on_clean_state(self) -> None:
        """A clean linear chain has no blocking issues; the gate is green."""

        state_data = _state_with_clean_scope()
        errors = _filter_blocking_issues(state_data)
        assert errors == []

    def test_run_preview_blocks_unsupported_linear_shape(self) -> None:
        """The fixed linear map's unsupported state gates preview."""

        errors = _filter_blocking_issues(_state_with_unsupported_linear_fork())
        assert any(i.kind == "unsupported_linear" for i in errors)


# ---------------------------------------------------------------------------
# Save pipeline gating (mirrors Run preview)
# ---------------------------------------------------------------------------


class TestSavePipelineGating:
    """The Save pipeline confirm button must obey the same gate."""

    def test_save_pipeline_filtered_to_severity_error(self) -> None:
        """``_filter_blocking_issues`` is used identically by the Save path."""

        state_data = _state_with_blocking_stub()
        errors = _filter_blocking_issues(state_data)
        assert errors
        assert all(i.severity == "error" for i in errors)

    def test_save_pipeline_aborts_when_any_error_issue(self) -> None:
        """The Save toast names the action verbatim so it's distinct from Run."""

        state_data = _state_with_blocking_stub()
        errors = _filter_blocking_issues(state_data)
        toast = _gate_toast_for_issue("save pipeline", errors[0])
        is_open, message, icon, header = toast
        assert is_open is True
        assert "Cannot save pipeline" in message
        assert "run preview" not in message  # No cross-talk between actions
        assert icon == "danger"
        assert header == "Validation"

    def test_save_pipeline_unblocked_by_advisory_only(
        self, monkeypatch: Any
    ) -> None:
        """Advisory-only states allow Save to proceed."""

        state_data = _state_with_advisory_only(monkeypatch)
        assert _filter_blocking_issues(state_data) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGatingEdgeCases:
    """Defensive paths the gate must tolerate without raising."""

    def test_filter_blocking_issues_none_state(self) -> None:
        """``None`` state data (first paint, before any publish) returns []."""

        assert _filter_blocking_issues(None) == []

    def test_filter_blocking_issues_legacy_state_passes_through(self) -> None:
        """Legacy linear-list state has no DAG validation; gate is transparent.

        The legacy schema has no ``selected_block_id`` attribute, so
        :func:`validate` can't be run.  The gate must return an empty
        list so the legacy GUI flag-off path never accidentally
        blocks Run / Save.
        """

        # Synthesise a legacy-looking payload (no _schema, has "nodes").
        legacy_state: Dict[str, Any] = {
            "root": {"nodes": [], "breadcrumb": []},
            "breadcrumb": [],
            "selected_node_id": None,
            "inspector_focus_aux": None,
        }
        assert _filter_blocking_issues(legacy_state) == []

    def test_filter_blocking_issues_corrupt_state_returns_empty(self) -> None:
        """Bad state shape (parse fails) returns [] rather than raising."""

        bad_state: Dict[str, Any] = {"_schema": "dag", "root": "this-is-not-a-dict"}
        # Should not raise — the helper logs and returns [].
        assert _filter_blocking_issues(bad_state) == []


# ---------------------------------------------------------------------------
# Internal helper used only by this module
# ---------------------------------------------------------------------------


def _load_issues_from_state(state_data: Dict[str, Any]) -> List[Issue]:
    """Re-run validate on the raw state for assertion purposes only."""

    from phenotypic.gui.builder._state import state_from_json

    state = state_from_json(state_data)
    return validate(state)
