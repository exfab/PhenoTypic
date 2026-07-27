"""Unit tests for Results/Analysis binding hand-off presentation."""

from __future__ import annotations

import pytest

from phenotypic.gui.shell._binding_ui import (
    binding_error_text,
    binding_ui_state,
)


def test_active_progress_surfaces_phase_counts_and_deduplication() -> None:
    """A running job produces an enabled, pollable progress card."""
    state = binding_ui_state(
        {
            "deduplicated": True,
            "poll_error": "HTTP 503",
            "job": {
                "status": "running",
                "phase": "inventory",
                "detail": "Scanning processing files.",
                "completed": 240,
                "total": 1_000,
                "attempt": 2,
                "cache_hit": True,
                "target": "/sandbox/large-output",
            },
        }
    )

    assert state.status == "Active"
    assert state.phase == "Scanning processing inventory"
    assert (state.progress_value, state.progress_max) == (240, 1_000)
    assert state.progress_label == "240 of 1000"
    assert state.cancel_disabled is False
    assert state.poll_disabled is False
    assert "Reused the active request." in state.diagnostic
    assert "Stable-read attempt 2." in state.diagnostic
    assert "Verified inventory cache hit." in state.diagnostic
    assert "Progress check unavailable: HTTP 503" in state.diagnostic


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("invalid", "Compatibility validation failed"),
        ("stale", "Snapshot consistency changed"),
        ("unavailable", "Binding is unavailable"),
    ],
)
def test_failure_diagnostic_classifies_terminal_cause(
    kind: str,
    expected: str,
) -> None:
    """Terminal diagnostics distinguish compatibility and consistency."""
    payload = {
        "job": {
            "status": "failed",
            "phase": "failed",
            "detail": "Results binding failed.",
            "error_kind": kind,
            "error": "fixture failure",
            "terminal": True,
        }
    }
    state = binding_ui_state(payload)

    assert expected in state.diagnostic
    assert "previous Results + Analysis publication is unchanged" in (
        state.diagnostic
    )
    assert state.cancel_disabled is True
    assert state.poll_disabled is True
    assert binding_error_text(payload) == "fixture failure"


def test_cancelled_and_superseded_explain_non_publication() -> None:
    """Cancellation and supersession explicitly preserve the live pair."""
    cancelled = binding_ui_state(
        {
            "job": {
                "status": "cancelled",
                "phase": "cancelled",
                "detail": "Results binding cancelled.",
                "terminal": True,
            }
        }
    )
    superseded = binding_ui_state(
        {
            "job": {
                "status": "superseded",
                "phase": "superseded",
                "detail": "Superseded.",
                "terminal": True,
            }
        }
    )

    assert "previous Results + Analysis publication is unchanged" in (
        cancelled.diagnostic
    )
    assert "superseded candidate was not published" in superseded.diagnostic


def test_success_reports_atomic_pair_and_fingerprint() -> None:
    """The terminal success names both sessions and the bound revision."""
    state = binding_ui_state(
        {
            "job": {
                "status": "succeeded",
                "phase": "complete",
                "detail": "Published.",
                "target": "/sandbox/small-output",
                "terminal": True,
            },
            "snapshot": {
                "processing_fingerprint": "abcdef0123456789",
            },
            "consistency": {
                "state": "coherent",
                "reasons": ["terminal manifest evidence is internally coherent"],
            },
        }
    )

    assert state.status == "Published"
    assert "small-output published atomically to Results + Analysis" in (
        state.diagnostic
    )
    assert "Snapshot abcdef012345" in state.diagnostic
    assert "Consistency: coherent." in state.diagnostic
