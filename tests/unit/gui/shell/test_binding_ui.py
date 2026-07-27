"""Unit tests for Results/Analysis binding hand-off presentation."""

from __future__ import annotations

import pytest

from phenotypic.gui.shell._binding_ui import (
    binding_error_text,
    binding_ui_state,
)


def test_pending_submission_is_visible_but_not_yet_actionable() -> None:
    """An optimistic submission never invents poll or cancellation authority."""
    state = binding_ui_state(
        {
            "status": "submitting",
            "submission_outcome": "pending",
            "submission_target": "results/large-output",
            "job": {
                "status": "submitting",
                "phase": "submitting",
                "detail": "Submitting the Results binding request.",
                "target": "results/large-output",
                "terminal": False,
                "authoritative": False,
            },
        }
    )

    assert state.status == "Submitting"
    assert state.phase == "Submitting request"
    assert state.progress_value is None
    assert state.progress_label == "Working"
    assert state.cancel_disabled is True
    assert state.poll_disabled is True
    assert "authoritative job identifier" in state.diagnostic
    assert "publication is unchanged" not in state.diagnostic


def test_pending_retry_retains_prior_authoritative_monitor() -> None:
    """A new unacknowledged request does not displace known job controls."""
    state = binding_ui_state(
        {
            "submission_outcome": "pending",
            "submission_target": "results/new-output",
            "job_id": "known-job",
            "poll_path": "/jobs/known-job",
            "cancel_path": "/jobs/known-job",
            "job": {
                "job_id": "known-job",
                "status": "running",
                "phase": "inventory",
                "completed": 40,
                "total": 100,
                "terminal": False,
            },
        }
    )

    assert state.status == "Active"
    assert state.progress_label == "40 of 100"
    assert state.cancel_disabled is False
    assert state.poll_disabled is False
    assert "newer binding request is awaiting acknowledgement" in (
        state.diagnostic
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


def test_unknown_submission_does_not_claim_the_live_pair_is_unchanged() -> None:
    """A response-lost POST is rendered as uncertainty, not terminal failure."""
    payload = {
        "status": "unknown",
        "submission_outcome": "unknown",
        "submission_error": "TypeError: response lost",
        "job": {
            "status": "unknown",
            "phase": "submission_unknown",
            "detail": (
                "The binding request may have been accepted, but its "
                "acknowledgement was not received."
            ),
            "terminal": False,
            "authoritative": False,
        },
    }

    state = binding_ui_state(payload)

    assert state.status == "Unconfirmed"
    assert state.phase == "Submission acknowledgement unavailable"
    assert "could not be confirmed and may have been accepted" in (
        state.diagnostic
    )
    assert "previous Results + Analysis publication is unchanged" not in (
        state.diagnostic
    )
    assert state.cancel_disabled is True
    assert state.poll_disabled is True
    assert "could not be confirmed and may have been accepted" in (
        binding_error_text(payload)
    )


def test_unknown_retry_keeps_prior_active_progress_and_warns() -> None:
    """An ambiguous retry leaves the previously authoritative monitor active."""
    payload = {
        "submission_outcome": "unknown",
        "submission_error": "HTTP 504",
        "job_id": "active-job",
        "poll_path": "/jobs/active-job",
        "cancel_path": "/jobs/active-job",
        "job": {
            "job_id": "active-job",
            "status": "running",
            "phase": "inventory",
            "completed": 40,
            "total": 100,
            "terminal": False,
        },
    }

    state = binding_ui_state(payload)

    assert state.status == "Active"
    assert state.progress_label == "40 of 100"
    assert state.cancel_disabled is False
    assert state.poll_disabled is False
    assert state.diagnostic == (
        "The latest submission could not be confirmed and may have been "
        "accepted (HTTP 504). Continuing to monitor the previously "
        "acknowledged job. Its progress does not establish whether the "
        "unacknowledged request published."
    )
    assert "publication has not changed" not in state.diagnostic
    assert "previous Results + Analysis publication is unchanged" not in (
        state.diagnostic
    )


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        ("failed", "The previously acknowledged job failed"),
        ("cancelled", "The previously acknowledged job was cancelled"),
    ],
)
def test_unknown_retry_never_claims_preservation_after_prior_job_terminal(
    status: str,
    detail: str,
) -> None:
    """A retained job's terminal state cannot resolve an uncertain retry."""

    state = binding_ui_state(
        {
            "submission_outcome": "unknown",
            "submission_error": "response lost",
            "job_id": "prior-job",
            "job": {
                "job_id": "prior-job",
                "status": status,
                "phase": status,
                "detail": f"Prior job {status}.",
                "terminal": True,
            },
        }
    )

    assert detail in state.diagnostic
    assert "unacknowledged request may still have published" in (
        state.diagnostic
    )
    assert "publication cannot be inferred" in state.diagnostic
    assert "publication is unchanged" not in state.diagnostic
    assert "not published" not in state.diagnostic


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
