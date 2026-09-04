"""The run-state reader: types, identity, verdict matrix, advisories.

Phase 1 Task 2 lands the state types only. The verdict matrix, the depth
behaviour and the degrade half of INV-VERDICT arrive with `resolve_run_state`
in Tasks 4 and 5.
"""

from __future__ import annotations

import dataclasses


def test_the_demoted_sources_live_only_under_diagnostics():
    """Spec §9: a predicate reaching into state.diagnostics is visibly wrong.

    This does not stop someone writing `if state.diagnostics.verified ==
    state.diagnostics.accepted`, but it does pin WHERE the demoted evidence
    lives. manifest counts and the event log were evidence; §4.2 demoted them.
    If they reappear as top-level RunState fields, the demotion has been undone.
    """
    from phenotypic.sdk_ import RunDiagnostics, RunState

    top = {f.name for f in dataclasses.fields(RunState)}
    assert top == {
        "completion",
        "identity",
        "images",
        "advisories",
        "diagnostics",
        "depth",
        "verified_at",
    }
    diag = {f.name for f in dataclasses.fields(RunDiagnostics)}
    assert diag == {"accepted", "verified", "failed"}, (
        "U-5 dropped manifest_completed/manifest_total/event_log_present after "
        "verifying zero consumers survive P6. Carrying demoted evidence into "
        "RunState is what keeps it alive as a quasi-evidence surface."
    )


def test_image_state_stages_carry_no_backfilled_key():
    """D-A: per-store metadata is written at promote time, so there is no
    backfill stage. `stages` stays an open map, so re-adding one later is
    additive -- but nothing in this phase may write or read that key."""
    from phenotypic.sdk_ import ImageState

    state = ImageState(
        work_id="w",
        dataset="d",
        image_stem="s",
        stages={"measured": {"at": "2026-09-03T00:00:00Z"}},
        verdict="verified",
        reason=None,
    )
    assert "backfilled" not in state.stages
