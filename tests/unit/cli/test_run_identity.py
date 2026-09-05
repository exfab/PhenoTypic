"""`restart_epoch`: the one tracked counter, and the fence it buys.

Spec §5.1 D4. Three groups, and the middle one is the reason the first exists.

**The counter** (`read_restart_epoch` / `bump_restart_epoch`) is the only
tracked value this design adds, and it is worthless if a restart resets it --
the whole point is to distinguish "deliberately fresh attempt" from "same
config again", which is exactly what a restart is.

**Rule 2's first half** (`_live_authority`): an authority counts only when it
reports work in flight *for the current identity*. P1 shipped the second half
-- the pid probe -- and could not build the first, because before this counter
existed the identity and the authority were read from the same file and the
comparison would have been a value against itself.

**The asymmetry** is deliberate and tested in both directions: the lifecycle
record is epoch-fenced, the GUI owner record is not. See
`test_a_live_gui_owner_still_reports_active_across_a_restart` for why that is
a decision rather than an omission.

**A note on what Task 1 does and does not write.** `bump_restart_epoch`
persists `.phenotypic/restart_epoch.json`; `processing_state.json`'s
`config.restart_epoch` -- which is what `RunIdentity.restart_epoch` reads --
is written by P2 Task 3's minting. So the fence tests here set the config
value explicitly rather than expecting a bump to move it. That the two homes
can differ is the design (CONFLICT-1), not a gap in the fixture: one is *the
counter*, the other is *the value this state was minted under*.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from phenotypic._cli._cli_identity import (
    bump_restart_epoch,
    read_restart_epoch,
)
from phenotypic.sdk_ import (
    clear_machine_state,
    gui_launch_owner_path,
    phenotypic_cache_dir,
    resolve_processing_state_path,
    resolve_run_state,
    restart_epoch_path,
    slurm_lifecycle_path,
)


@pytest.fixture
def incomplete_run(tmp_path):
    """A run whose verdict is `incomplete` unless an authority says otherwise.

    `incomplete` is the discriminator every fence test here needs: with a live
    authority the ladder returns `active`, and without one it falls through.
    A complete run would report `complete` either way and prove nothing.
    """
    from tests._output_layout import build_incomplete_run

    return build_incomplete_run(tmp_path)


def _set_config_restart_epoch(output_dir: Path, epoch: int) -> None:
    """Write `config.restart_epoch`, which P2 Task 3's minting will write.

    Edited directly rather than through a helper so this file does not depend
    on a writer that does not exist yet, and so the value under test is
    visible in the test that sets it.
    """
    path = resolve_processing_state_path(output_dir)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["config"]["restart_epoch"] = epoch
    path.write_text(json.dumps(document), encoding="utf-8")


def _publish_lifecycle(output_dir: Path, *, generation: str) -> dict:
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
    )

    return initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="slurm"
    )


def _lifecycle_record(output_dir: Path) -> dict:
    return json.loads(
        slurm_lifecycle_path(output_dir).read_text(encoding="utf-8")
    )


def _claim_as_live_gui(output_dir: Path) -> None:
    """Write a GUI owner record naming THIS process, which is alive.

    `os.getpid()` rather than a fabricated number: `_live_authority` probes
    the pid, so a record naming a dead process is refused by rule 2's *second*
    half and would make the epoch question unreachable.
    """
    path = gui_launch_owner_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": "running", "pid": os.getpid(), "generation": "gui-1"}
        ),
        encoding="utf-8",
    )


# -------------------------------------------------- the digest, D-C / §5.4


def test_scientific_config_digest_is_the_work_id_digest_itself():
    """D-C / §5.4: "not a new digest ... reused verbatim".

    §5.4's argument is that if the generation and `work_id` could disagree
    about what counts as scientific configuration, a change could invalidate
    per-image proofs without minting a new generation, or the reverse.
    Identity is the strongest form of agreement available, so this is an `is`
    check rather than an equality one: an equal-but-separate function is equal
    today and one edit away from not being, and nothing would fail at the
    moment it stopped.
    """
    from phenotypic._cli._cli_failure_tracker import (
        processing_configuration_digest,
    )
    from phenotypic._cli._cli_identity import scientific_config_digest

    assert scientific_config_digest is processing_configuration_digest


# ------------------------------------------------------------- the counter


def test_restart_epoch_survives_clear_machine_state(tmp_path):
    """D4. The counter is worthless if a restart resets it: the whole point is
    to distinguish "deliberately fresh attempt" from "same config again", and
    a restart is exactly the first."""
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    assert read_restart_epoch(tmp_path) == 0
    assert bump_restart_epoch(tmp_path) == 1
    assert bump_restart_epoch(tmp_path) == 2

    clear_machine_state(tmp_path)

    assert read_restart_epoch(tmp_path) == 2, (
        "clear_machine_state destroyed the restart epoch; the fence it exists "
        "for cannot survive the operation it exists to fence"
    )


def test_reading_a_corrupt_restart_epoch_is_zero_not_an_error(tmp_path):
    """INV-VERDICT's degrade half. A restart must not be blocked by an
    unparseable counter -- reading 0 understates the restarts and so fails to
    fence a stale worker, which is the pre-counter status quo. Raising would
    make one bad byte a reason the user cannot restart at all."""
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    restart_epoch_path(tmp_path).write_text("{not json", encoding="utf-8")

    assert read_restart_epoch(tmp_path) == 0


def test_a_boolean_is_not_a_restart_epoch(tmp_path):
    """`True` is an `int` in Python, so an unguarded reader accepts it as
    epoch 1 -- a fence silently advanced by a type error."""
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    restart_epoch_path(tmp_path).write_text(
        json.dumps({"restart_epoch": True}), encoding="utf-8"
    )

    assert read_restart_epoch(tmp_path) == 0


def test_a_failed_write_raises_rather_than_returning_quietly(tmp_path):
    """The asymmetry with `read_restart_epoch`, which degrades to 0.

    A silently swallowed write failure is *worse* than the pre-counter status
    quo: the next invocation reads the stale epoch, mints the generation the
    abandoned workers are already holding, and the fence passes for exactly
    the workers it exists to exclude. Reading a missing fence is recoverable;
    failing to write one is not.
    """
    cache = phenotypic_cache_dir(tmp_path)
    cache.mkdir(parents=True)
    os.chmod(cache, 0o500)
    try:
        if os.access(cache, os.W_OK):
            pytest.skip("cannot drop write permission here (running as root?)")

        with pytest.raises(OSError):
            bump_restart_epoch(tmp_path)
    finally:
        os.chmod(cache, 0o700)


# ------------------------------------------------- the writer, on its own


def test_the_lifecycle_record_carries_the_epoch_current_at_publication(
    tmp_path,
):
    """The writer's half of rule 2, pinned WITHOUT going through the fence.

    If this only held via `_live_authority`, a later change to the reader
    could make it vacuous and nothing would say so. The epoch is read at
    publication rather than passed in precisely so it cannot be a caller's
    belief -- so the test bumps the counter *between* two publications and
    asserts the second record moved with it.
    """
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    _publish_lifecycle(tmp_path, generation="gen-1")
    assert _lifecycle_record(tmp_path)["restart_epoch"] == 0

    bump_restart_epoch(tmp_path)
    slurm_lifecycle_path(tmp_path).unlink()
    _publish_lifecycle(tmp_path, generation="gen-2")

    assert _lifecycle_record(tmp_path)["restart_epoch"] == 1


def test_an_existing_active_fence_is_not_re_dated(tmp_path):
    """Re-publishing the same generation returns the standing record.

    Re-stamping would silently re-date an old fence to the current epoch,
    which is the precise failure the fence exists to prevent: a worker from
    before the restart would look current again.
    """
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    _publish_lifecycle(tmp_path, generation="gen-1")
    bump_restart_epoch(tmp_path)

    _publish_lifecycle(tmp_path, generation="gen-1")

    assert _lifecycle_record(tmp_path)["restart_epoch"] == 0


# ------------------------------------------------------ rule 2, first half


def test_a_current_authority_still_reports_the_run_active(incomplete_run):
    """The control. Every other fence test asserts something is refused, and
    all of them would pass against an implementation that refused
    everything -- including the one that reports no run as active, ever."""
    _publish_lifecycle(incomplete_run, generation="gen-1")
    _set_config_restart_epoch(incomplete_run, 0)

    state = resolve_run_state(incomplete_run, depth="deep")

    assert state.completion == "active"


def test_a_pre_restart_authority_does_not_report_the_run_active(
    incomplete_run,
):
    """Rule 2's first half, and the failure it excludes.

    A `--restart` mints a new epoch; a worker from the previous epoch is
    still draining and its lifecycle record still says `active`. Without the
    fence, rule 2 fires and the run looks alive on the strength of a worker
    the restart already abandoned -- a stale authority outranking a valid
    verdict, in the one direction P1 could not construct.
    """
    _publish_lifecycle(incomplete_run, generation="gen-1")
    _set_config_restart_epoch(incomplete_run, 1)

    state = resolve_run_state(incomplete_run, depth="deep")

    assert state.completion != "active", (
        "a lifecycle record from a superseded epoch reported the run alive"
    )


def test_a_record_without_an_epoch_still_counts_on_an_unrestarted_run(
    incomplete_run,
):
    """Backward compatibility, in the direction that must not regress.

    Records written before this field existed read as epoch 0. On a
    never-restarted run that is still current, so an existing SLURM launch
    must not be fenced by an upgrade.

    ``pop(..., None)`` rather than ``del``: the subject is *a record with no
    epoch*, and a ``del`` would raise -- coupling this test to the writer
    still stamping the field, which is a different test's job.
    """
    _publish_lifecycle(incomplete_run, generation="gen-1")
    record = _lifecycle_record(incomplete_run)
    record.pop("restart_epoch", None)
    slurm_lifecycle_path(incomplete_run).write_text(
        json.dumps(record), encoding="utf-8"
    )
    _set_config_restart_epoch(incomplete_run, 0)

    assert resolve_run_state(incomplete_run, depth="deep").completion == (
        "active"
    )


def test_a_record_without_an_epoch_is_fenced_on_a_restarted_run(
    incomplete_run,
):
    """The other direction of the same default, and the one that matters.

    ``_record_restart_epoch`` degrades a missing or corrupt field to ``0``,
    and the direction is the whole point: reading ``0`` makes a record look
    *older* than it may be, so a doubtful authority is **fenced** rather than
    believed. That moves the verdict away from ``active`` and toward
    ``incomplete`` -- INV-VERDICT's direction.

    Without this, the paired test above is satisfied by a default that
    degrades *upward*: ``sys.maxsize`` would also let a pre-field record count
    on an unrestarted run, while believing every stale authority on every
    restarted one. Both tests are needed to pin a default; one pins only that
    it exists.
    """
    _publish_lifecycle(incomplete_run, generation="gen-1")
    record = _lifecycle_record(incomplete_run)
    record.pop("restart_epoch", None)
    slurm_lifecycle_path(incomplete_run).write_text(
        json.dumps(record), encoding="utf-8"
    )
    _set_config_restart_epoch(incomplete_run, 1)

    assert resolve_run_state(incomplete_run, depth="deep").completion != (
        "active"
    ), "a record predating the epoch field was believed on a restarted run"


# --------------------------------------------- the asymmetry, pinned as one


def test_a_live_gui_owner_still_reports_active_across_a_restart(
    incomplete_run,
):
    """**The GUI owner record is deliberately NOT epoch-fenced.**

    This is the one place the fence covers half its surface on purpose, so it
    is the one place a future contributor "completes" the job by stamping an
    epoch onto the owner record too -- a one-line change that reads as
    finishing what Task 1 started, and that nothing else here would catch.

    Why the asymmetry is right: the owner record is a *local process* claim,
    already believed only while the pid it names is alive (P1's CAN-24 probe).
    That is a **stronger** check than an epoch comparison, not a weaker one --
    it asks whether the process exists rather than whether a number matches.
    A GUI still running across a restart is genuinely still running, which is
    not the stale-authority case the lifecycle fence is for. Fencing it would
    kill a live process's claim on the strength of a counter it never read.
    """
    _claim_as_live_gui(incomplete_run)
    _set_config_restart_epoch(incomplete_run, 3)

    state = resolve_run_state(incomplete_run, depth="deep")

    assert state.completion == "active", (
        "a live GUI owner was fenced by the restart epoch; the owner record "
        "is a process claim and is bounded by the pid probe, not the counter"
    )
