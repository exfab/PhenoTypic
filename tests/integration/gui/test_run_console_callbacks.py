"""Phase 6 regression tests for the Run console callback wiring.

Targets the bugs found in the Phase 6 post-impl review:

* H3: Local re-run on the same output dir would `RuntimeError` because
  the prior handle was never reaped (`runner.reap` is caller-driven).
  The Run callback now reaps before starting.
* C2: Validate runs registered with ``mode="local"`` blocked subsequent
  Local runs via the concurrency cap. They now register as
  ``mode="validate"`` and ``_local_run_active`` excludes them.
* H4: ``refresh_recents`` previously took ``RC_INTERVAL_LOG.n_intervals``
  as an Input, walking the sandbox every second. It now subscribes
  only to ``RC_STORE_RECENTS_REFRESH``.

These tests poke the implementation directly (not through Dash's
``/_dash-update-component`` round trip) where possible, since Dash's
callback dispatch isn't the seam that breaks. The aim is fast unit-ish
coverage for the post-review fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from phenotypic.gui.run_console._callbacks import _local_run_active
from phenotypic.gui.run_console._runner import LocalRunner
from phenotypic.gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot


@pytest.fixture()
def runner() -> LocalRunner:
    return LocalRunner()


@pytest.fixture()
def registry() -> RunRegistry:
    return RunRegistry()


# ---------------------------------------------------------------------------
# H3: rerun the same output dir
# ---------------------------------------------------------------------------

def test_local_runner_reap_unblocks_rerun(
    runner: LocalRunner, tmp_path: Path,
) -> None:
    """Reaping a completed handle drops it so the same run_id can re-start.

    Without the reap call, ``runner.start(run_id, ...)`` raises
    ``RuntimeError: run_id already running``. The Phase 6 fix calls
    ``runner.reap(run_id)`` in the Run callback before ``start``.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    argv = [sys.executable, "-c", "print('first')"]
    handle = runner.start("plate-0", argv, output_dir=output_dir)
    handle.process.wait(timeout=5.0)

    # Without reap, start() would refuse the second invocation.
    with pytest.raises(RuntimeError):
        runner.start("plate-0", argv, output_dir=output_dir)

    # Reaping drops the prior handle.
    rc = runner.reap("plate-0")
    assert rc == 0
    handle2 = runner.start("plate-0", argv, output_dir=output_dir)
    handle2.process.wait(timeout=5.0)
    runner.reap("plate-0")


# ---------------------------------------------------------------------------
# C2: validate runs do not block Local runs
# ---------------------------------------------------------------------------

def test_local_run_active_excludes_validate_records(
    runner: LocalRunner, registry: RunRegistry, tmp_path: Path,
) -> None:
    """``_local_run_active`` ignores ``mode="validate"`` records.

    Previously a long-running dry-run probe registered as
    ``mode="local"`` would block the Run button via the concurrency cap.
    The fix tags validate records distinctly so the cap only considers
    real Local runs.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Long-lived subprocess to act as the in-flight validation.
    handle = runner.start(
        "validate-1",
        [sys.executable, "-c", "import time; time.sleep(2)"],
        output_dir=output_dir,
    )
    registry.register(
        RunRecord(
            run_id="validate-1",
            mode="validate",
            output_dir=output_dir,
            rel_path="out",
            status="running",
        )
    )
    try:
        # The validate record is alive but ``_local_run_active`` returns False.
        assert runner.is_running("validate-1") is True
        assert _local_run_active(runner, registry) is False

        # A real Local record DOES make ``_local_run_active`` return True.
        registry.register(
            RunRecord(
                run_id="local-1",
                mode="local",
                output_dir=output_dir,
                rel_path="out",
                status="running",
            )
        )
        # No actual subprocess for local-1 yet, so ``runner.is_running``
        # returns False — the cap is False until a runner-tracked process
        # exists.
        assert _local_run_active(runner, registry) is False

        # Spawn a real subprocess for local-1.
        runner.start(
            "local-1",
            [sys.executable, "-c", "import time; time.sleep(2)"],
            output_dir=output_dir,
        )
        assert _local_run_active(runner, registry) is True
    finally:
        runner.stop("validate-1", grace_seconds=0.1)
        runner.stop("local-1", grace_seconds=0.1)
        runner.reap("validate-1")
        runner.reap("local-1")
        del handle


# ---------------------------------------------------------------------------
# Async SLURM submit infrastructure
# ---------------------------------------------------------------------------

def test_pending_slurm_dict_takes_only_completed_futures() -> None:
    """``_take_pending_slurm`` returns ``None`` for futures still in flight.

    The follow-up callback driven by the log-tail interval relies on this
    contract: a tick that arrives before the submission has resolved must
    leave the dict untouched and return ``None`` so the user keeps seeing
    the "submitting" banner.
    """
    from concurrent.futures import Future

    from phenotypic.gui.run_console._callbacks import (
        _stash_pending_slurm,
        _take_pending_slurm,
    )

    fut: Future[str] = Future()
    _stash_pending_slurm("slurm-pending-test", fut)

    # In flight → returns None, dict still holds the future.
    assert _take_pending_slurm("slurm-pending-test") is None

    # Resolve → next call returns the future and pops it.
    fut.set_result("done")
    taken = _take_pending_slurm("slurm-pending-test")
    assert taken is fut
    assert taken.result() == "done"
    # Already taken → next call returns None.
    assert _take_pending_slurm("slurm-pending-test") is None


# ---------------------------------------------------------------------------
# Shared source-image-root sync
# ---------------------------------------------------------------------------

def test_run_input_dir_builds_shared_source_payload(tmp_path: Path) -> None:
    from phenotypic.gui.run_console._callbacks import (
        _source_payload_for_input_dir,
    )

    plates = tmp_path / "plates"
    plates.mkdir()
    (plates / "plate.tif").write_bytes(b"")
    sandbox = SandboxRoot.from_path(tmp_path)

    payload = _source_payload_for_input_dir(sandbox, str(plates), None)

    assert payload is not None
    assert payload["abs_path"] == str(plates.resolve())
    assert payload["source"] == "run-console"
    assert payload["image_count"] == 1


def test_run_input_dir_does_not_rewrite_matching_shared_source(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.run_console._callbacks import (
        _source_payload_for_input_dir,
    )
    from phenotypic.gui.shell._source_context import source_payload_from_path

    plates = tmp_path / "plates"
    plates.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    current = source_payload_from_path(sandbox, plates, source="manual")

    assert current is not None
    assert _source_payload_for_input_dir(sandbox, str(plates), current) is None


def test_shared_source_initializes_empty_run_input(tmp_path: Path) -> None:
    from phenotypic.gui.run_console._callbacks import _input_dir_from_shared_source
    from phenotypic.gui.shell._source_context import source_payload_from_path

    plates = tmp_path / "plates"
    plates.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, plates, source="manual")

    assert _input_dir_from_shared_source(sandbox, payload, None) == str(
        plates.resolve()
    )


def test_shared_source_does_not_overwrite_non_empty_run_input(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.run_console._callbacks import _input_dir_from_shared_source
    from phenotypic.gui.shell._source_context import source_payload_from_path

    plates = tmp_path / "plates"
    existing = tmp_path / "existing"
    plates.mkdir()
    existing.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, plates, source="manual")

    assert (
        _input_dir_from_shared_source(sandbox, payload, str(existing))
        is None
    )


def test_form_state_includes_resolved_global_metadata_csv(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.run_console._callbacks import _form_inputs_to_state
    from phenotypic.gui.shell._metadata_context import metadata_payload_from_path
    from phenotypic.schema import METADATA

    csv_path = tmp_path / "layout.csv"
    csv_path.write_text(f"{METADATA.IMAGE_NAME},Treatment\nplate_a,control\n")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)

    state = _form_inputs_to_state(
        "/p.json",
        "/in",
        "/out",
        "local",
        [],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        metadata_payload=payload,
        sandbox=sandbox,
    )

    assert state["metadata_csv"] == str(csv_path.resolve())
