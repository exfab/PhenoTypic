#!/usr/bin/env python
"""Re-derive the load-bearing claims behind the JournalStorage tune backend.

The MCP-server spec (``docs/superpowers/specs/2026-08-12-phenotypic-mcp-server``)
proposes replacing the Postgres requirement for distributed SLURM tune studies
with Optuna's ``JournalStorage`` over a shared filesystem. These claims carry
that design, and a reader would otherwise take them on faith:

  C1  Optuna exposes ``JournalStorage`` + a file backend, and the *symlink* lock
      (the NFS-safe one) is selectable.
  C2a N concurrent OS processes can ask/tell against one journal-backed study
      without losing or duplicating trials. This is the claim that replaces
      "stand up a Postgres server".
  C2b **NEGATIVE CONTROL.** Re-runs C2a with a no-op lock that provides zero
      mutual exclusion. If that *also* passes, then C2a on this filesystem is
      measuring OS write atomicity, not the lock, and proves nothing about
      whether ``JournalFileSymlinkLock`` does any work here.
  C3  ``JournalStorage`` does not implement the heartbeat API, and PhenoTypic's
      duck-typed probes therefore degrade to "no heartbeat" rather than raising.
  C4  ``optuna.storages.fail_stale_trials`` is safe to call on a journal-backed
      study — PhenoTypic calls it unconditionally before every ask.
  C5  A ``journal://`` URL is NOT resolvable by Optuna's own string handling, so
      a scheme-dispatch layer is mandatory at every storage construction site.
  C6  The journal's write throughput exceeds a realistic fleet's TRIAL rate by
      orders of magnitude. Per-worker write rate does fall under lock contention
      (measured: ~4.7x from 1 to 16 workers), but the lock is held for a journal
      append, not for image evaluation — so the comparison that decides whether
      Postgres is still needed is write-rate vs trial-rate, not write-rate vs
      itself.

Why C2b exists
--------------
An earlier version of this script asserted C2a alone and was reported as
passing. Mutation testing showed it could not fail: replacing the lock with a
no-op still yielded 60/60 trials with no loss. ``JournalFileBackend.append_logs``
does ``open(path, "ab")`` → one ``write()`` → ``fsync()``, and POSIX guarantees
``O_APPEND`` write atomicity on a local filesystem regardless of any
application-level lock. So on APFS/ext4 the test cannot discriminate. C2b makes
that self-evident instead of silently flattering the design.

Read the DISCRIMINATION verdict, not just the ok/FAIL lines.

Depends only on the stdlib + optuna. Never imports ``phenotypic``.

Usage
-----
  uv run python .../optuna_journal_storage.py
  uv run python .../optuna_journal_storage.py --dir /path/on/shared/fs
  uv run python .../optuna_journal_storage.py --dir /nfs/... --require-discrimination

``--require-discrimination`` is the gate for L1 (§7): on the target cluster
filesystem, the run must show that a broken lock actually loses trials there.
Without that, a green C2a is not evidence.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

STUDY_NAME = "tune_cost_v1"  # mirrors phenotypic's _STUDY_NAME constant
N_PROCS = 4
TRIALS_PER_PROC = 15


def _fail(claim: str, detail: str) -> NoReturn:
    print(f"FAIL [{claim}] {detail}")
    sys.exit(1)


def _ok(claim: str, detail: str) -> None:
    print(f"ok   [{claim}] {detail}")


class _NoOpLock:
    """A lock providing zero mutual exclusion — the negative control.

    Matches ``BaseJournalFileLock``'s duck-typed surface (``acquire``/``release``).
    If the suite passes with this installed, the suite is not testing the lock.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def acquire(self) -> bool:
        return True

    def release(self) -> None:
        return None


def _make_storage(journal_path: Path, *, lock_mode: str):
    """Build a journal storage under the requested lock."""
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock

    if lock_mode == "symlink":
        lock = JournalFileSymlinkLock(str(journal_path))
    elif lock_mode == "noop":
        lock = _NoOpLock(str(journal_path))
    else:  # pragma: no cover
        raise ValueError(f"unknown lock_mode {lock_mode!r}")
    return optuna.storages.JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock))


def _worker(journal_path_str: str, n_trials: int, tag: int, lock_mode: str) -> None:
    """One process: attach to the shared study and drain ``n_trials`` ask/tells.

    Mirrors PhenoTypic's ask/tell engine loop rather than ``study.optimize`` —
    that is the access pattern the real workers use.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = _make_storage(Path(journal_path_str), lock_mode=lock_mode)
    study = optuna.create_study(
        storage=storage, study_name=STUDY_NAME, load_if_exists=True, direction="minimize"
    )
    for i in range(n_trials):
        trial = study.ask()
        x = trial.suggest_float("x", 0.0, 1.0)
        # Stamp provenance BEFORE telling: journal storage rejects any update to
        # a finished trial (UpdateFinishedTrialError), and PhenoTypic likewise
        # stamps its ``pheno_*`` user_attrs on the live trial.
        trial.set_user_attr("proc_tag", tag)
        trial.set_user_attr("seq", i)
        study.tell(trial, (x - 0.3) ** 2)


def _run_concurrent(tmp: Path, *, lock_mode: str, name: str) -> tuple[bool, str]:
    """Run the concurrent fan-out. Returns (all_writes_survived, detail)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    journal = tmp / f"{name}.log"

    # Pre-create the study exactly as PhenoTypic's submitter does.
    pre = _make_storage(journal, lock_mode=lock_mode)
    optuna.create_study(
        storage=pre, study_name=STUDY_NAME, load_if_exists=True, direction="minimize"
    )

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(str(journal), TRIALS_PER_PROC, tag, lock_mode))
        for tag in range(N_PROCS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)

    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        return False, f"worker exit codes {bad}"

    try:
        study = optuna.load_study(
            storage=_make_storage(journal, lock_mode=lock_mode), study_name=STUDY_NAME
        )
        trials = study.trials
    except Exception as exc:  # noqa: BLE001 - a corrupt journal is a legitimate outcome here
        return False, f"journal unreadable after the run: {exc!r}"

    expected = N_PROCS * TRIALS_PER_PROC
    if len(trials) != expected:
        return False, f"{len(trials)} trials persisted, expected {expected}"

    numbers = [t.number for t in trials]
    if len(set(numbers)) != len(numbers):
        return False, "duplicate trial numbers"

    per_tag: dict[int, set] = {}
    for t in trials:
        tag = t.user_attrs.get("proc_tag")
        seq = t.user_attrs.get("seq")
        if tag is None or seq is None:
            return False, f"trial {t.number} lost its user_attrs"
        per_tag.setdefault(int(tag), set()).add(seq)
    if len(per_tag) != N_PROCS:
        return False, f"only {len(per_tag)} of {N_PROCS} processes have surviving trials"
    for tag, seqs in sorted(per_tag.items()):
        if seqs != set(range(TRIALS_PER_PROC)):
            return False, f"process {tag} lost writes: got {sorted(seqs)}"

    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) != expected:
        return False, f"{len(completed)} COMPLETE, expected {expected}"

    return True, f"{expected} trials persisted intact"


def claim_1_api_surface() -> None:
    import optuna

    try:
        from optuna.storages.journal import (  # noqa: F401
            JournalFileBackend,
            JournalFileSymlinkLock,
        )
    except Exception as exc:  # noqa: BLE001
        _fail("C1", f"journal backend/lock not importable: {exc!r}")
    if not hasattr(optuna.storages, "JournalStorage"):
        _fail("C1", "optuna.storages.JournalStorage missing")
    _ok("C1", f"optuna {optuna.__version__}: JournalStorage + JournalFileSymlinkLock present")


def claim_2_concurrency_with_control(tmp: Path, *, require_discrimination: bool) -> None:
    locked_ok, locked_detail = _run_concurrent(tmp, lock_mode="symlink", name="locked")
    if not locked_ok:
        _fail("C2a", f"symlink-locked run did not survive: {locked_detail}")
    _ok("C2a", f"{N_PROCS} processes x {TRIALS_PER_PROC} trials — {locked_detail}")

    noop_ok, noop_detail = _run_concurrent(tmp, lock_mode="noop", name="noop")
    if noop_ok:
        print(
            "\n"
            "  DISCRIMINATION: NONE.\n"
            "  The no-op-lock control ALSO passed on this filesystem "
            f"({noop_detail}).\n"
            "  JournalFileBackend.append_logs does open(path,'ab') -> write() -> fsync(),\n"
            "  and POSIX guarantees O_APPEND atomicity on a local filesystem regardless\n"
            "  of any application-level lock. So C2a here measures OS write atomicity,\n"
            "  NOT JournalFileSymlinkLock. C2a is UNPROVEN on this filesystem.\n"
            "  Re-run with --dir on the target cluster's shared mount before trusting it.\n"
        )
        if require_discrimination:
            _fail(
                "C2b",
                "negative control passed — this filesystem cannot test the lock, "
                "so a green C2a is not evidence (--require-discrimination was set)",
            )
        _ok("C2b", "negative control ran; see DISCRIMINATION note above")
        return

    _ok(
        "C2b",
        f"negative control FAILED as required ({noop_detail}) — on this filesystem the "
        "lock is doing real work, so C2a is meaningful evidence",
    )


def claim_3_heartbeat_absent_and_probes_degrade(tmp: Path) -> None:
    """JournalStorage has no heartbeat; PhenoTypic's duck-typed probes must no-op.

    Reproduces the probes verbatim from
    ``src/phenotypic/tune/strategy/_optuna.py`` (``_heartbeat_interval`` at :57,
    ``_record_heartbeat`` at :64) rather than importing them.
    """
    import optuna

    storage = _make_storage(tmp / "hb.log", lock_mode="symlink")
    rdb = optuna.storages.RDBStorage(
        url=f"sqlite:///{tmp / 'hb.db'}", heartbeat_interval=60, grace_period=180
    )

    def heartbeat_interval(s):  # _optuna.py:57-61
        getter = getattr(s, "get_heartbeat_interval", None)
        interval = getter() if callable(getter) else getattr(s, "heartbeat_interval", None)
        return int(interval) if isinstance(interval, int) and interval > 0 else None

    def record_heartbeat(s, trial_id):  # _optuna.py:64-69
        recorder = getattr(s, "record_heartbeat", None)
        if not callable(recorder):
            return "declined"
        recorder(trial_id)
        return "recorded"

    if heartbeat_interval(storage) is not None:
        _fail("C3", "journal storage unexpectedly reports a heartbeat interval")
    if heartbeat_interval(rdb) != 60:
        _fail("C3", f"RDBStorage heartbeat interval {heartbeat_interval(rdb)!r}, expected 60")
    if record_heartbeat(storage, 1) != "declined":
        _fail("C3", "journal storage unexpectedly accepted record_heartbeat")

    _ok(
        "C3",
        "journal: no heartbeat (probes decline, no exception); RDB: interval=60 "
        "— stale-trial reclamation is LOST under the journal backend",
    )


def claim_4_fail_stale_trials_is_safe(tmp: Path) -> None:
    """PhenoTypic calls fail_stale_trials before every ask; it must not raise."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = _make_storage(tmp / "stale.log", lock_mode="symlink")
    study = optuna.create_study(
        storage=storage, study_name=STUDY_NAME, load_if_exists=True, direction="minimize"
    )
    trial = study.ask()
    study.tell(trial, 0.5)

    fail_stale = getattr(optuna.storages, "fail_stale_trials", None)
    if not callable(fail_stale):
        _fail("C4", "optuna.storages.fail_stale_trials missing")
    try:
        fail_stale(study)
    except Exception as exc:  # noqa: BLE001
        _fail("C4", f"fail_stale_trials raised on a journal-backed study: {exc!r}")

    if len(study.trials) != 1:
        _fail("C4", f"fail_stale_trials mutated trial count to {len(study.trials)}")
    _ok("C4", "fail_stale_trials is a safe no-op on a journal-backed study")


def claim_5_journal_scheme_needs_dispatch(tmp: Path) -> None:
    """A ``journal://`` URL must NOT resolve through Optuna's string handling.

    This is what makes a scheme-dispatch layer mandatory at every storage
    construction site rather than optional sugar.
    """
    import optuna

    url = f"journal:///{tmp / 'scheme.log'}"
    for name, fn in (
        ("RDBStorage", lambda: optuna.storages.RDBStorage(url)),
        ("get_storage", lambda: optuna.storages.get_storage(url)),
    ):
        try:
            fn()
        except Exception:  # noqa: BLE001 - any rejection proves the point
            continue
        _fail("C5", f"optuna.storages.{name} unexpectedly accepted a journal:// URL")

    # And JournalFileBackend construction has a write side effect, which matters
    # for any "read-only" path (the GUI Monitor opens with create=False).
    probe = tmp / "sideeffect.log"
    if probe.exists():
        _fail("C5", "precondition: probe path already exists")
    _make_storage(probe, lock_mode="symlink")
    if not probe.exists():
        _fail("C5", "expected JournalFileBackend to create the journal file")

    _ok(
        "C5",
        "journal:// is rejected by RDBStorage AND get_storage (scheme dispatch is "
        "mandatory); JournalFileBackend.__init__ CREATES the file (a 'read-only' "
        "open has a write side effect)",
    )


def claim_6_throughput_headroom(tmp: Path) -> None:
    """The journal's write rate must exceed a realistic fleet's TRIAL rate.

    Lock contention is real — per-worker write throughput falls as workers are
    added. The question that matters is not whether it falls but whether what
    remains is far above the rate a PhenoTypic fleet actually produces trials,
    because the lock is held for a journal append, NOT for image evaluation.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    per, workers = 25, 8
    journal = tmp / "throughput.log"
    optuna.create_study(
        storage=_make_storage(journal, lock_mode="symlink"),
        study_name=STUDY_NAME,
        load_if_exists=True,
        direction="minimize",
    )
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(str(journal), per, tag, "symlink"))
        for tag in range(workers)
    ]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    dt = max(time.time() - t0, 1e-6)

    if any(p.exitcode != 0 for p in procs):
        _fail("C6", "throughput workers did not all exit cleanly")

    rate = (per * workers) / dt

    # A realistic PhenoTypic trial evaluates >=2 images at seconds each, so a
    # fleet of this size produces trials at roughly workers/7s.
    fleet_trial_rate = workers / 7.0
    headroom = rate / fleet_trial_rate
    if headroom < 10.0:
        _fail(
            "C6",
            f"journal sustains {rate:.0f} trials/s vs a fleet rate of "
            f"{fleet_trial_rate:.2f}/s — only {headroom:.1f}x headroom; contention "
            "would actually matter and Postgres would be the right default",
        )
    _ok(
        "C6",
        f"{workers} workers: journal sustains {rate:.0f} append-trials/s vs a realistic "
        f"fleet rate of ~{fleet_trial_rate:.2f} trials/s (2 images x ~3.4 s each) — "
        f"{headroom:.0f}x headroom. Contention is measurable but irrelevant at the "
        "timescale image evaluation actually runs at.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        default=None,
        help="Run under this directory (point at the target shared filesystem for L1).",
    )
    ap.add_argument(
        "--require-discrimination",
        action="store_true",
        help="Exit non-zero unless the negative control actually fails here (the L1 gate).",
    )
    args = ap.parse_args()

    base = Path(args.dir) if args.dir else None
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pht-journal-validate-", dir=str(base) if base else None))
    print(f"# workdir: {tmp}")
    try:
        claim_1_api_surface()
        claim_2_concurrency_with_control(
            tmp, require_discrimination=args.require_discrimination
        )
        claim_3_heartbeat_absent_and_probes_degrade(tmp)
        claim_4_fail_stale_trials_is_safe(tmp)
        claim_5_journal_scheme_needs_dispatch(tmp)
        claim_6_throughput_headroom(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nAll claims re-derived. Read the DISCRIMINATION verdict above before trusting C2a.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
