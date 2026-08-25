"""Rename-promote commit protocol: uuid parts, move-aside, sweep, durability."""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_


def _fake_store(root: Path, marker: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(exist_ok=True)
    (root / "0" / "c.0.0.0").write_bytes(b"chunk")
    (root / "zarr.json").write_text(f'{{"marker": "{marker}"}}', encoding="utf-8")
    return root


def test_part_path_is_a_sibling_hidden_directory(tmp_path: Path) -> None:
    final = tmp_path / "plate_01.ome.zarr"
    part = ngff_.new_part_path(final)
    assert part.parent == final.parent
    assert part.name.startswith(".plate_01.ome.zarr.")
    assert part.name.endswith(".part")


def test_part_paths_are_distinct_across_concurrent_writers(tmp_path: Path) -> None:
    """A PID can be reused; a uuid4 cannot. Two writers must never share a dir."""
    final = tmp_path / "plate_01.ome.zarr"
    parts = {ngff_.new_part_path(final) for _ in range(64)}
    assert len(parts) == 64


def test_part_name_carries_no_pid(tmp_path: Path) -> None:
    part = ngff_.new_part_path(tmp_path / "plate_01.ome.zarr")
    assert str(os.getpid()) not in part.name.replace(".part", "")


def test_promote_onto_absent_target(tmp_path: Path) -> None:
    final = tmp_path / "plate_01.ome.zarr"
    part = _fake_store(ngff_.new_part_path(final), "new")
    result = ngff_.promote_store(part, final, fsync=False)
    assert result == final
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'
    assert not part.exists()


def test_promote_replaces_a_non_empty_existing_store(tmp_path: Path) -> None:
    """os.replace onto a non-empty directory raises ENOTEMPTY; the move-aside
    is what makes the promote work at all, on POSIX and on Windows alike."""
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")
    ngff_.promote_store(part, final, fsync=False)
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'


def test_nothing_writes_into_a_promoted_store(tmp_path: Path) -> None:
    """FLOW-5: a promoted store is **replaced wholesale**, never mutated in place.

    This is the invariant three separate subsystems now rest on, none of which
    can detect its violation on its own:

    * per-image completion markers fingerprint a store by its root ``zarr.json``
      alone (Task 3.8), because the promote writes the root **last**;
    * the results viewer's staleness scan is bounded to that same root (user
      ruling 2026-08-20), so an in-place chunk write would be invisible to it;
    * ``valid_staged_store`` treats a parseable root as evidence the whole store
      is complete.

    Each of those is *correct only while this holds*. Add one code path that
    opens an array inside a promoted store for writing and all three start
    lying, with nothing failing to say so -- which is precisely why the guard
    belongs here, at the primitive, rather than in any one of them.

    Proven by identity, not by content: the promoted directory is a **different
    inode** from the one it replaced, and so is every chunk beneath it. A
    merge-in-place implementation would leave the old directory in position with
    new bytes inside it, passing any content assertion while breaking all three
    readers above.
    """
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    before_dir = final.stat().st_ino
    before_chunk = (final / "0" / "c.0.0.0").stat().st_ino

    part = _fake_store(ngff_.new_part_path(final), "new")
    part_dir = part.stat().st_ino
    ngff_.promote_store(part, final, fsync=False)

    assert final.stat().st_ino != before_dir
    assert (final / "0" / "c.0.0.0").stat().st_ino != before_chunk
    # It is the *part* that is now in position -- a rename, not a copy.
    assert final.stat().st_ino == part_dir
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'


def test_promote_leaves_no_trash_behind(tmp_path: Path) -> None:
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")
    ngff_.promote_store(part, final, fsync=False)
    assert [p.name for p in tmp_path.iterdir()] == ["plate_01.ome.zarr"]


def test_bare_os_replace_onto_a_non_empty_directory_still_fails(tmp_path: Path) -> None:
    """Pins the reason the two-step move-aside is mandatory, not defensive."""
    src = _fake_store(tmp_path / "src", "a")
    dst = _fake_store(tmp_path / "dst", "b")
    with pytest.raises(OSError):
        os.replace(src, dst)


def test_sweep_removes_orphan_parts_and_trash(tmp_path: Path) -> None:
    """`min_age_seconds=0` because the fixtures are microseconds old.

    The production default is 6 h — see
    `test_the_sweep_spares_a_young_leftover`, which is the behaviour the age
    guard was added for.
    """
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    _fake_store(dataset / "keep.ome.zarr", "keep")
    _fake_store(dataset / ".keep.ome.zarr.deadbeef.part", "orphan")
    _fake_store(dataset / ".keep.ome.zarr.cafef00d.trash", "orphan")
    removed = ngff_.sweep_orphan_parts(tmp_path / "results", min_age_seconds=0)
    assert removed == 2
    assert (dataset / "keep.ome.zarr").is_dir()
    assert list(dataset.glob("*.part")) == []
    assert list(dataset.glob("*.trash")) == []


def test_sweep_is_idempotent_on_a_clean_tree(tmp_path: Path) -> None:
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    _fake_store(dataset / "keep.ome.zarr", "keep")
    assert ngff_.sweep_orphan_parts(tmp_path / "results", min_age_seconds=0) == 0


def test_the_sweep_spares_a_young_leftover(tmp_path: Path) -> None:
    """The whole point of the age guard: a uuid gives no liveness signal, so
    under a SLURM array a sibling task may be mid-write into this directory."""
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    live = _fake_store(dataset / ".keep.ome.zarr.deadbeef.part", "in flight")
    assert ngff_.sweep_orphan_parts(tmp_path / "results") == 0
    assert live.is_dir()


def test_durable_writes_honour_an_explicit_override(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.durable_writes_enabled(True) is True
    assert ngff_.durable_writes_enabled(False) is False


def test_durable_writes_default_off_locally(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.durable_writes_enabled(None) is False


def test_durable_writes_default_on_under_slurm(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert ngff_.durable_writes_enabled(None) is True


def test_durability_is_describable_for_the_run_start_log(monkeypatch) -> None:
    """The same command carries different guarantees in different places, so
    the resolved mode must be loggable, not merely resolvable."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert ngff_.describe_durability(None) == "durable writes: on (SLURM)"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.describe_durability(None) == "durable writes: off (local)"
    assert ngff_.describe_durability(True) == "durable writes: on (--durable-writes)"
    assert (
        ngff_.describe_durability(False) == "durable writes: off (--no-durable-writes)"
    )


def test_fsync_tree_runs_without_error_on_a_real_store(tmp_path: Path) -> None:
    store = _fake_store(tmp_path / "s.ome.zarr", "x")
    ngff_.fsync_tree(store)


@pytest.mark.skipif(os.name != "nt", reason="Windows path-prefix behaviour")
def test_long_path_prefixes_on_windows(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path).startswith("\\\\?\\")


@pytest.mark.skipif(os.name == "nt", reason="POSIX passthrough")
def test_long_path_is_a_passthrough_on_posix(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path) == str(tmp_path)


def test_store_path_segments_have_no_case_only_collisions() -> None:
    """NTFS is case-insensitive; asserted by test rather than by inspection."""
    segments = [
        ngff_.OME_GROUP,
        ngff_.LABELS_GROUP,
        ngff_.OBJMAP_LABEL,
        *ngff_.SERIES_ORDER,
    ]
    assert len({s.lower() for s in segments}) == len(segments)


# ---------------------------------------------------------------------------
# Crash-safety behaviour of the retry loop itself.
#
# ADDED BEYOND THE PLAN'S TEST BLOCK. A mutation survey of the plan's own
# fifteen promote tests killed the structural mutants (drop the move-aside,
# drop the trash cleanup, PID instead of uuid, unscoped discard glob, drop the
# sweep's age guard) but left every crash-safety mutant alive: collapsing the
# retry loop to one attempt, deleting the rollback, retrying non-retryable
# errors, and ignoring `fsync=True` all kept the suite green. Task 1.5 is a
# crash-safety mechanism, so those are the mutants that matter.
#
# Each test below names the mutant it kills.
# ---------------------------------------------------------------------------


def _replace_shim(monkeypatch, behaviour) -> list[tuple[str, str]]:
    """Install *behaviour* in front of ``os.replace`` and record every call.

    Returns the call log, which is what pins "how many attempts happened".
    """
    calls: list[tuple[str, str]] = []
    real = os.replace

    def shim(src, dst, *args, **kwargs):
        calls.append((str(src), str(dst)))
        outcome = behaviour(len(calls), str(src), str(dst))
        if isinstance(outcome, BaseException):
            raise outcome
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", shim)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    return calls


def test_a_transient_rename_failure_is_retried_not_surfaced(
    tmp_path: Path, monkeypatch
) -> None:
    """Kills the `PROMOTE_RETRY_ATTEMPTS -> 1` mutant.

    ERROR_SHARING_VIOLATION is what a Windows GUI or antivirus holding one of
    the store's ~40 files produces, and it clears on its own. Without the
    retry the promote surfaces it as a hard failure.
    """
    final = tmp_path / "plate_01.ome.zarr"
    part = _fake_store(ngff_.new_part_path(final), "new")
    sharing_violation = OSError(errno.EACCES, "sharing violation")
    sharing_violation.winerror = 32  # type: ignore[attr-defined]

    def behaviour(call_number: int, _src: str, _dst: str):
        return sharing_violation if call_number <= 2 else None

    calls = _replace_shim(monkeypatch, behaviour)
    assert ngff_.promote_store(part, final, fsync=False) == final
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'
    assert len(calls) == 3, "the first two attempts must have been retried"


def test_a_failed_promote_leaves_the_previous_store_in_place(
    tmp_path: Path, monkeypatch
) -> None:
    """Kills the `no rollback` mutant — the data-loss mode.

    The move-aside succeeds and the second rename never does. Without the
    rollback the previous store sits in `.trash` under a name no reader looks
    for, and `final` is simply gone: no copy at any path. The single-file HDF
    rename never had that mode, because a failed `os.replace(tmp, final)` left
    `final` untouched.
    """
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")

    def behaviour(_call_number: int, src: str, dst: str):
        if src.endswith(ngff_.PART_SUFFIX):
            return OSError(errno.ENOTEMPTY, "target busy")
        return None

    _replace_shim(monkeypatch, behaviour)
    with pytest.raises(OSError):
        ngff_.promote_store(part, final, fsync=False)
    assert final.is_dir(), "the previous store must never be left at no path"
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "old"}'


def test_a_hard_failure_is_not_retried_five_times(
    tmp_path: Path, monkeypatch
) -> None:
    """Kills the `retry everything` mutant.

    Retrying a genuine ENOSPC exhausts the backoff budget -- 3.1 s per image,
    an hour across 10k images -- before surfacing an error that was never
    going to clear. It must fail on the first attempt, with the previous store
    rolled back.
    """
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")

    def behaviour(_call_number: int, src: str, dst: str):
        if src.endswith(ngff_.PART_SUFFIX):
            return OSError(errno.ENOSPC, "no space left on device")
        return None

    calls = _replace_shim(monkeypatch, behaviour)
    with pytest.raises(OSError) as caught:
        ngff_.promote_store(part, final, fsync=False)
    assert caught.value.errno == errno.ENOSPC
    # move-aside, failed promote, rollback -- and then it stops.
    assert len(calls) == 3, f"a hard failure was retried: {calls}"
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "old"}'


def test_a_concurrent_promoter_appearing_mid_retry_is_benign(
    tmp_path: Path, monkeypatch
) -> None:
    """Kills the `hoist the existence check out of the loop` mutant.

    This is the exact race the in-loop re-evaluation is documented to close:
    `final` is absent when the promote starts, another writer creates it while
    the first attempt is failing, and a check-then-act done once would skip the
    move-aside forever and hit ENOTEMPTY on every remaining attempt.
    """
    final = tmp_path / "plate_01.ome.zarr"
    part = _fake_store(ngff_.new_part_path(final), "new")

    def behaviour(call_number: int, _src: str, _dst: str):
        if call_number == 1:
            _fake_store(final, "someone else")  # a concurrent promoter lands
            return OSError(errno.ENOENT, "transient")
        return None

    _replace_shim(monkeypatch, behaviour)
    assert ngff_.promote_store(part, final, fsync=False) == final
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'


def test_a_concurrent_winner_after_move_aside_is_reconciled_before_retry(
    tmp_path: Path, monkeypatch
) -> None:
    """A retry must not collide with the preceding attempt's trash.

    Writer A moves the old final aside. Writer B then publishes its own final
    before A can promote its part, so A's part-to-final rename fails. A must
    reconcile only the trash created by that attempt and retry with a fresh
    trash path; reusing the occupied path makes every later move-aside fail.
    """
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "writer A")

    def behaviour(call_number: int, src: str, _dst: str):
        if call_number == 2 and src == str(part):
            _fake_store(final, "writer B")
            return OSError(errno.ENOTEMPTY, "writer B won")
        return None

    calls = _replace_shim(monkeypatch, behaviour)

    assert ngff_.promote_store(part, final, fsync=False) == final
    assert (final / "zarr.json").read_text(encoding="utf-8") == (
        '{"marker": "writer A"}'
    )
    trash_destinations = [
        dst for src, dst in calls if src == str(final) and dst.endswith(".trash")
    ]
    assert len(trash_destinations) == 2
    assert len(set(trash_destinations)) == 2
    assert not any(path.name.endswith(".trash") for path in tmp_path.iterdir())


def test_a_concurrent_winner_during_rollback_is_reconciled_before_retry(
    tmp_path: Path, monkeypatch
) -> None:
    """A rollback collision must not mask a retryable promote failure.

    The first part-to-final rename fails while final is absent. Writer B lands
    after A decides to roll back but before A's trash-to-final rename. That
    rollback now collides with B's non-empty directory; B remains authoritative
    while A discards only its own superseded trash and retries.
    """
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "writer A")
    injected_winner = False

    def behaviour(call_number: int, src: str, dst: str):
        nonlocal injected_winner
        if call_number == 2 and src == str(part):
            return OSError(errno.ENOENT, "transient promote failure")
        if src.endswith(".trash") and dst == str(final) and not injected_winner:
            injected_winner = True
            _fake_store(final, "writer B")
        return None

    _replace_shim(monkeypatch, behaviour)

    assert ngff_.promote_store(part, final, fsync=False) == final
    assert injected_winner is True
    assert (final / "zarr.json").read_text(encoding="utf-8") == (
        '{"marker": "writer A"}'
    )
    assert not any(path.name.endswith(".trash") for path in tmp_path.iterdir())


def test_durable_promote_flushes_every_file_and_every_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Kills the `fsync ignored` and `files-only fsync` mutants.

    A durable file does not imply a durable directory entry, so a store whose
    nested `0/` dirent was never flushed is still a lost store after node
    loss. Recording the flushed paths is the only way to observe this without
    crashing a kernel.
    """
    final = tmp_path / "plate_01.ome.zarr"
    part = _fake_store(ngff_.new_part_path(final), "new")
    flushed: list[Path] = []
    monkeypatch.setattr(ngff_, "_fsync_path", lambda path: flushed.append(Path(path)))

    ngff_.promote_store(part, final, fsync=True)

    names = {path.name for path in flushed}
    assert {"zarr.json", "c.0.0.0"} <= names, f"file flushes missing: {names}"
    assert "0" in names, "the nested chunk directory's dirent was never flushed"
    assert part.name in names, "the .part root directory was never flushed"
    assert final.parent in flushed, "the rename's own dirent was never flushed"


def test_fsync_tree_flushes_directories_deepest_first(tmp_path: Path, monkeypatch) -> None:
    """A parent's entry must never be made durable before the child it names."""
    store = _fake_store(tmp_path / "s.ome.zarr", "x")
    flushed: list[Path] = []
    monkeypatch.setattr(ngff_, "_fsync_path", lambda path: flushed.append(Path(path)))
    ngff_.fsync_tree(store)
    directories = [path for path in flushed if path.is_dir()]
    assert directories, "no directory was flushed at all"
    depths = [len(path.parts) for path in directories]
    assert depths == sorted(depths, reverse=True), directories
