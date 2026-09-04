"""INV-VERDICT: nothing may improve a verdict except a successful deep
verification.

Spec §9.1 states the invariant and §14 calls these the highest-value tests in
the change. The current design's whole point is that it never trusts a cache,
so the correctness argument for introducing one has to be executable.

Each test corrupts the cache a different way and asserts that the corruption
can never license a *stronger* answer than a deep pass would give. A cache that
degrades to today's behaviour is correct; a cache that turns an incomplete run
into a complete one is the bug this file exists to prevent shipping.

D-B moved the cache in-process, so the "forge the file" cases here forge the
dict. The invariant is about what a cache may CAUSE, not where it lives, so it
binds identically. If S-5 had added an on-disk tier, Task 3 Step 8 would add
the JSON-corruption cases; it did not, so there are none.

**Scope, stated so nobody mistakes it for the whole invariant.** These tests
bind the cache's own surface. The end-to-end half -- "a forged cache cannot
make ``resolve_run_state`` say ``complete``" -- needs ``resolve_run_state``,
which is P1 Task 5, and lands with it. What is provable here is the mechanism
that end-to-end test depends on: an entry only ever *licenses* a re-use, the
licence is refused for every corruption below, and a refused licence means a
deep pass. See ``test_a_forged_entry_never_licenses_reuse``, which is the
cache-level form of the adversarial case.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import ImageState, clear_verification_cache
from phenotypic.sdk_._verification_cache import (
    CachedVerification,
    cached_states,
    entry_is_still_current,
    remember_states,
    tracked_output_count,
)

#: Two digests that differ in every byte, so a test that passes cannot be
#: passing on a prefix comparison.
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64

#: A fixed mtime, in ns. Tests that need "this file changed" or "this file
#: did not change" set it explicitly instead of racing the clock -- a
#: filesystem with coarse mtime granularity can otherwise give a rewrite the
#: same mtime as the write before it, and the test flakes green.
_PINNED_NS = 1_000_000_000


@pytest.fixture(autouse=True)
def _isolate_cache():
    """A module-level cache is shared state; a leaked entry makes the next
    test lie."""
    clear_verification_cache()
    yield
    clear_verification_cache()


def _image_state(stem: str, *, verdict: str = "verified") -> ImageState:
    return ImageState(
        work_id=f"work-{stem}",
        dataset="plate",
        image_stem=stem,
        stages={"measured": {"at": "2026-09-03T00:00:00Z"}},
        verdict=verdict,  # type: ignore[arg-type]
    )


def _stat_tuple(path: Path) -> tuple[int, int]:
    info = path.stat()
    return (info.st_size, info.st_mtime_ns)


def _artifact(root: Path, relative: str, payload: bytes = b"pixels") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _entry(
    root: Path, stem: str, *relatives: str, verdict: str = "verified"
) -> CachedVerification:
    return CachedVerification(
        state=_image_state(stem, verdict=verdict),
        stat_tuples={rel: _stat_tuple(root / rel) for rel in relatives},
    )


# ---------------------------------------------------------------- the fence


def test_a_stale_identity_never_matches(tmp_path):
    """Rule 1: no partial trust. A digest that does not match exactly yields
    ``None`` for the whole output, never a filtered subset."""
    artifact = "results/plate/a.bin"
    _artifact(tmp_path, artifact)
    remember_states(
        tmp_path, DIGEST_A, {"work-a": _entry(tmp_path, "a", artifact)}
    )

    assert cached_states(tmp_path, DIGEST_A) is not None
    assert cached_states(tmp_path, DIGEST_B) is None, (
        "an entry minted under a different identity was reused"
    )


def test_an_identity_change_replaces_the_output_entry_wholesale(tmp_path):
    """CAN-28. The bound comes from the FENCE, not from an eviction policy:
    only entries under the current identity for the current output are usable,
    so a new identity replaces the whole per-output map rather than
    accumulating alongside it.

    Audit §5, S22 and S23 are three findings about unbounded module globals in
    this codebase. This is bounded by 'images in the runs currently being asked
    about' -- tighter than the 200k-entry LRU an earlier draft proposed, and
    with no policy to get wrong.
    """
    artifact = "results/plate/a.bin"
    _artifact(tmp_path, artifact)
    remember_states(
        tmp_path, DIGEST_A, {"work-a": _entry(tmp_path, "a", artifact)}
    )
    remember_states(
        tmp_path, DIGEST_B, {"work-b": _entry(tmp_path, "a", artifact)}
    )

    assert cached_states(tmp_path, DIGEST_A) is None, (
        "stale identity entries survived"
    )
    assert set(cached_states(tmp_path, DIGEST_B)) == {"work-b"}
    assert tracked_output_count() == 1, "the output accumulated a second entry"


def test_two_spellings_of_one_output_share_one_slot(tmp_path):
    """``a`` and ``a/./`` and ``a/b/..`` are one output, so they are one slot.

    Without this, two callers naming the same tree each believe they hold the
    only entry for it, and the 'replaced wholesale' bound silently becomes
    'grows with however many spellings are in use'.
    """
    artifact = "results/plate/a.bin"
    _artifact(tmp_path, artifact)
    (tmp_path / "sub").mkdir()
    remember_states(
        tmp_path, DIGEST_A, {"work-a": _entry(tmp_path, "a", artifact)}
    )

    assert cached_states(tmp_path / "sub" / "..", DIGEST_A) is not None
    assert tracked_output_count() == 1


# --------------------------------------------------------- the currency rule


def test_a_forged_entry_never_licenses_reuse(tmp_path):
    """The adversarial case, at the surface it can be posed against here.

    The end-to-end form (P1 Task 5) forges every cached state to
    ``verdict="verified"`` with ``stat_tuples={}`` and asserts the verdict does
    not improve. What makes that hold is *this*: an entry declaring no
    artifacts is never current, so the resolver is never licensed to skip its
    deep pass and the forged verdict is never consulted.

    ``all()`` over an empty collection is ``True``, so the natural
    implementation makes a stat-tuple-less entry the STRONGEST entry in the
    cache instead of the weakest. That inversion is the whole bug.
    """
    forged = CachedVerification(
        state=_image_state("a", verdict="verified"), stat_tuples={}
    )
    assert entry_is_still_current(tmp_path, forged) is False


def test_ctime_is_not_part_of_the_currency_check(tmp_path, monkeypatch):
    """Audit S3 / spec §9.1: ``ctime_ns`` moves on chmod, chown, hardlink and
    ``rsync -a``, all routine on GPFS. ``size`` + ``mtime_ns`` already covers
    every write the publication contract makes, so a moved ``ctime_ns`` must
    invalidate nothing.

    **Do not go back to calling ``chmod`` here.** The obvious form of this
    test -- chmod the file, assert the entry survives -- is flaky by
    construction, and the first draft of it failed on xfs and tmpfs while
    passing on GPFS. The cause is not a filesystem capability: Linux stamps
    ``ctime`` from a *coarse* clock (``ktime_get_coarse_real_ts64``) whose
    resolution is the timer tick, typically 1-4 ms, so a chmod in the same
    tick as the file's creation produces a byte-identical ``ctime_ns`` and
    the test's own precondition guard fires. Measured on this cluster: an
    immediate chmod moved ``ctime_ns`` on GPFS and not on xfs or tmpfs, while
    a chmod after a 50 ms sleep moved it everywhere.

    Sleeping would make the test slow and still timing-dependent. Moving
    ``ctime_ns`` under the implementation's feet makes the claim
    deterministic on every filesystem, and makes it directly rather than
    incidentally: nothing about the file changes except the one field the
    currency check must not read.
    """
    artifact = "deliverables/overlays/plate/a.png"
    path = _artifact(tmp_path, artifact)
    # Pin mtime so the ONLY field that differs below is ctime_ns. Without
    # this the test would also pass against an implementation that ignored
    # the stat tuple entirely.
    os.utime(path, ns=(_PINNED_NS, _PINNED_NS))
    entry = _entry(tmp_path, "a", artifact)

    moved_ctime_ns = path.stat().st_ctime_ns + _PINNED_NS
    real_stat = Path.stat

    def _stat_with_moved_ctime(self, *args, **kwargs):
        info = real_stat(self, *args, **kwargs)
        return os.stat_result(
            (
                info.st_mode,
                info.st_ino,
                info.st_dev,
                info.st_nlink,
                info.st_uid,
                info.st_gid,
                info.st_size,
                int(info.st_atime),
                int(info.st_mtime),
                int(info.st_ctime),
            ),
            {
                "st_atime_ns": info.st_atime_ns,
                "st_mtime_ns": info.st_mtime_ns,
                "st_ctime_ns": moved_ctime_ns,
            },
        )

    monkeypatch.setattr(Path, "stat", _stat_with_moved_ctime)

    seen = path.stat()
    assert seen.st_ctime_ns == moved_ctime_ns, "the patch is not in effect"
    assert seen.st_mtime_ns == _PINNED_NS, "the patch disturbed mtime_ns"
    assert seen.st_size == entry.stat_tuples[artifact][0], (
        "the patch disturbed st_size"
    )

    assert entry_is_still_current(tmp_path, entry) is True, (
        "a moved ctime_ns invalidated the cache -- it has leaked into the "
        "currency check that audit S3 removed"
    )


def test_the_currency_check_never_reads_ctime(tmp_path, monkeypatch):
    """Structural companion to the test above, with no clock in it at all.

    The behavioural test proves a moved ``ctime_ns`` changes no answer. This
    one proves the stronger thing: the field is never *read*. It hands the
    checker a stat stand-in whose ``st_ctime_ns`` raises on access, so an
    implementation that consults ctime fails at the moment it does, rather
    than at whatever assertion happens to notice downstream.

    **Why not assert on ``entry.stat_tuples`` instead.** The obvious
    structural form -- ``assert len(stored) == 2`` -- pins ``_stat_tuple``,
    which is a *test helper* returning that pair as a literal. Adding
    ``st_ctime_ns`` to :func:`entry_is_still_current`'s comparison would not
    change ``entry.stat_tuples`` at all, so such a test stays green through
    exactly the mutation it was written to catch. Nothing in shipped code
    builds a stat tuple yet -- the recorder is ``resolve_run_state``, which
    is P1 Task 5 -- so until then the only structurally pinnable surface is
    the checker.
    """
    artifact = "deliverables/overlays/plate/a.png"
    path = _artifact(tmp_path, artifact)
    os.utime(path, ns=(_PINNED_NS, _PINNED_NS))
    entry = _entry(tmp_path, "a", artifact)

    real_stat = Path.stat

    class _StatWithExplodingCtime:
        """Every field the currency check may read, and one it may not."""

        def __init__(self, info: os.stat_result) -> None:
            self.st_mode = info.st_mode
            self.st_size = info.st_size
            self.st_mtime_ns = info.st_mtime_ns

        @property
        def st_ctime_ns(self) -> int:
            raise AssertionError(
                "the currency check read st_ctime_ns -- audit S3 removed it "
                "because it moves on chmod, chown, hardlink and rsync -a"
            )

    monkeypatch.setattr(
        Path,
        "stat",
        lambda self, *a, **k: _StatWithExplodingCtime(
            real_stat(self, *a, **k)
        ),
    )

    assert entry_is_still_current(tmp_path, entry) is True


def test_a_rewritten_artifact_is_not_current(tmp_path):
    """The stat tuple is the currency check; a changed artifact fails it."""
    artifact = "deliverables/overlays/plate/a.png"
    path = _artifact(tmp_path, artifact)
    entry = _entry(tmp_path, "a", artifact)

    path.write_bytes(b"pixels" + b"tamper")

    assert entry_is_still_current(tmp_path, entry) is False


def test_a_same_size_rewrite_is_caught_by_mtime(tmp_path):
    """Size alone is not the check. ``D2`` scopes the residual hole to an
    in-place edit preserving BOTH size and ``mtime_ns``; a rewrite preserving
    only size must still fail."""
    artifact = "deliverables/overlays/plate/a.png"
    path = _artifact(tmp_path, artifact, b"aaaaaa")
    # Both timestamps are set explicitly rather than left to the clock: on a
    # filesystem with coarse mtime granularity the rewrite below can land in
    # the same tick as the original write, and the test would flake green.
    os.utime(path, ns=(_PINNED_NS, _PINNED_NS))
    entry = _entry(tmp_path, "a", artifact)

    path.write_bytes(b"bbbbbb")
    os.utime(path, ns=(_PINNED_NS * 2, _PINNED_NS * 2))

    assert path.stat().st_size == 6, "the rewrite changed size, not just mtime"
    assert entry_is_still_current(tmp_path, entry) is False


def test_a_deleted_artifact_is_not_current(tmp_path):
    """A missing file degrades; it never raises. INV-VERDICT's degrade half."""
    artifact = "deliverables/overlays/plate/a.png"
    path = _artifact(tmp_path, artifact)
    entry = _entry(tmp_path, "a", artifact)

    path.unlink()

    assert entry_is_still_current(tmp_path, entry) is False


def test_a_store_directory_is_never_current(tmp_path):
    """A store is marker-bound as a DIRECTORY, and a directory's stat tuple is
    not a content fingerprint: rewriting ``tables/measurements/table.parquet``
    inside a store leaves the store root's ``mtime_ns`` untouched. That is spec
    §0's 'a valid root does not imply unchanged contents', reached through the
    cache instead of through the root.

    Fail closed, so a caller that fences a store by its directory gets a cache
    miss rather than a stale licence -- and is pushed to name the root
    ``zarr.json``, which is the file the marker descriptor already digests.
    """
    store = tmp_path / "results/plate/zarr/a.ome.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    entry = CachedVerification(
        state=_image_state("a"),
        stat_tuples={"results/plate/zarr/a.ome.zarr": _stat_tuple(store)},
    )

    assert entry_is_still_current(tmp_path, entry) is False


def test_an_unreadable_path_degrades_rather_than_raising(tmp_path):
    """``entry_is_still_current`` never raises. A path that cannot be stat'd
    at all -- here a component that is a file, so the traversal is ENOTDIR --
    is a cache miss, not an exception reaching the GUI's poll."""
    _artifact(tmp_path, "results/plate/a.bin")
    entry = CachedVerification(
        state=_image_state("a"),
        stat_tuples={"results/plate/a.bin/nested": (1, 1)},
    )

    assert entry_is_still_current(tmp_path, entry) is False


# ------------------------------------------------------------- no aliasing


def test_the_returned_map_cannot_be_forged_in_place(tmp_path):
    """A consumer holding the live dict could write a ``verified`` entry into
    the cache without ever verifying anything -- a positive verdict from a
    cache entry alone, which is exactly INV-VERDICT's prohibition."""
    artifact = "results/plate/a.bin"
    _artifact(tmp_path, artifact)
    remember_states(
        tmp_path, DIGEST_A, {"work-a": _entry(tmp_path, "a", artifact)}
    )

    states = cached_states(tmp_path, DIGEST_A)
    with pytest.raises(TypeError):
        states["work-forged"] = _entry(  # type: ignore[index]
            tmp_path, "forged", artifact
        )

    assert set(cached_states(tmp_path, DIGEST_A)) == {"work-a"}


def test_remember_states_does_not_alias_the_callers_map(tmp_path):
    """The caller keeps writing into its own dict as it verifies more images.
    If the cache aliased it, those unverified additions would appear in the
    cache under an identity they were never verified against."""
    artifact = "results/plate/a.bin"
    _artifact(tmp_path, artifact)
    caller_map = {"work-a": _entry(tmp_path, "a", artifact)}
    remember_states(tmp_path, DIGEST_A, caller_map)

    caller_map["work-later"] = _entry(tmp_path, "later", artifact)

    assert set(cached_states(tmp_path, DIGEST_A)) == {"work-a"}


# ------------------------------------------------------------------- clear


def test_clear_scoped_to_one_output_does_not_clear_another(tmp_path):
    """Rule 4. P2 wires the scoped form to ``clear_machine_state``, so a run
    resetting its own state must not throw away every other run's."""
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        _artifact(root, "results/plate/a.bin")
        remember_states(
            root,
            DIGEST_A,
            {"work-a": _entry(root, "a", "results/plate/a.bin")},
        )
    assert tracked_output_count() == 2

    clear_verification_cache(a)

    assert cached_states(a, DIGEST_A) is None
    assert cached_states(b, DIGEST_A) is not None, (
        "a scoped clear reached another output's entries"
    )
    assert tracked_output_count() == 1


def test_clearing_every_output_leaves_nothing_tracked(tmp_path):
    """The ``None`` form is what a fixture and a process teardown want, and is
    never what a run's own state reset wants."""
    for name in ("a", "b"):
        root = tmp_path / name
        _artifact(root, "results/plate/a.bin")
        remember_states(
            root,
            DIGEST_A,
            {"work-a": _entry(root, "a", "results/plate/a.bin")},
        )

    clear_verification_cache()

    assert tracked_output_count() == 0
    assert cached_states(tmp_path / "a", DIGEST_A) is None
