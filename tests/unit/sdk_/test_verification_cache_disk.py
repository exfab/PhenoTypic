"""INV-VERDICT for the on-disk tier: spec §9.1's six corruption cases.

U-11 reversed D-B and shipped ``.phenotypic/verification_cache.json``, and in
doing so weakened the sentence the in-process tier's safety argument rests on.
*"An entry only lets a previously deep-verified result stand"* means, in
process, **by this process, minutes ago**. On disk it means **by some
process** -- possibly an older build, possibly one that died mid-run, possibly
another user. So every one of §9.1's six ways a cache entry must fall through
to ``deep`` is a path along which a wrong ``complete`` could be manufactured
from a file this build did not write, and every one of them is tested here:

* **entry absent** --
  ``test_an_absent_entry_falls_through_to_deep``
* **stat tuple moved** --
  ``test_a_moved_stat_tuple_falls_through_to_deep``
* **recorded identity != current** --
  ``test_a_cache_from_another_identity_is_refused``,
  ``test_a_stale_identity_falls_through_to_deep``
* **file missing** --
  ``test_an_absent_cache_file_is_a_miss``,
  ``test_a_deleted_cache_file_falls_through_to_deep``
* **unreadable** --
  ``test_a_cache_file_that_cannot_be_read_is_a_miss``,
  ``test_an_unreadable_cache_file_falls_through_to_deep``
* **unparseable** --
  ``test_an_unparseable_cache_file_is_a_miss``,
  ``test_an_unparseable_cache_file_falls_through_to_deep``,
  ``test_one_malformed_entry_discards_the_whole_document``

Each case is pinned twice on purpose -- once at the reader's own surface, where
the assertion is small enough to read, and once end to end through
``resolve_run_state``, where the bug would actually live. The reader can be
correct while the resolver never calls it; the resolver can call it and then
ignore what it says.

**A separate file from ``test_verification_cache.py``, deliberately.** The
mutation-coverage gate is one harness per suite, so appending these to that
file would have made P1 Task 3's harness responsible for proving P2 Task 0's
tests. The harness for this suite is
``mutation_harnesses/p2_task0_disk_verification_cache.py``.

**What is NOT claimed here.** No test in this file shows that a forged
``verification_cache.json`` cannot manufacture a ``complete``. It can, if its
stat tuples happen to be current -- that is exactly the weakening U-11 accepted
and ``_verification_cache``'s module docstring states.
``test_a_forged_persisted_cache_cannot_manufacture_complete`` proves the
narrower and true thing: a forgery that does not carry current stat tuples is
refused, because the currency check is the only gate and it is still applied to
tier-2 entries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import (
    VERIFICATION_CACHE_VERSION,
    ImageState,
    clear_machine_state,
    clear_verification_cache,
    phenotypic_cache_dir,
    resolve_run_state,
    run_identity,
    verification_cache_path,
)
from phenotypic.sdk_._verification_cache import (
    CachedVerification,
    load_persisted_states,
    persist_states,
    remember_states,
    warm_states,
)

#: Two digests that differ in every byte, so a test that passes cannot be
#: passing on a prefix comparison.
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Tier 1 is a module global; a leaked entry makes the next test lie.

    Tier 2 needs no equivalent: it lives under ``tmp_path``, which pytest
    gives each test fresh.
    """
    clear_verification_cache()
    yield
    clear_verification_cache()


@pytest.fixture
def complete_run(tmp_path):
    from tests._output_layout import build_complete_run

    return build_complete_run(tmp_path)


@pytest.fixture
def incomplete_run(tmp_path):
    from tests._output_layout import build_incomplete_run

    return build_incomplete_run(tmp_path)


def _writable_output(tmp_path: Path) -> Path:
    """Return a tree with a ``.phenotypic/`` the cache is allowed to write to.

    ``persist_states`` never creates that directory (a reader that makes
    directories is not a reader), so a bare ``tmp_path`` is a *declined* write
    rather than a failed one -- which is a different test.
    """
    root = tmp_path / "out"
    phenotypic_cache_dir(root).mkdir(parents=True)
    return root


def _artifact(root: Path, relative: str, payload: bytes = b"pixels") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _entry(
    root: Path,
    stem: str,
    *relatives: str,
    verdict: str = "verified",
    reason: str | None = None,
) -> CachedVerification:
    return CachedVerification(
        state=ImageState(
            work_id=f"work-{stem}",
            dataset="plate",
            image_stem=stem,
            stages={"measured": {"at": "2026-09-04T00:00:00Z"}},
            verdict=verdict,  # type: ignore[arg-type]
            reason=reason,
        ),
        stat_tuples={
            rel: _stat_tuple(root / rel) for rel in relatives
        },
    )


def _stat_tuple(path: Path) -> tuple[int, int]:
    info = path.stat()
    return (info.st_size, info.st_mtime_ns)


def _read_document(root: Path) -> dict:
    return json.loads(verification_cache_path(root).read_bytes())


def _write_document(root: Path, document: object) -> None:
    verification_cache_path(root).write_text(
        json.dumps(document), encoding="utf-8"
    )


def _cold() -> None:
    """Simulate a new process: drop tier 1, leave tier 2 on disk."""
    clear_verification_cache()


def _current_digest(run: Path) -> str:
    identity = run_identity(run)
    assert identity is not None, "the fixture stopped writing a run identity"
    return identity.digest()


# ------------------------------------------------------- the reader contract


def test_the_reader_rebuilds_every_field_of_an_entry(tmp_path):
    """Every field round-trips, including the two nothing else here reads.

    ``stages`` is the load-bearing one: :func:`_advisories` is a projection
    over it with no I/O, which is what lets a shallow pass emit the same
    advisories as the deep pass it is standing in for. A reader that dropped
    ``stages`` would still produce correct *verdicts* and would silently stop
    producing the metadata-divergence and migrated-provenance advisories on
    every shallow pass served from disk.

    ``reason`` never appears in a file this build writes -- entries with no
    stat tuples are not persisted, and only those carry a reason -- so it is
    pinned here, at the reader, where a hand-written document can exercise it.
    """
    root = _writable_output(tmp_path)
    _write_document(
        root,
        {
            "schema_version": VERIFICATION_CACHE_VERSION,
            "identity_digest": DIGEST_A,
            "entries": {
                "work-a": {
                    "dataset": "plate",
                    "image_stem": "a",
                    "stages": {
                        "measured": {"at": "2026-09-04T00:00:00Z"},
                        "stage2": {"shard": 3},
                    },
                    "verdict": "unverified",
                    "reason": "declared artifact 'overlay' is missing",
                    "stat_tuples": {"results/plate/a.bin": [6, 1234]},
                }
            },
        },
    )

    loaded = load_persisted_states(root, DIGEST_A)

    assert loaded is not None
    assert dict(loaded) == {
        "work-a": CachedVerification(
            state=ImageState(
                work_id="work-a",
                dataset="plate",
                image_stem="a",
                stages={
                    "measured": {"at": "2026-09-04T00:00:00Z"},
                    "stage2": {"shard": 3},
                },
                verdict="unverified",
                reason="declared artifact 'overlay' is missing",
            ),
            stat_tuples={"results/plate/a.bin": (6, 1234)},
        )
    }


def test_persist_then_load_round_trips(tmp_path):
    """The writer's half of the same claim, at the writer's own surface.

    ``work_id`` is the map key and is deliberately not repeated inside the
    entry, so this also pins that the reader reconstructs it from the key
    rather than from a second copy the document could disagree with.
    """
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    entries = {"work-a": _entry(root, "a", "results/plate/a.bin")}

    assert persist_states(root, DIGEST_A, entries) is True

    assert dict(load_persisted_states(root, DIGEST_A) or {}) == entries


def test_a_cache_from_another_identity_is_refused(tmp_path):
    """§9.1 case 3, at the reader. No partial trust: an identity that does not
    match exactly yields ``None`` for the whole document."""
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    persist_states(
        root, DIGEST_A, {"work-a": _entry(root, "a", "results/plate/a.bin")}
    )

    assert load_persisted_states(root, DIGEST_A) is not None
    assert load_persisted_states(root, DIGEST_B) is None


def test_an_absent_cache_file_is_a_miss(tmp_path):
    """§9.1 case 4. A tree with no cache is the ordinary cold start, not an
    error, and certainly not an exception reaching the GUI's poll."""
    root = _writable_output(tmp_path)

    assert load_persisted_states(root, DIGEST_A) is None


def test_a_cache_file_that_cannot_be_read_is_a_miss(tmp_path):
    """§9.1 case 5. A directory where the file should be is deterministic on
    every filesystem and needs no ``chmod``, which
    ``test_ctime_is_not_part_of_the_currency_check`` documents as
    tick-granularity flaky. The code path is the same ``OSError`` a permission
    denial takes."""
    root = _writable_output(tmp_path)
    verification_cache_path(root).mkdir()

    assert load_persisted_states(root, DIGEST_A) is None


def test_an_unparseable_cache_file_is_a_miss(tmp_path):
    """§9.1 case 6. Bytes that are not JSON at all -- the shape a truncated or
    concurrently-clobbered file takes."""
    root = _writable_output(tmp_path)
    verification_cache_path(root).write_bytes(b'{"entries": {"a": ')

    assert load_persisted_states(root, DIGEST_A) is None


def test_one_malformed_entry_discards_the_whole_document(tmp_path):
    """§9.1 case 6, at entry granularity -- and the answer is all-or-nothing.

    Reading the half of a document that happens to parse is a decision to
    trust whatever wrote the half that did not. Tier 1 makes the same call for
    the same reason (``cached_states`` returns ``None`` for the whole output on
    an identity mismatch, never a filtered subset).
    """
    root = _writable_output(tmp_path)
    _write_document(
        root,
        {
            "schema_version": VERIFICATION_CACHE_VERSION,
            "identity_digest": DIGEST_A,
            "entries": {
                "work-a": {
                    "dataset": "plate",
                    "image_stem": "a",
                    "stages": {},
                    "verdict": "verified",
                    "stat_tuples": {"results/plate/a.bin": [6, 1234]},
                },
                "work-b": {
                    "dataset": "plate",
                    "image_stem": "b",
                    "stages": {},
                    "verdict": "verified",
                    "stat_tuples": "not a mapping",
                },
            },
        },
    )

    assert load_persisted_states(root, DIGEST_A) is None


def test_a_boolean_masquerading_as_a_stat_tuple_is_refused(tmp_path):
    """``True`` is an ``int`` in Python, so ``[true, true]`` would otherwise
    deserialize to the entirely plausible stat tuple ``(1, 1)``.

    A one-byte file whose mtime_ns happened to be 1 would then read as
    current. The window is absurd; the check is one predicate.
    """
    root = _writable_output(tmp_path)
    _write_document(
        root,
        {
            "schema_version": VERIFICATION_CACHE_VERSION,
            "identity_digest": DIGEST_A,
            "entries": {
                "work-a": {
                    "dataset": "plate",
                    "image_stem": "a",
                    "stages": {},
                    "verdict": "verified",
                    "stat_tuples": {"results/plate/a.bin": [True, True]},
                }
            },
        },
    )

    assert load_persisted_states(root, DIGEST_A) is None


def test_an_unrecognized_verdict_is_refused(tmp_path):
    """``ImageState.verdict`` is a closed set of three. A document naming a
    fourth was written by something whose notion of a verdict is not this
    build's, which is the whole of what the on-disk tier has to worry about."""
    root = _writable_output(tmp_path)
    _write_document(
        root,
        {
            "schema_version": VERIFICATION_CACHE_VERSION,
            "identity_digest": DIGEST_A,
            "entries": {
                "work-a": {
                    "dataset": "plate",
                    "image_stem": "a",
                    "stages": {},
                    "verdict": "trusted",
                    "stat_tuples": {"results/plate/a.bin": [6, 1234]},
                }
            },
        },
    )

    assert load_persisted_states(root, DIGEST_A) is None


def test_a_boolean_schema_version_is_refused(tmp_path):
    """`True == 1` in Python, so `schema_version: true` satisfies a plain
    comparison against version 1 (gate finding F11).

    This module already fails closed on `bool` in three other places -- the
    verdict check, and both halves of the stat tuple -- for exactly this
    reason. The omission here was inconsistent rather than reasoned, which is
    the kind that survives review: every individual guard looks deliberate.
    """
    root = _writable_output(tmp_path)
    _write_document(
        root,
        {
            "schema_version": True,
            "identity_digest": DIGEST_A,
            "entries": {},
        },
    )

    assert load_persisted_states(root, DIGEST_A) is None


def test_a_different_schema_version_is_refused(tmp_path):
    """The version is the only signal that a file was written by a build whose
    deep-verification RULES differed: the payload records what was checked,
    never how. That is why ``VERIFICATION_CACHE_VERSION``'s comment says to
    bump on a rules change and not only on a shape change."""
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    persist_states(
        root, DIGEST_A, {"work-a": _entry(root, "a", "results/plate/a.bin")}
    )
    document = _read_document(root)
    document["schema_version"] = VERIFICATION_CACHE_VERSION + 1
    _write_document(root, document)

    assert load_persisted_states(root, DIGEST_A) is None


# ------------------------------------------------------- the writer contract


def test_unverified_entries_are_not_persisted(tmp_path):
    """An entry with no stat tuples is permanently non-current, so persisting
    it grows the file by every unverified image while licensing nothing.

    On a 6,000-image run stopped a third of the way through, that is two
    thirds of the document written to say nothing.
    """
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    persist_states(
        root,
        DIGEST_A,
        {
            "work-a": _entry(root, "a", "results/plate/a.bin"),
            "work-b": _entry(
                root, "b", verdict="unverified", reason="no marker"
            ),
        },
    )

    assert set(load_persisted_states(root, DIGEST_A) or {}) == {"work-a"}


def test_persisting_into_a_read_only_output_is_not_an_error(tmp_path):
    """Spec §9.1: a tree the user cannot write must not become a tree the user
    cannot read.

    The failure this guards is not a lost cache -- that is free. It is a
    ``PermissionError`` propagating out of a *read* API and turning a
    perfectly readable archived run into one the GUI refuses to open.
    """
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    entries = {"work-a": _entry(root, "a", "results/plate/a.bin")}
    cache_dir = phenotypic_cache_dir(root)
    os.chmod(cache_dir, 0o500)
    try:
        if os.access(cache_dir, os.W_OK):
            pytest.skip("cannot drop write permission here (running as root?)")

        assert persist_states(root, DIGEST_A, entries) is False
    finally:
        os.chmod(cache_dir, 0o700)

    assert not verification_cache_path(root).exists()


def test_an_unserializable_stage_value_is_not_an_error(tmp_path):
    """"Never raises" has to cover serialization, not only I/O.

    ``stages`` is the **open** map of spec §6.1 -- ``stage1``/``stage2``/
    ``stage3``/``measured`` today, more later -- and nothing in this module
    constrains its values. Every value reaching the writer today came from
    marker JSON and is safe, so this is a promise about the contract rather
    than about today's callers: the first stage that carries a ``Path``, a
    ``set`` or a ``datetime`` must cost a cache, not a traceback out of a read
    API.

    **``stages`` is the trigger, not merely the payload**, which is worth
    stating because it makes this test co-fail with the "writer drops
    ``stages``" mutation in a way that reads like a coincidence and is not.
    That mutation serializes ``{}`` in place of the real map, so ``object()``
    never reaches ``json.dumps``, the write *succeeds*, and the first
    assertion here fails on ``True is False``. The mutation destroys this
    test's precondition rather than its subject. That coupling is a feature:
    if stage serialization ever moves or stops happening, this test fails
    loudly instead of quietly asserting nothing.
    """
    import dataclasses

    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    entry = _entry(root, "a", "results/plate/a.bin")
    poisoned = CachedVerification(
        state=dataclasses.replace(
            entry.state, stages={"measured": {"at": object()}}
        ),
        stat_tuples=entry.stat_tuples,
    )

    assert persist_states(root, DIGEST_A, {"work-a": poisoned}) is False

    assert not verification_cache_path(root).exists()


def test_persisting_never_creates_the_machine_state_directory(tmp_path):
    """A reader that makes directories is not a reader.

    ``resolve_run_state`` is documented to accept *any* directory, including
    one this package has never written to, and the on-disk tier must not turn
    that into a tree with a ``.phenotypic/`` in it. ``atomic_write_json``
    ``mkdir``s its parent, so declining has to be explicit.
    """
    root = tmp_path / "never-written-to"
    root.mkdir()
    _artifact(root, "results/plate/a.bin")
    entries = {"work-a": _entry(root, "a", "results/plate/a.bin")}

    assert persist_states(root, DIGEST_A, entries) is False

    assert not phenotypic_cache_dir(root).exists()


# ------------------------------------------------------------- the two tiers


def test_warm_states_prefers_tier_one(tmp_path):
    """Ordered by cost, not by trust: a long-lived process pays the JSON read
    once and never again.

    Both tiers license exactly the same thing, so preferring the in-process
    one changes only how long the answer takes.
    """
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    persist_states(
        root,
        DIGEST_A,
        {"work-on-disk": _entry(root, "a", "results/plate/a.bin")},
    )
    remember_states(
        root,
        DIGEST_A,
        {"work-in-memory": _entry(root, "a", "results/plate/a.bin")},
    )

    assert set(warm_states(root, DIGEST_A) or {}) == {"work-in-memory"}


def test_warm_states_falls_back_to_the_persisted_tier(tmp_path):
    """The whole point of U-11: the FIRST call in a NEW process is cheap."""
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    persist_states(
        root, DIGEST_A, {"work-a": _entry(root, "a", "results/plate/a.bin")}
    )
    clear_verification_cache()

    assert set(warm_states(root, DIGEST_A) or {}) == {"work-a"}


# ---------------------------------------------------------------------------
# End to end, through `resolve_run_state`. The reader can be correct while the
# resolver never calls it, and the resolver can call it and ignore the answer;
# neither is visible from the surface tests above.
# ---------------------------------------------------------------------------


def test_a_deep_pass_persists_the_cache(complete_run):
    """A deep pass writes tier 2, keyed on the identity it verified under."""
    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"

    document = _read_document(complete_run)

    assert document["schema_version"] == VERIFICATION_CACHE_VERSION
    assert document["identity_digest"] == _current_digest(complete_run)
    assert set(document["entries"]) == set(state.images)


def test_a_cold_process_reuses_the_persisted_tier(complete_run):
    """**The test that fails if the tier is written and never read.**

    Every other test in this file proves tier 2 cannot make an answer better
    than the truth. None of them would notice a file that nothing ever loads
    -- which degrades perfectly, ships U-11's 1403 s unfixed, and passes the
    entire corruption suite. This is the one that notices.
    """
    first = resolve_run_state(complete_run, depth="deep")
    assert first.depth == "deep"
    _cold()

    second = resolve_run_state(complete_run, depth="shallow")

    assert second.depth == "shallow", "the persisted tier was not consulted"
    assert second.completion == first.completion


def test_a_fully_warm_shallow_pass_does_not_rewrite_the_file(complete_run):
    """Tier 2 is written only when a pass actually deep-verified something.

    The observer ticks every 2 s and each browser tab polls every 5-10 s. A
    file rewritten on every one of those, at one entry per image, is a
    per-image-sized write on the exact cadences this cache exists to make
    cheap.

    Pinned with a key the reader ignores rather than with the file's mtime: a
    rewrite of identical content on a coarse-granularity filesystem can land
    in the same tick and leave size and ``mtime_ns`` untouched, and the test
    would flake green.
    """
    resolve_run_state(complete_run, depth="deep")
    document = _read_document(complete_run)
    document["witness"] = "survives unless the resolver rewrote the file"
    _write_document(complete_run, document)
    _cold()

    assert resolve_run_state(complete_run, depth="shallow").depth == "shallow"

    assert "witness" in _read_document(complete_run)


def test_an_absent_entry_falls_through_to_deep(complete_run):
    """§9.1 case 1, and spec §9.2's rolling-input case in miniature.

    Ten images landing in a 6,000-image run must cost ten deep verifications
    and 5,990 stats -- and the answer must report ``deep``, because part of it
    was. "Mostly shallow" is not a useful third value.
    """
    resolve_run_state(complete_run, depth="deep")
    document = _read_document(complete_run)
    dropped = sorted(document["entries"])[0]
    del document["entries"][dropped]
    _write_document(complete_run, document)
    _cold()

    state = resolve_run_state(complete_run, depth="shallow")

    assert state.depth == "deep"
    assert state.completion == "complete"


def test_a_moved_stat_tuple_falls_through_to_deep(complete_run):
    """§9.1 case 2, through the resolver.

    ``os.utime`` moves ``mtime_ns`` without touching a byte, so the deep pass
    behind the miss still says ``complete``: this isolates the currency check
    from the content check. A resolver that trusted a tier-2 entry without
    re-stating it would report ``shallow`` here, and would report ``complete``
    for a genuinely tampered artifact by the same code path.
    """
    resolve_run_state(complete_run, depth="deep")
    _cold()
    overlay = next(complete_run.rglob("overlays/**/*.png"), None)
    assert overlay is not None, "the fixture stopped writing an overlay"
    moved = overlay.stat().st_mtime_ns + 1_000_000_000
    os.utime(overlay, ns=(moved, moved))

    state = resolve_run_state(complete_run, depth="shallow")

    assert state.depth == "deep"
    assert state.completion == "complete"


def test_a_stale_identity_falls_through_to_deep(complete_run):
    """§9.1 case 3, through the resolver. ``bump_scientific_config_digest``
    rewrites ``config.pipeline_sha256``, which IS
    ``scientific_config_digest`` and is one of the five tokens
    ``RunIdentity.digest`` folds in."""
    from tests._output_layout import bump_scientific_config_digest

    resolve_run_state(complete_run, depth="deep")
    stale = _current_digest(complete_run)
    _cold()
    bump_scientific_config_digest(complete_run)
    # The file is still there and still parses; what moved is the identity.
    # Without this the assertion below would also pass for a tier that had
    # simply deleted, truncated or never written the file.
    assert _read_document(complete_run)["identity_digest"] == stale

    assert resolve_run_state(complete_run, depth="shallow").depth == "deep"


def test_a_deleted_cache_file_falls_through_to_deep(complete_run):
    """§9.1 case 4, through the resolver -- and the sanity check on every
    other end-to-end test here, since ``depth == "deep"`` is also what a
    working cache reports when it correctly refuses."""
    resolve_run_state(complete_run, depth="deep")
    _cold()
    verification_cache_path(complete_run).unlink()

    assert resolve_run_state(complete_run, depth="shallow").depth == "deep"


def test_an_unreadable_cache_file_falls_through_to_deep(complete_run):
    """§9.1 case 5, through the resolver."""
    resolve_run_state(complete_run, depth="deep")
    _cold()
    path = verification_cache_path(complete_run)
    path.unlink()
    path.mkdir()

    assert resolve_run_state(complete_run, depth="shallow").depth == "deep"


def test_an_unparseable_cache_file_falls_through_to_deep(complete_run):
    """§9.1 case 6, through the resolver."""
    resolve_run_state(complete_run, depth="deep")
    _cold()
    verification_cache_path(complete_run).write_bytes(b"\xff\xfe not json")

    assert resolve_run_state(complete_run, depth="shallow").depth == "deep"


def test_a_forged_persisted_cache_cannot_manufacture_complete(incomplete_run):
    """The adversarial case, in the form the on-disk tier can honestly claim.

    Every entry is forged to ``verdict="verified"`` with no stat tuples, which
    is what a file written by something that never verified anything looks
    like. The currency check is the only gate on a tier-2 entry, and it must
    still be applied: ``all()`` over an empty collection is ``True``, so the
    natural implementation makes a stat-tuple-less entry the STRONGEST entry
    in the cache instead of the weakest.

    **This is not a claim that a forged file is harmless.** A forgery carrying
    stat tuples that happen to be current would be indistinguishable from a
    real entry, and U-11 accepted that. What is pinned here is that forging
    the *verdict* alone buys nothing.
    """
    deep = resolve_run_state(incomplete_run, depth="deep")
    baseline = deep.completion
    assert baseline != "complete"
    _cold()
    _write_document(
        incomplete_run,
        {
            "schema_version": VERIFICATION_CACHE_VERSION,
            "identity_digest": _current_digest(incomplete_run),
            "entries": {
                work_id: {
                    "dataset": image.dataset,
                    "image_stem": image.image_stem,
                    "stages": {},
                    "verdict": "verified",
                    "reason": None,
                    "stat_tuples": {},
                }
                for work_id, image in deep.images.items()
            },
        },
    )

    after = resolve_run_state(incomplete_run, depth="shallow")

    assert after.completion == baseline, (
        "a forged persisted cache changed the verdict; a positive verdict "
        "must never come from a cache entry alone -- INV-VERDICT"
    )


def test_the_persisted_tier_carries_the_advisories_with_it(complete_run):
    """Advisories are projections over ``ImageState.stages`` with no I/O.

    That is what lets a shallow pass emit the same advisories as the deep pass
    it stands in for -- but only if ``stages`` survives the round trip through
    the file. A tier that dropped it would produce identical *verdicts* and
    silently stop telling the user that the configuration fence is unavailable
    for their migrated images, which is a wrong answer to a different
    question.

    ``provenance: "migrated"`` is used because it is the one advisory a
    fixture can raise today; the metadata-divergence advisory needs the store
    attribute P4 Task 2 has not written yet.
    """
    from phenotypic.sdk_ import image_completion_marker_path
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    marker_path = image_completion_marker_path(
        complete_run, FIXTURE_DATASET, FIXTURE_STEMS[0]
    )
    marker = json.loads(marker_path.read_bytes())
    marker["provenance"] = "migrated"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    deep = resolve_run_state(complete_run, depth="deep")
    assert any(
        "configuration fence is unavailable" in note
        for note in deep.advisories
    ), "the fixture no longer raises the migrated-provenance advisory"
    _cold()

    shallow = resolve_run_state(complete_run, depth="shallow")

    assert shallow.depth == "shallow"
    assert shallow.advisories == deep.advisories


def test_clear_machine_state_deletes_the_persisted_cache(tmp_path):
    """It is a cache, and it is **not** in the preserve set.

    ``clear_machine_state`` keeps ``terminal_failures.jsonl`` and -- from P2
    Task 1 -- ``restart_epoch.json``, because a counter that resets on the
    operation it fences is not a fence. The verification cache is the
    opposite: carrying a pre-restart verdict across the fence a restart exists
    to raise is exactly the failure the counter is there to prevent.
    """
    root = _writable_output(tmp_path)
    _artifact(root, "results/plate/a.bin")
    persist_states(
        root, DIGEST_A, {"work-a": _entry(root, "a", "results/plate/a.bin")}
    )
    assert verification_cache_path(root).is_file()

    clear_machine_state(root)

    assert not verification_cache_path(root).exists()
