"""The verification cache -- one tier in memory, one tier on disk.

Audit **S1** -- the finding spec §9.1 responds to -- proposed a *process-level*
cache keyed on the marker file's stat tuple. §9.1 escalated that to a file on
disk. Decision **D-B** (OPEN-QUESTIONS) took it back, and **U-11 reversed D-B**
on measured evidence: at N=6,657 on a cold node a deep pass is 1403 s and the
stat sweep the cache enables is ~37 s, because the cost is per-file *latency*
(a cold ``open+read`` is ~10x a cold ``stat``) rather than bytes.

So there are two tiers, and they answer different questions:

* **Tier 1, in process** (:data:`_CACHE`). Every cadence the audit measured is
  a repeated call inside ONE long-lived process -- the observer's 2 s tick, the
  viewer's 5-10 s poll, ``OutputRoot.discover``'s double read,
  ``OutputMutationGuard``'s double read. This tier makes the *second* call
  cheap.
* **Tier 2, on disk** (``.phenotypic/verification_cache.json``). Read when tier
  1 is cold, written after a pass that deep-verified something. This tier makes
  the *first call in a new process* cheap.

INVARIANT (INV-VERDICT) -- **nothing may improve a verdict except a successful
deep verification.** No function here returns a verdict to a caller that has
not deep-verified. :func:`entry_is_still_current` answers exactly one question:
*may a previously deep-verified result stand?* The caller supplied the verdict
from its own deep pass, and a ``True`` here merely licenses skipping that pass
next time. A stale, replaced or forged entry therefore degrades to today's
behaviour and never past it.

**The on-disk tier is a tenth tracked file in a change that removes nine, and
it is a cache -- not a tenth evidence source.** P7 Task 6's register lists it
under *cache*, never under tracked state. **Nothing branches on it and no
verdict is derived from it.** That is the whole of what keeps it a cache: it
can cause a re-verification, and it can cause a re-verification to be skipped,
and it can cause nothing else. Delete the file and every answer this package
gives is unchanged; only the time taken to give it moves.

**The on-disk tier weakens the sentence INV-VERDICT's safety argument rests
on, and inheriting the in-process wording would misrepresent it.** *"An entry
only lets a previously deep-verified result stand"* means, in tier 1, *by this
process, minutes ago* -- the process that wrote the entry is the process
reading it, running the code you are reading now. On disk it means **by some
process**:

* possibly an **older build** whose deep-verification rules differed. The
  payload records what was checked, never how, so
  ``VERIFICATION_CACHE_VERSION`` is the only signal available and it is only as
  good as the discipline of bumping it -- which is why
  :data:`~phenotypic.sdk_._io_constants.VERIFICATION_CACHE_VERSION` says to
  bump on a *rules* change and not only on a shape change.
* possibly one **still mid-write**. The write is a temp file plus
  ``os.replace`` (:func:`~phenotypic.sdk_._atomic_io.atomic_write_json`), so a
  reader sees the whole old file or the whole new one and never a torn one --
  but "the whole new one" can still be a file whose writer then died before
  finishing the run it describes.
* possibly **another user**, or the same user under a different build, sharing
  the output tree.

Writing to ``.phenotypic/verification_cache.json`` needs the same permission as
writing ``processing_state.json`` beside it, so the tier opens no new *access*.
What it changes is the *consequence* of that access: forging
``processing_state.json`` changes what a run claims to have accepted, and can
be detected by verifying it; forging this file changes whether anything was
checked at all, and a forgery whose stat tuples happen to be current is
indistinguishable from a real entry. The degradation story survives in full --
every doubt still falls through to ``deep`` -- but the guarantee is materially
weaker than tier 1's, and a reader is entitled to know which one they have.

``ctime_ns`` is deliberately absent from the stat tuple (audit **S3**): it
moves on ``chmod``, ownership change, hardlink and ``rsync -a``, all routine on
a shared filesystem, and ``size`` + ``mtime_ns`` already covers every write the
publication contract makes.

**CONTRACT FOR CALLERS: fence a store by its root ``zarr.json``, never by the
store directory.** ``stat_tuples`` names **regular files only**;
:func:`entry_is_still_current` fails closed on anything else, so a caller that
records a store *directory* gets a permanent cache miss rather than a wrong
answer. This is not fussiness about types. A store is marker-bound as a
directory (``_cli_completion._artifact_descriptor``), but a directory's
``st_size`` is a filesystem detail and its ``mtime_ns`` tracks only its own
entries -- rewriting ``tables/measurements/table.parquet`` inside a promoted
store leaves the store root's ``mtime_ns`` untouched. Fencing on the directory
would therefore license re-use of a verdict for contents that changed, which is
spec §0's *"a valid root does not imply unchanged contents"* arriving through
the cache instead of through the root. The root ``zarr.json`` is the file the
marker descriptor already digests, and it is the right thing to stat.

**Bounded by the fence, not by a policy** (CAN-28). ``LocalRunner._instances``,
``_terminal_job_cache`` (``_cli/_dashboard/_manifest_builder.py:73``) and
``_LAST_DUMPED`` (``gui/builder/_point_picker.py:549``) are three unbounded
module globals this codebase already carries as audit findings (§5, S22, S23),
so a fourth is not acceptable. But an LRU would be a *second* mechanism on top
of one that already bounds this: entries are identity-fenced, so only those
under the current identity for the current output can ever be used. Replacing
each output's map wholesale on an identity change is therefore both tighter and
simpler than evicting -- the bound is "images in the runs currently being asked
about", and it follows from the fence instead of being layered on top of it.

**No lock.** Entries are published by a single ``dict`` assignment of an
immutable ``(identity_digest, states)`` pair whose inner map is a private copy
that is never mutated afterwards, so a concurrent reader either sees the whole
old pair or the whole new one. That is what makes this safe under Dash's
threaded server without serialising the GUI's poll.

**No lock on disk either, and for a stronger reason.** Concurrent writers are
last-wins via ``atomic_write_json``, which is safe *precisely because the file
is never authoritative*: the worst a lost write can do is cost a later process
a deep pass it would otherwise have skipped. Taking ``flock`` here -- per
output, on every resolve, on the filesystem whose cross-node semantics
``sdk_/_file_locking.py:101`` already flags as the weaker POSIX option -- would
buy nothing a wrong answer could come from.

**Two rules the on-disk tier follows so that it stays a cache:**

1. **It never creates the directory it lives in.** ``persist_states`` writes
   only when ``.phenotypic/`` already exists, so resolving the state of a tree
   this package has never written to leaves that tree byte-for-byte alone. A
   reader that creates directories is not a reader.
2. **Every failure is silent and degrades to ``deep``.** A read-only output, a
   full disk, a truncated or forged or unparseable file, a file from another
   identity -- each is a cache miss, never an exception and never a log line
   the user has to act on. Spec §9.1: *a tree the user cannot write must not
   become a tree the user cannot read.*

``clear_machine_state`` deletes the file, because it deletes every child of
``.phenotypic/`` bar the terminal-failure journal. That is the intended
lifecycle and it needs no special case; what needs stating is that the cache is
**not** in the preserve set that ``restart_epoch.json`` is in -- a counter that
must survive a restart and a cache that must not are the two halves of the same
directory, and confusing them silently carries a pre-restart verdict across the
fence the restart exists to raise.

:func:`clear_verification_cache` clears **tier 1 only**. That is the same
asymmetry, from the other side: it is a reader-side memory reset, it touches no
file, and a caller that wants both tiers gone wants ``clear_machine_state``.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import get_args

from ._atomic_io import atomic_write_json
from ._io_constants import (
    VERIFICATION_CACHE_VERSION,
    phenotypic_cache_dir,
    verification_cache_path,
)

# NOT from ._run_state -- that direction is the cycle the split prevents.
from ._state_types import ImageState, Verdict

#: output_dir (resolved, as str) -> (identity_digest, {work_id: entry}).
#:
#: One slot per output, replaced wholesale. See the module docstring on why
#: there is no eviction policy and no lock.
_CACHE: dict[str, tuple[str, dict[str, "CachedVerification"]]] = {}


@dataclass(frozen=True)
class CachedVerification:
    """One image's deep-verified state plus the stat tuples that fence it.

    Attributes:
        state: The WHOLE :class:`~phenotypic.sdk_._state_types.ImageState`
            (CAN-14), not just a verdict. ``RunState.images`` needs ``stages``
            per image, so an entry carrying only ``(work_id, verdict)`` would
            force the shallow path to re-read every record JSON -- the ~10^4
            marker-reads half of audit §4's cost, left in place. Caching the
            state is what makes spec §9.2's claim true rather than half-true.
        stat_tuples: Run-root-relative POSIX path -> ``(size, mtime_ns)`` for
            every **regular file** the verdict was derived from. Relative, so
            an entry survives the tree being reached through a second mount,
            exactly as marker artifact descriptors do (ledger FLOW-3).
    """

    state: ImageState
    stat_tuples: Mapping[str, tuple[int, int]]


def _cache_key(output_dir: Path) -> str:
    """Return the canonical per-output cache key.

    Resolving collapses ``a``, ``a/``, ``a/../a`` and any symlinked route to
    the same slot, so two callers naming one output cannot end up with two
    entries that each think they are the only one. A path that cannot be
    resolved (a symlink loop) falls back to an absolute path rather than
    raising: a cache may never turn a readable output into an error.
    """
    try:
        return str(Path(output_dir).resolve())
    except OSError:
        return os.path.abspath(str(output_dir))


def cached_states(
    output_dir: Path, identity_digest: str
) -> Mapping[str, CachedVerification] | None:
    """Return this output's entries under ``identity_digest``, or ``None``.

    There is no partial trust: an identity that does not match **exactly**
    yields ``None`` for the whole output, never a filtered subset. Rule 1 of
    Task 3.

    The returned mapping is read-only. A caller adding freshly verified images
    builds its own ``dict`` and hands it to :func:`remember_states`; handing
    out the live map would let a consumer forge an entry in place, which is
    precisely the failure INV-VERDICT exists to prevent.

    Args:
        output_dir: Run output root.
        identity_digest: :meth:`RunIdentity.digest` for the run being resolved.

    Returns:
        A read-only ``work_id -> CachedVerification`` mapping, or ``None`` when
        this output has no entry or its entry was minted under a different
        identity.
    """
    entry = _CACHE.get(_cache_key(output_dir))
    if entry is None:
        return None
    stored_digest, states = entry
    if stored_digest != identity_digest:
        return None
    return MappingProxyType(states)


def remember_states(
    output_dir: Path,
    identity_digest: str,
    entries: Mapping[str, CachedVerification],
) -> None:
    """Replace this output's entries wholesale under ``identity_digest``.

    Replacement, not merge (rule 3). An identity change discards the previous
    map rather than accumulating beside it, which is what bounds the cache
    without an eviction path to get wrong (CAN-28).

    Args:
        output_dir: Run output root.
        identity_digest: :meth:`RunIdentity.digest` the entries were verified
            under.
        entries: The complete ``work_id -> CachedVerification`` map for this
            output. Copied, so a caller mutating its own dict afterwards
            cannot reach into the cache.
    """
    _CACHE[_cache_key(output_dir)] = (identity_digest, dict(entries))


def entry_is_still_current(
    output_dir: Path, entry: CachedVerification
) -> bool:
    """Return whether every file ``entry`` was verified from is unchanged.

    **This is not a verdict.** A ``True`` says only that a deep pass the caller
    already performed may stand; the caller supplies the verdict itself. Rule 2
    of Task 3: an empty ``stat_tuples`` map, a missing file, an ``OSError`` or
    any changed ``(size, mtime_ns)`` all return ``False``, and this function
    never raises.

    An empty map is ``False`` on purpose -- ``all()`` over nothing is ``True``,
    which would make a stat-tuple-less entry the strongest entry in the cache
    rather than the weakest.

    A **directory** is also ``False``. A store is marker-bound as a directory
    (``_cli_completion._artifact_descriptor``), but a directory's ``st_size``
    is a filesystem detail and its ``mtime_ns`` tracks only its own entries --
    rewriting ``tables/measurements/table.parquet`` inside a store leaves the
    store root's mtime untouched. A caller fencing a store must therefore name
    its root ``zarr.json``, the same file the marker descriptor digests, not
    the store directory.

    Args:
        output_dir: Run output root the entry's paths are relative to.
        entry: A previously remembered verification.

    Returns:
        ``True`` only when every named path is a regular file whose size and
        ``mtime_ns`` are unchanged.
    """
    if not entry.stat_tuples:
        return False
    root = Path(output_dir)
    for relative, expected in entry.stat_tuples.items():
        try:
            info = (root / relative).stat()
        except OSError:
            return False
        if not stat.S_ISREG(info.st_mode):
            return False
        # ctime_ns is absent by design -- see the module docstring (audit S3).
        if (info.st_size, info.st_mtime_ns) != tuple(expected):
            return False
    return True


# ---------------------------------------------------------------------------
# Tier 2: the on-disk cache (spec §9.1, U-11)
# ---------------------------------------------------------------------------

#: Document keys. Spelled once, because a reader and a writer disagreeing
#: about a key name is a permanent, silent cache miss -- the failure mode this
#: tier is least able to notice, since a miss is also its correct behaviour.
_SCHEMA_VERSION_KEY = "schema_version"
_IDENTITY_KEY = "identity_digest"
_ENTRIES_KEY = "entries"

#: The three legal ``ImageState.verdict`` values, taken from the type rather
#: than respelled. A forged file naming a fourth is refused.
_VERDICTS: frozenset[str] = frozenset(get_args(Verdict))


def _entry_to_json(entry: "CachedVerification") -> dict[str, object]:
    """Render one entry as JSON, minus its ``work_id``.

    ``work_id`` is the map key and is not repeated inside the value, so the
    document cannot disagree with itself about which image an entry describes
    -- the reader reconstructs it from the key.
    """
    state = entry.state
    return {
        "dataset": state.dataset,
        "image_stem": state.image_stem,
        "stages": {
            name: dict(body) for name, body in state.stages.items()
        },
        "verdict": state.verdict,
        "reason": state.reason,
        "stat_tuples": {
            relative: [size, mtime_ns]
            for relative, (size, mtime_ns) in entry.stat_tuples.items()
        },
    }


def _entry_from_json(
    work_id: str, payload: object
) -> "CachedVerification | None":
    """Rebuild one entry, or ``None`` if anything about it is off.

    Every field is type-checked rather than trusted. This is not defensive
    style for its own sake: the file may have been written by an older build,
    a half-finished run or another user (see the module docstring), so the
    reader's job is to decide whether it is looking at a document this build
    wrote, and a wrong type is the cheapest evidence that it is not.

    Deliberately **fails closed on ``bool``**. ``True`` is an ``int`` in
    Python, so ``[true, true]`` would otherwise deserialize to the perfectly
    plausible stat tuple ``(1, 1)``.
    """
    if not isinstance(payload, dict):
        return None
    dataset = payload.get("dataset")
    image_stem = payload.get("image_stem")
    verdict = payload.get("verdict")
    raw_reason = payload.get("reason")
    stages = payload.get("stages")
    raw_tuples = payload.get("stat_tuples")
    if not isinstance(dataset, str) or not isinstance(image_stem, str):
        return None
    if verdict not in _VERDICTS:
        return None
    # Spelled as a widening rather than a guard so the narrowed type is
    # `str | None` here and not `object`; the guard form types fine to a
    # reader and not to mypy.
    reason: str | None = raw_reason if isinstance(raw_reason, str) else None
    if raw_reason is not None and reason is None:
        return None
    if not isinstance(stages, dict) or not all(
        isinstance(name, str) and isinstance(body, dict)
        for name, body in stages.items()
    ):
        return None
    if not isinstance(raw_tuples, dict):
        return None
    tuples: dict[str, tuple[int, int]] = {}
    for relative, pair in raw_tuples.items():
        if not isinstance(relative, str):
            return None
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        size, mtime_ns = pair
        if not _is_plain_int(size) or not _is_plain_int(mtime_ns):
            return None
        tuples[relative] = (size, mtime_ns)
    return CachedVerification(
        state=ImageState(
            work_id=work_id,
            dataset=dataset,
            image_stem=image_stem,
            stages=stages,
            verdict=verdict,  # type: ignore[arg-type]
            reason=reason,
        ),
        stat_tuples=tuples,
    )


def _is_plain_int(value: object) -> bool:
    """Return whether ``value`` is an ``int`` that is not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def load_persisted_states(
    output_dir: Path, identity_digest: str
) -> Mapping[str, CachedVerification] | None:
    """Return tier 2's entries under ``identity_digest``, or ``None``.

    Spec §9.1 names six ways this must yield nothing, and every one of them is
    a path along which a wrong ``complete`` could otherwise be manufactured
    from a file this build did not write. Five are here -- **file missing**,
    **unreadable**, **unparseable**, **recorded identity != current**, and (via
    the whole-document refusal below) a structurally wrong **entry**; the
    sixth, **entry absent**, is the caller finding no key for a ``work_id``,
    and the seventh thing that must fall through -- a **moved stat tuple** --
    is :func:`entry_is_still_current`'s, on the entries this returns.

    **No partial trust, exactly as in tier 1.** One malformed entry discards
    the whole document rather than the entry: a file this build cannot fully
    account for was written by something else, and reading the half of it that
    happens to parse is a decision to trust that something else.

    Args:
        output_dir: Run output root.
        identity_digest: :meth:`RunIdentity.digest` for the run being resolved.

    Returns:
        A read-only ``work_id -> CachedVerification`` mapping, or ``None``.
        **Never raises.**
    """
    try:
        raw = verification_cache_path(Path(output_dir)).read_bytes()
    except OSError:
        return None
    try:
        # UnicodeDecodeError is a ValueError, so undecodable bytes and
        # malformed JSON are one case here, as they are one case to a caller.
        document = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(document, dict):
        return None
    if document.get(_SCHEMA_VERSION_KEY) != VERIFICATION_CACHE_VERSION:
        return None
    if document.get(_IDENTITY_KEY) != identity_digest:
        return None
    entries = document.get(_ENTRIES_KEY)
    if not isinstance(entries, dict):
        return None
    loaded: dict[str, CachedVerification] = {}
    for work_id, payload in entries.items():
        if not isinstance(work_id, str):
            return None
        rebuilt = _entry_from_json(work_id, payload)
        if rebuilt is None:
            return None
        loaded[work_id] = rebuilt
    return MappingProxyType(loaded)


def persist_states(
    output_dir: Path,
    identity_digest: str,
    entries: Mapping[str, CachedVerification],
) -> bool:
    """Write tier 2 for this output, best-effort. **Never raises.**

    Two rules from the module docstring are implemented here rather than
    merely described:

    1. **It never creates ``.phenotypic/``.** Resolving the state of a tree
       this package has never written to must leave that tree byte-for-byte
       alone, and ``atomic_write_json`` would otherwise ``mkdir`` the parent.
    2. **A failed write is a return value, not an exception.** A read-only
       output degrades to a deep pass on the next cold start, which is the
       behaviour that shipped before this tier existed.

    The catch covers serialization as well as I/O, and the payload is built
    inside it for that reason. ``stages`` is an **open** map (spec §6.1) whose
    values this module never constrains, so a stage body that is not
    JSON-serializable is a ``TypeError`` from ``json.dumps`` and one that is
    not a mapping is a ``ValueError`` from ``dict()``. Every value reaching
    here today came from marker JSON and is safe; "never raises" is a promise
    about the contract, not about today's callers.

    Entries with no ``stat_tuples`` are **omitted**. They are permanently
    non-current by :func:`entry_is_still_current`'s first rule, so persisting
    them would grow the file by every unverified image while licensing
    nothing.

    Args:
        output_dir: Run output root.
        identity_digest: :meth:`RunIdentity.digest` the entries were verified
            under.
        entries: The complete ``work_id -> CachedVerification`` map.

    Returns:
        ``True`` when the file was written. ``False`` for every reason it was
        not -- no machine-state directory, a read-only tree, a full disk. No
        caller should branch on a verdict because of it; the value exists so a
        test can tell "declined" from "raised".
    """
    root = Path(output_dir)
    if not phenotypic_cache_dir(root).is_dir():
        return False
    try:
        payload: dict[str, object] = {
            _SCHEMA_VERSION_KEY: VERIFICATION_CACHE_VERSION,
            _IDENTITY_KEY: identity_digest,
            _ENTRIES_KEY: {
                work_id: _entry_to_json(entry)
                for work_id, entry in entries.items()
                if entry.stat_tuples
            },
        }
        atomic_write_json(verification_cache_path(root), payload)
    except (OSError, TypeError, ValueError):
        return False
    return True


def warm_states(
    output_dir: Path, identity_digest: str
) -> Mapping[str, CachedVerification] | None:
    """Return the warmest entries: tier 1, else tier 2, else ``None``.

    The tiers are ordered by *cost*, not by trust -- a tier-2 entry licenses
    exactly what a tier-1 entry licenses and no more, because the caller runs
    :func:`entry_is_still_current` over whichever it gets. What the order buys
    is that a long-lived process pays the JSON read once, on its first cold
    call, and never again.

    Args:
        output_dir: Run output root.
        identity_digest: :meth:`RunIdentity.digest` for the run being resolved.

    Returns:
        A read-only ``work_id -> CachedVerification`` mapping, or ``None`` when
        neither tier holds entries for this output under this identity.
    """
    in_process = cached_states(output_dir, identity_digest)
    if in_process is not None:
        return in_process
    return load_persisted_states(output_dir, identity_digest)


def clear_verification_cache(output_dir: Path | None = None) -> None:
    """Drop one output's **tier-1** entries, or every output's.

    Rule 4 of Task 3. P2 wires the scoped form to ``clear_machine_state``, so
    discarding a run's tracked state also discards what was derived from it.

    **Tier 1 only, and deliberately.** This function touches no file: it is a
    reader-side memory reset, and that is what lets ``_run_state`` re-export it
    without exporting a writer (INV-LAYER). ``clear_machine_state`` is what
    removes the on-disk tier, by removing every child of ``.phenotypic/``. A
    caller that clears only this one and then resolves shallowly will still be
    served from disk -- which is correct, since nothing about the run changed,
    but it is not what "clear the cache" sounds like.

    Args:
        output_dir: The output to forget. ``None`` forgets all of them, which
            is what a test fixture wants and what a process teardown wants;
            it is never what a run's own state reset wants.
    """
    if output_dir is None:
        _CACHE.clear()
        return
    _CACHE.pop(_cache_key(output_dir), None)


def tracked_output_count() -> int:
    """Return how many outputs currently hold entries.

    Test-only introspection. It exists so that
    ``test_an_identity_change_replaces_the_output_entry_wholesale`` can assert
    the *absence* of accumulation, which is the property CAN-28 turns into the
    bound -- and which is invisible from :func:`cached_states` alone, since a
    stale identity and an evicted output both read as ``None``.
    """
    return len(_CACHE)
