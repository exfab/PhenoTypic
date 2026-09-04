"""The in-process verification cache.

Audit **S1** -- the finding spec §9.1 responds to -- proposed a *process-level*
cache keyed on the marker file's stat tuple. §9.1 escalated that to a file on
disk. Decision **D-B** (OPEN-QUESTIONS) took it back: every cadence the audit
measured is a repeated call inside ONE long-lived process (the observer's 2 s
tick, the viewer's 5-10 s poll, ``OutputRoot.discover``'s double read,
``OutputMutationGuard``'s double read), and an in-memory cache serves all of
them without adding a tracked artifact to a design whose purpose is removing
them.

INVARIANT (INV-VERDICT) -- **nothing may improve a verdict except a successful
deep verification.** No function here returns a verdict to a caller that has
not deep-verified. :func:`entry_is_still_current` answers exactly one question:
*may a previously deep-verified result stand?* The caller supplied the verdict
from its own deep pass, and a ``True`` here merely licenses skipping that pass
next time. A stale, replaced or forged entry therefore degrades to today's
behaviour and never past it.

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
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

# NOT from ._run_state -- that direction is the cycle the split prevents.
from ._state_types import ImageState

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


def clear_verification_cache(output_dir: Path | None = None) -> None:
    """Drop one output's entries, or every output's.

    Rule 4 of Task 3. P2 wires the scoped form to ``clear_machine_state``, so
    discarding a run's tracked state also discards what was derived from it.

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
