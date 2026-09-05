"""Run-identity minting: the writers spec §5.2 keeps out of ``sdk_``.

**Why this module is in ``phenotypic._cli`` and not beside the readers.**
Spec §5.2 makes the read/write asymmetry structural: ``sdk_/_run_state.py``
exports only readers, so a GUI import cannot reach a publisher. Everything
here **writes** -- :func:`bump_restart_epoch` persists a counter, and
``mint_run_identity`` (Task 3) can call it -- so it lives on the CLI side of
that line. ``sdk_/_io_constants`` owns the *path*; this module owns the
*value*.

**The counter has two homes and that is the design, not a duplication**
(CONFLICT-1). ``.phenotypic/restart_epoch.json`` is *the counter*;
``processing_state.json``'s ``config.restart_epoch`` is *the value the state
was minted under*. Their lifecycles differ on purpose --
``clear_machine_state`` preserves the file while deleting the state -- and a
design in which the two **can** disagree, where the disagreement is the
signal, is precisely a fence.

So :func:`read_restart_epoch` **never repairs or backfills**
``config.restart_epoch``. A state file whose config epoch differs from the
counter is a *fenced* state, and ``assert_identity_current``'s named-field
``RuntimeError`` is the only correct response. Code that silently reconciles
them turns the second home into a cache and deletes the fence -- which is what
the data-flow reviewer filed and the simplicity reviewer declined, and this is
the rule that decides between them.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..sdk_._atomic_io import atomic_write_json
from ..sdk_._io_constants import phenotypic_cache_dir, restart_epoch_path

__all__ = [
    "bump_restart_epoch",
    "read_restart_epoch",
]

#: The document key. Spelled once: a reader and a writer disagreeing about it
#: would reset the fence to 0 on every read, silently.
_EPOCH_KEY = "restart_epoch"


def read_restart_epoch(output_dir: Path) -> int:
    """Return the run's restart epoch, or ``0`` when absent. **Never raises.**

    INV-VERDICT's degrade half, applied to a counter: a corrupt, truncated or
    unreadable epoch file reads as ``0`` rather than blocking the run. The
    direction matters. Reading ``0`` understates how many restarts have
    happened, so a stale worker that should have been fenced is *not* fenced
    -- which is the behaviour that shipped before this counter existed, and
    is therefore no worse than the status quo. Raising instead would make an
    unparseable byte a reason a user cannot restart at all.

    Args:
        output_dir: Run output root. May be any directory, including one this
            package has never written to.

    Returns:
        The recorded epoch, or ``0``.
    """
    try:
        raw = restart_epoch_path(Path(output_dir)).read_bytes()
    except OSError:
        return 0
    try:
        # UnicodeDecodeError is a ValueError, so undecodable bytes and
        # malformed JSON are one case here, as they are one case to a caller.
        document = json.loads(raw)
    except ValueError:
        return 0
    if not isinstance(document, dict):
        return 0
    epoch = document.get(_EPOCH_KEY)
    # `bool` is excluded explicitly: it is an `int` subclass, and `True` is
    # not a restart epoch. A negative value is a corrupt field, not a counter
    # running backwards, so it degrades to 0 like any other.
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        return 0
    return epoch


def bump_restart_epoch(output_dir: Path) -> int:
    """Increment and persist the restart epoch. Returns the new value.

    **A writer, and the only one.** Called by ``--restart`` and by
    ``mint_run_identity(restart=True)``; calling it twice in one invocation
    gives that run two generations and burns an epoch, which is why Task 3's
    mint is structurally once-per-invocation.

    Unlike :func:`read_restart_epoch` this **does** raise. A restart whose
    fence cannot be persisted must not proceed believing it was: the next
    invocation would read the old epoch, mint the generation the abandoned
    workers are already holding, and the fence would pass for exactly the
    workers it exists to exclude. A loud failure is the only safe outcome, and
    it is the asymmetry between the two functions here -- reading a missing
    fence is recoverable, failing to write one is not.

    Args:
        output_dir: Run output root. Its ``.phenotypic/`` is created if
            absent, because unlike the verification cache this is tracked
            state a run is entitled to establish.

    Returns:
        The new epoch, always the previous value plus one.

    Raises:
        OSError: If the counter cannot be persisted.
    """
    root = Path(output_dir)
    phenotypic_cache_dir(root).mkdir(parents=True, exist_ok=True)
    updated = read_restart_epoch(root) + 1
    atomic_write_json(restart_epoch_path(root), {_EPOCH_KEY: updated})
    return updated
