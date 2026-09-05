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
from ..sdk_._digests import canonical_digest
from ..sdk_._io_constants import phenotypic_cache_dir, restart_epoch_path
from ._cli_failure_tracker import processing_configuration_digest

__all__ = [
    "bump_restart_epoch",
    "derive_processing_generation",
    "per_image_config_digest",
    "read_restart_epoch",
]

# D-C: §5.4 calls this the "scientific config digest"; the code has called it
# `processing_configuration_digest` since `work_id` was introduced. **Aliased,
# not wrapped.** §5.4's whole argument is that the generation and `work_id`
# must never disagree about what counts as scientific configuration, and
# identity is the only form of agreement that cannot drift -- a wrapper is
# equal today and is one edit away from not being.
#
# ---------------------------------------------------------------------------
# NAMED `per_image_config_digest`, NOT `scientific_config_digest`, AND THAT
# MATTERS MORE THAN IT LOOKS.
#
# `scientific_config_digest` was already taken, by a DIFFERENT value. Two
# digests, two questions, and conflating them is a silent data migration:
#
#   A. The PROOF-side token. `state.config["pipeline_sha256"]`, which is
#      `sha256(<pipeline JSON bytes>)` (`_cli_staged_resume.py:78`). It is
#      written into every aggregate and run proof under the key
#      `"scientific_config_digest"` (`_cli_completion.py:914,1020,1087`) and
#      read back as `RunIdentity.scientific_config_digest`
#      (`_run_state.py:277`). It answers **"did the pipeline change?"**, which
#      is what spec §5.3's table says.
#
#   B. This one. The per-image configuration digest folded into `work_id` --
#      `image_type`, `nrows`, `ncols`, `bit_depth`, `detect_mode`,
#      `drop_originals`, plus a mode-conditional group. **It contains no
#      pipeline bytes at all.** It answers "would this image have to be
#      reprocessed?", which is what spec §5.4 describes.
#
# A cannot answer B's question and B cannot answer A's. The generation folds
# in BOTH -- `sha256(pipeline_sha256 || per_image_config_digest ||
# restart_epoch)` -- which is only meaningful because they differ.
#
# **Do not "unify" them.** The tempting repair is to point the proofs at B.
# That rewrites the value in every aggregate and run proof already on disk, so
# every previously complete run reads `incomplete` until it is re-finalized --
# a migration wearing a rename's clothes. A has an on-disk representation and
# B does not, which is precisely why B is the one that got renamed here.
# ---------------------------------------------------------------------------
per_image_config_digest = processing_configuration_digest

#: The document key. Spelled once: a reader and a writer disagreeing about it
#: would reset the fence to 0 on every read, silently.
_EPOCH_KEY = "restart_epoch"


def derive_processing_generation(
    *,
    pipeline_sha256: str | None,
    per_image_config: str | None,
    restart_epoch: int,
) -> str:
    """Return the content-derived ``processing_generation`` (spec §5.1, D3).

    **Same inputs, same token.** That is the whole of D3: two invocations with
    the same configuration mint the same generation without either having read
    the other's state, which is what lets a SLURM worker starting cold fence
    itself correctly against a run it has never seen. A ``uuid4()`` cannot do
    that, and every site this replaces used one.

    **``inventory_digest`` is deliberately NOT a component (D7).** Generation
    fences *configuration*; ``inventory_digest`` fences *scope*, and they
    change on different schedules. Folding them together makes every arrival
    under a rolling input look like a configuration change -- resetting live
    progress and fencing in-flight workers, which on a 6,000-image rolling
    dataset is a daily occurrence rather than an edge case.

    Digested as a **mapping** rather than concatenated, so the components are
    self-describing and a fourth can be added without silently colliding with
    a different three-component run.

    Args:
        pipeline_sha256: The pipeline file's digest, or ``None`` when the run
            has no pipeline file. ``None`` and ``""`` are the same input --
            see the note below.
        per_image_config: :func:`per_image_config_digest`'s value, or ``None``
            for a caller that has no ``ExecutionConfig`` to derive one from.
            ``--mode migrate`` is that caller: a converted tree never recorded
            the per-image configuration, and U-10's rule is to mark what
            cannot be recovered rather than fabricate it.
        restart_epoch: The run's restart epoch.

    Returns:
        The 64-character generation digest.

    Note:
        Absent components are normalized to ``""`` rather than omitted, so a
        run with no pipeline file still mints a stable generation determined
        by the remaining components. That is defensible -- there is genuinely
        no pipeline to fence against -- but it does mean two pipeline-less
        runs with identical per-image config share a generation, which is
        exactly what D3 says should happen.
    """
    return canonical_digest(
        {
            "pipeline_sha256": pipeline_sha256 or "",
            "per_image_config_digest": per_image_config or "",
            "restart_epoch": restart_epoch,
        }
    )


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
