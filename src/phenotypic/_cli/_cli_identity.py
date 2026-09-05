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
from typing import TYPE_CHECKING

from ..sdk_._atomic_io import atomic_write_json
from ..sdk_._digests import canonical_digest
from ..sdk_._io_constants import phenotypic_cache_dir, restart_epoch_path
from ..sdk_._run_state import FINALIZATION_INPUT_SCHEMA_VERSION
from ..sdk_._state_types import RunIdentity
from ._cli_failure_tracker import processing_configuration_digest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._cli_types import ExecutionConfig

__all__ = [
    "bump_restart_epoch",
    "derive_processing_generation",
    "mint_run_identity",
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

#: Set on the ``ExecutionConfig`` by :func:`mint_run_identity` so a second
#: mint in one invocation is a loud failure rather than a silent second epoch
#: bump (CAN-21). An attribute on the config object rather than a module-level
#: registry: the config IS the invocation, so its lifetime is exactly the
#: scope the guard should have, and there is no `id()` reuse to get wrong.
_MINTED_FLAG = "_phenotypic_identity_minted"


def _metadata_digest_for(config: "ExecutionConfig") -> str | None:
    """Return the metadata snapshot's digest, or ``None`` without one.

    **Must stay identical to what the snapshot copier records.**
    ``phenotypicCLI.py:471`` computes ``sha256`` over the CSV's raw bytes and
    stamps it into ``config.metadata_sha256`` *after* state creation, so this
    recomputes rather than reads -- at mint time the state file does not yet
    carry it. A different computation here would make the minted
    ``finalization_input_digest`` disagree with the one every later reader
    derives from the state, and §7.4's late-metadata guarantee would fire on
    every run instead of only on an actual edit.
    """
    import hashlib

    from ..sdk_._io_constants import metadata_csv_deliverable_path

    path = getattr(config, "metadata_csv", None)
    if (
        path is None
        and not config.measure_only
        and not config.process_only_layer
        and config.output_dir is not None
    ):
        # The snapshot fallback, and the GUARD is the load-bearing half.
        #
        # `_prepare_incremental_startup` reassigns `config.metadata_csv` to
        # the existing `deliverables/metadata.csv` **after** the mint, so a
        # continuation that does not re-pass `--metadata` -- the default way
        # a run is continued -- saw `None` here and `state.config` recorded
        # the snapshot's real digest. The two then disagreed on every such
        # invocation, which is exactly the "fires on every run instead of on
        # a real edit" failure this function's docstring says it prevents.
        #
        # Unguarded, the fallback introduces the mirror-image bug: measure
        # and process runs **skip** the snapshot entirely, so pointing at
        # `deliverables/metadata.csv` would invent a digest for a file those
        # modes never wrote. The two arms fail in opposite directions, which
        # is why both are tested.
        path = metadata_csv_deliverable_path(config.output_dir)
    if path is None or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def mint_run_identity(
    config: "ExecutionConfig", *, restart: bool
) -> RunIdentity:
    """Mint the identity of a new or resumed invocation (spec §5.1, §5.4).

    **A writer.** ``restart=True`` bumps and persists the epoch, which is why
    this lives in ``phenotypic._cli`` rather than beside the readers in
    ``sdk_/_run_state.py``.

    **Mint exactly once per invocation, then thread the value (CAN-21).**
    Calling this twice in one run gives that run two generations and burns an
    epoch. ``ExecutionConfig.output_dir`` is ``Optional[Path]``
    (``_cli_types.py:102``), so this cannot make itself idempotent by
    re-reading; the rule is structural instead -- **the CLI entry point mints
    once and passes the** :class:`RunIdentity` **down**, and nothing below it
    calls this function. A second call on the same config object is a
    programming error and says so.

    **A resume is not a restart.** Only ``restart=True`` bumps. A resume mints
    the *same* generation, because its configuration has not changed -- which
    is the entire point of D3. If a resume bumped, every resume would fence
    its own in-flight workers, which is the failure D5 exists to prevent.

    Args:
        config: The invocation's execution configuration.
        restart: ``True`` for ``--restart``, which bumps and persists the
            epoch. ``False`` for a fresh run or a resume.

    Returns:
        A :class:`~phenotypic.sdk_.RunIdentity`. ``scheduler_epoch`` and
        ``owner_generation`` are ``None``: they are liveness facts owned by
        the SLURM lifecycle record and the GUI owner record, neither of which
        exists at mint time, and both are outside
        :meth:`RunIdentity.digest` by design.

    **``--restart --dry-run`` still bumps, and still creates
    ``.phenotypic/``** (gate finding F8). This is the only ``--dry-run``-
    reachable writer the identity change adds, and it is left as-is on
    purpose rather than made conditional:

    * the cost is bounded -- the counter is monotonic, so a dry run costs one
      generation value, never a wrong one;
    * ``--restart`` has already run ``clear_machine_state`` by this point, so
      a dry run under it has written to the tree regardless;
    * making the bump conditional on ``dry_run`` would put a second rule on a
      counter whose whole value is being unconditional. A fence with an
      exception is a fence someone has to remember.

    Worth knowing rather than worth fixing, but a ``--dry-run`` that writes
    tracked state deserves to say so where the write happens.

    Raises:
        RuntimeError: If called twice for the same ``config`` object.
    """
    if getattr(config, _MINTED_FLAG, False):
        raise RuntimeError(
            "run identity already minted for this invocation; mint once at "
            "the CLI entry point and thread the RunIdentity down (CAN-21). "
            "A second mint gives one run two generations and burns a restart "
            "epoch."
        )
    output_dir = config.output_dir
    if output_dir is None:
        raise RuntimeError(
            "cannot mint a run identity without an output directory"
        )
    epoch = (
        bump_restart_epoch(output_dir)
        if restart
        else read_restart_epoch(output_dir)
    )
    # Lazy: this module is imported BY `_cli_slurm_lifecycle`, so it stays
    # import-light at module scope to keep that edge acyclic.
    from ._cli_staged_resume import pipeline_content_digest

    pipeline_sha256 = (
        pipeline_content_digest(config.pipeline_json)
        if config.pipeline_json.is_file()
        else None
    )
    setattr(config, _MINTED_FLAG, True)
    return RunIdentity(
        processing_generation=derive_processing_generation(
            pipeline_sha256=pipeline_sha256,
            per_image_config=per_image_config_digest(config),
            restart_epoch=epoch,
        ),
        restart_epoch=epoch,
        # Liveness, not configuration -- see the Returns note.
        scheduler_epoch=None,
        owner_generation=None,
        # F2: **empty at mint, and populated by the reader.** Not a
        # placeholder -- the field is `canonical_digest(work_ids)`, a pure
        # function of `processing_state.json`, and the minter runs BEFORE
        # that state exists (`work_ids` is not populated until the main
        # state block, several hundred lines below the mint).
        #
        # Four sites compute it the reader's way and agree -- the reader
        # (`_run_state.py:276`) and the three proof writers
        # (`_cli_completion.py:904,1012,1086`). Only the minter disagreed,
        # deriving it from `image_manifest_digest` instead, so the minted
        # identity could never equal a read one: a 100% mismatch for the
        # first caller to compare them.
        #
        # The reader owns this field. `assert_identity_current` skips empty
        # tokens, and `_run_state.py:384` already sets it empty for the
        # unidentified case, so the empty string is the established
        # spelling for "not asserted here" rather than a new convention.
        inventory_digest="",
        # The PROOF-side token (A), which is `pipeline_sha256` verbatim. NOT
        # `per_image_config_digest` -- see this module's A/B block. Writing B
        # here would make every proof this run publishes disagree with every
        # proof already on disk.
        scientific_config_digest=pipeline_sha256 or "",
        finalization_input_digest=canonical_digest(
            {
                "schema_version": FINALIZATION_INPUT_SCHEMA_VERSION,
                "metadata_sha256": _metadata_digest_for(config),
                "include_dataset_column": config.include_dataset_column,
                "no_qc": config.no_qc,
            }
        ),
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
