"""Read-only resolution of a run's completion state.

**Readers only.** Spec §5.2 makes the read/write asymmetry structural: every
function that *publishes* state stays in :mod:`phenotypic._cli`, so a GUI
import of this module cannot reach one. INV-LAYER
(``tests/unit/sdk_/test_run_state_layering.py``) enforces both halves -- no
``phenotypic._cli`` import, and no writer in ``__all__``.

This module reads ``processing_state.json`` as plain JSON and never replays the
event log. That is possible because spec §4.2 demotes the event log out of the
evidence set and deletes ``processing_state.datasets.{completed,failed,started}``
from the file: what remains that a verdict depends on is ``config.work_ids``
and the digests, all literal JSON fields. See OPEN-QUESTIONS Q4.

The four frozen dataclasses are defined in
:mod:`phenotypic.sdk_._state_types` and re-exported here, which is where the
spec's function surface puts them. They live one module down so that
:mod:`phenotypic.sdk_._verification_cache` can cache whole ``ImageState``
objects without this module and that one importing each other.

**On the validation logic duplicated from ``_cli_completion``.** The marker
and proof readers below re-derive what ``valid_image_success``,
``valid_aggregate_snapshot`` and ``valid_run_completion`` decide today. That
is a second home for a format, which this change otherwise exists to remove,
and it is here because INV-LAYER forbids importing the CLI half while P1
moves no consumers. Two things keep it honest until **P6 Task 7** deletes the
CLI copies: every constant either side branches on lives in
``sdk_/_io_constants`` and is imported by both (``SUCCESS_MARKER_VERSION``,
the two artifact kinds, the two proof versions), and
``test_the_sdk_reader_agrees_with_the_cli_validator`` compares the two
implementations image by image over a real tree and its tamperings.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from . import _schema_shape
from ._digests import canonical_digest
from ._io_constants import (
    AGGREGATE_PROOF_VERSION,
    ARTIFACT_KIND_FILE,
    ARTIFACT_KIND_STORE,
    MIGRATION_REMEDY,
    RUN_PROOF_VERSION,
    SUCCESS_MARKER_VERSION,
    aggregate_publication_marker_path,
    datasets_needing_migration,
    gui_launch_owner_path,
    image_completion_marker_path,
    resolve_processing_state_path,
    run_completion_marker_path,
    slurm_lifecycle_path,
    source_image_stem,
    terminal_failures_jsonl_path,
    zarr_store_path,
)
from ._state_types import (
    Completion,
    Depth,
    ImageState,
    RunDiagnostics,
    RunIdentity,
    RunState,
    Verdict,
)
from ._verification_cache import (
    CachedVerification,
    clear_verification_cache,
    entry_is_still_current,
    persist_states,
    remember_states,
    warm_states,
)

#: Grows one name at a time, in the task that defines it. ``run_identity``,
#: ``assert_identity_current``, ``finalization_input_object`` and
#: ``resolve_run_state`` are named by spec §5.2 and belong here, but listing a
#: name this module does not yet bind is ruff **F822** -- an error under the
#: default ``F`` rule set this repo runs -- so each arrives with its own
#: implementation. Keeping the two in step is also what keeps every commit
#: importable, which is the phase-gate contract.
#:
#: ``clear_verification_cache`` is re-exported rather than defined here for the
#: same reason the four types are: spec §5.2 declares the public surface as
#: ``phenotypic.sdk_._run_state``, and the module split below it is a
#: cycle-breaking mechanism, not an interface change. It clears in-process
#: memory and touches no file, so it is not the kind of writer INV-LAYER keeps
#: out of this module.
__all__ = [
    "ImageState",
    "RunDiagnostics",
    "RunIdentity",
    "RunState",
    "assert_identity_current",
    "clear_verification_cache",
    "finalization_input_object",
    "resolve_run_state",
    "run_identity",
]

#: Spec §5.5's object schema. A new finalization input is a bump handled by
#: the reader, never a second tree migration.
FINALIZATION_INPUT_SCHEMA_VERSION = 1


def _read_state_config(output_dir: Path) -> dict[str, object] | None:
    """Return ``processing_state.json``'s ``config`` block, or ``None``.

    Plain JSON, no event-log replay -- see the module docstring and
    OPEN-QUESTIONS Q4. Every failure returns ``None`` rather than raising,
    which is INV-VERDICT's degrade half at its lowest level: an unreadable
    state file must make a run look *less* finished, not make a caller
    explode.

    Args:
        output_dir: Run output root. May be any directory.

    Returns:
        The ``config`` mapping, or ``None`` when the file is absent,
        unreadable, not JSON, not an object, or carries no ``config`` object.
    """
    try:
        raw = json.loads(
            resolve_processing_state_path(output_dir).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, TypeError):
        return None
    config = raw.get("config") if isinstance(raw, dict) else None
    return config if isinstance(config, dict) else None


def _read_json_object(path: Path) -> dict[str, object] | None:
    """Return one JSON object from ``path``, or ``None`` for anything else.

    The shared degrade path for every small sidecar this module reads -- the
    run proof, the aggregate proof, a per-image marker, the two liveness
    records. A truncated write, a directory in place of a file, a JSON array
    where an object belongs: all of them are ``None``, and none of them
    raise.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _optional_str(value: object) -> str | None:
    """Return ``value`` when it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


def _finalization_inputs(config: Mapping[str, object]) -> dict[str, object]:
    """Build spec §5.5's object from an already-read ``config`` block.

    Threaded rather than re-read: :func:`resolve_run_state` would otherwise
    open ``processing_state.json`` four times per call -- once for the
    config, once inside :func:`run_identity`, and twice more for the two
    finalization digests -- on a path the GUI polls every five seconds.
    """
    return {
        "schema_version": FINALIZATION_INPUT_SCHEMA_VERSION,
        "metadata_sha256": config.get("metadata_sha256"),
        "include_dataset_column": config.get("include_dataset_column"),
        "no_qc": config.get("no_qc", False),
    }


def finalization_input_object(output_dir: Path) -> dict[str, object]:
    """Return the versioned finalization-input object (spec §5.5).

    The object, not its digest: a caller that needs the digest passes this to
    :func:`phenotypic.sdk_._digests.canonical_digest`, and a caller that needs
    to *explain* a mismatch -- "the metadata snapshot changed" rather than
    "the digest changed" -- reads the fields.

    ``schema_version`` is what makes adding a fourth input a reader-side bump
    instead of a second tree migration.

    Args:
        output_dir: Run output root.

    Returns:
        ``{"schema_version", "metadata_sha256", "include_dataset_column",
        "no_qc"}``. Every value comes from ``config`` via ``.get`` and is
        ``None`` when the run never recorded it -- a shape with no state file
        at all still returns the four keys.
    """
    return _finalization_inputs(_read_state_config(output_dir) or {})


def _scheduler_epoch(output_dir: Path) -> str | None:
    """Return the SLURM launch generation currently fencing this output.

    Read here rather than imported from ``_cli_slurm_lifecycle`` because
    INV-LAYER forbids this module naming :mod:`phenotypic._cli`. The field is
    read for its *value*, not its liveness -- :func:`_liveness` is what asks
    whether the generation is still running.

    ``generation`` falls back to ``epoch``, matching the v1 records
    ``load_slurm_lifecycle`` still accepts.
    """
    record = _read_json_object(slurm_lifecycle_path(output_dir))
    if record is None:
        return None
    return _optional_str(record.get("generation")) or _optional_str(
        record.get("epoch")
    )


def _owner_generation(output_dir: Path) -> str | None:
    """Return the GUI launch generation recorded for this output."""
    record = _read_json_object(gui_launch_owner_path(output_dir))
    if record is None:
        return None
    return _optional_str(record.get("generation"))


def run_identity(output_dir: Path) -> RunIdentity | None:
    """Return this output's run identity, or ``None`` when it has no state.

    In P1 the tokens are read from the fields today's writers already
    produce: ``processing_generation`` (still a ``uuid4().hex`` until P2),
    ``pipeline_sha256``, the three finalization inputs, and ``work_ids``.
    ``restart_epoch`` defaults to ``0`` because P2 introduces its writer.
    That is what makes this phase independently landable -- the reader works
    on today's trees, before any writer moves.

    ``scheduler_epoch`` and ``owner_generation`` come from the two liveness
    records and are deliberately **outside** :meth:`RunIdentity.digest`: they
    are facts about processes, not about configuration, and folding them in
    would discard the verification cache every time a job is submitted
    against unchanged work.

    Args:
        output_dir: Run output root. May be any directory, including one this
            package has never written to.

    Returns:
        A :class:`RunIdentity`, or ``None`` when there is no readable
        processing state. Never raises.
    """
    config = _read_state_config(output_dir)
    if config is None:
        return None
    return _identity_from(output_dir, config)


def _identity_from(
    output_dir: Path, config: Mapping[str, object]
) -> RunIdentity:
    """Compose the identity from an already-read ``config`` block."""
    restart_epoch = config.get("restart_epoch", 0)
    return RunIdentity(
        processing_generation=str(config.get("processing_generation") or ""),
        # A non-integer is a corrupt field, and INV-VERDICT's degrade half
        # says a corrupt field must not raise out of a reader. `bool` is
        # excluded explicitly because it is an `int` subclass, and `True` is
        # not a restart epoch.
        restart_epoch=(
            restart_epoch
            if isinstance(restart_epoch, int)
            and not isinstance(restart_epoch, bool)
            else 0
        ),
        scheduler_epoch=_scheduler_epoch(output_dir),
        owner_generation=_owner_generation(output_dir),
        inventory_digest=canonical_digest(config.get("work_ids", {})),
        scientific_config_digest=str(config.get("pipeline_sha256") or ""),
        finalization_input_digest=canonical_digest(
            _finalization_inputs(config)
        ),
    )


#: The tokens :meth:`RunIdentity.digest` folds in, in the order
#: :func:`assert_identity_current` reports them. Comparing exactly these is
#: what makes "the identity is current" mean the same thing as "the cache
#: entry may stand": a caller holding an identity from before a job was
#: submitted has a stale ``scheduler_epoch`` and an unchanged configuration,
#: and that is not a reason to hard-error.
_IDENTITY_DIGEST_FIELDS = (
    "processing_generation",
    "restart_epoch",
    "inventory_digest",
    "scientific_config_digest",
    "finalization_input_digest",
)


def assert_identity_current(output_dir: Path, identity: RunIdentity) -> None:
    """Raise unless ``identity`` still describes ``output_dir``'s state.

    D6: a configuration change hard-errors, and it names the **specific**
    token that moved. A generic "identity changed" would make the
    content-derived generation a worse diagnostic than the ``uuid4`` it
    replaces, which would be a strange thing to ship in a change whose
    argument is that content-derived identity is better.

    Only the five tokens :meth:`RunIdentity.digest` folds in are compared --
    see :data:`_IDENTITY_DIGEST_FIELDS`.

    Args:
        output_dir: Run output root.
        identity: The identity the caller believes is current.

    Raises:
        RuntimeError: If the output has no readable processing state, or if
            any fenced token differs. The message names the first differing
            token and both values.
    """
    current = run_identity(output_dir)
    if current is None:
        raise RuntimeError(
            f"Run identity is unavailable: no readable processing state in "
            f"{output_dir}"
        )
    for field in _IDENTITY_DIGEST_FIELDS:
        expected = getattr(identity, field)
        found = getattr(current, field)
        if expected != found:
            raise RuntimeError(
                f"{field} changed: expected {expected!r}, found {found!r}"
            )


# ---------------------------------------------------------------------------
# Per-image verification
# ---------------------------------------------------------------------------

#: ``attributes.phenotypic.metadata_table.snapshot_sha256`` on a store's root
#: ``zarr.json`` -- which metadata snapshot the store's embedded tables were
#: built against (D-A). **P4 Task 2 writes it**; until then the key is absent
#: and the divergence advisory below simply never fires, which is the correct
#: behaviour for a tree that has not recorded the fact.
#:
#: The key is ``metadata_table`` and not ``metadata``, because
#: ``phenotypic.metadata`` is already taken by the ``{protected, public,
#: imported}`` image-metadata sections (``ngff_.py``'s
#: ``PhenotypicAttr.METADATA``). It is read from the **root**, not from the
#: Parquet footer where the digest lives today: a Parquet open per store, on
#: the deep path, from ``sdk_``, is not "one attribute read from a value the
#: store already carries".
_METADATA_TABLE_ATTR = "metadata_table"
_SNAPSHOT_SHA256_ATTR = "snapshot_sha256"

#: U-10's marking. ``--mode migrate`` (P7) publishes per-image records
#: carrying this, and such a record is accepted on **artifact validity alone**
#: -- no ``work_id`` comparison, because a pre-markers tree never had one to
#: compare against. :func:`resolve_run_state` says so in an advisory.
_PROVENANCE_MIGRATED = "migrated"

#: The stage a P1 marker maps onto. Until P3 replaces the reader, today's
#: ``image_complete/<ds>/<stem>.json`` is projected onto a **single-key**
#: ``stages`` map. The single key is a consequence of what today's marker
#: records, not the design: spec §6.1's record carries ``stage1``/``stage2``/
#: ``stage3``/``measured``, and P3 swaps the reader without touching any
#: caller.
_STAGE_MEASURED = "measured"

#: Statuses in the GUI owner record that assert work is in flight. Named
#: rather than derived from the registry's ``_RUN_STATUSES`` minus its
#: terminal set, because ``"unknown"`` is in neither: it is the absence of a
#: claim, and an absent claim must not read as a live worker.
_OWNER_STATUSES_IN_FLIGHT = frozenset({"running", "submitting"})

#: The identity a tree with no readable processing state gets. Every token is
#: empty, so it can never equal a real one and can never match a proof.
#: ``RunState.identity`` is not optional -- a caller that must branch on "is
#: this even a run?" reads the advisory, or calls :func:`run_identity`.
_UNIDENTIFIED = RunIdentity(
    processing_generation="",
    restart_epoch=0,
    scheduler_epoch=None,
    owner_generation=None,
    inventory_digest="",
    scientific_config_digest="",
    finalization_input_digest="",
)


def _stat_tuple(path: Path) -> tuple[int, int] | None:
    """Return ``(size, mtime_ns)`` for a regular file, else ``None``.

    ``ctime_ns`` is absent by design (audit S3) -- it moves on ``chmod``,
    ownership change, hardlink and ``rsync -a``, all routine on GPFS.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (info.st_size, info.st_mtime_ns)


def _digest_file(path: Path) -> str | None:
    """Return one file's SHA-256 hex digest, or ``None`` if unreadable.

    Streamed in 1 MiB chunks: a marker-bound artifact may be a multi-gigabyte
    Parquet, and the deep path walks every one of them.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _fenced_artifact_path(
    output_root: Path, descriptor: Mapping[str, object]
) -> str | None:
    """Verify one marker-bound artifact and return the path that fences it.

    The return value is deliberately the path to **stat next time**, not a
    bool: for a ``store`` descriptor that is the root ``zarr.json`` and not
    the store directory. A directory's ``mtime_ns`` tracks only its own
    entries, so rewriting ``tables/measurements/table.parquet`` inside a
    promoted store would leave a directory-fenced entry looking current --
    spec §0's "a valid root does not imply unchanged contents", reached
    through the cache. ``entry_is_still_current`` refuses directories outright
    (its CONTRACT FOR CALLERS), so getting this wrong costs a permanent cache
    miss rather than a wrong answer; returning the root here is what makes the
    cache work at all.

    Args:
        output_root: The strictly-resolved run output root.
        descriptor: One entry from a marker's ``artifacts`` map.

    Returns:
        The run-root-relative POSIX path to fence on, or ``None`` when the
        descriptor is malformed, escapes the root, or no longer matches disk.
    """
    from .ngff_ import STORE_ROOT_JSON

    if not isinstance(descriptor, dict):
        return None
    relative = descriptor.get("path")
    if not isinstance(relative, str):
        return None
    try:
        resolved = (output_root / relative).resolve()
        resolved.relative_to(output_root)
    except (OSError, ValueError):
        return None
    kind = descriptor.get("kind", ARTIFACT_KIND_FILE)
    if kind == ARTIFACT_KIND_STORE:
        root_json = resolved / STORE_ROOT_JSON
        digest = _digest_file(root_json)
        # `file_fingerprint`'s versioned "sha256:<hex>" spelling, which is
        # what `_artifact_descriptor` writes for a store and only for a store.
        if digest is None or f"sha256:{digest}" != descriptor.get("sha256"):
            return None
        if _stat_tuple(root_json) is None:
            return None
        return f"{relative}/{STORE_ROOT_JSON}"
    if kind == ARTIFACT_KIND_FILE:
        tuples = _stat_tuple(resolved)
        if tuples is None or tuples[0] != descriptor.get("size"):
            return None
        if _digest_file(resolved) != descriptor.get("sha256"):
            return None
        return relative
    # Fail closed: an unrecognized kind is a marker this build cannot
    # certify, never a file descriptor by default.
    return None


def _marker_rejection(
    marker: Mapping[str, object],
    *,
    work_id: str,
    dataset: str,
    image_stem: str,
) -> str | None:
    """Return why ``marker`` cannot certify this image, or ``None``.

    A sentence, not a bool, because the sentence lands in
    ``ImageState.reason`` and is what makes "which images are missing, and
    why?" answerable without re-running anything.
    """
    if marker.get("version") != SUCCESS_MARKER_VERSION:
        return (
            f"marker schema version {marker.get('version')!r} is not "
            f"{SUCCESS_MARKER_VERSION}"
        )
    if marker.get("dataset") != dataset:
        return "marker was written for a different dataset"
    if marker.get("image_stem") != image_stem:
        return "marker was written for a different image"
    if marker.get("provenance") != _PROVENANCE_MIGRATED:
        if marker.get("work_id") != work_id:
            return "marker was written for a different work_id"
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return "marker declares no artifacts"
    return None


def _store_metadata_snapshot(
    output_dir: Path, dataset: str, image_stem: str
) -> str | None:
    """Return which metadata snapshot a store was built against, if it says.

    One plain read of the store's root ``zarr.json`` -- the same small
    document the marker already digests, so it is warm in the page cache by
    the time this is called. ``None`` means the store does not record the
    fact, which is every store until P4 Task 2 starts writing the key.
    """
    from .ngff_ import STORE_ROOT_JSON, PhenotypicAttr

    payload = _read_json_object(
        zarr_store_path(output_dir, dataset, image_stem) / STORE_ROOT_JSON
    )
    if payload is None:
        return None
    attributes = payload.get("attributes")
    if not isinstance(attributes, dict):
        return None
    phenotypic = attributes.get(PhenotypicAttr.ROOT)
    if not isinstance(phenotypic, dict):
        return None
    table = phenotypic.get(_METADATA_TABLE_ATTR)
    if not isinstance(table, dict):
        return None
    return _optional_str(table.get(_SNAPSHOT_SHA256_ATTR))


def _verify_image(
    output_dir: Path,
    output_root: Path,
    *,
    dataset: str,
    image_name: str,
    work_id: str,
    failures: Mapping[str, str],
) -> CachedVerification:
    """Deep-verify one accepted image and record what fences the result.

    Args:
        output_dir: Run output root, as the caller spelled it.
        output_root: The same root, strictly resolved, for containment
            checks.
        dataset: Dataset name from the accepted inventory.
        image_name: Image filename from the accepted inventory.
        work_id: The work id the inventory accepted for that image.
        failures: ``work_id -> exception type`` from the terminal-failure
            journal.

    Returns:
        A :class:`CachedVerification`. Its ``stat_tuples`` are empty for
        anything not verified, which makes such an entry permanently
        non-current and so re-verified on every shallow pass -- exactly what
        an image that might yet succeed needs.
    """
    image_stem = source_image_stem(Path(image_name))
    marker_path = image_completion_marker_path(
        output_dir, dataset, image_stem
    )
    marker = _read_json_object(marker_path)
    fence: dict[str, tuple[int, int]] = {}
    provenance: str | None = None
    reason: str | None = "no readable success marker"
    if marker is not None:
        provenance = _optional_str(marker.get("provenance"))
        reason = _marker_rejection(
            marker, work_id=work_id, dataset=dataset, image_stem=image_stem
        )
        marker_tuple = _stat_tuple(marker_path)
        if reason is None and marker_tuple is None:
            reason = "success marker is not a readable regular file"
        elif reason is None and marker_tuple is not None:
            key = marker_path.relative_to(output_dir).as_posix()
            fence[key] = marker_tuple
        artifacts = marker.get("artifacts")
        if reason is None and isinstance(artifacts, dict):
            for name, descriptor in artifacts.items():
                fenced = _fenced_artifact_path(output_root, descriptor)
                if fenced is None:
                    reason = (
                        f"declared artifact {name!r} no longer matches disk"
                    )
                    break
                fenced_tuple = _stat_tuple(output_root / fenced)
                if fenced_tuple is None:
                    reason = f"declared artifact {name!r} cannot be stat'd"
                    break
                fence[fenced] = fenced_tuple

    if reason is None and marker is not None:
        stage: dict[str, object] = {
            "at": marker.get("completed_at"),
            "mode": marker.get("mode"),
        }
        # Derived, never tracked (D-A): the store already carries which
        # metadata snapshot it was built against, so the divergence advisory
        # is a projection over `images` rather than a second file to keep in
        # sync -- and it costs the shallow path nothing, because the value
        # rides in the cached ImageState.
        snapshot = _store_metadata_snapshot(output_dir, dataset, image_stem)
        if snapshot is not None:
            stage[_SNAPSHOT_SHA256_ATTR] = snapshot
        if provenance is not None:
            stage["provenance"] = provenance
        return CachedVerification(
            state=ImageState(
                work_id=work_id,
                dataset=dataset,
                image_stem=image_stem,
                stages={_STAGE_MEASURED: stage},
                verdict="verified",
            ),
            stat_tuples=fence,
        )

    failure = failures.get(work_id)
    verdict: Verdict = "failed" if failure is not None else "unverified"
    return CachedVerification(
        state=ImageState(
            work_id=work_id,
            dataset=dataset,
            image_stem=image_stem,
            stages={},
            verdict=verdict,
            reason=(
                f"terminal failure ({failure}); {reason}"
                if failure is not None
                else reason
            ),
        ),
        stat_tuples={},
    )


# ---------------------------------------------------------------------------
# The written authorities (spec §4.1)
# ---------------------------------------------------------------------------


def _accepted_inventory(
    work_ids: object,
) -> tuple[tuple[str, str, str], ...]:
    """Flatten ``config.work_ids`` into ``(dataset, image, work_id)`` rows.

    The **accepted inventory** authority. A directory listing is a different
    question -- "what is on disk" rather than "what did this run accept" --
    so nothing here walks the tree. Malformed rows are skipped rather than
    raised on, so a partially corrupt inventory degrades toward
    ``incomplete``.
    """
    if not isinstance(work_ids, dict):
        return ()
    rows: list[tuple[str, str, str]] = []
    for dataset, images in work_ids.items():
        if not isinstance(dataset, str) or not isinstance(images, dict):
            continue
        for image_name, work_id in images.items():
            if isinstance(image_name, str) and isinstance(work_id, str):
                rows.append((dataset, image_name, work_id))
    return tuple(rows)


def _terminal_failures(output_dir: Path) -> dict[str, str]:
    """Return ``work_id -> exception type`` from the terminal journal.

    The **terminal failures** authority: a failure leaves no artifact, so it
    cannot be derived from the tree.

    Read as plain lines rather than under ``_cli_file_locking``'s reader
    (INV-LAYER). The journal is append-only with whole-line atomic appends,
    so a concurrent write can at worst leave a torn final line, which parses
    as malformed and is skipped -- and an unreadable journal degrades to "no
    failures", which moves the verdict from ``failed`` toward ``incomplete``
    and never the other way.
    """
    try:
        content = terminal_failures_jsonl_path(output_dir).read_text(
            encoding="utf-8"
        )
    except (OSError, ValueError):
        return {}
    failures: dict[str, str] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        work_id = _optional_str(row.get("work_id"))
        if work_id is None:
            continue
        failures[work_id] = str(row.get("exception_type") or "unknown")
    return failures


def _live_authority(output_dir: Path) -> str | None:
    """Return a liveness authority that reports work in flight, or ``None``.

    Rule 2 of the verdict ladder has **two** halves, and the second is not
    decoration (CAN-24): nothing in this codebase repairs
    ``gui_launch_owner.json``, so a SIGKILLed GUI pins ``status: "running"``
    forever (audit S7, verified). An authority that is merely *asserting*
    work must therefore also be shown to be alive, or rule 2 pins the verdict
    at ``active`` permanently and the ladder is unsound.

    The SLURM lifecycle fence is the scheduler's claim, and the scheduler is
    the external system that owns it -- this module does not shell out to
    ``squeue`` (DEFERRED D-1 keeps the observer's decision tree out of this
    change), so an ``active`` fence is taken at face value. The GUI owner
    record is a *local process* claim, so it is only believed while the
    process it names is alive. A record with no ``pid`` -- a SLURM launch, or
    one rehydrated from disk -- cannot be probed and is therefore not
    believed here; the lifecycle fence is that run's liveness authority.

    Returns:
        The filename of the authority reporting live work, or ``None``.
    """
    lifecycle = _read_json_object(slurm_lifecycle_path(output_dir))
    if lifecycle is not None and lifecycle.get("active") is True:
        return slurm_lifecycle_path(output_dir).name
    owner = _read_json_object(gui_launch_owner_path(output_dir))
    if owner is not None and owner.get("status") in _OWNER_STATUSES_IN_FLIGHT:
        pid = owner.get("pid")
        if (
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and _process_is_alive(pid)
        ):
            return gui_launch_owner_path(output_dir).name
    return None


def _process_is_alive(pid: int) -> bool:
    """Return whether ``pid`` names a live process on this host.

    ``psutil`` is already a hard dependency of the CLI half; an environment
    without it degrades to "not alive", which moves the verdict away from
    ``active`` and toward ``incomplete`` -- INV-VERDICT's direction.
    """
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a hard dependency
        return False
    try:
        return bool(psutil.pid_exists(pid))
    except OSError:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# The content proofs
# ---------------------------------------------------------------------------


def _valid_run_proof(output_dir: Path) -> dict[str, object] | None:
    """Return the run proof when it is structurally valid, else ``None``."""
    marker = _read_json_object(run_completion_marker_path(output_dir))
    if marker is None:
        return None
    if marker.get("version") != RUN_PROOF_VERSION:
        return None
    if marker.get("status") != "complete":
        return None
    if marker.get("finalizer_succeeded") is not True:
        return None
    return marker


def _valid_aggregate_proof(output_dir: Path) -> dict[str, object] | None:
    """Return the aggregate proof when every required output still matches.

    This is the only place the deep path hashes a *run-level* artifact, and
    the cost is O(1) in images -- three or four deliverables -- which is why
    the shallow path re-checks it rather than caching it. Audit §4's cost is
    the ~10^4 per-image reads and hashes, and those are what the verification
    cache removes.
    """
    marker = _read_json_object(aggregate_publication_marker_path(output_dir))
    if marker is None or marker.get("version") != AGGREGATE_PROOF_VERSION:
        return None
    outputs = marker.get("required_outputs")
    if not isinstance(outputs, dict) or not outputs:
        return None
    try:
        output_root = Path(output_dir).resolve()
    except OSError:
        return None
    for descriptor in outputs.values():
        if _fenced_artifact_path(output_root, descriptor) is None:
            return None
    return marker


def _accepted_finalization_digests(
    config: Mapping[str, object],
) -> frozenset[str]:
    """Return every spelling of the finalization-input digest a proof may use.

    Two, for one release only. Spec §5.5 makes the object **versioned**, and
    :func:`finalization_input_object` returns that form; today's publishers
    (``_cli_completion.publish_aggregate_snapshot``) digest the same three
    values with no ``schema_version`` key. §5.5's own rule is that a schema
    change is *handled by the reader*, so the reader accepts both rather than
    requiring a tree migration. **P4 bumps the publishers and drops the
    unversioned spelling from this set.**

    Accepting both weakens nothing: both digests are functions of exactly
    ``metadata_sha256``, ``include_dataset_column`` and ``no_qc``, so a
    change to any of the three moves both.
    """
    versioned = _finalization_inputs(config)
    unversioned = {
        key: value
        for key, value in versioned.items()
        if key != "schema_version"
    }
    return frozenset(
        {canonical_digest(versioned), canonical_digest(unversioned)}
    )


def _run_proof_covers_current_inventory(
    output_dir: Path,
    config: Mapping[str, object],
    identity: RunIdentity,
    images: Mapping[str, ImageState],
    *,
    inventory_present: bool,
) -> bool:
    """Rule 1 of the verdict ladder, with **both** of §4.3's clauses (U-2).

    Clause 1 -- every accepted image has a valid proof -- is what makes
    completion O(N) in per-image proofs, and therefore what makes the
    verification cache load-bearing rather than marginal.

    Clause 2 is the **five** comparisons ``current_aggregate_is_current``
    makes today, not the one an earlier draft kept (CAN-4). Each is
    load-bearing: without ``inventory_digest`` a new image under a rolling
    input never invalidates completion; without
    ``finalization_input_digest`` §7.4's late-metadata guarantee stops
    working, since a metadata edit leaves ``work_ids`` untouched and nothing
    else notices; without ``scientific_config_digest`` a pipeline edit leaves
    the run reading ``complete``; without ``source_set_digest`` a partial
    shard set is undetectable (CAN-5); and ``source_image_count`` is a cheap
    arity cross-check on the same.

    ``--mode process`` takes a **different rule 1** and always has: a process
    run publishes no aggregate proof at all, so its ``source_set_digest`` and
    ``source_image_count`` do not exist and its ``finalization_input_digest``
    digests ``{"process_only_layer": ...}``. Three of the five comparisons
    are inapplicable rather than merely different, and ``_cli_completion``
    carries five carve-outs for exactly this. A flat conjunction that ignored
    them would make every process tree read ``incomplete`` forever (N-4).
    """
    if not inventory_present:
        # U-6: the pre-markers shape is schema 2.0.0 with no `work_ids`. There
        # is no accepted inventory for a proof to cover, so rule 1 cannot
        # fire -- and `requires_conversion` is what turns that into an
        # actionable message.
        return False
    if not all(image.verdict == "verified" for image in images.values()):
        return False
    proof = _valid_run_proof(output_dir)
    if proof is None:
        return False
    if str(proof.get("inventory_digest") or "") != identity.inventory_digest:
        return False
    if (
        str(proof.get("scientific_config_digest") or "")
        != identity.scientific_config_digest
    ):
        return False

    process_layer = config.get("process_only_layer")
    if process_layer:
        return str(proof.get("finalization_input_digest") or "") == (
            canonical_digest({"process_only_layer": process_layer})
        )

    if (
        str(proof.get("finalization_input_digest") or "")
        not in _accepted_finalization_digests(config)
    ):
        return False

    binding = _source_set_binding(output_dir, proof)
    if binding is None:
        return False
    verified = sorted(
        work_id
        for work_id, image in images.items()
        if image.verdict == "verified"
    )
    return (
        binding.get("source_set_digest") == canonical_digest(verified)
        and binding.get("source_image_count") == len(verified)
    )


def _source_set_binding(
    output_dir: Path, proof: Mapping[str, object]
) -> Mapping[str, object] | None:
    """Return the proof carrying ``source_set_digest``/``source_image_count``.

    U-4 cuts ``publication_id`` and puts ``source_set_digest`` in the **run**
    proof, so the aggregate-to-run binding is stated directly instead of
    through an opaque hash. That writer change lands in P4; until it does,
    today's run proof carries neither field and the values live in the
    aggregate proof, bound to the run proof by ``publication_id``.

    Both shapes are read here so that P1 lands on today's trees and keeps
    working across P4's writer bump, with no window in which the two
    comparisons silently stop being made -- which is the failure CAN-5 names.
    """
    if "source_set_digest" in proof:
        return proof
    aggregate = _valid_aggregate_proof(output_dir)
    if aggregate is None:
        return None
    if proof.get("publication_id") != aggregate.get("publication_id"):
        return None
    return aggregate


# ---------------------------------------------------------------------------
# Advisories -- derived, and never a gate
# ---------------------------------------------------------------------------


def _advisories(
    output_dir: Path,
    config: Mapping[str, object],
    images: Mapping[str, ImageState],
) -> tuple[str, ...]:
    """Return this run's advisories (spec §4.3).

    **An advisory is never a gate.** Each entry names a thing a reader may
    want to act on; none of them changes ``completion``. Today a
    half-migrated tree reaches ``contradictory`` and flags the whole output
    read-only for a reason the user cannot act on, which is the behaviour
    this replaces.
    """
    notes: list[str] = []

    # The schema-shape advisory, which is §4.3's READER half of the same
    # detection `_cli_schema_gate.refuse_unconverted_schema` uses as its
    # writer half. It replaces a hand-rolled test of signal 5 (`work_ids`
    # absent) that lived here: one detection, two audiences, one home.
    #
    # `SCHEMA_GATE_ARMED` is read through the module rather than imported as a
    # value ON PURPOSE. It is the flag's one mutable home, so a test arming
    # the advisory patches `_schema_shape` and this call sees it; binding the
    # value at import here would silently stop the test controlling anything,
    # which `test_the_gui_reports_rather_than_refuses` would then fail.
    #
    # Why it is gated at all: at P1 the legacy shape and the current shape are
    # the same shape, so `requires_conversion` returns CONVERT for every tree
    # the running build writes. An ungated advisory would banner "run
    # `--mode migrate`" on every GUI output until P3 -- advice the user cannot
    # act on, since migrate does not convert `.phenotypic/` until P7 Tasks 2,
    # 2b and 3. An advisory that is always on teaches people to ignore the one
    # that will matter.
    if _schema_shape.SCHEMA_GATE_ARMED:
        conversion = _schema_shape.describe_conversion_advisory(output_dir)
        if conversion is not None:
            notes.append(conversion)

    datasets = datasets_needing_migration(output_dir)
    if datasets:
        notes.append(
            f"Unconverted .h5 results remain in {', '.join(datasets)}. Run "
            f"`{MIGRATION_REMEDY}` to convert them; this is advisory and "
            "does not gate the verdict."
        )

    current_metadata = config.get("metadata_sha256")
    diverged = sorted(
        f"{image.dataset}/{image.image_stem}"
        for image in images.values()
        if _stage_value(image, _SNAPSHOT_SHA256_ATTR)
        not in (None, current_metadata)
    )
    if diverged:
        notes.append(
            "These stores were built against an earlier metadata snapshot "
            f"than the run's current one: {', '.join(diverged)}. Their "
            "measurements are valid; their embedded metadata predates the "
            "current deliverables/metadata.csv. Advisory only."
        )

    migrated = sorted(
        f"{image.dataset}/{image.image_stem}"
        for image in images.values()
        if _stage_value(image, "provenance") == _PROVENANCE_MIGRATED
    )
    if migrated:
        notes.append(
            "The configuration fence is unavailable for these migrated "
            f"images: {', '.join(migrated)}. They were accepted on artifact "
            "validity alone because the tree they came from never recorded a "
            "work_id, so a later run under a different pipeline will reuse "
            "them rather than reprocess them. Reprocessing any of them "
            "clears the marking. Advisory only."
        )
    return tuple(notes)


def _stage_value(image: ImageState, key: str) -> object | None:
    """Read one value out of an image's ``measured`` stage, if present.

    Advisories are projections over ``images`` with **no I/O** -- which is
    what lets the shallow path emit exactly the same advisories as the deep
    path it reuses, instead of losing them or paying a per-image read to keep
    them.
    """
    stage = image.stages.get(_STAGE_MEASURED)
    if not isinstance(stage, Mapping):
        return None
    return stage.get(key)


# ---------------------------------------------------------------------------
# The one reader
# ---------------------------------------------------------------------------


def _resolve_images(
    output_dir: Path,
    identity: RunIdentity,
    inventory: Sequence[tuple[str, str, str]],
    failures: Mapping[str, str],
    requested_depth: Depth,
) -> tuple[dict[str, ImageState], Depth]:
    """Resolve every accepted image, reusing the cache where it is current.

    Returns the images and **the depth actually performed**, which is not
    always the depth asked for: a ``"shallow"`` request over a cold or stale
    cache is a deep pass and says so. ``depth`` is what a caller reads to
    know whether the answer is authoritative, and "mostly shallow" is not a
    useful third value -- so any escalation at all reports ``"deep"``.

    ``warm_states`` reads tier 1 (in process) and falls back to tier 2 (
    ``.phenotypic/verification_cache.json``, U-11) -- but the loop below does
    not know or care which tier an entry came from, because both are gated by
    the same :func:`entry_is_still_current` call. That is what keeps the
    on-disk tier a cache: it changes which pass is skipped and never which
    verdict is reached.
    """
    try:
        output_root = Path(output_dir).resolve()
    except OSError:
        output_root = Path(output_dir).absolute()
    warm = (
        warm_states(output_dir, identity.digest())
        if requested_depth == "shallow"
        else None
    )
    entries: dict[str, CachedVerification] = {}
    escalated = False
    for dataset, image_name, work_id in inventory:
        entry = warm.get(work_id) if warm is not None else None
        if entry is not None and entry_is_still_current(output_dir, entry):
            entries[work_id] = entry
            continue
        escalated = True
        entries[work_id] = _verify_image(
            output_dir,
            output_root,
            dataset=dataset,
            image_name=image_name,
            work_id=work_id,
            failures=failures,
        )
    # Wholesale replacement under the current identity (CAN-28): entries
    # minted under any other identity are already unusable, so there is no
    # eviction policy to get wrong.
    remember_states(output_dir, identity.digest(), entries)
    # Tier 2 is written only when this pass actually deep-verified something
    # (U-11). A fully warm shallow pass changed nothing on disk, so rewriting
    # the file would put a per-image-sized write on the observer's 2 s tick
    # and the viewer's 5-10 s poll -- the two cadences the cache exists to
    # make cheap.
    if escalated:
        persist_states(output_dir, identity.digest(), entries)
    performed: Depth = (
        "shallow"
        if requested_depth == "shallow" and warm is not None and not escalated
        else "deep"
    )
    return {
        work_id: entry.state for work_id, entry in entries.items()
    }, performed


def resolve_run_state(
    output_dir: Path, *, depth: Depth = "deep"
) -> RunState:
    """Resolve one run's completion state (spec §4.3, §9).

    Verdict precedence is total and ordered (OPEN-QUESTIONS Q2):
    ``complete`` > ``active`` > ``failed`` > ``incomplete``. First match
    wins. ``contradictory`` does not exist.

    ``complete`` outranks ``active`` because a run proof covers the
    **current** inventory: a live worker at that point is either fenced by
    ``restart_epoch`` or belongs to a new invocation that has already changed
    the inventory, in which case rule 1 does not fire and this is not the
    case being decided. ``active`` outranks ``failed`` so that a failure from
    a previous attempt cannot mask an attempt currently retrying it.

    ``depth="shallow"`` re-stats the verification cache's recorded tuples --
    tier 1 in process, tier 2 from ``.phenotypic/verification_cache.json``
    when tier 1 is cold (U-11) -- and falls through to a deep pass for any
    image that is absent from the cache, moved, minted under a different
    identity, or unreadable. It **never** yields a positive verdict from a
    cache entry alone (INV-VERDICT): a cached entry can only ever license
    *skipping* a re-verification the caller already performed, and the
    run-level proofs are re-verified on every call regardless. A pass that
    deep-verified anything rewrites tier 2; a fully warm one writes nothing.

    Args:
        output_dir: Run output root. May be any directory, including one this
            package has never written to.
        depth: ``"deep"`` re-verifies every declared artifact's content and
            repopulates the cache. ``"shallow"`` re-stats instead. See spec
            §9's caller/depth table.

    Returns:
        A :class:`RunState`. **Never raises** for an unreadable or absent
        tree -- every parse failure degrades toward ``incomplete``
        (INV-VERDICT's degrade half). ``RunState.depth`` reports the depth
        actually performed, which for a cold ``"shallow"`` call is
        ``"deep"``.
    """
    output_dir = Path(output_dir)
    config = _read_state_config(output_dir)
    identity = None if config is None else _identity_from(output_dir, config)
    now = datetime.now(timezone.utc)
    if config is None or identity is None:
        return RunState(
            completion="incomplete",
            identity=_UNIDENTIFIED,
            images={},
            advisories=(
                "No readable processing state under this directory, so it "
                "has no run identity and no completion to establish.",
            ),
            diagnostics=RunDiagnostics(accepted=0, verified=0, failed=0),
            depth="deep",
            verified_at=now,
        )

    # `.get`, never a subscript, on every read of this mapping (flow-r4 N-4):
    # U-6's detection signal is the ABSENCE of `work_ids`, so a subscript
    # would raise KeyError from inside the one function whose job is to
    # classify that tree -- and this function's contract is that it never
    # raises.
    inventory_present = isinstance(config.get("work_ids"), dict)
    inventory = _accepted_inventory(config.get("work_ids"))
    images, performed = _resolve_images(
        output_dir,
        identity,
        inventory,
        _terminal_failures(output_dir),
        depth,
    )

    if _run_proof_covers_current_inventory(
        output_dir,
        config,
        identity,
        images,
        inventory_present=inventory_present,
    ):
        completion: Completion = "complete"
    elif _live_authority(output_dir) is not None:
        completion = "active"
    elif any(image.verdict == "failed" for image in images.values()):
        completion = "failed"
    else:
        completion = "incomplete"

    return RunState(
        completion=completion,
        identity=identity,
        images=images,
        advisories=_advisories(output_dir, config, images),
        diagnostics=RunDiagnostics(
            accepted=len(images),
            verified=sum(
                1 for image in images.values() if image.verdict == "verified"
            ),
            failed=sum(
                1 for image in images.values() if image.verdict == "failed"
            ),
        ),
        depth=performed,
        verified_at=now,
    )
