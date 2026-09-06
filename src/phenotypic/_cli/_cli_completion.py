"""Marker-last per-image success publication and validation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from phenotypic.sdk_ import (
    AGGREGATE_PROOF_VERSION,
    ARTIFACT_KIND_FILE,
    ARTIFACT_KIND_STORE,
    CommitGuard,
    DIR_IMAGE_COMPLETE,
    DIR_IMAGE_RECORDS,
    RUN_PROOF_VERSION,
    STORE_ROOT_JSON,
    SUCCESS_MARKER_VERSION,
    aggregate_publication_marker_path,
    atomic_write_json,
    file_fingerprint,
    image_completion_marker_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
    publication_commit,
    progress_dir,
    run_completion_marker_path,
    source_image_stem,
    validated_published_metadata_migration_targets,
)
from phenotypic.sdk_._digests import canonical_digest
from phenotypic.sdk_._image_record import (
    STAGE_MEASURED,
    read_image_record,
    record_rejection,
)
from phenotypic.sdk_._run_state import (
    fenced_artifact_path,
    marker_rejection,
)

# ``SUCCESS_MARKER_VERSION``, ``ARTIFACT_KIND_FILE``/``ARTIFACT_KIND_STORE``
# and the two proof versions are imported above from ``sdk_/_io_constants``
# and re-exported here, so every module that imports them from this one is
# unchanged. They moved because the run-state reader must check the same
# numbers and INV-LAYER forbids it importing this module: two copies of a
# version that gates a *completion* verdict is the one duplication that can
# silently manufacture a false ``complete``.

# ``STORE_ROOT_JSON`` (imported above) is the root metadata document an
# OME-Zarr store is fingerprinted by. ``promote_store`` writes it **last**, so
# its digest covers the whole promoted store: any later re-promote replaces it
# and invalidates the marker. Fingerprinting the store *directory* instead
# would be a constant function of the path -- ``paths_fingerprint`` emits one
# sentinel byte for a directory and does not recurse -- and would certify a
# store whose contents changed.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_descriptor(resolved: Path, relative: Path) -> dict[str, object]:
    """Describe one marker-bound artifact as a file or as a store.

    A store is a *directory*, which the file descriptor cannot express:
    ``_sha256`` opens its argument and ``stat().st_size`` on a directory is a
    filesystem detail, not a content fingerprint. Store descriptors therefore
    carry no ``size`` and key their ``sha256`` on the root ``zarr.json``'s
    **contents** via :func:`file_fingerprint` -- content-only, so the
    descriptor stays relocatable exactly like every file descriptor
    (``paths_fingerprint`` would fold the absolute resolved path in, and this
    tree is routinely reached through more than one mount; ledger FLOW-3).

    Args:
        resolved: The strictly-resolved artifact path.
        relative: Its path relative to the run output root.

    Returns:
        A JSON-serializable descriptor tagged with its ``kind``.

    Raises:
        FileNotFoundError: A store directory with no root ``zarr.json``.
    """
    if resolved.is_dir():
        return {
            "path": relative.as_posix(),
            "kind": ARTIFACT_KIND_STORE,
            "sha256": file_fingerprint(resolved / STORE_ROOT_JSON),
        }
    return {
        "path": relative.as_posix(),
        "kind": ARTIFACT_KIND_FILE,
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _store_artifact_matches(
    artifact: Path, descriptor: Mapping[str, object]
) -> bool:
    """Return whether a store on disk still matches its descriptor.

    A promoted store always has a root ``zarr.json``, so testing for that one
    regular file covers "not a directory", "not a store", and "an interrupted
    re-promote" in a single check.
    """
    root_json = artifact / STORE_ROOT_JSON
    if not root_json.is_file():
        return False
    return file_fingerprint(root_json) == descriptor.get("sha256")


def image_data_artifact(
    output_dir: Path,
    output_manager: object,
    dataset: str,
    image_stem: str,
) -> tuple[str, Path]:
    """Return the ``(key, path)`` of the per-image data artifact to certify.

    Every forward path now writes an OME-Zarr store; only a **legacy tree**
    carried over from an older release still has a per-image ``.h5``. Both
    have to be describable from one place.

    The artifact returned for a store is the **store directory**, and the
    marker records it as ``kind: "store"``. The digest still keys on the root
    ``zarr.json``'s contents (see :data:`STORE_ROOT_JSON`) -- naming the
    directory is what lets the marker say *what class of thing* it certifies,
    without any caller having to parse a path string to find out.

    Because the root ``zarr.json`` is written **last** by ``promote_store``,
    its digest covers the whole promoted store: any later re-promote replaces
    it and invalidates the marker. That is why no store write may follow
    ``publish_image_success`` on any path (ledger **FLOW-6**).

    The ``"hdf"`` fallback is **not** dead code. Nothing on a forward path
    writes a per-image ``.h5`` any more, but
    ``phenotypicCLI._migrate_legacy_success_evidence`` mints markers for
    **legacy trees**, which have an ``.h5`` and no store, and that is exactly
    the caller this branch serves.

    Args:
        output_dir: Run output root.
        output_manager: The run's :class:`OutputManager` (HDF fallback only).
        dataset: Dataset name.
        image_stem: Image stem.

    Returns:
        ``("store", <store>)`` when a store exists, else
        ``("hdf", results/<ds>/hdf/<stem>.h5)``.
    """
    from phenotypic.sdk_ import zarr_store_path

    store = zarr_store_path(output_dir, dataset, image_stem)
    if (store / STORE_ROOT_JSON).is_file():
        return ARTIFACT_KIND_STORE, store
    return "hdf", output_manager.get_output_path(  # type: ignore[attr-defined]
        dataset, "hdf", image_stem
    )


def publish_image_success(
    output_dir: Path,
    *,
    work_id: str,
    dataset: str,
    relative_image_path: str,
    image_stem: str,
    mode: str,
    attempt_id: str,
    lifecycle_epoch: str,
    artifacts: Mapping[str, Path],
    expected_artifact_descriptors: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    source_provenance: Mapping[str, object] | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Validate artifacts and atomically publish the image marker last."""
    if os.environ.get("SLURM_JOB_ID"):
        from ._cli_slurm_lifecycle import load_slurm_lifecycle

        lifecycle = load_slurm_lifecycle(output_dir)
        if lifecycle is not None and (
            lifecycle.get("generation") != lifecycle_epoch
            or lifecycle.get("active") is not True
        ):
            raise RuntimeError(
                "Cannot publish image success for a stale SLURM lifecycle"
            )
    # Resolves only. The DESCRIPTORS are built by `publish_image_record`,
    # which owns the record's shape -- this loop stays because it is what
    # turns an escaping artifact into a named `ValueError` rather than the
    # writer's generic failure, and because `_validate_expected_artifacts`
    # needs the resolved paths.
    resolved_artifacts: dict[str, Path] = {}
    output_root = output_dir.resolve()
    for name, artifact in artifacts.items():
        resolved = artifact.resolve(strict=True)
        try:
            resolved.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(
                f"Artifact escapes output root: {artifact}"
            ) from exc
        resolved_artifacts[name] = resolved

    def _validate_expected_artifacts() -> None:
        if expected_artifact_descriptors is None:
            return
        for name, expected in expected_artifact_descriptors.items():
            artifact = resolved_artifacts.get(name)
            if artifact is None:
                raise RuntimeError(
                    f"Expected marker artifact is missing: {name}"
                )
            actual = _artifact_descriptor(
                artifact,
                artifact.relative_to(output_root),
            )
            if actual != dict(expected):
                raise RuntimeError(
                    f"Marker artifact changed before publication: {name}"
                )

    _validate_expected_artifacts()

    # D1 is a CLEAN BREAK: the record replaces `image_complete/`, and nothing
    # dual-writes. A tree carrying `image_complete/` and no `images/` is a
    # legacy tree, which `--mode migrate` converts and every writing mode now
    # refuses -- which is why `SCHEMA_GATE_ARMED` flips in this same commit.
    # A dual write would leave the gate unable to tell the two shapes apart.
    #
    # DELEGATED, not restated. `publish_image_record` owns the record's shape,
    # its `stages` merge (CAN-6 rule 1) and the provenance default, so this
    # function contributes only the `measured` stage and its own artifact
    # revalidation. Rebuilding the payload here would be the second writer of
    # one schema, which is the defect P3 exists to remove.
    from ._cli_image_record import publish_image_record

    return publish_image_record(
        output_dir,
        work_id=work_id,
        dataset=dataset,
        image_stem=image_stem,
        relative_image_path=Path(relative_image_path).as_posix(),
        mode=mode,
        stages={
            STAGE_MEASURED: {
                "at": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                )
            }
        },
        artifacts=artifacts,
        attempt_id=attempt_id,
        lifecycle_epoch=lifecycle_epoch,
        source_provenance=source_provenance,
        pre_replace=(
            _validate_expected_artifacts
            if expected_artifact_descriptors is not None
            else None
        ),
        commit_guard=commit_guard,
    )


def valid_image_success(
    output_dir: Path,
    *,
    dataset: str,
    image_stem: str,
    work_id: str,
) -> bool:
    """Return whether the record and every declared artifact match disk.

    A ``bool`` over the shared readers, deliberately: the predicate is
    ``record_rejection`` and the artifact walk is ``fenced_artifact_path``,
    so this and ``resolve_run_state`` cannot return opposite verdicts for the
    same image. Keeping the signature keeps this function's ~20 callers
    untouched.

    **It reads the record now, not ``image_complete/``** (D1, clean break).
    The predicate moved with it, and moved *as one function* -- gate finding
    IMPL-F3 spent an increment merging two readers that disagreed on migrated
    markers, and re-splitting them here would have reproduced that defect
    against the new schema while looking like ordinary migration in the diff.

    Two clauses this function no longer spells and must not start spelling
    again: the ``work_id`` relaxation for a migrated record (U-10), and
    CAN-23's "a record with no artifacts certifies nothing" -- which matters
    more after the collapse than before it, because a Stage-2 worker now
    writes ``stages.stage2`` into this same file with no artifacts at all.
    Both live in ``record_rejection``.
    """
    record = read_image_record(output_dir, dataset, image_stem)
    if record is None:
        return False
    if (
        record_rejection(
            record,
            work_id=work_id,
            dataset=dataset,
            image_stem=image_stem,
        )
        is not None
    ):
        return False
    # `record_rejection` has already established that `artifacts` is a
    # non-empty `dict` (CAN-23), so neither guard below can fire today.
    # They are guards rather than a `cast` on purpose: a `cast` asserts the
    # ordering and goes silent if someone reorders these two blocks, while a
    # guard that survives the reorder returns `False` -- INV-VERDICT's degrade
    # half. Previously this narrowing was done by `AttributeError` in the
    # `except` clause below, i.e. by a crash the handler swallowed.
    #
    # `dict` and not `Mapping`, because this restates `record_rejection`'s own
    # `isinstance(artifacts, dict)` and has to use the same predicate: a
    # non-dict mapping passing here and failing there would put the two
    # readers back into the disagreement IMPL-F3 spent an increment merging.
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    try:
        output_root = output_dir.resolve()
        for descriptor in artifacts.values():
            if not isinstance(descriptor, dict):
                return False
            if fenced_artifact_path(output_root, descriptor) is None:
                return False
    except (OSError, ValueError, AttributeError):
        return False
    return True


def refresh_success_markers_after_metadata_migration(
    output_dir: Path,
    *,
    receipt_paths: Iterable[Path] = (),
) -> int:
    """Refresh marker descriptors for receipt-certified schema rewrites.

    Historical schema-3 metadata receipts could rewrite per-image files. This
    compatibility bridge preserves their existing success authority without
    blessing unrelated changes. Schema-4 bundle-durable receipts exclude
    Task-1-owned external Parquets and therefore need no marker refresh.

    Args:
        output_dir: Existing run-output root.
        receipt_paths: Receipts just validated by the current migration phase.
            Invalid explicit receipts fail the operation. Malformed historical
            scan candidates are ignored as incomplete recovery evidence.

    Returns:
        Number of success markers whose artifact descriptors were refreshed.

    Raises:
        RuntimeError: A marker-bound artifact changed outside a certified
            metadata migration transition.
    """
    from ._cli_state_management import load_processing_state

    output_root = Path(output_dir).resolve()
    state = load_processing_state(output_root)
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        return 0
    raw_work_ids = state.config.get("work_ids")
    if not isinstance(raw_work_ids, dict):
        return 0

    explicit = {Path(path).resolve() for path in receipt_paths}
    candidates = set(explicit)
    candidates.update(
        (output_root / ".phenotypic" / "metadata_migration").glob(
            "metadata-schema-*.json"
        )
    )
    results_root = output_root / "results"
    if results_root.is_dir():
        candidates.update(
            results_root.rglob(".metadata_migration/metadata-schema-*.json")
        )

    transitions: dict[Path, tuple[str, str]] = {}
    for receipt_path in sorted(candidates):
        try:
            certified = validated_published_metadata_migration_targets(
                receipt_path
            )
        except (OSError, TypeError, ValueError):
            if receipt_path in explicit:
                raise
            continue
        for artifact, source_fingerprint, post_fingerprint in certified:
            resolved = artifact.resolve()
            try:
                resolved.relative_to(output_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Metadata migration target escapes output root: {artifact}"
                ) from exc
            transition = (
                source_fingerprint.removeprefix("sha256:"),
                post_fingerprint.removeprefix("sha256:"),
            )
            previous = transitions.get(resolved)
            if previous is not None and previous != transition:
                raise RuntimeError(
                    f"Conflicting metadata migration receipts for {resolved}"
                )
            transitions[resolved] = transition

    refreshed = 0
    for dataset, raw_images in raw_work_ids.items():
        if not isinstance(dataset, str) or not isinstance(raw_images, dict):
            continue
        for image_name, work_id in raw_images.items():
            if not isinstance(image_name, str) or not isinstance(work_id, str):
                continue
            stem = source_image_stem(Path(image_name))
            marker_path = image_completion_marker_path(
                output_root, dataset, stem
            )
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                not isinstance(marker, dict)
                or marker.get("version") != SUCCESS_MARKER_VERSION
                or marker.get("work_id") != work_id
                or marker.get("dataset") != dataset
                or marker.get("image_stem") != stem
            ):
                continue
            artifacts = marker.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                continue

            changed = False
            for descriptor in artifacts.values():
                if not isinstance(descriptor, dict):
                    raise RuntimeError(
                        f"Invalid success marker: {marker_path}"
                    )
                relative = descriptor.get("path")
                if not isinstance(relative, str):
                    raise RuntimeError(
                        f"Invalid success marker: {marker_path}"
                    )
                artifact = (output_root / relative).resolve()
                try:
                    artifact.relative_to(output_root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Success marker artifact escapes output root: {artifact}"
                    ) from exc
                if descriptor.get("kind") == ARTIFACT_KIND_STORE:
                    # Metadata migration never rewrites a store, so a store
                    # descriptor can only be verified, never refreshed. It
                    # still has to be dispatched here: `_sha256` opens its
                    # argument as a file and would raise IsADirectoryError,
                    # and the size comparison below reads a key a store
                    # descriptor does not carry (ledger FLOW-31).
                    if not (artifact / STORE_ROOT_JSON).is_file():
                        raise RuntimeError(
                            f"Success marker artifact is missing: {artifact}"
                        )
                    if not _store_artifact_matches(artifact, descriptor):
                        raise RuntimeError(
                            "Uncertified artifact change prevents success "
                            f"marker refresh: {artifact}"
                        )
                    continue
                if not artifact.is_file():
                    raise RuntimeError(
                        f"Success marker artifact is missing: {artifact}"
                    )
                current_sha = _sha256(artifact)
                current_size = artifact.stat().st_size
                certified_transition = transitions.get(artifact)
                if certified_transition is None:
                    if (
                        descriptor.get("sha256") != current_sha
                        or descriptor.get("size") != current_size
                    ):
                        raise RuntimeError(
                            "Uncertified artifact change prevents success marker "
                            f"refresh: {artifact}"
                        )
                    continue
                source_sha, post_sha = certified_transition
                recorded_sha = descriptor.get("sha256")
                if current_sha != post_sha or recorded_sha not in {
                    source_sha,
                    post_sha,
                }:
                    raise RuntimeError(
                        "Metadata migration receipt does not bind success marker "
                        f"artifact: {artifact}"
                    )
                if recorded_sha == source_sha:
                    descriptor["sha256"] = post_sha
                    descriptor["size"] = current_size
                    changed = True
                elif descriptor.get("size") != current_size:
                    raise RuntimeError(
                        f"Success marker size does not match artifact: {artifact}"
                    )
            if changed:
                atomic_write_json(marker_path, marker)
                refreshed += 1
    return refreshed


def current_success_inventory(
    output_dir: Path,
) -> dict[str, frozenset[str]] | None:
    """Return the marker-validated image names of each dataset.

    **The processing inventory: every image that completed.** Contrast
    :func:`authorized_measurement_sources`, which answers a different
    question -- which completed images left something to aggregate. An
    image whose detector found no colonies belongs to this set and not to
    that one, because it publishes a success marker carrying its store
    and overlay but no ``measurements`` artifact.

    Conflating the two is what flagged whole runs read-only: a recompile
    sized its manifest from the aggregation basis, so a run with 4
    zero-detection images out of 36 declared ``total_images: 32`` against
    an inventory of 36, and the viewer's completion guard refused it.
    Both functions were right; the caller asked the wrong one.

    Args:
        output_dir: Root output directory.

    Returns:
        Dataset name to the image names carrying a valid success marker,
        or ``None`` for a legacy state that does not require markers --
        the same ``None`` contract as :func:`current_success_counts`, so a
        caller handles the legacy path once for both.

    Examples:
        The inventory a colony-counting run leaves behind, where one
        plate image grew nothing::

            >>> inventory = current_success_inventory(output_dir)
            ...     # doctest: +SKIP
            >>> sorted(inventory["plate"])  # doctest: +SKIP
            ['blank_control.tif', 'yeast_day3.tif']
    """
    walked = _walk_current_success(output_dir)
    if walked is None:
        return None
    return {
        dataset: frozenset(
            name for name, succeeded in images.items() if succeeded
        )
        for dataset, images in walked.items()
    }


def _walk_current_success(
    output_dir: Path,
) -> dict[str, dict[str, bool]] | None:
    """Validate every image the current state claims, once.

    The one traversal behind both :func:`current_success_inventory` and
    :func:`current_success_counts`, so a count can never disagree with
    the names it is a count of. Each answers by projecting differently:
    the inventory keeps the names that succeeded, the counts keep how
    many did against how many were claimed.

    Args:
        output_dir: Root output directory.

    Returns:
        Dataset name to a mapping of image name to whether its success
        marker validates, or ``None`` for a legacy state that does not
        require markers. A dataset whose images all failed maps to an
        all-``False`` mapping, which is distinct from being absent.
    """
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        return None
    raw_work_ids = state.config.get("work_ids")
    if not isinstance(raw_work_ids, dict):
        # No work-id projection: which images the state claims is known,
        # but not which of them succeeded, so none can be validated.
        return {
            str(dataset): {
                str(image): False for image in item.initial_images
            }
            for dataset, item in state.datasets.items()
        }

    walked: dict[str, dict[str, bool]] = {}
    for dataset, raw_images in raw_work_ids.items():
        if not isinstance(dataset, str) or not isinstance(raw_images, dict):
            continue
        claimed = walked.setdefault(dataset, {})
        for image_name, work_id in raw_images.items():
            if not isinstance(image_name, str) or not isinstance(work_id, str):
                continue
            claimed[image_name] = valid_image_success(
                output_dir,
                dataset=dataset,
                image_stem=source_image_stem(Path(image_name)),
                work_id=work_id,
            )
    return walked


def manifest_completion_inventory(
    output_dir: Path, dataset_names: Iterable[str]
) -> tuple[dict[str, int], dict[str, frozenset[str]] | None]:
    """Size a rebuilt manifest from what was **processed**.

    The pair ``build_manifest`` wants: per-dataset totals, and the image
    names those totals count. Extracted from the recompile path so the
    choice of basis is a named decision with a test on it rather than a
    line inside a CLI command -- the previous version was neither, and
    was wrong.

    **Completion accounting counts processed images, not measured ones.**
    An image whose detector found no colonies completes normally and
    publishes a success marker carrying its store and overlay, but no
    ``measurements`` artifact, so :func:`authorized_measurement_sources`
    -- correct for deciding what to aggregate -- omits it. Sizing a
    manifest that way under-counts every empty image, and the results
    viewer's completion guard reads the shortfall against the processing
    inventory as contradictory evidence and puts the run read-only.
    Measured on a 36-image run with 4 empty images: ``total_images: 32``
    against an inventory of 36.

    Args:
        output_dir: Root output directory.
        dataset_names: Datasets the manifest covers. A dataset with no
            surviving images is reported as zero rather than dropped, so
            the manifest keeps naming it.

    Returns:
        ``(totals, inventory)``. ``inventory`` is ``None`` only for a
        legacy state carrying no success markers, where totals fall back
        to counting per-image measurement Parquets on disk and no
        inventory can be established. Pass both to
        ``regenerate_dashboard_artifacts``: the totals alone leave it
        unable to reconcile them or to count completions when the event
        log is silent, which every recompile's is.
    """
    from phenotypic.sdk_ import dataset_measurements_dir

    names = list(dataset_names)
    walked = current_success_inventory(output_dir)
    if walked is not None:
        inventory = {
            name: walked.get(name, frozenset()) for name in names
        }
        return (
            {name: len(images) for name, images in inventory.items()},
            inventory,
        )

    totals: dict[str, int] = {}
    for name in names:
        meas_dir = dataset_measurements_dir(output_dir, name)
        totals[name] = (
            len(
                [
                    path
                    for path in meas_dir.glob("*.parquet")
                    if not path.name.startswith("_")
                ]
            )
            if meas_dir.is_dir()
            else 0
        )
    return totals, None


def current_success_counts(output_dir: Path) -> tuple[int, int] | None:
    """Return marker-validated ``(successful, total)`` for the current state.

    ``None`` identifies a legacy state that does not require general image
    success markers. Callers may retain their schema-2 compatibility path in
    that case, but schema-3 completion never depends on a manifest.

    Counts, not names -- :func:`current_success_inventory` is the same
    traversal keeping the names instead. Note ``total`` counts every
    image the state claims, so a run with a failed image reports
    ``successful < total``; the inventory simply omits that image.
    """
    walked = _walk_current_success(output_dir)
    if walked is None:
        return None
    successful = sum(
        sum(1 for succeeded in images.values() if succeeded)
        for images in walked.values()
    )
    total = sum(len(images) for images in walked.values())
    return successful, total


def _current_success_work_ids(output_dir: Path, work_ids: object) -> list[str]:
    """Return sorted current work IDs backed by valid success markers."""
    successful: list[str] = []
    if not isinstance(work_ids, dict):
        return successful
    for dataset, raw_images in work_ids.items():
        if not isinstance(dataset, str) or not isinstance(raw_images, dict):
            continue
        for image_name, work_id in raw_images.items():
            if (
                isinstance(image_name, str)
                and isinstance(work_id, str)
                and valid_image_success(
                    output_dir,
                    dataset=dataset,
                    image_stem=source_image_stem(Path(image_name)),
                    work_id=work_id,
                )
            ):
                successful.append(work_id)
    return sorted(successful)


def current_aggregate_is_current(output_dir: Path) -> bool | None:
    """Return whether aggregate evidence matches every current success.

    ``None`` identifies a legacy state. A valid partial aggregate is current
    when its source set is exactly the current marker-authorized success set,
    even though the whole run may remain terminal-incomplete.
    """
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        return None
    if state.config.get("process_only_layer"):
        return True
    aggregate = valid_aggregate_snapshot(output_dir)
    if aggregate is None:
        return False
    work_ids = state.config.get("work_ids", {})
    expected_finalization = canonical_digest(
        {
            "metadata_sha256": state.config.get("metadata_sha256"),
            "include_dataset_column": state.config.get(
                "include_dataset_column"
            ),
            "no_qc": state.config.get("no_qc", False),
        }
    )
    successful_work_ids = _current_success_work_ids(output_dir, work_ids)
    return (
        aggregate.get("inventory_digest") == canonical_digest(work_ids)
        and aggregate.get("finalization_input_digest") == expected_finalization
        and aggregate.get("scientific_config_digest")
        == state.config.get("pipeline_sha256")
        and aggregate.get("source_set_digest")
        == canonical_digest(successful_work_ids)
        and aggregate.get("source_image_count") == len(successful_work_ids)
    )


def current_run_is_complete(output_dir: Path) -> bool | None:
    """Return marker-derived current completion, or ``None`` for legacy state."""
    counts = current_success_counts(output_dir)
    if counts is None:
        return None
    successful, total = counts
    if successful != total:
        return False
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is not None and state.config.get("process_only_layer"):
        return True
    return current_aggregate_is_current(output_dir) is True


def _payload_authorizes(
    output_root: Path,
    payload: object,
    rejection: Callable[..., str | None],
) -> bool:
    """Return whether one certification payload may authorize its sources.

    The shape-specific predicate is passed in -- ``record_rejection`` for a
    record, ``marker_rejection`` for a legacy ``image_complete/`` marker --
    and the artifact walk after it is identical for both, because
    :func:`~phenotypic.sdk_._run_state.fenced_artifact_path` reads a
    descriptor, not a schema.

    The identity is taken **from the payload itself**, which is what this arm
    has always done: with no ``processing_state.json`` there is no
    ``work_ids`` map to check against, so the question is only whether the
    payload is internally coherent and its artifacts still match disk.
    """
    if not isinstance(payload, dict):
        return False
    try:
        dataset = str(payload["dataset"])
        stem = str(payload["image_stem"])
        work_id = str(payload["work_id"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        rejection(payload, work_id=work_id, dataset=dataset, image_stem=stem)
        is not None
    ):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    return all(
        isinstance(descriptor, dict)
        and fenced_artifact_path(output_root, descriptor) is not None
        for descriptor in artifacts.values()
    )


def _sources_without_state(output_dir: Path) -> dict[Path, str] | None:
    """Authorized sources for a tree with no ``success_markers_required``.

    **Both shapes, each on its own predicate.** After D1's clean break this
    arm globbed ``image_complete/`` and then asked ``valid_image_success`` --
    a *record* predicate -- whether a *legacy marker* was valid. The two
    consequences were both silent:

    * a legacy tree returned ``{}`` rather than its real sources, because
      every image failed a record read that a legacy tree cannot satisfy. And
      ``{}`` is not ``None``: ``_cli_chunk_writer`` treats a non-``None``
      result as embedded authority and skips its legacy fallback, while
      ``_cli_recompile_tables`` never reaches the branch that raises *"Legacy
      external measurement Parquets require --mode migrate"*. **A loud,
      actionable migration error became a silent no-op.**
    * a forward tree with no processing state returned ``None``, because
      ``image_complete/`` no longer exists for it to glob.

    So the record tree is scanned first and the legacy tree second, each
    gated by the predicate written for it. ``marker_rejection`` exists
    precisely for the second and this is its caller in ``src/`` -- its
    docstring's "no caller" note was true only in the window between the
    clean break and this repair.

    Returns:
        A mapping of authorized source Parquet to dataset, or ``None`` when
        neither tree exists at all -- which is the signal callers read as
        "fall back to legacy source discovery", and must not be confused with
        an empty mapping meaning "authority exists and authorizes nothing".
    """
    output_root = Path(output_dir).resolve()
    progress = progress_dir(output_dir)
    shapes = (
        (progress / DIR_IMAGE_RECORDS, record_rejection),
        (progress / DIR_IMAGE_COMPLETE, marker_rejection),
    )

    payload_paths = [
        (path, rejection)
        for root, rejection in shapes
        for path in sorted(root.glob("*/*.json"))
    ]
    if not payload_paths:
        return None

    sources: dict[Path, str] = {}
    for path, rejection in payload_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not _payload_authorizes(output_root, payload, rejection):
                continue
            relative = payload["artifacts"]["measurements"]["path"]
            source = (output_root / str(relative)).resolve()
            source.relative_to(output_root)
        except (
            KeyError,
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            continue
        sources[source] = str(payload["dataset"])
    return sources


def authorized_measurement_sources(
    output_dir: Path,
) -> dict[Path, str] | None:
    """Return current marker-authorized measurement Parquets by dataset.

    ``None`` requests the legacy source-discovery path. An empty mapping is a
    valid schema-3 result meaning that no successful measurements exist yet.
    """
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        return _sources_without_state(output_dir)
    raw_work_ids = state.config.get("work_ids")
    if not isinstance(raw_work_ids, dict):
        return {}

    sources: dict[Path, str] = {}
    output_root = output_dir.resolve()
    for dataset, raw_images in raw_work_ids.items():
        if not isinstance(dataset, str) or not isinstance(raw_images, dict):
            continue
        for image_name, work_id in raw_images.items():
            if not isinstance(image_name, str) or not isinstance(work_id, str):
                continue
            stem = source_image_stem(Path(image_name))
            if not valid_image_success(
                output_dir,
                dataset=dataset,
                image_stem=stem,
                work_id=work_id,
            ):
                continue
            # CAN-22: the RECORD, not `image_complete/`. This arm used to
            # re-open the legacy marker after `valid_image_success` passed --
            # and after D1's clean break that file is gone, so every image
            # would hit `OSError`, `continue`, and leave `sources` empty.
            #
            # That failure is silent and severe: `{}` is a VALID schema-3
            # result meaning "no successful measurements yet", so P4's
            # `finalize_run` would write an empty master and raise nothing.
            # A successful-looking run that discarded every measurement.
            record = read_image_record(output_dir, dataset, stem)
            if record is None:
                continue
            # Narrowed rather than indexed-and-caught. The old shape leaned
            # on `KeyError`/`TypeError` in the `except` to mean "malformed",
            # which reads as error handling but was the only thing typing the
            # payload -- and it shares its `continue` with the genuinely
            # different `OSError`/`ValueError` path resolution below. Each
            # `continue` here now names the one thing that was wrong.
            artifacts = record.get("artifacts")
            if not isinstance(artifacts, dict):
                continue
            descriptor = artifacts.get("measurements")
            if not isinstance(descriptor, dict):
                continue
            relative = descriptor.get("path")
            if not isinstance(relative, str):
                continue
            try:
                source = (output_root / relative).resolve()
                source.relative_to(output_root)
            except (OSError, ValueError):
                continue
            sources[source] = dataset
    return sources


def publish_aggregate_snapshot(
    output_dir: Path,
    *,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Publish marker-last integrity evidence for the canonical core snapshot."""
    from ._cli_state_management import load_processing_state

    state = load_processing_state(output_dir)
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        raise RuntimeError(
            "Aggregate marker publication requires current state"
        )
    counts = current_success_counts(output_dir)
    if counts is None or counts[0] == 0:
        raise RuntimeError("No marker-authorized measurements to publish")

    # D8: three descriptors, not four. `master_measurements.csv` is gone --
    # the un-joined master is no longer the file a human opens, the mirror is
    # -- and `finalize_run` writes no CSV, so certifying one would make every
    # forward finalization fail on a `resolve(strict=True)` for a file nothing
    # writes. `valid_aggregate_snapshot` and the sdk_ reader both validate
    # whatever the proof LISTS, so a three-entry proof validates on its own
    # terms.
    required_paths = {
        "master_parquet": master_measurements_parquet_path(output_dir),
        "measurements_csv": measurements_csv_path(output_dir),
        "measurements_parquet": measurements_parquet_path(output_dir),
    }
    output_root = output_dir.resolve()
    descriptors: dict[str, dict[str, object]] = {}
    for name, path in required_paths.items():
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(output_root)
        descriptors[name] = {
            "path": relative.as_posix(),
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }

    work_ids = state.config.get("work_ids", {})
    source_work_ids = _current_success_work_ids(output_dir, work_ids)
    # U-4: no `publication_id`. It was an opaque uuid4 whose only job was to
    # bind this proof to the run proof; the run proof now COPIES
    # `source_set_digest`/`source_image_count` from here instead, which states
    # the same binding in the clear and is checkable against live state.
    marker = {
        "version": AGGREGATE_PROOF_VERSION,
        "processing_generation": state.config.get("processing_generation"),
        "inventory_digest": canonical_digest(work_ids),
        "finalization_input_digest": canonical_digest(
            {
                "metadata_sha256": state.config.get("metadata_sha256"),
                "include_dataset_column": state.config.get(
                    "include_dataset_column"
                ),
                "no_qc": state.config.get("no_qc", False),
            }
        ),
        "scientific_config_digest": state.config.get("pipeline_sha256"),
        "source_set_digest": canonical_digest(sorted(source_work_ids)),
        "source_image_count": len(source_work_ids),
        "required_outputs": descriptors,
        "published_at": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    path = aggregate_publication_marker_path(output_dir)
    atomic_write_json(path, marker, commit_guard=commit_guard)
    return path


def valid_aggregate_snapshot(output_dir: Path) -> dict[str, object] | None:
    """Return a valid aggregate marker, rejecting any mixed core snapshot."""
    path = aggregate_publication_marker_path(output_dir)
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(marker, dict)
            or marker.get("version") != AGGREGATE_PROOF_VERSION
        ):
            return None
        outputs = marker.get("required_outputs")
        if not isinstance(outputs, dict) or not outputs:
            return None
        output_root = output_dir.resolve()
        for descriptor in outputs.values():
            if not isinstance(descriptor, dict):
                return None
            relative = descriptor.get("path")
            if not isinstance(relative, str):
                return None
            artifact = (output_root / relative).resolve()
            artifact.relative_to(output_root)
            if (
                not artifact.is_file()
                or artifact.stat().st_size != descriptor.get("size")
                or _sha256(artifact) != descriptor.get("sha256")
            ):
                return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return marker


def publish_run_completion_evidence(
    output_dir: Path,
    *,
    execution_epoch: str,
    gui_record_generation: str | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Publish all-success run evidence, idempotently for a no-op run."""
    from ._cli_state_management import load_processing_state

    completion = current_run_is_complete(output_dir)
    state = load_processing_state(output_dir)
    if (
        completion is None
        or state is None
        or not state.config.get("success_markers_required", False)
    ):
        path = run_completion_marker_path(output_dir)
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "generation": gui_record_generation or execution_epoch,
                "execution_epoch": execution_epoch,
                "mode": (
                    "local"
                    if gui_record_generation is not None
                    or execution_epoch == "local"
                    else "slurm"
                ),
                "status": "complete",
                "finalizer_succeeded": True,
                "completed_at": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
            commit_guard=commit_guard,
        )
        return path
    if completion is not True:
        raise RuntimeError(
            "Current run does not have complete publication evidence"
        )
    aggregate = (
        None
        if state.config.get("process_only_layer")
        else valid_aggregate_snapshot(output_dir)
    )
    work_ids = state.config.get("work_ids", {})
    payload = {
        "version": RUN_PROOF_VERSION,
        "processing_generation": state.config.get("processing_generation"),
        "inventory_digest": canonical_digest(work_ids),
        "finalization_input_digest": (
            aggregate.get("finalization_input_digest")
            if aggregate is not None
            else canonical_digest(
                {"process_only_layer": state.config.get("process_only_layer")}
            )
        ),
        "scientific_config_digest": state.config.get("pipeline_sha256"),
        # U-4: a COPY of the aggregate proof's values, exactly as
        # `publication_id` was copied. **Not recomputed.** The copy IS the
        # binding: it asserts "I was published against THAT aggregate", which
        # rule 1 then checks against a live re-derivation of the verified
        # image set. Recomputing here would assert only "here is my own view,
        # at my own moment" -- and a stale aggregate proof beside a fresh run
        # proof would then pass both checks independently, with nothing
        # noticing they disagree.
        "source_set_digest": (
            aggregate.get("source_set_digest")
            if aggregate is not None
            else None
        ),
        "source_image_count": (
            aggregate.get("source_image_count")
            if aggregate is not None
            else None
        ),
        "execution_epoch": execution_epoch,
        "gui_record_generation": gui_record_generation,
        "generation": gui_record_generation or execution_epoch,
        "mode": (
            "local"
            if gui_record_generation is not None or execution_epoch == "local"
            else "slurm"
        ),
        "status": "complete",
        "finalizer_succeeded": True,
        "completed_at": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    path = run_completion_marker_path(output_dir)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        existing = None
    # `publication_id` was minted fresh on every aggregate publication, so it
    # was the entry that made this comparison meaningful. With it gone the
    # entry would compare `None == None` and contribute nothing --
    # `source_set_digest` is the value that belongs there instead: it changes
    # exactly when the certified success set does.
    stable_keys = (
        "version",
        "inventory_digest",
        "finalization_input_digest",
        "scientific_config_digest",
        "source_set_digest",
        "status",
    )
    if isinstance(existing, dict) and all(
        existing.get(key) == payload.get(key) for key in stable_keys
    ):
        with publication_commit(commit_guard):
            return path
    atomic_write_json(path, payload, commit_guard=commit_guard)
    return path


def valid_run_completion(output_dir: Path) -> dict[str, object] | None:
    """Return current all-success run evidence, rejecting stale markers."""
    from ._cli_state_management import load_processing_state

    path = run_completion_marker_path(output_dir)
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or marker.get("status") != "complete":
        return None
    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        return marker if marker.get("finalizer_succeeded") is True else None
    if (
        marker.get("version") != RUN_PROOF_VERSION
        or current_run_is_complete(output_dir) is not True
    ):
        return None
    work_ids = state.config.get("work_ids", {})
    expected: dict[str, object] = {
        "inventory_digest": canonical_digest(work_ids),
        "scientific_config_digest": state.config.get("pipeline_sha256"),
    }
    if state.config.get("process_only_layer"):
        # The `publication_id = None` entry that stood here is DELETED, not
        # repointed. It asserted "a process-only run has no aggregate
        # binding"; with no such field to be absent it would compare
        # `None != None` and say nothing at all. A comparison that cannot
        # fail is worse than a deleted one, because it still reads as a guard.
        expected["finalization_input_digest"] = canonical_digest(
            {"process_only_layer": state.config.get("process_only_layer")}
        )
    else:
        aggregate = valid_aggregate_snapshot(output_dir)
        if aggregate is None:
            return None
        # U-4: the aggregate<->run binding, stated in the clear. This is the
        # live one -- five call sites in four modules read this function's
        # verdict -- so leaving it as a `None != None` tautology would stop
        # the binding being checked at all.
        expected["source_set_digest"] = aggregate.get("source_set_digest")
        expected["source_image_count"] = aggregate.get("source_image_count")
        expected["finalization_input_digest"] = aggregate.get(
            "finalization_input_digest"
        )
    if any(marker.get(key) != value for key, value in expected.items()):
        return None
    return marker
