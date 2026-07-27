"""Pure computation and transactional publication for Error analysis.

Preview callers use :func:`compute_all_category_analysis` without touching the
filesystem.  Explicit GUI publication and CLI finalization share the same
staging, checksum, lock, rollback, manifest, and receipt implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

import pandas as pd
import polars as pl

from phenotypic.analysis import (
    ErrorCutoffFinder,
    render_error_analysis_report,
)
from phenotypic.analysis._error_cutoffs import RESULT_COLUMNS, _RESULT_DTYPES
from phenotypic.gui.results_viewer._curation_labels import (
    LabelKey,
    _join_on_keys,
)
from phenotypic.sdk_ import (
    PARQUET_WRITE_OPTIONS,
    atomic_write_json,
    paths_fingerprint,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.gui.results_viewer._output_root import OutputRoot
    from phenotypic.sdk_ import BundleLayout

GoodMode = Literal["all_unlabeled", "verified"]
_PERSIST_COLUMNS: tuple[str, ...] = ("category", *RESULT_COLUMNS)
_MANIFEST_FILENAME = "error_analysis.manifest.json"
_RECEIPT_FILENAME = "error_analysis.publication.json"
_LOCK_FILENAME = ".error-analysis.publication.lock"
_JOURNAL_FILENAME = ".error-analysis.transaction.json"
_GENERATIONS_DIRNAME = ".error-analysis.generations"
_SOURCE_NAMES: tuple[str, ...] = (
    "master",
    "mirror",
    "labels",
    "custom_categories",
    "qc_database",
    "review_state",
)


def _is_lower_hex(value: object, *, length: int) -> TypeGuard[str]:
    """Return whether *value* is exact-length lower-case hexadecimal text."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


class ErrorPublicationConflict(RuntimeError):
    """Raised when an Error generation no longer matches its source snapshot."""


class ErrorPublicationValidationError(RuntimeError):
    """Raised when a staged or already-published generation is incomplete."""


@dataclass(frozen=True)
class CategoryAnalysis:
    """One configured category's complete computation result."""

    category: str
    label_count: int
    result: pd.DataFrame


@dataclass(frozen=True)
class ErrorAnalysisComputation:
    """All-category result derived from one source snapshot."""

    good_mode: GoodMode
    categories: tuple[CategoryAnalysis, ...]
    combined: pd.DataFrame
    verified: pd.DataFrame | None
    source_fingerprints: Mapping[str, str]
    generation: str


@dataclass(frozen=True)
class ErrorPublicationResult:
    """Outcome returned by an explicit generation publication."""

    generation: str
    category_count: int
    populated_category_count: int
    row_count: int
    artifact_names: tuple[str, ...]
    already_published: bool


def capture_error_source_fingerprints(
    layout: "BundleLayout",
) -> dict[str, str]:
    """Capture the complete named source set for Error computation.

    Args:
        layout: Resolved output/bundle topology.

    Returns:
        A name-to-content-fingerprint mapping. Missing files have a stable,
        explicit fingerprint through :func:`paths_fingerprint`.
    """
    paths = (
        layout.master_parquet,
        layout.mirror_parquet,
        layout.curation_labels_parquet,
        layout.custom_categories_json,
        layout.qc_duckdb,
        layout.qc_review_state_path,
    )
    return {
        name: paths_fingerprint((path,), root=layout.deliverables_base)
        for name, path in zip(_SOURCE_NAMES, paths, strict=True)
    }


def compute_all_category_analysis(
    master_df: pl.DataFrame,
    *,
    labels: Mapping[LabelKey, str],
    categories: tuple[str, ...],
    good_pdf: pd.DataFrame,
    good_mode: GoodMode,
    source_fingerprints: Mapping[str, str],
) -> ErrorAnalysisComputation:
    """Compute every configured category without performing filesystem writes.

    Empty categories remain first-class entries with a typed zero-row result,
    even though the combined tabular contract naturally has no row to tag for
    them. The manifest records those empty entries explicitly.

    Args:
        master_df: Clean master measurements containing labeled objects.
        labels: Snapshot of durable object-to-category assignments.
        categories: Complete configured vocabulary, including empty categories.
        good_pdf: One coherent good-baseline frame shared by every category.
        good_mode: Baseline mode used for this generation.
        source_fingerprints: Named source fingerprints captured before reads.

    Returns:
        Immutable all-category computation ready for explicit publication.
    """
    finder = ErrorCutoffFinder()
    computed: list[CategoryAnalysis] = []
    tagged_frames: list[pd.DataFrame] = []
    for category in categories:
        keys = [key for key, value in labels.items() if value == category]
        error_pdf = _join_on_keys(master_df, keys, "semi").to_pandas()
        if finder.enough_data(good_pdf, error_pdf):
            result = finder.analyze(good_pdf, error_pdf)
        else:
            result = _empty_result()
        result = result.loc[:, list(RESULT_COLUMNS)]
        computed.append(
            CategoryAnalysis(
                category=category,
                label_count=len(keys),
                result=result,
            )
        )
        if not result.empty:
            tagged = result.copy()
            tagged.insert(0, "category", category)
            tagged_frames.append(tagged.loc[:, list(_PERSIST_COLUMNS)])

    combined = (
        pd.concat(tagged_frames, ignore_index=True)
        if tagged_frames
        else _empty_combined()
    )
    verified = good_pdf.copy() if good_mode == "verified" else None
    generation = _generation_digest(
        source_fingerprints=source_fingerprints,
        good_mode=good_mode,
        categories=computed,
    )
    return ErrorAnalysisComputation(
        good_mode=good_mode,
        categories=tuple(computed),
        combined=combined,
        verified=verified,
        source_fingerprints=dict(source_fingerprints),
        generation=generation,
    )


def compute_gui_error_publication(
    output_root: "OutputRoot",
    *,
    filtered_state: "CurationLabels",
    good_mode: GoodMode,
) -> ErrorAnalysisComputation:
    """Read and compute one coherent GUI publication candidate.

    The curation store's in-process mutex and interprocess publication lock
    are held while the named source snapshot is captured and the in-memory
    labels/custom-category revision is verified. This prevents an old browser
    store from being paired with a newer disk fingerprint.
    """
    from phenotypic.gui.results_viewer._error_tab._data import (
        verified_good_keys,
    )
    from phenotypic.sdk_._file_locking import exclusive_path_lock

    if not output_root.mutation_snapshot_is_safe():
        raise ErrorPublicationConflict(
            "The bound output is active or its processing generation changed. "
            "Refresh the Results snapshot before publishing."
        )
    with filtered_state._lock:
        with exclusive_path_lock(filtered_state._publication_lock_path):
            before = capture_error_source_fingerprints(output_root.layout)
            current_curation = filtered_state._source_fingerprint(
                output_root.layout
            )
            if (
                filtered_state.stale
                or current_curation
                != filtered_state._expected_source_fingerprint
            ):
                raise ErrorPublicationConflict(
                    "Curation labels or category configuration changed outside "
                    "this binding. Refresh before publishing."
                )
            labels = dict(filtered_state.labels)
            categories = tuple(filtered_state.categories())
            if good_mode == "verified":
                good_keys = verified_good_keys(
                    output_root,
                    set(labels),
                )
                good_pdf = _join_on_keys(
                    output_root.clean_master_df,
                    good_keys,
                    "semi",
                ).to_pandas()
            else:
                good_pdf = filtered_state.filtered_df(
                    output_root.clean_master_df
                ).to_pandas()
            computation = compute_all_category_analysis(
                output_root.clean_master_df,
                labels=labels,
                categories=categories,
                good_pdf=good_pdf,
                good_mode=good_mode,
                source_fingerprints=before,
            )
            after = capture_error_source_fingerprints(output_root.layout)
            current_curation_after = filtered_state._source_fingerprint(
                output_root.layout
            )
            if (
                before != after
                or current_curation_after != current_curation
                or current_curation_after
                != filtered_state._expected_source_fingerprint
            ):
                raise ErrorPublicationConflict(
                    "Error inputs changed during computation. Refresh and "
                    "retry after the active writer settles."
                )
    if not output_root.mutation_snapshot_is_safe():
        raise ErrorPublicationConflict(
            "Error inputs changed during computation. Refresh and retry after "
            "the active writer settles."
        )
    return computation


def publish_error_analysis(
    layout: "BundleLayout",
    computation: ErrorAnalysisComputation,
    *,
    mutation_is_safe: Callable[[], bool],
    lock_timeout: float = 30.0,
    replace_file: Callable[[Path, Path], None] | None = None,
) -> ErrorPublicationResult:
    """Publish a complete Error generation with rollback and manifest-last CAS.

    Args:
        layout: Resolved output/bundle topology.
        computation: All-category computation derived from one snapshot.
        mutation_is_safe: Rechecks binding, processing generation, and active
            ownership while the publication lock is held.
        lock_timeout: Maximum seconds to wait for another Error publisher.
        replace_file: Optional test seam replacing ``source`` onto ``target``.

    Returns:
        Publication outcome, including whether an identical complete
        generation was already current.

    Raises:
        ErrorPublicationConflict: If sources, binding, processing, or active
            ownership changed.
        ErrorPublicationValidationError: If staged or canonical artifacts fail
            complete-set validation.
        OSError: If staging or publication fails after rollback.
    """
    base = layout.deliverables_base
    lock_path = base / _LOCK_FILENAME
    replace = replace_file or _replace_file
    with exclusive_path_lock(lock_path, timeout=lock_timeout):
        _require_publication_current(
            layout,
            computation,
            mutation_is_safe=mutation_is_safe,
        )
        recover_error_publication(layout)

        current = _read_json(base / _RECEIPT_FILENAME)
        if (
            isinstance(current, dict)
            and current.get("generation") == computation.generation
            and _validate_canonical_generation(base, current)
        ):
            return _result_from_receipt(current, already_published=True)

        token = uuid.uuid4().hex
        generation_dir = base / _GENERATIONS_DIRNAME / token
        staging_dir = generation_dir / "staged"
        backup_dir = generation_dir / "backup"
        staging_dir.mkdir(parents=True, exist_ok=False)
        backup_dir.mkdir(parents=True, exist_ok=False)
        journal_written = False
        try:
            manifest = _stage_generation(staging_dir, computation)
            _validate_staged_generation(staging_dir, manifest)
            receipt = {
                **manifest,
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(staging_dir / _RECEIPT_FILENAME, receipt)

            _require_publication_current(
                layout,
                computation,
                mutation_is_safe=mutation_is_safe,
            )
            targets = tuple(manifest["artifacts"]) + (
                _MANIFEST_FILENAME,
                _RECEIPT_FILENAME,
            )
            existing = _backup_targets(base, backup_dir, targets)
            atomic_write_json(
                base / _JOURNAL_FILENAME,
                {
                    "schema_version": 1,
                    "token": token,
                    "generation": computation.generation,
                    "targets": list(targets),
                    "existing": sorted(existing),
                },
            )
            journal_written = True
            try:
                for name in manifest["artifacts"]:
                    replace(staging_dir / name, base / name)
                _require_publication_current(
                    layout,
                    computation,
                    mutation_is_safe=mutation_is_safe,
                )
                replace(
                    staging_dir / _MANIFEST_FILENAME,
                    base / _MANIFEST_FILENAME,
                )
                _require_publication_current(
                    layout,
                    computation,
                    mutation_is_safe=mutation_is_safe,
                )
                replace(
                    staging_dir / _RECEIPT_FILENAME,
                    base / _RECEIPT_FILENAME,
                )
                # Receipt replacement is the commit boundary. Recheck afterward
                # so a source writer interleaved by the replacement cannot make a
                # stale generation authoritative.
                _require_publication_current(
                    layout,
                    computation,
                    mutation_is_safe=mutation_is_safe,
                )
                if not _validate_canonical_generation(base, receipt):
                    raise ErrorPublicationValidationError(
                        "Published Error generation failed checksum validation."
                    )
            except BaseException:
                _restore_targets(base, backup_dir, targets, existing)
                (base / _JOURNAL_FILENAME).unlink(missing_ok=True)
                shutil.rmtree(generation_dir, ignore_errors=True)
                _remove_empty_generations_dir(base)
                raise
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
            if not journal_written:
                shutil.rmtree(generation_dir, ignore_errors=True)
                _remove_empty_generations_dir(base)

        # The canonical receipt has committed and validated. From this point
        # onward, cleanup must never re-enter rollback: the journal may already
        # be absent, and a cleanup interruption is handled by the next
        # journal-free orphan sweep under this same writer lock.
        (base / _JOURNAL_FILENAME).unlink(missing_ok=True)
        shutil.rmtree(generation_dir, ignore_errors=True)
        _remove_empty_generations_dir(base)
        return _result_from_receipt(receipt, already_published=False)


def error_publication_lock_path(layout: "BundleLayout") -> Path:
    """Return the lock shared by GUI and CLI Error publishers."""
    return layout.deliverables_base / _LOCK_FILENAME


def recover_error_publication(layout: "BundleLayout") -> bool:
    """Recover a transaction interrupted before its receipt committed.

    The caller must hold :func:`error_publication_lock_path`. A receipt that
    selects and validates the journaled generation means publication committed;
    otherwise every canonical target is restored from the durable backup.

    Args:
        layout: Resolved output/bundle topology.

    Returns:
        ``True`` when a journal was found and resolved.
    """
    base = layout.deliverables_base
    journal_path = base / _JOURNAL_FILENAME
    if not journal_path.exists():
        _sweep_orphan_generations(base)
        return False
    payload = _read_json(journal_path)
    if payload is None:
        raise ErrorPublicationValidationError(
            "Error publication journal exists but cannot be decoded."
        )
    if not isinstance(payload, dict):
        raise ErrorPublicationValidationError(
            "Error publication journal is not a JSON object."
        )
    expected = {
        "schema_version",
        "token",
        "generation",
        "targets",
        "existing",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ErrorPublicationValidationError(
            "Error publication journal has invalid fields."
        )
    token = payload.get("token")
    generation = payload.get("generation")
    targets = payload.get("targets")
    existing = payload.get("existing")
    if (
        not _is_lower_hex(token, length=32)
        or not _is_lower_hex(generation, length=64)
        or not isinstance(targets, list)
        or not all(isinstance(name, str) for name in targets)
        or not isinstance(existing, list)
        or not all(isinstance(name, str) for name in existing)
    ):
        raise ErrorPublicationValidationError(
            "Error publication journal values are invalid."
        )
    base_targets = {
        computation_path_name("parquet"),
        computation_path_name("csv"),
        computation_path_name("html"),
        _MANIFEST_FILENAME,
        _RECEIPT_FILENAME,
    }
    allowed_target_sets = (
        base_targets,
        base_targets | {"verified.parquet"},
    )
    target_set = set(targets)
    existing_set = set(existing)
    if (
        target_set not in allowed_target_sets
        or len(target_set) != len(targets)
        or len(existing_set) != len(existing)
        or not existing_set.issubset(target_set)
    ):
        raise ErrorPublicationValidationError(
            "Error publication journal does not name one exact generation."
        )
    generation_dir = base / _GENERATIONS_DIRNAME / token
    backup_dir = generation_dir / "backup"
    if not backup_dir.is_dir():
        raise ErrorPublicationValidationError(
            "Error publication recovery backup directory is missing."
        )
    actual_backups: set[str] = set()
    for entry in backup_dir.iterdir():
        if entry.is_file() and entry.name in target_set:
            actual_backups.add(entry.name)
        elif entry.is_file() and _is_restore_temporary(
            entry.name,
            existing_set,
        ):
            continue
        else:
            raise ErrorPublicationValidationError(
                "Error publication backup inventory contains an unexpected "
                f"entry: {entry.name!r}."
            )
    if actual_backups != existing_set:
        raise ErrorPublicationValidationError(
            "Error publication journal does not match its durable backups."
        )
    receipt = _read_json(base / _RECEIPT_FILENAME)
    committed = (
        isinstance(receipt, dict)
        and receipt.get("generation") == generation
        and _validate_canonical_generation(base, receipt)
    )
    if not committed:
        _restore_targets(
            base,
            generation_dir / "backup",
            tuple(targets),
            set(existing),
        )
    shutil.rmtree(generation_dir / "staged", ignore_errors=True)
    journal_path.unlink(missing_ok=True)
    shutil.rmtree(generation_dir, ignore_errors=True)
    _remove_empty_generations_dir(base)
    return True


def _require_publication_current(
    layout: "BundleLayout",
    computation: ErrorAnalysisComputation,
    *,
    mutation_is_safe: Callable[[], bool],
) -> None:
    """Recheck binding/owner and every named source while lock is held."""
    if not mutation_is_safe():
        raise ErrorPublicationConflict(
            "The output binding is stale or actively owned; publication was "
            "blocked."
        )
    current_sources = capture_error_source_fingerprints(layout)
    if current_sources != dict(computation.source_fingerprints):
        raise ErrorPublicationConflict(
            "Error inputs changed after computation; nothing was published."
        )


def _stage_generation(
    staging_dir: Path,
    computation: ErrorAnalysisComputation,
) -> dict[str, Any]:
    """Write and describe a complete generation inside ``staging_dir``."""
    parquet_name = computation_path_name("parquet")
    csv_name = computation_path_name("csv")
    html_name = computation_path_name("html")
    combined = pl.from_pandas(computation.combined)
    combined.write_parquet(
        staging_dir / parquet_name,
        **PARQUET_WRITE_OPTIONS,
    )
    combined.write_csv(staging_dir / csv_name)
    reports = {item.category: item.result for item in computation.categories}
    (staging_dir / html_name).write_text(
        render_error_analysis_report(reports),
        encoding="utf-8",
    )
    artifact_names = [parquet_name, csv_name, html_name]
    if computation.verified is not None:
        verified_name = "verified.parquet"
        pl.from_pandas(computation.verified).write_parquet(
            staging_dir / verified_name,
            **PARQUET_WRITE_OPTIONS,
        )
        artifact_names.append(verified_name)

    artifacts = {
        name: _artifact_descriptor(staging_dir / name)
        for name in artifact_names
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generation": computation.generation,
        "good_mode": computation.good_mode,
        "sources": dict(computation.source_fingerprints),
        "categories": [
            {
                "category": item.category,
                "labels": item.label_count,
                "rows": len(item.result),
            }
            for item in computation.categories
        ],
        "rows": len(computation.combined),
        "artifacts": artifacts,
    }
    _write_json(staging_dir / _MANIFEST_FILENAME, manifest)
    return manifest


def _validate_staged_generation(
    staging_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Validate checksums, schemas, and complete category inventory."""
    artifacts = manifest.get("artifacts")
    categories = manifest.get("categories")
    if not isinstance(artifacts, dict) or not isinstance(categories, list):
        raise ErrorPublicationValidationError(
            "Error manifest lacks artifacts or categories."
        )
    required = {
        computation_path_name("parquet"),
        computation_path_name("csv"),
        computation_path_name("html"),
    }
    good_mode = manifest.get("good_mode")
    expected_artifacts = required | (
        {"verified.parquet"} if good_mode == "verified" else set()
    )
    if set(artifacts) != expected_artifacts:
        raise ErrorPublicationValidationError(
            "Error generation does not contain the exact artifact set."
        )
    for name, descriptor in artifacts.items():
        path = staging_dir / name
        if (
            not isinstance(name, str)
            or not isinstance(descriptor, dict)
            or not _descriptor_matches(path, descriptor)
        ):
            raise ErrorPublicationValidationError(
                f"Error artifact {name!r} failed checksum validation."
            )
    parquet = pl.read_parquet(staging_dir / computation_path_name("parquet"))
    csv = pl.read_csv(
        staging_dir / computation_path_name("csv"),
        infer_schema_length=0,
    )
    if parquet.columns != list(_PERSIST_COLUMNS):
        raise ErrorPublicationValidationError(
            "Error parquet does not match the persisted column contract."
        )
    if csv.columns != list(_PERSIST_COLUMNS):
        raise ErrorPublicationValidationError(
            "Error CSV does not match the persisted column contract."
        )
    if parquet.height != manifest.get("rows") or csv.height != parquet.height:
        raise ErrorPublicationValidationError(
            "Error artifact row counts disagree with the manifest."
        )
    names = [
        item.get("category") for item in categories if isinstance(item, dict)
    ]
    if len(names) != len(set(names)):
        raise ErrorPublicationValidationError(
            "Error manifest contains duplicate category entries."
        )


def _validate_canonical_generation(
    base: Path,
    receipt: Mapping[str, Any],
) -> bool:
    """Return whether every receipt-selected canonical artifact is complete."""
    receipt_fields = set(receipt)
    expected_receipt_fields = {
        "schema_version",
        "generation",
        "good_mode",
        "sources",
        "categories",
        "rows",
        "artifacts",
        "published_at",
    }
    artifacts = receipt.get("artifacts")
    good_mode = receipt.get("good_mode")
    sources = receipt.get("sources")
    categories = receipt.get("categories")
    generation = receipt.get("generation")
    rows = receipt.get("rows")
    published_at = receipt.get("published_at")
    expected_artifacts = {
        computation_path_name("parquet"),
        computation_path_name("csv"),
        computation_path_name("html"),
    } | ({"verified.parquet"} if good_mode == "verified" else set())
    if (
        receipt_fields != expected_receipt_fields
        or receipt.get("schema_version") != 1
        or good_mode not in ("all_unlabeled", "verified")
        or not _is_lower_hex(generation, length=64)
        or not isinstance(sources, dict)
        or set(sources) != set(_SOURCE_NAMES)
        or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in sources.items()
        )
        or not isinstance(categories, list)
        or not all(_valid_category_manifest_item(item) for item in categories)
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 0
        or not isinstance(published_at, str)
        or not isinstance(artifacts, dict)
        or set(artifacts) != expected_artifacts
    ):
        return False
    manifest = _read_json(base / _MANIFEST_FILENAME)
    expected_manifest = {
        key: value for key, value in receipt.items() if key != "published_at"
    }
    if manifest != expected_manifest:
        return False
    return all(
        isinstance(name, str)
        and isinstance(descriptor, dict)
        and _descriptor_matches(base / name, descriptor)
        for name, descriptor in artifacts.items()
    )


def _valid_category_manifest_item(item: object) -> bool:
    """Return whether one persisted category inventory item is well formed."""
    if not isinstance(item, dict) or set(item) != {
        "category",
        "labels",
        "rows",
    }:
        return False
    category = item.get("category")
    labels = item.get("labels")
    rows = item.get("rows")
    return (
        isinstance(category, str)
        and bool(category)
        and not isinstance(labels, bool)
        and isinstance(labels, int)
        and labels >= 0
        and not isinstance(rows, bool)
        and isinstance(rows, int)
        and rows >= 0
    )


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return {
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _descriptor_matches(path: Path, descriptor: Mapping[str, Any]) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size == descriptor.get("bytes")
            and _sha256(path) == descriptor.get("sha256")
        )
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_digest(
    *,
    source_fingerprints: Mapping[str, str],
    good_mode: GoodMode,
    categories: list[CategoryAnalysis],
) -> str:
    payload = {
        "schema_version": 1,
        "sources": dict(sorted(source_fingerprints.items())),
        "good_mode": good_mode,
        "categories": [
            {
                "category": item.category,
                "labels": item.label_count,
                "rows": len(item.result),
                "result": item.result.to_json(
                    orient="split",
                    double_precision=15,
                ),
            }
            for item in categories
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(dtype=_RESULT_DTYPES[column])
            for column in RESULT_COLUMNS
        }
    )


def _empty_combined() -> pd.DataFrame:
    dtypes = {"category": "object", **_RESULT_DTYPES}
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtypes[column])
            for column in _PERSIST_COLUMNS
        }
    )


def computation_path_name(kind: Literal["parquet", "csv", "html"]) -> str:
    """Return the canonical Error-analysis artifact name for ``kind``."""
    return f"error_analysis.{kind}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None


def _backup_targets(
    base: Path,
    backup_dir: Path,
    targets: tuple[str, ...],
) -> set[str]:
    existing: set[str] = set()
    for name in targets:
        source = base / name
        if source.is_file():
            shutil.copy2(source, backup_dir / name)
            existing.add(name)
    return existing


def _restore_targets(
    base: Path,
    backup_dir: Path,
    targets: tuple[str, ...],
    existing: set[str],
) -> None:
    for name in targets:
        target = base / name
        backup = backup_dir / name
        if name in existing:
            if not backup.is_file():
                raise ErrorPublicationValidationError(
                    f"Cannot restore prior Error artifact {name!r}."
                )
            _restore_from_durable_backup(backup, target)
        else:
            target.unlink(missing_ok=True)


def _restore_from_durable_backup(backup: Path, target: Path) -> None:
    """Atomically restore one target without consuming its crash backup."""
    temporary = backup.with_name(f".{target.name}.{uuid.uuid4().hex}.restore")
    try:
        shutil.copy2(backup, temporary)
        _replace_file(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _is_restore_temporary(name: str, targets: set[str]) -> bool:
    """Return whether *name* is one owned interrupted-restore temporary."""
    for target in targets:
        prefix = f".{target}."
        suffix = ".restore"
        if name.startswith(prefix) and name.endswith(suffix):
            token = name[len(prefix) : -len(suffix)]
            return _is_lower_hex(token, length=32)
    return False


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _remove_empty_generations_dir(base: Path) -> None:
    """Remove the generations container when no durable backup remains."""
    try:
        (base / _GENERATIONS_DIRNAME).rmdir()
    except OSError:
        pass


def _sweep_orphan_generations(base: Path) -> None:
    """Remove journal-free transaction directories under the writer lock."""
    generations = base / _GENERATIONS_DIRNAME
    if not generations.is_dir():
        return
    for candidate in generations.iterdir():
        token = candidate.name
        if candidate.is_dir() and _is_lower_hex(token, length=32):
            shutil.rmtree(candidate, ignore_errors=True)
    _remove_empty_generations_dir(base)


def _result_from_receipt(
    receipt: Mapping[str, Any],
    *,
    already_published: bool,
) -> ErrorPublicationResult:
    categories = receipt.get("categories")
    artifacts = receipt.get("artifacts")
    category_items = categories if isinstance(categories, list) else []
    artifact_items = artifacts if isinstance(artifacts, dict) else {}
    return ErrorPublicationResult(
        generation=str(receipt.get("generation", "")),
        category_count=len(category_items),
        populated_category_count=sum(
            int(item.get("rows", 0)) > 0
            for item in category_items
            if isinstance(item, dict)
        ),
        row_count=int(receipt.get("rows", 0)),
        artifact_names=tuple(sorted(str(name) for name in artifact_items)),
        already_published=already_published,
    )


__all__ = [
    "CategoryAnalysis",
    "ErrorAnalysisComputation",
    "ErrorPublicationConflict",
    "ErrorPublicationResult",
    "ErrorPublicationValidationError",
    "capture_error_source_fingerprints",
    "compute_all_category_analysis",
    "compute_gui_error_publication",
    "error_publication_lock_path",
    "publish_error_analysis",
    "recover_error_publication",
]
