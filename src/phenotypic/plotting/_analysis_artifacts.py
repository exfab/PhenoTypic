"""Named analysis artifact paths and manifest contracts.

The analysis manifest is the authoritative persisted index for analysis tables.
Artifact names are derived only from validated analysis IDs, and manifest paths are
validated again while reading so a copied or hand-edited bundle cannot escape its
deliverables directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from phenotypic.sdk_._atomic_io import atomic_write_json
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_._io_constants import (
    ANALYSIS_MANIFEST_SCHEMA_VERSION,
    AnalysisArtifactPaths as AnalysisArtifactPaths,
    analysis_manifest_path as analysis_manifest_path,
    named_analysis_csv_path as named_analysis_csv_path,
    named_analysis_parquet_path as named_analysis_parquet_path,
    named_analysis_paths as named_analysis_paths,
    validate_analysis_id as validate_analysis_id,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TRANSACTION_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_PUBLICATION_JOURNAL_FILENAME: Final = ".analysis-publication.json"
_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "class",
        "csv",
        "parquet",
        "rows",
        "columns",
        "csv_sha256",
        "parquet_sha256",
    }
)


class AnalysisManifestError(ValueError):
    """Raised when a persisted analysis manifest violates its contract."""


class AnalysisArtifactIntegrityError(AnalysisManifestError):
    """Raised when a manifest-selected artifact does not match its checksum."""


@dataclass(frozen=True)
class _AnalysisPublicationPaths:
    """All paths participating in one recoverable publication transaction."""

    token: str
    canonical: AnalysisArtifactPaths
    lock: Path
    journal: Path
    staged_csv: Path
    staged_parquet: Path
    backup_csv: Path
    backup_parquet: Path


def _analysis_publication_paths(
    deliverables_base: Path, analysis_id: str, token: str
) -> _AnalysisPublicationPaths:
    """Derive writer and recovery paths from one validated transaction key."""
    base = Path(deliverables_base)
    safe_id = validate_analysis_id(analysis_id)
    if not _TRANSACTION_TOKEN_PATTERN.fullmatch(token):
        raise ValueError("analysis publication token must be 32 lowercase hex digits")
    canonical = named_analysis_paths(base, safe_id)
    return _AnalysisPublicationPaths(
        token=token,
        canonical=canonical,
        lock=base / ".analysis-artifacts.lock",
        journal=base / _PUBLICATION_JOURNAL_FILENAME,
        staged_csv=base / f".{canonical.csv.name}.{token}.staged",
        staged_parquet=base / f".{canonical.parquet.name}.{token}.staged",
        backup_csv=base / f".{canonical.csv.name}.{token}.backup",
        backup_parquet=base / f".{canonical.parquet.name}.{token}.backup",
    )


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        A 64-character lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AnalysisManifestEntry:
    """Authoritative metadata for one named analysis artifact pair."""

    producer_class: str
    csv: str
    parquet: str
    rows: int
    columns: tuple[str, ...]
    csv_sha256: str
    parquet_sha256: str

    @classmethod
    def from_mapping(
        cls, analysis_id: str, payload: Mapping[str, Any]
    ) -> AnalysisManifestEntry:
        """Validate and decode one JSON manifest entry.

        Args:
            analysis_id: Mapping key that owns the entry.
            payload: Decoded JSON mapping.

        Returns:
            A validated immutable entry.

        Raises:
            AnalysisManifestError: If fields, values, or artifact paths are invalid.
        """
        analysis_id = _validate_manifest_analysis_id(analysis_id)
        if not isinstance(payload, Mapping):
            raise AnalysisManifestError(
                f"analysis manifest entry {analysis_id!r} must be an object"
            )
        fields = frozenset(payload)
        if fields != _ENTRY_FIELDS:
            missing = sorted(_ENTRY_FIELDS - fields)
            extra = sorted(fields - _ENTRY_FIELDS)
            raise AnalysisManifestError(
                f"analysis manifest entry {analysis_id!r} has invalid fields; "
                f"missing={missing}, extra={extra}"
            )

        producer_class = payload["class"]
        if not isinstance(producer_class, str) or not producer_class:
            raise AnalysisManifestError(
                f"analysis manifest entry {analysis_id!r} has an invalid class"
            )
        csv = _validate_artifact_filename(
            analysis_id, payload["csv"], suffix=".csv"
        )
        parquet = _validate_artifact_filename(
            analysis_id, payload["parquet"], suffix=".parquet"
        )
        rows = payload["rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise AnalysisManifestError(
                f"analysis manifest entry {analysis_id!r} has invalid rows"
            )
        columns_value = payload["columns"]
        if not isinstance(columns_value, list) or not all(
            isinstance(column, str) for column in columns_value
        ):
            raise AnalysisManifestError(
                f"analysis manifest entry {analysis_id!r} has invalid columns"
            )
        csv_sha256 = _validate_checksum(
            analysis_id, "csv_sha256", payload["csv_sha256"]
        )
        parquet_sha256 = _validate_checksum(
            analysis_id, "parquet_sha256", payload["parquet_sha256"]
        )
        return cls(
            producer_class=producer_class,
            csv=csv,
            parquet=parquet,
            rows=rows,
            columns=tuple(columns_value),
            csv_sha256=csv_sha256,
            parquet_sha256=parquet_sha256,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-compatible representation of this entry."""
        return {
            "class": self.producer_class,
            "csv": self.csv,
            "parquet": self.parquet,
            "rows": self.rows,
            "columns": list(self.columns),
            "csv_sha256": self.csv_sha256,
            "parquet_sha256": self.parquet_sha256,
        }


@dataclass(frozen=True)
class AnalysisManifest:
    """Versioned persisted index of named analysis artifacts."""

    analyses: Mapping[str, AnalysisManifestEntry]
    schema_version: int = ANALYSIS_MANIFEST_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AnalysisManifest:
        """Validate and decode a complete manifest mapping."""
        if not isinstance(payload, Mapping):
            raise AnalysisManifestError(
                "analysis manifest root must be an object"
            )
        fields = frozenset(payload)
        expected = frozenset({"schema_version", "analyses"})
        if fields != expected:
            raise AnalysisManifestError(
                "analysis manifest must contain only schema_version and analyses"
            )
        schema_version = payload["schema_version"]
        if schema_version != ANALYSIS_MANIFEST_SCHEMA_VERSION:
            raise AnalysisManifestError(
                "unsupported analysis manifest schema_version "
                f"{schema_version!r}; expected {ANALYSIS_MANIFEST_SCHEMA_VERSION}"
            )
        analyses_payload = payload["analyses"]
        if not isinstance(analyses_payload, Mapping):
            raise AnalysisManifestError(
                "analysis manifest analyses must be an object"
            )
        analyses: dict[str, AnalysisManifestEntry] = {}
        folded_ids: dict[str, str] = {}
        for analysis_id, entry_payload in analyses_payload.items():
            safe_id = _validate_manifest_analysis_id(analysis_id)
            previous = folded_ids.get(safe_id.casefold())
            if previous is not None and previous != safe_id:
                raise AnalysisManifestError(
                    "analysis manifest IDs collide case-insensitively: "
                    f"{previous!r} and {safe_id!r}"
                )
            folded_ids[safe_id.casefold()] = safe_id
            analyses[safe_id] = AnalysisManifestEntry.from_mapping(
                safe_id, entry_payload
            )
        return cls(analyses=analyses, schema_version=schema_version)

    def to_mapping(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible manifest mapping."""
        return {
            "schema_version": self.schema_version,
            "analyses": {
                analysis_id: self.analyses[analysis_id].to_mapping()
                for analysis_id in sorted(self.analyses)
            },
        }

    def with_entry(
        self, analysis_id: str, entry: AnalysisManifestEntry
    ) -> AnalysisManifest:
        """Return a manifest with ``analysis_id`` inserted or replaced."""
        safe_id = validate_analysis_id(analysis_id)
        _validate_entry_identity(safe_id, entry)
        analyses = dict(self.analyses)
        for existing_id in analyses:
            if existing_id != safe_id and existing_id.casefold() == safe_id.casefold():
                raise AnalysisManifestError(
                    "analysis IDs collide case-insensitively: "
                    f"{existing_id!r} and {safe_id!r}"
                )
        analyses[safe_id] = entry
        return AnalysisManifest(analyses=analyses)


def build_analysis_manifest_entry(
    *,
    analysis_id: str,
    producer_class: str,
    csv_path: Path,
    parquet_path: Path,
    rows: int,
    columns: Sequence[str],
) -> AnalysisManifestEntry:
    """Build a checksummed manifest entry from already-published artifacts.

    Both artifacts must use the exact validated ID-derived filenames. This helper is
    intended to run after atomic artifact replacement and before the manifest is
    published last.
    """
    safe_id = validate_analysis_id(analysis_id)
    expected_csv = f"{safe_id}.csv"
    expected_parquet = f"{safe_id}.parquet"
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    if csv_path.name != expected_csv or parquet_path.name != expected_parquet:
        raise ValueError(
            "analysis artifact filenames must be derived from analysis_id: "
            f"{expected_csv!r} and {expected_parquet!r}"
        )
    if csv_path.parent.resolve() != parquet_path.parent.resolve():
        raise ValueError(
            "analysis CSV and Parquet artifacts must share a directory"
        )
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError("rows must be a non-negative integer")
    column_tuple = tuple(columns)
    if not all(isinstance(column, str) for column in column_tuple):
        raise TypeError("columns must contain only strings")
    if not isinstance(producer_class, str) or not producer_class:
        raise ValueError("producer_class must be a non-empty string")
    return AnalysisManifestEntry(
        producer_class=producer_class,
        csv=expected_csv,
        parquet=expected_parquet,
        rows=rows,
        columns=column_tuple,
        csv_sha256=file_sha256(csv_path),
        parquet_sha256=file_sha256(parquet_path),
    )


def read_analysis_manifest(deliverables_base: Path) -> AnalysisManifest | None:
    """Read and validate the manifest, or return ``None`` when it is absent."""
    base = Path(deliverables_base).resolve()
    path = analysis_manifest_path(base)
    if not path.exists():
        return None
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise AnalysisManifestError(
            f"could not resolve analysis manifest {path}"
        ) from exc
    if resolved_path.parent != base:
        raise AnalysisManifestError("analysis manifest escapes deliverables")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisManifestError(
            f"could not read analysis manifest {path}"
        ) from exc
    try:
        return AnalysisManifest.from_mapping(payload)
    except AnalysisManifestError:
        raise
    except (TypeError, ValueError) as exc:
        raise AnalysisManifestError(
            f"invalid analysis manifest {path}"
        ) from exc


def write_analysis_manifest(
    deliverables_base: Path, manifest: AnalysisManifest
) -> Path:
    """Atomically publish an analysis manifest and return its path."""
    validated = AnalysisManifest.from_mapping(manifest.to_mapping())
    path = analysis_manifest_path(deliverables_base)
    atomic_write_json(path, validated.to_mapping())
    return path


def publish_analysis_manifest_entry(
    deliverables_base: Path,
    analysis_id: str,
    entry: AnalysisManifestEntry,
) -> Path:
    """Atomically insert or replace one manifest entry.

    Existing entries are retained. Callers must publish both named artifacts before
    invoking this helper so the manifest remains the last visible generation marker.
    """
    base = Path(deliverables_base)
    with exclusive_path_lock(base / ".analysis-manifest.lock"):
        manifest = read_analysis_manifest(base) or AnalysisManifest(analyses={})
        return write_analysis_manifest(
            base, manifest.with_entry(analysis_id, entry)
        )


def write_analysis_publication_journal(
    deliverables_base: Path,
    *,
    analysis_id: str,
    token: str,
    old_csv_exists: bool,
    old_parquet_exists: bool,
    entry: AnalysisManifestEntry,
) -> Path:
    """Persist recovery state before replacing class-named artifacts.

    The caller must hold ``.analysis-artifacts.lock``. The journal is removed
    only after the manifest pointer commits or recovery restores the previous
    canonical files.
    """
    safe_id = validate_analysis_id(analysis_id)
    if not _TRANSACTION_TOKEN_PATTERN.fullmatch(token):
        raise ValueError("analysis publication token must be 32 lowercase hex digits")
    _validate_entry_identity(safe_id, entry)
    if not isinstance(old_csv_exists, bool) or not isinstance(
        old_parquet_exists, bool
    ):
        raise TypeError("analysis publication existence flags must be bool")
    path = _analysis_publication_paths(
        deliverables_base, safe_id, token
    ).journal
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "analysis_id": safe_id,
            "token": token,
            "old_csv_exists": old_csv_exists,
            "old_parquet_exists": old_parquet_exists,
            "entry": entry.to_mapping(),
        },
    )
    return path


def recover_analysis_publication(deliverables_base: Path) -> bool:
    """Recover or finish an interrupted class-named artifact publication.

    The caller must hold ``.analysis-artifacts.lock``. If the manifest already
    contains the journaled entry, the generation committed and only temporary
    files are removed. Otherwise the canonical CSV/Parquet pair is restored to
    the generation still selected by the manifest.

    Returns:
        ``True`` when a journal was found and resolved, otherwise ``False``.
    """
    base = Path(deliverables_base)
    journal_path = base / _PUBLICATION_JOURNAL_FILENAME
    if not journal_path.exists():
        return False
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisManifestError(
            f"could not read analysis publication journal {journal_path}"
        ) from exc
    expected_fields = {
        "schema_version",
        "analysis_id",
        "token",
        "old_csv_exists",
        "old_parquet_exists",
        "entry",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise AnalysisManifestError("invalid analysis publication journal fields")
    if payload["schema_version"] != 1:
        raise AnalysisManifestError(
            "unsupported analysis publication journal schema_version"
        )
    analysis_id = _validate_manifest_analysis_id(payload["analysis_id"])
    token = payload["token"]
    if not isinstance(token, str) or not _TRANSACTION_TOKEN_PATTERN.fullmatch(
        token
    ):
        raise AnalysisManifestError("invalid analysis publication journal token")
    old_csv_exists = payload["old_csv_exists"]
    old_parquet_exists = payload["old_parquet_exists"]
    if not isinstance(old_csv_exists, bool) or not isinstance(
        old_parquet_exists, bool
    ):
        raise AnalysisManifestError(
            "invalid analysis publication journal existence flags"
        )
    entry = AnalysisManifestEntry.from_mapping(analysis_id, payload["entry"])
    paths = _analysis_publication_paths(base, analysis_id, token)

    manifest = read_analysis_manifest(base)
    committed = (
        manifest is not None and manifest.analyses.get(analysis_id) == entry
    )
    if not committed:
        _restore_analysis_artifact(
            paths.canonical.csv,
            paths.backup_csv,
            old_exists=old_csv_exists,
        )
        _restore_analysis_artifact(
            paths.canonical.parquet,
            paths.backup_parquet,
            old_exists=old_parquet_exists,
        )

    for temporary in (
        paths.staged_csv,
        paths.staged_parquet,
        paths.backup_csv,
        paths.backup_parquet,
    ):
        temporary.unlink(missing_ok=True)
    journal_path.unlink(missing_ok=True)
    return True


def _restore_analysis_artifact(
    canonical: Path, backup: Path, *, old_exists: bool
) -> None:
    """Restore one pre-transaction artifact from its durable backup."""
    if old_exists:
        if backup.exists():
            canonical.unlink(missing_ok=True)
            os.replace(backup, canonical)
        return
    canonical.unlink(missing_ok=True)


def resolve_manifest_artifact_path(
    deliverables_base: Path,
    analysis_id: str,
    entry: AnalysisManifestEntry,
    *,
    artifact: str = "parquet",
) -> Path:
    """Resolve and checksum-verify a manifest-selected artifact.

    Args:
        deliverables_base: Directory containing the manifest and artifact.
        analysis_id: ID owning ``entry``.
        entry: Validated manifest entry.
        artifact: Either ``"parquet"`` or ``"csv"``.

    Returns:
        The resolved artifact path.

    Raises:
        AnalysisManifestError: If the path escapes the deliverables directory or is
            missing.
        AnalysisArtifactIntegrityError: If the file checksum does not match.
    """
    safe_id = validate_analysis_id(analysis_id)
    _validate_entry_identity(safe_id, entry)
    if artifact == "parquet":
        filename = entry.parquet
        expected_checksum = entry.parquet_sha256
    elif artifact == "csv":
        filename = entry.csv
        expected_checksum = entry.csv_sha256
    else:
        raise ValueError("artifact must be 'csv' or 'parquet'")

    base = Path(deliverables_base).resolve()
    candidate = base / filename
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AnalysisManifestError(
            f"analysis {safe_id!r} {artifact} artifact is missing: {candidate}"
        ) from exc
    if resolved.parent != base:
        raise AnalysisManifestError(
            f"analysis {safe_id!r} {artifact} artifact escapes deliverables"
        )
    actual_checksum = file_sha256(resolved)
    if actual_checksum != expected_checksum:
        raise AnalysisArtifactIntegrityError(
            f"analysis {safe_id!r} {artifact} checksum mismatch: "
            f"expected {expected_checksum}, got {actual_checksum}"
        )
    return resolved


def _validate_manifest_analysis_id(value: Any) -> str:
    if not isinstance(value, str):
        raise AnalysisManifestError("analysis manifest IDs must be strings")
    try:
        return validate_analysis_id(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisManifestError(
            f"analysis manifest contains unsafe analysis ID {value!r}"
        ) from exc


def _validate_artifact_filename(
    analysis_id: str, value: Any, *, suffix: str
) -> str:
    expected = f"{analysis_id}{suffix}"
    if not isinstance(value, str) or value != expected:
        raise AnalysisManifestError(
            f"analysis manifest entry {analysis_id!r} must use artifact "
            f"filename {expected!r}"
        )
    if "/" in value or "\\" in value:
        raise AnalysisManifestError(
            f"analysis manifest entry {analysis_id!r} contains a path separator"
        )
    return value


def _validate_checksum(analysis_id: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise AnalysisManifestError(
            f"analysis manifest entry {analysis_id!r} has invalid {field}"
        )
    return value


def _validate_entry_identity(
    analysis_id: str, entry: AnalysisManifestEntry
) -> None:
    payload = entry.to_mapping()
    AnalysisManifestEntry.from_mapping(analysis_id, payload)
