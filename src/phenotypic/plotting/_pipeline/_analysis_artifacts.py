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
from phenotypic.sdk_._delimited import CSV_SUFFIX as _CSV_SUFFIX
from phenotypic.sdk_._delimited import TSV_SUFFIX as _TSV_SUFFIX
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_._io_constants import (
    ANALYSIS_MANIFEST_SCHEMA_VERSION,
    AnalysisArtifactPaths as AnalysisArtifactPaths,
    analysis_manifest_path as analysis_manifest_path,
    named_analysis_csv_path as named_analysis_csv_path,
    named_analysis_tsv_path as named_analysis_tsv_path,
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
        "tsv",
        "parquet",
        "rows",
        "columns",
        "tsv_sha256",
        "parquet_sha256",
    }
)
#: The last manifest version whose entries named a ``csv`` table. Its entries
#: still read: ``csv``/``csv_sha256`` map onto ``tsv``/``tsv_sha256`` (spec B3).
_LEGACY_CSV_SCHEMA_VERSION: Final = 1
_LEGACY_TO_CURRENT_ENTRY_KEYS: Final[Mapping[str, str]] = {
    "csv": "tsv",
    "csv_sha256": "tsv_sha256",
}
_SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset(
    {_LEGACY_CSV_SCHEMA_VERSION, ANALYSIS_MANIFEST_SCHEMA_VERSION}
)
#: Publication journal versions. ``1`` journaled a ``.csv`` generation, so its
#: staged and backup files carry ``.csv`` names; recovery still resolves it.
_JOURNAL_SCHEMA_VERSION: Final = 2
_LEGACY_CSV_JOURNAL_SCHEMA_VERSION: Final = 1


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
    table: Path
    staged_table: Path
    staged_parquet: Path
    backup_table: Path
    backup_parquet: Path


def _analysis_publication_paths(
    deliverables_base: Path,
    analysis_id: str,
    token: str,
    *,
    table_suffix: str = _TSV_SUFFIX,
) -> _AnalysisPublicationPaths:
    """Derive writer and recovery paths from one validated transaction key.

    Args:
        deliverables_base: Directory that holds the artifacts and manifest.
        analysis_id: The analysis whose generation is published.
        token: The transaction's 32-hex token.
        table_suffix: ``.tsv`` for every publication this release writes;
            ``.csv`` only to recover a version-1 journal, whose staged and
            backup files were named for the CSV table.
    """
    base = Path(deliverables_base)
    safe_id = validate_analysis_id(analysis_id)
    if not _TRANSACTION_TOKEN_PATTERN.fullmatch(token):
        raise ValueError("analysis publication token must be 32 lowercase hex digits")
    canonical = named_analysis_paths(base, safe_id)
    if table_suffix == _TSV_SUFFIX:
        table = canonical.tsv
    elif table_suffix == _CSV_SUFFIX:
        table = canonical.csv
    else:
        raise ValueError("table_suffix must be '.tsv' or '.csv'")
    return _AnalysisPublicationPaths(
        token=token,
        canonical=canonical,
        lock=base / ".analysis-artifacts.lock",
        journal=base / _PUBLICATION_JOURNAL_FILENAME,
        table=table,
        staged_table=base / f".{table.name}.{token}.staged",
        staged_parquet=base / f".{canonical.parquet.name}.{token}.staged",
        backup_table=base / f".{table.name}.{token}.backup",
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
    """Authoritative metadata for one named analysis artifact pair.

    ``tsv`` names the entry's text table. It is ``<id>.tsv`` for every entry
    this release publishes, and ``<id>.csv`` only for an entry read from a
    version-1 manifest (or carried forward from one into a later manifest),
    which describes a generation published before the TSV cutover.
    """

    producer_class: str
    tsv: str
    parquet: str
    rows: int
    columns: tuple[str, ...]
    tsv_sha256: str
    parquet_sha256: str

    @classmethod
    def from_mapping(
        cls,
        analysis_id: str,
        payload: Mapping[str, Any],
        *,
        schema_version: int = ANALYSIS_MANIFEST_SCHEMA_VERSION,
    ) -> AnalysisManifestEntry:
        """Validate and decode one JSON manifest entry.

        Args:
            analysis_id: Mapping key that owns the entry.
            payload: Decoded JSON mapping.
            schema_version: Version of the manifest the entry came from. A
                version-1 entry carries ``csv``/``csv_sha256``, which become
                ``tsv``/``tsv_sha256``.

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
        legacy = schema_version == _LEGACY_CSV_SCHEMA_VERSION
        if legacy:
            if "tsv" in payload or "tsv_sha256" in payload:
                raise AnalysisManifestError(
                    f"analysis manifest entry {analysis_id!r} mixes version-1 "
                    "csv fields with tsv fields"
                )
            payload = {
                _LEGACY_TO_CURRENT_ENTRY_KEYS.get(key, key): value
                for key, value in payload.items()
            }
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
        tsv = _validate_artifact_filename(
            analysis_id,
            payload["tsv"],
            suffixes=(_CSV_SUFFIX,) if legacy else (_TSV_SUFFIX, _CSV_SUFFIX),
        )
        parquet = _validate_artifact_filename(
            analysis_id, payload["parquet"], suffixes=(".parquet",)
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
        tsv_sha256 = _validate_checksum(
            analysis_id, "tsv_sha256", payload["tsv_sha256"]
        )
        parquet_sha256 = _validate_checksum(
            analysis_id, "parquet_sha256", payload["parquet_sha256"]
        )
        return cls(
            producer_class=producer_class,
            tsv=tsv,
            parquet=parquet,
            rows=rows,
            columns=tuple(columns_value),
            tsv_sha256=tsv_sha256,
            parquet_sha256=parquet_sha256,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the current-version JSON representation of this entry."""
        return {
            "class": self.producer_class,
            "tsv": self.tsv,
            "parquet": self.parquet,
            "rows": self.rows,
            "columns": list(self.columns),
            "tsv_sha256": self.tsv_sha256,
            "parquet_sha256": self.parquet_sha256,
        }


@dataclass(frozen=True)
class AnalysisManifest:
    """Versioned persisted index of named analysis artifacts.

    A decoded manifest is always current-version in memory: a version-1 file is
    upgraded while it is read, and :meth:`to_mapping` writes the current
    version, so re-publishing it rewrites the file in the current shape.
    """

    analyses: Mapping[str, AnalysisManifestEntry]
    schema_version: int = ANALYSIS_MANIFEST_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AnalysisManifest:
        """Validate and decode a complete manifest mapping (version 1 or 2)."""
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
        if (
            isinstance(schema_version, bool)
            or schema_version not in _SUPPORTED_SCHEMA_VERSIONS
        ):
            raise AnalysisManifestError(
                "unsupported analysis manifest schema_version "
                f"{schema_version!r}; expected one of "
                f"{sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
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
                safe_id, entry_payload, schema_version=schema_version
            )
        return cls(analyses=analyses)

    def to_mapping(self) -> dict[str, Any]:
        """Return a deterministic, current-version JSON-compatible mapping."""
        return {
            "schema_version": ANALYSIS_MANIFEST_SCHEMA_VERSION,
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
    tsv_path: Path,
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
    expected_tsv = f"{safe_id}{_TSV_SUFFIX}"
    expected_parquet = f"{safe_id}.parquet"
    tsv_path = Path(tsv_path)
    parquet_path = Path(parquet_path)
    if tsv_path.name != expected_tsv or parquet_path.name != expected_parquet:
        raise ValueError(
            "analysis artifact filenames must be derived from analysis_id: "
            f"{expected_tsv!r} and {expected_parquet!r}"
        )
    if tsv_path.parent.resolve() != parquet_path.parent.resolve():
        raise ValueError(
            "analysis TSV and Parquet artifacts must share a directory"
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
        tsv=expected_tsv,
        parquet=expected_parquet,
        rows=rows,
        columns=column_tuple,
        tsv_sha256=file_sha256(tsv_path),
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
    old_tsv_exists: bool,
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
    if not isinstance(old_tsv_exists, bool) or not isinstance(
        old_parquet_exists, bool
    ):
        raise TypeError("analysis publication existence flags must be bool")
    path = _analysis_publication_paths(
        deliverables_base, safe_id, token
    ).journal
    atomic_write_json(
        path,
        {
            "schema_version": _JOURNAL_SCHEMA_VERSION,
            "analysis_id": safe_id,
            "token": token,
            "old_tsv_exists": old_tsv_exists,
            "old_parquet_exists": old_parquet_exists,
            "entry": entry.to_mapping(),
        },
    )
    return path


def recover_analysis_publication(deliverables_base: Path) -> bool:
    """Recover or finish an interrupted class-named artifact publication.

    The caller must hold ``.analysis-artifacts.lock``. If the manifest already
    contains the journaled entry, the generation committed and only temporary
    files are removed. Otherwise the canonical TSV/Parquet pair is restored to
    the generation still selected by the manifest. A version-1 journal, left by
    an interrupted pre-TSV publication, is resolved against the ``.csv`` table
    it staged.

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
    if not isinstance(payload, Mapping):
        raise AnalysisManifestError("invalid analysis publication journal fields")
    journal_version = payload.get("schema_version")
    if isinstance(journal_version, bool) or journal_version not in (
        _JOURNAL_SCHEMA_VERSION,
        _LEGACY_CSV_JOURNAL_SCHEMA_VERSION,
    ):
        raise AnalysisManifestError(
            "unsupported analysis publication journal schema_version"
        )
    legacy = journal_version == _LEGACY_CSV_JOURNAL_SCHEMA_VERSION
    old_table_key = "old_csv_exists" if legacy else "old_tsv_exists"
    expected_fields = {
        "schema_version",
        "analysis_id",
        "token",
        old_table_key,
        "old_parquet_exists",
        "entry",
    }
    if set(payload) != expected_fields:
        raise AnalysisManifestError("invalid analysis publication journal fields")
    analysis_id = _validate_manifest_analysis_id(payload["analysis_id"])
    token = payload["token"]
    if not isinstance(token, str) or not _TRANSACTION_TOKEN_PATTERN.fullmatch(
        token
    ):
        raise AnalysisManifestError("invalid analysis publication journal token")
    old_table_exists = payload[old_table_key]
    old_parquet_exists = payload["old_parquet_exists"]
    if not isinstance(old_table_exists, bool) or not isinstance(
        old_parquet_exists, bool
    ):
        raise AnalysisManifestError(
            "invalid analysis publication journal existence flags"
        )
    entry = AnalysisManifestEntry.from_mapping(
        analysis_id,
        payload["entry"],
        schema_version=(
            _LEGACY_CSV_SCHEMA_VERSION if legacy else ANALYSIS_MANIFEST_SCHEMA_VERSION
        ),
    )
    paths = _analysis_publication_paths(
        base,
        analysis_id,
        token,
        table_suffix=_CSV_SUFFIX if legacy else _TSV_SUFFIX,
    )

    manifest = read_analysis_manifest(base)
    committed = (
        manifest is not None and manifest.analyses.get(analysis_id) == entry
    )
    if not committed:
        _restore_analysis_artifact(
            paths.table,
            paths.backup_table,
            old_exists=old_table_exists,
        )
        _restore_analysis_artifact(
            paths.canonical.parquet,
            paths.backup_parquet,
            old_exists=old_parquet_exists,
        )

    for temporary in (
        paths.staged_table,
        paths.staged_parquet,
        paths.backup_table,
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
        artifact: Either ``"parquet"`` or ``"tsv"`` (the entry's text table,
            which is a ``.csv`` for an entry from a version-1 manifest).

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
    elif artifact == "tsv":
        filename = entry.tsv
        expected_checksum = entry.tsv_sha256
    else:
        raise ValueError("artifact must be 'tsv' or 'parquet'")

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
    analysis_id: str, value: Any, *, suffixes: Sequence[str]
) -> str:
    expected = [f"{analysis_id}{suffix}" for suffix in suffixes]
    if not isinstance(value, str) or value not in expected:
        allowed = " or ".join(repr(name) for name in expected)
        raise AnalysisManifestError(
            f"analysis manifest entry {analysis_id!r} must use artifact "
            f"filename {allowed}"
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
