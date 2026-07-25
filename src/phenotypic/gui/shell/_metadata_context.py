"""Shared metadata CSV context for the GUI shell.

The metadata context mirrors the source-image-root context: browser storage is
transport only, and every filesystem consumer resolves payloads through this
module before use. Version 2 resolves only a sandbox-relative path whose stored
sandbox fingerprint matches the current launch.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from phenotypic.gui.shell._sandbox import (
    SandboxRoot,
    _is_safe_relative_path,
    _v1_selection_matches_sandbox,
)
from phenotypic.gui.shell._source_context import sandbox_fingerprint
from phenotypic.schema import METADATA

logger = logging.getLogger(__name__)

METADATA_CSV_PAYLOAD_VERSION = 2
_METADATA_CSV_PAYLOAD_V1 = 1
_METADATA_KIND: Literal["metadata_csv"] = "metadata_csv"
_UNAVAILABLE_LABEL = "Previous metadata unavailable in this sandbox"

MetadataResolutionState: TypeAlias = Literal[
    "unset",
    "resolved",
    "invalid",
    "unavailable",
    "fingerprint_mismatch",
]
MetadataLookupState: TypeAlias = Literal[
    "unset",
    "unavailable",
    "missing_image_name",
    "no_match",
    "matched",
]


class MetadataCsvValidation(TypedDict):
    """Filesystem facts recorded when a metadata CSV is selected."""

    exists: bool
    is_file: bool
    is_csv: bool
    readable: bool


class MetadataCsvPayload(TypedDict):
    """Version 2 JSON payload for the shared metadata CSV store.

    The final four path/validation fields are compatibility mirrors for
    out-of-scope consumers. Resolution never treats them as authoritative.
    """

    version: int
    kind: Literal["metadata_csv"]
    relative_path: str
    absolute_path_at_selection: str
    sandbox_fingerprint: str
    validation: MetadataCsvValidation
    selected_at: str
    has_image_name: bool
    row_count: int
    unique_image_names: bool
    abs_path: str
    rel_path: str
    label: str
    validated: bool


@dataclass(frozen=True)
class MetadataCsvResolution:
    """Typed result of resolving a browser metadata payload."""

    state: MetadataResolutionState
    path: Path | None = None
    payload_version: int | None = None

    @property
    def is_resolved(self) -> bool:
        """Whether the payload resolves to a current sandbox CSV."""
        return self.state == "resolved"


@dataclass(frozen=True)
class MetadataLookupResult:
    """Result of looking up one image stem in the selected metadata CSV."""

    state: MetadataLookupState
    image_stem: str
    rows: list[dict[str, str]]

    @property
    def row_count(self) -> int:
        """Number of metadata rows included in this lookup result."""
        return len(self.rows)


def metadata_payload_from_path(
    sandbox: SandboxRoot,
    path: Path | str,
) -> MetadataCsvPayload | None:
    """Return a version 2 metadata payload after explicit selection.

    Args:
        sandbox: Frozen-at-launch sandbox boundary.
        path: Candidate metadata CSV path.

    Returns:
        Versioned payload for browser storage, or ``None`` when the candidate
        escapes the sandbox, is not an existing ``.csv`` file, or cannot be
        read as CSV.
    """
    resolved = _resolve_candidate_csv(sandbox, path)
    if resolved is None:
        return None
    try:
        columns, rows = read_metadata_csv_table(resolved)
    except (csv.Error, OSError, UnicodeError):
        logger.warning("metadata CSV is unreadable: %s", resolved)
        return None
    try:
        rel_path = str(resolved.relative_to(sandbox.root))
    except ValueError:
        logger.warning("metadata CSV resolved outside sandbox: %s", resolved)
        return None

    image_values = [
        row.get(METADATA.IMAGE_NAME, "")
        for row in rows
        if row.get(METADATA.IMAGE_NAME, "") != ""
    ]
    has_image_name = METADATA.IMAGE_NAME in columns
    relative_path = "." if rel_path == "" else rel_path
    payload: MetadataCsvPayload = {
        "version": METADATA_CSV_PAYLOAD_VERSION,
        "kind": _METADATA_KIND,
        "relative_path": relative_path,
        "absolute_path_at_selection": str(resolved),
        "sandbox_fingerprint": sandbox_fingerprint(sandbox),
        "validation": {
            "exists": True,
            "is_file": True,
            "is_csv": True,
            "readable": True,
        },
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "has_image_name": has_image_name,
        "row_count": len(rows),
        "unique_image_names": has_image_name
        and len(image_values) == len(set(image_values)),
        # Compatibility mirrors. Resolution never trusts these fields.
        "abs_path": str(resolved),
        "rel_path": relative_path,
        "label": resolved.name or str(resolved),
        "validated": True,
    }
    return payload


def resolve_metadata_csv_state(
    sandbox: SandboxRoot,
    payload: object,
) -> MetadataCsvResolution:
    """Return the typed current-sandbox resolution of ``payload``."""
    if payload is None:
        return MetadataCsvResolution("unset")
    if not isinstance(payload, dict):
        return MetadataCsvResolution("invalid")

    version = payload.get("version")
    if version == METADATA_CSV_PAYLOAD_VERSION:
        return _resolve_v2_metadata(sandbox, payload)
    if version == _METADATA_CSV_PAYLOAD_V1:
        return _resolve_v1_metadata(sandbox, payload)
    return MetadataCsvResolution(
        "invalid",
        payload_version=version if type(version) is int else None,
    )


def resolve_metadata_csv(
    sandbox: SandboxRoot,
    payload: object,
) -> Path | None:
    """Compatibility wrapper returning only a resolved metadata CSV."""
    return resolve_metadata_csv_state(sandbox, payload).path


def metadata_csv_label(
    payload: object,
    *,
    sandbox: SandboxRoot | None = None,
) -> str:
    """Return the compact metadata label for the settings menu."""
    if sandbox is not None:
        resolution = resolve_metadata_csv_state(sandbox, payload)
        if resolution.state == "unset":
            return "metadata: unset"
        if resolution.path is not None:
            return f"metadata: {resolution.path.name or resolution.path}"
        if resolution.state in {"unavailable", "fingerprint_mismatch"}:
            return _UNAVAILABLE_LABEL
        return "metadata: invalid"
    return _legacy_metadata_csv_label(payload)


def metadata_csv_title(
    payload: object,
    *,
    sandbox: SandboxRoot | None = None,
) -> str:
    """Return the hover title for the current metadata CSV."""
    if sandbox is not None:
        resolution = resolve_metadata_csv_state(sandbox, payload)
        if resolution.path is not None:
            return str(resolution.path)
        if resolution.state in {"unavailable", "fingerprint_mismatch"}:
            return _UNAVAILABLE_LABEL
        return "No metadata CSV selected"
    if not isinstance(payload, dict):
        return "No metadata CSV selected"
    path = payload.get("absolute_path_at_selection", payload.get("abs_path"))
    if not isinstance(path, str) or not path:
        return "No metadata CSV selected"
    return path


def read_metadata_row_for_image_stem(
    sandbox: SandboxRoot,
    payload: object,
    image_stem: str,
) -> MetadataLookupResult:
    """Look up metadata values for ``image_stem`` in the selected CSV."""
    if payload is None:
        return MetadataLookupResult("unset", image_stem, [])
    path = resolve_metadata_csv(sandbox, payload)
    if path is None:
        return MetadataLookupResult("unavailable", image_stem, [])
    try:
        rows = _read_rows(path)
    except (csv.Error, OSError, UnicodeError):
        return MetadataLookupResult("unavailable", image_stem, [])
    if not rows or METADATA.IMAGE_NAME not in rows[0]:
        return MetadataLookupResult("missing_image_name", image_stem, [])

    matches = [row for row in rows if row.get(METADATA.IMAGE_NAME, "") == image_stem]
    if not matches:
        return MetadataLookupResult("no_match", image_stem, [])
    display_rows = [
        {
            str(key): str(value)
            for key, value in row.items()
            if key != METADATA.IMAGE_NAME
        }
        for row in matches
    ]
    return MetadataLookupResult("matched", image_stem, display_rows)


def _resolve_v2_metadata(
    sandbox: SandboxRoot,
    payload: dict[object, object],
) -> MetadataCsvResolution:
    if payload.get("kind") != _METADATA_KIND:
        return MetadataCsvResolution(
            "invalid",
            payload_version=METADATA_CSV_PAYLOAD_VERSION,
        )
    stored_fingerprint = payload.get("sandbox_fingerprint")
    relative_path = payload.get("relative_path")
    if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
        return MetadataCsvResolution(
            "invalid",
            payload_version=METADATA_CSV_PAYLOAD_VERSION,
        )
    if not isinstance(relative_path, str) or not _is_safe_relative_path(
        relative_path
    ):
        return MetadataCsvResolution(
            "invalid",
            payload_version=METADATA_CSV_PAYLOAD_VERSION,
        )
    if stored_fingerprint != sandbox_fingerprint(sandbox):
        return MetadataCsvResolution(
            "fingerprint_mismatch",
            payload_version=METADATA_CSV_PAYLOAD_VERSION,
        )
    return _resolve_current_metadata_path(
        sandbox,
        relative_path,
        payload_version=METADATA_CSV_PAYLOAD_VERSION,
    )


def _resolve_v1_metadata(
    sandbox: SandboxRoot,
    payload: dict[object, object],
) -> MetadataCsvResolution:
    raw_path = payload.get("abs_path")
    relative_path = payload.get("rel_path")
    if not isinstance(raw_path, str) or not raw_path:
        return MetadataCsvResolution(
            "invalid",
            payload_version=_METADATA_CSV_PAYLOAD_V1,
        )
    if not isinstance(relative_path, str) or not _is_safe_relative_path(
        relative_path
    ):
        return MetadataCsvResolution(
            "invalid",
            payload_version=_METADATA_CSV_PAYLOAD_V1,
        )
    if not _v1_selection_matches_sandbox(
        sandbox,
        raw_path=raw_path,
        relative_path=relative_path,
    ):
        return MetadataCsvResolution(
            "fingerprint_mismatch",
            payload_version=_METADATA_CSV_PAYLOAD_V1,
        )
    return _resolve_current_metadata_path(
        sandbox,
        relative_path,
        payload_version=_METADATA_CSV_PAYLOAD_V1,
    )


def _resolve_current_metadata_path(
    sandbox: SandboxRoot,
    relative_path: str,
    *,
    payload_version: int,
) -> MetadataCsvResolution:
    resolved = _resolve_candidate_csv(sandbox, relative_path)
    if resolved is None:
        return MetadataCsvResolution(
            "unavailable",
            payload_version=payload_version,
        )
    return MetadataCsvResolution(
        "resolved",
        path=resolved,
        payload_version=payload_version,
    )


def _resolve_candidate_csv(sandbox: SandboxRoot, path: Path | str) -> Path | None:
    try:
        resolved = sandbox.resolve(path)
    except (OSError, RuntimeError, ValueError):
        logger.warning("metadata CSV is not resolvable: %r", path)
        return None
    try:
        if not resolved.is_file():
            logger.warning("metadata CSV is not a file: %s", resolved)
            return None
    except (OSError, RuntimeError):
        logger.warning("metadata CSV stat failed: %s", resolved)
        return None
    if resolved.suffix.lower() != ".csv":
        logger.warning("metadata CSV does not use .csv suffix: %s", resolved)
        return None
    try:
        resolved.relative_to(sandbox.root)
    except ValueError:
        logger.warning("metadata CSV resolved outside sandbox: %s", resolved)
        return None
    return resolved


def _legacy_metadata_csv_label(payload: object) -> str:
    if payload is None:
        return "metadata: unset"
    if not isinstance(payload, dict):
        return "metadata: invalid"
    version = payload.get("version")
    if version not in {_METADATA_CSV_PAYLOAD_V1, METADATA_CSV_PAYLOAD_VERSION}:
        return "metadata: invalid"
    if version == METADATA_CSV_PAYLOAD_VERSION and payload.get("kind") != _METADATA_KIND:
        return "metadata: invalid"
    label = payload.get("label")
    if not isinstance(label, str) or not label:
        return "metadata: invalid"
    return f"metadata: {label}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]


def read_metadata_csv_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return ``(column_names, rows)`` for a metadata CSV.

    Decoded with ``utf-8-sig`` so an Excel-authored BOM is stripped and never
    prefixes a ``﻿`` onto the first column name (which would silently
    break the ``csv_image_col`` join). Matches :func:`_read_rows`.

    Args:
        path: Path to the metadata CSV (already sandbox-resolved by the caller).

    Returns:
        ``(columns, rows)`` where ``columns`` is the header order and each row
        is a ``{column: value}`` mapping (values stringified, ``None`` -> ``""``).
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]
    return columns, rows


__all__ = [
    "METADATA_CSV_PAYLOAD_VERSION",
    "MetadataCsvPayload",
    "MetadataCsvResolution",
    "MetadataLookupResult",
    "MetadataLookupState",
    "MetadataResolutionState",
    "metadata_csv_label",
    "metadata_csv_title",
    "metadata_payload_from_path",
    "read_metadata_csv_table",
    "read_metadata_row_for_image_stem",
    "resolve_metadata_csv",
    "resolve_metadata_csv_state",
]
