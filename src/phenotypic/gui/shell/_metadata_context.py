"""Shared metadata CSV context for the GUI shell.

The metadata context mirrors the source-image-root context: browser storage is
transport only, and every filesystem consumer resolves payloads through this
module before use.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.schema import METADATA

logger = logging.getLogger(__name__)

METADATA_CSV_PAYLOAD_VERSION = 1

MetadataLookupState: TypeAlias = Literal[
    "unset",
    "unavailable",
    "missing_image_name",
    "no_match",
    "matched",
]


class MetadataCsvPayload(TypedDict):
    """Versioned JSON payload for the shared metadata CSV store."""

    abs_path: str
    rel_path: str
    label: str
    validated: bool
    version: int
    has_image_name: bool
    row_count: int
    unique_image_names: bool


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
    """Return a validated metadata CSV payload for ``path``.

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
        rows = _read_rows(resolved)
    except OSError:
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
    has_image_name = bool(rows[0].keys()) and METADATA.IMAGE_NAME in rows[0]
    return {
        "abs_path": str(resolved),
        "rel_path": "." if rel_path == "" else rel_path,
        "label": resolved.name or str(resolved),
        "validated": True,
        "version": METADATA_CSV_PAYLOAD_VERSION,
        "has_image_name": has_image_name,
        "row_count": len(rows),
        "unique_image_names": has_image_name
        and len(image_values) == len(set(image_values)),
    }


def resolve_metadata_csv(
    sandbox: SandboxRoot,
    payload: object,
) -> Path | None:
    """Return a sandbox-contained metadata CSV path from ``payload``."""
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != METADATA_CSV_PAYLOAD_VERSION:
        return None
    if payload.get("validated") is not True:
        return None
    raw_path = payload.get("abs_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return _resolve_candidate_csv(sandbox, raw_path)


def metadata_csv_label(payload: object) -> str:
    """Return the compact metadata CSV label for the settings menu."""
    if payload is None:
        return "metadata: unset"
    if not isinstance(payload, dict):
        return "metadata: invalid"
    if payload.get("version") != METADATA_CSV_PAYLOAD_VERSION:
        return "metadata: invalid"
    if payload.get("validated") is not True:
        return "metadata: invalid"
    label = payload.get("label")
    if not isinstance(label, str) or not label:
        return "metadata: invalid"
    return f"metadata: {label}"


def metadata_csv_title(payload: object) -> str:
    """Return the full-path hover title for the metadata CSV label."""
    if not isinstance(payload, dict):
        return "No metadata CSV selected"
    path = payload.get("abs_path")
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
    except OSError:
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]


__all__ = [
    "METADATA_CSV_PAYLOAD_VERSION",
    "MetadataCsvPayload",
    "MetadataLookupResult",
    "MetadataLookupState",
    "metadata_csv_label",
    "metadata_csv_title",
    "metadata_payload_from_path",
    "read_metadata_row_for_image_stem",
    "resolve_metadata_csv",
]
