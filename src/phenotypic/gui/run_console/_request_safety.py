"""Authoritative path and metadata preflight for Run requests.

Browser stores transport confirmation receipts and preflight snapshots, but
they do not confer filesystem authority. Every Validate and Run action calls
the helpers in this module again immediately before allocating a generation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import polars as pl

from phenotypic._cli._cli_directory_scanner import scan_directory_structure
from phenotypic._cli._metadata_join import prepare_metadata_join_keys
from phenotypic.gui.shell._metadata_context import (
    MetadataResolutionState,
    resolve_metadata_csv_state,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import sandbox_fingerprint
from phenotypic.schema import (
    EXPERIMENT_METADATA,
    METADATA,
    REMBI_MODULE,
    header_to_module,
)

__all__ = [
    "MetadataPreflight",
    "OutputConfirmation",
    "RunRequestSafetyError",
    "build_metadata_preflight",
    "confirm_output_target",
    "metadata_preflight_from_json",
    "output_confirmation_from_json",
    "recheck_metadata_selection",
    "validate_output_confirmation",
]

_OUTPUT_CONFIRMATION_VERSION = 1
_METADATA_PREFLIGHT_VERSION = 2
_HASH_CHUNK_SIZE = 1024 * 1024

SourcePreflightState: TypeAlias = Literal[
    "unset",
    "resolved",
    "invalid",
    "unavailable",
]
MetadataCompatibilityState: TypeAlias = Literal[
    "absent",
    "pending",
    "compatible",
    "warning",
    "blocked",
]
MetadataChoice: TypeAlias = Literal["omit", "include"]


class RunRequestSafetyError(ValueError):
    """Raised when a Run request cannot be proven safe and current."""


@dataclass(frozen=True)
class OutputConfirmation:
    """Server-issued receipt for the exact typed output target."""

    raw_value: str
    canonical_path: str
    relative_path: str
    sandbox_fingerprint: str
    version: int = _OUTPUT_CONFIRMATION_VERSION

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe confirmation payload."""
        return {
            "version": self.version,
            "raw_value": self.raw_value,
            "canonical_path": self.canonical_path,
            "relative_path": self.relative_path,
            "sandbox_fingerprint": self.sandbox_fingerprint,
        }


@dataclass(frozen=True)
class MetadataPreflight:
    """Visible compatibility snapshot for one source and ambient metadata."""

    source_state: SourcePreflightState
    source_path: str | None
    source_fingerprint: str | None
    source_image_count: int
    metadata_state: MetadataResolutionState
    metadata_path: str | None
    metadata_fingerprint: str | None
    metadata_row_count: int
    join_columns: tuple[str, ...]
    unverified_join_columns: tuple[str, ...]
    matched_source_count: int
    unmatched_source_count: int
    metadata_only_count: int
    duplicate_key_count: int
    compatibility: MetadataCompatibilityState
    warnings: tuple[str, ...]
    request_fingerprint: str
    version: int = _METADATA_PREFLIGHT_VERSION

    @property
    def can_include(self) -> bool:
        """Whether ambient metadata resolves to a current readable CSV."""
        return self.metadata_state == "resolved" and self.metadata_path is not None

    @property
    def requires_acknowledgement(self) -> bool:
        """Whether explicit inclusion requires a warning acknowledgement."""
        return self.can_include and bool(self.warnings)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe preflight payload."""
        return {
            "version": self.version,
            "source_state": self.source_state,
            "source_path": self.source_path,
            "source_fingerprint": self.source_fingerprint,
            "source_image_count": self.source_image_count,
            "metadata_state": self.metadata_state,
            "metadata_path": self.metadata_path,
            "metadata_fingerprint": self.metadata_fingerprint,
            "metadata_row_count": self.metadata_row_count,
            "join_columns": list(self.join_columns),
            "unverified_join_columns": list(self.unverified_join_columns),
            "matched_source_count": self.matched_source_count,
            "unmatched_source_count": self.unmatched_source_count,
            "metadata_only_count": self.metadata_only_count,
            "duplicate_key_count": self.duplicate_key_count,
            "compatibility": self.compatibility,
            "warnings": list(self.warnings),
            "request_fingerprint": self.request_fingerprint,
        }


def _canonical_project_root(project_root: Path | None) -> Path:
    """Return the project root protected from use as a Run output."""
    candidate = Path.cwd() if project_root is None else project_root
    try:
        return candidate.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RunRequestSafetyError("project root cannot be resolved") from exc


def _resolve_typed_output(
    sandbox: SandboxRoot,
    typed_value: object,
    *,
    project_root: Path | None,
) -> tuple[str, Path, str]:
    """Resolve one non-empty typed target and reject protected roots."""
    if not isinstance(typed_value, str) or not typed_value.strip():
        raise RunRequestSafetyError("Type an output directory before confirming")
    raw_value = typed_value.strip()
    try:
        resolved = sandbox.resolve(raw_value)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunRequestSafetyError(
            "Output directory escapes the GUI sandbox or is invalid"
        ) from exc

    if resolved == sandbox.root:
        raise RunRequestSafetyError(
            "The GUI sandbox root cannot be used as a run output"
        )
    if resolved == _canonical_project_root(project_root):
        raise RunRequestSafetyError(
            "The PhenoTypic project root cannot be used as a run output"
        )
    try:
        if resolved.exists() and not resolved.is_dir():
            raise RunRequestSafetyError(
                "Output target exists and is not a directory"
            )
    except (OSError, RuntimeError) as exc:
        raise RunRequestSafetyError("Output target cannot be inspected") from exc
    try:
        relative_path = str(resolved.relative_to(sandbox.root))
    except ValueError as exc:  # pragma: no cover - SandboxRoot already proves this.
        raise RunRequestSafetyError("Output directory escapes the GUI sandbox") from exc
    if not relative_path or relative_path == ".":
        raise RunRequestSafetyError(
            "The GUI sandbox root cannot be used as a run output"
        )
    return raw_value, resolved, relative_path


def confirm_output_target(
    sandbox: SandboxRoot,
    typed_value: object,
    *,
    project_root: Path | None = None,
) -> OutputConfirmation:
    """Confirm exactly the typed output without creating or substituting it.

    Args:
        sandbox: Frozen GUI filesystem boundary.
        typed_value: Current output text field value.
        project_root: Protected project checkout. Defaults to the process
            working directory.

    Returns:
        A receipt preserving the canonical target, including a valid target
        that does not exist yet.

    Raises:
        RunRequestSafetyError: If the target is empty, protected, invalid, or
            outside the sandbox.
    """
    raw_value, resolved, relative_path = _resolve_typed_output(
        sandbox,
        typed_value,
        project_root=project_root,
    )
    return OutputConfirmation(
        raw_value=raw_value,
        canonical_path=str(resolved),
        relative_path=relative_path,
        sandbox_fingerprint=sandbox_fingerprint(sandbox),
    )


def output_confirmation_from_json(payload: object) -> OutputConfirmation:
    """Decode an output receipt without granting it filesystem authority."""
    if not isinstance(payload, dict):
        raise RunRequestSafetyError("Output target has not been confirmed")
    try:
        version = payload["version"]
        raw_value = payload["raw_value"]
        canonical_path = payload["canonical_path"]
        relative_path = payload["relative_path"]
        stored_sandbox = payload["sandbox_fingerprint"]
    except KeyError as exc:
        raise RunRequestSafetyError("Output confirmation is incomplete") from exc
    if version != _OUTPUT_CONFIRMATION_VERSION:
        raise RunRequestSafetyError("Output confirmation version is unsupported")
    values = (raw_value, canonical_path, relative_path, stored_sandbox)
    if not all(isinstance(value, str) and value for value in values):
        raise RunRequestSafetyError("Output confirmation is invalid")
    return OutputConfirmation(
        raw_value=raw_value,
        canonical_path=canonical_path,
        relative_path=relative_path,
        sandbox_fingerprint=stored_sandbox,
        version=version,
    )


def validate_output_confirmation(
    sandbox: SandboxRoot,
    typed_value: object,
    payload: object,
    *,
    project_root: Path | None = None,
) -> Path:
    """Re-resolve the typed target and require an exact current receipt."""
    receipt = output_confirmation_from_json(payload)
    current = confirm_output_target(
        sandbox,
        typed_value,
        project_root=project_root,
    )
    if receipt != current:
        raise RunRequestSafetyError(
            "Output confirmation is stale; confirm the exact typed path again"
        )
    return Path(current.canonical_path)


def _fingerprint_file(path: Path) -> str:
    """Return a content fingerprint for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_snapshot(
    sandbox: SandboxRoot,
    input_dir: object,
) -> tuple[
    SourcePreflightState,
    Path | None,
    str | None,
    tuple[tuple[str, Path], ...],
]:
    """Resolve and fingerprint the exact CLI image inventory."""
    if not isinstance(input_dir, str) or not input_dir.strip():
        return "unset", None, None, ()
    try:
        resolved = sandbox.resolve(input_dir.strip())
    except (OSError, RuntimeError, ValueError):
        return "invalid", None, None, ()
    try:
        if not resolved.is_dir():
            return "unavailable", resolved, None, ()
        datasets = scan_directory_structure(resolved)
        images = tuple(
            sorted(
                (
                    (dataset_name, path)
                    for dataset_name, paths in datasets.items()
                    for path in paths
                ),
                key=lambda item: (
                    item[0],
                    str(item[1].relative_to(resolved)),
                ),
            )
        )
        digest = hashlib.sha256()
        digest.update(str(resolved).encode("utf-8", errors="surrogateescape"))
        for dataset_name, image in images:
            stat = image.stat()
            relative = str(image.relative_to(resolved))
            digest.update(dataset_name.encode("utf-8", errors="surrogateescape"))
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    except (OSError, RuntimeError, ValueError):
        return "unavailable", resolved, None, ()
    return "resolved", resolved, f"sha256:{digest.hexdigest()}", images


def _source_join_key_frame(
    images: tuple[tuple[str, Path], ...],
) -> pl.DataFrame:
    """Project source inventory into keys emitted by CLI aggregation."""
    return pl.DataFrame(
        {
            str(METADATA.IMAGE_NAME): [
                image.stem for _dataset, image in images
            ],
            str(METADATA.SUFFIX): [
                image.suffix for _dataset, image in images
            ],
            str(EXPERIMENT_METADATA.DATASET): [
                dataset for dataset, _image in images
            ],
        }
    )


def _unverified_measurement_join_columns(
    metadata_columns: list[str],
    source_columns: list[str],
) -> tuple[str, ...]:
    """Return metadata columns that may join only after measurement."""
    modules = header_to_module()
    framework_image_headers = set(METADATA.get_headers())
    source_set = set(source_columns)
    return tuple(
        sorted(
            column
            for column in metadata_columns
            if column not in source_set
            and (
                modules.get(column) == REMBI_MODULE.ANALYZED_DATA
                or column in framework_image_headers
            )
        )
    )


def _request_fingerprint(fields: dict[str, object]) -> str:
    """Hash the server-derived source/metadata preflight fields."""
    encoded = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_metadata_preflight(
    sandbox: SandboxRoot,
    input_dir: object,
    metadata_payload: object,
) -> MetadataPreflight:
    """Build one visible, fingerprinted source/metadata compatibility snapshot.

    Args:
        sandbox: Frozen GUI filesystem boundary.
        input_dir: Current Run input-directory control.
        metadata_payload: Ambient shell metadata descriptor.

    Returns:
        A typed snapshot suitable for display and later action-time recheck.
    """
    source_state, source_path, source_hash, images = _source_snapshot(
        sandbox,
        input_dir,
    )
    metadata_resolution = resolve_metadata_csv_state(sandbox, metadata_payload)
    metadata_path = metadata_resolution.path
    metadata_hash: str | None = None
    metadata_rows = 0
    join_columns: tuple[str, ...] = ()
    unverified_join_columns: tuple[str, ...] = ()
    matched_source = 0
    unmatched_source = len(images)
    metadata_only = 0
    duplicate_keys = 0
    warnings: list[str] = []

    if metadata_resolution.state == "unset":
        compatibility: MetadataCompatibilityState = "absent"
    elif metadata_path is None:
        compatibility = "blocked"
        warnings.append(
            f"Ambient metadata descriptor is {metadata_resolution.state}."
        )
    else:
        try:
            metadata_hash = _fingerprint_file(metadata_path)
            metadata_frame = pl.read_csv(metadata_path)
            metadata_rows = metadata_frame.height
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
            pl.exceptions.PolarsError,
        ):
            compatibility = "blocked"
            warnings.append("Ambient metadata CSV cannot be read.")
        else:
            if source_state != "resolved":
                compatibility = "pending"
                warnings.insert(
                    0,
                    "Select a valid image source to check metadata compatibility.",
                )
            else:
                source_frame = _source_join_key_frame(images)
                prepared = prepare_metadata_join_keys(
                    source_frame,
                    metadata_frame,
                )
                analysis = prepared.analysis
                join_columns = analysis.columns
                unverified_join_columns = (
                    _unverified_measurement_join_columns(
                        metadata_frame.columns,
                        source_frame.columns,
                    )
                )
                matched_source = analysis.matched_measurement_count
                unmatched_source = analysis.unmatched_measurement_count
                metadata_only = analysis.unmatched_metadata_count
                duplicate_keys = analysis.duplicate_metadata_key_count
                if not join_columns:
                    warnings.append(
                        "Metadata shares no source-derived CLI join keys; "
                        "the production join may skip it."
                    )
                if unverified_join_columns:
                    warnings.append(
                        "Metadata also contains measurement-level columns "
                        "that the preflight cannot verify: "
                        + ", ".join(unverified_join_columns)
                        + ". The production join will use any of these columns "
                        "that measurements emit."
                    )
                if duplicate_keys:
                    warnings.append(
                        f"{duplicate_keys} duplicate metadata key row(s) on "
                        f"{', '.join(join_columns)} may fan out joined rows."
                    )
                if unmatched_source:
                    warnings.append(
                        f"{unmatched_source}/{len(images)} input images have "
                        "no metadata match on all preflight join keys and may "
                        "be dropped."
                    )
                if metadata_only:
                    warnings.append(
                        f"{metadata_only}/{metadata_rows} metadata rows do not "
                        "match an input image on all preflight join keys and "
                        "may become metadata-only rows."
                    )
                compatibility = "warning" if warnings else "compatible"

    fingerprint_fields: dict[str, object] = {
        "sandbox_fingerprint": sandbox_fingerprint(sandbox),
        "source_state": source_state,
        "source_path": str(source_path) if source_path is not None else None,
        "source_fingerprint": source_hash,
        "source_image_count": len(images),
        "metadata_state": metadata_resolution.state,
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "metadata_fingerprint": metadata_hash,
        "metadata_row_count": metadata_rows,
        "join_columns": join_columns,
        "unverified_join_columns": unverified_join_columns,
        "matched_source_count": matched_source,
        "unmatched_source_count": unmatched_source,
        "metadata_only_count": metadata_only,
        "duplicate_key_count": duplicate_keys,
        "compatibility": compatibility,
        "warnings": warnings,
    }
    return MetadataPreflight(
        source_state=source_state,
        source_path=str(source_path) if source_path is not None else None,
        source_fingerprint=source_hash,
        source_image_count=len(images),
        metadata_state=metadata_resolution.state,
        metadata_path=str(metadata_path) if metadata_path is not None else None,
        metadata_fingerprint=metadata_hash,
        metadata_row_count=metadata_rows,
        join_columns=join_columns,
        unverified_join_columns=unverified_join_columns,
        matched_source_count=matched_source,
        unmatched_source_count=unmatched_source,
        metadata_only_count=metadata_only,
        duplicate_key_count=duplicate_keys,
        compatibility=compatibility,
        warnings=tuple(warnings),
        request_fingerprint=_request_fingerprint(fingerprint_fields),
    )


def metadata_preflight_from_json(payload: object) -> MetadataPreflight:
    """Decode a preflight snapshot without trusting its filesystem claims."""
    if not isinstance(payload, dict):
        raise RunRequestSafetyError("Metadata preflight is unavailable")
    try:
        version = payload["version"]
        source_state = payload["source_state"]
        source_path = payload["source_path"]
        source_hash = payload["source_fingerprint"]
        source_count = payload["source_image_count"]
        metadata_state = payload["metadata_state"]
        metadata_path = payload["metadata_path"]
        metadata_hash = payload["metadata_fingerprint"]
        metadata_rows = payload["metadata_row_count"]
        join_column_values = payload["join_columns"]
        unverified_join_column_values = payload["unverified_join_columns"]
        matched = payload["matched_source_count"]
        unmatched = payload["unmatched_source_count"]
        metadata_only = payload["metadata_only_count"]
        duplicates = payload["duplicate_key_count"]
        compatibility = payload["compatibility"]
        warning_values = payload["warnings"]
        request_hash = payload["request_fingerprint"]
    except KeyError as exc:
        raise RunRequestSafetyError("Metadata preflight is incomplete") from exc
    if version != _METADATA_PREFLIGHT_VERSION:
        raise RunRequestSafetyError("Metadata preflight version is unsupported")
    if source_state not in {"unset", "resolved", "invalid", "unavailable"}:
        raise RunRequestSafetyError("Metadata preflight source state is invalid")
    if metadata_state not in {
        "unset",
        "resolved",
        "invalid",
        "unavailable",
        "fingerprint_mismatch",
    }:
        raise RunRequestSafetyError("Metadata descriptor state is invalid")
    if compatibility not in {
        "absent",
        "pending",
        "compatible",
        "warning",
        "blocked",
    }:
        raise RunRequestSafetyError("Metadata compatibility state is invalid")
    if not isinstance(warning_values, list) or not all(
        isinstance(item, str) for item in warning_values
    ):
        raise RunRequestSafetyError("Metadata preflight warnings are invalid")
    if not all(
        isinstance(values, list)
        and all(isinstance(item, str) for item in values)
        for values in (join_column_values, unverified_join_column_values)
    ):
        raise RunRequestSafetyError("Metadata preflight join columns are invalid")
    counts = (source_count, metadata_rows, matched, unmatched, metadata_only, duplicates)
    if not all(type(value) is int and value >= 0 for value in counts):
        raise RunRequestSafetyError("Metadata preflight counts are invalid")
    optional_strings = (
        source_path,
        source_hash,
        metadata_path,
        metadata_hash,
    )
    if not all(
        value is None or isinstance(value, str) for value in optional_strings
    ):
        raise RunRequestSafetyError("Metadata preflight fields are invalid")
    if not isinstance(request_hash, str) or not request_hash:
        raise RunRequestSafetyError("Metadata preflight fingerprint is invalid")
    return MetadataPreflight(
        source_state=source_state,
        source_path=source_path,
        source_fingerprint=source_hash,
        source_image_count=source_count,
        metadata_state=metadata_state,
        metadata_path=metadata_path,
        metadata_fingerprint=metadata_hash,
        metadata_row_count=metadata_rows,
        join_columns=tuple(join_column_values),
        unverified_join_columns=tuple(unverified_join_column_values),
        matched_source_count=matched,
        unmatched_source_count=unmatched,
        metadata_only_count=metadata_only,
        duplicate_key_count=duplicates,
        compatibility=compatibility,
        warnings=tuple(warning_values),
        request_fingerprint=request_hash,
        version=version,
    )


def recheck_metadata_selection(
    sandbox: SandboxRoot,
    *,
    input_dir: object,
    metadata_payload: object,
    choice: object,
    acknowledgement: object,
    preflight_payload: object,
) -> Path | None:
    """Recompute fingerprints and resolve an explicit metadata choice.

    Args:
        sandbox: Frozen GUI filesystem boundary.
        input_dir: Current Run input-directory control.
        metadata_payload: Current ambient shell metadata descriptor.
        choice: Explicit ``"omit"`` or ``"include"`` control value.
        acknowledgement: Checklist values containing ``"acknowledge"`` when
            the user accepted compatibility warnings.
        preflight_payload: Previously displayed preflight snapshot.

    Returns:
        The current metadata CSV path when explicitly included, otherwise
        ``None``.

    Raises:
        RunRequestSafetyError: If the visible snapshot is stale, inclusion is
            unavailable, or a required warning was not acknowledged.
    """
    if choice not in {"omit", "include"}:
        raise RunRequestSafetyError("Choose whether to omit or include metadata")
    displayed = metadata_preflight_from_json(preflight_payload)
    current = build_metadata_preflight(sandbox, input_dir, metadata_payload)
    if displayed.request_fingerprint != current.request_fingerprint:
        raise RunRequestSafetyError(
            "Source or metadata changed after preflight; reselect it to "
            "refresh the metadata preflight before continuing"
        )
    if choice == "omit":
        return None
    if not current.can_include or current.metadata_path is None:
        raise RunRequestSafetyError(
            "Ambient metadata is unavailable and cannot be included"
        )
    acknowledged = isinstance(acknowledgement, (list, tuple, set)) and (
        "acknowledge" in acknowledgement
    )
    if current.requires_acknowledgement and not acknowledged:
        raise RunRequestSafetyError(
            "Acknowledge the metadata compatibility warning before inclusion"
        )
    return Path(current.metadata_path)
