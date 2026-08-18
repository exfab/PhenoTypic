"""Shared source-image-root context for the GUI shell.

The source context is a small browser-store payload naming the directory of
source plate images. Browser storage is transport only; all filesystem consumers
must resolve payloads through :func:`resolve_source_image_root` before use.

Version 2 binds a selection to the sandbox in which the user made it. The
absolute path stored at selection time is diagnostic only; resolution uses the
sandbox-relative path after verifying the sandbox fingerprint. Version 1
payloads remain readable, but are never silently rewritten or rebound to a
different sandbox.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from phenotypic._services.sandbox import (  # noqa: F401 - re-exported
    sandbox_fingerprint,
)
from phenotypic.gui.shell._classifier import classify
from phenotypic.gui.shell._sandbox import (
    SandboxRoot,
    _is_safe_relative_path,
    _v1_selection_matches_sandbox,
)

logger = logging.getLogger(__name__)

SourceOrigin: TypeAlias = Literal[
    "sidebar",
    "run-console",
    "tune",
    "builder",
    "manual",
    "unknown",
]
SourceResolutionState: TypeAlias = Literal[
    "unset",
    "resolved",
    "invalid",
    "unavailable",
    "fingerprint_mismatch",
]

SOURCE_PAYLOAD_VERSION = 2
_SOURCE_PAYLOAD_V1 = 1
_SOURCE_KIND: Literal["image_source"] = "image_source"
_UNAVAILABLE_LABEL = "Previous source unavailable in this sandbox"


class SourceValidation(TypedDict):
    """Filesystem facts recorded when a source is explicitly selected."""

    exists: bool
    is_directory: bool


class SourcePayload(TypedDict):
    """Version 2 JSON payload for the shared source-image-root store.

    ``abs_path``, ``rel_path``, ``label``, and ``validated`` are compatibility
    mirrors for consumers that are outside the path-state migration scope.
    They are never authoritative during resolution.
    """

    version: int
    kind: Literal["image_source"]
    relative_path: str
    absolute_path_at_selection: str
    sandbox_fingerprint: str
    validation: SourceValidation
    selected_at: str
    source: SourceOrigin
    image_count: int | None
    abs_path: str
    rel_path: str
    label: str
    validated: bool


@dataclass(frozen=True)
class SourceResolution:
    """Typed result of resolving a browser source payload."""

    state: SourceResolutionState
    path: Path | None = None
    payload_version: int | None = None

    @property
    def is_resolved(self) -> bool:
        """Whether the payload resolves to a current sandbox directory."""
        return self.state == "resolved"


def source_payload_from_path(
    sandbox: SandboxRoot,
    path: Path | str,
    *,
    source: SourceOrigin,
) -> SourcePayload | None:
    """Return a version 2 source payload after explicit selection.

    Args:
        sandbox: Frozen-at-launch sandbox boundary.
        path: Candidate source image directory.
        source: Page or flow that set the value.

    Returns:
        Versioned payload for browser storage, or ``None`` when the candidate
        escapes the sandbox or is not an existing directory.
    """
    try:
        resolved = sandbox.resolve(path)
    except (OSError, RuntimeError, ValueError):
        logger.warning("source image root is not resolvable: %r", path)
        return None
    try:
        is_directory = resolved.is_dir()
    except (OSError, RuntimeError):
        logger.warning("source image root stat failed: %s", resolved)
        return None
    if not is_directory:
        logger.warning("source image root is not a directory: %s", resolved)
        return None

    try:
        rel_path = str(resolved.relative_to(sandbox.root))
    except ValueError:
        logger.warning("source image root resolved outside sandbox: %s", resolved)
        return None
    relative_path = "." if rel_path == "" else rel_path
    label = resolved.name or str(resolved)
    caps = classify(resolved)
    payload: SourcePayload = {
        "version": SOURCE_PAYLOAD_VERSION,
        "kind": _SOURCE_KIND,
        "relative_path": relative_path,
        "absolute_path_at_selection": str(resolved),
        "sandbox_fingerprint": sandbox_fingerprint(sandbox),
        "validation": {"exists": True, "is_directory": True},
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "image_count": caps.image_count,
        # Compatibility mirrors. Resolution never trusts these fields.
        "abs_path": str(resolved),
        "rel_path": relative_path,
        "label": label,
        "validated": True,
    }
    return payload


def resolve_source_image_root_state(
    sandbox: SandboxRoot,
    payload: object,
) -> SourceResolution:
    """Return the typed current-sandbox resolution of ``payload``."""
    if payload is None:
        return SourceResolution("unset")
    if not isinstance(payload, dict):
        return SourceResolution("invalid")

    version = payload.get("version")
    if version == SOURCE_PAYLOAD_VERSION:
        return _resolve_v2_source(sandbox, payload)
    if version == _SOURCE_PAYLOAD_V1:
        return _resolve_v1_source(sandbox, payload)
    return SourceResolution(
        "invalid",
        payload_version=version if type(version) is int else None,
    )


def resolve_source_image_root(
    sandbox: SandboxRoot,
    payload: object,
) -> Path | None:
    """Compatibility wrapper returning only a resolved source directory.

    Use :func:`resolve_source_image_root_state` when the caller needs to
    distinguish malformed, unavailable, and sandbox-mismatched selections.
    """
    return resolve_source_image_root_state(sandbox, payload).path


def source_label(
    payload: object,
    *,
    sandbox: SandboxRoot | None = None,
) -> str:
    """Return the compact label shown in the shell top bar.

    Passing ``sandbox`` enables authoritative V1/V2 resolution and unavailable
    messaging. Omitting it preserves the historical payload-only formatting
    API for compatibility with non-filesystem callers.
    """
    if sandbox is not None:
        resolution = resolve_source_image_root_state(sandbox, payload)
        if resolution.state == "unset":
            return "source: unset"
        if resolution.path is not None:
            return f"source: {resolution.path.name or resolution.path}"
        if resolution.state in {"unavailable", "fingerprint_mismatch"}:
            return _UNAVAILABLE_LABEL
        return "source: invalid"
    return _legacy_source_label(payload)


def source_title(
    payload: object,
    *,
    sandbox: SandboxRoot | None = None,
) -> str:
    """Return the hover title for the shell source label.

    A stale absolute path is not presented as the active source. It remains in
    the V2 payload solely for diagnostics.
    """
    if sandbox is not None:
        resolution = resolve_source_image_root_state(sandbox, payload)
        if resolution.path is not None:
            return str(resolution.path)
        if resolution.state in {"unavailable", "fingerprint_mismatch"}:
            return _UNAVAILABLE_LABEL
        return "No source image root selected"
    if not isinstance(payload, dict):
        return "No source image root selected"
    path = payload.get("absolute_path_at_selection", payload.get("abs_path"))
    if not isinstance(path, str) or not path:
        return "No source image root selected"
    return path


def _resolve_v2_source(
    sandbox: SandboxRoot,
    payload: dict[object, object],
) -> SourceResolution:
    if payload.get("kind") != _SOURCE_KIND:
        return SourceResolution("invalid", payload_version=SOURCE_PAYLOAD_VERSION)
    stored_fingerprint = payload.get("sandbox_fingerprint")
    relative_path = payload.get("relative_path")
    if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
        return SourceResolution("invalid", payload_version=SOURCE_PAYLOAD_VERSION)
    if not isinstance(relative_path, str) or not _is_safe_relative_path(
        relative_path
    ):
        return SourceResolution("invalid", payload_version=SOURCE_PAYLOAD_VERSION)
    if stored_fingerprint != sandbox_fingerprint(sandbox):
        return SourceResolution(
            "fingerprint_mismatch",
            payload_version=SOURCE_PAYLOAD_VERSION,
        )
    return _resolve_current_source_path(
        sandbox,
        relative_path,
        payload_version=SOURCE_PAYLOAD_VERSION,
    )


def _resolve_v1_source(
    sandbox: SandboxRoot,
    payload: dict[object, object],
) -> SourceResolution:
    raw_path = payload.get("abs_path")
    relative_path = payload.get("rel_path")
    if not isinstance(raw_path, str) or not raw_path:
        return SourceResolution("invalid", payload_version=_SOURCE_PAYLOAD_V1)
    if not isinstance(relative_path, str) or not _is_safe_relative_path(
        relative_path
    ):
        return SourceResolution("invalid", payload_version=_SOURCE_PAYLOAD_V1)
    if not _v1_selection_matches_sandbox(
        sandbox,
        raw_path=raw_path,
        relative_path=relative_path,
    ):
        return SourceResolution(
            "fingerprint_mismatch",
            payload_version=_SOURCE_PAYLOAD_V1,
        )
    return _resolve_current_source_path(
        sandbox,
        relative_path,
        payload_version=_SOURCE_PAYLOAD_V1,
    )


def _resolve_current_source_path(
    sandbox: SandboxRoot,
    relative_path: str,
    *,
    payload_version: int,
) -> SourceResolution:
    try:
        resolved = sandbox.resolve(relative_path)
        is_directory = resolved.is_dir()
    except (OSError, RuntimeError, ValueError):
        return SourceResolution("unavailable", payload_version=payload_version)
    if not is_directory:
        return SourceResolution("unavailable", payload_version=payload_version)
    return SourceResolution(
        "resolved",
        path=resolved,
        payload_version=payload_version,
    )


def _legacy_source_label(payload: object) -> str:
    if payload is None:
        return "source: unset"
    if not isinstance(payload, dict):
        return "source: invalid"
    version = payload.get("version")
    if version not in {_SOURCE_PAYLOAD_V1, SOURCE_PAYLOAD_VERSION}:
        return "source: invalid"
    if version == SOURCE_PAYLOAD_VERSION and payload.get("kind") != _SOURCE_KIND:
        return "source: invalid"
    label = payload.get("label")
    if not isinstance(label, str) or not label:
        return "source: invalid"
    return f"source: {label}"


__all__ = [
    "SOURCE_PAYLOAD_VERSION",
    "SourceOrigin",
    "SourcePayload",
    "SourceResolution",
    "SourceResolutionState",
    "resolve_source_image_root",
    "resolve_source_image_root_state",
    "sandbox_fingerprint",
    "source_label",
    "source_payload_from_path",
    "source_title",
]
