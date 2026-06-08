"""Shared source-image-root context for the GUI shell.

The source context is a small browser-store payload naming the directory of
source plate images. Browser storage is transport only; all filesystem consumers
must resolve payloads through :func:`resolve_source_image_root` before use.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from phenotypic.gui.shell._classifier import classify
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

SourceOrigin: TypeAlias = Literal[
    "sidebar",
    "run-console",
    "tune",
    "builder",
    "manual",
    "unknown",
]

SOURCE_PAYLOAD_VERSION = 1


class SourcePayload(TypedDict):
    """Versioned JSON payload for the shared source-image-root store."""

    abs_path: str
    rel_path: str
    label: str
    image_count: int | None
    source: SourceOrigin
    validated: bool
    version: int


def source_payload_from_path(
    sandbox: SandboxRoot,
    path: Path | str,
    *,
    source: SourceOrigin,
) -> SourcePayload | None:
    """Return a validated source-image-root payload for ``path``.

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
    except ValueError:
        logger.warning("source image root escapes sandbox: %r", path)
        return None
    if not resolved.is_dir():
        logger.warning("source image root is not a directory: %s", resolved)
        return None

    try:
        rel_path = str(resolved.relative_to(sandbox.root))
    except ValueError:
        logger.warning("source image root resolved outside sandbox: %s", resolved)
        return None
    label = resolved.name or str(resolved)
    caps = classify(resolved)
    return {
        "abs_path": str(resolved),
        "rel_path": "." if rel_path == "" else rel_path,
        "label": label,
        "image_count": caps.image_count,
        "source": source,
        "validated": True,
        "version": SOURCE_PAYLOAD_VERSION,
    }


def resolve_source_image_root(
    sandbox: SandboxRoot,
    payload: object,
) -> Path | None:
    """Return a sandbox-contained source image directory from ``payload``.

    Args:
        sandbox: Frozen-at-launch sandbox boundary.
        payload: Value read from ``SHELL_SOURCE_IMAGE_ROOT_STORE``.

    Returns:
        Resolved directory path, or ``None`` when the payload is malformed,
        stale, outside the sandbox, or not a directory.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != SOURCE_PAYLOAD_VERSION:
        return None
    if payload.get("validated") is not True:
        return None
    raw_path = payload.get("abs_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    try:
        resolved = sandbox.resolve(raw_path)
    except ValueError:
        logger.warning("stored source image root escapes sandbox: %r", raw_path)
        return None
    if not resolved.is_dir():
        logger.warning("stored source image root is not a directory: %s", resolved)
        return None
    return resolved


def source_label(payload: object) -> str:
    """Return the compact label shown in the shell top bar."""
    if payload is None:
        return "source: unset"
    if not isinstance(payload, dict):
        return "source: invalid"
    if payload.get("version") != SOURCE_PAYLOAD_VERSION:
        return "source: invalid"
    if payload.get("validated") is not True:
        return "source: invalid"
    label = payload.get("label")
    if not isinstance(label, str) or not label:
        return "source: invalid"
    return f"source: {label}"


def source_title(payload: object) -> str:
    """Return the full-path hover title for the shell source label."""
    if not isinstance(payload, dict):
        return "No source image root selected"
    path = payload.get("abs_path")
    if not isinstance(path, str) or not path:
        return "No source image root selected"
    return path


__all__ = [
    "SourceOrigin",
    "SourcePayload",
    "SOURCE_PAYLOAD_VERSION",
    "resolve_source_image_root",
    "source_label",
    "source_payload_from_path",
    "source_title",
]
