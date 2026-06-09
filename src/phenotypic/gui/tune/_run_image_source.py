"""Run image-source resolution for Tune Deploy."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from phenotypic.gui.shell._source_context import resolve_source_image_root

if TYPE_CHECKING:
    from phenotypic.gui.shell._sandbox import SandboxRoot


def resolve_run_images(
    sandbox: "SandboxRoot",
    shared_payload: object,
    override: str | None,
) -> str | None:
    """Resolve the image directory for a tune run.

    Args:
        sandbox: Frozen GUI sandbox.
        shared_payload: Browser-local shared source-image-root payload.
        override: Optional Run-form override path.

    Returns:
        The absolute in-sandbox image path, or ``None`` when unset/invalid.
    """
    if override:
        try:
            candidate = sandbox.resolve(Path(override).expanduser())
        except ValueError:
            return None
        if candidate.is_dir():
            return str(candidate)
        return None
    resolved = resolve_source_image_root(sandbox, shared_payload)
    return str(resolved) if resolved is not None else None


__all__ = ["resolve_run_images"]
