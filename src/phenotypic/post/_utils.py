from __future__ import annotations

_PREFIX = "Metadata_"


def _ensure_prefix(name: str) -> str:
    """Prepend Metadata_ if not already present."""
    return name if name.startswith(_PREFIX) else f"{_PREFIX}{name}"
