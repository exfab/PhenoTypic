"""Process-wide runtime settings for PhenoTypic.

This module is intentionally small. It owns runtime switches that affect
debugging or validation behavior across the current Python process. Algorithm
defaults, operation parameters, output paths, CLI defaults, and GUI design
tokens belong in their local modules instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

VALIDATE_OPS: bool = False
"""Whether operation and measurement integrity decorators validate arrays."""


def set_validate_ops(enabled: bool) -> None:
    """Set operation integrity validation for the current process.

    Args:
        enabled: Whether operation and measurement integrity decorators should
            validate protected arrays.
    """
    global VALIDATE_OPS
    VALIDATE_OPS = bool(enabled)


@contextmanager
def validation(enabled: bool) -> Iterator[None]:
    """Temporarily set operation integrity validation.

    Args:
        enabled: Temporary validation state for the context body.

    Yields:
        None.
    """
    previous = VALIDATE_OPS
    set_validate_ops(enabled)
    try:
        yield
    finally:
        set_validate_ops(previous)
