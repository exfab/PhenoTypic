"""
Shared utility functions for the PhenoTypic CLI.

Common functions used across multiple CLI modules to ensure consistent
behavior and validation across the command-line interface and SLURM
processing scripts.

Examples:
    >>> from phenotypic._cli._cli_utils import normalize_extension
    >>>
    >>> # Validate and normalize extension
    >>> ext = normalize_extension("png")  # Returns ".png"
    >>> ext = normalize_extension(".TIFF")  # Returns ".tiff"
    >>> ext = normalize_extension("")  # Returns ".tiff" (default)
    >>>
    >>> # Invalid extension raises error
    >>> ext = normalize_extension("exe")  # Raises click.BadParameter
"""

from __future__ import annotations

import click
from typing import Set

# Allowed image file extensions for PhenoTypic processing
ALLOWED_EXTENSIONS: Set[str] = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def normalize_extension(ext: str, default: str = ".tiff") -> str:
    """
    Normalize and validate file extension.

    Ensures extension has leading dot, is lowercase, and is a supported
    image format for PhenoTypic processing. This prevents invalid extensions
    from causing failures during file save operations.

    Args:
        ext: Extension string (with or without leading dot)
        default: Default extension if ext is empty (default: ".tiff")

    Returns:
        Normalized extension with leading dot in lowercase

    Raises:
        click.BadParameter: If extension is not in allowed list

    Examples:
        >>> normalize_extension("png")
        '.png'
        >>> normalize_extension(".TIFF")
        '.tiff'
        >>> normalize_extension("")
        '.tiff'
        >>> normalize_extension("exe")  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        click.exceptions.BadParameter: Unsupported extension '.exe'. Allowed: .jpg, .jpeg, .png, .tif, .tiff
    """
    if not ext:
        ext = default

    ext = ext.lower().strip()

    if not ext.startswith("."):
        ext = f".{ext}"

    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise click.BadParameter(
                f"Unsupported extension '{ext}'. Allowed: {allowed_str}"
        )

    return ext
