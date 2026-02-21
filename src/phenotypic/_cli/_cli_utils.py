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

import shutil
from typing import List, Set, Tuple

import click

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


def parse_slurm_args(slurm_args: "Sequence[str]") -> dict:
    """Parse space-separated KEY=VALUE pairs into dictionary.

    Args:
        slurm_args: Sequence of "KEY=VALUE" strings.

    Returns:
        Dictionary of parsed parameters.

    Raises:
        click.BadParameter: If parsing fails.
    """
    import ast

    parsed = {}
    for param in slurm_args:
        if "=" not in param:
            raise click.BadParameter(
                "--slurm must be KEY=VALUE pairs",
                param_hint="--slurm",
            )

        key, value = param.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise click.BadParameter(
                "SLURM parameter keys cannot be empty",
                param_hint="--slurm",
            )

        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value

        parsed[key] = parsed_value

    return parsed


def get_python_command(for_slurm: bool = False) -> Tuple[List[str], str]:
    """
    Detect available Python runner command for SLURM scripts.

    Checks if uv is available and returns the appropriate command parts
    for invoking Python in generated SLURM scripts. When uv is available,
    uses 'uv run python' to ensure the correct virtual environment and
    project context are used on worker nodes.

    Args:
        for_slurm: When True, return the direct venv Python interpreter
            path (``sys.executable``) instead of ``uv run python``.
            This avoids ``uv`` resolution overhead on SLURM worker nodes
            where the venv is already activated.

    Returns:
        Tuple of (command_parts, description) where:
        - command_parts: List of command strings (e.g., ["uv", "run", "python"])
        - description: Human-readable description for logging/display

    Examples:
        >>> cmd_parts, desc = get_python_command()
        >>> len(cmd_parts) >= 1
        True
    """
    if for_slurm:
        import sys

        return ([sys.executable], f"{sys.executable} (direct venv)")
    if shutil.which("uv"):
        return (["uv", "run", "python"], "uv run python (project environment)")
    return (["python"], "python (system)")
