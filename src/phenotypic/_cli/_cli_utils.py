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

import io
import json
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click

logger = logging.getLogger(__name__)

# Allowed image file extensions for PhenoTypic processing
ALLOWED_EXTENSIONS: Set[str] = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

# Thread-pinning snippet for SLURM bash scripts. Ensures Polars and NumPy
# respect the SLURM CPU allocation. Must appear before any Python import.
SLURM_THREAD_PIN_BASH = """\
# Pin Polars/NumPy thread pools to SLURM allocation (must happen before import)
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export POLARS_SKIP_CPU_CHECK=1"""


# Minimum file count to justify tar subprocess overhead.
_TAR_MIN_FILES = 8


def load_job_metadata(progress_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ``job_metadata.json`` from a progress directory.

    Args:
        progress_dir: Directory containing ``job_metadata.json``.

    Returns:
        Parsed dict, or ``None`` if the file is missing or unreadable.
    """
    meta_path = progress_dir / "job_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.warning("Failed to read job_metadata.json", exc_info=True)
        return None


def scan_parquets(
    parquet_files: List[Path],
) -> Dict[Path, "pl.LazyFrame"]:
    """Create lazy Parquet scans, using ``tar`` streaming on HPC filesystems.

    **Direct path** (< ``_TAR_MIN_FILES`` or no ``tar``):
    Returns ``pl.scan_parquet()`` lazy frames — no I/O occurs until the
    caller calls ``.collect()`` on the concatenated result.  This lets
    Polars optimise the full query plan in Rust.

    **Tar path** (>= ``_TAR_MIN_FILES`` with ``tar`` on ``$PATH``):
    Streams file bytes through ``tar cf -`` to reduce per-file metadata
    overhead on shared cluster filesystems (NFS, Lustre, GPFS), then
    wraps the eagerly-parsed DataFrames as ``.lazy()`` so callers can
    use a uniform lazy API.

    Args:
        parquet_files: Paths to Parquet files.

    Returns:
        Ordered dict mapping each original *Path* to a lazy frame.
        Files that could not be scanned/read are omitted.
    """
    import polars as pl

    if not parquet_files:
        return {}

    if len(parquet_files) >= _TAR_MIN_FILES and shutil.which("tar"):
        result = _read_via_tar(parquet_files)
        if result is not None:
            return result

    return _scan_direct(parquet_files)


def _scan_direct(parquet_files: List[Path]) -> Dict[Path, "pl.LazyFrame"]:
    """Create lazy scans per file (no I/O until collect)."""
    import polars as pl

    out: Dict[Path, pl.LazyFrame] = {}
    for p in parquet_files:
        try:
            out[p] = pl.scan_parquet(p)
        except Exception as exc:
            logger.warning("Failed to scan %s: %s", p, exc)
    return out


def _read_via_tar(
    parquet_files: List[Path],
) -> Optional[Dict[Path, "pl.LazyFrame"]]:
    """Stream file bytes through ``tar cf -``, parse, and wrap as lazy frames.

    Passes file paths via stdin (``tar -T -``) to avoid hitting OS
    ``ARG_MAX`` limits on large HPC runs with thousands of images.

    Returns ``None`` on subprocess failure so the caller can fall back.
    """
    import polars as pl

    # Build a lookup from the absolute string (as tar stores it, minus
    # the leading ``/``) back to the original Path object.
    abs_to_path: Dict[str, Path] = {}
    file_list_bytes: List[str] = []
    for p in parquet_files:
        abs_str = str(p.absolute())
        file_list_bytes.append(abs_str)
        # GNU tar strips the leading ``/``; store both forms for safety.
        abs_to_path[abs_str] = p
        abs_to_path[abs_str.lstrip("/")] = p

    # Feed paths via stdin (-T -) to avoid ARG_MAX limits.
    proc = subprocess.Popen(
        ["tar", "cf", "-", "-T", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None or proc.stdin is None:  # type narrowing
        proc.wait()
        return None

    # Write file list to stdin, then close to signal EOF.
    try:
        proc.stdin.write("\n".join(file_list_bytes).encode())
        proc.stdin.close()
    except OSError as exc:
        logger.warning("Failed to write file list to tar stdin: %s", exc)
        proc.stdout.close()
        proc.wait()
        return None

    out: Dict[Path, pl.LazyFrame] = {}
    try:
        with tarfile.open(mode="r|", fileobj=proc.stdout) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                original = abs_to_path.get(member.name)
                if original is None:
                    original = abs_to_path.get(member.name.lstrip("/"))
                if original is None:
                    logger.debug("Unmatched tar member: %s", member.name)
                    continue
                try:
                    df = pl.read_parquet(io.BytesIO(extracted.read()))
                    out[original] = df.lazy()
                except Exception as exc:
                    logger.warning("Failed to parse %s from tar stream: %s", member.name, exc)
    except Exception as exc:
        logger.warning("tar streaming failed (%s), falling back to lazy scan", exc)
        proc.stdout.close()
        proc.wait()
        return None

    proc.stdout.close()
    stderr_out = proc.stderr.read() if proc.stderr else b""
    proc.stderr.close() if proc.stderr else None
    rc = proc.wait()
    if stderr_out:
        logger.debug("tar stderr: %s", stderr_out.decode(errors="replace").strip())
    if rc != 0:
        logger.warning("tar exited with code %d, falling back to lazy scan", rc)
        return None

    return out


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

    Checks if pixi is available and returns the appropriate command parts
    for invoking Python in generated SLURM scripts. When pixi is available,
    uses 'pixi run python' to ensure the correct virtual environment and
    project context are used on worker nodes.

    Args:
        for_slurm: When True, return the direct venv Python interpreter
            path (``sys.executable``) instead of ``pixi run python``.
            This avoids ``pixi`` resolution overhead on SLURM worker nodes
            where the venv is already activated.

    Returns:
        Tuple of (command_parts, description) where:
        - command_parts: List of command strings (e.g., ["pixi", "run", "python"])
        - description: Human-readable description for logging/display

    Examples:
        >>> cmd_parts, desc = get_python_command()
        >>> len(cmd_parts) >= 1
        True
    """
    if for_slurm:
        import sys

        return ([sys.executable], f"{sys.executable} (direct venv)")
    if shutil.which("pixi"):
        return (["pixi", "run", "python"], "pixi run python (project environment)")
    return (["python"], "python (system)")
