"""Shared helpers for seeding fake CLI output directories in tests.

A PhenoTypic CLI run writes its user-facing deliverables under
``<output>/deliverables/`` (``master_measurements.*``, the
``measurements.*`` mirror, ``analysis.*``, per-feature splits,
``pipeline.json``, ``README.md``, ``dashboard.html``,
``analysis.html``, ``processing_report.html``). Per-image artifacts
(``results/<ds>/...``), QC outputs (``qc/``), progress sidecars
(``progress/``), and run state (``processing_state.json``) stay at the
output root.

Tests that synthesize one of these layouts should route through these
helpers (which compose from the production path-builders in
``phenotypic.tools_``) so the on-disk layout auto-tracks any future
relocation of the deliverables folder. Never hard-code
``tmp_path / "master_measurements.parquet"`` — it will silently drift.

The helpers accept either a polars or a pandas frame for the master.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from phenotypic.tools_ import (
    deliverables_dir,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
    pipeline_json_path,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl


def _ensure_deliverables(root: Path) -> Path:
    """Create ``<root>/deliverables/`` (the folder won't exist yet) and return it."""
    deliv = deliverables_dir(root)
    deliv.mkdir(parents=True, exist_ok=True)
    return deliv


def _to_polars(df: Any) -> "pl.DataFrame":
    """Coerce a pandas or polars frame to polars (for ``.write_parquet``/``.write_csv``)."""
    import polars as pl

    if isinstance(df, pl.DataFrame):
        return df
    # Assume a pandas frame (or anything polars can ingest from pandas).
    return pl.from_pandas(df)


def write_master(root: Path, master: Any, *, csv: bool = True, parquet: bool = True) -> Path:
    """Write ``master_measurements.{csv,parquet}`` under ``<root>/deliverables/``.

    Args:
        root: Run output root (``tmp_path``).
        master: A polars or pandas DataFrame.
        csv: Write the CSV master archive.
        parquet: Write the parquet master archive.

    Returns:
        The deliverables directory.
    """
    deliv = _ensure_deliverables(root)
    frame = _to_polars(master)
    if csv:
        frame.write_csv(master_measurements_csv_path(root))
    if parquet:
        frame.write_parquet(master_measurements_parquet_path(root))
    return deliv


def write_measurements_mirror(
    root: Path, df: Any, *, csv: bool = True, parquet: bool = True
) -> Path:
    """Write the post-applied ``measurements.{csv,parquet}`` mirror.

    This is the frame the GUI viewer reads/curates (see CLAUDE.md
    "Master vs. mirror outputs"). Lives under ``<root>/deliverables/``.
    """
    deliv = _ensure_deliverables(root)
    frame = _to_polars(df)
    if csv:
        frame.write_csv(measurements_csv_path(root))
    if parquet:
        frame.write_parquet(measurements_parquet_path(root))
    return deliv


def write_pipeline_json(root: Path, pipeline: Any) -> Path:
    """Serialize ``pipeline`` to ``<root>/deliverables/pipeline.json``.

    ``pipeline`` may be an ``ImagePipeline`` (uses ``.to_json()``) or a raw
    JSON string already produced by the caller.
    """
    _ensure_deliverables(root)
    path = pipeline_json_path(root)
    text = pipeline if isinstance(pipeline, str) else (pipeline.to_json() or "")
    path.write_text(text, encoding="utf-8")
    return path


def write_dashboard(root: Path, *, execution_mode: str = "local") -> Path:
    """Generate a real ``<root>/deliverables/dashboard.html`` via the producer."""
    from phenotypic._cli._dashboard._generator import generate_dashboard

    _ensure_deliverables(root)
    generate_dashboard(root, execution_mode=execution_mode)
    from phenotypic.tools_ import dashboard_html_path

    return dashboard_html_path(root)


def seed_output_dir(
    root: Path,
    master: Any,
    *,
    pipeline: Any | None = None,
    mirror: Any | None = None,
    dashboard: bool = False,
    results_dataset: str | None = None,
) -> Path:
    """Seed a fake CLI output directory under ``root``.

    Writes the master archive (always), and optionally the measurements
    mirror, ``pipeline.json``, a real ``dashboard.html``, and an empty
    per-image ``results/<dataset>/`` tree (so the shell classifier
    recognizes ``root`` as a CLI output).

    Args:
        root: Run output root (``tmp_path`` or a subdir of it).
        master: Master DataFrame (polars or pandas).
        pipeline: Optional ``ImagePipeline`` or JSON string for
            ``pipeline.json``.
        mirror: Optional frame for ``measurements.{csv,parquet}``. If
            ``None`` and ``master`` is given, no mirror is written (callers
            that need the mirror to match the master should pass it
            explicitly).
        dashboard: When True, generate a real ``dashboard.html``.
        results_dataset: When set, create ``results/<dataset>/`` at the
            root so the classifier's ``is_cli_output`` check passes.

    Returns:
        ``root`` (the output dir), for chaining.
    """
    write_master(root, master)
    if mirror is not None:
        write_measurements_mirror(root, mirror)
    if pipeline is not None:
        write_pipeline_json(root, pipeline)
    if dashboard:
        write_dashboard(root)
    if results_dataset is not None:
        (root / "results" / results_dataset).mkdir(parents=True, exist_ok=True)
    return root
