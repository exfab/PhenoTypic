"""Headless re-emit of the error-triage deliverables at CLI finalize.

Dash-free. Re-keys the durable ``qc/curation_labels.parquet`` onto the fresh
clean master (the SAME frame the GUI's ``CurationLabels`` loads, so headless
output is byte-consistent with live curation), re-writes the per-category
``deliverables/errors/*.parquet`` + the re-keyed labels parquet, and computes
``deliverables/error_analysis.{parquet,csv,html}`` across every labeled category
(all-unlabeled good baseline; verified mode is GUI-only).
``deliverables/measurements.parquet`` and ``deliverables/verified.parquet`` are
left untouched (spec §9 decisions 2 + 4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from phenotypic.analysis import ErrorCutoffFinder, render_error_analysis_report
from phenotypic.analysis._error_cutoffs import RESULT_COLUMNS, _RESULT_DTYPES
from phenotypic.tools_ import (
    curation_labels_parquet_path,
    error_analysis_csv_path,
    error_analysis_html_path,
    error_analysis_parquet_path,
)

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels

#: Columns of the persisted ``error_analysis.{parquet,csv}`` (the leading
#: ``category`` tag plus the category-free ``ErrorCutoffFinder`` result columns).
_PERSIST_COLUMNS: tuple[str, ...] = ("category", *RESULT_COLUMNS)


def reemit_error_deliverables(output_dir: Path, master_df: pl.DataFrame) -> None:
    """Re-emit errors/* + error_analysis.* from the durable labels store.

    No-op when there is no durable ``curation_labels.parquet`` (migration from a
    legacy ``measurements.parquet`` is a GUI-load concern only — in finalize the
    mirror is the *post-applied* seed, so the migration path would falsely import
    post-dropped rows as ``other``; see plan decision 3). Idempotent: stale
    per-category parquets are pruned by :meth:`CurationLabels.write_error_partitions`.

    Args:
        output_dir: The run output directory.
        master_df: The clean (pre-post) master frame being finalized — the SAME
            frame the GUI's ``CurationLabels`` loads, so headless == live.
    """
    if not curation_labels_parquet_path(output_dir).exists():
        return
    # Local import keeps the GUI package off the hot CLI import path; it is
    # Dash-free (verified) so this stays cheap.
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels

    store = CurationLabels.load(output_dir, master_df)  # re-keys onto fresh master
    if not store.labels:
        return
    store.write_error_partitions()  # errors/*.parquet + re-keyed labels parquet (no mirror)
    _write_error_analysis(output_dir, store, master_df)


def _write_error_analysis(
    output_dir: Path, store: "CurationLabels", master_df: pl.DataFrame
) -> None:
    """Run :class:`ErrorCutoffFinder` per category; write ``error_analysis.*``.

    The good baseline is the all-unlabeled set (``filtered_df`` drops every
    labeled row); the error set per category is the master rows carrying that
    category. The parquet/csv carry a leading ``category`` column; the HTML is
    one report with a section per category (decision 5).
    """
    good_pdf = store.filtered_df(master_df).to_pandas()  # all-unlabeled good
    finder = ErrorCutoffFinder()
    frames: list[pd.DataFrame] = []
    reports: dict[str, pd.DataFrame] = {}
    for category in sorted(set(store.labels.values())):
        error_keys = [k for k, c in store.labels.items() if c == category]
        error_pdf = _rows_for_keys(master_df, error_keys)
        res = finder.analyze(good_pdf, error_pdf)
        reports[category] = res
        if not res.empty:
            tagged = res.copy()
            tagged.insert(0, "category", category)
            frames.append(tagged[list(_PERSIST_COLUMNS)])

    combined = pd.concat(frames, ignore_index=True) if frames else _empty_combined()
    error_analysis_parquet_path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(combined, error_analysis_parquet_path(output_dir))
    _atomic_write_csv(combined, error_analysis_csv_path(output_dir))
    _atomic_write_text(
        render_error_analysis_report(reports), error_analysis_html_path(output_dir)
    )


def _empty_combined() -> pd.DataFrame:
    """Typed 0-row ``error_analysis`` frame (empty + populated schemas agree).

    Mirrors the engine's own ``_empty_result`` discipline so a labels-but-no-
    separation run writes a frame whose ``auc``/``cutoff``/... stay numeric
    instead of inferring ``object`` (L3).
    """
    dtypes = {"category": "object", **_RESULT_DTYPES}
    return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in _PERSIST_COLUMNS})


def _rows_for_keys(
    master_df: pl.DataFrame, keys: list[tuple[str, int]]
) -> pd.DataFrame:
    """Return the master rows whose ``(image_file, object_label)`` is in ``keys``.

    Mirrors ``CurationLabels._join_on_keys(..., "semi")`` and converts to pandas
    at the engine boundary (``ErrorCutoffFinder.analyze`` is pandas-typed).
    """
    from phenotypic.gui.results_viewer._curation_labels import _join_on_keys

    return _join_on_keys(master_df, keys, "semi").to_pandas()


# ---------------------------------------------------------------------------
# Atomic writers (temp + os.replace) — match the GUI/curation-store discipline
# so a concurrent viewer or a mid-write crash never sees a truncated artifact.
# ---------------------------------------------------------------------------


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` as parquet via a sibling temp file + replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pl.from_pandas(df).write_parquet(tmp)
    os.replace(tmp, path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` as CSV via a sibling temp file + replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pl.from_pandas(df).write_csv(tmp)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: Path) -> None:
    """Write ``text`` to ``path`` via a sibling temp file + replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
