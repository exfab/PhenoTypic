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

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from phenotypic.sdk_ import (
    BundleLayout,
    curation_labels_parquet_path,
    deliverables_dir,
)

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels

def reemit_error_deliverables(
    output_dir: Path, master_df: pl.DataFrame
) -> None:
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

    # The CLI always has a real output root, so build the full-run layout
    # directly (``BundleLayout.detect`` would require the master parquet to
    # already be on disk, which finalize writes *before* this call but
    # direct-reemit unit tests do not). ``CurationLabels.load`` now takes a
    # ``BundleLayout`` and falls back to the passed ``master_df`` when the
    # on-disk clean master is absent.
    layout = BundleLayout(
        deliverables_base=deliverables_dir(output_dir), output_root=output_dir
    )
    store = CurationLabels.load(layout, master_df)  # re-keys onto fresh master
    if not store.labels:
        return
    store.write_error_partitions()  # errors/*.parquet + re-keyed labels parquet (no mirror)
    _write_error_analysis(output_dir, store, master_df)


def _write_error_analysis(
    output_dir: Path, store: "CurationLabels", master_df: pl.DataFrame
) -> None:
    """Compute and transactionally publish every configured category.

    The good baseline is the all-unlabeled set (``filtered_df`` drops every
    labeled row). The pure computation and checksummed, rollback-capable
    generation publisher are shared with the GUI's explicit action.
    """
    good_pdf = store.filtered_df(master_df).to_pandas()  # all-unlabeled good
    layout = BundleLayout(
        deliverables_base=deliverables_dir(output_dir),
        output_root=output_dir,
    )
    from phenotypic.gui.results_viewer._error_tab._publication import (
        capture_error_source_fingerprints,
        compute_all_category_analysis,
        publish_error_analysis,
    )
    computation = compute_all_category_analysis(
        master_df,
        labels=dict(store.labels),
        categories=tuple(store.categories()),
        good_pdf=good_pdf,
        good_mode="all_unlabeled",
        source_fingerprints=capture_error_source_fingerprints(layout),
    )
    publish_error_analysis(
        layout,
        computation,
        # The CLI is the active authoritative owner of this output. Source
        # fingerprints and the shared Error lock still serialize it against
        # a stale GUI publisher.
        mutation_is_safe=lambda: True,
    )
