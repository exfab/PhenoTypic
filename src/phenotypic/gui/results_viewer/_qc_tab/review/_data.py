"""Pure data layer for the QC Review tab — recompute frame (+ legacy reads).

Every function here is side-effect-free (except the disk *reads* of the
``qc/`` artifact) and Dash-free. The Review callbacks now read the QC
worklist + members through the catalog-driven DuckDB API (:mod:`._db`);
this module retains only:

* **Recompute frame** — :func:`build_recompute_frame` reads the
  **post-applied + metadata-joined** ``measurements.parquet`` and
  anti-joins the curated removal set, producing exactly the frame the CLI
  feeds :func:`phenotypic.sdk_._qc_recipe._runner.run_qc` minus the user's
  removals (spec §D.5 / risk refinement #1 — NOT ``master − removed``).
* **Legacy flat-parquet readers** — :func:`load_qc_summary` /
  :func:`load_qc_members` / :func:`groupby_cols_for` / :func:`_eq_or_null`
  are retained ONLY for the Error-analysis tab's verified-good derivation,
  which is migrated onto :mod:`._db` in a later task; once that lands these
  are removed.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

#: Curation-key columns (mirrors ``_filtered_state.KEY_COLUMNS``; kept
#: local so this pure module never imports the Dash-coupled state module).
_KEY_IMAGE_FILE: str = "Metadata_ImageFile"
_KEY_OBJECT_LABEL: str = "Object_Label"
_KEY_DATASET: str = "Metadata_Dataset"
_KEY_TIME: str = "Metadata_Time"

#: Fixed lead/tail columns of the legacy ``qc_summary.parquet``. The
#: ``groupby`` columns sit between the ``class`` lead and the ``metric``
#: tail, so :func:`groupby_cols_for` recovers them by slicing out these
#: known names. (Legacy reader scaffolding for the Error tab only.)
_SUMMARY_LEAD: tuple[str, ...] = ("instance_id", "class")
_SUMMARY_TAIL: tuple[str, ...] = (
    "metric",
    "status",
    "flag",
    "n_members",
    "n_flagged",
    "rank",
)


# ---------------------------------------------------------------------------
# Legacy artifact readers (Error-tab verified-good only — removed once it
# migrates onto the DuckDB catalog)
# ---------------------------------------------------------------------------


def load_qc_summary(output_root: "OutputRoot") -> pl.DataFrame | None:
    """Read ``<root>/qc/qc_summary.parquet`` or ``None`` when absent.

    Args:
        output_root: The active results-viewer output root.

    Returns:
        The summary frame, or ``None`` when the artifact has not been
        written yet (no QC configured / never recompiled).
    """
    return _read_optional_parquet(output_root.layout.qc_summary_parquet)


def load_qc_members(output_root: "OutputRoot") -> pl.DataFrame | None:
    """Read ``<root>/qc/qc_members.parquet`` or ``None`` when absent."""
    return _read_optional_parquet(output_root.layout.qc_members_parquet)


def _read_optional_parquet(path: Path) -> pl.DataFrame | None:
    """Read a parquet, returning ``None`` (logged) on missing/corrupt file."""
    if not path.is_file():
        return None
    try:
        return pl.read_parquet(path)
    except Exception:  # noqa: BLE001 - defensive: a corrupt artifact is non-fatal
        logger.warning("Failed to read QC artifact %s", path, exc_info=True)
        return None


def groupby_cols_for(
    summary_df: pl.DataFrame, instance_id: str
) -> list[str]:
    """Return the ``groupby`` column names a module's summary rows carry.

    Recovered structurally: any summary column that is neither a fixed
    lead/tail column nor all-null for this instance's rows is a group key.
    Columns that belong to *other* modules (all-null here) are excluded so
    a union-schema summary (multiple checks with different ``groupby``)
    still yields the right keys per module.

    Args:
        summary_df: The full summary frame.
        instance_id: The module whose group keys are wanted.

    Returns:
        Ordered group-key column names for this module.
    """
    fixed = set(_SUMMARY_LEAD) | set(_SUMMARY_TAIL)
    candidate_cols = [c for c in summary_df.columns if c not in fixed]
    slice_df = summary_df.filter(pl.col("instance_id") == instance_id)
    if slice_df.is_empty():
        return []
    keep: list[str] = []
    for col in candidate_cols:
        # A genuine group key for this module has at least one non-null
        # value across its rows; a foreign module's key is all-null here.
        if slice_df.get_column(col).null_count() < slice_df.height:
            keep.append(col)
    return keep


def _eq_or_null(col: str, value: Any) -> pl.Expr:
    """Build an equality predicate that also matches a null group key.

    ``groupby(dropna=False)`` can produce a null group key; a plain ``==``
    never matches null in polars, so route null comparisons through
    ``is_null`` to keep null-keyed groups selectable.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pl.col(col).is_null()
    return pl.col(col).cast(pl.String) == str(value)


# ---------------------------------------------------------------------------
# Recompute frame (post-applied + metadata-joined, minus removals)
# ---------------------------------------------------------------------------


def build_recompute_frame(
    output_root: "OutputRoot",
    removed_keys: set[tuple[str, int]],
) -> "pd.DataFrame":
    """Return the curated frame to hand :func:`run_qc` for an in-session recompute.

    Reads the **post-applied + metadata-joined** ``measurements.parquet``
    (the exact frame the CLI feeds ``run_qc`` in ``finalize``) and
    anti-joins the live removal set, then converts to pandas (``run_qc``
    is pandas-typed). This is deliberately **not** ``get_curated_frame``
    (which is ``master − removed``): the master archive is post-free and
    metadata-free, so a metadata-only ``groupby`` column would ``KeyError``
    and the before→after delta would compare against a different frame
    than the CLI's artifact (spec §D risk refinement #1).

    Falls back to the master parquet only when the post-applied mirror is
    absent (mid-run / legacy output), matching ``OutputRoot.discover``'s
    own fallback so a recompute never hard-fails on a partial run.

    Args:
        output_root: The active output root.
        removed_keys: The curated ``(image_file, label)`` removal set, read
            under the ``FilteredMeasurements`` lock by the caller.

    Returns:
        A pandas DataFrame: the post-applied frame minus removed rows.
    """
    layout = output_root.layout
    mirror = layout.mirror_parquet
    if mirror.is_file():
        frame = pl.read_parquet(mirror)
    else:
        # Mid-run / legacy: the mirror has not been seeded. The master is
        # the only frame available; it lacks post/metadata columns, so a
        # metadata groupby will still KeyError inside run_qc and be skipped
        # with a warning — acceptable degradation, never a crash here.
        logger.info(
            "measurements.parquet absent; recompute falling back to master"
        )
        frame = pl.read_parquet(layout.master_parquet)

    curated = _anti_join_removed(frame, removed_keys)
    return curated.to_pandas()


def _anti_join_removed(
    frame: pl.DataFrame, removed_keys: set[tuple[str, int]]
) -> pl.DataFrame:
    """Drop rows whose ``(image_file, label)`` is in the removal set.

    Args:
        frame: The post-applied measurements frame.
        removed_keys: Curated removal keys.

    Returns:
        The frame with removed rows filtered out. Returned unchanged when
        the removal set is empty or the key columns are absent.
    """
    if not removed_keys:
        return frame
    if _KEY_IMAGE_FILE not in frame.columns or _KEY_OBJECT_LABEL not in frame.columns:
        return frame
    removed_df = pl.DataFrame(
        {
            _KEY_IMAGE_FILE: [k[0] for k in removed_keys],
            _KEY_OBJECT_LABEL: [k[1] for k in removed_keys],
        },
        schema={_KEY_IMAGE_FILE: pl.String, _KEY_OBJECT_LABEL: pl.Int64},
    )
    return (
        frame.with_columns(
            pl.col(_KEY_IMAGE_FILE).cast(pl.String),
            pl.col(_KEY_OBJECT_LABEL).cast(pl.Int64),
        )
        .join(removed_df, on=[_KEY_IMAGE_FILE, _KEY_OBJECT_LABEL], how="anti")
    )


__all__ = [
    "load_qc_summary",
    "load_qc_members",
    "groupby_cols_for",
    "build_recompute_frame",
]
