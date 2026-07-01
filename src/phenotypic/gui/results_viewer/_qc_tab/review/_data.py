"""Pure data layer for the QC Review tab — the in-session recompute frame.

Side-effect-free (except the disk *read* of ``measurements.parquet``) and
Dash-free. The QC worklist + members are now read through the
catalog-driven DuckDB API (:mod:`._db`); this module retains only the
recompute frame:

* :func:`build_recompute_frame` reads the **post-applied + metadata-joined**
  ``measurements.parquet`` and anti-joins the curated removal set, producing
  exactly the frame the CLI feeds
  :func:`phenotypic.sdk_._qc_recipe._runner.run_qc` minus the user's
  removals (spec §D.5 / risk refinement #1 — NOT ``master − removed``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from phenotypic.schema import CULTURE_METADATA, EXPERIMENT_METADATA, METADATA

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

#: Curation-key columns (mirrors ``_filtered_state.KEY_COLUMNS``; kept
#: local so this pure module never imports the Dash-coupled state module).
_KEY_IMAGE_FILE: str = str(METADATA.IMAGE_NAME)
_KEY_OBJECT_LABEL: str = "Object_Label"
_KEY_DATASET: str = str(EXPERIMENT_METADATA.DATASET)
_KEY_TIME: str = str(CULTURE_METADATA.TIME)


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
    "build_recompute_frame",
]
