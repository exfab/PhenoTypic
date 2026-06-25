"""Pure data layer for the QC Review tab — artifact reads + recompute frame.

Every function here is side-effect-free (except the disk *reads* of the
``qc/`` artifact) and Dash-free, so the worklist/summary/gallery
callbacks and their tests can exercise the load-bearing logic without
booting an app. The Review callbacks call into this module; this module
never imports Dash.

What lives here:

* **Artifact readers** — :func:`load_qc_summary` / :func:`load_qc_members`
  read the committed ``qc_summary.parquet`` / ``qc_members.parquet``
  (schema fixed by :mod:`phenotypic.sdk_._qc_recipe._runner`).
* **Module + worklist slicing** — :func:`module_options`,
  :func:`groupby_cols_for`, :func:`module_worklist` (worst-first, frozen).
* **Summary stats** — :func:`summary_stats`, which distinguishes
  **NaN/insufficient** groups from genuine ``pass`` and computes a
  robust (NaN/inf-safe) median metric (spec §D risk refinement).
* **Detail / gallery** — :func:`group_member_keys` and
  :func:`facet_keys_by_timepoint` resolve a group's tiles (and their
  per-timepoint facet rows) for :func:`gui._shared.tiles.build_tile_grid`.
* **Recompute frame** — :func:`build_recompute_frame` reads the
  **post-applied + metadata-joined** ``measurements.parquet`` and
  anti-joins the curated removal set, producing exactly the frame the CLI
  feeds :func:`phenotypic.sdk_._qc_recipe._runner.run_qc` minus the user's removals
  (spec §D.5 / risk refinement #1 — NOT ``master − removed``).
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

#: Fixed lead/tail columns of ``qc_summary.parquet`` (see
#: :mod:`phenotypic.sdk_._qc_recipe._runner`). The ``groupby`` columns sit between the
#: ``class`` lead and the ``metric`` tail, so we recover them by slicing
#: out these known names.
_SUMMARY_LEAD: tuple[str, ...] = ("instance_id", "class")
_SUMMARY_TAIL: tuple[str, ...] = (
    "metric",
    "status",
    "flag",
    "n_members",
    "n_flagged",
    "rank",
)

#: Fixed lead/tail of ``qc_members.parquet``.
_MEMBERS_LEAD: tuple[str, ...] = ("instance_id",)
_MEMBERS_TAIL: tuple[str, ...] = (
    _KEY_IMAGE_FILE,
    _KEY_OBJECT_LABEL,
    "member_value",
)


# ---------------------------------------------------------------------------
# Artifact readers
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


# ---------------------------------------------------------------------------
# Module + worklist slicing
# ---------------------------------------------------------------------------


def module_options(summary_df: pl.DataFrame | None) -> list[dict[str, str]]:
    """Return module-picker options from the distinct instances in a summary.

    Each option's ``label`` is ``"<Class> (<short-id>)"`` and its ``value``
    is the full ``instance_id``. Order follows first appearance in the
    summary (which ``run_qc`` writes in recipe order).

    Args:
        summary_df: The loaded ``qc_summary`` frame, or ``None``.

    Returns:
        A list of ``{"label", "value"}`` dicts (empty when no summary).
    """
    if summary_df is None or summary_df.is_empty():
        return []
    seen: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for instance_id, cls in zip(
        summary_df.get_column("instance_id").to_list(),
        summary_df.get_column("class").to_list(),
    ):
        iid = str(instance_id)
        if iid in seen_ids:
            continue
        seen_ids.add(iid)
        seen.append((iid, str(cls)))
    return [
        {"label": f"{cls} ({_short_id(iid)})", "value": iid}
        for iid, cls in seen
    ]


def _short_id(instance_id: str) -> str:
    """Return the trailing hex segment of an ``instance_id`` for display."""
    return instance_id.rsplit("-", 1)[-1] if "-" in instance_id else instance_id


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


def module_worklist(
    summary_df: pl.DataFrame, instance_id: str
) -> pl.DataFrame:
    """Return one module's summary rows, worst-first by the frozen ``rank``.

    The returned frame is the worklist's frozen order: ``run_qc`` already
    assigned a stable worst-first ``rank`` (NaN metrics last), so the
    worklist simply sorts by it. Re-sorting only happens when the user
    clicks ↻ Re-sort, which re-reads this after an in-session recompute.

    Args:
        summary_df: The full summary frame.
        instance_id: The module to slice.

    Returns:
        The module's rows sorted ascending by ``rank`` (0 = worst).
    """
    slice_df = summary_df.filter(pl.col("instance_id") == instance_id)
    if "rank" in slice_df.columns:
        slice_df = slice_df.sort("rank", nulls_last=True)
    return slice_df


# ---------------------------------------------------------------------------
# Summary header stats
# ---------------------------------------------------------------------------


def summary_stats(module_summary: pl.DataFrame) -> dict[str, Any]:
    """Compute the summary-header stat tiles for one module's worklist.

    Distinguishes **insufficient/NaN** groups (a metric that could not be
    computed — e.g. an ICC on a sparse grid) from genuine ``pass`` so the
    header never paints a no-signal group green (spec §D risk refinement).
    The median metric is **robust**: computed over finite values only, so
    a single ``inf`` (an unmatched-group ``Count``) or a swarm of ``NaN``
    cannot poison it.

    Args:
        module_summary: One module's summary rows (any order).

    Returns:
        A dict with integer counts ``total``/``fail``/``warn``/``pass``/
        ``insufficient`` plus ``colonies_removed`` placeholder ``0`` (the
        caller fills it from the live removal set) and ``median_metric``
        (``float`` or ``None`` when no finite metric exists).
    """
    total = int(module_summary.height)
    if total == 0:
        return {
            "total": 0,
            "fail": 0,
            "warn": 0,
            "pass": 0,
            "insufficient": 0,
            "colonies_removed": 0,
            "median_metric": None,
        }

    metrics = module_summary.get_column("metric").to_list()
    statuses = module_summary.get_column("status").to_list()

    fail = warn = passed = insufficient = 0
    finite_metrics: list[float] = []
    for metric, status in zip(metrics, statuses):
        is_nan = metric is None or (
            isinstance(metric, float) and math.isnan(metric)
        )
        if is_nan:
            insufficient += 1
        elif status == "fail":
            fail += 1
        elif status == "warn":
            warn += 1
        else:
            passed += 1
        if (
            metric is not None
            and isinstance(metric, (int, float))
            and math.isfinite(metric)
        ):
            finite_metrics.append(float(metric))

    median_metric = _robust_median(finite_metrics)
    return {
        "total": total,
        "fail": fail,
        "warn": warn,
        "pass": passed,
        "insufficient": insufficient,
        "colonies_removed": 0,
        "median_metric": median_metric,
    }


def _robust_median(values: list[float]) -> float | None:
    """Median over finite values only; ``None`` when the list is empty."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# ---------------------------------------------------------------------------
# Detail / gallery key resolution
# ---------------------------------------------------------------------------


def group_member_keys(
    members_df: pl.DataFrame,
    instance_id: str,
    groupby_cols: list[str],
    key_values: tuple[Any, ...],
    dataset_by_image: dict[str, str],
) -> list[tuple[str, str, int]]:
    """Resolve the ``(dataset, image_file, label)`` tiles for one group.

    Filters ``qc_members`` to the module + group key and joins each
    member's dataset from ``dataset_by_image`` (built once from the master
    frame) since the artifact carries only ``Metadata_ImageFile`` /
    ``Object_Label``. Members whose image is unknown to the dataset map
    are dropped (logged) rather than rendered with a bogus crop URL.

    Args:
        members_df: The full ``qc_members`` frame.
        instance_id: The module to slice.
        groupby_cols: The module's group-key column names (from
            :func:`groupby_cols_for`).
        key_values: The group-key value tuple, aligned to ``groupby_cols``.
        dataset_by_image: ``Metadata_ImageFile -> Metadata_Dataset`` map.

    Returns:
        ``(dataset, image_file, label)`` tuples in artifact (member) order,
        ready for :func:`gui._shared.tiles.build_tile_grid`.
    """
    predicate = pl.col("instance_id") == instance_id
    for col, value in zip(groupby_cols, key_values):
        if col not in members_df.columns:
            continue
        predicate = predicate & _eq_or_null(col, value)
    slice_df = members_df.filter(predicate)

    keys: list[tuple[str, str, int]] = []
    for image_file, label in zip(
        slice_df.get_column(_KEY_IMAGE_FILE).to_list(),
        slice_df.get_column(_KEY_OBJECT_LABEL).to_list(),
    ):
        image = str(image_file)
        dataset = dataset_by_image.get(image)
        if dataset is None:
            logger.debug(
                "QC member image %r has no dataset in master; skipping tile",
                image,
            )
            continue
        keys.append((dataset, image, int(label)))
    return keys


def _eq_or_null(col: str, value: Any) -> pl.Expr:
    """Build an equality predicate that also matches a null group key.

    ``groupby(dropna=False)`` can produce a null group key; a plain ``==``
    never matches null in polars, so route null comparisons through
    ``is_null`` to keep null-keyed groups selectable.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pl.col(col).is_null()
    return pl.col(col).cast(pl.String) == str(value)


def group_record(
    summary_df: pl.DataFrame,
    instance_id: str,
    groupby_cols: list[str],
    key_values: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Return one group's summary row as a dict, or ``None`` when absent.

    Args:
        summary_df: The full ``qc_summary`` frame.
        instance_id: The module to slice.
        groupby_cols: The module's group-key column names.
        key_values: The group-key value tuple aligned to ``groupby_cols``.

    Returns:
        The matching summary row as a ``{column: value}`` dict (the first
        match), or ``None`` when no row matches.
    """
    predicate = pl.col("instance_id") == instance_id
    for col, value in zip(groupby_cols, key_values):
        if col in summary_df.columns:
            predicate = predicate & _eq_or_null(col, value)
    row = summary_df.filter(predicate)
    if row.is_empty():
        return None
    return row.head(1).to_dicts()[0]


def facet_keys_by_timepoint(
    keys: list[tuple[str, str, int]],
    time_by_key: dict[tuple[str, int], Any],
) -> list[tuple[Any, list[tuple[str, str, int]]]]:
    """Group a flat tile-key list into per-timepoint facet rows.

    For time-course checks the detail gallery shows **one row per
    timepoint** (spec §D.2). Each tile's timepoint is looked up by its
    ``(image_file, label)`` key in ``time_by_key`` (resolved once from the
    master frame's ``Metadata_Time`` column). When no timepoint is known
    for any tile (``time_by_key`` empty / column absent), a single
    ``(None, keys)`` facet is returned so the caller renders one
    unfaceted gallery.

    Args:
        keys: ``(dataset, image_file, label)`` tuples for the group.
        time_by_key: ``(image_file, label) -> timepoint`` map.

    Returns:
        A list of ``(timepoint, keys)`` facet rows, ordered by timepoint
        (``None`` timepoints sort last). A single ``(None, keys)`` row when
        no timepoints are available.
    """
    if not time_by_key:
        return [(None, keys)]

    facets: dict[Any, list[tuple[str, str, int]]] = {}
    any_known = False
    for dataset, image_file, label in keys:
        timepoint = time_by_key.get((image_file, label))
        if timepoint is not None:
            any_known = True
        facets.setdefault(timepoint, []).append((dataset, image_file, label))

    if not any_known:
        return [(None, keys)]

    def _sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
        tp = item[0]
        # None timepoints sort last; everything else sorts by string form
        # (timepoints are typically small ints / floats / labels).
        return (1, "") if tp is None else (0, _time_sort_token(tp))

    return sorted(facets.items(), key=_sort_key)


def _time_sort_token(value: Any) -> str:
    """Zero-pad numeric timepoints so they sort numerically as strings."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{float(value):020.6f}"
    return str(value)


def dataset_by_image_map(output_root: "OutputRoot") -> dict[str, str]:
    """Build a ``Metadata_ImageFile -> Metadata_Dataset`` map from the master.

    Used to recover the dataset for each ``qc_members`` tile (the artifact
    carries only image/label). When ``Metadata_Dataset`` is absent (a
    single-dataset run), every image maps to the lone dataset directory.

    Args:
        output_root: The active output root (provides ``master_df``).

    Returns:
        Mapping from image file to its dataset name.
    """
    master = output_root.master_df
    if _KEY_DATASET in master.columns and _KEY_IMAGE_FILE in master.columns:
        pairs = (
            master.select([_KEY_IMAGE_FILE, _KEY_DATASET])
            .unique()
            .rows()
        )
        return {str(image): str(dataset) for image, dataset in pairs}
    return {}


def time_by_key_map(output_root: "OutputRoot") -> dict[tuple[str, int], Any]:
    """Build a ``(image_file, label) -> Metadata_Time`` map from the master.

    Empty when the run has no ``Metadata_Time`` column (not a time-course),
    which makes :func:`facet_keys_by_timepoint` fall back to a single
    unfaceted gallery.

    Args:
        output_root: The active output root (provides ``master_df``).

    Returns:
        Mapping from curation key to its timepoint value.
    """
    master = output_root.master_df
    needed = (_KEY_IMAGE_FILE, _KEY_OBJECT_LABEL, _KEY_TIME)
    if not all(col in master.columns for col in needed):
        return {}
    rows = master.select(list(needed)).rows()
    return {(str(image), int(label)): time for image, label, time in rows}


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
    "module_options",
    "groupby_cols_for",
    "module_worklist",
    "summary_stats",
    "group_record",
    "group_member_keys",
    "facet_keys_by_timepoint",
    "dataset_by_image_map",
    "time_by_key_map",
    "build_recompute_frame",
]
