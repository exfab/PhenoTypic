"""Time-axis predicate + (dataset, stem) record adapter for the Timeline tab.

The Results Timeline draws its axes from ``OutputRoot.master_df`` (the
post-applied mirror, which already carries joined ``Metadata_*`` columns).
The Y (row) axis reuses the colony-view ``selectable_axis_columns`` with the
50-cap removed (spec §16.5); the X (time) axis uses ``selectable_time_columns``
here — name/dtype-gated, *uncapped* (a long time-course is the whole point,
spec §15.2). One record is emitted per ``(dataset, stem)`` image pair that has
an overlay PNG; ``cell_ref`` is the ``(dataset, stem)`` tuple the thumbnail
route and the deep-zoom DZI route both consume.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._filtered_state import KEY_DATASET, KEY_IMAGE_FILE
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import (
    _MEASUREMENT_PREFIXES,
    _OBJECT_LABEL_COL,
)
from phenotypic.schema import MeasurementInfo

__all__ = [
    "selectable_time_columns",
    "is_large_time_axis",
    "has_eligible_time_axis",
    "build_timeline_records",
    "LARGE_TIME_AXIS_THRESHOLD",
]

#: Above this distinct-time-value count the toolbar shows a bucketing-warning
#: banner (bucketing UI itself is out of scope for v1 — spec §15.2).
LARGE_TIME_AXIS_THRESHOLD = 100


def _measurement_prefixes() -> frozenset[str]:
    """Authoritative ``<Category>_`` prefixes for measurement columns.

    The colony ``_MEASUREMENT_PREFIXES`` is a deliberately-small subset that
    works for the colony grid because its 50-cardinality cap drops the
    remaining numeric measurements. The timeline X axis is *uncapped*
    (spec §15.2), so a numeric measurement column (e.g. ``Size_Area``) would
    otherwise slip through the numeric-dtype eligibility path. We therefore
    union the colony baseline with every measurement category prefix derived
    from the public ``phenotypic.schema`` enums — excluding ``Metadata``/
    ``Object`` (valid axis/identity namespaces, gated separately).

    The subclass tree is walked *transitively*: ``MeasurementInfo`` has
    abstract intermediate bases (``PrimaryMeasure``/``DerivedMeasure``/…)
    whose ``category()`` raises, and the concrete category enums (``Size``,
    ``Shape``, …) hang off those intermediates — so ``__subclasses__()``
    alone (direct children only) would miss them.
    """
    prefixes: set[str] = set(_MEASUREMENT_PREFIXES)
    seen: set[type[MeasurementInfo]] = set()
    stack: list[type[MeasurementInfo]] = list(MeasurementInfo.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        try:
            category = cls.category()
        except Exception:  # abstract intermediate base — no category of its own
            continue
        if category in ("Metadata", "Object"):
            continue
        prefixes.add(f"{category}_")
    return frozenset(prefixes)


#: Computed once at import: the full measurement-column prefix set the
#: uncapped time-axis predicate excludes (colony baseline ∪ schema categories).
_TIME_AXIS_MEASUREMENT_PREFIXES: frozenset[str] = _measurement_prefixes()

#: Case-insensitive name match for a "Metadata_Time-like" column. Seeded from
#: Heatmap's hardcoded ``"Metadata_Time"`` (``_heatmap_tab/_callbacks.py:367``)
#: but generalized so e.g. ``Metadata_Timepoint`` / ``Metadata_ImageNumber``
#: also surface. Numeric/temporal dtype is an independent eligibility path.
_TIME_NAME_RE = re.compile(
    r"(?:^|_)(time|timepoint|imagenumber|frame)(?:_|$|\d)", re.IGNORECASE
)


def _is_time_like_name(col: str) -> bool:
    return bool(_TIME_NAME_RE.search(col))


def _is_ordered_dtype(dtype: pl.DataType) -> bool:
    # Numeric or temporal dtypes read as an ordered time axis without a name
    # match. Use the per-dtype predicates (pl.NUMERIC_DTYPES/TEMPORAL_DTYPES
    # are deprecated since polars 1.0).
    return bool(dtype.is_numeric() or dtype.is_temporal())


def selectable_time_columns(
    df: pl.DataFrame,
    column_value_sets: Mapping[str, list[str]],
) -> list[str]:
    """Return columns eligible as the timeline X (time) axis.

    A column is eligible iff it is NOT measurement-prefixed and NOT
    ``Object_Label``, AND either its name matches a ``Metadata_Time``-like
    pattern OR its dtype is numeric/temporal. There is **no cardinality cap**
    (spec §15.2). ``Metadata_*`` time-like names sort first, then everything
    else, alphabetically within each bucket.

    Args:
        df: The frame to inspect (typically the filtered master mirror).
        column_value_sets: Unused for eligibility but accepted for signature
            symmetry with ``selectable_axis_columns`` (callers thread the same
            pair through). Cardinality is intentionally NOT consulted.

    Returns:
        Eligible time-column names in bucketed sort order.
    """
    del column_value_sets  # eligibility is name/dtype-based, never cardinality
    eligible: list[str] = []
    schema = df.schema
    for col in df.columns:
        if col == _OBJECT_LABEL_COL:
            continue
        if any(col.startswith(prefix) for prefix in _TIME_AXIS_MEASUREMENT_PREFIXES):
            continue
        dtype = schema[col]
        if _is_time_like_name(col) or _is_ordered_dtype(dtype):
            eligible.append(col)

    def _bucket(name: str) -> int:
        return 0 if (name.startswith("Metadata_") and _is_time_like_name(name)) else 1

    eligible.sort(key=lambda name: (_bucket(name), name))
    return eligible


def is_large_time_axis(
    n_values: int, threshold: int = LARGE_TIME_AXIS_THRESHOLD
) -> bool:
    """Return ``True`` when the time axis has more than ``threshold`` distinct values."""
    return n_values > threshold


def has_eligible_time_axis(
    df: pl.DataFrame, column_value_sets: Mapping[str, list[str]]
) -> bool:
    """Return ``True`` iff at least one eligible time column exists (D9 empty state)."""
    return bool(selectable_time_columns(df, column_value_sets))


def build_timeline_records(
    output_root: OutputRoot,
    df: pl.DataFrame,
    *,
    row_col: str,
    time_col: str,
) -> list[dict[str, object]]:
    """Build ``build_matrix`` records from a filtered master slice.

    One record per ``(dataset, stem)`` pair surviving ``df`` that has an
    overlay PNG on disk. ``row_value``/``time_value`` are the row's values in
    ``row_col``/``time_col`` (stringified so they match ``build_matrix``'s own
    ``str(...)`` axis coercion downstream).

    Args:
        output_root: The viewer's output handle (overlay membership lookup).
        df: The filtered master mirror (the active filter slice).
        row_col: Y-axis column name.
        time_col: X (time)-axis column name.

    Returns:
        A list of ``{"row_value", "time_value", "cell_ref": (dataset, stem)}``
        dicts. ``cell_ref`` is the ``(dataset, stem)`` tuple consumed by the
        thumbnail + DZI routes.
    """
    needed = [KEY_DATASET, KEY_IMAGE_FILE, row_col, time_col]
    have = [c for c in dict.fromkeys(needed) if c in df.columns]
    slim = df.select(have).drop_nulls(subset=[KEY_DATASET, KEY_IMAGE_FILE]).unique()
    records: list[dict[str, object]] = []
    for record in slim.iter_rows(named=True):
        dataset = str(record[KEY_DATASET])
        image_file = str(record[KEY_IMAGE_FILE])
        stem = Path(image_file).stem if Path(image_file).suffix else image_file
        if not output_root.has_overlay(dataset, stem):
            continue
        records.append(
            {
                "row_value": str(record.get(row_col, "")),
                "time_value": str(record.get(time_col, "")),
                "cell_ref": (dataset, stem),
            }
        )
    return records
