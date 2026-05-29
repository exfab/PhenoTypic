"""Component IDs owned by the Heatmap tab.

The viewer-level ``_ids.py`` re-exposes :data:`TAB_HEATMAP_ID` and
:data:`STORE_QC_AUGMENTED_REVISION` so callbacks in other modules
(notably Wave E's QC tab) can subscribe without depending on this
sub-package.

Notes:
    The Heatmap tab does not use any pattern-matching ids - every
    control is a single static dropdown / slider so plain string ids
    are sufficient and grep-able.
"""
from __future__ import annotations

#: ``dbc.Tab`` value for the Heatmap view, mirrored from the viewer-level
#: ``_ids.py`` so this module is self-contained.
TAB_HEATMAP_ID = "tab-heatmap"

#: Hidden ``dcc.Store`` bumped by the QC tab callback after it has
#: finished writing ``CFG_QC_AUGMENTED_FRAME``. The heatmap render
#: callback subscribes to this revision (in addition to
#: ``STORE_REMOVED_KEYS``) so it never reads a stale frame between a
#: curation tick and the QC writer's completion. Spec lines 775-798.
STORE_QC_AUGMENTED_REVISION = "store-qc-augmented-revision"

#: Dropdown selecting the measurement (or QC metric) column whose
#: per-well values feed the heatmap. Options union
#: ``MeasurementSchema.columns_for("measurements")`` with any
#: ``QC_*_Metric`` columns present in the augmented frame.
HEATMAP_COLOR_PICKER_ID = "heatmap-color-picker"

#: Dropdown selecting the source image whose colonies are pivoted.
#: Restricted to unique ``Metadata_ImageFile`` values in the filtered
#: frame.
HEATMAP_IMAGE_PICKER_ID = "heatmap-image-picker"

#: Slider selecting the time-point filter. Marks are placed at every
#: unique numeric ``Metadata_Time`` value (not interpolated). Hidden via
#: :data:`HEATMAP_TIME_SLIDER_WRAPPER_ID` when only one time-point
#: exists, when the column is absent, or when coercion to numeric
#: yields all-NaN. See spec lines 1021-1028.
HEATMAP_TIME_SLIDER_ID = "heatmap-time-slider"

#: Wrapper ``html.Div`` around the time slider. Carrying the
#: ``display: none`` style here (rather than on the slider itself)
#: keeps the slider's value updates predictable when toggled by
#: ``_refresh_heatmap_controls``.
HEATMAP_TIME_SLIDER_WRAPPER_ID = "heatmap-time-slider-wrapper"

#: Inline caption below the time slider explaining a partial-NaN
#: ``Metadata_Time`` coercion (e.g. mixed numeric + "T0" / "baseline"
#: values). Empty string when no caption is needed.
HEATMAP_TIME_NON_NUMERIC_CAPTION_ID = "heatmap-time-non-numeric-caption"

#: Dropdown choosing the polars ``GroupBy.agg`` aggregator used to
#: collapse multi-row ``(Grid_RowNum, Grid_ColNum)`` bins (after the
#: image-file filter). For the typical one-row-per-well case the
#: aggregator is a no-op.
HEATMAP_AGGREGATOR_PICKER_ID = "heatmap-aggregator-picker"

#: ``dcc.Graph`` rendering the actual heatmap.
HEATMAP_FIGURE_ID = "heatmap-figure"

#: Explanation card rendered when ``Grid_RowNum`` / ``Grid_ColNum`` are
#: missing from the filtered frame (i.e. the pipeline did not run a
#: ``GridMeasureFeatures`` step). Spec lines 1029-1034.
HEATMAP_EMPTY_STATE_ID = "heatmap-empty-state"


__all__ = [
    "TAB_HEATMAP_ID",
    "STORE_QC_AUGMENTED_REVISION",
    "HEATMAP_COLOR_PICKER_ID",
    "HEATMAP_IMAGE_PICKER_ID",
    "HEATMAP_TIME_SLIDER_ID",
    "HEATMAP_TIME_SLIDER_WRAPPER_ID",
    "HEATMAP_TIME_NON_NUMERIC_CAPTION_ID",
    "HEATMAP_AGGREGATOR_PICKER_ID",
    "HEATMAP_FIGURE_ID",
    "HEATMAP_EMPTY_STATE_ID",
]
