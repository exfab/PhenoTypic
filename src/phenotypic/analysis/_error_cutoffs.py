"""Good-vs-error-category measurement screen with per-measurement cutoffs.

For one error category, :class:`ErrorCutoffFinder` compares the *good* baseline
distribution against the *error* distribution on every measurement column and
ranks the measurements by how cleanly they separate the two (AUC). Each
discriminative measurement gets a ROC/Youden's-J cutoff with the recall and
precision it achieves, plus the one-way ANOVA F/p and a Benjamini-Hochberg
FDR-adjusted p. The result is the table the Error-analysis tab reads so the
user can adopt a cutoff to filter similar bad data.

The engine is deliberately **GUI/IO-free and mode-agnostic**: it takes a *good*
frame and an *error* frame and does not know whether the good baseline is
"all unlabeled" or the verified-only set — the caller decides (spec §7).
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

#: Column-name prefixes treated as numeric **phenotype** measurements.
#: Absolute position (``Bbox_`` centroids/corners) is intentionally **excluded**
#: (resolved decision) — a "cutoff" on plate position is a spatial artifact, not
#: a phenotype filter. ``Texture`` (no trailing ``_``) catches every texture
#: matrix (``TextureGray_``, ``TextureColor_``, …), not just gray. This list is
#: defined independently — the colony grid's ``_MEASUREMENT_PREFIXES`` is a UI
#: axis-exclusion list, not an authoritative phenotype-measurement set.
MEASUREMENT_PREFIXES: tuple[str, ...] = (
    "Size_",
    "Shape_",
    "Intensity_",
    "Texture",
    "SymZones_",
    "GridSpatial_",
    "RadialExpansion_",
)

#: Output columns of :meth:`ErrorCutoffFinder.analyze`, in order.
RESULT_COLUMNS: tuple[str, ...] = (
    "measurement",
    "auc",
    "direction",
    "cutoff",
    "recall",
    "specificity",
    "good_flagged",
    "f_stat",
    "p_value",
    "p_bh",
    "good_n",
    "error_n",
)


class ErrorCutoffFinder(BaseModel):
    """Rank measurements by good-vs-error separability with suggested cutoffs.

    Args:
        min_error_n: Minimum error-class sample size; below it, :meth:`analyze`
            returns an empty frame (the statistics would be unstable).
        min_good_n: Minimum good-class sample size; same behaviour.
        measurement_prefixes: Column-name prefixes treated as numeric
            measurements. Defaults to :data:`MEASUREMENT_PREFIXES`.
    """

    model_config = ConfigDict(extra="forbid")

    min_error_n: int = 8
    min_good_n: int = 8
    measurement_prefixes: tuple[str, ...] = MEASUREMENT_PREFIXES

    def measurement_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the numeric measurement columns of ``df`` in column order.

        A column qualifies iff its name starts with one of
        :attr:`measurement_prefixes` and its dtype is numeric.

        Args:
            df: A measurement frame (good or error).

        Returns:
            The qualifying measurement column names.
        """
        out: list[str] = []
        for col in df.columns:
            if not col.startswith(self.measurement_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                out.append(col)
        return out

    def enough_data(self, good: pd.DataFrame, error: pd.DataFrame) -> bool:
        """Return whether both classes meet their minimum sample sizes."""
        return len(good) >= self.min_good_n and len(error) >= self.min_error_n
