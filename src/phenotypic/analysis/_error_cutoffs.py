"""Good-vs-error-category measurement screen with per-measurement cutoffs.

For one error category, :class:`ErrorCutoffFinder` compares the *good* baseline
distribution against the *error* distribution on every measurement column and
ranks the measurements by how cleanly they separate the two (AUC). Each
discriminative measurement gets a ROC/Youden's-J cutoff with the recall and
specificity it achieves (plus the count of good objects it would wrongly
flag), the one-way ANOVA F/p, and a Benjamini-Hochberg FDR-adjusted p. The
result is the table the Error-analysis tab reads so the user can adopt a cutoff
to filter similar bad data.

The engine is deliberately **GUI/IO-free and mode-agnostic**: it takes a *good*
frame and an *error* frame and does not know whether the good baseline is
"all unlabeled" or the verified-only set — the caller decides (spec §7).
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import f_oneway, false_discovery_control
from sklearn.metrics import roc_auc_score, roc_curve

#: Column-name prefixes treated as numeric **phenotype** measurements.
#: Absolute position (``Bbox_`` centroids/corners) is intentionally **excluded**
#: (resolved decision) — a "cutoff" on plate position is a spatial artifact, not
#: a phenotype filter. ``Texture`` (no trailing ``_``) matches the whole
#: ``Texture``-prefixed namespace regardless of the matrix/scale suffix. This
#: list is defined independently — the colony grid's ``_MEASUREMENT_PREFIXES``
#: is a UI axis-exclusion list, not an authoritative phenotype-measurement set.
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

    def analyze(self, good: pd.DataFrame, error: pd.DataFrame) -> pd.DataFrame:
        """Screen every measurement for good-vs-error separation.

        Args:
            good: The good-baseline frame (caller chooses all-unlabeled vs
                verified-only — the engine is agnostic).
            error: The frame of objects labelled with the target error
                category.

        Returns:
            A frame with one row per measurement, columns
            :data:`RESULT_COLUMNS`, sorted by ``auc`` (separability) descending.
            Empty (0 rows, same columns) when :meth:`enough_data` is ``False``
            or no measurement column has enough non-NaN values in both classes.

        Examples:
            >>> import numpy as np, pandas as pd
            >>> rng = np.random.default_rng(0)
            >>> good = pd.DataFrame({"Size_Area": rng.normal(0, 1, 40)})
            >>> error = pd.DataFrame({"Size_Area": rng.normal(5, 1, 12)})
            >>> res = ErrorCutoffFinder().analyze(good, error)
            >>> res.iloc[0]["measurement"], bool(res.iloc[0]["auc"] > 0.9)
            ('Size_Area', True)
        """
        empty = pd.DataFrame(columns=list(RESULT_COLUMNS))
        if not self.enough_data(good, error):
            return empty

        rows: list[dict[str, object]] = []
        for col in self.measurement_columns(good):
            if col not in error.columns:
                continue
            g = pd.to_numeric(good[col], errors="coerce").dropna().to_numpy()
            e = pd.to_numeric(error[col], errors="coerce").dropna().to_numpy()
            scored = self._score_measurement(g, e)
            if scored is None:
                continue
            scored["measurement"] = col
            rows.append(scored)

        if not rows:
            return empty

        res = pd.DataFrame(rows)
        # Benjamini-Hochberg across the screened measurements.
        res["p_bh"] = false_discovery_control(res["p_value"].to_numpy(), method="bh")
        res = res.sort_values("auc", ascending=False, ignore_index=True)
        return res[list(RESULT_COLUMNS)]

    @staticmethod
    def _score_measurement(
        g: npt.NDArray[np.float64], e: npt.NDArray[np.float64]
    ) -> dict[str, object] | None:
        """Score one measurement: ANOVA F/p, AUC + direction, gap-midpoint cutoff.

        Args:
            g: Good-class values (NaN-free 1-D array).
            e: Error-class values (NaN-free 1-D array).

        Returns:
            A dict of the per-measurement statistics, or ``None`` when either
            class has < 2 values, the combined values are constant, or the
            ANOVA F/p is non-finite (degenerate) — so no non-finite p ever
            reaches the BH step.
        """
        if len(g) < 2 or len(e) < 2:
            return None
        scores = np.concatenate([g, e])
        if np.ptp(scores) == 0:  # all identical -> nothing to separate
            return None
        y = np.concatenate([np.zeros(len(g)), np.ones(len(e))])  # 1 = error

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence scipy ConstantInputWarning
            f_stat, p_value = f_oneway(g, e)
        # Degenerate (e.g. each class internally constant -> F=inf/p=nan): skip
        # BEFORE the value can poison BH-FDR (false_discovery_control rejects NaN).
        if not np.isfinite(f_stat) or not np.isfinite(p_value):
            return None

        auc_raw = roc_auc_score(y, scores)  # P(error score > good score)
        if auc_raw >= 0.5:
            direction = ">"  # error is the HIGH side; flag when value > cutoff
            separability = auc_raw
            fpr, tpr, thr = roc_curve(y, scores)
        else:
            direction = "<"  # error is the LOW side; flag when value < cutoff
            separability = 1.0 - auc_raw
            fpr, tpr, thr = roc_curve(y, -scores)
            thr = -thr  # map thresholds back to the measurement scale

        # Youden's J optimal operating point (skip roc_curve's +/-inf point).
        valid = np.isfinite(thr)
        k = int(np.argmax(tpr[valid] - fpr[valid]))
        thr_v = float(thr[valid][k])
        tpr_k = float(tpr[valid][k])
        fpr_k = float(fpr[valid][k])

        # Midpoint-nudge (resolved decision): the ROC threshold is an *attained*
        # value, so shift the cutoff to the midpoint between it and the nearest
        # observed value on the good side. This puts it in the gap (mid-gap on
        # perfect separation) and makes >/< strictness irrelevant. The nudge
        # stays between two adjacent observed values, so no point crosses it —
        # tpr/fpr at the operating point are unchanged.
        if direction == ">":
            lower = scores[scores < thr_v]
            cutoff = float((thr_v + lower.max()) / 2) if lower.size else thr_v
        else:
            upper = scores[scores > thr_v]
            cutoff = float((thr_v + upper.min()) / 2) if upper.size else thr_v

        return {
            "auc": float(separability),
            "direction": direction,
            "cutoff": cutoff,
            "recall": tpr_k,                              # fraction of errors caught
            "specificity": float(1.0 - fpr_k),           # fraction of good kept
            "good_flagged": int(round(fpr_k * len(g))),  # # good wrongly flagged
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "good_n": int(len(g)),
            "error_n": int(len(e)),
        }
