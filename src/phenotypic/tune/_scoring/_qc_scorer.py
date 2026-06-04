"""The no-ground-truth Count objective.

Wraps :class:`phenotypic.analysis.ExpectedVsDetectedCount` — for each ``groupby``
unit it compares the detected colony count against an expected count from a
layout frame, yielding ``QC_Count_Metric = |detected - expected| / expected``
(``inf`` when a measurement group has no metadata counterpart). The metric is a
*lower-is-better* divergence in ``[0, ∞)``; ``_threshold_anchored`` flips and
normalizes it to a *higher-is-better* term in ``[0, 1]`` anchored on the check's
``fail_threshold``.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

import pandas as pd

from phenotypic.analysis import ExpectedVsDetectedCount

from ._scorer import Scorer


def _threshold_anchored(metric: float, fail_threshold: float) -> float:
    """Map a lower-is-better divergence to a higher-is-better score in ``[0, 1]``.

    ``t = exp(-ln2 * metric / fail_threshold)`` — so ``metric == 0`` → ``1.0``,
    ``metric == fail_threshold`` → ``0.5``, ``metric == inf`` → ``0.0``.

    Args:
        metric: The non-negative divergence (``|detected-expected|/expected``);
            ``inf`` for an unmatched group.
        fail_threshold: The metric value the check treats as a hard fail; the
            half-score anchor.

    Returns:
        The normalized score in ``[0, 1]`` (higher = better).
    """
    if not math.isfinite(metric):
        return 0.0
    if metric <= 0.0:
        return 1.0
    return math.exp(-math.log(2.0) * metric / fail_threshold)


class QCScorer(Scorer):
    """Count-only quality objective backed by ``ExpectedVsDetectedCount``.

    Args:
        check: A configured count check. **Configure it from a metadata path**
            (``metadata="layout.csv"``) so ``metadata_source`` persists and the
            scorer round-trips through ``tuning_spec.json``; a check built from
            an in-memory frame cannot be rebuilt from JSON.

    Examples:
        >>> import pandas as pd
        >>> from phenotypic.analysis import ExpectedVsDetectedCount
        >>> from phenotypic.tune import QCScorer
        >>> layout = pd.DataFrame(
        ...     {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
        ... )
        >>> scorer = QCScorer(
        ...     check=ExpectedVsDetectedCount(
        ...         metadata=layout, groupby=["Metadata_ImageName"]
        ...     )
        ... )
        >>> measured = pd.DataFrame(
        ...     {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
        ... )
        >>> round(scorer.score_image(None, measured)["Count"], 3)
        1.0
    """

    term_name: ClassVar[str] = "Count"

    check: ExpectedVsDetectedCount

    def availability(self) -> bool:
        """``True`` when the check resolved a non-empty layout frame."""
        return not self.check.metadata.empty

    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Return ``{"Count": t}`` — the normalized per-image count score.

        Runs the count check on ``measurements``, normalizes each group's
        ``QC_Count_Metric`` via :func:`_threshold_anchored`, and averages across
        groups (a single-plate frame has one group, so the mean is that group's
        score). An empty frame scores ``0.0``.

        Args:
            image: Unused (the count objective reads only the frame).
            measurements: The candidate pipeline's measurement frame.

        Returns:
            ``{"Count": <score in [0, 1]>}`` (higher = better).
        """
        if measurements is None or len(measurements) == 0:
            return {self.term_name: 0.0}
        augmented = self.check.analyze(measurements)
        metric_col = self.check.metric_col()
        per_group = augmented.groupby(self.check.groupby, dropna=False)[
            metric_col
        ].first()
        fail = float(self.check.fail_threshold)
        score = float(
            per_group.map(lambda m: _threshold_anchored(float(m), fail)).mean()
        )
        return {self.term_name: score}
