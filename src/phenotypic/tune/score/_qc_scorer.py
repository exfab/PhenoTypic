"""The no-ground-truth Count objective.

Wraps :class:`phenotypic.analysis.ExpectedVsDetectedCount` — for each ``groupby``
unit it compares the detected colony count against an expected count from a
layout frame, yielding ``QC_Count_Metric = |detected - expected| / expected``
(``inf`` when a measurement group has no metadata counterpart). The metric is a
*lower-is-better* divergence in ``[0, ∞)``; ``_threshold_anchored`` folds it to a
**natural goodness** term in ``[0, 1]`` anchored on the check's
``fail_threshold`` (``_TERM_SENSE = HIGHER_BETTER``), which the base
:meth:`Scorer.score_image` then complements into **cost** (lower = better — the
optimizer minimizes).
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

import pandas as pd

from phenotypic.analysis import ExpectedVsDetectedCount

from ._orient import Sense
from ._scorer import Scorer


def _threshold_anchored(metric: float, fail_threshold: float) -> float:
    """Fold a lower-is-better divergence to a natural-goodness value in ``[0, 1]``.

    ``t = exp(-ln2 * metric / fail_threshold)`` — so ``metric == 0`` → ``1.0``,
    ``metric == fail_threshold`` → ``0.5``, ``metric == inf`` → ``0.0``. This is
    the scorer's **natural** (higher-better) value; the base ``score_image``
    complements it into cost.

    Args:
        metric: The non-negative divergence (``|detected-expected|/expected``);
            ``inf`` for an unmatched group.
        fail_threshold: The metric value the check treats as a hard fail; the
            half-score anchor.

    Returns:
        The natural-goodness value in ``[0, 1]`` (higher = better; the base
        complements it to cost).
    """
    if not math.isfinite(metric):
        return 0.0
    if metric <= 0.0:
        return 1.0
    return math.exp(-math.log(2.0) * metric / fail_threshold)


def fold_expected_vs_detected_count(
    check: ExpectedVsDetectedCount, measurements: pd.DataFrame
) -> float:
    """Fold a count check's per-group divergence into one natural-goodness value.

    The shared count-tier reduction used by :class:`QCScorer`,
    :class:`SupervisedScorer` (count tier), and :class:`ReferenceFreeScorer`
    (optional count term): run ``check`` over ``measurements``, anchor each
    ``groupby`` group's ``QC_Count_Metric`` on the check's ``fail_threshold`` via
    :func:`_threshold_anchored`, and average across groups — turning the
    lower-is-better divergence into a natural-goodness ``[0, 1]`` value (the base
    ``score_image`` complements it to cost). An empty or ``None`` frame scores
    ``0.0`` (every caller's empty-frame floor).

    Args:
        check: A configured :class:`ExpectedVsDetectedCount` count check.
        measurements: The candidate pipeline's measurement frame.

    Returns:
        The averaged anchored count value in ``[0, 1]`` (natural goodness, higher
        = better; the base complements it to cost); ``0.0`` for an empty or
        ``None`` frame.
    """
    if measurements is None or len(measurements) == 0:
        return 0.0
    augmented = check.analyze(measurements)
    metric_col = check.metric_col()
    per_group = augmented.groupby(check.groupby, dropna=False)[metric_col].first()
    fail = float(check.fail_threshold)
    return float(
        per_group.map(lambda m: _threshold_anchored(float(m), fail)).mean()
    )


class QCScorer(Scorer):
    """Count-only quality objective backed by ``ExpectedVsDetectedCount``.

    Args:
        check: A configured count check. **Configure it from a metadata path**
            (``metadata="layout.csv"``) so the layout path persists under the
            check's ``metadata`` field and the scorer round-trips through
            ``tuning_spec.json``; a check built from an in-memory frame cannot
            be rebuilt from JSON.

    Examples:
        >>> import pandas as pd
        >>> from phenotypic.analysis import ExpectedVsDetectedCount
        >>> from phenotypic.tune.score import QCScorer
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
        >>> # a perfect 96-well count match → zero cost
        >>> round(scorer.score_image(None, measured)["Count"], 3)
        0.0
    """

    term_name: ClassVar[str] = "Count"
    #: The folded count term is a bounded [0,1] goodness; the base complements
    #: it (1 - value) into cost.
    _TERM_SENSE = Sense.HIGHER_BETTER

    check: ExpectedVsDetectedCount

    def availability(self) -> bool:
        """``True`` when the check resolved a non-empty layout frame.

        Reads the check's *resolved* layout frame (``_metadata``) rather
        than the raw ``metadata`` field, which may hold a path string.
        """
        return not self.check._metadata.empty

    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Return ``{"Count": t}`` — the natural per-image count goodness.

        Runs the count check on ``measurements``, normalizes each group's
        ``QC_Count_Metric`` via :func:`_threshold_anchored`, and averages across
        groups (a single-plate frame has one group, so the mean is that group's
        score). An empty frame scores ``0.0``. The base :meth:`score_image`
        complements this natural goodness in ``[0, 1]`` to cost.

        Args:
            image: Unused (the count objective reads only the frame).
            measurements: The candidate pipeline's measurement frame.

        Returns:
            ``{"Count": <natural goodness in [0, 1]>}`` (the base complements it
            to cost).
        """
        return {
            self.term_name: fold_expected_vs_detected_count(
                self.check, measurements
            )
        }
