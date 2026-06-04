"""The supervised (ground-truth) scoring objective — Phase 4 chunk B (4.4).

``SupervisedScorer`` scores a candidate segmentation **against ground-truth
annotations** resolved through a path-configured :class:`GroundTruthMasks`
loader. It is **modality-tiered** (supervised-scorers §3): the GT source's
modality decides which tier runs, and a tier emits exactly the term(s) it can
honestly compute, so the scorer degrades gracefully as annotations get cheaper:

* **mask tier** (``gt.modality() == "mask"``): resolve the per-image GT mask,
  match predicted objects to GT objects (per-grid-cell on a ``GridImage``, or
  IoU-greedy as a non-gridded fallback), score each matched pair with the
  chosen *single* region metric (Dice **xor** IoU — never both, §1 composition
  note), and **macro-average per image** into the ``"Region"`` term.
* **count tier** (``gt.modality() == "count"``): **reuse**
  :class:`phenotypic.analysis.ExpectedVsDetectedCount` (the master's count QC
  check — do *not* re-implement counting, §1 "reuse, don't duplicate") to get
  each group's ``|detected − expected| / expected`` divergence, fold it to a
  *higher-is-better* ``[0, 1]`` score anchored on the check's ``fail_threshold``,
  and emit the ``"CountMAE"`` term.
* **none tier** (abstain): no resolvable GT → :meth:`availability` is ``False``
  and the engine degrades to the configured fallback objective.

The two region metrics are **monotonically related** (``Dice = 2·IoU/(1+IoU)``)
so they rank candidates identically; the scorer therefore carries exactly one
(``region_metric``), guarded by a ``field_validator``.

# TODO(DEFERRED-WORK §1): validate against real annotated plates. v1 ships the
# modality-tiered term machinery (mask region-overlap + count-MAE reuse) and the
# availability gate; the numeric correctness of the region term vs. a real
# annotated calibration set — and the choice of region metric / matching τ that
# best tracks the visual optimum — is deferred until such a set exists. No
# numeric-vs-real-GT test runs here (tests pin construction, term shape,
# availability tiers, count-check reuse, and round-trip only).
"""
from __future__ import annotations

from typing import Any, ClassVar, Literal, Optional, TypeAlias

import numpy as np
import pandas as pd
from pydantic import ConfigDict, field_validator

from phenotypic.analysis import ExpectedVsDetectedCount

from ._gt_loader import GroundTruthMasks
from ._matching import MatchPair, match_iou_greedy, match_per_grid_cell
from ._metrics import dice, iou
from ._qc_scorer import _threshold_anchored
from ._scorer import Scorer

#: The single region-overlap metric — Dice **xor** IoU (never both: they rank
#: candidates identically, so a panel carrying both adds no information,
#: supervised-scorers §1 composition note). A type-only closed set.
RegionMetric: TypeAlias = Literal["dice", "iou"]

#: The object-matching strategy — per-grid-cell (the arrayed-plate default) or
#: IoU-greedy (the non-gridded fallback). A type-only closed set.
MatchStrategy: TypeAlias = Literal["grid_cell", "iou_greedy"]


class SupervisedScorer(Scorer):
    """Ground-truth segmentation/count objective, tiered by GT modality.

    The :attr:`gt` loader's :meth:`GroundTruthMasks.modality` selects the tier
    that runs: a directory of per-image masks → the mask (region-overlap) tier;
    a ``.csv``/``.parquet`` count table (paired with a configured
    :attr:`count_check`) → the count tier; nothing resolvable → abstain. Each
    runnable tier contributes exactly one term — ``"Region"`` (mask) or
    ``"CountMAE"`` (count) — so :meth:`score_image` returns a tier-appropriate
    mapping (empty when no tier can score a given image).

    Args:
        gt: The path-configured ground-truth loader. Its ``gt_masks_source``
            is the serializable handle and its :meth:`GroundTruthMasks.modality`
            drives tier selection and :meth:`availability`.
        region_metric: The single region-overlap metric for the mask tier —
            ``"dice"`` (default) or ``"iou"``. The two rank identically; carry
            exactly one (Dice **xor** IoU, supervised-scorers §1).
        match_strategy: How predicted objects are paired with GT objects on the
            mask tier — ``"grid_cell"`` (default; the arrayed plate's grid is the
            spatial prior, no τ) or ``"iou_greedy"`` (the non-gridded fallback,
            accepting pairs with IoU > :attr:`iou_tau`).
        iou_tau: The IoU acceptance threshold for ``"iou_greedy"`` matching
            (ignored for ``"grid_cell"``). Default ``0.5`` gives a provably
            one-to-one assignment.
        count_check: The path-configured :class:`ExpectedVsDetectedCount` reused
            for the count tier (do not re-implement counting). Required for the
            count tier to run; ``None`` makes the count tier abstain.

    Raises:
        pydantic.ValidationError: If ``region_metric`` is not exactly one of
            ``"dice"`` / ``"iou"`` (the Dice-xor-IoU guard).

    Examples:
        Construct against a count-table GT source and inspect availability — the
        synthetic plate is the runnable doctest target (construction + the
        modality-tiered availability gate only; numeric GT scoring is deferred):

        >>> import tempfile
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from phenotypic.analysis import ExpectedVsDetectedCount
        >>> from phenotypic.tune import GroundTruthMasks, SupervisedScorer
        >>> tmp = Path(tempfile.mkdtemp())
        >>> counts = tmp / "counts.csv"
        >>> _ = pd.DataFrame(
        ...     {"Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
        ...      "Object_Label": list(range(96))}
        ... ).to_csv(counts, index=False)
        >>> scorer = SupervisedScorer(
        ...     gt=GroundTruthMasks(gt_masks_source=counts),
        ...     count_check=ExpectedVsDetectedCount(
        ...         metadata=str(counts), groupby=["Metadata_ImageName"]
        ...     ),
        ... )
        >>> scorer.gt.modality()
        'count'
        >>> scorer.availability()  # count tier runnable (check configured)
        True

        A sourceless loader abstains, so the engine degrades to the fallback:

        >>> SupervisedScorer(gt=GroundTruthMasks(gt_masks_source=None)).availability()
        False
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: The mask-tier region-overlap term name.
    region_term_name: ClassVar[str] = "Region"
    #: The count-tier folded-divergence term name.
    count_term_name: ClassVar[str] = "CountMAE"

    gt: GroundTruthMasks
    region_metric: RegionMetric = "dice"
    match_strategy: MatchStrategy = "grid_cell"
    iou_tau: float = 0.5
    count_check: Optional[ExpectedVsDetectedCount] = None

    @field_validator("region_metric", mode="before")
    @classmethod
    def _exactly_one_region_metric(cls, value: object) -> RegionMetric:
        """Guard the single-region-metric contract (Dice **xor** IoU).

        The two region metrics rank candidates identically, so the panel carries
        exactly one. This rejects anything outside the closed
        :data:`RegionMetric` set (e.g. a ``"both"`` sentinel) at construction.

        Args:
            value: The raw ``region_metric`` input.

        Returns:
            The validated metric name (``"dice"`` or ``"iou"``).

        Raises:
            ValueError: If ``value`` is not exactly ``"dice"`` or ``"iou"``.
        """
        if value == "dice":
            return "dice"
        if value == "iou":
            return "iou"
        raise ValueError(
            "region_metric must be exactly one of 'dice' or 'iou' "
            "(Dice xor IoU — they rank identically, so carry only one); "
            f"got {value!r}"
        )

    def availability(self) -> bool:
        """Whether some GT tier can run as configured (modality-tiered).

        Reads :meth:`GroundTruthMasks.modality`: the **mask** tier is available
        whenever a mask source resolves; the **count** tier is available only
        when a :attr:`count_check` is also configured (the tier reuses it); the
        **none** modality abstains. When this returns ``False`` the engine
        degrades to the configured fallback objective.

        Returns:
            ``True`` if the mask tier (mask modality) or the count tier (count
            modality **and** a configured ``count_check``) can run; ``False``
            otherwise.
        """
        modality = self.gt.modality()
        if modality == "mask":
            return True
        if modality == "count":
            return self.count_check is not None
        return False

    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Score one image against its ground truth, by GT modality tier.

        Dispatches on :meth:`GroundTruthMasks.modality`:

        * **mask** — resolve the per-image GT mask via
          :meth:`GroundTruthMasks.masks_for` (keyed by ``image.name``); when it
          resolves, match predicted vs. GT objects (:mod:`._matching`), score
          each matched pair with the chosen single region metric
          (:mod:`._metrics`), and **macro-average per image** into ``"Region"``.
          An unresolved name yields no term (nothing to score).
        * **count** — reuse the configured :attr:`count_check`
          (:class:`ExpectedVsDetectedCount`) to get each group's normalized
          count divergence, fold it to a higher-is-better ``[0, 1]`` score
          anchored on ``fail_threshold``, and emit ``"CountMAE"``.
        * **none** — abstain (empty mapping).

        Args:
            image: The processed image — a ``GridImage`` (duck-typed: ``name``,
                ``objmap``, ``grid.get_section_map``) on the mask tier; unused on
                the count tier.
            measurements: The candidate pipeline's measurement frame.

        Returns:
            A mapping with the runnable tier's term — ``{"Region": ...}`` (mask),
            ``{"CountMAE": ...}`` (count) — or ``{}`` when no tier can score this
            image (no GT mask for the name, or count tier without a check).
        """
        modality = self.gt.modality()
        if modality == "mask":
            return self._score_mask_tier(image, measurements)
        if modality == "count":
            return self._score_count_tier(measurements)
        return {}

    # ------------------------------------------------------------------ #
    # mask tier — match → per-pair region metric → macro-average
    # ------------------------------------------------------------------ #
    def _score_mask_tier(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Macro-average the per-pair region metric into the ``"Region"`` term.

        Args:
            image: The ``GridImage`` whose ``name`` keys the GT mask and whose
                ``objmap`` / ``grid`` drive matching.
            measurements: Unused (the mask tier reads the object map, not the
                frame) — accepted for signature parity.

        Returns:
            ``{"Region": <macro-averaged metric in [0, 1]>}`` when the image's GT
            mask resolves; ``{}`` when no GT mask exists for ``image.name``.
        """
        del measurements  # mask tier reads image.objmap, not the frame
        gt_mask = self.gt.masks_for(getattr(image, "name", ""))
        if gt_mask is None:
            return {}
        pairs = self._match(image, gt_mask)
        score = self._macro_average_region(image, gt_mask, pairs)
        return {self.region_term_name: score}

    def _match(self, image: Any, gt_mask: Any) -> list[MatchPair]:
        """Pair predicted vs. GT objects by the configured strategy.

        Args:
            image: The ``GridImage`` carrying the predicted ``objmap`` (and, for
                ``"grid_cell"``, the ``grid.get_section_map``).
            gt_mask: The per-image ground-truth label/boolean array.

        Returns:
            The ``(pred_label, gt_label)`` pairs from :mod:`._matching`.
        """
        if self.match_strategy == "grid_cell":
            return match_per_grid_cell(image, gt_mask)
        return match_iou_greedy(image.objmap[:], gt_mask, tau=self.iou_tau)

    def _macro_average_region(
        self, image: Any, gt_mask: Any, pairs: list[MatchPair]
    ) -> float:
        """Macro-average the single region metric over every match pair.

        Each pair contributes the chosen metric between the predicted object's
        mask and the GT object's mask; an unmatched object (``None`` on a side)
        is scored against an empty mask, so a missed or spurious object scores
        ``0.0`` (a non-empty vs. empty overlap, §A.5). With no objects on either
        side the image is perfectly (vacuously) scored ``1.0``.

        Args:
            image: The ``GridImage`` carrying the predicted ``objmap``.
            gt_mask: The per-image ground-truth label array.
            pairs: The match pairs from :meth:`_match`.

        Returns:
            The macro-averaged region metric in ``[0, 1]`` (higher = better).
        """
        if not pairs:
            return 1.0  # nothing predicted, nothing annotated → vacuous match
        pred = np.asarray(image.objmap[:])
        gt = np.asarray(gt_mask)
        metric = dice if self.region_metric == "dice" else iou
        empty = np.zeros_like(gt, dtype=bool)
        scores: list[float] = []
        for pred_label, gt_label in pairs:
            pred_obj = (pred == pred_label) if pred_label is not None else empty
            gt_obj = (gt == gt_label) if gt_label is not None else empty
            scores.append(metric(pred_obj, gt_obj))
        return float(sum(scores) / len(scores))

    # ------------------------------------------------------------------ #
    # count tier — reuse ExpectedVsDetectedCount (do NOT re-implement)
    # ------------------------------------------------------------------ #
    def _score_count_tier(self, measurements: pd.DataFrame) -> dict[str, float]:
        """Fold the reused count divergence into the ``"CountMAE"`` term.

        Reuses the configured :attr:`count_check`
        (:class:`ExpectedVsDetectedCount`) exactly as :class:`QCScorer` does —
        runs the check, anchors each group's ``QC_Count_Metric`` on the check's
        ``fail_threshold`` via :func:`_threshold_anchored`, and averages across
        groups — turning the lower-is-better divergence into a higher-is-better
        ``[0, 1]`` score. An empty frame, or a count tier with no configured
        check, scores ``0.0``.

        Args:
            measurements: The candidate pipeline's measurement frame.

        Returns:
            ``{"CountMAE": <score in [0, 1]>}`` (higher = better).
        """
        if self.count_check is None:
            return {self.count_term_name: 0.0}
        if measurements is None or len(measurements) == 0:
            return {self.count_term_name: 0.0}
        augmented = self.count_check.analyze(measurements)
        metric_col = self.count_check.metric_col()
        per_group = augmented.groupby(self.count_check.groupby, dropna=False)[
            metric_col
        ].first()
        fail = float(self.count_check.fail_threshold)
        score = float(
            per_group.map(lambda m: _threshold_anchored(float(m), fail)).mean()
        )
        return {self.count_term_name: score}
