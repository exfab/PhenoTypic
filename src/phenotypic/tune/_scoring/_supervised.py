"""The supervised (ground-truth) scoring objective — Phase 4 chunk B (4.4).

``SupervisedScorer`` scores a candidate segmentation **against ground-truth
annotations** resolved through a path-configured :class:`GroundTruthMasks`
loader. It is **modality-tiered** (supervised-scorers §3): the GT source's
modality decides which tier runs, and a tier emits exactly the term(s) it can
honestly compute, so the scorer degrades gracefully as annotations get cheaper:

* **mask tier** (``gt.modality() == "mask"``): resolve the per-image GT mask,
  turn it into an **instance-label** array per the loader's ``gt_format``
  (``"instance"`` GT is used as-is; ``"binary"`` foreground GT is split into
  connected components within each **geometric, detection-independent** grid
  cell, so spatially-separated colonies in one cell become distinct GT instances
  while the full GT extent is preserved), match predicted objects to those GT
  instances (``"binary"`` always via global IoU-greedy so missed/over-extended
  GT is scored honestly; ``"instance"`` via the configured strategy — per-grid-
  cell on a ``GridImage`` or IoU-greedy), score each matched pair with the chosen
  *single* region metric (Dice **xor** IoU — never both, §1 composition note),
  and **macro-average per image** into the ``"Region"`` term.
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
from skimage import measure

from phenotypic.analysis import ExpectedVsDetectedCount

from ._gt_loader import GroundTruthMasks
from ._matching import MatchPair, match_iou_greedy, match_per_grid_cell
from ._metrics import dice, iou
from ._orient import Sense
from ._qc_scorer import fold_expected_vs_detected_count
from ._scorer import Scorer

#: The single region-overlap metric — Dice **xor** IoU (never both: they rank
#: candidates identically, so a panel carrying both adds no information,
#: supervised-scorers §1 composition note). A type-only closed set.
RegionMetric: TypeAlias = Literal["dice", "iou"]

#: The object-matching strategy — per-grid-cell (the arrayed-plate default) or
#: IoU-greedy (the non-gridded fallback). A type-only closed set.
MatchStrategy: TypeAlias = Literal["grid_cell", "iou_greedy"]

#: Sentinel cell id for pixels **outside the grid** in the geometric cell map
#: (margins beyond the first/last grid edge). Excluded when deriving per-cell
#: binary-GT instances so off-grid foreground is never split into a GT object.
#: Grid cells themselves use the row-major id ``row * ncols + col`` (``>= 0``).
_OUTSIDE_GRID = -1


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
    #: Both tier terms (Region = Dice/IoU; CountMAE = folded count goodness) are
    #: bounded [0,1] goodness; the base complements them into cost.
    _TERM_SENSE = Sense.HIGHER_BETTER

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

    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Natural goodness terms by GT modality tier (the base complements to cost).

        Dispatches on :meth:`GroundTruthMasks.modality`:

        * **mask** — resolve the per-image GT mask via
          :meth:`GroundTruthMasks.masks_for` (keyed by ``image.name``); when it
          resolves, match predicted vs. GT objects (:mod:`._matching`), score
          each matched pair with the chosen single region metric
          (:mod:`._metrics`), and **macro-average per image** into ``"Region"``.
          An unresolved name yields no term (nothing to score).
        * **count** — reuse the configured :attr:`count_check`
          (:class:`ExpectedVsDetectedCount`) to get each group's normalized
          count divergence, fold it to a natural goodness ``[0, 1]`` value
          anchored on ``fail_threshold``, and emit ``"CountMAE"``. The base
          :meth:`score_image` complements each term to cost.
        * **none** — abstain (empty mapping).

        Args:
            image: The processed image — a ``GridImage`` (duck-typed: ``name``,
                ``objmap``, ``grid.get_section_map``) on the mask tier; unused on
                the count tier.
            measurements: The candidate pipeline's measurement frame.

        Returns:
            A mapping with the runnable tier's natural goodness term —
            ``{"Region": ...}`` (mask), ``{"CountMAE": ...}`` (count) — or
            ``{}`` when no tier can score this image. Values are natural goodness
            in ``[0, 1]``; the base ``score_image`` orients them to cost.
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

        The loaded GT is first turned into an **instance-label** array per the
        loader's :attr:`GroundTruthMasks.gt_format` (:meth:`_gt_instances`) —
        ``"instance"`` GT is used as-is; ``"binary"`` GT is split into per-cell
        connected components on a **geometric** (detection-independent) cell map.
        That instance array then drives matching and per-pair scoring.

        Matching is dispatched by format:

        * ``"binary"`` always uses :func:`match_iou_greedy` (pure global IoU, no
          section-map dependency) so a GT instance in a cell the prediction
          missed stays **unmatched** (scored ``0.0`` = correct miss penalty) and
          an over-extended GT is **not clipped** to the prediction's pixels.
        * ``"instance"`` honours the configured :attr:`match_strategy` (the
          arrayed-plate ``"grid_cell"`` default, or ``"iou_greedy"``) — the GT
          already carries authoritative per-object labels, so the grid prior is
          safe to use.

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
        gt_instances = self._gt_instances(image, gt_mask)
        if self.gt.gt_format == "binary":
            # Pure global IoU: detection-independent, so misses/over-extension
            # are scored honestly (no clipping to the prediction's footprint).
            pairs = match_iou_greedy(
                image.objmap[:], gt_instances, tau=self.iou_tau
            )
        else:
            pairs = self._match(image, gt_instances)
        score = self._macro_average_region(image, gt_instances, pairs)
        return {self.region_term_name: score}

    def _gt_instances(self, image: Any, gt_mask: Any) -> np.ndarray:
        """Resolve the loaded GT into an integer **instance-label** array.

        Dispatches on the loader's :attr:`GroundTruthMasks.gt_format`:

        * ``"instance"`` — the GT is already an integer instance-label array
          (each colony a distinct positive label); it is used **as-is** (the
          loader now preserves on-disk integer labels, so the existing matcher
          path is correct).
        * ``"binary"`` — the GT is a foreground mask; per-cell instances are
          derived from a **geometric, detection-independent** cell map
          (:meth:`_geometric_cell_map_or_none`, built from the grid *edges* — not
          from ``get_section_map()``, which is prediction-derived and would clip
          the GT to the prediction's pixels and erase entirely-missed cells).
          With ``foreground = gt > 0``, each grid cell is connected-component
          labelled independently on ``foreground & (cell_map == cell_id)``, with
          a running label offset so every component is globally unique. This
          **splits spatially-separated colonies in one cell into distinct
          instances automatically** (only truly-touching clumps stay merged —
          those need ``"instance"`` GT) and **preserves the full GT extent** so
          the downstream IoU reflects under-segmentation honestly. An image with
          no grid (not a ``GridImage``) falls back to a single global
          :func:`skimage.measure.label` of the foreground.

        Args:
            image: The processed image (a ``GridImage`` exposes
                ``grid.get_row_edges()`` / ``get_col_edges()``; a plain image
                does not).
            gt_mask: The loaded GT array (integer labels or a binary foreground,
                dtype preserved by :meth:`GroundTruthMasks.masks_for`).

        Returns:
            An integer instance-label array the same shape as ``gt_mask``
            (``0`` = background, positive = distinct GT colony instances).
        """
        gt = np.asarray(gt_mask)
        if self.gt.gt_format == "instance":
            return gt.astype(np.int64, copy=False)
        # gt_format == "binary": derive per-cell connected-component instances on
        # a GEOMETRIC (detection-independent) cell map.
        foreground = gt > 0
        cell_map = self._geometric_cell_map_or_none(image, foreground.shape)
        if cell_map is None:
            # No grid → a single global connected-components labelling.
            return measure.label(foreground).astype(np.int64, copy=False)
        gt_instances = np.zeros(foreground.shape, dtype=np.int64)
        running_max = 0
        # Off-grid margin (_OUTSIDE_GRID = -1) is excluded so foreground beyond
        # the grid edges is never promoted into a spurious instance.
        cell_ids = [
            int(c) for c in np.unique(cell_map) if int(c) != _OUTSIDE_GRID
        ]
        for cell_id in cell_ids:
            cell_fg = foreground & (cell_map == cell_id)
            if not cell_fg.any():
                continue
            cell_labels = measure.label(cell_fg)
            cell_max = int(cell_labels.max())
            if cell_max == 0:
                continue
            # Offset so this cell's components are globally unique.
            gt_instances[cell_labels > 0] = (
                cell_labels[cell_labels > 0] + running_max
            )
            running_max += cell_max
        return gt_instances

    @staticmethod
    def _geometric_cell_map_or_none(
        image: Any, shape: tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """A per-pixel, **detection-independent** grid-cell-id map from grid edges.

        Builds the cell map purely from the grid line positions
        (``grid.get_row_edges()`` / ``get_col_edges()``, each strictly increasing
        with length ``n+1``) — never from ``get_section_map()`` (which is derived
        from the predicted ``objmap`` and so only labels detected-colony pixels).
        Each image row/column is digitized into its grid row/col; pixels inside
        the grid get the row-major cell id ``row * ncols + col`` (``>= 0``) and
        pixels in the margins beyond the first/last edge get
        :data:`_OUTSIDE_GRID` (``-1``).

        Args:
            image: The processed image; only a ``GridImage`` exposes the grid
                edge accessors and ``grid.nrows`` / ``grid.ncols``.
            shape: The GT mask's ``(H, W[, ...])`` shape; the first two entries
                are the image height and width the map is built over.

        Returns:
            An ``(H, W)`` integer cell-id map (cells ``>= 0`` row-major, margin
            ``-1``), or ``None`` when the image carries no usable grid (so the
            binary-GT derivation falls back to a single global labelling).
        """
        grid = getattr(image, "grid", None)
        if grid is None:
            return None
        row_getter = getattr(grid, "get_row_edges", None)
        col_getter = getattr(grid, "get_col_edges", None)
        nrows = getattr(grid, "nrows", None)
        ncols = getattr(grid, "ncols", None)
        if (
            row_getter is None
            or col_getter is None
            or nrows is None
            or ncols is None
        ):
            return None
        height, width = int(shape[0]), int(shape[1])
        row_edges = np.asarray(row_getter())
        col_edges = np.asarray(col_getter())
        # Digitize each pixel index into its grid row/col; subtract 1 so the
        # first interior bin is 0. Pixels before edge[0] -> -1, pixels at/after
        # edge[-1] -> n (both filtered by the valid mask below).
        row_bin = np.digitize(np.arange(height), row_edges) - 1
        col_bin = np.digitize(np.arange(width), col_edges) - 1
        valid_row = (row_bin >= 0) & (row_bin < int(nrows))
        valid_col = (col_bin >= 0) & (col_bin < int(ncols))
        valid = valid_row[:, None] & valid_col[None, :]
        cell_id = row_bin[:, None] * int(ncols) + col_bin[None, :]
        return np.where(valid, cell_id, _OUTSIDE_GRID).astype(np.int64)

    def _match(self, image: Any, gt_instances: Any) -> list[MatchPair]:
        """Pair predicted vs. GT objects by the configured strategy.

        Args:
            image: The ``GridImage`` carrying the predicted ``objmap`` (and, for
                ``"grid_cell"``, the ``grid.get_section_map``).
            gt_instances: The per-image ground-truth **instance-label** array
                (from :meth:`_gt_instances`).

        Returns:
            The ``(pred_label, gt_label)`` pairs from :mod:`._matching`.
        """
        if self.match_strategy == "grid_cell":
            return match_per_grid_cell(image, gt_instances)
        return match_iou_greedy(image.objmap[:], gt_instances, tau=self.iou_tau)

    def _macro_average_region(
        self, image: Any, gt_instances: Any, pairs: list[MatchPair]
    ) -> float:
        """Macro-average the single region metric over every match pair.

        Each pair contributes the chosen metric between the predicted object's
        mask and the GT object's mask; an unmatched object (``None`` on a side)
        is scored against an empty mask, so a missed or spurious object scores
        ``0.0`` (a non-empty vs. empty overlap, §A.5). With no objects on either
        side the image is perfectly (vacuously) scored ``1.0``.

        Args:
            image: The ``GridImage`` carrying the predicted ``objmap``.
            gt_instances: The per-image ground-truth **instance-label** array
                (from :meth:`_gt_instances`); must share the predicted objmap's
                shape.
            pairs: The match pairs from :meth:`_match`.

        Returns:
            The macro-averaged region metric in ``[0, 1]`` (higher = better).

        Raises:
            ValueError: If the predicted objmap and ``gt_instances`` differ in
                shape (they must be pixel-aligned for per-object overlap).
        """
        if not pairs:
            return 1.0  # nothing predicted, nothing annotated → vacuous match
        pred = np.asarray(image.objmap[:])
        gt = np.asarray(gt_instances)
        if pred.shape != gt.shape:
            raise ValueError(
                "predicted objmap and GT instances must share a shape for "
                f"per-object overlap; got pred {pred.shape} vs. GT {gt.shape}"
            )
        metric = dice if self.region_metric == "dice" else iou
        # Build the empty fallback from the *pred* shape (pred and gt are
        # shape-checked equal above, so either works; pred is the candidate).
        empty = np.zeros(pred.shape, dtype=bool)
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
        return {
            self.count_term_name: fold_expected_vs_detected_count(
                self.count_check, measurements
            )
        }
