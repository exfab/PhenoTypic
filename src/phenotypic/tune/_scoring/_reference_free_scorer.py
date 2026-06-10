"""The reference-free (no-ground-truth) segmentation-quality objective.

``ReferenceFreeScorer`` scores a candidate segmentation **without a
ground-truth mask** using a lean set of domain proxy terms drawn from the
companion catalogue
(``docs/superpowers/specs/param-sweep-redesign/reference-free-segmentation-metrics.md``):

* **ShapeRegularity** — mean of per-colony shape-prior plausibility
  (``Shape_Solidity``, ``Shape_Circularity``, ``1 − Shape_Eccentricity``),
  reusing the public ``phenotypic.schema`` measurement columns (family C.2).
* **Contrast** — Otsu between-class separation ``η = σ²_B / σ²_T`` of the
  image's foreground vs. background grayscale (family A.3); a fixed-normalized
  ``[0, 1]`` separation, *not* a min–max-over-the-grid score.
* **SizeCV** — within-replicate coefficient of variation of ``Size_Area``,
  folded by the bounded ``1 / (1 + CV)`` transform (family C.3).
* **Count** — *optional* expected-vs-detected grid count, reused from
  ``QCScorer`` when a path-configured ``count_check`` is supplied (family C.6).

Every term is **fixed-normalized** to ``[0, 1]`` (higher = better) so the
optimum cannot migrate when the parameter grid's endpoints change — the "Böck
trap" (``§B.3``) that min–max-over-the-tested-set normalization falls into.

The scorer is **gated** behind meta-validation (``D1``): :meth:`meta_validate`
correlates the proxy against ground truth and caches an enable/abstain flag;
:meth:`availability` is the cheap cached-boolean read. Until the gate runs and
passes, :meth:`availability` returns ``False`` so the engine **fails safe** by
degrading to :class:`QCScorer`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, PrivateAttr

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.schema import SHAPE, SIZE

from ._orient import Sense
from ._qc_scorer import fold_expected_vs_detected_count
from ._scorer import Scorer

#: Spearman ρ at or above which the proxy is trusted enough to *enable* the
#: scorer (engineering bar, reference-free-metrics §E.4 — inference, not a
#: cited cutoff).
_ENABLE_RHO: float = 0.7
#: Spearman ρ at or above which fully *unattended* auto-tuning is allowed.
_UNATTENDED_RHO: float = 0.8


def _clamp01(value: float) -> float:
    """Clamp a scalar into ``[0, 1]`` (``NaN`` → ``0.0``).

    Fixed normalization with hard external bounds — the anti-gaming remedy for
    the Böck instability (``§B.3``): the unit interval is decoupled from the
    search grid. Defensive against measurement quirks (e.g. the synthetic
    plate's ``Shape_Solidity`` exceeding ``1``).

    Args:
        value: Any real scalar.

    Returns:
        ``value`` confined to ``[0, 1]``; ``0.0`` when ``value`` is ``NaN``.
    """
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _bounded_inverse(dispersion: float) -> float:
    """Fold a non-negative dispersion to ``[0, 1]`` via ``1 / (1 + x)``.

    Chen & Murphy's bounded-score trick (``§C.1``/``§C.3``): ``x == 0`` (no
    dispersion) → ``1.0`` (best), growing dispersion → ``0``, ``inf`` → ``0``.
    Monotone decreasing, so lower variability scores higher.

    Args:
        dispersion: A non-negative spread statistic (e.g. a coefficient of
            variation); ``inf`` for an undefined/degenerate group.

    Returns:
        The bounded score in ``[0, 1]`` (higher = better).
    """
    if not math.isfinite(dispersion):
        return 0.0
    if dispersion <= 0.0:
        return 1.0
    return 1.0 / (1.0 + dispersion)


class ReferenceFreeScorer(Scorer):
    """No-ground-truth proxy objective, gated behind meta-validation.

    The lean proxy set (``§0a``/``§0b``) reuses the public ``phenotypic.schema``
    measurement columns — ``Shape_*`` and ``Size_*`` — for the shape and size
    terms (no geometry is recomputed), reads the image grayscale + mask for the
    contrast term, and optionally reuses :class:`QCScorer`'s expected-vs-detected
    grid count. All terms are fixed-normalized to ``[0, 1]`` (higher = better).

    The scorer is **unavailable until meta-validated**: :meth:`availability`
    returns ``False`` (so the engine degrades to :class:`QCScorer`) until
    :meth:`meta_validate` correlates the proxy against ground truth and caches a
    passing flag.

    Args:
        count_check: An optional configured expected-vs-detected count check —
            **configure it from a metadata path** (``metadata="layout.csv"``)
            so it round-trips through ``tuning_spec.json`` (mirrors
            :class:`QCScorer`). When supplied, a ``"Count"`` term is added.
        replicate_groupby: Optional metadata columns that group sibling
            replicate colonies; the size-CV term is computed *within* each group
            and averaged (``§C.3``). ``None`` treats the whole frame as one
            group.
        gt_masks_source: Optional path to a ground-truth mask source consumed by
            the meta-validation gate (a name→mask mapping, mirroring
            :class:`QCScorer`'s metadata-path discipline so it serializes).
            ``None`` means the gate cannot validate and abstains.
        min_area: Minimum ``Size_Area`` (px) below which a colony is excluded
            from the shape term — perimeter-derived metrics are unreliable at a
            few pixels (``§C`` watch-out (a)). Default ``0`` keeps every colony.

    Examples:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.measure import MeasureShape, MeasureSize
        >>> from phenotypic.tune import ReferenceFreeScorer
        >>> image = load_synth_yeast_plate()
        >>> pipe = ImagePipeline(
        ...     ops=[OtsuDetector()], meas=[MeasureShape(), MeasureSize()]
        ... )
        >>> measurements = pipe.measure(image, apply_post=False)
        >>> scorer = ReferenceFreeScorer()
        >>> sorted(scorer.score_image(image, measurements))
        ['Contrast', 'ShapeRegularity', 'SizeCV']
        >>> scorer.availability()  # fail-safe: gate has not run yet
        False
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: The optional grid-count term name (present iff ``count_check`` is set).
    count_term_name: ClassVar[str] = "Count"

    #: Every proxy term is bounded [0,1] goodness (fixed-normalized); the base
    #: complements each into cost.
    _TERM_SENSE: ClassVar[Sense] = Sense.HIGHER_BETTER

    count_check: Optional[ExpectedVsDetectedCount] = None
    replicate_groupby: Optional[list[str]] = None
    gt_masks_source: Optional[Path] = None
    min_area: int = 0

    #: Run-local cached gate verdict (default not-yet-run → ``False``,
    #: fail-safe). A ``PrivateAttr`` so it never serializes — a reloaded scorer
    #: re-validates and stays unavailable until it does.
    _meta_validated: bool = PrivateAttr(default=False)

    #: The raw proxy-vs-GT Spearman ρ from the last :meth:`meta_validate` run,
    #: kept distinct from the enable verdict so the stricter *unattended* bar
    #: (:data:`_UNATTENDED_RHO`) is checked against the actual correlation rather
    #: than conflated with the 0.7 enable decision. Run-local (``PrivateAttr``,
    #: never serialized); ``-inf`` until the gate has run.
    _last_rho: float = PrivateAttr(default=float("-inf"))

    # ----------------------------------------------------------------- #
    # availability + the meta-validation gate (Task 2)
    # ----------------------------------------------------------------- #
    def availability(self) -> bool:
        """Whether the proxy has passed meta-validation (cheap cached read).

        The cached :attr:`_meta_validated` flag is the only thing read — no GT
        is loaded and no correlation is recomputed. Defaults to ``False``
        (fail-safe) until :meth:`meta_validate` runs and passes, so the engine
        degrades to :class:`QCScorer` rather than trusting an unvalidated proxy.

        Returns:
            ``True`` only after a passing :meth:`meta_validate`; ``False``
            otherwise.
        """
        return self._meta_validated

    def meta_validate(self, gt_images: Any, grid: Any) -> bool:
        """Gate the proxy against ground truth, caching the enable/abstain flag.

        Loads ground-truth masks via :meth:`_load_gt_masks`, computes the
        proxy-vs-GT Spearman rank correlation, and caches
        :attr:`_meta_validated` ``= ρ ≥`` :data:`_ENABLE_RHO`. **Fail-safe**: if
        no GT is configured (or it resolves to nothing), the gate abstains and
        the flag stays ``False``, so the engine keeps falling back to
        :class:`QCScorer`.

        Args:
            gt_images: The annotated calibration images to validate against
                (duck-typed; passed through to :meth:`_load_gt_masks`).
            grid: The candidate parameter grid the proxy is asked to rank
                (its argmax is compared against the true best in the deferred,
                real-GT validation).

        Returns:
            ``True`` if the proxy passed (``ρ ≥`` enable threshold) and the
            scorer is now :meth:`availability`-enabled; ``False`` on abstain.

        # TODO(DEFERRED-WORK §1): validate against real annotated plates — run
        # the proxy-vs-GT correlation on a real GT set and confirm the
        # enable/abstain thresholds (and the argmax test) numerically. v1 ships
        # only the gate machinery + abstain logic; ``_proxy_gt_spearman`` is a
        # structural stub until an annotated calibration set exists.
        """
        masks = self._load_gt_masks(gt_images)
        if not masks:
            # No ground truth → cannot certify the proxy → abstain (fail-safe).
            self._meta_validated = False
            self._last_rho = float("-inf")
            return False
        rho = self._proxy_gt_spearman(gt_images, masks, grid)
        # Store the raw ρ so the stricter unattended bar reads the actual value;
        # the enable verdict and the unattended verdict are distinct thresholds.
        self._last_rho = float(rho)
        self._meta_validated = bool(rho >= _ENABLE_RHO)
        return self._meta_validated

    def is_unattended_safe(self) -> bool:
        """Whether the last gate run cleared the *unattended* auto-tuning bar.

        A strictly stronger read than :meth:`availability`: unattended auto-tuning
        needs ``ρ ≥`` :data:`_UNATTENDED_RHO` (0.8), checked against the **raw**
        correlation :attr:`_last_rho` recorded by :meth:`meta_validate` — *not*
        conflated with the 0.7 enable verdict. Because ``_last_rho`` is ``-inf``
        until the gate runs, this returns ``False`` before meta-validation and
        whenever the gate abstained (no GT), so it implies :meth:`availability`
        but a proxy that only cleared the 0.7 enable bar is **not** unattended-safe.

        Returns:
            ``True`` only when the last gate run's ρ reached the unattended
            threshold (``≥`` :data:`_UNATTENDED_RHO`); ``False`` otherwise.
        """
        return self._last_rho >= _UNATTENDED_RHO

    def _load_gt_masks(self, gt_images: Any) -> dict[str, Any]:
        """Resolve the configured ground-truth mask source to a name→mask map.

        Mirrors :class:`QCScorer`'s metadata-path discipline: the source is a
        serializable :attr:`gt_masks_source` path, resolved here at validation
        time. ``None`` (no GT configured) resolves to an empty mapping so
        :meth:`meta_validate` abstains.

        Args:
            gt_images: The annotated images whose names key the returned map
                (unused by the v1 structural loader beyond presence).

        Returns:
            A mapping of image name → ground-truth mask. Empty when no GT source
            is configured.

        # TODO(DEFERRED-WORK §1): read real annotated masks from
        # ``gt_masks_source`` (a directory of per-image masks) and key them by
        # image name. v1 resolves the path only — an existing source yields no
        # masks yet, so the gate abstains until real annotations exist.
        """
        if self.gt_masks_source is None:
            return {}
        # Path resolves but carries no annotations in v1 → abstain (fail-safe).
        return {}

    def _proxy_gt_spearman(
        self, gt_images: Any, masks: dict[str, Any], grid: Any
    ) -> float:
        """Spearman ρ between the proxy score and the GT reference metric.

        The acceptance statistic of the meta-validation gate (``§E.4``): a rank
        correlation (robust to monotone nonlinearity) between the proxy's
        per-candidate scores and a GT-based reference (Dice/Jaccard).

        Args:
            gt_images: The annotated calibration images.
            masks: The resolved name→mask map from :meth:`_load_gt_masks`.
            grid: The candidate parameter grid scored by both the proxy and GT.

        Returns:
            The Spearman rank-correlation coefficient in ``[-1, 1]``.

        # TODO(DEFERRED-WORK §1): compute the real proxy-vs-GT Spearman ρ. v1
        # is a structural stub (returns ``0.0`` → abstain) overridden in tests;
        # wiring the real correlation needs an annotated calibration set.
        """
        return 0.0

    # ----------------------------------------------------------------- #
    # the proxy terms (Task 1)
    # ----------------------------------------------------------------- #
    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Score one image's segmentation as fixed-normalized proxy terms.

        Args:
            image: The processed image — the contrast term reads its
                ``gray``/``objmask`` accessors; the other terms read the frame.
            measurements: The candidate pipeline's measurement frame, carrying
                the category-prefixed ``Shape_*``/``Size_*`` columns.

        Returns:
            A mapping of proxy term → natural goodness in ``[0, 1]`` (higher =
            better): ``ShapeRegularity``, ``Contrast``, ``SizeCV``, and — when a
            ``count_check`` is configured — ``Count``. The base complements these
            goodness values to cost. Keys are stable across images so the
            ``Evaluator`` can aggregate per term.
        """
        empty = measurements is None or len(measurements) == 0
        terms: dict[str, float] = {
            "ShapeRegularity": self._shape_regularity(measurements, empty),
            "Contrast": self._contrast(image),
            "SizeCV": self._size_cv(measurements, empty),
        }
        if self.count_check is not None:
            terms[self.count_term_name] = self._count(measurements, empty)
        return terms

    def _shape_regularity(
        self, measurements: pd.DataFrame, empty: bool
    ) -> float:
        """Mean shape-prior plausibility from the ``Shape_*`` columns.

        Averages the available higher-is-better shape signals — solidity,
        circularity, and ``1 − eccentricity`` — each clamped to ``[0, 1]``
        (``§C.2``). Reuses the public schema columns; **never** recomputes
        geometry from the mask. Colonies smaller than :attr:`min_area` are
        excluded (perimeter metrics are unreliable at a few pixels). Returns the
        neutral floor ``0.0`` when no usable shape column is present.

        Args:
            measurements: The candidate measurement frame.
            empty: Whether the frame is empty.

        Returns:
            The mean shape-regularity score in ``[0, 1]``.
        """
        if empty:
            return 0.0
        frame = measurements
        if self.min_area > 0 and str(SIZE.AREA) in frame.columns:
            frame = frame[frame[str(SIZE.AREA)] >= self.min_area]
            if len(frame) == 0:
                return 0.0
        per_colony_means: list[float] = []
        if str(SHAPE.SOLIDITY) in frame.columns:
            per_colony_means.append(
                float(frame[str(SHAPE.SOLIDITY)].map(_clamp01).mean())
            )
        if str(SHAPE.CIRCULARITY) in frame.columns:
            per_colony_means.append(
                float(frame[str(SHAPE.CIRCULARITY)].map(_clamp01).mean())
            )
        if str(SHAPE.ECCENTRICITY) in frame.columns:
            per_colony_means.append(
                float(
                    frame[str(SHAPE.ECCENTRICITY)]
                    .map(lambda e: _clamp01(1.0 - e))
                    .mean()
                )
            )
        if not per_colony_means:
            return 0.0
        return _clamp01(sum(per_colony_means) / len(per_colony_means))

    def _contrast(self, image: Any) -> float:
        """Otsu between-class separation ``η = σ²_B / σ²_T`` of fg vs. bg.

        Reads ``image.gray`` and ``image.objmask`` to split foreground from
        background, then returns the between-class fraction of total grayscale
        variance (``§A.3``) — already in ``[0, 1]`` (fixed-normalized, no
        min–max over the grid). Returns ``0.0`` for a degenerate image (no
        variance, or an all-foreground / all-background mask).

        Args:
            image: The processed image exposing ``gray``/``objmask`` accessors;
                ``None`` (no image) scores ``0.0``.

        Returns:
            The separation score ``η`` in ``[0, 1]`` (higher = better).
        """
        if image is None:
            return 0.0
        try:
            gray = np.asarray(image.gray[:], dtype=np.float64)
            mask = np.asarray(image.objmask[:], dtype=bool)
        except (AttributeError, TypeError):
            return 0.0
        total_var = float(gray.var())
        if total_var <= 0.0:
            return 0.0
        w_fg = float(mask.mean())
        if w_fg <= 0.0 or w_fg >= 1.0:
            return 0.0
        mu_fg = float(gray[mask].mean())
        mu_bg = float(gray[~mask].mean())
        between_var = w_fg * (1.0 - w_fg) * (mu_fg - mu_bg) ** 2
        return _clamp01(between_var / total_var)

    def _size_cv(self, measurements: pd.DataFrame, empty: bool) -> float:
        """Within-replicate size uniformity from ``Size_Area`` (``§C.3``).

        Computes the coefficient of variation of ``Size_Area`` *within* each
        :attr:`replicate_groupby` group (the whole frame is one group when
        unset), averages the per-group CVs, then folds the result to ``[0, 1]``
        via :func:`_bounded_inverse` (lower variability → higher score). A
        single-colony group contributes ``CV = 0`` (perfectly uniform).

        Args:
            measurements: The candidate measurement frame.
            empty: Whether the frame is empty.

        Returns:
            The size-uniformity score in ``[0, 1]`` (higher = better); ``0.0``
            when ``Size_Area`` is absent or empty.
        """
        area_col = str(SIZE.AREA)
        if empty or area_col not in measurements.columns:
            return 0.0
        if self.replicate_groupby and all(
            col in measurements.columns for col in self.replicate_groupby
        ):
            groups = [
                grp[area_col]
                for _, grp in measurements.groupby(
                    self.replicate_groupby, dropna=False
                )
            ]
        else:
            groups = [measurements[area_col]]
        cvs = [self._coefficient_of_variation(grp) for grp in groups]
        if not cvs:
            return 0.0
        return _bounded_inverse(float(np.mean(cvs)))

    @staticmethod
    def _coefficient_of_variation(values: pd.Series) -> float:
        """The coefficient of variation ``σ / μ`` of one replicate group.

        Uses the **sample** standard deviation (``ddof=1``, Bessel's correction):
        a replicate group is a *sample* of the colony-size population, not the
        whole population, so the unbiased estimator is the correct dispersion
        (and it matches pandas' ``Series.std`` default).

        Args:
            values: A group's ``Size_Area`` values.

        Returns:
            Sample ``std / mean`` (``0.0`` for a single value or a non-positive
            mean).
        """
        clean = values.dropna()
        if len(clean) < 2:
            return 0.0
        mean = float(clean.mean())
        if mean <= 0.0:
            return 0.0
        return float(clean.std(ddof=1)) / mean

    def _count(self, measurements: pd.DataFrame, empty: bool) -> float:
        """The reused expected-vs-detected grid count term (``§C.6``).

        Delegates to the configured :attr:`count_check` exactly as
        :class:`QCScorer` does — runs the check, anchors each group's
        ``QC_Count_Metric`` on the check's ``fail_threshold`` via
        :func:`_threshold_anchored`, and averages across groups.

        Args:
            measurements: The candidate measurement frame.
            empty: Whether the frame is empty.

        Returns:
            The normalized count score in ``[0, 1]`` (higher = better); ``0.0``
            for an empty frame.
        """
        if empty or self.count_check is None:
            return 0.0
        return fold_expected_vs_detected_count(self.count_check, measurements)
