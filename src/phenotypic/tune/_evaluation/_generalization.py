"""The held-out generalization pass + the ``generalization.json`` report.

Phase 4.5 part 2 closes the robust-eval split (part 1) by *consuming* it: after
the search has run on **calibration plates only**, this module re-evaluates the
winner on the untouched **held-out** plates and reports the true generalization
gap — a user-facing deliverable, never a change to the winner.

The verdict is report-only on two axes:

- The **overfit gate** (:func:`compute_generalization_gap`) flags a
  calibration→held-out score drop only when it exceeds **both** a relative and
  an absolute margin (the ``HeldOutConfig`` policy), so a tiny absolute drop on
  a low-scoring objective is not raised as overfit, and neither is a large
  relative drop that is absolutely negligible.
- The **3-tier report** mirrors the split's ``kind`` (:func:`run_held_out`): a
  whole-group hold-out is the strongest cross-batch test (``"group"``); a
  within-group hold-out carries a weaker-guarantee caveat (``"within_group"``);
  a data-poor split reserved nothing, so there is **no** real held-out gap and
  the report falls back to a calibration-stability estimate (``"none"``,
  ``cv_deferred=True``) — the §8 CV-estimate is deferred (see DEFERRED-WORK.md).

optuna-free (numpy is fine; this module uses neither); the held-out evaluation
re-uses the spec's own ``Evaluator`` so the held-out pass is the same code path
as calibration, just on the reserved plates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, TypeAlias

from ._aggregate_math import _relative
from ._split import SplitKind

#: The generalization estimate's provenance — a real held-out pass
#: (``"held_out"``) or the data-poor calibration-stability proxy
#: (``"calibration_stability"``). A closed value set, reused as a typed field.
Estimate: TypeAlias = Literal["held_out", "calibration_stability"]

#: The within-group hold-out caveat note (a weaker, within-group guarantee).
_WITHIN_GROUP_NOTE = (
    "held-out plates share a group with calibration — within-group "
    "generalization estimate only (no cross-group test)"
)

#: The data-poor fallback note (no untouched held-out set; CV deferred).
_DATA_POOR_NOTE = (
    "no untouched held-out plates (data-poor split) — calibration-stability "
    "estimate (CV deferred)"
)

#: The dataset-drift note (the loaded plates no longer match the persisted split).
_DATASET_CHANGED_NOTE = (
    "loaded dataset no longer matches the persisted split — the held-out "
    "verdict reflects the original split membership, not the current plates"
)


def compute_generalization_gap(
    cal_score: float,
    heldout_score: float,
    *,
    rel_margin: float,
    abs_margin: float,
) -> tuple[float, float, bool]:
    """The BOTH-thresholds overfit gate on a calibration→held-out score drop.

    Computes the relative and absolute drops and flags overfit only when **both**
    margins are exceeded (the :class:`~phenotypic.tune.HeldOutConfig` policy):

    - ``relative_drop = (cal_score - heldout_score) / max(|cal_score|, eps)``;
    - ``absolute_drop = cal_score - heldout_score``;
    - ``flagged = (relative_drop > rel_margin) and (absolute_drop > abs_margin)``.

    Requiring both guards two false positives: a *large relative* drop that is
    *absolutely negligible* (e.g. ``0.04 → 0.03`` — 25% relative, 0.01 absolute),
    and a *large absolute* drop on an objective whose calibration score was so
    high the relative slack already covers it.

    Args:
        cal_score: The winner's calibration (in-search) score (higher = better;
            under the cost convention the caller passes the goodness-equivalent
            ``1 − cal_cost`` — see Note).
        heldout_score: The winner's held-out score (higher = better; under the
            cost convention the caller passes ``1 − heldout_cost`` — see Note).
        rel_margin: The relative-drop margin (``HeldOutConfig.gap_margin_relative``).
        abs_margin: The absolute-drop margin (``HeldOutConfig.gap_margin_absolute``).

    Returns:
        A ``(relative_drop, absolute_drop, flagged)`` triple. The drops are
        signed (negative when the held-out score *improved*); ``flagged`` is
        ``True`` only when both exceed their margins.

    Note:
        This function is direction-agnostic. Under the cost convention the
        caller passes goodness-equivalents (``1 − cost``) so the unchanged
        formula is the standard loss-space gap (``heldout_cost − cal_cost``).

    Examples:
        >>> rel, absolute, flagged = compute_generalization_gap(
        ...     0.9, 0.5, rel_margin=0.15, abs_margin=0.05
        ... )
        >>> round(rel, 3), round(absolute, 3), flagged
        (0.444, 0.4, True)
        >>> # A tiny absolute drop is never flagged, even at 25% relative.
        >>> compute_generalization_gap(0.04, 0.03, rel_margin=0.15, abs_margin=0.05)[2]
        False
    """
    absolute_drop = float(cal_score) - float(heldout_score)
    relative_drop = _relative(absolute_drop, float(cal_score))
    flagged = (relative_drop > rel_margin) and (absolute_drop > abs_margin)
    return relative_drop, absolute_drop, flagged


@dataclass(frozen=True)
class GeneralizationReport:
    """The winner's held-out generalization verdict — a frozen deliverable.

    Serialized to ``deliverables/generalization.json`` (via :meth:`to_dict`). It
    is **report-only**: the winner is never changed by the held-out pass, and the
    immutable trial journal is untouched (the per-trial ``Trial.gap`` stays the
    calibration dispersion; the TRUE held-out gap lives only here).

    Args:
        kind: The split tier this verdict was produced under — ``"group"``
            (whole-group hold-out, the strongest cross-batch test),
            ``"within_group"`` (a weaker, within-group hold-out), or ``"none"``
            (data-poor — no real held-out gap, calibration-stability fallback).
        calibration_score: The winner's calibration (in-search) score.
        heldout_score: The winner's held-out score, or ``None`` for the data-poor
            fallback (no untouched held-out set was evaluated).
        relative_drop: The relative calibration→held-out drop, or ``None`` for the
            data-poor fallback.
        absolute_drop: The absolute calibration→held-out drop, or ``None`` for the
            data-poor fallback.
        gap: The true held-out generalization gap (``= absolute_drop``), or
            ``None`` for the data-poor fallback.
        flagged: ``True`` when the overfit gate fired (both margins exceeded);
            always ``False`` for the data-poor fallback. Report-only — it never
            changes the winner.
        estimate: ``"held_out"`` when a real held-out pass ran, else
            ``"calibration_stability"`` (the data-poor proxy).
        cv_deferred: ``True`` for the data-poor fallback — the §8 cross-validation
            estimate is deferred; the report substitutes a calibration-stability
            proxy (see DEFERRED-WORK.md).
        within_group_caveat: ``True`` for ``kind="within_group"`` — the held-out
            plates share a group with calibration, so the guarantee is weaker.
        dataset_changed: ``True`` when the loaded plates no longer match the
            persisted split's ``dataset_identity`` (the verdict still reflects the
            original split membership; resume reuses the persisted split).
        warning: A human-readable caveat (within-group / data-poor / dataset-drift),
            or ``None`` when the verdict carries the strongest guarantee.
        gap_margin_relative: The relative margin the overfit gate used.
        gap_margin_absolute: The absolute margin the overfit gate used.
        calibration_stability: The winner's per-trial calibration dispersion
            (``Trial.gap``) — the data-poor proxy for a held-out gap; ``None``
            when a real held-out pass ran (or the winner had no gap signal).
    """

    kind: SplitKind
    calibration_score: float
    heldout_score: Optional[float]
    relative_drop: Optional[float]
    absolute_drop: Optional[float]
    gap: Optional[float]
    flagged: bool
    estimate: Estimate
    cv_deferred: bool
    within_group_caveat: bool
    dataset_changed: bool
    warning: Optional[str]
    gap_margin_relative: float
    gap_margin_absolute: float
    calibration_stability: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON-serializable mapping written to ``generalization.json``.

        Returns:
            A plain ``dict`` of the report fields (all JSON-native scalars), ready
            for ``json.dumps(..., indent=2)``.
        """
        return asdict(self)


def _select_held_out(split: Any, images_by_name: dict[str, Any]) -> list:
    """The loaded plates whose ``name`` is in the split's held-out list.

    Name-membership (RESOLVED design): a held-out plate is one whose ``name`` is
    in ``split.held_out``; a plate absent from the current load is simply skipped
    (the held-out set shrinks rather than erroring).

    Args:
        split: The resolved :class:`~phenotypic.tune._evaluation._split.Split`.
        images_by_name: ``{image.name: image}`` of the loaded plates.

    Returns:
        The held-out plate objects present in the current load.
    """
    return [images_by_name[name] for name in split.held_out if name in images_by_name]


def run_held_out(
    spec: Any,
    winner: Any,
    split: Any,
    images_by_name: dict[str, Any],
    *,
    current_identity: Optional[str] = None,
) -> GeneralizationReport:
    """Re-evaluate the ``winner`` on the held-out plates → a :class:`GeneralizationReport`.

    The report-only generalization pass (RESOLVED design — held-out orchestration
    lives in the run layer, never the engine). It re-runs the winner's parameters
    through the spec's own :class:`~phenotypic.tune.Evaluator` over the **held-out
    plates only**, then builds a 3-tier verdict by ``split.kind``:

    - ``"group"``: a real held-out gap (the strongest cross-batch test);
    - ``"within_group"``: a real held-out gap **plus** a weaker-guarantee caveat;
    - ``"none"``: data-poor — no untouched held-out set, so **no** real gap
      (``gap=None``, ``flagged=False``); the report falls back to a
      calibration-stability estimate carrying the winner's ``Trial.gap`` (the
      per-trial calibration dispersion) with ``cv_deferred=True``.

    The winner is **never** changed; ``Trial.gap`` is **not** mutated (Option A).
    When ``current_identity`` differs from ``split.dataset_identity`` the report's
    ``dataset_changed`` is set with a drift warning (the split is reused verbatim
    on resume; this only annotates the verdict).

    Args:
        spec: The resolved tuning spec — only ``spec.evaluator``, ``spec.pipeline``,
            ``spec.scorer``, and ``spec.held_out`` (the gap margins) are read.
        winner: The winning trial — ``winner.params``, ``winner.score``, and
            ``winner.gap`` are read.
        split: The resolved split (its ``kind`` / ``held_out`` / ``group_key`` /
            ``within_group_caveat`` / ``dataset_identity`` drive the verdict).
        images_by_name: ``{image.name: image}`` of the loaded plates.
        current_identity: The current dataset identity (a mismatch vs
            ``split.dataset_identity`` sets ``dataset_changed``); ``None`` skips
            the drift check.

    Returns:
        The :class:`GeneralizationReport` to write to ``generalization.json``.
    """
    held_out = spec.held_out
    rel_margin = float(held_out.gap_margin_relative)
    abs_margin = float(held_out.gap_margin_absolute)
    cal_score = float(winner.score)
    dataset_changed = (
        current_identity is not None
        and current_identity != split.dataset_identity
    )

    held_out_images = _select_held_out(split, images_by_name)

    # Data-poor (or an empty held-out set): no real held-out gap. Fall back to a
    # calibration-stability estimate carrying the winner's per-trial dispersion.
    if split.kind == "none" or not held_out_images:
        warning = _compose_warning(_DATA_POOR_NOTE, dataset_changed)
        return GeneralizationReport(
            kind=split.kind,
            calibration_score=cal_score,
            heldout_score=None,
            relative_drop=None,
            absolute_drop=None,
            gap=None,
            flagged=False,
            estimate="calibration_stability",
            cv_deferred=True,
            within_group_caveat=bool(split.within_group_caveat),
            dataset_changed=dataset_changed,
            warning=warning,
            gap_margin_relative=rel_margin,
            gap_margin_absolute=abs_margin,
            calibration_stability=(
                None if winner.gap is None else float(winner.gap)
            ),
        )

    # A real held-out pass: re-evaluate the winner on the reserved plates once,
    # via the spec's own Evaluator (same code path as calibration, full fidelity).
    result = spec.evaluator.evaluate(
        spec.pipeline, spec.scorer, winner.params, held_out_images
    )
    heldout_score = float(result.score)
    # Cost convention: pass goodness-equivalents (1 - cost) so the unchanged
    # accuracy-space formula (cal_g - heldout_g) equals the standard loss-space
    # gap (heldout_cost - cal_cost), positive = overfit. No bespoke sign flip.
    relative_drop, absolute_drop, flagged = compute_generalization_gap(
        1.0 - cal_score,
        1.0 - heldout_score,
        rel_margin=rel_margin,
        abs_margin=abs_margin,
    )

    note = _WITHIN_GROUP_NOTE if split.within_group_caveat else None
    warning = _compose_warning(note, dataset_changed)
    return GeneralizationReport(
        kind=split.kind,
        calibration_score=cal_score,
        heldout_score=heldout_score,
        relative_drop=relative_drop,
        absolute_drop=absolute_drop,
        gap=absolute_drop,
        flagged=flagged,
        estimate="held_out",
        cv_deferred=False,
        within_group_caveat=bool(split.within_group_caveat),
        dataset_changed=dataset_changed,
        warning=warning,
        gap_margin_relative=rel_margin,
        gap_margin_absolute=abs_margin,
        calibration_stability=None,
    )


def _compose_warning(note: Optional[str], dataset_changed: bool) -> Optional[str]:
    """Join the tier note + the dataset-drift note into one warning string.

    Args:
        note: The tier-specific caveat (within-group / data-poor), or ``None``.
        dataset_changed: Whether the dataset-drift note should be appended.

    Returns:
        The composed warning, or ``None`` when neither caveat applies.
    """
    parts = [p for p in (note, _DATASET_CHANGED_NOTE if dataset_changed else None) if p]
    if not parts:
        return None
    return "; ".join(parts)
