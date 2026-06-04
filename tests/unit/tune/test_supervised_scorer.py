"""Tests for the supervised (ground-truth) scorer — Phase 4 chunk B (4.4).

Per plan §0a+§0b and the spec's §3 tier table, these tests pin only the
**structure** of the scorer: term shape per runnable GT tier, the
modality-tiered availability matrix (mask / count / abstain), the genuine
reuse of :class:`phenotypic.analysis.ExpectedVsDetectedCount` for the count
tier (a spy on ``.analyze``), registry round-trip through ``TuningSpec``, and
the one-region-metric (Dice xor IoU) construction guard. **GT validation is
DEFERRED** (no numeric-vs-real-GT assertions here).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune._scoring._gt_loader import GroundTruthMasks
from phenotypic.tune._scoring._supervised import SupervisedScorer


# --------------------------------------------------------------------------- #
# helpers — a mask directory (mask tier) and a count table (count tier)
# --------------------------------------------------------------------------- #
def _mask_dir(tmp_path: Path, name: str = "plate1") -> Path:
    """A GT source directory holding one boolean mask → modality "mask"."""
    d = tmp_path / "gt_masks"
    d.mkdir()
    np.save(d / f"{name}.npy", np.array([[True, True, False], [False, True, False]]))
    return d


def _count_table(tmp_path: Path, n: int = 96, name: str = "p1") -> Path:
    """A GT source CSV of per-image expected counts → modality "count"."""
    csv = tmp_path / "counts.csv"
    pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    ).to_csv(csv, index=False)
    return csv


def _count_check(tmp_path: Path, n: int = 96, name: str = "p1") -> ExpectedVsDetectedCount:
    """A path-configured count check (so ``metadata_source`` round-trips)."""
    return ExpectedVsDetectedCount(
        metadata=str(_count_table(tmp_path, n, name)),
        groupby=["Metadata_ImageName"],
    )


def _measurements(n: int = 96, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


class _FakeGridImage:
    """A minimal duck-typed GridImage for the mask-tier match path.

    Exposes ``name``, ``objmap`` (label array via ``[:]``), and
    ``grid.get_section_map`` — the surface ``match_per_grid_cell`` reads.
    """

    def __init__(self, name: str, labels: np.ndarray, sections: np.ndarray) -> None:
        self.name = name
        self.objmap = _ArrayAccessor(labels)
        self.grid = _FakeGrid(sections)


class _ArrayAccessor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def __getitem__(self, key: object) -> np.ndarray:
        return self._arr


class _FakeGrid:
    def __init__(self, sections: np.ndarray) -> None:
        self._sections = sections

    def get_section_map(self) -> np.ndarray:
        return self._sections


# --------------------------------------------------------------------------- #
# one-region-metric guard (Dice xor IoU)
# --------------------------------------------------------------------------- #
def test_region_metric_dice_is_default(tmp_path):
    scorer = SupervisedScorer(gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path)))
    assert scorer.region_metric == "dice"


def test_region_metric_iou_is_accepted(tmp_path):
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path)),
        region_metric="iou",
    )
    assert scorer.region_metric == "iou"


def test_region_metric_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        SupervisedScorer(
            gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path)),
            region_metric="both",  # not a single Dice xor IoU
        )


# --------------------------------------------------------------------------- #
# availability matrix — mask / count / abstain
# --------------------------------------------------------------------------- #
def test_availability_mask_tier(tmp_path):
    scorer = SupervisedScorer(gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path)))
    assert scorer.gt.modality() == "mask"
    assert scorer.availability() is True


def test_availability_count_tier(tmp_path):
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_count_table(tmp_path)),
        count_check=_count_check(tmp_path),
    )
    assert scorer.gt.modality() == "count"
    assert scorer.availability() is True


def test_availability_count_tier_without_check_abstains(tmp_path):
    # The count tier needs a count check to run; without one it cannot score.
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_count_table(tmp_path)),
        count_check=None,
    )
    assert scorer.gt.modality() == "count"
    assert scorer.availability() is False


def test_availability_none_tier_abstains():
    scorer = SupervisedScorer(gt=GroundTruthMasks(gt_masks_source=None))
    assert scorer.gt.modality() == "none"
    assert scorer.availability() is False


# --------------------------------------------------------------------------- #
# term shape — one term per runnable tier
# --------------------------------------------------------------------------- #
def test_mask_tier_emits_region_term(tmp_path):
    # GT mask exactly equals the prediction → macro-averaged Region == 1.0.
    labels = np.array([[1, 1, 0], [0, 1, 0]])
    sections = np.array([[1, 1, 1], [1, 1, 1]])
    gt_dir = tmp_path / "gt_masks"
    gt_dir.mkdir()
    np.save(gt_dir / "plateA.npy", labels.astype(bool))
    image = _FakeGridImage("plateA", labels, sections)
    scorer = SupervisedScorer(gt=GroundTruthMasks(gt_masks_source=gt_dir))
    out = scorer.score_image(image, _measurements())
    assert set(out) == {"Region"}
    assert out["Region"] == pytest.approx(1.0)


def test_mask_tier_unresolved_name_returns_empty_terms(tmp_path):
    # No GT mask for this image's stem → no Region term (nothing to score).
    labels = np.array([[1, 1, 0], [0, 1, 0]])
    sections = np.array([[1, 1, 1], [1, 1, 1]])
    image = _FakeGridImage("unseen_plate", labels, sections)
    scorer = SupervisedScorer(gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path)))
    out = scorer.score_image(image, _measurements())
    assert out == {}


def test_count_tier_emits_count_mae_term(tmp_path):
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_count_table(tmp_path, n=96, name="p1")),
        count_check=_count_check(tmp_path, n=96, name="p1"),
    )
    out = scorer.score_image(None, _measurements(96, "p1"))
    assert set(out) == {"CountMAE"}
    # perfect count match → folded score 1.0 (higher is better)
    assert out["CountMAE"] == pytest.approx(1.0)


def test_count_tier_at_fail_threshold_is_half(tmp_path):
    # expected 100, detected 90 → metric 0.10 == fail_threshold → folded 0.5
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_count_table(tmp_path, n=100, name="p1")),
        count_check=_count_check(tmp_path, n=100, name="p1"),
    )
    out = scorer.score_image(None, _measurements(90, "p1"))
    assert out["CountMAE"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# the count term genuinely reuses ExpectedVsDetectedCount (spy on .analyze)
# --------------------------------------------------------------------------- #
def test_count_term_reuses_expected_vs_detected_count(tmp_path):
    from unittest import mock

    check = _count_check(tmp_path, n=96, name="p1")
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_count_table(tmp_path, n=96, name="p1")),
        count_check=check,
    )
    # Spy on the bound method via the class (the pydantic model rejects
    # per-instance method assignment under validate_assignment): wrap the real
    # ExpectedVsDetectedCount.analyze so we prove the count tier delegates to it.
    with mock.patch.object(
        ExpectedVsDetectedCount,
        "analyze",
        autospec=True,
        side_effect=ExpectedVsDetectedCount.analyze,
    ) as spy:
        scorer.score_image(None, _measurements(96, "p1"))
    assert spy.called, "count tier must call ExpectedVsDetectedCount.analyze"
    # the check that was called is exactly the configured count_check
    assert spy.call_args.args[0] is check


# --------------------------------------------------------------------------- #
# registry round-trip through TuningSpec via ScorerField
# --------------------------------------------------------------------------- #
def test_supervised_scorer_round_trips_in_tuning_spec(tmp_path):
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import (
        Budget,
        Categorical,
        Evaluator,
        GridConfig,
        Knob,
        SearchSpace,
    )
    from phenotypic.tune._spec import TuningSpec

    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_count_table(tmp_path)),
        region_metric="iou",
        count_check=_count_check(tmp_path),
    )
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),)
        ),
        scorer=scorer,
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    back = TuningSpec.model_validate_json(spec.model_dump_json())
    assert isinstance(back.scorer, SupervisedScorer)
    assert back.scorer.region_metric == "iou"
    assert back.scorer.gt.modality() == "count"
    assert back.scorer.count_check is not None
    # the path-configured count check still scores after reload
    assert back.scorer.score_image(None, _measurements())["CountMAE"] == pytest.approx(1.0)


def test_supervised_scorer_direct_round_trip(tmp_path):
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path)),
        region_metric="dice",
    )
    back = SupervisedScorer.model_validate_json(scorer.model_dump_json())
    assert back.region_metric == "dice"
    assert back.match_strategy == "grid_cell"
    assert str(back.gt.gt_masks_source) == str(scorer.gt.gt_masks_source)
