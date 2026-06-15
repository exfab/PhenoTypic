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
from phenotypic.grid import AutoGridFinder
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
    assert out["Region"] == pytest.approx(0.0)


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
    # perfect count match → zero cost (lower is better)
    assert out["CountMAE"] == pytest.approx(0.0)


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
    assert back.scorer.score_image(None, _measurements())["CountMAE"] == pytest.approx(0.0)


def test_supervised_scorer_direct_round_trip(tmp_path):
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path)),
        region_metric="dice",
    )
    back = SupervisedScorer.model_validate_json(scorer.model_dump_json())
    assert back.region_metric == "dice"
    assert back.match_strategy == "grid_cell"
    assert str(back.gt.gt_masks_source) == str(scorer.gt.gt_masks_source)


# --------------------------------------------------------------------------- #
# B1 — binary GT modality on REAL load_synth_yeast_plate() data
#
# These tests use the *real* GridImage grid (geometric, detection-independent
# cell map built from grid edges) — NO hand-built full-coverage section map. A
# wrapper serves a custom predicted objmap over the genuine grid so the binary
# GT (the plate's true foreground) is derived per geometric cell and matched by
# pure global IoU (so misses/over-extension are scored honestly, never clipped
# to the prediction's pixels).
# --------------------------------------------------------------------------- #
class _PredOverRealGrid:
    """Forwards the real ``GridImage`` grid but serves a custom predicted objmap.

    Exposes ``name``, ``grid`` (the real grid — its ``get_row_edges`` /
    ``get_col_edges`` / ``nrows`` / ``ncols`` drive the geometric cell map), and
    ``objmap`` (the custom prediction). This is the realistic surface the binary
    path reads — it does **not** stub ``get_section_map``.
    """

    def __init__(self, image: object, pred_objmap: np.ndarray) -> None:
        self._image = image
        self._pred = np.asarray(pred_objmap)
        self.name = getattr(image, "name", "")

    @property
    def grid(self) -> object:
        return self._image.grid  # type: ignore[attr-defined]

    @property
    def objmap(self) -> "_RealArrayAccessor":
        return _RealArrayAccessor(self._pred)


class _RealArrayAccessor:
    """A minimal objmap accessor that honors slicing (``[:]``) and ``.shape``."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def __getitem__(self, key: object) -> np.ndarray:
        return self._arr[key]  # type: ignore[index]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape


def _binary_gt_dir_for_plate(tmp_path: Path, image: object) -> Path:
    """A GT dir holding the plate's true foreground as a BINARY mask."""
    d = tmp_path / "gt_binary"
    d.mkdir()
    foreground = np.asarray(image.objmap[:]) > 0  # type: ignore[attr-defined]
    np.save(d / f"{image.name}.npy", foreground)  # type: ignore[attr-defined]
    return d


def test_binary_gt_missed_cell_pulls_score_below_all_detected(tmp_path):
    """(a) A GT colony in a cell the prediction detects NOTHING in must pull the
    score below the all-detected score — proving the geometric (not
    prediction-derived) cell map keeps the missed colony's GT instance alive, so
    it is matched against nothing and scored 0 (correct miss penalty). The old
    ``get_section_map()`` path would have erased that GT instance entirely (no
    detected pixels → no cell id), hiding the miss.
    """
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    # Pin the per-axis AutoGridFinder so the (anisotropic) synth plate yields a
    # clean 1:1 colony->cell grid. The default CenteredAutoGridFinder fits a single
    # isotropic pitch (non-square pitch is an explicit non-goal), which shares 8
    # cells on this fixture and would break the perfect-score baseline this scorer
    # test depends on. This test exercises the scorer's geometric cell map, not the
    # default finder choice.
    image.grid_finder = AutoGridFinder(nrows=image.nrows, ncols=image.ncols)
    true_objmap = np.asarray(image.objmap[:]).copy()
    gt_dir = _binary_gt_dir_for_plate(tmp_path, image)
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=gt_dir), region_metric="iou"
    )
    assert scorer.gt.gt_format == "binary"

    # All-detected: the prediction is the plate's own object map.
    all_detected = scorer.score_image(
        _PredOverRealGrid(image, true_objmap), _measurements()
    )["Region"]

    # Missed cell: zero out one whole colony so its grid cell has no prediction.
    dropped = int(np.unique(true_objmap[true_objmap > 0])[0])
    missed = true_objmap.copy()
    missed[missed == dropped] = 0
    missed_score = scorer.score_image(
        _PredOverRealGrid(image, missed), _measurements()
    )["Region"]

    assert all_detected == pytest.approx(0.0)
    assert missed_score > all_detected
    # Diagnostic: exactly one of N GT instances is now unmatched (scored 0), so
    # the macro-average goodness is (N-1)/N → cost is 1/N — confirming the missed
    # cell stayed scoreable rather than vanishing from the GT.
    n = int(image.num_objects)
    assert missed_score == pytest.approx(1 / n)


def test_binary_gt_under_segmentation_is_visible_not_clipped(tmp_path):
    """(b) When the GT foreground in a DETECTED cell is larger than the predicted
    colony, the region IoU must reflect the TRUE GT extent (score < 1) — the GT
    is not clipped to the prediction's pixels. The old ``get_section_map()`` path
    masked the GT to detected pixels, which would inflate this toward 1.
    """
    from skimage.morphology import binary_erosion, disk

    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    # Pin the per-axis AutoGridFinder so the (anisotropic) synth plate yields a
    # clean 1:1 colony->cell grid (see the companion test for the rationale).
    image.grid_finder = AutoGridFinder(nrows=image.nrows, ncols=image.ncols)
    true_objmap = np.asarray(image.objmap[:]).copy()
    gt_dir = _binary_gt_dir_for_plate(tmp_path, image)
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=gt_dir), region_metric="iou"
    )

    all_detected = scorer.score_image(
        _PredOverRealGrid(image, true_objmap), _measurements()
    )["Region"]

    # Under-segment ONE colony: shrink its predicted footprint while the GT
    # foreground keeps the colony's full extent.
    target = int(np.unique(true_objmap[true_objmap > 0])[1])
    target_mask = true_objmap == target
    eroded = binary_erosion(target_mask, disk(3))
    under = true_objmap.copy()
    under[target_mask & ~eroded] = 0  # predicted colony is now smaller than GT
    assert int((target_mask & ~eroded).sum()) > 0  # genuinely shrank it
    under_score = scorer.score_image(
        _PredOverRealGrid(image, under), _measurements()
    )["Region"]

    assert all_detected == pytest.approx(0.0)
    # The shrunken pair's IoU is now < 1 (GT extent preserved), so the macro
    # average goodness drops → cost rises above the all-detected cost — under-segmentation is visible.
    assert under_score > all_detected


def test_binary_gt_separating_two_colonies_beats_merging_on_real_grid(tmp_path):
    """Separating two real colonies that share ONE geometric grid cell scores
    higher than merging them — proving per-geometric-cell connected components
    derive two GT instances (the old single-pseudo-object collapse, which made a
    merge score *perfectly*, is gone). Uses the real grid; no artificial map.
    """
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    true_objmap = np.asarray(image.objmap[:]).copy()

    # Pick two distinct real colonies; build a "separated" prediction (their own
    # labels) and a "merged" prediction (both relabelled to a single object).
    labels = list(np.unique(true_objmap[true_objmap > 0])[:2])
    a, b = int(labels[0]), int(labels[1])
    two_only = np.zeros_like(true_objmap)
    two_only[true_objmap == a] = a
    two_only[true_objmap == b] = b

    # Restrict the GT to just these two colonies so the macro-average is over a
    # small, sensitive set (and so "merge" genuinely fuses two GT instances).
    gt_two = (true_objmap == a) | (true_objmap == b)
    gt_two_dir = tmp_path / "gt_two"
    gt_two_dir.mkdir()
    np.save(gt_two_dir / f"{image.name}.npy", gt_two)
    scorer_two = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=gt_two_dir), region_metric="iou"
    )

    separated = two_only.copy()  # two distinct predicted objects
    merged = np.zeros_like(true_objmap)
    merged[gt_two] = 1  # ONE predicted object spanning both colonies

    sep_region = scorer_two.score_image(
        _PredOverRealGrid(image, separated), _measurements()
    )["Region"]
    mrg_region = scorer_two.score_image(
        _PredOverRealGrid(image, merged), _measurements()
    )["Region"]

    assert sep_region < mrg_region
    assert sep_region == pytest.approx(0.0)
    assert mrg_region > 0.0


def test_binary_gt_without_grid_falls_back_to_global_components(tmp_path):
    """A plain (non-grid) image falls back to a single global CC labelling: two
    separated blobs still become two GT instances, so a 2-object prediction
    matches both."""

    class _PlainImage:
        def __init__(self, name: str, labels: np.ndarray) -> None:
            self.name = name
            self.objmap = _ArrayAccessor(labels)
            # No ``grid`` attribute → _gt_instances takes the global-CC branch.

    foreground = np.array([[1, 1, 0, 0, 1, 1]])
    gt_dir = tmp_path / "gt_plain"
    gt_dir.mkdir()
    np.save(gt_dir / "plateP.npy", foreground.astype(bool))
    # iou_greedy is the non-gridded matcher (grid_cell needs a grid).
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=gt_dir),
        match_strategy="iou_greedy",
        region_metric="iou",
    )
    image = _PlainImage("plateP", np.array([[1, 1, 0, 0, 2, 2]]))
    region = scorer.score_image(image, _measurements())["Region"]
    assert region == pytest.approx(0.0)


def test_binary_gt_geometric_cell_map_is_detection_independent():
    """The geometric cell map covers the whole grid regardless of detections:
    it labels every cell (0..nrows*ncols-1) from the grid EDGES, unlike
    ``get_section_map()`` which only labels detected-colony pixels. This is what
    keeps entirely-missed cells scoreable."""
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    scorer = SupervisedScorer(gt=GroundTruthMasks(gt_masks_source=None))
    shape = np.asarray(image.objmap[:]).shape
    cell_map = scorer._geometric_cell_map_or_none(image, shape)
    assert cell_map is not None
    n_cells = int(image.grid.nrows) * int(image.grid.ncols)
    # Every grid cell id appears, and they do NOT depend on where colonies were
    # detected (the map is built purely from grid edges).
    assert set(np.unique(cell_map[cell_map >= 0]).tolist()) == set(range(n_cells))


def test_instance_gt_round_trips_through_the_matcher(tmp_path):
    """An integer-labelled (instance) GT is used as-is by the matcher: a
    prediction equal to the GT labels matches every object perfectly."""
    labels = np.array([[1, 1, 0, 0, 2, 2]])
    sections = np.array([[1, 1, 1, 1, 1, 1]])
    d = tmp_path / "gt_instance"
    d.mkdir()
    np.save(d / "plateI.npy", labels.astype(np.int32))  # integer labels, NOT bool
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=d, gt_format="instance"),
        region_metric="iou",
    )
    image = _FakeGridImage("plateI", labels.copy(), sections)
    out = scorer.score_image(image, _measurements())
    assert out["Region"] == pytest.approx(0.0)


def test_instance_gt_two_objects_one_missed_is_penalized(tmp_path):
    """With instance GT, missing one of two GT objects halves the macro-average
    (one perfect pair + one unmatched GT scored 0)."""
    gt_labels = np.array([[1, 1, 0, 0, 2, 2]])
    sections = np.array([[1, 1, 1, 1, 1, 1]])
    d = tmp_path / "gt_instance2"
    d.mkdir()
    np.save(d / "plateM.npy", gt_labels.astype(np.int32))
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=d, gt_format="instance"),
        region_metric="iou",
    )
    # Prediction detects only the first GT object; the second is missed.
    image = _FakeGridImage("plateM", np.array([[1, 1, 0, 0, 0, 0]]), sections)
    out = scorer.score_image(image, _measurements())
    # one perfect IoU (goodness 1.0) + one unmatched GT (goodness 0.0) → macro goodness 0.5 → cost 0.5
    assert out["Region"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# B1 — gt_format round-trips through TuningSpec / model_dump_json
# --------------------------------------------------------------------------- #
def test_gt_format_round_trips_through_tuning_spec(tmp_path):
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
        gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path), gt_format="instance"),
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
    assert back.scorer.gt.gt_format == "instance"


def test_gt_format_round_trips_through_scorer_dump(tmp_path):
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_mask_dir(tmp_path), gt_format="instance"),
    )
    back = SupervisedScorer.model_validate_json(scorer.model_dump_json())
    assert back.gt.gt_format == "instance"
