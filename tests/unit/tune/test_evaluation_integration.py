from __future__ import annotations

import gc
import math
import weakref

import pandas as pd
import pytest
from pydantic import PrivateAttr

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import Evaluator
from phenotypic.tune.score import QCScorer
from phenotypic.tune.score._scorer import Scorer


def _layout_csv(tmp_path, n: int, image_name: str = "Synthetic96PlateWithObjects"):
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {
            "Metadata_ImageName": [image_name] * n,
            "Object_Label": list(range(n)),
        }
    ).to_csv(csv, index=False)
    return str(csv)


def _distinct_plates(n: int) -> list:
    return [load_synth_yeast_plate() for _ in range(n)]


def _measured_count(pipeline: ImagePipeline) -> int:
    measured = pipeline.apply_and_measure(
        load_synth_yeast_plate(), inplace=False, apply_post=False
    )
    return len(measured)


def test_perfect_count_scores_one(tmp_path):
    base = ImagePipeline(ops=[OtsuDetector()])
    expected_count = _measured_count(base)
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout_csv(tmp_path, expected_count),
            groupby=["Metadata_ImageName"],
        )
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.n_images == 1
    assert result.terms["Count"] == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


def test_count_mismatch_scores_below_one(tmp_path):
    base = ImagePipeline(ops=[OtsuDetector()])
    detected_count = _measured_count(base)
    expected_count = max(detected_count + 1, round(detected_count * 1.25))
    metric = abs(detected_count - expected_count) / expected_count
    check = ExpectedVsDetectedCount(
        metadata=_layout_csv(tmp_path, expected_count),
        groupby=["Metadata_ImageName"],
    )
    expected_cost = 1.0 - math.exp(-math.log(2.0) * metric / check.fail_threshold)
    scorer = QCScorer(
        check=check
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.score == pytest.approx(expected_cost, abs=1e-6)


# --------------------------------------------------------------------------- #
# The Evaluator must score the image the candidate actually produced
# --------------------------------------------------------------------------- #
class _RecordingScorer(Scorer):
    """Records what the Evaluator handed it, and how many copies are still live.

    ``_live_predecessors[k]`` is how many of the images scored *before* image
    ``k`` were still reachable at the moment image ``k`` was scored. The copies
    are per-image scratch, so every entry should be ``0``.
    """

    _num_objects: list[int] = PrivateAttr(default_factory=list)
    _refs: list[weakref.ref] = PrivateAttr(default_factory=list)
    _live_predecessors: list[int] = PrivateAttr(default_factory=list)

    def _score_terms(self, image, measurements) -> dict[str, float]:
        self._num_objects.append(int(image.num_objects))
        gc.collect()  # so a copy held only by a reference cycle still counts
        self._live_predecessors.append(
            sum(1 for ref in self._refs if ref() is not None)
        )
        self._refs.append(weakref.ref(image))
        return {"X": 0.0}


def test_scorer_receives_the_candidates_detection_not_the_input_objmap():
    # ``load_synth_yeast_plate()`` ships WITH an objmap already populated, so a
    # scorer handed the untouched input reads a stale, candidate-independent
    # segmentation -- which is exactly how the Region term came out invariant to
    # the sampled hyperparameters. Guard first that the two counts differ, so
    # this test cannot pass vacuously.
    base = ImagePipeline(ops=[OtsuDetector()])
    untouched = load_synth_yeast_plate().num_objects
    detected = base.apply(image=load_synth_yeast_plate(), inplace=False).num_objects
    assert untouched != detected, (
        "fixture and candidate detect the same object count; this test can no "
        "longer distinguish the input objmap from the candidate's own"
    )

    image = load_synth_yeast_plate()
    scorer = _RecordingScorer()
    Evaluator().evaluate(base, scorer, {}, [image])

    assert scorer._num_objects == [detected]
    # ...and the shared calibration image is still pristine for the next trial.
    assert image.num_objects == untouched


def test_processed_copies_are_released_between_images():
    # The processed copies are per-image scratch. Retaining them for the whole
    # pass -- e.g. to keep their ``id()``s distinct for an identity-keyed test
    # scorer -- holds the entire calibration set in memory in processed form
    # (rgb + gray + detect_mat + objmap), on every trial of every study. Peak
    # cost is linear in the set size and in the pixel count per plate.
    base = ImagePipeline(ops=[OtsuDetector()])
    scorer = _RecordingScorer()
    Evaluator().evaluate(base, scorer, {}, _distinct_plates(3))

    assert len(scorer._refs) == 3
    assert scorer._live_predecessors == [0, 0, 0]
