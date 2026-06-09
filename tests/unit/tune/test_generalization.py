"""4.5p2 D — the held-out generalization pass + ``generalization.json`` report.

D1: ``compute_generalization_gap`` — the BOTH-thresholds overfit gate (a
calibration→held-out score drop is flagged only when it exceeds **both** the
relative and absolute margins), plus the frozen ``GeneralizationReport``.
D2: ``run_held_out`` — the report-only held-out pass on the winner across the
3-tier split policy (``"group"`` / ``"within_group"`` / ``"none"``) and the
dataset-changed flag.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import QCScorer
from phenotypic.tune._evaluation._generalization import (
    GeneralizationReport,
    compute_generalization_gap,
    run_held_out,
)
from phenotypic.tune._evaluation._split import Split, _dataset_identity


# -- D1: the BOTH-thresholds overfit gate -------------------------------------


def test_gap_flagged_when_both_margins_exceeded():
    rel, absolute, flagged = compute_generalization_gap(
        0.9, 0.5, rel_margin=0.15, abs_margin=0.05
    )
    assert rel == pytest.approx(0.4 / 0.9)  # ≈ 0.444
    assert absolute == pytest.approx(0.4)
    assert flagged is True


def test_not_flagged_when_absolute_drop_tiny():
    # 25% relative drop, but only a 0.01 absolute drop → the absolute floor
    # (0.05) is not exceeded, so the BOTH-thresholds gate does not flag.
    rel, absolute, flagged = compute_generalization_gap(
        0.04, 0.03, rel_margin=0.15, abs_margin=0.05
    )
    assert rel == pytest.approx(0.25)
    assert absolute == pytest.approx(0.01)
    assert flagged is False


def test_not_flagged_when_comparable():
    rel, absolute, flagged = compute_generalization_gap(
        0.8, 0.79, rel_margin=0.15, abs_margin=0.05
    )
    assert flagged is False


def test_report_is_frozen_and_json_serializable():
    report = GeneralizationReport(
        kind="group",
        calibration_score=0.9,
        heldout_score=0.5,
        relative_drop=0.444,
        absolute_drop=0.4,
        gap=0.4,
        flagged=True,
        estimate="held_out",
        cv_deferred=False,
        within_group_caveat=False,
        dataset_changed=False,
        warning=None,
        gap_margin_relative=0.15,
        gap_margin_absolute=0.05,
    )
    payload = report.to_dict()
    assert payload["kind"] == "group"
    assert payload["heldout_score"] == 0.5
    assert payload["flagged"] is True
    # Round-trips through JSON without custom encoders.
    import json

    assert json.loads(json.dumps(payload)) == payload


# -- D2: the report-only held-out pass on the winner --------------------------


class _Winner:
    """A minimal stand-in for a winning ``Trial`` (only ``params``/``gap`` read)."""

    def __init__(self, params: dict, *, score: float = 0.9, gap: float | None = 0.12):
        self.params = params
        self.score = score
        self.gap = gap


class _NamedImage:
    """A minimal plate stand-in carrying a ``name`` (no detection needed)."""

    def __init__(self, name: str):
        self.name = name


class _StubScorer:
    """A scorer whose ``check.groupby`` lets ``infer_group_key`` resolve a column."""

    def __init__(self, score: float):
        self._score = score


class _StubEvaluator:
    """Records the images it was handed and returns a fixed held-out score."""

    def __init__(self, heldout_score: float):
        self._heldout_score = heldout_score
        self.seen_images: list = []
        self.seen_params: Any = None

    def evaluate(self, base, scorer, params, images, **_kwargs):
        self.seen_images = list(images)
        self.seen_params = params

        class _Result:
            score = self._heldout_score

        # bind the captured score
        _Result.score = self._heldout_score
        return _Result()


class _StubSpec:
    """A spec stand-in exposing only the fields ``run_held_out`` reads."""

    def __init__(self, evaluator, *, pipeline="PIPE", scorer=None):
        self.evaluator = evaluator
        self.pipeline = pipeline
        self.scorer = scorer
        from phenotypic.tune._evaluation import HeldOutConfig

        self.held_out = HeldOutConfig()


def _images_by_name(names):
    return {n: _NamedImage(n) for n in names}


def test_run_held_out_group():
    cal = ["c0", "c1", "c2"]
    held = ["h0", "h1"]
    images_by_name = _images_by_name(cal + held)
    identity = "deadbeef" * 8
    split = Split(
        calibration=cal,
        held_out=held,
        kind="group",
        group_key="Metadata_Group",
        dataset_identity=identity,
        within_group_caveat=False,
    )
    evaluator = _StubEvaluator(heldout_score=0.5)
    spec = _StubSpec(evaluator)
    winner = _Winner({"a": 1}, score=0.9, gap=0.12)

    report = run_held_out(spec, winner, split, images_by_name)

    assert report.kind == "group"
    assert report.calibration_score == pytest.approx(0.9)
    assert report.heldout_score == pytest.approx(0.5)
    assert report.gap == pytest.approx(0.4)
    assert report.cv_deferred is False
    assert report.within_group_caveat is False
    assert report.estimate == "held_out"
    # Evaluated ONLY the held-out plates (by name-membership).
    assert {im.name for im in evaluator.seen_images} == set(held)


def test_run_held_out_within_group_caveat():
    cal = ["c0", "c1", "c2"]
    held = ["h0", "h1"]
    images_by_name = _images_by_name(cal + held)
    split = Split(
        calibration=cal,
        held_out=held,
        kind="within_group",
        group_key="Metadata_Group",
        dataset_identity="ab" * 32,
        within_group_caveat=True,
    )
    evaluator = _StubEvaluator(heldout_score=0.6)
    spec = _StubSpec(evaluator)
    winner = _Winner({"a": 1}, score=0.9, gap=0.1)

    report = run_held_out(spec, winner, split, images_by_name)

    assert report.kind == "within_group"
    assert report.gap == pytest.approx(0.3)
    assert report.within_group_caveat is True
    assert report.cv_deferred is False
    assert report.warning is not None  # within-group caveat note


def test_run_held_out_data_poor_fallback():
    cal = ["c0", "c1", "c2"]
    images_by_name = _images_by_name(cal)
    split = Split(
        calibration=cal,
        held_out=[],
        kind="none",
        group_key=None,
        dataset_identity="cd" * 32,
        within_group_caveat=False,
    )
    evaluator = _StubEvaluator(heldout_score=0.5)  # must NOT be called
    spec = _StubSpec(evaluator)
    winner = _Winner({"a": 1}, score=0.9, gap=0.12)

    report = run_held_out(spec, winner, split, images_by_name)

    assert report.kind == "none"
    assert report.gap is None
    assert report.heldout_score is None
    assert report.flagged is False
    assert report.estimate == "calibration_stability"
    assert report.cv_deferred is True
    assert report.calibration_stability == pytest.approx(0.12)
    assert report.warning is not None
    # The held-out evaluator was never invoked (no untouched held-out set).
    assert evaluator.seen_images == []


def test_run_held_out_dataset_changed_flag():
    cal = ["c0", "c1", "c2"]
    held = ["h0", "h1"]
    images_by_name = _images_by_name(cal + held)
    split = Split(
        calibration=cal,
        held_out=held,
        kind="group",
        group_key="Metadata_Group",
        dataset_identity="00" * 32,  # stale identity
        within_group_caveat=False,
    )
    evaluator = _StubEvaluator(heldout_score=0.5)
    spec = _StubSpec(evaluator)
    winner = _Winner({"a": 1}, score=0.9, gap=0.12)

    current = _dataset_identity(list(images_by_name.values()))
    report = run_held_out(
        spec, winner, split, images_by_name, current_identity=current
    )

    assert report.dataset_changed is True
    assert report.warning is not None


def test_run_held_out_with_real_qc_scorer(tmp_path):
    """A faithful end-to-end-ish check using the real ``QCScorer.evaluate`` stub.

    Uses the stub evaluator (no real detection) but a real ``QCScorer`` so the
    report carries a real scorer reference; verifies the group-tier path builds
    the expected fields.
    """
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["h0"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        )
    )
    cal = ["c0", "c1", "c2"]
    held = ["h0", "h1"]
    images_by_name = _images_by_name(cal + held)
    split = Split(
        calibration=cal,
        held_out=held,
        kind="group",
        group_key="Metadata_ImageName",
        dataset_identity=_dataset_identity(list(images_by_name.values())),
        within_group_caveat=False,
    )
    evaluator = _StubEvaluator(heldout_score=0.7)
    spec = _StubSpec(evaluator, scorer=scorer)
    winner = _Winner({"a": 1}, score=0.85, gap=0.05)

    report = run_held_out(spec, winner, split, images_by_name)
    assert report.kind == "group"
    assert report.heldout_score == pytest.approx(0.7)
    assert report.dataset_changed is False
