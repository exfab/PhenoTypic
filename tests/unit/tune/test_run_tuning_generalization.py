"""4.5p2 D3 — ``run_tuning`` writes the split + ``generalization.json``.

Integration on a tiny multi-plate synthetic set: ``run_tuning`` resolves the
held-out split, runs the search on **calibration plates only**, and writes the
generalization report. The data-poor single-plate path falls back to the
calibration-stability estimate. Resume reuses the persisted split.

The synthetic plates are clones of ``load_synth_yeast_plate()`` re-named and
tagged with a ``Metadata_Group`` column so the split reaches the ``"group"`` /
``"within_group"`` tiers (a non-empty held-out set); ``_load_images`` is
monkeypatched to return them (the on-disk scan is exercised elsewhere).
"""
from __future__ import annotations

import json

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    QCScorer,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune._tune_cli import _run as run_mod
from phenotypic.tune._tune_cli._run import run_tuning


def _named_plate(name: str, group: str | None = None):
    """A clone of the synthetic yeast plate, re-named (+ optional group tag)."""
    image = load_synth_yeast_plate()
    image.name = name
    if group is not None:
        image.metadata["Metadata_Group"] = group
    return image


def _multi_plate_set(n: int = 8) -> list:
    """``n`` plates split across two groups → a ``"group"`` held-out tier."""
    groups = ["A", "B"]
    return [
        _named_plate(f"plate_{i:02d}", group=groups[i % 2]) for i in range(n)
    ]


def _layout_csv(tmp_path, names) -> str:
    """A layout CSV with 96 expected objects per plate name (the synth plate)."""
    rows = []
    for name in names:
        rows.extend(
            {"Metadata_ImageName": name, "Object_Label": j} for j in range(96)
        )
    csv = tmp_path / "layout.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def _spec(tmp_path, names) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(check=ExpectedVsDetectedCount(
            metadata=_layout_csv(tmp_path, names),
            groupby=["Metadata_ImageName"],
        )),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_writes_split_and_generalization(tmp_path):
    images = _multi_plate_set(8)
    names = [im.name for im in images]
    spec = _spec(tmp_path, names)
    out = tmp_path / "out"

    run_tuning(spec, images, out)

    split_path = io.split_assignment_path(out)
    gen_path = io.generalization_path(out)
    assert split_path.exists()
    assert gen_path.exists()
    payload = json.loads(gen_path.read_text())
    # The report carries the documented fields.
    for key in (
        "kind", "calibration_score", "heldout_score", "gap", "flagged",
        "estimate", "cv_deferred", "within_group_caveat", "dataset_changed",
        "gap_margin_relative", "gap_margin_absolute",
    ):
        assert key in payload
    assert payload["kind"] in {"group", "within_group", "none"}


def test_search_runs_on_calibration_only(tmp_path, monkeypatch):
    images = _multi_plate_set(8)
    names = [im.name for im in images]
    spec = _spec(tmp_path, names)
    out = tmp_path / "out"

    seen: dict = {}
    original = run_mod.TuningEngine.optimize

    def _spy_optimize(self, search_images):
        seen["names"] = {im.name for im in search_images}
        return original(self, search_images)

    monkeypatch.setattr(run_mod.TuningEngine, "optimize", _spy_optimize)

    run_tuning(spec, images, out)

    split = json.loads(io.split_assignment_path(out).read_text())
    held = set(split["held_out"])
    assert held  # a non-empty held-out set (group / within-group tier)
    # The search NEVER touched the held-out plates.
    assert seen["names"].isdisjoint(held)
    assert seen["names"] == set(split["calibration"])


def test_resume_reuses_split_generalization(tmp_path):
    images = _multi_plate_set(8)
    names = [im.name for im in images]
    spec = _spec(tmp_path, names)
    out = tmp_path / "out"

    run_tuning(spec, images, out)
    first_split = io.split_assignment_path(out).read_text()

    # A second run to the same -o reuses the persisted split verbatim.
    run_tuning(spec, list(images), out)
    second_split = io.split_assignment_path(out).read_text()
    assert second_split == first_split


def test_data_poor_writes_fallback_generalization(tmp_path):
    # A single plate → kind="none" → the calibration-stability fallback.
    images = [_named_plate("plate_only")]
    spec = _spec(tmp_path, ["plate_only"])
    out = tmp_path / "out"

    run_tuning(spec, images, out)

    gen = json.loads(io.generalization_path(out).read_text())
    assert gen["kind"] == "none"
    assert gen["gap"] is None
    assert gen["heldout_score"] is None
    assert gen["flagged"] is False
    assert gen["estimate"] == "calibration_stability"
    assert gen["cv_deferred"] is True
    # The Phase-1 deliverables still exist (behavior otherwise unchanged).
    assert io.best_pipeline_path(out).exists()
    assert io.trials_parquet_path(out).exists()
