"""End-to-end minimize-cost acceptance smoke (multi-plate, seeded).

Drives the REAL tuner (``run_tuning`` -> ``TuningEngine.optimize`` -> an Optuna
**minimize** study) over several synthetic plates (the canonical
``load_synth_yeast_plate`` fixture, copied under distinct names so the
cross-image robust aggregate runs). Asserts the cutover end-to-end: the study
minimizes and carries the convention stamp, the winner is the lowest-cost
trial, and the achieved cost is low (a good pipeline is reachable) — the
whole-system proof that complements the synthetic-objective regressions in
``tests/unit/tune/test_cost_convention_regression.py``.

Implementation note: the raw ``make_synthetic_plate`` arrays detect only
~84/96 colonies with ``GaussianBlur + OtsuDetector`` at any resolution (it is
detector-limited, not resolution-limited), so a small-sigma config could not
reach the ``best_value < 0.5`` "good pipeline reachable" bar. The pre-rendered
``load_synth_yeast_plate`` fixture detects exactly 96 with ``sigma=1.0`` and
merges/erases colonies at large sigma, so it gives the clean, discriminating
count signal this acceptance gate needs without loosening any threshold.
"""
from __future__ import annotations

import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")  # the `tune` extra

from phenotypic import GridImage, ImagePipeline  # noqa: E402
from phenotypic.analysis import ExpectedVsDetectedCount  # noqa: E402
from phenotypic.data import load_synth_yeast_plate  # noqa: E402
from phenotypic.detect import OtsuDetector  # noqa: E402
from phenotypic.enhance import GaussianBlur  # noqa: E402
from phenotypic.sdk_ import _io_constants as io  # noqa: E402
from phenotypic.tune import (  # noqa: E402
    Budget,
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune.score import QCScorer  # noqa: E402
from phenotypic.tune.strategy import OptunaConfig  # noqa: E402
from phenotypic.tune._tune_cli._run import _STUDY_NAME, run_tuning  # noqa: E402

_NROWS, _NCOLS = 8, 12
_EXPECTED = _NROWS * _NCOLS  # 96 colonies per plate


def _seeded_plates(n: int = 4) -> list[GridImage]:
    """``n`` cleanly-detectable plates under DISTINCT names.

    The canonical ``load_synth_yeast_plate`` fixture detects exactly 96 colonies
    with ``GaussianBlur(sigma=1.0) + OtsuDetector`` (and collapses at large
    sigma), so the count signal is clean and the small-sigma config is reachably
    low-cost. Copies carry distinct ``name``s so the multi-image cross-plate
    robust aggregate path executes (the count is identical per plate, so the
    discriminating axis is the sigma knob, not plate-to-plate noise).
    """
    plates = []
    for i in range(n):
        plate = load_synth_yeast_plate()
        plate.name = f"plate_{i:02d}"
        plates.append(plate)
    return plates


def _layout_csv(tmp_path, names) -> str:
    """A layout CSV declaring 96 expected objects per plate (for the count check)."""
    rows = [
        {"Metadata_ImageName": name, "Object_Label": j}
        for name in names
        for j in range(_EXPECTED)
    ]
    csv = tmp_path / "layout.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def _spec(tmp_path, names) -> TuningSpec:
    # GaussianBlur sigma is the discriminating knob: a small sigma keeps colonies
    # separable (count ~= 96 -> low cost); a large sigma merges/erases them
    # (count far from 96 -> high cost). Minimization must prefer the small sigma.
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="0.sigma", domain=Categorical(choices=(1.0, 12.0, 40.0))),)
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=_layout_csv(tmp_path, names),
                groupby=["Metadata_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=6, sampler="tpe", seed=0),
        budget=Budget(),
    )


def test_minimize_cost_end_to_end_winner_is_low_cost(tmp_path):
    images = _seeded_plates(4)
    names = [im.name for im in images]
    out = tmp_path / "out"

    run_tuning(_spec(tmp_path, names), images, out)

    # 1. Deliverables were written.
    assert io.best_pipeline_path(out).exists()
    assert io.trials_parquet_path(out).exists()

    # 2. The Optuna study MINIMIZES and carries the convention stamp (no silent
    #    maximize; the name was bumped to the cost-era study).
    db = io.resolve_study_db_path(out)
    study = optuna.load_study(storage=f"sqlite:///{db}", study_name=_STUDY_NAME)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert study.user_attrs.get("tune_convention") == "minimize-cost-v1"

    # 3. The winner is the LOWEST-cost trial, and a good pipeline is reachable.
    values = [t.value for t in study.trials if t.value is not None]
    assert values, "no completed trials"
    assert study.best_value == pytest.approx(min(values))
    assert study.best_value < 0.5  # small-sigma config detects ~96 -> low cost
    # Minimization discriminated: when >=2 distinct configs were evaluated, the
    # best is strictly better than the worst tried (the large-sigma config wrecks
    # the count -> high cost).
    if len(set(values)) > 1:
        assert study.best_value < max(values)
