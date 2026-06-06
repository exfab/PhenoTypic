"""Integration tests for the Space view (Task C2).

The Space view infers a search space from the bound run's
``tuning_spec.json`` (preferred) or ``pipeline.json`` (fallback), renders one
editable knob row per flat / presence knob (nested leaves read-only), and exports
the edited space back to ``deliverables/tuning_spec.json`` via the pure
:func:`~phenotypic.gui.tune._space.space_to_spec`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _runnable_spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A round-trippable existing ``TuningSpec`` written to a run dir."""
    from phenotypic import ImagePipeline
    from phenotypic.analysis import ExpectedVsDetectedCount
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import GaussianBlur
    from phenotypic.tune import (
        Budget,
        Categorical,
        Evaluator,
        Knob,
        QCScorer,
        RandomConfig,
        SearchSpace,
        TuningSpec,
    )

    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["plate1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),)
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=RandomConfig(n_trials=17),
        budget=Budget(n_trials=23),
    )


def _space_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A loaded tune app whose run dir carries a tuning_spec.json + a journal."""
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import trials_parquet_path, tuning_spec_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    spec = _runnable_spec(tmp_path)
    spec_path = tuning_spec_path(tmp_path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(spec.model_dump_json(indent=2))

    parquet = trials_parquet_path(tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[
            Trial(number=0, params={"0.sigma": 1.0}, score=0.5, terms={}, n_images=2),
        ]
    ).to_parquet(parquet)

    root = TuneRunRoot.discover(tmp_path)
    return create_app(root=root, url_prefix="/tune/"), root


def test_space_view_renders_knob_rows_and_export(tmp_path: Path) -> None:
    app, _root = _space_app(tmp_path)
    layout = str(app.layout)
    # One editable row per inferred flat knob; the Export button + note.
    assert "tune-space-knob-row" in layout
    assert "tune-btn-space-export" in layout
    assert "tune-space-note" in layout
    # The sigma float-range knob's low/high inputs are present.
    assert "0.sigma" in layout


def test_space_view_empty_when_no_pipeline(tmp_path: Path) -> None:
    """A run dir with neither spec nor pipeline shows the pick-a-pipeline prompt."""
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    parquet = trials_parquet_path(tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[Trial(number=0, params={}, score=0.5, terms={}, n_images=1)]
    ).to_parquet(parquet)
    root = TuneRunRoot.discover(tmp_path)
    app = create_app(root=root, url_prefix="/tune/")
    layout = str(app.layout)
    assert "tune-space-empty" in layout


def test_export_writes_tuning_spec_preserving_scorer(tmp_path: Path) -> None:
    """The export helper writes a spec that preserves the run's scorer/strategy."""
    from phenotypic.gui.tune._callbacks import write_space_spec
    from phenotypic.tools_ import tuning_spec_path
    from phenotypic.tune import TuningSpec

    _app, root = _space_app(tmp_path)
    written = write_space_spec(root, edits={})
    assert written == tuning_spec_path(tmp_path)

    reloaded = TuningSpec.model_validate_json(written.read_text())
    # OQ8: the run's scorer / strategy / budget survive the re-export.
    assert type(reloaded.scorer).__name__ == "QCScorer"
    assert type(reloaded.strategy).__name__ == "RandomConfig"
    assert reloaded.strategy.n_trials == 17
    assert reloaded.budget.n_trials == 23
    # The exported search space is the inferred flat knobs (sigma + the Otsu/blur
    # flat fields), not the original single-knob space.
    keys = {k.key for k in reloaded.search_space.knobs}
    assert "0.sigma" in keys
