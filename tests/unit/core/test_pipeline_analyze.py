"""Unit tests for ``ImagePipeline.analyze`` / ``_analyze_steps`` and the
``filters`` / ``model`` JSON round-trip introduced for the analysis GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import (
    EdgeCorrector,
    LinearSoftplusModel,
    LogGrowthModel,
    TukeyOutlierRemover,
)


@pytest.fixture(scope="module")
def area_measurements() -> pd.DataFrame:
    """Real per-image measurement frame shipped with the repo.

    Columns include ``Metadata_Strain``, ``Metadata_Time``, ``Shape_Area``
    and the grouping/metadata columns the analysis classes rely on.
    """
    data_path = (
        Path(__file__).parents[3]
        / "src"
        / "phenotypic"
        / "data"
        / "meas"
        / "area_meas.csv"
    )
    return pd.read_csv(data_path)


class TestAnalyzeContract:
    """Contract for ``ImagePipelineCore.analyze`` / ``_analyze_steps``."""

    def test_no_model_raises(self):
        pipe = ImagePipeline()
        with pytest.raises(ValueError, match="no analysis model configured"):
            pipe.analyze(pd.DataFrame())

    def test_filters_only_chain_runs_via_analyze_steps(self, area_measurements):
        pipe = ImagePipeline(
            filters=[
                TukeyOutlierRemover(
                    on="Shape_Area",
                    groupby=["Metadata_Strain"],
                    k=3.0,
                ),
            ],
        )
        steps = pipe._analyze_steps(area_measurements)
        assert len(steps) == 1
        label, df = steps[0]
        assert label == "TukeyOutlierRemover"
        # Outlier removal should not introduce rows.
        assert len(df) <= len(area_measurements)

    def test_analyze_runs_filter_then_model(self, area_measurements):
        pipe = ImagePipeline(
            filters=[
                TukeyOutlierRemover(
                    on="Shape_Area",
                    groupby=["Metadata_Strain"],
                    k=3.0,
                ),
            ],
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        fit = pipe.analyze(area_measurements)
        # Model fits emit one row per group.
        assert isinstance(fit, pd.DataFrame)
        assert len(fit) <= area_measurements["Metadata_Strain"].nunique()
        assert len(fit) > 0

    def test_analyze_steps_includes_terminal_model_entry(self, area_measurements):
        pipe = ImagePipeline(
            filters=[
                TukeyOutlierRemover(
                    on="Shape_Area",
                    groupby=["Metadata_Strain"],
                    k=3.0,
                ),
            ],
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        steps = pipe._analyze_steps(area_measurements)
        assert [label for label, _ in steps] == [
            "TukeyOutlierRemover",
            "LogGrowthModel",
        ]


class TestSetters:
    def test_set_filters_list_dedupes(self):
        pipe = ImagePipeline()
        pipe.set_filters([
            TukeyOutlierRemover(on="Shape_Area", groupby=["x"]),
            TukeyOutlierRemover(on="Shape_Area", groupby=["x"]),
        ])
        keys = list(pipe.get_filters().keys())
        assert keys == ["TukeyOutlierRemover", "TukeyOutlierRemover_1"]

    def test_set_filters_dict_preserves_keys(self):
        pipe = ImagePipeline()
        flt = TukeyOutlierRemover(on="Shape_Area", groupby=["x"])
        pipe.set_filters({"my_tukey": flt})
        assert "my_tukey" in pipe.get_filters()

    def test_set_filters_rejects_invalid_type(self):
        pipe = ImagePipeline()
        with pytest.raises(TypeError):
            pipe.set_filters("not a list or dict")

    def test_set_model_clears_with_none(self):
        pipe = ImagePipeline(
            model=LogGrowthModel(on="x", groupby=["y"], n_jobs=1),
        )
        assert pipe.get_model() is not None
        pipe.set_model(None)
        assert pipe.get_model() is None

    def test_set_model_rejects_non_model_fitter(self):
        pipe = ImagePipeline()
        # SetAnalyzer (not ModelFitter) — must reject.
        with pytest.raises(TypeError, match="ModelFitter"):
            pipe.set_model(TukeyOutlierRemover(on="x", groupby=["y"]))


class TestJSONRoundTrip:
    def test_filters_and_model_round_trip(self):
        edge = EdgeCorrector(
            on="Shape_Area",
            groupby=["Metadata_Plate"],
            top_n=5,
            nrows=16,
            ncols=24,
        )
        tukey = TukeyOutlierRemover(
            on="Shape_Area",
            groupby=["Metadata_Plate"],
            k=2.0,
        )
        model = LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Plate"],
            time_label="Metadata_Time",
            lam=1.5,
            beta=3.0,
            n_jobs=1,
        )
        pipe = ImagePipeline(filters=[edge, tukey], model=model)
        loaded = ImagePipeline.from_json(pipe.to_json())

        assert list(loaded.get_filters().keys()) == [
            "EdgeCorrector",
            "TukeyOutlierRemover",
        ]
        loaded_edge = loaded.get_filters()["EdgeCorrector"]
        assert isinstance(loaded_edge, EdgeCorrector)
        assert loaded_edge.top_n == 5
        assert loaded_edge.nrows == 16
        assert loaded_edge.ncols == 24

        loaded_tukey = loaded.get_filters()["TukeyOutlierRemover"]
        assert isinstance(loaded_tukey, TukeyOutlierRemover)
        assert loaded_tukey.k == 2.0

        loaded_model = loaded.get_model()
        assert isinstance(loaded_model, LogGrowthModel)
        assert loaded_model.lam == 1.5
        assert loaded_model.beta == 3.0
        assert loaded_model.n_jobs == 1

    def test_empty_pipeline_emits_empty_filters_and_null_model(self):
        config = json.loads(ImagePipeline().to_json())
        assert config["filters"] == {}
        assert config["model"] is None

    def test_legacy_json_without_keys_loads_with_defaults(self):
        legacy = json.dumps({
            "name": "legacy",
            "pipe_cfgs": {},
            "meas": {},
            "post": {},
        })
        loaded = ImagePipeline.from_json(legacy)
        assert loaded.get_filters() == {}
        assert loaded.get_model() is None

    def test_model_in_json_must_deserialize_to_model_fitter(self):
        bad = json.dumps({
            "name": "bad",
            "pipe_cfgs": {},
            "meas": {},
            "post": {},
            "filters": {},
            # SetAnalyzer subclass that is not a ModelFitter — must error.
            "model": {
                "class": "EdgeCorrector",
                "params": {
                    "on": "Shape_Area",
                    "groupby": ["Metadata_Plate"],
                },
            },
        })
        with pytest.raises(TypeError, match="ModelFitter"):
            ImagePipeline.from_json(bad)

    def test_linear_softplus_model_round_trip(self):
        # LinearSoftplusModel uses ``num_workers`` (not ``n_jobs``) at the
        # constructor; SetAnalyzer stores it as ``n_jobs``. The alias map
        # must round-trip both shapes.
        model = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Plate"],
            time_label="Metadata_Time",
            num_workers=2,
        )
        pipe = ImagePipeline(model=model)
        loaded = ImagePipeline.from_json(pipe.to_json())
        loaded_model = loaded.get_model()
        assert isinstance(loaded_model, LinearSoftplusModel)
        assert loaded_model.n_jobs == 2
