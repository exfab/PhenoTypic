"""Identity and serialization contract for ``ImagePipeline.plots``."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from pydantic import BaseModel

from phenotypic import ImagePipeline
from phenotypic.analysis import GridOccupancy, ICC, LinearLagModel
from phenotypic.measure import MeasureSymmetricZones
from phenotypic.abc_.plotting import PhtPlot
from phenotypic._core._pipeline_parts._serializable_pipeline import (
    PipelineLoadWarning,
)
from phenotypic.plotting import (
    AnalysisInput,
    PipelineObjectRef,
    PlotBinding,
    PlotColonyMetricOverTime,
    PlotMeasTimeSeries,
)
from phenotypic.sdk_._qc_recipe import QcRecipeEntry


def test_none_normalizes_to_empty_bindings() -> None:
    assert ImagePipeline(plots=None).get_plots() == []


def test_set_plots_normalizes_objects_for_runtime_and_serialization() -> None:
    plot = PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])
    pipeline = ImagePipeline()

    pipeline.set_plots([plot])

    binding = pipeline.get_plots()[0]
    assert isinstance(binding, PlotBinding)
    assert binding.plot is plot
    loaded = ImagePipeline.from_json(pipeline.to_json())
    assert isinstance(loaded.get_plots()[0], PlotBinding)


def test_measurer_reference_round_trip_preserves_shared_identity() -> None:
    zones = MeasureSymmetricZones()
    pipeline = ImagePipeline(meas={"zones": zones}, plots=[zones])

    loaded = ImagePipeline.from_json(pipeline.to_json())

    binding = loaded.get_plots()[0]
    assert binding.ref is not None
    assert binding.ref.slot == "meas"
    assert binding.ref.key == "zones"
    assert binding.plot is loaded.get_meas()["zones"]


def test_model_reference_round_trip_preserves_shared_identity() -> None:
    model = LinearLagModel(
        on="Size_Area", groupby=["MetadataGenetic_Strain"]
    )
    loaded = ImagePipeline.from_json(
        ImagePipeline(model=model, plots=[model]).to_json()
    )
    assert loaded.get_plots()[0].plot is loaded.get_model()


def test_equal_but_distinct_inline_plot_is_not_made_a_reference() -> None:
    configured = PlotMeasTimeSeries(
        environment_by=["env"], replicate_by=["rep"]
    )
    equal_inline = configured.model_copy(deep=True)
    pipeline = ImagePipeline(plots=[configured, PlotBinding(id="copy", plot=equal_inline)])
    assert pipeline.get_plots()[0].plot is configured
    assert pipeline.get_plots()[1].plot is equal_inline
    assert pipeline.get_plots()[0].ref is None
    assert pipeline.get_plots()[1].ref is None


def test_inline_builtin_round_trip_uses_module_and_qualname() -> None:
    plot = PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])
    pipeline = ImagePipeline(plots=[plot])
    payload = json.loads(pipeline.to_json())
    assert payload["plots"][0]["inline"]["module"] == type(plot).__module__
    loaded = ImagePipeline.from_json(json.dumps(payload))
    assert loaded.get_plots()[0].plot == plot


def test_colony_metric_plot_round_trip_preserves_on_and_overrides() -> None:
    plot = PlotColonyMetricOverTime(
        on="Shape_MedianRadius",
        groupby=["MetadataCondition_Treatment"],
        replicate_label="MetadataSample_TechnicalReplicate",
    )

    serialized = ImagePipeline(plots=[plot]).to_json()
    payload = json.loads(serialized)
    params = payload["plots"][0]["inline"]["params"]
    loaded = ImagePipeline.from_json(serialized)

    loaded_plot = loaded.get_plots()[0].plot
    assert set(params) == {
        "connect",
        "groupby",
        "on",
        "replicate_label",
        "strain_label",
        "time",
    }
    assert isinstance(loaded_plot, PlotColonyMetricOverTime)
    assert loaded_plot == plot


def test_duplicate_ids_fail_with_ids_in_message() -> None:
    first = PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])
    second = first.model_copy(deep=True)
    with pytest.raises(ValidationError, match="duplicate plot binding ids"):
        ImagePipeline(plots=[first, second])


def test_ids_colliding_after_path_sanitization_fail() -> None:
    first = PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])
    second = first.model_copy(deep=True)
    with pytest.raises(
        ValidationError, match="collide after filesystem sanitization"
    ):
        ImagePipeline(
            plots=[
                PlotBinding(id="growth qc", plot=first),
                PlotBinding(id="growth-qc", plot=second),
            ]
        )


def test_qc_reference_requires_plot_qc_capability() -> None:
    entry = QcRecipeEntry(
        cls=ICC,
        params={},
        instance_id="qc-icc",
    )
    with pytest.raises(ValidationError, match="does not implement PlotQc"):
        ImagePipeline(
            qc=[entry],
            plots=[
                PlotBinding(
                    id="icc-output",
                    ref=PipelineObjectRef(slot="qc", key=entry.instance_id),
                )
            ],
        )


def test_qc_reference_accepts_plot_qc_capability() -> None:
    entry = QcRecipeEntry(
        cls=GridOccupancy,
        params={},
        instance_id="qc-grid",
    )
    pipeline = ImagePipeline(
        qc=[entry],
        plots=[
            PlotBinding(
                id="grid-output",
                ref=PipelineObjectRef(slot="qc", key=entry.instance_id),
            )
        ],
    )
    assert pipeline.get_plots()[0].ref is not None


def test_qc_reference_accepts_explicit_analysis_input() -> None:
    entry = QcRecipeEntry(
        cls=GridOccupancy,
        params={},
        instance_id="qc-grid",
    )
    pipeline = ImagePipeline(
        qc=[entry],
        plots=[
            PlotBinding(
                id="grid-output",
                ref=PipelineObjectRef(slot="qc", key=entry.instance_id),
                input=AnalysisInput(analysis_id="LinearLagModel"),
            )
        ],
    )

    assert pipeline.get_plots()[0].input == AnalysisInput(
        analysis_id="LinearLagModel"
    )


def test_measurement_plot_rejects_analysis_input() -> None:
    plot = PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])
    with pytest.raises(ValidationError, match="always consumes the measurement mirror"):
        ImagePipeline(
            plots=[
                PlotBinding(
                    id="measurements",
                    plot=plot,
                    input=AnalysisInput(analysis_id="LinearLagModel"),
                )
            ]
        )


def test_nested_pipeline_with_plots_is_rejected() -> None:
    nested = ImagePipeline(
        plots=[PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])]
    )
    with pytest.raises(ValidationError, match="nested pipeline operation"):
        ImagePipeline(ops={"nested": nested})


def test_plain_pht_plot_rejects_missing_actionable_lifecycle() -> None:
    class _PlainPlot(BaseModel, PhtPlot):
        pass

    with pytest.raises(ValidationError, match="exactly one actionable"):
        ImagePipeline(plots=[_PlainPlot()])


def test_tolerant_load_skips_plot_reference_to_skipped_model() -> None:
    model = LinearLagModel(on="Size_Area", groupby=["strain"])
    payload = json.loads(ImagePipeline(model=model, plots=[model]).to_json())
    payload["model"]["class"] = "RemovedLinearLagModel"
    warnings: list[PipelineLoadWarning] = []

    loaded = ImagePipeline.from_json(
        json.dumps(payload),
        skip_unknown_analyzers=True,
        load_warnings=warnings,
    )

    assert loaded.get_model() is None
    assert loaded.get_plots() == []
    assert [(warning.slot, warning.name) for warning in warnings] == [
        ("model", "model")
    ]


def test_tolerant_load_skips_unknown_inline_plot_class() -> None:
    plot = PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])
    payload = json.loads(ImagePipeline(plots=[plot]).to_json())
    payload["plots"][0]["inline"]["qualname"] = "RemovedPlot"
    warnings: list[PipelineLoadWarning] = []

    loaded = ImagePipeline.from_json(
        json.dumps(payload),
        skip_unknown_analyzers=True,
        load_warnings=warnings,
    )

    assert loaded.get_plots() == []
    assert len(warnings) == 1
    assert warnings[0].slot == "plot"
    assert warnings[0].name == "PlotMeasTimeSeries"
