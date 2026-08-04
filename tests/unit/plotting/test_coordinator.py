"""Lifecycle dispatch and output layout for configured plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, PrivateAttr
from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic.analysis import GridOccupancy
from phenotypic.abc_.plotting import (
    PlotAnalysis,
    PlotImage,
    PlotMeas,
    PlotOutput,
    PlotPage,
    PlotQc,
)
from phenotypic.plotting._pipeline import (
    AnalysisInput,
    AnalysisRegistry,
    MeasurementInput,
    PlotBinding,
    PlotCoordinator,
    PipelineObjectRef,
    QcPlotSubject,
)
from phenotypic.sdk_ import plots_dir
from phenotypic.sdk_._qc_recipe import QcRecipeEntry


class _ImagePlot(BaseModel, PlotImage):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _seen: object = PrivateAttr(default=None)

    def inspect(self, subject=None, *, for_save=False, **overrides):
        self._seen = subject
        return plt.figure()


class _FailingImagePlot(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise RuntimeError("plot publication failed")


class _MeasPlot(BaseModel, PlotMeas):
    _seen: object = PrivateAttr(default=None)

    def inspect(self, subject=None, *, for_save=False, **overrides):
        self._seen = subject
        return plt.figure()


class _MultiImagePlot(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        return PlotOutput(
            pages=(
                PlotPage(key="first", figure=plt.figure()),
                PlotPage(key="second", figure=plt.figure()),
            )
        )


class _AnalysisPlot(BaseModel, PlotAnalysis):
    _seen: object = PrivateAttr(default=None)

    def inspect(self, subject=None, *, for_save=False, **overrides):
        self._seen = subject
        return plt.figure()


class RefreshableAnalysisPlot(_AnalysisPlot):
    _analyzed: object = PrivateAttr(default=None)

    def analyze(self, measurements):
        self._analyzed = measurements
        return pd.DataFrame({"fit": [1.0]})


class _QcPlot(BaseModel, PlotQc):
    _seen: object = PrivateAttr(default=None)

    def inspect(self, subject=None, *, for_save=False, **overrides):
        self._seen = subject
        return plt.figure()


def test_image_plot_uses_deliverables_plot_layout(tmp_path) -> None:
    plot = _ImagePlot()
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=plot)])
    subject = object()
    PlotCoordinator(pipeline, tmp_path).emit_image(
        subject, dataset="dataset", image_stem="plate-1"
    )
    assert plot._seen is subject
    written = list((plots_dir(tmp_path) / "image" / "dataset").glob("*.png"))
    assert len(written) == 1
    assert written[0].name.startswith("plate-1-")


def test_image_plot_strict_mode_propagates_publication_failure(tmp_path) -> None:
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="image", plot=_FailingImagePlot())]
    )

    with pytest.raises(RuntimeError, match="plot publication failed"):
        PlotCoordinator(pipeline, tmp_path).emit_image(
            object(),
            dataset="dataset",
            image_stem="plate-1",
            strict=True,
        )


def test_image_plot_disambiguates_sanitized_and_casefold_collisions(
    tmp_path,
) -> None:
    plot = _ImagePlot()
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=plot)])
    coordinator = PlotCoordinator(pipeline, tmp_path)

    for image_stem in ("plate 1", "plate-1", "Plate-1"):
        coordinator.emit_image(
            object(), dataset="dataset", image_stem=image_stem
        )

    directory = plots_dir(tmp_path) / "image" / "dataset"
    written = sorted(path.name.casefold() for path in directory.glob("*.png"))
    assert len(written) == 3
    assert len(set(written)) == 3


def test_image_plot_output_name_is_stable_for_reruns(tmp_path) -> None:
    plot = _ImagePlot()
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=plot)])
    coordinator = PlotCoordinator(pipeline, tmp_path)

    coordinator.emit_image(object(), dataset="dataset", image_stem="plate 1")
    first = list((plots_dir(tmp_path) / "image" / "dataset").glob("*.png"))
    coordinator.emit_image(object(), dataset="dataset", image_stem="plate 1")
    second = list((plots_dir(tmp_path) / "image" / "dataset").glob("*.png"))

    assert first == second


def test_multi_page_image_plot_disambiguates_invocation_directories(
    tmp_path,
) -> None:
    plot = _MultiImagePlot()
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=plot)])
    coordinator = PlotCoordinator(pipeline, tmp_path)

    for image_stem in ("plate 1", "plate-1", "Plate-1"):
        coordinator.emit_image(
            object(), dataset="dataset", image_stem=image_stem
        )

    directory = plots_dir(tmp_path) / "image" / "dataset"
    invocation_paths = sorted(path for path in directory.iterdir() if path.is_dir())
    invocations = [path.name.casefold() for path in invocation_paths]
    assert len(invocations) == 3
    assert len(set(invocations)) == 3
    assert all((path / "manifest.json").exists() for path in invocation_paths)


def test_measurement_plot_receives_current_table(tmp_path) -> None:
    plot = _MeasPlot()
    table = pd.DataFrame({"Size_Area": [1.0]})
    pipeline = ImagePipeline(plots=[PlotBinding(id="measurements", plot=plot)])
    PlotCoordinator(pipeline, tmp_path).emit_measurements(table)
    assert plot._seen is table
    assert (plots_dir(tmp_path) / "measurements" / "default.png").exists()


def test_explicit_plots_base_avoids_double_deliverables_join(tmp_path) -> None:
    plot = _MeasPlot()
    table = pd.DataFrame({"Size_Area": [1.0]})
    pipeline = ImagePipeline(plots=[PlotBinding(id="measurements", plot=plot)])
    standalone_plots = tmp_path / "bundle" / "plots"

    PlotCoordinator(
        pipeline,
        tmp_path / "bundle",
        plots_base=standalone_plots,
    ).emit_measurements(table)

    assert (standalone_plots / "measurements" / "default.png").exists()
    assert not (tmp_path / "bundle" / "deliverables").exists()


def test_analysis_input_is_resolved_on_each_dispatch(tmp_path) -> None:
    plot = _AnalysisPlot()
    selected = pd.DataFrame({"lag": [2.0]})
    registry = AnalysisRegistry()
    registry.register("LinearLagModel", selected)
    pipeline = ImagePipeline(
        plots=[
            PlotBinding(
                id="analysis",
                plot=plot,
                input=AnalysisInput(analysis_id="LinearLagModel"),
            )
        ]
    )
    PlotCoordinator(pipeline, tmp_path).emit_analyses(
        pd.DataFrame({"Size_Area": [1.0]}), registry
    )
    assert plot._seen is selected


def test_analysis_producer_reuses_its_fitted_state_without_table_subject(
    tmp_path,
) -> None:
    plot = _AnalysisPlot()
    selected = pd.DataFrame({"lag": [2.0]})
    registry = AnalysisRegistry()
    registry.register("LinearLagModel", selected, producer=plot)
    pipeline = ImagePipeline(
        plots=[
            PlotBinding(
                id="analysis",
                plot=plot,
                input=AnalysisInput(analysis_id="LinearLagModel"),
            )
        ]
    )

    PlotCoordinator(pipeline, tmp_path).emit_analyses(
        pd.DataFrame({"Size_Area": [1.0]}), registry
    )

    assert plot._seen is None


def test_measurement_dependency_refreshes_analysis_producer_before_plot(
    tmp_path,
) -> None:
    plot = RefreshableAnalysisPlot()
    measurements = pd.DataFrame({"Size_Area": [1.0]})
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="analysis", plot=plot)]
    )

    refreshed = PlotCoordinator(pipeline, tmp_path).emit_analyses(
        measurements,
        AnalysisRegistry(),
        updated_input=MeasurementInput(),
        refresh_producers=True,
    )

    assert plot._analyzed is measurements
    assert plot._seen is None
    assert refreshed == ("RefreshableAnalysisPlot",)


def test_analysis_update_emits_only_matching_dependency(tmp_path) -> None:
    matching = _AnalysisPlot()
    other = _AnalysisPlot()
    selected = pd.DataFrame({"lag": [2.0]})
    registry = AnalysisRegistry()
    registry.register("LinearLagModel", selected)
    pipeline = ImagePipeline(
        plots=[
            PlotBinding(
                id="matching",
                plot=matching,
                input=AnalysisInput(analysis_id="LinearLagModel"),
            ),
            PlotBinding(
                id="other",
                plot=other,
                input=AnalysisInput(analysis_id="OtherModel"),
            ),
        ]
    )

    PlotCoordinator(pipeline, tmp_path).emit_analyses(
        pd.DataFrame({"Size_Area": [1.0]}),
        registry,
        updated_input=AnalysisInput(analysis_id="LinearLagModel"),
    )

    assert matching._seen is selected
    assert other._seen is None


def test_analysis_update_refreshes_reused_producer_with_default_input(
    tmp_path,
) -> None:
    plot = _AnalysisPlot()
    selected = pd.DataFrame({"lag": [2.0]})
    registry = AnalysisRegistry()
    registry.register("LinearLagModel", selected, producer=plot)
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="reused-model", plot=plot)]
    )

    PlotCoordinator(pipeline, tmp_path).emit_analyses(
        pd.DataFrame({"Size_Area": [1.0]}),
        registry,
        updated_input=AnalysisInput(analysis_id="LinearLagModel"),
    )

    assert plot._seen is None
    assert (plots_dir(tmp_path) / "reused-model" / "default.png").exists()


def test_qc_plot_receives_exact_successful_check_and_selected_input(
    tmp_path,
) -> None:
    plot = _QcPlot()
    measurements = pd.DataFrame({"Size_Area": [1.0]})
    analyzed_check = object()
    successful = SimpleNamespace(instance_id="qc", check=analyzed_check)
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="qc", plot=plot)]
    )
    qc_database = tmp_path / "qc.duckdb"

    PlotCoordinator(pipeline, tmp_path).emit_qc(
        measurements,
        AnalysisRegistry(),
        successful_modules={"qc": successful},
        qc_database=qc_database,
    )

    assert isinstance(plot._seen, QcPlotSubject)
    assert plot._seen.input_table is measurements
    assert plot._seen.analyzed_check is analyzed_check
    assert plot._seen.qc_database is qc_database


def test_measurement_update_refreshes_matching_standalone_qc_plot(
    tmp_path,
) -> None:
    plot = _QcPlot()
    measurements = pd.DataFrame({"Size_Area": [1.0]})
    pipeline = ImagePipeline(plots=[PlotBinding(id="qc", plot=plot)])

    PlotCoordinator(pipeline, tmp_path).emit_dependent_qc(
        measurements,
        AnalysisRegistry(),
        updated_input=MeasurementInput(),
    )

    assert isinstance(plot._seen, QcPlotSubject)
    assert plot._seen.input_table is measurements


def test_analysis_update_refreshes_only_matching_standalone_qc_plot(
    tmp_path,
) -> None:
    matching = _QcPlot()
    other = _QcPlot()
    selected = pd.DataFrame({"lag": [2.0]})
    registry = AnalysisRegistry()
    registry.register("LinearLagModel", selected)
    pipeline = ImagePipeline(
        plots=[
            PlotBinding(
                id="matching",
                plot=matching,
                input=AnalysisInput(analysis_id="LinearLagModel"),
            ),
            PlotBinding(
                id="other",
                plot=other,
                input=AnalysisInput(analysis_id="OtherModel"),
            ),
        ]
    )

    PlotCoordinator(pipeline, tmp_path).emit_dependent_qc(
        pd.DataFrame({"Size_Area": [1.0]}),
        registry,
        updated_input=AnalysisInput(analysis_id="LinearLagModel"),
    )

    assert isinstance(matching._seen, QcPlotSubject)
    assert matching._seen.input_table is selected
    assert other._seen is None


def test_qc_reference_uses_instance_id_when_output_id_is_custom(tmp_path) -> None:
    entry = QcRecipeEntry(
        cls=GridOccupancy,
        params={},
        instance_id="qc-grid-source",
    )
    analyzed_plot = _QcPlot()
    successful = SimpleNamespace(
        instance_id=entry.instance_id,
        check=analyzed_plot,
    )
    pipeline = ImagePipeline(
        qc=[entry],
        plots=[
            PlotBinding(
                id="custom-grid-output",
                ref=PipelineObjectRef(slot="qc", key=entry.instance_id),
            )
        ],
    )
    measurements = pd.DataFrame({"Size_Area": [1.0]})

    PlotCoordinator(pipeline, tmp_path).emit_qc(
        measurements,
        AnalysisRegistry(),
        successful_modules={entry.instance_id: successful},
    )

    assert isinstance(analyzed_plot._seen, QcPlotSubject)
    assert analyzed_plot._seen.qc_instance_id == entry.instance_id
    assert analyzed_plot._seen.analyzed_check is analyzed_plot
    assert (
        plots_dir(tmp_path) / "custom-grid-output" / "default.png"
    ).exists()
