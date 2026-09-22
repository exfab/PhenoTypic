"""Lifecycle dispatch and output layout for configured plots."""

from __future__ import annotations

import re

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


def test_a_multi_page_plotly_image_plot_writes_exactly_one_bundle(tmp_path) -> None:
    """The gigabyte trap: one plotly.min.js per RUN, never per directory.

    Every figure elsewhere in this file is matplotlib, so no bundle is ever
    written and this default was untested. Measured before the fix: three
    images produced three 4.8 MB copies, one per image directory --
    extrapolating to 7.45 GB on a 1,536-image plate, while emitting a
    correct-looking relative src in every page.

    Asserts the count across the WHOLE tree rather than the absence of a copy
    in one place: a per-directory bundle is still "a bundle that exists", so
    only a global count distinguishes hoisted from duplicated.
    """
    import plotly.graph_objects as go
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotImage, PlotOutput, PlotPage, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import PlotCoordinator

    class _MultiPagePlotly(BaseModel, PlotImage):
        @figure(title="unused", backend="plotly", primary=True)
        def _never(self, image):  # pragma: no cover - inspect() overrides
            raise AssertionError

        def inspect(self, subject=None, *, for_save=False, **overrides):
            return PlotOutput(pages=(
                PlotPage(key="first", figure=go.Figure(), label="First"),
                PlotPage(key="second", figure=go.Figure(), label="Second"),
                # Fails in _render_page (unsupported figure), so a record is
                # written -- which is what makes the hoisting check below
                # able to fail.
                PlotPage(key="broken", figure=object(), label="Broken"),
            ))

    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_MultiPagePlotly()])
    coordinator = PlotCoordinator(pipeline, tmp_path)
    for stem in ("img-A", "img-B", "img-C"):
        coordinator.emit_image(object(), dataset="ds", image_stem=stem)

    bundles = sorted(tmp_path.rglob("plotly.min.js"))
    assert len(bundles) == 1, (
        f"expected one hoisted bundle, found {len(bundles)}: "
        f"{[str(b.relative_to(tmp_path)) for b in bundles]}"
    )
    assert bundles[0].parent == tmp_path / "deliverables" / "plots"

    pages = sorted(tmp_path.rglob("*.html"))
    assert len(pages) == 6, f"expected 6 pages, found {len(pages)}"

    # D2: the failure record is hoisted for the same reason as the bundle, and
    # was equally unguarded. Spec §3 puts it at deliverables/plots/, not one
    # per page directory -- a per-directory record is still "a record that
    # exists", so again only a whole-tree location check distinguishes the
    # two. One page per image fails, so there IS a record to locate; an
    # earlier form asserted absence in a run where nothing failed, which
    # passed whether or not the record was hoisted.
    records = sorted(tmp_path.rglob(".failures.jsonl"))
    assert records == [tmp_path / "deliverables" / "plots" / ".failures.jsonl"], (
        records
    )
    entries = _failure_entries(tmp_path)
    assert [entry["lifecycle"] for entry in entries] == ["page"] * 3


def _failure_entries(tmp_path) -> list[dict]:
    """Parse the one durable failure record under ``deliverables/plots``."""
    import json

    record = plots_dir(tmp_path) / ".failures.jsonl"
    return [
        json.loads(line)
        for line in record.read_text(encoding="utf-8").splitlines()
    ]


def test_a_raising_figure_leaves_the_run_green_and_is_recorded(tmp_path) -> None:
    from phenotypic.abc_.plotting import figure
    from phenotypic.detect import OtsuDetector

    class _Exploding(BaseModel, PlotMeas):
        @figure(title="T", backend="plotly", primary=True)
        def t(self, subject):
            raise RuntimeError("figure exploded")

    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_Exploding()])
    PlotCoordinator(pipeline, tmp_path).emit_measurements(pd.DataFrame())

    entries = _failure_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["binding_id"] == "_Exploding"
    assert entries[0]["plot_class"] == "_Exploding"
    assert entries[0]["lifecycle"] == "measurements"
    assert entries[0]["error"] == "RuntimeError: figure exploded"


class _RaisingImagePlot(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise ValueError("image plot exploded")


# No leading underscore: emit_analyses looks the producer up by class name,
# and the registry rejects an id starting with "_" -- which raises before
# inspect() and would record the id validation instead of this failure.
class RaisingAnalysisPlot(BaseModel, PlotAnalysis):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise ValueError("analysis plot exploded")


class _RaisingQcPlot(BaseModel, PlotQc):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise ValueError("qc plot exploded")


def _emit_image(coordinator) -> None:
    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")


def _emit_analyses(coordinator) -> None:
    coordinator.emit_analyses(pd.DataFrame(), AnalysisRegistry())


def _emit_qc(coordinator) -> None:
    coordinator.emit_qc(pd.DataFrame(), AnalysisRegistry())


def _emit_dependent_qc(coordinator) -> None:
    coordinator.emit_dependent_qc(
        pd.DataFrame(), AnalysisRegistry(), updated_input=MeasurementInput()
    )


def _emit_dependent_qc_unresolvable(coordinator) -> None:
    # The input names a table the registry cannot resolve, so the failure is
    # raised by emit_dependent_qc's OWN try, before _emit_aggregate is reached.
    coordinator.emit_dependent_qc(
        pd.DataFrame(),
        AnalysisRegistry(),
        updated_input=AnalysisInput(analysis_id="MissingModel"),
    )


@pytest.mark.parametrize(
    ("plot", "plot_input", "emit", "lifecycle", "image_fields", "error_pattern"),
    [
        pytest.param(
            _RaisingImagePlot(), None, _emit_image, "image",
            {"dataset": "ds", "image_stem": "plate-1"}, r"ValueError: image plot exploded",
            id="emit_image",
        ),
        pytest.param(
            RaisingAnalysisPlot(), None, _emit_analyses, "analysis",
            {}, r"ValueError: analysis plot exploded", id="emit_analyses",
        ),
        pytest.param(
            _RaisingQcPlot(), None, _emit_qc, "qc",
            {}, r"ValueError: qc plot exploded", id="emit_qc-via-aggregate",
        ),
        # "qc", NOT "qc dependency". The string is supplied at the
        # emit_dependent_qc call site and only FORWARDED by _emit_aggregate's
        # handler, so a fix made in that handler changes nothing.
        pytest.param(
            _RaisingQcPlot(), None, _emit_dependent_qc, "qc",
            {}, r"ValueError: qc plot exploded", id="emit_dependent_qc-via-aggregate",
        ),
        pytest.param(
            _RaisingQcPlot(), AnalysisInput(analysis_id="MissingModel"),
            _emit_dependent_qc_unresolvable, "qc",
            {}, r"AnalysisNotFoundError: .*MissingModel.*", id="emit_dependent_qc-own-handler",
        ),
    ],
)
def test_every_emit_point_records_one_failure_under_a_closed_lifecycle(
    tmp_path, plot, plot_input, emit, lifecycle, image_fields, error_pattern
) -> None:
    """Each coordinator handler writes exactly one record, never two.

    ``_emit_aggregate`` swallows without re-raising, so an aggregate failure
    must reach the record once -- from the innermost handler -- and not again
    from the emit method's own handler around it.
    """
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="failing", plot=plot, input=plot_input)]
    )

    emit(PlotCoordinator(pipeline, tmp_path))

    entries = _failure_entries(tmp_path)
    assert len(entries) == 1, entries
    entry = entries[0]
    assert entry["binding_id"] == "failing"
    assert entry["plot_class"] == type(plot).__name__
    assert entry["lifecycle"] == lifecycle
    assert re.fullmatch(error_pattern, entry["error"]), entry["error"]
    assert {
        key: entry[key] for key in ("dataset", "image_stem") if key in entry
    } == image_fields


def test_emit_qc_prelude_failure_names_the_right_binding(tmp_path) -> None:
    """F4: the prelude is the ONLY window where the misattribution lives.

    The prelude is everything in ``emit_qc``'s loop before ``binding`` is
    assigned by ``model_copy``. An earlier draft of this test patched
    ``MeasurementInput.__init__``, which runs one line AFTER that assignment,
    so it passed on the unfixed code and proved nothing. ``modules.get`` is
    inside the prelude, and is the way in.
    """
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import figure
    from phenotypic.detect import OtsuDetector

    class _FirstQc(BaseModel, PlotQc):
        @figure(title="First", backend="plotly", primary=True)
        def t(self, subject):
            return go.Figure()

    class _SecondQc(BaseModel, PlotQc):
        @figure(title="Second", backend="plotly", primary=True)
        def t(self, subject):
            return go.Figure()

    pipeline = ImagePipeline(
        ops={"d": OtsuDetector()}, plots=[_FirstQc(), _SecondQc()]
    )

    class _RaisingOnSecond(dict):
        """Raises inside the prelude for the SECOND binding only.

        Raising on the FIRST would make the misnaming assertion tautological:
        with no prior iteration there is no stale ``binding`` to be
        misattributed to. Raising on the second is what makes a missing
        ``binding = None`` reset detectable -- the record would then say
        ``_FirstQc`` for a failure in ``_SecondQc``.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def get(self, key, default=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("prelude exploded")
            return super().get(key, default)

    # Non-empty: `modules = successful_modules or {}` discards a falsy mapping,
    # which would skip the injection entirely.
    modules = _RaisingOnSecond({"unused": object()})

    PlotCoordinator(pipeline, tmp_path).emit_qc(
        pd.DataFrame(),
        AnalysisRegistry(tmp_path / "deliverables"),
        successful_modules=modules,
    )

    assert modules.calls == 2, "premise: the prelude ran once per binding"
    entries = _failure_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["binding_id"] == "_SecondQc", (
        "a stale binding from the previous iteration was misattributed"
    )
    # The class that WOULD have emitted is unknowable from inside the prelude:
    # for a QC reference, `configured.plot` is the recipe entry, not the check
    # substituted for it. An explicit marker keeps that row distinguishable
    # from one whose class was actually observed.
    assert entries[0]["plot_class"] == "<unresolved>"
    assert entries[0]["lifecycle"] == "qc"
    assert entries[0]["error"] == "RuntimeError: prelude exploded"

    # ...and the FIRST binding, which ran before the failure, still published.
    assert (plots_dir(tmp_path) / "_FirstQc").is_dir()


class _SingleFigurePlotlyImagePlot(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        import plotly.graph_objects as go

        return go.Figure()


def test_a_single_figure_plotly_image_plot_publishes_html(
    tmp_path, monkeypatch
) -> None:
    """B1: the flat single-page path is the one every bare figure takes.

    A bare figure normalizes to one ``"default"`` page, which bypasses
    ``publish_plot_output`` entirely. Before it was routed through
    ``_render_page`` it wrote PNG only, so without Chrome an image plot
    published nothing at all.
    """
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="image", plot=_SingleFigurePlotlyImagePlot())]
    )
    coordinator = PlotCoordinator(pipeline, tmp_path)
    for image_stem in ("plate-1", "plate-2"):
        coordinator.emit_image(object(), dataset="ds", image_stem=image_stem)

    directory = plots_dir(tmp_path) / "image" / "ds"
    pages = sorted(path.name for path in directory.glob("*.html"))
    assert len(pages) == 2, pages
    # `<stem>-<hash>`, NOT the page key: routing through publish_plot_output
    # would name every image `default.html` and each would overwrite the last.
    assert pages[0].startswith("plate-1-") and pages[1].startswith("plate-2-")
    assert not (directory / "manifest.json").exists()
    assert list(directory.glob("*.png")) == []

    bundle = plots_dir(tmp_path) / "plotly.min.js"
    assert sorted(tmp_path.rglob("plotly.min.js")) == [bundle]
    html = (directory / pages[0]).read_text(encoding="utf-8")
    sources = re.findall(r'src="([^"]*plotly\.min\.js)"', html)
    assert len(sources) == 1, sources
    assert (directory / sources[0]).resolve() == bundle.resolve(), sources

    # Chrome being absent is a missing CAPABILITY, not a failure: the PNG was
    # never attempted, so the durable record stays empty (spec §2, "Scope of
    # the durable record"). The CLI preflight announcement covers this case.
    assert list(tmp_path.rglob(".failures.jsonl")) == []


class _UnsupportedFigureImagePlot(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        return object()


def test_a_flat_image_render_failure_records_the_real_exception_class(
    tmp_path,
) -> None:
    """The render error is recorded as raised -- its class is the diagnostic.

    ``_render_page`` returns exceptions, not strings. Wrapping one as
    ``RuntimeError(exc)`` would spell this ``"RuntimeError: unsupported
    figure type ..."``; for an error that already IS a RuntimeError the wrap
    comes out right by coincidence, which is why this uses a TypeError.
    """
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="image", plot=_UnsupportedFigureImagePlot())]
    )

    PlotCoordinator(pipeline, tmp_path).emit_image(
        object(), dataset="ds", image_stem="plate-1"
    )

    entries = _failure_entries(tmp_path)
    # Two records by design: the first says which renderer failed, the second
    # (from emit_image's handler) that the image got no file at all.
    assert len(entries) == 2, entries
    render, image = entries
    assert render["error"].startswith(
        "TypeError: unsupported figure type builtins.object"
    ), render["error"]
    assert image["error"].startswith("RuntimeError: plot 'image' produced no file")
    # Formatted once, by _format_error -- not a doubled "TypeError: TypeError:".
    assert image["error"].count("TypeError") == 1, image["error"]
    for entry in entries:
        assert entry["binding_id"] == "image"
        assert entry["plot_class"] == "_UnsupportedFigureImagePlot"
        assert entry["lifecycle"] == "image"
        assert (entry["dataset"], entry["image_stem"]) == ("ds", "plate-1")


# --- F1: a refused guard voids the refresh; it is not a plot failure --------


def _emit_image_flat(coordinator) -> None:
    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")


def _emit_measurements(coordinator) -> None:
    coordinator.emit_measurements(pd.DataFrame())


def _emit_analyses_refreshable(coordinator) -> None:
    coordinator.emit_analyses(pd.DataFrame(), AnalysisRegistry())


def _emit_qc_default(coordinator) -> None:
    coordinator.emit_qc(pd.DataFrame(), AnalysisRegistry())


_EVERY_EMIT_POINT = [
    pytest.param(_ImagePlot, _emit_image_flat, id="emit_image-flat"),
    pytest.param(_MultiImagePlot, _emit_image_flat, id="emit_image-multi-page"),
    pytest.param(_MeasPlot, _emit_measurements, id="emit_measurements"),
    # No leading underscore: see RaisingAnalysisPlot.
    pytest.param(
        RefreshableAnalysisPlot, _emit_analyses_refreshable, id="emit_analyses"
    ),
    pytest.param(_QcPlot, _emit_qc_default, id="emit_qc"),
    pytest.param(_QcPlot, _emit_dependent_qc, id="emit_dependent_qc"),
]


@pytest.mark.parametrize(("plot_class", "emit"), _EVERY_EMIT_POINT)
def test_a_refused_publication_guard_propagates_and_writes_nothing(
    tmp_path, plot_class, emit
) -> None:
    """Probe A, through the coordinator.

    The GUI's guard refuses when the output tree changed under it (a CLI run
    in progress, say). Recording that as a plot failure wrote
    ``.failures.jsonl`` and ``.failures.lock`` into the very tree the guard
    had just said not to touch.
    """
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    pipeline = ImagePipeline(plots=[PlotBinding(id="guarded", plot=plot_class())])
    coordinator = PlotCoordinator(
        pipeline, tmp_path, publication_guard=lambda: False
    )

    with pytest.raises(PlotPublicationBlocked):
        emit(coordinator)

    assert sorted(tmp_path.rglob("*")) == []


class _FencingCommitGuard:
    """A ``commit_guard`` that admits *allow* commits, then fences.

    Shaped like the staged worker's guard: entering it raises the lifecycle
    fence's own exception, which is NOT a ``PlotPublicationBlocked``.
    """

    def __init__(self, allow: int = 0) -> None:
        self.allow = allow
        self.entered = 0

    def __call__(self):
        from contextlib import contextmanager

        from phenotypic._cli._cli_slurm_lifecycle import (
            SlurmGenerationInactiveError,
        )

        @contextmanager
        def _guard():
            self.entered += 1
            if self.entered > self.allow:
                raise SlurmGenerationInactiveError("epoch superseded")
            yield

        return _guard()


@pytest.mark.parametrize(
    ("plot_class", "emit", "allow"),
    [
        pytest.param(_ImagePlot, _emit_image_flat, 0, id="flat-page"),
        pytest.param(_MultiImagePlot, _emit_image_flat, 0, id="multi-page-page"),
        # Two page commits succeed; the MANIFEST commit is the one fenced.
        pytest.param(_MultiImagePlot, _emit_image_flat, 2, id="multi-page-manifest"),
        pytest.param(_MeasPlot, _emit_measurements, 0, id="aggregate-page"),
    ],
)
def test_a_fenced_commit_propagates_with_its_cause_and_records_nothing(
    tmp_path, plot_class, emit, allow
) -> None:
    """Probe B: a superseded worker must stop, not log a plot failure.

    The fence is found by walking ``__cause__``, which is how Stage 3 and the
    full path recognise it behind whatever wraps it.
    """
    from phenotypic._cli._cli_slurm_lifecycle import (
        slurm_generation_inactive_cause,
    )
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    guard = _FencingCommitGuard(allow=allow)
    pipeline = ImagePipeline(plots=[PlotBinding(id="fenced", plot=plot_class())])
    coordinator = PlotCoordinator(pipeline, tmp_path, commit_guard=guard)

    with pytest.raises(PlotPublicationBlocked) as raised:
        emit(coordinator)

    assert guard.entered == allow + 1
    assert slurm_generation_inactive_cause(raised.value) is not None
    assert list(tmp_path.rglob(".failures.jsonl")) == []
    assert list(tmp_path.rglob("*.tmp")) == []


def test_the_flat_path_commits_through_the_commit_guard(tmp_path) -> None:
    """M1: one guarded commit per file the flat path writes."""
    guard = _FencingCommitGuard(allow=10)
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=_ImagePlot())])

    PlotCoordinator(pipeline, tmp_path, commit_guard=guard).emit_image(
        object(), dataset="ds", image_stem="plate-1"
    )

    assert len(list((plots_dir(tmp_path) / "image" / "ds").glob("*.png"))) == 1
    assert guard.entered == 1


def test_the_flat_path_rechecks_the_publication_guard_before_commit(
    tmp_path,
) -> None:
    """M1: a guard that flips after the entry check still stops the write.

    The flat path checks the guard once on entry. Only the check inside the
    commit, which ``_render_page`` makes with the guard it is handed, can see
    a snapshot that changed while the figure was rendering.
    """
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    answers = iter([True])
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=_ImagePlot())])
    coordinator = PlotCoordinator(
        pipeline, tmp_path, publication_guard=lambda: next(answers, False)
    )

    with pytest.raises(PlotPublicationBlocked):
        coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")

    assert list(tmp_path.rglob("*.png")) == []
    assert list(tmp_path.rglob(".failures.jsonl")) == []


def test_a_strict_flat_failure_keeps_the_renderer_as_its_cause(tmp_path) -> None:
    """``strict=True`` must still let a caller find the renderer's exception."""
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="image", plot=_UnsupportedFigureImagePlot())]
    )

    with pytest.raises(RuntimeError, match="produced no file") as raised:
        PlotCoordinator(pipeline, tmp_path).emit_image(
            object(), dataset="ds", image_stem="plate-1", strict=True
        )

    assert isinstance(raised.value.__cause__, TypeError)


# --- M2 / M3: a partial render is published, recorded once, and not raised --


def test_a_partial_flat_render_publishes_what_it_can_and_records_once(
    tmp_path, monkeypatch
) -> None:
    """HTML succeeds, PNG fails: the page is published and one record says why.

    Only the everything-failed case was tested before, so recording only on
    total failure, or raising on any error, both survived.
    """
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._adapter import FigureAdapter

    def _raster_fails(figure, path):
        raise OSError("raster exploded")

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)
    monkeypatch.setattr(FigureAdapter, "save_png", staticmethod(_raster_fails))
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="image", plot=_SingleFigurePlotlyImagePlot())]
    )

    PlotCoordinator(pipeline, tmp_path).emit_image(
        object(), dataset="ds", image_stem="plate-1"
    )

    directory = plots_dir(tmp_path) / "image" / "ds"
    assert len(list(directory.glob("*.html"))) == 1
    assert list(directory.glob("*.png")) == []
    entries = _failure_entries(tmp_path)
    assert [entry["error"] for entry in entries] == ["OSError: raster exploded"]


# --- F2: a rerun must not leave the previous generation's sibling ----------


class _SwitchableImagePlot(BaseModel, PlotImage):
    """Returns a Plotly or a matplotlib figure, as the test chooses."""

    backend: str = "plotly"

    def inspect(self, subject=None, *, for_save=False, **overrides):
        if self.backend == "plotly":
            import plotly.graph_objects as go

            return go.Figure()
        return plt.figure()


def _fake_png(figure, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG fake")


def test_a_rerun_without_chrome_removes_the_previous_png(
    tmp_path, monkeypatch
) -> None:
    """Probe C: first run with Chrome, second without.

    With no manifest on the flat path, a surviving PNG beside a fresh HTML is
    indistinguishable from a matching pair -- it would be read as this run's.
    """
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._adapter import FigureAdapter

    plot = _SwitchableImagePlot(backend="plotly")
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=plot)])
    coordinator = PlotCoordinator(pipeline, tmp_path)
    directory = plots_dir(tmp_path) / "image" / "ds"

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)
    monkeypatch.setattr(FigureAdapter, "save_png", staticmethod(_fake_png))
    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")
    assert len(list(directory.glob("*.png"))) == 1, "premise: run 1 wrote a PNG"

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")

    assert list(directory.glob("*.png")) == []
    assert len(list(directory.glob("*.html"))) == 1


def test_a_failed_flat_rerun_keeps_both_previous_renderings(
    tmp_path, monkeypatch
) -> None:
    """A rerun that publishes nothing must not half-clean the old pair.

    Removal is for a page this run DID publish, as on the manifest path. If the
    HTML save fails on a Chrome-less rerun, deleting the old PNG would leave the
    old HTML alone on disk, reading as this run's output.
    """
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._adapter import FigureAdapter

    plot = _SwitchableImagePlot(backend="plotly")
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=plot)])
    coordinator = PlotCoordinator(pipeline, tmp_path)
    directory = plots_dir(tmp_path) / "image" / "ds"

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)
    monkeypatch.setattr(FigureAdapter, "save_png", staticmethod(_fake_png))
    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")
    before = sorted(path.name for path in directory.iterdir())
    assert [n.rsplit(".", 1)[1] for n in before] == ["html", "png"], before

    def _failing_html(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    monkeypatch.setattr(FigureAdapter, "save_html", staticmethod(_failing_html))
    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")

    assert sorted(path.name for path in directory.iterdir()) == before


def test_a_rerun_as_matplotlib_removes_the_previous_html(
    tmp_path, monkeypatch
) -> None:
    """The same, when the plot itself changed backend between runs."""
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    plot = _SwitchableImagePlot(backend="plotly")
    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=plot)])
    coordinator = PlotCoordinator(pipeline, tmp_path)
    directory = plots_dir(tmp_path) / "image" / "ds"

    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")
    assert len(list(directory.glob("*.html"))) == 1, "premise: run 1 wrote HTML"

    plot.backend = "mpl"
    coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")

    assert list(directory.glob("*.html")) == []
    assert len(list(directory.glob("*.png"))) == 1


# --- M4 / M5 -----------------------------------------------------------------


class _PlotlyMeasPlot(BaseModel, PlotMeas):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        import plotly.graph_objects as go

        return go.Figure()


def test_a_plotly_aggregate_uses_the_one_hoisted_bundle(
    tmp_path, monkeypatch
) -> None:
    """M4: the aggregate path, not only the image paths, hoists the bundle."""
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="measurements", plot=_PlotlyMeasPlot())]
    )

    PlotCoordinator(pipeline, tmp_path).emit_measurements(pd.DataFrame())

    assert sorted(tmp_path.rglob("plotly.min.js")) == [
        plots_dir(tmp_path) / "plotly.min.js"
    ]
    assert (plots_dir(tmp_path) / "measurements" / "default.html").exists()


def test_the_flat_path_closes_its_matplotlib_figure(tmp_path) -> None:
    """M5: a 1,536-image plate must not accumulate 1,536 open figures.

    On the success path ``FigureAdapter.save_png`` already closes a
    matplotlib figure in its own ``finally``, so asserting there cannot fail
    whatever the coordinator does. The coordinator's close is load-bearing
    only when ``save_png`` is never reached -- here, a publication guard that
    refuses at the entry check, after ``inspect()`` has built the figure.
    """
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    pipeline = ImagePipeline(plots=[PlotBinding(id="image", plot=_ImagePlot())])
    coordinator = PlotCoordinator(
        pipeline, tmp_path, publication_guard=lambda: False
    )
    before = set(plt.get_fignums())

    with pytest.raises(PlotPublicationBlocked):
        coordinator.emit_image(object(), dataset="ds", image_stem="plate-1")

    assert set(plt.get_fignums()) == before


# --- F2, manifest directories: the same stale sibling, behind a manifest ----


class _MultiPagePlotlyImagePlot(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        import plotly.graph_objects as go

        return PlotOutput(pages=(
            PlotPage(key="first", figure=go.Figure(), label="First"),
            PlotPage(key="second", figure=go.Figure(), label="Second"),
        ))


def _assert_manifest_matches_disk(directory) -> None:
    """The non-hidden page files are exactly what the manifest names."""
    import json

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    named = {name for page in manifest["pages"] for name in page["files"].values()}
    on_disk = {
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.name != "manifest.json"
    }
    assert on_disk == named, (on_disk, named)


def _emit_twice_chrome_then_none(monkeypatch, coordinator, emit) -> None:
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._adapter import FigureAdapter

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)
    monkeypatch.setattr(FigureAdapter, "save_png", staticmethod(_fake_png))
    emit(coordinator)
    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    emit(coordinator)


def test_an_aggregate_rerun_without_chrome_removes_the_previous_png(
    tmp_path, monkeypatch
) -> None:
    """The manifest stops naming the PNG; the PNG must stop existing with it.

    Otherwise ``default.png`` from the Chrome run sits beside the new
    ``default.html`` under a manifest that denies it exists.
    """
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="measurements", plot=_PlotlyMeasPlot())]
    )
    directory = plots_dir(tmp_path) / "measurements"

    _emit_twice_chrome_then_none(
        monkeypatch,
        PlotCoordinator(pipeline, tmp_path),
        lambda coordinator: coordinator.emit_measurements(pd.DataFrame()),
    )

    assert list(directory.glob("*.png")) == []
    assert (directory / "default.html").is_file()
    _assert_manifest_matches_disk(directory)


def test_a_multi_page_image_rerun_without_chrome_removes_the_previous_pngs(
    tmp_path, monkeypatch
) -> None:
    pipeline = ImagePipeline(
        plots=[PlotBinding(id="image", plot=_MultiPagePlotlyImagePlot())]
    )

    _emit_twice_chrome_then_none(
        monkeypatch,
        PlotCoordinator(pipeline, tmp_path),
        lambda coordinator: coordinator.emit_image(
            object(), dataset="ds", image_stem="plate-1"
        ),
    )

    (directory,) = [
        path for path in (plots_dir(tmp_path) / "image" / "ds").iterdir()
        if path.is_dir()
    ]
    assert list(directory.glob("*.png")) == []
    assert len(list(directory.glob("*.html"))) == 2
    _assert_manifest_matches_disk(directory)
