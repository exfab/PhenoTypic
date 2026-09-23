"""Rendering capability and the hoisted plotly.min.js bundle."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.plotting._pipeline._backends import (
    chrome_available,
    ensure_plotlyjs_bundle,
    plotlyjs_src_for,
)


def test_the_bundle_is_written_once(tmp_path: Path) -> None:
    first = ensure_plotlyjs_bundle(tmp_path)
    assert first == tmp_path / "plotly.min.js"
    assert first.stat().st_size > 1_000_000
    stamp = first.stat().st_mtime_ns

    second = ensure_plotlyjs_bundle(tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns == stamp, "bundle was rewritten"


def test_the_bundle_is_rewritten_if_truncated(tmp_path: Path) -> None:
    bundle = ensure_plotlyjs_bundle(tmp_path)
    bundle.write_text("corrupted")
    assert ensure_plotlyjs_bundle(tmp_path).stat().st_size > 1_000_000


@pytest.mark.parametrize(
    "page_dir, expected",
    [
        ("plots/sym", "../plotly.min.js"),
        ("plots/sym/ds-1", "../../plotly.min.js"),
        ("plots/sym/ds-1/A01-abc123", "../../../plotly.min.js"),
    ],
)
def test_the_src_resolves_from_every_layout(
    tmp_path: Path, page_dir: str, expected: str
) -> None:
    """Aggregate, single-page image, and multi-page image layouts."""
    bundle = tmp_path / "plots" / "plotly.min.js"
    assert plotlyjs_src_for(tmp_path / page_dir, bundle) == expected


def test_the_probe_is_memoised(monkeypatch) -> None:
    calls = {"n": 0}

    def _counting_to_image(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("Kaleido requires Google Chrome to be installed.")

    import plotly.io as pio

    monkeypatch.setattr(pio, "to_image", _counting_to_image)
    assert chrome_available() is False
    assert chrome_available() is False
    assert calls["n"] == 1, "the probe ran more than once"


def test_the_probe_reports_success(monkeypatch) -> None:
    import plotly.io as pio

    monkeypatch.setattr(pio, "to_image", lambda *a, **k: b"\x89PNG")
    assert chrome_available() is True


# --------------------------------------------------------------------------
# Preflight (Task 7). For a Chrome-less single-page image directory the
# announcement below is the only record of why no PNG exists, so these tests
# pin that it names every affected binding -- including plots that declare no
# @figure backend at all, which is most of the production plot classes.
# --------------------------------------------------------------------------


def _hide_module(monkeypatch, module_name: str) -> None:
    """Make ``import <module_name>`` (and its submodules) raise ImportError."""
    import builtins

    real_import = builtins.__import__

    def _hiding_import(name, *args, **kwargs):
        if name == module_name or name.startswith(f"{module_name}."):
            raise ImportError(f"no {module_name} here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _hiding_import)


def test_preflight_warns_about_missing_chrome_without_raising(monkeypatch) -> None:
    import plotly.graph_objects as go
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    class _P(BaseModel, PlotMeas):
        @figure(title="T", backend="plotly", primary=True)
        def t(self, subject):
            return go.Figure()

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_P()])

    warnings_out = preflight_plot_backends(pipeline)

    assert len(warnings_out) == 1
    assert "_P" in warnings_out[0]
    assert "plotly_get_chrome" in warnings_out[0]


def test_preflight_is_silent_when_no_plots_are_configured() -> None:
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    assert preflight_plot_backends(
        ImagePipeline(ops={"d": OtsuDetector()})
    ) == []


def test_preflight_is_silent_when_chrome_is_available(monkeypatch) -> None:
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting import PlotDetectModes, PlotDiagnostics
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)
    pipeline = ImagePipeline(
        ops={"d": OtsuDetector()},
        plots=[PlotDiagnostics(), PlotDetectModes()],
    )

    assert preflight_plot_backends(pipeline) == []


def test_preflight_does_not_name_image_plots_that_store_no_png(monkeypatch) -> None:
    """Image plots are judged by their declared ``store``, not by Chrome.

    Both are PlotImage: PlotDiagnostics has a Plotly primary storing the
    default ``plotly-json``, and PlotDetectModes overrides ``inspect`` with no
    ``@figure``, so each page stores its backend default. Neither renders a
    PNG, so a missing Chrome costs them nothing (spec 2026-09-22 §2). The
    undeclared non-image case is
    ``test_preflight_treats_an_undecorated_inspect_override_as_undeclared``.
    """
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting import PlotDetectModes, PlotDiagnostics
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    assert PlotDetectModes().iter_figures() == [], (
        "premise: PlotDetectModes declares no @figure backend"
    )
    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(
        ops={"d": OtsuDetector()},
        plots=[PlotDiagnostics(), PlotDetectModes()],
    )

    assert preflight_plot_backends(pipeline) == []


def test_preflight_names_qc_recipe_bindings(monkeypatch) -> None:
    """A QC-recipe binding holds a config entry, not a plot; it is still named."""
    from phenotypic import ImagePipeline
    from phenotypic.analysis import GridOccupancy
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import (
        PipelineObjectRef,
        PlotBinding,
        _backends,
    )
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry

    entry = QcRecipeEntry(cls=GridOccupancy, params={}, instance_id="qc-grid")
    pipeline = ImagePipeline(
        ops={"d": OtsuDetector()},
        qc=[entry],
        plots=[
            PlotBinding(
                id="grid-output",
                ref=PipelineObjectRef(slot="qc", key=entry.instance_id),
            )
        ],
    )
    assert not hasattr(pipeline.get_plots()[0].plot, "iter_figures"), (
        "premise: a QC-recipe binding's plot is not a PhtPlot instance"
    )
    monkeypatch.setattr(_backends, "chrome_available", lambda: False)

    (line,) = preflight_plot_backends(pipeline)

    assert "grid-output" in line


def test_preflight_does_not_probe_chrome_without_plotly_candidates(
    monkeypatch,
) -> None:
    """A pipeline that cannot produce Plotly never launches a browser.

    The probe's termination is unproven for a present-but-hung Chrome and this
    runs before SLURM submission, so it must not be reached needlessly.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure as MplFigure
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    class _M(BaseModel, PlotMeas):
        @figure(title="T", backend="mpl", primary=True)
        def t(self, subject):
            return MplFigure()

    def _must_not_probe() -> bool:
        raise AssertionError("chrome_available() was called")

    monkeypatch.setattr(_backends, "chrome_available", _must_not_probe)
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_M()])

    assert preflight_plot_backends(pipeline) == []


def test_preflight_raises_when_a_declared_library_is_missing(monkeypatch) -> None:
    """Missing Chrome is a warning; a missing BACKEND is not."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure as MplFigure
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline._backends import (
        PlotBackendUnavailable,
        preflight_plot_backends,
    )

    class _M(BaseModel, PlotMeas):
        @figure(title="T", backend="mpl", primary=True)
        def t(self, subject):
            return MplFigure()

    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_M()])
    _hide_module(monkeypatch, "matplotlib")

    with pytest.raises(PlotBackendUnavailable, match="_M"):
        preflight_plot_backends(pipeline)


def test_preflight_raises_when_plotly_is_missing(monkeypatch) -> None:
    """The same rule for the other declared backend, checked before the probe."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting import PlotDiagnostics
    from phenotypic.plotting._pipeline._backends import (
        PlotBackendUnavailable,
        preflight_plot_backends,
    )

    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[PlotDiagnostics()])
    _hide_module(monkeypatch, "plotly")

    with pytest.raises(PlotBackendUnavailable, match="PlotDiagnostics"):
        preflight_plot_backends(pipeline)


# --- the CLI seam: validate_pipeline ---------------------------------------


def _write_pipeline_json(tmp_path: Path, pipeline) -> Path:
    path = tmp_path / "pipeline.json"
    path.write_text(pipeline.to_json())
    return path


@pytest.fixture
def _fresh_announcements(monkeypatch):
    """Isolate the once-per-process announcement memo from other tests."""
    from phenotypic._cli import _cli_validation

    monkeypatch.setattr(_cli_validation, "_ANNOUNCED_PLOT_WARNINGS", set())


def test_validation_announces_missing_chrome_once_per_process(
    tmp_path: Path, monkeypatch, caplog, _fresh_announcements
) -> None:
    """The warning reaches a WARNING record naming the bindings, exactly once.

    ``--dry-run`` validates the same pipeline twice in one process (the main
    path, then ``execute_dry_run`` -> ``full_validation``); the second call
    must not repeat the announcement.
    """
    import logging

    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_validation import validate_pipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting import PlotMeasTimeSeries
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    path = _write_pipeline_json(
        tmp_path,
        ImagePipeline(
            ops={"d": OtsuDetector()},
            # A PlotMeas: image plots storing no PNG are never announced.
            plots=[PlotMeasTimeSeries(environment_by=["env"], replicate_by=["rep"])],
        ),
    )

    with caplog.at_level(logging.WARNING, logger="phenotypic._cli._cli_validation"):
        assert validate_pipeline(path) == (True, None)
        assert validate_pipeline(path) == (True, None)

    records = [
        r for r in caplog.records
        if r.name == "phenotypic._cli._cli_validation"
        and r.levelno == logging.WARNING
    ]
    assert len(records) == 1, [r.getMessage() for r in records]
    message = records[0].getMessage()
    assert "PlotMeasTimeSeries" in message
    assert "plotly_get_chrome" in message


def test_validation_fails_cleanly_when_a_backend_is_unavailable(
    tmp_path: Path, monkeypatch, _fresh_announcements
) -> None:
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_validation import validate_pipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting import PlotDiagnostics
    from phenotypic.plotting._pipeline import _backends

    path = _write_pipeline_json(
        tmp_path,
        ImagePipeline(ops={"d": OtsuDetector()}, plots=[PlotDiagnostics()]),
    )

    def _unavailable(pipeline):
        raise _backends.PlotBackendUnavailable("plotly missing: PlotDiagnostics")

    monkeypatch.setattr(_backends, "preflight_plot_backends", _unavailable)

    valid, message = validate_pipeline(path)

    assert valid is False
    # The pipeline LOADED; saying "Failed to load pipeline" would send the
    # user looking at their JSON instead of their environment.
    assert message == "Plot backend unavailable: plotly missing: PlotDiagnostics"


# --- classification: by what inspect() renders, not by every declaration ---


def _must_not_probe() -> bool:
    raise AssertionError("chrome_available() was called")


def _plotly_sentence_and_rest(line: str) -> tuple[str, str]:
    """Split the Chrome warning into its Plotly sentence and everything else."""
    marker = "Plotly plots will publish HTML only, without PNG:"
    head, sep, tail = line.partition(marker)
    assert sep, f"no Plotly sentence in: {line}"
    plotly_sentence, _, rest = tail.partition(".")
    return plotly_sentence, head + rest


def test_preflight_classifies_by_the_primary_figure_not_every_figure(
    monkeypatch,
) -> None:
    """Probe D: primary mpl, secondary plotly. inspect() returns matplotlib.

    Unioning every declaration named this plot in the Chrome warning, which
    claimed it would publish without PNG -- false, it publishes a PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import plotly.graph_objects as go
    from matplotlib.figure import Figure as MplFigure
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    class Mixed(BaseModel, PlotMeas):
        @figure(title="Primary", backend="mpl", primary=True)
        def primary(self, subject):
            return MplFigure()

        @figure(title="Secondary", backend="plotly")
        def secondary(self, subject):
            return go.Figure()

    monkeypatch.setattr(_backends, "chrome_available", _must_not_probe)
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[Mixed()])

    assert preflight_plot_backends(pipeline) == []


def test_preflight_names_a_plotly_primary_in_the_plotly_sentence(
    monkeypatch,
) -> None:
    """The reverse of Probe D: a secondary mpl figure changes nothing.

    With matplotlib hidden it must not raise either: inspect() never renders
    the mpl figure, so its library is not required for publication.
    """
    import matplotlib

    matplotlib.use("Agg")
    import plotly.graph_objects as go
    from matplotlib.figure import Figure as MplFigure
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    class PlotlyFirst(BaseModel, PlotMeas):
        @figure(title="Primary", backend="plotly", primary=True)
        def primary(self, subject):
            return go.Figure()

        @figure(title="Secondary", backend="mpl")
        def secondary(self, subject):
            return MplFigure()

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[PlotlyFirst()])
    _hide_module(monkeypatch, "matplotlib")

    (line,) = preflight_plot_backends(pipeline)

    plotly_sentence, rest = _plotly_sentence_and_rest(line)
    assert "PlotlyFirst" in plotly_sentence
    assert "PlotlyFirst" not in rest


def test_preflight_treats_an_undecorated_inspect_override_as_undeclared(
    monkeypatch,
) -> None:
    """An overridden inspect() renders none of the class's @figure methods."""
    import plotly.graph_objects as go
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    class Overridden(BaseModel, PlotMeas):
        @figure(title="Declared", backend="plotly", primary=True)
        def declared(self, subject):
            return go.Figure()

        def inspect(self, subject=None, *, for_save=False, **overrides):
            return go.Figure()

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[Overridden()])

    (line,) = preflight_plot_backends(pipeline)

    assert "Plotly plots will publish" not in line
    assert "declare no figure backend" in line
    assert "Overridden" in line


def test_preflight_honours_a_decorated_inspect_override(monkeypatch) -> None:
    """``@figure`` ON ``inspect`` declares what inspect renders.

    This is the MeasureSymZones / MeasureOrientationZones shape: the override
    IS the declaration, and the wrapper enforces the backend at render time.
    Demoting it to "undeclared" would drop the importability check for it.
    """
    import plotly.graph_objects as go
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import (
        PlotBackendUnavailable,
        preflight_plot_backends,
    )

    class DecoratedInspect(BaseModel, PlotMeas):
        @figure(title="Inspect", backend="plotly", primary=True)
        def inspect(self, subject=None, *, for_save=False):
            return go.Figure()

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[DecoratedInspect()])

    (line,) = preflight_plot_backends(pipeline)
    plotly_sentence, _rest = _plotly_sentence_and_rest(line)
    assert "DecoratedInspect" in plotly_sentence

    _hide_module(monkeypatch, "plotly")
    with pytest.raises(PlotBackendUnavailable, match="DecoratedInspect"):
        preflight_plot_backends(pipeline)


def _qc_recipe_pipeline(qc_cls):
    """A pipeline whose only plot is a QC-recipe reference to *qc_cls*."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import PipelineObjectRef, PlotBinding
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry

    entry = QcRecipeEntry(cls=qc_cls, params={}, instance_id="qc-synthetic")
    pipeline = ImagePipeline(
        ops={"d": OtsuDetector()},
        qc=[entry],
        plots=[
            PlotBinding(
                id="qc-output",
                ref=PipelineObjectRef(slot="qc", key=entry.instance_id),
            )
        ],
    )
    assert not hasattr(pipeline.get_plots()[0].plot, "iter_figures"), (
        "premise: a QC-recipe binding holds the entry, not a PhtPlot instance"
    )
    return pipeline


def test_preflight_reads_an_inherited_figure_on_a_qc_recipe_class(
    monkeypatch,
) -> None:
    """The class-level branch, against a real declaration (M6).

    No QC class in ``src/`` declares a figure, so the only existing QC test
    (GridOccupancy, which overrides inspect) could not tell a working class
    walk from an empty one. This class INHERITS an mpl primary.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure as MplFigure

    from phenotypic.abc_.plotting import PlotQc, figure
    from phenotypic.analysis.abc_ import QualityCheck
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import (
        PlotBackendUnavailable,
        preflight_plot_backends,
    )

    class MplQcBase(QualityCheck, PlotQc):
        @figure(title="Flags", backend="mpl", primary=True)
        def flags(self, subject):
            return MplFigure()

    class MplQcChild(MplQcBase):
        pass

    pipeline = _qc_recipe_pipeline(MplQcChild)
    monkeypatch.setattr(_backends, "chrome_available", _must_not_probe)

    assert preflight_plot_backends(pipeline) == []

    _hide_module(monkeypatch, "matplotlib")
    with pytest.raises(PlotBackendUnavailable, match="qc-output"):
        preflight_plot_backends(pipeline)


def test_preflight_respects_an_undecorated_override_on_a_qc_recipe_class(
    monkeypatch,
) -> None:
    """An undecorated override removes the inherited figure (M7).

    The parent declares plotly; the child overrides that method without
    ``@figure``, so it declares nothing and inspect() has no figure to render.
    """
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotQc, figure
    from phenotypic.analysis.abc_ import QualityCheck
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    class PlotlyQcBase(QualityCheck, PlotQc):
        @figure(title="Flags", backend="plotly", primary=True)
        def flags(self, subject):
            return go.Figure()

    class ShadowingQc(PlotlyQcBase):
        def flags(self, subject):
            return go.Figure()

    pipeline = _qc_recipe_pipeline(ShadowingQc)
    monkeypatch.setattr(_backends, "chrome_available", lambda: False)

    (line,) = preflight_plot_backends(pipeline)

    assert "Plotly plots will publish" not in line
    assert "declare no figure backend" in line
    assert "qc-output" in line


def test_image_plots_need_chrome_only_when_they_declare_png(monkeypatch):
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotImage, figure
    from phenotypic.plotting._pipeline import _backends

    class Img(BaseModel, PlotImage):
        @figure(title="t", backend="plotly", primary=True)
        def draw(self, image):
            raise AssertionError

    class ImgPng(BaseModel, PlotImage):
        @figure(title="t", backend="plotly", primary=True, store=("png",))
        def draw(self, image):
            raise AssertionError

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    assert _backends.preflight_plot_backends(ImagePipeline(plots=[Img()])) == []
    [line] = _backends.preflight_plot_backends(ImagePipeline(plots=[ImgPng()]))
    assert "ImgPng" in line
