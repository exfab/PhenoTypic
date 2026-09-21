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


def test_preflight_names_plots_that_declare_no_backend(monkeypatch) -> None:
    """A plot that overrides ``inspect`` with no ``@figure`` is still named.

    PlotDetectModes is a PlotImage -- exactly the single-page image path whose
    only record of a missing PNG is this announcement -- and it declares no
    figure backend. Iterating ``iter_figures`` alone would leave it out.
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

    (line,) = preflight_plot_backends(pipeline)

    assert "PlotDiagnostics" in line
    assert "PlotDetectModes" in line


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
    from phenotypic.plotting import PlotDetectModes, PlotDiagnostics
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    path = _write_pipeline_json(
        tmp_path,
        ImagePipeline(
            ops={"d": OtsuDetector()},
            plots=[PlotDiagnostics(), PlotDetectModes()],
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
    assert "PlotDiagnostics" in message
    assert "PlotDetectModes" in message
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
    assert "PlotBackendUnavailable" in message
    assert "PlotDiagnostics" in message
