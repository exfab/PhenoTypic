"""build_image_figures: in memory, finest-grained failures (spec §1, §3 step 1)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pydantic import BaseModel

from phenotypic import ImagePipeline
from phenotypic.abc_.plotting import PlotImage, PlotOutput, PlotPage, figure
from phenotypic.plotting._pipeline._store_figures import (
    build_image_figures,
    normalize_figure_error,
)


class Bars(BaseModel, PlotImage):
    @figure(title="bars", backend="plotly", primary=True)
    def draw(self, image):
        import plotly.graph_objects as go

        return go.Figure(go.Bar(x=["a"], y=[1]))


class BarsWithPng(BaseModel, PlotImage):
    @figure(title="bars", backend="plotly", primary=True, store=("plotly-json", "png"))
    def draw(self, image):
        import plotly.graph_objects as go

        return go.Figure(go.Bar(x=["a"], y=[1]))


class MplLine(BaseModel, PlotImage):
    @figure(title="line", backend="mpl", primary=True)
    def draw(self, image):
        from matplotlib.figure import Figure

        fig = Figure()
        fig.subplots().plot([0, 1])
        return fig


class HandBuiltPages(BaseModel, PlotImage):
    """Overrides inspect(): no spec, so each page takes its backend default."""

    def inspect(self, subject=None, *, for_save=False, **overrides):
        import plotly.graph_objects as go
        from matplotlib.figure import Figure

        return PlotOutput(pages=(
            PlotPage(key="A b", figure=go.Figure(), label="First"),
            PlotPage(key="a-b", figure=Figure()),
            PlotPage(key="odd", figure=object()),
            PlotPage(key="np", figure=go.Figure(), metadata={"n": np.int64(3)}),
        ))


class Explodes(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise RuntimeError(f"bad object at {hex(id(self))}")


class ReturnsNone(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        return None


def _build(*plots):
    return build_image_figures(ImagePipeline(plots=list(plots)), object())


def test_no_image_binding_builds_nothing():
    assert build_image_figures(ImagePipeline(), object()) is None


def test_default_plotly_stores_plotly_json_only():
    stored = _build(Bars())
    [binding] = stored.bindings
    assert (binding.binding_id, binding.directory) == ("Bars", "Bars")
    [page] = binding.pages
    assert [(f.format, f.filename) for f in page.files] == [
        ("plotly-json", "default.plotly.json")
    ]
    assert json.loads(page.files[0].data)["data"][0]["type"] == "bar"
    assert stored.failed == ()


def test_mpl_default_stores_png():
    [page] = _build(MplLine()).bindings[0].pages
    assert [f.format for f in page.files] == ["png"]
    assert page.backend == "mpl"
    assert page.files[0].data.startswith(b"\x89PNG")


def test_a_declared_png_without_chrome_fails_that_format_only(monkeypatch):
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    stored = _build(BarsWithPng())
    [page] = stored.bindings[0].pages
    assert [f.format for f in page.files] == ["plotly-json"]
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("BarsWithPng", "default", "png")
    assert failure.error.startswith("PlotBackendUnavailable: ")


def test_a_declared_png_with_chrome_stores_what_kaleido_returns(monkeypatch):
    import plotly.io as pio

    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)
    monkeypatch.setattr(pio, "to_image", lambda fig, format: b"\x89PNG kaleido")
    [page] = _build(BarsWithPng()).bindings[0].pages
    assert [(f.format, f.data) for f in page.files][1] == ("png", b"\x89PNG kaleido")


def test_hand_built_pages_use_backend_defaults_and_collision_safe_names():
    stored = _build(HandBuiltPages())
    [binding] = stored.bindings
    names = {page.key: [f.filename for f in page.files] for page in binding.pages}
    assert names["A b"] == ["A-b.plotly.json"]
    [mpl_name] = names["a-b"]
    assert mpl_name.endswith(".png") and mpl_name != "A-b.png"
    assert "odd" not in names and "np" not in names
    by_page = {f.page: f for f in stored.failed}
    assert by_page["odd"].format is None
    assert by_page["odd"].error.startswith("TypeError: unsupported figure type")
    assert by_page["np"].format is None
    assert by_page["np"].error.startswith("TypeError: ")


def test_inspect_raising_omits_the_binding_and_normalises_the_address():
    stored = _build(Explodes(), Bars())
    assert [b.binding_id for b in stored.bindings] == ["Bars"]
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("Explodes", None, None)
    assert failure.error == "RuntimeError: bad object at 0x…"


def test_inspect_returning_none_is_a_failure_not_an_absence():
    stored = _build(ReturnsNone())
    assert stored.bindings == ()
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("ReturnsNone", None, None)


def test_a_binding_whose_every_page_failed_is_absent(monkeypatch):
    from phenotypic.plotting._pipeline import _backends

    class PngOnly(BaseModel, PlotImage):
        @figure(title="p", backend="plotly", primary=True, store=("png",))
        def draw(self, image):
            import plotly.graph_objects as go

            return go.Figure()

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    stored = _build(PngOnly())
    assert stored.bindings == ()
    assert [(f.page, f.format) for f in stored.failed] == [("default", "png")]


def test_an_unexpected_error_outside_inspect_stays_inside_the_binding(monkeypatch):
    from phenotypic.plotting._pipeline import _store_figures

    def _boom(plot):
        raise LookupError("resolver broke")

    monkeypatch.setattr(_store_figures, "declared_figure_spec", _boom)
    stored = _build(Bars())
    assert stored.bindings == ()
    assert stored.failed[0].error == "LookupError: resolver broke"


def test_normalize_figure_error_replaces_every_address():
    assert normalize_figure_error(ValueError("0xdead and 0xBEEF1")) == "ValueError: 0x… and 0x…"


def test_figures_are_closed_after_serialization(monkeypatch):
    from phenotypic.plotting._pipeline import _store_figures

    closed = []
    monkeypatch.setattr(_store_figures.FigureAdapter, "close", staticmethod(closed.append))
    _build(MplLine())
    assert len(closed) == 1


@pytest.mark.parametrize("backend", ["plotly", "mpl"])
def test_the_default_serializer_is_stable_across_processes(backend):
    """Spec §4: fresh interpreters (fresh hash seeds, fresh addresses) agree."""
    make = {
        "plotly": "import plotly.graph_objects as go\nfig = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2]))\n",
        "mpl": "from matplotlib.figure import Figure\nfig = Figure()\nfig.subplots().plot([1, 2, 3], [3, 1, 2])\n",
    }[backend]
    fmt = "plotly-json" if backend == "plotly" else "png"
    code = make + textwrap.dedent(f"""
        import hashlib
        from phenotypic.plotting._pipeline._store_formats import serialize_store_format
        data = serialize_store_format({fmt!r}, fig, binding_id="b", page_key="default")
        print(hashlib.sha256(data).hexdigest())
    """)

    def digest(seed: str) -> str:
        import os

        env = {**os.environ, "PYTHONHASHSEED": seed}
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            check=True, env=env,
        ).stdout.strip()

    assert digest("1") == digest("2")
