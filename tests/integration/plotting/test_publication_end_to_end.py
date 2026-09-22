"""Real image, real detector, every publication path -- composed.

Unit tests prove each piece in isolation. These prove they compose on a real
pipeline: a synthetic yeast plate, a real ``OtsuDetector``, real measurements.
Nothing in the code under test is patched. Chrome is never assumed either way:
every rendering assertion is written against ``chrome_available()``, the
machine's real, freshly probed capability, so the same test is correct on a
node with Chrome and on one without.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from phenotypic import ImagePipeline
from phenotypic.abc_.plotting import PlotImage, PlotMeas, figure
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.plotting._pipeline import PlotCoordinator, chrome_available

_OPS = {"detect": OtsuDetector()}
_MEAS = {"size": MeasureSize()}


@pytest.fixture(autouse=True)
def _reset_chrome_probe():
    """A verdict memoised by an earlier test must not stand in for this one's.

    The same reset ``tests/unit/plotting/conftest.py`` applies; that conftest
    does not reach this directory.
    """
    from phenotypic.plotting._pipeline._backends import reset_chrome_probe

    reset_chrome_probe()
    yield
    reset_chrome_probe()


@pytest.fixture(scope="module")
def measured_plate():
    """One real plate, detected and measured once for the whole module."""
    pipeline = ImagePipeline(ops=_OPS, meas=_MEAS)
    image = load_synth_yeast_plate()
    measurements = pipeline.apply_and_measure(image, inplace=True)
    assert image.num_objects > 0, "premise: Otsu found colonies on the plate"
    assert len(measurements) > 0, "premise: the plate produced measurements"
    return image, measurements


def _plots(tmp_path: Path) -> Path:
    return tmp_path / "deliverables" / "plots"


class ObjectCount(BaseModel, PlotImage):
    @figure(title="Object count", backend="plotly", primary=True)
    def count(self, image):
        import plotly.graph_objects as go

        return go.Figure(go.Bar(x=["objects"], y=[image.num_objects]))


class ObjectCountMpl(BaseModel, PlotImage):
    @figure(title="Object count", backend="mpl", primary=True)
    def count(self, image):
        from matplotlib.figure import Figure

        fig = Figure()
        fig.subplots().bar(["objects"], [image.num_objects])
        return fig


class ColonyCount(BaseModel, PlotMeas):
    @figure(title="Colonies measured", backend="plotly", primary=True)
    def count(self, measurements):
        import plotly.graph_objects as go

        return go.Figure(go.Bar(x=["colonies"], y=[len(measurements)]))


class ExplodingImagePlot(BaseModel, PlotImage):
    @figure(title="Never renders", backend="plotly", primary=True)
    def count(self, image):
        raise RuntimeError("colony count unavailable")


def test_a_plotly_image_plot_publishes_html_and_one_hoisted_bundle(
    tmp_path: Path, measured_plate
) -> None:
    image, _measurements = measured_plate
    pipeline = ImagePipeline(ops=_OPS, meas=_MEAS, plots=[ObjectCount()])

    PlotCoordinator(pipeline, tmp_path).emit_image(
        image, dataset="ds 1", image_stem="plate_01", strict=True
    )

    plots = _plots(tmp_path)
    directory = plots / "ObjectCount" / "ds-1"
    pages = list(directory.glob("plate_01-*.html"))
    assert len(pages) == 1, "expected exactly one published HTML page"

    assert sorted(tmp_path.rglob("plotly.min.js")) == [plots / "plotly.min.js"]
    assert 'src="../../plotly.min.js"' in pages[0].read_text(encoding="utf-8")

    # PNG presence tracks the real capability of this machine.
    assert pages[0].with_suffix(".png").is_file() is chrome_available()

    # The flat path writes no manifest, and nothing was recorded as failed.
    assert not (directory / "manifest.json").exists()
    assert list(tmp_path.rglob(".failures.jsonl")) == []


def test_a_matplotlib_image_plot_publishes_png_only_and_no_bundle(
    tmp_path: Path, measured_plate
) -> None:
    image, _measurements = measured_plate
    pipeline = ImagePipeline(ops=_OPS, meas=_MEAS, plots=[ObjectCountMpl()])

    PlotCoordinator(pipeline, tmp_path).emit_image(
        image, dataset="ds 1", image_stem="plate_01", strict=True
    )

    directory = _plots(tmp_path) / "ObjectCountMpl" / "ds-1"
    assert len(list(directory.glob("plate_01-*.png"))) == 1
    assert list(tmp_path.rglob("*.html")) == []
    # Matplotlib has no HTML form, so nothing should have fetched the bundle.
    assert list(tmp_path.rglob("plotly.min.js")) == []
    assert list(tmp_path.rglob(".failures.jsonl")) == []


def test_an_aggregate_plot_manifest_matches_what_is_on_disk(
    tmp_path: Path, measured_plate
) -> None:
    _image, measurements = measured_plate
    pipeline = ImagePipeline(ops=_OPS, meas=_MEAS, plots=[ColonyCount()])

    PlotCoordinator(pipeline, tmp_path).emit_measurements(measurements)

    directory = _plots(tmp_path) / "ColonyCount"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    can_rasterise = chrome_available()

    assert manifest["schema_version"] == 2
    assert manifest["failed"] == []
    assert manifest["renderers"] == {
        "html": "available",
        "png": "available" if can_rasterise else "unavailable: chrome not found",
    }
    (page,) = manifest["pages"]
    assert page["backend"] == "plotly"
    assert set(page["files"]) == ({"html", "png"} if can_rasterise else {"html"})

    # The manifest names exactly the page files present -- no more, no fewer.
    on_disk = {
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.name != "manifest.json"
    }
    assert on_disk == set(page["files"].values())


def test_a_failing_plot_is_recorded_once_and_does_not_stop_its_neighbour(
    tmp_path: Path, measured_plate
) -> None:
    image, _measurements = measured_plate
    # The failing plot runs FIRST, so its neighbour publishing proves the loop
    # continued past the failure rather than merely preceding it.
    pipeline = ImagePipeline(
        ops=_OPS, meas=_MEAS, plots=[ExplodingImagePlot(), ObjectCount()]
    )

    PlotCoordinator(pipeline, tmp_path).emit_image(
        image, dataset="ds 1", image_stem="plate_01"
    )

    plots = _plots(tmp_path)
    assert sorted(tmp_path.rglob(".failures.jsonl")) == [plots / ".failures.jsonl"]
    entries = [
        json.loads(line)
        for line in (plots / ".failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(entries) == 1, entries
    (entry,) = entries
    assert entry["binding_id"] == "ExplodingImagePlot"
    assert entry["lifecycle"] == "image"
    assert (entry["dataset"], entry["image_stem"]) == ("ds 1", "plate_01")
    assert entry["error"] == "RuntimeError: colony count unavailable"

    assert len(list((plots / "ObjectCount" / "ds-1").glob("plate_01-*.html"))) == 1
    assert not (plots / "ExplodingImagePlot").exists()
