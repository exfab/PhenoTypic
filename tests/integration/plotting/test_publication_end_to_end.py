"""Real image, real detector, every publication path -- composed.

Unit tests prove each piece in isolation. These prove they compose on a real
pipeline: a synthetic yeast plate, a real ``OtsuDetector``, real measurements.
Nothing in the code under test is patched. Chrome is never assumed either way:
every rendering assertion holds on a node with Chrome and on one without --
the image plots store only their backend default, which needs no Chrome, and
the aggregate assertions are written against ``chrome_available()``, the
machine's real, freshly probed capability.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import BaseModel

from phenotypic import ImagePipeline
from phenotypic.abc_.plotting import PlotImage, PlotMeas, figure
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.plotting._pipeline import PlotCoordinator, chrome_available
from phenotypic.plotting._pipeline._store_figures import build_image_figures
from tests.unit.plotting._store_fixtures import TEST_RUN

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


def _image_directory(plot_directory: Path) -> Path:
    """The one ``<stem>-<hash>`` directory an image plot published for plate_01."""
    [directory] = plot_directory.glob("plate_01-*")
    return directory


def _publish_through_store(
    pipeline: ImagePipeline, image, tmp_path: Path, *, expect_clean: bool = True
) -> None:
    """build -> real store -> copy-out, the path every CLI mode takes.

    The store sits beside ``tmp_path``, not in it, so the tree assertions
    below see only deliverables. It is removed once copy-out has read it.
    """
    stored = build_image_figures(pipeline, image, run=TEST_RUN)
    if expect_clean:
        # Replaces `strict=True`: a failed build must not pass quietly.
        assert stored.failed == ()
    store = tmp_path.parent / f"{tmp_path.name}-store.ome.zarr"
    try:
        store = image.save2zarr(store, figures=stored)
        PlotCoordinator(pipeline, tmp_path).publish_store_figures(
            store, run_id=TEST_RUN.run_id, dataset="ds 1", image_stem="plate_01"
        )
    finally:
        shutil.rmtree(store, ignore_errors=True)


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

    _publish_through_store(pipeline, image, tmp_path)

    plots = _plots(tmp_path)
    directory = _image_directory(plots / "ObjectCount" / "ds-1")
    pages = list(directory.rglob("*.html"))
    assert len(pages) == 1, "expected exactly one published HTML page"

    # The page sits in its plot folder, two levels deeper than before, so the
    # relative src to the one hoisted bundle is two levels longer too.
    assert sorted(tmp_path.rglob("plotly.min.js")) == [plots / "plotly.min.js"]
    assert 'src="../../../../plotly.min.js"' in pages[0].read_text(encoding="utf-8")

    # The stored default is `plotly-json`, copied out beside its HTML; no
    # PNG is stored, so none is published whatever this machine can render.
    assert len(list(directory.rglob("*.plotly.json"))) == 1
    assert list(directory.rglob("*.png")) == []

    # A manifest directory with nothing failed, naming exactly the files copied
    # out (the pre-folder flat path wrote none; there is no flat case now).
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["failed"] == []
    (page,) = manifest["pages"]
    assert set(page["files"]) == {"html", "plotly-json"}
    assert {directory / name for name in page["files"].values()} == {
        pages[0], *directory.rglob("*.plotly.json")
    }
    assert list(tmp_path.rglob(".failures.jsonl")) == []


def test_a_matplotlib_image_plot_publishes_png_only_and_no_bundle(
    tmp_path: Path, measured_plate
) -> None:
    image, _measurements = measured_plate
    pipeline = ImagePipeline(ops=_OPS, meas=_MEAS, plots=[ObjectCountMpl()])

    _publish_through_store(pipeline, image, tmp_path)

    directory = _image_directory(_plots(tmp_path) / "ObjectCountMpl" / "ds-1")
    assert len(list(directory.rglob("*.png"))) == 1
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

    assert manifest["schema_version"] == 3
    assert manifest["failed"] == []
    assert manifest["renderers"] == {
        "html": "available",
        "png": "available" if can_rasterise else "unavailable: chrome not found",
    }
    (page,) = manifest["pages"]
    assert page["backend"] == "plotly"
    assert set(page["files"]) == ({"html", "png"} if can_rasterise else {"html"})

    # The manifest names exactly the page files present -- no more, no fewer --
    # as paths relative to the manifest, each inside the page's plot folder.
    on_disk = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.name != "manifest.json"
    }
    assert on_disk == set(page["files"].values())
    assert {Path(name).parent.as_posix() for name in page["files"].values()} == {
        page["plot"]
    }


def test_a_failing_plot_is_recorded_once_and_does_not_stop_its_neighbour(
    tmp_path: Path, measured_plate
) -> None:
    image, _measurements = measured_plate
    # The failing plot runs FIRST, so its neighbour publishing proves the loop
    # continued past the failure rather than merely preceding it.
    pipeline = ImagePipeline(
        ops=_OPS, meas=_MEAS, plots=[ExplodingImagePlot(), ObjectCount()]
    )

    _publish_through_store(pipeline, image, tmp_path, expect_clean=False)

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

    neighbour = _image_directory(plots / "ObjectCount" / "ds-1")
    assert len(list(neighbour.rglob("*.html"))) == 1
    assert not (plots / "ExplodingImagePlot").exists()
