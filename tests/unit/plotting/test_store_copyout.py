"""Copy-out: promoted store -> today's deliverables layout (spec §3 step 3)."""
from __future__ import annotations

import json

import pytest

from phenotypic.plotting._pipeline._coordinator import _image_output_stem
from phenotypic.plotting._pipeline._store_copyout import publish_store_figures
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
)
from tests.unit.plotting._store_fixtures import figure_store

_STEM = _image_output_stem("ds 1", "plate_01")


def _plotly_json() -> bytes:
    import plotly.graph_objects as go

    return go.Figure(go.Bar(x=["a"], y=[1])).to_json().encode()


def _page(key="default", label=None, formats=("plotly-json",), backend="plotly"):
    table = {
        "plotly-json": ("application/vnd.plotly.v1+json", ".plotly.json", _plotly_json()),
        "png": ("image/png", ".png", b"\x89PNG"),
    }
    stem = key.replace(" ", "-")
    return StoredFigurePage(key, label, backend, {"k": 1}, tuple(
        StoredFigureFile(fmt, table[fmt][0], f"{stem}{table[fmt][1]}", table[fmt][2])
        for fmt in formats
    ))


def _one(*pages, binding="sym", failed=()):
    return StoredFigures((StoredFigureBinding(binding, "MeasureSymZones", binding, pages),), failed)


def _publish(tmp_path, store, **kw):
    plots = tmp_path / "deliverables" / "plots"
    publish_store_figures(store, plots, dataset="ds 1", image_stem="plate_01", **kw)
    return plots


def _lines(plots):
    text = (plots / ".failures.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_a_single_default_page_lands_flat_with_generated_html(tmp_path):
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    base = plots / "sym" / "ds-1"
    assert sorted(p.name for p in base.iterdir()) == [f"{_STEM}.html", f"{_STEM}.plotly.json"]
    assert 'src="../../plotly.min.js"' in (base / f"{_STEM}.html").read_text(encoding="utf-8")
    assert (plots / "plotly.min.js").is_file()
    assert not (base / "manifest.json").exists()


def test_multi_page_writes_a_directory_and_manifest_v2(tmp_path):
    pages = (_page("first", "First"), _page("second", formats=("plotly-json", "png")))
    failed = (StoredFigureFailure("sym", "second", "png", "OSError: partial"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages, failed=failed)))
    directory = plots / "sym" / "ds-1" / _STEM
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert [p["key"] for p in manifest["pages"]] == ["first", "second"]
    assert manifest["pages"][0]["files"] == {"plotly-json": "First.plotly.json", "html": "First.html"}
    assert manifest["pages"][1]["files"]["png"] == "second.png"
    assert manifest["pages"][1]["partial"] == ["OSError: partial"]
    assert manifest["pages"][0]["metadata"] == {"k": 1}
    assert manifest["renderers"] == {"html": "available"}
    assert 'src="../../../plotly.min.js"' in (directory / "First.html").read_text(encoding="utf-8")
    # No plot_classes passed: a published binding's class comes from the store.
    [record] = _lines(plots)
    assert (record["page"], record["format"], record["plot_class"]) == (
        "second", "png", "MeasureSymZones"
    )


def test_a_failed_html_generation_is_partial_on_a_published_page(tmp_path, monkeypatch):
    from phenotypic.plotting._pipeline import _store_copyout

    def _html_breaks(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_store_copyout, "_write_html_from_json", _html_breaks)
    pages = (_page("first"), _page("second"))
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages)))
    manifest = json.loads(
        (plots / "sym" / "ds-1" / _STEM / "manifest.json").read_text(encoding="utf-8")
    )
    first = manifest["pages"][0]
    assert first["files"] == {"plotly-json": "first.plotly.json"}
    assert first["partial"] == ["OSError: disk full"]
    assert [r["page"] for r in _lines(plots)] == ["first", "second"]


def test_a_refused_guard_mid_page_leaves_no_half_page(tmp_path, monkeypatch):
    """S11 parity with `_render_page`: the copied JSON goes with the refusal."""
    from phenotypic.plotting._pipeline import PlotPublicationBlocked, _store_copyout

    def _html_refused(*args, **kwargs):
        raise PlotPublicationBlocked("refused")

    monkeypatch.setattr(_store_copyout, "_write_html_from_json", _html_refused)
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    base = tmp_path / "deliverables" / "plots" / "sym" / "ds-1"
    assert not (base / f"{_STEM}.plotly.json").exists()


def test_a_tampered_file_is_recorded_and_not_copied(tmp_path):
    store = figure_store(tmp_path / "s", _one(_page()))
    (store / "figures/sym/default.plotly.json").write_bytes(b"tampered")
    plots = _publish(tmp_path, store, plot_classes={"sym": "MeasureSymZones"})
    assert not list((plots / "sym").rglob("*.plotly.json"))
    [record] = _lines(plots)
    assert record["error"].startswith("ValueError: ") and "sha256" in record["error"]
    assert (record["page"], record["format"], record["plot_class"]) == (
        "default", "plotly-json", "MeasureSymZones"
    )


def test_descriptor_failures_are_recorded_verbatim_with_their_class(tmp_path):
    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom at 0x…"),)
    plots = _publish(
        tmp_path, figure_store(tmp_path / "s", _one(_page(), failed=failed)),
        plot_classes={"orient": "MeasureOrientationZones"},
    )
    [record] = _lines(plots)
    assert record["error"] == "RuntimeError: boom at 0x…"
    assert record["plot_class"] == "MeasureOrientationZones"
    assert "page" not in record and "format" not in record
    assert (record["binding_id"], record["dataset"], record["image_stem"]) == (
        "orient", "ds 1", "plate_01"
    )


def test_a_republished_page_loses_its_leftover_renderings(tmp_path):
    base = tmp_path / "deliverables" / "plots" / "sym" / "ds-1"
    base.mkdir(parents=True)
    (base / f"{_STEM}.png").write_bytes(b"a png from a run that stored png")
    _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    assert not (base / f"{_STEM}.png").exists()


def test_a_page_that_publishes_nothing_keeps_its_previous_files(tmp_path):
    base = tmp_path / "deliverables" / "plots" / "sym" / "ds-1"
    base.mkdir(parents=True)
    for suffix in (".html", ".png"):
        (base / f"{_STEM}{suffix}").write_bytes(b"previous")
    store = figure_store(tmp_path / "s", _one(_page()))
    (store / "figures/sym/default.plotly.json").write_bytes(b"tampered")
    _publish(tmp_path, store)
    assert sorted(p.name for p in base.iterdir()) == [f"{_STEM}.html", f"{_STEM}.png"]


def test_a_store_without_figures_publishes_nothing(tmp_path):
    store = tmp_path / "s" / "p.ome.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"phenotypic": {}}}),
        encoding="utf-8",
    )
    plots = _publish(tmp_path, store)
    assert not plots.exists()


def test_an_unreadable_store_is_one_record(tmp_path):
    store = tmp_path / "s" / "p.ome.zarr"
    store.mkdir(parents=True)
    [record] = _lines(_publish(tmp_path, store))
    assert record["binding_id"] == "<store>"
    assert record["error"].startswith("FileNotFoundError: ")


def test_an_unreadable_store_asks_the_guard_before_recording(tmp_path):
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    store = tmp_path / "s" / "p.ome.zarr"
    store.mkdir(parents=True)
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, store, publication_guard=lambda: False)
    assert not (tmp_path / "deliverables").exists()


def test_the_coordinator_names_a_failed_binding_by_its_pipeline_class(tmp_path):
    """The descriptor has no class for a binding whose inspect() raised."""
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotImage, figure
    from phenotypic.plotting._pipeline import PlotCoordinator
    from tests.unit.plotting._store_fixtures import emit_image_via_store

    class Bars(BaseModel, PlotImage):
        @figure(title="bars", backend="plotly", primary=True)
        def draw(self, image):
            import plotly.graph_objects as go

            return go.Figure(go.Bar(x=["a"], y=[1]))

    class Explodes(BaseModel, PlotImage):
        def inspect(self, subject=None, *, for_save=False, **overrides):
            raise RuntimeError("boom")

    coordinator = PlotCoordinator(ImagePipeline(plots=[Bars(), Explodes()]), tmp_path)
    emit_image_via_store(coordinator, dataset="ds", image_stem="plate-1")
    plots = tmp_path / "deliverables" / "plots"
    stem = _image_output_stem("ds", "plate-1")
    assert (plots / "Bars" / "ds" / f"{stem}.html").is_file()
    [record] = _lines(plots)
    assert (record["binding_id"], record["plot_class"]) == ("Explodes", "Explodes")
    assert record["error"] == "RuntimeError: boom"


def test_a_refused_guard_propagates_before_anything_is_written(tmp_path):
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom"),)
    store = figure_store(tmp_path / "s", _one(_page(), failed=failed))
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, store, publication_guard=lambda: False)
    assert not (tmp_path / "deliverables").exists()
