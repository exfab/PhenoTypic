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
    # The stored JSON copied fine: the record names the rendering, not the store.
    assert [(r["page"], r["format"]) for r in _lines(plots)] == [
        ("first", "html"), ("second", "html")
    ]


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
    from phenotypic.plotting._pipeline import PlotBinding, PlotCoordinator
    from tests.unit.plotting._store_fixtures import emit_image_via_store

    class Bars(BaseModel, PlotImage):
        @figure(title="bars", backend="plotly", primary=True)
        def draw(self, image):
            import plotly.graph_objects as go

            return go.Figure(go.Bar(x=["a"], y=[1]))

    class Explodes(BaseModel, PlotImage):
        def inspect(self, subject=None, *, for_save=False, **overrides):
            raise RuntimeError("boom")

    # An id unlike the class, so a fallback to the id cannot pass for the class.
    pipeline = ImagePipeline(plots=[Bars(), PlotBinding(id="exploder", plot=Explodes())])
    coordinator = PlotCoordinator(pipeline, tmp_path)
    emit_image_via_store(coordinator, dataset="ds", image_stem="plate-1")
    plots = tmp_path / "deliverables" / "plots"
    stem = _image_output_stem("ds", "plate-1")
    assert (plots / "Bars" / "ds" / f"{stem}.html").is_file()
    [record] = _lines(plots)
    assert (record["binding_id"], record["plot_class"]) == ("exploder", "Explodes")
    assert record["error"] == "RuntimeError: boom"
    # MINOR-13: the helper's store, kept outside tmp_path, does not leak.
    assert not list(tmp_path.parent.glob(f"{tmp_path.name}-store-*"))


def test_a_refused_guard_propagates_before_anything_is_written(tmp_path):
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom"),)
    store = figure_store(tmp_path / "s", _one(_page(), failed=failed))
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, store, publication_guard=lambda: False)
    assert not (tmp_path / "deliverables").exists()


# --- MAJOR-2: layout and manifest `failed` cover every page, failed ones too ---


def _manifest(plots, binding="sym"):
    path = plots / binding / "ds-1" / _STEM / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_page_that_failed_outright_stays_in_the_manifest(tmp_path):
    """TwoOneBad: the store has page `good`; `bad` stored nothing."""
    failed = (StoredFigureFailure("sym", "bad", None, "TypeError: unsupported figure type"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page("good"), failed=failed)))
    manifest = _manifest(plots)
    assert [p["key"] for p in manifest["pages"]] == ["good"]
    assert manifest["failed"] == [
        {"key": "bad", "label": None, "error": "TypeError: unsupported figure type"}
    ]


def test_a_failed_second_page_does_not_flip_the_layout_to_flat(tmp_path):
    """DefaultPlusBad: two pages produced, so a directory, as the old writer did."""
    failed = (StoredFigureFailure("sym", "bad", None, "TypeError: nope"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page(), failed=failed)))
    base = plots / "sym" / "ds-1"
    assert not (base / f"{_STEM}.plotly.json").exists()
    assert (base / _STEM / "default.plotly.json").is_file()
    manifest = _manifest(plots)
    assert [p["key"] for p in manifest["pages"]] == ["default"]
    assert [f["key"] for f in manifest["failed"]] == ["bad"]


def test_every_page_failed_replaces_the_previous_manifest(tmp_path):
    """No published page, yet the directory's manifest must describe this run."""
    directory = tmp_path / "deliverables" / "plots" / "sym" / "ds-1" / _STEM
    directory.mkdir(parents=True)
    (directory / "a.html").write_bytes(b"previous")
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "pages": [{"key": "a"}], "failed": []}),
        encoding="utf-8",
    )
    failed = (
        StoredFigureFailure("sym", "a", "png", "PlotBackendUnavailable: no chrome"),
        StoredFigureFailure("sym", "b", None, "TypeError: nope"),
        StoredFigureFailure("sym", "a", "plotly-json", "OSError: second"),
    )
    plots = _publish(
        tmp_path, figure_store(tmp_path / "s", StoredFigures((), failed)),
        plot_classes={"sym": "MeasureSymZones"},
    )
    manifest = _manifest(plots)
    assert manifest["pages"] == []
    assert manifest["class"] == "MeasureSymZones"
    assert manifest["failed"] == [
        {"key": "a", "label": None, "error": "PlotBackendUnavailable: no chrome"},
        {"key": "b", "label": None, "error": "TypeError: nope"},
    ]
    # A page that published nothing keeps its previous files.
    assert (directory / "a.html").read_bytes() == b"previous"


def test_a_failed_lone_default_page_writes_nothing(tmp_path):
    """Flat case: a page that published nothing keeps its previous files, so
    there is nothing to write -- not even the binding's directory."""
    failed = (StoredFigureFailure("sym", "default", "png", "PlotBackendUnavailable: x"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", StoredFigures((), failed)))
    assert [r["page"] for r in _lines(plots)] == ["default"]
    assert not (plots / "sym").exists()


def test_a_binding_level_failure_is_a_record_only(tmp_path):
    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", StoredFigures((), failed)))
    assert [r["binding_id"] for r in _lines(plots)] == ["orient"]
    assert not (plots / "orient").exists()


def test_the_manifest_class_is_the_pipelines_class(tmp_path):
    """MINOR-8: the same class the failure records carry."""
    failed = (StoredFigureFailure("sym", "second", "png", "OSError: partial"),)
    pages = (_page("first"), _page("second", formats=("plotly-json", "png")))
    plots = _publish(
        tmp_path, figure_store(tmp_path / "s", _one(*pages, failed=failed)),
        plot_classes={"sym": "Renamed"},
    )
    assert _manifest(plots)["class"] == "Renamed"
    assert [r["plot_class"] for r in _lines(plots)] == ["Renamed"]


# --- MINOR-7: the descriptor is checked, not trusted ---


def _edit_descriptor(store, edit):
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    edit(root["attributes"]["phenotypic"]["figures"])
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")


def test_an_unknown_schema_version_is_skipped_and_recorded(tmp_path):
    store = figure_store(tmp_path / "s", _one(_page()))
    _edit_descriptor(store, lambda d: d.update(schema_version=2))
    plots = _publish(tmp_path, store)
    assert not (plots / "sym").exists()
    [record] = _lines(plots)
    assert record["binding_id"] == "<store>"
    assert "schema_version 2" in record["error"]


def test_a_path_outside_figures_is_a_per_file_failure(tmp_path):
    """Correct sha256, so only the containment check can refuse it."""
    store = figure_store(tmp_path / "s", _one(_page()))
    data = (store / "figures/sym/default.plotly.json").read_bytes()
    (store / "outside.plotly.json").write_bytes(data)

    def _escape(descriptor):
        descriptor["bindings"]["sym"]["pages"][0]["files"][0]["path"] = (
            "figures/../outside.plotly.json"
        )

    _edit_descriptor(store, _escape)
    plots = _publish(tmp_path, store)
    assert not list((plots / "sym").rglob("*.plotly.json"))
    [record] = _lines(plots)
    assert (record["page"], record["format"]) == ("default", "plotly-json")
    assert "outside figures/" in record["error"]
