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
from tests.unit.plotting._store_fixtures import TEST_RUN, figure_store, run_path

_STEM = _image_output_stem("ds 1", "plate_01")


def _plotly_json() -> bytes:
    import plotly.graph_objects as go

    return go.Figure(go.Bar(x=["a"], y=[1])).to_json().encode()


def _page(key="default", label=None, formats=("plotly-json",), backend="plotly",
          plot=None, flat=False):
    """A page as the builder stores it: in the folder of its cleaned plot name
    (plot defaults to the key), or *flat*, as a version 1 run stored it."""
    table = {
        "plotly-json": ("application/vnd.plotly.v1+json", ".plotly.json", _plotly_json()),
        "png": ("image/png", ".png", b"\x89PNG"),
    }
    stem = key.replace(" ", "-")
    plot = None if flat else (plot or key)
    return StoredFigurePage(
        key, label, backend, {"k": 1},
        tuple(StoredFigureFile(fmt, table[fmt][0], f"{stem}{table[fmt][1]}", table[fmt][2])
              for fmt in formats),
        plot=plot,
        directory=None if flat else plot.replace(" ", "-"),
    )


def _one(*pages, binding="sym", failed=()):
    return StoredFigures(
        TEST_RUN, (StoredFigureBinding(binding, "MeasureSymZones", binding, pages),), failed
    )


def _publish(tmp_path, store, **kw):
    plots = tmp_path / "deliverables" / "plots"
    kw.setdefault("run_id", TEST_RUN.run_id)
    publish_store_figures(store, plots, dataset="ds 1", image_stem="plate_01", **kw)
    return plots


def _lines(plots):
    text = (plots / ".failures.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_a_single_default_page_lands_in_its_plot_folder_with_generated_html(tmp_path):
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    directory = plots / "sym" / "ds-1" / _STEM
    assert sorted(p.name for p in (directory / "default").iterdir()) == [
        "default.html", "default.plotly.json",
    ]
    html = (directory / "default" / "default.html").read_text(encoding="utf-8")
    assert 'src="../../../../plotly.min.js"' in html
    assert (plots / "plotly.min.js").is_file()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["pages"][0]["plot"] == "default"
    assert manifest["pages"][0]["files"] == {
        "plotly-json": "default/default.plotly.json", "html": "default/default.html",
    }


def test_multi_page_writes_a_directory_and_manifest_v3(tmp_path):
    pages = (_page("first", "First"), _page("second", formats=("plotly-json", "png")))
    failed = (StoredFigureFailure("sym", "second", "png", "OSError: partial", plot="second"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages, failed=failed)))
    directory = plots / "sym" / "ds-1" / _STEM
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert [(p["key"], p["plot"]) for p in manifest["pages"]] == [
        ("first", "first"), ("second", "second"),
    ]
    # Decision P1: named by key, as the store names them; the label stays a label.
    assert manifest["pages"][0]["files"] == {
        "plotly-json": "first/first.plotly.json", "html": "first/first.html",
    }
    assert manifest["pages"][0]["label"] == "First"
    assert manifest["pages"][1]["files"]["png"] == "second/second.png"
    assert (directory / "second" / "second.png").read_bytes() == b"\x89PNG"
    assert manifest["pages"][1]["partial"] == ["OSError: partial"]
    assert manifest["pages"][0]["metadata"] == {"k": 1}
    assert manifest["renderers"] == {"html": "available"}
    html = (directory / "first" / "first.html").read_text(encoding="utf-8")
    assert 'src="../../../../plotly.min.js"' in html
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
    assert first["files"] == {"plotly-json": "first/first.plotly.json"}
    assert first["partial"] == ["OSError: disk full"]
    # The stored JSON copied fine: the record names the rendering, not the store.
    assert [(r["page"], r["format"]) for r in _lines(plots)] == [
        ("first", "html"), ("second", "html")
    ]


def test_a_refused_guard_mid_page_leaves_no_half_page(tmp_path, monkeypatch):
    """S11 parity with `_render_page`: the copied JSON goes with the refusal."""
    from phenotypic.plotting._pipeline import PlotPublicationBlocked, _store_copyout

    calls = []

    def _html_refused(data, directory, *args, **kwargs):
        calls.append(directory)
        raise PlotPublicationBlocked("refused")

    monkeypatch.setattr(_store_copyout, "_write_html_from_json", _html_refused)
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    image_dir = tmp_path / "deliverables" / "plots" / "sym" / "ds-1" / _STEM
    # Mid-page: the JSON was copied into the plot folder before the refusal.
    assert calls == [image_dir / "default"]
    assert list(image_dir.rglob("*.plotly.json")) == []


def test_a_tampered_file_is_recorded_and_not_copied(tmp_path):
    store = figure_store(tmp_path / "s", _one(_page()))
    stored = store / run_path("sym/default/default.plotly.json")
    assert stored.is_file()
    stored.write_bytes(b"tampered")
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
    folder = tmp_path / "deliverables" / "plots" / "sym" / "ds-1" / _STEM / "default"
    folder.mkdir(parents=True)
    (folder / "default.png").write_bytes(b"a png from a run that stored png")
    _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    assert sorted(p.name for p in folder.iterdir()) == ["default.html", "default.plotly.json"]


def test_a_page_that_publishes_nothing_keeps_its_previous_files(tmp_path):
    folder = tmp_path / "deliverables" / "plots" / "sym" / "ds-1" / _STEM / "default"
    folder.mkdir(parents=True)
    for suffix in (".html", ".png"):
        (folder / f"default{suffix}").write_bytes(b"previous")
    store = figure_store(tmp_path / "s", _one(_page()))
    stored = store / run_path("sym/default/default.plotly.json")
    assert stored.is_file()
    stored.write_bytes(b"tampered")
    plots = _publish(tmp_path, store)
    # This pass did publish this image folder: its manifest says the page
    # failed, and the page's previous files stay whole.
    assert [f["key"] for f in _manifest(plots)["failed"]] == ["default"]
    assert sorted(p.name for p in folder.iterdir()) == ["default.html", "default.png"]


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
    assert (plots / "Bars" / "ds" / stem / "default" / "default.html").is_file()
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
    # As the builder records it: a bare page's plot is its key.
    failed = (
        StoredFigureFailure("sym", "bad", None, "TypeError: unsupported figure type", plot="bad"),
    )
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page("good"), failed=failed)))
    manifest = _manifest(plots)
    assert [p["key"] for p in manifest["pages"]] == ["good"]
    assert manifest["failed"] == [
        {"key": "bad", "plot": "bad", "label": None,
         "error": "TypeError: unsupported figure type"}
    ]


def test_a_failed_page_keeps_its_plot_in_the_manifest(tmp_path):
    """Review Focus 5."""
    failed = (StoredFigureFailure("sym", "roi_1", None, "TypeError: nope", plot="tiles"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page(), failed=failed)))
    manifest = _manifest(plots)
    assert manifest["failed"] == [
        {"key": "roi_1", "plot": "tiles", "label": None, "error": "TypeError: nope"}
    ]


def test_a_failure_belongs_to_the_page_of_its_key_and_plot(tmp_path):
    """One key in two plots is two pages: a failure in one is not the other's."""
    failed = (
        StoredFigureFailure("sym", "roi_0", "png", "OSError: tiles png", plot="tiles"),
        StoredFigureFailure("sym", "roi_0", None, "TypeError: delta", plot="delta_e"),
    )
    store = figure_store(tmp_path / "s", _one(_page("roi_0", plot="tiles"), failed=failed))
    manifest = _manifest(_publish(tmp_path, store))
    [page] = manifest["pages"]
    assert (page["plot"], page["partial"]) == ("tiles", ["OSError: tiles png"])
    assert manifest["failed"] == [
        {"key": "roi_0", "plot": "delta_e", "label": None, "error": "TypeError: delta"}
    ]


def test_one_key_in_two_plots_is_two_files(tmp_path):
    pages = (_page("summary", plot="tiles"), _page("summary", plot="delta_e"))
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages)))
    directory = plots / "sym" / "ds-1" / _STEM
    assert [p["files"]["plotly-json"] for p in _manifest(plots)["pages"]] == [
        "tiles/summary.plotly.json", "delta_e/summary.plotly.json",
    ]
    assert (directory / "tiles" / "summary.plotly.json").is_file()
    assert (directory / "delta_e" / "summary.plotly.json").is_file()


def test_the_plot_folder_comes_from_the_stored_path_not_the_plot_name(tmp_path):
    """`plot` is the logical name; the folder is the store's cleaned one."""
    plots = _publish(
        tmp_path, figure_store(tmp_path / "s", _one(_page("roi_0", plot="Tile overlay")))
    )
    directory = plots / "sym" / "ds-1" / _STEM
    [page] = _manifest(plots)["pages"]
    assert page["plot"] == "Tile overlay"
    assert page["files"] == {
        "plotly-json": "Tile-overlay/roi_0.plotly.json", "html": "Tile-overlay/roi_0.html",
    }
    assert sorted(p.name for p in directory.iterdir() if p.is_dir()) == ["Tile-overlay"]
    assert (directory / "Tile-overlay" / "roi_0.html").is_file()


def test_a_disambiguated_folder_and_file_are_copied_as_the_store_named_them(tmp_path):
    """P1 / D4: folder and file come from the stored path, including where the
    store had to disambiguate them, so neither can be re-derived from
    `plot` or `key`."""
    from phenotypic.plotting._pipeline._writer import plot_page_paths

    ids = [("Tiles", "roi 0"), ("tiles", "roi 0")]
    paths = plot_page_paths([(plot, key, key) for plot, key in ids])
    # Premise: the second folder carries the digest suffix; the stem is not the key.
    assert paths[0] == ("Tiles", "roi-0") and paths[1][0].startswith("tiles-")
    pages = tuple(
        StoredFigurePage(
            key, None, "plotly", {"k": 1},
            (StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                              f"{stem}.plotly.json", _plotly_json()),),
            plot=plot, directory=folder,
        )
        for (plot, key), (folder, stem) in zip(ids, paths)
    )
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages)))
    directory = plots / "sym" / "ds-1" / _STEM
    assert [p["files"] for p in _manifest(plots)["pages"]] == [
        {"plotly-json": f"{folder}/{stem}.plotly.json", "html": f"{folder}/{stem}.html"}
        for folder, stem in paths
    ]
    for folder, stem in paths:
        assert (directory / folder / f"{stem}.html").is_file()


def test_a_flat_v1_page_is_copied_into_the_image_folder(tmp_path):
    """A version 1 run stored pages flat, with no `plot` anywhere."""
    failed = (StoredFigureFailure("sym", "default", "png", "OSError: partial"),)
    store = figure_store(tmp_path / "s", _one(_page(flat=True), failed=failed))

    def _as_v1(descriptor):
        descriptor["schema_version"] = 1
        run = descriptor["runs"][TEST_RUN.run_id]
        for entry in [*run["bindings"]["sym"]["pages"], *run["failed"]]:
            del entry["plot"]

    _edit_descriptor(store, _as_v1)
    assert (store / run_path("sym/default.plotly.json")).is_file()
    plots = _publish(tmp_path, store)
    directory = plots / "sym" / "ds-1" / _STEM
    assert sorted(p.name for p in directory.iterdir() if not p.name.startswith(".")) == [
        "default.html", "default.plotly.json", "manifest.json",
    ]
    assert 'src="../../../plotly.min.js"' in (directory / "default.html").read_text(
        encoding="utf-8"
    )
    [page] = _manifest(plots)["pages"]
    assert page["plot"] is None
    assert page["files"] == {"plotly-json": "default.plotly.json", "html": "default.html"}
    assert page["partial"] == ["OSError: partial"]


def test_a_page_with_no_copyable_file_is_failed_with_its_plot(tmp_path):
    """I4: the "no stored file could be copied out" entry carries `plot`."""
    store = figure_store(tmp_path / "s", _one(_page("roi_0", plot="tiles")))
    stored = store / run_path("sym/tiles/roi_0.plotly.json")
    assert stored.is_file()
    stored.write_bytes(b"tampered")
    plots = _publish(tmp_path, store)
    manifest = _manifest(plots)
    assert manifest["pages"] == []
    assert manifest["failed"] == [
        {"key": "roi_0", "plot": "tiles", "label": None,
         "error": "no stored file could be copied out"}
    ]
    assert not (plots / "sym" / "ds-1" / _STEM / "tiles").exists()


def test_a_page_whose_copy_fails_leaves_no_empty_plot_folder(tmp_path, monkeypatch):
    from phenotypic.plotting._pipeline import _store_copyout

    def _disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_store_copyout, "_atomic_write", _disk_full)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page("roi_0", plot="tiles"))))
    # Premise: the read passed, so the copy (after the mkdir) is what failed.
    assert [(r["format"], r["error"]) for r in _lines(plots)] == [
        ("plotly-json", "OSError: disk full")
    ]
    manifest = _manifest(plots)
    assert manifest["pages"] == []
    assert [(f["key"], f["plot"]) for f in manifest["failed"]] == [("roi_0", "tiles")]
    assert not (plots / "sym" / "ds-1" / _STEM / "tiles").exists()


def test_a_failed_copy_keeps_the_plot_folder_its_sibling_published_into(tmp_path, monkeypatch):
    """Only a folder the failing page created is removed, never a shared one."""
    from phenotypic.plotting._pipeline import _store_copyout

    real = _store_copyout._atomic_write

    def _second_page_fails(destination, *args, **kwargs):
        if destination.name.startswith("roi_1"):
            raise OSError("disk full")
        return real(destination, *args, **kwargs)

    monkeypatch.setattr(_store_copyout, "_atomic_write", _second_page_fails)
    pages = (_page("roi_0", plot="tiles"), _page("roi_1", plot="tiles"))
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages)))
    manifest = _manifest(plots)
    assert [p["key"] for p in manifest["pages"]] == ["roi_0"]
    assert [(f["key"], f["plot"]) for f in manifest["failed"]] == [("roi_1", "tiles")]
    assert sorted(p.name for p in (plots / "sym" / "ds-1" / _STEM / "tiles").iterdir()) == [
        "roi_0.html", "roi_0.plotly.json",
    ]


def test_a_refused_guard_creates_no_plot_folder(tmp_path):
    """I5: the guard is asked before a plot folder is created."""
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    store = figure_store(tmp_path / "s", _one(_page("roi_0", plot="tiles")))
    # Premise: an admitting guard does create the plot folder, so its absence
    # below is the refusal's doing.
    admitted = _publish(tmp_path / "admitted", store, publication_guard=lambda: True)
    assert (admitted / "sym" / "ds-1" / _STEM / "tiles" / "roi_0.plotly.json").is_file()
    # Entry check, the image folder's mkdir, inside its lock, then the plot folder.
    calls = iter([True, True, True, False])
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, store, publication_guard=lambda: next(calls, False))
    image_dir = tmp_path / "deliverables" / "plots" / "sym" / "ds-1" / _STEM
    assert image_dir.is_dir()                  # the refusal came after the image folder ...
    assert not (image_dir / "tiles").exists()  # ... and before the plot folder


def test_every_page_failed_replaces_the_previous_manifest(tmp_path):
    """No published page, yet the directory's manifest must describe this run."""
    directory = tmp_path / "deliverables" / "plots" / "sym" / "ds-1" / _STEM
    directory.mkdir(parents=True)
    (directory / "a").mkdir(parents=True)
    (directory / "a" / "a.html").write_bytes(b"previous")
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 3, "pages": [{"key": "a", "plot": "a"}], "failed": []}),
        encoding="utf-8",
    )
    failed = (
        StoredFigureFailure("sym", "a", "png", "PlotBackendUnavailable: no chrome", plot="a"),
        StoredFigureFailure("sym", "b", None, "TypeError: nope", plot="b"),
        StoredFigureFailure("sym", "a", "plotly-json", "OSError: second", plot="a"),
    )
    plots = _publish(
        tmp_path, figure_store(tmp_path / "s", StoredFigures(TEST_RUN, (), failed)),
        plot_classes={"sym": "MeasureSymZones"},
    )
    manifest = _manifest(plots)
    assert manifest["pages"] == []
    assert manifest["class"] == "MeasureSymZones"
    assert manifest["failed"] == [
        {"key": "a", "plot": "a", "label": None, "error": "PlotBackendUnavailable: no chrome"},
        {"key": "b", "plot": "b", "label": None, "error": "TypeError: nope"},
    ]
    # A page that published nothing keeps its previous files.
    assert (directory / "a" / "a.html").read_bytes() == b"previous"


def test_a_failed_lone_default_page_writes_a_manifest_of_its_failure(tmp_path):
    """No flat case any more: a lone `default` is a plot like any other."""
    failed = (
        StoredFigureFailure("sym", "default", "png", "PlotBackendUnavailable: x", plot="default"),
    )
    plots = _publish(tmp_path, figure_store(tmp_path / "s", StoredFigures(TEST_RUN, (), failed)))
    assert [r["page"] for r in _lines(plots)] == ["default"]
    manifest = _manifest(plots)
    assert manifest["pages"] == []
    assert manifest["failed"] == [
        {"key": "default", "plot": "default", "label": None,
         "error": "PlotBackendUnavailable: x"}
    ]
    # A page that published nothing gets no plot folder.
    assert not (plots / "sym" / "ds-1" / _STEM / "default").exists()


def test_a_binding_level_failure_is_a_record_only(tmp_path):
    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", StoredFigures(TEST_RUN, (), failed)))
    assert [r["binding_id"] for r in _lines(plots)] == ["orient"]
    assert not (plots / "orient").exists()


def test_the_manifest_class_is_the_pipelines_class(tmp_path):
    """MINOR-8: the same class the failure records carry."""
    failed = (StoredFigureFailure("sym", "second", "png", "OSError: partial", plot="second"),)
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
    _edit_descriptor(store, lambda d: d.update(schema_version=3))
    plots = _publish(tmp_path, store)
    assert not (plots / "sym").exists()
    [record] = _lines(plots)
    assert record["binding_id"] == "<store>"
    assert "schema_version 3" in record["error"]


def test_a_path_outside_figures_is_a_per_file_failure(tmp_path):
    """Correct sha256, so only the containment check can refuse it."""
    store = figure_store(tmp_path / "s", _one(_page()))
    data = (store / run_path("sym/default/default.plotly.json")).read_bytes()
    (store / "outside.plotly.json").write_bytes(data)

    def _escape(descriptor):
        descriptor["runs"][TEST_RUN.run_id]["bindings"]["sym"]["pages"][0]["files"][0]["path"] = (
            "figures/../outside.plotly.json"
        )

    _edit_descriptor(store, _escape)
    plots = _publish(tmp_path, store)
    assert not list((plots / "sym").rglob("*.plotly.json"))
    [record] = _lines(plots)
    assert (record["page"], record["format"]) == ("default", "plotly-json")
    assert "outside figures/" in record["error"]


# --- §1a: the copy-out publishes this run's folder, and only it ---


def test_only_this_runs_folder_is_published(tmp_path):
    from phenotypic.sdk_._image_figures import FigureRun

    other = FigureRun(date="2026-09-21", pipeline_sha256="cd" * 32)
    earlier = StoredFigures(
        other, (StoredFigureBinding("old", "Old", "old", (_page(),)),),
        (StoredFigureFailure("old", None, None, "RuntimeError: earlier run"),),
    )
    store = figure_store(tmp_path / "s", earlier, _one(_page()))
    plots = _publish(tmp_path, store)
    assert sorted(p.name for p in plots.iterdir() if p.is_dir()) == ["sym"]
    assert not (plots / ".failures.jsonl").exists()
    assert (plots / "sym" / "ds-1" / _STEM / "default" / "default.plotly.json").is_file()


def test_a_store_with_no_folder_for_this_run_publishes_nothing(tmp_path):
    store = figure_store(tmp_path / "s", _one(_page()))
    plots = _publish(tmp_path, store, run_id="2026-01-01-000000000000")
    assert not plots.exists()
