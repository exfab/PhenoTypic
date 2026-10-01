"""build_image_figures: in memory, finest-grained failures (spec §1, §3 step 1)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pydantic import BaseModel

from phenotypic import ImagePipeline
from phenotypic.abc_.plotting import (
    FigureInputUnavailable,
    PlotImage,
    PlotOutput,
    PlotPage,
    figure,
)
from phenotypic.plotting._pipeline._store_figures import (
    build_image_figures,
    normalize_figure_error,
)
from phenotypic.plotting._pipeline._writer import plot_page_paths
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_figure_run,
    write_image_figures,
)
from tests.unit.plotting._store_fixtures import TEST_RUN, figure_store


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
    return build_image_figures(ImagePipeline(plots=list(plots)), object(), run=TEST_RUN)


def test_no_image_binding_builds_nothing():
    assert build_image_figures(ImagePipeline(), object(), run=None) is None


def test_the_built_value_names_its_run():
    assert _build(Bars()).run == TEST_RUN


def test_an_image_binding_without_a_run_is_refused():
    """A run folder cannot be named without its pipeline digest (spec §1a)."""
    with pytest.raises(ValueError, match="run folder"):
        build_image_figures(ImagePipeline(plots=[Bars()]), object(), run=None)


def test_figure_run_for_reads_the_journal_or_takes_what_it_is_given():
    from types import SimpleNamespace

    from phenotypic.plotting._pipeline._store_figures import figure_run_for

    sha = "ef" * 32
    journal = {"applications": [{"pipeline": {"source_path": "p.json", "sha256": sha}}]}
    image = SimpleNamespace(_metadata=SimpleNamespace(provenance_journal=journal))
    assert figure_run_for(image, date="2026-09-22").run_id == "2026-09-22-efefefefefef"
    other = "01" * 32
    assert figure_run_for(image, date="2026-09-22", pipeline_sha256=other).pipeline_sha256 == other
    assert figure_run_for(object(), date="2026-09-22") is None


def test_a_run_that_cannot_be_named_writes_no_figures_and_says_so(tmp_path):
    """Spec §3: no figure error fails an image. An unnamed run folder -- no
    digest, a malformed date -- is one `.failures.jsonl` line, never a raise."""
    from phenotypic.plotting._pipeline._store_figures import (
        RUN_FOLDER_FAILURE,
        name_figure_run,
    )
    from phenotypic.sdk_ import plot_failures_jsonl_path

    bindings = ImagePipeline(plots=[Bars()]).get_plots()
    sha = "ef" * 32
    assert name_figure_run(bindings, object(), plots_base=tmp_path) is None
    assert name_figure_run(
        bindings, object(), date="2026-13-45", pipeline_sha256=sha, plots_base=tmp_path,
        dataset="ds", image_stem="plate",
    ) is None
    assert name_figure_run(bindings, object(), plots_base=None) is None
    lines = [
        json.loads(line)
        for line in plot_failures_jsonl_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    assert [line["binding_id"] for line in lines] == [RUN_FOLDER_FAILURE] * 2
    assert "no pipeline digest" in lines[0]["error"]
    assert (lines[1]["dataset"], lines[1]["image_stem"]) == ("ds", "plate")
    assert "YYYY-MM-DD" in lines[1]["error"]
    # A named run passes through; no image binding needs no run and records nothing.
    run = name_figure_run(bindings, object(), date="2026-09-22", pipeline_sha256=sha)
    assert run.run_id == "2026-09-22-efefefefefef"
    assert name_figure_run(ImagePipeline().get_plots(), object(), plots_base=tmp_path) is None
    assert len(plot_failures_jsonl_path(tmp_path).read_text(encoding="utf-8").splitlines()) == 2


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
    # Bare pages are each their own plot (spec D2): the digest is on the folder.
    where = {page.key: (page.directory, [f.filename for f in page.files]) for page in binding.pages}
    assert where["A b"] == ("A-b", ["A-b.plotly.json"])
    folder, [mpl_name] = where["a-b"]
    assert re.fullmatch(r"a-b-[0-9a-f]{8}", folder) and mpl_name == "a-b.png"
    assert "odd" not in where and "np" not in where
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


def test_an_empty_plot_output_is_a_failure_not_an_absence():
    class NoPages(BaseModel, PlotImage):
        def inspect(self, subject=None, *, for_save=False, **overrides):
            return PlotOutput(pages=())

    stored = _build(NoPages())
    assert stored.bindings == ()
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("NoPages", None, None)


def _with_metadata(metadata):
    """A good page, then a page carrying *metadata*."""

    class Meta(BaseModel, PlotImage):
        def inspect(self, subject=None, *, for_save=False, **overrides):
            import plotly.graph_objects as go

            return PlotOutput(pages=(
                PlotPage(key="good", figure=go.Figure(), metadata={"n": 1}),
                PlotPage(key="meta", figure=go.Figure(), metadata=metadata),
            ))

    return Meta()


_UNSTORABLE_METADATA = {
    "unsortable keys": {1: "x", "b": 2},
    "nan": {"v": float("nan")},
    "infinity": {"v": float("inf")},
}


@pytest.mark.parametrize("metadata", _UNSTORABLE_METADATA.values(), ids=_UNSTORABLE_METADATA)
def test_metadata_no_strict_writer_accepts_refuses_that_page(metadata):
    stored = _build(_with_metadata(metadata))
    assert [p.key for p in stored.bindings[0].pages] == ["good"]
    [failure] = stored.failed
    assert (failure.page, failure.format) == ("meta", None)


def test_metadata_is_stored_as_a_reader_reads_it_back():
    stored = _build(_with_metadata({2: (1, 2), 1: "x"}))
    meta = next(p for p in stored.bindings[0].pages if p.key == "meta")
    assert list(meta.metadata.items()) == [("1", "x"), ("2", [1, 2])]
    assert stored.failed == ()


@pytest.fixture(scope="module")
def plate():
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    return Image(load_synth_yeast_plate())


def _strict_json(text: str):
    def _refuse(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(text, parse_constant=_refuse)


@pytest.mark.parametrize("metadata", _UNSTORABLE_METADATA.values(), ids=_UNSTORABLE_METADATA)
def test_unstorable_metadata_still_publishes_the_store_in_every_mode(
    tmp_path, plate, metadata
):
    """Spec §5: "metadata not JSON-native ... the store still publishes (all
    modes, including the measure-mode root rewrite)" -- with a strict root."""
    import pandas as pd

    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._image_figures import read_image_figures_descriptor
    from phenotypic.sdk_._measurement_tables import replace_image_tables

    stored = _build(_with_metadata(metadata))
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=stored)
    _strict_json((store / "zarr.json").read_text(encoding="utf-8"))
    replace_image_tables(
        store,
        prepare_image_tables(pd.DataFrame({"Object_Label": [1]}), None),
        objmap_target=ngff_.objmap_path("rgb"),
        figures=stored,
    )
    root = _strict_json((store / "zarr.json").read_text(encoding="utf-8"))
    descriptor = root["attributes"]["phenotypic"]["figures"]
    assert descriptor == read_image_figures_descriptor(store)
    run = descriptor["runs"][TEST_RUN.run_id]
    [binding] = run["bindings"].values()
    assert [p["key"] for p in binding["pages"]] == ["good"]
    assert [(f["page"], f["format"]) for f in run["failed"]] == [("meta", None)]


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


# --------------------------------------------------------------------------
# §3a: a figure only apply() can draw is kept within its own run folder, or
# listed as unavailable -- never copied between run folders
# --------------------------------------------------------------------------


class ApplyState(BaseModel, PlotImage):
    """Draws while its apply-state exists; afterwards raises the signal.

    ``draw`` gives a labelled page with metadata plus a page that fails, so a
    keep has an entry, its files and a page-level failure to keep.
    """

    mode: str = "draw"

    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        if self.mode == "gone":
            raise FigureInputUnavailable("the as-shot pixels are gone")
        if self.mode == "explode":
            raise RuntimeError("the gate refused this frame")
        fig = Figure()
        fig.subplots().plot([0, 1])
        return PlotOutput(pages=(
            PlotPage(key="tiles", figure=fig, label="Tiles", metadata={"roi": 1}),
            PlotPage(key="odd", figure=object()),
        ))


class OtherClass(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise FigureInputUnavailable("not here")


def _files(store):
    files = {
        path.relative_to(store).as_posix(): path.read_bytes()
        for path in sorted((store / "figures").rglob("*")) if path.is_file()
    }
    assert files, f"no figure files under {store}"   # an empty == empty proves nothing
    return files


def _first_store(tmp_path, mode="draw", run=TEST_RUN):
    from tests.unit.plotting._store_fixtures import figure_store

    first = build_image_figures(
        ImagePipeline(plots=[ApplyState(mode=mode)]), object(), run=run
    )
    return first, figure_store(tmp_path / "first", first)


def _keep(store, *plots, run=TEST_RUN):
    return build_image_figures(
        ImagePipeline(plots=list(plots)), object(), run=run, keep_from=store
    )


def test_unavailable_with_nothing_to_keep_is_listed_not_failed():
    stored = _build(ApplyState(mode="gone"), Bars())
    assert [b.binding_id for b in stored.bindings] == ["Bars"]
    assert stored.failed == ()
    assert stored.unavailable == ("ApplyState",)


def test_a_binding_in_the_same_runs_folder_is_kept_byte_identical(tmp_path):
    from phenotypic.sdk_._image_figures import read_image_figures_descriptor

    from tests.unit.plotting._store_fixtures import figure_store

    first, store = _first_store(tmp_path)
    assert [(f.page, f.format) for f in first.failed] == [("odd", None)]
    kept = _keep(store, ApplyState(mode="gone"))
    assert kept == first
    again = figure_store(tmp_path / "again", kept)
    assert read_image_figures_descriptor(again) == read_image_figures_descriptor(store)
    assert _files(again) == _files(store)


def test_another_runs_folder_is_never_copied_from(tmp_path):
    from phenotypic.sdk_._image_figures import FigureRun

    earlier = FigureRun(date="2026-09-01", pipeline_sha256=TEST_RUN.pipeline_sha256)
    _first, store = _first_store(tmp_path, run=earlier)
    kept = _keep(store, ApplyState(mode="gone"))
    assert kept.bindings == () and kept.failed == ()
    assert kept.unavailable == ("ApplyState",)


def test_a_tampered_file_is_a_binding_failure_and_keeps_nothing(tmp_path):
    _first, store = _first_store(tmp_path)
    [png] = (store / "figures" / TEST_RUN.run_id / "ApplyState").rglob("*.png")
    png.write_bytes(png.read_bytes() + b"\0")
    kept = _keep(store, ApplyState(mode="gone"))
    assert kept.bindings == () and kept.unavailable == ()
    [failure] = kept.failed
    assert (failure.binding, failure.page, failure.format) == ("ApplyState", None, None)
    assert "does not match its sha256" in failure.error


def test_a_stored_binding_level_failure_is_kept_verbatim(tmp_path):
    first, store = _first_store(tmp_path, mode="explode")
    assert first.bindings == () and len(first.failed) == 1
    assert _keep(store, ApplyState(mode="gone")) == first


def test_a_folder_that_never_held_the_binding_lists_it_unavailable(tmp_path):
    from tests.unit.plotting._store_fixtures import figure_store

    store = figure_store(tmp_path, _build(Bars()))
    kept = _keep(store, ApplyState(mode="gone"), Bars())
    assert [b.binding_id for b in kept.bindings] == ["Bars"]
    assert kept.failed == () and kept.unavailable == ("ApplyState",)


def test_a_figure_stored_by_another_class_is_not_kept(tmp_path):
    from phenotypic.plotting._pipeline import PlotBinding

    _first, store = _first_store(tmp_path)
    kept = _keep(store, PlotBinding(id="ApplyState", plot=OtherClass()))
    assert kept.bindings == ()
    [failure] = kept.failed
    assert "was drawn by 'ApplyState', not OtherClass" in failure.error


def test_a_drawable_binding_is_redrawn_not_kept(tmp_path):
    first, store = _first_store(tmp_path)
    # A stored copy that would refuse to be kept: only a redraw can succeed.
    pngs = list((store / "figures" / TEST_RUN.run_id / "ApplyState").rglob("*.png"))
    assert pngs     # tampering nothing could not tell a redraw from a keep
    for png in pngs:
        png.write_bytes(b"tampered")
    rebuilt = _keep(store, ApplyState(mode="draw"), Bars())
    assert [b.binding_id for b in rebuilt.bindings] == ["ApplyState", "Bars"]
    assert rebuilt.bindings[0] == first.bindings[0]


# --------------------------------------------------------------------------
# Staged Stage 3: Stage 1's plots are kept, never drawn, and merged with
# Stage 3's own build into the one run folder
# --------------------------------------------------------------------------


def test_keep_image_figures_keeps_what_stage_one_wrote(tmp_path):
    from phenotypic.plotting._pipeline._store_figures import keep_image_figures

    first, store = _first_store(tmp_path)
    # Its inspect() would now raise: keeping must never ask it to draw.
    kept = keep_image_figures(
        store, ImagePipeline(plots=[ApplyState(mode="explode")]).get_plots(), run=TEST_RUN
    )
    assert kept == first


def test_keep_image_figures_lists_a_binding_stage_one_never_wrote(tmp_path):
    from phenotypic.plotting._pipeline._store_figures import keep_image_figures

    _first, store = _first_store(tmp_path)
    kept = keep_image_figures(
        store, ImagePipeline(plots=[ApplyState(), Bars()]).get_plots(), run=TEST_RUN
    )
    assert [b.binding_id for b in kept.bindings] == ["ApplyState"]
    assert kept.unavailable == ("Bars",)
    assert keep_image_figures(store, ImagePipeline().get_plots(), run=TEST_RUN) is None
    with pytest.raises(ValueError, match="run folder"):
        keep_image_figures(store, ImagePipeline(plots=[Bars()]).get_plots(), run=None)


def test_merge_joins_one_runs_parts_and_refuses_two_runs():
    from phenotypic.plotting._pipeline._store_figures import merge_stored_figures
    from phenotypic.sdk_._image_figures import FigureRun

    stage1 = _build(ApplyState(mode="gone"))
    stage3 = _build(Bars())
    merged = merge_stored_figures(stage1, None, stage3)
    assert merged.run == TEST_RUN
    assert [b.binding_id for b in merged.bindings] == ["Bars"]
    assert merged.unavailable == ("ApplyState",)
    assert merge_stored_figures(None, None) is None
    other = FigureRun(date="2026-10-01", pipeline_sha256=TEST_RUN.pipeline_sha256)
    elsewhere = build_image_figures(ImagePipeline(plots=[Bars()]), object(), run=other)
    with pytest.raises(ValueError, match="different runs"):
        merge_stored_figures(stage3, elsewhere)


# --------------------------------------------------------------------------
# Plot folders (spec 2026-09-30 §2): pages are built into
# `<binding>/<plot>/<file>`; a kept v1 page stays flat, a kept v2 page stays
# in its folder
# --------------------------------------------------------------------------


class PerRoiPages(BaseModel, PlotImage):
    """Two `tiles` pages and a bare `delta_e`, like the calibration overlay."""

    mode: str = "draw"

    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        if self.mode == "gone":
            raise FigureInputUnavailable("the as-shot pixels are gone")

        def fig(y):
            f = Figure()
            f.subplots().plot([0, y])
            return f

        return PlotOutput(pages=(
            PlotPage(key="roi_0", plot="tiles", figure=fig(1)),
            PlotPage(key="roi_1", plot="tiles", figure=fig(2)),
            PlotPage(key="delta_e", figure=fig(3)),
        ))


def test_plot_page_paths_groups_pages_by_plot():
    assert plot_page_paths([
        ("tiles", "roi_0", "roi_0"), ("tiles", "roi_1", "roi_1"), ("delta_e", "delta_e", "delta_e"),
    ]) == [("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e")]


def test_plot_folders_that_collide_get_distinct_names():
    """Review Focus 1: case-folded collisions get the stable digest suffix."""
    paths = plot_page_paths([("Tiles", "a", "a"), ("tiles", "a", "a")])
    folders = [folder for folder, _stem in paths]
    assert folders[0] == "Tiles"
    assert re.fullmatch(r"tiles-[0-9a-f]{8}", folders[1])
    assert [stem for _folder, stem in paths] == ["a", "a"]       # one per folder, no clash
    assert paths == plot_page_paths([("Tiles", "a", "a"), ("tiles", "a", "a")])  # stable
    # First appearance takes the clean name, whichever spelling comes first.
    reversed_folders = [folder for folder, _stem in plot_page_paths([
        ("tiles", "a", "a"), ("Tiles", "a", "a"),
    ])]
    assert reversed_folders[0] == "tiles"
    assert re.fullmatch(r"Tiles-[0-9a-f]{8}", reversed_folders[1])


def test_reserved_names_never_become_plot_folders():
    """Minor 1: a plot named like the group document or the manifest."""
    folders = [folder for folder, _stem in plot_page_paths([
        ("zarr.json", "a", "a"), ("Manifest.JSON", "b", "b"),
    ])]
    assert all(re.fullmatch(r"(zarr|Manifest)\.(json|JSON)-[0-9a-f]{8}", f) for f in folders)


def test_two_pages_of_one_plot_whose_stems_collide_get_distinct_files():
    """I2: the per-plot file pass, not only the folder pass, is collision-safe."""
    paths = plot_page_paths([("tiles", "A b", "A b"), ("tiles", "a-b", "a-b")])
    assert paths[0] == ("tiles", "A-b")
    assert paths[1][0] == "tiles" and re.fullmatch(r"a-b-[0-9a-f]{8}", paths[1][1])


def test_a_multi_plot_output_is_built_into_plot_folders():
    [binding] = _build(PerRoiPages()).bindings
    assert [(p.plot, p.directory, p.key, p.files[0].filename) for p in binding.pages] == [
        ("tiles", "tiles", "roi_0", "roi_0.png"),
        ("tiles", "tiles", "roi_1", "roi_1.png"),
        ("delta_e", "delta_e", "delta_e", "delta_e.png"),
    ]


def test_a_foldered_v2_binding_is_kept_byte_identical(tmp_path):
    first = _build(PerRoiPages())
    store = figure_store(tmp_path / "first", first)
    stored_pages = read_figure_run(store, TEST_RUN.run_id)["bindings"]["PerRoiPages"]["pages"]
    assert [p["files"][0]["path"].split("/")[3:] for p in stored_pages] == [
        ["tiles", "roi_0.png"], ["tiles", "roi_1.png"], ["delta_e", "delta_e.png"],
    ]   # really `<binding>/<plot>/<file>`, so the keep below reads plot folders
    kept = _keep(store, PerRoiPages(mode="gone"))
    assert kept == first
    assert _files(figure_store(tmp_path / "again", kept)) == _files(store)


def _flat_v1_store(tmp_path, *, plot_claimed):
    """A run folder an older PhenoTypic wrote: flat page, no `plot`, version 1."""
    flat = StoredFigures(TEST_RUN, (StoredFigureBinding(
        "PerRoiPages", "PerRoiPages", "PerRoiPages",
        (StoredFigurePage("tiles", None, "mpl", {},
                          (StoredFigureFile("png", "image/png", "tiles.png", b"\x89PNG v1"),)),),
    ),), ())
    store = figure_store(tmp_path / "v1", flat)
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    figures = root["attributes"]["phenotypic"]["figures"]
    figures["schema_version"] = 1
    [page] = figures["runs"][TEST_RUN.run_id]["bindings"]["PerRoiPages"]["pages"]
    if plot_claimed is None:
        page.pop("plot")                      # exactly as 0.19 wrote it
    else:
        page["plot"] = plot_claimed           # a v2 claim over a flat path
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    return store


def test_a_kept_flat_page_is_written_flat_with_a_null_plot(tmp_path):
    """I6 / P3: the null plot survives the write and the file stays flat."""
    kept = _keep(_flat_v1_store(tmp_path, plot_claimed=None), PerRoiPages(mode="gone"))
    again = figure_store(tmp_path / "again", kept)
    [page] = read_figure_run(again, TEST_RUN.run_id)["bindings"]["PerRoiPages"]["pages"]
    assert page["plot"] is None
    assert page["files"][0]["path"] == f"figures/{TEST_RUN.run_id}/PerRoiPages/tiles.png"


def test_a_binding_spread_over_two_binding_folders_is_refused(tmp_path):
    """I6: the single-binding-folder check still holds with plot folders."""
    store = figure_store(tmp_path / "s", _build(PerRoiPages()))
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    pages = root["attributes"]["phenotypic"]["figures"]["runs"][TEST_RUN.run_id]["bindings"]["PerRoiPages"]["pages"]
    moved = pages[2]["files"][0]
    assert len(moved["path"].split("/")) == 5    # a plot-folder page, not a flat one
    source = store / moved["path"]
    moved["path"] = moved["path"].replace("/PerRoiPages/", "/Elsewhere/")
    (store / moved["path"]).parent.mkdir(parents=True)
    source.rename(store / moved["path"])
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    kept = _keep(store, PerRoiPages(mode="gone"))
    assert kept.bindings == ()
    assert "one directory" in kept.failed[0].error


def test_a_page_failure_records_its_plot_through_the_descriptor(tmp_path):
    """I6 / Review Focus 5: the builder stamps `plot` on page failures."""
    stored = _build(HandBuiltPages())
    by_page = {f.page: f for f in stored.failed}
    assert by_page["odd"].plot == "odd" and by_page["np"].plot == "np"
    (tmp_path / "s").mkdir()
    fragment = write_image_figures(tmp_path / "s", stored)
    failed = fragment["figures"]["runs"][TEST_RUN.run_id]["failed"]
    assert {(f["page"], f["plot"]) for f in failed} >= {("odd", "odd"), ("np", "np")}


def test_a_flat_v1_binding_is_kept_flat(tmp_path):
    """Review Focus 2: an older PhenoTypic wrote this run's folder flat."""
    kept = _keep(_flat_v1_store(tmp_path, plot_claimed=None), PerRoiPages(mode="gone"))
    [page] = kept.bindings[0].pages
    assert (page.plot, page.directory, page.files[0].filename) == (None, None, "tiles.png")


def test_a_page_whose_plot_disagrees_with_its_path_is_refused(tmp_path):
    kept = _keep(_flat_v1_store(tmp_path, plot_claimed="tiles"), PerRoiPages(mode="gone"))
    assert kept.bindings == ()
    [failure] = kept.failed
    assert (failure.binding, failure.page) == ("PerRoiPages", None)
    assert "not laid out" in failure.error


class PlottedFailures(BaseModel, PlotImage):
    """Failing pages whose plot is not their key, like calibration's `tiles`."""

    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        def fig():
            f = Figure()
            f.subplots().plot([0, 1])
            return f

        return PlotOutput(pages=(
            PlotPage(key="roi_0", plot="tiles", figure=fig()),
            PlotPage(key="roi_1", plot="tiles", figure=object()),   # whole-page failure
            PlotPage(key="roi_2", plot="tiles", figure=fig()),      # per-format failure, below
        ))


def test_failures_of_plotted_pages_record_the_plot_not_the_key(tmp_path, monkeypatch):
    """I1: both failure sites stamp the page's plot, which T4 matches on."""
    from phenotypic.plotting._pipeline import _store_figures

    real = _store_figures.serialize_store_format

    def flaky(fmt, figure, *, binding_id, page_key):
        if page_key == "roi_2":
            raise OSError("disk full")
        return real(fmt, figure, binding_id=binding_id, page_key=page_key)

    monkeypatch.setattr(_store_figures, "serialize_store_format", flaky)
    stored = _build(PlottedFailures())
    assert {(f.page, f.format, f.plot) for f in stored.failed} == {
        ("roi_1", None, "tiles"), ("roi_2", "png", "tiles"),
    }
    (tmp_path / "s").mkdir()
    failed = write_image_figures(tmp_path / "s", stored)["figures"]["runs"][TEST_RUN.run_id]["failed"]
    assert {(f["page"], f["plot"]) for f in failed} == {("roi_1", "tiles"), ("roi_2", "tiles")}


class SpacedPlot(BaseModel, PlotImage):
    """A plot name that does not clean to itself: folder `Tile-overlay`."""

    mode: str = "draw"

    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        if self.mode == "gone":
            raise FigureInputUnavailable("gone")
        f = Figure()
        f.subplots().plot([0, 1])
        return PlotOutput(pages=(PlotPage(key="roi_0", plot="Tile overlay", figure=f),))


def test_a_kept_page_keeps_the_folder_its_path_names_not_its_plot_name(tmp_path):
    """I2: the kept folder comes from the stored path, never from `plot`."""
    first = _build(SpacedPlot())
    assert first.bindings[0].pages[0].directory == "Tile-overlay"
    store = figure_store(tmp_path / "first", first)
    kept = _keep(store, SpacedPlot(mode="gone"))
    assert kept == first
    assert _files(figure_store(tmp_path / "again", kept)) == _files(store)


def _tamper_pages(store, edit) -> None:
    """Apply *edit* to the stored `PerRoiPages` pages and write the root back."""
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    run = root["attributes"]["phenotypic"]["figures"]["runs"][TEST_RUN.run_id]
    edit(run["bindings"]["PerRoiPages"]["pages"])
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")


def _refused_as_not_laid_out(store) -> None:
    kept = _keep(store, PerRoiPages(mode="gone"))
    assert kept.bindings == ()
    [failure] = kept.failed
    assert (failure.binding, failure.page) == ("PerRoiPages", None)
    assert "not laid out" in failure.error


def test_a_page_with_no_plot_over_a_plot_folder_is_refused(tmp_path):
    """Minor 2: the other direction of the layout check -- no `plot`, foldered path."""
    store = figure_store(tmp_path / "s", _build(PerRoiPages()))

    def drop_plot(pages):
        assert len(pages[0]["files"][0]["path"].split("/")) == 5   # really foldered
        pages[0].pop("plot")

    _tamper_pages(store, drop_plot)
    _refused_as_not_laid_out(store)


def test_a_page_spread_over_two_plot_folders_is_refused(tmp_path):
    """Minor 2: one page whose files sit in two plot folders of one binding."""
    store = figure_store(tmp_path / "s", _build(PerRoiPages()))

    def spread(pages):
        # `delta_e/delta_e.png`: a real file with a valid sha256, in another folder.
        borrowed = dict(pages[2]["files"][0])
        assert borrowed["path"].split("/")[3] != pages[0]["files"][0]["path"].split("/")[3]
        pages[0]["files"].append(borrowed)

    _tamper_pages(store, spread)
    _refused_as_not_laid_out(store)
