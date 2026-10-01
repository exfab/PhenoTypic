"""Backend-neutral output normalization and Matplotlib publication."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import pytest
from dash import html

from phenotypic.abc_.plotting import (
    PlotOutput,
    PlotPage,
    canonical_group_key,
)
from phenotypic.plotting._pipeline import (
    FigureAdapter,
    publish_plot_output,
)


def test_matplotlib_dash_adaptation_closes_figure() -> None:
    figure = plt.figure()
    figure_number = figure.number

    component = FigureAdapter.to_dash_component(
        figure,
        class_name="preview",
        image_style={"maxWidth": "100%"},
        mpl_savefig_kwargs={"dpi": 72},
    )

    assert isinstance(component, html.Img)
    assert component.className == "preview"
    assert component.style == {"maxWidth": "100%"}
    assert component.src.startswith("data:image/png;base64,")
    assert not plt.fignum_exists(figure_number)


def test_matplotlib_dash_adaptation_closes_figure_on_failure(
    monkeypatch,
) -> None:
    figure = plt.figure()
    figure_number = figure.number

    def fail_savefig(*args, **kwargs) -> None:
        raise RuntimeError("raster failed")

    monkeypatch.setattr(figure, "savefig", fail_savefig)

    with pytest.raises(RuntimeError, match="raster failed"):
        FigureAdapter.to_dash_component(figure)

    assert not plt.fignum_exists(figure_number)


def test_duplicate_page_keys_are_rejected() -> None:
    fig = plt.figure()
    with pytest.raises(ValueError, match="duplicate page keys"):
        PlotOutput(
            pages=(PlotPage("same", fig), PlotPage("same", plt.figure()))
        )
    plt.close("all")


def test_matplotlib_pages_publish_with_collision_safe_names(tmp_path) -> None:
    first = plt.figure()
    second = plt.figure()
    output = PlotOutput(
        pages=(
            # One plot, so the names must be made unique within one folder:
            # "A-b" and "a-b" collide case-folded.
            PlotPage("first", first, label="A b", plot="same"),
            PlotPage("second", second, label="a-b", plot="same"),
        )
    )
    manifest = publish_plot_output(output, tmp_path, plot_id="demo")
    files = [entry["files"]["png"] for entry in manifest["pages"]]
    digest = hashlib.sha256(b"second").hexdigest()[:8]
    assert files == ["same/A-b.png", f"same/a-b-{digest}.png"]
    assert all((tmp_path / name).is_file() for name in files)
    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert persisted["class"] == "demo"
    assert {page["backend"] for page in persisted["pages"]} == {"matplotlib"}


def test_hash_suffix_is_rechecked_for_page_filename_collision(tmp_path) -> None:
    third_key = "third"
    reserved_suffix = hashlib.sha256(third_key.encode("utf-8")).hexdigest()[:8]
    output = PlotOutput(
        pages=(
            PlotPage("first", plt.figure(), label="A", plot="same"),
            PlotPage(
                "preempted-suffix",
                plt.figure(),
                label=f"A-{reserved_suffix}",
                plot="same",
            ),
            PlotPage(third_key, plt.figure(), label="A", plot="same"),
        )
    )

    manifest = publish_plot_output(output, tmp_path, plot_id="collision")

    files = [page["files"]["png"] for page in manifest["pages"]]
    assert len(files) == 3
    assert all(name.startswith("same/") for name in files)
    assert len({name.casefold() for name in files}) == 3
    assert all((tmp_path / name).is_file() for name in files)


def test_unsupported_page_fails_without_suppressing_sibling(tmp_path) -> None:
    output = PlotOutput(
        pages=(
            PlotPage("bad", object()),
            PlotPage("good", plt.figure()),
        )
    )
    manifest = publish_plot_output(output, tmp_path, plot_id="demo")
    assert [page["key"] for page in manifest["pages"]] == ["good"]
    assert (tmp_path / "good" / "good.png").is_file()
    assert [(f["key"], f["plot"]) for f in manifest["failed"]] == [("bad", "bad")]
    # A page that rendered nothing leaves no empty plot folder behind.
    assert not (tmp_path / "bad").exists()


def test_a_failed_plotted_page_records_its_plot_and_keeps_the_shared_folder(
    tmp_path,
) -> None:
    output = PlotOutput(pages=(
        PlotPage(key="good", plot="tiles", figure=plt.figure()),
        PlotPage(key="bad", plot="tiles", figure=object()),
    ))
    manifest = publish_plot_output(output, tmp_path, plot_id="demo")
    assert [(f["key"], f["plot"]) for f in manifest["failed"]] == [
        ("bad", "tiles")
    ]
    # The folder holds the good page, so the failed page does not remove it.
    assert sorted(p.name for p in (tmp_path / "tiles").iterdir()) == [
        "good.png"
    ]


def test_a_refused_plot_folder_closes_the_page_figure(tmp_path) -> None:
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    before = set(plt.get_fignums())
    # Entry, inside the lock, then the plot folder.
    calls = iter([True, True, False])
    with pytest.raises(PlotPublicationBlocked):
        publish_plot_output(
            PlotOutput(pages=(
                PlotPage(key="a", plot="tiles", figure=plt.figure()),
            )),
            tmp_path / "agg", plot_id="agg",
            publication_guard=lambda: next(calls, False),
        )
    assert set(plt.get_fignums()) == before
    assert not (tmp_path / "agg" / "tiles").exists()


def test_a_plot_folder_that_cannot_be_created_fails_only_its_page(
    tmp_path, monkeypatch
) -> None:
    """Like a render error, and like copy-out: a page failure, recorded,
    while its siblings publish -- not an aborted publication."""
    from pathlib import Path

    real_mkdir = Path.mkdir

    def _mkdir(self, *args, **kwargs):
        if self.name == "locked":
            raise PermissionError("read-only")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _mkdir)
    refused = plt.figure()
    output = PlotOutput(pages=(
        PlotPage(key="a", plot="locked", figure=refused),
        PlotPage(key="b", plot="open", figure=plt.figure()),
    ))
    manifest = publish_plot_output(output, tmp_path, plot_id="demo")
    assert [(p["key"], p["files"]) for p in manifest["pages"]] == [
        ("b", {"png": "open/b.png"})
    ]
    assert manifest["failed"] == [{
        "key": "a", "plot": "locked", "label": None,
        "error": "PermissionError: read-only",
    }]
    assert (tmp_path / "open" / "b.png").is_file()
    assert not (tmp_path / "locked").exists()
    assert not plt.fignum_exists(refused.number)
    [entry] = [
        json.loads(line)
        for line in (tmp_path / ".failures.jsonl").read_text().splitlines()
    ]
    assert entry["error"] == "PermissionError: read-only"


def test_concurrent_plot_publications_do_not_mix_generations(
    tmp_path, monkeypatch
) -> None:
    from phenotypic.plotting._pipeline import _writer

    class _FakeFigure:
        def __init__(self, generation: str) -> None:
            self.generation = generation

    def save_png(figure, path, **_kwargs) -> None:
        time.sleep(0.01)
        path.write_text(figure.generation, encoding="utf-8")

    monkeypatch.setattr(FigureAdapter, "save_png", save_png)
    # Classify _FakeFigure as matplotlib so this stays a PNG-only test. The
    # writer no longer calls FigureAdapter.backend_name -- _render_page asks
    # figure_backend_of, which would classify _FakeFigure as None and divert
    # every page to "failed", leaving manifest["pages"] empty and testing
    # nothing. Patching the writer's own binding works only because
    # figure_backend_of is imported at module level there.
    monkeypatch.setattr(_writer, "figure_backend_of", lambda _figure: "mpl")
    monkeypatch.setattr(FigureAdapter, "close", lambda _figure: None)

    def output(generation: str) -> PlotOutput:
        return PlotOutput(
            pages=tuple(
                PlotPage(
                    key=key,
                    figure=_FakeFigure(generation),
                    metadata={"generation": generation},
                )
                for key in ("first", "second")
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                publish_plot_output,
                output(generation),
                tmp_path,
                plot_id="concurrent",
            )
            for generation in ("one", "two")
        ]
        for future in futures:
            future.result()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    generations = {
        page["metadata"]["generation"] for page in manifest["pages"]
    }
    assert len(generations) == 1
    generation = generations.pop()
    assert {
        (tmp_path / page["files"]["png"]).read_text(encoding="utf-8")
        for page in manifest["pages"]
    } == {generation}


def test_canonical_group_keys_preserve_types_and_temporals() -> None:
    assert canonical_group_key([("value", 1)]) != canonical_group_key(
        [("value", "1")]
    )
    key = canonical_group_key(
        [
            ("float", 1.5),
            ("date", date(2026, 7, 16)),
            ("datetime", datetime(2026, 7, 16, 12, 30)),
            ("duration", timedelta(seconds=2, microseconds=5)),
        ]
    )
    assert "0x1.8000000000000p+0" in key
    assert "2026-07-16T12:30:00" in key
    assert "2000005000" in key


def test_canonical_group_key_rejects_nonfinite_float() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_group_key([("value", float("inf"))])


def test_a_plotly_page_publishes_html_without_chrome(tmp_path, monkeypatch) -> None:
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)

    output = PlotOutput(pages=(
        PlotPage(key="only", figure=go.Figure(go.Scatter(y=[1, 2])), label="Only"),
    ))
    manifest = publish_plot_output(output, tmp_path / "sym", plot_id="sym")

    page = manifest["pages"][0]
    assert page["files"] == {"html": "only/Only.html"}
    assert (tmp_path / "sym" / "only" / "Only.html").is_file()
    assert list((tmp_path / "sym").rglob("*.png")) == []


def test_the_html_references_the_hoisted_bundle(tmp_path, monkeypatch) -> None:
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)

    plots_base = tmp_path / "plots"
    output = PlotOutput(pages=(PlotPage(key="only", figure=go.Figure(), label="Only"),))
    publish_plot_output(
        output, plots_base / "sym", plot_id="sym", plots_base=plots_base
    )

    html = (plots_base / "sym" / "only" / "Only.html").read_text()
    assert 'src="../../plotly.min.js"' in html
    assert (plots_base / "plotly.min.js").is_file()
    # The 4.8 MB bundle must NOT be duplicated into the page or plot directory.
    assert list((plots_base / "sym").rglob("plotly.min.js")) == []


def test_a_matplotlib_page_publishes_png_only(tmp_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import publish_plot_output

    output = PlotOutput(pages=(PlotPage(key="only", figure=Figure(), label="Only"),))
    manifest = publish_plot_output(output, tmp_path / "m", plot_id="m")

    assert manifest["pages"][0]["files"] == {"png": "only/Only.png"}
    assert manifest["renderers"] == {"png": "available"}


def test_a_partial_rendering_failure_is_not_lost(tmp_path, monkeypatch) -> None:
    """S2: HTML succeeds, PNG fails -- the PNG failure must still be recorded.

    Without this the page lands in "pages" with one file and the other
    renderer's failure vanishes: best-effort silently meaning silent, one
    level below where §3 fixed it.
    """
    import json

    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, _writer, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)

    def _png_explodes(*args, **kwargs):
        raise RuntimeError("raster exploded")

    monkeypatch.setattr(_writer.FigureAdapter, "save_png", _png_explodes)

    plots_base = tmp_path / "plots"
    output = PlotOutput(pages=(PlotPage(key="only", figure=go.Figure(), label="Only"),))
    manifest = publish_plot_output(
        output, plots_base / "sym", plot_id="sym", plots_base=plots_base
    )

    page = manifest["pages"][0]
    assert page["files"] == {"html": "only/Only.html"}     # HTML still published
    # Exact, not a substring. A substring match is satisfied by
    # "RuntimeError: RuntimeError: raster exploded", which is what an earlier
    # draft wrote when _render_page returned pre-formatted strings that
    # record_plot_failure then formatted again.
    assert page["partial"] == ["RuntimeError: raster exploded"]

    record = plots_base / ".failures.jsonl"
    assert record.is_file(), "S3: the writer must write .failures.jsonl"
    entry = json.loads(record.read_text().splitlines()[0])
    assert entry["error"] == "RuntimeError: raster exploded"


def test_every_page_failing_yields_an_explanatory_manifest(tmp_path, monkeypatch) -> None:
    """A zero-page manifest must say why, not merely assert nothing."""
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _writer, publish_plot_output

    def _boom(*args, **kwargs):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(_writer.FigureAdapter, "save_html", _boom)
    monkeypatch.setattr(_writer.FigureAdapter, "save_png", _boom)

    output = PlotOutput(pages=(
        PlotPage(key="a", figure=go.Figure(), label="A"),
    ))
    manifest = publish_plot_output(output, tmp_path / "p", plot_id="p")

    assert manifest["schema_version"] == 3
    assert manifest["pages"] == []
    assert len(manifest["failed"]) == 1
    assert manifest["failed"][0]["key"] == "a"
    assert manifest["failed"][0]["plot"] == "a"
    # Its plot folder was created for it, and is not left behind empty.
    assert not (tmp_path / "p" / "a").exists()
    # Exact, for the same reason as the partial-failure test: a substring is
    # satisfied by a doubled "RuntimeError: RuntimeError: " prefix too. The
    # value is machine-independent -- errors[0] is always the HTML failure,
    # whether or not this machine also attempted a PNG.
    assert manifest["failed"][0]["error"] == "RuntimeError: renderer exploded"


def test_renderers_records_why_png_is_missing(tmp_path, monkeypatch) -> None:
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    output = PlotOutput(pages=(
        PlotPage(key="a", figure=go.Figure(), label="A"),
    ))
    manifest = publish_plot_output(output, tmp_path / "p", plot_id="p")

    assert manifest["renderers"]["html"] == "available"
    assert "chrome" in manifest["renderers"]["png"].lower()


def test_a_mixed_backend_directory_reports_png_as_partial(
    tmp_path, monkeypatch
) -> None:
    """One Plotly page and one matplotlib page in one directory, no Chrome.

    Neither "available" nor "unavailable" is true of this directory: the
    matplotlib page has a PNG and the Plotly page does not. ``renderers`` is
    the key a reader consults *instead of* walking every page, so it has to
    say so rather than pick whichever answer the last branch wrote.

    The per-page ``files`` map is correct in every case and is asserted here
    too, because it is what distinguishes a reporting bug from a publication
    bug.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)

    plots_base = tmp_path / "plots"
    output = PlotOutput(pages=(
        PlotPage(key="interactive", figure=go.Figure(), label="Interactive"),
        PlotPage(key="raster", figure=Figure(), label="Raster"),
    ))
    manifest = publish_plot_output(
        output, plots_base / "mixed", plot_id="mixed", plots_base=plots_base
    )

    assert manifest["renderers"] == {
        "html": "available",
        "png": "available: matplotlib only; chrome not found",
    }
    by_key = {page["key"]: page for page in manifest["pages"]}
    assert by_key["interactive"]["files"] == {"html": "interactive/Interactive.html"}
    assert by_key["raster"]["files"] == {"png": "raster/Raster.png"}
    assert manifest["failed"] == []


def test_pages_land_in_plot_folders_with_a_v3_manifest(tmp_path) -> None:
    def figure():
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        return fig

    output = PlotOutput(pages=(
        PlotPage(key="a", plot="tiles", label="Tile A", figure=figure()),
        PlotPage(key="b", plot="tiles", figure=figure()),
        PlotPage(key="chart", figure=figure()),
    ))
    manifest = publish_plot_output(
        output, tmp_path / "agg", plot_id="agg", plots_base=tmp_path
    )
    assert manifest["schema_version"] == 3
    assert [(p["plot"], p["key"], p["files"]["png"]) for p in manifest["pages"]] == [
        ("tiles", "a", "tiles/Tile-A.png"),
        ("tiles", "b", "tiles/b.png"),
        ("chart", "chart", "chart/chart.png"),
    ]
    for page in manifest["pages"]:
        assert (tmp_path / "agg" / page["files"]["png"]).is_file()


def test_a_rerun_removes_the_old_rendering_inside_its_plot_folder(
    tmp_path, monkeypatch
) -> None:
    import plotly.graph_objects as go

    from phenotypic.plotting._pipeline import _backends

    stale = tmp_path / "agg" / "tiles" / "a.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    monkeypatch.setattr(_backends, "chrome_available", lambda: False)  # HTML only

    publish_plot_output(
        PlotOutput(pages=(PlotPage(key="a", plot="tiles", figure=go.Figure()),)),
        tmp_path / "agg", plot_id="agg", plots_base=tmp_path,
    )

    assert not stale.exists()
    assert (tmp_path / "agg" / "tiles" / "a.html").is_file()


def test_the_guard_is_asked_before_a_plot_folder_is_created(tmp_path) -> None:
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    def output() -> PlotOutput:
        return PlotOutput(pages=(PlotPage(key="a", plot="tiles", figure=plt.figure()),))

    # Premise: an admitting guard does create the plot folder, so its absence
    # below is the refusal's doing.
    publish_plot_output(
        output(), tmp_path / "admitted", plot_id="agg", publication_guard=lambda: True
    )
    assert (tmp_path / "admitted" / "tiles" / "a.png").is_file()
    # Entry check, inside the lock, then the plot folder.
    calls = iter([True, True, False])
    with pytest.raises(PlotPublicationBlocked):
        publish_plot_output(
            output(), tmp_path / "agg", plot_id="agg",
            publication_guard=lambda: next(calls, False),
        )
    assert (tmp_path / "agg").is_dir()            # refused after the binding folder ...
    assert not (tmp_path / "agg" / "tiles").exists()  # ... and before the plot folder
