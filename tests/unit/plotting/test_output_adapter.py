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
            PlotPage("first", first, label="A/B"),
            PlotPage("second", second, label="A B"),
        )
    )
    manifest = publish_plot_output(output, tmp_path, plot_id="demo")
    files = [entry["file"] for entry in manifest["pages"]]
    assert len(files) == 2
    assert len({name.casefold() for name in files}) == 2
    assert all((tmp_path / name).exists() for name in files)
    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert persisted["class"] == "demo"
    assert {page["backend"] for page in persisted["pages"]} == {"matplotlib"}


def test_hash_suffix_is_rechecked_for_page_filename_collision(tmp_path) -> None:
    third_key = "third"
    reserved_suffix = hashlib.sha256(third_key.encode("utf-8")).hexdigest()[:8]
    output = PlotOutput(
        pages=(
            PlotPage("first", plt.figure(), label="A"),
            PlotPage(
                "preempted-suffix",
                plt.figure(),
                label=f"A-{reserved_suffix}",
            ),
            PlotPage(third_key, plt.figure(), label="A"),
        )
    )

    manifest = publish_plot_output(output, tmp_path, plot_id="collision")

    files = [page["file"] for page in manifest["pages"]]
    assert len(files) == 3
    assert len({name.casefold() for name in files}) == 3
    assert all((tmp_path / name).exists() for name in files)


def test_unsupported_page_fails_without_suppressing_sibling(tmp_path) -> None:
    output = PlotOutput(
        pages=(
            PlotPage("bad", object()),
            PlotPage("good", plt.figure()),
        )
    )
    manifest = publish_plot_output(output, tmp_path, plot_id="demo")
    assert [page["key"] for page in manifest["pages"]] == ["good"]


def test_concurrent_plot_publications_do_not_mix_generations(
    tmp_path, monkeypatch
) -> None:
    class _FakeFigure:
        def __init__(self, generation: str) -> None:
            self.generation = generation

    def save_png(figure, path, **_kwargs) -> None:
        time.sleep(0.01)
        path.write_text(figure.generation, encoding="utf-8")

    monkeypatch.setattr(FigureAdapter, "save_png", save_png)
    monkeypatch.setattr(FigureAdapter, "backend_name", lambda _figure: "fake")
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
        (tmp_path / page["file"]).read_text(encoding="utf-8")
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
