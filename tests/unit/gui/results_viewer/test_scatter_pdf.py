"""Export must produce a PDF with visible ink, not merely a valid file."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("pypdf")

from phenotypic.gui.results_viewer._scatter_tab._figure import (  # noqa: E402
    CUSTOMDATA_COL,
)
from phenotypic.gui.results_viewer._scatter_tab._pdf import (  # noqa: E402
    export_sections_pdf,
)
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec  # noqa: E402


@pytest.fixture(scope="module")
def chrome_or_skip() -> None:
    """Skip, loudly, when kaleido has no browser to drive.

    Both rendering tests need Chrome. Without this they do not skip --
    they hard-fail with kaleido's RuntimeError, which reads as a broken
    export rather than a missing prerequisite.
    """
    import shutil

    if not (
        any(shutil.which(b) for b in ("google-chrome", "chromium", "chrome"))
        or Path.home().joinpath(".cache/kaleido").exists()
        or os.environ.get("BROWSER_PATH")
    ):
        pytest.skip("no Chrome for kaleido; run `uv run plotly_get_chrome`")


def _frame(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    return pl.DataFrame(
        {
            "x": rng.integers(0, 8, n).tolist(),
            "y": rng.normal(10, 2, n).tolist(),
            "s": ["A" if i % 2 else "B" for i in range(n)],
            CUSTOMDATA_COL: list(range(n)),
        }
    )


@pytest.fixture()
def recorded_pages(monkeypatch):
    """Stand in for kaleido, recording the figure handed to each page.

    Everything this fixture supports runs WITHOUT Chrome. That is the
    point: the trace-type substitution is a correctness requirement
    (kaleido renders Scattergl as blank axes, silently), and a guard that
    only runs where Chrome happens to be installed guards nothing on the
    machine this branch is built on.
    """
    import kaleido
    from pypdf import PdfWriter

    seen: list[tuple[object, Path]] = []

    def _fake_write_fig_sync(fig, path, **kwargs):
        seen.append((fig, Path(path)))
        one_page = PdfWriter()
        one_page.add_blank_page(width=72, height=72)
        with open(path, "wb") as fh:
            one_page.write(fh)

    monkeypatch.setattr(kaleido, "write_fig_sync", _fake_write_fig_sync)
    return seen


def test_the_export_path_never_hands_kaleido_a_webgl_trace(recorded_pages) -> None:
    """The regression pin for the silent-blank-export failure, Chrome-free.

    kaleido renders ``Scattergl`` as blank axes with exit code 0 and no
    warning: 624 non-white pixels against 46,886 for SVG. Substituting
    ``go.Scatter`` at the export boundary is therefore a correctness
    requirement, and removing it produces a clean, well-formed, entirely
    empty PDF that every structural assertion passes.
    """
    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    export_sections_pdf(df, spec, ["A", "B"])

    assert recorded_pages, "kaleido was never called"
    for fig, _ in recorded_pages:
        assert fig.data, "a page was rendered with no traces at all"
        assert all(t.type == "scatter" for t in fig.data), (
            f"page carries webgl traces: {[t.type for t in fig.data]}"
        )


def test_each_page_carries_only_its_own_section(recorded_pages) -> None:
    """One page per section means one section's rows per page.

    Without this, an export that ignored ``section_col`` and drew the
    whole frame on every page would still produce the right page count.
    """
    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    export_sections_pdf(df, spec, ["A", "B"])

    assert len(recorded_pages) == 2
    expected = {
        value: set(
            df.filter(pl.col("s") == value)[CUSTOMDATA_COL].to_list()
        )
        for value in ("A", "B")
    }
    for (fig, _), value in zip(recorded_pages, ("A", "B")):
        drawn = {int(cd[0]) for t in fig.data for cd in t.customdata}
        assert drawn == expected[value], f"page {value} drew {len(drawn)} rows"


def test_the_page_is_the_requested_size_in_inches(recorded_pages) -> None:
    """``width_in``/``height_in`` are inches, so the conversion is 96.

    Kaleido reads plotly's ``width``/``height`` as pixels at 96 DPI, so
    16 in is 1536 px. The literal 96 is spelled here rather than imported
    from ``_pdf`` on purpose: importing the constant would make this test
    agree with whatever the module happens to say, which is exactly how
    ``* 72`` passed here while rendering a 12-inch page.

    This pins the INTENT, not the outcome: it reads the figure's own
    ``layout.width``, so it runs without Chrome.
    ``test_the_rendered_page_measures_the_requested_inches`` is the half
    that measures the artifact, and it is what caught ``* 72``.
    """
    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    export_sections_pdf(df, spec, ["A"], width_in=16, height_in=12)

    fig, _ = recorded_pages[0]
    assert (fig.layout.width, fig.layout.height) == (16 * 96, 12 * 96)


def test_an_empty_section_list_still_renders_one_page(recorded_pages) -> None:
    """The zero-page guard, exercised on the path that reaches it.

    This must configure a section column AND pass no sections. An earlier
    version of this test passed a spec with no ``section_col`` at all,
    which takes a different branch entirely -- so removing the guard
    changed nothing and the test reported coverage it did not have. Found
    by mutation, not by reading.

    Measured: an empty ``PdfWriter`` does not raise. It emits a valid
    311-byte, 0-page PDF -- the same shape of failure as the blank export
    this module exists to prevent.
    """
    from pypdf import PdfReader

    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    out = export_sections_pdf(df, spec, [])

    assert len(recorded_pages) == 1, "exactly one page, not zero and not two"
    assert PdfReader(io.BytesIO(out)).get_num_pages() == 1

    # And that page carries the data. A one-page document drawn from an
    # empty frame would satisfy the count and still be blank.
    fig, _ = recorded_pages[0]
    drawn = {int(cd[0]) for t in fig.data for cd in t.customdata}
    assert drawn == set(df[CUSTOMDATA_COL].to_list())


def test_no_section_grouping_renders_one_page(recorded_pages) -> None:
    """"Section: none" is a legitimate configuration, not an error."""
    from pypdf import PdfReader

    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y")
    out = export_sections_pdf(df, spec, [])

    assert len(recorded_pages) == 1
    assert PdfReader(io.BytesIO(out)).get_num_pages() == 1


def test_a_missing_chrome_is_reported_not_raised(monkeypatch) -> None:
    """No Chrome must degrade to an actionable message, never a crash.

    Runs WITHOUT Chrome and without skipping, which is the point: every
    other rendering test here skips when kaleido has no browser, so the
    one behaviour a user without Chrome actually meets was the one
    behaviour nothing pinned. It already worked; this keeps it working.

    kaleido's error is recognised by its text, so this also fails if that
    text stops containing "chrome" -- the export would then surface
    kaleido's raw message instead of the install hint. Still a
    RuntimeError, so still caught by the callback, but no longer telling
    the user what to do about it.
    """
    import kaleido

    def _no_chrome(*_args, **_kwargs):
        raise RuntimeError("Chrome not found. Please install Chrome.")

    monkeypatch.setattr(kaleido, "write_fig_sync", _no_chrome)

    with pytest.raises(RuntimeError) as excinfo:
        export_sections_pdf(
            _frame(),
            FigureSpec(x_col="x", y_col="y", section_col="s"),
            ["A", "B"],
        )
    assert "plotly_get_chrome" in str(excinfo.value), (
        "the hint must name the command that fixes it"
    )


def test_one_page_is_written_per_section(chrome_or_skip) -> None:
    from pypdf import PdfReader

    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    out = export_sections_pdf(df, spec, ["A", "B"])
    assert PdfReader(io.BytesIO(out)).get_num_pages() == 2


def test_the_rendered_page_measures_the_requested_inches(chrome_or_skip) -> None:
    """The outcome half of the page-size pin.

    Asserting on ``fig.layout.width`` only says what we asked plotly for.
    The MediaBox is what a reader actually gets, and it is the only thing
    that settles the px-per-inch rate the conversion rests on. It has
    settled it twice against a wrong answer: the plan's ``* 100`` gives
    16.67 in and our ``* 72`` gave 12.00, both from carrying a number out
    of the context that measured it. Kaleido reads plotly's width as
    pixels at 96 DPI, so ``16 in`` is 1536 px and lands as 1152 pt.

    A PDF point IS 1/72 in -- that part of the ``* 72`` reasoning was
    never wrong. What it missed is that the MediaBox is downstream of a
    px->pt conversion kaleido performs itself.
    """
    from pypdf import PdfReader

    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    out = export_sections_pdf(df, spec, ["A"], width_in=16, height_in=12)

    box = PdfReader(io.BytesIO(out)).pages[0].mediabox
    assert (round(float(box.width)), round(float(box.height))) == (
        16 * 72,
        12 * 72,
    ), f"page is {float(box.width) / 72:.2f} x {float(box.height) / 72:.2f} in"


def test_the_exported_page_contains_ink_not_just_axes(chrome_or_skip, tmp_path) -> None:
    """The regression pin for the silent-blank-export failure.

    NOT marked ``slow``. ``addopts`` carries ``-m 'not slow'``, so a slow
    marker here would deselect the only defence against a blank export on
    every default run -- a green suite over 23 empty pages.

    Measured against a CONTROL rather than an absolute floor, and counting
    any ink rather than dark ink. Both corrections come from the same
    mistake, and it is the mistake the spec's own evidence was free of:
    spec 11.1's 289-vs-36,608 separation was measured on a figure with
    opaque markers, and this figure's markers carry ``marker_opacity=0.5``
    -- 50% navy over white is luminance ~149, so ``gray < 128`` sees none
    of them. The first run of this test on a real browser reported 196
    dark pixels for a page that was in fact drawn correctly.

    A control page built from an empty frame, through the same builder and
    the same layout, isolates exactly the marker layer. That is also the
    shape of the spec's argument -- 624 non-white for gl equalling the
    count for a figure with *zero* traces -- so the test now measures the
    thing the evidence measured instead of approximating it.
    """
    import kaleido
    from PIL import Image as PILImage

    from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
    from phenotypic.gui.results_viewer._scatter_tab._figure import (
        build_scatter_figure,
    )

    rng = np.random.default_rng(2)
    n = 500
    df = pl.DataFrame(
        {
            # Continuous, unlike `_frame`'s eight integer x values, so 500
            # markers spread over the panel instead of stacking into
            # stripes that mostly overlap.
            "x": rng.normal(0.0, 1.0, n).tolist(),
            "y": rng.normal(10.0, 2.0, n).tolist(),
            CUSTOMDATA_COL: list(range(n)),
        }
    )
    spec = FigureSpec(x_col="x", y_col="y")

    def _ink(frame: pl.DataFrame, name: str) -> int:
        fig = build_scatter_figure(
            frame, spec, plan_facets(df, spec), for_export=True
        )
        fig.update_layout(
            width=800,
            height=600,
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
        )
        png = tmp_path / name
        kaleido.write_fig_sync(fig, png)  # format follows the extension
        gray = np.asarray(PILImage.open(png).convert("L"))
        return int((gray < 250).sum())

    drawn = _ink(df, "page.png")
    blank = _ink(df.clear(), "control.png")

    # CALIBRATION -- read this before changing the fixture. A measured
    # sweep of `gray < 128` (dark pixels only) on this renderer:
    #
    #   n_points  marker_size  opacity   dark px
    #         60            6      0.5       129
    #         60           12      1.0     6,231
    #      1,000            6      0.5     7,800
    #      5,000            6      0.5    21,792
    #      5,000            6      1.0    28,153   <- spec 11.1's figure
    #
    # Spec 11.1's threshold of 2,000 was calibrated on the last row. The
    # top row is what this test used to draw, which is why it called a
    # correctly rendered page blank.
    #
    # The control is what makes that table advisory rather than binding:
    # the assertion is a DIFFERENCE against the same figure with no data,
    # so it moves with the fixture instead of having to be re-derived
    # whenever the point count, marker size or opacity changes. 500
    # markers at size 6 cover several thousand pixels even allowing for
    # overlap, and `< 250` counts the semi-transparent ones the dark
    # threshold misses -- so 5,000 sits far from both ends without
    # needing a 5,000-point render in every suite run.
    assert drawn - blank > 5000, (
        f"exported page has no ink beyond axes: {drawn} px against a "
        f"{blank} px control"
    )


def test_a4_lands_within_a_point_of_its_non_integer_inches(
    chrome_or_skip,
) -> None:
    """A4 landscape is 11.69 x 8.27 in, the first fraction this path sees.

    Every earlier page size was whole inches, so the inch -> px -> pt
    chain had only ever been exercised where each step lands on an
    integer. A4 does not: 11.69 in is 1122.24 px, and kaleido's own
    px -> pt conversion has to carry the fraction through.

    **Measured, and the tolerance is honest about what it rests on.**
    Asking for 11.69 x 8.27 in yields a MediaBox of 841.92 x 594.96 pt,
    which is 11.6933 x 8.2633 in -- errors of 0.24 and 0.48 pt. The two
    whole-inch presets land exact (1152.00 x 864.00 and 792.00 x 612.00).

    A half-point bound derived from "the MediaBox is written in points"
    would pass here at 0.48 with almost nothing to spare, and it does not
    predict 0.48 anyway: rounding the pixel dimension to an integer first
    caps the error at 0.375 pt, and the observed value exceeds that. The
    quantization kaleido and Chrome perform between the two is not
    something this test derives, so the bound is **1 pt** -- twice the
    largest error measured, and still an order of magnitude tighter than
    any page-size mistake this guards. Under 1 pt the page is correct to
    0.014 in; a wrong px-per-inch rate is off by inches, which is what
    caught ``* 100`` and ``* 72`` before it.
    """
    from pypdf import PdfReader

    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    out = export_sections_pdf(df, spec, ["A"], width_in=11.69, height_in=8.27)

    box = PdfReader(io.BytesIO(out)).pages[0].mediabox
    assert float(box.width) == pytest.approx(11.69 * 72, abs=1.0), (
        f"page is {float(box.width) / 72:.4f} in wide"
    )
    assert float(box.height) == pytest.approx(8.27 * 72, abs=1.0), (
        f"page is {float(box.height) / 72:.4f} in tall"
    )
