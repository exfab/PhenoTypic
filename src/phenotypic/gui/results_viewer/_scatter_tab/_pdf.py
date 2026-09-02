"""Render each section to a PDF page and merge them into one document."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
from phenotypic.gui.results_viewer._scatter_tab._figure import (
    build_scatter_figure,
)
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

_CHROME_HINT = (
    "PDF export needs Chrome for kaleido. Install it once with "
    "`uv run plotly_get_chrome`, then retry."
)

#: Kaleido reads plotly's ``width``/``height`` as **pixels at 96 DPI**, so
#: this is what turns the inch-denominated page controls into the page
#: they name. Measured, not derived: a 1152x864 figure renders a MediaBox
#: of 864x648 pt, which is 12x9 in -- exactly 96 px per inch.
#:
#: Both earlier values were wrong in the same way, from the same habit of
#: carrying a number out of the context that measured it. The plan's 100
#: came from a PNG raster geometry and gives 16.67 in; 72 came from
#: reasoning that PDF points are 1/72 in -- true of the MediaBox, but the
#: MediaBox is downstream of a px->pt conversion kaleido does itself --
#: and gives 12 in. Only a rendered MediaBox settles it, which is why
#: ``test_the_rendered_page_measures_the_requested_inches`` exists.
_PIXELS_PER_INCH = 96

#: Stands in for the section value when no section grouping is chosen.
_NO_SECTION = object()


def export_sections_pdf(
    df: pl.DataFrame,
    spec: FigureSpec,
    sections: list[str],
    *,
    width_in: int = 16,
    height_in: int = 12,
) -> bytes:
    """Render one page per section and merge them.

    Every page is built with ``for_export=True`` so its traces are SVG
    ``go.Scatter``. WebGL export is not reliable across machines: with
    identical inputs it rendered as blank axes on one compute node and
    correctly on two others, silently and with exit code 0. The mechanism
    is unidentified. SVG costs nothing at one section per page, so the
    export path does not depend on a renderer that sometimes works.

    An empty ``sections`` renders a single page over the whole frame
    rather than a zero-page document. That is not defensiveness: an empty
    ``PdfWriter`` does not raise, it emits a valid 311-byte PDF holding
    nothing (measured), which is the same failure shape as the blank
    export this module exists to prevent.

    Args:
        df: The plottable frame across all sections.
        spec: The figure configuration.
        sections: Section values, in page order. Empty means "no section
            grouping" and yields one page.
        width_in: Page width in inches.
        height_in: Page height in inches.

    Returns:
        The merged PDF as bytes.

    Raises:
        RuntimeError: If kaleido cannot find Chrome.
    """
    import kaleido
    from pypdf import PdfWriter

    section_col = (
        spec.section_col
        if spec.section_col and spec.section_col in df.columns
        else None
    )
    # A section list the frame cannot honour would otherwise render N
    # identical pages under N different titles -- a document that states
    # something untrue rather than one that is merely empty.
    pages: list[object] = (
        list(sections) if section_col is not None else [_NO_SECTION]
    )

    writer = PdfWriter()
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        for n, value in enumerate(pages or [_NO_SECTION]):
            page_df = df
            title_text = ""
            if value is not _NO_SECTION and section_col is not None:
                page_df = df.filter(
                    pl.col(section_col).cast(pl.String) == str(value)
                )
                title_text = f"{section_col}: {value}"
            fig = build_scatter_figure(
                page_df,
                spec,
                plan_facets(page_df, spec),
                for_export=True,
            )
            fig.update_layout(
                title=dict(
                    text=title_text,
                    font=dict(size=spec.sizes["section"]),
                ),
                width=width_in * _PIXELS_PER_INCH,
                height=height_in * _PIXELS_PER_INCH,
            )
            out = Path(tmp) / f"page_{n:04d}.pdf"
            try:
                kaleido.write_fig_sync(fig, out)
            except RuntimeError as exc:  # kaleido's missing-Chrome error
                if "chrome" in str(exc).lower():
                    raise RuntimeError(_CHROME_HINT) from exc
                raise
            writer.append(str(out))
        # Written inside the block on purpose. `append` happens to slurp
        # the file eagerly, so this would survive outside it today -- but
        # that is a property of pypdf's reader, not of this code, and it
        # is not one a future reader should have to know.
        writer.write(buf)

    return buf.getvalue()
