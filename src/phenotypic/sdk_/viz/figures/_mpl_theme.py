"""Matplotlib theme for PhenoTypic's static figures.

This module carries the defaults in ``DESIGN.md`` "Figures": figures sized for an
A4 page and set in DejaVu Sans, matplotlib's bundled face, so a figure renders the
same on macOS, Windows and Linux. The theme uses the 10 / 12 / 14 pt size ladder at
printed size, the published Okabe-Ito order, white backgrounds without gridlines,
TrueType font embedding and reproducible export metadata.

These are defaults, not constraints. A figure that needs something else sets it
after entering :func:`phenotypic_mpl_context`, and a contributor's explicit choice
always wins. Interactive Plotly charts in the GUI follow :mod:`._theme` instead.

matplotlib reads ``pdf.fonttype``, ``svg.fonttype`` and ``svg.hashsalt`` when a
file is saved, not when the figure is built, so a figure saved outside the theme is
exported with matplotlib's own defaults. :func:`export_figure` saves inside the
theme with the reproducibility metadata for the file's format.

``matplotlib`` is imported lazily inside the functions so importing this module
(and the ``phenotypic.sdk_.viz.figures`` package) stays cheap.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

from phenotypic.sdk_._palette import OKABE_ITO_PUBLISHED

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "FIGURE_WIDTHS_MM",
    "export_figure",
    "figure_size_mm",
    "phenotypic_mpl_context",
    "phenotypic_rc",
]

MM_PER_INCH: float = 25.4

#: Optional named figure widths in millimetres, for convenience only. ``full`` is
#: the text width of an A4 page (210 mm) with 1 in side margins, 210 - 2 x 25.4;
#: ``half`` fits two figures side by side on that width with a 5 mm gutter. A
#: figure may use any size its content needs, multi-panel figures included.
FIGURE_WIDTHS_MM: dict[str, float] = {"full": 159.2, "half": 77.1}

#: Point sizes at printed size (DESIGN.md "Figures").
_TICK_PT: float = 10.0
_LABEL_PT: float = 12.0

#: Fixed so SVG element ids, and with them the file bytes, repeat across runs.
_SVG_HASHSALT: str = "phenotypic"

#: Formats :func:`export_figure` writes, mapped to the metadata that keeps each
#: one reproducible.
_EXPORT_METADATA: dict[str, dict[str, None] | None] = {
    "pdf": {"CreationDate": None},
    "svg": {"Date": None},
    "png": None,
}


def phenotypic_rc() -> dict[str, Any]:
    """Return the PhenoTypic matplotlib ``rcParams`` dict (DESIGN.md "Figures").

    Suitable for ``matplotlib.rcParams.update(...)`` or
    ``matplotlib.rc_context(phenotypic_rc())``. All text is DejaVu Sans; tick
    labels and legend entries are 10 pt; axis, colorbar and legend titles,
    annotations and all other text are 12 pt. The Okabe-Ito published order anchors ``axes.prop_cycle``, so
    the first plotted series is black.

    Returns:
        A fresh ``dict`` of rcParam name -> value.
    """
    from cycler import cycler

    return {
        # Typography: matplotlib's bundled face, so output is identical on every OS.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": _LABEL_PT,
        "axes.labelsize": _LABEL_PT,
        "axes.titlesize": _LABEL_PT,
        "xtick.labelsize": _TICK_PT,
        "ytick.labelsize": _TICK_PT,
        "legend.fontsize": _TICK_PT,
        "legend.title_fontsize": _LABEL_PT,
        # Lines and ticks on the same scale as the type.
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "lines.linewidth": 1.75,
        "lines.markersize": 4.0,
        # Minimal chrome: white ground, no grid, bottom and left spines only.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        # Color.
        "axes.prop_cycle": cycler(color=list(OKABE_ITO_PUBLISHED)),
        "image.cmap": "cividis",
        # Export: TrueType text in PDF / PS, editable text in SVG, stable SVG ids.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": _SVG_HASHSALT,
    }


@contextmanager
def phenotypic_mpl_context() -> Iterator[None]:
    """Context manager applying :func:`phenotypic_rc` via ``rc_context``.

    Use around figure construction so the theme is scoped and never leaks into
    a caller's global ``rcParams``::

        with phenotypic_mpl_context():
            fig = analyzer.show()
            fig.savefig(buf, format="png")

    Yields:
        ``None``; the themed rcParams are active for the duration of the block.
    """
    import matplotlib as mpl

    with mpl.rc_context(phenotypic_rc()):
        yield


def figure_size_mm(
    width: Literal["full", "half"] | float, height_mm: float
) -> tuple[float, float]:
    """Return a matplotlib ``figsize`` in inches for a size given in millimetres.

    Args:
        width: A preset name from :data:`FIGURE_WIDTHS_MM` (``"full"`` or
            ``"half"``) or any width in millimetres.
        height_mm: Figure height in millimetres. No upper bound is enforced:
            a dense multi-panel figure may need more than one A4 page height,
            and it is the author's call.

    Returns:
        ``(width_in, height_in)`` for ``plt.figure(figsize=...)``.

    Raises:
        ValueError: If ``width`` names no preset, or either dimension is not
            positive.

    Examples:
        >>> from phenotypic.sdk_.viz.figures import figure_size_mm
        >>> w, h = figure_size_mm("half", 60)
        >>> round(w * 25.4, 1), round(h * 25.4, 1)
        (77.1, 60.0)
    """
    if isinstance(width, str):
        if width not in FIGURE_WIDTHS_MM:
            raise ValueError(
                f"unknown figure width {width!r}; expected one of "
                f"{sorted(FIGURE_WIDTHS_MM)} or a width in mm"
            )
        width_mm = FIGURE_WIDTHS_MM[width]
    else:
        width_mm = float(width)
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(f"figure size must be positive, got {width_mm} x {height_mm} mm")
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def export_figure(fig: Figure, path: str | Path, *, dpi: float = 300) -> Path:
    """Save ``fig`` for print with the theme's export settings applied.

    Saving happens inside :func:`phenotypic_mpl_context`, because matplotlib reads
    the font-embedding and SVG-id settings at save time. PDF output drops its
    creation date and SVG output its date, so two exports of the same figure are
    byte-identical. The figure keeps the size it was built at; no tight bounding
    box is applied, since that would change every point size.

    Args:
        fig: The matplotlib figure to save.
        path: Destination file; the suffix picks the format (``.pdf``, ``.svg``
            or ``.png``).
        dpi: Resolution for raster output and for raster images embedded in
            vector output.

    Returns:
        The destination path.

    Raises:
        ValueError: If the suffix is not a supported format.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from matplotlib.figure import Figure
        >>> from phenotypic.sdk_.viz.figures import export_figure, figure_size_mm
        >>> fig = Figure(figsize=figure_size_mm("half", 50))
        >>> _ = fig.add_subplot().plot([0, 12, 24], [0.1, 1.4, 2.6])
        >>> out = export_figure(fig, Path(tempfile.mkdtemp()) / "growth.pdf")
        >>> out.suffix
        '.pdf'
    """
    destination = Path(path)
    fmt = destination.suffix.lower().lstrip(".")
    if fmt not in _EXPORT_METADATA:
        raise ValueError(
            f"unsupported export format {destination.suffix!r}; "
            f"expected one of {sorted('.' + name for name in _EXPORT_METADATA)}"
        )
    with phenotypic_mpl_context():
        fig.savefig(destination, format=fmt, dpi=dpi, metadata=_EXPORT_METADATA[fmt])
    return destination
