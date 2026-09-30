"""The matplotlib figure theme applies the DESIGN.md "Figures" defaults."""

from __future__ import annotations

import matplotlib as mpl
import pytest
from matplotlib import font_manager
from matplotlib.figure import Figure

from phenotypic.sdk_._palette import OKABE_ITO_PUBLISHED
from phenotypic.sdk_.viz.figures import (
    FIGURE_WIDTHS_MM,
    export_figure,
    figure_size_mm,
    phenotypic_mpl_context,
    phenotypic_rc,
)


def _growth_figure() -> Figure:
    """A small two-series growth plot, built outside any theme context."""
    fig = Figure(figsize=figure_size_mm("half", 50))
    ax = fig.add_subplot()
    ax.plot([0, 12, 24, 36], [0.1, 0.9, 2.1, 2.6], label="WT")
    ax.plot([0, 12, 24, 36], [0.1, 0.6, 1.5, 2.0], label="ura3Δ")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Colony area (mm²)")
    ax.legend()
    return fig


def test_typography_is_bundled_dejavu_sans_at_print_sizes() -> None:
    rc = phenotypic_rc()
    assert rc["font.sans-serif"] == ["DejaVu Sans"]
    assert rc["mathtext.fontset"] == "dejavusans"
    assert rc["xtick.labelsize"] == rc["ytick.labelsize"] == rc["legend.fontsize"] == 7.0
    assert rc["axes.labelsize"] == rc["legend.title_fontsize"] == 8.0


def test_theme_font_resolves_without_fallback() -> None:
    """DejaVu Sans ships with matplotlib, so no OS can substitute another face."""
    path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False
    )
    assert "DejaVuSans" in path


def test_chrome_is_minimal_and_white() -> None:
    rc = phenotypic_rc()
    assert rc["figure.facecolor"] == rc["axes.facecolor"] == rc["savefig.facecolor"] == "white"
    assert rc["axes.grid"] is False
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
    assert rc["legend.frameon"] is False


def test_series_follow_published_okabe_ito_order() -> None:
    colors = [entry["color"] for entry in phenotypic_rc()["axes.prop_cycle"]]
    assert colors == list(OKABE_ITO_PUBLISHED)
    assert colors[0] == "#000000"
    assert "#F0E442" not in colors


def test_export_settings_embed_text() -> None:
    rc = phenotypic_rc()
    assert rc["pdf.fonttype"] == rc["ps.fonttype"] == 42
    assert rc["svg.fonttype"] == "none"
    assert rc["svg.hashsalt"]


def test_context_does_not_leak_into_global_rcparams() -> None:
    before = mpl.rcParams["pdf.fonttype"]
    with phenotypic_mpl_context():
        assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["pdf.fonttype"] == before


@pytest.mark.parametrize(("preset", "expected_mm"), sorted(FIGURE_WIDTHS_MM.items()))
def test_figure_size_mm_converts_presets(preset: str, expected_mm: float) -> None:
    width_in, height_in = figure_size_mm(preset, 60)
    assert width_in * 25.4 == pytest.approx(expected_mm)
    assert height_in * 25.4 == pytest.approx(60)


def test_figure_size_mm_accepts_a_width_in_mm() -> None:
    assert figure_size_mm(100, 50)[0] * 25.4 == pytest.approx(100)


def test_figure_size_mm_allows_tall_multi_panel_figures() -> None:
    """No page-height cap: a dense multi-panel figure may be taller than A4."""
    assert figure_size_mm("full", 297)[1] * 25.4 == pytest.approx(297)


@pytest.mark.parametrize(
    ("width", "height"),
    [("quarter", 50), (0, 50), ("full", 0), (-10, 50)],
)
def test_figure_size_mm_rejects_bad_sizes(width, height) -> None:
    with pytest.raises(ValueError):
        figure_size_mm(width, height)


def test_export_pdf_embeds_truetype_even_when_built_outside_the_theme(tmp_path) -> None:
    """The font-type setting is read at save time, which export_figure owns."""
    pdf = export_figure(_growth_figure(), tmp_path / "growth.pdf").read_bytes()
    # Type 42 embeds the TrueType program (/FontFile2) under a CID font; the
    # matplotlib default would write /Type3 glyph procedures instead.
    assert b"/FontFile2" in pdf
    assert b"/Type3" not in pdf


def test_export_svg_keeps_text_as_text(tmp_path) -> None:
    svg = export_figure(_growth_figure(), tmp_path / "growth.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "Time (h)" in svg


@pytest.mark.parametrize("suffix", [".pdf", ".svg", ".png"])
def test_export_is_byte_reproducible(tmp_path, suffix: str) -> None:
    first = export_figure(_growth_figure(), tmp_path / f"a{suffix}").read_bytes()
    second = export_figure(_growth_figure(), tmp_path / f"b{suffix}").read_bytes()
    assert first == second


def test_export_keeps_the_built_size(tmp_path) -> None:
    """No tight bounding box: the saved page is the size the figure was built at."""
    from PIL import Image

    fig = _growth_figure()
    png = export_figure(fig, tmp_path / "growth.png", dpi=100)
    width_px, height_px = Image.open(png).size
    # matplotlib truncates the pixel size, so allow the fractional pixel.
    assert abs(width_px - fig.get_figwidth() * 100) < 1
    assert abs(height_px - fig.get_figheight() * 100) < 1


def test_export_rejects_unknown_format(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported export format"):
        export_figure(_growth_figure(), tmp_path / "growth.jpg")
