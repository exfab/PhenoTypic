"""Render the sample figures shown in ``figure-typography.html``.

Each decision axis on the chooser page (font family, size ladder, series
colours, plot chrome, panel labels) maps to one keyword below. The script
renders every combination the page can display into ``samples/`` as SVG with
text converted to paths, so the browser shows exactly the glyphs matplotlib
drew rather than substituting a local font.

All data are synthetic and fixed-seed. The colony crop comes from
``load_synth_yeast_plate()``; its scale bar is labelled in pixels because the
synthetic plate carries no physical calibration.

Usage::

    uv run python docs/superpowers/artifacts/2026-09-29-figure-typography/render_samples.py \
        --nunito-dir <folder holding NunitoSans-{Regular,SemiBold,Bold,Italic}.ttf>
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("svg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "samples"

MM_PER_INCH = 25.4

#: Full text width of an A4 page (210 mm) with Word's default 1 in side margins:
#: 210 - 2 x 25.4 = 159.2 mm. The chooser page states this assumption.
FULL_WIDTH_MM = 159.2
FIGURE_HEIGHT_MM = 104.0

FONTS = {
    "dejavu": {
        "label": "DejaVu Sans",
        "rc": {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
               "mathtext.fontset": "dejavusans"},
        "tick_family": None,
    },
    "dejavu-mono": {
        "label": "DejaVu Sans + DejaVu Sans Mono ticks",
        "rc": {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
               "font.monospace": ["DejaVu Sans Mono"], "mathtext.fontset": "dejavusans"},
        "tick_family": "monospace",
    },
    "stix": {
        "label": "STIX (serif)",
        "rc": {"font.family": "serif", "font.serif": ["STIXGeneral"],
               "mathtext.fontset": "stix"},
        "tick_family": None,
    },
    "nunito": {
        "label": "Nunito Sans (bundled TTF)",
        "rc": {"font.family": "sans-serif", "font.sans-serif": ["Nunito Sans"],
               "mathtext.fontset": "custom", "mathtext.rm": "Nunito Sans",
               "mathtext.it": "Nunito Sans:italic", "mathtext.bf": "Nunito Sans:bold",
               "mathtext.sf": "Nunito Sans", "mathtext.cal": "Nunito Sans"},
        "tick_family": None,
    },
}

#: Point sizes at final printed size. ``body`` is axis labels and legend text,
#: ``small`` is tick labels and annotations, ``panel`` is the panel letter.
LADDERS = {
    "compact": {"body": 7.0, "small": 6.0, "panel": 8.0, "axes_lw": 0.5, "line_lw": 1.0,
                "marker": 2.5, "tick_len": 2.5},
    "balanced": {"body": 8.0, "small": 7.0, "panel": 10.0, "axes_lw": 0.6, "line_lw": 1.25,
                 "marker": 3.0, "tick_len": 3.0},
    "generous": {"body": 9.0, "small": 8.0, "panel": 11.0, "axes_lw": 0.75, "line_lw": 1.5,
                 "marker": 3.5, "tick_len": 3.5},
}

PALETTES = {
    # Current DESIGN.md order: brand navy stands in for Okabe-Ito black.
    "house": ["#003660", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#CC79A7"],
    # Okabe-Ito in the order Wong (2011) prints it, yellow (#F0E442) skipped because
    # it fails as a line or point colour on white.
    "canonical": ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00",
                  "#CC79A7"],
}

CHROME = {
    "minimal": {"figure_bg": "#FFFFFF", "grid": False, "titles": False},
    "guides": {"figure_bg": "#FFFFFF", "grid": True, "titles": True},
    "current": {"figure_bg": "#FBFEF8", "grid": True, "titles": True, "navy_titles": True},
}

PANEL_STYLES = {
    "lower": lambda i: "abcd"[i],
    "upper": lambda i: "ABCD"[i],
    "paren": lambda i: f"({'abcd'[i]})",
}

#: Yeast nomenclature: deletion alleles are lowercase italic; "WT" stays roman.
STRAINS = ["WT", "ura3Δ", "his3Δ", "leu2Δ"]
ITALIC = [False, True, True, True]


def _register_nunito(nunito_dir: Path | None) -> bool:
    """Add the Nunito Sans TTFs to matplotlib's font manager if present.

    Args:
        nunito_dir: Folder holding the four static Nunito Sans TTFs, or ``None``.

    Returns:
        ``True`` when all four faces were registered.
    """
    if nunito_dir is None:
        return False
    faces = ["Regular", "SemiBold", "Bold", "Italic"]
    paths = [nunito_dir / f"NunitoSans-{face}.ttf" for face in faces]
    if not all(path.exists() for path in paths):
        return False
    for path in paths:
        font_manager.fontManager.addfont(str(path))
    return True


def _synthetic_data(seed: int = 7) -> dict[str, np.ndarray]:
    """Build fixed-seed growth, endpoint and plate-map data.

    Args:
        seed: Seed for :func:`numpy.random.default_rng`.

    Returns:
        Arrays for each panel.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(0, 49, 3.0)
    carrying = np.array([3.2, 2.6, 2.9, 2.1])
    rate = np.array([0.22, 0.17, 0.20, 0.14])
    lag = np.array([6.0, 9.0, 7.0, 11.0])
    replicates = 6
    curves = []
    for k, r, lam in zip(carrying, rate, lag):
        base = k / (1 + np.exp(-r * (hours - lam - 10)))
        noise = rng.normal(0, 0.06 * k, size=(replicates, hours.size))
        curves.append(np.clip(base + noise, 0, None))
    endpoint = [c[:, -1] + rng.normal(0, 0.08, replicates) for c in curves]
    rows, cols = 16, 24
    yy, xx = np.mgrid[0:rows, 0:cols]
    edge = np.minimum.reduce([yy, xx, rows - 1 - yy, cols - 1 - xx])
    plate = 2.4 + 0.35 * (edge == 0) + 0.15 * (edge == 1) + rng.normal(0, 0.12, (rows, cols))
    return {"hours": hours, "curves": np.array(curves), "endpoint": np.array(endpoint),
            "plate": plate}


def _colony_crop() -> np.ndarray:
    """Return a grayscale crop of the synthetic yeast plate.

    Returns:
        A 2-D float array in ``[0, 1]``.
    """
    from phenotypic.data import load_synth_yeast_plate

    gray = load_synth_yeast_plate().gray[:]
    crop = gray[150:390, 200:520]
    return (crop - crop.min()) / max(float(np.ptp(crop)), 1e-9)


def _rc_for(font: str, ladder: str, chrome: str) -> dict:
    """Assemble rcParams for one combination.

    Args:
        font: Key into :data:`FONTS`.
        ladder: Key into :data:`LADDERS`.
        chrome: Key into :data:`CHROME`.

    Returns:
        An rcParams mapping.
    """
    lad = LADDERS[ladder]
    chr_ = CHROME[chrome]
    rc = {
        "svg.fonttype": "path",
        "font.size": lad["body"],
        "axes.labelsize": lad["body"],
        "axes.titlesize": lad["body"],
        "legend.fontsize": lad["small"],
        "legend.title_fontsize": lad["small"],
        "xtick.labelsize": lad["small"],
        "ytick.labelsize": lad["small"],
        "axes.linewidth": lad["axes_lw"],
        "xtick.major.width": lad["axes_lw"],
        "ytick.major.width": lad["axes_lw"],
        "xtick.major.size": lad["tick_len"],
        "ytick.major.size": lad["tick_len"],
        "lines.linewidth": lad["line_lw"],
        "lines.markersize": lad["marker"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": chr_["grid"],
        "axes.grid.axis": "y" if chrome == "guides" else "both",
        "grid.color": "#E8ECF2",
        "grid.linewidth": 0.5,
        "figure.facecolor": chr_["figure_bg"],
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#333333" if chrome != "current" else "#DDE3ED",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333" if chrome != "current" else "#8892A4",
        "ytick.color": "#333333" if chrome != "current" else "#8892A4",
        "axes.titlecolor": "#003660" if chr_.get("navy_titles") else "#222222",
        "legend.frameon": False,
        "savefig.facecolor": chr_["figure_bg"],
    }
    rc.update(FONTS[font]["rc"])
    return rc


def render_figure(font: str, ladder: str, palette: str, chrome: str, panel: str,
                  data: dict, crop: np.ndarray, path: Path, *,
                  legend_outside: bool = False) -> None:
    """Render one 2 x 2 sample figure to ``path``.

    Args:
        font: Key into :data:`FONTS`.
        ladder: Key into :data:`LADDERS`.
        palette: Key into :data:`PALETTES`.
        chrome: Key into :data:`CHROME`.
        panel: Key into :data:`PANEL_STYLES`.
        data: Output of :func:`_synthetic_data`.
        crop: Output of :func:`_colony_crop`.
        path: Destination ``.svg`` path.
        legend_outside: Place the growth legend to the right of its panel, so
            it never covers data however large the text is.
    """
    lad = LADDERS[ladder]
    colors = PALETTES[palette]
    titles = CHROME[chrome]["titles"]
    tick_family = FONTS[font]["tick_family"]
    size = (FULL_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
    with plt.rc_context(_rc_for(font, ladder, chrome)):
        fig, axes = plt.subplots(2, 2, figsize=size, layout="constrained")
        ax_growth, ax_end, ax_plate, ax_img = axes.ravel()

        hours = data["hours"]
        for i, (curve, label) in enumerate(zip(data["curves"], STRAINS)):
            mean, sd = curve.mean(0), curve.std(0, ddof=1)
            ax_growth.plot(hours, mean, color=colors[i], label=label)
            ax_growth.fill_between(hours, mean - sd, mean + sd, color=colors[i], alpha=0.15,
                                   linewidth=0)
        ax_growth.set_xlabel("Time (h)")
        ax_growth.set_ylabel("Colony area (mm$^2$)")
        ax_growth.set_xlim(0, 48)
        ax_growth.set_xticks(range(0, 49, 12))
        if legend_outside:
            ax_growth.legend(title="S. cerevisiae", loc="upper left",
                             bbox_to_anchor=(1.02, 1.0), handlelength=1.4, borderaxespad=0)
        else:
            ax_growth.legend(title="S. cerevisiae", loc="best", handlelength=1.4)
        legend = ax_growth.get_legend()
        legend.get_title().set_fontstyle("italic")
        for text, italic in zip(legend.get_texts(), ITALIC):
            text.set_fontstyle("italic" if italic else "normal")

        rng = np.random.default_rng(3)
        for i, values in enumerate(data["endpoint"]):
            x = i + rng.uniform(-0.12, 0.12, values.size)
            ax_end.scatter(x, values, s=lad["marker"] ** 2, color=colors[i], zorder=3,
                           linewidths=0)
            ax_end.hlines(values.mean(), i - 0.25, i + 0.25, color="#222222",
                          linewidth=lad["line_lw"], zorder=4)
        ax_end.set_xticks(range(len(STRAINS)), STRAINS)
        for tick, italic in zip(ax_end.get_xticklabels(), ITALIC):
            tick.set_fontstyle("italic" if italic else "normal")
        ax_end.set_xlim(-0.6, len(STRAINS) - 0.4)
        ax_end.set_ylabel("Area at 48 h (mm$^2$)")
        ax_end.grid(False, axis="x")

        im = ax_plate.imshow(data["plate"], cmap="cividis", aspect="equal",
                             interpolation="nearest")
        ax_plate.set_xticks([0, 11, 23], ["1", "12", "24"])
        ax_plate.set_yticks([0, 7, 15], ["A", "H", "P"])
        ax_plate.tick_params(length=0)
        ax_plate.grid(False)
        for spine in ax_plate.spines.values():
            spine.set_visible(False)
        ax_plate.set_xlabel("Column")
        ax_plate.set_ylabel("Row")
        bar = fig.colorbar(im, ax=ax_plate, shrink=0.9, pad=0.02)
        bar.set_label("Colony area (mm$^2$)")
        bar.outline.set_linewidth(lad["axes_lw"])

        ax_img.imshow(crop, cmap="gray", interpolation="nearest")
        ax_img.set_axis_off()
        h, w = crop.shape
        ax_img.plot([w - 110, w - 10], [h - 14, h - 14], color="white", linewidth=2.0,
                    solid_capstyle="butt")
        ax_img.text(w - 60, h - 22, "100 px", color="white", ha="center", va="bottom",
                    fontsize=lad["small"])

        if titles:
            ax_growth.set_title("Growth on YPD, 30 °C", loc="left")
            ax_end.set_title("Endpoint colony area", loc="left")
            ax_plate.set_title("384-pin plate map", loc="left")
            ax_img.set_title("Colony image (grayscale)", loc="left")

        _angle_crowded_xticklabels(fig, ax_end)

        if tick_family:
            for ax in (ax_growth, ax_end, ax_plate):
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    if tick.get_text() not in STRAINS:
                        tick.set_family(tick_family)
            for tick in bar.ax.get_yticklabels():
                tick.set_family(tick_family)

        for i, ax in enumerate((ax_growth, ax_end, ax_plate, ax_img)):
            if titles:
                ax.text(-0.02, 1.0, PANEL_STYLES[panel](i), transform=ax.transAxes,
                        fontsize=lad["panel"], fontweight="bold", ha="right", va="bottom")
            else:
                # A left-aligned title, so the layout engine reserves room for the
                # letter and it never collides with tick labels or the axis title.
                ax.set_title(PANEL_STYLES[panel](i), loc="left", fontsize=lad["panel"],
                             fontweight="bold")

        fig.savefig(path, format="svg", metadata={"Date": None})
        plt.close(fig)
    _strip_doctype(path)


def _angle_crowded_xticklabels(fig, ax) -> None:
    """Angle ``ax``'s x tick labels by 30 degrees if any two of them overlap.

    Category labels (strain names) are the first thing to collide as the text
    grows; angling them is the usual remedy and costs a little height.

    Args:
        fig: The figure holding ``ax``.
        ax: Axes whose x tick labels are checked.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()]
    if any(boxes[i].overlaps(boxes[i + 1]) for i in range(len(boxes) - 1)):
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


def _strip_doctype(path: Path) -> None:
    """Remove the DOCTYPE declaration matplotlib writes into SVG output.

    The artifact host refuses supporting XML files that carry DTD declarations,
    and browsers render the SVG identically without it.

    Args:
        path: SVG file to rewrite in place.
    """
    text = path.read_text(encoding="utf-8")
    path.write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", text, count=1), encoding="utf-8")


def render_all(nunito_dir: Path | None) -> dict:
    """Render every combination the chooser page can display.

    Args:
        nunito_dir: Folder holding the Nunito Sans TTFs, or ``None`` to skip them.

    Returns:
        A manifest describing the rendered files.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    fonts = list(FONTS)
    if not _register_nunito(nunito_dir):
        fonts.remove("nunito")
    data = _synthetic_data()
    crop = _colony_crop()
    manifest = {"full_width_mm": FULL_WIDTH_MM, "height_mm": FIGURE_HEIGHT_MM,
                "fonts": fonts, "files": []}
    for font, ladder, palette, chrome, panel in itertools.product(
            fonts, LADDERS, PALETTES, CHROME, PANEL_STYLES):
        name = f"{font}__{ladder}__{palette}__{chrome}__{panel}.svg"
        render_figure(font, ladder, palette, chrome, panel, data, crop, OUT / name)
        manifest["files"].append(name)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------------------
# Round 2: size-ladder candidates, rendered with the decided round-1 choices
# (DejaVu Sans, published Okabe-Ito order, minimal chrome, "(a)" panel labels).
# ---------------------------------------------------------------------------

#: Candidate ladders: tick / axis-label / panel-letter sizes in pt at printed
#: size, with line and marker weights scaled alongside.
LADDER_CANDIDATES = {
    "l7": {"small": 7.0, "body": 8.0, "panel": 10.0, "axes_lw": 0.6, "line_lw": 1.25,
           "marker": 3.0, "tick_len": 3.0},
    "l8": {"small": 8.0, "body": 9.0, "panel": 12.0, "axes_lw": 0.7, "line_lw": 1.4,
           "marker": 3.5, "tick_len": 3.5},
    "l9": {"small": 9.0, "body": 10.0, "panel": 12.0, "axes_lw": 0.75, "line_lw": 1.5,
           "marker": 3.5, "tick_len": 3.5},
    "l10": {"small": 10.0, "body": 11.0, "panel": 14.0, "axes_lw": 0.8, "line_lw": 1.6,
            "marker": 4.0, "tick_len": 4.0},
    "l10b": {"small": 10.0, "body": 12.0, "panel": 14.0, "axes_lw": 0.8, "line_lw": 1.75,
             "marker": 4.0, "tick_len": 4.0},
    "l14": {"small": 14.0, "body": 16.0, "panel": 18.0, "axes_lw": 1.0, "line_lw": 2.0,
            "marker": 5.0, "tick_len": 5.0},
}

LADDER_OUT = OUT / "ladder"


def _visible_texts(fig) -> list:
    """Every drawn, non-empty text artist in ``fig``.

    matplotlib keeps label artists for ticks that fall outside an axis's view
    limits and never draws them, so tick labels are kept only when their tick
    sits inside the visible interval.
    """
    from matplotlib.text import Text

    tick_labels: set[int] = set()
    drawn: list = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            span = (high - low) or 1.0
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for label in (tick.label1, tick.label2):
                    tick_labels.add(id(label))
                    inside = low - 1e-9 * span <= tick.get_loc() <= high + 1e-9 * span
                    if inside and label.get_visible() and label.get_text().strip():
                        drawn.append(label)
    others = [t for t in fig.findobj(Text)
              if id(t) not in tick_labels and t.get_visible() and t.get_text().strip()]
    return drawn + others


def _layout_metrics(fig, data_axes: list) -> dict:
    """Measure how much of ``fig`` holds data and whether any text collides.

    Args:
        fig: A drawn figure on the Agg canvas.
        data_axes: The axes that hold data (colorbars and image panels excluded).

    Returns:
        ``data_area`` (share of the figure inside ``data_axes``), ``overlaps``
        (pairs of overlapping text boxes) and ``clipped`` (texts that extend
        past the figure edge).
    """
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    area = sum(a.get_window_extent(renderer).width * a.get_window_extent(renderer).height
               for a in data_axes)
    texts = _visible_texts(fig)
    boxes = [t.get_window_extent(renderer) for t in texts]
    overlaps = sum(1 for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                   if boxes[i].overlaps(boxes[j]) and boxes[i].width and boxes[j].width
                   and not _angled_neighbors_clear(texts[i], texts[j], fig))
    clipped = sum(1 for b in boxes if b.x0 < fig_box.x0 - 0.5 or b.x1 > fig_box.x1 + 0.5
                  or b.y0 < fig_box.y0 - 0.5 or b.y1 > fig_box.y1 + 0.5)
    hidden = _data_points_under_text(fig, renderer)
    return {"data_area": round(area / (fig_box.width * fig_box.height), 3),
            "overlaps": overlaps, "clipped": clipped, "hidden": hidden}


def _angled_neighbors_clear(a, b, fig) -> bool:
    """True when two angled tick labels only overlap as bounding boxes.

    An angled label's axis-aligned box is much larger than its glyphs, so two
    neighbors' boxes overlap even when the text does not. Two labels angled by
    theta on the same baseline clear each other when their spacing times
    sin(theta) exceeds the text height.
    """
    import math

    angle = a.get_rotation()
    if angle == 0 or b.get_rotation() != angle:
        return False
    (xa, ya), (xb, yb) = a.get_transform().transform(a.get_position()), \
        b.get_transform().transform(b.get_position())
    if abs(ya - yb) > 0.5:
        return False
    height_px = max(a.get_fontsize(), b.get_fontsize()) * 1.2 * fig.dpi / 72
    return abs(xa - xb) * math.sin(math.radians(angle)) > height_px


def _data_points_under_text(fig, renderer) -> int:
    """Count plotted data points that sit under a legend or a text label.

    Text that covers data is the other way a larger ladder chops a figure, and
    a text-to-text overlap count cannot see it.
    """
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D

    hidden = 0
    for ax in fig.axes:
        covers = [t.get_window_extent(renderer) for t in ax.texts
                  if t.get_visible() and t.get_text().strip()]
        legend = ax.get_legend()
        if legend is not None:
            covers.append(legend.get_window_extent(renderer))
        if not covers:
            continue
        points = []
        for artist in ax.get_children():
            if isinstance(artist, Line2D) and artist.get_visible() and len(artist.get_xydata()):
                points.extend(ax.transData.transform(artist.get_xydata()))
            elif isinstance(artist, PathCollection) and artist.get_visible():
                offsets = artist.get_offsets()
                if len(offsets):
                    points.extend(ax.transData.transform(offsets))
        hidden += sum(1 for x, y in points if any(b.contains(x, y) for b in covers))
    return hidden


def _save_svg(fig, path: Path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None})
    _strip_doctype(path)


def _half_growth_figure(data: dict) -> tuple:
    """A single growth panel at half A4 text width (77.1 x 60 mm)."""
    fig, ax = plt.subplots(figsize=(77.1 / MM_PER_INCH, 60 / MM_PER_INCH),
                           layout="constrained")
    colors = PALETTES["canonical"]
    for i, (curve, label) in enumerate(zip(data["curves"], STRAINS)):
        mean, sd = curve.mean(0), curve.std(0, ddof=1)
        ax.plot(data["hours"], mean, color=colors[i], label=label)
        ax.fill_between(data["hours"], mean - sd, mean + sd, color=colors[i], alpha=0.15,
                        linewidth=0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Colony area (mm$^2$)")
    ax.set_xticks(range(0, 49, 12))
    legend = ax.legend(loc="best", handlelength=1.2)
    for text, italic in zip(legend.get_texts(), ITALIC):
        text.set_fontstyle("italic" if italic else "normal")
    return fig, [ax]


def _facet_figure(data: dict, panel_fmt) -> tuple:
    """A 3 x 3 small-multiples grid of growth curves at full A4 text width."""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(3, 3, figsize=(FULL_WIDTH_MM / MM_PER_INCH, 120 / MM_PER_INCH),
                             sharex=True, sharey=True, layout="constrained")
    hours = data["hours"]
    names = ["WT", "ura3\u0394", "his3\u0394", "leu2\u0394", "lys2\u0394", "trp1\u0394",
             "met15\u0394", "ade2\u0394", "can1\u0394"]
    for k, (ax, name) in enumerate(zip(axes.ravel(), names)):
        rate, cap, lag = 0.12 + 0.012 * k, 2.0 + 0.15 * (k % 4), 6 + k
        base = cap / (1 + np.exp(-rate * (hours - lag - 10)))
        reps = base + rng.normal(0, 0.05 * cap, (4, hours.size))
        for rep in reps:
            ax.plot(hours, rep, color="#BBBBBB", linewidth=0.6)
        ax.plot(hours, reps.mean(0), color=PALETTES["canonical"][0])
        ax.text(0.04, 0.96, name, transform=ax.transAxes, va="top",
                fontstyle="normal" if k == 0 else "italic")
        ax.set_xticks([0, 24, 48])
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Area (mm$^2$)")
    return fig, list(axes.ravel())


def render_ladder_candidates() -> dict:
    """Render every ladder candidate three ways and record its layout metrics.

    Returns:
        A manifest mapping each ladder id to its sizes, file names and metrics.
    """
    plt.switch_backend("agg")
    LADDER_OUT.mkdir(parents=True, exist_ok=True)
    data = _synthetic_data()
    crop = _colony_crop()
    manifest: dict = {}
    for ladder_id, ladder in LADDER_CANDIDATES.items():
        LADDERS[ladder_id] = ladder
        entry: dict = {"sizes": [ladder["small"], ladder["body"], ladder["panel"]],
                       "files": {}, "metrics": {}}
        with plt.rc_context(_rc_for("dejavu", ladder_id, "minimal")):
            captured: dict = {}
            original = plt.Figure.savefig

            def capture(self, *args, **kwargs):
                captured["fig"] = self
                return original(self, *args, **kwargs)

            plt.Figure.savefig = capture
            try:
                grid_path = LADDER_OUT / f"{ladder_id}__grid.svg"
                render_figure("dejavu", ladder_id, "canonical", "minimal", "paren", data,
                              crop, grid_path, legend_outside=True)
            finally:
                plt.Figure.savefig = original
            fig = captured["fig"]
            fig.canvas.draw()
            data_axes = [a for a in fig.axes if a.get_label() != "<colorbar>"][:3]
            entry["metrics"]["grid"] = _layout_metrics(fig, data_axes)
            entry["files"]["grid"] = grid_path.name
            plt.close(fig)

            for view, builder in (("half", lambda: _half_growth_figure(data)),
                                  ("facets", lambda: _facet_figure(data, None))):
                fig, data_axes = builder()
                fig.canvas.draw()
                entry["metrics"][view] = _layout_metrics(fig, data_axes)
                path = LADDER_OUT / f"{ladder_id}__{view}.svg"
                _save_svg(fig, path)
                entry["files"][view] = path.name
                plt.close(fig)
        manifest[ladder_id] = entry
    (LADDER_OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nunito-dir", type=Path, default=None)
    parser.add_argument("--ladders-only", action="store_true",
                        help="Render only the round-2 size-ladder candidates.")
    args = parser.parse_args()
    if args.ladders_only:
        result = render_ladder_candidates()
        print(json.dumps({k: v["metrics"] for k, v in result.items()}, indent=1))
    else:
        result = render_all(args.nunito_dir)
        print(f"rendered {len(result['files'])} figures into {OUT}")
