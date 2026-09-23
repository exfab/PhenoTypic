"""What one ``CalibrateColorRpcc`` run looked like, and a figure of it.

:class:`CalibrationOverlayRecord` is plain data -- the as-shot ROI crops, every
tile's boxes, the chart patch it was matched to, its status and ΔE00 -- built
by the operation during ``apply()`` and kept even when the frame is refused.
:func:`render_calibration_overlay` draws it and reads nothing else, so a record
persisted elsewhere can be drawn anywhere.

See the spec: ``docs/superpowers/specs/2026-09-22-checker-calibration-overlay/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ._checker_measure import TileMeasurement
    from ._checker_qc import QcRecord
    from ._checker_roi import CheckerLattice

Verdict = Literal["corrected", "corrected_with_warnings", "skipped", "refused"]
TileStatus = Literal["used", "partly_covered", "rejected", "excluded", "empty"]
Box = tuple[float, float, float, float]
Rgb = tuple[float, float, float]


class TileOverlay(BaseModel):
    """One tile: where it was, what it was matched to, how it came out.

    Attributes:
        row: Tile row within its ROI lattice.
        col: Tile column within its ROI lattice.
        patch: Chart patch the tile was identified as.
        status: See the spec's tile-status table.
        full_box: ``(y0, y1, x0, x1)`` of the whole tile, ROI-local pixels.
        core_box: ``(y0, y1, x0, x1)`` of the pixels the medoid came from.
        measured_srgb: The medoid pixel's own sRGB; ``None`` for an empty tile.
        reference_srgb: The chart's reference colour, sRGB-encoded.
        impurity: Contaminated fraction; ``None`` for an empty tile.
        delta_e_before: ΔE00 before correction; ``None`` unless the fit ran.
        delta_e_after: ΔE00 after correction; ``None`` unless the fit ran.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    row: int
    col: int
    patch: str
    status: TileStatus
    full_box: Box
    core_box: Box
    measured_srgb: Rgb | None
    reference_srgb: Rgb
    impurity: float | None
    delta_e_before: float | None
    delta_e_after: float | None


class RoiOverlay(BaseModel):
    """One ROI: its as-shot pixels, its tiles and what the gate said.

    Attributes:
        unidentified_boxes: ``(full_box, core_box)`` pairs of a lattice that
            was found but refused before any tile was identified -- a
            tile-count or placement refusal, or every box outside the ROI.
            Empty whenever ``tiles`` is not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    roi_index: int
    label: str | None
    crop: np.ndarray
    lattice_found: bool
    n_tile_columns: int
    flags: list[str]
    warnings: list[str]
    tiles: list[TileOverlay]
    unidentified_boxes: list[tuple[Box, Box]] = Field(default_factory=list)


class CalibrationOverlayRecord(BaseModel):
    """Everything :func:`render_calibration_overlay` draws, and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_name: str | None
    verdict: Verdict
    degree: int
    n_fitted: int | None
    n_expected: int
    refusal: str | None
    rois: list[RoiOverlay]


@dataclass
class RoiDraft:
    """What the ROI loop learned about one ROI, before the frame's outcome."""

    roi_index: int
    label: str | None
    crop: np.ndarray
    lattice: CheckerLattice | None = None
    tiles: list[tuple[TileMeasurement, str]] = field(default_factory=list)
    claimed: bool = False


def _tile_status(
        tile: TileMeasurement,
        patch: str,
        *,
        claimed: bool,
        fitted: bool,
        rejected: set[str],
        impurity_limit: float,
) -> TileStatus:
    """The spec's tile-status table, in its order of precedence.

    A tile whose ROI lost a patch collision is ``excluded`` even when the
    winner's patch of that name was rejected: the rejection is not its own.
    A rejection is shown without an accepted fit, since it explains a
    post-rejection rank refusal.
    """
    if not tile.n_pixels:
        return "empty"
    if not claimed:
        return "excluded"
    if patch in rejected:
        return "rejected"
    if not fitted:
        return "excluded"
    if tile.impurity == tile.impurity and tile.impurity > impurity_limit:
        return "partly_covered"
    return "used"


def build_overlay_record(
        *,
        image_name: str | None,
        verdict: Verdict,
        refusal: str | None,
        degree: int,
        n_expected: int,
        n_fitted: int | None,
        drafts: Sequence[RoiDraft],
        qc: Sequence[QcRecord],
        reference_srgb: Mapping[str, Rgb],
        fitted_patches: Mapping[str, Mapping[str, float]] | None,
        rejected: set[str],
        impurity_limit: float,
        core_trim: float,
) -> CalibrationOverlayRecord:
    """Assemble the record from the ROI loop's drafts and the frame's outcome.

    Args:
        image_name: The image's name, for the figure title.
        verdict: See the spec's verdict table.
        refusal: The refusal message when ``verdict == "refused"``.
        degree: The configured polynomial degree.
        n_expected: Patches the chart has.
        n_fitted: Patches that reached the fit, or ``None`` with no fit.
        drafts: One per ROI, in ROI order. Each crop is marked read-only.
        qc: The gate's records; matched to drafts by ``roi_index``.
        reference_srgb: Patch name -> reference colour, sRGB-encoded.
        fitted_patches: ``ColorCheckerProfile.diagnostics["patches"]`` when
            the fit ran and was accepted, else ``None``.
        rejected: Patches outlier rejection removed; shown even when
            ``fitted_patches`` is ``None`` (a post-rejection rank refusal).
        impurity_limit: ``QcLimits.max_tile_impurity``.
        core_trim: The operation's ``core_trim``, for the core boxes.

    Returns:
        The frozen record.
    """
    qc_by_roi = {record.roi_index: record for record in qc}
    rois = []
    for draft in drafts:
        gate = qc_by_roi.get(draft.roi_index)
        tiles: list[TileOverlay] = []
        unidentified: list[tuple[Box, Box]] = []
        if draft.lattice is not None:
            lattice = draft.lattice
            full = {(r, c): (y0, y1, x0, x1)
                    for r, c, y0, y1, x0, x1 in lattice.boxes(rot=lattice.rot)}
            core = {(r, c): (y0, y1, x0, x1)
                    for r, c, y0, y1, x0, x1 in lattice.boxes(core=core_trim, rot=lattice.rot)}
            if not draft.tiles:
                # Refused after its lattice was found: there is no identity
                # to show, but the detected boxes are what explain the refusal.
                unidentified = [(full[key], core[key]) for key in full]
            for measured, patch in draft.tiles:
                status = _tile_status(
                        measured, patch, claimed=draft.claimed,
                        fitted=fitted_patches is not None, rejected=rejected,
                        impurity_limit=impurity_limit,
                )
                scored = (
                    fitted_patches.get(patch)
                    if fitted_patches is not None
                    and status in ("used", "partly_covered", "rejected")
                    else None
                )
                key = (measured.row, measured.col)
                tiles.append(TileOverlay(
                        row=measured.row, col=measured.col, patch=patch, status=status,
                        full_box=full[key], core_box=core[key],
                        measured_srgb=measured.srgb if measured.n_pixels else None,
                        reference_srgb=reference_srgb[patch],
                        impurity=float(measured.impurity) if measured.n_pixels else None,
                        delta_e_before=None if scored is None else float(scored["deltaE00_before"]),
                        delta_e_after=None if scored is None else float(scored["deltaE00_after"]),
                ))
        # The record is frozen; so are its pixels. The draft's crop is already
        # an owned copy, so this costs nothing.
        draft.crop.flags.writeable = False
        rois.append(RoiOverlay(
                roi_index=draft.roi_index,
                label=draft.label,
                crop=draft.crop,
                lattice_found=draft.lattice is not None,
                n_tile_columns=0 if draft.lattice is None else len(draft.lattice.columns),
                flags=list(gate.flags) if gate is not None else [],
                warnings=list(gate.warnings) if gate is not None else [],
                tiles=tiles,
                unidentified_boxes=unidentified,
        ))
    return CalibrationOverlayRecord(
            image_name=image_name, verdict=verdict, degree=degree,
            n_fitted=n_fitted, n_expected=n_expected, refusal=refusal, rois=rois,
    )



# -- the figure ----------------------------------------------------------------
#
# Text takes the theme's defaults -- font, size and colour -- throughout. The
# only colours set on it are the ΔE00 bands, which carry meaning. Every size
# below is derived from text measured in that default style.

#: ΔE00 after-correction bands (spec §Colour): <= GOOD is good, <= FAIR fair.
DELTA_E_GOOD = 2.0
DELTA_E_FAIR = 5.0

#: Okabe-Ito semantic colours per tile status (DESIGN.md §01).
STATUS_COLOURS: dict[str, str] = {
    "used": "#009E73", "partly_covered": "#E69F00", "rejected": "#D55E00",
    "excluded": "#BBBBBB", "empty": "#BBBBBB",
}
_DASHED = frozenset({"rejected", "excluded", "empty"})
#: Darkened text variants of the data colours, legible on a light ground.
_DELTA_E_TEXT = {"good": "#007a5a", "fair": "#a86f00", "poor": "#b04a00"}
_SWATCH_EDGE = "#2e3a4e"
_SUFFIX = {"used": "", "rejected": " · rejected", "excluded": " · excluded",
           "empty": " · empty"}

_IMAGE_W_IN = 1.6        # nominal image width; grows when labels need room
_SWATCH_IN = 0.14        # each of the measured | reference swatches
_GAP_IN = 0.06           # image <-> swatch, swatch <-> text
_EDGE_IN = 0.08          # outer padding
_GROUP_GAP_IN = 0.3      # between ROI groups
_LINE_GAP = 1.15         # line pitch as a multiple of text height
_KEY_ROW_GAP = 1.35      # key entry pitch, as a multiple of a two-line block
_MARKER_PAD = 0.15       # tile-number box padding, in multiples of the font size
_MARKER_ROOM = 1.6       # smallest core box side, as a multiple of the marker
_CORE_LW = 1.8


def delta_e_band(value: float) -> str:
    """``"good"``, ``"fair"`` or ``"poor"`` for a ΔE00 after-correction value."""
    if value <= DELTA_E_GOOD:
        return "good"
    return "fair" if value <= DELTA_E_FAIR else "poor"


class _TextMeter:
    """Measures text in the default style, in inches, on an Agg renderer."""

    def __init__(self, probe, dpi: float) -> None:
        self._probe = probe
        self._renderer = probe.canvas.get_renderer()
        self._dpi = dpi
        artist = probe.text(0, 0, "")
        #: The default text size, in points.
        self.fontsize_pt: float = artist.get_fontsize()
        artist.remove()

    def size(self, text: str) -> tuple[float, float]:
        artist = self._probe.text(0, 0, text)
        box = artist.get_window_extent(self._renderer)
        artist.remove()
        return box.width / self._dpi, box.height / self._dpi


@dataclass
class _Block:
    name: str
    delta_e: str
    colour: str | None
    width: float


@dataclass
class _RoiPlan:
    """Every size of one ROI group, in inches.

    Vertically, top to bottom: ``title_h``, a gap, ``top_pad`` (label overhang
    above the image), ``image_h``, ``bottom_pad``, ``key_h``, ``note_h``.
    Horizontally: ``left_w``, ``image_w``, ``right_w``.
    """

    title: str
    title_w: float
    title_h: float
    side: bool
    left_w: float
    image_w: float
    image_h: float
    right_w: float
    width: float
    blocks: dict[tuple[int, int], _Block]
    name_h: float
    de_h: float
    split: float = 0.0
    top_pad: float = 0.0
    bottom_pad: float = 0.0
    key_h: float = 0.0
    key_rows: int = 1
    key_entry_w: float = 0.0
    key_row_h: float = 0.0
    number_w: float = 0.0
    notes: list[str] = field(default_factory=list)
    note_line_h: float = 0.0
    note_h: float = 0.0

    @property
    def block_h(self) -> float:
        """One two-line label: name, the split between the lines, ΔE line."""
        return self.name_h + self.split + self.de_h

    @property
    def height(self) -> float:
        return (self.title_h + _GAP_IN + self.top_pad + self.image_h + self.bottom_pad
                + self.key_h + self.note_h)


def _name_line(tile: TileOverlay) -> str:
    if tile.status == "partly_covered":
        return f"{tile.patch} · {(tile.impurity or 0.0) * 100:.0f}% covered"
    return tile.patch + _SUFFIX[tile.status]


def _delta_e_line(tile: TileOverlay) -> tuple[str, str | None]:
    """The ΔE00 line and its band colour; ``None`` keeps the default colour.

    ASCII ``->``, not an arrow glyph: the theme's default font may lack U+2192.
    """
    if tile.delta_e_before is None or tile.delta_e_after is None:
        return "ΔE00 not fitted", None
    return (f"ΔE00 {tile.delta_e_before:.1f} -> {tile.delta_e_after:.1f}",
            _DELTA_E_TEXT[delta_e_band(tile.delta_e_after)])


def _min_centre_gap_px(tiles: Sequence[TileOverlay]) -> float | None:
    """Smallest vertical gap between neighbouring tile centres in one column."""
    gaps = []
    for col in {t.col for t in tiles}:
        centres = sorted((t.full_box[0] + t.full_box[1]) / 2 for t in tiles if t.col == col)
        gaps += [b - a for a, b in zip(centres, centres[1:])]
    return min(gaps) if gaps else None


def _wrap_to_width(message: str, width: float, meter: _TextMeter) -> list[str]:
    """Greedy word wrap by measured width; a word wider than *width* stands alone."""
    lines: list[str] = []
    for word in message.split():
        trial = f"{lines[-1]} {word}" if lines else word
        if lines and meter.size(trial)[0] <= width:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def _plan_side(plan: _RoiPlan, roi: RoiOverlay) -> None:
    """Two label columns beside the image; the image grows until labels fit."""
    h_px = roi.crop.shape[0]
    gap_px = _min_centre_gap_px(roi.tiles)
    if gap_px:
        have = gap_px * plan.image_h / h_px
        scale = max(1.0, plan.block_h * _LINE_GAP / have)
        plan.image_w, plan.image_h = plan.image_w * scale, plan.image_h * scale
    swatches = _GAP_IN + 2 * _SWATCH_IN + _GAP_IN
    widths = {c: max((b.width for (_, cc), b in plan.blocks.items() if cc == c), default=None)
              for c in (0, 1)}
    plan.left_w = _EDGE_IN + (widths[0] + swatches if widths[0] is not None else 0.0)
    plan.right_w = _EDGE_IN + (widths[1] + swatches if widths[1] is not None else 0.0)
    plan.width = plan.left_w + plan.image_w + plan.right_w

    # A label block straddles its tile's centre row: the name above it, the
    # ΔE line below. A tile near the crop's top or bottom edge would push its
    # block past the image, into the ROI title or the notes; reserve that room.
    up = max(plan.name_h + plan.split / 2, _SWATCH_IN / 2)
    down = max(plan.de_h + plan.split / 2, _SWATCH_IN / 2)
    for t in roi.tiles:
        from_top = (t.full_box[0] + t.full_box[1]) / 2 * plan.image_h / h_px
        plan.top_pad = max(plan.top_pad, up - from_top)
        plan.bottom_pad = max(plan.bottom_pad, down - (plan.image_h - from_top))


def _plan_key(plan: _RoiPlan, roi: RoiOverlay, meter: _TextMeter) -> None:
    """Numbered tiles and a key below the image, one key column per tile column."""
    w_px = roi.crop.shape[1]
    sizes = [meter.size(str(n)) for n in range(1, len(roi.tiles) + 1)]
    number_w = max(w for w, _ in sizes)
    number_h = max(h for _, h in sizes)
    # The marker sits inside the core box; the box must stay visible around
    # it, or the tile's status colour is hidden behind its number.
    smallest_px = min(min(t.core_box[1] - t.core_box[0], t.core_box[3] - t.core_box[2])
                      for t in roi.tiles)
    pad_in = 2 * _MARKER_PAD * meter.fontsize_pt / 72
    marker = max(number_w, number_h) + pad_in
    scale = max(1.0, marker * _MARKER_ROOM / (smallest_px * plan.image_w / w_px))
    plan.image_w, plan.image_h = plan.image_w * scale, plan.image_h * scale
    entry_w = (number_w + _GAP_IN + 2 * _SWATCH_IN + _GAP_IN
               + max(b.width for b in plan.blocks.values()) + _EDGE_IN)
    rows = max(1, max(t.row for t in roi.tiles) + 1)
    columns = math.ceil(len(roi.tiles) / rows)
    plan.width = max(_EDGE_IN + plan.image_w + _EDGE_IN, columns * entry_w)
    plan.left_w = plan.right_w = (plan.width - plan.image_w) / 2
    plan.key_rows, plan.key_entry_w, plan.number_w = rows, entry_w, number_w
    plan.key_row_h = plan.block_h * _KEY_ROW_GAP
    plan.key_h = _GAP_IN + rows * plan.key_row_h


def _plan_roi(roi: RoiOverlay, meter: _TextMeter) -> _RoiPlan:
    """Every size for one ROI group, in inches, from measured text."""
    h_px, w_px = roi.crop.shape[:2]
    title = f"ROI {roi.roi_index}" + (f" · {roi.label}" if roi.label else "")
    title_w, title_h = meter.size(title)
    blocks, name_h, de_h = {}, 0.0, 0.0
    for t in roi.tiles:
        name = _name_line(t)
        delta_e, colour = _delta_e_line(t)
        n_w, n_h = meter.size(name)
        d_w, d_h = meter.size(delta_e)
        blocks[(t.row, t.col)] = _Block(name, delta_e, colour, max(n_w, d_w))
        name_h, de_h = max(name_h, n_h), max(de_h, d_h)
    # A little air between a label's two lines, so descenders on the name
    # never touch the ΔE line below.
    plan = _RoiPlan(title, title_w, title_h, True, 0.0, _IMAGE_W_IN,
                    _IMAGE_W_IN * h_px / w_px, 0.0, 0.0, blocks, name_h, de_h,
                    split=(_LINE_GAP - 1) * max(name_h, de_h))
    if roi.n_tile_columns <= 2 or not roi.tiles:
        _plan_side(plan, roi)
    else:
        plan.side = False
        _plan_key(plan, roi, meter)

    # Flags, then warnings, wrapped by measured width to the group. A word
    # wider than the group widens it, so no line can leave its column.
    messages = [*roi.flags, *roi.warnings]
    widest_word = max((meter.size(w)[0] for m in messages for w in m.split()), default=0.0)
    content_w = max(title_w, widest_word + 2 * _EDGE_IN)
    if content_w > plan.width:                   # widen both sides equally
        extra = (content_w - plan.width) / 2
        plan.left_w, plan.right_w, plan.width = (plan.left_w + extra, plan.right_w + extra,
                                                 content_w)
    for message in messages:
        plan.notes += _wrap_to_width(message, plan.width - 2 * _EDGE_IN, meter)
    if plan.notes:
        plan.note_line_h = max(meter.size(line)[1] for line in plan.notes) * _LINE_GAP
        plan.note_h = _GAP_IN + len(plan.notes) * plan.note_line_h
    return plan


def _figure_title(record: CalibrationOverlayRecord) -> str:
    fitted = ("not fitted" if record.n_fitted is None
              else f"{record.n_fitted}/{record.n_expected} patches fitted")
    return (f"{record.image_name or 'unnamed image'} · {record.verdict.replace('_', ' ')}"
            f" · degree {record.degree} · {fitted}")


def _swatches(ax, x: float, cy: float, height: float, tile: TileOverlay, rectangle) -> None:
    """The measured | reference pair, left edge at *x*, centred on *cy*."""
    for k, rgb in enumerate((tile.measured_srgb, tile.reference_srgb)):
        style: dict[str, object] = ({"facecolor": rgb} if rgb is not None
                                    else {"facecolor": "none", "hatch": "////"})
        ax.add_patch(rectangle((x + k * _SWATCH_IN, cy - height / 2), _SWATCH_IN, height,
                               edgecolor=_SWATCH_EDGE, linewidth=0.4, **style))


def render_calibration_overlay(
        record: CalibrationOverlayRecord,
        *,
        figsize: tuple[float, float] | None = None,
        dpi: float = 160,
) -> Figure:
    """Draw each ROI's tiles, the patch each was matched to, and its ΔE00.

    Every size is computed in inches from measured text, so no two labels
    overlap and none leaves the figure (spec §2).

    Args:
        record: From ``CalibrateColorRpcc.calibration_record``.
        figsize: Optional ``(width, height)`` in inches. Must be at least the
            computed size; the content is centred in any extra room.
        dpi: Resolution the text is measured and drawn at.

    Returns:
        A ``matplotlib.figure.Figure`` with an Agg canvas attached.

    Raises:
        ValueError: If *figsize* is smaller than the labels need.
    """
    from matplotlib import font_manager, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from phenotypic.sdk_.viz.figures._mpl_theme import phenotypic_mpl_context

    # Text is measured here but drawn whenever the caller saves the figure,
    # usually outside the theme. A generic family ("sans-serif") is looked up
    # in rcParams at *draw* time, so the theme's font would be measured and
    # matplotlib's default one drawn -- wider, so labels would clip. Naming
    # the family the theme resolves to makes every text carry that font from
    # its creation; font, size and colour otherwise stay the theme defaults.
    with phenotypic_mpl_context():
        default_family = font_manager.FontProperties(
                fname=font_manager.findfont(font_manager.FontProperties())).get_name()
    with phenotypic_mpl_context(), rc_context({"font.family": [default_family]}):
        probe = Figure(dpi=dpi)
        FigureCanvasAgg(probe)
        meter = _TextMeter(probe, dpi)
        title = _figure_title(record)
        title_w, title_h = meter.size(title)
        plans = [_plan_roi(roi, meter) for roi in record.rois]

        content_w = sum(p.width for p in plans) + _GROUP_GAP_IN * max(0, len(plans) - 1)
        top_band = title_h + 2 * _EDGE_IN
        need_w = max(content_w, title_w) + 2 * _EDGE_IN
        need_h = top_band + max((p.height for p in plans), default=0.0) + _EDGE_IN
        if figsize is not None and (figsize[0] < need_w - 1e-9 or figsize[1] < need_h - 1e-9):
            raise ValueError(
                    f"figsize={figsize} is too small for these labels; this record "
                    f"needs at least ({need_w:.2f}, {need_h:.2f}) inches."
            )
        fig_w, fig_h = figsize if figsize is not None else (need_w, need_h)
        fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        FigureCanvasAgg(fig)

        def axes(x, bottom, w, h, **kwargs):
            return fig.add_axes((x / fig_w, bottom / fig_h, w / fig_w, h / fig_h), **kwargs)

        fig.text(0.5, 1 - _EDGE_IN / fig_h, title, ha="center", va="top")
        x = (fig_w - content_w) / 2
        y_top = fig_h - top_band - (fig_h - need_h) / 2
        for roi, plan in zip(record.rois, plans):
            _draw_roi(roi, plan, axes, x, y_top, Rectangle)
            x += plan.width + _GROUP_GAP_IN
        return fig


def _draw_roi(roi: RoiOverlay, plan: _RoiPlan, axes, x: float, y_top: float,
              rectangle) -> None:
    """Title, image with boxes, labels or key, and notes for one ROI group."""
    h_px, w_px = roi.crop.shape[:2]
    title_ax = axes(x, y_top - plan.title_h, plan.width, plan.title_h)
    title_ax.set_xlim(0, plan.width)
    title_ax.set_ylim(0, plan.title_h)
    title_ax.set_axis_off()
    # Centred over the image, but never past the group's own edges.
    half = plan.title_w / 2
    title_x = min(max(plan.left_w + plan.image_w / 2, half), plan.width - half)
    title_ax.text(title_x, 0.0, plan.title, ha="center", va="bottom")

    img_top = y_top - plan.title_h - _GAP_IN - plan.top_pad
    img_bottom = img_top - plan.image_h
    ax = axes(x + plan.left_w, img_bottom, plan.image_w, plan.image_h)
    ax.imshow(roi.crop, interpolation="nearest", aspect="auto")
    ax.set_xlim(-0.5, w_px - 0.5)
    ax.set_ylim(h_px - 0.5, -0.5)
    ax.set_axis_off()

    # A refused lattice's boxes carry no identity: drawn like excluded tiles,
    # and never labelled.
    boxes = [(t.full_box, t.core_box, t.status) for t in roi.tiles]
    boxes += [(full, core, "excluded") for full, core in roi.unidentified_boxes]
    for (y0, y1, x0, x1), (cy0, cy1, cx0, cx1), status in boxes:
        ax.add_patch(rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                               edgecolor="white", linewidth=0.6, linestyle=(0, (1.5, 1.5)),
                               alpha=0.8))
        ax.add_patch(rectangle((cx0 - 0.5, cy0 - 0.5), cx1 - cx0, cy1 - cy0, fill=False,
                               edgecolor=STATUS_COLOURS[status], linewidth=_CORE_LW,
                               linestyle="--" if status in _DASHED else "-"))

    if plan.side:
        _draw_side_labels(roi, plan, axes, x, img_bottom, ax, rectangle)
        bottom = img_bottom - plan.bottom_pad
    else:
        _draw_key(roi, plan, axes, x, img_bottom, ax, rectangle)
        bottom = img_bottom - plan.key_h

    if plan.notes:
        note_ax = axes(x, bottom - plan.note_h, plan.width, plan.note_h)
        note_ax.set_xlim(0, plan.width)
        note_ax.set_ylim(plan.note_h, 0)
        note_ax.set_axis_off()
        for i, line in enumerate(plan.notes):
            note_ax.text(_EDGE_IN, _GAP_IN + i * plan.note_line_h, line, ha="left", va="top")


def _draw_side_labels(roi: RoiOverlay, plan: _RoiPlan, axes, x: float, img_bottom: float,
                      ax, rectangle) -> None:
    """Column-0 labels on the left, column-1 on the right, each at its tile's row."""
    h_px = roi.crop.shape[0]
    sides = {0: axes(x, img_bottom, plan.left_w, plan.image_h, sharey=ax),
             1: axes(x + plan.left_w + plan.image_w, img_bottom, plan.right_w,
                     plan.image_h, sharey=ax)}
    for col, side_ax in sides.items():
        side_ax.set_xlim(0, plan.left_w if col == 0 else plan.right_w)
        side_ax.set_axis_off()
    px_per_in = h_px / plan.image_h
    swatch_h, half_split = _SWATCH_IN * px_per_in, plan.split / 2 * px_per_in
    for tile in roi.tiles:
        block = plan.blocks[(tile.row, tile.col)]
        cy = (tile.full_box[0] + tile.full_box[1]) / 2 - 0.5
        if tile.col == 0:
            side_ax, sw_x = sides[0], plan.left_w - _GAP_IN - 2 * _SWATCH_IN
            tx, ha = sw_x - _GAP_IN, "right"
        else:
            side_ax, sw_x = sides[1], _GAP_IN
            tx, ha = _GAP_IN + 2 * _SWATCH_IN + _GAP_IN, "left"
        _swatches(side_ax, sw_x, cy, swatch_h, tile, rectangle)
        side_ax.text(tx, cy - half_split, block.name, ha=ha, va="bottom")
        side_ax.text(tx, cy + half_split, block.delta_e, ha=ha, va="top", color=block.colour)


def _draw_key(roi: RoiOverlay, plan: _RoiPlan, axes, x: float, img_bottom: float,
              ax, rectangle) -> None:
    """Number every tile and list the numbers below, column-major like the card."""
    ordered = sorted(roi.tiles, key=lambda t: (t.col, t.row))
    key_ax = axes(x, img_bottom - plan.key_h, plan.width, plan.key_h)
    key_ax.set_xlim(0, plan.width)
    key_ax.set_ylim(plan.key_h, 0)
    key_ax.set_axis_off()
    # Each block sits wholly inside its row: the line-gap slack split evenly
    # above and below, the name above the split line and the ΔE line below.
    slack = (plan.key_row_h - plan.block_h) / 2
    for number, tile in enumerate(ordered, start=1):
        y0, y1, x0, x1 = tile.core_box
        ax.text((x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5, str(number), ha="center",
                va="center", bbox=dict(boxstyle=f"round,pad={_MARKER_PAD}",
                                       facecolor="white", edgecolor="none", alpha=0.85))
        col, row = divmod(number - 1, plan.key_rows)
        ex = col * plan.key_entry_w
        ey = _GAP_IN + row * plan.key_row_h + slack + plan.name_h + plan.split / 2
        block = plan.blocks[(tile.row, tile.col)]
        key_ax.text(ex + plan.number_w, ey, str(number), ha="right", va="center")
        sw_x = ex + plan.number_w + _GAP_IN
        _swatches(key_ax, sw_x, ey, _SWATCH_IN, tile, rectangle)
        tx = sw_x + 2 * _SWATCH_IN + _GAP_IN
        key_ax.text(tx, ey - plan.split / 2, block.name, ha="left", va="bottom")
        key_ax.text(tx, ey + plan.split / 2, block.delta_e, ha="left", va="top",
                    color=block.colour)
