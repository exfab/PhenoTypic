"""Find the tile lattice of a colour-checker card inside an ROI.

Two entry points, chosen by whether a lattice from an earlier frame is
available:

* :func:`fit_lattice` builds one from the ROI alone, **including how many
  tiles there are** -- patches are plateaus in a cross-channel spread signal
  and the gutters between them are the minima, so counting qualifying plateaus
  gives the grid before any phase is fitted.  Use it to build a prior on a
  reference frame, and for one-off images.
* :func:`refine` moves a stored lattice onto this frame.  ``"rigid"`` measures
  the displacement on one unclipped column and applies it to the whole
  lattice; ``"ecc"`` registers the band against a stored reference band;
  ``"frozen"`` trusts the prior unchanged.

Why the rigid variant exists: re-snapping each column independently lags on a
column the frame border clips, because that column's *visible extent* changes
with the shift, so its measured centre travels about half as far as the card
does.  Measuring on the unclipped column and moving the lattice as one body
roughly triples the horizontal capture range.

Deliberately not ported from the prototype: blob-and-RANSAC segmentation and
its cascade (1.419 and 1.825 s per band against the rigid snap's 0.357, with no
measured gain), and phase-whitened cross-correlation, which recovers synthetic
shifts exactly but collapses the horizontal estimate to zero on real
cross-session pairs -- a failure that hides behind a perfect synthetic test.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from ._checker_roi import CheckerLattice, ColumnLattice

RefineMethod = Literal["rigid", "ecc", "frozen"]

#: Fraction of the ROI trimmed off each end before building the 1-D signals,
#: keeping the card's own outer border out of them.
SIGNAL_MARGIN = 0.05

#: Smoothing widths for the row and column signals, in pixels.
ROW_SMOOTH = 9
COL_SMOOTH = 7

#: A plateau shorter than this is noise, not a patch.  It is what rejects the
#: refracted ghost copies of a card seen through a plate wall: the ghosts are
#: too blurred to produce a qualifying plateau at all.
MIN_PLATEAU_PX = 20

#: Duty cycle (tile height as a fraction of pitch) searched by the phase fit.
DUTY_RANGE = (0.55, 0.94)


class RefineResult(NamedTuple):
    """Outcome of moving a lattice onto a frame.

    Attributes:
        lattice: The placed :class:`CheckerLattice`.
        dy: Vertical displacement applied, in pixels.
        dx: Horizontal displacement applied, in pixels.
        rot: Rotation applied, in radians.
        confidence: Method-specific: the phase-fit contrast score for
            ``"rigid"``, the ECC correlation for ``"ecc"``, ``1.0`` for
            ``"frozen"``.  Comparable only within a method.
        method: Which refinement produced it.
    """

    lattice: CheckerLattice
    dy: float
    dx: float
    rot: float
    confidence: float
    method: str


def _window(length: int, margin: float = SIGNAL_MARGIN) -> slice:
    """Central slice of an axis, trimming *margin* off each end."""
    cut = int(round(length * margin))
    return slice(cut, max(cut + 1, length - cut))


def column_signal(lab: np.ndarray, smooth: int = COL_SMOOTH) -> np.ndarray:
    """Per-column cross-channel Lab spread down the ROI.

    High where a column of patches runs down the image, low in the gutters
    and on uniform background.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of the ROI.
        smooth: Uniform-filter width applied to the profile.

    Returns:
        A ``(W,)`` profile.
    """
    from scipy.ndimage import uniform_filter1d

    rows = _window(lab.shape[0])
    spread = lab[rows].std(axis=0)
    return uniform_filter1d(np.sqrt((spread ** 2).sum(axis=1)), smooth)


def row_signal(
        lab: np.ndarray,
        x0: float,
        x1: float,
        smooth: int = ROW_SMOOTH,
) -> tuple[np.ndarray, int]:
    """Per-row Lab distance from a column's own background colour.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of the ROI.
        x0: Left edge of the column to profile.
        x1: Right edge of the column to profile.
        smooth: Uniform-filter width applied to the profile.

    Returns:
        ``(profile, offset)`` where *offset* is the ROI row the profile starts
        at, so ``profile[i]`` describes ROI row ``offset + i``.
    """
    from scipy.ndimage import uniform_filter1d

    rows = _window(lab.shape[0])
    lo, hi = int(max(0, x0)), int(min(lab.shape[1], x1))
    if hi <= lo:
        return np.zeros(0), rows.start
    strip = np.median(lab[rows, lo:hi], axis=1)
    distance = np.sqrt(((strip - np.median(strip, axis=0)) ** 2).sum(axis=1))
    return uniform_filter1d(distance, smooth), rows.start


def plateaus(
        signal: np.ndarray,
        threshold: float,
        min_length: int = MIN_PLATEAU_PX,
) -> list[tuple[int, int]]:
    """Runs where *signal* stays above *threshold* for at least *min_length*.

    Args:
        signal: A 1-D profile.
        threshold: Level a run must exceed.
        min_length: Shortest run counted, in samples.

    Returns:
        ``(start, stop)`` index pairs, half-open.
    """
    above = (np.asarray(signal) > threshold).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, above, 0]))
    return [
        (int(a), int(b))
        for a, b in zip(edges[::2], edges[1::2])
        if b - a >= min_length
    ]


#: Where the column threshold sits between the profile's background level and
#: its peak.  A midpoint threshold is set by the *strongest* column and so
#: misses a low-chroma one: on a real band the neutral column reads 23.5
#: against the saturated column's 46.4 over a background of 13, which a
#: midpoint (29.7) rejects and a 0.3 fraction (23.0) keeps.
COLUMN_THRESHOLD_FRACTION = 0.30


def _plateau_threshold(signal: np.ndarray) -> float:
    """Midpoint between the signal's floor and its peak."""
    return 0.5 * (float(np.percentile(signal, 5)) + float(signal.max()))


def _column_threshold(
        signal: np.ndarray,
        fraction: float = COLUMN_THRESHOLD_FRACTION,
) -> float:
    """Threshold set relative to the profile's background, not its midpoint."""
    background = float(np.median(signal))
    return background + fraction * (float(signal.max()) - background)


def edge_profile(lab: np.ndarray, x0: float, x1: float, smooth: int = ROW_SMOOTH):
    """Vertical Lab gradient magnitude down a column: one spike per patch edge.

    Used to measure the row pitch.  Unlike the distance-from-background
    profile, this stays periodic even when neighbouring patches are similar
    enough that their plateaus merge -- which is the case that makes plateau
    counting under-count rows on a real card.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of the ROI.
        x0: Left edge of the column.
        x1: Right edge of the column.
        smooth: Uniform-filter width.

    Returns:
        ``(profile, offset)``, indexed like :func:`row_signal`.
    """
    from scipy.ndimage import uniform_filter1d

    rows = _window(lab.shape[0])
    lo, hi = int(max(0, x0)), int(min(lab.shape[1], x1))
    if hi <= lo:
        return np.zeros(0), rows.start
    strip = np.median(lab[rows, lo:hi], axis=1)
    gradient = np.linalg.norm(np.gradient(strip, axis=0), axis=1)
    return uniform_filter1d(gradient, smooth), rows.start


def _pooled_pitch(
        profiles: list[np.ndarray],
        min_pitch: int = 40,
        max_pitch: int | None = None,
) -> float:
    """Row pitch from the pooled autocorrelation of several edge profiles.

    Pooling matters: on a real band, per-column autocorrelation returned the
    true pitch for two of four columns, a harmonic (2x) for one and a
    spurious short period for another.  Summing the normalised
    autocorrelations lets the columns that agree outvote the ones that do not.

    Harmonics are then resolved downward -- if half the winning lag also
    scores well, the card has twice as many rows as the raw peak suggests.

    Args:
        profiles: Edge profiles, one per candidate column.
        min_pitch: Shortest pitch considered, in pixels.
        max_pitch: Longest pitch considered.  Defaults to a third of the
            shortest profile.

    Returns:
        The pitch in pixels.

    Raises:
        ValueError: If no profile is long enough to measure a pitch.
    """
    usable = [p for p in profiles if p.size > 3 * min_pitch]
    if not usable:
        raise ValueError(
                "No column is tall enough to measure a row pitch; the ROI may "
                "be cropped too tightly around the card."
        )
    limit = max_pitch or max(min_pitch + 1, min(p.size // 3 for p in usable))

    pooled = np.zeros(limit)
    for profile in usable:
        centred = profile - profile.mean()
        correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
        correlation = correlation / (correlation[0] + 1e-9)
        pooled[: min(limit, correlation.size)] += correlation[:limit]

    peak = min_pitch + int(np.argmax(pooled[min_pitch:limit]))
    half = peak // 2
    if half >= min_pitch and pooled[half] > 0.6 * pooled[peak]:
        peak = half
    return float(peak)


def fit_phase(
        signal: np.ndarray,
        n_periods: int,
        pitch: float,
        start_range: tuple[float, float] | None = None,
        duty_range: tuple[float, float] = DUTY_RANGE,
        step: float = 0.5,
) -> tuple[float, float, float] | None:
    """Fit the phase and duty of an *n_periods* square wave of known pitch.

    Args:
        signal: A 1-D profile with one peak per tile.
        n_periods: Number of tiles expected along the axis.
        pitch: Tile-to-tile spacing, in samples.
        start_range: Bounds on the first tile's leading edge.
        duty_range: Bounds on the tile fraction of each period.
        step: Search step for the start position, in samples.

    Returns:
        ``(start, duty, score)`` maximising the mean signal inside the tiles
        minus the mean just outside them, or ``None`` if nothing fits.
    """
    samples = np.arange(len(signal), dtype=float)
    span = len(signal) - n_periods * pitch
    low, high = start_range if start_range else (0.0, span)
    low = max(low, -0.5 * pitch)
    high = min(high, span + 0.5 * pitch)
    if high < low:
        return None

    best: tuple[float, float, float] | None = None
    for duty in np.arange(*duty_range, 0.01):
        for start in np.arange(low, high, step):
            offset = (samples - start) % pitch
            inside = (
                (offset < duty * pitch)
                & (samples >= start)
                & (samples < start + n_periods * pitch)
            )
            outside = (
                ~inside
                & (samples >= start - 0.3 * pitch)
                & (samples < start + n_periods * pitch + 0.3 * pitch)
            )
            if inside.sum() < 10 or outside.sum() < 10:
                continue
            score = float(signal[inside].mean() - signal[outside].mean())
            if best is None or score > best[-1]:
                best = (float(start), float(duty), score)
    return best


def _pitch_for_row_count(
        lab: np.ndarray,
        col_runs: list[tuple[int, int]],
        n_rows: int,
) -> float:
    """Pitch that best explains *n_rows* tiles down each column.

    Two stages.  First the pooled edge-profile autocorrelation, searched only
    over pitches that *could* fit ``n_rows`` tiles in the ROI -- that range
    constraint is what removes the harmonics and spurious short periods a
    free autocorrelation returns (per-column peaks of 60, 256, 256 and 514 px
    were observed on one real frame, of which only 256 is real).  Then a
    local refinement scored by the same tile-versus-gutter contrast the phase
    fit uses.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of the ROI.
        col_runs: Detected ``(x0, x1)`` column extents.
        n_rows: Tiles expected down each column.

    Returns:
        The pitch, in pixels.

    Raises:
        ValueError: If the ROI is too short to hold *n_rows* tiles.
    """
    edges = [edge_profile(lab, x0, x1)[0] for x0, x1 in col_runs]
    rows = [row_signal(lab, x0, x1) for x0, x1 in col_runs]
    usable_edges = [e for e in edges if e.size > n_rows * 2]
    if not usable_edges:
        raise ValueError(f"ROI is too short to hold {n_rows} tiles down a column.")

    span = min(e.size for e in usable_edges)
    # A pitch below span/(3*n) would leave two thirds of the ROI empty; above
    # span/n the tiles cannot fit at all.
    low = max(4, int(span / (3.0 * n_rows)))
    high = max(low + 2, int(span / n_rows))

    pooled = np.zeros(high + 1)
    for profile in usable_edges:
        centred = profile - profile.mean()
        correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
        correlation = correlation / (correlation[0] + 1e-9)
        pooled[: min(high + 1, correlation.size)] += correlation[: high + 1]
    coarse = float(low + int(np.argmax(pooled[low : high + 1])))

    best: tuple[float, float] | None = None
    for pitch in np.arange(max(low, coarse - 6.0), min(high, coarse + 6.0) + 0.5, 0.5):
        total = 0.0
        for signal, _offset in rows:
            if signal.size <= n_rows:
                continue
            fitted = fit_phase(signal, n_rows, float(pitch))
            if fitted is not None:
                total += fitted[-1]
        if best is None or total > best[-1]:
            best = (float(pitch), total)
    return best[0] if best else coarse


def fit_lattice(
        lab: np.ndarray,
        grid: tuple[int, int] | None = None,
        min_plateau_px: int = MIN_PLATEAU_PX,
        column_threshold_fraction: float = COLUMN_THRESHOLD_FRACTION,
) -> CheckerLattice:
    """Build a lattice from an ROI alone, inferring the tile grid.

    Columns are the runs of :func:`column_signal` above a background-relative
    threshold.  The row pitch is the pooled autocorrelation of every column's
    :func:`edge_profile`, and the row count follows from the pitch and the
    span the edges cover.  Only then is the phase refined per column.  Nothing
    about the grid is supplied.

    This is the bootstrap path -- use it to build a prior on a clean reference
    frame, then :func:`refine` that prior onto production frames.  Check its
    output: a card whose columns differ greatly in contrast, or whose patches
    are too similar vertically to show edges, is where inference is hardest.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of one ROI.
        grid: Optional ``(nrows, ncols)`` of the tile block. Supplying it
            replaces the row-count inference with a far more reliable
            one-parameter pitch fit, and makes a wrong column count an error
            rather than a silently odd lattice. It says nothing about which
            chart patches are present or how the card is oriented -- that is
            always derived. Measured on six real bands, column detection is
            correct on all six while the inferred row count is not, so supply
            it when bootstrapping from a real card.
        min_plateau_px: Shortest run counted as a column.
        column_threshold_fraction: See :data:`COLUMN_THRESHOLD_FRACTION`.

    Returns:
        The fitted :class:`CheckerLattice`.

    Raises:
        ValueError: If no columns are found, or no pitch can be measured --
            which is what an ROI containing no card looks like.
    """
    col_profile = column_signal(lab)
    col_runs = plateaus(
            col_profile,
            _column_threshold(col_profile, column_threshold_fraction),
            min_plateau_px,
    )
    if not col_runs:
        raise ValueError(
                "No patch columns found in this ROI: the cross-channel spread "
                "signal has no run above threshold. Either the rectangle misses "
                "the card, or the card is too blurred to resolve."
        )

    edges = [edge_profile(lab, x0, x1)[0] for x0, x1 in col_runs]

    if grid is not None:
        n_rows, n_cols = grid
        if len(col_runs) != n_cols:
            raise ValueError(
                    f"Expected {n_cols} patch column(s) in this ROI but found "
                    f"{len(col_runs)} at {col_runs}. Either the rectangle spans "
                    "the wrong region or a column is too low-contrast to resolve."
            )
        pitch = _pitch_for_row_count(lab, col_runs, n_rows)
    else:
        pitch = _pooled_pitch(edges)
        span = len(edges[0])
        n_rows = max(1, int(round(span / pitch)))
        while n_rows > 1 and n_rows * pitch > span + 0.5 * pitch:
            n_rows -= 1

    columns = []
    for (x0, x1), profile in zip(col_runs, edges):
        row_profile, offset = row_signal(lab, x0, x1)
        fitted = fit_phase(row_profile, n_rows, pitch)
        if fitted is None:
            start, duty = float(offset), 0.8
        else:
            start, duty = fitted[0] + offset, fitted[1]
        columns.append(
                ColumnLattice(
                        x0=int(x0), x1=int(x1), start=float(start),
                        pitch=float(pitch), duty=float(duty),
                )
        )
    return CheckerLattice(columns=columns, nrows=n_rows)


def _anchor_index(lattice: CheckerLattice, roi_width: int) -> int:
    """Index of the column furthest from either ROI edge.

    That column is the one least likely to be clipped by the frame border, and
    so the one whose measured centre tracks the card's true displacement.
    """
    clearances = [
        min(column.x0, roi_width - column.x1) for column in lattice.columns
    ]
    return int(np.argmax(clearances))


def _anchor_search_window(
        lattice: CheckerLattice,
        index: int,
        roi_width: int,
        search_x: int,
) -> tuple[int, int]:
    """Horizontal search bounds for the anchor column.

    Clamped to the midpoints with the neighbouring columns, so a wide search
    on a densely-packed card cannot reach into the next column and lock onto
    it.  Without the clamp, columns spaced closer than ``2 * search_x`` let
    the neighbour's plateau win and the measured displacement is wrong by a
    whole column pitch -- silently, because the plateau looks perfectly
    healthy.
    """
    anchor = lattice.columns[index]
    low, high = anchor.x0 - search_x, anchor.x1 + search_x
    for other_index, other in enumerate(lattice.columns):
        if other_index == index:
            continue
        other_centre = (other.x0 + other.x1) / 2.0
        anchor_centre = (anchor.x0 + anchor.x1) / 2.0
        midpoint = (other_centre + anchor_centre) / 2.0
        if other_centre < anchor_centre:
            low = max(low, midpoint)
        else:
            high = min(high, midpoint)
    return int(max(0, low)), int(min(roi_width, high))


def refine_rigid(
        lab: np.ndarray,
        prior: CheckerLattice,
        anchor_col: int | None = None,
        search_x: int = 45,
        search_y: int = 55,
) -> RefineResult:
    """Move *prior* onto this frame as one rigid body, reference-free.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of the ROI.
        prior: Lattice from an earlier frame.
        anchor_col: Column to measure the horizontal shift on.  Defaults to
            the column furthest from the ROI edges.
        search_x: Horizontal search half-width, in pixels.
        search_y: Vertical search half-width, in pixels.

    Returns:
        A :class:`RefineResult`.
    """
    width = lab.shape[1]
    index = _anchor_index(prior, width) if anchor_col is None else anchor_col
    anchor = prior.columns[index]

    profile = column_signal(lab)
    low, high = _anchor_search_window(prior, index, width, search_x)
    segment = profile[low:high]
    anchor_centre = (anchor.x0 + anchor.x1) / 2.0
    runs = plateaus(segment, _plateau_threshold(segment), min_length=MIN_PLATEAU_PX)
    if runs:
        # Nearest to the prior, not longest: with the window clamped to the
        # anchor's own territory the true plateau is the nearest one, and
        # preferring length breaks ties toward whichever neighbour leaked in.
        centres = [(low + a + low + b) / 2.0 for a, b in runs]
        dx = min(centres, key=lambda c: abs(c - anchor_centre)) - anchor_centre
    else:
        dx = 0.0

    best: tuple[float, float] | None = None
    for column in prior.columns:
        row_profile, offset = row_signal(lab, column.x0 + dx, column.x1 + dx)
        if row_profile.size == 0:
            continue
        fitted = fit_phase(
                row_profile, prior.nrows, column.pitch,
                start_range=(
                    column.start - offset - search_y,
                    column.start - offset + search_y,
                ),
        )
        if fitted is None:
            continue
        start, _duty, score = fitted
        if best is None or score > best[-1]:
            best = (start + offset - column.start, score)
    dy, confidence = best if best else (0.0, 0.0)

    return RefineResult(
            lattice=prior.translated(dy=dy, dx=dx),
            dy=float(dy), dx=float(dx), rot=0.0,
            confidence=float(confidence), method="rigid",
    )


def _registration_image(lab: np.ndarray) -> np.ndarray:
    """Contrast-normalised Lab magnitude, as ECC expects."""
    magnitude = np.sqrt((np.asarray(lab, dtype=np.float64) ** 2).sum(axis=-1))
    return (
        (magnitude - magnitude.mean()) / (magnitude.std() + 1e-9)
    ).astype(np.float32)


def refine_ecc(
        lab: np.ndarray,
        prior: CheckerLattice,
        reference_lab: np.ndarray,
        iterations: int = 200,
        eps: float = 1e-6,
) -> RefineResult:
    """Register the ROI against a stored reference band with OpenCV ECC.

    Widest capture range of the three methods, and its correlation doubles as
    a confidence signal -- at the cost of shipping a reference band alongside
    the prior.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of the ROI on this frame.
        prior: Lattice as fitted on the reference frame.
        reference_lab: ``(H, W, 3)`` CIE Lab of the reference band.
        iterations: ECC iteration cap.
        eps: ECC termination tolerance.

    Returns:
        A :class:`RefineResult`.  On a registration failure the displacement
        is zero and the confidence is ``0.0``, so the QC gate refuses the
        band rather than a wrong lattice being used.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If the reference band is a different shape.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
                "refine='ecc' needs OpenCV (opencv-python). Use refine='rigid' "
                "for a reference-free alternative with no extra dependency."
        ) from exc

    if np.shape(reference_lab)[:2] != np.shape(lab)[:2]:
        raise ValueError(
                f"Reference band {np.shape(reference_lab)[:2]} and ROI "
                f"{np.shape(lab)[:2]} must be the same shape."
        )

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        correlation, warp = cv2.findTransformECC(
                _registration_image(reference_lab), _registration_image(lab),
                warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5,
        )
    except cv2.error:
        return RefineResult(prior, 0.0, 0.0, 0.0, 0.0, "ecc")

    dy, dx = float(warp[1, 2]), float(warp[0, 2])
    rot = float(np.arctan2(warp[1, 0], warp[0, 0]))
    lattice = prior.translated(dy=dy, dx=dx)
    lattice = lattice.model_copy(update={"rot": rot})
    return RefineResult(lattice, dy, dx, rot, float(correlation), "ecc")


def refine(
        lab: np.ndarray,
        prior: CheckerLattice,
        method: RefineMethod = "rigid",
        reference_lab: np.ndarray | None = None,
        anchor_col: int | None = None,
) -> RefineResult:
    """Move *prior* onto this frame by the named method.

    Args:
        lab: ``(H, W, 3)`` CIE Lab of the ROI.
        prior: Lattice from an earlier frame.
        method: ``"rigid"``, ``"ecc"`` or ``"frozen"``.
        reference_lab: Reference band, required by ``"ecc"``.
        anchor_col: Passed to :func:`refine_rigid`.

    Returns:
        A :class:`RefineResult`.

    Raises:
        ValueError: If *method* is unknown, or ``"ecc"`` is asked for without
            a reference band.
    """
    if method == "frozen":
        return RefineResult(prior, 0.0, 0.0, 0.0, 1.0, "frozen")
    if method == "rigid":
        return refine_rigid(lab, prior, anchor_col=anchor_col)
    if method == "ecc":
        if reference_lab is None:
            raise ValueError(
                    "refine='ecc' needs reference_lab, the band this lattice was "
                    "fitted on. Use refine='rigid' if no reference band is stored."
            )
        return refine_ecc(lab, prior, reference_lab)
    raise ValueError(
            f"Unknown refine method {method!r}; expected 'rigid', 'ecc' or 'frozen'."
    )
