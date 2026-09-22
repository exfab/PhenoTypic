"""Colour-chip detectors for the SnP plate rig.

The rig clamps a ColorChecker half-card (2 columns x 6 rows) at each edge of
every plate frame, in a near-identical position from frame to frame.  Two
properties of this geometry drive the design:

* the **outer** column of each half-card is clipped by the frame border;
* the transparent plate wall lies *inboard* of both cards — it does not cross
  either column — but it casts refracted ghost copies of the neutral column
  into the plate region of the right band (x ~100-150), which a blind
  segmenter could mistake for the real thing;
* the card does move a little between sessions (order 10 px), so a frozen
  crop is not enough either.

Detection is therefore posed as *constrained refinement of a stored rig
prior* rather than blind search.  All detectors share the prior lattice and
differ in how they refine it:

``M1_translation``  phase-correlation of the whole band against a reference
                    band; rigid (dy, dx) shift of the prior.
``M1_euclidean``    the same but with rotation, via OpenCV ECC.
``M2_profile``      reference-free: re-fits the periodic row lattice and the
                    column plateaus inside a window around the prior.
``M3_blob``         segments patch-like blobs and fits a lattice to their
                    centroids with a prior-seeded RANSAC.

Identity is a separate stage, not a locator: :func:`assign_identity` decides
which chart patch sits in each lattice cell by scoring the eight discrete
orientations a 2x6 half-card can take.  (An earlier ``M4_identity``, which
solved a free Hungarian assignment against a non-illuminant-adapted
reference, has been removed: a free permutation lets labels wander onto
whichever reference fits best.)
"""
from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage import color

NROWS = 6
NCOLS = 2


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------
@dataclass
class ColumnPrior:
    x0: int
    x1: int
    start: float     # band y of the first tile's top edge
    pitch: float     # tile-to-tile spacing down the column
    duty: float      # tile height as a fraction of pitch


@dataclass
class SidePrior:
    cols: list
    nrows: int = NROWS

    @classmethod
    def from_dict(cls, d):
        return cls([ColumnPrior(**c) for c in d["cols"]], d.get("nrows", NROWS))

    def to_dict(self):
        return {"cols": [asdict(c) for c in self.cols], "nrows": self.nrows}

    def boxes(self, dy=0.0, dx=0.0, core=0.0, rot=0.0):
        """Tile boxes in band coordinates.

        Args:
            dy, dx: rigid translation applied to the whole lattice.
            core: fraction of each box trimmed away (0.3 keeps the central 70%).
            rot: rotation in radians about the lattice centroid.

        Returns:
            List of ``(row, col, y0, y1, x0, x1)``.
        """
        out = []
        for j, c in enumerate(self.cols):
            h = c.duty * c.pitch
            for k in range(self.nrows):
                y0 = c.start + k * c.pitch
                out.append([k, j, y0, y0 + h, float(c.x0), float(c.x1)])
        arr = np.array([[o[2], o[3], o[4], o[5]] for o in out], float)
        if rot:
            cy = arr[:, :2].mean(); cx = arr[:, 2:].mean()
            ys = arr[:, :2].mean(axis=1) - cy
            xs = arr[:, 2:].mean(axis=1) - cx
            ny = ys * np.cos(rot) - xs * np.sin(rot) + cy
            nx = ys * np.sin(rot) + xs * np.cos(rot) + cx
            hh = (arr[:, 1] - arr[:, 0]) / 2
            ww = (arr[:, 3] - arr[:, 2]) / 2
            arr = np.column_stack([ny - hh, ny + hh, nx - ww, nx + ww])
        arr[:, :2] += dy
        arr[:, 2:] += dx
        if core:
            my = core * (arr[:, 1] - arr[:, 0]) / 2
            mx = core * (arr[:, 3] - arr[:, 2]) / 2
            arr = np.column_stack([arr[:, 0] + my, arr[:, 1] - my,
                                   arr[:, 2] + mx, arr[:, 3] - mx])
        return [(o[0], o[1], *a) for o, a in zip(out, arr)]

    def centers(self, **kw):
        b = self.boxes(**kw)
        return np.array([[(y0 + y1) / 2, (x0 + x1) / 2] for _, _, y0, y1, x0, x1 in b])


def load_prior(path="rig_prior.json"):
    d = json.load(open(path))
    return {s: SidePrior.from_dict(d[s]) for s in ("left", "right")}, d


# ---------------------------------------------------------------------------
# 1-D signals
# ---------------------------------------------------------------------------
def row_signal(lab, x0, x1, ys=(150, 1980), smooth=9):
    """Per-row Lab distance from the column's own background colour."""
    prof = np.median(lab[ys[0]:ys[1], int(x0):int(x1)], axis=1)
    d = np.sqrt(((prof - np.median(prof, axis=0)) ** 2).sum(axis=1))
    return uniform_filter1d(d, smooth), ys[0]


def col_signal(lab, ys=(300, 1850), smooth=7):
    """Per-column Lab standard deviation down the card."""
    s = lab[ys[0]:ys[1]].std(axis=0)
    return uniform_filter1d(np.sqrt((s ** 2).sum(axis=1)), smooth)


def fit_phase(sig, n, pitch, duty_rng=(0.55, 0.94), start_rng=None, step=0.5):
    """Fit the phase and duty of an n-period square wave of known pitch.

    Returns ``(start, duty, score)`` where *score* is the mean signal inside
    the tiles minus the mean just outside them.
    """
    y = np.arange(len(sig), dtype=float)
    lo, hi = start_rng if start_rng else (0.0, len(sig) - n * pitch)
    lo = max(lo, -0.5 * pitch); hi = min(hi, len(sig) - n * pitch + 0.5 * pitch)
    best = None
    for duty in np.arange(*duty_rng, 0.01):
        for start in np.arange(lo, hi, step):
            r = (y - start) % pitch
            ins = (r < duty * pitch) & (y >= start) & (y < start + n * pitch)
            out = (~ins) & (y >= start - 0.3 * pitch) & (y < start + n * pitch + 0.3 * pitch)
            if ins.sum() < 10 or out.sum() < 10:
                continue
            sc = sig[ins].mean() - sig[out].mean()
            if best is None or sc > best[-1]:
                best = (float(start), float(duty), float(sc))
    return best


def plateaus(sig, thr, minlen=25):
    m = (sig > thr).astype(np.int8)
    idx = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    return [(int(a), int(b)) for a, b in zip(idx[::2], idx[1::2]) if b - a >= minlen]


# ---------------------------------------------------------------------------
# Registration windows: the part of each band that is card, not plate or ghost
# ---------------------------------------------------------------------------
REG_WIN = {"left": (slice(150, 1980), slice(0, 205)),
           "right": (slice(150, 1980), slice(160, 340))}


def _prep(lab, side):
    """Contrast-normalised L*-plus-chroma image used for registration."""
    w = REG_WIN[side]
    l = lab[w]
    g = np.sqrt(l[..., 0] ** 2 + l[..., 1] ** 2 + l[..., 2] ** 2)
    g = (g - g.mean()) / (g.std() + 1e-9)
    return g.astype(np.float32)


# ---------------------------------------------------------------------------
# M1 -- registration of the band to a reference band
# ---------------------------------------------------------------------------
def M1_translation(lab, ref_lab, side, upsample=20, normalization=None):
    """Subpixel rigid shift of the prior by cross-correlation.

    ``normalization=None`` (plain masked cross-correlation) is the default:
    phase whitening recovers synthetic shifts exactly but collapses the
    horizontal estimate to zero on real cross-session pairs, where the
    illumination differs.

    Returns ``dict(dy, dx, rot=0, confidence)`` with *confidence* =
    ``1 - normalised registration error``.
    """
    from skimage.registration import phase_cross_correlation
    a, b = _prep(ref_lab, side), _prep(lab, side)
    shift, error, _ = phase_cross_correlation(a, b, upsample_factor=upsample,
                                              normalization=normalization)
    return dict(dy=float(-shift[0]), dx=float(-shift[1]), rot=0.0,
                confidence=float(1.0 - min(error, 1.0)))


def M1_euclidean(lab, ref_lab, side, iters=200, eps=1e-6):
    """Shift plus rotation via OpenCV ECC (MOTION_EUCLIDEAN).

    Returns ``dict(dy, dx, rot, confidence)`` with *rot* in radians and
    *confidence* the ECC correlation coefficient.
    """
    import cv2
    a, b = _prep(ref_lab, side), _prep(lab, side)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return dict(dy=np.nan, dx=np.nan, rot=np.nan, confidence=0.0)
    # warp[:, 2] is the reference -> moving translation in (x, y) order, which
    # is already the offset to add to the prior lattice
    rot = float(np.arctan2(warp[1, 0], warp[0, 0]))
    return dict(dy=float(warp[1, 2]), dx=float(warp[0, 2]), rot=rot,
                confidence=float(cc))


# ---------------------------------------------------------------------------
# M2 -- reference-free profile snap
# ---------------------------------------------------------------------------
def M2_profile(lab, prior, side, search_x=28, search_y=40, pitch_tol=0.0):
    """Re-fit the lattice from the band's own row/column profiles.

    Column extents are re-snapped from the per-column Lab standard deviation
    inside a window around the prior; the row phase is re-fitted as a
    6-period square wave with the prior's pitch.  No reference image is used,
    only the prior's search windows and the expected grid shape.

    Returns ``dict(prior, dy, dx, rot, confidence, scores)`` where *prior* is
    the refined :class:`SidePrior` and *dy*/*dx* are the mean shifts implied
    relative to the input prior.
    """
    cs = col_signal(lab)
    W = lab.shape[1]
    out_cols, dys, dxs, scores = [], [], [], []
    for j, c in enumerate(prior.cols):
        lo, hi = max(0, c.x0 - search_x), min(W, c.x1 + search_x)
        seg = cs[lo:hi]
        thr = 0.5 * (np.percentile(seg, 5) + seg.max())
        pl = plateaus(seg, thr, minlen=20)
        if pl:
            a, b = max(pl, key=lambda p: p[1] - p[0])
            nx0, nx1 = lo + a, lo + b
        else:
            nx0, nx1 = c.x0, c.x1
        pitch = c.pitch
        sig, off = row_signal(lab, nx0, nx1)
        best = fit_phase(sig, prior.nrows, pitch,
                         start_rng=(c.start - off - search_y, c.start - off + search_y))
        if best is None:
            best = (c.start - off, c.duty, 0.0)
        start, duty, sc = best
        out_cols.append(ColumnPrior(int(nx0), int(nx1), float(start + off), pitch, float(duty)))
        dys.append(start + off - c.start)
        dxs.append(((nx0 + nx1) - (c.x0 + c.x1)) / 2.0)
        scores.append(sc)
    refined = SidePrior(out_cols, prior.nrows)
    return dict(prior=refined, dy=float(np.mean(dys)), dx=float(np.mean(dxs)),
                rot=0.0, confidence=float(np.min(scores)), scores=scores)


#: For each side, the index of the column that is NOT clipped by the frame
#: border — the one further from the image edge.
ANCHOR_COL = {"left": 1, "right": 0}


def M2_rigid(lab, prior, side, search_x=45, search_y=55, anchor=None):
    """Reference-free rigid refinement anchored on the unclipped column.

    ``M2_profile`` re-snaps each column's extent independently, which makes it
    lag on the clipped outer column: as the card moves, that column's *visible*
    extent grows or shrinks, so its measured centre travels about half as far
    as the card does.  This variant measures the shift on the unclipped inner
    column only and applies it rigidly to the whole lattice, keeping the
    prior's tile widths.  Rows are taken from whichever column fits with the
    higher contrast score.

    Returns ``dict(prior, dy, dx, rot, confidence)``.
    """
    j = ANCHOR_COL[side] if anchor is None else anchor
    c = prior.cols[j]
    cs = col_signal(lab)
    lo, hi = max(0, c.x0 - search_x), min(lab.shape[1], c.x1 + search_x)
    seg = cs[lo:hi]
    pl = plateaus(seg, 0.5 * (np.percentile(seg, 5) + seg.max()), minlen=20)
    if pl:
        a, b = max(pl, key=lambda p: p[1] - p[0])
        dx = ((lo + a) + (lo + b)) / 2.0 - (c.x0 + c.x1) / 2.0
    else:
        dx = 0.0

    best = None
    for jj, cc in enumerate(prior.cols):
        x0, x1 = cc.x0 + dx, cc.x1 + dx
        sig, off = row_signal(lab, max(0, x0), min(lab.shape[1], x1))
        fit = fit_phase(sig, prior.nrows, cc.pitch,
                        start_rng=(cc.start - off - search_y, cc.start - off + search_y))
        if fit is None:
            continue
        start, duty, sc = fit
        if best is None or sc > best[-1]:
            best = (start + off - cc.start, sc)
    dy, score = best if best else (0.0, 0.0)

    cols = [ColumnPrior(int(round(cc.x0 + dx)), int(round(cc.x1 + dx)),
                        cc.start + dy, cc.pitch, cc.duty) for cc in prior.cols]
    return dict(prior=SidePrior(cols, prior.nrows), dy=float(dy), dx=float(dx),
                rot=0.0, confidence=float(score))


# ---------------------------------------------------------------------------
# M3 -- blob segmentation with a lattice fit
# ---------------------------------------------------------------------------
def blob_candidates(lab, med=9, win=21, border_q=0.75, erode=9,
                    min_area=2500, max_area=45000):
    """Segment patch cores by local-variance borders, natively in the band.

    This is the excolor border-fill idea run on the frame as captured: median
    filter, local Lab variance, threshold at a quantile (the absolute noise
    level differs by 2.5x between sessions, so a fixed threshold does not
    transfer), close, fill, erode, label.  No reflect padding, so no
    fabricated pixels.

    Returns ``(regions, mask)`` with *regions* a list of RegionProperties.
    """
    from scipy.ndimage import uniform_filter, median_filter, binary_fill_holes
    from skimage import measure, morphology

    sm = np.dstack([median_filter(lab[..., c], size=med) for c in range(3)])
    var = np.zeros(lab.shape[:2])
    for ch in range(3):
        m = uniform_filter(sm[..., ch], win)
        var += uniform_filter(sm[..., ch] ** 2, win) - m ** 2
    std = np.sqrt(np.maximum(var, 0))

    border = std > np.quantile(std, border_q)
    border = morphology.closing(border, morphology.footprint_rectangle((11, 3)))
    border[:, :2] = True; border[:, -2:] = True
    border[:2] = True; border[-2:] = True
    inside = binary_fill_holes(border) & ~border
    inside = morphology.erosion(inside, morphology.footprint_rectangle((erode, erode)))

    lbl = measure.label(inside)
    regs = [r for r in measure.regionprops(lbl)
            if min_area <= r.area <= max_area
            and r.solidity > 0.85
            and 1.2 < (r.bbox[2] - r.bbox[0]) / max(r.bbox[3] - r.bbox[1], 1) < 7.0]
    return regs, inside


def M3_blob(lab, prior, side, tol=35.0, search=60, **kw):
    """Fit the prior lattice to segmented blob centroids by translation RANSAC.

    Every blob centroid proposes a translation onto every prior node; the
    translation with the most inliers (within *tol* px, at most *search* px
    from the prior) wins, and the lattice is then refined on its inliers.
    Occluded or missed tiles are recovered from the fitted lattice.

    Returns ``dict(prior, dy, dx, rot, confidence, n_blobs, n_inliers)``; the
    returned lattice carries all twelve tiles, including any the segmentation
    missed.
    """
    regs, _ = blob_candidates(lab, **kw)
    nodes = prior.centers()
    ncols = len(prior.cols)
    if not regs:
        return dict(prior=prior, dy=0.0, dx=0.0, rot=0.0, confidence=0.0,
                    n_blobs=0, n_inliers=0)
    cents = np.array([r.centroid for r in regs])

    best = (0, 0.0, 0.0, None)
    for c in cents:
        for nd in nodes:
            d = c - nd
            if abs(d[0]) > search or abs(d[1]) > search:
                continue
            dd = np.abs(cents[:, None, :] - (nodes[None, :, :] + d))
            hit = (dd[..., 0] < tol) & (dd[..., 1] < tol)
            n = int(hit.any(axis=1).sum())
            if n > best[0]:
                best = (n, d[0], d[1], hit)
    n_in, dy, dx, hit = best

    # Per-column offsets: a single global dx does not fit.  Measured on the
    # reference frame, blob centroids sit toward the frame border relative to
    # their prior node on both sides (left residuals -9..-4 px at node x=38.5,
    # right +1..+6 px at node x=310), so the two columns of a side need
    # separate x offsets.
    dys, dxs = [], [[] for _ in range(ncols)]
    if hit is not None:
        order = [(k, j) for j in range(ncols) for k in range(prior.nrows)]
        for i, nidx in zip(*np.nonzero(hit)):
            dys.append(cents[i, 0] - nodes[nidx, 0])
            dxs[order[nidx][1]].append(cents[i, 1] - nodes[nidx, 1])
    dy = float(np.median(dys)) if dys else float(dy)
    cols = []
    for j, c in enumerate(prior.cols):
        ox = float(np.median(dxs[j])) if dxs[j] else float(dx)
        cols.append(ColumnPrior(int(round(c.x0 + ox)), int(round(c.x1 + ox)),
                                c.start + dy, c.pitch, c.duty))
    return dict(prior=SidePrior(cols, prior.nrows), dy=dy,
                dx=float(np.mean([np.median(d) if d else dx for d in dxs])), rot=0.0,
                confidence=n_in / len(nodes), n_blobs=len(regs), n_inliers=int(n_in))


# ---------------------------------------------------------------------------
# Reference chart and tile identity
# ---------------------------------------------------------------------------
#: Which chart row each (side, column index) holds, and the running order.
#: The card is mounted rotated, so the chart's six columns run down the image
#: and its four rows are split two per card.
LAYOUT = {("left", 0): 2, ("left", 1): 1, ("right", 0): 4, ("right", 1): 3}
CHART_COLS = "ABCDEF"


def load_reference(path="ColorChecker24_After_Nov2014.txt"):
    """Read the X-Rite reference file into ``{'A1': (L, a, b), ...}``."""
    ref = {}
    for line in open(path, encoding="latin-1"):
        parts = line.strip().split("\t")
        if len(parts) == 4 and len(parts[0]) == 2 and parts[0][0] in CHART_COLS:
            ref[parts[0]] = tuple(float(p.replace(",", ".")) for p in parts[1:])
    if len(ref) != 24:
        raise ValueError(f"expected 24 reference patches, parsed {len(ref)}")
    return ref


def tile_ids(side, nrows=NROWS):
    """Chart identifiers for the tiles of one side, in (row, col) order.

    This is the layout the rig happens to use. It is the *expected* answer, not
    an assumption baked into detection: :func:`assign_identity` recovers it
    from colour without being told.
    """
    return [f"{CHART_COLS[k]}{LAYOUT[(side, j)]}" for j in (0, 1) for k in range(nrows)]


@lru_cache(maxsize=4)
def reference_linear_rgb(illuminant="D65"):
    """Reference chart as linear sRGB, chromatically adapted to *illuminant*.

    Reads the chart from colour-science rather than a local Lab dump, and
    Bradford-adapts from the chart's own illuminant (D50) to the working one.
    Skipping that adaptation — e.g. by pushing the D50 Lab values through a
    D65-assuming ``lab2rgb`` — displaces the saturated patches by up to 6.4
    Lab units (A3 6.38, B2 5.60, F3 4.83) while leaving the neutrals within
    0.27.

    Scope of that claim: the displacement is measured, the consequence is not.
    Swapping the un-adapted reference for this one changed *no* labels in the
    A/B on this rig — both gave left 12/12 and right 8/12; the mislabels were
    fixed by the scoring change in :func:`assign_identity`, not by adaptation.
    Adapt anyway, because the un-adapted comparison is wrong on principle and
    the error sits precisely where the patches are closest together, but do
    not credit it with a correctness gain it has not been shown to deliver.
    """
    import colour
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    wp = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][illuminant]
    names = list(cc.data.keys())
    XYZ = colour.adaptation.chromatic_adaptation_VonKries(
        colour.xyY_to_XYZ(np.array([cc.data[n] for n in names])),
        colour.xy_to_XYZ(cc.illuminant), colour.xy_to_XYZ(wp), transform="Bradford")
    lin = colour.XYZ_to_sRGB(XYZ, illuminant=wp, apply_cctf_encoding=False)
    return {f"{CHART_COLS[i % 6]}{i // 6 + 1}": lin[i] for i in range(24)}


def orientation_hypotheses():
    """Every discrete way a 2x6 half-card can sit in the lattice."""
    H = []
    for rows in ((1, 2), (2, 1), (3, 4), (4, 3)):
        for flip in (False, True):
            order = list(CHART_COLS)[::-1] if flip else list(CHART_COLS)
            H.append((f"rows{rows}{'_flip' if flip else ''}",
                      [f"{c}{rows[0]}" for c in order] + [f"{c}{rows[1]}" for c in order]))
    return H


def _identity_features(lin_rgb):
    """Gain-invariant colour features: chromaticity plus relative luminance."""
    s = lin_rgb.sum(1, keepdims=True) + 1e-9
    return np.column_stack([lin_rgb[:, :2] / s * 3.0,
                            lin_rgb[:, 1] / (lin_rgb[:, 1].max() + 1e-9)])


def assign_identity(obs_lin, clean=None, ref=None):
    """Recover which chart patch sits in each lattice cell — colour-agnostically.

    Detection (the M-methods) never uses patch colour, so a rotated or swapped
    card is still located. This step then decides *which* patch is where by
    enumerating the discrete orientations and scoring each one directly:
    under a hypothesis, lattice position k must be ``ids[k]``. Scoring
    directly rather than solving a free assignment matters — a free
    permutation lets labels wander onto whichever reference fits best and
    silently mislabels occluded tiles.

    Args:
        obs_lin: ``(12, 3)`` observed linear RGB per tile, in lattice order.
        clean: optional boolean mask; only these tiles vote on the hypothesis.
            Occluded tiles are still labelled, they just do not choose.
        ref: reference map; defaults to :func:`reference_linear_rgb`.

    Returns:
        ``dict(hypothesis, assignment, score, runner_up, margin)``. *margin* is
        the gap to the second-best hypothesis. Measured over 40 bands with all
        twelve tiles voting: left 0.365-0.409, right 0.367-0.484. A margin
        below ~0.1 means the orientation is not determined by the tiles that
        voted.

    Known limitation of *clean*: on the one real occlusion in the test set
    (five of six neutrals covered), restricting the vote to the six clean
    tiles picks the wrong hypothesis and labels 0/12 correctly, at margin
    0.044. Six same-row tiles do not constrain the orientation. The low
    margin is the only thing that catches this, so treat ``margin < 0.1`` as
    "orientation undetermined" and fall back to the expected layout rather
    than to the returned assignment. With all twelve tiles voting the same
    band is labelled 12/12 correctly, so the mask is worth passing only when
    a majority of tiles are clean.
    """
    ref = ref or reference_linear_rgb()
    clean = np.ones(len(obs_lin), bool) if clean is None else np.asarray(clean)
    fo = _identity_features(np.asarray(obs_lin, float))
    scored = []
    for name, ids in orientation_hypotheses():
        fe = _identity_features(np.vstack([ref[t] for t in ids]))
        scored.append((float(np.linalg.norm(fo[clean] - fe[clean], axis=1).mean()), name, ids))
    scored.sort(key=lambda t: t[0])
    return dict(score=scored[0][0], hypothesis=scored[0][1], assignment=scored[0][2],
                runner_up=scored[1][0], margin=scored[1][0] - scored[0][0])


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure_tiles(rgb, side_prior, side, core=0.4):
    """Measure every tile of a side from a detected lattice.

    Returns a list of dicts with the tile's chart id, box, median linear RGB,
    median Lab, and the within-tile dE spread from its own median.

    Encoding, stated once because everything here depends on it. *rgb* must be
    the **linear** band (RawTherapee 16-bit / 65535). R/G/B are reported as
    plain medians of it, and those are the values the identity stage and the
    downstream fit consume.

    The ``L``/``a``/``b`` and ``spread_dE`` fields are a different matter: they
    come from ``skimage.rgb2lab``, which expects sRGB-encoded input and is
    being handed linear values. They are therefore a *consistent internal
    contrast measure*, not calibrated CIE Lab or CIE dE2000, and the same is
    true of every threshold expressed in those units here (``tile_impurity``'s
    6.0, the veil analysis). Re-running the impurity check with a properly
    sRGB-encoded band changes no gate decision on the test set — the same five
    tiles exceed 5 % on the occluded band and none elsewhere — and gives
    slightly smaller numbers, so the thresholds as set are the conservative
    side of the choice. Do not quote these figures as perceptual dE2000.
    """
    from skimage.color import rgb2lab, deltaE_ciede2000
    H, W = rgb.shape[:2]
    ids = tile_ids(side, side_prior.nrows)
    out = []
    for (k, j, y0, y1, x0, x1), tid in zip(side_prior.boxes(core=core), ids):
        ys, ye = max(0, int(round(y0))), min(H, int(round(y1)))
        xs, xe = max(0, int(round(x0))), min(W, int(round(x1)))
        patch = rgb[ys:ye, xs:xe]
        if patch.size == 0:
            out.append(dict(tile=tid, row=k, col=j, n_px=0))
            continue
        lab = rgb2lab(patch.reshape(-1, 1, 3)).reshape(-1, 3)
        med = np.median(lab, axis=0)
        spread = deltaE_ciede2000(np.broadcast_to(med, lab.shape), lab)
        out.append(dict(tile=tid, row=int(k), col=int(j), n_px=int(patch.shape[0] * patch.shape[1]),
                        y0=float(y0), y1=float(y1), x0=float(x0), x1=float(x1),
                        R=float(np.median(patch[..., 0])), G=float(np.median(patch[..., 1])),
                        B=float(np.median(patch[..., 2])),
                        L=float(med[0]), a=float(med[1]), b=float(med[2]),
                        spread_dE=float(np.median(spread))))
    return out


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
METHODS = ("M0_frozen", "M1_translation", "M1_euclidean", "M2_profile",
           "M2_rigid", "M3_blob", "M23_cascade")


def detect(lab, prior, side, method="M1_translation", ref_lab=None):
    """Locate the twelve tiles of one half-card.

    Args:
        lab: Lab image of the band.
        prior: :class:`SidePrior` for this side.
        side: ``"left"`` or ``"right"``.
        method: one of :data:`METHODS`.
        ref_lab: reference band, required by the M1 methods.

    Returns:
        ``dict`` with the refined ``prior``, the implied ``dy``/``dx``/``rot``,
        a method-specific ``confidence``, and any extra diagnostics.
    """
    if method == "M0_frozen":
        return dict(prior=prior, dy=0.0, dx=0.0, rot=0.0, confidence=1.0)
    if method == "M1_translation":
        r = M1_translation(lab, ref_lab, side)
    elif method == "M1_euclidean":
        r = M1_euclidean(lab, ref_lab, side)
    elif method == "M2_profile":
        return M2_profile(lab, prior, side)
    elif method == "M2_rigid":
        return M2_rigid(lab, prior, side)
    elif method == "M3_blob":
        return M3_blob(lab, prior, side)
    elif method == "M23_cascade":
        # coarse: blob RANSAC pulls the prior into range; fine: rigid snap
        coarse = M3_blob(lab, prior, side)
        fine = M2_rigid(lab, coarse["prior"], side)
        fine["dy"] += coarse["dy"]
        fine["dx"] += coarse["dx"]
        fine["n_inliers"] = coarse["n_inliers"]
        fine["confidence"] = min(fine["confidence"] / 20.0, 1.0) * coarse["confidence"]
        return fine
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    cols = [ColumnPrior(int(round(c.x0 + r["dx"])), int(round(c.x1 + r["dx"])),
                        c.start + r["dy"], c.pitch, c.duty) for c in prior.cols]
    r["prior"] = SidePrior(cols, prior.nrows)
    return r


def detect_frame(img, priors, ref_bands, method="M1_euclidean", core=0.4,
                 band_y=(950, 3100), band_w=340, ref=None, qc=True):
    """Locate and measure all 24 tiles of one plate frame.

    Args:
        img: full frame, float RGB in [0, 1].
        priors: ``{"left": SidePrior, "right": SidePrior}`` from :func:`load_prior`.
        ref_bands: ``{"left": lab, "right": lab}`` reference bands for the M1 methods.
        method: detector to use (see :data:`METHODS`).
        core: fraction of each tile box trimmed before measuring.
        band_y, band_w: search-band geometry, matching the stored prior.
        ref: parsed reference chart; read from disk when omitted.
        qc: also run :func:`qc_record` on each side.

    Returns:
        ``dict`` with ``tiles`` (24 measurement dicts, chart-id keyed by
        ``tile``), ``fits`` (the refined lattice per side) and ``qc``.
    """
    from skimage.color import rgb2lab
    if ref is None:
        ref = load_reference()
    y0, y1 = band_y
    W = img.shape[1]
    out = {"tiles": [], "fits": {}, "qc": {}}
    for side, x0 in (("left", 0), ("right", W - band_w)):
        rgb = img[y0:y1, x0:x0 + band_w]
        lab = rgb2lab(rgb)
        r = detect(lab, priors[side], side, method, ref_lab=ref_bands[side])
        out["fits"][side] = r["prior"]
        for m in measure_tiles(rgb, r["prior"], side, core=core):
            m.update(side=side, band_y0=y0, band_x0=x0)
            out["tiles"].append(m)
        if qc:
            out["qc"][side] = qc_record(rgb, lab, priors[side], side,
                                        ref_bands[side], ref=ref)
    return out


# ---------------------------------------------------------------------------
# QC gate for unattended batches
# ---------------------------------------------------------------------------
#: ``min_identity`` is 10, not 12: on real bands the Hungarian match swaps at
#: most one adjacent pair of genuinely similar patches (purplish blue vs blue
#: flower on the left card; the two mid-grey neutrals on the right), so the
#: check detects a missing, flipped or wrong card (0-1 of 12) rather than
#: certifying every individual tile.
#: ``min_identity`` is 10, not 12: on real bands the Hungarian match swaps at
#: most one adjacent pair of genuinely similar patches (purplish blue vs blue
#: flower on the left card; the two mid-grey neutrals on the right), so the
#: check detects a missing, flipped or wrong card (0-1 of 12) rather than
#: certifying every individual tile.
#:
#: ``max_impurity`` is the occlusion detector.  On the one real obstruction
#: event in the 20-frame subset (a bright object sweeping across the right
#: card's neutral column) it reads 18.9-28.3 % against <= 1.13 % on all 39
#: other bands.  The spread, ECC and identity signals also fire on that band,
#: so it is corroboration rather than the sole catch — but it is the only one
#: of the four that both needs no reference image AND measures the tiles
#: themselves.  That matters because M2_rigid's own confidence on that band
#: (35.1) sits inside its clean-band range (25.4-36.0), so the reference-free
#: primary cannot report an occlusion by itself.
#:
#: ``max_disagreement`` is 8 px for the reference panel and 12 px for the
#: reference-free one.  Both are calibrated on the subset: the reference panel
#: spans <= 5.8 px on clean bands and 10.4 px on the occluded one; the
#: reference-free pair spans <= 7.8 px clean and 22.2 px occluded.  Earlier
#: versions polled M2_profile and M3_blob as well, whose column estimates are
#: erratic, and consequently flagged 16 of 40 clean bands.
QC_LIMITS = dict(max_shift=30.0, max_disagreement=8.0, min_ecc=0.90,
                 min_identity=10, max_impurity=0.05, max_tile_impurity=0.05,
                 max_robust_shift=1.5, max_clipped=0.20)

#: The impurity signals answer *is something there*; ``max_robust_shift``
#: answers *did it move the answer*, and only the second is grounds to reject
#: a band.  A per-channel median already absorbs a minority of outliers: on
#: tile F4 of ``d000378`` 8.7 % of pixels are contaminated, yet the median
#: shifts by 0.57 dE and lands 3.6 dE from the clean-frame consensus against a
#: 2.8 dE spread among the clean frames themselves — detectable, not harmful.
#: The barcode sticker shifts the same statistic by 2.5-18 dE.  So per-tile
#: impurity is reported as a WARNING and the robust shift decides.
#: ``max_clipped`` is a third, independent fault: a channel pinned at the
#: sensor floor is not an outlier, so neither impurity nor the robust shift
#: sees it, and no estimator can recover a value that was never recorded.

#: ``max_impurity`` tests the 12-tile MEAN, ``max_tile_impurity`` the worst
#: single tile.  Both are needed and they catch different things.  The mean
#: only moves when a defect covers much of the card — the barcode sticker did,
#: across five of six neutrals.  A defect confined to two or three tiles is
#: diluted by the nine clean ones and passes: on the Sept-2025 calibration
#: band ``d000220_300_038`` right, tiles C3 0.144, B3 0.072 and F4 0.061 all
#: exceed the per-tile limit while the mean is 0.028 and the band passed.
#: Testing the worst tile as well closes that gap.  The limit is mode-aware
#: because the two panels place boxes differently.  Over the 48-band set,
#: worst-tile impurity on the 38 bands with no known defect reaches 0.045 in
#: reference-free mode but 0.057 in reference mode, the latter a cluster of
#: ~0.052-0.057 readings on tile C4 across nine frames — an artefact of
#: ``M1_euclidean``'s box placement catching a patch edge, not an occluder.
#: The mildest genuine defect in the set is 0.063 (tile C2 of the Sept-2025
#: reference frame, a third of its green channel floored).  So reference-free
#: separates the two populations by a factor of 1.4 at a 0.05 limit, while
#: reference mode needs 0.06 and separates them by only 0.006 — one more
#: reason to prefer the reference-free panel.

#: Methods polled for the displacement-spread signal.
#: Per-tile impurity limit by mode; see the note on QC_LIMITS.
TILE_IMPURITY_LIMIT = {"reference": 0.06, "reference_free": 0.05}

QC_PANEL = {"reference": ("M1_translation", "M1_euclidean", "M2_rigid"),
            "reference_free": ("M2_rigid", "M2_profile")}


def tile_impurity(rgb, side_prior, core=0.4, thr=6.0, med=9):
    """Fraction of core-box pixels far from their tile's own median colour.

    Computed on a median-filtered copy so the number reflects spatial
    contamination — an occluder, a border, a reflection crossing the tile —
    rather than sensor noise.  Returns the per-tile array.
    """
    from scipy.ndimage import median_filter
    from skimage.color import rgb2lab, deltaE_ciede2000
    sm = np.dstack([median_filter(rgb[..., c], size=med) for c in range(3)])
    H, W = sm.shape[:2]
    out = []
    for _, _, y0, y1, x0, x1 in side_prior.boxes(core=core):
        ys, ye = max(0, int(y0)), min(H, int(y1))
        xs, xe = max(0, int(x0)), min(W, int(x1))
        p = sm[ys:ye, xs:xe]
        if p.size == 0:
            out.append(np.nan); continue
        l = rgb2lab(p.reshape(-1, 1, 3)).reshape(-1, 3)
        d = deltaE_ciede2000(np.broadcast_to(np.median(l, axis=0), l.shape), l)
        out.append(float((d > thr).mean()))
    return np.array(out)


def ids_over(imp, meas, limit):
    """Chart ids of the tiles whose impurity exceeds *limit*, worst first."""
    hits = [(float(v), m.get("tile", "?")) for v, m in zip(imp, meas) if v > limit]
    return ", ".join(t for _, t in sorted(hits, reverse=True))


def tile_clipped(rgb, side_prior, core=0.4, lo=1/65535, hi=0.999):
    """Fraction of each tile's pixels with any channel at the sensor's floor
    or ceiling.

    A different fault from contamination and invisible to both
    ``tile_impurity`` and :func:`tile_robust_shift`: clipping is uniform, not
    an outlier, so the median sits squarely on the clipped value and no
    robustness helps.  All three Sept-2025 calibration frames are affected —
    tile B3's red channel reads exactly zero over two thirds of the tile
    against a reference of 0.046 — including the frame used as the stored
    registration reference.  The 20 plate frames are clean.
    """
    H, W = rgb.shape[:2]
    out = []
    for _, _, y0, y1, x0, x1 in side_prior.boxes(core=core):
        p = rgb[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))]
        if p.size == 0:
            out.append(np.nan); continue
        out.append(float(((p <= lo) | (p >= hi)).any(-1).mean()))
    return np.array(out)


def tile_robust_shift(rgb, side_prior, core=0.4, keep=0.6, med=9):
    """How far each tile's median colour moves when atypical pixels are dropped.

    ``tile_impurity`` counts contaminated pixels; this measures whether that
    contamination actually moved the answer.  A per-channel median already
    absorbs a minority of outliers, so a tile can be visibly contaminated and
    still report the right colour — the two numbers answer different
    questions, and the gate needs both: impurity says *something is there*,
    this says *and it shifted the measurement*.

    Returns the per-tile dE between the median over all core pixels and the
    median over the *keep* fraction closest to the tile's own centre.
    """
    from scipy.ndimage import median_filter
    from skimage.color import rgb2lab, deltaE_ciede2000
    sm = np.dstack([median_filter(rgb[..., c], size=med) for c in range(3)])
    H, W = sm.shape[:2]
    out = []
    for _, _, y0, y1, x0, x1 in side_prior.boxes(core=core):
        ys, ye = max(0, int(y0)), min(H, int(y1))
        xs, xe = max(0, int(x0)), min(W, int(x1))
        p = sm[ys:ye, xs:xe].reshape(-1, 3)
        if len(p) < 20:
            out.append(np.nan); continue
        lab = rgb2lab(p.reshape(-1, 1, 3)).reshape(-1, 3)
        centre = np.median(lab, axis=0)
        d = deltaE_ciede2000(np.broadcast_to(centre, lab.shape), lab)
        sel = d <= np.quantile(d, keep)
        a = rgb2lab(np.median(p, axis=0).reshape(1, 1, 3))
        b = rgb2lab(np.median(p[sel], axis=0).reshape(1, 1, 3))
        out.append(float(deltaE_ciede2000(a, b)[0, 0]))
    return np.array(out)


def qc_record(rgb, lab, prior, side, ref_lab=None, limits=None, ref=None,
              mode="reference"):
    """Run the detector panel on one band and decide whether to trust it.

    Five signals have to pass: displacement from the prior, registration
    correlation (reference mode only), spread between the panel's estimates,
    colour identity, and tile impurity.  Any one failing flags the band.

    Args:
        mode: ``"reference"`` uses ECC as primary and polls the reference
            panel; ``"reference_free"`` uses ``M2_rigid`` as primary, polls
            the reference-free pair, and skips the ECC signal entirely
            (``ref_lab`` may then be None).

    Returns a dict of the measured signals plus ``ok`` and ``flags``.
    """
    lim = dict(QC_LIMITS, max_tile_impurity=TILE_IMPURITY_LIMIT[mode], **(limits or {}))
    if mode == "reference_free":
        lim.setdefault("_disagree", 12.0)
        lim["max_disagreement"] = (limits or {}).get("max_disagreement", 12.0)
    panel = QC_PANEL[mode]
    res = {m: detect(lab, prior, side, m, ref_lab=ref_lab) for m in panel}
    dys = np.array([res[m]["dy"] for m in panel])
    dxs = np.array([res[m]["dx"] for m in panel])
    disagree = float(max(dys.max() - dys.min(), dxs.max() - dxs.min()))

    primary = "M1_euclidean" if mode == "reference" else "M2_rigid"
    best = res[primary]
    meas = measure_tiles(rgb, best["prior"], side)
    obs = np.array([[m.get("R", np.nan), m.get("G", np.nan), m.get("B", np.nan)]
                    for m in meas])
    ident_res = assign_identity(obs, ref=reference_linear_rgb())
    expected = tile_ids(side, prior.nrows)
    ident = dict(n_correct=int(sum(a == e for a, e in zip(ident_res["assignment"], expected))),
                 hypothesis=ident_res["hypothesis"], margin=ident_res["margin"])
    imp = tile_impurity(rgb, best["prior"])
    imp_mean = float(np.nanmean(imp))

    flags, warnings_ = [], []
    shift0 = float(np.hypot(best["dy"], best["dx"]))
    if shift0 > lim["max_shift"]:
        flags.append(f"displacement {shift0:.1f} px > {lim['max_shift']}")
    if disagree > lim["max_disagreement"]:
        flags.append(f"methods disagree by {disagree:.1f} px")
    if mode == "reference" and res["M1_euclidean"]["confidence"] < lim["min_ecc"]:
        flags.append(f"ECC correlation {res['M1_euclidean']['confidence']:.2f} low")
    if ident["n_correct"] < lim["min_identity"]:
        flags.append(f"identity {ident['n_correct']}/12")
    if ident["margin"] < 0.1:
        flags.append(f"orientation undetermined (margin {ident['margin']:.3f})")
    imp_max = float(np.nanmax(imp))
    n_over = int(np.nansum(np.asarray(imp) > lim["max_tile_impurity"]))
    if imp_mean > lim["max_impurity"]:
        flags.append(f"tile impurity {imp_mean*100:.1f} % mean — something is covering the card")
    shift = tile_robust_shift(rgb, best["prior"])
    clip = tile_clipped(rgb, best["prior"])
    sh_max, cl_max = float(np.nanmax(shift)), float(np.nanmax(clip))
    if sh_max > lim["max_robust_shift"]:
        flags.append(f"tile colour displaced by contamination, worst {sh_max:.1f} dE "
                     f"({ids_over(shift, meas, lim['max_robust_shift'])})")
    if cl_max > lim["max_clipped"]:
        flags.append(f"channel clipped at the sensor limit on {int(np.nansum(clip > lim['max_clipped']))} "
                     f"tile(s), worst {cl_max*100:.0f} % of pixels "
                     f"({ids_over(clip, meas, lim['max_clipped'])})")
    if imp_max > lim["max_tile_impurity"]:
        warnings_.append(f"{n_over} tile(s) impure, worst {imp_max*100:.1f} % "
                         f"({ids_over(imp, meas, lim['max_tile_impurity'])}) — "
                         "contamination present; see robust_shift_max for its effect")
    return dict(side=side, mode=mode, shift_px=shift0, disagreement_px=disagree,
                robust_shift_max=sh_max, clipped_max=cl_max,
                warnings="; ".join(warnings_),
                ecc_cc=res["M1_euclidean"]["confidence"] if mode == "reference" else np.nan,
                identity=ident["n_correct"], impurity_mean=imp_mean,
                impurity_max=imp_max, n_tiles_impure=n_over,
                ok=not flags, flags="; ".join(flags))
