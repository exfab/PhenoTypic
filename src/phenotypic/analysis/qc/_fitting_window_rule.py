"""Rule D — a growth-curve fitting window read off a replicate-CV time series.

A numpy-only transcription of a polars reference implementation, two scripts
kept outside this repository:

* ``prototype_rule_d.py`` — ``_derivatives``, ``abrupt_bar``, ``_sustained``
  and ``rule_d``.
* ``c10_cv_by_ph.py`` — the three names Rule D imports from it:
  ``_EDGE_FRAMES``, ``_MIN_SERIES_FRAMES`` and ``_cv_series``, whose body is
  what :func:`mean_cv_series` reproduces.

Docstrings below cite those scripts by name and function. Every deviation from
them is stated where it occurs.

What the rule does, in the reference's own words: one **shared absolute** bar
for every condition, a window that *opens* at the left edge of the post-hump CV
minimum's basin, *continues* through gradual drift of any magnitude, *ends* at
the first sustained adverse acceleration excursion or at the first sustained
rise in the raw series' roughness, and *abstains* entirely on a condition whose
own acceleration noise swallows the shared bar.

Three numerical properties are load-bearing and easy to lose in a port:

* **The bar is pooled over every series, not computed per series.**
  :func:`abrupt_bar` concatenates every admissible series' ``|acceleration|``
  and returns **one** float: "Pooled rather than per-curve: a per-curve bar is
  exactly the unequal treatment this rule exists to remove." A per-series bar
  changes windows silently. One consequence: **a condition alone in its run is
  never too noisy** at the default settings (see :func:`abrupt_bar`), so the
  :attr:`WindowOutcome.TOO_NOISY` exit only appears alongside other
  conditions.
* **The rolling means are polars' ``center=True`` rolling means**, which are
  *null* unless the whole window is present — see
  :func:`_rolling_mean_centered`.
* **The CV uses the sample standard deviation** (``ddof=1``), because that is
  what polars' ``Series.std()`` computes. ``numpy.std`` defaults to ``ddof=0``
  and would scale every CV by ``sqrt((n - 1) / n)`` — 18% low at ``n = 3``, the
  common case.

Nothing here reads a pipeline, a plot or an image; the only non-stdlib imports
are numpy, pandas (for :func:`mean_cv_series` alone) and the public measurement
schema, for the spelling of ``Object_Label``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Iterable, NamedTuple, Sequence

import numpy as np

from phenotypic.schema import OBJECT

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "ABRUPT_K",
    "CV_MEAN_FLOOR",
    "Derivatives",
    "EDGE_FRAMES",
    "MAD_TO_SIGMA",
    "MIN_BAR_OVER_NOISE",
    "MIN_REPLICATES",
    "MIN_SERIES_FRAMES",
    "MeanCvSeries",
    "NEAR_MIN_TOLERANCE",
    "PERSIST",
    "ROUGH_K",
    "ROUGH_LAG",
    "ROUGH_ROLL",
    "ROUGH_SPAN",
    "RuleDResult",
    "SMOOTH",
    "WindowOutcome",
    "abrupt_bar",
    "derivatives",
    "mean_cv_series",
    "rule_d",
    "sustained",
]

# --- tuned constants -----------------------------------------------------------------
#
# Every value is read off the reference; the comment gives its name there. They
# are module constants only so that they can serve as the keyword defaults
# below -- nothing in this module reads them directly.

#: ``_SMOOTH`` (``prototype_rule_d.py``). Trend window, in frames. Wider than
#: c10's own 5: "Differencing twice amplifies noise, and 9 frames halves the
#: acceleration noise floor, which is what makes a shared absolute bar mean
#: anything."
SMOOTH = 9

#: ``_ABRUPT_K`` (``prototype_rule_d.py``). How far above the pooled quiet level
#: an acceleration must go to count as abrupt.
ABRUPT_K = 10.0

#: ``_PERSIST`` (``prototype_rule_d.py``). Frames an excursion must persist to
#: count.
PERSIST = 2

#: ``_ROUGH_ROLL`` (``prototype_rule_d.py``). Window of the roughness rolling
#: mean.
ROUGH_ROLL = 7

#: ``_ROUGH_K`` (``prototype_rule_d.py``). Multiple of a condition's own recent
#: roughness that ends a window.
ROUGH_K = 2.5

#: ``_ROUGH_LAG`` (``prototype_rule_d.py``). The roughness baseline is a
#: *trailing* median; this is how far back it starts.
ROUGH_LAG = 3

#: ``_ROUGH_SPAN`` (``prototype_rule_d.py``). How wide that trailing baseline
#: window is.
ROUGH_SPAN = 12

#: ``_NEAR_MIN_TOLERANCE`` (``prototype_rule_d.py``). How close to its own
#: minimum the smoothed CV must come for a frame to belong to the minimum's
#: basin.
NEAR_MIN_TOLERANCE = 0.10

#: ``_MIN_BAR_OVER_NOISE`` (``prototype_rule_d.py``). Multiples of a condition's
#: *own* acceleration noise the shared bar must exceed for the test to
#: discriminate.
MIN_BAR_OVER_NOISE = 3.0

#: ``_EDGE_FRAMES`` (``c10_cv_by_ph.py``), imported by the prototype. Frames at
#: each end that cannot be evaluated.
EDGE_FRAMES = 3

#: ``_MIN_SERIES_FRAMES`` (``c10_cv_by_ph.py``), likewise imported. **A plain
#: literal here, deliberately not re-derived.** Upstream it is
#: ``_SMOOTH_WINDOW + 2 * _EDGE_FRAMES`` where ``_SMOOTH_WINDOW`` is c10's own
#: **5**-frame smoother -- not the **9**-frame :data:`SMOOTH` Rule D actually
#: smooths with. So the guard admits an 11-frame series with five interior
#: frames under a nine-frame smoother, and the next guard (``too few evaluable
#: frames``) is what catches it in practice. Transcribed, not fixed.
MIN_SERIES_FRAMES = 11

#: ``MIN_REPLICATES`` (``c10_cv_by_ph.py``). Below this many *detected*
#: replicates a dispersion estimate is noise about noise.
MIN_REPLICATES = 3

#: ``c10_cv_by_ph._cv_by_time``: a replicate group whose mean metric is not above
#: this does not count at that frame. Easy to miss, and it moves both
#: ``n_counted`` and the frame-wise mean.
CV_MEAN_FLOOR = 1e-9

#: The consistency constant that turns a median absolute deviation into a normal
#: sigma estimate; ``1.4826`` verbatim in both ``abrupt_bar`` and ``rule_d`` of
#: ``prototype_rule_d.py``.
MAD_TO_SIGMA = 1.4826


class WindowOutcome(str, Enum):
    """Which of :func:`rule_d`'s six exits produced a result.

    One member per ``return`` in the reference's ``rule_d``
    (``prototype_rule_d.py``), so the value is a *structured* answer to "why is
    there no window" rather than a sentence to be re-parsed. The reference's
    figure code recovers a label from the sentence by regex (``_short_reason``)
    and loses information doing it: it maps ``too few evaluable frames``, ``no
    smoothed value`` and ``abrupt event immediately after the start`` all onto
    one label.

    Subclasses :class:`str` (the ``OpCacheKey`` pattern, ``sdk_/_opcache.py``)
    so a member is both an importable constant and a plain string in a table.
    """

    #: The last ``return`` — a window was found.
    FOUND = "found"
    #: The series is shorter than ``min_series_frames``.
    TOO_FEW_FRAMES = "too_few_frames"
    #: Fewer than four finite accelerations inside the edge margins.
    TOO_FEW_EVALUABLE_FRAMES = "too_few_evaluable_frames"
    #: The shared bar sits inside this curve's own noise. Never reached by a
    #: condition alone in its run at the default settings (see
    #: :func:`abrupt_bar`).
    TOO_NOISY = "too_noisy"
    #: The smoothed series is entirely NaN.
    NO_SMOOTHED_VALUE = "no_smoothed_value"
    #: The window would be empty: its end is at or before its start. Named for
    #: an abrupt event, but reached in practice when a series' best agreement
    #: comes in its last evaluable frames, so no event can follow the start.
    ABRUPT_EVENT_AT_START = "abrupt_event_at_start"

    def __str__(self) -> str:
        """The plain string value, so the member formats as a table cell.

        Examples:
            >>> str(WindowOutcome.TOO_NOISY)
            'too_noisy'
        """
        return str(self.value)


class Derivatives(NamedTuple):
    """The eight arrays :func:`derivatives` returns, all in absolute CV units.

    Field order matches the return tuple of ``prototype_rule_d._derivatives``
    exactly, so positional unpacking of either is the same operation.
    """

    #: Frames, ascending.
    frames: np.ndarray
    #: The measured series, reordered to match :attr:`frames`.
    values: np.ndarray
    #: The centred rolling-mean trend of :attr:`values` (the reference's
    #: ``smooth_v``). NaN wherever the window is not full.
    smooth_v: np.ndarray
    #: ``frames[1:]`` — the frames the rate and acceleration are indexed on.
    rate_frames: np.ndarray
    #: Smoothed signed rate, per frame.
    rate: np.ndarray
    #: Acceleration, per frame squared. NaN at index 0 by construction.
    accel: np.ndarray
    #: Boolean: acceleration is finite *and* the index is inside the edge margins.
    evaluable: np.ndarray
    #: Rolling mean of ``|values - smooth_v|`` — the scatter the smoother
    #: discards, indexed like :attr:`values`.
    rough: np.ndarray


class RuleDResult(NamedTuple):
    """Rule D's answer for one series.

    The reference returns ``(window, events, reason)``; this adds the structured
    :attr:`outcome` between the events and the sentence, because the sentence is
    lossy. A deliberate deviation from the reference's return shape.
    """

    #: ``(start_frame, end_frame)`` inclusive, in ``frames`` units, or ``None``.
    window: tuple[int, int] | None
    #: Sustained adverse acceleration excursions, as ``(start, end)`` frame pairs
    #: on ``rate_frames``. Non-empty only once the noise guard is passed.
    events: list[tuple[int, int]]
    #: Which exit produced this result.
    outcome: WindowOutcome
    #: The reference's own sentence, verbatim; ``""`` when a window was found.
    detail: str


class MeanCvSeries(NamedTuple):
    """The per-frame mean replicate CV, and how many groups stand behind it."""

    #: Counted frames, ascending. One entry per frame with a defined mean CV.
    frames: np.ndarray
    #: The mean, across counted replicate groups, of each group's CV.
    mean_cv: np.ndarray
    #: How many replicate groups were counted at that frame (the reference's
    #: ``cells``, in ``c10_cv_by_ph._cv_series``).
    n_counted: np.ndarray


def _rolling_mean_centered(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean with polars' "whole window or nothing" null rule.

    The reference smooths with ``pl.Series(x).rolling_mean(window_size=w,
    center=True)`` in three places in ``prototype_rule_d._derivatives``. Two
    properties of that call are what has to be reproduced, and neither is what a
    naive ``np.convolve`` or a ``min_periods=1`` rolling mean does:

    1. **``min_samples`` defaults to ``window_size``**, so the result is *null*
       unless the whole window is present. ``.to_numpy()`` renders those nulls
       as NaN, which is why the first and last ``w // 2`` entries of every
       smoothed array here are NaN and why :data:`EDGE_FRAMES` exists at all.
    2. **A NaN inside a full window propagates** into that window's mean — NaN
       is a float value in polars, not a null, so it is summed.

    ``center=True`` places the window for index ``i`` at
    ``[i - w // 2, i - w // 2 + w - 1]``; for the odd windows the reference uses
    (9 and 7) that is the symmetric ``[i - w // 2, i + w // 2]``.

    Args:
        values: The series to smooth. Coerced to ``float``.
        window: Window width in samples.

    Returns:
        An array the same length as *values*, NaN where the window is not full.

    Raises:
        ValueError: If *window* is not a positive integer.

    Examples:
        Asserted rather than printed: ``numpy``'s array repr pads differently
        under ``set_printoptions(legacy=...)``, which any earlier import in the
        process can set, so an example that pinned the spacing would flake.

        >>> out = _rolling_mean_centered(np.arange(5.0), 3)
        >>> out[1:4].tolist()
        [1.0, 2.0, 3.0]
        >>> bool(np.isnan(out[[0, 4]]).all())
        True
        >>> bool(np.isnan(_rolling_mean_centered(np.arange(2.0), 3)).all())
        True
    """
    if window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    series = np.asarray(values, dtype=float)
    out = np.full(series.size, np.nan, dtype=float)
    if series.size < window:
        return out
    offset = window // 2
    view = np.lib.stride_tricks.sliding_window_view(series, window)
    out[offset : offset + view.shape[0]] = view.mean(axis=1)
    return out


def derivatives(
    frames: np.ndarray,
    values: np.ndarray,
    *,
    smooth: int = SMOOTH,
    rough_roll: int = ROUGH_ROLL,
    edge_frames: int = EDGE_FRAMES,
) -> Derivatives:
    """Value, trend, signed rate, acceleration and roughness — no normalisation.

    Line-for-line ``prototype_rule_d._derivatives``. In the reference's words:
    "Rates are per frame and acceleration per frame squared, both divided by the
    real frame spacing because several series skip frames." A zero spacing is
    replaced by 1.0 rather than producing an infinity, for both the rate and the
    acceleration, which matters because the frame axis can repeat a value.

    ``evaluable`` is masked off at both ends with the reference's own
    ``[len(x) - edge_frames:]`` spelling rather than ``[-edge_frames:]``: the two
    differ at ``edge_frames == 0``, where the negative form would blank the whole
    array.

    Args:
        frames: The frame axis. Sorted ascending internally, and the values are
            reordered with it.
        values: The measured series, one entry per frame.
        smooth: Trend window. Reference :data:`SMOOTH`.
        rough_roll: Roughness window. Reference :data:`ROUGH_ROLL`.
        edge_frames: Frames blanked at each end of ``evaluable``. Reference
            :data:`EDGE_FRAMES`.

    Returns:
        A :class:`Derivatives`.

    Examples:
        A ramp has a constant rate and, once the edges are excluded, no finite
        acceleration to speak of:

        >>> d = derivatives(np.arange(20), np.arange(20.0) * 0.01)
        >>> float(np.nanmax(np.abs(d.rate[np.isfinite(d.rate)]) - 0.01)) < 1e-12
        True
        >>> bool(np.allclose(d.accel[d.evaluable], 0.0))
        True
    """
    order = np.argsort(frames)
    frames = np.asarray(frames)[order]
    values = np.asarray(values, dtype=float)[order]

    smooth_v = _rolling_mean_centered(values, smooth)

    spacing = np.diff(frames).astype(float)
    spacing[spacing == 0] = 1.0
    rate = _rolling_mean_centered(np.diff(values) / spacing, smooth)
    rate_frames = frames[1:]

    accel = np.full_like(rate, np.nan)
    if len(rate) > 1:
        step = np.diff(rate_frames).astype(float)
        step[step == 0] = 1.0
        accel[1:] = np.diff(rate) / step

    rough = _rolling_mean_centered(np.abs(values - smooth_v), rough_roll)

    evaluable = np.isfinite(accel)
    evaluable[:edge_frames] = False
    evaluable[len(evaluable) - edge_frames :] = False
    return Derivatives(
        frames, values, smooth_v, rate_frames, rate, accel, evaluable, rough
    )


def abrupt_bar(
    series: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    min_series_frames: int = MIN_SERIES_FRAMES,
    abrupt_k: float = ABRUPT_K,
    smooth: int = SMOOTH,
    rough_roll: int = ROUGH_ROLL,
    edge_frames: int = EDGE_FRAMES,
) -> float:
    """One abrupt-change bar, pooled over **every** series given.

    ``prototype_rule_d.abrupt_bar`` loops over every condition, concatenates
    their evaluable ``|acceleration|`` and returns a single float: "Pooled rather
    than per-curve: a per-curve bar is exactly the unequal treatment this rule
    exists to remove." Every condition of a run is judged against that one bar.

    The bar is a quiet level plus ``abrupt_k`` robust scales of the quiet half:
    take the pooled magnitudes at or below their own median (the "quiet" half),
    then ``median(quiet) + abrupt_k * MAD(quiet) * 1.4826``.

    The reference loop passes each series through :func:`derivatives` and keeps
    only ``accel[evaluable]``, so a series shorter than *min_series_frames* is
    skipped entirely — it contributes nothing to the bar even though it will
    still be *judged* against it.

    **A condition alone in its run is never too noisy.** With one series the
    pool is that series' own accelerations, so the quiet scale computed here is
    exactly the noise :func:`rule_d` measures for it, and the bar is its quiet
    level plus ``abrupt_k`` of those noise units. The headroom ``bar / noise`` is
    therefore at least ``abrupt_k``, which at the defaults (10) is well above
    ``min_bar_over_noise`` (3). :attr:`WindowOutcome.TOO_NOISY` only happens
    when other, quieter conditions pull the pooled bar down: whether a condition
    gets a window can depend on what else is in the run.

    Args:
        series: ``(frames, values)`` pairs — one per condition. **All of them**;
            handing this one condition's series is the single most damaging way
            to get Rule D wrong.
        min_series_frames: Series shorter than this contribute nothing.
        abrupt_k: Multiples of the quiet scale the bar sits above the quiet level.
        smooth: Passed through to :func:`derivatives`.
        rough_roll: Passed through to :func:`derivatives`.
        edge_frames: Passed through to :func:`derivatives`.

    Returns:
        The bar, or NaN when no series contributes a finite acceleration. The
        reference raises from ``np.concatenate([])`` in that case; NaN is
        harmless because a run in which nothing contributes is also a run in
        which every series exits :attr:`WindowOutcome.TOO_FEW_FRAMES` before the
        bar is read. A deliberate deviation from the reference.

    Examples:
        Two noisy series pooled give a bar above either one's jitter:

        >>> rng = np.random.default_rng(0)
        >>> f = np.arange(40)
        >>> pair = [(f, rng.normal(0.1, 0.001, 40)) for _ in range(2)]
        >>> bar = abrupt_bar(pair)
        >>> bar > 0
        True
    """
    pool: list[np.ndarray] = []
    for frames, values in series:
        if len(frames) < min_series_frames:
            continue
        computed = derivatives(
            frames,
            values,
            smooth=smooth,
            rough_roll=rough_roll,
            edge_frames=edge_frames,
        )
        pool.append(np.abs(computed.accel[computed.evaluable]))
    if not pool:
        return float("nan")
    pooled = np.concatenate(pool)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("nan")
    quiet = pooled[pooled <= np.median(pooled)]
    level = float(np.median(quiet))
    scale = float(np.median(np.abs(quiet - level)) * MAD_TO_SIGMA)
    return level + abrupt_k * scale


def sustained(
    mask: np.ndarray,
    evaluable: np.ndarray,
    *,
    persist: int = PERSIST,
) -> list[tuple[int, int]]:
    """Index runs where *mask* holds, and *evaluable* allows it, for *persist* frames.

    Line-for-line ``prototype_rule_d._sustained``, including the closing clause:
    a run still open when the loop ends is admitted on
    ``len(mask) - start >= persist``, which is the same comparison the in-loop
    branch makes with ``i`` one past the run. The returned bounds are
    **inclusive** indices.

    ``>=`` is the comparison, not ``>``: a run of exactly *persist* frames
    counts.

    Args:
        mask: Per-index condition.
        evaluable: Per-index admissibility; a non-evaluable index breaks a run.
        persist: Minimum run length, in indices.

    Returns:
        Inclusive ``(start, end)`` index pairs, in ascending order.

    Examples:
        >>> m = np.array([False, True, True, False, True, False])
        >>> ev = np.ones(6, dtype=bool)
        >>> sustained(m, ev, persist=2)
        [(1, 2)]
        >>> sustained(m, ev, persist=1)
        [(1, 2), (4, 4)]
    """
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(len(mask)):
        if evaluable[i] and mask[i]:
            start = i if start is None else start
        else:
            if start is not None and i - start >= persist:
                out.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= persist:
        out.append((start, len(mask) - 1))
    return out


def rule_d(
    frames: np.ndarray,
    values: np.ndarray,
    bar: float,
    *,
    smooth: int = SMOOTH,
    persist: int = PERSIST,
    rough_roll: int = ROUGH_ROLL,
    rough_k: float = ROUGH_K,
    rough_lag: int = ROUGH_LAG,
    rough_span: int = ROUGH_SPAN,
    near_min_tolerance: float = NEAR_MIN_TOLERANCE,
    min_bar_over_noise: float = MIN_BAR_OVER_NOISE,
    edge_frames: int = EDGE_FRAMES,
    min_series_frames: int = MIN_SERIES_FRAMES,
) -> RuleDResult:
    """Start at best replicate agreement; run until something breaks.

    Line-for-line ``prototype_rule_d.rule_d``, with the sentence replaced by a
    :class:`RuleDResult` that carries both the sentence and a
    :class:`WindowOutcome`. The reference's ``n_cells`` argument is dropped: it
    is never read in the body.

    The six exits, in the order they are reached:

    1. ``len(frames) < min_series_frames`` → :attr:`WindowOutcome.TOO_FEW_FRAMES`.
    2. Fewer than four finite evaluable accelerations →
       :attr:`WindowOutcome.TOO_FEW_EVALUABLE_FRAMES`.
    3. ``bar / noise < min_bar_over_noise`` → :attr:`WindowOutcome.TOO_NOISY`,
       where ``noise`` is this curve's own robust acceleration sigma. A zero
       noise gives infinite headroom, so a perfectly smooth curve is assessable
       rather than abstained on. A curve judged against a bar pooled from itself
       alone never takes this exit at the defaults (see :func:`abrupt_bar`).
    4. An all-NaN trend → :attr:`WindowOutcome.NO_SMOOTHED_VALUE`.
    5. ``end <= start`` → :attr:`WindowOutcome.ABRUPT_EVENT_AT_START`. With no
       event after the start, ``end`` is the last evaluable frame, so a series
       whose minimum falls at or after that frame (one still converging when
       the run ends) reaches this exit with no event after the start.
    6. Otherwise a window, :attr:`WindowOutcome.FOUND`.

    How the window is placed:

    * **Start** at the left edge of the *basin* containing the trend's argmin —
      the maximal run of frames within *near_min_tolerance* of the minimum that
      contains the argmin itself. Taking the first near-minimum frame anywhere
      lets an early dip during the establishment fall open a window far from
      the minimum it has no connection to.
    * The abrupt-event search begins at ``search_from``, the basin's right edge
      plus the smoother's half-width: "A minimum is BY DEFINITION a
      positive-acceleration feature", so searching from the start would fire on
      the curvature of the feature chosen as the start.
    * **End** one frame before the first sustained *adverse* (``accel > bar``)
      excursion that begins after ``search_from``; with no such event, the last
      evaluable frame. Only deterioration counts — "a sharp
      improvement in agreement is not a reason to stop."
    * A **second** end criterion catches the measurement itself becoming
      unstable: the roughness rising above *rough_k* times its
      own *trailing* median for *persist* consecutive frames. The baseline is
      trailing, not a global quantile, because "a rise is only a rise relative
      to what came immediately before it."

    Args:
        frames: The frame axis of the series.
        values: The measured series (a mean replicate CV, in the reference).
        bar: The pooled abrupt-change bar from :func:`abrupt_bar`. **Pooled over
            the whole run**, never per series.
        smooth: Trend window. Reference :data:`SMOOTH`.
        persist: Frames an excursion must persist. Reference :data:`PERSIST`.
        rough_roll: Roughness window. Reference :data:`ROUGH_ROLL`.
        rough_k: Multiple of trailing roughness that ends a window. Reference
            :data:`ROUGH_K`.
        rough_lag: Lag of the trailing roughness baseline. Reference
            :data:`ROUGH_LAG`.
        rough_span: Span of that baseline. Reference :data:`ROUGH_SPAN`.
        near_min_tolerance: Basin tolerance. Reference :data:`NEAR_MIN_TOLERANCE`.
        min_bar_over_noise: Abstention floor. Reference
            :data:`MIN_BAR_OVER_NOISE`.
        edge_frames: Frames blanked at each end. Reference :data:`EDGE_FRAMES`.
        min_series_frames: Shortest series assessed. Reference
            :data:`MIN_SERIES_FRAMES`.

    Returns:
        A :class:`RuleDResult`.

    Examples:
        A series too short to assess names its own length:

        >>> short = rule_d(np.arange(6), np.full(6, 0.1), bar=0.01)
        >>> short.outcome is WindowOutcome.TOO_FEW_FRAMES
        True
        >>> short.detail
        'only 6 frames'
    """
    if len(frames) < min_series_frames:
        return RuleDResult(
            None, [], WindowOutcome.TOO_FEW_FRAMES, f"only {len(frames)} frames"
        )
    f, v, smooth_v, rf, rate, accel, ev, rough = derivatives(
        frames,
        values,
        smooth=smooth,
        rough_roll=rough_roll,
        edge_frames=edge_frames,
    )

    # Can this condition's acceleration be tested against the shared bar at all?
    finite_a = accel[ev][np.isfinite(accel[ev])]
    if len(finite_a) < 4:
        return RuleDResult(
            None, [], WindowOutcome.TOO_FEW_EVALUABLE_FRAMES, "too few evaluable frames"
        )
    mag = np.abs(finite_a)
    quiet = mag[mag <= np.median(mag)]
    noise = float(np.median(np.abs(quiet - np.median(quiet))) * MAD_TO_SIGMA)
    headroom = bar / noise if noise > 0 else np.inf
    if headroom < min_bar_over_noise:
        return RuleDResult(
            None,
            [],
            WindowOutcome.TOO_NOISY,
            (
                f"bar is {headroom:.1f}x this curve's own noise "
                f"(<{min_bar_over_noise:g}): not assessable"
            ),
        )

    # START -- the left edge of the minimum's own basin.
    finite = np.isfinite(smooth_v)
    if not finite.any():
        return RuleDResult(None, [], WindowOutcome.NO_SMOOTHED_VALUE, "no smoothed value")
    j = int(np.nanargmin(smooth_v))
    floor = float(smooth_v[j])
    near = finite & (smooth_v <= floor * (1 + near_min_tolerance))
    left = j
    while left - 1 >= 0 and near[left - 1]:
        left -= 1
    right = j
    while right + 1 < len(near) and near[right + 1]:
        right += 1
    start = int(f[left])
    # The search begins past the basin plus the smoothing half-width, so the
    # turn at the minimum cannot be mistaken for a fault.
    search_from = int(f[min(right + smooth // 2, len(f) - 1)])

    # END -- the first sustained ADVERSE acceleration excursion after the start.
    adverse = ev & (accel > bar)
    events = [
        (int(rf[a]), int(rf[b])) for a, b in sustained(adverse, ev, persist=persist)
    ]
    after = [e for e in events if e[0] > search_from]
    last_ev = int(rf[ev][-1]) if ev.any() else start
    end = (after[0][0] - 1) if after else last_ev

    # SECOND end criterion: the measurement itself becoming unstable.
    ok_r = np.isfinite(rough)
    ok_r[:edge_frames] = False
    ok_r[len(ok_r) - edge_frames :] = False
    run = 0
    for i in range(len(f)):
        if not ok_r[i] or f[i] < start + rough_lag + 4:
            continue
        prior = ok_r & (f >= f[i] - rough_lag - rough_span) & (f <= f[i] - rough_lag)
        if prior.sum() < 4:
            continue
        base = float(np.nanmedian(rough[prior]))
        if not np.isfinite(base) or base <= 0:
            continue
        run = run + 1 if rough[i] > rough_k * base else 0
        if run >= persist:
            end = min(end, int(f[i - persist + 1]))
            break

    if end <= start:
        return RuleDResult(
            None,
            events,
            WindowOutcome.ABRUPT_EVENT_AT_START,
            "abrupt event immediately after the start",
        )
    return RuleDResult((start, end), events, WindowOutcome.FOUND, "")


def mean_cv_series(
    data: "pd.DataFrame",
    *,
    on: str,
    replicate_by: Sequence[str],
    time_col: str,
    min_replicates: int = MIN_REPLICATES,
) -> MeanCvSeries:
    """The per-frame mean replicate CV that Rule D consumes.

    ``c10_cv_by_ph._cv_series`` composed with ``_cv_by_time`` and ``_detected``,
    with ``statistic="mean"`` (which is what ``prototype_rule_d._series``
    passes):

    1. Keep only real detections.
    2. Group by ``replicate_by + [time_col]``; take ``mean``, ``sd`` and ``n``
       of *on*.
    3. Keep a group only where ``n >= min_replicates`` **and**
       ``mean > 1e-9``. Both guards, not just the first.
    4. ``cv = sd / mean``.
    5. Per frame, the **mean** of those CVs, and ``n_counted`` = how many groups
       were counted.

    Two details decide whether the numbers match:

    * **``sd`` is the sample standard deviation.** polars' ``Series.std()`` is
      ``ddof=1`` and so is :meth:`pandas.core.groupby.GroupBy.std`;
      ``numpy.std`` is ``ddof=0`` and would scale every CV by
      ``sqrt((n - 1) / n)``.
    * **``n`` counts rows; ``mean``/``sd`` skip nulls.** ``pl.len()`` counts
      every row in the group whether or not its metric is null, while
      ``.mean()``/``.std()`` ignore nulls — and pandas' ``size``/``mean``/``std``
      aggregations split the same way, so the composition is reproduced without
      a special case.

    **This function is order-deterministic; the reference is not.** polars'
    ``group_by`` is unordered, so the reference sums each frame's mean in
    whatever order the groups come out in, and it is measurably
    nondeterministic: two runs over the identical frame can give pooled bars that
    differ in the last digits. Here the rows are sorted by
    ``replicate_by + [time_col] + [on]`` before aggregating and both groupings
    are sorted, so the result is a function of the *set* of input rows and not of
    their arrangement — which is what lets the tables built on it be
    byte-reproducible. It costs about one bit in the last place relative to the
    reference, three orders of magnitude inside the parity tolerance. Sorting the
    group keys alone does **not** achieve this, because pandas accumulates within
    a group in frame order. A deliberate deviation from the reference.

    The reference selects detections with two engineered boolean columns (an
    unpinned-detection flag and a detected flag, ``c10_cv_by_ph._detected``),
    plus a non-null condition key. Neither flag exists on a PhenoTypic
    measurements frame, so this reproduces the same subset NaN-natively, which
    also makes it correct on a frame that never went through the CLI:

    * A metadata-only phantom row carries a null ``Object_Label`` — the test
      :meth:`ExpectedVsDetectedCount._detected_count` already uses — so those
      rows are dropped. No flag column is consulted.
    * An unpinned detection carries no plate, strain or condition, so its
      grouping keys are null and :meth:`pandas.DataFrame.groupby` (``dropna=True``
      by default) drops it, exactly as the reference's unpinned-detection and
      non-null-condition filters do.

    Args:
        data: One condition's rows. Never the whole run: the reference filters
            to one condition first, so the caller is responsible for grouping
            by condition.
        on: The metric column whose replicate CV is tracked: a radius, for
            example, rather than an area.
        replicate_by: Columns that define one replicate group (for example,
            the strain).
        time_col: The per-image frame index. Must hold whole numbers; elapsed
            hours do not work, because images of different plates never share an
            hour value.
        min_replicates: Detected replicates a group needs at a frame to count.

    Returns:
        A :class:`MeanCvSeries`. Empty arrays when nothing is counted, which
        :func:`rule_d` reports as ``only 0 frames`` — not a seventh outcome.

    Raises:
        KeyError: If *on*, *time_col* or any of *replicate_by* is missing.
        ValueError: If *replicate_by* is empty, or if *time_col* holds a value
            that is not a whole number.

    Examples:
        Three replicates at each of two frames. ``on``, ``replicate_by`` and
        ``time_col`` are whatever the caller names them — nothing here assumes a
        measurement vocabulary, so the example does not borrow one. The only
        column this function spells for itself is the object label, which it
        asks the schema for rather than hard-coding.

        >>> import pandas as pd
        >>> from phenotypic.schema import OBJECT
        >>> rows = pd.DataFrame({
        ...     "strain": ["a"] * 6,
        ...     "frame": [1, 1, 1, 2, 2, 2],
        ...     str(OBJECT.LABEL): [1, 2, 3, 1, 2, 3],
        ...     "radius": [10.0, 11.0, 12.0, 20.0, 20.0, 20.0],
        ... })
        >>> series = mean_cv_series(
        ...     rows, on="radius", replicate_by=["strain"], time_col="frame"
        ... )
        >>> series.frames.tolist()
        [1, 2]

        Frame 2's replicates agree exactly, so its CV is zero — a counted frame
        with no dispersion, not an uncounted one:

        >>> [round(x, 6) for x in series.mean_cv.tolist()]
        [0.090909, 0.0]
        >>> series.n_counted.tolist()
        [1, 1]
    """
    import pandas as pd

    keys = list(replicate_by)
    if not keys:
        raise ValueError("replicate_by must name at least one column")
    missing = [c for c in (*keys, time_col, on) if c not in data.columns]
    if missing:
        raise KeyError(f"mean_cv_series: columns absent from the frame: {missing}")

    label = str(OBJECT.LABEL)
    rows = data
    if label in rows.columns:
        rows = rows.loc[rows[label].notna()]

    group_cols = [*keys, time_col]
    if rows.empty:
        return MeanCvSeries(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
        )

    # Canonical row order BEFORE aggregating. Sorting the group *keys* alone is
    # not enough: pandas accumulates each group's mean and sd in frame order, so
    # a shuffled input still moves the last bit of some frames. Sorting by the
    # keys *and the metric* makes the reduction order a function of the rows
    # themselves. `kind="stable"` so that rows tying on all of them cannot
    # reorder either.
    rows = rows.sort_values([*group_cols, on], kind="stable", na_position="last")
    stats = (
        rows.groupby(group_cols, dropna=True, sort=True, observed=True)[on]
        .agg(["mean", "std", "size"])
        .reset_index()
    )
    counted = stats.loc[
        (stats["size"] >= min_replicates) & (stats["mean"] > CV_MEAN_FLOOR)
    ].copy()
    if counted.empty:
        return MeanCvSeries(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
        )
    counted["cv"] = counted["std"] / counted["mean"]

    per_frame = (
        counted.groupby(time_col, dropna=True, sort=True, observed=True)["cv"]
        .agg(["mean", "size"])
        .reset_index()
    )
    axis = pd.to_numeric(per_frame[time_col], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(axis)) or not np.all(axis == np.floor(axis)):
        raise ValueError(
            f"mean_cv_series: {time_col!r} must hold whole frame numbers, "
            f"got {sorted(set(per_frame[time_col].tolist()))[:5]}"
        )
    return MeanCvSeries(
        axis.astype(np.int64),
        per_frame["mean"].to_numpy(dtype=float),
        per_frame["size"].to_numpy(dtype=np.int64),
    )
