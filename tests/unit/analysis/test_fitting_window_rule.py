"""Rule D port: golden fixture, behavioural controls, and the CV step.

Two kinds of test live here and they answer different questions.

* The **golden fixture** (``tests/fixtures/rule_d/``) answers "is this *that*
  algorithm?" It was recorded by running the reference scripts themselves
  (``prototype_rule_d.py``, ``c10_cv_by_ph.py``) over a synthetic series — see
  the fixture README. It pins every output Rule D has: window, outcome, the
  reference's sentence, and the sustained-event list, plus the CV step's
  series.
* The **behavioural controls** answer "does it behave like this *kind* of
  algorithm?" The synthetic run reaches five of ``rule_d``'s six exits; the
  sixth gets a series built to reach it, every exit gets a control of its own,
  and the properties that a golden fixture cannot see — the bar is pooled, the
  run minimum is ``>=`` not ``>``, the CV is a sample standard deviation — get
  their own controls.

Neither kind asserts a hand-computed window for a synthetic series: the golden
fixture is the only place numbers are pinned, and they come from the reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis.qc._fitting_window_rule import (
    ABRUPT_K,
    EDGE_FRAMES,
    MAD_TO_SIGMA,
    MIN_BAR_OVER_NOISE,
    MIN_REPLICATES,
    MIN_SERIES_FRAMES,
    NEAR_MIN_TOLERANCE,
    PERSIST,
    ROUGH_K,
    ROUGH_LAG,
    ROUGH_ROLL,
    ROUGH_SPAN,
    SMOOTH,
    WindowOutcome,
    _rolling_mean_centered,
    abrupt_bar,
    derivatives,
    mean_cv_series,
    rule_d,
    sustained,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "rule_d"
WINDOWS = FIXTURES / "synthetic_windows.json"
SERIES = FIXTURES / "synthetic_series.parquet"

#: The reference is nondeterministic in its last digits (polars' ``group_by`` is
#: unordered), so every float comparison against it is relative and sits many
#: orders of magnitude below any spread that could move a window, rather than
#: being exact.
BAR_RTOL = 1e-12
CV_RTOL = 1e-12

#: The three recorded runs, each with its own pooled bar.
RUNS = ("page", "coverage", "alone")


@pytest.fixture(scope="module")
def recording() -> dict:
    """The whole recording: constants, provenance, and the three runs."""
    return json.loads(WINDOWS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden(recording: dict) -> dict:
    """The ``coverage`` run: every synthetic condition, judged together."""
    return recording["runs"]["coverage"]


@pytest.fixture(scope="module")
def golden_series(golden: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every condition's ``(frames, values)`` — the input ``abrupt_bar`` pools."""
    return [_series_of(c) for c in golden["conditions"]]


def _series_of(condition: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(condition["frames"], dtype=np.int64),
        np.asarray(condition["values"], dtype=float),
    )


def _by_name(run: dict) -> dict[str, dict]:
    return {c["condition"]: c for c in run["conditions"]}


# --- the constants ------------------------------------------------------------------


def test_the_tuned_constants_match_the_reference():
    """Each value is the one read off the reference."""
    assert SMOOTH == 9
    assert ABRUPT_K == 10.0
    assert PERSIST == 2
    assert ROUGH_ROLL == 7
    assert ROUGH_K == 2.5
    assert ROUGH_LAG == 3
    assert ROUGH_SPAN == 12
    assert NEAR_MIN_TOLERANCE == 0.10
    assert MIN_BAR_OVER_NOISE == 3.0
    assert EDGE_FRAMES == 3
    assert MIN_REPLICATES == 3


def test_the_tuned_constants_match_the_recording(recording: dict):
    """The same values, as the generator read them off the reference modules."""
    assert recording["constants"] == {
        "smooth": SMOOTH,
        "abrupt_k": ABRUPT_K,
        "persist": PERSIST,
        "rough_roll": ROUGH_ROLL,
        "rough_k": ROUGH_K,
        "rough_lag": ROUGH_LAG,
        "rough_span": ROUGH_SPAN,
        "near_min_tolerance": NEAR_MIN_TOLERANCE,
        "min_bar_over_noise": MIN_BAR_OVER_NOISE,
        "edge_frames": EDGE_FRAMES,
        "min_series_frames": MIN_SERIES_FRAMES,
        "min_replicates": MIN_REPLICATES,
    }


def test_min_series_frames_is_a_literal_eleven_not_a_derived_fifteen():
    """The length guard is c10's ``5 + 2 * 3``, not Rule D's ``9 + 2 * 3``.

    ``c10_cv_by_ph.py`` derives ``_MIN_SERIES_FRAMES`` from c10's own
    5-frame smoother, while Rule D smooths with 9 (``prototype_rule_d.py``).
    Re-deriving it here from :data:`SMOOTH` would give 15 and silently reject
    four-frame-wider series than the reference does.
    """
    assert MIN_SERIES_FRAMES == 11
    assert MIN_SERIES_FRAMES != SMOOTH + 2 * EDGE_FRAMES


# --- the rolling mean ---------------------------------------------------------------


@pytest.mark.parametrize("window", [3, 4, 7, 9])
def test_rolling_mean_matches_polars_center_true(window: int):
    """The oracle: polars itself, which is what the reference calls.

    ``prototype_rule_d._derivatives`` uses
    ``rolling_mean(window_size=w, center=True)``. polars is a first-class
    dependency here, so the transcription is pinned against the real thing
    rather than against a description of it.
    """
    import polars as pl

    rng = np.random.default_rng(20260925)
    values = rng.normal(size=40)
    values[7] = np.nan  # a NaN inside a full window must propagate
    values[31] = np.nan

    expected = pl.Series(values).rolling_mean(window_size=window, center=True).to_numpy()
    actual = _rolling_mean_centered(values, window)

    np.testing.assert_array_equal(np.isnan(expected), np.isnan(actual))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0, atol=1e-15)


def test_rolling_mean_is_null_until_the_window_is_full():
    """Not ``min_periods=1``: the first and last ``w // 2`` entries are NaN."""
    out = _rolling_mean_centered(np.arange(7.0), 5)
    assert np.isnan(out[[0, 1, 5, 6]]).all()
    np.testing.assert_allclose(out[2:5], [2.0, 3.0, 4.0])


def test_a_nan_value_is_skipped_not_treated_as_zero():
    """A NaN poisons its window rather than pulling the mean toward zero."""
    values = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    out = _rolling_mean_centered(values, 3)
    assert np.isnan(out[1:4]).all()
    assert not (out[1:4] < 1.0).any()


# --- the golden fixture -------------------------------------------------------------


def test_the_recording_was_made_from_the_committed_series(recording: dict):
    """The JSON's runs and the parquet they were recorded from cannot drift apart."""
    digest = hashlib.sha256(SERIES.read_bytes()).hexdigest()
    assert recording["series_sha256"] == digest


def test_the_golden_fixture_covers_what_it_was_built_to(recording: dict):
    """The shape of each run, checked before any per-condition assertion.

    A fixture that silently lost conditions would still pass every
    per-condition assertion below, so the shape of the recording comes first.
    The coverage run reaches five of the six exits; ``no_smoothed_value`` cannot
    be reached from data and has a behavioural control instead.
    """
    runs = recording["runs"]
    coverage = [c["outcome"] for c in runs["coverage"]["conditions"]]
    assert len(coverage) == 11
    assert coverage.count("found") == 7
    for outcome in (
        "too_noisy",
        "too_few_frames",
        "too_few_evaluable_frames",
        "abrupt_event_at_start",
    ):
        assert coverage.count(outcome) == 1, outcome

    page = {c["condition"]: c for c in runs["page"]["conditions"]}
    assert sorted(page) == ["Condition A", "Condition B", "Condition C", "Condition D"]
    assert [page[n]["outcome"] for n in sorted(page)] == [
        "found", "found", "found", "too_noisy",
    ]
    assert [c["condition"] for c in runs["alone"]["conditions"]] == ["Condition D"]


@pytest.mark.parametrize("run_name", RUNS)
def test_every_condition_reproduces_the_reference(recording: dict, run_name: str):
    """Window, outcome, sentence and event list, for every condition of a run."""
    run = recording["runs"][run_name]
    mismatches = []
    for condition in run["conditions"]:
        result = rule_d(*_series_of(condition), run["bar"])
        expected_window = (
            tuple(condition["window"]) if condition["window"] is not None else None
        )
        actual = (
            result.window,
            result.outcome.value,
            result.detail,
            list(result.events),
        )
        expected = (
            expected_window,
            condition["outcome"],
            condition["reason"],
            [tuple(e) for e in condition["events"]],
        )
        if actual != expected:
            mismatches.append(f"{condition['condition']}: {actual!r} != {expected!r}")
    assert not mismatches, "\n".join(mismatches)


def test_the_outcome_is_a_member_not_a_reparsed_sentence(golden: dict):
    """``rule_d`` hands back the enum; the sentence is carried alongside it."""
    noisy = next(c for c in golden["conditions"] if c["outcome"] == "too_noisy")
    result = rule_d(*_series_of(noisy), golden["bar"])
    assert result.outcome is WindowOutcome.TOO_NOISY
    assert result.detail.startswith("bar is ")
    assert result.detail.endswith("not assessable")


def test_a_one_based_frame_axis_shifts_every_window_by_exactly_one(
    golden: dict, golden_series: list
):
    """The CLI's frame index is 1-based, and Rule D does not care.

    The property belongs to ``rule_d``, not to the check that calls it, which is
    why it is pinned here: it is the justification for a user-facing decision
    (a 1-based ``FrameIndex``) taken far from the code it constrains.

    Rule D compares frames only to frames: ``np.diff(frames)`` for spacing,
    ``f[left]`` and ``f[min(right + smooth // 2, …)]`` for the window edges,
    ``e[0] > search_from`` for events, and ``f[i] - start`` in the roughness gate.
    So adding one to the axis must shift ``start`` and ``end`` by exactly one and
    change nothing else. Were that false, a 1-based axis would silently change
    *outcomes*, not just window labels.
    """
    variant = []
    for condition, (frames, values) in zip(
        golden["conditions"], golden_series, strict=True
    ):
        base = rule_d(frames, values, golden["bar"])
        shifted = rule_d(frames + 1, values, golden["bar"])
        expected = (
            None if base.window is None else (base.window[0] + 1, base.window[1] + 1)
        )
        if shifted.outcome is not base.outcome or shifted.window != expected:
            variant.append(
                f"{condition['condition']}: {base.window}/{base.outcome.value} -> "
                f"{shifted.window}/{shifted.outcome.value}"
            )
    assert not variant, "\n".join(variant)


def test_the_pooled_bar_is_frame_origin_invariant(golden_series: list):
    """The other half: a shifted axis must not move the shared bar.

    ``abrupt_bar`` divides by the real frame spacing, so only differences reach
    it — but the bar is pooled over the whole run, so if it *did* move, every
    condition's verdict would move with it and the per-condition shift test above
    would still pass. Compared exactly, not at a tolerance.
    """
    shifted = [(frames + 1, values) for frames, values in golden_series]
    assert abrupt_bar(shifted) == abrupt_bar(golden_series)


# --- the pooled bar -----------------------------------------------------------------


@pytest.mark.parametrize("run_name", RUNS)
def test_the_bar_pooled_over_every_series_reproduces_the_reference(
    recording: dict, run_name: str
):
    """``abrupt_bar`` over a run's series returns that run's one bar."""
    run = recording["runs"][run_name]
    series = [_series_of(c) for c in run["conditions"]]
    assert abrupt_bar(series) == pytest.approx(run["bar"], rel=BAR_RTOL)


def test_the_three_runs_have_three_different_bars(recording: dict):
    """The bar is a property of the whole run, not of any one condition."""
    bars = [recording["runs"][name]["bar"] for name in RUNS]
    assert len({round(b, 12) for b in bars}) == 3


def test_dropping_one_series_moves_the_bar(golden: dict, golden_series: list):
    """The bar is a property of the whole run, so the pool matters.

    The series dropped is one long enough to contribute; dropping a series
    shorter than ``MIN_SERIES_FRAMES`` changes nothing (tested below).
    """
    index = next(
        i for i, (frames, _) in enumerate(golden_series)
        if len(frames) >= MIN_SERIES_FRAMES
    )
    reduced = abrupt_bar(golden_series[:index] + golden_series[index + 1:])
    assert reduced != pytest.approx(golden["bar"], rel=BAR_RTOL)


def test_a_per_series_bar_disagrees_with_the_golden_windows(
    golden: dict, golden_series: list
):
    """A per-curve bar is the single most damaging way to get Rule D wrong.

    It is also invisible to a fixture built from its own output, which is why
    this control exists: it re-runs every condition against *its own* bar and
    asserts the answers move.
    """
    changed = []
    for condition, (frames, values) in zip(
        golden["conditions"], golden_series, strict=True
    ):
        own_bar = abrupt_bar([(frames, values)])
        result = rule_d(frames, values, own_bar)
        expected = tuple(condition["window"]) if condition["window"] else None
        if (result.window, result.outcome.value) != (expected, condition["outcome"]):
            changed.append(condition["condition"])
    assert changed, "a per-series bar reproduced every pooled-bar answer"


def test_a_series_shorter_than_the_guard_contributes_nothing_to_the_bar(
    golden_series: list,
):
    """``prototype_rule_d.abrupt_bar`` skips it — it is still judged against the bar."""
    short = (np.arange(8, dtype=np.int64), np.linspace(0.1, 0.2, 8))
    assert abrupt_bar([*golden_series, short]) == pytest.approx(
        abrupt_bar(golden_series), rel=0, abs=0
    )


def test_an_empty_pool_gives_nan_rather_than_raising():
    """A deliberate deviation: the reference raises from ``np.concatenate([])``."""
    assert np.isnan(abrupt_bar([(np.arange(5), np.zeros(5))]))


# --- a condition alone (the bar is pooled) ------------------------------------------


def _own_noise(frames: np.ndarray, values: np.ndarray) -> float:
    """The curvature noise ``rule_d`` measures for one series, recomputed."""
    d = derivatives(frames, values)
    mag = np.abs(d.accel[d.evaluable])
    mag = mag[np.isfinite(mag)]
    quiet = mag[mag <= np.median(mag)]
    return float(np.median(np.abs(quiet - np.median(quiet))) * MAD_TO_SIGMA)


def test_the_abstaining_condition_gets_a_window_when_it_is_alone(recording: dict):
    """Condition D abstains in both pooled runs, and gets a window on its own.

    The recorded ``alone`` run is the reference's own answer for D by itself, so
    this is the page's sentence "No window means the rule could not choose one
    for that condition *in that run*" pinned against the reference: D's verdict
    depends on what else is in the run.
    """
    runs = recording["runs"]
    for name in ("page", "coverage"):
        assert _by_name(runs[name])["Condition D"]["outcome"] == "too_noisy", name
    alone = runs["alone"]["conditions"][0]
    assert alone["outcome"] == "found"
    frames, values = _series_of(alone)
    result = rule_d(frames, values, abrupt_bar([(frames, values)]))
    assert result.outcome is WindowOutcome.FOUND
    assert result.window == tuple(alone["window"])


def test_a_condition_alone_is_never_too_noisy(golden_series: list):
    """Review focus: with one series in the pool, ``too_noisy`` is unreachable.

    The bar pooled from one series is that series' quiet level plus
    ``ABRUPT_K`` of the very noise ``rule_d`` measures for it, so the headroom
    is at least ``ABRUPT_K`` (10), above ``MIN_BAR_OVER_NOISE`` (3). Checked on
    every recorded series that is long enough to be assessed, and on 300 seeded
    random series spanning quiet to very noisy, including ones that are too
    noisy in a pooled run.
    """
    rng = np.random.default_rng(20261104)
    random_series = []
    for _ in range(300):
        n = int(rng.integers(MIN_SERIES_FRAMES + 3, 80))
        frames = np.arange(1, n + 1, dtype=np.int64)
        values = np.abs(
            0.1
            + rng.normal(0.0, rng.uniform(1e-4, 0.2), n)
            + rng.uniform(-0.01, 0.01) * frames
        )
        random_series.append((frames, values))

    for frames, values in [*golden_series, *random_series]:
        if len(frames) < MIN_SERIES_FRAMES:
            continue
        bar = abrupt_bar([(frames, values)])
        result = rule_d(frames, values, bar)
        assert result.outcome is not WindowOutcome.TOO_NOISY
        noise = _own_noise(frames, values)
        if noise > 0 and np.isfinite(bar):
            assert bar / noise >= ABRUPT_K


def test_a_condition_alone_can_be_too_noisy_only_below_the_floor():
    """The guarantee is ``abrupt_k >= min_bar_over_noise``, not unconditional.

    Lowering ``abrupt_k`` below the abstention floor brings ``too_noisy`` back
    for a condition alone, which is what makes the previous test a statement
    about the defaults rather than about every setting.
    """
    rng = np.random.default_rng(7)
    frames = np.arange(1, 41, dtype=np.int64)
    values = 0.1 + rng.normal(0.0, 0.01, 40)
    bar = abrupt_bar([(frames, values)], abrupt_k=0.0)
    result = rule_d(frames, values, bar)
    assert result.outcome is WindowOutcome.TOO_NOISY


# --- sustained ----------------------------------------------------------------------


def test_a_run_of_exactly_persist_frames_counts():
    """``>=`` in ``prototype_rule_d._sustained``, not ``>``."""
    mask = np.array([False, True, True, False, False])
    evaluable = np.ones(5, dtype=bool)
    assert sustained(mask, evaluable, persist=2) == [(1, 2)]
    assert sustained(mask, evaluable, persist=3) == []


def test_a_single_frame_excursion_is_not_sustained():
    """One frame above the bar is noise, not an event (``_PERSIST``)."""
    mask = np.array([False, True, False, True, False, True, False])
    assert sustained(mask, np.ones(7, dtype=bool), persist=PERSIST) == []


def test_a_run_still_open_at_the_end_is_admitted():
    """The closing clause uses ``len(mask) - start``."""
    mask = np.array([False, False, True, True])
    assert sustained(mask, np.ones(4, dtype=bool), persist=2) == [(2, 3)]


def test_a_non_evaluable_frame_breaks_a_run():
    """``evaluable`` gates the run, it does not merely mask the output."""
    mask = np.ones(6, dtype=bool)
    evaluable = np.array([True, True, False, True, True, True])
    assert sustained(mask, evaluable, persist=3) == [(3, 5)]


# --- behavioural controls: one per exit ---------------------------------------------


def _smooth_v_series(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A CV-shaped curve: a fall to an interior minimum, then a gentle rise."""
    frames = np.arange(n, dtype=np.int64)
    values = 0.08 - 0.004 * np.minimum(frames, 12) + 0.0015 * np.maximum(frames - 12, 0)
    return frames, values


def _noisy_series() -> tuple[np.ndarray, np.ndarray]:
    """Flat, plus white noise an order of magnitude above any real signal."""
    rng = np.random.default_rng(7)
    return np.arange(40, dtype=np.int64), 0.1 + rng.normal(0.0, 0.01, 40)


def _falling_series() -> tuple[np.ndarray, np.ndarray]:
    """A monotone fall, so the trend's minimum is the last smoothed frame."""
    frames = np.arange(30, dtype=np.int64)
    return frames, 0.2 * np.exp(-frames / 5.0)


def _overflowing_series() -> tuple[np.ndarray, np.ndarray]:
    """Values so large their nine-term sum overflows to ``inf``.

    The differences between them are ordinary, so the rate and acceleration
    stay finite while the trend does not — the only way past the earlier guards
    to :attr:`WindowOutcome.NO_SMOOTHED_VALUE`.
    """
    return np.arange(40, dtype=np.int64), np.full(40, 3e307)


def test_exit_found_on_a_clean_v_shaped_curve():
    """A window that opens at the basin and runs to the last evaluable frame."""
    frames, values = _smooth_v_series(40)
    result = rule_d(frames, values, bar=1.0)
    assert result.outcome is WindowOutcome.FOUND
    assert result.detail == ""
    assert result.window is not None
    start, end = result.window
    assert start < end
    assert start >= int(frames[0])
    assert end <= int(frames[-1])


def test_exit_too_few_frames_at_ten_and_not_at_eleven():
    """The guard is ``len(frames) < 11``."""
    ten = rule_d(np.arange(10, dtype=np.int64), np.full(10, 0.1), bar=1.0)
    assert ten.outcome is WindowOutcome.TOO_FEW_FRAMES
    assert ten.detail == "only 10 frames"

    eleven = rule_d(np.arange(11, dtype=np.int64), np.full(11, 0.1), bar=1.0)
    assert eleven.outcome is not WindowOutcome.TOO_FEW_FRAMES


def test_exit_too_few_evaluable_frames_at_thirteen_and_not_at_fourteen():
    """The 11-frame quirk in the length guard, made visible.

    With a nine-frame smoother and three edge frames, a length-``L`` series has
    exactly ``L - 10`` finite evaluable accelerations, so the "< 4" guard at
    in ``rule_d`` bites until ``L == 14`` — three frames past the length guard's 11.
    That gap is the transcribed quirk of ``_MIN_SERIES_FRAMES``.
    """
    frames, values = _smooth_v_series(13)
    thirteen = rule_d(frames, values, bar=1.0)
    assert thirteen.outcome is WindowOutcome.TOO_FEW_EVALUABLE_FRAMES
    assert thirteen.detail == "too few evaluable frames"

    frames, values = _smooth_v_series(14)
    fourteen = rule_d(frames, values, bar=1.0)
    assert fourteen.outcome is not WindowOutcome.TOO_FEW_EVALUABLE_FRAMES


def test_exit_too_noisy_when_the_bar_sits_inside_the_curves_own_noise():
    """Flat plus white noise, judged against a bar far below that noise."""
    frames, values = _noisy_series()
    result = rule_d(frames, values, bar=1e-9)
    assert result.outcome is WindowOutcome.TOO_NOISY
    assert result.detail.startswith("bar is ")
    assert f"(<{MIN_BAR_OVER_NOISE:g})" in result.detail
    assert result.window is None


def test_exit_too_noisy_is_a_ratio_not_a_sample_size():
    """The same noisy curve passes once the bar clears its noise.

    ``_MIN_BAR_OVER_NOISE`` replaced a cell-count cutoff in the reference; the
    test is ``bar / sigma``, so raising the bar alone flips the verdict.
    """
    frames, values = _noisy_series()
    assert rule_d(frames, values, bar=1.0).outcome is not WindowOutcome.TOO_NOISY


def test_exit_no_smoothed_value_needs_a_non_finite_trend_over_finite_slopes():
    """The one exit ordinary data cannot reach.

    The ``no smoothed value`` exit fires only when the *trend* is nowhere finite while the
    acceleration still has four finite evaluable entries. NaN cannot do it: a
    finite ``rate[k]`` needs ten consecutive finite values, and ten consecutive
    finite values contain a full nine-frame window, so the trend is finite
    there. Overflow can: the differences between huge values are ordinary, but
    their nine-term sum is ``inf``. Recorded as a reference finding, not fixed.
    """
    frames, values = _overflowing_series()
    with np.errstate(over="ignore"):
        result = rule_d(frames, values, bar=1.0)
    assert result.outcome is WindowOutcome.NO_SMOOTHED_VALUE
    assert result.detail == "no smoothed value"


def test_exit_abrupt_event_at_start_when_the_minimum_is_the_last_frame():
    """A monotone fall puts the basin past the last evaluable frame.

    ``end`` then defaults to ``last_ev``, which cannot exceed ``start``, and
    the ``end <= start`` exit fires.
    """
    frames, values = _falling_series()
    result = rule_d(frames, values, bar=1e6)
    assert result.outcome is WindowOutcome.ABRUPT_EVENT_AT_START
    assert result.detail == "abrupt event immediately after the start"
    assert result.window is None


def _roughening_series(
    onset: int = 38, quiet: float = 0.0002, loud: float = 0.0016
) -> tuple[np.ndarray, np.ndarray]:
    """A smooth rising trend whose *residual scatter* jumps at *onset*.

    Exactly the failure the second end criterion exists for
    (``prototype_rule_d.rule_d``): "the smoothed trend stays smooth while the raw
    series bounces around it", so no derivative test can see it.

    The scatter is a deterministic alternating square wave rather than noise, so
    the test needs no seed and cannot flake: a 9-frame centred mean of ``±s``
    leaves a residual of magnitude ``s(1 ± 1/9)``, so the 7-frame roughness mean
    tracks ``s`` closely and steps by the same 8x the scatter does.
    """
    frames = np.arange(60, dtype=np.int64)
    trend = 0.05 + 0.0008 * frames
    scale = np.where(frames < onset, quiet, loud)
    return frames, trend + scale * np.where(frames % 2 == 0, 1.0, -1.0)


def test_the_roughness_criterion_ends_a_window_the_derivatives_cannot():
    """Rule D's *second* end criterion, on its own.

    The acceleration bar is set out of reach (``bar=1e6``), so nothing here can
    end the window except the roughness block — which must cut it near the
    scatter onset instead of running to the last evaluable frame.
    """
    frames, values = _roughening_series()
    result = rule_d(frames, values, bar=1e6)
    assert result.outcome is WindowOutcome.FOUND
    assert result.events == []
    assert result.window is not None
    start, end = result.window
    assert 36 <= end <= 48, f"window {result.window} does not end near the onset"
    assert end < int(frames[-1])
    del start


def test_raising_rough_k_disables_the_roughness_end():
    """The pair that proves the previous test is about the roughness block.

    With ``rough_k`` beyond reach, ``rough[i] > rough_k * base`` can never hold,
    so the same series must run to its last evaluable frame — same ``start``, a
    strictly later ``end``. This is the keyword-driven equivalent of deleting the
    block, and it is how the golden conditions decided by roughness are found
    below.
    """
    frames, values = _roughening_series()
    decided = rule_d(frames, values, bar=1e6)
    disabled = rule_d(frames, values, bar=1e6, rough_k=1e18)
    assert disabled.outcome is WindowOutcome.FOUND
    assert decided.window is not None and disabled.window is not None
    assert disabled.window[0] == decided.window[0]
    assert disabled.window[1] > decided.window[1]


def test_one_golden_condition_is_decided_by_the_roughness_criterion(golden: dict):
    """So the golden fixture is known to cover the block, not assumed to.

    If a future change stopped routing any condition through the roughness end,
    the fixture would quietly stop guarding those twenty lines and this test says
    so. Condition C's replicates split apart late; the split is seen by the
    roughness test before any adverse acceleration starts after the basin.
    """
    roughness_decided = {}
    for condition in golden["conditions"]:
        frames, values = _series_of(condition)
        base = rule_d(frames, values, golden["bar"])
        disabled = rule_d(frames, values, golden["bar"], rough_k=1e18)
        if base.window != disabled.window:
            roughness_decided[condition["condition"]] = (base.window, disabled.window)
    assert roughness_decided == {"Condition C": ((9, 17), (9, 26))}


def test_one_golden_condition_is_ended_by_an_adverse_event(golden: dict):
    """The *first* end criterion, on recorded data.

    Condition J drifts gradually (which the window rides through), then its rate
    jumps. The window ends one frame before the first sustained adverse
    excursion that starts after the basin, and switching the roughness test off
    does not move it, so the event is what ends it.
    """
    j = _by_name(golden)["Condition J"]
    frames, values = _series_of(j)
    result = rule_d(frames, values, golden["bar"])
    disabled = rule_d(frames, values, golden["bar"], rough_k=1e18)
    assert result.outcome is WindowOutcome.FOUND
    assert result.window is not None
    assert disabled.window == result.window
    assert (result.window[1] + 1, result.events[-1][1]) == tuple(result.events[-1])
    assert result.window[1] + 1 > result.window[0]


def test_the_empty_window_exit_is_reached_on_recorded_data(golden: dict):
    """Condition K is still converging when the run ends.

    Its best agreement falls in the last evaluable frames, so the window would
    be empty. Any event it has comes before the start: none follows it.
    """
    k = _by_name(golden)["Condition K"]
    frames, values = _series_of(k)
    result = rule_d(frames, values, golden["bar"])
    assert result.outcome is WindowOutcome.ABRUPT_EVENT_AT_START
    d = derivatives(frames, values)
    last_evaluable = int(d.rate_frames[d.evaluable][-1])
    assert int(frames[int(np.nanargmin(d.smooth_v))]) >= last_evaluable


def test_every_outcome_is_reached_by_the_controls():
    """A roster check: six exits, six members, none left unexercised."""
    with np.errstate(over="ignore"):
        reached = {
            rule_d(*_smooth_v_series(40), bar=1.0).outcome,
            rule_d(np.arange(10, dtype=np.int64), np.full(10, 0.1), bar=1.0).outcome,
            rule_d(*_smooth_v_series(13), bar=1.0).outcome,
            rule_d(*_noisy_series(), bar=1e-9).outcome,
            rule_d(*_overflowing_series(), bar=1.0).outcome,
            rule_d(*_falling_series(), bar=1e6).outcome,
        }
    assert reached == set(WindowOutcome)


# --- derivatives --------------------------------------------------------------------


def test_a_repeated_frame_gives_unit_spacing_rather_than_an_infinity():
    """A zero step is replaced with 1.0, for the rate and the acceleration."""
    frames = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    values = np.linspace(0.1, 0.2, len(frames))
    computed = derivatives(frames, values)
    assert not np.isinf(computed.rate[np.isfinite(computed.rate)]).any()
    assert not np.isinf(computed.accel[np.isfinite(computed.accel)]).any()


def test_edge_frames_zero_leaves_the_tail_evaluable():
    """``[len(x) - 0:]`` is empty; ``[-0:]`` would blank the whole array."""
    frames, values = _smooth_v_series(30)
    computed = derivatives(frames, values, edge_frames=0)
    finite = np.isfinite(computed.accel)
    np.testing.assert_array_equal(computed.evaluable, finite)


def test_the_frame_axis_is_sorted_and_the_values_follow_it():
    """Both arrays are reordered together."""
    order = np.array([5, 0, 3, 1, 4, 2])
    frames = np.arange(6, dtype=np.int64)[order]
    values = np.arange(6, dtype=float)[order]
    computed = derivatives(frames, values)
    np.testing.assert_array_equal(computed.frames, np.arange(6))
    np.testing.assert_array_equal(computed.values, np.arange(6.0))


# --- the CV step --------------------------------------------------------------------


def _cv_frame() -> pd.DataFrame:
    """Two strains over three frames, with one group below each guard."""
    return pd.DataFrame(
        {
            "Metadata_Strain": (
                ["a"] * 3 + ["b"] * 3 + ["a"] * 3 + ["b"] * 2 + ["a"] * 3
            ),
            "Metadata_FrameIndex": [0] * 6 + [1] * 5 + [2] * 3,
            "Object_Label": [1, 2, 3] * 2 + [1, 2, 3] + [1, 2] + [1, 2, 3],
            "Size_RobustMeanRadius": [
                10.0, 11.0, 12.0,   # a @ 0  -> counts
                20.0, 20.0, 20.0,   # b @ 0  -> counts, cv 0
                5.0, 6.0, 7.0,      # a @ 1  -> counts
                30.0, 31.0,         # b @ 1  -> only 2 replicates, dropped
                0.0, 0.0, 0.0,      # a @ 2  -> mean 0, below the 1e-9 floor
            ],
        }
    )


def test_the_cv_uses_the_sample_standard_deviation():
    """``ddof=1``; ``ddof=0`` would be 18% low at n=3 (``c10_cv_by_ph._cv_by_time``)."""
    series = mean_cv_series(
        _cv_frame().query("Metadata_FrameIndex == 0 and Metadata_Strain == 'a'"),
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    values = np.array([10.0, 11.0, 12.0])
    sample = float(np.std(values, ddof=1) / values.mean())
    population = float(np.std(values, ddof=0) / values.mean())
    assert series.mean_cv[0] == pytest.approx(sample, rel=1e-12)
    assert series.mean_cv[0] != pytest.approx(population, rel=1e-6)


def test_a_group_below_min_replicates_is_not_counted():
    """Frame 1 keeps strain ``a`` only: two replicates are not a dispersion."""
    series = mean_cv_series(
        _cv_frame(),
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    assert series.frames.tolist() == [0, 1]
    assert series.n_counted.tolist() == [2, 1]


def test_a_group_whose_mean_is_at_the_floor_is_not_counted():
    """The second guard in ``c10_cv_by_ph._cv_by_time``, easy to drop.

    Frame 2's only group has three replicates but a zero mean, so the frame
    disappears from the series entirely rather than arriving with a zero CV.
    """
    series = mean_cv_series(
        _cv_frame(),
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    assert 2 not in series.frames.tolist()


def test_an_uncounted_frame_is_absent_rather_than_zero():
    """Rule D's ``frames`` axis is the counted frames, not a padded grid."""
    frame = _cv_frame()
    frame.loc[frame["Metadata_FrameIndex"] == 0, "Size_RobustMeanRadius"] = np.nan
    series = mean_cv_series(
        frame,
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    assert 0 not in series.frames.tolist()
    assert (series.mean_cv != 0).all()


def test_a_phantom_row_does_not_lift_a_group_over_min_replicates():
    """The guard for the phantom filter — the one the *other* phantom test is not.

    What the ``Object_Label`` filter actually buys: ``size`` counts rows whether
    or not the metric is null, so a phantom **inflates ``n`` inside an existing
    under-populated group** while contributing nothing to ``mean``/``std``.
    Strain ``b`` at frame 1 has two real replicates and is correctly dropped by
    ``n >= min_replicates``; one phantom would lift it to three and have it
    counted with a two-replicate CV.

    Without the filter this gives ``n_counted == [2, 2]`` and a different
    ``mean_cv[1]``.
    """
    frame = _cv_frame()
    phantom = (
        frame[
            (frame["Metadata_Strain"] == "b") & (frame["Metadata_FrameIndex"] == 1)
        ]
        .iloc[[0]]
        .copy()
    )
    phantom["Object_Label"] = np.nan
    phantom["Size_RobustMeanRadius"] = np.nan
    padded = pd.concat([frame, phantom], ignore_index=True)
    series = mean_cv_series(
        padded,
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    assert series.frames.tolist() == [0, 1]
    assert series.n_counted.tolist() == [2, 1]


def test_a_phantom_row_in_a_group_of_its_own_changes_nothing():
    """A weaker property than it looks, and **not** the guard for the phantom filter.

    Rows for a strain that was detected nowhere form a group whose ``mean`` is
    NaN, and ``NaN > CV_MEAN_FLOOR`` is False — so the ``mean > 1e-9`` guard
    drops them whether or not the ``Object_Label`` filter exists. This test
    therefore passes with that filter deleted, which is why
    :func:`test_a_phantom_row_does_not_lift_a_group_over_min_replicates` exists
    beside it. Kept because "a nowhere-detected strain does not perturb the
    series" is still worth pinning; it just is not evidence for the filter.
    """
    frame = _cv_frame()
    phantom = frame.iloc[[0]].copy()
    phantom["Object_Label"] = np.nan
    phantom["Size_RobustMeanRadius"] = np.nan
    phantom["Metadata_Strain"] = "c"
    padded = pd.concat([frame, *[phantom] * 3], ignore_index=True)

    plain = mean_cv_series(
        frame,
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    with_phantoms = mean_cv_series(
        padded,
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    np.testing.assert_array_equal(plain.frames, with_phantoms.frames)
    np.testing.assert_allclose(plain.mean_cv, with_phantoms.mean_cv)
    np.testing.assert_array_equal(plain.n_counted, with_phantoms.n_counted)


def test_a_row_with_no_grouping_key_is_dropped():
    """An unpinned detection has no strain, so ``groupby`` drops it — as the
    reference's unpinned-detection filter does (``c10_cv_by_ph._detected``)."""
    frame = _cv_frame()
    unpinned = frame.iloc[:3].copy()
    unpinned["Metadata_Strain"] = None
    padded = pd.concat([frame, unpinned], ignore_index=True)
    plain = mean_cv_series(
        frame,
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    with_unpinned = mean_cv_series(
        padded,
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    np.testing.assert_array_equal(plain.n_counted, with_unpinned.n_counted)


def test_a_frame_with_no_object_label_column_still_returns_a_series():
    """The untested branch the phantom filter rests on, documented rather than defended.

    ``if label in rows.columns`` means a frame without the column gets **no**
    detection filter at all. There is no correct alternative — without the column
    phantoms are indistinguishable from detections — and every CLI measurements
    mirror carries it, so this is not a defect. What it means in practice: on a
    label-free frame the function still computes a series, and on one that also
    contained phantoms those phantoms would be counted. This pins the first half
    and states the second.
    """
    frame = _cv_frame()
    with_label = mean_cv_series(
        frame,
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    without_label = mean_cv_series(
        frame.drop(columns=["Object_Label"]),
        on="Size_RobustMeanRadius",
        replicate_by=["Metadata_Strain"],
        time_col="Metadata_FrameIndex",
    )
    # No phantoms in this frame, so dropping the column must change nothing.
    np.testing.assert_array_equal(with_label.frames, without_label.frames)
    np.testing.assert_array_equal(with_label.mean_cv, without_label.mean_cv)
    np.testing.assert_array_equal(with_label.n_counted, without_label.n_counted)


def test_a_missing_column_is_named():
    with pytest.raises(KeyError, match="Shape_Nonexistent"):
        mean_cv_series(
            _cv_frame(),
            on="Shape_Nonexistent",
            replicate_by=["Metadata_Strain"],
            time_col="Metadata_FrameIndex",
        )


def test_a_non_integral_time_axis_is_refused():
    """Elapsed hours are not a frame index."""
    frame = _cv_frame()
    frame["Metadata_Hours"] = frame["Metadata_FrameIndex"] * 0.5
    with pytest.raises(ValueError, match="whole frame numbers"):
        mean_cv_series(
            frame,
            on="Size_RobustMeanRadius",
            replicate_by=["Metadata_Strain"],
            time_col="Metadata_Hours",
        )


# --- the CV step against the reference ----------------------------------------------


@pytest.fixture(scope="module")
def series_rows() -> pd.DataFrame:
    """The synthetic table the recording was made from."""
    return pd.read_parquet(SERIES)


def _cv_kwargs(recording: dict) -> dict:
    return {
        "on": recording["on"],
        "replicate_by": list(recording["replicate_by"]),
        "time_col": recording["time_col"],
    }


def _condition_rows(rows: pd.DataFrame, recording: dict, name: str) -> pd.DataFrame:
    return rows[rows[recording["condition_col"]] == name]


def test_the_series_actually_contains_what_the_counting_step_must_ignore(
    series_rows: pd.DataFrame, recording: dict
):
    """Otherwise the detection rules are asserted on the fixture but never exercised.

    The reference selects detections with two flags a PhenoTypic frame does not
    have; the fixture carries the same subsets in PhenoTypic's own form. A
    phantom (undetected) row has a null ``Object_Label`` and a true
    ``QC_MetadataOnly``; an unpinned detection has a label and no replicate
    group; and some detected rows have no metric value at all.
    """
    label = series_rows["Object_Label"]
    replicate = series_rows[recording["replicate_by"][0]]
    metric = series_rows[recording["on"]]
    np.testing.assert_array_equal(
        label.isna().to_numpy(), series_rows["QC_MetadataOnly"].to_numpy()
    )
    assert label.isna().sum() > 0
    assert metric[label.isna()].isna().all()
    assert (replicate.isna() & label.notna()).sum() > 0
    assert (label.notna() & metric.isna()).sum() > 0


def test_both_counting_guards_bite_on_the_fixture(
    series_rows: pd.DataFrame, recording: dict
):
    """A group below ``MIN_REPLICATES`` and a group whose mean is undefined.

    Computed straight from the rows, independently of ``mean_cv_series``, so a
    regenerated fixture that stopped exercising either guard fails here.
    """
    kwargs = _cv_kwargs(recording)
    rows = series_rows[series_rows["Object_Label"].notna()]
    stats = rows.groupby(
        [recording["condition_col"], *kwargs["replicate_by"], kwargs["time_col"]],
        dropna=True,
    )[kwargs["on"]].agg(["size", "mean"])
    assert (stats["size"] < MIN_REPLICATES).sum() > 0
    assert ((stats["size"] >= MIN_REPLICATES) & stats["mean"].isna()).sum() > 0


def test_the_cv_series_matches_the_reference(
    series_rows: pd.DataFrame, golden: dict, recording: dict
):
    """Every synthetic condition's counted frames, mean CV and group count."""
    for condition in golden["conditions"]:
        series = mean_cv_series(
            _condition_rows(series_rows, recording, condition["condition"]),
            min_replicates=recording["constants"]["min_replicates"],
            **_cv_kwargs(recording),
        )
        np.testing.assert_array_equal(
            series.frames, np.asarray(condition["frames"], dtype=np.int64),
            err_msg=condition["condition"],
        )
        np.testing.assert_allclose(
            series.mean_cv,
            np.asarray(condition["values"], dtype=float),
            rtol=CV_RTOL,
            atol=0.0,
            err_msg=condition["condition"],
        )
        np.testing.assert_array_equal(
            series.n_counted, np.asarray(condition["n_counted"], dtype=np.int64),
            err_msg=condition["condition"],
        )


def test_the_cv_series_does_not_depend_on_input_row_order(
    series_rows: pd.DataFrame, recording: dict
):
    """The port is order-deterministic where the reference is not.

    polars' ``group_by`` is unordered, and the reference is measurably
    nondeterministic because of it. This port sorts both grouping levels, so
    its output is a function of the *set* of rows.

    The comparison is **exact**, not ``allclose``: the whole claim is about the
    last bits, so a tolerance would make the test unable to fail. It runs on the
    whole synthetic table, whose dense condition (ten groups of six colonies)
    is there so that summation order shows: without the row sort, a shuffled
    input moves the last bits of most frames. Sorting the group keys alone is
    not enough, because pandas accumulates each group's mean and sd in input
    order — the rows themselves have to be sorted.
    """
    shuffled = series_rows.sample(frac=1.0, random_state=20260925).reset_index(
        drop=True
    )
    kwargs = _cv_kwargs(recording)
    for name in sorted(series_rows[recording["condition_col"]].unique()):
        plain = mean_cv_series(_condition_rows(series_rows, recording, name), **kwargs)
        reordered = mean_cv_series(_condition_rows(shuffled, recording, name), **kwargs)
        np.testing.assert_array_equal(plain.frames, reordered.frames, err_msg=name)
        np.testing.assert_array_equal(plain.mean_cv, reordered.mean_cv, err_msg=name)
        np.testing.assert_array_equal(plain.n_counted, reordered.n_counted, err_msg=name)


@pytest.mark.parametrize("run_name", RUNS)
def test_the_rows_feed_rule_d_the_same_windows(
    series_rows: pd.DataFrame, recording: dict, run_name: str
):
    """End to end: rows -> CV series -> pooled bar -> Rule D -> the golden window.

    The bar is computed by the port from the port's own series, not read from
    the recording, so this is the whole chain a ``FittingWindow`` runs.
    """
    run = recording["runs"][run_name]
    kwargs = _cv_kwargs(recording)
    series = {
        c["condition"]: mean_cv_series(
            _condition_rows(series_rows, recording, c["condition"]), **kwargs
        )
        for c in run["conditions"]
    }
    bar = abrupt_bar([(s.frames, s.mean_cv) for s in series.values()])
    assert bar == pytest.approx(run["bar"], rel=BAR_RTOL)
    for condition in run["conditions"]:
        s = series[condition["condition"]]
        result = rule_d(s.frames, s.mean_cv, bar)
        assert result.outcome.value == condition["outcome"], condition["condition"]
        assert result.window == (
            tuple(condition["window"]) if condition["window"] else None
        ), condition["condition"]


# --- doctests -----------------------------------------------------------------------


def test_the_module_doctests_run():
    """The examples in the docstrings are executable, and are executed.

    Every example prints a Python scalar, list or bool rather than a numpy array
    repr. Array reprs pad differently under
    ``numpy.set_printoptions(legacy=...)``, which any earlier import in the
    pytest process can set, so an example pinning the spacing passes alone and
    fails in a shard — which is exactly how this test first failed.
    """
    import doctest

    from phenotypic.analysis.qc import _fitting_window_rule

    results = doctest.testmod(_fitting_window_rule, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
