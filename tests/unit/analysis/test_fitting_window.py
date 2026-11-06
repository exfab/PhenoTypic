"""The ``FittingWindow`` quality check — the pipeline half of Rule D.

``tests/unit/analysis/test_fitting_window_rule.py`` pins the *algorithm* against
the reference. This module pins the *check*: that it pools the abrupt-change bar
over the whole run rather than per condition, that a condition with no window
fails, that its DuckDB table is at the condition grain, and that its figure
carries one page per condition.

**The golden tests drive the check from rows.** ``tests/fixtures/rule_d/`` holds
the synthetic table the reference was run on and the reference's answers for
three runs of it, so the check is tested where a user starts — a measurements
frame — against the reference's own windows and outcomes. Floats are compared
with a tolerance (the reference is nondeterministic in its last digits; see the
fixture README); windows and outcomes are compared exactly.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis.qc import FittingWindow
from phenotypic.analysis.qc._fitting_window_rule import (
    WindowOutcome,
    abrupt_bar,
    mean_cv_series,
    rule_d,
)
from phenotypic.schema import CULTURE, OBJECT, QUALITY_WINDOW, SAMPLE, SIZE

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "rule_d"

CONDITION = "Metadata_Condition"
REPLICATE = str(SAMPLE.BIO_REPLICATE)
TIME_COL = str(CULTURE.FRAME_INDEX)
HOURS = str(CULTURE.TIME)
METRIC = str(SIZE.ROBUST_MEAN_RADIUS)
LABEL = str(OBJECT.LABEL)

#: The reference's own pooled bar varies in its last digits (fixture README).
FLOAT_RTOL = 1e-9

OUTCOME = str(QUALITY_WINDOW.OUTCOME)
START = str(QUALITY_WINDOW.START)
END = str(QUALITY_WINDOW.END)
REASON = str(QUALITY_WINDOW.REASON)
BEST_CV = str(QUALITY_WINDOW.BEST_CV)
BAR = str(QUALITY_WINDOW.BAR)
N_GROUPS = str(QUALITY_WINDOW.N_GROUPS)
N_FRAMES = str(QUALITY_WINDOW.N_FRAMES_COUNTED)


# --- fixtures -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def recording() -> dict:
    """The reference's answers for the three recorded runs."""
    return json.loads(
        (FIXTURES / "synthetic_windows.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    """The synthetic table the recording was made from."""
    return pd.read_parquet(FIXTURES / "synthetic_series.parquet")


def _make_check(**overrides) -> FittingWindow:
    """The check, configured for the synthetic table."""
    params = {
        "on": METRIC,
        "groupby": [CONDITION],
        "replicate_by": [REPLICATE],
        "time_col": TIME_COL,
    }
    params.update(overrides)
    return FittingWindow(**params)


def _only(rows: pd.DataFrame, names) -> pd.DataFrame:
    return rows[rows[CONDITION].isin(list(names))]


@pytest.fixture(scope="module")
def analyzed(rows: pd.DataFrame) -> FittingWindow:
    """The check, analyzed over every synthetic condition (the coverage run)."""
    check = _make_check()
    check.analyze(rows)
    return check


def _verdicts(check: FittingWindow) -> dict[str, pd.Series]:
    """The condition-grain table, keyed by condition name."""
    return {row[CONDITION]: row for _, row in check.to_table().iterrows()}


def _window_of(row: pd.Series) -> tuple[int, int] | None:
    if pd.isna(row[START]):
        return None
    return int(row[START]), int(row[END])


PAIR = ("Condition A", "Condition D")


@pytest.fixture(scope="module")
def pair_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Two conditions — enough for anything structural, and cheap."""
    return _only(rows, PAIR)


# --- the reference, end to end ------------------------------------------------------


@pytest.mark.parametrize("run_name", ["page", "coverage", "alone"])
def test_every_condition_reproduces_the_reference(
    recording: dict, rows: pd.DataFrame, run_name: str
):
    """Window, outcome and sentence for every condition, from rows, through the check."""
    run = recording["runs"][run_name]
    check = _make_check()
    check.analyze(_only(rows, run["conditions_in_run"]))
    verdicts = _verdicts(check)
    assert sorted(verdicts) == sorted(run["conditions_in_run"])

    mismatches = []
    for condition in run["conditions"]:
        row = verdicts[condition["condition"]]
        expected_window = (
            tuple(condition["window"]) if condition["window"] is not None else None
        )
        actual = (row[OUTCOME], _window_of(row), row[REASON])
        expected = (condition["outcome"], expected_window, condition["reason"])
        if actual != expected:
            mismatches.append(f"{condition['condition']}: {actual!r} != {expected!r}")
    assert not mismatches, "\n".join(mismatches)
    assert float(verdicts[run["conditions"][0]["condition"]][BAR]) == pytest.approx(
        run["bar"], rel=FLOAT_RTOL
    )


def test_the_example_series_is_the_recorded_page_series(rows: pd.DataFrame):
    """The class docstring's input is the series the reference was run on.

    ``make_synthetic_window_series`` plus ``AddFrameIndex`` must give conditions
    A to D of the fixture, value for value, so the example's windows rest on the
    reference's recording rather than on the port's say-so.
    """
    from phenotypic.data import make_synthetic_window_series
    from phenotypic.post import AddFrameIndex

    example = AddFrameIndex(plate_by=["PlateID"]).apply(make_synthetic_window_series())
    page = _only(rows, ("Condition A", "Condition B", "Condition C", "Condition D"))
    keys = [CONDITION, REPLICATE, TIME_COL]
    # The fixture numbers colonies 1-12 within a condition; the example 1-4 per
    # plate. Rank within (condition, replicate, frame) gives both the same key.
    ours = example.assign(
        _k=example.groupby(keys)[LABEL].rank(method="first")
    ).sort_values([*keys, "_k"])
    theirs = page.assign(
        _k=page.groupby(keys)[LABEL].rank(method="first")
    ).sort_values([*keys, "_k"])
    # The metric is float arithmetic, so its last bit depends on the CPU: a
    # Cascade Lake node differs from the recording node by up to 4e-16
    # relative. 1e-12 is far above that and far below any real change (a
    # different seed or formula moves values by whole units).
    np.testing.assert_allclose(
        ours[METRIC].to_numpy(), theirs[METRIC].to_numpy(), rtol=1e-12, atol=0
    )
    np.testing.assert_array_equal(
        ours[TIME_COL].astype("int64").to_numpy(), theirs[TIME_COL].to_numpy()
    )
    np.testing.assert_array_equal(ours[HOURS].to_numpy(), theirs[HOURS].to_numpy())


# --- categorical condition groups --------------------------------------------------


@pytest.mark.parametrize("multiple_groupers", [False, True])
@pytest.mark.parametrize("include_missing", [False, True])
def test_categorical_conditions_match_observed_string_groups(
    rows: pd.DataFrame, multiple_groupers: bool, include_missing: bool
):
    """Filtering categorical conditions must not create empty QC groups."""
    frame = rows.copy()
    groupby = [CONDITION]
    if multiple_groupers:
        batch = "Metadata_TestBatch"
        frame[batch] = np.where(frame[CONDITION] == PAIR[0], "first", "second")
        groupby.append(batch)
    if include_missing:
        frame.loc[frame[CONDITION] == PAIR[1], CONDITION] = np.nan
    selected = frame[CONDITION].eq(PAIR[0]) | (
        frame[CONDITION].isna() if include_missing else frame[CONDITION].eq(PAIR[1])
    )
    ordinary = frame.loc[selected].copy()
    for column in groupby:
        frame[column] = pd.Categorical(frame[column])
    categorical = frame.loc[selected].copy()
    before = categorical.copy(deep=True)

    expected = _make_check(groupby=groupby)
    expected_result = expected.analyze(ordinary)
    actual = _make_check(groupby=groupby)
    actual_result = actual.analyze(categorical)

    pairs = [
        (actual_result.sort_index(), expected_result.sort_index()),
        (actual.summary(), expected.summary()),
        (actual.to_table(), expected.to_table()),
    ]
    for observed, control in pairs:
        observed = observed.copy()
        for column in groupby:
            observed[column] = observed[column].astype(object)
        pd.testing.assert_frame_equal(
            observed.sort_values(groupby).reset_index(drop=True),
            control.sort_values(groupby).reset_index(drop=True),
        )
    assert len(actual.to_table()) == 2
    assert [page.key for page in actual.inspect().pages] == [
        page.key for page in expected.inspect().pages
    ]
    pd.testing.assert_frame_equal(categorical, before)


@pytest.mark.parametrize("categorical", [False, True])
def test_filtering_all_conditions_returns_empty_outputs(
    pair_frame: pd.DataFrame, categorical: bool
):
    """An empty selection has no verdicts, including retained category levels."""
    frame = pair_frame.copy()
    if categorical:
        frame[CONDITION] = pd.Categorical(frame[CONDITION])
    check = _make_check()
    result = check.analyze(frame.iloc[:0])

    assert result.empty
    assert check.summary().empty
    table = check.to_table()
    assert table.empty
    assert list(table.columns) == [
        CONDITION,
        *(str(member) for member in QUALITY_WINDOW),
        check.metric_col(),
        check.status_col(),
        check.flag_col(),
    ]


# --- the pooled bar -----------------------------------------------------------------


def test_the_bar_is_one_value_on_every_row(analyzed: FittingWindow):
    """A run-level constant, not a per-condition number.

    This is the single most damaging way to get the port wrong: a per-condition
    bar type-checks, runs, and silently reclassifies conditions.
    """
    results = analyzed.results()
    assert results[BAR].nunique() == 1
    assert results[CONDITION].nunique() > 1


def test_only_assessable_conditions_contribute_to_the_pooled_bar(
    recording: dict, rows: pd.DataFrame, analyzed: FittingWindow
):
    """Which conditions the bar is pooled over — measured, one drop at a time.

    ``abrupt_bar`` skips a series shorter than ``min_series_frames`` entirely,
    so such a condition **cannot** move the bar while still being *judged*
    against it. In the synthetic run that is condition H (8 frames). Every
    other condition moves the bar when dropped.
    """
    full_bar = float(analyzed.results()[BAR].iloc[0])
    names = recording["runs"]["coverage"]["conditions_in_run"]
    short = [
        c["condition"]
        for c in recording["runs"]["coverage"]["conditions"]
        if len(c["frames"]) < analyzed.min_series_frames
    ]
    assert short == ["Condition H"]

    moved, unmoved = [], []
    for name in names:
        reduced = _make_check()
        reduced.analyze(_only(rows, [n for n in names if n != name]))
        bar = float(reduced.results()[BAR].iloc[0])
        (unmoved if bar == full_bar else moved).append(name)

    assert unmoved == short
    assert len(moved) == len(names) - 1


def test_a_conditions_verdict_depends_on_the_rest_of_the_run(
    rows: pd.DataFrame, analyzed: FittingWindow
):
    """The per-condition-bar mutant, killed at the check level.

    Analyzing one condition on its own *is* a per-condition bar. If the check
    pooled per condition, every verdict here would match its verdict in the
    full run. Condition D's does not.
    """
    baseline = _verdicts(analyzed)
    disagreements = []
    for name in baseline:
        alone = _make_check()
        alone.analyze(_only(rows, [name]))
        (row,) = _verdicts(alone).values()
        here = (row[OUTCOME], _window_of(row))
        there = (baseline[name][OUTCOME], _window_of(baseline[name]))
        if here != there:
            disagreements.append(name)
    assert "Condition D" in disagreements


def test_a_condition_analyzed_alone_is_never_too_noisy(rows: pd.DataFrame):
    """Review focus: the pooled bar makes ``too_noisy`` a statement about a run.

    Every synthetic condition, analyzed on its own at the default settings,
    gets an outcome other than ``too_noisy`` — including D, which is too noisy
    in both pooled runs.
    """
    outcomes = {}
    for name in sorted(rows[CONDITION].unique()):
        alone = _make_check()
        alone.analyze(_only(rows, [name]))
        (row,) = _verdicts(alone).values()
        outcomes[name] = row[OUTCOME]
    assert str(WindowOutcome.TOO_NOISY) not in outcomes.values(), outcomes
    assert outcomes["Condition D"] == str(WindowOutcome.FOUND)


# --- verdicts -----------------------------------------------------------------------


def test_the_coverage_run_splits_seven_pass_four_fail(analyzed: FittingWindow):
    statuses = [row[analyzed.status_col()] for row in _verdicts(analyzed).values()]
    assert len(statuses) == 11
    assert statuses.count("pass") == 7
    assert statuses.count("fail") == 4
    assert statuses.count("warn") == 0


def test_every_non_found_outcome_fails(analyzed: FittingWindow):
    """No window is always a failure."""
    for name, row in _verdicts(analyzed).items():
        metric = row[analyzed.metric_col()]
        if row[OUTCOME] == str(WindowOutcome.FOUND):
            assert np.isfinite(metric), name
            assert row[analyzed.status_col()] != "fail", name
            assert not bool(row[analyzed.flag_col()]), name
        else:
            assert metric == np.inf, name
            assert row[analyzed.status_col()] == "fail", name
            assert bool(row[analyzed.flag_col()]), name


def test_a_warning_condition_is_not_flagged(rows: pd.DataFrame):
    """The condition-grain flag follows ``fail`` only, as the base class's does.

    With the default thresholds every found window here passes, so a flag
    written as "not pass" would look identical. Tighter thresholds put some
    found windows in the warn band, where the flag must stay down.
    """
    check = _make_check(warn_threshold=0.03, fail_threshold=0.05)
    check.analyze(rows)
    table = check.to_table()
    statuses = set(table[check.status_col()])
    assert {"pass", "warn", "fail"} <= statuses
    pd.testing.assert_series_equal(
        table[check.flag_col()],
        table[check.status_col()] == "fail",
        check_names=False,
    )
    member_flags = check.results().groupby(CONDITION)[check.flag_col()].first()
    assert table.set_index(CONDITION)[check.flag_col()].to_dict() == member_flags.to_dict()


def test_a_windowless_condition_with_the_best_minimum_still_fails(
    recording: dict, analyzed: FittingWindow
):
    """Why the metric is ``+inf`` without a window, and not the series minimum.

    Condition K is still converging when the run ends, so its CV reaches the
    lowest value of any condition — and Rule D cannot place a window. A metric
    of "window mean if found, else the series minimum" would rank K the best
    condition in the run and pass it. The adopted metric fails it.
    """
    verdicts = _verdicts(analyzed)
    k = verdicts["Condition K"]
    recorded = next(
        c for c in recording["runs"]["coverage"]["conditions"]
        if c["condition"] == "Condition K"
    )
    assert k[OUTCOME] == str(WindowOutcome.ABRUPT_EVENT_AT_START)
    assert k[BEST_CV] == pytest.approx(min(recorded["values"]), rel=FLOAT_RTOL)

    def drafted(record) -> float:
        if record[OUTCOME] == str(WindowOutcome.FOUND):
            return float(record[analyzed.metric_col()])
        return float(record[BEST_CV])

    scores = {name: drafted(record) for name, record in verdicts.items()}
    assert scores["Condition K"] == min(scores.values())
    assert scores["Condition K"] < analyzed.warn_threshold

    assert k[analyzed.metric_col()] == np.inf
    assert k[analyzed.status_col()] == "fail"


def test_the_metric_is_the_window_mean_of_the_measured_series(
    rows: pd.DataFrame, analyzed: FittingWindow
):
    row = _verdicts(analyzed)["Condition A"]
    series = mean_cv_series(
        _only(rows, ["Condition A"]),
        on=METRIC,
        replicate_by=[REPLICATE],
        time_col=TIME_COL,
    )
    start, end = _window_of(row)
    inside = (series.frames >= start) & (series.frames <= end)
    assert row[analyzed.metric_col()] == pytest.approx(
        float(series.mean_cv[inside].mean())
    )


# --- the frame base -----------------------------------------------------------------


def test_a_shifted_time_axis_shifts_every_window_by_exactly_one(rows: pd.DataFrame):
    """``AddFrameIndex`` is 1-based; a 0-based axis must change no outcome.

    Rule D compares frames with each other and never with an absolute number,
    so the offset must move every window by exactly one and change no outcome.
    """
    one_based = _make_check()
    one_based.analyze(rows)
    zero_based = _make_check()
    zero_based.analyze(rows.assign(**{TIME_COL: rows[TIME_COL] - 1}))

    before, after = _verdicts(one_based), _verdicts(zero_based)
    assert set(before) == set(after)
    for name in before:
        assert before[name][OUTCOME] == after[name][OUTCOME], name
        window = _window_of(before[name])
        expected = None if window is None else (window[0] - 1, window[1] - 1)
        assert _window_of(after[name]) == expected, name
    assert float(one_based.results()[BAR].iloc[0]) == float(
        zero_based.results()[BAR].iloc[0]
    )


def test_a_string_time_column_gives_the_same_table_as_an_integer_one(
    rows: pd.DataFrame,
):
    """A frame index that arrives as text must not change the order.

    Sorted as text, frame 10 lands between 1 and 2 — which scrambles the series
    axis Rule D's rolling windows and window bounds are computed over. Every
    published number would still look plausible, so only a comparison against
    the numeric run can see it.
    """
    assert int(rows[TIME_COL].max()) >= 10

    numeric = _make_check()
    numeric.analyze(rows)
    textual = _make_check()
    textual.analyze(rows.assign(**{TIME_COL: rows[TIME_COL].astype(str)}))

    pd.testing.assert_frame_equal(numeric.to_table(), textual.to_table())

    # ``rule_d`` re-sorts its frames, so the verdict alone survives a text axis;
    # the figure does not. Its trend is computed on sorted frames and drawn
    # against the series' own frame order, so a scrambled series misplaces it.
    for ours, theirs in zip(
        numeric.inspect().pages, textual.inspect().pages, strict=True
    ):
        assert ours.key == theirs.key
        for a, b in zip(ours.figure.data, theirs.figure.data, strict=True):
            np.testing.assert_array_equal(np.asarray(a.x), np.asarray(b.x))
            np.testing.assert_array_equal(
                np.asarray(a.y, dtype=float), np.asarray(b.y, dtype=float)
            )


def test_analyze_hands_back_the_callers_time_column_unchanged(rows: pd.DataFrame):
    """The text frame index is read as numbers for the series, not rewritten.

    The coercion lives on a copy: the returned frame carries the caller's own
    column, and the caller's frame is not touched.
    """
    textual = rows.assign(**{TIME_COL: rows[TIME_COL].astype(str)})
    before = textual.copy()

    result = _make_check().analyze(textual)

    pd.testing.assert_frame_equal(textual, before)
    pd.testing.assert_series_equal(
        result[TIME_COL].sort_index(), textual[TIME_COL].sort_index()
    )
    assert pd.api.types.is_object_dtype(result[TIME_COL])


def test_the_bar_is_nan_when_no_condition_is_long_enough(pair_frame: pd.DataFrame):
    """Nothing contributes to the pool, so there is no bar — and nothing reads it.

    Every condition exits ``too_few_frames`` before the bar is consulted, so a
    NaN bar cannot change a verdict; it is published as NaN rather than as a
    number nothing produced.
    """
    check = _make_check(min_series_frames=10_000)
    check.analyze(pair_frame)
    assert check.results()[BAR].isna().all()
    assert {row[OUTCOME] for row in _verdicts(check).values()} == {
        str(WindowOutcome.TOO_FEW_FRAMES)
    }


def test_hours_are_refused_as_a_time_axis(rows: pd.DataFrame):
    """The window is in frames; hours cannot line plates up and are refused."""
    rows = rows.assign(**{HOURS: rows[HOURS] + 0.5})
    with pytest.raises(ValueError, match="whole frame numbers"):
        _make_check(time_col=HOURS).analyze(rows)


# --- emitted columns ----------------------------------------------------------------


def test_the_flag_is_a_bool_and_the_outcome_is_a_string(analyzed: FittingWindow):
    """The categorical answer is a separate column from the boolean flag."""
    results = analyzed.results()
    assert results[analyzed.flag_col()].dtype == bool
    outcomes = results[OUTCOME]
    assert all(isinstance(value, str) for value in outcomes.unique())
    assert set(outcomes.unique()) <= {member.value for member in WindowOutcome}


def test_the_window_bounds_are_null_unless_a_window_was_found(
    analyzed: FittingWindow,
):
    for row in _verdicts(analyzed).values():
        if row[OUTCOME] == str(WindowOutcome.FOUND):
            assert _window_of(row) is not None
            assert row[REASON] == ""
        else:
            assert pd.isna(row[START]) and pd.isna(row[END])
            assert row[REASON]


def test_n_groups_is_the_rounded_median_group_count(
    recording: dict, analyzed: FittingWindow
):
    """Condition E has ten replicate groups; the rest three."""
    verdicts = _verdicts(analyzed)
    for condition in recording["runs"]["coverage"]["conditions"]:
        expected = int(round(float(np.median(condition["n_counted"]))))
        assert int(verdicts[condition["condition"]][N_GROUPS]) == expected
    assert int(verdicts["Condition E"][N_GROUPS]) == 10


def _uneven_group_count_frame() -> pd.DataFrame:
    """Two conditions whose counted-group count is not the same at every frame.

    ``plan`` is the number of replicate groups present at each frame, and a
    group is counted only when it reaches ``min_replicates``, so the per-frame
    counted-group counts are exactly ``plan``'s values. Condition X gives
    ``[3, 4]`` (median 3.5) and Y gives ``[3, 3, 4]`` (median 3).
    """
    plan = {"X": [3, 4], "Y": [3, 3, 4]}
    out = []
    for name, counts in plan.items():
        for frame, n_groups in enumerate(counts):
            for index in range(n_groups):
                for label, offset in enumerate((-1, 0, 1), start=1):
                    out.append({
                        CONDITION: name,
                        REPLICATE: index,
                        TIME_COL: frame,
                        LABEL: label,
                        METRIC: 10.0 * (1.0 + offset * 0.05),
                    })
    return pd.DataFrame(out)


def test_n_groups_rounds_the_median_it_does_not_truncate():
    """``[3, 4]`` is 3.5 groups, and the schema says so — so it reports 4.

    Truncating reported 3, understating the sample size the verdict rests on.
    ``min`` and truncation both give 3 for X, and ``max`` gives 4 for Y.
    """
    check = _make_check()
    check.analyze(_uneven_group_count_frame())
    verdicts = _verdicts(check)
    assert int(verdicts["X"][N_GROUPS]) == 4
    assert int(verdicts["Y"][N_GROUPS]) == 3


def test_n_frames_counted_is_the_series_length(
    recording: dict, analyzed: FittingWindow
):
    verdicts = _verdicts(analyzed)
    for condition in recording["runs"]["coverage"]["conditions"]:
        assert int(verdicts[condition["condition"]][N_FRAMES]) == len(
            condition["frames"]
        )


def test_a_condition_with_no_counted_frame_reports_zero_and_fails(
    pair_frame: pd.DataFrame,
):
    """Zero counted frames is not a seventh outcome — it is ``only 0 frames``."""
    # Two colonies per group is below ``min_replicates``, so nothing counts.
    frame = pair_frame.loc[pair_frame[LABEL] % 4 < 2]
    check = _make_check()
    check.analyze(frame)
    for row in _verdicts(check).values():
        assert int(row[N_FRAMES]) == 0
        assert row[OUTCOME] == str(WindowOutcome.TOO_FEW_FRAMES)
        assert row[REASON] == "only 0 frames"
        assert pd.isna(row[BEST_CV])
        assert pd.isna(row[N_GROUPS])
        assert row[check.status_col()] == "fail"


# --- the DuckDB table ---------------------------------------------------------------


def test_to_table_is_one_row_per_condition(pair_frame: pd.DataFrame):
    """The inherited member-level table would write one row per colony."""
    check = _make_check()
    check.analyze(pair_frame)
    table = check.to_table()
    assert len(table) == 2
    assert len(pair_frame) > 2
    assert list(table.columns)[0] == CONDITION
    assert LABEL not in table.columns
    for member in QUALITY_WINDOW:
        assert str(member) in table.columns
    for column in (check.metric_col(), check.status_col(), check.flag_col()):
        assert column in table.columns


def _qc_pipeline(**overrides) -> ImagePipeline:
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry

    params = {
        "on": METRIC,
        "groupby": [CONDITION],
        "replicate_by": [REPLICATE],
        "time_col": TIME_COL,
        **overrides,
    }
    return ImagePipeline(
        qc=[
            QcRecipeEntry(
                cls=FittingWindow,
                params=params,
                instance_id="qc-Window-0123abcd",
                enabled=True,
            )
        ]
    )


def test_the_qc_module_row_is_well_formed_and_the_table_is_condition_grain(
    tmp_path: Path, pair_frame: pd.DataFrame
):
    """Through ``run_qc``, which is what actually writes ``qc.duckdb``."""
    import duckdb

    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    modules = run_qc(pair_frame, _qc_pipeline(), tmp_path)
    assert [m.instance_id for m in modules] == ["qc-Window-0123abcd"]

    con = duckdb.connect(str(qc_duckdb_path(tmp_path)), read_only=True)
    try:
        row = con.execute(
            "SELECT class, name, groupby_cols, metric_col, status_col, flag_col, "
            "on_col, member_key_cols, supports_object_curation, higher_is_bad, "
            "extra_cols, warn_threshold, fail_threshold, table_name, time_col "
            "FROM qc_modules WHERE instance_id = ?",
            ["qc-Window-0123abcd"],
        ).fetchone()
        assert row is not None
        (
            cls_name, name, groupby_cols, metric_col, status_col, flag_col,
            on_col, member_key_cols, curation, higher_is_bad, extra_cols,
            warn, fail, table_name, time_col,
        ) = row
        # The catalog reads a check's time axis off ``time_label``; this check
        # spells the parameter ``time_col``, so without the alias a
        # time-course check records a null time column.
        assert time_col == TIME_COL
        assert cls_name == "FittingWindow"
        assert name == "Window"
        assert json.loads(groupby_cols) == [CONDITION]
        assert metric_col == FittingWindow.metric_col()
        assert status_col == FittingWindow.status_col()
        assert flag_col == FittingWindow.flag_col()
        assert on_col == METRIC
        assert json.loads(member_key_cols) == []
        assert bool(curation) is False
        assert bool(higher_is_bad) is True
        assert set(json.loads(extra_cols)) == {str(m) for m in QUALITY_WINDOW}
        assert (float(warn), float(fail)) == (0.10, 0.30)

        count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        assert count == 2
    finally:
        con.close()


def test_an_infinite_metric_survives_duckdb(tmp_path: Path, pair_frame: pd.DataFrame):
    """The ``+inf`` of a windowless condition has to reach the consumers.

    ``min_series_frames`` makes every condition windowless without depending on
    any noise statistic, so both conditions carry ``+inf``.
    """
    import duckdb

    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    run_qc(pair_frame, _qc_pipeline(min_series_frames=10_000), tmp_path)
    con = duckdb.connect(str(qc_duckdb_path(tmp_path)), read_only=True)
    try:
        (table_name,) = con.execute(
            "SELECT table_name FROM qc_modules WHERE instance_id = ?",
            ["qc-Window-0123abcd"],
        ).fetchone()
        metrics = [
            value
            for (value,) in con.execute(
                f'SELECT "{FittingWindow.metric_col()}" FROM "{table_name}"'
            ).fetchall()
        ]
    finally:
        con.close()
    assert len(metrics) == 2
    assert all(math.isinf(value) and value > 0 for value in metrics), metrics


# --- the check agrees with the layer beneath it -------------------------------------


def _rule_answer(frame: pd.DataFrame, check: FittingWindow) -> dict:
    """Drive the rule functions directly: the same three calls, by hand.

    ``mean_cv_series`` per condition, ``abrupt_bar`` once over all of them,
    ``rule_d`` per condition against that bar — written out here rather than
    delegated to the check, so the check's own plumbing (which condition's
    state reaches which group) is what is being compared.
    """
    series = {}
    for key, group in frame.groupby(list(check.groupby), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        series[key] = mean_cv_series(
            group,
            on=check.on,
            replicate_by=list(check.replicate_by),
            time_col=check.time_col,
        )
    bar = abrupt_bar([(s.frames, s.mean_cv) for s in series.values()])
    return {
        key: (rule_d(s.frames, s.mean_cv, bar), bar, s) for key, s in series.items()
    }


def test_the_table_is_what_the_rule_functions_return(
    rows: pd.DataFrame, analyzed: FittingWindow
):
    """Rows in, one pooled bar, the rule's answer out — for that condition.

    The golden test pins the whole chain against the reference. This one pins
    the plumbing underneath it: each condition's window, outcome, sentence,
    bar, series minimum and counted-frame count are the rule's for **that**
    condition and not a neighbour's.
    """
    expected = _rule_answer(rows, analyzed)
    table = analyzed.to_table()
    assert len(table) == len(expected)
    for _, row in table.iterrows():
        result, bar, series = expected[(row[CONDITION],)]
        assert _window_of(row) == result.window, row[CONDITION]
        assert row[OUTCOME] == result.outcome.value, row[CONDITION]
        assert row[REASON] == result.detail, row[CONDITION]
        assert row[BAR] == bar, row[CONDITION]
        assert row[BEST_CV] == pytest.approx(float(series.mean_cv.min()))
        assert int(row[N_FRAMES]) == len(series.frames), row[CONDITION]


def test_a_failed_analyze_leaves_the_previous_runs_state(
    monkeypatch, pair_frame: pd.DataFrame, rows: pd.DataFrame
):
    """Half a run is not a run.

    If ``QualityCheck.analyze`` raises after the stash is set, the instance
    would otherwise hold run N's verdicts beside run N-1's frame, and the
    figure would draw one while the table described the other.
    """
    from phenotypic.analysis.abc_._quality_check import QualityCheck

    check = _make_check()
    check.analyze(pair_frame)
    before_bar = check._bar
    before_keys = [page.key for page in check.inspect().pages]

    def boom(self, data):
        raise RuntimeError("the metric coercion failed")

    monkeypatch.setattr(QualityCheck, "analyze", boom)
    with pytest.raises(RuntimeError, match="metric coercion"):
        check.analyze(_only(rows, ["Condition B"]))

    assert check._bar == before_bar
    assert [page.key for page in check.inspect().pages] == before_keys


# --- the figure ---------------------------------------------------------------------


def test_inspect_returns_one_page_per_condition(analyzed: FittingWindow):
    output = analyzed.inspect()
    keys = [page.key for page in output.pages]
    assert len(keys) == 11
    assert keys == sorted(keys)
    assert len(set(keys)) == 11
    assert set(output.pages[0].metadata) == {CONDITION}


def test_inspect_names_the_outcome_only_when_there_is_no_window(
    analyzed: FittingWindow,
):
    verdicts = _verdicts(analyzed)
    for page in analyzed.inspect().pages:
        row = verdicts[page.metadata[CONDITION]]
        title = str(page.figure.layout.title.text)
        if row[OUTCOME] == str(WindowOutcome.FOUND):
            assert "no window" not in title
        else:
            assert "no window" in title
            assert row[OUTCOME] in title


def test_inspect_before_analyze_raises():
    with pytest.raises(RuntimeError, match="analyze"):
        _make_check().inspect()


def test_an_all_numeric_frame_keys_its_pages_on_the_integer_it_was_given():
    """``group.iloc[0]`` would read a condition value ``7`` back as ``7.0``.

    On a frame with no non-numeric column, a row-wise read upcasts every value
    to float, and the page key would then depend on which other columns the
    frame carries. The readable half of each key is the integer the user wrote.
    """
    frame = _uneven_group_count_frame()
    frame[CONDITION] = frame[CONDITION].map({"X": 7, "Y": 10})
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns)
    check = _make_check()
    check.analyze(frame)
    keys = {page.key.rsplit("-", 1)[0] for page in check.inspect().pages}
    assert keys == {"7", "10"}


# --- construction -------------------------------------------------------------------


def test_an_empty_replicate_by_is_refused():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at least one column"):
        _make_check(replicate_by=[])


def test_the_thresholds_default_to_the_figures_cv_bands():
    check = _make_check()
    assert (check.warn_threshold, check.fail_threshold) == (0.10, 0.30)


def test_the_tuning_constants_default_to_the_reference_values():
    from phenotypic.analysis.qc import _fitting_window_rule as rule

    check = _make_check()
    assert check.smooth == rule.SMOOTH
    assert check.abrupt_k == rule.ABRUPT_K
    assert check.persist == rule.PERSIST
    assert check.rough_roll == rule.ROUGH_ROLL
    assert check.rough_k == rule.ROUGH_K
    assert check.rough_lag == rule.ROUGH_LAG
    assert check.rough_span == rule.ROUGH_SPAN
    assert check.near_min_tolerance == rule.NEAR_MIN_TOLERANCE
    assert check.min_bar_over_noise == rule.MIN_BAR_OVER_NOISE
    assert check.edge_frames == rule.EDGE_FRAMES
    assert check.min_series_frames == rule.MIN_SERIES_FRAMES
    assert check.min_replicates == rule.MIN_REPLICATES


def test_a_tuning_override_reaches_the_rule(pair_frame: pd.DataFrame):
    """The constants are exposed as parameters, not baked in."""
    default = _make_check()
    default.analyze(pair_frame)
    strict = _make_check(min_series_frames=10_000)
    strict.analyze(pair_frame)

    strict_outcomes = {row[OUTCOME] for row in _verdicts(strict).values()}
    assert strict_outcomes == {str(WindowOutcome.TOO_FEW_FRAMES)}
    assert {row[OUTCOME] for row in _verdicts(default).values()} != strict_outcomes


def test_a_lowered_abrupt_k_brings_too_noisy_back_for_a_condition_alone(
    rows: pd.DataFrame,
):
    """The single-condition guarantee holds at the defaults, not at any setting."""
    check = _make_check(abrupt_k=0.0)
    check.analyze(_only(rows, ["Condition D"]))
    (row,) = _verdicts(check).values()
    assert row[OUTCOME] == str(WindowOutcome.TOO_NOISY)


def test_a_missing_column_is_named(pair_frame: pd.DataFrame):
    check = _make_check()
    with pytest.raises(KeyError, match=TIME_COL):
        check.analyze(pair_frame.drop(columns=[TIME_COL]))


# --- the docs generator -------------------------------------------------------------


def test_the_check_is_a_registered_producer_of_its_schema():
    """Review focus: without a producer the Measurements reference fails the build."""
    from phenotypic.util import measurement_producers

    producer = next(
        p for p in measurement_producers() if p.output_key == "FittingWindow"
    )
    assert QUALITY_WINDOW in producer.primary_infos
    assert FittingWindow.output_header(QUALITY_WINDOW.START) == START


# --- the source guard ---------------------------------------------------------------

#: Column families this check must never spell for itself: they are all user
#: parameters. ``QC_`` is excluded on purpose — those are the check's *own*
#: output columns, pinned to the schema below.
_COLUMN_LITERAL = re.compile(
    r"""["'](?:Metadata|Grid|Shape|Size|Intensity|Color|Texture|Object|Computed"""
    r"""|Bbox|Neighbor|Orientation|Symmetric|Radial|Curation)_[A-Za-z]"""
)

_GUARDED_MODULES = ("_fitting_window.py", "_expected_positions.py")

#: Docstring example lines are user code — what a reader types — not columns
#: the check requires, so they are exempt.
_DOCTEST_LINE = re.compile(r"^\s*(>>>|\.\.\.)")


def test_no_measurement_or_metadata_column_is_spelled_in_the_source():
    """``on``, ``groupby``, ``replicate_by`` and ``time_col`` are user parameters.

    A literal outside a docstring example is a column this check silently
    requires.
    """
    source_dir = (
        Path(__file__).resolve().parents[3] / "src" / "phenotypic" / "analysis" / "qc"
    )
    offenders = []
    for name in _GUARDED_MODULES:
        path = source_dir / name
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _DOCTEST_LINE.match(line):
                continue
            if _COLUMN_LITERAL.search(line):
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert not offenders, (
        "hard-coded column literals (ask the schema, or take a parameter):\n"
        + "\n".join(offenders)
    )


def test_the_emitted_headers_come_from_the_schema():
    """What the doctest's literals are pinned to."""
    assert OUTCOME == "QC_Window_Outcome"
    assert START == "QC_Window_Start"
    assert END == "QC_Window_End"
    assert REASON == "QC_Window_Reason"
    assert BEST_CV == "QC_Window_BestCV"
    assert BAR == "QC_Window_Bar"
    assert N_GROUPS == "QC_Window_NGroups"
    assert N_FRAMES == "QC_Window_NFramesCounted"
    assert FittingWindow.metric_col() == "QC_Window_Metric"
    assert FittingWindow.flag_col() == "QC_Window_Flag"
    assert FittingWindow.status_col() == "QC_Window_Status"


def test_every_outcome_value_is_explained_in_the_enum():
    """The measurement-docstring split: per-value detail lives on the enum."""
    desc = QUALITY_WINDOW.OUTCOME.desc
    for member in WindowOutcome:
        assert member.value in desc, member.value


def test_the_outcome_members_carry_no_machine_written_biology():
    for member in QUALITY_WINDOW:
        assert member.bio_desc == ""
        assert member.image is None


# --- doctests -----------------------------------------------------------------------


def test_the_module_doctests_run():
    """The class example: AddFrameIndex, FittingWindow, a filter, a fit on hours."""
    import doctest

    from phenotypic.analysis.qc import _fitting_window

    results = doctest.testmod(_fitting_window, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0


def test_the_synthetic_series_doctests_run():
    import doctest

    from phenotypic.data import _synthetic_window_series

    results = doctest.testmod(_synthetic_window_series, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
