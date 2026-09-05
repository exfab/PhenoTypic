"""The Scatter figure builder is pure: no Dash, no I/O."""

from __future__ import annotations

import numpy as np
import polars as pl

from phenotypic.gui._design import OI_GREY
from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
from phenotypic.gui.results_viewer._scatter_tab._figure import (
    CUSTOMDATA_COL,
    REMOVED_COL,
    REMOVED_LABEL,
    build_scatter_figure,
)
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec


def _frame(n: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "x": rng.integers(0, 8, n).tolist(),
            "y": rng.normal(10, 2, n).tolist(),
            "r": ["0" if i % 2 else "1" for i in range(n)],
            "c": ["0" if i % 3 else "1" for i in range(n)],
            "hue": ["a" if i % 2 else "b" for i in range(n)],
            CUSTOMDATA_COL: list(range(n)),
        }
    )


def _spec() -> FigureSpec:
    return FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c", hue_col="hue")


def test_the_screen_figure_uses_webgl_traces() -> None:
    """SVG go.Scatter cannot render at this project's point counts."""
    df, spec = _frame(), _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))
    assert fig.data, "no traces were added"
    assert all(t.type == "scattergl" for t in fig.data)


def test_the_export_figure_uses_svg_traces() -> None:
    """kaleido renders Scattergl as blank axes -- 624 non-white px against
    46,886 for SVG, with no warning and exit code 0. The export pass must
    substitute the trace type or every PDF is empty."""
    df, spec = _frame(), _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec), for_export=True)
    assert fig.data
    assert all(t.type == "scatter" for t in fig.data)


def test_a_series_absent_from_the_first_cell_still_reaches_the_legend() -> None:
    """``showlegend=first_cell`` drops any series missing from cell 1.

    On a sparse frame -- 23 strains over 36 images in the fixture -- that
    is the common case, not a corner: a hue that happens not to appear in
    the top-left facet vanishes from the legend while its points are drawn.
    Track which series have been given a legend entry instead.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "r": ["0", "0", "1", "1"],
            "c": ["0", "0", "0", "0"],
            "hue": ["a", "a", "b", "b"],  # 'b' never appears in cell (0,0)
            CUSTOMDATA_COL: [0, 1, 2, 3],
        }
    )
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c", hue_col="hue")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    drawn = {t.name for t in fig.data}
    in_legend = {t.name for t in fig.data if t.showlegend}
    assert drawn == in_legend, f"drawn {drawn} but only {in_legend} in the legend"


def test_each_series_is_legended_exactly_once() -> None:
    """The other half of the legend contract.

    Tracking "have I legended this series" must not degrade into "legend
    everything": a hue drawn in six facets would then get six identical
    legend entries. Only the pair of assertions pins the behaviour.
    """
    df, spec = _frame(), _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    legended = [t.name for t in fig.data if t.showlegend]
    assert sorted(legended) == sorted(set(legended)), (
        f"a series was legended more than once: {legended}"
    )
    assert len(fig.data) > len(legended), (
        "this frame must draw a series in more than one facet, or the test "
        "cannot detect duplicate legend entries"
    )


def test_one_hue_keeps_one_colour_across_every_facet() -> None:
    """A hue must be the same colour in every panel.

    The frame is built to defeat the two ways this goes wrong, because a
    frame that does not is a test that passes against both:

    * **A hue appears in more than one facet row.** Otherwise a colour
      derived from the row index is constant per series by accident.
    * **Cell (0,0) is missing hue 'a'.** Otherwise a per-cell counter --
      enumerating the hues *present in this cell* rather than the figure's
      global hue order -- assigns the same index everywhere and agrees.

    Both mutations shift a hue's colour between panels while the legend
    keeps showing one swatch, so the legend describes a figure other than
    the one on screen and nothing raises.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "r": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "c": ["0", "0", "1", "1", "0", "0", "1", "1"],
            # (r=0, c=0) carries 'b' only -- 'a' is absent from it.
            "hue": ["b", "b", "a", "b", "a", "b", "a", "b"],
            CUSTOMDATA_COL: [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c", hue_col="hue")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    by_series: dict[str, set[str]] = {}
    for trace in fig.data:
        by_series.setdefault(trace.name, set()).add(trace.marker.color)

    assert len(by_series) == 2, f"expected two hue series, got {by_series}"
    assert all(len(colours) == 1 for colours in by_series.values()), (
        f"a hue changed colour between facets: {by_series}"
    )
    assert len({next(iter(c)) for c in by_series.values()}) == 2, (
        "two hues must not share a colour"
    )
    # The arrangement the assertions above depend on. Without this, a
    # frame edit could quietly make the test vacuous again.
    drawn_per_series = [t.name for t in fig.data]
    assert drawn_per_series.count("hue=a") >= 2, (
        "hue 'a' must span more than one facet or the row-index mutation "
        "is invisible"
    )


def test_every_point_carries_the_index_of_its_own_row() -> None:
    """Correspondence, not membership. This is the assertion that let B1 pass.

    The original asserted ``seen <= set(range(df.height))`` -- that every
    index falls in a plausible range. A filtered-frame index satisfies that
    perfectly while pointing at the wrong row, which is exactly how the
    filtered-vs-master defect survived a careful read. Assert that the index
    on a point resolves to the row whose x and y that point actually carries.
    """
    df, spec = _frame(), _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    n_points = 0
    for trace in fig.data:
        for x, y, cd in zip(trace.x, trace.y, trace.customdata):
            row = df.row(int(cd[0]), named=True)
            assert row["x"] == x and row["y"] == y, (
                f"index {int(cd[0])} points at ({row['x']}, {row['y']}) but "
                f"the marker is at ({x}, {y})"
            )
            n_points += 1
    assert n_points == df.height, "every row must be drawn exactly once"


def test_the_index_column_is_carried_through_never_recomputed() -> None:
    """The builder must emit the index it was GIVEN, not the row's position.

    The test above cannot see the difference, because there the index
    column happens to equal the positional index. That is precisely the
    coincidence B1 hid behind: a builder that called ``with_row_index``
    itself would satisfy it while pointing every click at the wrong colony.
    Here the frame is a filtered, re-sorted slice whose indices point into
    a larger master frame, so passing through and recomputing disagree.
    """
    master = _frame(40)
    # A filtered, re-ordered subset -- what the tab actually hands the builder.
    section = master.filter(pl.col("r") == "1").sort("y", descending=True)
    assert section.height not in (0, master.height), "the slice must be proper"

    spec = FigureSpec(x_col="x", y_col="y", col_col="c", hue_col="hue")
    fig = build_scatter_figure(section, spec, plan_facets(section, spec))

    seen = [int(cd[0]) for t in fig.data for cd in t.customdata]
    assert sorted(seen) == sorted(section[CUSTOMDATA_COL].to_list()), (
        "the builder recomputed the index instead of carrying it through"
    )
    for trace in fig.data:
        for x, y, cd in zip(trace.x, trace.y, trace.customdata):
            row = master.row(int(cd[0]), named=True)
            assert row["x"] == x and row["y"] == y, (
                f"index {int(cd[0])} resolves against master to "
                f"({row['x']}, {row['y']}) but the marker is at ({x}, {y})"
            )


def test_an_empty_facet_still_occupies_its_cell() -> None:
    """A missing (row, col) combination must not collapse the geometry.

    Four cells are planned, three carry data. The grid must still be 2x2 --
    a facet that silently disappears shifts every panel after it and
    misreads as data rather than as absence.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [1.0, 2.0, 3.0],
            "r": ["0", "0", "1"],
            "c": ["0", "1", "0"],  # (r=1, c=1) is absent
            CUSTOMDATA_COL: [0, 1, 2],
        }
    )
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c")
    plan = plan_facets(df, spec)
    assert len(plan.rows) == 2 and len(plan.cols) == 2

    fig = build_scatter_figure(df, spec, plan)
    # Probe the geometry through axis DOMAINS rather than axis count. How
    # many axis objects `make_subplots` creates depends on how plotly
    # implements sharing; how many distinct strips of the paper they cover
    # is the grid itself. A 2x2 is two column bands and two row bands; a
    # collapsed 1x3 would read as three and one.
    lay = fig.layout.to_plotly_json()
    x_bands = {
        tuple(v["domain"])
        for k, v in lay.items()
        if k.startswith("xaxis") and isinstance(v, dict) and v.get("domain")
    }
    y_bands = {
        tuple(v["domain"])
        for k, v in lay.items()
        if k.startswith("yaxis") and isinstance(v, dict) and v.get("domain")
    }
    assert (len(x_bands), len(y_bands)) == (2, 2), f"{x_bands} x {y_bands}"
    assert len(fig.data) == 3, "one trace per non-empty cell"

    # The three traces must occupy three DISTINCT cells: a builder that
    # dropped the empty cell by shifting the others would still add three
    # traces, but two of them would land in the same panel.
    assert len({(t.xaxis, t.yaxis) for t in fig.data}) == 3


def test_shared_axes_give_every_facet_one_range() -> None:
    df, spec_ = _frame(), _spec()
    fig = build_scatter_figure(df, spec_, plan_facets(df, spec_))
    ranges = {
        tuple(v["range"])
        for k, v in fig.layout.to_plotly_json().items()
        if k.startswith("yaxis") and isinstance(v, dict) and v.get("range")
    }
    # `<= 1` would pass on ZERO ranges too, which is exactly what
    # share_axes=False produces -- so the assertion must be `== 1` or it
    # cannot detect the regression it is named for.
    assert len(ranges) == 1, f"expected one shared y range, got {ranges}"


def test_unshared_axes_do_not_set_one_range() -> None:
    """The negative half. Without this, the test above proves nothing."""
    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c", share_axes=False)
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))
    ranges = {
        tuple(v["range"])
        for k, v in fig.layout.to_plotly_json().items()
        if k.startswith("yaxis") and isinstance(v, dict) and v.get("range")
    }
    assert len(ranges) != 1


def test_an_empty_string_facet_value_selects_only_its_own_rows() -> None:
    """`""` is a real value, not a "this axis is unused" sentinel.

    ``plan_facets`` returns ``[""]`` for a missing or all-null column, so a
    builder that reads ``value != ""`` as "filter this axis" will, on a
    column that genuinely contains an empty string, draw EVERY row into
    that panel. The symptom is duplicated points, which reads as a data
    problem rather than a filtering one.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1.0, 2.0, 3.0, 4.0],
            "r": ["", "", "z", "z"],
            CUSTOMDATA_COL: [0, 1, 2, 3],
        }
    )
    spec = FigureSpec(x_col="x", y_col="y", row_col="r")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    drawn = [int(cd[0]) for t in fig.data for cd in t.customdata]
    assert sorted(drawn) == [0, 1, 2, 3], f"rows drawn {sorted(drawn)}"


def test_an_all_null_facet_column_draws_one_panel_with_every_row() -> None:
    """The other side of the sentinel question.

    An all-null facet column yields no values at all, and the grid must
    degrade to a single panel carrying every row -- not to a panel that
    filters on null and draws nothing.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [1.0, 2.0, 3.0],
            "r": [None, None, None],
            CUSTOMDATA_COL: [0, 1, 2],
        },
        schema_overrides={"r": pl.String},
    )
    spec = FigureSpec(x_col="x", y_col="y", row_col="r")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    drawn = [int(cd[0]) for t in fig.data for cd in t.customdata]
    assert sorted(drawn) == [0, 1, 2]


def _curated_frame() -> pl.DataFrame:
    """Two facet rows, two hues, and one removed colony in each row.

    Removed rows span both facets so the single-legend-entry property is
    testable, and hue 'c' is carried ONLY by removed rows so a leak from
    the curation series into the hue channel is visible.
    """
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "r": ["0", "0", "0", "1", "1", "1"],
            "hue": ["a", "b", "c", "a", "b", "c"],
            REMOVED_COL: [False, False, True, False, False, True],
            CUSTOMDATA_COL: [0, 1, 2, 3, 4, 5],
        }
    )


def test_removed_colonies_draw_as_a_grey_x_series() -> None:
    """The curation toggle's on state: shown, but visibly set apart."""
    df = _curated_frame()
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", hue_col="hue")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    removed = [t for t in fig.data if t.name == REMOVED_LABEL]
    assert removed, f"no curation series; drew {[t.name for t in fig.data]}"
    assert all(t.marker.color == OI_GREY for t in removed)
    assert all(t.marker.symbol == "x" for t in removed)

    drawn = {int(cd[0]) for t in removed for cd in t.customdata}
    assert drawn == {2, 5}, "the grey series must carry exactly the removed rows"


def test_removed_colonies_vanish_when_the_toggle_is_off() -> None:
    """The off state drops the rows, rather than drawing them as normal."""
    df = _curated_frame()
    spec = FigureSpec(
        x_col="x", y_col="y", row_col="r", hue_col="hue", show_removed=False
    )
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    assert not [t for t in fig.data if t.name == REMOVED_LABEL]
    drawn = {int(cd[0]) for t in fig.data for cd in t.customdata}
    assert drawn == {0, 1, 3, 4}, (
        f"removed rows must not be drawn at all, got {sorted(drawn)}"
    )


def test_the_curation_series_stays_out_of_the_hue_channel() -> None:
    """A removed colony belongs to no hue, and opens no hue series.

    Hue 'c' is carried only by removed rows. If the split happened after
    the hue channel were derived, 'c' would open a legend entry for a
    series that draws nothing -- or worse, draw removed colonies in a
    colour that says they are live data.
    """
    df = _curated_frame()
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", hue_col="hue")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    names = {t.name for t in fig.data}
    assert "hue=c" not in names, f"a removed-only hue opened a series: {names}"
    assert names == {"hue=a", "hue=b", REMOVED_LABEL}

    for trace in fig.data:
        if trace.name == REMOVED_LABEL:
            continue
        drawn = {int(cd[0]) for cd in trace.customdata}
        assert not (drawn & {2, 5}), (
            f"trace {trace.name!r} drew a removed colony: {sorted(drawn)}"
        )


def test_the_curation_series_takes_one_legend_entry_for_the_figure() -> None:
    """Removed colonies appear in both facets; the legend says so once."""
    df = _curated_frame()
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", hue_col="hue")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    removed = [t for t in fig.data if t.name == REMOVED_LABEL]
    assert len(removed) == 2, "one curation trace per facet that has one"
    assert sum(bool(t.showlegend) for t in removed) == 1


def test_a_frame_without_the_curation_column_is_unchanged() -> None:
    """Absence means "nothing is removed", for every caller that has none.

    Both toggle states must agree with the no-column frame, or wiring
    curation in would silently change what every existing caller renders.
    """
    df = _curated_frame().drop(REMOVED_COL)
    plain = FigureSpec(x_col="x", y_col="y", row_col="r", hue_col="hue")
    hidden = FigureSpec(
        x_col="x", y_col="y", row_col="r", hue_col="hue", show_removed=False
    )

    def _drawn(spec: FigureSpec) -> set[int]:
        fig = build_scatter_figure(df, spec, plan_facets(df, spec))
        return {int(cd[0]) for t in fig.data for cd in t.customdata}

    assert _drawn(plain) == {0, 1, 2, 3, 4, 5}
    assert _drawn(hidden) == {0, 1, 2, 3, 4, 5}


def test_hiding_removed_colonies_narrows_the_shared_axis_range() -> None:
    """The shared range covers what is drawn, not what was filtered out.

    An outlier that is hidden must not go on reserving the space it would
    have occupied -- every panel would keep an empty margin sized for a
    point the user asked not to see.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [1.0, 2.0, 500.0],
            REMOVED_COL: [False, False, True],
            CUSTOMDATA_COL: [0, 1, 2],
        }
    )

    def _y_range(show: bool) -> tuple[float, float]:
        spec = FigureSpec(x_col="x", y_col="y", show_removed=show)
        fig = build_scatter_figure(df, spec, plan_facets(df, spec))
        return tuple(fig.layout.to_plotly_json()["yaxis"]["range"])

    assert _y_range(True)[1] > 100.0, "the outlier is drawn, so it is covered"
    assert _y_range(False)[1] < 100.0, (
        "the outlier is hidden, so the range must not still reach it"
    )


def test_a_frame_with_no_rows_returns_a_figure_rather_than_raising() -> None:
    """A filter can empty the frame; the tab must render, not 500."""
    df = _frame(0)
    spec = _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))
    assert fig.data == ()
