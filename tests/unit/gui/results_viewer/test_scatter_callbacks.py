"""The Scatter tab's Dash wiring: what it subscribes to, and in what order.

Four of these tests exist because the defect they pin produces a
plausible answer rather than an error, and so cannot be found by reading:

* :func:`test_the_click_index_is_stamped_before_the_filter` -- an index
  built on the filtered frame resolves to a real but *wrong* colony.
* :func:`test_a_stale_fingerprint_is_refused_at_the_call_site` -- reading
  the live fingerprint on both sides of the comparison leaves a guard
  that always passes.
* :func:`test_the_grey_removed_series_is_drawn_from_the_curation_store`
  -- a join that quietly produces an all-False column renders exactly as
  "nothing is removed", which is how the toggle read before it had a
  producer at all.
* :func:`test_every_id_the_callbacks_bind_is_mounted` -- an Input bound
  to an unmounted id is the silent half of the id contract; a renamed
  symbol raises at import and a renamed value stays in sync.

Every header here is asked of ``phenotypic.schema``. Spelling one by hand
is this branch's most-repeated defect: the string reads correctly,
matches no measurer, and the assertion then passes for an unrelated
reason.
"""

from __future__ import annotations

from pathlib import Path

import dash
import polars as pl
import pytest

from phenotypic.gui._config import MOUNT_HOME, SCATTER_CROPS_URL_SEGMENT
from phenotypic.gui.results_viewer import _ids as rv_ids
from phenotypic.gui.results_viewer._filter_state import (
    METHOD_IS_ANY_OF,
    FilterSpec,
)
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_DATASET,
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._scatter_tab import _ids as ids
from phenotypic.gui.results_viewer._scatter_tab._callbacks import (
    build_render_state,
    clamp_section_index,
    click_index,
    crop_url,
    export_payload,
    inspector_payload,
    join_curation_flags,
    legend_layout,
    paged_section_index,
    prepare_frame,
    register_callbacks,
    resolve_inspector_click,
    section_values,
    store_int,
)
from phenotypic.gui.results_viewer._scatter_tab._figure import (
    CUSTOMDATA_COL,
    REMOVED_COL,
    REMOVED_LABEL,
)
from phenotypic.gui.results_viewer._scatter_tab._inspector import resolve_click
from phenotypic.gui.results_viewer._scatter_tab._layout import (
    LEGEND_CORNER_DEFAULT,
    build_scatter_tab_body,
)
from phenotypic.gui.results_viewer._scatter_tab._spec import CURATION_PHANTOM_COL
from phenotypic.schema import CULTURE, SIZE
from tests._dash_layout import mounted_string_ids, walk_components
from tests._output_layout import (
    write_complete_manifest,
    write_master,
    write_measurements_mirror,
)

_AREA = str(SIZE.AREA)
_TIME = str(CULTURE.TIME)
_STRAIN = "Metadata_Strain"

#: Ids the layout mounts that no Python callback binds, each for a
#: reason. Without naming them the "declared implies bound" direction of
#: the id contract cannot be asserted at all; with a blanket exemption it
#: would assert nothing.
_SELF_WIRED = {
    # dbc wires the popover to its target itself (``trigger="legacy"``).
    ids.SCATTER_CONFIG_TOGGLE,
    ids.SCATTER_CONFIG_POPOVER,
    # Driven by results_viewer.js section H off its data attributes; the
    # Python side binds the store it writes, not the handle.
    ids.SCATTER_INSPECTOR_SPLITTER,
}


def _master() -> pl.DataFrame:
    """Six colonies over three images plus one phantom.

    Six rather than three so a filter can drop rows without leaving the
    carried index equal to its row position by coincidence -- the
    coincidence a filtered-frame index hides behind.

    The phantom is the seventh row: metadata for a strain that was never
    detected, with a null ``Object_Label``. The mirror really carries
    these (121 of the verification fixture's 844 rows), and they are what
    ``plottable`` exists to remove.
    """
    return pl.DataFrame(
        {
            KEY_DATASET: ["d1"] * 7,
            KEY_IMAGE_FILE: ["a", "a", "b", "b", "c", "c", "d"],
            KEY_OBJECT_LABEL: pl.Series(
                [1, 2, 1, 2, 1, 2, None], dtype=pl.Int64
            ),
            _STRAIN: [
                "BY4741",
                "BY4741",
                "S288C",
                "S288C",
                "W303",
                "W303",
                "W303",
            ],
            _TIME: pl.Series(
                [0.0, 6.0, 0.0, 6.0, 0.0, 6.0, None], dtype=pl.Float64
            ),
            _AREA: pl.Series(
                [10.0, 20.0, 11.0, 21.0, 12.0, 22.0, None], dtype=pl.Float64
            ),
            CURATION_PHANTOM_COL: pl.Series(
                [False] * 6 + [True], dtype=pl.Boolean
            ),
        }
    )


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """A discoverable run over :func:`_master`."""
    master = _master()
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    (tmp_path / "results" / "d1" / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        (overlays / f"{stem}.png").touch()
    write_complete_manifest(tmp_path, total_images=4)
    return OutputRoot.discover(
        tmp_path, cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache"
    )


@pytest.fixture()
def dash_app_and_root(output_root: OutputRoot) -> tuple[dash.Dash, OutputRoot]:
    """A bare app the Scatter callbacks can be registered onto.

    Bare rather than a full ``create_app``: registering only this tab's
    callbacks is what lets the id-contract tests below attribute every
    bound id to Scatter rather than to one of the six other surfaces.
    """
    return dash.Dash(__name__), output_root


def _render(output_root: OutputRoot, **overrides: object):
    """Render with the tab's own defaults, overridden per test."""
    kwargs: dict[str, object] = {
        "section_col": _STRAIN,
        "row_col": None,
        "col_col": None,
        "x_col": _TIME,
        "y_col": _AREA,
        "hue_col": None,
        "shape_col": None,
        "show_removed": True,
        "section_index": 0,
        "filter_payload": None,
        "removed_payload": None,
        "legend_payload": None,
    }
    kwargs.update(overrides)
    return build_render_state(output_root, **kwargs)  # type: ignore[arg-type]


def _carried(figure) -> list[int]:
    """Every point's carried master index, across every trace."""
    return [
        int(entry[0])
        for trace in figure.data
        for entry in (trace.customdata or [])
    ]


def _bound_ids(app: dash.Dash) -> set[str]:
    """Every string component id the app's callbacks reference.

    Dash 4.1 stores dependencies as plain dicts keyed ``"id"`` /
    ``"property"``, not as objects -- ``dep.component_id`` raises. The
    repo's working example is ``test_filter_panel.py:216-219``.
    """
    bound: set[str] = set()
    for entry in app.callback_map.values():
        for spec in list(entry["inputs"]) + list(entry.get("state") or []):
            component_id = spec["id"] if isinstance(spec, dict) else spec
            if isinstance(component_id, str):
                bound.add(component_id)
    # Outputs are not stored as a dependency list; they are encoded in the
    # callback_map KEY, in the format ``tests/_dash_layout.py`` decodes.
    for key in app.callback_map:
        for segment in key.strip(".").split("..."):
            segment = segment.strip(".").split("@", 1)[0]
            if "." not in segment or segment.startswith("{"):
                continue
            bound.add(segment.rsplit(".", 1)[0])
    return bound


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------


def test_scatter_subscribes_to_the_shared_refresh_revision(
    dash_app_and_root,
) -> None:
    """One Refresh must move every surface together.

    A Scatter-local refresh button would let the tab disagree with Plate
    and Colony about which snapshot it is showing.
    """
    app, output_root = dash_app_and_root
    register_callbacks(app, output_root)

    inputs = {
        spec["id"]
        for entry in app.callback_map.values()
        for spec in entry["inputs"]
    }
    assert rv_ids.STORE_PLOT_REFRESH_REVISION in inputs
    assert rv_ids.STORE_FILTER_SPEC in inputs, (
        "the filter store must be an Input, not a State -- decision Q4 is "
        "that Scatter shares filters, so the figure must rebuild on a "
        "filter edit rather than going stale"
    )
    assert rv_ids.STORE_REMOVED_KEYS in inputs, (
        "the curation store must be an Input: it is the only source of "
        "the grey removed series, so as a State the toggle would show a "
        "curation write only once something else moved"
    )


# ---------------------------------------------------------------------------
# 13.0 -- the id contract, in both directions
# ---------------------------------------------------------------------------


def test_every_id_the_callbacks_bind_is_mounted(
    dash_app_and_root, output_root: OutputRoot
) -> None:
    """Binding to an id nothing mounts is the silent half of the contract.

    Renaming an id's *value* stays in sync, because the layout and the
    callbacks both reference the symbol; renaming the *symbol* raises
    ``AttributeError`` at import. Binding an Input to an unmounted id
    gives, depending on ``suppress_callback_exceptions``, either a
    registration error or a callback that simply never fires.
    """
    app, _ = dash_app_and_root
    register_callbacks(app, output_root)

    from phenotypic.gui.results_viewer._app import create_app

    mounted = mounted_string_ids(create_app(output_root))
    missing = _bound_ids(app) - mounted
    assert not missing, f"bound but never mounted: {sorted(missing)}"


def test_every_declared_scatter_id_is_bound_by_a_callback(
    dash_app_and_root, output_root: OutputRoot
) -> None:
    """The other direction: mounted chrome nothing reads.

    Task 12's ``test_every_declared_id_is_actually_mounted`` proves every
    declared id reaches the DOM. That is satisfied by a control no
    callback consults -- which is chrome that looks live and is not.
    """
    app, _ = dash_app_and_root
    register_callbacks(app, output_root)

    declared = {getattr(ids, name) for name in ids.__all__}
    unbound = declared - _bound_ids(app) - _SELF_WIRED
    assert not unbound, f"declared and mounted, but nothing reads: {sorted(unbound)}"


# ---------------------------------------------------------------------------
# The click index anchors to master_df
# ---------------------------------------------------------------------------


def test_the_click_index_is_stamped_before_the_filter(
    output_root: OutputRoot,
) -> None:
    """Index on ``master_df``, then filter -- never the other way round.

    A filter that genuinely drops rows is load-bearing here. Over an
    unfiltered frame the carried index equals the row position by
    coincidence, so the round trip passes against the very defect it
    names; the degeneracy assertion below fails loudly if the fixture
    ever drifts back into that shape.
    """
    keep = ["S288C", "W303"]
    payload = [
        {"column": _STRAIN, "method": METHOD_IS_ANY_OF, "values": keep}
    ]
    assert FilterSpec.from_store(payload).apply_to(output_root.master_df).height < (
        output_root.master_df.height
    ), "the filter must actually drop rows or this test proves nothing"

    figure, _, fingerprint = _render(
        output_root, section_col=None, filter_payload=payload
    )
    carried = _carried(figure)

    assert carried, "the filtered frame still has points to draw"
    assert carried != list(range(len(carried))), (
        "the carried indices coincide with row positions, so this test "
        "cannot distinguish a master-anchored index from a filtered one"
    )
    for index in carried:
        colony = resolve_click(
            output_root.master_df, index, fingerprint, fingerprint
        )
        assert colony is not None
        row = output_root.master_df.filter(
            (pl.col(KEY_IMAGE_FILE) == colony.stem)
            & (pl.col(KEY_OBJECT_LABEL) == colony.label)
        )
        assert row[_STRAIN][0] in keep, (
            f"index {index} resolved to a colony the filter excluded -- "
            "the index was stamped on the filtered frame"
        )


def test_the_fingerprint_a_figure_stores_is_the_bindings_own(
    output_root: OutputRoot,
) -> None:
    """The producer half of the staleness guard."""
    _, _, fingerprint = _render(output_root)

    assert fingerprint == output_root.consumed_state_fingerprint


def test_a_stale_fingerprint_is_refused_at_the_call_site(
    output_root: OutputRoot,
) -> None:
    """The stored fingerprint is compared, never re-read.

    This is the test the tautology cannot pass: reading
    ``consumed_state_fingerprint`` on both sides of the comparison makes
    the stale case resolve like any other, and the guard -- still
    present, still reading correctly -- stops nothing.
    """
    click = {"points": [{"customdata": [1]}]}

    live = resolve_inspector_click(
        output_root, click, output_root.consumed_state_fingerprint
    )
    stale = resolve_inspector_click(output_root, click, "a-previous-snapshot")

    assert live is not None, "a current fingerprint must resolve"
    assert stale is None, (
        "a figure drawn before the snapshot changed must be refused, not "
        "resolved against the frame that replaced it"
    )


def test_a_click_that_is_not_on_a_point_leaves_the_inspector_alone(
    output_root: OutputRoot,
) -> None:
    """An errant click on the axis must not discard what is on show."""
    assert click_index({"points": []}) is None
    assert click_index(None) is None

    is_open, title, colony, rows = inspector_payload(
        output_root, {"points": []}, output_root.consumed_state_fingerprint
    )

    assert is_open is dash.no_update
    assert title is dash.no_update
    assert colony is dash.no_update
    assert rows is dash.no_update


def test_a_stale_click_says_so_rather_than_opening_nothing(
    output_root: OutputRoot,
) -> None:
    """A refusal a user cannot see reads as a broken click."""
    is_open, title, colony, rows = inspector_payload(
        output_root, {"points": [{"customdata": [1]}]}, "a-previous-snapshot"
    )

    assert is_open is True
    assert "efresh" in str(title)
    assert colony is None
    assert rows == []


def test_a_resolved_click_names_its_colony_and_its_measurements(
    output_root: OutputRoot,
) -> None:
    """The whole point of the panel."""
    is_open, title, colony, rows = inspector_payload(
        output_root,
        {"points": [{"customdata": [1]}]},
        output_root.consumed_state_fingerprint,
    )

    assert is_open is True
    assert colony == {"dataset": "d1", "stem": "a", "label": 2}
    assert "label 2" in str(title)
    rendered = " ".join(
        str(getattr(node, "children", ""))
        for block in rows
        for node in walk_components(block)
    )
    assert _AREA in rendered


# ---------------------------------------------------------------------------
# 13.1 -- the curation join
# ---------------------------------------------------------------------------


def test_the_grey_removed_series_is_drawn_from_the_curation_store(
    output_root: OutputRoot,
) -> None:
    """Without the join the toggle renders nothing, as it did before it.

    A join that silently produces an all-False column is indistinguishable
    from "nothing is removed" -- which is exactly the failure this half of
    the wiring was added to end -- so the assertion is that the series
    exists AND carries the right colony.
    """
    removed = [[str(output_root.master_df[KEY_IMAGE_FILE][0]), 1]]

    figure, _, fingerprint = _render(
        output_root, section_col=None, removed_payload=removed
    )

    grey = [trace for trace in figure.data if trace.name == REMOVED_LABEL]
    assert grey, "a removed colony must draw as the grey curation series"
    carried = [int(entry[0]) for entry in grey[0].customdata]
    resolved = [
        resolve_click(output_root.master_df, i, fingerprint, fingerprint)
        for i in carried
    ]
    assert [(r.stem, r.label) for r in resolved if r] == [("a", 1)]


def test_removed_colonies_disappear_when_the_toggle_is_off(
    output_root: OutputRoot,
) -> None:
    """``show_removed=False`` drops the rows, it does not merely recolour."""
    removed = [["a", 1]]

    shown, _, _ = _render(
        output_root, section_col=None, removed_payload=removed
    )
    hidden, _, _ = _render(
        output_root,
        section_col=None,
        removed_payload=removed,
        show_removed=False,
    )

    assert len(_carried(shown)) == len(_carried(hidden)) + 1
    assert not [t for t in hidden.data if t.name == REMOVED_LABEL]


def test_a_repeated_curation_key_does_not_duplicate_its_colony(
    output_root: OutputRoot,
) -> None:
    """A left join over a key list is only safe if the list is unique.

    Dash stores are written by several surfaces; a key arriving twice
    would otherwise draw one colony as two points, silently doubling its
    weight in a figure people read as counts.
    """
    once, _, _ = _render(
        output_root, section_col=None, removed_payload=[["a", 1]]
    )
    twice, _, _ = _render(
        output_root, section_col=None, removed_payload=[["a", 1], ["a", 1]]
    )

    assert len(_carried(twice)) == len(_carried(once))


def test_a_phantom_is_dropped_by_the_phantom_filter_not_by_luck(
    output_root: OutputRoot,
) -> None:
    """A metadata-only row has no colony, no coordinates and no crop.

    The coordinate columns here are the *grouping* column, deliberately.
    A phantom's measurements are null, so with a measurement on either
    axis the null-coordinate drop removes it too and this test cannot
    tell which mechanism did it -- it would pass with ``plottable`` gone.
    ``Metadata_Strain`` is populated on the phantom, so only the phantom
    filter can account for the missing row.
    """
    frame, dropped = prepare_frame(
        output_root,
        x_col=_STRAIN,
        y_col=_STRAIN,
        filter_payload=None,
        removed_payload=None,
    )

    assert dropped == 0, "no row lacks a value on this axis"
    assert frame.height == 6, "the seventh row is the phantom"


def _labelled_frame(dtype) -> pl.DataFrame:
    """Three colonies in one image, labels 1/2/3 stored as ``dtype``.

    Built from ``String`` because every target dtype accepts it. An
    ``Int64`` source does not cast to ``Categorical`` at all, so a
    fixture built that way would error out of the case it exists to
    describe.
    """
    return pl.DataFrame(
        {
            KEY_IMAGE_FILE: ["a", "a", "a"],
            KEY_OBJECT_LABEL: pl.Series(
                ["1", "2", "3"], dtype=pl.String
            ).cast(dtype),
        }
    )


@pytest.mark.parametrize(
    "dtype",
    [pl.Int64, pl.Int32, pl.Float64, pl.String],
    ids=["Int64", "Int32", "Float64", "String"],
)
def test_a_curation_key_matches_its_colony_whatever_the_label_dtype(
    dtype,
) -> None:
    """The label must join on its VALUE, not on how it is stored.

    ``Object_Label`` is ``Int64`` in every real master, but the join is
    the one place a narrower or string-typed label would fail silently
    rather than loudly, so the supported dtypes are stated rather than
    assumed.
    """
    flagged = join_curation_flags(_labelled_frame(dtype), [["a", 2]])

    assert flagged[REMOVED_COL].to_list() == [False, True, False], (
        f"a {dtype} label joined on something other than its value"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A category-coded label is an accepted precondition, not a "
        "supported input. Polars casts both Categorical AND Enum to their "
        "physical CATEGORY CODES, so label 2 reads as code 2 -- the third "
        "colony, not the second. Not guarded because nothing produces the "
        "state: CurationLabels pins the key side to Int64 by declared "
        "schema (_curation_labels.py:829-836), the real master carries "
        "Object_Label as Int64 with no categorical column in 149, and "
        "_curation_labels.py:638 already casts this same column unguarded "
        "on the same frame -- more strictly, without strict=False. "
        "Guarding it here alone would be dead code diverging from shipped "
        "precedent. STRICT ON PURPOSE: if this starts passing, the cast "
        "changed and the decision needs revisiting."
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [pl.Categorical, pl.Enum(["1", "2", "3"])],
    ids=["Categorical", "Enum"],
)
def test_a_category_coded_label_is_out_of_scope_and_here_is_why(dtype) -> None:
    """A standing, executable record of a trap we chose not to guard.

    Both dtypes are listed because both were measured to fail, and Enum
    is the one a later reader would drop as an unused case -- it looks
    like defensive padding beside Categorical and is not.

    Written because the next person to read the join will wonder, and a
    comment alone cannot tell them whether the behaviour is still what it
    was.
    """
    flagged = join_curation_flags(_labelled_frame(dtype), [["a", 2]])

    assert flagged[REMOVED_COL].to_list() == [False, True, False]


def test_a_null_label_matches_no_curation_key() -> None:
    """Every phantom has one, and a phantom is not a curatable object."""
    frame = pl.DataFrame(
        {
            KEY_IMAGE_FILE: ["a", "a"],
            KEY_OBJECT_LABEL: pl.Series([1, None], dtype=pl.Int64),
        }
    )

    flagged = join_curation_flags(frame, [["a", 1]])

    assert flagged[REMOVED_COL].to_list() == [True, False]


def test_a_store_payload_is_read_as_a_number_or_refused() -> None:
    """``bool`` is an ``int`` subclass, and that is the whole trap.

    ``int(True)`` is ``1``, so a truthy flag arriving where a count is
    expected would silently become "section 1" or "1 px wide". The two
    guards are asserted separately because they are separately
    removable: without the bool branch ``True`` becomes ``1``, without
    the type branch ``None`` raises instead of refusing.
    """
    assert store_int(3) == 3
    assert store_int(3.7) == 3, "a width arrives from getBoundingClientRect"
    assert store_int("360") == 360
    assert store_int(True) is None, "a flag is not a count"
    assert store_int(False) is None
    assert store_int(None) is None
    assert store_int("not a number") is None
    assert store_int([1]) is None
    # `int` raises OverflowError for an infinity and ValueError for a NaN.
    # That split is an accident of `int`, not a distinction this function
    # makes, so both must refuse rather than one refusing and one raising.
    assert store_int(float("inf")) is None
    assert store_int(float("-inf")) is None
    assert store_int(float("nan")) is None


def test_a_frame_without_curation_keys_gets_no_flag_column(
    output_root: OutputRoot,
) -> None:
    """Absence is the contract for "nothing is removed".

    ``_figure._split_on_curation`` reads a missing column as "nothing
    removed"; inventing an all-False column for a frame that cannot carry
    curation keys would claim more than this module knows.
    """
    keyless = pl.DataFrame({"x": [1.0], "y": [2.0]})

    assert REMOVED_COL not in join_curation_flags(keyless, [["a", 1]]).columns


def test_curation_is_joined_before_the_phantom_filter(
    output_root: OutputRoot,
) -> None:
    """Ordering, stated as an executable fact.

    ``prepare_frame`` must hand the figure builder a frame that already
    carries the Boolean flag, whose indices are still master positions,
    and from which the phantoms are gone. The index assertion is the
    load-bearing one: it is what fails if the join or the filter is moved
    ahead of :func:`~.._inspector.index_frame`.
    """
    frame, _ = prepare_frame(
        output_root,
        x_col=_TIME,
        y_col=_AREA,
        filter_payload=None,
        removed_payload=[["a", 1]],
    )

    assert frame.schema[REMOVED_COL] == pl.Boolean
    assert not frame[CURATION_PHANTOM_COL].to_list().count(True)
    assert frame[CUSTOMDATA_COL].to_list() == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Pager, legend, crop URL, export
# ---------------------------------------------------------------------------


def test_the_pager_clamps_into_the_section_list(output_root: OutputRoot) -> None:
    """A live run can retire the section the pager was on."""
    assert clamp_section_index(9, 3) == 2
    assert clamp_section_index(-4, 3) == 0
    assert clamp_section_index(None, 3) == 0
    assert clamp_section_index(1, 0) == 0, "no sections is still page one"


def test_sections_come_from_the_frame_in_page_order(
    output_root: OutputRoot,
) -> None:
    """The pages a render will produce, from the frame it will draw."""
    frame, _ = prepare_frame(
        output_root,
        x_col=_TIME,
        y_col=_AREA,
        filter_payload=None,
        removed_payload=None,
    )

    assert section_values(frame, _STRAIN) == ["BY4741", "S288C", "W303"]
    assert section_values(frame, None) == []
    assert section_values(frame, "Metadata_NotAColumn") == []


def test_the_pager_arrows_step_in_the_directions_they_are_labelled(
) -> None:
    """Which button means which way, not the arithmetic.

    The clamp is already pinned separately. What was unreachable until
    this logic left the callback closure is the mapping from the clicked
    id to the direction -- and a swapped pair walks backwards from a
    button labelled "next" without raising anything.
    """
    assert paged_section_index(ids.SCATTER_NEXT_BTN, 2, 5) == 3
    assert paged_section_index(ids.SCATTER_PREV_BTN, 2, 5) == 1


def test_the_pager_cannot_step_off_either_end() -> None:
    """The clamp still applies to a stepped index, not just a stored one."""
    assert paged_section_index(ids.SCATTER_NEXT_BTN, 4, 5) == 4
    assert paged_section_index(ids.SCATTER_PREV_BTN, 0, 5) == 0


def test_the_section_index_selects_which_section_is_drawn(
    output_root: OutputRoot,
) -> None:
    """The pager has to move the figure, not just the chip.

    Asserted on the carried master indices rather than on the point
    count: every section here has two colonies, so a count would be
    satisfied by drawing the wrong section, which is exactly the failure
    a pager makes.
    """
    figure, label, fingerprint = _render(output_root, section_index=1)

    # Rows 2 and 3 of the master are the S288C colonies; 0-1 are BY4741.
    assert _carried(figure) == [2, 3]
    assert "S288C" in label
    assert "(2 / 3)" in label
    resolved = [
        resolve_click(output_root.master_df, i, fingerprint, fingerprint)
        for i in _carried(figure)
    ]
    assert [r.stem for r in resolved if r] == ["b", "b"]


def test_the_legend_settings_reach_the_rendered_figure(
    output_root: OutputRoot,
) -> None:
    """Producer tested, consumer tested, WIRING untested.

    ``legend_layout`` is asserted as a pure function elsewhere and the
    controls that feed it are asserted to be bound. Neither notices if
    the render stops applying the result -- both legend controls become
    dead chrome with a green suite. That is the identical shape as the
    ``show_removed`` gap which had a UI and no producer, so it is worth a
    test that spans the seam rather than either side of it.
    """
    collapsed, _, _ = _render(
        output_root, legend_payload={"corner": "top-left", "collapsed": True}
    )
    placed, _, _ = _render(
        output_root, legend_payload={"corner": "top-left", "collapsed": False}
    )

    assert collapsed.layout.showlegend is False
    assert placed.layout.showlegend is True
    assert placed.layout.legend.xanchor == "left"
    assert placed.layout.legend.yanchor == "top"


def test_a_null_grouping_value_is_dropped_rather_than_becoming_a_page() -> None:
    """Spec section 10: never a silent omission, never a "(none)" page.

    Built by hand rather than taken off the fixture, and deliberately so.
    The run fixture has no null strain, so asserting this against it would
    pass whether the null-drop existed or not -- it would be a test of
    three values that happen not to include one.
    """
    frame = pl.DataFrame({_STRAIN: ["BY4741", None, "S288C", None]})

    assert section_values(frame, _STRAIN) == ["BY4741", "S288C"]


def test_the_legend_moves_corner_and_collapses() -> None:
    """Both halves of spec section 9's legend row reach the figure."""
    default = legend_layout(None)
    top_left = legend_layout({"corner": "top-left", "collapsed": False})
    collapsed = legend_layout({"corner": "top-left", "collapsed": True})

    assert default["showlegend"] is True
    assert default["legend"] == legend_layout(
        {"corner": LEGEND_CORNER_DEFAULT}
    )["legend"]
    assert top_left["legend"]["xanchor"] == "left"
    assert top_left["legend"]["yanchor"] == "top"
    assert collapsed["showlegend"] is False


def test_the_contours_control_changes_only_the_query_parameter() -> None:
    """Spec section 7: the toggle re-requests, it does not re-resolve."""
    colony = {"dataset": "d1", "stem": "a", "label": 2}

    with_contours = crop_url(MOUNT_HOME, colony, contours=1)
    without = crop_url(MOUNT_HOME, colony, contours=0)

    assert with_contours.split("?")[0] == without.split("?")[0]
    assert with_contours.endswith("contours=1")
    assert without.endswith("contours=0")


def test_the_crop_url_targets_the_scatter_segment_under_the_mount_prefix() -> None:
    """The hub mounts the viewer under a prefix; the crop must follow it."""
    url = crop_url(
        "/results/", {"dataset": "d1", "stem": "a", "label": 2}, contours=1
    )

    assert url.startswith(f"/results/{SCATTER_CROPS_URL_SEGMENT}/d1/a/2.png?")


def test_an_unselected_colony_renders_no_crop() -> None:
    """An empty ``src`` draws nothing, which is what an empty panel wants."""
    assert crop_url(MOUNT_HOME, None, contours=1) == ""
    assert crop_url(MOUNT_HOME, {"dataset": "d1"}, contours=1) == ""


def test_the_export_button_does_nothing_until_it_is_clicked(
    output_root: OutputRoot,
) -> None:
    """A ``dcc.Download`` written on mount would download on page load."""
    download, status = export_payload(
        output_root,
        n_clicks=0,
        section_col=_STRAIN,
        row_col=None,
        col_col=None,
        x_col=_TIME,
        y_col=_AREA,
        hue_col=None,
        shape_col=None,
        show_removed=True,
        filter_payload=None,
        removed_payload=None,
    )

    assert download is dash.no_update
    assert status == ""


def test_the_export_refuses_an_unconfigured_figure_out_loud(
    output_root: OutputRoot,
) -> None:
    """A run with no numeric column leaves Y unset; say so."""
    download, status = export_payload(
        output_root,
        n_clicks=1,
        section_col=_STRAIN,
        row_col=None,
        col_col=None,
        x_col=_TIME,
        y_col=None,
        hue_col=None,
        shape_col=None,
        show_removed=True,
        filter_payload=None,
        removed_payload=None,
    )

    assert download is dash.no_update
    assert status


def test_the_inspector_handle_declares_its_own_splitter_wiring(
    output_root: OutputRoot,
) -> None:
    """The shared splitter names no surface; the handle names both ids.

    A handle missing ``data-splitter-target`` is not a handle at all to
    ``results_viewer.js`` section H -- it simply never attaches, with
    nothing raising on either side.
    """
    body = build_scatter_tab_body(output_root)
    handle = next(
        node
        for node in walk_components(body)
        if getattr(node, "id", None) == ids.SCATTER_INSPECTOR_SPLITTER
    )

    props = handle.to_plotly_json()["props"]
    assert props["data-splitter-target"] == ids.SCATTER_INSPECTOR
    assert props["data-splitter-store"] == ids.STORE_SCATTER_INSPECTOR_WIDTH
