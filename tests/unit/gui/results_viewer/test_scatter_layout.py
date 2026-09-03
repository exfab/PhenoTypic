"""The Scatter tab body: option lists, defaults, and what a filter can do.

The tab resolves its dropdown options at build time rather than in a
callback, unlike the Colony view. That is a deliberate choice recorded in
``_scatter_tab/_layout.py``, and it has one consequence worth an
executable statement: a filter can narrow a selected column to a single
value without the option list noticing. This module pins that the figure
path treats that as an ordinary figure, which is what makes the choice
safe -- see :func:`test_a_column_narrowed_to_one_value_still_plans_and_renders`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
import pytest

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._scatter_tab import _ids as ids
from phenotypic.gui.results_viewer._scatter_tab._facets import (
    COMPUTED_FRAME_INDEX,
)
from phenotypic.gui.results_viewer._scatter_tab._layout import (
    CONFIG_POPOVER_WIDTH_PX,
    build_scatter_tab_body,
)
from phenotypic.schema import CULTURE, IMAGE, SIZE
from phenotypic.sdk_ import is_metadata_header, master_measurements_parquet_path

#: Real headers, asked of the schema. Spelling one by hand is this
#: branch's most-repeated defect: the string reads correctly, matches no
#: measurer, and the assertion then passes or fails for an unrelated
#: reason.
_AREA = str(SIZE.AREA)
_IMAGE_NAME = str(IMAGE.IMAGE_NAME)
_FRAME_INDEX = str(CULTURE.FRAME_INDEX)


def _output_for(frame: pl.DataFrame, tmp_path: Path) -> OutputRoot:
    """Discover an ``OutputRoot`` over one hand-built display frame.

    Mirrors the ``built_results_layout`` fixture's recipe: the layout
    builder needs a discoverable root with one measured image, so the
    master parquet plus one overlay is enough.

    Args:
        frame: The display frame the viewer should load.
        tmp_path: Per-test directory to build the run under.

    Returns:
        A discovered :class:`OutputRoot`.
    """
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = tmp_path / "deliverables" / "overlays" / "d1"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / "a.png").touch()
    target = master_measurements_parquet_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    return OutputRoot.discover(
        tmp_path, cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache"
    )


def _walk(component: object) -> Iterator[object]:
    """Yield ``component`` and every descendant, depth-first."""
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None:
        yield from _walk(children)


def _control(body: object, control_id: str) -> object:
    """Return the mounted control carrying ``control_id``."""
    for node in _walk(body):
        if getattr(node, "id", None) == control_id:
            return node
    raise AssertionError(f"{control_id} is not mounted in the Scatter body")


def _option_values(control: object) -> list[str]:
    """Return a dropdown's offered values, in offered order."""
    return [entry["value"] for entry in control.options]


@pytest.fixture()
def three_strain_frame() -> pl.DataFrame:
    """A frame with one grouping column, one measurement, one empty column.

    ``Metadata_Strain`` carries three values, so it is a legal grouping
    column. ``Metadata_Empty`` is declared **Float64 and all-null** on
    purpose: that is the only shape the non-empty value-set guard in
    ``_numeric_columns`` actually catches. A bare ``[None] * 6`` infers
    polars ``Null`` dtype, which ``is_numeric_column`` already rejects on
    dtype -- so a fixture built that way would pass whether the guard
    existed or not, which is the failure this fixture is written to
    avoid rather than a detail of it.
    """
    return pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 6,
            _IMAGE_NAME: ["a"] * 6,
            "Metadata_Strain": ["BY4741", "BY4741", "S288C", "S288C", "W303", "W303"],
            "Metadata_Time": [0.0, 6.0, 0.0, 6.0, 0.0, 6.0],
            _AREA: [10.0, 20.0, 11.0, 21.0, 12.0, 22.0],
            "Metadata_Empty": pl.Series([None] * 6, dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_section_group_defaults_to_a_metadata_column(
    three_strain_frame, tmp_path
) -> None:
    """Spec section 9: open on the first metadata column with 2-50 values."""
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    section = _control(body, ids.SCATTER_SECTION_COL)
    assert section.value == "Metadata_Strain"
    assert is_metadata_header(section.value)


def test_y_axis_defaults_to_a_measurement_not_a_numeric_metadata_column(
    three_strain_frame, tmp_path
) -> None:
    """``Metadata_Time`` is numeric and sorts first; Y must still pick Area.

    The frame is built so the naive "first numeric column" answer is the
    wrong one -- otherwise this test would pass against an implementation
    that never consults the measurement prefixes at all.
    """
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    assert _control(body, ids.SCATTER_Y_COL).value == _AREA


def test_x_axis_falls_back_to_the_derived_frame_index(
    three_strain_frame, tmp_path
) -> None:
    """With no capture ordinal in the run, X offers and picks the derived one."""
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    x_axis = _control(body, ids.SCATTER_X_COL)
    assert x_axis.value == COMPUTED_FRAME_INDEX
    assert COMPUTED_FRAME_INDEX in _option_values(x_axis)


def test_x_axis_prefers_the_runs_own_capture_ordinal(
    three_strain_frame, tmp_path
) -> None:
    """A run that records the capture ordinal is plotted against it.

    Also pins that the schema member the layout resolves really does
    land in the metadata namespace: the column is offered under the name
    ``CULTURE.FRAME_INDEX`` stringifies to, and if that stopped being a
    metadata header the default would silently revert to the derived
    index with nothing failing.
    """
    assert is_metadata_header(_FRAME_INDEX)
    frame = three_strain_frame.with_columns(
        pl.Series(_FRAME_INDEX, [0, 1, 0, 1, 0, 1], dtype=pl.Int32)
    )

    body = build_scatter_tab_body(_output_for(frame, tmp_path))

    assert _control(body, ids.SCATTER_X_COL).value == _FRAME_INDEX


def test_a_numeric_column_with_no_values_is_offered_as_no_axis_at_all(
    three_strain_frame, tmp_path
) -> None:
    """A column typed numeric but never populated is not an axis.

    ``is_numeric_column`` answers this one on ``schema[column]
    .is_numeric()`` alone and never looks at the values, so only the
    non-empty value-set guard in ``_numeric_columns`` keeps it out of
    the list. See the fixture for why the dtype is load-bearing.
    """
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    assert "Metadata_Empty" not in _option_values(_control(body, ids.SCATTER_X_COL))
    assert "Metadata_Empty" not in _option_values(_control(body, ids.SCATTER_Y_COL))


def test_the_facet_and_channel_roles_open_unset(
    three_strain_frame, tmp_path
) -> None:
    """Rows, columns, hue and shape default to a single undivided series."""
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    for control_id in (
        ids.SCATTER_ROW_COL,
        ids.SCATTER_COL_COL,
        ids.SCATTER_HUE_COL,
        ids.SCATTER_SHAPE_COL,
    ):
        assert _control(body, control_id).value is None, control_id


def test_removed_colonies_are_shown_by_default(three_strain_frame, tmp_path) -> None:
    """Spec section 9: the curation toggle opens on."""
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    assert _control(body, ids.SCATTER_SHOW_REMOVED).value is True


def test_every_declared_id_is_actually_mounted(
    three_strain_frame, tmp_path
) -> None:
    """An id in ``_ids.__all__`` that nothing mounts is a callback that binds
    to nothing.

    Task 13 registers this tab's callbacks against these ids, and it has
    not been written yet, so nothing else currently checks that the two
    halves agree. A *symbol* rename would announce itself as an
    ``AttributeError`` when Task 13 imports it; declaring an id and never
    mounting it is the half that fails quietly, and it is the half this
    catches.
    """
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    # Two kinds of mounted id, and a declared name may be either. A
    # literal id mounts as a string; a pattern-matching id mounts as a
    # dict, and what `_ids` declares for one of those is the `type` token
    # its instances share -- `SCATTER_STYLE_STEP` is never an id by
    # itself, it is the `type` of eight pairs of them. Collecting only
    # string ids would report every such token as unmounted; collecting
    # both into one set raises `TypeError: unhashable type: 'dict'`.
    mounted_strings: set[str] = set()
    mounted_types: set[str] = set()
    for node in _walk(body):
        node_id = getattr(node, "id", None)
        if isinstance(node_id, str):
            mounted_strings.add(node_id)
        elif isinstance(node_id, dict) and isinstance(node_id.get("type"), str):
            mounted_types.add(node_id["type"])

    declared = {getattr(ids, name) for name in ids.__all__}
    unmounted = declared - mounted_strings - mounted_types
    assert not unmounted, f"declared but never mounted: {unmounted}"


# ---------------------------------------------------------------------------
# The consequence of resolving options at build time
# ---------------------------------------------------------------------------


def test_a_column_narrowed_to_one_value_still_plans_and_renders(
    three_strain_frame,
) -> None:
    """A filter can empty an option list's premise; that must be harmless.

    Options describe the run, so a column offered as a section group can
    be narrowed by the shared filter to a single value -- or to none --
    after the fact. The Colony view avoids this by re-deriving its
    options and moving the selection; Scatter holds the selection still,
    which is only safe if one value is an ordinary figure rather than an
    edge case. This is the test that makes that claim checkable.
    """
    from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
    from phenotypic.gui.results_viewer._scatter_tab._figure import (
        CUSTOMDATA_COL,
        build_scatter_figure,
    )
    from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

    one_strain = three_strain_frame.filter(
        pl.col("Metadata_Strain") == "BY4741"
    ).with_columns(pl.Series(CUSTOMDATA_COL, [0, 1], dtype=pl.Int32))
    assert one_strain["Metadata_Strain"].n_unique() == 1

    spec = FigureSpec(x_col="Metadata_Time", y_col=_AREA, row_col="Metadata_Strain")
    plan = plan_facets(one_strain, spec)

    assert plan.rows == ["BY4741"]
    assert plan.truncated is False
    figure = build_scatter_figure(one_strain, spec, plan)
    assert len(figure.data) >= 1


def test_a_column_filtered_away_entirely_still_plans(three_strain_frame) -> None:
    """The degenerate end of the same case: zero rows, still one panel."""
    from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
    from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

    empty = three_strain_frame.filter(pl.col("Metadata_Strain") == "nonesuch")
    assert empty.height == 0

    plan = plan_facets(
        empty, FigureSpec(x_col="Metadata_Time", y_col=_AREA, row_col="Metadata_Strain")
    )

    # `_values`'s [""] fallback: an empty axis must collapse to one panel,
    # never to zero.
    assert plan.rows == [""]
    assert plan.cols == [""]


def test_the_settings_popover_declares_one_fixed_width(
    three_strain_frame, tmp_path
) -> None:
    """Both ``width`` and ``maxWidth``, and no control setting a floor.

    A ``maxWidth`` alone caps the wide sections and lets the narrow ones
    shrink, which is half the movement this removes -- measured, the
    popover ran 279 px on Style against 320 px on Data.

    The second assertion is the one that keeps it fixed. Any descendant
    declaring a ``minWidth`` can demand more than the popover grants and
    reintroduce exactly the per-section variation, and it would do so
    silently: the popover still has a width, it is just not the width it
    declared.
    """
    body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))
    popover = next(
        node
        for node in _walk(body)
        if getattr(node, "id", None) == ids.SCATTER_CONFIG_POPOVER
    )

    style = popover.style
    assert style["width"] == f"{CONFIG_POPOVER_WIDTH_PX}px"
    assert style["maxWidth"] == f"{CONFIG_POPOVER_WIDTH_PX}px"

    # Not "no min-width anywhere" -- the stepper readouts declare 3rem so
    # the number does not jitter its neighbours as it goes 9 -> 14 -> 100,
    # and 48 px inside a 266 px row cannot push anything. What matters is
    # that no declared minimum EXCEEDS the width the popover grants, which
    # is the only way one can win against it.
    content_px = CONFIG_POPOVER_WIDTH_PX - (2 + 32 + 40)
    offenders = []
    for node in _walk(popover):
        style = getattr(node, "style", None)
        if not isinstance(style, dict) or "minWidth" in style is None:
            continue
        raw = style.get("minWidth")
        if raw is None:
            continue
        if isinstance(raw, str) and raw.endswith("px"):
            demand = float(raw.removesuffix("px"))
        elif isinstance(raw, str) and raw.endswith("rem"):
            demand = float(raw.removesuffix("rem")) * 16
        elif isinstance(raw, (int, float)):
            demand = float(raw)
        else:
            continue
        if demand > content_px:
            offenders.append((getattr(node, "id", None), raw))
    assert not offenders, (
        f"min-width exceeding the popover's {content_px}px of content: "
        f"{offenders}"
    )


def test_the_popover_is_wider_than_the_chrome_it_must_hold(
    three_strain_frame, tmp_path
) -> None:
    """The width must leave a usable control, not merely be a number.

    ``.popover-body`` padding, ``.accordion-body`` padding and the
    popover border take 74 px before any control is drawn -- measured off
    the computed styles. Asserting the remainder clears the 240 px the
    dropdowns used to ask for keeps the constant honest if someone tunes
    it down.
    """
    _body = build_scatter_tab_body(_output_for(three_strain_frame, tmp_path))

    chrome_px = 2 + 32 + 40
    assert CONFIG_POPOVER_WIDTH_PX - chrome_px >= 240
