"""Colony cards showing a measurement: the scale, the gaps, and the chrome.

The load-bearing test here is
:func:`test_the_scale_spans_the_values_in_view_not_the_whole_column`. Scaling
to the column's global range passes every smoke test -- a value appears, a
colour appears, nothing raises -- and produces a uniformly-coloured grid the
moment the user filters, which is the only time the feature is worth having.
So the fixture deliberately hands ``build_grid`` values for colonies that are
*not* on the grid, and asserts the visible extremes still reach the ends of
the ramp.

The other three pin what must NOT change: a colony with no row renders as it
did before, and a tinted card still carries its radial trigger and its
selection checkbox. Curation chrome is a background this feature paints
behind, never a layer it replaces.
"""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import polars as pl
from dash import html
from dash.development.base_component import Component

from phenotypic.gui._shared._measurement_tint import sequential_tint
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import build_grid
from phenotypic.schema import IMAGE

from tests._output_layout import write_master

COLUMN = "Shape_Area"

#: The four colonies the grid renders, and the areas they carry.
IN_VIEW: dict[tuple[str, int], float] = {
    ("img-001", 1): 100.0,
    ("img-001", 2): 200.0,
    ("img-002", 1): 300.0,
    ("img-002", 2): 400.0,
}

#: Colonies the store measured that this grid does NOT show -- filtered out,
#: or belonging to another image. Their values dwarf the visible ones, so a
#: scale built over the whole mapping instead of the visible subset squashes
#: every rendered card into the ramp's bottom 4%.
OUT_OF_VIEW: dict[tuple[str, int], float] = {
    ("img-009", 1): 0.0,
    ("img-009", 2): 10_000.0,
}


def _make_output_root(tmp_path: Path) -> OutputRoot:
    """A two-image run whose four colonies fill a 2x2 grid."""
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate1"] * 4,
            str(IMAGE.IMAGE_NAME): ["img-001", "img-001", "img-002", "img-002"],
            "Object_Label": [1, 2, 1, 2],
            "Bbox_MinRR": [0, 5, 10, 15],
            "Bbox_MaxRR": [40, 45, 50, 55],
            "Bbox_MinCC": [0, 5, 10, 15],
            "Bbox_MaxCC": [40, 45, 50, 55],
            "Bbox_CenterRR": [20, 25, 30, 35],
            "Bbox_CenterCC": [20, 25, 30, 35],
            "Grid_RowNum": [1, 2, 1, 2],
            "Grid_ColNum": [1, 1, 2, 2],
        }
    )
    (tmp_path / "results" / "plate1" / "measurements").mkdir(parents=True)
    write_master(tmp_path, master)
    return OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )


def _walk(node: object):
    """Yield every component in a rendered tree, depth first."""
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None:
        yield from _walk(children)


def _measurement_labels(component: Component) -> list[tuple[str, str]]:
    """Every card's ``(text, tint)`` pair, in render order."""
    found = []
    for node in _walk(component):
        if (
            isinstance(node, html.Span)
            and getattr(node, "className", None) == "colony-cell-measurement"
        ):
            found.append((node.children, node.style["background"]))
    return found


def _render(
    tmp_path: Path,
    *,
    values: dict[tuple[str, int], float] | None,
    column: str | None = COLUMN,
):
    root = _make_output_root(tmp_path)
    return build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
        measurement_column=column,
        measurement_values=values,
    )


# ---------------------------------------------------------------------------
# The scale
# ---------------------------------------------------------------------------


def test_the_scale_spans_the_values_in_view_not_the_whole_column(
    tmp_path: Path,
) -> None:
    """The one failure a smoke test cannot see.

    ``measurement_values`` carries 0 and 10,000 for colonies the grid does
    not render. Scale over the mapping and the visible 100..400 occupy the
    ramp's first 4%, so every card comes out the same near-white. Scale over
    what is drawn and the visible extremes reach both ends.
    """
    component, _order = _render(
        tmp_path, values={**IN_VIEW, **OUT_OF_VIEW}
    )
    tints = {tint for _text, tint in _measurement_labels(component)}
    assert len(tints) == len(IN_VIEW), "each visible value gets its own shade"
    assert sequential_tint(0.0) in tints, "the smallest visible value anchors 0"
    assert sequential_tint(1.0) in tints, "the largest visible value anchors 1"


def test_each_card_wears_its_own_value(tmp_path: Path) -> None:
    """The text on a card is the value joined to that card's ``Object_Label``."""
    component, order = _render(tmp_path, values=IN_VIEW)
    labels = _measurement_labels(component)
    assert [text for text, _tint in labels] == [
        str(int(IN_VIEW[key])) for key in order
    ]


def test_a_degenerate_scale_does_not_divide_by_zero(tmp_path: Path) -> None:
    """Every visible colony measuring the same is a result, not a fault."""
    flat = dict.fromkeys(IN_VIEW, 42.0)
    component, _order = _render(tmp_path, values=flat)
    labels = _measurement_labels(component)
    assert len(labels) == len(IN_VIEW)
    assert {tint for _text, tint in labels} == {sequential_tint(0.0)}


# ---------------------------------------------------------------------------
# The gaps
# ---------------------------------------------------------------------------


def test_a_colony_with_no_row_renders_untinted_with_no_text(
    tmp_path: Path,
) -> None:
    """Post-measurement operations remove objects. That is not an error."""
    partial = {
        key: value
        for key, value in IN_VIEW.items()
        if key != ("img-002", 2)
    }
    component, order = _render(tmp_path, values=partial)
    assert len(order) == len(IN_VIEW)
    assert len(_measurement_labels(component)) == len(partial)


def test_no_column_chosen_renders_todays_grid(tmp_path: Path) -> None:
    """The default state, and the state of every unmeasured run."""
    component, order = _render(tmp_path, values=None, column=None)
    assert len(order) == len(IN_VIEW)
    assert _measurement_labels(component) == []


def test_values_without_a_column_render_nothing(tmp_path: Path) -> None:
    """A stale value mapping must not tint a grid with no column chosen."""
    component, _order = _render(tmp_path, values=IN_VIEW, column=None)
    assert _measurement_labels(component) == []


# ---------------------------------------------------------------------------
# The legend
# ---------------------------------------------------------------------------


def test_a_legend_names_the_column_and_its_visible_range(
    tmp_path: Path,
) -> None:
    """Without it the tint means nothing."""
    component, _order = _render(tmp_path, values=IN_VIEW)
    texts = [
        node.children
        for node in _walk(component)
        if isinstance(node, html.Span) and isinstance(node.children, str)
    ]
    assert COLUMN in texts
    assert "100" in texts
    assert "400" in texts


def test_no_legend_without_a_column(tmp_path: Path) -> None:
    component, _order = _render(tmp_path, values=None, column=None)
    assert not [
        node
        for node in _walk(component)
        if getattr(node, "className", None) == "colony-measurement-legend"
    ]


# ---------------------------------------------------------------------------
# Curation chrome is untouched
# ---------------------------------------------------------------------------


def test_a_tinted_card_still_carries_its_radial_trigger(
    tmp_path: Path,
) -> None:
    """The tint is a background, never a replacement for the curation layer."""
    component, order = _render(tmp_path, values=IN_VIEW)
    triggers = [
        node
        for node in _walk(component)
        if isinstance(node, dbc.Button)
        and isinstance(node.id, dict)
        and node.id.get("type") == "colony-radial-trigger"
    ]
    assert len(triggers) == len(order)


def test_a_tinted_card_still_carries_its_selection_checkbox(
    tmp_path: Path,
) -> None:
    component, order = _render(tmp_path, values=IN_VIEW)
    checkboxes = [
        node
        for node in _walk(component)
        if getattr(node, "className", "") == "colony-cell-checkbox"
    ]
    assert len(checkboxes) == len(order)
