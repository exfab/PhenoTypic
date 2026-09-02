"""Above the cap, cells virtualize rather than all mounting.

Spec section 6.2 records the cap as measured during D1, so these tests read
it from the single source rather than restating a literal -- a cap that
appears in two places drifts.

The last two tests are the curation half. Virtualization is a rendering
budget, and it must not quietly become a change to what the grid contains:
every populated cell keeps its place in ``grid_order`` (so a shift+click
range and a bulk mark still reach it) and every MOUNTED cell keeps its
radial trigger. A mounted cell missing its radial is a curation regression;
a virtualized cell without one is correct, because it has no tile.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from dash.development.base_component import Component

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import (
    COLONY_VIEW_CELL_CAP,
    build_grid,
    plan_visible_cells,
)
from phenotypic.schema import IMAGE
from tests._output_layout import write_master


def test_cells_beyond_the_cap_are_not_mounted() -> None:
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP * 2)]
    visible = plan_visible_cells(cells, focus_index=0)
    assert len(visible) <= COLONY_VIEW_CELL_CAP


def test_the_focused_cell_is_always_visible() -> None:
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP * 2)]
    focus = COLONY_VIEW_CELL_CAP + 5
    visible = plan_visible_cells(cells, focus_index=focus)
    assert any(c["id"] == focus for c in visible)


def test_a_grid_under_the_cap_mounts_every_cell() -> None:
    """The cap must not virtualize a grid that fits.

    Every real plate layout below the cap has to render whole -- a window
    that always trimmed would silently drop cells from a 96-colony grid.
    """
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP)]
    assert plan_visible_cells(cells, focus_index=0) == cells


def test_the_window_stays_full_at_either_end() -> None:
    """Clamped, not centred: an edge focus still mounts a full cap.

    A window centred unconditionally on ``focus_index`` would mount half a
    cap at index 0 -- fewer cells exactly where the user starts.
    """
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP * 3)]
    for focus in (0, len(cells) - 1):
        assert len(plan_visible_cells(cells, focus_index=focus)) == (
            COLONY_VIEW_CELL_CAP
        )


def _wide_output_root(tmp_path: Path, n_cells: int) -> OutputRoot:
    """An output root whose master frame fills an ``n_cells``-cell grid.

    One colony per (row, col) pair, so ``build_grid`` renders exactly
    ``n_cells`` populated cells and nothing aggregates.
    """
    side = 1
    while side * side < n_cells:
        side += 1
    rows = [(i // side, i % side) for i in range(n_cells)]
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate1"] * n_cells,
            str(IMAGE.IMAGE_NAME): ["img-001"] * n_cells,
            "Object_Label": list(range(1, n_cells + 1)),
            "Bbox_MinRR": [0] * n_cells,
            "Bbox_MaxRR": [40] * n_cells,
            "Bbox_MinCC": [0] * n_cells,
            "Bbox_MaxCC": [40] * n_cells,
            "Bbox_CenterRR": [20] * n_cells,
            "Bbox_CenterCC": [20] * n_cells,
            "Grid_RowNum": [r for r, _ in rows],
            "Grid_ColNum": [c for _, c in rows],
        }
    )
    (tmp_path / "results" / "plate1" / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    write_master(tmp_path, master)
    return OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )


def _radial_trigger_count(component: Component) -> int:
    """Count ``colony-radial-trigger`` pattern ids in a component tree."""
    found = 0
    stack: list[object] = [component]
    while stack:
        node = stack.pop()
        node_id = getattr(node, "id", None)
        if isinstance(node_id, dict) and (
            node_id.get("type") == "colony-radial-trigger"
        ):
            found += 1
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return found


def test_build_grid_mounts_at_most_the_cap_and_keeps_the_full_key_order(
    tmp_path: Path,
) -> None:
    """Above the cap: capped radials, complete ``grid_order``."""
    n_cells = COLONY_VIEW_CELL_CAP + 40
    root = _wide_output_root(tmp_path, n_cells)

    component, grid_order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
    )

    # Every populated cell keeps its key, so range selection and bulk-mark
    # still reach a cell that happens to be virtualized out of the mount.
    assert len(grid_order) == n_cells
    # ...but only the mounted ones carry chrome.
    assert _radial_trigger_count(component) == COLONY_VIEW_CELL_CAP


def test_every_mounted_cell_carries_its_radial(tmp_path: Path) -> None:
    """Below the cap nothing virtualizes and every cell keeps its radial."""
    n_cells = 24
    root = _wide_output_root(tmp_path, n_cells)

    component, grid_order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
    )

    assert len(grid_order) == n_cells
    assert _radial_trigger_count(component) == n_cells


def test_the_focused_cell_mounts_even_past_the_cap(tmp_path: Path) -> None:
    """A focus beyond the first window moves the mounted cells with it."""
    n_cells = COLONY_VIEW_CELL_CAP * 2
    root = _wide_output_root(tmp_path, n_cells)

    kwargs = dict(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
    )
    first, order_first = build_grid(**kwargs, focus_index=0)
    last, order_last = build_grid(**kwargs, focus_index=n_cells - 1)

    assert order_first == order_last
    assert _radial_trigger_count(first) == COLONY_VIEW_CELL_CAP
    assert _radial_trigger_count(last) == COLONY_VIEW_CELL_CAP

    def _mounted_keys(component: Component) -> set[tuple[str, int]]:
        keys: set[tuple[str, int]] = set()
        stack: list[object] = [component]
        while stack:
            node = stack.pop()
            node_id = getattr(node, "id", None)
            if isinstance(node_id, dict) and (
                node_id.get("type") == "colony-radial-trigger"
            ):
                keys.add((node_id["image_file"], int(node_id["label"])))
            children = getattr(node, "children", None)
            if isinstance(children, (list, tuple)):
                stack.extend(children)
            elif children is not None:
                stack.append(children)
        return keys

    assert _mounted_keys(first) != _mounted_keys(last), (
        "the mounted window did not move with focus_index -- the cap would "
        "make the tail of a large grid permanently unreachable"
    )
