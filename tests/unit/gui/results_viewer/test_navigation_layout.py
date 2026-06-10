"""Layout tests for Results Viewer icon navigation controls."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from dash import dcc

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer import _viewer_card
from phenotypic.gui.results_viewer._heatmap_tab import _ids as heatmap_ids
from phenotypic.gui.results_viewer._heatmap_tab import _layout as heatmap_layout
from phenotypic.gui.results_viewer.colony_view import _layout as colony_layout
from phenotypic.gui.shell import _sidebar
from phenotypic.gui.shell._ids import sidebar_entry_id
from phenotypic.gui.shell._sandbox import SandboxRoot


def _walk(component: object) -> Iterator[object]:
    """Yield ``component`` and every descendant, depth-first."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _props(component: object) -> dict:
    """Return a Dash component's serializable props."""
    if hasattr(component, "to_plotly_json"):
        return component.to_plotly_json().get("props", {})
    return {}


def _find_by_id(component: object, target: object) -> object:
    for node in _walk(component):
        if getattr(node, "id", None) == target:
            return node
    raise AssertionError(f"missing component id {target!r}")


def test_plate_card_renders_icon_image_navigation_buttons() -> None:
    card = _viewer_card.layout("card-1", object())  # type: ignore[arg-type]
    prev = _find_by_id(card, ids.card_picker_prev_id("card-1"))
    next_ = _find_by_id(card, ids.card_picker_next_id("card-1"))

    assert getattr(prev, "children") == "‹"
    assert getattr(next_, "children") == "›"
    assert _props(prev)["aria-label"] == "Previous image"
    assert _props(next_)["aria-label"] == "Next image"


def test_heatmap_picker_strip_renders_icon_image_navigation_buttons() -> None:
    strip = heatmap_layout._build_picker_strip(  # noqa: SLF001 - layout unit.
        [{"label": "Size_Area", "value": "Size_Area"}],
        [{"label": "plate_001.tif", "value": "plate_001.tif"}],
    )
    prev = _find_by_id(strip, heatmap_ids.HEATMAP_IMAGE_PREV_ID)
    next_ = _find_by_id(strip, heatmap_ids.HEATMAP_IMAGE_NEXT_ID)

    assert getattr(prev, "children") == "‹"
    assert getattr(next_, "children") == "›"
    assert _props(prev)["aria-label"] == "Previous image"
    assert _props(next_)["aria-label"] == "Next image"


def test_colony_layout_replaces_tile_size_slider_with_stepper() -> None:
    body = colony_layout.layout(object())  # type: ignore[arg-type]
    minus = _find_by_id(body, ids.COLONY_TILE_SIZE_MINUS)
    readout = _find_by_id(body, ids.COLONY_TILE_SIZE_READOUT)
    plus = _find_by_id(body, ids.COLONY_TILE_SIZE_PLUS)

    assert getattr(minus, "children") == "−"
    assert getattr(readout, "children") == "150 px"
    assert getattr(plus, "children") == "+"
    assert not any(isinstance(node, dcc.Slider) for node in _walk(body))


def test_shell_sidebar_folder_row_uses_folder_icon_not_chevron(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "plate1").mkdir()
    sandbox = SandboxRoot.from_path(sandbox_dir)
    tree = _sidebar.render_tree(sandbox, include_hidden=False, include_external=False)
    row = _find_by_id(tree, sidebar_entry_id("plate1"))
    row_text = "".join(str(text) for text in _walk(row) if isinstance(text, str))

    assert "📁" in row_text
    assert "‹" not in row_text
    assert "›" not in row_text
    assert _props(row)["aria-label"] == "Expand folder: plate1"
