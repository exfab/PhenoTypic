"""Layout tests for Results Viewer icon navigation controls."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def _find_plate_card_stepper_group(component: object) -> object:
    expected_ids = [
        ids.card_picker_prev_id("card-1"),
        ids.card_picker_next_id("card-1"),
    ]
    for node in _walk(component):
        if "btn-group" not in str(getattr(node, "className", "")):
            continue
        children = getattr(node, "children", None)
        if not isinstance(children, (list, tuple)):
            continue
        child_ids = [getattr(child, "id", None) for child in children]
        if child_ids == expected_ids:
            return node
    raise AssertionError("missing plate-card image stepper button group")


def test_plate_card_renders_icon_image_navigation_buttons() -> None:
    card = _viewer_card.layout("card-1", object())  # type: ignore[arg-type]
    prev = _find_by_id(card, ids.card_picker_prev_id("card-1"))
    next_ = _find_by_id(card, ids.card_picker_next_id("card-1"))

    assert getattr(prev, "children") == "‹"
    assert getattr(next_, "children") == "›"
    assert _props(prev)["aria-label"] == "Previous image"
    assert _props(next_)["aria-label"] == "Next image"


def test_plate_card_uses_browse_stepper_button_schema() -> None:
    card = _viewer_card.layout("card-1", object())  # type: ignore[arg-type]
    group = _find_plate_card_stepper_group(card)
    prev = _find_by_id(card, ids.card_picker_prev_id("card-1"))
    next_ = _find_by_id(card, ids.card_picker_next_id("card-1"))

    assert getattr(group, "className") == "btn-group"
    assert _props(group)["role"] == "group"
    assert _props(group)["aria-label"] == "Step through images"
    assert "browse-step-button" in getattr(prev, "className", "")
    assert "browse-step-button" in getattr(next_, "className", "")
    assert "card-picker-nav-btn" in getattr(prev, "className", "")
    assert "card-picker-nav-btn" in getattr(next_, "className", "")

    css = Path("src/phenotypic/gui/results_viewer/_assets/results_viewer.css")
    css_text = css.read_text(encoding="utf-8")
    assert ".browse-step-button" in css_text
    assert "min-width: 3rem;" in css_text


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
    # has_results=False → no per-image HDFs, so build_layer_toggle returns None
    # and the colony layout under test (tile-size stepper) builds without it.
    body = colony_layout.layout(SimpleNamespace(has_results=False))  # type: ignore[arg-type]
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


# ---------------------------------------------------------------------------
# Sticky tab-row Filters button placement (Feature A, Task 9)
# ---------------------------------------------------------------------------

import polars as pl

from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._layout import _build_header, build_app_layout
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import master_measurements_parquet_path
from phenotypic.schema import METADATA


def _iter(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _iter(child)


def _ids_in(component) -> set:
    return {getattr(c, "id", None) for c in _iter(component)}


def _make_output(tmp_path: Path) -> OutputRoot:
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = tmp_path / "deliverables" / "overlays" / "d1"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {"MetadataExperiment_Dataset": ["d1"], str(METADATA.IMAGE_NAME): ["a"], "Size_Area": [1.0]}
    )
    target = master_measurements_parquet_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)
    (overlay_dir / "a.png").touch()
    return OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )


def test_header_no_longer_contains_filters_toggle(tmp_path) -> None:
    out = _make_output(tmp_path)
    header = _build_header(out)
    assert ids.BTN_FILTERS_TOGGLE not in _ids_in(header)


def test_app_layout_keeps_filters_toggle_near_tabs(tmp_path) -> None:
    out = _make_output(tmp_path)
    state = CurationLabels.load(out.layout, out.clean_master_df)
    layout = build_app_layout(out, state)
    all_ids = _ids_in(layout)
    assert ids.BTN_FILTERS_TOGGLE in all_ids
    assert ids.FILTER_TOGGLE_BADGE_ID in all_ids
    assert ids.TABS_ID in all_ids
