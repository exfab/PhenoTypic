"""Layout structure: the filter panel lives in a right offcanvas, tabs full-width."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pytest
from PIL import Image as PILImage

import phenotypic.gui.results_viewer as results_viewer_pkg
from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filtered_state import FilteredMeasurements
from phenotypic.gui.results_viewer._layout import build_app_layout
from phenotypic.gui.results_viewer._output_root import OutputRoot
from tests._output_layout import write_master, write_measurements_mirror


def _seed_output(tmp_path: Path) -> Path:
    """Write a minimal CLI output dir the OutputRoot can discover.

    ``OutputRoot.discover`` requires the
    ``<root>/deliverables/master_measurements.parquet`` layout plus a
    ``<root>/results/<dataset>/`` directory and overlays under
    ``<root>/deliverables/overlays/<dataset>/``.
    """
    out = tmp_path / "results" / "Example"
    out.mkdir(parents=True)
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds1", "ds1"],
            "Metadata_ImageFile": ["a.tif", "b.tif"],
            "Object_Label": [1, 2],
            "Size_Area": [100.0, 200.0],
        }
    )
    write_master(out, master)
    write_measurements_mirror(out, master)

    (out / "results" / "ds1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = out / "deliverables" / "overlays" / "ds1"
    overlays.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        PILImage.new("RGB", (64, 64), (200, 0, 0)).save(overlays / f"{stem}.png")
    return out


@pytest.fixture
def output_root(tmp_path: Path) -> OutputRoot:
    return OutputRoot.discover(_seed_output(tmp_path))


def _walk(component: Any) -> Iterator[Any]:
    """Yield a component and all of its descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


def _ids_in(component: Any) -> set[Any]:
    out: set[Any] = set()
    for node in _walk(component):
        node_id = getattr(node, "id", None)
        if node_id is not None:
            out.add(node_id if isinstance(node_id, str) else str(node_id))
    return out


def _viewer_css() -> str:
    """Return the packaged results-viewer stylesheet."""
    css_path = Path(results_viewer_pkg.__file__).parent / "_assets" / "results_viewer.css"
    return css_path.read_text(encoding="utf-8")


def test_offcanvas_hosts_filter_panel_and_toggle_present(
    output_root: OutputRoot,
) -> None:
    filtered = FilteredMeasurements.load(Path(output_root.root), output_root.master_df)
    layout = build_app_layout(output_root, filtered)

    # The offcanvas exists and contains the filter panel's row container + add button.
    offcanvas = next(
        n for n in _walk(layout) if getattr(n, "id", None) == ids.OFFCANVAS_FILTER_ID
    )
    assert offcanvas._type == "Offcanvas"
    assert getattr(offcanvas, "placement", None) == "end"
    assert getattr(offcanvas, "is_open", None) is False
    inner = _ids_in(offcanvas)
    assert ids.FILTER_ROWS_CONTAINER_ID in inner
    assert ids.BTN_ADD_FILTER_ROW in inner
    assert ids.FILTER_MATCH_COUNT_ID in inner

    # The top-bar toggle + badge are present in the overall tree.
    all_ids = _ids_in(layout)
    assert ids.BTN_FILTERS_TOGGLE in all_ids
    assert ids.FILTER_TOGGLE_BADGE_ID in all_ids


def test_body_has_no_lg_sidebar_columns(output_root: OutputRoot) -> None:
    filtered = FilteredMeasurements.load(Path(output_root.root), output_root.master_df)
    layout = build_app_layout(output_root, filtered)
    # No dbc.Col with an lg=3/lg=9 split should remain (full-width content).
    for node in _walk(layout):
        if getattr(node, "_type", None) == "Col":
            assert getattr(node, "lg", None) not in (3, 9)


def test_filter_toggle_keeps_stable_control_height() -> None:
    """The zero-height sticky action strip must not shrink the Filters button."""
    css = _viewer_css()
    assert "align-items: flex-start;" in css
    assert "#btn-filters-toggle {" in css
    assert "min-height: 31px;" in css
    assert "line-height: 1.5;" in css
