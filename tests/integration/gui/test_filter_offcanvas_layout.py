"""Layout structure: the filter panel lives in a right offcanvas, tabs full-width."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filtered_state import FilteredMeasurements
from phenotypic.gui.results_viewer._layout import build_app_layout
from phenotypic.gui.results_viewer._output_root import OutputRoot


def _seed_output(tmp_path: Path) -> Path:
    """Write a minimal CLI output dir the OutputRoot can discover.

    ``OutputRoot.discover`` requires the
    ``<root>/results/<dataset>/overlays/<stem>.png`` layout, so seed the
    overlays alongside the master + mirror parquets.
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
    master.write_parquet(out / "master_measurements.parquet")
    master.write_parquet(out / "measurements.parquet")

    overlays = out / "results" / "ds1" / "overlays"
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
