"""Timeline tab body: ids present + empty-state predicate wiring (pure)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
from PIL import Image as PILImage

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view import _ids
from phenotypic.gui.results_viewer.timeline_view._layout import (
    build_timeline_grid_component,
    layout,
)
from tests._output_layout import write_master, write_measurements_mirror
from phenotypic.schema import IMAGE


def _walk(component):
    stack = [component]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)


def _walk_ids(component) -> set[str]:
    return {
        cid
        for node in _walk(component)
        if isinstance((cid := getattr(node, "id", None)), str)
    }


def _walk_classnames(component) -> set[str]:
    classes: set[str] = set()
    for node in _walk(component):
        cls = getattr(node, "className", None)
        if isinstance(cls, str):
            classes.update(cls.split())
    return classes


def _root(tmp_path: Path, *, with_time: bool) -> OutputRoot:
    cli_out = tmp_path / "out"
    cols = {
        "Metadata_Dataset": ["ds", "ds"],
        str(IMAGE.IMAGE_NAME): ["a", "b"],
        "Metadata_PlateNum": ["1", "2"],
        "Object_Label": [1, 2],
        "Size_Area": [1.0, 2.0],
    }
    if with_time:
        cols["Metadata_ImageNumber"] = pl.Series([1, 2], dtype=pl.Int64)
    df = pl.DataFrame(cols)
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    (cli_out / "results" / "ds" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = cli_out / "deliverables" / "overlays" / "ds"
    overlays.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        PILImage.new("RGB", (40, 30), (1, 2, 3)).save(overlays / f"{stem}.png")
    return OutputRoot.discover(
        cli_out,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )


def test_layout_mounts_all_focus_navigate_chrome(tmp_path: Path) -> None:
    ids = _walk_ids(layout(_root(tmp_path, with_time=True)))
    for cid in (
        _ids.TIMELINE_GRID,
        _ids.TIMELINE_Y_DROPDOWN,
        _ids.TIMELINE_X_DROPDOWN,
        _ids.TIMELINE_TILE_SIZE_READOUT,
        _ids.TIMELINE_NAV_UP,
        _ids.TIMELINE_NAV_DOWN,
        _ids.TIMELINE_NAV_LEFT,
        _ids.TIMELINE_NAV_RIGHT,
        _ids.TIMELINE_POSITION,
        _ids.TIMELINE_POPOUT_MODAL,
    ):
        assert cid in ids


def test_empty_state_when_no_time_column(tmp_path: Path) -> None:
    root = _root(tmp_path, with_time=False)
    _component, show_empty, _n = build_timeline_grid_component(
        root, root.master_df, row_col="Metadata_PlateNum", time_col=None, tile_size=150
    )
    assert show_empty is True


def test_grid_renders_when_time_column_present(tmp_path: Path) -> None:
    root = _root(tmp_path, with_time=True)
    _component, show_empty, n = build_timeline_grid_component(
        root,
        root.master_df,
        row_col="Metadata_PlateNum",
        time_col="Metadata_ImageNumber",
        tile_size=150,
    )
    assert show_empty is False
    assert n == 2  # two distinct image numbers


def test_controller_required_classes_present(tmp_path: Path) -> None:
    # C2: timeline.js is surface-agnostic and finds controls BY CLASS scoped to
    # .timeline-body. The Results layout must carry every controller-required
    # class or the controller silently finds nothing (the e2e is not the only net).
    classes = _walk_classnames(layout(_root(tmp_path, with_time=True)))
    required = {
        "timeline-body",
        "timeline-viewport",
        "timeline-grid-container",
        "timeline-nav-up",
        "timeline-nav-down",
        "timeline-nav-left",
        "timeline-nav-right",
        "timeline-position",
        "timeline-popout-bridge",
    }
    missing = required - classes
    assert not missing, f"layout is missing controller classes: {sorted(missing)}"


def test_one_column_matrix_when_filtered_to_single_time(tmp_path: Path) -> None:
    # S3: filtering down to a single image-number must still render a sensible
    # 1-column matrix (not an empty/degenerate grid).
    root = _root(tmp_path, with_time=True)
    single = root.master_df.filter(pl.col("Metadata_ImageNumber") == 1)
    _component, show_empty, n = build_timeline_grid_component(
        root,
        single,
        row_col="Metadata_PlateNum",
        time_col="Metadata_ImageNumber",
        tile_size=150,
    )
    assert show_empty is False
    assert n == 1  # one time column survives the filter


def test_representative_reaching_builders_is_the_raw_tuple(tmp_path: Path) -> None:
    # C3: build_matrix keeps the representative as the raw (dataset, stem) TUPLE
    # (min by str(...), object stored), so ref_builder=lambda r: encode_cell_ref(*r)
    # and the url_builder both receive a tuple — not a stringified pair. Assert the
    # emitted data-ref/data-src reflect encode_cell_ref(dataset, stem), proving the
    # representative was unpackable.
    root = _root(tmp_path, with_time=True)
    component, _show_empty, _n = build_timeline_grid_component(
        root,
        root.master_df,
        row_col="Metadata_PlateNum",
        time_col="Metadata_ImageNumber",
        tile_size=150,
    )
    refs: set[str] = set()
    for node in _walk(component):
        to_json = getattr(node, "to_plotly_json", None)
        if to_json is None:
            continue
        props = to_json().get("props", {})
        ref = props.get("data-ref")
        if isinstance(ref, str):
            refs.add(ref)
    # At least one populated cell's data-ref is encode_cell_ref("ds", stem) ==
    # "ds/<stem>" — proving build_matrix kept the representative as the raw
    # (dataset, stem) tuple that ref_builder/url_builder unpacked.
    assert any(r.startswith("ds/") for r in refs)
