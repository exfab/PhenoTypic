"""Fixtures for the Results-viewer unit suite.

Hosts the six per-image store fixtures the Viv source contract is asserted
against -- ``rgb_store``, ``gray_only_store``, ``store_with_original``,
``label_less_store``, ``stage1_store`` and the ``store_at_extent`` factory.
Every one is written by the real writer, so a test can only agree with a
store the writer actually produces.

A ``live_viewer`` fixture used to live here, holding a full-run output root
plus a registered DZI tile blueprint. It went with the routes: the results
Plate reads store chunks in the browser now, so there is no server-rendered
pyramid and no DZI cache left to invalidate. The equivalent Flask-boundary
coverage for the byte route is ``test_zarr_routes.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phenotypic import Image
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE, OBJECT
from phenotypic.sdk_._measurement_tables import (
    PreparedImageTables,
)
from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON



@pytest.fixture()
def built_results_layout(tmp_path: Path):
    """The results viewer's top-level component tree over a minimal run.

    Mirrors ``_make_output`` in ``test_navigation_layout.py``: the layout
    builder only needs a discoverable output root with one measured image,
    so the fixture writes the master parquet and one overlay rather than a
    full store-backed run.
    """
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.gui.results_viewer._layout import build_app_layout
    from phenotypic.sdk_ import master_measurements_parquet_path

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = tmp_path / "deliverables" / "overlays" / "d1"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    target = master_measurements_parquet_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"],
            str(IMAGE.IMAGE_NAME): ["a"],
            "Size_Area": [1.0],
        }
    ).write_parquet(target)
    (overlay_dir / "a.png").touch()

    output_root = OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )
    state = CurationLabels.load(output_root.layout, output_root.clean_master_df)
    return build_app_layout(output_root, state)


# --------------------------------------------------------------------------
# Store fixtures for the Viv source contract.
#
# Every one is written by the REAL writer (``save2zarr`` /
# ``save_intermediate_zarr``), never by hand-editing a ``zarr.json``. A
# hand-built store would let these tests agree with a layout no writer
# produces -- which is the failure mode the whole "read it, do not infer it"
# rule exists to prevent.
# --------------------------------------------------------------------------


def _plate_array(extent: int) -> np.ndarray:
    """A landscape RGB array whose longest edge is *extent*.

    Low-entropy on purpose: the ladder assertions care about the recorded
    level count, and zstd over structured pixels writes a 4000x3000 store in
    ~2 s instead of the tens of seconds uniform noise costs.
    """
    height, width = extent * 3 // 4, extent
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = 40
    arr[: height // 2, : width // 2, 1] = 200
    arr[height // 2 :, width // 2 :, 2] = 120
    return arr


def _measurement_table() -> PreparedImageTables:
    """One embedded per-object measurement payload.

    Its presence is what writes ``attributes.phenotypic.tables``, which is
    the only reliable discriminator between "Stage 1 done, Stage 3 pending"
    and "finished, detector found nothing" -- see task 3.4.
    """
    return PreparedImageTables(
        measurements=pd.DataFrame(
            {str(OBJECT.LABEL): [1], "Size_Area": [16.0]}
        ),
        metadata=None,
        measurement_columns=("Size_Area",),
        join_status="not_requested",
        join_keys=(),
        metadata_snapshot_sha256="",
    )


def _level_shapes(store: Path, layer: str) -> list[tuple[int, ...]]:
    """Every pyramid level's shape, READ from the store's own metadata.

    Not computed by halving here: the point of each assertion built on this
    is that the store records the ladder, so re-deriving it in the test would
    make the test agree with itself rather than with the writer.

    Args:
        store: Path to a promoted ``*.ome.zarr`` directory.
        layer: A series name (``"rgb"``) or a resolved label path.

    Returns:
        One shape tuple per level, level 0 first.
    """
    root = json.loads((store / STORE_ROOT_JSON).read_text(encoding="utf-8"))
    block = root["attributes"][PhenotypicAttr.ROOT]
    member = block[PhenotypicAttr.SERIES].get(layer, layer)
    levels = int(block[PhenotypicAttr.PYRAMID]["levels"])
    shapes = []
    for level in range(levels):
        meta = store / member / str(level) / STORE_ROOT_JSON
        shapes.append(
            tuple(json.loads(meta.read_text(encoding="utf-8"))["shape"])
        )
    return shapes


@pytest.fixture(scope="session")
def store_at_extent(tmp_path_factory) -> Callable[[int], Path]:
    """Factory writing a real store whose longest edge is a given extent.

    Session-scoped and memoized: the 4000 px store is the subject of several
    tests and costs ~2 s to write, so writing it once is the difference
    between a fast module and a slow one.

    Returns:
        ``extent -> Path`` of a promoted store written by ``save2zarr``.
    """
    base = tmp_path_factory.mktemp("stores_by_extent")
    written: dict[int, Path] = {}

    def _write(extent: int) -> Path:
        if extent not in written:
            image = Image(arr=_plate_array(extent))
            written[extent] = image.save2zarr(
                base / f"extent-{extent}.ome.zarr"
            )
        return written[extent]

    return _write


@pytest.fixture()
def rgb_store(tmp_path: Path) -> Path:
    """An ordinary finished run store: ``rgb`` primary, measured."""
    image = Image(arr=_plate_array(128))
    return image.save2zarr(
        tmp_path / "plate.ome.zarr", measurement_table=_measurement_table()
    )


@pytest.fixture()
def gray_only_store(tmp_path: Path) -> Path:
    """A store with no RGB layer, so ``gray`` is the primary series.

    ``_series_names`` omits ``rgb`` entirely when it is empty, which moves
    both the primary series and the objmap label under ``gray`` -- the case
    that proves nothing hard-codes ``rgb/labels/objmap``.
    """
    image = Image(arr=_plate_array(128)[:, :, 0])
    return image.save2zarr(
        tmp_path / "grayplate.ome.zarr",
        measurement_table=_measurement_table(),
    )


@pytest.fixture()
def store_with_original(tmp_path: Path) -> Path:
    """A store whose series list exceeds the three canonical names.

    ``_write_store_part`` appends ``"original"`` whenever the image carries
    a retained original, exactly as ``_cli_process_single`` arranges before
    processing begins.
    """
    image = Image(arr=_plate_array(128))
    image._retain_original()
    return image.save2zarr(tmp_path / "orig.ome.zarr")


@pytest.fixture()
def label_less_store(tmp_path: Path) -> Path:
    """A preview store with no label image at all.

    ``save_intermediate_zarr`` sets ``write_objmap = "objmap" in layers``, so
    the ``labels`` key is omitted from the block ENTIRELY rather than emitted
    empty. Most builder-preview stores are like this.
    """
    image = Image(arr=_plate_array(128))
    return image.save_intermediate_zarr(
        tmp_path / "prev.ome.zarr", layers=("gray",)
    )


@pytest.fixture()
def stage1_store(tmp_path: Path) -> Path:
    """A store as Stage 1 leaves it: zeros objmap, no measurements table.

    The landed staged engine keeps Stage 2 read-only, so between Stage 1 and
    Stage 3 the in-store objmap is all zeros and no ``tables`` descriptor
    exists. Both facts are properties of a plain ``save2zarr`` on an
    unmeasured image, which is what Stage 1 performs.
    """
    image = Image(arr=_plate_array(128))
    return image.save2zarr(tmp_path / "stage1plate.ome.zarr")
