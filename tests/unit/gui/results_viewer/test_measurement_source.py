"""What the picker offers, and what it does when a run measured nothing.

The state this module exists for is the **empty** one. An absent ``tables``
descriptor is normal -- a ``--mode process`` run never measures -- and the
viewer already retracted one inference that read it as "measurement pending"
(see ``_store_source.py``). The picker must be empty and the cards must
render as they did, with nothing anywhere claiming something is still coming.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phenotypic import Image
from phenotypic.gui.results_viewer._measurement_source import (
    displayable_measurement_columns,
    measurement_values_for,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE, OBJECT
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    PreparedEmbeddedMeasurementTable,
    zarr_store_path,
)

from tests._output_layout import write_master

DATASET = "d1"
MEASURED = "img-measured"
UNMEASURED = "img-unmeasured"


def _table() -> PreparedEmbeddedMeasurementTable:
    return PreparedEmbeddedMeasurementTable(
        frame=pd.DataFrame(
            {
                str(OBJECT.LABEL): [1, 2],
                "Shape_Area": [12.0, 512.0],
                "ColorLab_MedoidColorHex": ["#a08866", "#62605f"],
            }
        ),
        measurement_columns=(
            str(OBJECT.LABEL),
            "Shape_Area",
            "ColorLab_MedoidColorHex",
        ),
        join_status="not_requested",
        join_keys=(),
        metadata_snapshot_sha256="",
    )


@pytest.fixture(scope="module")
def run_template(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("measurement-source-run")
    write_master(
        root,
        pl.DataFrame(
            {
                "Metadata_Dataset": [DATASET] * 4,
                str(IMAGE.IMAGE_NAME): [
                    MEASURED,
                    MEASURED,
                    UNMEASURED,
                    UNMEASURED,
                ],
                "Object_Label": [1, 2, 1, 2],
                "Shape_Area": [12.0, 512.0, 1.0, 2.0],
                "ColorLab_MedoidColorHex": ["#a08866", "#62605f"] * 2,
                "Bbox_CenterRR": [16.0] * 4,
                "Bbox_CenterCC": [16.0] * 4,
            }
        ),
    )
    (root / "results" / DATASET / "measurements").mkdir(parents=True)
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    measured = zarr_store_path(root, DATASET, MEASURED)
    measured.parent.mkdir(parents=True, exist_ok=True)
    Image(arr=pixels).save2zarr(measured, measurement_table=_table())
    Image(arr=pixels).save2zarr(zarr_store_path(root, DATASET, UNMEASURED))
    return root


@pytest.fixture
def run(run_template: Path, tmp_path: Path) -> Path:
    root = tmp_path / "run"
    shutil.copytree(run_template, root, symlinks=True)
    return root


def _output_root(root: Path) -> OutputRoot:
    return OutputRoot.discover(
        root, cache_root=root.parent / ".test-phenotypic-viewer-cache"
    )


# ---------------------------------------------------------------------------
# What the picker offers
# ---------------------------------------------------------------------------


def test_the_picker_offers_the_stores_own_numeric_columns(run: Path) -> None:
    output_root = _output_root(run)
    columns = displayable_measurement_columns(
        output_root, [(DATASET, MEASURED)]
    )
    assert "Shape_Area" in columns
    # A declared column with no scale over it. Filtered out of the picker
    # rather than offered text-only -- the smaller of the two options the
    # spec left open.
    assert "ColorLab_MedoidColorHex" not in columns


def test_the_picker_is_sorted(run: Path) -> None:
    output_root = _output_root(run)
    columns = displayable_measurement_columns(
        output_root, [(DATASET, MEASURED)]
    )
    assert list(columns) == sorted(columns)


def test_a_store_with_no_tables_descriptor_offers_nothing(run: Path) -> None:
    """The empty picker is a normal state, not a pending one."""
    output_root = _output_root(run)
    assert (
        displayable_measurement_columns(output_root, [(DATASET, UNMEASURED)])
        == ()
    )


def test_one_unmeasured_image_does_not_empty_the_picker(run: Path) -> None:
    """A mixed run still offers what its measured stores declare."""
    output_root = _output_root(run)
    columns = displayable_measurement_columns(
        output_root, [(DATASET, UNMEASURED), (DATASET, MEASURED)]
    )
    assert "Shape_Area" in columns


def test_a_newer_store_schema_does_not_empty_the_picker(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unsupported store is skipped rather than failing the callback."""
    from phenotypic.gui.results_viewer import _measurement_source as source

    monkeypatch.setattr(
        source,
        "_columns_for_store",
        lambda *_args: (_ for _ in ()).throw(ValueError("newer schema")),
    )
    assert (
        displayable_measurement_columns(
            _output_root(run), [(DATASET, MEASURED)]
        )
        == ()
    )


# ---------------------------------------------------------------------------
# What the values look like
# ---------------------------------------------------------------------------


def test_values_are_keyed_by_dataset_image_and_object_label(run: Path) -> None:
    output_root = _output_root(run)
    values = measurement_values_for(
        output_root, [(DATASET, MEASURED)], "Shape_Area"
    )
    assert values == {
        (DATASET, MEASURED, 1): 12.0,
        (DATASET, MEASURED, 2): 512.0,
    }


def test_an_unmeasured_image_contributes_no_keys(run: Path) -> None:
    output_root = _output_root(run)
    values = measurement_values_for(
        output_root, [(DATASET, MEASURED), (DATASET, UNMEASURED)], "Shape_Area"
    )
    assert set(values) == {
        (DATASET, MEASURED, 1),
        (DATASET, MEASURED, 2),
    }


def test_duplicate_stems_in_two_datasets_keep_distinct_values(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dataset identity prevents a later store overwriting an earlier one."""
    from phenotypic.gui.results_viewer import _measurement_source as source

    identity = (1, 2, 3, 4)
    monkeypatch.setattr(
        source,
        "_store_and_identity",
        lambda _root, dataset, _stem: (Path(dataset), identity),
    )
    monkeypatch.setattr(
        source,
        "_column_for_store",
        lambda store, _identity, _column: ((1, 10.0 if store == "d1" else 20.0),),
    )
    values = measurement_values_for(
        _output_root(run), [("d1", "same"), ("d2", "same")], "Shape_Area"
    )
    assert values == {
        ("d1", "same", 1): 10.0,
        ("d2", "same", 1): 20.0,
    }


def test_a_column_no_store_declares_yields_nothing_and_does_not_raise(
    run: Path,
) -> None:
    output_root = _output_root(run)
    assert (
        measurement_values_for(
            output_root, [(DATASET, MEASURED)], "Shape_NoSuchThing"
        )
        == {}
    )


def test_a_rewritten_table_is_read_fresh(run: Path) -> None:
    """The cache keys on the payload's generation, not on its path.

    A stale read here would show the user the values from before a
    ``--mode recompile``, with nothing failing to say so.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_root = _output_root(run)
    first = measurement_values_for(
        output_root, [(DATASET, MEASURED)], "Shape_Area"
    )
    assert first[(DATASET, MEASURED, 1)] == 12.0

    payload = zarr_store_path(run, DATASET, MEASURED) / (
        MEASUREMENT_TABLE_RELATIVE_PATH
    )
    table = pq.read_table(payload)
    rewritten = table.set_column(
        table.column_names.index("Shape_Area"),
        "Shape_Area",
        pa.array([99.0, 512.0], type=pa.float64()),
    ).replace_schema_metadata(table.schema.metadata)
    pq.write_table(rewritten, payload)

    second = measurement_values_for(
        output_root, [(DATASET, MEASURED)], "Shape_Area"
    )
    assert second[(DATASET, MEASURED, 1)] == 99.0
