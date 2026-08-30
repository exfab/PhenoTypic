"""Embedded per-image measurement-table storage contracts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from phenotypic import Image
from phenotypic._cli._cli_output_manager import (
    OutputManager,
    prepare_embedded_measurement_table,
)
from phenotypic.schema import OBJECT
from tests._ngff_conformance import assert_store_conforms

from phenotypic.sdk_ import (
    EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    PhenotypicAttr,
    read_phenotypic_attributes,
)


def _image() -> Image:
    image = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="plate")
    image.objmap[:] = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 2, 2, 0],
            [0, 1, 1, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    return image


def _manager(output_dir: Path) -> OutputManager:
    return OutputManager.from_config(
        output_dir,
        ext=".tiff",
        include_dataset_column=True,
        save_overlays=False,
    )


def test_final_store_embeds_an_interoperable_measurement_table(
    tmp_path: Path,
) -> None:
    """Dropping the table payload, descriptor, or Parquet metadata must fail."""
    measurements = pd.DataFrame(
        {
            str(OBJECT.LABEL): [1, 2],
            "Shape_Area": [4.0, 4.0],
            "Metadata_ImageName": ["plate.tif", "plate.tif"],
        }
    )
    store = _manager(tmp_path).save_image_store(
        _image(),
        "dataset-a",
        "plate",
        work_id="work-1",
        measurements=measurements,
    )

    assert store is not None
    assert_store_conforms(store)
    table_path = store / MEASUREMENT_TABLE_RELATIVE_PATH
    assert table_path.is_file()
    assert (
        json.loads((store / "tables" / "zarr.json").read_text())["node_type"]
        == "group"
    )
    assert (
        json.loads(
            (store / "tables" / "measurements" / "zarr.json").read_text()
        )["node_type"]
        == "group"
    )

    descriptor = read_phenotypic_attributes(store)[PhenotypicAttr.TABLES][
        "measurements"
    ]
    assert descriptor == {
        "schema_version": 1,
        "type": "object_measurements",
        "format": "parquet",
        "path": "tables/measurements/table.parquet",
        "measurement_columns": [
            "Object_Label",
            "Shape_Area",
            "Metadata_ImageName",
            "Metadata_Dataset",
        ],
        "target": {
            "column": "Object_Label",
            "path": "rgb/labels/objmap",
        },
    }

    arrow_table = pq.read_table(table_path)
    assert arrow_table.column_names == descriptor["measurement_columns"]
    metadata = {
        key.decode(): value.decode()
        for key, value in (arrow_table.schema.metadata or {}).items()
    }
    keys = EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS
    assert metadata[keys.JOIN_STATUS] == "not_requested"
    assert metadata[keys.JOIN_KIND] == "right"
    assert metadata[keys.JOIN_LEFT] == "metadata"
    assert metadata[keys.JOIN_RIGHT] == "measurements"
    assert json.loads(metadata[keys.JOIN_KEYS]) == []
    assert metadata[keys.METADATA_SNAPSHOT_SHA256] == ""
    assert (
        json.loads(metadata[keys.MEASUREMENT_COLUMNS])
        == descriptor["measurement_columns"]
    )

    duckdb_rows = (
        duckdb.connect()
        .execute(
            "SELECT Object_Label, Shape_Area FROM read_parquet(?) ORDER BY Object_Label",
            [str(table_path)],
        )
        .fetchall()
    )
    assert duckdb_rows == [(1, 4.0), (2, 4.0)]


def test_metadata_is_right_joined_before_embedding(tmp_path: Path) -> None:
    """A left join would leak metadata-only rows or drop measured-only rows."""
    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text(
        "Object_Label,Metadata_Strain\n1,WT\n3,metadata-only\n",
        encoding="utf-8",
    )
    baseline = pd.DataFrame(
        {
            "Object_Label": [1, 2],
            "Shape_Area": [8.0, 9.0],
        }
    )

    prepared = prepare_embedded_measurement_table(baseline, metadata_csv)

    assert prepared.measurement_columns == ("Object_Label", "Shape_Area")
    assert prepared.join_status == "joined"
    assert prepared.join_keys == ("Object_Label",)
    assert (
        prepared.metadata_snapshot_sha256
        == hashlib.sha256(metadata_csv.read_bytes()).hexdigest()
    )
    assert prepared.frame["Object_Label"].tolist() == [1, 2]
    assert prepared.frame.loc[0, "Metadata_Strain"] == "WT"
    assert pd.isna(prepared.frame.loc[1, "Metadata_Strain"])
    assert "metadata-only" not in prepared.frame["Metadata_Strain"].tolist()


def test_duplicate_metadata_keys_fan_out_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Deduplicating metadata would silently discard requested fan-out."""
    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text(
        "Object_Label,Metadata_Strain\n1,WT-a\n1,WT-b\n",
        encoding="utf-8",
    )
    baseline = pd.DataFrame({"Object_Label": [1], "Shape_Area": [8.0]})

    with caplog.at_level(logging.WARNING):
        prepared = prepare_embedded_measurement_table(baseline, metadata_csv)

    assert prepared.frame["Metadata_Strain"].tolist() == ["WT-a", "WT-b"]
    assert "duplicate keys" in caplog.text


def test_no_common_metadata_keys_keeps_measurements_unchanged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Inventing a join key must not discard or expand measured rows."""
    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text("Metadata_Strain\nWT\n", encoding="utf-8")
    baseline = pd.DataFrame({"Object_Label": [1], "Shape_Area": [8.0]})

    with caplog.at_level(logging.WARNING):
        prepared = prepare_embedded_measurement_table(baseline, metadata_csv)

    pd.testing.assert_frame_equal(prepared.frame, baseline)
    assert prepared.join_status == "no_common_keys"
    assert prepared.join_keys == ()
    assert "no columns in common" in caplog.text


def test_zero_object_table_is_valid_and_keeps_its_ordered_schema(
    tmp_path: Path,
) -> None:
    """Zero rows must not erase the descriptor's baseline schema."""
    measurements = pd.DataFrame(
        {
            "Object_Label": pd.Series(dtype="int64"),
            "Shape_Area": pd.Series(dtype="float64"),
        }
    )
    store = _manager(tmp_path).save_image_store(
        _image(),
        "dataset-a",
        "empty",
        measurements=measurements,
    )

    assert store is not None
    descriptor = read_phenotypic_attributes(store)[PhenotypicAttr.TABLES][
        "measurements"
    ]
    assert descriptor["measurement_columns"] == [
        "Object_Label",
        "Shape_Area",
        "Metadata_Dataset",
    ]
    table = pq.read_table(store / MEASUREMENT_TABLE_RELATIVE_PATH)
    assert table.num_rows == 0
    assert table.column_names == descriptor["measurement_columns"]
