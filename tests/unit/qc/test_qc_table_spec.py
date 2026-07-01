"""Tests for the self-describing QualityCheck table contract."""

import pandas as pd

from phenotypic.analysis.qc import MaxModifiedZScore
from phenotypic.schema import METADATA


def _frame():
    return pd.DataFrame(
        {
            str(METADATA.IMAGE_NAME): ["a.png", "a.png", "b.png", "b.png"],
            "Object_Label": [1, 2, 1, 2],
            "Plate": ["P1", "P1", "P1", "P1"],
            "Metadata_Time": [0, 0, 1, 1],
            "Size_Area": [10.0, 11.0, 100.0, 9.0],
        }
    )


def test_to_table_carries_check_specific_columns():
    chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
    chk.analyze(_frame())
    table = chk.to_table()
    # Member-level: one row per input object, with the check's QC columns.
    assert len(table) == 4
    assert "QC_ZMax_Metric" in table.columns
    assert "QC_ZMax_Status" in table.columns
    assert "QC_ZMax_Median" in table.columns  # check-specific extra (kept!)
    assert {str(METADATA.IMAGE_NAME), "Object_Label", "Plate"} <= set(table.columns)


def test_table_spec_describes_roles():
    chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
    chk.analyze(_frame())
    spec = chk.table_spec("qc-ZMax-deadbeef")
    assert spec.instance_id == "qc-ZMax-deadbeef"
    assert spec.cls_name == "MaxModifiedZScore"
    assert spec.groupby_cols == ["Plate"]
    assert spec.metric_col == "QC_ZMax_Metric"
    assert spec.status_col == "QC_ZMax_Status"
    assert spec.supports_object_curation is True
    assert spec.member_key_cols == [str(METADATA.IMAGE_NAME), "Object_Label"]
    assert spec.time_col == "Metadata_Time"   # ZMax declares a time_label field
    assert spec.higher_is_bad is True
    assert "QC_ZMax_Median" in spec.extra_cols


def test_grid_occupancy_is_group_level_and_diagnostic_only():
    import pandas as pd

    from phenotypic.analysis.qc import GridOccupancy

    metadata = pd.DataFrame(
        {str(METADATA.IMAGE_NAME): ["a.png"] * 4, "cell_label": [1, 2, 3, 4]}
    )
    measured = pd.DataFrame(
        {
            str(METADATA.IMAGE_NAME): ["a.png", "a.png"],
            "Object_Label": [1, 2],
            "cell_label": [1, 2],
        }
    )
    # cell_label defaults to "Grid_RowMajorIdx"; point it at this frame's column.
    chk = GridOccupancy(
        metadata=metadata, groupby=[str(METADATA.IMAGE_NAME)], cell_label="cell_label"
    )
    chk.analyze(measured)
    spec = chk.table_spec("qc-Occupancy-cafef00d")
    assert spec.supports_object_curation is False

    table = chk.to_table()
    # Group-level: one row per group (here one image), not per colony.
    assert len(table) == 1
    assert {"QC_Occupancy_Filled", "QC_Occupancy_Expected"} <= set(table.columns)
