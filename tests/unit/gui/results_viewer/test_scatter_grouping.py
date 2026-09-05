"""Columns group by the measurer that emits them, from the run's own config."""

from __future__ import annotations

from phenotypic.gui.results_viewer._scatter_tab._grouping import group_columns

MEAS = {
    "MeasureShape": {"class": "MeasureShape", "params": {}},
    "MeasureIntensity": {"class": "MeasureIntensity", "params": {}},
    "MeasureColor": {
        "class": "MeasureColor",
        "params": {"include_XYZ": False, "include_xy": False},
    },
    "MeasureTexture": {"class": "MeasureTexture", "params": {"scale": [5]}},
    "MeasureNeighborDist": {"class": "MeasureNeighborDist", "params": {}},
}


def test_exact_headers_group_by_measurer() -> None:
    """Also asserts the NEGATIVE, or a total-failure bug passes.

    If the implementation calls ``get_measurement_infoclasses`` on the class
    rather than an instance it raises, every measurer is skipped, and every
    column lands in Unattributed. Asserting only "Shape_Area is in
    MeasureShape" would KeyError, but asserting membership without
    asserting absence let three sibling tests pass against exactly that bug.
    """
    groups = group_columns(["Shape_Area", "Intensity_MeanIntensity"], MEAS)
    assert "Shape_Area" in groups["MeasureShape"]
    assert "Intensity_MeanIntensity" in groups["MeasureIntensity"]
    assert "Unattributed" not in groups, (
        "no column here is unclaimed; an Unattributed group means the "
        "measurers were never successfully constructed"
    )


def test_parameterized_schemas_fall_back_to_category() -> None:
    """TEXTURE.get_headers requires a `scale` argument.

    Naively this raises TypeError and dumps every Texture_ column into
    Unattributed -- 65 of 148 columns on the verification fixture.
    """
    groups = group_columns(["Texture_Contrast-deg000-scale05"], MEAS)
    assert groups["MeasureTexture"] == ["Texture_Contrast-deg000-scale05"]


def test_measurer_params_change_the_claimed_headers() -> None:
    """The same column must flip groups when the run's params change.

    Asserting only the "off" case passes against an implementation that
    claims NOTHING -- everything is Unattributed, so the column is
    correctly absent from MeasureColor for entirely the wrong reason. The
    "on" case is what makes this discriminate.

    The column is read off ``ColorXYZ`` itself rather than written out
    from the category name. An earlier draft used ``ColorXYZ_X``, which
    no schema emits, so the "on" half failed for a reason unrelated to
    what it tests -- the third fictional-header defect on this branch.
    """
    from phenotypic.schema import ColorXYZ

    column = ColorXYZ.get_headers()[0]

    off = group_columns([column], MEAS)
    assert column not in off.get("MeasureColor", [])
    assert column in off["Unattributed"]

    on_cfg = dict(MEAS)
    on_cfg["MeasureColor"] = {
        "class": "MeasureColor",
        "params": {"include_XYZ": True, "include_xy": True},
    }
    on = group_columns([column], on_cfg)
    assert column in on["MeasureColor"], (
        "with include_XYZ=True the column must be claimed -- if it is still "
        "Unattributed the measurer is not being constructed from its params"
    )


def test_metadata_is_one_flat_group_and_curation_is_its_own() -> None:
    groups = group_columns(
        ["Metadata_Strain", "Metadata_PlateID", "QC_MetadataOnly"], MEAS
    )
    assert set(groups["Metadata"]) == {"Metadata_Strain", "Metadata_PlateID"}
    assert groups["Curation"] == ["QC_MetadataOnly"]


def test_unclaimed_columns_land_in_unattributed() -> None:
    """Mixes claimed and unclaimed, so "everything is unattributed" fails."""
    groups = group_columns(
        ["Object_Label", "Bbox_CenterRR", "Grid_RowNum", "Shape_Area"], MEAS
    )
    assert set(groups["Unattributed"]) == {
        "Object_Label",
        "Bbox_CenterRR",
        "Grid_RowNum",
    }
    assert groups["MeasureShape"] == ["Shape_Area"]
