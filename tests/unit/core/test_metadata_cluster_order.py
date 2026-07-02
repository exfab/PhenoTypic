"""Cluster-ordering: the shared order_measurement_columns helper + mirror wiring."""

from __future__ import annotations


def test_order_measurement_columns_full_contract():
    from phenotypic.sdk_ import order_measurement_columns

    # Deliberately shuffled input spanning every partition.
    cols = [
        "Grid_RowNum",                 # info
        "MetadataImage_ImageName",     # framework image (trailing)
        "Shape_Area",                  # measurement
        "MetadataGenetic_Strain",      # front metadata (Strain cluster)
        "Object_Label",                # info (leads info block by name, not position)
        "MetadataSample_SampleID",     # front metadata (Identity cluster, leads)
        "MetadataCondition_Media",     # front metadata (Condition cluster)
        "Metadata_UnknownTag",         # uncategorized user metadata -> end of front
    ]

    ordered = order_measurement_columns(cols)

    assert ordered == [
        # front metadata: Identity (Sample) -> Strain -> Condition -> uncategorized
        "MetadataSample_SampleID",
        "MetadataGenetic_Strain",
        "MetadataCondition_Media",
        "Metadata_UnknownTag",
        # measurements
        "Shape_Area",
        # framework image block
        "MetadataImage_ImageName",
        # per-object info block
        "Object_Label",
        "Grid_RowNum",
    ]


def test_order_measurement_columns_no_metadata():
    from phenotypic.sdk_ import order_measurement_columns

    cols = ["Shape_Area", "Object_Label", "Bbox_MinRR", "Intensity_MeanIntensity"]
    ordered = order_measurement_columns(cols)
    # Measurements keep relative order; info block trails.
    assert ordered == [
        "Shape_Area",
        "Intensity_MeanIntensity",
        "Object_Label",
        "Bbox_MinRR",
    ]


def test_info_block_prefix_is_collision_free():
    """Only GRID/BBOX emit Grid_/Bbox_ headers; GridLinReg_/GridSpread_ must not."""
    import phenotypic.schema as schema
    from phenotypic.schema import MeasurementInfo

    for name in schema.__all__:
        obj = getattr(schema, name)
        if not (isinstance(obj, type) and issubclass(obj, MeasurementInfo)
                and obj is not MeasurementInfo and list(obj)):
            continue
        if obj.category() in {"Grid", "Bbox"}:
            continue
        for member in obj:
            assert not member.value.startswith("Grid_"), member.value
            assert not member.value.startswith("Bbox_"), member.value


def test_insert_metadata_front_block_cluster_order():
    """insert_metadata places user tags in cluster order (Sample/Identity before Strain)."""
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.sdk_ import is_metadata_header
    import pandas as pd

    img = load_synth_yeast_plate()
    # Set tags out of cluster order on purpose.
    img.metadata["Strain"] = "BY4741"        # MetadataGenetic_ (Strain cluster)
    img.metadata["SampleID"] = "S1"          # MetadataSample_ (Identity cluster, leads)
    img.metadata["Media"] = "YPD"            # MetadataCondition_ (Condition cluster)

    df = img.metadata.insert_metadata(pd.DataFrame({"Object_Label": [1]}))
    meta_cols = [c for c in df.columns if is_metadata_header(c)]

    # Identity (Sample) precedes Strain precedes Condition.
    assert meta_cols.index("MetadataSample_SampleID") < meta_cols.index(
        "MetadataGenetic_Strain"
    )
    assert meta_cols.index("MetadataGenetic_Strain") < meta_cols.index(
        "MetadataCondition_Media"
    )
