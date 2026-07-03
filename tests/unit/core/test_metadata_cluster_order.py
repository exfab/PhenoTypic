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


def test_order_measurement_columns_multiple_uncategorized_sort_last_alpha():
    """Several unknown Metadata_* tags all trail known metadata, alpha among themselves."""
    from phenotypic.sdk_ import order_measurement_columns

    cols = [
        "Metadata_Zebra",           # uncategorized
        "MetadataGenetic_Strain",   # known front metadata
        "Metadata_Apple",           # uncategorized
        "Shape_Area",               # measurement
        "Object_Label",             # info
    ]
    ordered = order_measurement_columns(cols)
    assert ordered == [
        "MetadataGenetic_Strain",   # known metadata leads the front block
        "Metadata_Apple",           # unknowns trail known, alpha-sorted
        "Metadata_Zebra",
        "Shape_Area",
        "Object_Label",
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


def test_finalize_mirror_applies_cluster_order(tmp_path):
    """The polars mirror frame from finalize is canonical-ordered, even after a
    --metadata join that lands external columns front-in-CSV-order."""
    import polars as pl
    from phenotypic._cli._cli_output_manager import finalize_post_master_outputs

    # Clean master (metadata-free) with a join anchor column present in both frames.
    master = pl.DataFrame(
        {
            "MetadataImage_ImageName": ["plateA"],
            "Object_Label": [1],
            "Grid_RowNum": [1],
            "Shape_Area": [123.0],
        }
    )
    # External metadata CSV with columns in NON-canonical order.
    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text(
        "MetadataImage_ImageName,MetadataCondition_Media,MetadataGenetic_Strain,MetadataSample_SampleID\n"
        "plateA,YPD,BY4741,S1\n"
    )

    out_dir = tmp_path / "run"
    out_dir.mkdir()

    post_df = finalize_post_master_outputs(
        out_dir, master, pipeline=None, metadata_csv=meta_csv, no_qc=True
    )

    assert post_df.columns == [
        # front metadata: Identity(Sample) -> Strain -> Condition
        "MetadataSample_SampleID",
        "MetadataGenetic_Strain",
        "MetadataCondition_Media",
        # measurements
        "Shape_Area",
        # framework image block
        "MetadataImage_ImageName",
        # per-object info block
        "Object_Label",
        "Grid_RowNum",
    ]


def test_join_metadata_prefixes_bare_columns(tmp_path):
    """join_metadata prefixes bare CSV attribute columns (not join keys) so the
    mirror orderer treats them as front metadata, matching the pandas path."""
    import polars as pl
    from phenotypic._cli._cli_output_manager import join_metadata

    df = pl.DataFrame({"MetadataImage_ImageName": ["a"], "Shape_Area": [1.0]})
    csv = tmp_path / "m.csv"
    # Join key already prefixed; attribute columns are BARE (Strain known, Foo unknown).
    csv.write_text("MetadataImage_ImageName,Strain,Foo\na,BY4741,bar\n")

    out = join_metadata(df, csv)

    assert "MetadataGenetic_Strain" in out.columns  # bare known label -> per-topic
    assert "Metadata_Foo" in out.columns            # bare unknown label -> generic
    assert "Strain" not in out.columns and "Foo" not in out.columns
    assert "MetadataImage_ImageName" in out.columns  # join key kept its raw name


def test_join_metadata_prefixed_then_ordered_lands_in_front(tmp_path):
    """A bare CSV attribute column, once prefixed, orders into the front block."""
    import polars as pl
    from phenotypic.sdk_ import order_measurement_columns
    from phenotypic._cli._cli_output_manager import join_metadata

    df = pl.DataFrame(
        {"MetadataImage_ImageName": ["a"], "Shape_Area": [1.0], "Object_Label": [1]}
    )
    csv = tmp_path / "m.csv"
    csv.write_text("MetadataImage_ImageName,Strain\na,BY4741\n")

    joined = join_metadata(df, csv)
    ordered = order_measurement_columns(joined.columns)

    # Strain (now MetadataGenetic_Strain) leads, ahead of the measurements.
    assert ordered.index("MetadataGenetic_Strain") < ordered.index("Shape_Area")
