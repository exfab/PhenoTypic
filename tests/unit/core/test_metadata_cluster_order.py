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
        # measurements (the metadata-join flag has no producer enum and rides
        # along here; every row matched, so it is all-False)
        "Shape_Area",
        "QC_MetadataOnly",
        # framework image block
        "MetadataImage_ImageName",
        # per-object info block
        "Object_Label",
        "Grid_RowNum",
    ]
    assert post_df["QC_MetadataOnly"].to_list() == [False]


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


def test_join_metadata_leaves_schema_header_columns_unprefixed(tmp_path):
    """A supplied column that is already a schema header (e.g. Grid_RowNum) must
    NOT get a Metadata_ prefix appended — only genuinely bare labels are prefixed."""
    import polars as pl
    from phenotypic._cli._cli_output_manager import join_metadata

    df = pl.DataFrame({"MetadataImage_ImageName": ["a"], "Shape_Area": [1.0]})
    csv = tmp_path / "m.csv"
    # Grid_RowNum is a real info-block header; Strain is a bare metadata label.
    csv.write_text("MetadataImage_ImageName,Grid_RowNum,Strain\na,3,BY4741\n")

    out = join_metadata(df, csv)

    assert "Grid_RowNum" in out.columns              # supplied header left as-is
    assert "Metadata_Grid_RowNum" not in out.columns  # NOT prefixed
    assert "MetadataGenetic_Strain" in out.columns   # bare label still prefixed


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


def test_join_metadata_inner_emits_no_phantom_flag(tmp_path):
    """The default ``how="inner"`` is unchanged — no phantom rows, no flag column.

    Backward-compat pin for the two inner call sites. ``_cli_chunk_writer`` joins
    mid-run against a PARTIAL measurements frame, so a left join there would flag
    every not-yet-processed strain as undetected in every checkpoint. The
    sentinel column must never leak either.
    """
    import polars as pl
    from phenotypic._cli._cli_output_manager import join_metadata
    from phenotypic.schema import METADATA_MATCH

    df = pl.DataFrame({"plate": ["A"], "Shape_Area": [1.0]})
    csv = tmp_path / "m.csv"
    # Plate B is metadata-only: an inner join must drop it silently, as today.
    csv.write_text("plate,Strain\nA,BY4741\nB,BY4742\n")

    out = join_metadata(df, csv)

    assert out.height == 1
    assert out["plate"].to_list() == ["A"]
    assert str(METADATA_MATCH.METADATA_ONLY) not in out.columns
    assert not [c for c in out.columns if c.startswith("__phenotypic")]


def test_join_metadata_left_keeps_metadata_row_order(tmp_path):
    """``maintain_order="left"`` — the mirror's row order follows the CSV."""
    import polars as pl
    from phenotypic._cli._cli_output_manager import join_metadata
    from phenotypic.schema import METADATA_MATCH

    df = pl.DataFrame({"plate": ["C", "A"], "Shape_Area": [3.0, 1.0]})
    csv = tmp_path / "m.csv"
    csv.write_text("plate,Strain\nA,s1\nB,s2\nC,s3\n")

    out = join_metadata(df, csv, how="left")

    assert out["plate"].to_list() == ["A", "B", "C"]
    assert out[str(METADATA_MATCH.METADATA_ONLY)].to_list() == [False, True, False]
    assert not [c for c in out.columns if c.startswith("__phenotypic")]


def test_join_metadata_duplicate_keys_warn_is_height_independent(tmp_path, caplog):
    """Duplicate keys are detected from the metadata frame's own uniqueness.

    Fixture is chosen so the join's output height EQUALS the measurement height
    (2 duplicate metadata rows for plate A fan out over 1 measured A, while
    plate B is measured but unmatched). Any height-delta inference sees no change
    and misses the duplicates entirely; ``n_unique`` still catches them.
    """
    import logging

    import polars as pl
    from phenotypic._cli._cli_output_manager import join_metadata

    df = pl.DataFrame({"plate": ["A", "B"], "Shape_Area": [1.0, 2.0]})
    csv = tmp_path / "m.csv"
    csv.write_text("plate,Strain\nA,s1\nA,s2\n")

    with caplog.at_level(logging.WARNING):
        out = join_metadata(df, csv, how="left")

    assert out.height == df.height  # the height-delta blind spot
    assert "duplicate keys" in caplog.text


def test_join_metadata_measurement_fanout_never_warns_duplicates(tmp_path, caplog):
    """One metadata key -> many colonies is the NORMAL case; it must not warn."""
    import logging

    import polars as pl
    from phenotypic._cli._cli_output_manager import join_metadata

    df = pl.DataFrame({"plate": ["A", "A", "A"], "Shape_Area": [1.0, 2.0, 3.0]})
    csv = tmp_path / "m.csv"
    csv.write_text("plate,Strain\nA,s1\n")

    with caplog.at_level(logging.WARNING):
        out = join_metadata(df, csv, how="left")

    assert out.height == 3
    assert "duplicate keys" not in caplog.text
