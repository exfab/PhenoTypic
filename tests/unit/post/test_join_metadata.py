"""Behaviour of :class:`phenotypic.post.JoinMetadata`.

The op exists because measurement frames carry no experimental annotation:
a pipeline measures image name, grid position, object label and features, so
anything that groups colonies by strain / medium / pH has to be joined in.
That makes the failure modes worth pinning explicitly -- a join that silently
fans out, silently matches nothing, or silently renames a key produces a frame
that looks right and groups wrong.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from phenotypic.post import JoinMetadata


@pytest.fixture
def layout(tmp_path):
    """A two-plate, four-well layout keyed by image name and grid cell."""
    path = tmp_path / "layout.csv"
    pd.DataFrame(
        {
            "Metadata_ImageName": ["p1", "p1", "p2", "p2"],
            "Grid_RowNum": [1, 3, 1, 3],
            "Grid_ColNum": [1, 2, 1, 2],
            "Metadata_Strain": ["WT", "mut", "WT", "mut"],
            "Metadata_pH": [4, 4, 7, 7],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def measurements():
    """Measurements for both plates, in a deliberately non-sorted order."""
    return pd.DataFrame(
        {
            "Metadata_ImageName": ["p2", "p1", "p1", "p2"],
            "Grid_RowNum": [3, 1, 3, 1],
            "Grid_ColNum": [2, 1, 2, 1],
            "Object_Label": [1, 2, 3, 4],
            "Size_Area": [10.0, 20.0, 30.0, 40.0],
        }
    )


KEYS = ["Metadata_ImageName", "Grid_RowNum", "Grid_ColNum"]


def test_joins_annotation_onto_each_row(layout, measurements):
    out = JoinMetadata(metadata=layout, on=KEYS).apply(measurements)
    assert list(out.Metadata_Strain) == ["mut", "WT", "mut", "WT"]
    assert list(out.Metadata_pH) == [7, 4, 4, 7]


def test_preserves_row_count_and_order(layout, measurements):
    out = JoinMetadata(metadata=layout, on=KEYS).apply(measurements)
    assert len(out) == len(measurements)
    # Object_Label is the identity carried through; order must be untouched so a
    # caller can assign the result back alongside the original frame.
    assert list(out.Object_Label) == list(measurements.Object_Label)
    assert list(out.index) == list(measurements.index)


def test_grid_key_is_not_rewritten_to_a_metadata_spelling(layout, measurements):
    """``Grid_RowNum`` must stay ``Grid_RowNum``.

    ``ensure_metadata_prefix`` spares a name only when it is already in a
    Metadata category, so prefixing unconditionally would turn a valid
    ``Grid_RowNum`` key into ``Metadata_Grid_RowNum`` and fail against a table
    that spells it correctly.
    """
    op = JoinMetadata(metadata=layout, on=KEYS)
    assert op.on == KEYS
    out = op.apply(measurements)
    assert "Metadata_Grid_RowNum" not in out.columns
    assert "Metadata_Grid_ColNum" not in out.columns


def _write(path, frame):
    frame.to_csv(path, index=False)
    return path


def test_bare_annotation_column_takes_the_metadata_prefix(tmp_path, measurements):
    """An annotation column the measurements do not carry is metadata."""
    path = _write(
        tmp_path / "bare.csv",
        pd.DataFrame(
            {
                "Metadata_ImageName": ["p1", "p1", "p2", "p2"],
                "Grid_RowNum": [1, 3, 1, 3],
                "Grid_ColNum": [1, 2, 1, 2],
                "Medium": ["YPD", "SC", "YPD", "SC"],
            }
        ),
    )
    out = JoinMetadata(metadata=path, on=KEYS).apply(measurements)
    assert "Medium" not in out.columns
    assert list(out.Metadata_Medium) == ["SC", "YPD", "SC", "YPD"]


def test_raw_column_shared_with_the_measurements_keeps_its_name(tmp_path):
    """A raw key both frames carry is a join key, not an annotation.

    Prefixing it would rename the table side to ``Metadata_plate`` and match
    nothing against the measurement frame's ``plate``.
    """
    path = _write(
        tmp_path / "raw_key.csv",
        pd.DataFrame({"plate": ["A", "B"], "Strain": ["WT", "mut"]}),
    )
    frame = pd.DataFrame({"plate": ["B", "A"], "Object_Label": [1, 2]})
    out = JoinMetadata(metadata=path, on=["plate"]).apply(frame)
    assert "Metadata_plate" not in out.columns
    assert list(out.plate) == ["B", "A"]
    assert list(out.Metadata_Strain) == ["mut", "WT"]


def test_key_the_measurements_spell_prefixed_is_prefixed_on_the_table(tmp_path):
    """The preserve check is against the measurement frame, not the table.

    ``plate`` is raw in the table but the measurements carry
    ``Metadata_plate``, so it is not a shared raw column and takes the prefix.
    """
    path = _write(
        tmp_path / "prefixed_key.csv",
        pd.DataFrame({"plate": ["A", "B"], "Strain": ["WT", "mut"]}),
    )
    frame = pd.DataFrame({"Metadata_plate": ["B", "A"], "Object_Label": [1, 2]})
    out = JoinMetadata(metadata=path, on=["plate"]).apply(frame)
    assert "plate" not in out.columns
    assert list(out.Metadata_Strain) == ["mut", "WT"]


def test_known_non_metadata_schema_header_keeps_its_name(tmp_path, measurements):
    """A schema header is never reclassified as metadata, shared or not."""
    path = _write(
        tmp_path / "schema_header.csv",
        pd.DataFrame(
            {
                "Metadata_ImageName": ["p1", "p1", "p2", "p2"],
                "Grid_RowNum": [1, 3, 1, 3],
                "Grid_ColNum": [1, 2, 1, 2],
                "Shape_Circularity": [0.1, 0.2, 0.3, 0.4],
            }
        ),
    )
    out = JoinMetadata(metadata=path, on=KEYS).apply(measurements)
    assert "Shape_Circularity" in out.columns
    assert "Metadata_Shape_Circularity" not in out.columns


def test_joined_column_already_in_the_frame_is_refused(tmp_path, measurements):
    """A shared non-key column would split into ``_x``/``_y`` copies."""
    path = _write(
        tmp_path / "clash.csv",
        pd.DataFrame(
            {
                "Metadata_ImageName": ["p1", "p1", "p2", "p2"],
                "Grid_RowNum": [1, 3, 1, 3],
                "Grid_ColNum": [1, 2, 1, 2],
                "Size_Area": [1.0, 2.0, 3.0, 4.0],
            }
        ),
    )
    with pytest.raises(ValueError, match="already in the measurement frame"):
        JoinMetadata(metadata=path, on=KEYS).apply(measurements)


def test_naming_rule_matches_the_cli_metadata_join(tmp_path, measurements):
    """``JoinMetadata`` and the CLI ``--metadata`` join name columns alike."""
    import polars as pl

    from phenotypic._cli._metadata_join import normalize_external_metadata_columns

    table = pd.DataFrame(
        {
            "ImageName": ["p1", "p1", "p2", "p2"],
            "Grid_RowNum": [1, 3, 1, 3],
            "Grid_ColNum": [1, 2, 1, 2],
            "Medium": ["YPD", "SC", "YPD", "SC"],
            "Strain": ["WT", "mut", "WT", "mut"],
        }
    )
    path = _write(tmp_path / "parity.csv", table)
    out = JoinMetadata(
        metadata=path, on=["ImageName", "Grid_RowNum", "Grid_ColNum"]
    ).apply(measurements)

    cli = normalize_external_metadata_columns(
        pl.from_pandas(measurements), pl.from_pandas(table)
    )
    assert set(cli.columns) <= set(out.columns)


def test_bare_label_falls_back_to_the_prefixed_spelling(layout, measurements):
    op = JoinMetadata(metadata=layout, on=["ImageName", "Grid_RowNum", "Grid_ColNum"])
    assert op.on[0] == "Metadata_ImageName"
    assert len(op.apply(measurements)) == len(measurements)


def test_columns_subset_brings_only_what_was_asked(layout, measurements):
    out = JoinMetadata(
        metadata=layout, on=KEYS, columns=["Metadata_Strain"]
    ).apply(measurements)
    assert "Metadata_Strain" in out.columns
    assert "Metadata_pH" not in out.columns


def test_duplicate_key_is_refused_rather_than_fanning_out(tmp_path, measurements):
    """A duplicated key would multiply measurement rows.

    This is the dangerous failure: a left join against a table with two rows per
    key returns MORE rows than it was given, silently inflating every count and
    group statistic computed downstream. It must fail loudly, at construction.
    """
    path = tmp_path / "dupes.csv"
    pd.DataFrame(
        {
            "Metadata_ImageName": ["p1", "p1"],
            "Grid_RowNum": [1, 1],
            "Grid_ColNum": [1, 1],
            "Metadata_Strain": ["WT", "mut"],  # same cell, two strains
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate row"):
        JoinMetadata(metadata=path, on=KEYS)


def test_strict_raises_when_a_measurement_row_matches_nothing(layout, measurements):
    orphan = pd.concat(
        [
            measurements,
            pd.DataFrame(
                {
                    "Metadata_ImageName": ["p3"],
                    "Grid_RowNum": [5],
                    "Grid_ColNum": [9],
                    "Object_Label": [9],
                    "Size_Area": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="matched no"):
        JoinMetadata(metadata=layout, on=KEYS).apply(orphan)


def test_non_strict_leaves_unmatched_rows_null(layout, measurements):
    orphan = pd.concat(
        [
            measurements,
            pd.DataFrame(
                {
                    "Metadata_ImageName": ["p3"],
                    "Grid_RowNum": [5],
                    "Grid_ColNum": [9],
                    "Object_Label": [9],
                    "Size_Area": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    out = JoinMetadata(metadata=layout, on=KEYS, strict=False).apply(orphan)
    assert len(out) == len(orphan)
    assert pd.isna(out.Metadata_Strain.iloc[-1])


def test_key_dtype_mismatch_still_matches(layout, measurements):
    """A CSV-read int64 key against a float/object frame key.

    pandas reports a dtype-mismatched merge as an all-NaN join rather than an
    error, so without explicit alignment this looks like "no metadata matched"
    and, under strict, blames the data.
    """
    shifted = measurements.assign(Grid_RowNum=measurements.Grid_RowNum.astype(float))
    out = JoinMetadata(metadata=layout, on=KEYS).apply(shifted)
    assert out.Metadata_Strain.notna().all()


def test_missing_key_in_the_table_fails_at_construction(layout):
    with pytest.raises(KeyError, match="key column"):
        JoinMetadata(metadata=layout, on=["Metadata_ImageName", "Nonexistent_Key"])


def test_missing_key_in_the_frame_fails_at_apply(layout, measurements):
    op = JoinMetadata(metadata=layout, on=KEYS)
    with pytest.raises(KeyError, match="measurement frame"):
        op.apply(measurements.drop(columns=["Grid_ColNum"]))


def test_missing_path_fails_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        JoinMetadata(metadata=tmp_path / "absent.csv", on=["Metadata_ImageName"])


def test_unsupported_suffix_is_rejected(tmp_path):
    path = tmp_path / "layout.xlsx"
    path.write_text("not really a spreadsheet")
    with pytest.raises(ValueError, match="unsupported metadata suffix"):
        JoinMetadata(metadata=path, on=["Metadata_ImageName"])


def test_empty_on_is_rejected(layout):
    with pytest.raises(ValueError, match="at least one key column"):
        JoinMetadata(metadata=layout, on=[])


def test_parquet_table_is_supported(tmp_path, measurements):
    path = tmp_path / "layout.parquet"
    pd.DataFrame(
        {
            "Metadata_ImageName": ["p1", "p1", "p2", "p2"],
            "Grid_RowNum": [1, 3, 1, 3],
            "Grid_ColNum": [1, 2, 1, 2],
            "Metadata_Strain": ["WT", "mut", "WT", "mut"],
        }
    ).to_parquet(path)
    out = JoinMetadata(metadata=path, on=KEYS).apply(measurements)
    assert out.Metadata_Strain.notna().all()


def test_round_trips_through_json(layout, measurements):
    """The path is what persists -- a reloaded op must re-read the same table."""
    op = JoinMetadata(metadata=layout, on=KEYS, columns=["Metadata_Strain"])
    payload = op.model_dump_json()
    assert str(layout) in json.loads(payload)["metadata"]

    restored = JoinMetadata.model_validate_json(payload)
    pd.testing.assert_frame_equal(restored.apply(measurements), op.apply(measurements))


def test_runs_as_a_pipeline_post_op(layout, measurements):
    """`post` ops run inside `measure(apply_post=True)`, which is what puts the
    joined columns in front of a scorer."""
    from phenotypic import ImagePipeline

    pipe = ImagePipeline(post=[JoinMetadata(metadata=layout, on=KEYS)])
    assert len(pipe.post) == 1
    # Serialization must keep the op resolvable by bare class name. Use
    # `from_json`, not the raw pydantic validator: class-tagged op payloads are
    # resolved by `_find_class_in_phenotypic`, which only `from_json` invokes.
    restored = ImagePipeline.from_json(pipe.to_json())
    assert type(list(restored.post.values())[0]).__name__ == "JoinMetadata"
