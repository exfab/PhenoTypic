"""CLI compatibility tests for metadata namespace decoupling."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenotypic._cli._cli_chunk_writer import (
    _dedupe_on_colony_key,
    _incremental_combined,
    _update_dataset_parquet,
)
from phenotypic._cli._cli_output_manager import join_metadata
from phenotypic._cli._cli_output_manager import finalize_post_master_outputs
from phenotypic._cli._cli_parquet_agg import aggregate_parquet_files
from phenotypic._cli._metadata_join import prepare_metadata_join_keys
from phenotypic import ImagePipeline
from phenotypic.abc_._post_measurement import PostMeasurement
from phenotypic.post import MergeMetadata
from phenotypic.schema import (
    EXPERIMENT,
    GENETIC,
    IMAGE,
    OBJECT,
    MetadataInfo,
)
from phenotypic.sdk_ import (
    DATASET_AGGREGATED_PARQUET,
    DIR_MEASUREMENTS,
    DIR_RESULTS,
    measurements_csv_path,
    measurements_parquet_path,
)


def _legacy_header(member: MetadataInfo) -> str:
    """Return the exact previous-release spelling for a metadata member."""
    legacy_categories = {
        "IMAGE": "MetadataImage",
        "GENETIC": "MetadataGenetic",
        "SAMPLE": "MetadataSample",
        "PLATE": "MetadataPlate",
        "CONDITION": "MetadataCondition",
        "CULTURE": "MetadataCulture",
        "EXPERIMENT": "MetadataExperiment",
        "STUDY": "MetadataStudy",
        "ACQUISITION": "MetadataAcquisition",
    }
    return f"{legacy_categories[type(member).__name__]}_{member.label}"


@pytest.mark.parametrize(
    "source_header",
    [
        IMAGE.IMAGE_NAME.label,
        str(IMAGE.IMAGE_NAME),
        _legacy_header(IMAGE.IMAGE_NAME),
    ],
)
def test_join_accepts_bare_current_and_future_flat_keys(
    tmp_path: Path,
    source_header: str,
) -> None:
    """All supported external key spellings join to the live emitted header."""
    source = tmp_path / "metadata.csv"
    source.write_text(
        f"{source_header},{GENETIC.STRAIN.label}\nplate_a,BY4741\n",
        encoding="utf-8",
    )
    original_bytes = source.read_bytes()
    measurements = pl.DataFrame(
        {str(IMAGE.IMAGE_NAME): ["plate_a"], "Shape_Area": [12.0]}
    )

    joined = join_metadata(measurements, source)

    assert joined[str(IMAGE.IMAGE_NAME)].to_list() == ["plate_a"]
    assert joined[str(GENETIC.STRAIN)].to_list() == ["BY4741"]
    assert source_header not in joined.columns or source_header == str(
        IMAGE.IMAGE_NAME
    )
    assert source.read_bytes() == original_bytes


def test_join_coalesces_complementary_current_and_future_columns(
    tmp_path: Path,
) -> None:
    """Partially migrated external columns coalesce into the live spelling."""
    current = str(GENETIC.STRAIN)
    future = _legacy_header(GENETIC.STRAIN)
    source = tmp_path / "metadata.csv"
    source.write_text(
        f"plate,{current},{future}\nA,BY4741,\nB,,BY4742\n",
        encoding="utf-8",
    )
    measurements = pl.DataFrame(
        {"plate": ["A", "B"], "Shape_Area": [10.0, 11.0]}
    )

    joined = join_metadata(measurements, source)

    assert joined[current].to_list() == ["BY4741", "BY4742"]
    assert future not in joined.columns


def test_join_rejects_conflicting_duplicate_columns_without_mutating_source(
    tmp_path: Path,
) -> None:
    """Conflicting aliases fail loudly and leave the external file untouched."""
    current = str(GENETIC.STRAIN)
    future = _legacy_header(GENETIC.STRAIN)
    source = tmp_path / "metadata.csv"
    source.write_text(
        f"plate,{current},{future}\nA,BY4741,BY4742\n",
        encoding="utf-8",
    )
    original_bytes = source.read_bytes()

    with pytest.raises(ValueError, match="conflicting non-null values"):
        join_metadata(
            pl.DataFrame({"plate": ["A"], "Shape_Area": [10.0]}),
            source,
        )

    assert source.read_bytes() == original_bytes


def test_prepare_join_normalizes_on_copies() -> None:
    """In-memory measurement and external frames remain byte-for-byte logical copies."""
    measurements = pl.DataFrame(
        {_legacy_header(IMAGE.IMAGE_NAME): ["plate_a"], "Shape_Area": [1.0]}
    )
    metadata = pl.DataFrame(
        {IMAGE.IMAGE_NAME.label: ["plate_a"], GENETIC.STRAIN.label: ["WT"]}
    )
    original_measurements = measurements.clone()
    original_metadata = metadata.clone()

    prepared = prepare_metadata_join_keys(measurements, metadata)

    assert prepared.analysis.columns == (str(IMAGE.IMAGE_NAME),)
    assert prepared.metadata[str(GENETIC.STRAIN)].to_list() == ["WT"]
    assert measurements.equals(original_measurements)
    assert metadata.equals(original_metadata)


@pytest.mark.parametrize(
    "stored_header",
    [
        IMAGE.IMAGE_NAME.label,
        str(IMAGE.IMAGE_NAME),
        _legacy_header(IMAGE.IMAGE_NAME),
    ],
)
def test_parquet_aggregation_accepts_all_metadata_spellings_and_emits_current(
    tmp_path: Path,
    stored_header: str,
) -> None:
    """Recompile readers normalize old and future inputs without changing C3 output."""
    source = tmp_path / "plate_a.parquet"
    pl.DataFrame(
        {
            stored_header: ["plate_a"],
            _legacy_header(EXPERIMENT.DATASET): ["already-present"],
            "Size_Area": [42],
        }
    ).write_parquet(source)

    aggregated = aggregate_parquet_files(
        [source],
        {source: "derived-dataset"},
        include_dataset_column=True,
    )

    assert aggregated is not None
    assert aggregated[str(IMAGE.IMAGE_NAME)].to_list() == ["plate_a"]
    assert aggregated[str(EXPERIMENT.DATASET)].to_list() == ["already-present"]
    assert aggregated["Size_Area"].to_list() == [42]
    assert stored_header not in aggregated.columns or stored_header == str(
        IMAGE.IMAGE_NAME
    )
    assert _legacy_header(EXPERIMENT.DATASET) not in aggregated.columns


def _future_flat_colonies() -> pl.DataFrame:
    """Return two colonies stored with the future flat metadata spelling."""
    return pl.DataFrame(
        {
            _legacy_header(EXPERIMENT.DATASET): ["dataset", "dataset"],
            _legacy_header(IMAGE.IMAGE_NAME): ["plate_a", "plate_b"],
            str(OBJECT.LABEL): [1, 1],
            "Size_Area": [10, 20],
        }
    )


def _current_colony() -> pl.DataFrame:
    """Return one colony stored with the current metadata spelling."""
    return pl.DataFrame(
        {
            str(EXPERIMENT.DATASET): ["dataset"],
            str(IMAGE.IMAGE_NAME): ["plate_c"],
            str(OBJECT.LABEL): [1],
            "Size_Area": [30],
        }
    )


def test_incremental_combined_normalizes_existing_future_flat_rows(
    tmp_path: Path,
) -> None:
    """Rolling resume normalizes persisted rows before colony-key deduplication."""
    existing_path = tmp_path / "analysis_full.parquet"
    _future_flat_colonies().write_parquet(existing_path)

    combined = _incremental_combined(_current_colony(), existing_path)
    deduplicated = _dedupe_on_colony_key(combined, context="test")

    assert deduplicated.height == 3
    assert set(deduplicated[str(IMAGE.IMAGE_NAME)]) == {
        "plate_a",
        "plate_b",
        "plate_c",
    }
    assert _legacy_header(IMAGE.IMAGE_NAME) not in deduplicated.columns


def test_dataset_resume_normalizes_existing_future_flat_rows(
    tmp_path: Path,
) -> None:
    """Dataset resume preserves same-labeled colonies from distinct images."""
    aggregate_path = (
        tmp_path
        / DIR_RESULTS
        / "dataset"
        / DIR_MEASUREMENTS
        / DATASET_AGGREGATED_PARQUET
    )
    aggregate_path.parent.mkdir(parents=True)
    _future_flat_colonies().write_parquet(aggregate_path)

    _update_dataset_parquet(tmp_path, "dataset", _current_colony())

    resumed = pl.read_parquet(aggregate_path)
    assert resumed.height == 3
    assert set(resumed[str(IMAGE.IMAGE_NAME)]) == {
        "plate_a",
        "plate_b",
        "plate_c",
    }
    assert _legacy_header(IMAGE.IMAGE_NAME) not in resumed.columns


def test_finalize_external_alias_conflict_aborts_before_mirror_publication(
    tmp_path: Path,
) -> None:
    """A conflicting external alias pair cannot fall back to a clean mirror."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    metadata_csv = tmp_path / "metadata.csv"
    current = str(GENETIC.STRAIN)
    future = _legacy_header(GENETIC.STRAIN)
    metadata_csv.write_text(
        f"{IMAGE.IMAGE_NAME},{current},{future}\nplate_a,WT,mutant\n",
        encoding="utf-8",
    )
    master = pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["plate_a"],
            str(OBJECT.LABEL): [1],
            "Size_Area": [10],
        }
    )

    with pytest.raises(ValueError, match="conflicting non-null values"):
        finalize_post_master_outputs(
            output_dir,
            master,
            ImagePipeline(),
            metadata_csv=metadata_csv,
            no_qc=True,
        )

    assert not measurements_csv_path(output_dir).exists()
    assert not measurements_parquet_path(output_dir).exists()


def test_finalize_post_alias_conflict_aborts_before_mirror_publication(
    tmp_path: Path,
) -> None:
    """A post-operation alias conflict cannot publish a partially applied mirror."""

    class AddConflictingStrainAliases(PostMeasurement):
        """Introduce conflicting equivalent columns for the next post op."""

        def _operate(self, df):
            result = df.copy()
            result[str(GENETIC.STRAIN)] = "WT"
            result[_legacy_header(GENETIC.STRAIN)] = "mutant"
            return result

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    master = pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["plate_a"],
            str(OBJECT.LABEL): [1],
            "Size_Area": [10],
        }
    )
    pipeline = ImagePipeline(
        post=[
            AddConflictingStrainAliases(),
            MergeMetadata(
                columns=["Strain", "ImageName"],
                label="SampleID",
            ),
        ]
    )

    with pytest.raises(ValueError, match="conflicting non-null values"):
        finalize_post_master_outputs(
            output_dir,
            master,
            pipeline,
            no_qc=True,
        )

    assert not measurements_csv_path(output_dir).exists()
    assert not measurements_parquet_path(output_dir).exists()


def test_finalize_last_post_alias_conflict_aborts_before_mirror_publication(
    tmp_path: Path,
) -> None:
    """The final post output is normalized before either mirror is written."""

    class AddConflictingStrainAliases(PostMeasurement):
        """Return a frame containing conflicting equivalent metadata columns."""

        def _operate(self, df):
            result = df.copy()
            result[str(GENETIC.STRAIN)] = "WT"
            result[_legacy_header(GENETIC.STRAIN)] = "mutant"
            return result

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    master = pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["plate_a"],
            str(OBJECT.LABEL): [1],
            "Size_Area": [10],
        }
    )

    with pytest.raises(ValueError, match="conflicting non-null values"):
        finalize_post_master_outputs(
            output_dir,
            master,
            ImagePipeline(post=[AddConflictingStrainAliases()]),
            no_qc=True,
        )

    assert not measurements_csv_path(output_dir).exists()
    assert not measurements_parquet_path(output_dir).exists()
