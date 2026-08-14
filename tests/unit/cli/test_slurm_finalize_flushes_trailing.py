"""Trailing images must reach the master when the image count is not a
multiple of the SLURM checkpoint interval.

Checkpoint sentinels are emitted only after every ``checkpoint_interval``
images and there is no terminal sentinel
(``_cli_slurm_array_scripts._build_entry_list``), so the last partial group is
never chunked. Final aggregation resolves its sources via
``discover_measurement_sources``, which prefers
``_dataset_aggregated.parquet`` over the individual per-image parquets — so
those trailing rows exist on disk and are absent from the published master.
"""

from pathlib import Path

import polars as pl

from phenotypic._cli._cli_chunk_writer import flush_unchunked_measurements
from phenotypic._cli._cli_output_manager import aggregate_measurements
from phenotypic.schema import EXPERIMENT_METADATA, METADATA, OBJECT
from phenotypic.sdk_ import (
    DATASET_AGGREGATED_PARQUET,
    chunk_state_path,
    dataset_measurements_dir,
    master_measurements_parquet_path,
)

DATASET = "plate_a"


def _write_per_image(output_dir: Path, stems: list[str]) -> None:
    """Write one per-image measurement parquet per stem, as a worker would."""
    meas_dir = dataset_measurements_dir(output_dir, DATASET)
    meas_dir.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        pl.DataFrame(
            {
                str(EXPERIMENT_METADATA.DATASET): [DATASET],
                str(METADATA.IMAGE_NAME): [stem],
                str(OBJECT.LABEL): [1],
                "Shape_Area": [float(i + 1)],
            }
        ).write_parquet(meas_dir / f"{stem}.parquet")


def _master_image_names(output_dir: Path) -> set[str]:
    master = pl.read_parquet(master_measurements_parquet_path(output_dir))
    return set(master[str(METADATA.IMAGE_NAME)].to_list())


def test_trailing_images_reach_the_master(tmp_path: Path) -> None:
    """Images written after the last checkpoint sentinel must not be dropped.

    Reproduces the SLURM shape: a checkpoint consumes the images seen so far,
    then the remaining ``n % interval`` images land with no further sentinel,
    and the dependent finalizer aggregates.
    """
    chunked = ["img_001", "img_002", "img_003"]
    _write_per_image(tmp_path, chunked)

    # A checkpoint sentinel fires: everything so far is chunked into
    # _dataset_aggregated.parquet.
    flush_unchunked_measurements(tmp_path)

    # The trailing image lands. No further sentinel is emitted, because
    # _build_entry_list appends one only every `checkpoint_interval` images.
    _write_per_image(tmp_path, ["img_004"])

    # The dependent finalizer runs.
    aggregate_measurements(output_dir=tmp_path, dataset_names=[DATASET])

    got = _master_image_names(tmp_path)
    assert got == {"img_001", "img_002", "img_003", "img_004"}, (
        f"master is missing trailing images: got {sorted(got)}"
    )


def test_no_trailing_images_is_unaffected(tmp_path: Path) -> None:
    """The evenly-divisible case must keep working — the flush is a no-op."""
    stems = ["img_001", "img_002"]
    _write_per_image(tmp_path, stems)
    flush_unchunked_measurements(tmp_path)

    aggregate_measurements(output_dir=tmp_path, dataset_names=[DATASET])

    assert _master_image_names(tmp_path) == set(stems)


def test_unchunked_run_is_not_chunked_by_aggregation(tmp_path: Path) -> None:
    """A local run has no chunk state, and aggregation must not create one.

    Local and staged-local runs are never chunked: aggregation reads their
    per-image parquets directly, so they do not have this defect. Flushing them
    would manufacture ``progress/chunks/`` and ``_dataset_aggregated.parquet``
    artifacts they never had — fixing nothing and changing their output layout.
    """
    stems = ["img_001", "img_002", "img_003"]
    _write_per_image(tmp_path, stems)

    aggregate_measurements(output_dir=tmp_path, dataset_names=[DATASET])

    assert _master_image_names(tmp_path) == set(stems)
    assert not chunk_state_path(tmp_path).is_file(), (
        "aggregation created chunk state for an unchunked run"
    )
    assert not (
        dataset_measurements_dir(tmp_path, DATASET)
        / DATASET_AGGREGATED_PARQUET
    ).is_file(), "aggregation created an aggregate for an unchunked run"
