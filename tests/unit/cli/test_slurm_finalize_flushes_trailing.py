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
from phenotypic._cli._cli_recompile_slurm_scripts import build_recompile_tasks
from phenotypic._cli._measurement_sources import discover_measurement_sources
from phenotypic.schema import EXPERIMENT, IMAGE, OBJECT
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
                str(EXPERIMENT.DATASET): [DATASET],
                str(IMAGE.IMAGE_NAME): [stem],
                str(OBJECT.LABEL): [1],
                "Shape_Area": [float(i + 1)],
            }
        ).write_parquet(meas_dir / f"{stem}.parquet")


def _master_image_names(output_dir: Path) -> set[str]:
    master = pl.read_parquet(master_measurements_parquet_path(output_dir))
    return set(master[str(IMAGE.IMAGE_NAME)].to_list())


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

    # Pin the mechanism: source discovery returns the aggregate alone, so the
    # trailing image is invisible to aggregation. Without this the test would
    # also pass if a future change made discovery fall back to the individual
    # parquets — which would hide a regression in the flush.
    sources = discover_measurement_sources(tmp_path, [DATASET])
    assert [p.path.name for p in sources] == [DATASET_AGGREGATED_PARQUET], (
        f"expected discovery to prefer the aggregate, got {sources}"
    )

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


def test_slurm_recompile_shards_include_trailing_images(tmp_path: Path) -> None:
    """The SLURM recompile path must see the trailing images too.

    ``build_recompile_tasks`` resolves shards at submit time and never reaches
    ``aggregate_measurements`` — the worker concatenates the shards itself. So
    the flush has to happen here as well, or the command a user runs *to
    recover from a short master* rebuilds the same short master, and
    ``--mode recompile`` and ``--mode recompile --slurm`` disagree on one
    directory.
    """
    _write_per_image(tmp_path, ["img_001", "img_002", "img_003"])
    flush_unchunked_measurements(tmp_path)
    _write_per_image(tmp_path, ["img_004"])

    tasks = build_recompile_tasks(
        output_dir=tmp_path,
        dataset_names=[DATASET],
        include_dataset_column=True,
        overlay_alpha=0.5,
        shard_size=100,
    )

    sharded = [
        Path(f).name
        for t in tasks
        for f in t.get("files", [])
    ]
    assert sharded, f"no measurement shards emitted: {tasks}"

    # Every image must be reachable from the shards — either because the
    # aggregate now contains it (post-flush) or because it is listed directly.
    rows = pl.concat(
        [
            pl.read_parquet(dataset_measurements_dir(tmp_path, DATASET) / name)
            for name in sharded
        ],
        how="diagonal_relaxed",
    )
    assert set(rows[str(IMAGE.IMAGE_NAME)].to_list()) == {
        "img_001",
        "img_002",
        "img_003",
        "img_004",
    }, f"recompile shards omit trailing images: {sharded}"


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
