"""The chunk writer must not lose images when a checkpoint task is killed.

``_aggregate_chunks_locked`` consumes per-image parquets and records them in
``chunk_state.json`` so later checkpoints skip them. If that state is committed
*before* the data it describes reaches ``_dataset_aggregated.parquet``, a task
killed in between leaves those images permanently marked as consumed while
their rows are absent from the aggregate — and since final aggregation prefers
the aggregate, they never reach the master. No later flush can recover them,
because the state says there is nothing to flush.

SLURM kills checkpoint tasks routinely: walltime, preemption, node failure.
"""

import json
from pathlib import Path

import polars as pl
import pytest

from phenotypic._cli import _cli_chunk_writer as chunk_writer
from phenotypic._cli._cli_chunk_writer import (
    _update_dataset_parquet,
    flush_unchunked_measurements,
)
from phenotypic.schema import EXPERIMENT, IMAGE, OBJECT
from phenotypic.sdk_ import (
    CHUNK_MANIFEST_JSON,
    DATASET_AGGREGATED_PARQUET,
    ChunkManifestKey,
    chunk_state_path,
    dataset_measurements_dir,
    master_measurements_parquet_path,
    progress_dir,
)

DATASET = "plate_a"


def _write_per_image(
    output_dir: Path,
    stems: list[str],
    *,
    dataset: str = DATASET,
    colonies: int = 1,
    dataset_column: bool = True,
    name_column: str | None = "stem",
) -> None:
    """Write per-image parquets in a chosen on-disk shape.

    The shape matters. Per-image parquets do not all carry the colony-key
    columns: ``--no-dataset-column`` omits ``Metadata_Dataset``, and
    ``Metadata_ImageName`` can be absent or hold a UUID (``Image.name`` falls
    back to ``str(self.uuid)``). The chunk writer's reader normalizes all
    three, so a test helper that only ever writes the already-normalized
    shape cannot see a reader that skips normalization.

    Args:
        dataset_column: Emit ``Metadata_Dataset``.
        name_column: ``"stem"`` writes the file stem, ``"uuid"`` writes a
            per-image UUID-shaped value, ``None`` omits the column.
    """
    meas_dir = dataset_measurements_dir(output_dir, dataset)
    meas_dir.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        columns: dict[str, list[object]] = {}
        if dataset_column:
            columns[str(EXPERIMENT.DATASET)] = [dataset] * colonies
        if name_column == "stem":
            columns[str(IMAGE.IMAGE_NAME)] = [stem] * colonies
        elif name_column == "uuid":
            columns[str(IMAGE.IMAGE_NAME)] = [
                f"3f2b7c1a-0000-4000-8000-{i:012d}"
            ] * colonies
        columns[str(OBJECT.LABEL)] = list(range(1, colonies + 1))
        columns["Shape_Area"] = [float(i + 1)] * colonies
        pl.DataFrame(columns).write_parquet(meas_dir / f"{stem}.parquet")


def _aggregate(output_dir: Path, dataset: str = DATASET) -> pl.DataFrame | None:
    agg = dataset_measurements_dir(output_dir, dataset) / DATASET_AGGREGATED_PARQUET
    return pl.read_parquet(agg) if agg.is_file() else None


def _aggregate_image_names(output_dir: Path, dataset: str = DATASET) -> list[str]:
    agg = _aggregate(output_dir, dataset)
    if agg is None:
        return []
    return agg[str(IMAGE.IMAGE_NAME)].to_list()


def test_kill_before_aggregate_write_does_not_lose_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task killed before the aggregate lands must leave work recoverable.

    Simulates the kill by making the aggregate write raise, then retries the
    flush cleanly — as the next checkpoint sentinel or the finalize-time flush
    would. The images must reach the aggregate on that retry.
    """
    _write_per_image(tmp_path, ["img_001", "img_002"])

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated task kill before the aggregate write")

    monkeypatch.setattr(chunk_writer, "_update_dataset_parquet", _boom)
    with pytest.raises(OSError):
        flush_unchunked_measurements(tmp_path)

    monkeypatch.undo()
    flush_unchunked_measurements(tmp_path)

    assert sorted(_aggregate_image_names(tmp_path)) == ["img_001", "img_002"], (
        "images consumed by the killed task never reached the aggregate"
    )


def test_retry_after_kill_does_not_duplicate_any_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retried checkpoint must not double-count in *any* retry-exposed file.

    Committing chunk state last means a killed task re-chunks images whose rows
    may already have landed. Three artifacts are written before that commit —
    the rolling master, ``analysis_full.parquet``, and the chunk manifest — and
    each must absorb the repeat. An earlier revision deduplicated only the
    dataset aggregate, so a kill in the aggregate loop left every colony
    doubled in the master the module exists to publish mid-run.

    The kill is simulated at the *latest* possible point, after the master and
    manifest writes, which is the window that duplicates.
    """
    _write_per_image(tmp_path, ["img_001", "img_002"])

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated kill after the master and manifest writes")

    monkeypatch.setattr(chunk_writer, "_update_dataset_parquet", _boom)
    with pytest.raises(OSError):
        flush_unchunked_measurements(tmp_path)
    monkeypatch.undo()

    flush_unchunked_measurements(tmp_path)

    expected = {"img_001", "img_002"}
    assert set(_aggregate_image_names(tmp_path)) == expected
    assert len(_aggregate_image_names(tmp_path)) == 2, "aggregate duplicated"

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert master.height == 2, (
        f"rolling master duplicated after retry: {master.height} rows"
    )

    manifest = json.loads(
        (progress_dir(tmp_path) / CHUNK_MANIFEST_JSON).read_text()
    )
    names = [c[ChunkManifestKey.NAME] for c in manifest[ChunkManifestKey.CHUNKS]]
    assert len(names) == len(set(names)), f"manifest repeated a chunk: {names}"
    assert manifest[ChunkManifestKey.TOTAL_ROWS] == 2, (
        f"manifest total_rows double-counted: {manifest}"
    )


def test_dedup_is_skipped_when_the_colony_key_is_incomplete(
    tmp_path: Path,
) -> None:
    """A frame without ``Object_Label`` must keep every row.

    Nothing guarantees that column. Filtering the key down to whichever columns
    are present would leave ``(dataset, image)``, and ``unique`` on that keeps
    one row per image — silently deleting every other colony. Duplicates are
    recoverable; deleted colonies are not.
    """
    meas_dir = dataset_measurements_dir(tmp_path, DATASET)
    meas_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            str(EXPERIMENT.DATASET): [DATASET, DATASET, DATASET],
            str(IMAGE.IMAGE_NAME): ["img_001", "img_001", "img_001"],
            "Shape_Area": [1.0, 2.0, 3.0],
        }
    )

    _update_dataset_parquet(tmp_path, DATASET, frame)

    agg = pl.read_parquet(meas_dir / DATASET_AGGREGATED_PARQUET)
    assert agg.height == 3, (
        f"rows dropped by dedup on a partial key: {agg.height} of 3 survived"
    )


@pytest.mark.parametrize(
    "dataset_column, name_column",
    [
        pytest.param(True, "stem", id="normalized"),
        pytest.param(False, "stem", id="no-dataset-column"),
        pytest.param(True, None, id="no-image-name"),
        pytest.param(True, "uuid", id="uuid-image-name"),
    ],
)
def test_corrupt_aggregate_is_rebuilt_not_discarded(
    tmp_path: Path, dataset_column: bool, name_column: str | None
) -> None:
    """An unreadable aggregate must be rebuilt from its source, not replaced.

    The aggregate is a cache of ``results/<ds>/measurements/*.parquet``. When it
    cannot be read, writing only the incoming chunk destroys every previously
    aggregated colony — and chunk state still lists those sources as consumed,
    so no later flush recovers them. Since final aggregation publishes the
    master from this file, those colonies never reach the user.

    Parametrized over the on-disk shapes because the rebuild must go through
    the same identity normalization that built the aggregate. Reading the
    per-image parquets raw looks equivalent and is not: two of the three
    columns the reader fixes up are colony-key columns, so a raw re-read
    followed by dedup collapses unrelated colonies instead of restoring them
    — silently, since the rebuild's own row count is taken before the dedup.
    """
    _write_per_image(
        tmp_path,
        ["img_001", "img_002"],
        colonies=2,
        dataset_column=dataset_column,
        name_column=name_column,
    )
    flush_unchunked_measurements(tmp_path)
    assert sorted(set(_aggregate_image_names(tmp_path))) == [
        "img_001",
        "img_002",
    ]

    agg_path = (
        dataset_measurements_dir(tmp_path, DATASET) / DATASET_AGGREGATED_PARQUET
    )
    agg_path.write_bytes(b"not a parquet file at all")

    # A later checkpoint brings one new image.
    _write_per_image(
        tmp_path,
        ["img_003"],
        colonies=2,
        dataset_column=dataset_column,
        name_column=name_column,
    )
    flush_unchunked_measurements(tmp_path)

    rebuilt = _aggregate(tmp_path)
    assert rebuilt is not None
    assert sorted(set(_aggregate_image_names(tmp_path))) == [
        "img_001",
        "img_002",
        "img_003",
    ], "prior colonies were discarded instead of rebuilt from the per-image parquets"
    assert rebuilt.height == 6, (
        f"expected 3 images x 2 colonies, got {rebuilt.height} rows — the "
        f"rebuild dropped or duplicated colonies"
    )
    assert (
        rebuilt[str(EXPERIMENT.DATASET)].null_count() == 0
    ), "rebuild left Metadata_Dataset null, breaking dedup across the boundary"


def test_corrupt_aggregate_is_preserved_for_diagnosis(tmp_path: Path) -> None:
    """The unreadable bytes must survive, not be overwritten in place."""
    _write_per_image(tmp_path, ["img_001"])
    flush_unchunked_measurements(tmp_path)

    meas_dir = dataset_measurements_dir(tmp_path, DATASET)
    (meas_dir / DATASET_AGGREGATED_PARQUET).write_bytes(b"corrupt bytes")

    _write_per_image(tmp_path, ["img_002"])
    flush_unchunked_measurements(tmp_path)

    # Assert the exact name, not a glob match: it pins the numbering the
    # docstring promises, and the `_` prefix is what keeps the file out of
    # `_scan_unchunked_parquets` and `discover_measurement_sources`.
    preserved = meas_dir / "_dataset_aggregated.corrupt1.parquet"
    assert preserved.is_file(), (
        f"corrupt aggregate was overwritten: {sorted(p.name for p in meas_dir.iterdir())}"
    )
    assert preserved.read_bytes() == b"corrupt bytes"


def test_quarantine_falls_back_to_copy_when_rename_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mount that refuses rename-over must not cost us the evidence.

    The caller overwrites the aggregate either way, so if the rename fails and
    nothing else happens, the unreadable bytes are gone — on exactly the
    network filesystems (GPFS, NFS, FUSE) whose torn writes produce them.
    """
    _write_per_image(tmp_path, ["img_001"])
    flush_unchunked_measurements(tmp_path)

    meas_dir = dataset_measurements_dir(tmp_path, DATASET)
    (meas_dir / DATASET_AGGREGATED_PARQUET).write_bytes(b"corrupt bytes")

    real_replace = Path.replace

    def refuse_quarantine_rename(self: Path, target: object) -> Path:
        if "corrupt" in Path(str(target)).name:
            raise OSError("simulated rename-over refusal")
        return real_replace(self, target)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "replace", refuse_quarantine_rename)

    _write_per_image(tmp_path, ["img_002"])
    flush_unchunked_measurements(tmp_path)
    monkeypatch.undo()

    preserved = meas_dir / "_dataset_aggregated.corrupt1.parquet"
    assert preserved.is_file(), "rename failed and the bytes were not copied aside"
    assert preserved.read_bytes() == b"corrupt bytes"
    # The rebuild must still have happened.
    assert sorted(_aggregate_image_names(tmp_path)) == ["img_001", "img_002"]


def test_corrupt_aggregate_recovery_is_scoped_to_one_dataset(
    tmp_path: Path,
) -> None:
    """Corruption in one plate must not disturb another's aggregate."""
    other = "plate_b"
    _write_per_image(tmp_path, ["img_001"])
    _write_per_image(tmp_path, ["img_001"], dataset=other)
    flush_unchunked_measurements(tmp_path)

    before = _aggregate(tmp_path, other)
    assert before is not None

    (
        dataset_measurements_dir(tmp_path, DATASET) / DATASET_AGGREGATED_PARQUET
    ).write_bytes(b"corrupt bytes")
    _write_per_image(tmp_path, ["img_002"])
    _write_per_image(tmp_path, ["img_002"], dataset=other)
    flush_unchunked_measurements(tmp_path)

    assert sorted(_aggregate_image_names(tmp_path)) == ["img_001", "img_002"]
    assert sorted(_aggregate_image_names(tmp_path, other)) == [
        "img_001",
        "img_002",
    ]
    # The quarantine landed in the corrupt dataset's directory only.
    assert (
        dataset_measurements_dir(tmp_path, DATASET)
        / "_dataset_aggregated.corrupt1.parquet"
    ).is_file()
    assert not list(
        dataset_measurements_dir(tmp_path, other).glob("*corrupt*")
    )


def test_unreadable_per_image_parquet_is_retried_not_consumed(
    tmp_path: Path,
) -> None:
    """A file that could not be read must not be recorded as chunked.

    The scanner used to mark every file it *discovered* as consumed, before
    anything read it, and the reader skipped unreadable files with a bare
    warning. A torn or momentarily unavailable per-image parquet was therefore
    marked consumed forever: no later flush retries it, and since final
    aggregation prefers the aggregate, that image never reaches the master.
    """
    _write_per_image(tmp_path, ["img_001", "img_002"])
    torn = dataset_measurements_dir(tmp_path, DATASET) / "img_003.parquet"
    torn.write_bytes(b"a torn write, mid-copy")

    flush_unchunked_measurements(tmp_path)
    assert sorted(_aggregate_image_names(tmp_path)) == ["img_001", "img_002"]

    state = json.loads(chunk_state_path(tmp_path).read_text())
    assert f"{DATASET}/img_003.parquet" not in state["chunked_files"], (
        "an unread file was recorded as chunked, so no flush will retry it"
    )

    # The write completes (or the transient read error clears).
    _write_per_image(tmp_path, ["img_003"])
    flush_unchunked_measurements(tmp_path)

    assert sorted(_aggregate_image_names(tmp_path)) == [
        "img_001",
        "img_002",
        "img_003",
    ], "the recovered image never reached the aggregate"


def test_reprocessing_the_same_images_does_not_duplicate(
    tmp_path: Path,
) -> None:
    """Appending the same colony twice must not double it.

    Committing chunk state after the data means a killed task can re-chunk
    images whose rows already landed. That is only safe if the append is
    idempotent on the colony key.
    """
    _write_per_image(tmp_path, ["img_001"])
    frame = pl.read_parquet(
        dataset_measurements_dir(tmp_path, DATASET) / "img_001.parquet"
    )

    _update_dataset_parquet(tmp_path, DATASET, frame)
    _update_dataset_parquet(tmp_path, DATASET, frame)

    assert _aggregate_image_names(tmp_path) == ["img_001"], (
        "re-appending the same colony duplicated it in the aggregate"
    )


def test_reappend_keeps_the_newer_measurement(tmp_path: Path) -> None:
    """When a colony is re-measured, the later value wins.

    This is the `--restart` shape: the aggregate survives, the images are
    re-measured, and the new rows are appended over the old.
    """
    _write_per_image(tmp_path, ["img_001"])
    meas = dataset_measurements_dir(tmp_path, DATASET) / "img_001.parquet"
    first = pl.read_parquet(meas)
    _update_dataset_parquet(tmp_path, DATASET, first)

    second = first.with_columns(pl.lit(99.0).alias("Shape_Area"))
    _update_dataset_parquet(tmp_path, DATASET, second)

    agg = pl.read_parquet(
        dataset_measurements_dir(tmp_path, DATASET)
        / DATASET_AGGREGATED_PARQUET
    )
    assert agg.height == 1, f"expected one row, got {agg.height}"
    assert agg["Shape_Area"].to_list() == [99.0], (
        "the older measurement won; the later one should"
    )
