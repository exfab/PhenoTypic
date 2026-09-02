"""One migration predicate, applied to every consumer.

The spec's guard was "output contains **only** ``.h5`` results fails with a
pointer", tested through ``--mode recompile`` alone. But migration is
explicitly resumable, so a **half-migrated** tree is the expected state after
any interruption -- and it is neither "only ``.h5``" nor fully converted, so
it passed the guard. ``--mode full`` would then silently reprocess every
unconverted image from source. The GUI had no detection at all: it discovers
the tree, lists every image, and resolves unconverted ones to ``None``,
rendering silently empty (ledger MIG-8).
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.sdk_ import datasets_needing_migration


def test_a_half_migrated_tree_is_detected(half_migrated_run: Path) -> None:
    """The expected state after any interruption -- migration is resumable."""
    assert datasets_needing_migration(half_migrated_run) == ["ds"]


def test_a_fully_migrated_tree_is_clean(migrated_run: Path) -> None:
    assert datasets_needing_migration(migrated_run) == []


def test_the_predicate_is_PER_IMAGE_not_per_dataset(
    half_migrated_run: Path,
) -> None:
    """A dataset-level "has .h5 and has no zarr/ dir" test misses this shape.

    The half-migrated tree's converted and unconverted images are in the
    SAME dataset, so ``zarr/`` exists and holds a valid store -- and the
    dataset still needs migrating.
    """
    from phenotypic.sdk_ import dataset_zarr_dir
    from phenotypic.sdk_.ngff_ import STORE_SUFFIX

    zarr_dir = dataset_zarr_dir(half_migrated_run, "ds")
    assert zarr_dir.is_dir(), "the dataset must already have a zarr/ dir"
    assert list(zarr_dir.glob(f"*{STORE_SUFFIX}")), "and at least one store"
    assert datasets_needing_migration(half_migrated_run) == ["ds"]


def test_the_predicate_gates_on_VALIDITY_not_existence(
    migrated_run: Path,
) -> None:
    """A store written at an older ``store_schema_version`` is present but the
    loader refuses it, so an existence test reads that tree as clean while
    every image fails to open."""
    import json

    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON, PhenotypicAttr

    assert datasets_needing_migration(migrated_run) == []

    store = zarr_store_path(migrated_run, "ds", "img")
    root = store / STORE_ROOT_JSON
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"][PhenotypicAttr.ROOT][
        PhenotypicAttr.STORE_SCHEMA_VERSION
    ] = 2
    root.write_text(json.dumps(payload), encoding="utf-8")

    assert datasets_needing_migration(migrated_run) == ["ds"]


def test_a_tree_with_no_hdf_at_all_is_clean(tmp_path: Path) -> None:
    """A modern run has nothing to migrate and must not be flagged."""
    (tmp_path / "results" / "ds" / "zarr").mkdir(parents=True)
    assert datasets_needing_migration(tmp_path) == []


def test_a_missing_results_tree_is_clean(tmp_path: Path) -> None:
    assert datasets_needing_migration(tmp_path) == []
