"""Both ``--mode migrate`` passes, in order, through the real entry point.

Every test in ``tests/unit/sdk_/test_migration_republishes_state.py`` calls
``migrate_run_hdf_to_zarr`` directly, which is **pass 2 only**. Pass 1 lives
in the CLI driver, so the interaction MIG-15 is about -- pass 1 rewriting the
parquets that pass 2's markers fingerprint -- is exercised by nothing else.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from click.testing import CliRunner

from phenotypic._cli._cli_completion import (
    current_aggregate_is_current,
    current_success_counts,
    valid_run_completion,
    valid_image_success,
)
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    dataset_measurements_dir,
    dataset_overlays_dir,
    datasets_needing_migration,
    zarr_store_path,
)
from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON, valid_staged_store

from tests.unit.sdk_._migration_fixtures import LegacyRun


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_full_migrate_leaves_the_run_valid_and_idle(
    finished_legacy_run: LegacyRun,
) -> None:
    """The test MIG-15 predicts will fail against an images-first plan.

    Pass 1 rewrites ``results/<ds>/measurements/*.parquet``, and every
    per-image completion marker carries that parquet's size and sha256. Run
    the image pass first and the marker republication fingerprints parquets
    that the non-image pass then rewrites -- silently reintroducing the exact
    failure the republication exists to prevent, on the default path.
    """
    tree = finished_legacy_run.path
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )
    assert result.exit_code == 0, result.output

    for stem in finished_legacy_run.stems:
        assert valid_staged_store(zarr_store_path(tree, "ds", stem))
        assert valid_image_success(
            tree,
            dataset="ds",
            image_stem=stem,
            work_id=finished_legacy_run.work_id_for(stem),
        ), stem
    assert current_aggregate_is_current(tree) is True
    completion = valid_run_completion(tree)
    assert completion is not None
    assert completion["version"] == 2


def test_fixture_shaped_run_completes_32_measured_and_four_zero_object_images(
    finished_legacy_run: LegacyRun,
) -> None:
    """The reported field failure shape completes all 36 legitimate images."""
    import h5py

    legacy_run = finished_legacy_run.path
    hdf_dir = legacy_run / "results" / "ds" / "hdf"
    measurements = dataset_measurements_dir(legacy_run, "ds")
    source_stem = finished_legacy_run.stems[0]
    source_hdf = hdf_dir / f"{source_stem}.h5"
    source_table = measurements / f"{source_stem}.parquet"

    for index in range(30):
        stem = f"measured-{index:02d}"
        shutil.copy2(source_hdf, hdf_dir / f"{stem}.h5")
        shutil.copy2(source_table, measurements / f"{stem}.parquet")
    for index in range(4):
        stem = f"zero-{index:02d}"
        target = hdf_dir / f"{stem}.h5"
        shutil.copy2(source_hdf, target)
        with h5py.File(target, mode="a") as handle:
            handle["layers/objmap"][:] = 0

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--njobs",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert current_success_counts(legacy_run) == (36, 36)
    assert len(list(dataset_overlays_dir(legacy_run, "ds").glob("*.png"))) == 36
    assert current_aggregate_is_current(legacy_run) is True
    assert valid_run_completion(legacy_run) is not None
    assert datasets_needing_migration(legacy_run) == []


def test_the_migrated_tree_does_no_work_on_the_next_full_run(
    finished_legacy_run: LegacyRun,
) -> None:
    """The end-to-end consequence, through the CLI on both sides."""
    tree = finished_legacy_run.path
    assert (
        CliRunner()
        .invoke(phenotypic_cli, ["--mode", "migrate", "--output", str(tree)])
        .exit_code
        == 0
    )

    roots = {
        stem: _digest(zarr_store_path(tree, "ds", stem) / STORE_ROOT_JSON)
        for stem in finished_legacy_run.stems
    }
    parquets = {
        stem: _digest(dataset_measurements_dir(tree, "ds") / f"{stem}.parquet")
        for stem in finished_legacy_run.stems
    }

    second = CliRunner().invoke(phenotypic_cli, finished_legacy_run.full_run_args())

    assert second.exit_code == 0, second.output
    for stem, digest in roots.items():
        assert (
            _digest(zarr_store_path(tree, "ds", stem) / STORE_ROOT_JSON) == digest
        ), f"{stem} store was re-promoted"
    for stem, digest in parquets.items():
        assert (
            _digest(dataset_measurements_dir(tree, "ds") / f"{stem}.parquet")
            == digest
        ), f"{stem} was re-measured"


def test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate(
    finished_legacy_run: LegacyRun,
) -> None:
    """Immutable input provenance, through the whole two-pass driver.

    The unit-level guard calls pass 2 alone; this one covers pass 1, the
    marker republication, the aggregate republish, and the canonical view.
    """
    tree = finished_legacy_run.path
    snapshot = tree / "deliverables" / "metadata.csv"
    before = snapshot.read_bytes()

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )

    assert result.exit_code == 0, result.output
    assert snapshot.read_bytes() == before
    assert not (tree / "deliverables" / "metadata.original.csv").exists()
    assert (tree / "deliverables" / "metadata.canonical.csv").is_file()


def test_one_manifest_image_primitive_publishes_complete_scientific_authority(
    finished_legacy_run: LegacyRun,
) -> None:
    """The shared worker core completes one real demoted image without discovery."""
    from phenotypic._cli._cli_migrate import run_metadata_pass
    from phenotypic._cli._cli_migrate_image import migrate_image_task
    from phenotypic._cli._cli_migrate_manifest import discover_migration_tasks
    from phenotypic.sdk_ import metadata_csv_deliverable_path

    tree = finished_legacy_run.path
    metadata = metadata_csv_deliverable_path(tree)
    assert not run_metadata_pass(tree, dry_run=False).failures
    task = discover_migration_tasks(tree)[0]

    result = migrate_image_task(
        tree,
        task,
        metadata_csv=metadata,
        overlay_alpha=0.3,
        dry_run=False,
    )

    assert valid_staged_store(task.store_path)
    assert result.marker_digest == _digest(task.marker_path)
    assert valid_image_success(
        tree,
        dataset=task.dataset,
        image_stem=task.stem,
        work_id=result.work_id,
    )


def test_delete_sources_reclaims_only_after_the_markers_validate(
    finished_legacy_run: LegacyRun,
) -> None:
    """``--delete-sources`` is the only irreversible step in this phase.

    Its precondition is deliberately stronger than ``valid_staged_store``:
    a value-level re-read comparison **and** a passing ``valid_image_success``
    after republication.
    """
    tree = finished_legacy_run.path
    hdf_dir = tree / "results" / "ds" / "hdf"
    assert list(hdf_dir.glob("*.h5"))

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(tree), "--delete-sources"],
    )

    assert result.exit_code == 0, result.output
    assert not list(hdf_dir.glob("*.h5"))
    for stem in finished_legacy_run.stems:
        assert valid_image_success(
            tree,
            dataset="ds",
            image_stem=stem,
            work_id=finished_legacy_run.work_id_for(stem),
        ), stem
