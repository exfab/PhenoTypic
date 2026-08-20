"""Both ``--mode migrate`` passes, in order, through the real entry point.

Every test in ``tests/unit/sdk_/test_migration_republishes_state.py`` calls
``migrate_run_hdf_to_zarr`` directly, which is **pass 2 only**. Pass 1 lives
in the CLI driver, so the interaction MIG-15 is about -- pass 1 rewriting the
parquets that pass 2's markers fingerprint -- is exercised by nothing else.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from click.testing import CliRunner

from phenotypic._cli._cli_completion import (
    current_aggregate_is_current,
    valid_image_success,
)
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import dataset_measurements_dir, zarr_store_path
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
