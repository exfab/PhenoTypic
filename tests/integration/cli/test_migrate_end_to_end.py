"""Both ``--mode migrate`` passes, in order, through the real entry point.

Every test in ``tests/unit/sdk_/test_migration_republishes_state.py`` calls
``migrate_run_hdf_to_zarr`` directly, which is **pass 2 only**. Pass 1 lives
in the CLI driver, so the interaction MIG-15 is about -- pass 1 rewriting the
parquets that pass 2's markers fingerprint -- is exercised by nothing else.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace

from click.testing import CliRunner

from phenotypic._cli._cli_completion import (
    current_aggregate_is_current,
    current_success_counts,
    valid_aggregate_snapshot,
    valid_run_completion,
    valid_image_success,
)
from phenotypic._cli._cli_migrate import migration_terminal_status_path
from phenotypic._cli._cli_migrate_manifest import (
    migration_image_seal_path,
    migration_reclaim_seal_path,
)
from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    dataset_measurements_dir,
    dataset_overlays_dir,
    datasets_needing_migration,
    image_completion_marker_path,
    load_image_from_store,
    phenotypic_cache_dir,
    zarr_store_path,
)
from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON, valid_staged_store

from tests.unit.sdk_._migration_fixtures import LegacyRun


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    """Return a path-and-content digest for one published artifact tree."""
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _script_indices(script: Path) -> list[int]:
    """Read the concrete work indexes emitted into one array script."""
    text = script.read_text(encoding="utf-8")
    entries = text.split("TASK_INDICES=(\n", 1)[1].split("\n)", 1)[0]
    return [int(entry.strip()) for entry in entries.splitlines()]


def _run_generated_migration_worker_commands(plan: object) -> SimpleNamespace:
    """Synchronously execute every generated migration worker command in order.

    This is intentionally a dispatcher fake, not a replacement implementation:
    its command and indexes come from the emitted scripts, then the real worker
    Click entry point consumes their immutable config.  That makes the test
    cover the actual local-vs-SLURM handoff without requiring ``sbatch``.
    """
    from phenotypic._cli._cli_migrate_slurm import MigrationSlurmPlan
    from phenotypic._cli._cli_migrate_worker import migration_worker_cli

    assert isinstance(plan, MigrationSlurmPlan)
    executed: list[tuple[str, int | None]] = []
    for script in (*plan.flat_scripts, plan.finalizer_script):
        command_line = next(
            line.strip()
            for line in script.read_text(encoding="utf-8").splitlines()
            if "-m phenotypic._cli._cli_migrate_worker" in line
        )
        parts = shlex.split(command_line)
        config_index = parts.index("--config")
        config = parts[config_index + 1]
        command = parts[config_index + 2]
        indexed = "--index" in parts
        for index in _script_indices(script):
            args = ["--config", config, command]
            if indexed:
                args.extend(["--index", str(index)])
            result = CliRunner().invoke(migration_worker_cli, args)
            assert result.exit_code == 0, (
                f"{script.name} index {index} failed:\n{result.output}"
            )
            executed.append((command, index if indexed else None))
    assert [command for command, _ in executed] == [
        "metadata",
        *("image" for _ in range(plan.task_count)),
        "seal",
        "finalize",
    ]
    return SimpleNamespace(job_ids=["1"])


def _published_migration_snapshot(tree: Path, stems: tuple[str, ...]) -> dict[str, object]:
    """Capture the migrated scientific/publication state, excluding scheduler control."""
    image_markers: dict[str, object] = {}
    for stem in stems:
        marker = json.loads(
            image_completion_marker_path(tree, "ds", stem).read_text(encoding="utf-8")
        )
        assert valid_image_success(
            tree,
            dataset="ds",
            image_stem=stem,
            work_id=str(marker["work_id"]),
        )
        image_markers[stem] = {
            key: marker[key]
            for key in ("version", "dataset", "image_stem", "work_id", "artifacts")
        }
    stores = {stem: zarr_store_path(tree, "ds", stem) for stem in stems}
    aggregate = valid_aggregate_snapshot(tree)
    completion = valid_run_completion(tree)
    assert aggregate is not None
    assert completion is not None
    return {
        "store_conformance": {
            stem: valid_staged_store(store) for stem, store in stores.items()
        },
        "store_content": {
            stem: _tree_digest(store) for stem, store in stores.items()
        },
        "embedded_tables": {
            stem: _digest(store / MEASUREMENT_TABLE_RELATIVE_PATH)
            for stem, store in stores.items()
        },
        "overlays": {
            stem: _digest(dataset_overlays_dir(tree, "ds") / f"{stem}.png")
            for stem in stems
        },
        "image_markers": image_markers,
        "aggregate_marker": {
            "current": current_aggregate_is_current(tree),
            **{
                key: aggregate[key]
                for key in (
                    "version",
                    "inventory_digest",
                    "finalization_input_digest",
                    "scientific_config_digest",
                    "source_set_digest",
                    "source_image_count",
                )
            },
        },
        "run_completion_marker": {
            key: completion[key]
            for key in (
                "version",
                "mode",
                "status",
                "finalizer_succeeded",
                "inventory_digest",
                "finalization_input_digest",
                "scientific_config_digest",
                "processing_generation",
            )
        },
        "success_counts": current_success_counts(tree),
    }


def _summary_counters(output: str) -> tuple[str, ...]:
    """Keep only the four durable pass-counter lines from a CLI summary."""
    return tuple(
        line.strip()
        for line in output.splitlines()
        if line.lstrip().startswith((
            "Pass 1 (metadata headers, non-image targets):",
            "Pass 2 (per-image .h5 -> .ome.zarr):",
            "Pass 3 (external Parquet -> embedded table):",
            "Pass 4 (store -> overlay PNG):",
        ))
    )


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
    lifecycle = load_slurm_lifecycle(tree)
    assert lifecycle is not None
    assert lifecycle["mode"] == "migrate"
    assert lifecycle["active"] is False
    generation = str(lifecycle["generation"])
    image_seal = json.loads(
        migration_image_seal_path(
            phenotypic_cache_dir(tree), generation
        ).read_text(encoding="utf-8")
    )
    terminal_status = json.loads(
        migration_terminal_status_path(
            phenotypic_cache_dir(tree), generation
        ).read_text(encoding="utf-8")
    )
    assert image_seal["clean"] is True
    assert terminal_status["status"] == "succeeded"
    assert not migration_reclaim_seal_path(
        phenotypic_cache_dir(tree), generation
    ).exists()


def test_local_and_synchronous_slurm_migration_publish_equivalent_runs(
    finished_legacy_run: LegacyRun,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The generated worker chain reaches the same science as local migration.

    A synchronous dispatcher fake runs each emitted worker command in the
    generated metadata -> image -> seal -> finalizer order.  The comparison
    deliberately covers publication authority and user-facing artifacts, not
    the generation-scoped scheduler control files which must differ between
    local and SLURM execution.
    """
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    local_tree = tmp_path / "local"
    slurm_tree = tmp_path / "slurm"
    shutil.copytree(finished_legacy_run.path, local_tree)
    shutil.copytree(finished_legacy_run.path, slurm_tree)

    local = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(local_tree)]
    )
    assert local.exit_code == 0, local.output

    monkeypatch.setattr(
        migrate,
        "submit_migration_slurm_plan",
        lambda plan, **_kwargs: _run_generated_migration_worker_commands(plan),
    )
    slurm = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(slurm_tree),
            "--slurm",
            "slurm_partition=short",
            "--wait",
        ],
    )
    assert slurm.exit_code == 0, slurm.output

    assert _summary_counters(local.output) == _summary_counters(slurm.output)
    local_snapshot = _published_migration_snapshot(local_tree, finished_legacy_run.stems)
    slurm_snapshot = _published_migration_snapshot(slurm_tree, finished_legacy_run.stems)
    assert local_snapshot == slurm_snapshot

    # This is the post-migration CLI consumption seam.  Browse has its own
    # atomic listing and URL-resolution contract on the process branch.
    scanned = scan_store_outputs(slurm_tree)
    assert [dataset.name for dataset in scanned] == ["ds"]
    assert [store.name for store in scanned[0].images] == [
        f"{stem}.ome.zarr" for stem in finished_legacy_run.stems
    ]
    for store in scanned[0].images:
        assert load_image_from_store(store).shape == (128, 128, 3)

    for tree, rerun_args in (
        (local_tree, ["--mode", "migrate", "--output", str(local_tree)]),
        (
            slurm_tree,
            [
                "--mode",
                "migrate",
                "--output",
                str(slurm_tree),
                "--slurm",
                "slurm_partition=short",
                "--wait",
            ],
        ),
    ):
        before = _published_migration_snapshot(tree, finished_legacy_run.stems)
        rerun = CliRunner().invoke(phenotypic_cli, rerun_args)
        assert rerun.exit_code == 0, rerun.output
        assert "converted 0" in rerun.output
        assert _published_migration_snapshot(tree, finished_legacy_run.stems) == before


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
    lifecycle = load_slurm_lifecycle(tree)
    assert lifecycle is not None
    reclaim_seal = json.loads(
        migration_reclaim_seal_path(
            phenotypic_cache_dir(tree), str(lifecycle["generation"])
        ).read_text(encoding="utf-8")
    )
    assert reclaim_seal["deletion_requested"] is True
    assert reclaim_seal["clean"] is True
