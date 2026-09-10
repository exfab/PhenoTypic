"""Both ``--mode migrate`` passes, in order, through the real entry point.

Every test in ``tests/unit/sdk_/test_migration_republishes_state.py`` calls
``migrate_run_hdf_to_zarr`` directly, which is **pass 2 only**. Pass 1 lives
in the CLI driver, so the interaction MIG-15 is about -- pass 1 rewriting the
parquets that pass 2's markers fingerprint -- is exercised by nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil

from click.testing import CliRunner
import pytest

from phenotypic._cli._cli_completion import (
    _current_aggregate_is_current,
    _current_success_counts,
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
    image_record_path,
    load_image_from_store,
    deliverables_dir,
    metadata_migration_authority,
    phenotypic_cache_dir,
    processing_state_path,
    zarr_store_path,
)
from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON, valid_staged_store

from tests.unit.sdk_._migration_fixtures import (
    LegacyRun,
    build_completed_run,
    demote_run_to_hdf,
)


WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt", reason="requires real Windows handle semantics"
)


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


def _run_generated_migration_worker_commands(
    *,
    chunk_scripts: list[Path],
    dispatcher_scripts: list[Path],
    finalizer_script: Path | None = None,
    continuation_dependency_kind: str = "afterany",
    output_dir: Path | None = None,
    generation: str | None = None,
) -> tuple[list[str], None]:
    """Synchronously execute generated worker commands after real chain setup.

    The fake replaces only the scheduler submission primitive. Production code
    has already generated and resolved the dispatcher chain, including its
    ``afterany`` dependencies and finalizer. Commands and indexes come from
    the emitted worker scripts, then the real worker Click entry point consumes
    their immutable config.
    """
    from phenotypic._cli._cli_migrate_worker import migration_worker_cli

    assert finalizer_script is not None
    assert output_dir is not None
    assert generation is not None
    assert continuation_dependency_kind == "afterany"
    assert len(dispatcher_scripts) == len(chunk_scripts) - 1
    for dispatcher in dispatcher_scripts:
        text = dispatcher.read_text(encoding="utf-8")
        assert f"--generation {generation}" in text
        assert "--dependency-kind afterany" in text

    executed: list[tuple[str, int | None]] = []
    task_count: int | None = None
    for script in (*chunk_scripts, finalizer_script):
        command_line = next(
            line.strip()
            for line in script.read_text(encoding="utf-8").splitlines()
            if "-m phenotypic._cli._cli_migrate_worker" in line
        )
        parts = shlex.split(command_line)
        config_index = parts.index("--config")
        config = parts[config_index + 1]
        command = parts[config_index + 2]
        task_count = json.loads(Path(config).read_text(encoding="utf-8"))["task_count"]
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
    assert task_count is not None
    assert [command for command, _ in executed] == [
        "metadata",
        *("image" for _ in range(task_count)),
        "seal",
        "finalize",
    ]
    return ["1", "2"], None


def _retained_source_snapshot(tree: Path, stems: tuple[str, ...]) -> dict[str, object]:
    """Capture retained legacy HDF and immutable metadata input bytes."""
    hdf_dir = tree / "results" / "ds" / "hdf"
    hdf = {stem: hdf_dir / f"{stem}.h5" for stem in stems}
    assert all(path.is_file() for path in hdf.values())
    metadata = tree / "deliverables" / "metadata.csv"
    assert metadata.is_file()
    return {
        "hdf": {stem: _digest(path) for stem, path in hdf.items()},
        "metadata": metadata.read_bytes(),
    }


def _published_migration_snapshot(tree: Path, stems: tuple[str, ...]) -> dict[str, object]:
    """Capture the migrated scientific/publication state, excluding scheduler control."""
    image_markers: dict[str, object] = {}
    for stem in stems:
        marker = json.loads(
            image_record_path(tree, "ds", stem).read_text(encoding="utf-8")
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
    store_conformance = {
        stem: valid_staged_store(store) for stem, store in stores.items()
    }
    assert all(store_conformance.values())
    assert _current_aggregate_is_current(tree) is True
    success_counts = _current_success_counts(tree)
    assert success_counts == (2, 2)
    return {
        "store_conformance": store_conformance,
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
            "current": True,
            **{
                key: aggregate[key]
                for key in (
                    "version",
                    "inventory_digest",
                    "finalization_input_digest",
                    "scientific_config_digest",
                    "source_set_digest",
                    "source_image_count",
                    "required_outputs",
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
        "success_counts": success_counts,
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


@WINDOWS_ONLY
def test_windows_click_fresh_local_migrate_reaches_terminal_success(
    finished_legacy_run: LegacyRun,
) -> None:
    """The real Click entry point completes and publishes Windows authority."""
    tree = finished_legacy_run.path

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )

    assert result.exit_code == 0, result.output
    lifecycle = load_slurm_lifecycle(tree)
    assert lifecycle is not None
    generation = str(lifecycle["generation"])
    terminal = json.loads(
        migration_terminal_status_path(
            phenotypic_cache_dir(tree), generation
        ).read_text(encoding="utf-8")
    )
    assert terminal["status"] == "succeeded"
    assert metadata_migration_authority(tree).status_path.is_file()


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
    assert _current_aggregate_is_current(tree) is True
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
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs
    from phenotypic._cli import _cli_slurm_submission as slurm_submission

    local_tree = tmp_path / "local"
    slurm_tree = tmp_path / "slurm"
    shutil.copytree(finished_legacy_run.path, local_tree)
    shutil.copytree(finished_legacy_run.path, slurm_tree)
    local_sources = _retained_source_snapshot(local_tree, finished_legacy_run.stems)
    slurm_sources = _retained_source_snapshot(slurm_tree, finished_legacy_run.stems)
    assert local_sources == slurm_sources

    local = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(local_tree)]
    )
    assert local.exit_code == 0, local.output

    monkeypatch.setattr(
        slurm_submission,
        "submit_drip_feed_start",
        _run_generated_migration_worker_commands,
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

    expected_first_counters = (
        "Pass 1 (metadata headers, non-image targets): migrated 0 target(s)",
        "Pass 2 (per-image .h5 -> .ome.zarr): converted 2, skipped 0",
        "Pass 3 (external Parquet -> embedded table): migrated 2, skipped 0",
        "Pass 4 (store -> overlay PNG): rendered 0, preserved 2",
    )
    expected_noop_counters = (
        "Pass 1 (metadata headers, non-image targets): migrated 0 target(s)",
        "Pass 2 (per-image .h5 -> .ome.zarr): converted 0, skipped 2",
        "Pass 3 (external Parquet -> embedded table): migrated 0, skipped 2",
        "Pass 4 (store -> overlay PNG): rendered 0, preserved 2",
    )
    assert _summary_counters(local.output) == expected_first_counters
    assert _summary_counters(slurm.output) == expected_first_counters
    assert _retained_source_snapshot(local_tree, finished_legacy_run.stems) == local_sources
    assert _retained_source_snapshot(slurm_tree, finished_legacy_run.stems) == slurm_sources
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
        sources_before = _retained_source_snapshot(tree, finished_legacy_run.stems)
        rerun = CliRunner().invoke(phenotypic_cli, rerun_args)
        assert rerun.exit_code == 0, rerun.output
        assert _summary_counters(rerun.output) == expected_noop_counters
        assert _published_migration_snapshot(tree, finished_legacy_run.stems) == before
        assert _retained_source_snapshot(tree, finished_legacy_run.stems) == sources_before


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
    assert _current_success_counts(legacy_run) == (36, 36)
    assert len(list(dataset_overlays_dir(legacy_run, "ds").glob("*.png"))) == 36
    assert _current_aggregate_is_current(legacy_run) is True
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_hdf_to_zarr._republish_image_marker rewrites the legacy marker "
        "(:614,:647) and writes no record, so valid_image_success is false "
        "for every migrated image. P7 U-10: republish as a record with "
        "provenance='migrated'. Full rationale beside the shared marker in "
        "tests/unit/sdk_/test_migration_republishes_state.py."
    ),
)
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_hdf_to_zarr._republish_image_marker rewrites the legacy marker "
        "(:614,:647) and writes no record, so valid_image_success is false "
        "for every migrated image -- and reclaim gates on exactly that, so "
        "sources are retained. P7 U-10. Full rationale beside the shared "
        "marker in tests/unit/sdk_/test_migration_republishes_state.py."
    ),
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


# ---------------------------------------------------------------------------
# P7 Task 2b -- the v0.17.3 floor shape
# ---------------------------------------------------------------------------

def _build_v0_17_3_tree(workspace: Path) -> Path:
    """Return a tree in the **pre-markers** shape, built by the real CLI.

    MIG-15: nothing in ``tests/`` builds this today. ``make_markerless`` is the
    closest and is **not** it -- it sets ``success_markers_required = False``
    (present, falsey) and **retains content-derived ``work_ids``**, so it is a
    modern tree with its evidence stripped rather than a tree from before the
    concepts existed. A test built on it exercises ``_configured_work_id``'s
    *hit* path, which is the blind spot MIG-10 found.

    The floor shape, per U-6's ruling that detection is by shape and not by a
    version number: ``version="2.0.0"``, **no ``work_ids`` key**, **no**
    ``success_markers_required`` key, ``datasets.<ds>.completed`` populated,
    per-image ``.h5``, no store, no ``image_complete/``.
    """
    from tests.unit.sdk_._migration_fixtures import (
        build_completed_run,
        demote_run_to_hdf,
        run_stems,
    )

    output = build_completed_run(workspace, ("a", "b"))
    stems = run_stems(output)
    demote_run_to_hdf(output, keep_markers=False)

    path = processing_state_path(output)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = "2.0.0"
    config = payload.get("config", {})
    # The two keys whose ABSENCE is the shape. `pop`, not set-to-falsey:
    # signal 5 keys on `work_ids` being absent, and a present-but-empty dict
    # would clear it while describing a tree that never existed.
    config.pop("work_ids", None)
    config.pop("success_markers_required", None)
    payload["config"] = config
    for entry in payload.get("datasets", {}).values():
        entry["completed"] = [f"{stem}.png" for stem in stems]
    path.write_text(json.dumps(payload), encoding="utf-8")

    # `demote_run_to_hdf` writes `LEGACY_METADATA_CSV` into the deliverables,
    # and that snapshot names `img.png` / `img2.png` -- not this run's `a`/`b`.
    # `build_completed_run` was invoked with **no** `--metadata`, so the tree it
    # produced had no snapshot at all; the demotion adds one that contradicts
    # the state beside it. Left in place it admits an image set disjoint from
    # the inputs, and a later resume refuses on continuation compatibility
    # before reading a single record -- which looks exactly like a reuse
    # failure and is not one.
    snapshot = deliverables_dir(output) / "metadata.csv"
    snapshot.unlink(missing_ok=True)
    return output


def test_a_pre_markers_tree_converts_end_to_end(tmp_path: Path) -> None:
    """CAN-7 / U-1. The floor predates markers AND stores.

    This is the shape that must work, not an edge case -- and it is built
    through the **real** HDF migrator, because a hand-planted fixture cannot
    catch the class of drift that let the gap survive the first draft.

    **What this test is actually deciding.** The round-2 reversal argued the
    promoter must be ported because it mints the content-derived id a resume
    re-derives. U-10 dissolved that: a record marked ``PROVENANCE_MIGRATED``
    skips the ``work_id`` comparison entirely (``sdk_/_run_state.py:587``), and
    Task 2 stamps that on every record it writes. The reason U-10 left standing
    is narrower -- that ``datasets.<ds>.completed`` is the only record of what
    finished, so migrate needs it to know which images to publish for.

    But the shipped migrator already seeds ``work_ids`` with
    ``_migration_work_id`` (``_cli_migrate.py:634``), already publishes a
    record per converted ``.h5`` (``_cli_migrate_image.py:589``), and already
    publishes run-completion evidence (``_cli_migrate.py:1264``). So this test
    asks whether any port is needed at all, rather than assuming one is.
    """
    from phenotypic.sdk_ import resolve_run_state

    tree = _build_v0_17_3_tree(tmp_path)
    config = json.loads(
        processing_state_path(tree).read_text(encoding="utf-8")
    )["config"]
    assert config.get("success_markers_required") is None
    assert "work_ids" not in config

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )
    assert result.exit_code == 0, result.output

    state = resolve_run_state(tree, depth="deep")
    assert len(state.images) == 2, (
        f"records were not produced for both images: {sorted(state.images)}"
    )
    assert state.completion == "complete", (
        f"{state.completion}: {state.advisories}"
    )

    # INV-DISCHARGEABLE, on a tree that CAN be discharged.
    #
    # The unit matrix in `test_schema_gate.py` cannot assert this for a
    # pre-markers shape, and not because its fixtures are careless: the
    # discharge of signals 3 and 5 runs through *converting image data*.
    # `_completed_is_fully_consumed` needs a record per named image, and
    # `_ensure_migration_processing_state` builds its inventory from real
    # `.h5` tasks or real `*.ome.zarr` stores (`_cli_migrate.py:604-621`),
    # returning early at `:662` when it finds neither. A schema-shape fixture
    # has neither by construction.
    #
    # This tree does: `demote_run_to_hdf` leaves real per-image `.h5`, migrate
    # converts them, and both mechanisms fire. So the claim lives here, where
    # it can be true, rather than in a matrix that can only ever fail it.
    from phenotypic.sdk_._schema_shape import requires_conversion

    assert requires_conversion(tree) is None, (
        "a pre-markers tree with real image data did not discharge in one "
        "migrate: " + str(requires_conversion(tree))
    )




def _initial_images(root: Path) -> dict[str, list[str]]:
    """Return each dataset's ``initial_images``, sorted, straight from disk."""
    payload = json.loads(
        processing_state_path(root).read_text(encoding="utf-8")
    )
    return {
        name: sorted(entry.get("initial_images", []))
        for name, entry in payload.get("datasets", {}).items()
    }


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TOLERATED, not fixed (user ruling 2026-09-09). A migrated tree is "
        "not continuable and should be restarted, so the admitted-set "
        "comparison this pollution corrupts is never reached for such a tree "
        "-- the migrated-tree refusal fires first. The defect is real (one "
        "rule applied at two of its three sites in "
        "`_ensure_migration_processing_state`) and is left alone because "
        "repairing it rewrites machine state on the one phase that cannot be "
        "rolled back, for zero behavioural gain. STRICT so that a future "
        "repair turns this red and has to be acknowledged rather than "
        "silently changing what the refusal above is documenting."
    ),
)
def test_migrate_does_not_pollute_the_admitted_image_set(
    tmp_path: Path,
) -> None:
    """A migrated tree must still be continuable. Today it is not.

    ``initial_images`` is the run's own record of which files it admitted, and
    it is the **only** thing a continuation compares against:
    ``_validate_resume_input_images`` prefers it whenever it is non-empty
    (``phenotypicCLI.py:752-755``) and matches it against a fresh scan of
    ``--input`` by ``image.name``.

    ``_ensure_migration_processing_state`` unions ``state_names`` into it
    (``_cli_migrate.py:653``). On a tree whose ``work_ids`` does not already
    carry the image *filename* -- the floor shape, where ``work_ids`` is
    absent entirely -- ``state_names`` falls back to the bare **stem**
    (``:638-648``), so the set becomes ``{a.png, b.png, a, b}``. The two stems
    match no file on disk, and every later ``--mode full`` refuses with
    *"Image set mismatch"* before reading a single record.

    **This is why the CAN-7 investigation kept coming back inconclusive**: the
    resume stops on the admitted set, which is upstream of any ``work_id``
    comparison, so no amount of reasoning about provenance could reach it.

    Non-floor trees are unaffected: their ``work_ids`` carries ``a.png``, the
    lookup at ``:641-646`` hits, ``state_names`` holds filenames, and the union
    is a no-op. The parametrisation below asserts both halves, so a fix that
    silences the floor case by disabling the union everywhere would fail the
    other arm.
    """
    tree = _build_v0_17_3_tree(tmp_path)
    before = _initial_images(tree)
    assert before == {"ds": ["a.png", "b.png"]}, before

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )
    assert result.exit_code == 0, result.output

    assert _initial_images(tree) == before, (
        "migrate added entries to the admitted image set; every later "
        "continuation now refuses on an image set that names files which "
        "have never existed on disk"
    )


def test_migrate_leaves_a_modern_trees_admitted_set_alone(
    tmp_path: Path,
) -> None:
    """The control, and the arm a careless fix would break.

    A tree whose ``work_ids`` already carries the image filename resolves
    ``state_names`` to that filename, so the union is a no-op and this passes
    today. It is here so that a fix which simply stops unioning cannot be
    mistaken for a correct one: the fallback still has to reach an
    ``initial_images`` that genuinely holds nothing for a stem.
    """
    from tests.unit.sdk_._migration_fixtures import (
        build_completed_run,
        demote_run_to_hdf,
    )

    tree = build_completed_run(tmp_path, ("a", "b"))
    demote_run_to_hdf(tree, keep_markers=False)
    (deliverables_dir(tree) / "metadata.csv").unlink(missing_ok=True)
    before = _initial_images(tree)

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )
    assert result.exit_code == 0, result.output

    assert _initial_images(tree) == before, before


def test_a_migrated_tree_refuses_continuation_and_names_the_remedy(
    tmp_path: Path,
) -> None:
    """The ruled behaviour (user, 2026-09-09), and the refusal contract.

    A migrated tree is legitimately not continuable -- migration rebuilds the
    admitted image set from the tree instead of from the original run's record
    of it -- and the ruling is that such a tree *"should be considered as if
    not ran then, and do a full restart"*.

    What the code owed was not a repair but an **actionable** refusal. It
    already declined; it declined with *"The input image set has changed"*,
    which is true of the symptom and useless as advice. This asserts the
    refusal now names `--restart`.

    **Told, never done.** Clearing machine state and reprocessing every image
    is a destructive step costing hours; firing it automatically from a
    condition the user did not ask about is the hidden state transition U-7
    refused, and worse than U-7's case, which destroyed nothing.
    """
    tree = _build_v0_17_3_tree(tmp_path)
    migrated = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )
    assert migrated.exit_code == 0, migrated.output
    assert (
        phenotypic_cache_dir(tree) / "migration_manifest.json"
    ).is_file(), "the signal this refusal keys on is absent"

    resumed = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(tmp_path / "pipeline.json"),
            "--input",
            str(tmp_path / "ds"),
            "-o",
            str(tree),
            "--njobs",
            "1",
            "--skip-validation",
            "--force-local",
        ],
    )

    assert resumed.exit_code == 1, resumed.output
    assert "--mode migrate" in resumed.output
    assert "--restart" in resumed.output, (
        "the refusal must name the command that fixes it; a refusal the user "
        f"cannot act on is the bug class this change exists to remove\\n{resumed.output}"
    )


def test_an_ordinary_image_set_change_keeps_its_own_message(
    tmp_path: Path,
) -> None:
    """The control. A non-migrated tree must not be told to restart.

    An image-set mismatch on a forward tree usually means the user moved or
    deleted inputs, and `--restart` there would destroy a run that is fine.
    The migrated-tree advice is gated on the manifest precisely so it cannot
    reach this case.
    """
    from tests.unit.sdk_._migration_fixtures import build_completed_run

    tree = build_completed_run(tmp_path, ("a", "b"))
    assert not (
        phenotypic_cache_dir(tree) / "migration_manifest.json"
    ).is_file()
    (tmp_path / "ds" / "a.png").unlink()

    resumed = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(tmp_path / "pipeline.json"),
            "--input",
            str(tmp_path / "ds"),
            "-o",
            str(tree),
            "--njobs",
            "1",
            "--skip-validation",
            "--force-local",
        ],
    )

    assert resumed.exit_code == 1, resumed.output
    assert "--restart" not in resumed.output, (
        "a forward tree with a changed input set was told to restart, which "
        f"would destroy a run that is fine\\n{resumed.output}"
    )


# ---------------------------------------------------------------------------
# MIG-23: the four `republish_aggregate` returns mean two different things
# ---------------------------------------------------------------------------
#
# `republish_aggregate` (`sdk_/_hdf_to_zarr.py:694`) returns False from four
# places. Two are the **documented no-op** its own docstring describes -- "a
# legacy tree with no markers is a documented no-op, not an exception (ledger
# MIG-23)" -- and two are genuine **failures**. `_cli_migrate.py:1059` raises on
# all four, which is the only safe reading available to a caller that cannot
# tell them apart.
#
# The two `xfail(strict=True)` markers here were REMOVED when the fix
# landed, not because they were inconvenient: strict is what turned
# their passing into a failure that had to be acknowledged, and
# removing the mark IS the acknowledgement. `republish_aggregate` now
# raises on its faults (`_hdf_to_zarr.py:742`, `:790`) and returns
# False only for a no-op, of which there proved to be THREE, not two:
# the empty authorized set is decided explicitly at `:782` rather than
# by catching `publish_aggregate_snapshot`'s message.
#
# These four tests pin the CONTRACT rather than the boolean: what matters is
# what `--mode migrate` reports, not what an internal helper returns. A fix
# that makes all four non-fatal would pass the schema-gate test and be strictly
# worse than the bug, so the two fatal arms are asserted as hard as the two
# no-op ones.


def _legacy_tree_without_markers(tmp_path: Path) -> Path:
    """A migrated-shape tree with state but nothing marker-authorized."""
    tree = build_completed_run(tmp_path, ("a", "b"))
    demote_run_to_hdf(tree, keep_markers=False)
    (deliverables_dir(tree) / "metadata.csv").unlink(missing_ok=True)
    return tree


def test_a_tree_with_no_authorized_markers_migrates_cleanly(
    tmp_path: Path,
) -> None:
    """No-op arm 1: `success_markers_required` falsey, or state absent.

    `republish_aggregate`'s docstring names this population directly -- *"a
    pre-markers archive is a likely migration subject; aborting there would
    leave the stores written and the run reported as failed"* -- which is
    exactly the outcome asserted against here.
    """
    tree = _legacy_tree_without_markers(tmp_path)

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )

    assert result.exit_code == 0, result.output
    assert "returned false" not in result.output


def test_a_tree_with_zero_images_migrates_cleanly(tmp_path: Path) -> None:
    """No-op arm 2: state present and authorized, but nothing to aggregate.

    Distinct from arm 1 on purpose. A fix that keys only on
    `success_markers_required` clears arm 1 and leaves this one raising, and
    the schema-gate matrix would still be blocked -- its shapes carry
    `success_markers_required: True` with no images.
    """
    tree = build_completed_run(tmp_path, ("a", "b"))
    demote_run_to_hdf(tree, keep_markers=False)
    (deliverables_dir(tree) / "metadata.csv").unlink(missing_ok=True)
    payload = json.loads(
        processing_state_path(tree).read_text(encoding="utf-8")
    )
    payload["config"]["success_markers_required"] = True
    processing_state_path(tree).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )

    assert result.exit_code == 0, result.output


def test_a_corrupt_processing_state_still_fails_the_migrate(
    tmp_path: Path,
) -> None:
    """Failure arm 1, and **this must keep failing**.

    `republish_aggregate` returns False here from
    `except (KeyError, TypeError, ValueError)` -- a *corrupt* state, not an
    absent one. A fix that makes every False non-fatal turns this into a
    successful migrate over a tree nobody can read, on the one phase that
    cannot be rolled back by reverting code. That is strictly worse than the
    bug being fixed, and it is why this arm is asserted as hard as the no-op
    ones.

    Passes today. It is here so that it cannot start failing quietly.
    """
    tree = build_completed_run(tmp_path, ("a", "b"))
    demote_run_to_hdf(tree, keep_markers=False)
    processing_state_path(tree).write_text("{truncated", encoding="utf-8")

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )

    assert result.exit_code != 0, (
        "a migrate over an unreadable processing state reported success"
    )


def test_an_unwritable_deliverables_dir_still_fails_the_migrate(
    tmp_path: Path,
) -> None:
    """Failure arm 2, and this must keep failing too.

    `republish_aggregate`'s outer `except (OSError, RuntimeError, ValueError)`
    covers a publication that was attempted and failed -- as distinct from one
    that was correctly not attempted. Same reasoning as arm 1: silence here
    would report success over a tree whose deliverables were never written.
    """
    tree = build_completed_run(tmp_path, ("a", "b"))
    demote_run_to_hdf(tree, keep_markers=False)
    (deliverables_dir(tree) / "metadata.csv").unlink(missing_ok=True)
    blocked = deliverables_dir(tree) / "master_measurements.parquet"
    blocked.unlink(missing_ok=True)
    blocked.mkdir(parents=True, exist_ok=True)

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(tree)]
    )

    assert result.exit_code != 0, (
        "a migrate that could not write its deliverables reported success"
    )
