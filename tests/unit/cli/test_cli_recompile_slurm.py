"""Unit tests for recompile-specific SLURM task scripts and worker."""

from __future__ import annotations

import json
import os
import subprocess
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
from click.testing import CliRunner

from phenotypic.sdk_ import (
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
    progress_dir,
    slurm_scripts_dir,
    zarr_store_path,
)
from phenotypic.schema import EXPERIMENT, IMAGE

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Recompile SLURM worker tests require non-Windows paths",
)


def _write_parquet(
    path: Path,
    values: list[int],
    *,
    image_names: list[str] | None = None,
) -> None:
    """Write a tiny measurement Parquet with optional image identities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[int] | list[str]] = {"Size_Area": values}
    if image_names is not None:
        columns[str(IMAGE.IMAGE_NAME)] = image_names
    pl.DataFrame(columns).write_parquet(path)


def test_recompile_slurm_dispatcher_submits_and_writes_metadata(
    tmp_path: Path,
) -> None:
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text("Metadata_Dataset,Treatment\nplate_a,drug\n")
    _write_parquet(
        output_dir / "results" / "plate_a" / "measurements" / "img1.parquet",
        [1],
    )
    scripts = [
        slurm_scripts_dir(output_dir) / "recompile" / "chunk0.sh",
        slurm_scripts_dir(output_dir) / "recompile" / "chunk1.sh",
    ]
    generated_tasks: list[dict[str, object]] = []

    def _fake_generate(
        tasks: list[dict[str, object]],
        output_dir: Path,
        slurm_args: dict[str, object],
        array_limit: int,
        attempt_id: str | None = None,
    ) -> list[Path]:
        generated_tasks.extend(tasks)
        assert slurm_args == {"slurm_partition": "compute"}
        assert array_limit == 77
        manifest_path = (
            progress_dir(output_dir)
            / "recompile"
            / "attempts"
            / str(attempt_id)
            / "task_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        return scripts

    submission = SimpleNamespace(job_ids=["12345"], flat_scripts=scripts)

    with (
        patch(
            "phenotypic.phenotypicCLI.get_slurm_array_limit",
            return_value=77,
        ) as mock_limit,
        patch(
            "phenotypic.phenotypicCLI.generate_recompile_slurm_scripts",
            side_effect=_fake_generate,
        ) as mock_generate,
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
            return_value=submission,
        ) as mock_submit,
        patch("phenotypic._cli._dashboard.generate_dashboard") as mock_dashboard,
        patch(
            "phenotypic.phenotypicCLI._wait_for_recompile_finalizer_status"
        ) as mock_wait,
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=metadata_csv,
            include_dataset_column=False,
            overlay_alpha=0.4,
            checkpoint_interval=10,
            slurm_args={"slurm_partition": "compute"},
            wait=False,
        )

    mock_limit.assert_called_once_with()
    mock_generate.assert_called_once()
    mock_submit.assert_called_once()
    assert mock_submit.call_args.kwargs["flat_chunk_scripts"] == scripts
    assert mock_submit.call_args.kwargs["output_dir"] == output_dir
    assert mock_submit.call_args.kwargs["slurm_args"] == {
        "slurm_partition": "compute"
    }
    mock_dashboard.assert_called_once_with(output_dir, execution_mode="slurm")
    mock_wait.assert_not_called()

    assert generated_tasks[-1]["task_type"] == "finalize"
    assert generated_tasks[-1]["metadata_csv"] == str(metadata_csv)

    metadata = json.loads(
        (progress_dir(output_dir) / "job_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["execution_mode"] == "slurm"
    assert metadata["datasets"] == {"plate_a": {"total": 1, "images": ["img1"]}}
    assert metadata["chunk_scripts"] == [str(path) for path in scripts]
    assert metadata["chunk_job_ids"] == {"0": "12345"}
    assert metadata["include_dataset_column"] is False
    assert metadata["metadata_csv"] == str(metadata_csv)
    assert metadata["input_path"] == str(output_dir)
    assert metadata["recompile"]["task_manifest"] == str(
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / metadata["recompile"]["attempt_id"]
        / "task_manifest.json"
    )
    assert metadata["recompile"]["finalizer_task_index"] == len(generated_tasks) - 1


def test_recompile_slurm_dispatcher_waits_for_finalizer(
    tmp_path: Path,
) -> None:
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    _write_parquet(
        output_dir / "results" / "plate_a" / "measurements" / "img1.parquet",
        [1],
    )
    script = slurm_scripts_dir(output_dir) / "recompile" / "chunk0.sh"
    finalizer_indices: list[int] = []

    def _fake_generate(
        tasks: list[dict[str, object]],
        output_dir: Path,
        slurm_args: dict[str, object],
        array_limit: int,
        attempt_id: str | None = None,
    ) -> list[Path]:
        assert slurm_args == {"slurm_partition": "compute"}
        assert array_limit == 100
        finalizer_indices.append(len(tasks) - 1)
        manifest_path = (
            progress_dir(output_dir)
            / "recompile"
            / "attempts"
            / str(attempt_id)
            / "task_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        return [script]

    submission = SimpleNamespace(job_ids=["12345"], flat_scripts=[script])

    with (
        patch("phenotypic.phenotypicCLI.get_slurm_array_limit", return_value=100),
        patch(
            "phenotypic.phenotypicCLI.generate_recompile_slurm_scripts",
            side_effect=_fake_generate,
        ),
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
            return_value=submission,
        ),
        patch("phenotypic._cli._dashboard.generate_dashboard"),
        patch(
            "phenotypic.phenotypicCLI._wait_for_recompile_finalizer_status"
        ) as mock_wait,
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={"slurm_partition": "compute"},
            wait=True,
        )

    assert mock_wait.call_count == 1
    assert mock_wait.call_args.args == (output_dir, finalizer_indices[0])
    assert mock_wait.call_args.kwargs["recompile_finalizer_status_path"].name == (
        f"task_{finalizer_indices[0]}.json"
    )


def test_recompile_slurm_dispatcher_falls_back_to_local_when_no_scripts(
    tmp_path: Path,
) -> None:
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    (output_dir / "results" / "plate_a" / "measurements").mkdir(parents=True)

    with (
        patch(
            "phenotypic.phenotypicCLI.generate_recompile_slurm_scripts",
            return_value=[],
        ),
        patch(
            "phenotypic.phenotypicCLI._handle_recompile",
        ) as mock_local,
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
        ) as mock_submit,
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=0,
            slurm_args={"slurm_partition": "compute"},
            wait=False,
        )

    mock_local.assert_called_once_with(
        output_dir, None, True, 0.3, -1, no_qc=False
    )
    mock_submit.assert_not_called()


def test_wait_for_recompile_finalizer_status_completed_and_failed(
    tmp_path: Path,
) -> None:
    from phenotypic.phenotypicCLI import _wait_for_recompile_finalizer_status

    output_dir = tmp_path / "out"
    status_dir = progress_dir(output_dir) / "recompile" / "status"
    status_dir.mkdir(parents=True)
    (status_dir / "task_3.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )

    _wait_for_recompile_finalizer_status(
        output_dir, 3, poll_interval=0.001, timeout=0.01
    )

    (status_dir / "task_3.json").write_text(
        json.dumps({"status": "failed", "error": "finalize failed"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="finalize failed"):
        _wait_for_recompile_finalizer_status(
            output_dir, 3, poll_interval=0.001, timeout=0.01
        )


def test_generate_recompile_scripts_write_manifest_and_worker_arrays(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_FINALIZE,
        TASK_MEASUREMENTS,
        TASK_OVERLAY,
        build_recompile_tasks,
        generate_recompile_slurm_scripts,
        recompile_task_status_path,
    )

    output_dir = tmp_path / "out"
    _write_parquet(
        output_dir
        / "results"
        / "plate_a"
        / "measurements"
        / "_dataset_aggregated.parquet",
        [1],
        image_names=["plate_a"],
    )
    _write_parquet(
        output_dir
        / "results"
        / "plate_a"
        / "measurements"
        / "ignored.parquet",
        [99],
    )
    _write_parquet(
        output_dir / "results" / "plate_b" / "measurements" / "img_b1.parquet",
        [2],
    )
    _write_parquet(
        output_dir / "results" / "plate_b" / "measurements" / "img_b2.parquet",
        [3],
    )
    # Overlay discovery walks the per-image OME-Zarr stores. A store is a
    # DIRECTORY, so this fixture has to be one -- a stub file would be
    # skipped and the assertion below would pass for the wrong reason.
    store_path = zarr_store_path(output_dir, "plate_b", "img_b1")
    store_path.mkdir(parents=True)
    (store_path / "zarr.json").write_text("{}", encoding="utf-8")

    tasks = build_recompile_tasks(
        output_dir=output_dir,
        dataset_names=["plate_a", "plate_b"],
        include_dataset_column=False,
        overlay_alpha=0.42,
        shard_size=1,
    )

    assert [task["task_type"] for task in tasks].count(TASK_FINALIZE) == 1
    assert tasks[-1]["task_type"] == TASK_FINALIZE

    measurement_tasks = [
        t for t in tasks if t["task_type"] == TASK_MEASUREMENTS
    ]
    assert measurement_tasks[0]["files"] == [
        str(
            output_dir
            / "results"
            / "plate_a"
            / "measurements"
            / "_dataset_aggregated.parquet"
        )
    ]
    assert all(t["include_dataset_column"] is False for t in measurement_tasks)

    overlay_tasks = [t for t in tasks if t["task_type"] == TASK_OVERLAY]
    assert overlay_tasks == [
        {
            "task_type": TASK_OVERLAY,
            "dataset_name": "plate_b",
            "store_path": str(store_path),
            "overlay_alpha": 0.42,
        }
    ]

    finalizer = tasks[-1]
    assert finalizer["dataset_names"] == ["plate_a", "plate_b"]
    assert finalizer["include_dataset_column"] is False
    assert finalizer["metadata_csv"] is None
    assert finalizer["expected_non_finalizer_tasks"] == len(tasks) - 1

    scripts = generate_recompile_slurm_scripts(
        tasks=tasks,
        output_dir=output_dir,
        slurm_args={},
        array_limit=2,
        attempt_id="attempt-script-args",
    )

    manifest_path = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / "attempt-script-args"
        / "task_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [
        {key: value for key, value in task.items() if key != "slurm_generation"}
        for task in manifest["tasks"]
    ] == tasks
    assert all(
        task["slurm_generation"] == "attempt-script-args"
        for task in manifest["tasks"]
    )
    assert len(scripts) == 3
    assert (
        scripts[-1].read_text(encoding="utf-8").count("_cli_recompile_worker")
        == 1
    )
    assert '--task-index "$CURRENT_TASK_INDEX"' in scripts[-1].read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-0" in scripts[-1].read_text(encoding="utf-8")
    script_text = scripts[0].read_text(encoding="utf-8")
    assert "+    --slurm-generation" not in script_text
    assert "--slurm-generation attempt-script-args" in script_text
    assert "--attempt-id attempt-script-args" in script_text
    assert "--terminal-status-path" in script_text
    assert str(
        recompile_task_status_path(manifest_path, len(tasks) - 1)
    ) in script_text


def test_measurement_worker_writes_shard_with_dataset_and_image_file(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_recompile_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "measurement-worker"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    img2 = output_dir / "results" / "plate_a" / "measurements" / "img2.parquet"
    img1 = output_dir / "results" / "plate_a" / "measurements" / "img1.parquet"
    _write_parquet(img2, [20])
    _write_parquet(img1, [10])
    manifest_path = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "shard_id": 7,
                        "files": [str(img2), str(img1)],
                        "include_dataset_column": True,
                        "slurm_generation": generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "--output-dir",
            str(output_dir),
            "--task-manifest",
            str(manifest_path),
            "--task-index",
            "0",
            "--slurm-generation",
            generation,
            "--attempt-id",
            generation,
        ],
    )

    assert result.exit_code == 0, result.output
    shard = pl.read_parquet(
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "measurement_shards"
        / "shard_7.parquet"
    )
    assert shard.sort("Size_Area").select(
        [str(EXPERIMENT.DATASET), str(IMAGE.IMAGE_NAME), "Size_Area"]
    ).to_dicts() == [
        {
            str(EXPERIMENT.DATASET): "plate_a",
            str(IMAGE.IMAGE_NAME): "img1",
            "Size_Area": 10,
        },
        {
            str(EXPERIMENT.DATASET): "plate_a",
            str(IMAGE.IMAGE_NAME): "img2",
            "Size_Area": 20,
        },
    ]
    status = json.loads(
        (
            manifest_path.parent / "status" / "task_0.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"


def test_overlay_worker_records_save_failure_as_completed_nonfatal(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_recompile_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "overlay-worker"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    store_path = zarr_store_path(output_dir, "plate_a", "img1")
    store_path.mkdir(parents=True)
    (store_path / "zarr.json").write_text("{}", encoding="utf-8")
    manifest_path = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "overlay",
                        "dataset_name": "plate_a",
                        "store_path": str(store_path),
                        "overlay_alpha": 0.7,
                        "slurm_generation": generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # The subject is the worker's failure handling, not store IO, so the
    # loader is stubbed out entirely rather than a real store written.
    with (
        patch(
            "phenotypic._cli._cli_recompile_worker.load_image_from_store",
            return_value=object(),
        ),
        patch(
            "phenotypic._cli._cli_output_manager.OutputManager.save_overlay",
            side_effect=RuntimeError("png failed"),
        ),
    ):
        result = CliRunner().invoke(
            main,
            [
                "--output-dir",
                str(output_dir),
                "--task-manifest",
                str(manifest_path),
                "--task-index",
                "0",
                "--slurm-generation",
                generation,
                "--attempt-id",
                generation,
            ],
        )

    assert result.exit_code == 0, result.output
    status = json.loads(
        (
            manifest_path.parent / "status" / "task_0.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"
    assert status["overlay_failed"] is True
    assert "png failed" in status["error"]


def test_finalizer_writes_master_outputs_and_rebuilds_dashboard(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_recompile_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "finalizer-worker"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    attempt_dir = (
        progress_dir(output_dir) / "recompile" / "attempts" / generation
    )
    shard_dir = attempt_dir / "measurement_shards"
    _write_parquet(shard_dir / "shard_1.parquet", [2])
    _write_parquet(shard_dir / "shard_0.parquet", [1])
    status_dir = attempt_dir / "status"
    status_dir.mkdir(parents=True)
    (status_dir / "task_0.json").write_text(
        json.dumps({"status": "completed", "task_type": "measurements"}),
        encoding="utf-8",
    )
    manifest_path = attempt_dir / "task_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "slurm_generation": generation,
                    },
                    {
                        "task_type": "finalize",
                        "dataset_names": ["plate_a"],
                        "include_dataset_column": True,
                        "metadata_csv": None,
                        "expected_non_finalizer_tasks": 1,
                        "slurm_generation": generation,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with (
        patch(
            "phenotypic._cli._dashboard._manifest_builder.build_manifest"
        ) as mock_manifest,
        patch(
            "phenotypic._cli._dashboard._generator.generate_dashboard"
        ) as mock_dashboard,
        patch(
            "phenotypic._cli._cli_output_manager._load_pipeline_from_output_dir",
            return_value=None,
        ),
    ):
        result = CliRunner().invoke(
            main,
            [
                "--output-dir",
                str(output_dir),
                "--task-manifest",
                str(manifest_path),
                "--task-index",
                "1",
                "--slurm-generation",
                generation,
                "--attempt-id",
                generation,
            ],
        )

    assert result.exit_code == 0, result.output
    assert master_measurements_csv_path(output_dir).exists()
    assert master_measurements_parquet_path(output_dir).exists()
    # Recompile finalizer also seeds the GUI's editable measurements copy.
    assert measurements_csv_path(output_dir).exists()
    assert measurements_parquet_path(output_dir).exists()
    assert pl.read_csv(master_measurements_csv_path(output_dir))[
        "Size_Area"
    ].to_list() == [
        1,
        2,
    ]
    assert (
        pl.read_csv(measurements_csv_path(output_dir))["Size_Area"].to_list()
        == [1, 2]
    )
    mock_manifest.assert_called_once()
    mock_dashboard.assert_called_once_with(output_dir, execution_mode="local")
    status = json.loads(
        (
            manifest_path.parent / "status" / "task_1.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"


def test_finalizer_blocks_publication_on_unknown_failed_task(
    tmp_path: Path,
) -> None:
    """Bootstrap failures abort master and post publication."""
    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "blocked-finalizer"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    task = {"expected_non_finalizer_tasks": 1}
    with (
        patch.object(
            worker,
            "_wait_for_non_finalizer_statuses",
            return_value=[{"task_type": "unknown", "status": "failed"}],
        ),
        patch.object(worker, "_write_master_outputs_from_shards") as master,
        patch.object(worker, "_run_post_master_steps") as post,
        pytest.raises(RuntimeError, match="non-finalizer recompile task"),
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest,
            task,
            slurm_generation=generation,
        )

    master.assert_not_called()
    post.assert_not_called()


def test_finalizer_does_not_publish_after_master_parquet_failure(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A mixed master CSV/Parquet generation cannot receive fresh authority."""
    import shutil

    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import (
        aggregate_publication_marker_path,
        master_measurements_csv_path,
        master_measurements_parquet_path,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    generation = "master-parquet-failure"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    attempt_dir = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
    )
    shard_dir = attempt_dir / "measurement_shards"
    _write_parquet(shard_dir / "shard_0.parquet", [999999])
    manifest_path = attempt_dir / "task_manifest.json"
    manifest_path.write_text(
        json.dumps({"tasks": []}),
        encoding="utf-8",
    )
    marker_path = aggregate_publication_marker_path(output_dir)
    marker_before = marker_path.read_bytes()
    parquet_before = master_measurements_parquet_path(output_dir).read_bytes()
    real_atomic_write = worker.atomic_write_with_writer

    def _fail_master_parquet(
        path: Path,
        writer: object,
        **kwargs: object,
    ) -> None:
        if Path(path) == master_measurements_parquet_path(output_dir):
            raise OSError("simulated master Parquet failure")
        real_atomic_write(path, writer, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(
            worker,
            "atomic_write_with_writer",
            _fail_master_parquet,
        ),
        patch.object(worker, "_run_post_master_steps"),
        patch.object(worker, "_regenerate_recompile_dashboard"),
        pytest.raises(OSError, match="simulated master Parquet failure"),
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest_path,
            {"expected_non_finalizer_tasks": 0},
            slurm_generation=generation,
        )

    assert pl.read_csv(master_measurements_csv_path(output_dir))[
        "Size_Area"
    ].to_list() == [999999]
    assert (
        master_measurements_parquet_path(output_dir).read_bytes()
        == parquet_before
    )
    assert marker_path.read_bytes() == marker_before


def test_slurm_recompile_schedules_table_bound_to_missing_overlay(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Submit-time invalid overlay authority is recoverable array work."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        TASK_OVERLAY,
        build_recompile_tasks,
    )
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stems = run_stems(output_dir)
    missing_stem = stems[0]
    (
        dataset_overlays_dir(output_dir, DATASET) / f"{missing_stem}.png"
    ).unlink()

    tasks = build_recompile_tasks(
        output_dir,
        [DATASET],
        include_dataset_column=True,
        overlay_alpha=0.3,
        shard_size=1,
    )

    measurement_files = {
        Path(path)
        for task in tasks
        if task["task_type"] == TASK_MEASUREMENTS
        for path in task["files"]
    }
    assert measurement_files == {
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
        for stem in stems
    }
    table = (
        zarr_store_path(output_dir, DATASET, missing_stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    measurement_task = next(
        task
        for task in tasks
        if task["task_type"] == TASK_MEASUREMENTS
        and str(table) in task["files"]
    )
    assert measurement_task["overlay_repairs"] == [
        {
            "dataset_name": DATASET,
            "store_path": str(table.parents[2]),
            "table_path": str(table),
            "overlay_alpha": 0.3,
        }
    ]
    assert not any(
        task["task_type"] == TASK_OVERLAY
        and task.get("store_path") == str(table.parents[2])
        for task in tasks
    )


def test_slurm_overlay_worker_restores_marker_authority(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A successful overlay repair republishes its complete image marker."""
    import shutil

    from phenotypic._cli._cli_completion import (
        authorized_measurement_sources,
        valid_image_success,
    )
    from phenotypic._cli._cli_recompile_worker import (
        _restore_overlay_marker_authority,
        _run_overlay_task,
    )
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
    )
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.unlink()
    generation = "overlay-authority"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )

    overlay_task = {
        "task_type": "overlay",
        "dataset_name": DATASET,
        "store_path": str(store),
        "overlay_alpha": 0.6,
        "restore_marker_authority": True,
    }
    result = _run_overlay_task(
        output_dir,
        overlay_task,
        slurm_generation=generation,
    )

    assert result == {"status": "completed", "overlay_failed": False}
    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    sources = authorized_measurement_sources(output_dir)
    assert sources is not None
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    assert table in sources

    # A changed non-overlay artifact must never be re-fingerprinted as valid.
    frame = pl.read_parquet(table).with_columns(
        pl.lit("changed").alias("Metadata_ReviewProbe")
    )
    frame.write_parquet(table)
    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    task_manifest = tmp_path / "recompile-task-manifest.json"
    task_manifest.write_text(
        json.dumps({"tasks": [overlay_task]}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="non-overlay artifact changed"):
        _restore_overlay_marker_authority(output_dir, task_manifest)

    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )


def test_finalizer_refreshes_nested_overlay_repair_authority(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Finalization sees repairs nested in a co-located measurement task."""
    import shutil

    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_recompile_worker import (
        _restore_overlay_marker_authority,
    )
    from phenotypic.sdk_ import dataset_overlays_dir
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.write_bytes(overlay.read_bytes() + b"repaired")
    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    task_manifest = tmp_path / "recompile-task-manifest.json"
    task_manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "overlay_repairs": [
                            {
                                "dataset_name": DATASET,
                                "store_path": str(store),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _restore_overlay_marker_authority(output_dir, task_manifest)

    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )

def test_measurement_worker_derives_embedded_image_names_from_store(
    tmp_path: Path,
) -> None:
    """SLURM shards preserve the identity of each fixed-name embedded table."""
    from phenotypic._cli._cli_recompile_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    output_dir = tmp_path / "out"
    generation = "embedded-name-worker"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    tables = [
        zarr_store_path(output_dir, "plate_a", stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
        for stem in ("img2", "img1")
    ]
    for value, table in zip((20, 10), tables, strict=True):
        _write_parquet(table, [value])
    manifest_path = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "shard_id": 8,
                        "files": [str(table) for table in tables],
                        "include_dataset_column": True,
                        "slurm_generation": generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "phenotypic._cli._cli_recompile_tables.recompile_embedded_measurement_table"
    ):
        result = CliRunner().invoke(
            main,
            [
                "--output-dir",
                str(output_dir),
                "--task-manifest",
                str(manifest_path),
                "--task-index",
                "0",
                "--slurm-generation",
                generation,
                "--attempt-id",
                generation,
            ],
        )

    assert result.exit_code == 0, result.output
    shard = pl.read_parquet(
        manifest_path.parent / "measurement_shards" / "shard_8.parquet"
    )
    assert shard.sort("Size_Area")[str(IMAGE.IMAGE_NAME)].to_list() == [
        "img1",
        "img2",
    ]



def test_finalizer_overlay_refresh_locks_store_before_lifecycle(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Final overlay refresh cannot invert the worker store/lifecycle order."""
    import shutil
    from contextlib import contextmanager
    from typing import Iterator

    import phenotypic._cli._cli_recompile_slurm_scripts as scripts
    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic.sdk_ import dataset_overlays_dir
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.write_bytes(overlay.read_bytes() + b"repaired")
    manifest_path = tmp_path / "task-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "overlay_repairs": [
                            {
                                "dataset_name": DATASET,
                                "store_path": str(store),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    lifecycle_active = False
    store_active = False
    inverted = False
    lifecycle_entered_inside_store = False

    @contextmanager
    def _lifecycle_guard(*args: object, **kwargs: object) -> Iterator[None]:
        nonlocal lifecycle_active, lifecycle_entered_inside_store
        lifecycle_entered_inside_store |= store_active
        lifecycle_active = True
        try:
            yield
        finally:
            lifecycle_active = False

    @contextmanager
    def _store_lock(*args: object, **kwargs: object) -> Iterator[None]:
        nonlocal store_active, inverted
        inverted |= lifecycle_active
        store_active = True
        try:
            yield
        finally:
            store_active = False

    with (
        patch.object(
            worker,
            "generation_publication_guard",
            _lifecycle_guard,
        ),
        patch.object(scripts, "exclusive_path_lock", _store_lock),
        patch.object(
            worker,
            "_wait_for_non_finalizer_statuses",
            return_value=[],
        ),
        patch.object(
            worker,
            "_write_master_outputs_from_shards",
            return_value=None,
        ),
        patch.object(worker, "_run_post_master_steps"),
        patch.object(worker, "_regenerate_recompile_dashboard"),
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest_path,
            {"expected_non_finalizer_tasks": 0},
            slurm_generation="lock-order",
        )

    assert not inverted
    assert lifecycle_entered_inside_store


def test_superseded_finalizer_cannot_refresh_overlay_marker(
    _completed_run_two: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final marker refresh is fenced before a stale worker can mutate it."""
    import shutil

    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
    )
    from phenotypic.sdk_ import (
        dataset_overlays_dir,
        image_completion_marker_path,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.write_bytes(b"replacement overlay bytes")
    marker_path = image_completion_marker_path(output_dir, DATASET, stem)
    marker_before = marker_path.read_bytes()
    generation = "superseded-overlay-finalizer"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    assert deactivate_generation(output_dir, generation)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    manifest_path = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "overlay",
                        "dataset_name": DATASET,
                        "store_path": str(store),
                        "restore_marker_authority": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.object(
            worker,
            "_write_master_outputs_from_shards",
            side_effect=SlurmGenerationInactiveError("superseded"),
        ),
        pytest.raises(SlurmGenerationInactiveError, match="superseded"),
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest_path,
            {"expected_non_finalizer_tasks": 0},
            slurm_generation=generation,
        )

    assert marker_path.read_bytes() == marker_before


@pytest.mark.parametrize("corrupt_artifact", ["measurements", "store"])
def test_missing_overlay_recovery_rejects_other_corrupt_artifact(
    _completed_run_two: Path,
    tmp_path: Path,
    corrupt_artifact: str,
) -> None:
    """Any second invalid artifact makes measured-overlay recovery fatal."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    (dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png").unlink()
    if corrupt_artifact == "measurements":
        import pyarrow.parquet as pq

        payload = pq.read_table(table)
        pq.write_table(payload, table, compression="gzip")
    else:
        (store / "zarr.json").write_text('{"corrupt": true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
        )
def test_missing_overlay_recovery_rejects_store_symlink_outside_output(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A measured recovery store resolving outside the output root is fatal."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic.sdk_ import dataset_overlays_dir
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    outside_store = tmp_path / f"{stem}-outside.ome.zarr"
    shutil.copytree(store, outside_store)
    shutil.rmtree(store)
    store.symlink_to(outside_store, target_is_directory=True)
    (dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png").unlink()

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
        )
def test_measurement_worker_refreshes_marker_with_active_slurm_generation(
    _completed_run_two: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real worker publishes its active generation, not the old marker epoch."""
    import shutil

    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        build_recompile_tasks,
    )
    from phenotypic._cli._cli_recompile_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
        image_completion_marker_path,
    )
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.unlink()
    generation = "actual-slurm-worker-generation"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    tasks = build_recompile_tasks(
        output_dir,
        [DATASET],
        include_dataset_column=True,
        overlay_alpha=0.6,
        shard_size=1,
        attempt_id=generation,
    )
    task = next(
        item
        for item in tasks
        if item["task_type"] == TASK_MEASUREMENTS
        and str(table) in item["files"]
    )
    manifest_path = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"tasks": [task]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "987654")

    result = CliRunner().invoke(
        main,
        [
            "--output-dir",
            str(output_dir),
            "--task-manifest",
            str(manifest_path),
            "--task-index",
            "0",
            "--slurm-generation",
            generation,
            "--attempt-id",
            generation,
        ],
    )

    assert result.exit_code == 0, result.output
    marker = json.loads(
        image_completion_marker_path(output_dir, DATASET, stem).read_text(
            encoding="utf-8"
        )
    )
    assert overlay.is_file()
    assert marker["lifecycle_epoch"] == generation
    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )


def test_recoverable_overlay_and_table_share_one_slurm_task(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Same-store overlay and table mutation cannot race in separate tasks."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        build_recompile_tasks,
    )
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    (dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png").unlink()

    tasks = build_recompile_tasks(
        output_dir,
        [DATASET],
        include_dataset_column=True,
        overlay_alpha=0.3,
        shard_size=1,
        attempt_id="ordered-recovery",
    )
    same_store_tasks = [
        task
        for task in tasks
        if str(table) in task.get("files", [])
        or task.get("store_path") == str(store)
    ]

    assert len(same_store_tasks) == 1
    task = same_store_tasks[0]
    assert task["task_type"] == TASK_MEASUREMENTS
    assert task["overlay_repairs"] == [
        {
            "dataset_name": DATASET,
            "store_path": str(store),
            "table_path": str(table),
            "overlay_alpha": 0.3,
        }
    ]


def test_retry_schedules_table_replaced_before_marker_publish_crash(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Durable exact replacement evidence makes an interrupted table retryable."""
    import shutil

    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        build_recompile_tasks,
    )
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        image_completion_marker_path,
    )
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)

    with (
        patch(
            "phenotypic._cli._cli_recompile_tables._republish_table_marker",
            side_effect=RuntimeError("simulated crash before marker publish"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        recompile_embedded_measurement_table(
            output_dir,
            table,
            DATASET,
            metadata,
        )

    assert (
        pl.read_parquet(table)["Metadata_Review"].to_list()
        == ["replacement"] * pl.read_parquet(table).height
    )
    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    old_marker = image_completion_marker_path(output_dir, DATASET, stem)
    assert old_marker.is_file()

    tasks = build_recompile_tasks(
        output_dir,
        [DATASET],
        include_dataset_column=True,
        overlay_alpha=0.3,
        shard_size=1,
        attempt_id="retry-after-table-replace",
    )
    scheduled = {
        Path(path)
        for task in tasks
        if task["task_type"] == TASK_MEASUREMENTS
        for path in task["files"]
    }

    assert table in scheduled

    recompile_embedded_measurement_table(
        output_dir,
        table,
        DATASET,
        metadata,
    )

    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    assert not recompile_table_transition_path(
        output_dir, DATASET, stem
    ).exists()



def test_retry_rejects_self_referential_transition_payload(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Transition evidence cannot nominate the changed canonical table itself."""
    import hashlib
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    with (
        patch(
            "phenotypic._cli._cli_recompile_tables._republish_table_marker",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    payload = pq.read_table(table)
    area_index = payload.column_names.index("Shape_Area")
    payload = payload.set_column(
        area_index,
        "Shape_Area",
        pa.array(
            [999999] * payload.num_rows,
            type=payload.schema.field(area_index).type,
        ),
    )
    pq.write_table(payload, table)
    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    transition["prepared_path"] = table.relative_to(output_dir).as_posix()
    transition["prepared_size"] = table.stat().st_size
    transition["prepared_sha256"] = hashlib.sha256(
        table.read_bytes()
    ).hexdigest()
    transition_path.write_text(json.dumps(transition), encoding="utf-8")

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
            attempt_id="reject-self-reference",
        )




@pytest.mark.parametrize(
    "prepared_path_case",
    ["malformed-name", "outside-root", "symlink", "hardlink"],
)
def test_retry_rejects_noncanonical_transition_staging_path(
    _completed_run_two: Path,
    tmp_path: Path,
    prepared_path_case: str,
) -> None:
    """Only a private canonical regular staging payload can authorize retry."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    with (
        patch(
            "phenotypic._cli._cli_recompile_tables._republish_table_marker",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    staged = output_dir / str(transition["prepared_path"])
    if prepared_path_case == "malformed-name":
        malformed = staged.with_name(f"{stem}.not-a-uuid.parquet")
        shutil.copy2(staged, malformed)
        transition["prepared_path"] = malformed.relative_to(
            output_dir
        ).as_posix()
    elif prepared_path_case == "outside-root":
        outside = tmp_path / "outside.parquet"
        shutil.copy2(staged, outside)
        transition["prepared_path"] = outside.relative_to(
            output_dir, walk_up=True
        ).as_posix()
    elif prepared_path_case == "symlink":
        outside = tmp_path / "outside.parquet"
        shutil.copy2(staged, outside)
        staged.unlink()
        staged.symlink_to(outside)
    else:
        staged.unlink()
        staged.hardlink_to(table)
    transition_path.write_text(json.dumps(transition), encoding="utf-8")

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
            attempt_id=f"reject-{prepared_path_case}",
        )



@pytest.mark.parametrize("tamper_case", ["current-bytes", "baseline"])
def test_retry_rejects_altered_payload_or_measurement_baseline(
    _completed_run_two: Path,
    tmp_path: Path,
    tamper_case: str,
) -> None:
    """Retry requires exact intended bytes and the unchanged table contract."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    with (
        patch(
            "phenotypic._cli._cli_recompile_tables._republish_table_marker",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    if tamper_case == "current-bytes":
        payload = pq.read_table(table)
        area_index = payload.column_names.index("Shape_Area")
        payload = payload.set_column(
            area_index,
            "Shape_Area",
            pa.array(
                [999999] * payload.num_rows,
                type=payload.schema.field(area_index).type,
            ),
        )
        pq.write_table(payload, table)
    else:
        root_path = store / "zarr.json"
        root = json.loads(root_path.read_text(encoding="utf-8"))
        root["attributes"]["phenotypic"]["tables"]["measurements"][
            "measurement_columns"
        ].append("Shape_MissingBaseline")
        root_path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
            attempt_id=f"reject-{tamper_case}",
        )


@pytest.mark.parametrize("evidence_case", ["missing-prior", "stale-marker"])
def test_retry_rejects_stale_or_unbound_prior_table_evidence(
    _completed_run_two: Path,
    tmp_path: Path,
    evidence_case: str,
) -> None:
    """Recovery requires the exact marker and prior table fingerprint."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    with (
        patch(
            "phenotypic._cli._cli_recompile_tables._republish_table_marker",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    if evidence_case == "missing-prior":
        transition.pop("prior_table_size", None)
        transition.pop("prior_table_sha256", None)
        transition_path.write_text(json.dumps(transition), encoding="utf-8")
    else:
        from phenotypic.sdk_ import image_completion_marker_path

        marker_path = image_completion_marker_path(output_dir, DATASET, stem)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["stale_marker_probe"] = True
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
            attempt_id="reject-unbound-prior-table",
        )




def test_recompile_rejects_staged_bytes_changed_after_journal(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Canonical replacement must promote exactly the journaled staged bytes."""
    import shutil

    import phenotypic._cli._cli_recompile_recovery as recovery
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    real_begin = recovery.begin_recompile_table_transition

    def _begin_then_corrupt(*args: object, **kwargs: object) -> Path:
        staged = real_begin(*args, **kwargs)  # type: ignore[arg-type]
        staged.write_bytes(b"changed after journal publication")
        return staged

    with (
        patch(
            "phenotypic._cli._cli_recompile_tables."
            "begin_recompile_table_transition",
            _begin_then_corrupt,
        ),
        pytest.raises(RuntimeError, match="transition|staged"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )


def test_retry_cleans_orphan_after_crash_before_transition_journal(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A retry removes staged bytes left before the journal became durable."""
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )

    with (
        patch(
            "phenotypic._cli._cli_recompile_recovery._write_json_at",
            side_effect=RuntimeError("crash before transition journal"),
        ),
        pytest.raises(RuntimeError, match="crash before transition journal"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    assert not transition_path.exists()
    assert len(list(transition_path.parent.glob(f"{stem}.*.parquet"))) == 1

    recompile_embedded_measurement_table(
        output_dir, table, DATASET, metadata
    )

    assert not transition_path.exists()
    assert list(transition_path.parent.glob(f"{stem}.*.parquet")) == []



def test_retry_recovers_crash_after_marker_publish_before_cleanup(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A valid new marker makes leftover transition evidence safely retryable."""
    import shutil

    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )

    with (
        patch(
            "phenotypic._cli._cli_recompile_tables."
            "clear_recompile_table_transition",
            side_effect=RuntimeError("crash before cleanup"),
        ),
        pytest.raises(RuntimeError, match="crash before cleanup"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    assert transition_path.exists()

    recompile_embedded_measurement_table(
        output_dir, table, DATASET, metadata
    )

    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    assert not transition_path.exists()
    assert list(transition_path.parent.glob(f"{stem}.*.parquet")) == []


def test_retry_refuses_unjournaled_invalid_measurement_table(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """An arbitrary marker-invalid table cannot be omitted into a partial run."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    pl.read_parquet(table).with_columns(
        pl.lit("arbitrary").alias("Metadata_Unjournaled")
    ).write_parquet(table)

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
            attempt_id="reject-unjournaled-table",
        )


def test_slurm_recompile_rejects_nonrecoverable_measurement_overlay(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A missing overlay plus changed table cannot degrade to best-effort work."""
    import shutil

    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    (dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png").unlink()
    pl.read_parquet(table).with_columns(
        pl.lit("changed").alias("Metadata_OtherInvalidity")
    ).write_parquet(table)

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
            attempt_id="reject-nonrecoverable-overlay",
        )



def test_clear_transition_never_deletes_forged_in_root_payload(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Cleanup must not unlink an arbitrary path named by forged evidence."""
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        clear_recompile_table_transition,
        recompile_table_transition_path,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )
    transition_path.parent.mkdir(parents=True, exist_ok=True)
    victim = output_dir / "must-survive.parquet"
    victim.write_bytes(b"not transition staging")
    transition_path.write_text(
        json.dumps(
            {"prepared_path": victim.relative_to(output_dir).as_posix()}
        ),
        encoding="utf-8",
    )

    clear_recompile_table_transition(output_dir, DATASET, stem)

    assert victim.read_bytes() == b"not transition staging"
    assert not transition_path.exists()


def test_retry_recovers_crash_after_transition_journal_before_promotion(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A durable journal with the prior table intact is safely replaceable."""
    import shutil

    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )

    with (
        patch(
            "phenotypic._cli._cli_recompile_tables."
            "promote_recompile_table_transition",
            side_effect=RuntimeError("crash after transition journal"),
        ),
        pytest.raises(RuntimeError, match="crash after transition journal"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    assert transition_path.exists()
    assert len(list(transition_path.parent.glob(f"{stem}.*.parquet"))) == 1
    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )

    recompile_embedded_measurement_table(
        output_dir, table, DATASET, metadata
    )

    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )
    assert not transition_path.exists()
    assert list(transition_path.parent.glob(f"{stem}.*.parquet")) == []

@pytest.mark.parametrize(
    "redirect_component",
    ["dataset-root", "transition-parent"],
)
def test_begin_transition_rejects_symlink_root_without_external_writes(
    _completed_run_two: Path,
    tmp_path: Path,
    redirect_component: str,
) -> None:
    """A redirected transition directory cannot receive or delete payloads."""
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    transition_root = recompile_table_transition_path(
        output_dir, DATASET, stem
    ).parent
    external = tmp_path / "external-transition-root"
    external.mkdir()
    if redirect_component == "dataset-root":
        redirect = transition_root
        payload_dir = external
    else:
        redirect = transition_root.parent
        payload_dir = external / DATASET
        payload_dir.mkdir()
    redirect.parent.mkdir(parents=True, exist_ok=True)
    victim = payload_dir / f"{stem}.{'a' * 32}.parquet"
    victim.write_bytes(b"external victim")
    redirect.symlink_to(external, target_is_directory=True)

    with pytest.raises((RuntimeError, ValueError)):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    assert victim.read_bytes() == b"external victim"
    assert sorted(path.name for path in payload_dir.iterdir()) == [victim.name]


@pytest.mark.parametrize("forgery", ["receipt-symlink", "external-hardlink"])
def test_retry_rejects_linked_transition_evidence(
    _completed_run_two: Path,
    tmp_path: Path,
    forgery: str,
) -> None:
    """Transition receipts and staged payloads must have one canonical link."""
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    with (
        patch(
            "phenotypic._cli._cli_recompile_tables._republish_table_marker",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    transition_path = recompile_table_transition_path(
        output_dir, DATASET, stem
    )
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    staged = output_dir / str(transition["prepared_path"])
    if forgery == "receipt-symlink":
        external_receipt = tmp_path / "external-transition.json"
        external_receipt.write_bytes(transition_path.read_bytes())
        transition_path.unlink()
        transition_path.symlink_to(external_receipt)
    else:
        external_alias = tmp_path / "external-stage-alias.parquet"
        external_alias.hardlink_to(staged)

    with pytest.raises(RuntimeError, match="measurement authority"):
        build_recompile_tasks(
            output_dir,
            [DATASET],
            include_dataset_column=True,
            overlay_alpha=0.3,
            shard_size=1,
            attempt_id=f"reject-{forgery}",
        )


def test_overlay_refresh_holds_generation_guard_only_for_marker_commit(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Marker discovery and hashing stay outside the lifecycle commit window."""
    import shutil
    from contextlib import contextmanager
    from typing import Iterator

    import phenotypic._cli._cli_recompile_slurm_scripts as scripts
    from phenotypic.sdk_ import dataset_overlays_dir
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.write_bytes(overlay.read_bytes() + b"repaired")
    lifecycle_active = False
    recovery_guard_states: list[bool] = []
    guard_entries = 0
    real_recovery = scripts._overlay_recovery_marker

    def _observe_recovery(*args: object, **kwargs: object) -> object:
        recovery_guard_states.append(lifecycle_active)
        return real_recovery(*args, **kwargs)  # type: ignore[arg-type]

    @contextmanager
    def _commit_guard() -> Iterator[None]:
        nonlocal lifecycle_active, guard_entries
        guard_entries += 1
        lifecycle_active = True
        try:
            yield
        finally:
            lifecycle_active = False

    with patch.object(
        scripts,
        "_overlay_recovery_marker",
        side_effect=_observe_recovery,
    ):
        assert scripts.refresh_overlay_marker_authority(
            output_dir,
            DATASET,
            stem,
            store,
            commit_guard=_commit_guard,
        )

    assert recovery_guard_states == [False]
    assert guard_entries == 1

def test_begin_transition_parent_swap_cannot_touch_external_directory(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A validated parent swap cannot redirect staging or orphan cleanup."""
    import shutil

    import phenotypic._cli._cli_recompile_recovery as recovery
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    root = recompile_table_transition_path(output_dir, DATASET, stem).parent
    root.mkdir(parents=True)
    displaced = tmp_path / "displaced-transition-root"
    external = tmp_path / "external-transition-root"
    external.mkdir()
    victim = external / f"{stem}.{'b' * 32}.parquet"
    victim.write_bytes(b"external victim")
    real_write = recovery._write_validated_parquet
    swapped = False

    def _swap_then_write(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        if not swapped:
            root.rename(displaced)
            root.symlink_to(external, target_is_directory=True)
            swapped = True
        real_write(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(
            recovery,
            "_write_validated_parquet",
            _swap_then_write,
        ),
        pytest.raises((RuntimeError, ValueError)),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    assert victim.read_bytes() == b"external victim"
    assert sorted(path.name for path in external.iterdir()) == [victim.name]


def test_clear_transition_parent_swap_cannot_delete_external_files(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Cleanup remains bound to the opened transition directory identity."""
    import os
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        clear_recompile_table_transition,
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["replacement"],
        }
    ).write_csv(metadata)
    with (
        patch(
            "phenotypic._cli._cli_recompile_tables._republish_table_marker",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        recompile_embedded_measurement_table(
            output_dir, table, DATASET, metadata
        )

    receipt = recompile_table_transition_path(output_dir, DATASET, stem)
    transition = json.loads(receipt.read_text(encoding="utf-8"))
    staged_name = Path(str(transition["prepared_path"])).name
    root = receipt.parent
    displaced = tmp_path / "displaced-transition-root"
    external = tmp_path / "external-transition-root"
    external.mkdir()
    external_stage = external / staged_name
    external_receipt = external / receipt.name
    external_stage.write_bytes(b"external staged victim")
    external_receipt.write_bytes(b"external receipt victim")
    real_unlink = os.unlink
    swapped = False

    def _swap_then_unlink(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped:
            root.rename(displaced)
            root.symlink_to(external, target_is_directory=True)
            swapped = True
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    with patch.object(os, "unlink", _swap_then_unlink):
        clear_recompile_table_transition(output_dir, DATASET, stem)

    assert external_stage.read_bytes() == b"external staged victim"
    assert external_receipt.read_bytes() == b"external receipt victim"


def test_recovery_source_discovery_rejects_symlink_root_before_enumeration(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Recovery discovery fails closed instead of walking a redirected root."""
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        recoverable_recompile_measurement_sources,
        recompile_table_transition_path,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    root = recompile_table_transition_path(output_dir, DATASET, stem).parent
    root.parent.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external-transition-root"
    external.mkdir()
    (external / "external.json").write_text("{}", encoding="utf-8")
    root.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="transition directory"):
        recoverable_recompile_measurement_sources(output_dir, [DATASET])


def test_stale_slurm_overlay_worker_does_not_publish_rendered_bytes(
    _completed_run_two: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale generation is fenced before canonical overlay replacement."""
    import shutil

    from phenotypic._cli._cli_recompile_worker import _run_overlay_task
    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
    )
    from phenotypic.sdk_ import dataset_overlays_dir
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.unlink()
    generation = "stale-overlay-render"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    assert deactivate_generation(output_dir, generation)
    monkeypatch.setenv("SLURM_JOB_ID", "stale-overlay-worker")

    with pytest.raises(SlurmGenerationInactiveError):
        _run_overlay_task(
            output_dir,
            {
                "task_type": "overlay",
                "dataset_name": DATASET,
                "store_path": str(store),
                "overlay_alpha": 0.3,
                "restore_marker_authority": True,
            },
            slurm_generation=generation,
        )

    assert not overlay.exists()


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"),
    reason="FIFO blocking regression requires POSIX FIFO support",
)
def test_transition_fifo_evidence_is_rejected_without_blocking(
    tmp_path: Path,
) -> None:
    """A FIFO receipt cannot block recovery while the store lock is held."""
    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    receipt = recompile_table_transition_path(output_dir, "ds", "img")
    receipt.parent.mkdir(parents=True)
    os.mkfifo(receipt)
    ready = tmp_path / "ready.txt"
    completed = tmp_path / "completed.txt"
    probe = (
        "from pathlib import Path; import sys; "
        "from phenotypic._cli._cli_recompile_recovery import "
        "recoverable_recompile_measurement_sources; "
        "Path(sys.argv[2]).write_text('ready', encoding='utf-8'); "
        "result = recoverable_recompile_measurement_sources("
        "Path(sys.argv[1]), ['ds']); "
        "Path(sys.argv[3]).write_text(repr(result), encoding='utf-8')"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            probe,
            str(output_dir),
            str(ready),
            str(completed),
        ]
    )
    import_deadline = time.monotonic() + 15.0
    while not ready.exists() and time.monotonic() < import_deadline:
        assert process.poll() is None
        time.sleep(0.01)
    assert ready.is_file(), "FIFO recovery probe did not finish importing"
    try:
        return_code = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
        pytest.fail("FIFO receipt blocked recovery discovery")

    assert return_code == 0
    assert completed.read_text(encoding="utf-8") == "{}"


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(),
    reason="directory fsync ordering probe requires procfs",
)
def test_recompile_fsyncs_transaction_directories_in_publication_order(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Durable directory commits follow receipt, table, marker, cleanup order."""
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        image_completion_marker_path,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    marker = image_completion_marker_path(output_dir, DATASET, stem)
    transition_dir = recompile_table_transition_path(
        output_dir,
        DATASET,
        stem,
    ).parent
    missing_component_parents: list[Path] = []
    component = output_dir
    for name in transition_dir.relative_to(output_dir).parts:
        candidate = component / name
        if not candidate.exists():
            missing_component_parents.append(component)
        component = candidate
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["durable"],
        }
    ).write_csv(metadata)
    real_fsync = os.fsync
    directory_syncs: list[Path] = []

    def _record_fsync(file_descriptor: int) -> None:
        identity = os.fstat(file_descriptor)
        if stat.S_ISDIR(identity.st_mode):
            directory_syncs.append(
                Path(os.readlink(f"/proc/self/fd/{file_descriptor}"))
            )
        real_fsync(file_descriptor)

    with patch.object(os, "fsync", _record_fsync):
        recompile_embedded_measurement_table(
            output_dir,
            table,
            DATASET,
            metadata,
        )

    table_parent_index = directory_syncs.index(table.parent)
    marker_parent_index = directory_syncs.index(marker.parent)
    transition_indices = [
        index
        for index, directory in enumerate(directory_syncs)
        if directory == transition_dir
    ]
    assert all(
        parent in directory_syncs[:table_parent_index]
        for parent in missing_component_parents
    )
    assert len(
        [index for index in transition_indices if index < table_parent_index]
    ) >= 2
    assert table_parent_index < marker_parent_index
    assert any(index > marker_parent_index for index in transition_indices)


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(),
    reason="directory fsync fault probe requires procfs",
)
def test_marker_directory_fsync_failure_preserves_transition_evidence(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A marker durability failure aborts before receipt and stage cleanup."""
    import shutil

    from phenotypic._cli._cli_recompile_recovery import (
        recompile_table_transition_path,
    )
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        image_completion_marker_path,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    table = (
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    marker_parent = image_completion_marker_path(
        output_dir,
        DATASET,
        stem,
    ).parent
    receipt = recompile_table_transition_path(output_dir, DATASET, stem)
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [stem],
            "Metadata_Review": ["durable"],
        }
    ).write_csv(metadata)
    real_fsync = os.fsync

    def _fail_marker_directory(file_descriptor: int) -> None:
        identity = os.fstat(file_descriptor)
        target = Path(os.readlink(f"/proc/self/fd/{file_descriptor}"))
        if stat.S_ISDIR(identity.st_mode) and target == marker_parent:
            raise OSError("simulated marker directory fsync failure")
        real_fsync(file_descriptor)

    with (
        patch.object(os, "fsync", _fail_marker_directory),
        pytest.raises(OSError, match="marker directory fsync failure"),
    ):
        recompile_embedded_measurement_table(
            output_dir,
            table,
            DATASET,
            metadata,
        )

    assert receipt.is_file()
    transition = json.loads(receipt.read_text(encoding="utf-8"))
    staged = output_dir / str(transition["prepared_path"])
    assert staged.is_file()

def test_transition_recovery_fails_closed_without_safe_directory_primitives(
    tmp_path: Path,
) -> None:
    """Recovery refuses access when identity-bound primitives are unavailable."""
    import phenotypic._cli._cli_recompile_recovery as recovery

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with (
        patch.object(
            recovery,
            "_IDENTITY_BOUND_DIRECTORY_OPERATIONS",
            False,
        ),
        pytest.raises(RuntimeError, match="cannot safely access"),
    ):
        recovery.recoverable_recompile_measurement_sources(
            output_dir,
            ["ds"],
        )
