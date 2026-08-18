"""Unit tests for recompile-specific SLURM task scripts and worker."""

from __future__ import annotations

import json
import sys
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
    hdf_path = output_dir / "results" / "plate_b" / "hdf" / "img_b1.h5"
    hdf_path.parent.mkdir(parents=True)
    hdf_path.write_text("stub", encoding="utf-8")

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
            "hdf_path": str(hdf_path),
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
    hdf_path = output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    hdf_path.parent.mkdir(parents=True)
    hdf_path.write_text("stub", encoding="utf-8")
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
                        "hdf_path": str(hdf_path),
                        "overlay_alpha": 0.7,
                        "slurm_generation": generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeH5:
        attrs = {"phenotypic_class": "Image"}

        def __enter__(self) -> "FakeH5":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    class FakeImage:
        @classmethod
        def load_hdf5(cls, _path: Path) -> "FakeImage":
            return cls()

    with (
        patch("h5py.File", return_value=FakeH5()),
        patch("phenotypic.Image", FakeImage),
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
