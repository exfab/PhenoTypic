"""Unit tests for recompile-specific SLURM task scripts and worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
from click.testing import CliRunner

from phenotypic.sdk_ import (
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


#: The legacy-marker deferral is CLOSED. P4 repointed all five
#: production call sites onto the per-image record, keeping the legacy
#: `image_complete/` shape as a second arm under its own predicate and
#: its own version constant -- `_image_authority_shapes` in
#: `_cli_recompile_recovery.py` is the one home for that pairing, and
#: carries the retirement condition (P7 arms the schema gate). The
#: `_RECOMPILE_READS_THE_LEGACY_MARKER_UNTIL_P4` xfail marker that stood
#: here went with the repoint, in the same commit: it was `strict=True`,
#: so leaving it would have turned every test it decorated red.


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


def test_finalizer_publishes_when_only_overlay_task_failed(
    tmp_path: Path,
) -> None:
    """An optional overlay failure cannot suppress measurement publication."""
    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic._cli._cli_recompile_slurm_scripts import TASK_OVERLAY
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "overlay-failure-nonblocking"
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
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    task = {"expected_non_finalizer_tasks": 1}
    with (
        patch.object(
            worker,
            "_wait_for_non_finalizer_statuses",
            return_value=[{"task_type": TASK_OVERLAY, "status": "failed"}],
        ),
        patch.object(
            worker, "_run_post_master_steps", return_value=None
        ) as post,
        patch.object(worker, "_regenerate_recompile_dashboard") as dashboard,
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest,
            task,
            slurm_generation=generation,
        )

    # `_run_post_master_steps` IS the master step since P4 collapsed
    # `_write_master_outputs_from_shards` into it, so it is the probe for
    # "the finalizer reached publication" that the deleted function was.
    post.assert_called_once()
    dashboard.assert_called_once()


def test_finalizer_blocks_when_measurement_task_failed(
    tmp_path: Path,
) -> None:
    """A failed measurement shard leaves the aggregate incomplete."""
    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic._cli._cli_recompile_slurm_scripts import TASK_MEASUREMENTS
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "measurement-failure-blocking"
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
            return_value=[
                {"task_type": TASK_MEASUREMENTS, "status": "failed"}
            ],
        ),
        patch.object(
            worker, "_run_post_master_steps", return_value=None
        ) as post,
        patch.object(worker, "_regenerate_recompile_dashboard"),
        pytest.raises(RuntimeError, match="blocking non-finalizer recompile"),
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest,
            task,
            slurm_generation=generation,
        )

    post.assert_not_called()


@pytest.mark.parametrize(
    ("task_type", "worker_name"),
    [
        ("measurements", "_run_measurement_task"),
        ("overlay", "_run_overlay_task"),
    ],
)
def test_failed_non_finalizer_keeps_generation_active(
    tmp_path: Path,
    task_type: str,
    worker_name: str,
) -> None:
    """Only the finalizer owns teardown of the shared generation."""
    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    generation = f"non-finalizer-{task_type}"
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
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": task_type,
                        "slurm_generation": generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.object(worker, worker_name, side_effect=RuntimeError("failed")),
        pytest.raises(RuntimeError, match="failed"),
    ):
        worker.run_recompile_task(
            output_dir,
            manifest,
            0,
            slurm_generation=generation,
            attempt_id=generation,
        )

    assert generation_is_active(output_dir, generation)
    status = json.loads(
        (manifest.parent / "status" / "task_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "failed"


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
    assert master_measurements_parquet_path(output_dir).exists()
    # D8: the master is parquet-only, on this path as on the forward one.
    assert not (
        output_dir / "deliverables" / "master_measurements.csv"
    ).exists()
    # Recompile finalizer also seeds the GUI's editable measurements copy.
    assert measurements_csv_path(output_dir).exists()
    assert measurements_parquet_path(output_dir).exists()
    assert pl.read_parquet(master_measurements_parquet_path(output_dir))[
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
        patch.object(
            worker, "_run_post_master_steps", return_value=None
        ) as post,
        pytest.raises(RuntimeError, match="non-finalizer recompile task"),
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest,
            task,
            slurm_generation=generation,
        )

    post.assert_not_called()


def test_finalizer_does_not_publish_after_master_parquet_failure(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A finalizer that could not write the master publishes no new authority.

    **D8 inverted which write is the gate.** The CSV used to be required and
    the Parquet best-effort ("CSV was saved"); now the Parquet *is* the master,
    so its failure has to stop finalization outright. The failure mode this
    guards is a run that reports success having written no master at all.

    The failure is swallowed rather than raised -- ``finalize_run`` writes the
    master through ``_guarded_terminal_best_effort`` and returns ``None`` --
    so the assertions are on what is *not* on disk afterwards rather than on
    an exception, plus one positive control that the blocked write was
    reached at all. See the comment above them for why that control replaced
    a sentinel column.
    """
    import shutil

    import phenotypic.sdk_ as sdk_
    import phenotypic._cli._cli_recompile_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import (
        aggregate_publication_marker_path,
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
    # STANDING RULE: both assertions below are equalities against a "before"
    # snapshot, and `b"" == b""` would satisfy them on a fixture that never
    # published a master or a proof. Establish that this run did.
    assert marker_path.is_file(), "fixture published no aggregate proof"
    marker_before = marker_path.read_bytes()
    master_path = master_measurements_parquet_path(output_dir)
    assert master_path.is_file(), "fixture published no master"
    parquet_before = master_path.read_bytes()
    assert parquet_before, "fixture's master is empty"
    real_atomic_write = sdk_.atomic_write_with_writer
    blocked: list[Path] = []

    def _fail_master_parquet(
        path: Path,
        writer: object,
        **kwargs: object,
    ) -> None:
        if Path(path) == master_path:
            blocked.append(Path(path))
            raise OSError("simulated master Parquet failure")
        real_atomic_write(path, writer, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(
            sdk_,
            "atomic_write_with_writer",
            _fail_master_parquet,
        ),
        patch.object(worker, "_regenerate_recompile_dashboard"),
    ):
        worker._run_finalizer_task(
            output_dir,
            manifest_path,
            {"expected_non_finalizer_tasks": 0},
            slurm_generation=generation,
        )

    # STANDING RULE, and the reason this is not the assertion it replaces.
    # Both equalities below say "nothing changed", which is equally true of a
    # run that never reached the master write: an empty shard glob, a merge
    # that produced no frame, a finalizer that raised earlier.
    #
    # The pre-D8 test established the write had happened by reading the
    # shard's 999999 back out of the master CSV -- which worked because that
    # CSV *was* the shard concat. **D8 removes the object that assertion read
    # from.** The CSV is deleted and the Parquet write is blocked, so the only
    # master on disk afterwards is the fixture's own, whose measurers are
    # Shape/Intensity/Texture/Color and which therefore carries no `Size_*`
    # column at all. Carrying the sentinel across that retarget kept the
    # column's spelling and lost its subject.
    #
    # What survives the inversion is the attempt. Asserting on it is stronger
    # than the sentinel ever was here, because it fails for the empty-shard
    # case too, which no reading of the master can detect.
    assert blocked == [master_path], (
        "the master write was never attempted, so the equalities below hold "
        "for a run that did nothing rather than for one that was stopped"
    )
    assert master_path.read_bytes() == parquet_before
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
    # REAL stores, not a bare `table.parquet`. The shard worker reads each
    # table's descriptor from its store root (P7 Task 4), so a rootless table
    # was never a reachable input.
    #
    # Each store also gets a per-image RECORD. The shard now reports the work
    # identities it merged (`_merged_source_work_ids`), so a source with no
    # authority payload raises rather than being silently dropped from the
    # set the aggregate proof is published against.
    import pandas as pd

    from phenotypic._cli._cli_completion import publish_image_success

    from .conftest import _image, _manager

    tables = []
    for stem, value in (("img2", 20), ("img1", 10)):
        store = _manager(output_dir).save_image_store(
            _image(stem),
            "plate_a",
            stem,
            work_id=f"work-{stem}",
            measurements=pd.DataFrame(
                {"Object_Label": [1, 2], "Size_Area": [value, value + 1]}
            ),
        )
        assert store is not None, f"the forward writer failed to promote {stem}"
        publish_image_success(
            output_dir,
            work_id=f"work-{stem}",
            dataset="plate_a",
            relative_image_path=f"{stem}.tiff",
            image_stem=stem,
            mode="full",
            attempt_id="attempt-1",
            lifecycle_epoch=generation,
            artifacts={
                "measurements": store / MEASUREMENT_TABLE_RELATIVE_PATH,
                "store": store,
            },
        )
        tables.append(store / MEASUREMENT_TABLE_RELATIVE_PATH)
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
        "img1",
        "img2",
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
            worker, "_run_post_master_steps", return_value=None
        ),
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
        image_record_path,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.write_bytes(b"replacement overlay bytes")
    record_path = image_record_path(output_dir, DATASET, stem)
    record_before = record_path.read_bytes()
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
            "_run_post_master_steps",
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

    assert record_path.read_bytes() == record_before


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
        image_record_path,
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
    record = json.loads(
        image_record_path(output_dir, DATASET, stem).read_text(
            encoding="utf-8"
        )
    )
    assert overlay.is_file()
    assert record["lifecycle_epoch"] == generation
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
    """A FIFO receipt cannot block recovery while the store lock is held.

    **The budget below is 30 s and must not be re-tightened.** It was 1.0 s,
    and the test failed intermittently on a loaded HPCC login node -- at
    ``92be762a``, before the recompile-rewrite removal, so the flake is not
    anyone's change. Measured on this node with the same interpreter:

    * importing ``_cli_recompile_recovery`` (and so ``phenotypic``): **5.3 s**
    * ``recoverable_recompile_measurement_sources`` over the FIFO: **0.001 s**
    * total subprocess wall time: **6.3 s**

    The import has its own 15 s window above, so the 1 s was never covering
    it. What the 1 s covered was the call **plus interpreter teardown**, and
    teardown of the imported stack is ~1.06 s -- just over the budget, which
    is why the result flipped run to run rather than failing consistently.

    So the number was measuring CPython shutdown, not the property. The
    property is *"a FIFO does not block recovery indefinitely"*, and the call
    satisfies it by three orders of magnitude: ``_read_regular_file_at``
    opens with ``O_NONBLOCK | O_NOFOLLOW`` and rejects the FIFO on ``fstat``
    before reading a byte. 30 s is far above the teardown cost and far below
    any blocking open, which would never return at all.
    """
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
        return_code = process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
        pytest.fail("FIFO receipt blocked recovery discovery")

    assert return_code == 0
    assert completed.read_text(encoding="utf-8") == "{}"


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


def _strip_measurement_descriptor(store: Path) -> None:
    """Make the projection exclude *store*, the way it excludes one for real.

    ``project_embedded_measurement_table`` excludes a store whose root
    declares no ``tables.measurements`` -- "a normal state", per
    ``read_embedded_measurement_descriptor`` -- with an advisory rather than
    a raise. Removing the entry from the promoted root is that state exactly;
    nothing else about the store changes, so its table is still on disk and
    still named by the shard.
    """
    from phenotypic.sdk_ import PhenotypicAttr, STORE_ROOT_JSON

    root_path = store / STORE_ROOT_JSON
    root = json.loads(root_path.read_text(encoding="utf-8"))
    tables = root["attributes"][PhenotypicAttr.ROOT][PhenotypicAttr.TABLES]
    del tables["measurements"]
    root_path.write_text(json.dumps(root), encoding="utf-8")


def _publish_shard_store(
    output_dir: Path, dataset: str, stem: str, value: int, generation: str
) -> Path:
    """Promote one real store with a record, and return its table path."""
    import pandas as pd

    from phenotypic._cli._cli_completion import publish_image_success
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    from .conftest import _image, _manager

    store = _manager(output_dir).save_image_store(
        _image(stem),
        dataset,
        stem,
        work_id=f"work-{stem}",
        measurements=pd.DataFrame(
            {"Object_Label": [1, 2], "Size_Area": [value, value + 1]}
        ),
    )
    assert store is not None, f"the forward writer failed to promote {stem}"
    publish_image_success(
        output_dir,
        work_id=f"work-{stem}",
        dataset=dataset,
        relative_image_path=f"{stem}.tiff",
        image_stem=stem,
        mode="full",
        attempt_id="attempt-1",
        lifecycle_epoch=generation,
        artifacts={
            "measurements": store / MEASUREMENT_TABLE_RELATIVE_PATH,
            "store": store,
        },
    )
    return store


def _run_measurement_shard(
    output_dir: Path, generation: str, tables: list[Path], shard_id: int
) -> tuple[object, Path]:
    """Write a one-task manifest and run the measurement shard worker."""
    from phenotypic._cli._cli_recompile_worker import main

    manifest_path = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "shard_id": shard_id,
                        "files": [str(table) for table in tables],
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
    return result, manifest_path


def test_a_shard_whose_stores_are_all_excluded_writes_an_empty_shard(
    tmp_path: Path,
) -> None:
    """Gap (b). Every store excluded is not a shard failure.

    ``write_measurement_shard`` already writes an empty shard rather than
    skipping one, because the finalizer counts shard FILES against a carried
    K and a missing file reads there as a dead worker. The recompile shard
    raised ``No valid measurements found for shard`` instead, failing the
    whole recompile for a state the forward path treats as ordinary.

    **The raise is kept for the case that is genuinely broken** -- sources
    read but no frame produced -- which the empty-shard branch is careful not
    to swallow; only "the projection excluded everything" is now benign.
    """
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    output_dir = tmp_path / "out"
    generation = "all-excluded-shard"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    store = _publish_shard_store(output_dir, "plate_a", "img1", 10, generation)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    _strip_measurement_descriptor(store)

    # STANDING RULE: the store must really be excluded, or an ordinary shard
    # would be written and the assertions below would be about nothing.
    from phenotypic._cli._cli_parquet_agg import (
        project_embedded_measurement_table,
    )

    assert project_embedded_measurement_table(table) is None

    result, manifest_path = _run_measurement_shard(
        output_dir, generation, [table], shard_id=3
    )

    assert result.exit_code == 0, result.output
    shard_path = (
        manifest_path.parent / "measurement_shards" / "shard_3.parquet"
    )
    assert shard_path.is_file(), (
        "the shard file is missing; the finalizer counts files against a "
        "carried K and would read this as a dead worker"
    )
    assert pl.read_parquet(shard_path).height == 0

    status = json.loads(
        (manifest_path.parent / "status" / "task_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "completed"
    assert status["source_work_ids"] == [], (
        "the excluded store was reported as merged, so the aggregate proof "
        "would certify an image the master does not carry"
    )


def test_a_shard_reports_only_the_stores_it_actually_merged(
    tmp_path: Path,
) -> None:
    """Gap (a), the producing half: what the shard records is what it merged.

    Two stores, one excluded by the projection. Pre-change the shard recorded
    nothing at all and the finalizer re-derived the source set from the live
    markers, which name **both**.
    """
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    output_dir = tmp_path / "out"
    generation = "partly-excluded-shard"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    good = _publish_shard_store(output_dir, "plate_a", "good", 10, generation)
    bad = _publish_shard_store(output_dir, "plate_a", "bad", 20, generation)
    _strip_measurement_descriptor(bad)
    tables = [
        good / MEASUREMENT_TABLE_RELATIVE_PATH,
        bad / MEASUREMENT_TABLE_RELATIVE_PATH,
    ]

    result, manifest_path = _run_measurement_shard(
        output_dir, generation, tables, shard_id=0
    )

    assert result.exit_code == 0, result.output
    shard = pl.read_parquet(
        manifest_path.parent / "measurement_shards" / "shard_0.parquet"
    )
    assert set(shard[str(IMAGE.IMAGE_NAME)].to_list()) == {"good"}

    status = json.loads(
        (manifest_path.parent / "status" / "task_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["source_work_ids"] == ["work-good"], (
        "the shard reported a source it did not merge"
    )


def _run_finalizer_over(
    tmp_path: Path, generation: str, statuses: list[dict]
) -> object:
    """Run the real finalizer task over hand-written non-finalizer statuses.

    The public entry (``run_recompile_task`` through the worker CLI), not
    ``_run_post_master_steps``: driving the private function would make the
    pre-change failure a ``TypeError`` about a new keyword argument, which
    proves the signature moved rather than that the behaviour was wrong.
    Here the pre-change failure is that ``planned_work_ids`` never reaches
    ``finalize_run``, which is the defect itself.
    """
    from phenotypic._cli._cli_recompile_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / generation
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    attempt_dir = (
        progress_dir(output_dir) / "recompile" / "attempts" / generation
    )
    (attempt_dir / "measurement_shards").mkdir(parents=True)
    status_dir = attempt_dir / "status"
    status_dir.mkdir(parents=True)
    for index, status in enumerate(statuses):
        (status_dir / f"task_{index}.json").write_text(
            json.dumps(status), encoding="utf-8"
        )
    manifest_path = attempt_dir / "task_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": status["task_type"],
                        "slurm_generation": generation,
                    }
                    for status in statuses
                ]
                + [
                    {
                        "task_type": "finalize",
                        "dataset_names": ["plate_a"],
                        "include_dataset_column": True,
                        "metadata_csv": None,
                        "expected_non_finalizer_tasks": len(statuses),
                        "slurm_generation": generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    captured: list[object] = []

    def _capture(output_dir: Path, **kwargs: object) -> None:
        captured.append(kwargs.get("planned_work_ids", "ABSENT"))
        return None

    with (
        patch("phenotypic._cli._cli_finalize_run.finalize_run", new=_capture),
        patch("phenotypic._cli._dashboard._manifest_builder.build_manifest"),
        patch("phenotypic._cli._dashboard._generator.generate_dashboard"),
    ):
        result = CliRunner().invoke(
            main,
            [
                "--output-dir",
                str(output_dir),
                "--task-manifest",
                str(manifest_path),
                "--task-index",
                str(len(statuses)),
                "--slurm-generation",
                generation,
                "--attempt-id",
                generation,
            ],
        )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1, (
        f"finalize_run was called {len(captured)} times, not once"
    )
    return captured[0]


def test_the_recompile_finalizer_publishes_against_what_its_shards_merged(
    tmp_path: Path,
) -> None:
    """Gap (a), the consuming half: the union reaches ``finalize_run``.

    The finalizer used to pass no ``planned_work_ids``, so ``finalize_run``
    fell back to ``_work_ids_for_sources`` over the sources it selected
    itself -- the live authorized set, which includes stores the shards
    excluded.

    An overlay status contributes nothing, and a shard that merged nothing
    contributes an empty list rather than being ignored.
    """
    forwarded = _run_finalizer_over(
        tmp_path,
        "planned-forwarded",
        [
            {
                "task_type": "measurements",
                "status": "completed",
                "source_work_ids": ["work-b", "work-a"],
            },
            {
                "task_type": "measurements",
                "status": "completed",
                "source_work_ids": [],
            },
            {"task_type": "overlay", "status": "completed"},
        ],
    )

    assert forwarded == ["work-a", "work-b"], (
        "the finalizer did not forward the set its shards recorded merging"
    )


def test_the_recompile_finalizer_forwards_nothing_when_a_shard_could_not_say(
    tmp_path: Path,
) -> None:
    """The other direction, and it is not symmetry for its own sake.

    ``[]`` is a shard's claim that it merged nothing; a **missing** key is no
    claim at all, which is what a legacy-external-Parquet shard reports
    (:func:`~phenotypic._cli._cli_recompile_worker._run_measurement_task`).
    Forwarding ``[]`` for that would publish a proof asserting zero images
    over a master built from every authorized source -- a worse error than
    the live re-derivation this change replaced, because it understates
    rather than overstates.

    Two shapes, because a single one cannot separate "absent" from "empty":
    an attempt with no measurement task at all, and one where a measurement
    status carries no ``source_work_ids``.
    """
    assert (
        _run_finalizer_over(
            tmp_path,
            "planned-no-measurement-task",
            [{"task_type": "overlay", "status": "completed"}],
        )
        == "ABSENT"
    )
    assert (
        _run_finalizer_over(
            tmp_path,
            "planned-partial",
            [
                {
                    "task_type": "measurements",
                    "status": "completed",
                    "source_work_ids": ["work-a"],
                },
                {"task_type": "measurements", "status": "completed"},
            ],
        )
        == "ABSENT"
    ), (
        "a partial union was forwarded; it understates the master exactly "
        "as badly as a live re-derivation overstates it"
    )
