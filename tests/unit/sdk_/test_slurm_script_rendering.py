"""Tests for shared SLURM array script rendering helpers."""

from __future__ import annotations

import sys
import shlex
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="SLURM script tests require POSIX chmod"
)


def test_slurm_array_script_renders_directives_task_list_and_body(
    tmp_path: Path,
) -> None:
    """Rendered scripts should include directives, task mapping, and body."""
    from phenotypic.sdk_.slurm import SlurmArrayScriptSpec

    log_path = tmp_path / "logs" / "job_%A_%a.log"
    spec = SlurmArrayScriptSpec(
        job_name="pht-test",
        slurm_args={"slurm_partition": "short", "mem_gb": 8, "time": 90},
        log_path=log_path,
        task_indices=[5, 8, 13],
        prelude='echo "prelude"',
        body='echo "body $CURRENT_TASK_INDEX"',
        comments=["# Chunk: 0"],
    )

    rendered = spec.render()

    assert rendered.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=pht-test" in rendered
    assert f"#SBATCH --output={log_path.as_posix()}" in rendered
    assert f"#SBATCH --error={log_path.as_posix()}" in rendered
    assert "#SBATCH --partition=short" in rendered
    assert "#SBATCH --mem=8G" in rendered
    assert "#SBATCH --time=01:30:00" in rendered
    assert "#SBATCH --array=0-2" in rendered
    assert "# Chunk: 0" in rendered
    assert "set -e\nset -u" in rendered
    assert 'echo "prelude"' in rendered
    assert "TASK_INDICES=(\n    5\n    8\n    13\n)" in rendered
    assert (
        'CURRENT_TASK_INDEX="${TASK_INDICES[$SLURM_ARRAY_TASK_ID]}"'
        in rendered
    )
    assert 'echo "body $CURRENT_TASK_INDEX"' in rendered
    assert "EXIT_CODE=$?" in rendered
    assert "Exit Code: $EXIT_CODE" in rendered
    assert rendered.rstrip().endswith("exit $EXIT_CODE")


def test_slurm_array_script_supports_custom_entry_variables(
    tmp_path: Path,
) -> None:
    """Forward image arrays should keep IMAGE_LIST/CURRENT_IMAGE names."""
    from phenotypic.sdk_.slurm import SlurmArrayScriptSpec

    image_path = tmp_path / "image with space.tif"
    spec = SlurmArrayScriptSpec(
        job_name="pht-images",
        slurm_args={},
        log_path=tmp_path / "images_%A_%a.log",
        task_indices=[str(image_path), "__SENTINEL__"],
        array_name="IMAGE_LIST",
        current_var="CURRENT_IMAGE",
        missing_task_id_message="ERROR: SLURM_ARRAY_TASK_ID not set",
        bounds_error_message=(
            "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds image list size "
            "${#IMAGE_LIST[@]}"
        ),
        body='echo "image $CURRENT_IMAGE"',
    )

    rendered = spec.render()

    assert "IMAGE_LIST=(" in rendered
    assert shlex.quote(str(image_path)) in rendered
    assert "__SENTINEL__" in rendered
    assert 'CURRENT_IMAGE="${IMAGE_LIST[$SLURM_ARRAY_TASK_ID]}"' in rendered
    assert 'echo "image $CURRENT_IMAGE"' in rendered


def test_slurm_array_script_renders_signal_and_requeue_directives(
    tmp_path: Path,
) -> None:
    """Stage-2 GPU scripts should carry walltime-survival directives."""
    from phenotypic.sdk_.slurm import SlurmArrayScriptSpec

    spec = SlurmArrayScriptSpec(
        job_name="pht-stage2",
        slurm_args={"slurm_partition": "gpu"},
        log_path=tmp_path / "stage2_%A_%a.log",
        task_indices=[0],
        body="run-stage-2",
        signal_grace=120,
        requeue=True,
    )

    rendered = spec.render()

    assert "#SBATCH --signal=B:TERM@120" in rendered
    assert "#SBATCH --requeue" in rendered


def test_write_slurm_array_script_writes_executable_file(
    tmp_path: Path,
) -> None:
    """The write helper should create parents and chmod the script executable."""
    from phenotypic.sdk_.slurm import (
        SlurmArrayScriptSpec,
        write_slurm_array_script,
    )

    spec = SlurmArrayScriptSpec(
        job_name="pht-write",
        slurm_args={},
        log_path=tmp_path / "logs" / "write_%A_%a.log",
        task_indices=[0],
        body="echo ok",
    )
    script_path = tmp_path / "scripts" / "array.sh"

    written = write_slurm_array_script(script_path, spec)

    assert written == script_path
    assert script_path.read_text(encoding="utf-8") == spec.render()
    assert script_path.stat().st_mode & 0o111
