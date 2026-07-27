"""Tests for shared SLURM array script rendering helpers."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
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


def test_slurm_array_script_restores_namespaced_pythonpath_before_prelude(
    tmp_path: Path,
) -> None:
    """Every shared batch script should restore the submitted import path."""
    from phenotypic.sdk_.slurm import (
        SLURM_PYTHONPATH_BOOTSTRAP_BASH,
        SLURM_PYTHONPATH_ENV_VAR,
        SlurmArrayScriptSpec,
    )

    spec = SlurmArrayScriptSpec(
        job_name="pht-pythonpath",
        slurm_args={},
        log_path=tmp_path / "pythonpath_%A_%a.log",
        task_indices=[0],
        prelude='echo "prelude"',
        body="python-worker",
    )

    rendered = spec.render()

    assert SLURM_PYTHONPATH_BOOTSTRAP_BASH in rendered
    assert SLURM_PYTHONPATH_ENV_VAR in rendered
    assert (
        rendered.index(SLURM_PYTHONPATH_BOOTSTRAP_BASH)
        < rendered.index('echo "prelude"')
        < rendered.index("python-worker")
    )


def test_sbatch_submission_environment_snapshots_pythonpath() -> None:
    """Submission should preserve the exact caller path in a safe namespace."""
    from phenotypic.sdk_.slurm import (
        SLURM_PYTHONPATH_ENV_VAR,
        sbatch_submission_environment,
    )

    source = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/reviewed/src:/reviewed/tests",
    }

    submitted = sbatch_submission_environment(source)

    assert submitted == {
        **source,
        SLURM_PYTHONPATH_ENV_VAR: source["PYTHONPATH"],
    }
    assert SLURM_PYTHONPATH_ENV_VAR not in source


@pytest.mark.parametrize("python_path", [None, ""])
def test_sbatch_submission_environment_omits_empty_snapshot(
    python_path: str | None,
) -> None:
    """Missing and empty paths should not propagate stale bootstrap state."""
    from phenotypic.sdk_.slurm import (
        SLURM_PYTHONPATH_ENV_VAR,
        sbatch_submission_environment,
    )

    source = {
        "PATH": "/usr/bin",
        SLURM_PYTHONPATH_ENV_VAR: "/stale/path",
    }
    if python_path is not None:
        source["PYTHONPATH"] = python_path

    submitted = sbatch_submission_environment(source)

    assert SLURM_PYTHONPATH_ENV_VAR not in submitted
    assert submitted.get("PYTHONPATH") == python_path


@pytest.mark.parametrize(
    ("snapshot", "python_path", "expected"),
    [
        ("/reviewed/src:/reviewed/tests", None, "/reviewed/src:/reviewed/tests"),
        (
            "/reviewed/src:/reviewed/tests",
            "/site/modules",
            "/reviewed/src:/reviewed/tests:/site/modules",
        ),
        ("/reviewed/src", "/reviewed/src", "/reviewed/src"),
        (
            "/reviewed path/$NOT_EXPANDED",
            None,
            "/reviewed path/$NOT_EXPANDED",
        ),
        ("", "/site/modules", "/site/modules"),
        (None, None, "<unset>"),
    ],
)
def test_pythonpath_bootstrap_handles_present_empty_and_absent_values(
    snapshot: str | None,
    python_path: str | None,
    expected: str,
) -> None:
    """The shell bootstrap should restore, prepend, or no-op deterministically."""
    from phenotypic.sdk_.slurm import (
        SLURM_PYTHONPATH_BOOTSTRAP_BASH,
        SLURM_PYTHONPATH_ENV_VAR,
    )

    environment = {"PATH": os.environ["PATH"]}
    if snapshot is not None:
        environment[SLURM_PYTHONPATH_ENV_VAR] = snapshot
    if python_path is not None:
        environment["PYTHONPATH"] = python_path

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"{SLURM_PYTHONPATH_BOOTSTRAP_BASH}\n"
                'printf "%s" "${PYTHONPATH-<unset>}"'
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )

    assert result.stdout == expected


@pytest.mark.parametrize(
    ("time_value", "expected"),
    [
        ("00:10:00", "#SBATCH --time=00:10:00"),
        ("1-04:00:00", "#SBATCH --time=1-04:00:00"),
    ],
)
def test_slurm_array_script_accepts_canonical_duration_strings(
    tmp_path: Path,
    time_value: str,
    expected: str,
) -> None:
    from phenotypic.sdk_.slurm import SlurmArrayScriptSpec

    spec = SlurmArrayScriptSpec(
        job_name="pht-time",
        slurm_args={"time": time_value},
        log_path=tmp_path / "time_%A_%a.log",
        task_indices=[0],
        body="echo ok",
    )

    assert expected in spec.render()


def test_slurm_array_script_renders_delayed_bounded_cpu_profile(
    tmp_path: Path,
) -> None:
    """The live cancellation profile holds a CPU job without requesting a GPU."""
    from phenotypic.sdk_.slurm import SlurmArrayScriptSpec

    spec = SlurmArrayScriptSpec(
        job_name="pht-live-cancel",
        slurm_args={
            "slurm_partition": "short",
            "slurm_begin": "now+2minutes",
            "slurm_cpus_per_task": 1,
            "slurm_mem": "4G",
            "time": "00:10:00",
        },
        log_path=tmp_path / "cancel_%A_%a.log",
        task_indices=[0],
        body="echo held",
    )

    rendered = spec.render()

    assert "#SBATCH --partition=short" in rendered
    assert "#SBATCH --begin=now+2minutes" in rendered
    assert "#SBATCH --cpus-per-task=1" in rendered
    assert "#SBATCH --mem=4G" in rendered
    assert "#SBATCH --time=00:10:00" in rendered
    assert "#SBATCH --gpus" not in rendered


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
