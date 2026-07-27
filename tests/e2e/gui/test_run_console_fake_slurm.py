"""Browser-level Run Console lifecycle test with fake SLURM executables."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw
from playwright.sync_api import Page

from phenotypic import ImagePipeline
from phenotypic._cli._cli_slurm_lifecycle import lifecycle_state_path
from phenotypic._cli._cli_staged_orchestration import update_job_dependency
from phenotypic.detect import OtsuDetector
from phenotypic.sdk_ import (
    CONFIG_SUFFIX_PIPELINE,
    dashboard_html_path,
    ensure_typed_json_suffix,
    gui_launch_owner_path,
    job_metadata_path,
    run_completion_marker_path,
)
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server
from tests.e2e.gui.test_run_console import _set_action_controls


_FAKE_SLURM = """#!{interpreter}
import json
import os
import fcntl
import re
import subprocess
import sys
import time
from pathlib import Path

state_path = Path(os.environ["PHENOTYPIC_FAKE_SLURM_STATE"])
lock_path = state_path.with_suffix(".lock")
command = Path(sys.argv[0]).name
args = sys.argv[1:]

def load():
    if not state_path.exists():
        return {{"next_id": 4700, "jobs": {{}}}}
    return json.loads(state_path.read_text(encoding="utf-8"))

def save(state):
    temporary = state_path.with_suffix(f".{{os.getpid()}}.tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.replace(state_path)

def update_job(job_id, **fields):
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = load()
        if job_id in state["jobs"]:
            state["jobs"][job_id].update(fields)
            save(state)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

if args and args[0] == "__run":
    job_id, script = args[1], Path(args[2])
    while True:
        state = load()
        dependencies = state["jobs"][job_id].get("dependencies", [])
        dependency_kind = state["jobs"][job_id].get(
            "dependency_kind", "afterok"
        )
        dependency_states = [
            state["jobs"].get(dependency, {{}}).get("state")
            for dependency in dependencies
        ]
        terminal_states = {{"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}}
        if dependency_kind == "afterok" and any(
            value in terminal_states - {{"COMPLETED"}}
            for value in dependency_states
        ):
            update_job(job_id, state="CANCELLED", completed_at=time.time())
            raise SystemExit(1)
        if dependency_kind == "afterany" and all(
            value in terminal_states for value in dependency_states
        ):
            break
        if dependency_kind == "afterok" and all(
            value == "COMPLETED" for value in dependency_states
        ):
            break
        time.sleep(0.02)
    update_job(job_id, state="RUNNING", started_at=time.time())
    match = re.search(
        r"^#SBATCH --array=0-(\\d+)",
        script.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    last_index = int(match.group(1)) if match else 0
    delays = json.loads(
        os.environ.get("PHENOTYPIC_FAKE_SLURM_TASK_DELAYS", "{{}}")
    )
    processes = {{}}
    task_states = {{}}
    for task_id in range(last_index + 1):
        env = os.environ.copy()
        env.update(
            {{
                "SLURM_ARRAY_TASK_ID": str(task_id),
                "SLURM_ARRAY_JOB_ID": job_id,
                "SLURM_JOB_ID": job_id,
            }}
        )
        delay = max(0.0, float(delays.get(str(task_id), 0.0)))
        command_args = ["bash", str(script)]
        if delay:
            command_args = [
                "bash",
                "-c",
                f"sleep {{delay}}; exec bash \\"$1\\"",
                "_",
                str(script),
            ]
        processes[task_id] = subprocess.Popen(
            command_args,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        task_states[str(task_id)] = {{
            "started_at": time.time(),
            "state": "RUNNING",
        }}
    update_job(job_id, tasks=task_states)
    returncodes = {{}}
    while len(returncodes) < len(processes):
        for task_id, process in processes.items():
            if task_id in returncodes:
                continue
            task_returncode = process.poll()
            if task_returncode is None:
                continue
            returncodes[task_id] = task_returncode
            task_states[str(task_id)] = {{
                **task_states[str(task_id)],
                "completed_at": time.time(),
                "returncode": task_returncode,
                "state": (
                    "COMPLETED" if task_returncode == 0 else "FAILED"
                ),
            }}
            update_job(job_id, tasks=task_states)
        time.sleep(0.01)
    returncode = next(
        (value for value in returncodes.values() if value != 0),
        0,
    )
    update_job(
        job_id,
        state="COMPLETED" if returncode == 0 else "FAILED",
        completed_at=time.time(),
        tasks=task_states,
    )
    raise SystemExit(returncode)

if command == "sbatch":
    comment = ""
    if "--comment" in args:
        comment = args[args.index("--comment") + 1]
    dependencies = []
    dependency_kind = "afterok"
    dependency_value = None
    if "--dependency" in args:
        dependency_value = args[args.index("--dependency") + 1]
    for arg in args:
        if arg.startswith("--dependency="):
            dependency_value = arg.split("=", 1)[1]
    if dependency_value is not None:
        dependency_kind, dependency_ids = dependency_value.split(":", 1)
        dependencies.extend(dependency_ids.split(","))
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = load()
        job_id = str(state["next_id"])
        state["next_id"] += 1
        state["jobs"][job_id] = {{
            "comment": comment,
            "dependency_kind": dependency_kind,
            "dependencies": dependencies,
            "state": "PENDING",
        }}
        save(state)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    if os.environ.get("PHENOTYPIC_FAKE_SLURM_AUTORUN") == "1":
        subprocess.Popen(
            [sys.executable, __file__, "__run", job_id, args[-1]],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(job_id)
elif command == "squeue":
    state = load()
    if any("%k" in arg for arg in args):
        for job_id, job in sorted(state["jobs"].items()):
            if job["state"] in {{"PENDING", "RUNNING"}}:
                print(f"{{job_id}}|{{job['comment']}}")
    elif any("%T" in arg for arg in args):
        requested = None
        if "--jobs" in args:
            requested = set(args[args.index("--jobs") + 1].split(","))
        for job_id, job in sorted(state["jobs"].items()):
            if requested is not None and job_id not in requested:
                continue
            if job["state"] in {{"PENDING", "RUNNING"}}:
                print(f"{{job_id}}|{{job['state']}}")
elif command == "sacct":
    state = load()
    if any("Comment" in arg for arg in args):
        for job_id, job in sorted(state["jobs"].items()):
            print(f"{{job_id}}|{{job['comment']}}")
    elif any("State" in arg for arg in args):
        requested = None
        if "--jobs" in args:
            requested = set(args[args.index("--jobs") + 1].split(","))
        for job_id, job in sorted(state["jobs"].items()):
            if requested is None or job_id in requested:
                print(f"{{job_id}}|{{job['state']}}")
elif command == "scancel":
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = load()
        cancelled = []
        for job_id in args:
            if job_id in state["jobs"]:
                state["jobs"][job_id]["state"] = "CANCELLED"
                cancelled.append(job_id)
        state["cancelled"] = sorted(
            set(state.get("cancelled", [])) | set(cancelled)
        )
        save(state)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
elif command == "scontrol":
    if not args or args[0] != "update":
        raise SystemExit(2)
    job_id = next(
        value.split("=", 1)[1]
        for value in args
        if value.startswith("JobId=")
    )
    dependency = next(
        value.split("=", 1)[1]
        for value in args
        if value.startswith("Dependency=")
    )
    dependency_kind, dependency_ids = dependency.split(":", 1)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = load()
        if job_id not in state["jobs"]:
            raise SystemExit(1)
        state["jobs"][job_id]["dependency_kind"] = dependency_kind
        state["jobs"][job_id]["dependencies"] = [
            item for item in re.split(r"[:,]", dependency_ids) if item
        ]
        save(state)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
"""


def _write_fake_slurm_bin(root: Path, state_path: Path) -> Path:
    """Write deterministic scheduler commands sharing one JSON state file."""
    bin_dir = root / "fake-slurm-bin"
    bin_dir.mkdir()
    script = _FAKE_SLURM.format(interpreter=sys.executable)
    for command in ("sbatch", "squeue", "sacct", "scancel", "scontrol"):
        executable = bin_dir / command
        executable.write_text(script, encoding="utf-8")
        executable.chmod(0o755)
    state_path.write_text(
        json.dumps({"next_id": 4700, "jobs": {}}),
        encoding="utf-8",
    )
    return bin_dir


def _serve_fake_slurm_hub(
    tmp_path: Path,
    *,
    autorun: bool,
) -> Iterator[tuple[str, Path, Path]]:
    """Boot the real hub with fake scheduler commands ahead of ``PATH``."""
    sandbox = _build_sandbox(tmp_path)
    state_path = tmp_path / "fake-slurm-state.json"
    bin_dir = _write_fake_slurm_bin(tmp_path, state_path)
    env = {
        "PATH": os.pathsep.join((str(bin_dir), os.environ.get("PATH", ""))),
        "PHENOTYPIC_FAKE_SLURM_STATE": str(state_path),
    }
    if autorun:
        env["PHENOTYPIC_FAKE_SLURM_AUTORUN"] = "1"
    # The production launcher enables background observers by default, but
    # its pytest guard sees the parent's phase variable. Remove that variable
    # only while Popen captures its environment, then restore it immediately.
    pytest_phase = os.environ.pop("PYTEST_CURRENT_TEST", None)
    server = _start_live_server(sandbox, env_overrides=env)
    try:
        url = next(server)
    finally:
        if pytest_phase is not None:
            os.environ["PYTEST_CURRENT_TEST"] = pytest_phase
    try:
        yield url, sandbox, state_path
    finally:
        server.close()


@pytest.fixture
def fake_slurm_hub(tmp_path: Path) -> Iterator[tuple[str, Path, Path]]:
    """Boot a fake scheduler whose submitted jobs remain pending."""
    yield from _serve_fake_slurm_hub(tmp_path, autorun=False)


@pytest.fixture
def fake_slurm_success_hub(
    tmp_path: Path,
) -> Iterator[tuple[str, Path, Path]]:
    """Boot a fake scheduler that executes dependency-ordered jobs."""
    yield from _serve_fake_slurm_hub(tmp_path, autorun=True)


def _wait_for_status(
    output_dir: Path,
    expected: set[str],
    *,
    timeout: float = 20.0,
) -> dict[str, object]:
    """Wait for one durable owner status emitted by the server process."""
    owner_path = gui_launch_owner_path(output_dir)
    deadline = time.monotonic() + timeout
    last: dict[str, object] | None = None
    while time.monotonic() < deadline:
        if owner_path.is_file():
            last = json.loads(owner_path.read_text(encoding="utf-8"))
            if str(last.get("status")) in expected:
                return last
        time.sleep(0.05)
    raise AssertionError(f"owner did not reach {sorted(expected)}: {last!r}")


def _wait_for_scheduler_terminal(
    state_path: Path,
    *,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Wait until every submitted fake scheduler job is terminal."""
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
    deadline = time.monotonic() + timeout
    last: dict[str, object] | None = None
    while time.monotonic() < deadline:
        last = json.loads(state_path.read_text(encoding="utf-8"))
        jobs = last.get("jobs", {})
        if (
            isinstance(jobs, dict)
            and len(jobs) >= 2
            and all(
                isinstance(job, dict)
                and str(job.get("state")) in terminal_states
                for job in jobs.values()
            )
        ):
            return last
        time.sleep(0.05)
    raise AssertionError(f"scheduler jobs did not become terminal: {last!r}")


def _submit_process_export(
    tmp_path: Path,
    *,
    valid_image: bool,
) -> tuple[Path, dict[str, object]]:
    """Submit a process/export run through production CLI and fake ``sbatch``."""
    state_path = tmp_path / "fake-process-slurm-state.json"
    bin_dir = _write_fake_slurm_bin(tmp_path, state_path)
    pipeline_base = tmp_path / "process-pipeline.json"
    ImagePipeline(ops=[OtsuDetector()]).to_json(pipeline_base)
    pipeline_path = ensure_typed_json_suffix(
        pipeline_base,
        CONFIG_SUFFIX_PIPELINE,
    )
    input_dir = tmp_path / "process-input"
    input_dir.mkdir()
    image_path = input_dir / "plate.tiff"
    if valid_image:
        PILImage.new("RGB", (32, 32), (120, 80, 40)).save(image_path)
    else:
        image_path.write_bytes(b"invalid-tiff")
        PILImage.new("RGB", (32, 32), (80, 120, 40)).save(
            input_dir / "slow-sibling.tiff"
        )
    output_dir = tmp_path / "process-output"
    env = os.environ.copy()
    env.update(
        {
            "PATH": os.pathsep.join(
                (str(bin_dir), os.environ.get("PATH", ""))
            ),
            "PHENOTYPIC_FAKE_SLURM_AUTORUN": "1",
            "PHENOTYPIC_FAKE_SLURM_STATE": str(state_path),
        }
    )
    if not valid_image:
        env["PHENOTYPIC_FAKE_SLURM_TASK_DELAYS"] = json.dumps({"1": 0.75})
    submitted = subprocess.run(
        [
            sys.executable,
            "-m",
            "phenotypic",
            "--pipeline",
            str(pipeline_path),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--mode",
            "process",
            "--layer",
            "gray",
            "--slurm",
            "slurm_partition=compute",
            "--skip-validation",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert submitted.returncode == 0, submitted.stderr
    return output_dir, _wait_for_scheduler_terminal(state_path)


def _terminal_finalizer_job(
    scheduler_state: dict[str, object],
) -> tuple[str, dict[str, object]]:
    """Return the fake scheduler row carrying the terminal-finalizer token."""
    jobs = scheduler_state["jobs"]
    assert isinstance(jobs, dict)
    matches = [
        (str(job_id), job)
        for job_id, job in jobs.items()
        if isinstance(job, dict)
        and str(job.get("comment", "")).endswith(":finalizer")
    ]
    assert len(matches) == 1
    return matches[0]


def test_ordinary_slurm_submit_and_cancel_is_generation_fenced(
    page: Page,
    fake_slurm_hub: tuple[str, Path, Path],
) -> None:
    """The browser action binds the submitted epoch and cancels it to quiescence."""
    hub_url, sandbox, scheduler_state_path = fake_slurm_hub
    pipeline_base = sandbox / "ordinary-pipeline.json"
    ImagePipeline(ops=[OtsuDetector()]).to_json(pipeline_base)
    pipeline_path = ensure_typed_json_suffix(
        pipeline_base,
        CONFIG_SUFFIX_PIPELINE,
    )
    input_dir = sandbox / "ordinary-input"
    input_dir.mkdir()
    (input_dir / "plate.tiff").write_bytes(b"not-read-by-submitter")
    output_dir = sandbox / "results" / "FakeSlurmOrdinary"
    output_dir.mkdir()

    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-run")
    _set_action_controls(
        page,
        pipeline=pipeline_path,
        input_dir=input_dir,
        output_dir=output_dir,
        modes=["slurm"],
    )
    page.locator("#rc-btn-run").click()

    submitted = _wait_for_status(
        output_dir,
        {"queued", "running", "reconciling"},
    )
    metadata = json.loads(
        job_metadata_path(output_dir).read_text(encoding="utf-8")
    )
    lifecycle = json.loads(
        lifecycle_state_path(output_dir).read_text(encoding="utf-8")
    )
    assert submitted["generation"]
    assert metadata["gui_record_generation"] == submitted["generation"]
    assert metadata["slurm_generation"] == lifecycle["generation"]
    assert metadata["slurm_job_ids"]

    page.locator("#rc-btn-cancel").click()
    cancelled = _wait_for_status(output_dir, {"cancelled"})
    scheduler_state = json.loads(
        scheduler_state_path.read_text(encoding="utf-8")
    )
    lifecycle = json.loads(
        lifecycle_state_path(output_dir).read_text(encoding="utf-8")
    )

    assert cancelled["terminal_at"]
    assert lifecycle["active"] is False
    assert scheduler_state["cancelled"]


def test_ordinary_slurm_array_and_finalizer_publish_terminal_artifacts(
    page: Page,
    fake_slurm_success_hub: tuple[str, Path, Path],
) -> None:
    """A real array and dependent finalizer drive the owner to ``complete``."""
    hub_url, sandbox, scheduler_state_path = fake_slurm_success_hub
    pipeline_base = sandbox / "success-pipeline.json"
    ImagePipeline(ops=[OtsuDetector()]).to_json(pipeline_base)
    pipeline_path = ensure_typed_json_suffix(
        pipeline_base,
        CONFIG_SUFFIX_PIPELINE,
    )
    input_dir = sandbox / "success-input"
    input_dir.mkdir()
    plate = PILImage.new("RGB", (64, 64), (0, 0, 0))
    ImageDraw.Draw(plate).ellipse((16, 16, 48, 48), fill=(255, 255, 255))
    plate.save(input_dir / "plate.tiff")
    output_dir = sandbox / "results" / "FakeSlurmSuccess"
    output_dir.mkdir()

    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-btn-run")
    _set_action_controls(
        page,
        pipeline=pipeline_path,
        input_dir=input_dir,
        output_dir=output_dir,
        modes=["slurm"],
    )
    page.locator("#rc-btn-run").click()

    completed = _wait_for_status(output_dir, {"complete", "failed"}, timeout=90)
    scheduler_state = json.loads(
        scheduler_state_path.read_text(encoding="utf-8")
    )
    metadata = json.loads(
        job_metadata_path(output_dir).read_text(encoding="utf-8")
    )
    marker = json.loads(
        run_completion_marker_path(output_dir).read_text(encoding="utf-8")
    )
    lifecycle = json.loads(
        lifecycle_state_path(output_dir).read_text(encoding="utf-8")
    )

    assert completed["status"] == "complete", completed.get("status_detail")
    assert completed["lifecycle_epoch"] == lifecycle["generation"]
    assert marker["generation"] == metadata["slurm_generation"]
    assert dashboard_html_path(output_dir).is_file()
    assert lifecycle["active"] is False
    assert all(
        job["state"] == "COMPLETED"
        for job in scheduler_state["jobs"].values()
    )


def test_process_export_finalizer_waits_for_successful_chunk(
    tmp_path: Path,
) -> None:
    """The process finalizer records ``afterany`` and publishes success."""
    output_dir, scheduler_state = _submit_process_export(
        tmp_path,
        valid_image=True,
    )
    finalizer_id, finalizer = _terminal_finalizer_job(scheduler_state)
    jobs = scheduler_state["jobs"]
    assert isinstance(jobs, dict)
    chunk_ids = [str(job_id) for job_id in jobs if str(job_id) != finalizer_id]

    assert finalizer["dependency_kind"] == "afterany"
    assert finalizer["dependencies"] == chunk_ids
    assert finalizer["state"] == "COMPLETED"
    assert run_completion_marker_path(output_dir).is_file()


def test_process_export_finalizer_runs_after_failed_chunk_without_marker(
    tmp_path: Path,
) -> None:
    """``afterany`` releases the finalizer, which rejects a failed manifest."""
    output_dir, scheduler_state = _submit_process_export(
        tmp_path,
        valid_image=False,
    )
    finalizer_id, finalizer = _terminal_finalizer_job(scheduler_state)
    jobs = scheduler_state["jobs"]
    assert isinstance(jobs, dict)
    chunk_ids = [str(job_id) for job_id in jobs if str(job_id) != finalizer_id]

    assert finalizer["dependency_kind"] == "afterany"
    assert finalizer["dependencies"] == chunk_ids
    assert finalizer["state"] == "FAILED"
    assert any(jobs[chunk_id]["state"] == "FAILED" for chunk_id in chunk_ids)
    chunk = jobs[chunk_ids[0]]
    assert all(
        task["state"] in {"COMPLETED", "FAILED"}
        for task in chunk["tasks"].values()
    )
    assert finalizer["started_at"] >= chunk["completed_at"]
    assert not run_completion_marker_path(output_dir).exists()


def test_staged_dependency_retargeting_uses_fake_scontrol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staged controller's retarget path updates a pending fake job."""
    state_path = tmp_path / "fake-staged-slurm-state.json"
    bin_dir = _write_fake_slurm_bin(tmp_path, state_path)
    state = {
        "next_id": 4703,
        "jobs": {
            "4700": {"comment": "chunk", "state": "RUNNING"},
            "4701": {"comment": "controller", "state": "PENDING"},
            "4702": {"comment": "stage2", "state": "RUNNING"},
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(bin_dir), os.environ.get("PATH", ""))),
    )
    monkeypatch.setenv("PHENOTYPIC_FAKE_SLURM_STATE", str(state_path))

    assert update_job_dependency("4701", ["4700", "4702"]) is True

    updated = json.loads(state_path.read_text(encoding="utf-8"))
    controller = updated["jobs"]["4701"]
    assert controller["dependency_kind"] == "afterany"
    assert controller["dependencies"] == ["4700", "4702"]
