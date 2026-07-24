"""Browser-level Run Console lifecycle test with fake SLURM executables."""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page

from phenotypic import ImagePipeline
from phenotypic._cli._cli_slurm_lifecycle import lifecycle_state_path
from phenotypic.detect import OtsuDetector
from phenotypic.sdk_ import (
    CONFIG_SUFFIX_PIPELINE,
    ensure_typed_json_suffix,
    gui_launch_owner_path,
    job_metadata_path,
)
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server
from tests.e2e.gui.test_run_console import _set_action_controls


_FAKE_SLURM = """#!{interpreter}
import json
import os
import sys
from pathlib import Path

state_path = Path(os.environ["PHENOTYPIC_FAKE_SLURM_STATE"])
command = Path(sys.argv[0]).name
args = sys.argv[1:]

def load():
    if not state_path.exists():
        return {{"next_id": 4700, "jobs": {{}}}}
    return json.loads(state_path.read_text(encoding="utf-8"))

def save(state):
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.replace(state_path)

state = load()
if command == "sbatch":
    job_id = str(state["next_id"])
    state["next_id"] += 1
    comment = ""
    if "--comment" in args:
        comment = args[args.index("--comment") + 1]
    state["jobs"][job_id] = {{"comment": comment, "state": "PENDING"}}
    save(state)
    print(job_id)
elif command == "squeue":
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
    cancelled = []
    for job_id in args:
        if job_id in state["jobs"]:
            state["jobs"][job_id]["state"] = "CANCELLED"
            cancelled.append(job_id)
    state["cancelled"] = sorted(
        set(state.get("cancelled", [])) | set(cancelled)
    )
    save(state)
"""


def _write_fake_slurm_bin(root: Path, state_path: Path) -> Path:
    """Write deterministic scheduler commands sharing one JSON state file."""
    bin_dir = root / "fake-slurm-bin"
    bin_dir.mkdir()
    script = _FAKE_SLURM.format(interpreter=sys.executable)
    for command in ("sbatch", "squeue", "sacct", "scancel"):
        executable = bin_dir / command
        executable.write_text(script, encoding="utf-8")
        executable.chmod(0o755)
    state_path.write_text(
        json.dumps({"next_id": 4700, "jobs": {}}),
        encoding="utf-8",
    )
    return bin_dir


@pytest.fixture
def fake_slurm_hub(tmp_path: Path) -> Iterator[tuple[str, Path, Path]]:
    """Boot the real hub with fake scheduler commands ahead of ``PATH``."""
    sandbox = _build_sandbox(tmp_path)
    state_path = tmp_path / "fake-slurm-state.json"
    bin_dir = _write_fake_slurm_bin(tmp_path, state_path)
    env = {
        "PATH": os.pathsep.join((str(bin_dir), os.environ.get("PATH", ""))),
        "PHENOTYPIC_FAKE_SLURM_STATE": str(state_path),
    }
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
