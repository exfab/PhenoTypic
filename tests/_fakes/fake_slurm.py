"""Deterministic fake SLURM executables sharing one JSON state file.

``sbatch``, ``squeue``, ``sacct``, ``scancel``, ``scontrol`` and ``sinfo``
are one Python script. With ``PHENOTYPIC_FAKE_SLURM_AUTORUN=1``, ``sbatch``
runs each job once its ``--dependency`` is satisfied, every array index as a
separate process, and records the outcome -- so a whole CLI run, finalizer
chain included, executes for real with only the scheduler replaced.

Shared by the Run Console e2e suite (``tests/e2e/gui``) and the CLI
integration suite (``tests/integration/cli``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


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

if command == "sbatch" and "--test-only" in args:
    # The CLI run preflight validates each profile with ``sbatch
    # --test-only`` (script on stdin). Real sbatch answers on stderr and
    # registers nothing, so neither does the fake.
    sys.stdin.read()
    print("sbatch: Job 0 to start at now using 1 processors on nodes fake", file=sys.stderr)
elif command == "sbatch":
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
elif command == "sinfo":
    # Partition GRES for the preflight's GPU-partition check.
    print("gpu:1")
elif command == "scontrol" and args[:2] == ["show", "partition"]:
    name = args[2] if len(args) > 2 else "fake"
    print(f"PartitionName={{name}}")
    print("   AllowGroups=ALL Default=YES")
    print("   MaxTime=UNLIMITED MinNodes=0")
elif command == "scontrol" and args[:2] == ["show", "config"]:
    print("EnforcePartLimits       = NO")
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


def write_fake_slurm_bin(root: Path, state_path: Path) -> Path:
    """Write deterministic scheduler commands sharing one JSON state file."""
    bin_dir = root / "fake-slurm-bin"
    bin_dir.mkdir()
    script = _FAKE_SLURM.format(interpreter=sys.executable)
    for command in ("sbatch", "squeue", "sacct", "scancel", "scontrol", "sinfo"):
        executable = bin_dir / command
        executable.write_text(script, encoding="utf-8")
        executable.chmod(0o755)
    state_path.write_text(
        json.dumps({"next_id": 4700, "jobs": {}}),
        encoding="utf-8",
    )
    return bin_dir
