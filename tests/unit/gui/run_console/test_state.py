"""Unit tests for ``phenotypic.gui.run_console._state``.

The Run console form's per-session state lives behind a tiny mutable
dataclass plus a JSON round-trip pair. The argv translator is shared
across local + SLURM modes — it's the single source of truth for CLI
flags, so it gets the most coverage here.
"""
from __future__ import annotations

import pytest

from phenotypic.gui.run_console._state import (
    RunConsoleState,
    run_state_from_json,
    run_state_to_json,
    to_argv,
)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_default_state_round_trips_cleanly() -> None:
    state = RunConsoleState()
    payload = run_state_to_json(state)
    assert run_state_from_json(payload) == state


def test_full_state_round_trips_cleanly() -> None:
    state = RunConsoleState(
        pipeline_path="/p/pipeline.json",
        input_dir="/p/in",
        output_dir="/p/out",
        mode="slurm",
        dry_run=True,
        resume=True,
        save_inspect=True,
        advanced_args={"sample": 4, "nrows": 8, "ncols": 12, "image_type": "tif"},
        slurm_args={
            "partition": "compute",
            "time": "01:00:00",
            "mem": "16G",
            "cpus_per_task": 4,
            "gpus": 1,
            "extra": {"qos": "bench"},
        },
    )
    assert run_state_from_json(run_state_to_json(state)) == state


def test_from_json_tolerates_missing_fields() -> None:
    """Older preset files (saved before fields were added) still load."""
    state = run_state_from_json({"pipeline_path": "/p.json"})
    assert state.pipeline_path == "/p.json"
    assert state.input_dir is None
    assert state.mode == "local"  # defaulted
    assert state.dry_run is False
    assert state.save_inspect is False


def test_from_json_normalises_extras_to_strings() -> None:
    state = run_state_from_json(
        {"slurm_args": {"extra": {"qos": "bench", "priority": 5}}}
    )
    extras = state.slurm_args["extra"]
    assert extras == {"qos": "bench", "priority": "5"}


def test_from_json_rejects_unknown_mode_with_default() -> None:
    """Unknown ``mode`` strings degrade to ``"local"`` rather than raising."""
    state = run_state_from_json({"mode": "kubernetes"})
    assert state.mode == "local"


def test_empty_string_path_coerced_to_none() -> None:
    state = run_state_from_json({"pipeline_path": "   ", "input_dir": ""})
    assert state.pipeline_path is None
    assert state.input_dir is None


# ---------------------------------------------------------------------------
# to_argv
# ---------------------------------------------------------------------------

def test_to_argv_emits_required_positional_args() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
    )
    assert to_argv(state) == ["/p.json", "/in", "-o", "/out"]


def test_to_argv_includes_dry_run_flag() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json", input_dir="/in", output_dir="/out", dry_run=True,
    )
    argv = to_argv(state)
    assert "--dry-run" in argv


def test_to_argv_includes_resume_flag() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json", input_dir="/in", output_dir="/out", resume=True,
    )
    argv = to_argv(state)
    assert "--resume" in argv


def test_to_argv_includes_save_inspect_flag() -> None:
    """``--save-inspect`` is appended when ``state.save_inspect=True``."""
    state = RunConsoleState(
        pipeline_path="/p.json", input_dir="/in", output_dir="/out", save_inspect=True,
    )
    argv = to_argv(state)
    assert "--save-inspect" in argv


def test_to_argv_omits_save_inspect_flag_when_disabled() -> None:
    """No ``--save-inspect`` token unless explicitly enabled."""
    state = RunConsoleState(
        pipeline_path="/p.json", input_dir="/in", output_dir="/out",
    )
    argv = to_argv(state)
    assert "--save-inspect" not in argv


def test_to_argv_threads_advanced_args() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
        advanced_args={
            "sample": 2,
            "nrows": 8,
            "ncols": 12,
            "image_type": "jpg",
            "workers": 4,
        },
    )
    argv = to_argv(state)
    # Skip the first 4 (positional + -o + output_dir); the rest is
    # ``--flag value`` pairs in unspecified order.
    tail = argv[4:]
    pair_dict: dict[str, str] = {}
    for i in range(0, len(tail), 2):
        pair_dict[tail[i]] = tail[i + 1]
    assert pair_dict["--sample"] == "2"
    assert pair_dict["--nrows"] == "8"
    assert pair_dict["--ncols"] == "12"
    assert pair_dict["--image-type"] == "jpg"
    # ``workers`` maps to the CLI's ``--n-jobs``.
    assert pair_dict["--n-jobs"] == "4"


def test_to_argv_skips_unset_advanced_args() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
        advanced_args={"sample": None, "image_type": ""},
    )
    argv = to_argv(state)
    assert "--sample" not in argv
    assert "--image-type" not in argv


def test_to_argv_does_not_emit_log_level() -> None:
    """``log_level`` is intentionally not forwarded — CLI lacks the flag."""
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
        advanced_args={"log_level": "DEBUG"},
    )
    argv = to_argv(state)
    assert "--log-level" not in argv


def test_to_argv_does_not_inject_slurm_flags() -> None:
    """SLURM argv extension is the SLURM runner's responsibility, not state's."""
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
        mode="slurm",
        slurm_args={"partition": "compute"},
    )
    argv = to_argv(state)
    assert "--slurm" not in argv


def test_to_argv_raises_on_missing_required_fields() -> None:
    state = RunConsoleState(input_dir="/in")
    with pytest.raises(ValueError) as exc:
        to_argv(state)
    msg = str(exc.value)
    assert "pipeline_path" in msg
    assert "output_dir" in msg


# ---------------------------------------------------------------------------
# Mutability sanity
# ---------------------------------------------------------------------------

def test_state_is_mutable() -> None:
    """Match builder convention — frozen would force replace() everywhere."""
    state = RunConsoleState()
    state.pipeline_path = "/p.json"
    state.advanced_args["sample"] = 3
    assert state.pipeline_path == "/p.json"
    assert state.advanced_args == {"sample": 3}
