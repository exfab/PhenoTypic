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
    state_from_controls,
    to_argv,
)
from phenotypic.gui.shell._metadata_context import metadata_payload_from_path
from phenotypic.gui.shell._sandbox import SandboxRoot


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
        metadata_csv="/p/layout.csv",
        mode="slurm",
        dry_run=True,
        retry_failures=True,
        advanced_args={"sample": 4, "nrows": 8, "ncols": 12, "image_type": "tif"},
        slurm_args={
            "partition": "compute",
            "time": "01:00:00",
            "mem": "16G",
            "cpus_per_task": 4,
            "gpus": 1,
            "extra": {"qos": "bench"},
        },
        gpu_slurm_args=(
            "slurm_partition=gpu",
            "slurm_gpus_per_node=1",
        ),
        gpu_shards=3,
    )
    assert run_state_from_json(run_state_to_json(state)) == state


def test_from_json_tolerates_missing_fields() -> None:
    """Older preset files (saved before fields were added) still load."""
    state = run_state_from_json({"pipeline_path": "/p.json"})
    assert state.pipeline_path == "/p.json"
    assert state.input_dir is None
    assert state.mode == "local"  # defaulted
    assert state.dry_run is False
    assert not hasattr(state, "save_inspect")


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


def test_from_json_tolerates_legacy_gpu_profile_shapes() -> None:
    state = run_state_from_json(
        {
            "gpu_slurm_args": {
                "slurm_partition": "gpu",
                "slurm_account": "lab",
            },
            "gpu_shards": "invalid",
        }
    )
    assert state.gpu_slurm_args == (
        "slurm_partition=gpu",
        "slurm_account=lab",
    )
    assert state.gpu_shards == 1


# ---------------------------------------------------------------------------
# Raw controls
# ---------------------------------------------------------------------------


def _raw_controls(sandbox: SandboxRoot) -> dict[str, object]:
    return {
        "pipeline_path": "pipeline.json",
        "input_dir": "images",
        "output_dir": "output",
        "mode": "slurm",
        "flags": ["dry_run", "retry_failures"],
        "sample": 2,
        "nrows": 8,
        "ncols": 12,
        "image_type": "gridimage",
        "workers": 4,
        "log_level": "INFO",
        "slurm_partition": "compute",
        "slurm_time": "10",
        "slurm_mem": "16G",
        "slurm_cpus": 4,
        "slurm_gpus": 0,
        "slurm_extra": "account=lab\nqos=normal",
        "metadata_payload": None,
        "sandbox": sandbox,
        "gpu_slurm": "slurm_partition=gpu\ntime=01:30:00",
        "gpu_shards": 2,
    }


def test_state_from_controls_uses_raw_visible_values(tmp_path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "pipeline.json").write_text("{}", encoding="utf-8")
    sandbox = SandboxRoot.from_path(tmp_path)
    controls = _raw_controls(sandbox)

    state = state_from_controls(**controls)

    assert state.pipeline_path == str(tmp_path / "pipeline.json")
    assert state.input_dir == str(tmp_path / "images")
    assert state.output_dir == str(tmp_path / "output")
    assert state.mode == "slurm"
    assert state.dry_run is True
    assert state.retry_failures is True
    assert state.advanced_args == {
        "sample": 2,
        "nrows": 8,
        "ncols": 12,
        "image_type": "GridImage",
        "workers": 4,
        "log_level": "INFO",
    }
    assert state.slurm_args == {
        "partition": "compute",
        "time": "00:10:00",
        "mem": "16G",
        "cpus_per_task": 4,
        "gpus": 0,
        "extra": {"account": "lab", "qos": "normal"},
    }
    assert state.gpu_slurm_args == (
        "slurm_partition=gpu",
        "time=01:30:00",
    )
    assert state.gpu_shards == 2


def test_state_from_controls_resolves_metadata_payload(tmp_path) -> None:
    metadata = tmp_path / "layout.csv"
    metadata.write_text("Metadata_ImageName\nplate_a\n", encoding="utf-8")
    sandbox = SandboxRoot.from_path(tmp_path)
    controls = _raw_controls(sandbox)
    controls["mode"] = "local"
    controls["metadata_payload"] = metadata_payload_from_path(
        sandbox, metadata
    )

    state = state_from_controls(**controls)

    assert state.metadata_csv == str(metadata)


def test_state_from_controls_rejects_empty_slurm_profile(tmp_path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    controls = _raw_controls(sandbox)
    for key in (
        "slurm_partition",
        "slurm_time",
        "slurm_mem",
        "slurm_cpus",
        "slurm_gpus",
        "slurm_extra",
    ):
        controls[key] = None

    with pytest.raises(ValueError, match="nonempty CPU SLURM profile"):
        state_from_controls(**controls)


def test_state_from_controls_rejects_path_escape(tmp_path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    controls = _raw_controls(sandbox)
    controls["pipeline_path"] = tmp_path.parent / "outside.json"

    with pytest.raises(ValueError, match="outside the GUI sandbox"):
        state_from_controls(**controls)


# ---------------------------------------------------------------------------
# to_argv
# ---------------------------------------------------------------------------

def test_to_argv_emits_required_explicit_path_options() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
    )
    assert to_argv(state) == [
        "--mode",
        "full",
        "--pipeline",
        "/p.json",
        "--input",
        "/in",
        "--output",
        "/out",
    ]


def test_to_argv_includes_metadata_when_selected() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
        metadata_csv="/layout.csv",
    )
    argv = to_argv(state)
    assert argv[argv.index("--metadata") + 1] == "/layout.csv"


def test_to_argv_omits_metadata_when_unset() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
    )
    argv = to_argv(state)
    assert "--metadata" not in argv


def test_to_argv_includes_dry_run_flag() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json", input_dir="/in", output_dir="/out", dry_run=True,
    )
    argv = to_argv(state)
    assert "--dry-run" in argv


def test_to_argv_includes_retry_failures_flag() -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir="/out",
        retry_failures=True,
    )
    argv = to_argv(state)
    assert "--retry-failures" in argv


def test_legacy_resume_preset_is_ignored() -> None:
    state = run_state_from_json(
        {
            "resume": True,
            "pipeline_path": "/p.json",
            "input_dir": "/in",
            "output_dir": "/out",
        }
    )
    assert "--resume" not in to_argv(state)


def test_legacy_save_inspect_preset_is_ignored() -> None:
    state = run_state_from_json({"save_inspect": True})
    assert not hasattr(state, "save_inspect")


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
    # Skip the first 8 (mode + path options + output); the rest is
    # ``--flag value`` pairs in unspecified order.
    tail = argv[8:]
    pair_dict: dict[str, str] = {}
    for i in range(0, len(tail), 2):
        pair_dict[tail[i]] = tail[i + 1]
    assert pair_dict["--sample"] == "2"
    assert pair_dict["--nrows"] == "8"
    assert pair_dict["--ncols"] == "12"
    assert pair_dict["--image-type"] == "jpg"
    # ``workers`` maps to the CLI's ``--njobs``.
    assert pair_dict["--njobs"] == "4"


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
