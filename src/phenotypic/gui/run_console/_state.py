"""Per-session UI state model for the Run console (Phase 6).

This module defines :class:`RunConsoleState`, the small dataclass backing
the form widgets in the Run console. It is intentionally pure-Python so it
travels through ``dcc.Store`` without any Dash dependency. The shape mirrors
the convention used by :mod:`phenotypic.gui.builder._state`: a mutable
dataclass plus a pair of ``state_to_json`` / ``state_from_json`` helpers.

Note: this is UI scratch state, not the process-wide ``RunRegistry`` (which
lives in :mod:`phenotypic.gui.shell._runs_registry` and survives Run console
rebuild).

The free function :func:`to_argv` translates a state into the argv tail used
by both the local subprocess runner and the SLURM submitter shell-out, so the
two execution modes share a single source of truth for CLI flags.

Examples:
    Round-trip a state through JSON:

    >>> from phenotypic.gui.run_console._state import (
    ...     RunConsoleState, run_state_to_json, run_state_from_json,
    ... )
    >>> state = RunConsoleState(
    ...     pipeline_path="/p/pipeline.json",
    ...     input_dir="/p/in",
    ...     output_dir="/p/out",
    ...     mode="local",
    ...     dry_run=True,
    ... )
    >>> payload = run_state_to_json(state)
    >>> run_state_from_json(payload).dry_run
    True

Module split (Phase 1a, Task 8)
-------------------------------
``RunConsoleState``, ``to_argv``, the JSON round-trip, and their pure coercion
helpers now live in :mod:`phenotypic._services.argv` so the MCP server can build
the same command line without importing a GUI package. They are re-exported here
unchanged, so every existing import of this module keeps working.

What stays: :func:`state_from_controls` and :func:`_resolve_control_path`. Both
take a :class:`SandboxRoot` and resolve the metadata-CSV payload through
``gui.shell._metadata_context``, which makes them genuinely GUI-facing — moving
them would drag ``phenotypic.gui`` into the Dash-free tier.
"""
from __future__ import annotations

from phenotypic._services.argv import (  # noqa: F401
    RunConsoleState,
    _coerce_optional_int,
    _coerce_optional_str,
    _normalize_image_type,
    _parse_key_value_lines,
    run_state_from_json,
    run_state_to_json,
    to_argv,
)
from phenotypic._services.sandbox import SandboxRoot
from phenotypic.gui.shell._metadata_context import resolve_metadata_csv
from phenotypic.sdk_.slurm import parse_slurm_time
from phenotypic.sdk_.typing_ import ExecutionMode

__all__ = [
    "RunConsoleState",
    "run_state_to_json",
    "run_state_from_json",
    "state_from_controls",
    "to_argv",
]


def _resolve_control_path(
    value: object,
    *,
    field_name: str,
    sandbox: SandboxRoot,
) -> str | None:
    """Resolve an optional raw path control through ``sandbox``."""
    text = _coerce_optional_str(value)
    if text is None:
        return None
    try:
        return str(sandbox.resolve(text))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{field_name} is outside the GUI sandbox") from exc

def state_from_controls(  # noqa: PLR0913
    *,
    pipeline_path: object,
    input_dir: object,
    output_dir: object,
    mode: object,
    flags: object,
    sample: object,
    nrows: object,
    ncols: object,
    image_type: object,
    workers: object,
    log_level: object,
    slurm_partition: object,
    slurm_time: object,
    slurm_mem: object,
    slurm_cpus: object,
    slurm_gpus: object,
    slurm_extra: object,
    metadata_payload: object,
    sandbox: SandboxRoot,
    gpu_slurm: object = None,
    gpu_shards: object = 1,
) -> RunConsoleState:
    """Build authoritative run state directly from raw Dash controls.

    Args:
        pipeline_path: Pipeline picker value.
        input_dir: Input-directory picker value.
        output_dir: Output-directory picker value.
        mode: Visible execution mode, ``"local"`` or ``"slurm"``.
        flags: Visible checklist values (``dry_run`` and ``resume``).
        sample: Optional positive sample size.
        nrows: Optional positive grid row count.
        ncols: Optional positive grid column count.
        image_type: Optional CLI image class.
        workers: Optional positive local worker count.
        log_level: Optional presentation-only log level.
        slurm_partition: Optional CPU-profile partition.
        slurm_time: Optional positive minutes or SLURM duration.
        slurm_mem: Optional CPU-profile memory request.
        slurm_cpus: Optional positive CPU count.
        slurm_gpus: Optional nonnegative common GPU count.
        slurm_extra: Optional CPU-profile ``key=value`` lines.
        metadata_payload: Shared metadata-context payload.
        sandbox: Frozen GUI sandbox used to resolve every path.
        gpu_slurm: Optional GPU-stage delta ``key=value`` lines.
        gpu_shards: Positive number of whole-GPU Stage-2 shards.

    Returns:
        A validated :class:`RunConsoleState` reflecting the controls supplied
        in this call, without consulting a derived browser store.

    Raises:
        ValueError: If a path escapes the sandbox, a typed control is invalid,
            or SLURM mode has no common CPU profile.
    """
    if mode not in {"local", "slurm"}:
        raise ValueError("mode must be 'local' or 'slurm'")
    execution_mode: ExecutionMode = "slurm" if mode == "slurm" else "local"

    flag_values = (
        {
            str(item)
            for item in flags
            if isinstance(item, str)
        }
        if isinstance(flags, (list, tuple, set, frozenset))
        else set()
    )

    advanced_candidates: dict[str, object] = {
        "sample": _coerce_optional_int(
            sample, field_name="sample", minimum=1
        ),
        "nrows": _coerce_optional_int(
            nrows, field_name="nrows", minimum=1
        ),
        "ncols": _coerce_optional_int(
            ncols, field_name="ncols", minimum=1
        ),
        "image_type": _normalize_image_type(image_type),
        "workers": _coerce_optional_int(
            workers, field_name="workers", minimum=1
        ),
        "log_level": _coerce_optional_str(log_level),
    }
    advanced_args = {
        key: value
        for key, value in advanced_candidates.items()
        if value is not None
    }

    cpu_extra_tokens = _parse_key_value_lines(
        slurm_extra, field_name="Extra SLURM"
    )
    cpu_extra = dict(token.split("=", 1) for token in cpu_extra_tokens)
    canonical_time = parse_slurm_time(slurm_time)
    typed_slurm: dict[str, object] = {
        "partition": _coerce_optional_str(slurm_partition),
        "time": canonical_time,
        "mem": _coerce_optional_str(slurm_mem),
        "cpus_per_task": _coerce_optional_int(
            slurm_cpus, field_name="SLURM CPUs per task", minimum=1
        ),
        "gpus": _coerce_optional_int(
            slurm_gpus, field_name="SLURM GPUs", minimum=0
        ),
    }
    slurm_args = {
        key: value for key, value in typed_slurm.items() if value is not None
    }
    if cpu_extra:
        slurm_args["extra"] = cpu_extra
    if execution_mode == "slurm" and not slurm_args:
        raise ValueError("SLURM mode requires a nonempty CPU SLURM profile")

    metadata_csv = resolve_metadata_csv(sandbox, metadata_payload)
    return RunConsoleState(
        pipeline_path=_resolve_control_path(
            pipeline_path, field_name="pipeline_path", sandbox=sandbox
        ),
        input_dir=_resolve_control_path(
            input_dir, field_name="input_dir", sandbox=sandbox
        ),
        output_dir=_resolve_control_path(
            output_dir, field_name="output_dir", sandbox=sandbox
        ),
        metadata_csv=str(metadata_csv) if metadata_csv is not None else None,
        mode=execution_mode,
        dry_run="dry_run" in flag_values,
        resume="resume" in flag_values,
        advanced_args=advanced_args,
        slurm_args=slurm_args,
        gpu_slurm_args=_parse_key_value_lines(
            gpu_slurm, field_name="GPU-stage SLURM"
        ),
        gpu_shards=(
            _coerce_optional_int(
                gpu_shards, field_name="GPU shards", minimum=1
            )
            or 1
        ),
    )
