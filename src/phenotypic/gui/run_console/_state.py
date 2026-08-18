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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phenotypic.gui.shell._metadata_context import resolve_metadata_csv
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.sdk_.slurm import parse_slurm_time
from phenotypic.sdk_.typing_ import ExecutionMode


__all__ = [
    "RunConsoleState",
    "run_state_to_json",
    "run_state_from_json",
    "state_from_controls",
    "to_argv",
]


# Recognised keys for ``RunConsoleState.advanced_args``. Anything else the
# user stuffs in is silently ignored at argv translation time so the GUI
# can over-broadcast without breaking the CLI shell-out.
_ADVANCED_KEYS: frozenset[str] = frozenset(
    {"sample", "nrows", "ncols", "image_type", "workers", "log_level"}
)

# Recognised keys for ``RunConsoleState.slurm_args``. ``extra`` is the
# free-form ``--slurm k=v`` pass-through bucket.
_SLURM_KEYS: frozenset[str] = frozenset(
    {"partition", "time", "mem", "cpus_per_task", "gpus", "extra"}
)


@dataclass
class RunConsoleState:
    """Form state for the Run console.

    Attributes:
        pipeline_path: Absolute path (string) to the pipeline JSON the run
            should execute. ``None`` means the form has not been completed.
        input_dir: Absolute path (string) to the directory of input images.
            ``None`` means unset.
        output_dir: Absolute path (string) to the desired output directory.
            ``None`` means unset.
        metadata_csv: Absolute path (string) to the optional metadata CSV
            selected in the global GUI settings. ``None`` means unset.
        mode: Either ``"local"`` (spawn a subprocess in the GUI process)
            or ``"slurm"`` (shell-out to ``python -m phenotypic ... --slurm
            ...`` and let SLURM take over).
        dry_run: When ``True``, append ``--dry-run`` to the CLI argv so the
            CLI just validates inputs without processing images.
        retry_failures: When ``True``, append ``--retry-failures`` for this
            invocation without clearing terminal history.
        advanced_args: Optional advanced flag bucket. Recognised keys:
            ``sample`` (int), ``nrows`` (int), ``ncols`` (int),
            ``image_type`` (str), ``workers`` (int), ``log_level`` (str).
            Any of these may be ``None``; unknown keys are ignored at argv
            time.
        slurm_args: Optional SLURM flag bucket. Recognised keys:
            ``partition`` (str), ``time`` (str), ``mem`` (str),
            ``cpus_per_task`` (int), ``gpus`` (int), and ``extra`` — a
            ``dict[str, str]`` of free-form key/value pairs that are
            forwarded as additional ``--slurm k=v`` repeats.
        gpu_slurm_args: Ordered GPU-stage SLURM delta ``key=value`` tokens.
        gpu_shards: Number of whole-GPU Stage-2 shard tasks.
    """

    pipeline_path: str | None = None
    input_dir: str | None = None
    output_dir: str | None = None
    metadata_csv: str | None = None
    mode: ExecutionMode = "local"
    dry_run: bool = False
    retry_failures: bool = False
    advanced_args: dict[str, Any] = field(default_factory=dict)
    slurm_args: dict[str, Any] = field(default_factory=dict)
    gpu_slurm_args: tuple[str, ...] = ()
    gpu_shards: int = 1


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def run_state_to_json(state: RunConsoleState) -> dict[str, Any]:
    """Convert ``state`` to a JSON-friendly dict.

    The resulting dict contains only stdlib types (``dict``, ``list``,
    ``str``, ``int``, ``float``, ``bool``, ``None``) provided the values
    inside ``advanced_args`` and ``slurm_args`` are themselves JSON-friendly
    (the contract the GUI layer is expected to uphold).

    Args:
        state: The state to serialize.

    Returns:
        Dict with the same field names as :class:`RunConsoleState`.
    """

    return {
        "pipeline_path": state.pipeline_path,
        "input_dir": state.input_dir,
        "output_dir": state.output_dir,
        "metadata_csv": state.metadata_csv,
        "mode": state.mode,
        "dry_run": bool(state.dry_run),
        "retry_failures": bool(state.retry_failures),
        "advanced_args": dict(state.advanced_args or {}),
        "slurm_args": _slurm_args_to_json(state.slurm_args or {}),
        "gpu_slurm_args": list(state.gpu_slurm_args),
        "gpu_shards": state.gpu_shards,
    }


def run_state_from_json(payload: dict[str, Any]) -> RunConsoleState:
    """Inverse of :func:`run_state_to_json`. Tolerant of missing keys.

    Args:
        payload: Dict previously produced by :func:`run_state_to_json` or
            an equivalent JSON document. Older preset files are allowed to
            be missing fields — defaults match :class:`RunConsoleState`.

    Returns:
        A reconstructed :class:`RunConsoleState`.
    """

    if not isinstance(payload, dict):
        return RunConsoleState()

    raw_mode = payload.get("mode", "local")
    mode: ExecutionMode
    mode = "slurm" if raw_mode == "slurm" else "local"

    advanced_raw = payload.get("advanced_args") or {}
    advanced_args: dict[str, Any] = (
        dict(advanced_raw) if isinstance(advanced_raw, dict) else {}
    )

    slurm_raw = payload.get("slurm_args") or {}
    slurm_args: dict[str, Any] = (
        _slurm_args_to_json(slurm_raw) if isinstance(slurm_raw, dict) else {}
    )
    gpu_slurm_args = _gpu_slurm_args_from_json(
        payload.get("gpu_slurm_args", ())
    )
    gpu_shards = _positive_int_or_default(payload.get("gpu_shards"), default=1)

    return RunConsoleState(
        pipeline_path=_coerce_optional_str(payload.get("pipeline_path")),
        input_dir=_coerce_optional_str(payload.get("input_dir")),
        output_dir=_coerce_optional_str(payload.get("output_dir")),
        metadata_csv=_coerce_optional_str(payload.get("metadata_csv")),
        mode=mode,
        dry_run=bool(payload.get("dry_run", False)),
        retry_failures=bool(payload.get("retry_failures", False)),
        advanced_args=advanced_args,
        slurm_args=slurm_args,
        gpu_slurm_args=gpu_slurm_args,
        gpu_shards=gpu_shards,
    )


def _coerce_optional_str(value: Any) -> str | None:
    """Return ``value`` as ``str`` if non-empty, otherwise ``None``.

    Empty strings are coerced to ``None`` so a user clearing a form field
    round-trips as "unset" rather than as a literal empty path.

    Args:
        value: Candidate value pulled from a JSON payload.

    Returns:
        The trimmed string, or ``None`` if ``value`` is missing/empty.
    """

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slurm_args_to_json(slurm_args: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly copy of ``slurm_args`` with a normalised ``extra`` dict.

    Args:
        slurm_args: Source dict, possibly containing arbitrary keys.

    Returns:
        Shallow copy where ``extra`` (if present) is forced to a
        ``dict[str, str]``.
    """

    out: dict[str, Any] = {}
    for key, value in slurm_args.items():
        if key == "extra":
            extra: dict[str, str] = {}
            if isinstance(value, dict):
                for k, v in value.items():
                    if k is None or v is None:
                        continue
                    extra[str(k)] = str(v)
            out["extra"] = extra
        else:
            out[key] = value
    return out


def _gpu_slurm_args_from_json(value: object) -> tuple[str, ...]:
    """Return a tolerant tuple of GPU-stage ``key=value`` tokens."""
    if isinstance(value, dict):
        return tuple(
            f"{key}={item}"
            for key, item in value.items()
            if key is not None and item is not None
        )
    if isinstance(value, str):
        candidates: object = value.splitlines()
    else:
        candidates = value
    if not isinstance(candidates, (list, tuple)):
        return ()
    return tuple(
        text
        for item in candidates
        if (text := str(item).strip())
    )


def _positive_int_or_default(value: object, *, default: int) -> int:
    """Return a positive integer or ``default`` for tolerant preset loading."""
    try:
        parsed = _coerce_optional_int(value, field_name="value", minimum=1)
    except ValueError:
        return default
    return default if parsed is None else parsed


def _coerce_optional_int(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int | None:
    """Coerce one optional Dash numeric control to an integer."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be an integer >= {minimum}"
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    if parsed < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return parsed


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


def _normalize_image_type(value: object) -> str | None:
    """Normalize the optional CLI image class selected by the form."""
    text = _coerce_optional_str(value)
    if text is None:
        return None
    normalized = text.casefold()
    if normalized == "image":
        return "Image"
    if normalized == "gridimage":
        return "GridImage"
    raise ValueError("image_type must be 'Image' or 'GridImage'")


def _parse_key_value_lines(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Parse and validate newline-delimited ``key=value`` controls."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.splitlines()
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raise ValueError(f"{field_name} must contain key=value entries")

    tokens: list[str] = []
    for raw_item in raw_items:
        text = str(raw_item).strip()
        if not text:
            continue
        key, separator, raw_value = text.partition("=")
        key = key.strip()
        item_value = raw_value.strip()
        if not separator or not key or not item_value:
            raise ValueError(
                f"{field_name} entries must use nonempty key=value syntax"
            )
        if key in {"time", "slurm_time"}:
            item_value = parse_slurm_time(item_value) or ""
        tokens.append(f"{key}={item_value}")
    return tuple(tokens)


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
        flags: Visible checklist values (``dry_run`` and ``retry_failures``).
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
        retry_failures="retry_failures" in flag_values,
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


# ---------------------------------------------------------------------------
# CLI argv translation
# ---------------------------------------------------------------------------


def to_argv(state: RunConsoleState) -> list[str]:
    """Translate ``state`` into the argv tail of ``python -m phenotypic ...``.

    The returned list does **not** include the leading executable / module
    spec — callers prepend ``[sys.executable, "-m", "phenotypic"]`` (or the
    equivalent shell incantation). ``--slurm k=v`` pairs are not added here;
    SLURM-specific argv extension is the SLURM runner's responsibility.

    Args:
        state: The form state to translate. Must have non-``None``
            ``pipeline_path``, ``input_dir``, and ``output_dir``.

    Returns:
        List of argv tokens, e.g.
        ``["--mode", "full", "--pipeline", "pipeline.json", "--input", "/in",
        "--output", "/out", "--dry-run"]``.

    Raises:
        ValueError: If any of ``pipeline_path``, ``input_dir``, or
            ``output_dir`` is missing. The caller is responsible for
            filesystem-existence checks; this function only enforces that
            the slots are populated.
    """

    missing: list[str] = []
    if not state.pipeline_path:
        missing.append("pipeline_path")
    if not state.input_dir:
        missing.append("input_dir")
    if not state.output_dir:
        missing.append("output_dir")
    if missing:
        raise ValueError(
            "RunConsoleState is missing required field(s): "
            + ", ".join(missing)
        )

    # Narrowing: the three slots are non-None past the guard above.
    pipeline_path = str(state.pipeline_path)
    input_dir = str(state.input_dir)
    output_dir = str(state.output_dir)

    argv: list[str] = [
        "--mode",
        "full",
        "--pipeline",
        pipeline_path,
        "--input",
        input_dir,
        "--output",
        output_dir,
    ]

    if state.metadata_csv:
        argv.extend(["--metadata", str(state.metadata_csv)])

    if state.dry_run:
        argv.append("--dry-run")
    if state.retry_failures:
        argv.append("--retry-failures")

    advanced = state.advanced_args or {}
    sample = advanced.get("sample")
    if sample is not None:
        argv.extend(["--sample", str(sample)])

    nrows = advanced.get("nrows")
    if nrows is not None:
        argv.extend(["--nrows", str(nrows)])

    ncols = advanced.get("ncols")
    if ncols is not None:
        argv.extend(["--ncols", str(ncols)])

    image_type = advanced.get("image_type")
    if image_type:
        argv.extend(["--image-type", str(image_type)])

    workers = advanced.get("workers")
    if workers is not None:
        argv.extend(["--njobs", str(workers)])

    # ``log_level`` is intentionally not forwarded: the CLI does not expose
    # a ``--log-level`` flag. We keep the field in state so future CLI work
    # can pick it up without a state schema migration.
    _ = _ADVANCED_KEYS  # silence unused-warning while documenting the set
    _ = _SLURM_KEYS

    return argv
