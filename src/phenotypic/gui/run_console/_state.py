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

from phenotypic.tools_.typing_ import ExecutionMode


__all__ = [
    "RunConsoleState",
    "run_state_to_json",
    "run_state_from_json",
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
        mode: Either ``"local"`` (spawn a subprocess in the GUI process)
            or ``"slurm"`` (shell-out to ``python -m phenotypic ... --slurm
            ...`` and let SLURM take over).
        dry_run: When ``True``, append ``--dry-run`` to the CLI argv so the
            CLI just validates inputs without processing images.
        resume: When ``True``, append ``--resume`` so the CLI picks up where
            a previous run left off.
        save_inspect: When ``True``, append ``--save-inspect`` to the CLI
            argv so :meth:`MeasureFeatures.inspect` figures are saved as
            PNGs per processed image.
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
    """

    pipeline_path: str | None = None
    input_dir: str | None = None
    output_dir: str | None = None
    mode: ExecutionMode = "local"
    dry_run: bool = False
    resume: bool = False
    save_inspect: bool = False
    advanced_args: dict[str, Any] = field(default_factory=dict)
    slurm_args: dict[str, Any] = field(default_factory=dict)


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
        "mode": state.mode,
        "dry_run": bool(state.dry_run),
        "resume": bool(state.resume),
        "save_inspect": bool(state.save_inspect),
        "advanced_args": dict(state.advanced_args or {}),
        "slurm_args": _slurm_args_to_json(state.slurm_args or {}),
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

    return RunConsoleState(
        pipeline_path=_coerce_optional_str(payload.get("pipeline_path")),
        input_dir=_coerce_optional_str(payload.get("input_dir")),
        output_dir=_coerce_optional_str(payload.get("output_dir")),
        mode=mode,
        dry_run=bool(payload.get("dry_run", False)),
        resume=bool(payload.get("resume", False)),
        save_inspect=bool(payload.get("save_inspect", False)),
        advanced_args=advanced_args,
        slurm_args=slurm_args,
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
        ``["--pipeline", "pipeline.json", "--input", "/in", "-o", "/out",
        "--dry-run"]``.

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
        "--pipeline",
        pipeline_path,
        "--input",
        input_dir,
        "-o",
        output_dir,
    ]

    if state.dry_run:
        argv.append("--dry-run")
    if state.resume:
        argv.append("--resume")
    if state.save_inspect:
        argv.append("--save-inspect")

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
        argv.extend(["--n-jobs", str(workers)])

    # ``log_level`` is intentionally not forwarded: the CLI does not expose
    # a ``--log-level`` flag. We keep the field in state so future CLI work
    # can pick it up without a state schema migration.
    _ = _ADVANCED_KEYS  # silence unused-warning while documenting the set
    _ = _SLURM_KEYS

    return argv
