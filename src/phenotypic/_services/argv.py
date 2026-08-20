"""Run-console form state and CLI argv construction.

``RunConsoleState`` is the small dataclass backing the Run console form, and
``to_argv`` translates one into the argv tail of ``python -m phenotypic ...``.
They live together because ``to_argv``'s signature *is* the dataclass: splitting
them would make this module import back up into :mod:`phenotypic.gui`, inverting
the layering the architecture asserts.

``to_argv`` renders the mode-independent tail; ``slurm_argv_extension`` renders
the profile's ``--slurm`` pairs; ``to_subprocess_argv`` is the **single**
composition point that joins them with the launcher and the GPU-staging flags.
The three stay separate because a local run must not carry SLURM argv, and they
are all here because spec ``05-deploy-and-slurm.md`` §5.4 digests the composed
list — two renderings of it would be two digests.

Also hosts the ``phenotypic.tune run`` semantic-tail and argv builders, promoted
from ``gui/tune/_run_argv.py`` so the tune surface and the MCP server share one
spelling of the command line.

Nothing here imports Dash, Flask, or :mod:`phenotypic.gui` — enforced by
``tests/unit/services/test_import_purity.py`` and
``tests/unit/services/test_argv_promotion.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from phenotypic.sdk_.slurm import parse_slurm_time
from phenotypic.sdk_.typing_ import ExecutionMode

__all__ = [
    "RunConsoleState",
    "run_state_to_json",
    "run_state_from_json",
    "slurm_argv_extension",
    "to_argv",
    "to_subprocess_argv",
    "tune_run_argv",
    "tune_run_argv_from_tail",
    "tune_run_tail",
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

# ``slurm_args`` keys emitted as ``--slurm key=value`` in this fixed order.
# Order is load-bearing: spec 05 §5.4 digests the rendered argv, so a
# reordering here moves ``argv_digest`` for every already-minted plan token.
# Promoted from ``gui/run_console/_slurm.py`` with the emitter that reads it.
_SLURM_DIRECT_KEYS: tuple[str, ...] = (
    "partition",
    "time",
    "mem",
    "cpus_per_task",
    "gpus",
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
        restart: When ``True``, append ``--restart`` so the CLI clears the
            previous machine-state and starts the run over. Mutually
            exclusive with the CLI's ``--overwrite``, which this state
            deliberately cannot express: ``--overwrite`` deletes the output
            tree, and a caller that only holds a serialized state should not
            be able to ask for that.
        image_manifest: Absolute path (string) to a ``.images`` manifest
            listing the exact subset of ``input_dir`` to process. ``None``
            means "every image under ``input_dir``". The manifest is passed
            *alongside* ``--input``, never instead of it: work IDs are
            derived relative to ``--input``, so pointing it at the manifest
            file would collapse every work ID to a bare basename.
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
    restart: bool = False
    image_manifest: str | None = None
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
        "restart": bool(state.restart),
        "image_manifest": state.image_manifest,
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
        restart=bool(payload.get("restart", False)),
        image_manifest=_coerce_optional_str(payload.get("image_manifest")),
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

# ---------------------------------------------------------------------------
# CLI argv translation
# ---------------------------------------------------------------------------


def to_argv(state: RunConsoleState) -> list[str]:
    """Translate ``state`` into the argv tail of ``python -m phenotypic ...``.

    The returned list does **not** include the leading executable / module
    spec — callers prepend ``[sys.executable, "-m", "phenotypic"]`` (or the
    equivalent shell incantation). ``--slurm k=v`` pairs are not added here;
    SLURM-specific argv extension is :func:`slurm_argv_extension`'s job, and
    :func:`to_subprocess_argv` is the one composition point that joins the
    two. Keeping them apart is deliberate: a local run must not carry SLURM
    argv, and duplicating either half would silently double-emit.

    ``--input`` is always emitted, including when ``image_manifest`` is set.
    The manifest names a subset *of* ``--input``; it never replaces it,
    because ``work_id_for_image`` derives each image's identity from its path
    relative to ``--input``.

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

    if state.image_manifest:
        argv.extend(["--image-manifest", str(state.image_manifest)])

    if state.metadata_csv:
        argv.extend(["--metadata", str(state.metadata_csv)])

    if state.dry_run:
        argv.append("--dry-run")
    if state.retry_failures:
        argv.append("--retry-failures")
    if state.restart:
        argv.append("--restart")

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


def slurm_argv_extension(slurm_args: Mapping[str, Any]) -> list[str]:
    """Build the repeated ``--slurm key=value`` arguments for one profile.

    Promoted from ``gui/run_console/_slurm.py`` so the GUI submitter and the
    MCP server render the same pairs. The GUI also uses an empty
    return as its "no SLURM profile configured" predicate, so an empty
    ``slurm_args`` must keep yielding ``[]`` rather than raising.

    Args:
        slurm_args: The ``RunConsoleState.slurm_args`` bucket. Recognised
            direct keys are emitted first in :data:`_SLURM_DIRECT_KEYS` order;
            an ``extra`` sub-mapping is appended as free-form pairs.

    Returns:
        Flat token list, e.g. ``["--slurm", "partition=short", "--slurm",
        "mem=4G"]``. Empty when nothing is configured.

    Examples:
        >>> slurm_argv_extension({"partition": "short", "mem": "4G"})
        ['--slurm', 'partition=short', '--slurm', 'mem=4G']
        >>> slurm_argv_extension({})
        []
    """
    if not slurm_args:
        return []
    pairs: list[tuple[str, str]] = []
    for key in _SLURM_DIRECT_KEYS:
        value = slurm_args.get(key)
        if value is not None and value != "":
            pairs.append((key, str(value)))
    extra = slurm_args.get("extra") or {}
    if isinstance(extra, Mapping):
        for extra_key, extra_value in extra.items():
            if (
                extra_key is not None
                and extra_value is not None
                and str(extra_key)
                and str(extra_value)
            ):
                pairs.append((str(extra_key), str(extra_value)))
    argv: list[str] = []
    for key, value in pairs:
        argv.extend(["--slurm", f"{key}={value}"])
    return argv


def to_subprocess_argv(
    state: RunConsoleState,
    *,
    python: str | None = None,
) -> list[str]:
    """Assemble the complete ``python -m phenotypic`` argument vector.

    **This is the one composition point.** ``to_argv`` renders the
    mode-independent tail and deliberately excludes SLURM argv; this function
    is where that tail meets the profile's ``--slurm`` pairs and the
    GPU-staging flags. Spec ``05-deploy-and-slurm.md`` §5.4 digests exactly
    this list, so any caller that assembles the halves itself would produce a
    digest nobody else can reproduce — and a caller that added the SLURM
    pairs *and* called this would emit every one of them twice.

    Args:
        state: The form state to translate.
        python: Python executable, defaulting to :data:`sys.executable`.

    Returns:
        Full argv including the launcher, ready for ``subprocess`` or for
        hashing into ``argv_digest``.

    Raises:
        ValueError: Propagated from :func:`to_argv` when a required path slot
            is unset.
    """
    argv = [
        python or sys.executable,
        "-m",
        "phenotypic",
        *to_argv(state),
        *slurm_argv_extension(state.slurm_args or {}),
    ]
    for token in state.gpu_slurm_args:
        argv.extend(["--gpu-slurm", token])
    if state.gpu_shards != 1:
        argv.extend(["--gpu-shards", str(state.gpu_shards)])
    return argv


def tune_run_tail(
    *,
    spec_path: str,
    images_dir: str,
    output_dir: str,
    strategy: str | None,
    n_trials: int | None,
    storage_url: str | None,
    n_workers: int | None,
    slurm_partition: str | None,
    slurm_mem: str | None,
    slurm_time: str | None,
    held_out_fraction: float | None,
    cv_group: str | None,
    slurm: bool,
    screen: bool,
    slurm_args: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build the launcher-independent ``run`` argument tail.

    Args:
        spec_path: Tuning spec path passed as the ``run`` positional argument.
        images_dir: Image directory passed to ``-i``.
        output_dir: Output directory passed to ``-o``.
        strategy: Optional CLI ``--strategy`` override. ``None`` preserves the
            strategy, seed, and storage configured in the tuning spec.
        n_trials: Trial budget, omitted for exhaustive grid search.
        storage_url: Optional Optuna storage URL.
        n_workers: Optional SLURM worker count.
        slurm_partition: Optional SLURM partition.
        slurm_mem: Optional SLURM memory request.
        slurm_time: Optional SLURM wall time.
        held_out_fraction: Optional robust-eval fraction override.
        cv_group: Optional robust-eval grouping-column override.
        slurm: Whether to append ``--slurm``.
        screen: Whether to append ``--screen``.
        slurm_args: Free-form ``#SBATCH`` keys the three sugar arguments above
            cannot express — ``slurm_account`` (mandatory for UCR HPCC's
            ``exfab`` and ``preempt`` partitions), ``slurm_qos``,
            ``slurm_cpus_per_task``, ``slurm_gpus_per_node``. Rendered as
            repeated ``--slurm key=value`` in iteration order, **only when**
            ``slurm`` is true: on this CLI the presence of ``--slurm`` is what
            requests submission, so emitting a pair for a local run would turn
            it into a cluster job. Empty keys and values are skipped.

    Returns:
        Tokens beginning with the ``run`` subcommand.

    Raises:
        ValueError: If a required path is empty.
    """
    missing = [
        name
        for name, value in (
            ("spec_path", spec_path),
            ("images_dir", images_dir),
            ("output_dir", output_dir),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "tune_run_tail missing required field(s): " + ", ".join(missing)
        )

    tail = [
        "run",
        spec_path,
        "-i",
        images_dir,
        "-o",
        output_dir,
    ]
    if strategy:
        tail += ["--strategy", strategy]
    if n_trials is not None and strategy != "grid":
        tail += ["--n-trials", str(n_trials)]
    if storage_url:
        tail += ["--storage-url", storage_url]
    if n_workers is not None:
        tail += ["--n-workers", str(n_workers)]
    if slurm_partition:
        tail += ["--slurm-partition", slurm_partition]
    if slurm_mem:
        tail += ["--slurm-mem", slurm_mem]
    if slurm_time:
        tail += ["--slurm-time", slurm_time]
    if held_out_fraction is not None:
        tail += ["--held-out-fraction", str(held_out_fraction)]
    if cv_group:
        tail += ["--cv-group", cv_group]
    if slurm:
        pairs = _tune_slurm_pairs(slurm_args)
        if pairs:
            tail += pairs
        else:
            tail.append("--slurm")
    if screen:
        tail.append("--screen")
    return tail


def _tune_slurm_pairs(slurm_args: Mapping[str, Any] | None) -> list[str]:
    """Render ``slurm_args`` as repeated ``--slurm key=value`` tokens.

    Each rendered pair carries the submission request on its own, so a
    non-empty result replaces the bare ``--slurm`` rather than joining it.
    """
    if not slurm_args:
        return []
    argv: list[str] = []
    for key, value in slurm_args.items():
        if key is None or value is None:
            continue
        key_text, value_text = str(key), str(value)
        if not key_text or not value_text:
            continue
        argv.extend(["--slurm", f"{key_text}={value_text}"])
    return argv


def tune_run_argv_from_tail(
    tail: Sequence[str],
    *,
    python: str | None = None,
) -> list[str]:
    """Prefix one validated semantic tail with the executable module launcher."""
    return [python or sys.executable, "-m", "phenotypic.tune", *tail]


def tune_run_argv(
    *,
    spec_path: str,
    images_dir: str,
    output_dir: str,
    strategy: str | None,
    n_trials: int | None,
    storage_url: str | None,
    n_workers: int | None,
    slurm_partition: str | None,
    slurm_mem: str | None,
    slurm_time: str | None,
    held_out_fraction: float | None,
    cv_group: str | None,
    slurm: bool,
    screen: bool,
    slurm_args: Mapping[str, Any] | None = None,
    python: str | None = None,
) -> list[str]:
    """Build the full launch argv for a tune run.

    Args:
        spec_path: Tuning spec path passed as the ``run`` positional argument.
        images_dir: Image directory passed to ``-i``.
        output_dir: Output directory passed to ``-o``.
        strategy: CLI ``--strategy`` override.
        n_trials: Trial budget, omitted for exhaustive grid search.
        storage_url: Optional Optuna storage URL.
        n_workers: Optional SLURM worker count.
        slurm_partition: Optional SLURM partition.
        slurm_mem: Optional SLURM memory request.
        slurm_time: Optional SLURM wall time.
        held_out_fraction: Optional robust-eval fraction override.
        cv_group: Optional robust-eval grouping-column override.
        slurm: Whether to append ``--slurm``.
        screen: Whether to append ``--screen``.
        slurm_args: Free-form ``--slurm key=value`` passthrough; see
            :func:`tune_run_tail`.
        python: Python executable, defaulting to :data:`sys.executable`.

    Returns:
        Full argv, including Python executable and module entry point.

    Raises:
        ValueError: If ``spec_path``, ``images_dir``, or ``output_dir`` is empty.
    """
    tail = tune_run_tail(
        spec_path=spec_path,
        images_dir=images_dir,
        output_dir=output_dir,
        strategy=strategy,
        n_trials=n_trials,
        storage_url=storage_url,
        n_workers=n_workers,
        slurm_partition=slurm_partition,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        held_out_fraction=held_out_fraction,
        cv_group=cv_group,
        slurm=slurm,
        screen=screen,
        slurm_args=slurm_args,
    )
    return tune_run_argv_from_tail(tail, python=python)
