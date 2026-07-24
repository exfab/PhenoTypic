"""Validated Tune launch commands shared by preview, copy, and deployment."""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.tune._run_argv import (
    tune_run_argv,
    tune_run_argv_from_tail,
    tune_run_tail,
)
from phenotypic.tune.strategy._config import STRATEGY_CHOICES

ExecutionTarget = Literal["local", "slurm"]
StorageMode = Literal["local", "environment"]

DEFAULT_STORAGE_ENV = "PHENOTYPIC_STORAGE_URL"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PORTABLE_PREFIX = ("uv", "run", "python", "-m", "phenotypic.tune")


@dataclass(frozen=True)
class ValidatedTuneCommand:
    """One server-side command plan for preview, copy, and execution.

    Secret-bearing fields are excluded from ``repr``. Browser callbacks may
    return only ``display_tokens`` or ``portable_tokens``.
    """

    argv: tuple[str, ...] = field(repr=False)
    semantic_tail: tuple[str, ...] = field(repr=False)
    display_tokens: tuple[str, ...]
    portable_tokens: tuple[str, ...]
    spec_path: Path | None
    images_dir: Path | None
    output_dir: Path | None
    execution_target: ExecutionTarget
    issues: tuple[str, ...]
    copy_eligible: bool

    @property
    def deploy_eligible(self) -> bool:
        """Whether deployment may execute :attr:`argv`."""
        return not self.issues

    def display_command(self) -> str:
        """Return the redacted GUI-equivalent shell command."""
        return render_tokens(self.display_tokens, preserve_env_refs=True)

    def portable_command(self) -> str:
        """Return the redacted portable project command."""
        return render_tokens(self.portable_tokens, preserve_env_refs=True)


def _resolve_existing(
    sandbox: SandboxRoot,
    value: str | None,
    *,
    label: str,
    directory: bool,
) -> tuple[Path | None, list[str]]:
    """Resolve one required existing path and collect a reader-facing issue."""
    if not value or not value.strip():
        return None, [f"Set {label}."]
    try:
        path = sandbox.resolve(value.strip())
    except ValueError:
        return None, [f"{label.capitalize()} escapes the GUI sandbox."]
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        return path, [f"{label.capitalize()} is not an existing {kind}: {path}"]
    return path, []


def _resolve_output(
    sandbox: SandboxRoot,
    value: str | None,
) -> tuple[Path | None, list[str]]:
    """Resolve the output path, allowing a not-yet-created directory."""
    if not value or not value.strip():
        return None, ["Set output directory."]
    try:
        path = sandbox.resolve(value.strip())
    except ValueError:
        return None, ["Output directory escapes the GUI sandbox."]
    if path.exists() and not path.is_dir():
        return path, [f"Output path is not a directory: {path}"]
    return path, []


def _storage_tokens(
    *,
    sandbox: SandboxRoot,
    mode: StorageMode,
    local_path: str | None,
    environment_name: str | None,
    environ: Mapping[str, str],
) -> tuple[str | None, str | None, list[str]]:
    """Return actual/redacted storage values without exposing credentials."""
    if mode == "local":
        if not local_path or not local_path.strip():
            return None, None, []
        try:
            path = sandbox.resolve(local_path.strip())
        except ValueError:
            return None, None, ["Local storage path escapes the GUI sandbox."]
        if path.exists() and path.is_dir():
            return None, None, [f"Local storage path is a directory: {path}"]
        url = f"sqlite:///{path}"
        return url, url, []

    name = (environment_name or DEFAULT_STORAGE_ENV).strip()
    if not _ENV_NAME.fullmatch(name):
        return None, None, ["Storage environment variable name is invalid."]
    value = environ.get(name)
    if not value:
        return None, f"${name}", [
            f"Server environment variable {name} is not configured."
        ]
    return value, f"${name}", []


def build_tune_command(
    *,
    sandbox: SandboxRoot,
    spec_path: str | None,
    images_dir: str | None,
    output_dir: str | None,
    strategy: str | None,
    n_trials: int | None,
    storage_mode: StorageMode = "local",
    storage_local_path: str | None = None,
    storage_environment_name: str | None = DEFAULT_STORAGE_ENV,
    n_workers: int | None = None,
    slurm_partition: str | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    held_out_fraction: float | None = None,
    cv_group: str | None = None,
    slurm: bool = False,
    screen: bool = False,
    environ: Mapping[str, str] | None = None,
    additional_issues: Sequence[str] = (),
) -> ValidatedTuneCommand:
    """Build and validate the single authoritative Tune command object.

    Args:
        sandbox: Frozen GUI filesystem boundary.
        spec_path: Authored tuning spec.
        images_dir: Calibration image directory.
        output_dir: Tune output directory, which may not exist yet.
        strategy: Search strategy passed to the CLI.
        n_trials: Optional non-grid trial budget.
        storage_mode: Local SQLite path or server environment variable.
        storage_local_path: SQLite database path in local mode.
        storage_environment_name: Name only, never a credential value.
        n_workers: Optional worker count.
        slurm_partition: Optional SLURM partition.
        slurm_mem: Optional SLURM memory request.
        slurm_time: Optional SLURM time request.
        held_out_fraction: Optional robust-evaluation override.
        cv_group: Optional cross-validation group.
        slurm: Whether the execution target is SLURM.
        screen: Whether two-round screening is enabled.
        environ: Server environment mapping. Defaults to :data:`os.environ`.
        additional_issues: Preflight issues owned by the resulting plan.

    Returns:
        A validated plan. Invalid plans retain every issue and have no argv.
    """
    issues: list[str] = list(additional_issues)
    resolved_spec, path_issues = _resolve_existing(
        sandbox, spec_path, label="tuning spec", directory=False
    )
    issues.extend(path_issues)
    resolved_images, path_issues = _resolve_existing(
        sandbox, images_dir, label="image source", directory=True
    )
    issues.extend(path_issues)
    resolved_output, path_issues = _resolve_output(sandbox, output_dir)
    issues.extend(path_issues)

    normalized_strategy = (strategy or "").strip()
    if not normalized_strategy:
        issues.append("Choose a tuning strategy.")
    elif normalized_strategy not in STRATEGY_CHOICES:
        issues.append(f"Unknown tuning strategy: {normalized_strategy}")
    if n_trials is not None and n_trials <= 0:
        issues.append("Trial budget must be positive.")
    if n_workers is not None and n_workers <= 0:
        issues.append("Worker count must be positive.")
    if held_out_fraction is not None and not 0 <= held_out_fraction <= 1:
        issues.append("Held-out fraction must be between 0 and 1.")

    if storage_mode not in {"local", "environment"}:
        actual_storage = None
        redacted_storage = None
        issues.append("Choose a valid storage mode.")
    else:
        actual_storage, redacted_storage, storage_issues = _storage_tokens(
            sandbox=sandbox,
            mode=storage_mode,
            local_path=storage_local_path,
            environment_name=storage_environment_name,
            environ=os.environ if environ is None else environ,
        )
        issues.extend(storage_issues)

    semantic_tail: list[str] = []
    display_tail: list[str] = []
    if (
        resolved_spec is not None
        and resolved_images is not None
        and resolved_output is not None
        and normalized_strategy
    ):
        semantic_tail = tune_run_tail(
            spec_path=str(resolved_spec),
            images_dir=str(resolved_images),
            output_dir=str(resolved_output),
            strategy=normalized_strategy,
            n_trials=n_trials,
            storage_url=actual_storage,
            n_workers=n_workers,
            slurm_partition=slurm_partition,
            slurm_mem=slurm_mem,
            slurm_time=slurm_time,
            held_out_fraction=held_out_fraction,
            cv_group=cv_group,
            slurm=slurm,
            screen=screen,
        )
        display_tail = tune_run_tail(
            spec_path=str(resolved_spec),
            images_dir=str(resolved_images),
            output_dir=str(resolved_output),
            strategy=normalized_strategy,
            n_trials=n_trials,
            storage_url=redacted_storage,
            n_workers=n_workers,
            slurm_partition=slurm_partition,
            slurm_mem=slurm_mem,
            slurm_time=slurm_time,
            held_out_fraction=held_out_fraction,
            cv_group=cv_group,
            slurm=slurm,
            screen=screen,
        )

    argv = (
        tune_run_argv_from_tail(semantic_tail)
        if semantic_tail and not issues
        else []
    )
    display_tokens = (
        tune_run_argv_from_tail(display_tail)
        if display_tail
        else []
    )
    portable_tokens = [*_PORTABLE_PREFIX, *display_tail] if display_tail else []
    placeholders = any(
        token.startswith("<") and token.endswith(">")
        for token in portable_tokens
    )
    return ValidatedTuneCommand(
        argv=tuple(argv),
        semantic_tail=tuple(semantic_tail),
        display_tokens=tuple(display_tokens),
        portable_tokens=tuple(portable_tokens),
        spec_path=resolved_spec,
        images_dir=resolved_images,
        output_dir=resolved_output,
        execution_target="slurm" if slurm else "local",
        issues=tuple(issues),
        copy_eligible=bool(portable_tokens) and not issues and not placeholders,
    )


def render_tokens(
    tokens: Sequence[str],
    *,
    preserve_env_refs: bool = False,
) -> str:
    """Render shell-safe tokens, optionally preserving ``$ENV`` expansion."""
    rendered = []
    for token in tokens:
        if preserve_env_refs and re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", token):
            rendered.append(token)
        else:
            rendered.append(shlex.quote(token))
    return " ".join(rendered)


def render_launch_command(
    spec_path: str,
    input_dir: str,
    output_dir: str,
    *,
    strategy: str,
    n_trials: int | None,
    storage_url: str | None,
    n_workers: int | None = None,
    slurm_partition: str | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    held_out_fraction: float | None = None,
    cv_group: str | None = None,
    screen: bool = False,
    slurm: bool = False,
) -> str:
    """Render the legacy command string through the shared argv builder."""
    tokens = tune_run_argv(
        spec_path=spec_path,
        images_dir=input_dir,
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
        python="python",
    )
    return render_tokens(tokens)


__all__ = [
    "DEFAULT_STORAGE_ENV",
    "ExecutionTarget",
    "StorageMode",
    "ValidatedTuneCommand",
    "build_tune_command",
    "render_launch_command",
    "render_tokens",
]
