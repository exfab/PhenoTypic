"""Dash-free tuning-spec authoring: infer a space, edit it, run it, export it.

The one implementation of everything the Tune surfaces do *around* a tuning
run — the Dash views in :mod:`phenotypic.gui.tune` and the MCP server both call
in here, so neither owns behaviour the other has to reimplement.

**Search space** — the pure half of what used to be ``gui/tune/_space.py``:

* :func:`_build_search_space` — infer a pipeline's knobs, drop the nested ones,
  apply the caller's per-knob edits, and assemble a
  :class:`~phenotypic.tune.SearchSpace`.
* :func:`apply_space_edits` — the same edit pass over an *existing* space, so an
  authored :class:`~phenotypic.tune.TuningSpec` keeps its configured search space
  byte-for-byte when there is nothing to change.
* :func:`space_to_spec` — the OQ8 config-preserving builder. From an existing
  spec it replaces **only** ``search_space`` and keeps the run's scorer /
  strategy / budget / evaluator / held-out policy. From a bare ``ImagePipeline``
  it defaults the scorer (an unconfigured
  :class:`~phenotypic.tune.score.QCScorer` — the "review in Launch" signal),
  strategy, and budget.
* :func:`_load_space_source` — read a run's authored spec, else its base
  pipeline, else ``None``.

The editable knobs are the flat (:class:`~phenotypic.tune.targets.Param`) and
presence (:class:`~phenotypic.tune.targets.Presence`) targets; nested
(:class:`~phenotypic.tune.targets.Nested`) leaves are **dropped** from the
exported space (v1 tunes flat + presence only) — a surface that wants to show
them read-only re-infers them itself.

**Launch command** — from ``gui/tune/_command.py``: :func:`build_tune_command`
resolves every user-supplied path against a
:class:`~phenotypic._services.sandbox.SandboxRoot`, validates the strategy /
budget / storage choices, and returns one :class:`ValidatedTuneCommand`
carrying both the argv to execute and a **redacted** token list safe to show a
browser. :func:`render_tokens` is the only shell-quoting path.

**Export** — from ``gui/tune/_export.py``: read a finished run's winning (or
Pareto) parameters and write the tuned pipeline, with
:func:`prepare_best_from_run` / :func:`publish_prepared_export` splitting the
build from the atomic write so a caller can validate before publishing.

Nothing here may import Dash, and nothing here may import a rendering surface:
the split exists so both the Space view and the MCP server can call one tested
implementation. The Dash rendering lives in
:mod:`phenotypic.gui.tune._space_view`, which imports *this* module. Like the
rest of the tune tier, importing this module must never drag ``optuna`` into
``sys.modules``.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
)
from urllib.parse import urlsplit

from phenotypic import ImagePipeline
from phenotypic._services.argv import (
    tune_run_argv,
    tune_run_argv_from_tail,
    tune_run_tail,
)
from phenotypic._services.sandbox import SandboxRoot
from phenotypic.sdk_ import (
    atomic_write_text,
    best_params_path,
    best_pipeline_path,
    pareto_best_pipeline_path,
    resolve_pipeline_config_path,
    resolve_tuning_spec_path,
)
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune._spec import TuningSpec
from phenotypic.tune.strategy._config import STRATEGY_CHOICES

if TYPE_CHECKING:
    from phenotypic.tune import InferredSearchSpace, Knob, SearchSpace


class _RunRootLike(Protocol):
    """The one attribute :func:`_load_space_source` needs from a run handle.

    Structural rather than nominal on purpose: the concrete type is
    ``phenotypic.gui.tune._run_root.TuneRunRoot``, and naming it here — even
    under ``TYPE_CHECKING`` — would make this module import the GUI, which is
    the boundary this tier exists to hold.
    """

    @property
    def path(self) -> "Path":
        """The tune output directory (the run's ``--output`` root)."""


def _is_tuning_spec(obj: Any) -> bool:
    """Duck-type whether ``obj`` is a ``TuningSpec`` (vs. a bare pipeline).

    A ``TuningSpec`` carries the ``pipeline`` + ``search_space`` + ``scorer``
    triple; a bare ``ImagePipeline`` exposes ``get_ops`` but none of those. The
    duck-test keeps this module from importing the spec type at call time (the
    optuna-free / cheap-import contract).
    """
    return all(
        hasattr(obj, attr) for attr in ("pipeline", "search_space", "scorer")
    )


def _editable_knobs(inferred: "InferredSearchSpace") -> "list[Knob]":
    """The flat + presence knobs (drop depth-1 nested leaves; v1-editable set)."""
    return [
        knob for knob in inferred.knobs if type(knob.target).__name__ != "Nested"
    ]


def _apply_edits(
    knob: "Knob", edits: Mapping[str, Mapping[str, Any]]
) -> "Optional[Knob]":
    """Apply one knob's edits, returning the edited knob or ``None`` if untuned.

    A ``tunable: False`` edit drops the knob from the exported space; numeric
    ``low`` / ``high`` / ``log`` edits override a range domain's bounds and a
    ``choices`` edit overrides a categorical domain. A knob with no entry in
    ``edits`` passes through unchanged.

    Args:
        knob: The inferred knob to edit.
        edits: The per-key edit map (``{key: {"low": …, "tunable": …, …}}``).

    Returns:
        The (possibly re-bounded) knob, or ``None`` when the user toggled it off.
    """
    edit = edits.get(knob.key)
    if edit is None:
        return knob
    if edit.get("tunable") is False:
        return None
    domain = knob.domain
    kind = domain.kind
    if kind in ("float_range", "int_range"):
        updates: dict[str, Any] = {}
        if edit.get("low") is not None:
            updates["low"] = edit["low"]
        if edit.get("high") is not None:
            updates["high"] = edit["high"]
        if edit.get("log") is not None:
            updates["log"] = bool(edit["log"])
        if updates:
            domain = domain.model_copy(update=updates)
    elif kind == "categorical" and edit.get("choices") is not None:
        recovered = _recover_typed_choices(domain, edit["choices"])
        domain = domain.model_copy(update={"choices": recovered})
    return knob.model_copy(update={"domain": domain})


def _recover_typed_choices(
    domain: Any, selected: "Iterable[Any]"
) -> "tuple[Any, ...]":
    """Map a checklist's stringified ``selected`` values back to typed members.

    The Space categorical checklist renders each option's ``value`` as
    ``str(choice)`` (a checklist value must be a string), so the export ``State``
    returns the **stringified** subset. Only the knob's own ``domain.choices``
    knows the real types, so recover them here: build ``{str(c): c}`` over the
    original members and map each selected string back through it, dropping any
    value not present (a stale / unknown selection). This preserves the member
    types — ``"True"`` → ``True``, ``"1.0"`` → ``1.0`` — AND handles subset
    narrowing, so the exported ``Categorical`` is semantically faithful and
    ``build_pipeline`` applies the override correctly.

    Args:
        domain: The knob's original ``Categorical`` domain (the type source).
        selected: The stringified subset the checklist returned.

    Returns:
        The selected original members, typed, in the domain's declaration order.
    """
    selected_strings = {str(value) for value in selected}
    # Iterate the original choices to keep the domain's declaration order and
    # drop any stale / unknown selection (a value not in the original set).
    return tuple(
        choice for choice in domain.choices if str(choice) in selected_strings
    )


def _build_search_space(
    pipeline: "ImagePipeline", edits: Mapping[str, Mapping[str, Any]]
) -> "SearchSpace":
    """Infer the editable (flat + presence) knobs, apply ``edits``, build a space.

    Mines ``pipeline`` with :func:`~phenotypic.tune.infer_search_space`, keeps the
    flat + presence knobs (nested leaves are dropped — v1 tunes those only),
    applies the per-knob edits, and assembles a :class:`~phenotypic.tune.SearchSpace`.

    Args:
        pipeline: The base pipeline to mine.
        edits: The per-key edit map (empty → the inferred defaults).

    Returns:
        The exported :class:`~phenotypic.tune.SearchSpace`.
    """
    from phenotypic.tune import SearchSpace, infer_search_space

    inferred = infer_search_space(pipeline)
    knobs = []
    for knob in _editable_knobs(inferred):
        edited = _apply_edits(knob, edits)
        if edited is not None:
            knobs.append(edited)
    return SearchSpace(knobs=tuple(knobs))


def apply_space_edits(
    search_space: "SearchSpace",
    edits: Mapping[str, Mapping[str, Any]],
) -> "SearchSpace":
    """Apply domain edits to an existing space without re-inferring it.

    This is the Setup path for an existing :class:`TuningSpec`: its configured
    search space is preserved byte-for-byte when ``edits`` is empty, while
    explicit editor changes affect only the named knobs.
    """
    from phenotypic.tune import SearchSpace

    if not edits:
        return search_space
    knobs = []
    for knob in search_space.knobs:
        edited = _apply_edits(knob, edits)
        if edited is not None:
            knobs.append(edited)
    return SearchSpace(knobs=tuple(knobs))


def _default_qc_scorer() -> Any:
    """A fresh, unconfigured ``QCScorer`` — the "review in Launch" signal.

    Built over an **empty** layout frame so :meth:`QCScorer.availability` is
    ``False``: the Space view surfaces this as a "configure the scorer's metadata
    in Launch before tuning" note. A fresh-from-pipeline export cannot fabricate a
    real layout (there is no run dir with a ``tuning_spec.json`` to inherit one
    from), so the user must point the scorer at a metadata path in Launch. Such a
    scorer is intentionally not JSON-round-trippable (an in-memory frame has no
    source path) — the export-from-existing-spec path is the round-trippable one.
    """
    import pandas as pd

    from phenotypic.analysis import ExpectedVsDetectedCount
    from phenotypic.schema import METADATA
    from phenotypic.tune.score import QCScorer

    image_name = str(METADATA.IMAGE_NAME)
    empty = pd.DataFrame({image_name: [], "Object_Label": []})
    return QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=empty, groupby=[image_name]
        )
    )


def space_to_spec(
    pipeline_or_spec: "Union[ImagePipeline, TuningSpec]",
    edits: Mapping[str, Mapping[str, Any]],
) -> "TuningSpec":
    """Build a ``TuningSpec`` from a pipeline or an existing spec (OQ8).

    The config-preserving export. When ``pipeline_or_spec`` is an existing
    :class:`~phenotypic.tune.TuningSpec` (the run already had a
    ``tuning_spec.json``), the result replaces **only** its ``search_space`` and
    keeps the run's ``scorer`` / ``strategy`` / ``budget`` / ``evaluator`` /
    ``held_out`` verbatim — the user is re-shaping the search, not the objective.
    When it is a bare :class:`~phenotypic.ImagePipeline` (a fresh start from a
    ``pipeline.json``), the result defaults the scorer (an unconfigured
    :class:`~phenotypic.tune.score.QCScorer` — :func:`_default_qc_scorer`), a
    :class:`~phenotypic.tune.strategy.GridConfig` strategy, a default
    :class:`~phenotypic.tune.Budget`, and a default
    :class:`~phenotypic.tune.Evaluator`; the unconfigured scorer is the
    "review these in Launch" signal.

    Either way the exported ``search_space`` is the inferred **flat + presence**
    knobs (nested leaves dropped) with the user's per-knob ``edits`` applied.

    Args:
        pipeline_or_spec: A live pipeline (fresh) or an existing tuning spec.
        edits: The per-key edit map (``{key: {"low": …, "tunable": …}}``); empty
            keeps the inferred defaults.

    Returns:
        The exported :class:`~phenotypic.tune.TuningSpec`.
    """
    from phenotypic.tune import Budget, Evaluator
    from phenotypic.tune.strategy import GridConfig

    if _is_tuning_spec(pipeline_or_spec):
        spec: "TuningSpec" = pipeline_or_spec  # type: ignore[assignment]
        new_space = _build_search_space(spec.pipeline, edits)
        # Replace ONLY the search space; the run's objective / optimizer / budget
        # carry verbatim (OQ8). ``model_copy`` re-runs the after-validators, so
        # the new space's targets are re-checked against the unchanged pipeline.
        return spec.model_copy(update={"search_space": new_space})

    pipeline: "ImagePipeline" = pipeline_or_spec  # type: ignore[assignment]
    new_space = _build_search_space(pipeline, edits)
    return TuningSpec(
        pipeline=pipeline,
        search_space=new_space,
        scorer=_default_qc_scorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def _load_space_source(
    root: "_RunRootLike",
) -> "Optional[Union[TuningSpec, ImagePipeline]]":
    """Load the run's existing spec, else its base pipeline, else ``None``.

    Prefers an existing ``deliverables/tuning_spec.json`` (so the export preserves
    its scorer / strategy / budget — OQ8); falls back to a bare
    ``deliverables/pipeline.json``. A read / parse failure degrades to ``None`` so
    the view shows the pick-a-pipeline prompt rather than raising.

    Args:
        root: The bound tune output handle.

    Returns:
        The loaded ``TuningSpec`` (preferred), the ``ImagePipeline`` (fallback),
        or ``None`` when neither resolves.
    """
    spec_path = resolve_tuning_spec_path(root.path)
    if spec_path.is_file():
        spec = _try_load_spec(spec_path)
        if spec is not None:
            return spec
    pipe_path = resolve_pipeline_config_path(root.path)
    if pipe_path.is_file():
        return _try_load_pipeline(pipe_path)
    return None


def _try_load_spec(spec_path: Any) -> "Optional[TuningSpec]":
    """Load a ``TuningSpec`` from ``spec_path``, or ``None`` on any failure.

    A ``tuning_spec.json`` may instead hold an ``InferredSearchSpace`` proposal
    (what ``auto-space`` writes) — that is not a full spec, so it fails to
    validate here and the loader falls back to the pipeline path.
    """
    try:
        return TuningSpec.model_validate_json(spec_path.read_text())
    except Exception:  # noqa: BLE001 - a proposal / corrupt file degrades to None
        return None


def _try_load_pipeline(pipe_path: Any) -> "Optional[ImagePipeline]":
    """Load an ``ImagePipeline`` from ``pipe_path``, or ``None`` on any failure."""
    try:
        return ImagePipeline.from_json(pipe_path.read_text())
    except Exception:  # noqa: BLE001 - a corrupt pipeline.json degrades to None
        return None


# ---------------------------------------------------------------------------
# Launch command (was gui/tune/_command.py)
# ---------------------------------------------------------------------------


ExecutionTarget = Literal["local", "slurm"]
StorageMode = Literal["spec", "local", "environment"]

DEFAULT_STORAGE_ENV = "PHENOTYPIC_STORAGE_URL"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PORTABLE_PREFIX = ("uv", "run", "python", "-m", "phenotypic.tune")
_INLINE_PASSWORD_ISSUE = (
    "The configured storage URL embeds an inline password. "
    "Use ~/.pgpass or PGPASSWORD instead."
)


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


def storage_url_preflight_issue(value: str | None) -> str | None:
    """Return a credential-safe issue for an inline-password URL."""
    if not value:
        return None
    try:
        has_password = urlsplit(value).password is not None
    except ValueError:
        return None
    return _INLINE_PASSWORD_ISSUE if has_password else None


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
    if mode == "spec":
        return None, None, []
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
    credential_issue = storage_url_preflight_issue(value)
    if credential_issue is not None:
        return None, f"${name}", [credential_issue]
    return value, f"${name}", []


def build_tune_command(
    *,
    sandbox: SandboxRoot,
    spec_path: str | None,
    images_dir: str | None,
    output_dir: str | None,
    strategy: str | None,
    n_trials: int | None,
    effective_strategy: str | None = None,
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
        strategy: Optional search-strategy CLI override.
        effective_strategy: Strategy selected by the authored spec when no
            override is needed. Used only for validation.
        n_trials: Optional non-grid trial budget.
        storage_mode: Authored-spec storage, a local SQLite path, or a server
            environment variable.
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

    normalized_strategy = (strategy or "").strip() or None
    normalized_effective_strategy = (
        (effective_strategy or "").strip() or normalized_strategy
    )
    if not normalized_effective_strategy:
        issues.append("Choose a tuning strategy.")
    elif normalized_effective_strategy not in STRATEGY_CHOICES:
        issues.append(
            f"Unknown tuning strategy: {normalized_effective_strategy}"
        )
    if normalized_strategy and normalized_strategy not in STRATEGY_CHOICES:
        issues.append(f"Unknown tuning strategy override: {normalized_strategy}")
    if n_trials is not None and n_trials <= 0:
        issues.append("Trial budget must be positive.")
    effective_n_trials = (
        None if normalized_effective_strategy == "grid" else n_trials
    )
    if n_workers is not None and n_workers <= 0:
        issues.append("Worker count must be positive.")
    if held_out_fraction is not None and not 0 <= held_out_fraction <= 1:
        issues.append("Held-out fraction must be between 0 and 1.")

    if storage_mode not in {"spec", "local", "environment"}:
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
        and normalized_effective_strategy
    ):
        semantic_tail = tune_run_tail(
            spec_path=str(resolved_spec),
            images_dir=str(resolved_images),
            output_dir=str(resolved_output),
            strategy=normalized_strategy,
            n_trials=effective_n_trials,
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
            n_trials=effective_n_trials,
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


# ---------------------------------------------------------------------------
# Pipeline export (was gui/tune/_export.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedPipelineExport:
    """Complete in-memory pipeline export awaiting atomic publication."""

    path: Path
    payload: str


def export_winning_pipeline(
    base: ImagePipeline,
    params: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write the single-objective tuned winner pipeline."""
    pipeline = build_pipeline(base, params)
    path = best_pipeline_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.to_json(path)
    return path


def export_pareto_pipeline(
    base: ImagePipeline,
    params: dict[str, Any],
    output_dir: Path,
    *,
    objective: str,
) -> Path:
    """Write a per-objective Pareto tuned pipeline."""
    pipeline = build_pipeline(base, params)
    path = pareto_best_pipeline_path(output_dir, objective)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.to_json(path)
    return path


def _params_from_best_params_payload(payload: object) -> dict[str, Any]:
    """Extract flat knob params from canonical or legacy best-params payloads."""
    if not isinstance(payload, dict):
        raise ValueError("best params must be a JSON object")
    wrapped = payload.get("params")
    if isinstance(wrapped, dict):
        return wrapped
    if "params" in payload:
        raise ValueError("best params 'params' must be a JSON object")
    return payload


def prepare_best_from_run(output_dir: Path) -> PreparedPipelineExport:
    """Read winner inputs and build the complete pipeline payload in memory."""
    spec_path = resolve_tuning_spec_path(output_dir)
    if not spec_path.is_file():
        raise FileNotFoundError(f"tuning spec not found: {spec_path}")

    params_path = best_params_path(output_dir)
    if not params_path.is_file():
        raise FileNotFoundError(f"best params not found: {params_path}")

    spec = TuningSpec.model_validate_json(spec_path.read_text())
    params = _params_from_best_params_payload(json.loads(params_path.read_text()))
    pipeline = build_pipeline(spec.pipeline, params)
    payload = pipeline.to_json()
    if not isinstance(payload, str):  # pragma: no cover
        raise RuntimeError("in-memory pipeline serialization returned no payload")
    return PreparedPipelineExport(
        path=best_pipeline_path(output_dir),
        payload=payload,
    )


def publish_prepared_export(prepared: PreparedPipelineExport) -> Path:
    """Atomically publish one fully prepared pipeline payload."""
    atomic_write_text(prepared.path, prepared.payload)
    return prepared.path


def export_best_from_run(output_dir: Path) -> Path:
    """Read a completed run's winner params and atomically write its pipeline."""
    return publish_prepared_export(prepare_best_from_run(output_dir))


__all__ = [
    "PreparedPipelineExport",
    "ValidatedTuneCommand",
    "apply_space_edits",
    "build_tune_command",
    "export_best_from_run",
    "export_pareto_pipeline",
    "export_winning_pipeline",
    "prepare_best_from_run",
    "publish_prepared_export",
    "render_launch_command",
    "render_tokens",
    "space_to_spec",
    "storage_url_preflight_issue",
]
