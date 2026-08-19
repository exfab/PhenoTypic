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

**Setup authoring** — from ``gui/tune/_setup_authoring.py``: resolve the
sandbox-relative pipeline / metadata paths a picker hands back, cache a
versioned :class:`SetupDraft`, and atomically write the authored
``.json.pht-tune`` spec into the sandbox's tune-presets directory.

**Validation** — from ``gui/tune/_validation.py``: :func:`validate_setup` and
:func:`preflight_issues` return the :class:`Issue` list that blocks Continue /
Deploy. :func:`grid_feasibility` came with them from ``gui/tune/_domain_editor``
— it is a pure predicate over a :class:`~phenotypic.tune.SearchSpace` and
``preflight_issues`` is its only production caller, so it belongs beside it
rather than one import above the boundary.

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

import hashlib
import json
import os
import re
import secrets
import shlex
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    Union,
)
from urllib.parse import urlsplit

from pydantic import ValidationError
from typing_extensions import NotRequired

from phenotypic import ImagePipeline
from phenotypic._services.argv import (
    tune_run_argv,
    tune_run_argv_from_tail,
    tune_run_tail,
)
from phenotypic._services.sandbox import (
    SandboxRoot,
    _is_safe_relative_path,
    _v1_selection_matches_sandbox,
    sandbox_fingerprint,
)
from phenotypic.analysis import ExpectedVsDetectedCount
# The single accepted upward import in this module, and the reason for it.
# ``resolve_metadata_csv`` is a five-line compatibility wrapper over
# ``resolve_metadata_csv_state`` in a 596-line browser-payload resolver that
# transitively reaches ``gui.shell._source_context`` -> ``._classifier``.
# Promoting that chain is a design decision about inverting the payload
# dependency, not a side effect of this cluster, so it is allowlisted in
# ``tests/unit/services/test_import_purity.py`` and TEMPORARY: the entry is
# tracked for removal when a later phase promotes or inverts the resolver.
from phenotypic.gui.shell._metadata_context import resolve_metadata_csv
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    CONFIG_SUFFIX_TUNING,
    PIPELINE_CONFIG_SUFFIXES,
    atomic_write_text,
    best_params_path,
    best_pipeline_path,
    matches_any_suffix,
    pareto_best_pipeline_path,
    resolve_pipeline_config_path,
    resolve_tuning_spec_path,
)
# Not re-exported from ``phenotypic.sdk_``'s ``__init__`` -- same as
# ``IMAGE_EXTS``, which ``gui/_config`` also imports from the private module.
from phenotypic.sdk_._io_constants import tune_presets_dir
from phenotypic.tune.score import QCScorer
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune._search_space import FloatRange, IntRange, SearchSpace
from phenotypic.tune._spec import TuningSpec
from phenotypic.tune.strategy._config import STRATEGY_CHOICES

if TYPE_CHECKING:
    from phenotypic.tune import InferredSearchSpace, Knob


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
    from phenotypic.schema import IMAGE
    from phenotypic.tune.score import QCScorer

    image_name = str(IMAGE.IMAGE_NAME)
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


# ---------------------------------------------------------------------------
# Setup / preflight validation (was gui/tune/_validation.py, plus
# grid_feasibility from gui/tune/_domain_editor.py)
# ---------------------------------------------------------------------------

Blocks = Literal["continue", "deploy", "both"]


@dataclass(frozen=True)
class Issue:
    """One Setup validation problem."""

    section: str
    message: str
    blocks: Blocks = "both"


def grid_feasibility(space: SearchSpace) -> tuple[bool, str]:
    """Return whether every knob can be enumerated by grid search."""
    for knob in space.knobs:
        domain = knob.domain
        if isinstance(domain, FloatRange) and domain.step is None:
            return (
                False,
                f"Grid unavailable: {knob.key} is a continuous float. "
                "Add a step, pin it, or use Optuna.",
            )
    return True, "All active knobs are enumerable."


def validate_setup(
    space: SearchSpace, *, scorer_kind: str, metadata_present: bool
) -> list[Issue]:
    """Return blocking Setup issues."""
    issues: list[Issue] = []
    if len(space.knobs) == 0:
        issues.append(Issue("search_space", "No active knobs to tune."))
    for knob in space.knobs:
        domain = knob.domain
        if isinstance(domain, (FloatRange, IntRange)) and not domain.high > domain.low:
            issues.append(Issue("search_space", f"{knob.key}: low must be < high."))
    if scorer_kind == "qc" and not metadata_present:
            issues.append(Issue("scorer", "QC scorer needs a metadata CSV."))
    return issues


def preflight_issues(space: SearchSpace, *, strategy: str) -> list[Issue]:
    """Return Run/Deploy preflight issues."""
    if strategy != "grid":
        return []
    ok, message = grid_feasibility(space)
    if ok:
        return []
    return [Issue("strategy", message, blocks="deploy")]


def can_deploy(setup_issues: list[Issue], run_issues: list[Issue]) -> bool:
    """Return whether no issue blocks Deploy."""
    for issue in [*setup_issues, *run_issues]:
        if issue.blocks in {"deploy", "both"}:
            return False
    return True


def spec_path_issue(spec_path: str | None) -> Issue | None:
    """Return a Deploy issue when the CLI spec path is not a tuning spec."""
    if not spec_path:
        return Issue("spec", "Choose a pipeline or tuning spec first.", blocks="deploy")
    if matches_any_suffix(spec_path, (CONFIG_SUFFIX_TUNING,)):
        return None
    return Issue(
        "spec",
        "Deploy needs an authored .json.pht-tune spec. Pipeline-only paths "
        "must be converted in Setup before launch.",
        blocks="deploy",
    )


# ---------------------------------------------------------------------------
# Setup authoring (was gui/tune/_setup_authoring.py)
# ---------------------------------------------------------------------------


SetupPathKind = Literal["pipeline", "metadata"]
SetupPathSource = Literal["typed", "picker", "shared", "unset"]

SETUP_DRAFT_VERSION = 2
_SETUP_DRAFT_CACHE_SIZE = 256
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
_METADATA_SUFFIXES = frozenset({".csv", ".parquet"})
_PIPELINE_SUFFIXES = PIPELINE_CONFIG_SUFFIXES | frozenset({CONFIG_SUFFIX_TUNING})


class SetupPathPayload(TypedDict):
    """Versioned, sandbox-bound current-session picker payload."""

    version: int
    kind: SetupPathKind
    relative_path: str
    absolute_path_at_selection: str
    sandbox_fingerprint: str
    selected_at: str
    selection_id: NotRequired[str]


@dataclass(frozen=True)
class SetupPathResolution:
    """Resolved Setup path plus its precedence source and all issues."""

    path: Path | None
    source: SetupPathSource
    issues: tuple[str, ...] = ()

    def to_store(self) -> dict[str, object]:
        """Return a JSON-serializable Dash-store payload."""
        return {
            "path": str(self.path) if self.path is not None else None,
            "source": self.source,
            "issues": list(self.issues),
        }


def setup_path_resolution_from_store(
    value: object,
    *,
    sandbox: SandboxRoot | None = None,
    kind: SetupPathKind | None = None,
) -> SetupPathResolution:
    """Parse a path-resolution store, optionally rechecking its sandbox path."""
    if not isinstance(value, dict):
        return SetupPathResolution(None, "unset")
    path = value.get("path")
    source = value.get("source")
    issues = value.get("issues")
    if (
        (path is not None and not isinstance(path, str))
        or source not in {"typed", "picker", "shared", "unset"}
        or not isinstance(issues, list)
    ):
        return SetupPathResolution(
            None,
            "unset",
            ("Setup path state is invalid; select the file again.",),
        )
    resolution = SetupPathResolution(
        Path(path) if path else None,
        source,
        tuple(str(issue) for issue in issues),
    )
    if sandbox is None or kind is None or resolution.path is None:
        return resolution
    checked = _candidate_path(
        sandbox,
        str(resolution.path),
        kind=kind,
        source=resolution.source,
    )
    return SetupPathResolution(
        checked.path,
        checked.source,
        tuple(dict.fromkeys((*resolution.issues, *checked.issues))),
    )


@dataclass(frozen=True)
class SetupAuthoringResult:
    """In-memory authored spec or its complete validation issue list."""

    spec: TuningSpec | None
    source_is_spec: bool
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Whether the full authored spec validated."""
        return self.spec is not None and not self.issues


@dataclass(frozen=True)
class SetupDraft:
    """One revisioned, validated interpretation of all Setup controls.

    ``revision`` binds the resolved paths, current source bytes, search-space
    edits, scorer choice, validation issues, and validated spec JSON. The full
    object remains server-side because an existing spec may contain credentials.
    """

    revision: str
    source_revision: str
    pipeline_path: str | None
    pipeline_source: SetupPathSource
    metadata_path: str | None
    metadata_source: SetupPathSource
    replace_scorer: bool
    source_is_spec: bool
    edits: dict[str, dict[str, object]]
    source_fingerprint: str
    metadata_fingerprint: str
    scorer_name: str | None
    issues: tuple[str, ...]
    spec_json: str | None

    @property
    def is_valid(self) -> bool:
        """Whether this draft contains a fully validated tuning spec."""
        return self.spec_json is not None and not self.issues

    def to_store(self) -> dict[str, object]:
        """Return a redacted summary that never includes authored spec content.

        This summary is diagnostic only. Browser callbacks use
        :class:`SetupDraftCache.publish`, which adds an unguessable server-cache
        handle while retaining only this revision in client transport.
        """
        return {
            "version": SETUP_DRAFT_VERSION,
            "revision": self.revision,
        }


class SetupDraftCache:
    """Bounded per-app server cache for credential-bearing Setup drafts."""

    def __init__(self, *, max_entries: int = _SETUP_DRAFT_CACHE_SIZE) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._drafts: OrderedDict[str, SetupDraft] = OrderedDict()
        self._handles_by_revision: dict[str, str] = {}
        self._lock = RLock()

    def publish(self, draft: SetupDraft) -> dict[str, object]:
        """Cache ``draft`` and return its credential-free browser receipt."""
        with self._lock:
            handle = self._handles_by_revision.get(draft.revision)
            if handle is None or handle not in self._drafts:
                handle = secrets.token_urlsafe(32)
                self._handles_by_revision[draft.revision] = handle
            self._drafts[handle] = draft
            self._drafts.move_to_end(handle)
            while len(self._drafts) > self._max_entries:
                evicted_handle, evicted = self._drafts.popitem(last=False)
                if self._handles_by_revision.get(evicted.revision) == evicted_handle:
                    self._handles_by_revision.pop(evicted.revision, None)
        return {
            "version": SETUP_DRAFT_VERSION,
            "handle": handle,
            "revision": draft.revision,
        }

    def resolve(self, value: object) -> SetupDraft | None:
        """Resolve an unmodified browser receipt to its server-side draft."""
        if not isinstance(value, dict) or value.get("version") != SETUP_DRAFT_VERSION:
            return None
        if set(value) != {"version", "handle", "revision"}:
            return None
        handle = value.get("handle")
        revision = value.get("revision")
        if (
            not isinstance(handle, str)
            or not handle
            or not isinstance(revision, str)
            or not revision
        ):
            return None
        with self._lock:
            draft = self._drafts.get(handle)
            if draft is None or draft.revision != revision:
                return None
            self._drafts.move_to_end(handle)
            return draft


@dataclass(frozen=True)
class SetupWriteReceipt:
    """Immutable provenance for the exact authored bytes written by Continue."""

    path: Path
    draft_revision: str
    source_fingerprint: str
    metadata_fingerprint: str
    authored_fingerprint: str


def _safe_stem(path: Path) -> str:
    """Return a filesystem-safe stem for a GUI-authored spec."""
    stem = _SAFE_STEM.sub("-", path.stem).strip(".-")
    return stem or "tuning-spec"


def path_content_fingerprint(path: Path | None) -> str:
    """Return a canonical path-and-content identity without exposing content.

    Args:
        path: File to identify, or ``None`` for an unset optional input.

    Returns:
        A SHA-256 digest over the canonical path and current file bytes. Missing
        and unreadable paths receive stable sentinel identities.
    """
    if path is None:
        return hashlib.sha256(b"unset").hexdigest()
    try:
        canonical = path.expanduser().resolve(strict=False)
    except OSError:
        canonical = path.expanduser().absolute()
    try:
        content = canonical.read_bytes()
        state = b"file"
    except OSError:
        content = b""
        state = b"unavailable"
    digest = hashlib.sha256()
    digest.update(state)
    digest.update(b"\0")
    digest.update(str(canonical).encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    digest.update(content)
    return digest.hexdigest()


def authored_content_fingerprint(path: Path) -> str:
    """Return the SHA-256 digest of one authored spec's current bytes."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _canonical_edits(
    edits: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    """Return a detached, JSON-safe edit mapping in deterministic key order."""
    encoded = json.dumps(
        edits or {},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in sorted(decoded.items())
        if isinstance(value, dict)
    }


def _content_revision(payload: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 revision for JSON-safe content."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _setup_source_revision(
    *,
    pipeline_path: str | None,
    pipeline_source: SetupPathSource,
    source_fingerprint: str,
) -> str:
    """Return the revision controlling source-dependent editor rendering."""
    return _content_revision(
        {
            "pipeline_path": pipeline_path,
            "pipeline_source": pipeline_source,
            "source_fingerprint": source_fingerprint,
        }
    )


def setup_draft_from_store(
    value: object,
    *,
    cache: SetupDraftCache,
) -> SetupDraft | None:
    """Resolve a redacted browser receipt through its per-app server cache."""
    return cache.resolve(value)


def load_pipeline_or_spec(path: Path) -> ImagePipeline | TuningSpec:
    """Load a selected pipeline or existing tuning spec from disk."""
    text = path.read_text(encoding="utf-8")
    if matches_any_suffix(path, (CONFIG_SUFFIX_TUNING,)):
        return TuningSpec.model_validate_json(text)
    return ImagePipeline.from_json(text)


def setup_path_payload(
    sandbox: SandboxRoot,
    path: str | Path,
    *,
    kind: SetupPathKind,
) -> SetupPathPayload | None:
    """Build a fresh picker payload after validating one selected file."""
    try:
        resolved = sandbox.resolve(path)
    except ValueError:
        return None
    suffixes = _PIPELINE_SUFFIXES if kind == "pipeline" else _METADATA_SUFFIXES
    if not resolved.is_file() or not matches_any_suffix(resolved, suffixes):
        return None
    return {
        "version": 2,
        "kind": kind,
        "relative_path": resolved.relative_to(sandbox.root).as_posix() or ".",
        "absolute_path_at_selection": str(resolved),
        "sandbox_fingerprint": sandbox_fingerprint(sandbox),
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "selection_id": secrets.token_hex(16),
    }


def resolve_picker_payload(
    sandbox: SandboxRoot,
    payload: object,
    *,
    kind: SetupPathKind,
) -> Path | None:
    """Resolve a selected V1/V2 descriptor against the current sandbox.

    V2 is fingerprint-bound. V1 remains readable only when its absolute and
    sandbox-relative mirrors still identify the same current-sandbox file.
    """
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    if version == 2:
        if (
            payload.get("kind") != kind
            or payload.get("sandbox_fingerprint")
            != sandbox_fingerprint(sandbox)
        ):
            return None
        relative = payload.get("relative_path")
        if not isinstance(relative, str) or not _is_safe_relative_path(relative):
            return None
    elif version == 1:
        raw_path = payload.get("abs_path", payload.get("path"))
        relative = payload.get("rel_path", payload.get("relative_path"))
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or not isinstance(relative, str)
            or not _is_safe_relative_path(relative)
            or not _v1_selection_matches_sandbox(
                sandbox,
                raw_path=raw_path,
                relative_path=relative,
            )
        ):
            return None
    else:
        return None
    candidate = setup_path_payload(sandbox, relative, kind=kind)
    return (
        Path(candidate["absolute_path_at_selection"])
        if candidate is not None
        else None
    )


def _candidate_path(
    sandbox: SandboxRoot,
    candidate: str,
    *,
    kind: SetupPathKind,
    source: SetupPathSource,
) -> SetupPathResolution:
    """Validate one precedence-selected path without falling through."""
    try:
        path = sandbox.resolve(candidate.strip())
    except ValueError:
        return SetupPathResolution(
            None,
            source,
            (f"{kind.capitalize()} path escapes the GUI sandbox.",),
        )
    suffixes = _PIPELINE_SUFFIXES if kind == "pipeline" else _METADATA_SUFFIXES
    issues = []
    if not path.is_file():
        issues.append(f"{kind.capitalize()} path is not an existing file: {path}")
    if not matches_any_suffix(path, suffixes):
        suffix_text = ", ".join(sorted(suffixes))
        issues.append(
            f"{kind.capitalize()} path must use one of: {suffix_text}"
        )
    return SetupPathResolution(path, source, tuple(issues))


def resolve_setup_path(
    *,
    sandbox: SandboxRoot,
    kind: SetupPathKind,
    typed_path: str | None,
    picker_payload: object,
    shared_payload: object,
) -> SetupPathResolution:
    """Resolve Setup input using typed, picker, shared, then unset precedence."""
    if typed_path and typed_path.strip():
        return _candidate_path(
            sandbox, typed_path, kind=kind, source="typed"
        )

    picked = resolve_picker_payload(sandbox, picker_payload, kind=kind)
    if picked is not None:
        return SetupPathResolution(picked, "picker")

    if kind == "metadata":
        shared = resolve_metadata_csv(sandbox, shared_payload)
        if shared is not None:
            return SetupPathResolution(shared, "shared")
    else:
        shared = resolve_picker_payload(sandbox, shared_payload, kind="pipeline")
        if shared is not None:
            return SetupPathResolution(shared, "shared")
        legacy_candidates: list[str] = []
        if isinstance(shared_payload, str):
            legacy_candidates.append(shared_payload)
        elif isinstance(shared_payload, dict) and shared_payload.get("version") is None:
            for key in ("relative_path", "rel_path", "path", "abs_path"):
                candidate = shared_payload.get(key)
                if isinstance(candidate, str) and candidate:
                    legacy_candidates.append(candidate)
                    break
        for candidate in legacy_candidates:
            resolution = _candidate_path(
                sandbox,
                candidate,
                kind="pipeline",
                source="shared",
            )
            if not resolution.issues:
                return resolution

    return SetupPathResolution(None, "unset")


def authored_setup_spec_path(
    *,
    sandbox_root: Path,
    source_path: Path,
    metadata_path: Path | None = None,
    authored_content: str = "",
) -> Path:
    """Return a collision-safe GUI preset path for one authored spec.

    The readable source stem is only a label. The digest also binds the
    canonical source and metadata identities plus the exact authored content,
    so equal stems in different directories and changed inputs cannot alias.
    """
    identity = {
        "source": path_content_fingerprint(source_path),
        "metadata": path_content_fingerprint(metadata_path),
        "authored": hashlib.sha256(authored_content.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    suffix = hashlib.sha256(encoded).hexdigest()[:20]
    return (
        tune_presets_dir(sandbox_root)
        / f"{_safe_stem(source_path)}-{suffix}.setup.json.pht-tune"
    )


def _validation_messages(exc: ValidationError) -> list[str]:
    """Format every pydantic issue with its precise model location."""
    messages = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        prefix = f"{location}: " if location else ""
        messages.append(prefix + str(error.get("msg", "invalid value")))
    return messages



def _normalize_setup_metadata_groupby(column: str) -> str:
    """Normalize a metadata-only Setup group-by without relabeling locators.

    Setup authors a QC scorer from a layout table, so an unqualified unknown
    label is generic metadata. Qualified values may instead be object
    locators or measurement keys, and the mixed-reference helper leaves them
    unchanged.
    """
    from phenotypic.gui.shell._metadata_context import (
        normalize_metadata_column_reference,
    )

    return normalize_metadata_column_reference(str(column))


def build_authored_setup_spec(
    *,
    pipeline_or_spec_path: Path,
    metadata_path: Path | None,
    edits: Mapping[str, Mapping[str, object]] | None = None,
    replace_scorer: bool = False,
    metadata_groupby: list[str] | None = None,
) -> SetupAuthoringResult:
    """Construct and fully validate a Setup-authored spec in memory."""
    issues: list[str] = []
    if not pipeline_or_spec_path.is_file():
        return SetupAuthoringResult(
            None,
            False,
            (f"Pipeline/spec file does not exist: {pipeline_or_spec_path}",),
        )
    try:
        source = load_pipeline_or_spec(pipeline_or_spec_path)
    except (OSError, ValueError, ValidationError):
        return SetupAuthoringResult(
            None,
            False,
            ("Could not load pipeline/spec; review the server log.",),
        )

    source_is_spec = isinstance(source, TuningSpec)
    needs_metadata = not source_is_spec or replace_scorer
    if needs_metadata:
        if metadata_path is None:
            issues.append(
                "Metadata is required for the metadata-backed QC scorer."
            )
        elif not metadata_path.is_file():
            issues.append(f"Metadata file does not exist: {metadata_path}")
        elif not matches_any_suffix(metadata_path, _METADATA_SUFFIXES):
            issues.append("Metadata must be a CSV or Parquet file.")
    elif metadata_path is not None and not metadata_path.is_file():
        issues.append(f"Selected metadata file does not exist: {metadata_path}")

    try:
        if isinstance(source, TuningSpec):
            spec = source
            if edits:
                space = apply_space_edits(source.search_space, edits)
                spec = spec.model_copy(update={"search_space": space})
        else:
            spec = space_to_spec(source, edits=edits or {})

        if needs_metadata and metadata_path is not None and metadata_path.is_file():
            groupby = [
                _normalize_setup_metadata_groupby(column)
                for column in (metadata_groupby or [str(IMAGE.IMAGE_NAME)])
            ]
            scorer = QCScorer(
                check=ExpectedVsDetectedCount(
                    metadata=str(metadata_path),
                    groupby=groupby,
                )
            )
            spec = spec.model_copy(update={"scorer": scorer})

        if not spec.search_space.knobs:
            issues.append("No active knobs to tune.")
        validated = TuningSpec.model_validate_json(spec.model_dump_json())
    except ValidationError as exc:
        issues.extend(_validation_messages(exc))
        validated = None
    except (TypeError, ValueError):
        issues.append("Could not apply Setup edits; review the server log.")
        validated = None

    if issues:
        return SetupAuthoringResult(None, source_is_spec, tuple(issues))
    return SetupAuthoringResult(validated, source_is_spec, ())


def build_setup_draft(
    *,
    pipeline: SetupPathResolution,
    metadata: SetupPathResolution,
    edits: Mapping[str, Mapping[str, object]] | None = None,
    replace_scorer: bool = False,
) -> SetupDraft:
    """Build the sole revisioned Setup state from resolved controls.

    Args:
        pipeline: Precedence-resolved pipeline or tuning-spec path.
        metadata: Precedence-resolved optional metadata path.
        edits: Raw search-space editor values keyed by knob.
        replace_scorer: Whether an existing scorer should be explicitly replaced.

    Returns:
        A self-consistent draft carrying the validated spec JSON or all issues.
    """
    canonical_edits = _canonical_edits(edits)
    pipeline_path = str(pipeline.path) if pipeline.path is not None else None
    metadata_path = str(metadata.path) if metadata.path is not None else None
    source_fingerprint = path_content_fingerprint(pipeline.path)
    metadata_fingerprint = path_content_fingerprint(metadata.path)
    issues = [*pipeline.issues, *metadata.issues]
    source_is_spec = False
    scorer_name: str | None = None
    spec_json: str | None = None
    if pipeline.path is None:
        if not issues:
            issues.append("Choose a pipeline or existing tuning spec.")
    elif not pipeline.issues:
        result = build_authored_setup_spec(
            pipeline_or_spec_path=pipeline.path,
            metadata_path=metadata.path,
            edits=canonical_edits,
            replace_scorer=replace_scorer,
        )
        source_is_spec = result.source_is_spec
        issues.extend(result.issues)
        if result.is_valid and result.spec is not None and not issues:
            scorer_name = type(result.spec.scorer).__name__
            spec_json = result.spec.model_dump_json(indent=2)

    source_revision = _setup_source_revision(
        pipeline_path=pipeline_path,
        pipeline_source=pipeline.source,
        source_fingerprint=source_fingerprint,
    )
    normalized_issues = tuple(dict.fromkeys(issues))
    revision_payload = {
        "source_revision": source_revision,
        "pipeline_path": pipeline_path,
        "pipeline_source": pipeline.source,
        "metadata_path": metadata_path,
        "metadata_source": metadata.source,
        "replace_scorer": replace_scorer,
        "source_is_spec": source_is_spec,
        "edits": canonical_edits,
        "source_fingerprint": source_fingerprint,
        "metadata_fingerprint": metadata_fingerprint,
        "scorer_name": scorer_name,
        "issues": normalized_issues,
        "spec_json": spec_json,
    }
    return SetupDraft(
        revision=_content_revision(revision_payload),
        source_revision=source_revision,
        pipeline_path=pipeline_path,
        pipeline_source=pipeline.source,
        metadata_path=metadata_path,
        metadata_source=metadata.source,
        replace_scorer=replace_scorer,
        source_is_spec=source_is_spec,
        edits=canonical_edits,
        source_fingerprint=source_fingerprint,
        metadata_fingerprint=metadata_fingerprint,
        scorer_name=scorer_name,
        issues=normalized_issues,
        spec_json=spec_json,
    )


def write_setup_draft_receipt(
    *,
    sandbox_root: Path,
    draft: SetupDraft,
) -> SetupWriteReceipt:
    """Atomically write ``draft`` and return its immutable provenance.

    The source and optional metadata fingerprints are rechecked immediately
    before writing so Continue cannot publish a draft whose files changed after
    validation.
    """
    if not draft.is_valid or draft.spec_json is None or draft.pipeline_path is None:
        raise ValueError("\n".join(draft.issues) or "Setup draft is invalid.")
    sandbox = SandboxRoot.from_path(sandbox_root)
    source_path = sandbox.resolve(draft.pipeline_path)
    metadata_path = (
        sandbox.resolve(draft.metadata_path) if draft.metadata_path else None
    )
    if path_content_fingerprint(source_path) != draft.source_fingerprint:
        raise ValueError("Pipeline/spec changed after Setup validation.")
    if path_content_fingerprint(metadata_path) != draft.metadata_fingerprint:
        raise ValueError("Metadata changed after Setup validation.")
    validated = TuningSpec.model_validate_json(draft.spec_json)
    authored_content = validated.model_dump_json(indent=2)
    target = authored_setup_spec_path(
        sandbox_root=sandbox_root,
        source_path=source_path,
        metadata_path=metadata_path,
        authored_content=authored_content,
    )
    atomic_write_text(target, authored_content)
    return SetupWriteReceipt(
        path=target,
        draft_revision=draft.revision,
        source_fingerprint=draft.source_fingerprint,
        metadata_fingerprint=draft.metadata_fingerprint,
        authored_fingerprint=hashlib.sha256(
            authored_content.encode("utf-8")
        ).hexdigest(),
    )


def write_setup_draft(*, sandbox_root: Path, draft: SetupDraft) -> Path:
    """Compatibility wrapper returning only the authored spec path."""
    return write_setup_draft_receipt(
        sandbox_root=sandbox_root,
        draft=draft,
    ).path


def write_authored_setup_spec(
    *,
    sandbox_root: Path,
    pipeline_or_spec_path: Path,
    metadata_path: Path | None = None,
    edits: Mapping[str, Mapping[str, object]] | None = None,
    replace_scorer: bool = False,
    metadata_groupby: list[str] | None = None,
) -> Path:
    """Atomically validate and write one GUI-authored tuning spec."""
    if not pipeline_or_spec_path.is_file():
        raise FileNotFoundError(pipeline_or_spec_path)
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    result = build_authored_setup_spec(
        pipeline_or_spec_path=pipeline_or_spec_path,
        metadata_path=metadata_path,
        edits=edits,
        replace_scorer=replace_scorer,
        metadata_groupby=metadata_groupby,
    )
    if not result.is_valid or result.spec is None:
        raise ValueError("\n".join(result.issues))
    authored_content = result.spec.model_dump_json(indent=2)
    target = authored_setup_spec_path(
        sandbox_root=sandbox_root,
        source_path=pipeline_or_spec_path,
        metadata_path=metadata_path,
        authored_content=authored_content,
    )
    atomic_write_text(target, authored_content)
    return target


__all__ = [
    "Issue",
    "PreparedPipelineExport",
    "ValidatedTuneCommand",
    "apply_space_edits",
    "build_tune_command",
    "can_deploy",
    "export_best_from_run",
    "export_pareto_pipeline",
    "export_winning_pipeline",
    "prepare_best_from_run",
    "publish_prepared_export",
    "preflight_issues",
    "render_launch_command",
    "render_tokens",
    "space_to_spec",
    "spec_path_issue",
    "storage_url_preflight_issue",
    "validate_setup",
]
