"""Dash-free tuning-spec authoring: infer a space, edit it, build a ``TuningSpec``.

This is the pure half of what used to be ``gui/tune/_space.py``. It carries the
three functions the Space view is built on plus the loaders that feed them:

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

Nothing here may import Dash, and nothing here may import a rendering surface:
the split exists so both the Space view and the MCP server can call one tested
implementation. The Dash rendering lives in
:mod:`phenotypic.gui.tune._space_view`, which imports *this* module. Like the
rest of the tune tier, importing this module must never drag ``optuna`` into
``sys.modules``.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Union,
)

if TYPE_CHECKING:
    from pathlib import Path

    from phenotypic import ImagePipeline
    from phenotypic.tune import InferredSearchSpace, Knob, SearchSpace, TuningSpec


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
    from phenotypic.tune import (
    Budget,
    Evaluator,
    TuningSpec,
)
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
    from phenotypic.sdk_ import resolve_pipeline_config_path, resolve_tuning_spec_path

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
    from phenotypic.tune import TuningSpec

    try:
        return TuningSpec.model_validate_json(spec_path.read_text())
    except Exception:  # noqa: BLE001 - a proposal / corrupt file degrades to None
        return None


def _try_load_pipeline(pipe_path: Any) -> "Optional[ImagePipeline]":
    """Load an ``ImagePipeline`` from ``pipe_path``, or ``None`` on any failure."""
    from phenotypic import ImagePipeline

    try:
        return ImagePipeline.from_json(pipe_path.read_text())
    except Exception:  # noqa: BLE001 - a corrupt pipeline.json degrades to None
        return None


__all__ = [
    "apply_space_edits",
    "space_to_spec",
]
