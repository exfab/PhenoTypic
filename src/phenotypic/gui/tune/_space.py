"""Space view — infer a search space, render knob forms, export a spec (C2).

The Space view turns a configured pipeline (or an existing tuning run) into an
editable search space and writes the result back to ``tuning_spec.json``. Three
pure pieces back it (all unit-tested headless, all optuna-free):

* :func:`_knob_form` — render one inferred :class:`~phenotypic.tune.Knob` as a
  ``dbc.Row`` editor. ``FloatRange`` → two numeric inputs (low / high) + a log
  switch; ``IntRange`` → two integer inputs (+ log); ``Categorical`` → a
  checklist of the choices. Every editable knob also carries a per-knob
  ``tunable`` switch. A depth-1 ``Nested`` knob renders **read-only / disabled**
  (nested leaves are surfaced but not v1-editable).
* :func:`space_to_spec` — the OQ8 config-preserving builder. From an existing
  :class:`~phenotypic.tune.TuningSpec` it replaces **only** ``search_space`` and
  keeps the run's scorer / strategy / budget / evaluator / held-out policy. From
  a bare ``ImagePipeline`` it defaults the scorer (an unconfigured
  :class:`~phenotypic.tune.QCScorer` — the "review in Launch" signal), strategy,
  and budget.
* :func:`build_space_view` — the Dash view body: one :func:`_knob_form` row per
  inferred flat / presence knob, the disabled nested rows, and the Export button.

The editable knobs are the flat (:class:`~phenotypic.tune.targets.Param`) and
presence (:class:`~phenotypic.tune.targets.Presence`) targets; nested
(:class:`~phenotypic.tune.targets.Nested`) leaves are shown read-only and are
**dropped** from the exported space (v1 tunes flat + presence only). Like the
rest of :mod:`phenotypic.gui.tune`, importing this module must never drag
``optuna`` into ``sys.modules``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html

from phenotypic.gui.tune import _ids as ids

if TYPE_CHECKING:
    from phenotypic import ImagePipeline
    from phenotypic.gui.tune._run_root import TuneRunRoot  # noqa: F401
    from phenotypic.tune import InferredSearchSpace, Knob, SearchSpace, TuningSpec


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
    from phenotypic.tune import QCScorer

    empty = pd.DataFrame({"Metadata_ImageName": [], "Object_Label": []})
    return QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=empty, groupby=["Metadata_ImageName"]
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
    :class:`~phenotypic.tune.QCScorer` — :func:`_default_qc_scorer`), a
    :class:`~phenotypic.tune.GridConfig` strategy, a default
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
        GridConfig,
        TuningSpec,
    )

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


# ---------------------------------------------------------------------------
# Knob forms (Dash)
# ---------------------------------------------------------------------------

#: The "review this guess" badge text for an inference-flagged knob.
_REVIEW_BADGE: str = "review"


def _tunable_toggle(knob: "Knob", *, disabled: bool) -> dbc.Checklist:
    """A per-knob on/off ``tunable`` switch (on by default, off when disabled)."""
    return dbc.Checklist(
        id={"type": ids.TUNE_SPACE_TUNABLE, "key": knob.key},
        options=[{"label": "tunable", "value": "on", "disabled": disabled}],
        value=[] if disabled else ["on"],
        switch=True,
        className="tune-space-tunable",
    )


def _range_inputs(knob: "Knob", *, is_int: bool, disabled: bool) -> list[Any]:
    """Two numeric low/high inputs + a log switch for a range domain.

    The caller has already dispatched on ``knob.domain.kind`` being a range, so
    ``low``/``high``/``log`` are read via ``getattr`` (the ``Domain`` union does
    not statically expose them — they live on the range subtypes only).
    """
    step = 1 if is_int else "any"
    return [
        html.Label("low", className="tune-space-bound-label"),
        dbc.Input(
            id={"type": ids.TUNE_SPACE_LOW, "key": knob.key},
            type="number",
            value=getattr(knob.domain, "low", None),
            step=step,
            disabled=disabled,
            className="tune-space-bound",
        ),
        html.Label("high", className="tune-space-bound-label"),
        dbc.Input(
            id={"type": ids.TUNE_SPACE_HIGH, "key": knob.key},
            type="number",
            value=getattr(knob.domain, "high", None),
            step=step,
            disabled=disabled,
            className="tune-space-bound",
        ),
        dbc.Checklist(
            id={"type": ids.TUNE_SPACE_LOG, "key": knob.key},
            options=[{"label": "log", "value": "on", "disabled": disabled}],
            value=["on"] if getattr(knob.domain, "log", False) else [],
            switch=True,
            className="tune-space-log",
        ),
    ]


def _categorical_input(knob: "Knob", *, disabled: bool) -> list[Any]:
    """A checklist of the categorical choices (all pre-checked)."""
    choices = list(getattr(knob.domain, "choices", ()))
    return [
        dbc.Checklist(
            id={"type": ids.TUNE_SPACE_CHOICES, "key": knob.key},
            options=[
                {"label": str(choice), "value": str(choice), "disabled": disabled}
                for choice in choices
            ],
            value=[str(choice) for choice in choices],
            className="tune-space-choices",
        )
    ]


def _domain_editor(knob: "Knob", *, disabled: bool) -> list[Any]:
    """Dispatch a knob's domain to its editor widgets (range / categorical)."""
    kind = knob.domain.kind
    if kind == "float_range":
        return _range_inputs(knob, is_int=False, disabled=disabled)
    if kind == "int_range":
        return _range_inputs(knob, is_int=True, disabled=disabled)
    if kind == "categorical":
        return _categorical_input(knob, disabled=disabled)
    # ``fixed`` (and any future domain) renders read-only — nothing to tune.
    return [html.Span(str(getattr(knob.domain, "value", "")), className="tune-space-fixed")]


def _knob_form(knob: "Knob") -> dbc.Row:
    """Render one inferred ``Knob`` as a ``dbc.Row`` editor.

    A flat / presence knob is fully editable: the domain editor
    (:func:`_domain_editor`) plus a per-knob ``tunable`` switch. A depth-1
    ``Nested`` knob is **read-only / disabled** — its leaf is surfaced for
    visibility but v1 does not tune nested ops, so every input carries
    ``disabled=True`` and the knob is dropped from the exported space.

    Args:
        knob: The knob to render.

    Returns:
        A :class:`dbc.Row` carrying the knob's key, domain editor, and toggles.
    """
    disabled = type(knob.target).__name__ == "Nested"
    label_children: list[Any] = [
        html.Span(knob.key, className="tune-space-key"),
    ]
    if knob.needs_review:
        label_children.append(
            dbc.Badge(_REVIEW_BADGE, color="warning", className="tune-space-review")
        )
    if disabled:
        label_children.append(
            html.Span("nested (read-only)", className="tune-space-nested-note")
        )
    cells: list[Any] = [
        dbc.Col(label_children, className="tune-space-key-col"),
        dbc.Col(_domain_editor(knob, disabled=disabled), className="tune-space-domain-col"),
        dbc.Col(_tunable_toggle(knob, disabled=disabled), className="tune-space-toggle-col"),
    ]
    return dbc.Row(
        cells,
        id={"type": ids.TUNE_SPACE_KNOB_ROW, "key": knob.key},
        className="tune-space-knob-row",
    )


__all__ = ["space_to_spec", "build_space_view"]


def build_space_view(root: "TuneRunRoot") -> html.Div:
    """Render the Space view body for the bound run ``root``.

    Loads the run's base pipeline (from its ``tuning_spec.json`` when present,
    else its ``pipeline.json``), infers the search space, and renders one
    :func:`_knob_form` per inferred knob (flat / presence editable; nested
    read-only), plus the Export button and a status note. When neither a spec nor
    a pipeline can be loaded, a short prompt is shown instead of the form.

    Args:
        root: The validated tune output handle.

    Returns:
        The Space view body.
    """
    source = _load_space_source(root)
    if source is None:
        return html.Div(
            html.P(
                "No pipeline found for this run: the Space view needs a "
                "deliverables/tuning_spec.json or deliverables/pipeline.json to "
                "infer a search space from."
            ),
            className="tune-space tune-space-empty",
        )

    from phenotypic.tune import infer_search_space

    # ``source`` is a TuningSpec or a bare pipeline (duck-typed); the spec carries
    # the base pipeline on ``.pipeline``, a bare pipeline is itself the base.
    pipeline = getattr(source, "pipeline", source)
    inferred = infer_search_space(pipeline)
    rows = [_knob_form(knob) for knob in inferred.knobs]

    note = ""
    if not _is_tuning_spec(source):
        note = (
            "Fresh search space from pipeline.json -- the scorer, strategy, and "
            "budget will be defaulted; review them in Launch before tuning."
        )

    return html.Div(
        [
            html.P(
                "Edit the inferred search space, then export it to "
                "deliverables/tuning_spec.json.",
                className="tune-space-intro",
            ),
            html.Div(rows, className="tune-space-knobs"),
            html.Div(
                [
                    dbc.Button(
                        "Export tuning_spec.json",
                        id=ids.TUNE_BTN_SPACE_EXPORT,
                        color="primary",
                        n_clicks=0,
                    ),
                    html.Span(
                        note,
                        id=ids.TUNE_SPACE_NOTE,
                        className="tune-space-note",
                    ),
                ],
                className="tune-space-export-bar",
            ),
        ],
        className="tune-space",
    )


def _load_space_source(
    root: "TuneRunRoot",
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
