"""Space view — render the inferred search space as editable Dash knob forms.

This is the Dash half of what used to be ``gui/tune/_space.py``. Everything that
computes a search space lives in :mod:`phenotypic._services.tune_spec`, which
this module imports; the dependency never runs the other way, so the MCP server
can build the same specs without Dash on its import path.

* :func:`_knob_form` — render one inferred :class:`~phenotypic.tune.Knob` as a
  ``dbc.Row`` editor. ``FloatRange`` → two numeric inputs (low / high) + a log
  switch; ``IntRange`` → two integer inputs (+ log); ``Categorical`` → a
  checklist of the choices. Every editable knob also carries a per-knob
  ``tunable`` switch. A depth-1 ``Nested`` knob renders **read-only / disabled**
  (nested leaves are surfaced but not v1-editable).
* :func:`setup_knob_forms` — the Setup surface's rows, preserving an authored
  spec's configured space.
* :func:`build_space_view` — the Space view body: one :func:`_knob_form` row per
  inferred flat / presence knob, the disabled nested rows, and the Export button.

Nested leaves are shown read-only here and are **dropped** from the exported
space by :func:`~phenotypic._services.tune_spec._build_search_space` (v1 tunes
flat + presence only). Like the rest of :mod:`phenotypic.gui.tune`, importing
this module must never drag ``optuna`` into ``sys.modules``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html

from phenotypic._services.tune_spec import _is_tuning_spec, _load_space_source
from phenotypic.gui.tune import _ids as ids

if TYPE_CHECKING:
    from phenotypic.gui.tune._run_root import TuneRunRoot  # noqa: F401
    from phenotypic.tune import Knob


#: The "review this guess" badge text for an inference-flagged knob.
_REVIEW_BADGE: str = "review"


def _tunable_toggle(
    knob: "Knob",
    *,
    disabled: bool,
    selected_when_disabled: bool = False,
    component_type: str = ids.TUNE_SPACE_TUNABLE,
) -> dbc.Checklist:
    """A per-knob on/off ``tunable`` switch (on by default, off when disabled)."""
    return dbc.Checklist(
        id={"type": component_type, "key": knob.key},
        options=[{"label": "tunable", "value": "on", "disabled": disabled}],
        value=["on"] if not disabled or selected_when_disabled else [],
        switch=True,
        className="tune-space-tunable",
    )


def _range_inputs(
    knob: "Knob",
    *,
    is_int: bool,
    disabled: bool,
    low_type: str = ids.TUNE_SPACE_LOW,
    high_type: str = ids.TUNE_SPACE_HIGH,
    log_type: str = ids.TUNE_SPACE_LOG,
) -> list[Any]:
    """Two numeric low/high inputs + a log switch for a range domain.

    The caller has already dispatched on ``knob.domain.kind`` being a range, so
    ``low``/``high``/``log`` are read via ``getattr`` (the ``Domain`` union does
    not statically expose them — they live on the range subtypes only).
    """
    step = 1 if is_int else "any"
    return [
        html.Label("low", className="tune-space-bound-label"),
        dbc.Input(
            id={"type": low_type, "key": knob.key},
            type="number",
            value=getattr(knob.domain, "low", None),
            step=step,
            disabled=disabled,
            className="tune-space-bound",
        ),
        html.Label("high", className="tune-space-bound-label"),
        dbc.Input(
            id={"type": high_type, "key": knob.key},
            type="number",
            value=getattr(knob.domain, "high", None),
            step=step,
            disabled=disabled,
            className="tune-space-bound",
        ),
        dbc.Checklist(
            id={"type": log_type, "key": knob.key},
            options=[{"label": "log", "value": "on", "disabled": disabled}],
            value=["on"] if getattr(knob.domain, "log", False) else [],
            switch=True,
            className="tune-space-log",
        ),
    ]


def _categorical_input(
    knob: "Knob",
    *,
    disabled: bool,
    component_type: str = ids.TUNE_SPACE_CHOICES,
) -> list[Any]:
    """A checklist of the categorical choices (all pre-checked)."""
    choices = list(getattr(knob.domain, "choices", ()))
    return [
        dbc.Checklist(
            id={"type": component_type, "key": knob.key},
            options=[
                {"label": str(choice), "value": str(choice), "disabled": disabled}
                for choice in choices
            ],
            value=[str(choice) for choice in choices],
            className="tune-space-choices",
        )
    ]


def _domain_editor(
    knob: "Knob",
    *,
    disabled: bool,
    low_type: str = ids.TUNE_SPACE_LOW,
    high_type: str = ids.TUNE_SPACE_HIGH,
    log_type: str = ids.TUNE_SPACE_LOG,
    choices_type: str = ids.TUNE_SPACE_CHOICES,
) -> list[Any]:
    """Dispatch a knob's domain to its editor widgets (range / categorical)."""
    kind = knob.domain.kind
    if kind == "float_range":
        return _range_inputs(
            knob,
            is_int=False,
            disabled=disabled,
            low_type=low_type,
            high_type=high_type,
            log_type=log_type,
        )
    if kind == "int_range":
        return _range_inputs(
            knob,
            is_int=True,
            disabled=disabled,
            low_type=low_type,
            high_type=high_type,
            log_type=log_type,
        )
    if kind == "categorical":
        return _categorical_input(
            knob,
            disabled=disabled,
            component_type=choices_type,
        )
    # ``fixed`` (and any future domain) renders read-only — nothing to tune.
    return [html.Span(str(getattr(knob.domain, "value", "")), className="tune-space-fixed")]


def _knob_form(
    knob: "Knob",
    *,
    setup: bool = False,
    preserve_disabled: bool = False,
) -> dbc.Row:
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
    low_type = ids.TUNE_SETUP_SPACE_LOW if setup else ids.TUNE_SPACE_LOW
    high_type = ids.TUNE_SETUP_SPACE_HIGH if setup else ids.TUNE_SPACE_HIGH
    log_type = ids.TUNE_SETUP_SPACE_LOG if setup else ids.TUNE_SPACE_LOG
    choices_type = (
        ids.TUNE_SETUP_SPACE_CHOICES if setup else ids.TUNE_SPACE_CHOICES
    )
    tunable_type = (
        ids.TUNE_SETUP_SPACE_TUNABLE if setup else ids.TUNE_SPACE_TUNABLE
    )
    row_type = ids.TUNE_SETUP_SPACE_KNOB_ROW if setup else ids.TUNE_SPACE_KNOB_ROW
    cells: list[Any] = [
        dbc.Col(label_children, className="tune-space-key-col"),
        dbc.Col(
            _domain_editor(
                knob,
                disabled=disabled,
                low_type=low_type,
                high_type=high_type,
                log_type=log_type,
                choices_type=choices_type,
            ),
            className="tune-space-domain-col",
        ),
        dbc.Col(
            _tunable_toggle(
                knob,
                disabled=disabled,
                selected_when_disabled=preserve_disabled,
                component_type=tunable_type,
            ),
            className="tune-space-toggle-col",
        ),
    ]
    return dbc.Row(
        cells,
        id={"type": row_type, "key": knob.key},
        className="tune-space-knob-row",
    )


def setup_knob_forms(source: Any) -> list[dbc.Row]:
    """Render editable Setup rows while preserving an existing spec's space."""
    from phenotypic.tune import infer_search_space

    if _is_tuning_spec(source):
        configured = list(source.search_space.knobs)
        existing_keys = {knob.key for knob in configured}
        inferred = infer_search_space(source.pipeline)
        inferred_nested = [
            knob
            for knob in inferred.knobs
            if type(knob.target).__name__ == "Nested"
            and knob.key not in existing_keys
        ]
        return [
            *[
                _knob_form(knob, setup=True, preserve_disabled=True)
                for knob in configured
            ],
            *[_knob_form(knob, setup=True) for knob in inferred_nested],
        ]
    return [
        _knob_form(knob, setup=True)
        for knob in infer_search_space(source).knobs
    ]


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


__all__ = [
    "build_space_view",
    "setup_knob_forms",
]
