"""Session-scoped plotting controls for filter / model section cards.

Analyzer visualization methods (``show`` / ``dash``) take *plotting*
parameters — ``figsize``, ``collapsed``, ``cmap``, ``legend``,
``max_groups``, ``tmax`` — that are plain method arguments, **not**
pydantic fields. They therefore have no schema and never serialize into
``pipeline.json``.

This module introspects whichever visualization method
:func:`._render.render_plot` will actually call and builds a small
``html.Details`` disclosure of widgets for the explicitly-named params.
The widget values live in a session-scoped ``dcc.Store`` (see
:data:`._ids.ANALYSIS_PLOT_PREFS_STORE`) keyed ``f"{kind}-{index}-{name}"``.

Only the *explicitly-named* signature parameters are surfaced. Documented
``**kwargs`` extras (``dpi``, ``facecolor``, ``legend_fontsize`` …) are not
in the signature and stay power-user-only — call ``.show()`` directly.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.analysis.abc_ import SetAnalyzer
from phenotypic.gui._design import COLOR_MUTED
from phenotypic.gui.analysis import _ids as ids

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

#: Method parameters never surfaced as widgets — ``ax`` is an internal
#: matplotlib hand-off and ``criteria`` is a nested filter dict with no
#: sensible scalar widget.
_EXCLUDED_PARAMS: frozenset[str] = frozenset({"ax", "criteria"})

#: Widget dtypes a :class:`PlotParamSpec` can carry.
PlotDType = str  # one of: "bool", "number", "tuple", "str"


@dataclass(frozen=True)
class PlotParamSpec:
    """One introspected plotting parameter.

    Attributes:
        name: The method-argument name (e.g. ``"figsize"``).
        dtype: Widget dtype — ``"bool"`` / ``"number"`` / ``"tuple"`` /
            ``"str"``.
        default: The parameter's signature default (``None`` when the
            signature has no default).
    """

    name: str
    dtype: PlotDType
    default: Any


def _viz_method(node: Any) -> Any:
    """Return the bound visualization method ``render_plot`` will call.

    ``render_plot`` tries ``dash`` first and falls back to ``show`` on
    ``NotImplementedError``. A subclass that leaves ``dash`` inherited
    from :class:`SetAnalyzer` raises that error, so ``show`` is what
    actually runs; a subclass that overrides ``dash`` (every
    ``ModelFitter``) uses ``dash``.
    """
    overrides_dash = type(node).dash is not SetAnalyzer.dash
    return node.dash if overrides_dash else node.show


def _classify(default: Any, annotation: Any) -> PlotDType:
    """Map a parameter's default / annotation to a widget dtype."""
    # The default value is the strongest signal when present.
    if isinstance(default, tuple):
        return "tuple"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (int, float)):
        return "number"
    if isinstance(default, str):
        return "str"
    # Default is None / empty — fall back to the annotation text.
    ann = "" if annotation is inspect.Parameter.empty else str(annotation).lower()
    if "tuple" in ann:
        return "tuple"
    if "bool" in ann:
        return "bool"
    if "int" in ann or "float" in ann:
        return "number"
    return "str"


def plotting_params(node: Any) -> list[PlotParamSpec]:
    """Introspect *node*'s active visualization method into widget specs.

    Args:
        node: A :class:`SetAnalyzer` (filter) or ``ModelFitter`` (model)
            instance.

    Returns:
        One :class:`PlotParamSpec` per explicitly-named parameter of the
        method :func:`._render.render_plot` will call, excluding
        ``self``, ``*args``, ``**kwargs`` and :data:`_EXCLUDED_PARAMS`.
    """
    try:
        sig = inspect.signature(_viz_method(node))
    except (ValueError, TypeError):  # pragma: no cover - defensive
        logger.warning("Could not introspect viz method on %s", type(node).__name__)
        return []

    specs: list[PlotParamSpec] = []
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.name in _EXCLUDED_PARAMS:
            continue
        default = None if param.default is inspect.Parameter.empty else param.default
        specs.append(
            PlotParamSpec(
                name=param.name,
                dtype=_classify(default, param.annotation),
                default=default,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Store-key helpers
# ---------------------------------------------------------------------------

def _store_key(kind: str, index: int, name: str) -> str:
    """Flat key under which one widget's value lives in the prefs store."""
    return f"{kind}-{index}-{name}"


def collect_plot_kwargs(
    kind: str,
    index: int,
    node: Any,
    prefs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble ``render_plot`` kwargs for one section from the prefs store.

    Only params the user has actually set a meaningful value for are
    returned; everything else is omitted so the analyzer falls back to
    its own signature default. ``tuple`` params are reassembled from
    their two ``"__0"`` / ``"__1"`` axis widgets and dropped unless both
    axes carry a value.

    Args:
        kind: Section kind (``"filter"`` / ``"model"``).
        index: Section index within its stack (always ``0`` for models).
        node: The analyzer instance — used only to introspect specs.
        prefs: The raw session-store dict, or ``None``.

    Returns:
        A kwargs dict safe to splat into :func:`._render.render_plot`.
    """
    prefs = prefs or {}
    kwargs: dict[str, Any] = {}
    for spec in plotting_params(node):
        if spec.dtype == "tuple":
            raw0 = prefs.get(_store_key(kind, index, f"{spec.name}__0"))
            raw1 = prefs.get(_store_key(kind, index, f"{spec.name}__1"))
            if raw0 in (None, "") or raw1 in (None, ""):
                continue
            kwargs[spec.name] = (raw0, raw1)
            continue

        raw = prefs.get(_store_key(kind, index, spec.name))
        if raw is None or raw == "":
            continue
        if spec.dtype == "bool":
            kwargs[spec.name] = bool(raw)
        else:  # "number" / "str" — dcc.Input already typed these
            kwargs[spec.name] = raw
    return kwargs


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _widget_for(
    spec: PlotParamSpec,
    kind: "ids.PlotSectionKind",
    index: int,
    prefs: dict[str, Any],
) -> Any:
    """Build the labelled widget(s) for one :class:`PlotParamSpec`."""
    label = html.Label(
        spec.name,
        className="analysis-plot-param-label",
        style={"color": COLOR_MUTED, "marginRight": "0.5rem"},
    )

    if spec.dtype == "tuple":
        seed = spec.default if isinstance(spec.default, tuple) else (None, None)
        inputs = [
            dcc.Input(
                id=ids.plot_param_id(kind, index, f"{spec.name}__{axis}"),
                type="number",
                value=prefs.get(
                    _store_key(kind, index, f"{spec.name}__{axis}"),
                    seed[axis] if axis < len(seed) else None,
                ),
                debounce=True,
                style={"width": "70px", "marginRight": "0.25rem"},
            )
            for axis in (0, 1)
        ]
        return html.Div([label, *inputs], className="analysis-plot-param")

    seeded = prefs.get(_store_key(kind, index, spec.name), spec.default)
    if spec.dtype == "bool":
        widget: Any = dbc.Switch(
            id=ids.plot_param_id(kind, index, spec.name),
            value=bool(seeded),
        )
    elif spec.dtype == "number":
        widget = dcc.Input(
            id=ids.plot_param_id(kind, index, spec.name),
            type="number",
            value=seeded,
            debounce=True,
            style={"width": "90px"},
        )
    else:  # "str"
        widget = dcc.Input(
            id=ids.plot_param_id(kind, index, spec.name),
            type="text",
            value=seeded if isinstance(seeded, str) else "",
            debounce=True,
            style={"width": "120px"},
        )
    return html.Div([label, widget], className="analysis-plot-param")


def plot_controls_form(
    kind: "ids.PlotSectionKind",
    index: int,
    node: Any,
    prefs: dict[str, Any] | None = None,
) -> Any:
    """Build the Display-settings disclosure + Preview button + plot slot.

    Args:
        kind: Section kind (``"filter"`` / ``"model"``).
        index: Section index within its stack (``0`` for the model).
        node: The analyzer instance whose viz method is introspected.
        prefs: Current session-store dict; widget values re-seed from it
            so they survive section-stack rebuilds.

    Returns:
        An ``html.Div`` ready to append to a section card body.
    """
    prefs = prefs or {}
    specs = plotting_params(node)

    disclosure_body = (
        [_widget_for(spec, kind, index, prefs) for spec in specs]
        if specs
        else [html.Em("No plotting parameters.", style={"color": COLOR_MUTED})]
    )

    return html.Div(
        [
            html.Details(
                [
                    html.Summary(
                        "Display settings",
                        className="analysis-plot-controls-summary",
                    ),
                    html.Div(
                        disclosure_body,
                        className="analysis-plot-controls-body",
                    ),
                ],
                className="analysis-plot-controls-disclosure",
            ),
            dbc.Button(
                "Preview",
                id=ids.preview_button_id(kind, index),
                color="secondary",
                size="sm",
                n_clicks=0,
                className="analysis-preview-button",
                style={"marginTop": "0.5rem"},
            ),
            dcc.Loading(
                html.Div(
                    id=ids.plot_slot_id(kind, index),
                    className="analysis-plot-slot",
                ),
                type="default",
            ),
        ],
        className="analysis-plot-controls",
    )


__all__ = [
    "PlotParamSpec",
    "plotting_params",
    "collect_plot_kwargs",
    "plot_controls_form",
]
