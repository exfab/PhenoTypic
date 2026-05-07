"""Layout builders for the analysis sub-app.

The page is a vertical stepper:

1. Output-root header (path + reload).
2. Pipeline summary chip.
3. Recompile banner (post edits need a CLI re-run to land in master).
4. Post section stack — table preview UX (col-name + top-5 before/after).
5. Filter section stack — plot preview UX (autodetect dash vs PNG).
6. Model section (single) — plot preview UX.
7. Sticky run console (button + spinner + status).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._design import (
    COLOR_BLUE,
    COLOR_GOLD,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    COLOR_WHITE,
)
from phenotypic.gui._operation_registry import OperationRegistry, get_registry
from phenotypic.gui._param_forms import param_form
from phenotypic.gui.analysis import _ids as ids

if TYPE_CHECKING:
    from phenotypic.gui.analysis._recipe_state import RecipeState
    from phenotypic.gui.results_viewer._output_root import OutputRoot

# Class names for analysis filter / model registries (v1 fixed list).
_FILTER_CHOICES = ["EdgeCorrector", "TukeyOutlierRemover"]
_MODEL_CHOICES = ["LogGrowthModel", "LinearSoftplusModel"]
_POST_CHOICES = ["PrependString", "AppendString", "ExpandMetadata", "MergeMetadata"]


def build_app_layout(
    output_root: "OutputRoot",
    recipe: "RecipeState",
) -> html.Div:
    """Assemble the analysis page body.

    Args:
        output_root: Validated CLI output root (provides
            ``master_measurements.parquet`` and the curated
            ``measurements.parquet``).
        recipe: Loaded :class:`RecipeState` for ``<output>/pipeline.json``.

    Returns:
        Top-level ``html.Div`` ready to drop into the shell's main pane.
    """
    return html.Div(
        [
            _build_output_header(output_root),
            _build_pipeline_header(recipe),
            _build_stale_banner(),
            _build_recompile_banner(),
            _build_post_panel(recipe),
            _build_filter_panel(recipe),
            _build_model_panel(recipe),
            _build_run_console(recipe),
            dcc.Store(
                id=ids.ANALYSIS_PIPELINE_STORE,
                data=recipe.last_json or recipe.pipeline.to_json() or "{}",
            ),
        ],
        id=ids.ANALYSIS_PAGE,
        className="analysis-page",
    )


def build_empty_state_layout() -> html.Div:
    """Layout shown when the hub mounts ``/analysis/`` without an output root.

    Mirrors the results viewer's empty-state hand-off banner: the user
    picks a CLI output entry in the sidebar, the banner fills in with
    the selection, and clicking ↩ Open in analysis POSTs to the shared
    ``/sandbox/api/viewer/output-root`` endpoint, which releases both
    the viewer and the analysis ToolSession so the next request to
    ``/analysis/`` rebuilds against the bound output root.
    """
    handoff_banner = html.Div(
        [
            html.Span(
                "Selected: ",
                className="analysis-empty-handoff-prefix",
            ),
            html.Code(
                "(none)",
                id=ids.EMPTY_HANDOFF_LABEL,
                className="analysis-empty-handoff-label",
            ),
            dbc.Button(
                "↩ Open in analysis",
                id=ids.EMPTY_HANDOFF_OPEN_BUTTON,
                color="primary",
                size="sm",
                disabled=True,
                className="analysis-empty-handoff-open ms-2",
                n_clicks=0,
            ),
        ],
        id=ids.EMPTY_HANDOFF_BANNER,
        className="analysis-empty-handoff-banner",
        style={
            "display": "none",
            "alignItems": "center",
            "gap": "0.5rem",
            "marginTop": "1rem",
            "padding": "0.5rem 0.75rem",
            "background": COLOR_SURFACE,
            "border": f"1px solid {COLOR_BLUE}",
            "borderRadius": "6px",
        },
    )

    error_slot = html.Div(
        "",
        id=ids.EMPTY_HANDOFF_ERROR,
        className="analysis-empty-handoff-error text-danger small",
        style={"marginTop": "0.5rem", "minHeight": "1.25rem"},
    )

    return html.Div(
        [
            html.Div(
                [
                    html.H2(
                        "No output selected",
                        style={"color": COLOR_NAVY},
                    ),
                    html.P(
                        "Pick a CLI output directory in the sidebar, "
                        "then click ↩ Open in analysis to bind it. The "
                        "binding is shared with the results viewer — "
                        "both tools rebuild against the chosen output "
                        "in lock-step.",
                    ),
                    handoff_banner,
                    error_slot,
                ],
            ),
        ],
        id=ids.ANALYSIS_PAGE,
        className="analysis-page analysis-empty",
        style={"padding": "2rem"},
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_output_header(output_root: "OutputRoot") -> html.Div:
    return html.Div(
        [
            html.Strong("Output: "),
            html.Code(str(output_root.root)),
        ],
        id=ids.ANALYSIS_OUTPUT_HEADER,
        className="analysis-output-header",
        style={
            "padding": "0.5rem 1rem",
            "background": COLOR_SURFACE,
            "borderBottom": "1px solid #ddd",
        },
    )


def pipeline_header_children(recipe: "RecipeState") -> list:
    """Build the inner children of the pipeline-header bar.

    Shared by :func:`_build_pipeline_header` (initial render) and the
    callback that refreshes the header on every recipe mutation, so the
    summary string lives in one place.
    """
    pipeline = recipe.pipeline
    n_post = len(pipeline.get_post())
    n_filters = len(pipeline.get_filters())
    model = pipeline.get_model()
    model_name = type(model).__name__ if model is not None else "—"
    summary = (
        f"{len(pipeline.get_ops())} ops · {len(pipeline.get_meas())} meas · "
        f"{n_post} post · {n_filters} filters · model: {model_name}"
    )
    return [
        html.Strong(f"Pipeline: {pipeline.name}"),
        html.Span(
            summary,
            className="analysis-pipeline-summary",
            style={"marginLeft": "1rem", "color": COLOR_MUTED},
        ),
    ]


def _build_pipeline_header(recipe: "RecipeState") -> html.Div:
    return html.Div(
        pipeline_header_children(recipe),
        id=ids.ANALYSIS_PIPELINE_HEADER,
        className="analysis-pipeline-header",
        style={"padding": "0.5rem 1rem"},
    )


def _build_recompile_banner() -> html.Div:
    return html.Div(
        [
            html.Span("ℹ "),
            "Post edits change per-image measurement. Re-run the CLI "
            "(",
            html.Code("python -m phenotypic --recompile <output>"),
            ") to apply post changes to ",
            html.Code("master_measurements.parquet"),
            ".",
        ],
        id=ids.ANALYSIS_RECOMPILE_BANNER,
        className="analysis-recompile-banner",
        style={
            "padding": "0.5rem 1rem",
            "background": "#fff8e1",
            "borderLeft": f"4px solid {COLOR_GOLD}",
            "margin": "0.5rem 1rem",
        },
    )


def _build_stale_banner() -> html.Div:
    return html.Div(
        id=ids.ANALYSIS_STALE_BANNER,
        className="analysis-stale-banner",
        style={"display": "none"},
    )


def _build_post_panel(recipe: "RecipeState") -> html.Div:
    return _build_section_panel(
        title="Post operations (metadata transforms)",
        section_label="post",
        choices=_POST_CHOICES,
        add_dropdown_id=ids.ANALYSIS_POST_ADD_DROPDOWN,
        stack_id=ids.ANALYSIS_POST_STACK,
        recipe=recipe,
    )


def _build_filter_panel(recipe: "RecipeState") -> html.Div:
    return _build_section_panel(
        title="Filters",
        section_label="filter",
        choices=_FILTER_CHOICES,
        add_dropdown_id=ids.ANALYSIS_FILTER_ADD_DROPDOWN,
        stack_id=ids.ANALYSIS_FILTER_STACK,
        recipe=recipe,
    )


def _build_section_panel(
    *,
    title: str,
    section_label: "ids.SectionKind",
    choices: list[str],
    add_dropdown_id: str,
    stack_id: str,
    recipe: "RecipeState",
) -> html.Div:
    return html.Div(
        [
            html.H3(title),
            html.Div(
                build_section_stack(stack_id, section_label, recipe),
                id=stack_id,
                className=f"analysis-{section_label}-stack",
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id=add_dropdown_id,
                        options=[{"label": c, "value": c} for c in choices],
                        placeholder=f"Add {section_label}…",
                        style={"width": "260px", "display": "inline-block"},
                    ),
                ],
                style={"margin": "0.5rem 0"},
            ),
        ],
        style={"padding": "0.5rem 1rem"},
    )


def build_section_stack(
    stack_id: str,
    kind: "ids.SectionKind",
    recipe: "RecipeState",
    registry: OperationRegistry | None = None,
) -> list:
    """Build the list of section cards inside a stack.

    Each card now hosts a fully editable :func:`param_form` rendered
    against the section's analyzer instance. Widget ids are scoped via
    ``form_id_prefix=f"analysis-{kind}-{index}"`` so the analysis-side
    pattern-matching callback can map any param edit back to the correct
    section without colliding with the builder's prefixes.
    """
    pipeline = recipe.pipeline
    items: list[tuple[str, Any]]
    if kind == "post":
        items = list(pipeline.get_post().items())
    elif kind == "filter":
        items = list(pipeline.get_filters().items())
    else:
        return []

    if registry is None:
        registry = get_registry()

    cards: list = []
    for index, (name, instance) in enumerate(items):
        info = registry.get(type(instance).__name__)
        body = (
            _section_form(info, instance, kind=kind, index=index)
            if info is not None
            else html.Em(
                f"No registry info for {type(instance).__name__}",
                style={"color": COLOR_MUTED},
            )
        )
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(name),
                            html.Button(
                                "×",
                                id=ids.section_remove_button_id(kind, index),
                                n_clicks=0,
                                className="analysis-section-remove",
                                style={
                                    "float": "right",
                                    "border": "none",
                                    "background": "transparent",
                                    "fontSize": "1.2rem",
                                    "cursor": "pointer",
                                },
                                title=f"Remove {name}",
                            ),
                        ],
                        className=f"analysis-{kind}-section-header",
                    ),
                    html.Div(
                        body,
                        className=f"analysis-{kind}-section-params",
                    ),
                ],
                id=ids.post_section_id(index)
                if kind == "post"
                else ids.filter_section_id(index),
                className=f"analysis-{kind}-section",
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "4px",
                    "padding": "0.5rem 0.75rem",
                    "margin": "0.25rem 0",
                    "background": COLOR_SURFACE,
                },
            )
        )
    return cards


def _section_form(info, instance, *, kind: str, index: int):
    """Render a ``param_form`` for a section's analyzer instance."""
    return param_form(
        info,
        current_values={
            k: v for k, v in vars(instance).items() if not k.startswith("_")
        },
        form_id_prefix=f"analysis-{kind}-{index}",
    )


def _build_model_panel(recipe: "RecipeState") -> html.Div:
    pipeline = recipe.pipeline
    model = pipeline.get_model()
    return html.Div(
        [
            html.H3("Model (endpoint)"),
            html.Div(
                _build_model_section(model) if model is not None
                else html.Span(
                    "No model configured.", style={"color": COLOR_MUTED}
                ),
                id=ids.ANALYSIS_MODEL_SECTION,
                className="analysis-model-section",
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id=ids.ANALYSIS_MODEL_DROPDOWN,
                        options=[
                            {"label": "(no model)", "value": ""},
                            *(
                                {"label": c, "value": c}
                                for c in _MODEL_CHOICES
                            ),
                        ],
                        value=type(model).__name__ if model is not None else "",
                        clearable=False,
                        style={"width": "260px", "display": "inline-block"},
                    ),
                ],
                style={"margin": "0.5rem 0"},
            ),
        ],
        style={"padding": "0.5rem 1rem"},
    )


def _build_model_section(model: object) -> html.Div:
    info = get_registry().get(type(model).__name__)
    body: Any
    if info is not None:
        body = _section_form(info, model, kind="model", index=0)
    else:
        body = html.Em(
            f"No registry info for {type(model).__name__}",
            style={"color": COLOR_MUTED},
        )
    return html.Div(
        [
            html.Strong(type(model).__name__),
            html.Div(body),
        ],
        style={
            "border": f"1px solid {COLOR_NAVY}",
            "borderRadius": "4px",
            "padding": "0.5rem 0.75rem",
            "background": COLOR_SURFACE,
        },
    )


def _build_run_console(recipe: "RecipeState") -> html.Div:
    has_model = recipe.pipeline.get_model() is not None
    return html.Div(
        [
            dbc.Button(
                "Run analysis",
                id=ids.ANALYSIS_RUN_BUTTON,
                color="primary",
                disabled=not has_model,
                n_clicks=0,
            ),
            dcc.Loading(
                id=ids.ANALYSIS_RUN_SPINNER,
                children=html.Div(
                    "",
                    id=ids.ANALYSIS_RUN_STATUS,
                    className="analysis-run-status",
                    style={"display": "inline-block", "marginLeft": "1rem"},
                ),
                type="default",
            ),
        ],
        className="analysis-run-console",
        style={
            "position": "sticky",
            "bottom": 0,
            "padding": "0.75rem 1rem",
            "background": COLOR_WHITE,
            "borderTop": "1px solid #ddd",
        },
    )


