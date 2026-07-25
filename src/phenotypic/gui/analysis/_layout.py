"""Layout builders for the analysis sub-app.

The page is a vertical stepper:

1. Output-root header (path + reload).
2. Pipeline summary chip.
3. Recompile banner (post edits need a CLI re-run to land in master).
4. Post section stack — table preview UX (col-name + top-5 before/after).
5. Filter section stack — plot preview UX (PlotAnalysis vs PNG).
6. Model section (single) — plot preview UX.
7. Sticky run console (button + spinner + status).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._config import MOUNT_HOME
from phenotypic.gui._design import (
    COLOR_BLUE,
    COLOR_BORDER,
    COLOR_GOLD,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    COLOR_WHITE,
    FONT_SIZE_HEADER_2,
    OI_VERMILION,
    RADIUS,
    SPACING_4,
)
from phenotypic.gui._operation_registry import OperationRegistry, get_registry
from phenotypic.gui._param_forms import param_form
from phenotypic.gui._shared import SHARED_LOGO_PATH
from phenotypic.gui.analysis import _ids as ids
from phenotypic.gui.analysis._plot_controls import plot_controls_form

if TYPE_CHECKING:
    from phenotypic.gui.analysis._recipe_state import RecipeState
    from phenotypic.gui.results_viewer._output_root import OutputRoot

#: Type alias for the column-list provider plumbed into ``param_form``.
ColumnsProvider = Callable[[str], list[str]]

def _choices_for_category(category: str) -> list[str]:
    """Return sorted analyzer/post class names registered under ``category``.

    Pulled from the shared :class:`OperationRegistry` so any
    ``SetAnalyzer`` / ``ModelFitter`` / ``PostMeasurement`` subclass
    discoverable in ``phenotypic.analysis`` / ``phenotypic.post`` is
    automatically offered in the analysis sub-app's add-dropdowns.
    """
    return sorted(info.name for info in get_registry().get_by_category(category))


#: Registry category whose analyzers render in the dedicated edge stack.
_EDGE_CATEGORY = "Edge Correction"


def filter_items_for_kind(
    pipeline: Any,
    kind: str,
    registry: OperationRegistry | None = None,
) -> list[tuple[str, Any]]:
    """Split the shared ``pipeline._filters`` dict by GUI section kind.

    The pipeline stores every non-model ``SetAnalyzer`` in one
    ``get_filters()`` dict; the GUI shows outlier filters and edge
    correctors in separate stacks. This returns the ordered
    ``(key, instance)`` sublist whose registry category maps to *kind*
    (``"edge"`` for ``"Edge Correction"``, ``"filter"`` for everything
    else). Local list position is the section's stable index for the
    remove/edit/preview callbacks.
    """
    registry = registry or get_registry()
    out: list[tuple[str, Any]] = []
    for key, inst in pipeline.get_filters().items():
        info = registry.get(type(inst).__name__)
        category = info.category if info is not None else "Filter"
        item_kind = "edge" if category == _EDGE_CATEGORY else "filter"
        if item_kind == kind:
            out.append((key, inst))
    return out


def build_app_layout(
    output_root: "OutputRoot",
    recipe: "RecipeState",
    *,
    url_prefix: str = MOUNT_HOME,
    columns_provider: Optional[ColumnsProvider] = None,
    binding_generation: str | None = None,
    refresh_supported: bool = True,
) -> html.Div:
    """Assemble the analysis page body.

    Args:
        output_root: Validated CLI output root (provides
            ``master_measurements.parquet`` and the curated
            ``measurements.parquet``).
        recipe: Loaded :class:`RecipeState` for ``<output>/pipeline.json``.
        url_prefix: Mount-point prefix used to resolve the dashboard
            logo URL in the output header. Defaults to ``MOUNT_HOME``
            ("/") for standalone launches; the hub passes
            ``MOUNT_ANALYSIS``.
        columns_provider: Optional callable resolving a column source
            (``"measurements"`` / ``"master_measurements"``) to a list
            of column names. Threaded through to the filter/model
            section forms so column-ref params render as live
            dropdowns. Standalone launches that build the layout before
            the app boots can leave this ``None``; ``create_app``
            passes :meth:`MeasurementSchema.columns_for`.
        binding_generation: Optional shell generation embedded in the page.
        refresh_supported: Whether the host supports in-process rebinding.

    Returns:
        Top-level ``html.Div`` ready to drop into the shell's main pane.
    """
    children: list[Any] = [
            _build_output_header(
                output_root,
                url_prefix=url_prefix,
                refresh_supported=refresh_supported,
            ),
            _build_pipeline_header(recipe),
            _build_stale_banner(),
            _build_load_warnings_banner(recipe),
            _build_recompile_banner(),
            _build_post_panel(recipe),
            _build_filter_panel(recipe, columns_provider=columns_provider),
            _build_edge_panel(recipe, columns_provider=columns_provider),
            _build_model_panel(recipe, columns_provider=columns_provider),
            _build_run_console(recipe),
            dcc.Interval(
                id=ids.ANALYSIS_SNAPSHOT_INTERVAL,
                interval=10_000,
                n_intervals=0,
            ),
            dcc.Store(
                id=ids.ANALYSIS_PIPELINE_STORE,
                data={
                    "revision": 0,
                    "pipeline_json": (
                        recipe.last_json
                        or recipe.pipeline.to_json()
                        or "{}"
                    ),
                },
            ),
            dcc.Store(
                id=ids.ANALYSIS_PIPELINE_EVENT_STORE,
                data=None,
            ),
            dcc.Store(
                id=ids.ANALYSIS_PIPELINE_GATE_ACK_STORE,
                data=None,
            ),
            dcc.Store(
                id=ids.ANALYSIS_PLOT_PREFS_STORE,
                storage_type="session",
                data={},
            ),
        ]
    if binding_generation is not None:
        children.insert(
            0,
            dcc.Store(
                id=ids.ANALYSIS_BINDING_GENERATION,
                data=binding_generation,
            ),
        )
    return html.Div(
        children,
        id=ids.ANALYSIS_PAGE,
        className="analysis-page",
    )


def build_active_snapshot_layout(
    output_root: "OutputRoot",
    *,
    url_prefix: str = MOUNT_HOME,
    binding_generation: str | None = None,
) -> html.Div:
    """Build a mutation-free Analysis placeholder for active processing."""
    children = [
        _build_output_header(output_root, url_prefix=url_prefix),
        dcc.Interval(
            id=ids.ANALYSIS_SNAPSHOT_INTERVAL,
            interval=5_000,
            n_intervals=0,
        ),
        dbc.Alert(
            [
                html.H5("Analysis is read-only while processing"),
                html.P(
                    "Analysis authoring and publication callbacks are not "
                    "loaded for this active output."
                ),
                html.P(
                    "When processing finishes, use Refresh snapshot to load "
                    "one stable Results and Analysis revision."
                ),
            ],
            color="warning",
            className="m-4",
        ),
    ]
    if binding_generation is not None:
        children.insert(
            0,
            dcc.Store(
                id=ids.ANALYSIS_BINDING_GENERATION,
                data=binding_generation,
            ),
        )
    return html.Div(
        children,
        id=ids.ANALYSIS_PAGE,
        className="analysis-page analysis-active-snapshot",
    )


def build_empty_state_layout(
    *,
    binding_generation: str | None = None,
) -> html.Div:
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
            "borderRadius": RADIUS,
        },
    )

    error_slot = html.Div(
        "",
        id=ids.EMPTY_HANDOFF_ERROR,
        className="analysis-empty-handoff-error text-danger small",
        style={"marginTop": "0.5rem", "minHeight": "1.25rem"},
    )

    children: list[Any] = [
        html.Div(
                [
                    html.H2(
                        "No output selected",
                        style={"color": COLOR_NAVY},
                    ),
                    html.P(
                        "Pick a CLI output directory in the sidebar, "
                        "then click ↩ Open in analysis to bind it. The "
                        "binding is shared with the results viewer -- "
                        "both tools rebuild against the chosen output "
                        "in lock-step.",
                    ),
                    handoff_banner,
                    error_slot,
                ],
            ),
        ]
    if binding_generation is not None:
        children.insert(
            0,
            dcc.Store(
                id=ids.ANALYSIS_BINDING_GENERATION,
                data=binding_generation,
            ),
        )
    return html.Div(
        children,
        id=ids.ANALYSIS_PAGE,
        className="analysis-page analysis-empty",
        style={"padding": "2rem"},
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_output_header(
    output_root: "OutputRoot",
    *,
    url_prefix: str = MOUNT_HOME,
    refresh_supported: bool = True,
) -> html.Div:
    return html.Div(
        [
            html.Img(
                src=f"{url_prefix}{SHARED_LOGO_PATH}",
                alt="PhenoTypic",
                className="analysis-output-header__logo",
            ),
            html.Strong("Output: "),
            html.Code(str(output_root.root)),
            html.Span(" · Snapshot "),
            html.Span(
                output_root.snapshot.captured_at.astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                )
            ),
            html.Span(" · "),
            html.Code(
                output_root.snapshot.processing_fingerprint[:12],
                title=output_root.snapshot.processing_fingerprint,
            ),
            dbc.Badge(
                (
                    "Active run snapshot"
                    if output_root.snapshot.active_run
                    else "Current"
                ),
                id=ids.ANALYSIS_SNAPSHOT_STATUS,
                color=(
                    "warning"
                    if output_root.snapshot.active_run
                    else "success"
                ),
                className="ms-2",
            ),
            dbc.Button(
                "Refresh snapshot",
                id=ids.ANALYSIS_REFRESH_SNAPSHOT,
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
                disabled=(
                    output_root.snapshot.active_run or not refresh_supported
                ),
                className="ms-2",
            ),
            html.Span(
                id=ids.ANALYSIS_REFRESH_ERROR,
                className="text-danger ms-2",
            ),
        ],
        id=ids.ANALYSIS_OUTPUT_HEADER,
        className="analysis-output-header",
        style={
            "padding": "0.5rem 1rem",
            "background": COLOR_SURFACE,
            "borderBottom": f"1px solid {COLOR_BORDER}",
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
    n_edge = len(filter_items_for_kind(pipeline, "edge"))
    n_filters = len(filter_items_for_kind(pipeline, "filter"))
    model = pipeline.get_model()
    model_name = type(model).__name__ if model is not None else "(none)"
    summary = (
        f"{len(pipeline.get_ops())} ops · {len(pipeline.get_meas())} meas · "
        f"{n_post} post · {n_filters} filters · {n_edge} edge · "
        f"model: {model_name}"
    )
    return [
        html.Strong(f"Pipeline: {pipeline.name}"),
        html.Span(
            summary,
            className="analysis-pipeline-summary",
            style={"marginLeft": SPACING_4, "color": COLOR_MUTED},
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
            html.Code("python -m phenotypic --mode recompile --output <output>"),
            ") to apply post changes to ",
            html.Code("measurements.parquet"),
            " (",
            html.Code("master_measurements.parquet"),
            " stays a clean, pre-post archive).",
        ],
        id=ids.ANALYSIS_RECOMPILE_BANNER,
        className="analysis-recompile-banner",
        style={
            "padding": "0.5rem 1rem",
            "background": "rgba(230, 159, 0, 0.10)",
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


def _build_load_warnings_banner(recipe: "RecipeState") -> html.Div:
    """Banner listing opaque analyzer entries retained during tolerant load.

    Renders only when ``recipe.load_warnings`` is non-empty. Each entry
    names the missing class plus its slot (filter vs model) so the user
    can select a replacement. The exact raw nodes remain on disk and are
    merged through unrelated saves.
    """
    if not recipe.load_warnings:
        return html.Div(
            id=ids.ANALYSIS_LOAD_WARNINGS_BANNER,
            className="analysis-load-warnings-banner",
            style={"display": "none"},
        )

    bullets = [
        html.Li(
            [
                html.Code(w.class_name),
                f" (slot: {w.slot}",
                f", key: {w.name}" if w.slot == "filter" else "",
                ")",
            ]
        )
        for w in recipe.load_warnings
    ]
    return html.Div(
        [
            html.Strong("Unavailable analyzer entries"),
            html.Div(
                [
                    "These classes were referenced in ",
                    html.Code(str(recipe.path)),
                    " but are no longer available in this version of "
                    "phenotypic. They are unavailable in the live editor, "
                    "but their exact JSON remains on disk and is preserved "
                    "through unrelated edits. Selecting a live node for the "
                    "same slot explicitly replaces that opaque entry.",
                ],
                style={"marginTop": "0.25rem", "fontSize": "0.9em"},
            ),
            html.Ul(bullets, style={"margin": "0.5rem 0 0 1rem"}),
        ],
        id=ids.ANALYSIS_LOAD_WARNINGS_BANNER,
        className="analysis-load-warnings-banner",
        style={
            "padding": "0.5rem 1rem",
            "background": "rgba(213, 94, 0, 0.08)",
            "borderLeft": f"4px solid {OI_VERMILION}",
            "margin": "0.5rem 1rem",
            "color": COLOR_NAVY,
        },
    )


def _build_post_panel(recipe: "RecipeState") -> html.Div:
    return _build_section_panel(
        title="Post operations (metadata transforms)",
        section_label="post",
        choices=_choices_for_category("Post"),
        add_dropdown_id=ids.ANALYSIS_POST_ADD_DROPDOWN,
        stack_id=ids.ANALYSIS_POST_STACK,
        recipe=recipe,
    )


def _build_filter_panel(
    recipe: "RecipeState",
    *,
    columns_provider: Optional[ColumnsProvider] = None,
) -> html.Div:
    return _build_section_panel(
        title="Filters",
        section_label="filter",
        choices=_choices_for_category("Filter"),
        add_dropdown_id=ids.ANALYSIS_FILTER_ADD_DROPDOWN,
        stack_id=ids.ANALYSIS_FILTER_STACK,
        recipe=recipe,
        columns_provider=columns_provider,
    )


def _build_edge_panel(
    recipe: "RecipeState",
    *,
    columns_provider: Optional[ColumnsProvider] = None,
) -> html.Div:
    return _build_section_panel(
        title="Edge Correction",
        section_label="edge",
        choices=_choices_for_category("Edge Correction"),
        add_dropdown_id=ids.ANALYSIS_EDGE_ADD_DROPDOWN,
        stack_id=ids.ANALYSIS_EDGE_STACK,
        recipe=recipe,
        columns_provider=columns_provider,
    )


def _build_section_panel(
    *,
    title: str,
    section_label: "ids.SectionKind",
    choices: list[str],
    add_dropdown_id: str,
    stack_id: str,
    recipe: "RecipeState",
    columns_provider: Optional[ColumnsProvider] = None,
) -> html.Div:
    return html.Div(
        [
            html.H3(title),
            html.Div(
                build_section_stack(
                    stack_id,
                    section_label,
                    recipe,
                    columns_provider=columns_provider,
                ),
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
    *,
    columns_provider: Optional[ColumnsProvider] = None,
    plot_prefs: Optional[dict] = None,
) -> list:
    """Build the list of section cards inside a stack.

    Each card now hosts a fully editable :func:`param_form` rendered
    against the section's analyzer instance. Widget ids are scoped via
    ``form_id_prefix=f"analysis-{kind}-{index}"`` so the analysis-side
    pattern-matching callback can map any param edit back to the correct
    section without colliding with the builder's prefixes.

    Filter and edge cards additionally host a :func:`plot_controls_form` — a
    Display-settings disclosure plus a Preview button. Its widget values
    re-seed from ``plot_prefs`` (the session-scoped plotting-preference
    store) so display tweaks survive stack rebuilds. Post cards carry no
    plot controls.
    """
    pipeline = recipe.pipeline
    if registry is None:
        registry = get_registry()
    items: list[tuple[str, Any]]
    if kind == "post":
        items = list(pipeline.get_post().items())
    elif kind in ("filter", "edge"):
        items = filter_items_for_kind(pipeline, kind, registry)
    else:
        return []

    cards: list = []
    for index, (name, instance) in enumerate(items):
        info = registry.get(type(instance).__name__)
        body: Any = (
            _section_form(
                info,
                instance,
                kind=kind,
                index=index,
                columns_provider=columns_provider,
            )
            if info is not None
            else html.Em(
                f"No registry info for {type(instance).__name__}",
                style={"color": COLOR_MUTED},
            )
        )
        # Filter and edge cards carry a plotting-preview affordance; post cards
        # keep the table-preview path and get no plot controls.
        if kind in ("filter", "edge"):
            # ``kind`` is narrowed to a ``PlotSectionKind`` by the guard above,
            # but mypy can't infer that from a tuple-membership test.
            body = [
                body,
                plot_controls_form(
                    cast("ids.PlotSectionKind", kind), index, instance, plot_prefs
                ),
            ]
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
                                    "fontSize": FONT_SIZE_HEADER_2,
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
                else (
                    ids.edge_section_id(index)
                    if kind == "edge"
                    else ids.filter_section_id(index)
                ),
                className=f"analysis-{kind}-section",
                style={
                    "border": f"1px solid {COLOR_BORDER}",
                    "borderRadius": RADIUS,
                    "padding": "0.5rem 0.75rem",
                    "margin": "0.25rem 0",
                    "background": COLOR_SURFACE,
                },
            )
        )
    return cards


def _section_form(
    info,
    instance,
    *,
    kind: str,
    index: int,
    columns_provider: Optional[ColumnsProvider] = None,
):
    """Render a ``param_form`` for a section's analyzer instance."""
    return param_form(
        info,
        current_values={
            k: v for k, v in vars(instance).items() if not k.startswith("_")
        },
        form_id_prefix=f"analysis-{kind}-{index}",
        columns_provider=columns_provider,
    )


def _build_model_panel(
    recipe: "RecipeState",
    *,
    columns_provider: Optional[ColumnsProvider] = None,
    plot_prefs: Optional[dict] = None,
) -> html.Div:
    pipeline = recipe.pipeline
    model = pipeline.get_model()
    return html.Div(
        [
            html.H3("Model (endpoint)"),
            html.Div(
                _build_model_section(
                    model,
                    columns_provider=columns_provider,
                    plot_prefs=plot_prefs,
                )
                if model is not None
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
                                for c in _choices_for_category("Model")
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


def _build_model_section(
    model: object,
    *,
    columns_provider: Optional[ColumnsProvider] = None,
    plot_prefs: Optional[dict] = None,
) -> html.Div:
    info = get_registry().get(type(model).__name__)
    body: Any
    if info is not None:
        body = _section_form(
            info,
            model,
            kind="model",
            index=0,
            columns_provider=columns_provider,
        )
    else:
        body = html.Em(
            f"No registry info for {type(model).__name__}",
            style={"color": COLOR_MUTED},
        )
    return html.Div(
        [
            html.Strong(type(model).__name__),
            html.Div(body),
            plot_controls_form("model", 0, model, plot_prefs),
        ],
        style={
            "border": f"1px solid {COLOR_NAVY}",
            "borderRadius": RADIUS,
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
            "borderTop": f"1px solid {COLOR_BORDER}",
        },
    )


