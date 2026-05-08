"""Callbacks for the analysis sub-app.

Wires user interactions to :class:`RecipeState`: adding/removing post or
filter sections, choosing the endpoint model, and triggering the inline
``Run analysis`` button. Every mutation persists to
``<output>/pipeline.json`` via :meth:`RecipeState.save`.

v1 keeps section authoring minimal — selecting a class from the
dropdown instantiates it with sensible default parameters; tuning
parameters from the GUI is deferred to v2 (the user can hand-edit
``pipeline.json`` in the meantime).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from dash import ALL, Input, Output, State, callback_context, html, no_update

from phenotypic.gui._config import (
    CFG_MEASUREMENT_SCHEMA,
    CFG_OUTPUT_ROOT,
    CFG_RECIPE_STATE,
    MEASUREMENTS_PARQUET,
)
from phenotypic.gui._design import COLOR_MUTED
from phenotypic.gui.analysis import _ids as ids
from phenotypic.gui.analysis._layout import (
    build_section_stack,
    pipeline_header_children,
)

if TYPE_CHECKING:
    import dash

logger = logging.getLogger(__name__)


# v1 placeholder defaults — users tune by editing pipeline.json until
# per-section param forms ship in v2.
_POST_DEFAULTS: dict[str, dict[str, Any]] = {
    "PrependString": {"to_column": "Metadata_Strain", "string": "strain_"},
    "AppendString": {"to_column": "Metadata_Strain", "string": "_x"},
    "ExpandMetadata": {"on_column": "Metadata_ImageFile",
                       "split_pattern": "_",
                       "new_columns": ["A", "B"]},
    "MergeMetadata": {"metadata_path": "metadata.csv", "on": "Metadata_ImageFile"},
}
_FILTER_DEFAULTS: dict[str, dict[str, Any]] = {
    "EdgeCorrector": {"on": "Shape_Area", "groupby": ["Metadata_Strain"]},
    "TukeyOutlierRemover": {"on": "Shape_Area", "groupby": ["Metadata_Strain"]},
}
_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "LogGrowthModel": {"on": "Shape_Area", "groupby": ["Metadata_Strain"],
                       "time_label": "Metadata_Time", "n_jobs": 1},
    "LinearSoftplusModel": {"on": "Shape_Area",
                            "groupby": ["Metadata_Strain"],
                            "time_label": "Metadata_Time"},
}


def register_callbacks(app: "dash.Dash") -> None:
    """Register every callback on *app*.

    The ``app.server.config`` must already carry:

    - :data:`CFG_OUTPUT_ROOT` — the validated
      :class:`~phenotypic.gui.results_viewer._output_root.OutputRoot`.
    - :data:`CFG_RECIPE_STATE` — the loaded :class:`RecipeState`.
    - :data:`CFG_MEASUREMENT_SCHEMA` — the
      :class:`~phenotypic.gui.analysis._schema_cache.MeasurementSchema`
      instance whose ``columns_for`` method is plumbed into every
      ``build_section_stack`` rebuild so column-aware widgets stay in
      sync with the on-disk measurements file.
    """
    server = app.server

    def _columns_provider(source: str) -> list:
        """Resolve a ColumnSource to columns from the live schema cache."""
        schema = server.config.get(CFG_MEASUREMENT_SCHEMA)
        if schema is None:
            return []
        return schema.columns_for(source)

    @app.callback(
        Output(ids.ANALYSIS_POST_STACK, "children"),
        Output(ids.ANALYSIS_PIPELINE_HEADER, "children"),
        Output(ids.ANALYSIS_PIPELINE_STORE, "data", allow_duplicate=True),
        Input(ids.ANALYSIS_POST_ADD_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _add_post(class_name: str | None):
        if not class_name:
            return no_update, no_update, no_update
        recipe = server.config[CFG_RECIPE_STATE]
        instance = _instantiate("post", class_name)
        if instance is None:
            return no_update, no_update, no_update
        post_dict = recipe.pipeline.get_post()
        post_dict[_unique_key(post_dict, class_name)] = instance
        recipe.pipeline.set_post(post_dict)
        recipe.save()
        return (
            build_section_stack(
                ids.ANALYSIS_POST_STACK, "post", recipe,
                columns_provider=_columns_provider,
            ),
            _pipeline_summary(recipe),
            recipe.last_json,
        )

    @app.callback(
        Output(ids.ANALYSIS_FILTER_STACK, "children"),
        Output(ids.ANALYSIS_PIPELINE_HEADER, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_PIPELINE_STORE, "data", allow_duplicate=True),
        Input(ids.ANALYSIS_FILTER_ADD_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _add_filter(class_name: str | None):
        if not class_name:
            return no_update, no_update, no_update
        recipe = server.config[CFG_RECIPE_STATE]
        instance = _instantiate("filter", class_name)
        if instance is None:
            return no_update, no_update, no_update
        filters_dict = recipe.pipeline.get_filters()
        filters_dict[_unique_key(filters_dict, class_name)] = instance
        recipe.pipeline.set_filters(filters_dict)
        recipe.save()
        return (
            build_section_stack(
                ids.ANALYSIS_FILTER_STACK, "filter", recipe,
                columns_provider=_columns_provider,
            ),
            _pipeline_summary(recipe),
            recipe.last_json,
        )

    @app.callback(
        Output(ids.ANALYSIS_MODEL_SECTION, "children"),
        Output(ids.ANALYSIS_PIPELINE_HEADER, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_RUN_BUTTON, "disabled"),
        Output(ids.ANALYSIS_PIPELINE_STORE, "data", allow_duplicate=True),
        Input(ids.ANALYSIS_MODEL_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _set_model(class_name: str):
        recipe = server.config[CFG_RECIPE_STATE]
        if class_name == "":
            recipe.pipeline.set_model(None)
        else:
            instance = _instantiate("model", class_name)
            if instance is None:
                return no_update, no_update, no_update, no_update
            recipe.pipeline.set_model(instance)
        recipe.save()

        from phenotypic.gui.analysis._layout import (
            _build_model_section,  # type: ignore[attr-defined]
        )
        model = recipe.pipeline.get_model()
        section = (
            _build_model_section(model, columns_provider=_columns_provider)
            if model is not None
            else html.Span("No model configured.", style={"color": COLOR_MUTED})
        )
        return (
            section,
            _pipeline_summary(recipe),
            model is None,
            recipe.last_json,
        )

    @app.callback(
        Output(ids.ANALYSIS_POST_STACK, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_FILTER_STACK, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_PIPELINE_HEADER, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_PIPELINE_STORE, "data", allow_duplicate=True),
        # ``ALL`` is Dash's pattern-matching wildcard; the strict Literal/int
        # signature on ``section_remove_button_id`` doesn't model it.
        Input(ids.section_remove_button_id(ALL, ALL), "n_clicks"),  # type: ignore[arg-type]
        State(ids.ANALYSIS_PIPELINE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _remove_section(n_clicks_list, _store):
        # Pattern-matching callback fires on *every* button (including
        # zero-click initial state). Filter to the actually-triggered
        # button via callback_context.
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update
        triggered = ctx.triggered[0]
        if not triggered["value"]:
            return no_update, no_update, no_update, no_update
        triggered_id = ctx.triggered_id
        if not isinstance(triggered_id, dict):
            return no_update, no_update, no_update, no_update

        kind = triggered_id["kind"]
        index = triggered_id["index"]

        recipe = server.config[CFG_RECIPE_STATE]
        if kind == "post":
            items = list(recipe.pipeline.get_post().items())
        elif kind == "filter":
            items = list(recipe.pipeline.get_filters().items())
        else:
            return no_update, no_update, no_update, no_update

        if not (0 <= index < len(items)):
            return no_update, no_update, no_update, no_update

        items.pop(index)
        new_dict = dict(items)
        if kind == "post":
            recipe.pipeline.set_post(new_dict)
        else:
            recipe.pipeline.set_filters(new_dict)
        recipe.save()

        return (
            build_section_stack(
                ids.ANALYSIS_POST_STACK, "post", recipe,
                columns_provider=_columns_provider,
            ),
            build_section_stack(
                ids.ANALYSIS_FILTER_STACK, "filter", recipe,
                columns_provider=_columns_provider,
            ),
            _pipeline_summary(recipe),
            recipe.last_json,
        )

    # ----- Per-section param edit ----- #
    # One mega fan-in callback over every param-* widget kind (mirrors the
    # builder's pattern). Only triggers whose ``prefix`` starts with
    # ``"analysis-"`` are routed; other prefixes (e.g. the builder's
    # node-uuid prefixes that may share the same widget types) fall
    # through as no-ops.
    @app.callback(
        Output(ids.ANALYSIS_POST_STACK, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_FILTER_STACK, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_MODEL_SECTION, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_PIPELINE_HEADER, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_PIPELINE_STORE, "data", allow_duplicate=True),
        Input({"type": "param-bool", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-num", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-str", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-enum", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-list", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-tuple", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-multi-tag", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-multi-value", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-column-scalar", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-column-multi", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-column-mode", "prefix": ALL, "name": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _on_param_edit(*_values: Any):
        ctx = callback_context
        if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
            return no_update, no_update, no_update, no_update, no_update
        prefix = ctx.triggered_id.get("prefix", "")
        if not isinstance(prefix, str) or not prefix.startswith("analysis-"):
            return no_update, no_update, no_update, no_update, no_update

        recipe = server.config[CFG_RECIPE_STATE]
        applied, kind = _apply_param_edit(recipe, ctx)
        if not applied:
            return no_update, no_update, no_update, no_update, no_update

        # Only rebuild the touched stack — the other two are unchanged
        # so we send ``no_update`` and avoid wasted Dash component diffs.
        post_out = no_update
        filter_out = no_update
        model_out: Any = no_update
        if kind == "post":
            post_out = build_section_stack(
                ids.ANALYSIS_POST_STACK, "post", recipe,
                columns_provider=_columns_provider,
            )
        elif kind == "filter":
            filter_out = build_section_stack(
                ids.ANALYSIS_FILTER_STACK, "filter", recipe,
                columns_provider=_columns_provider,
            )
        elif kind == "model":
            from phenotypic.gui.analysis._layout import _build_model_section

            model = recipe.pipeline.get_model()
            model_out = (
                _build_model_section(model, columns_provider=_columns_provider)
                if model is not None
                else html.Span("No model configured.", style={"color": COLOR_MUTED})
            )
        return (
            post_out,
            filter_out,
            model_out,
            _pipeline_summary(recipe),
            recipe.last_json,
        )

    @app.callback(
        Output(ids.ANALYSIS_RUN_STATUS, "children"),
        Input(ids.ANALYSIS_RUN_BUTTON, "n_clicks"),
        prevent_initial_call=True,
    )
    def _run_analysis(n_clicks: int):
        if not n_clicks:
            return no_update
        recipe = server.config[CFG_RECIPE_STATE]
        output_root = server.config[CFG_OUTPUT_ROOT]
        if recipe.pipeline.get_model() is None:
            return html.Span(
                "No model configured.", style={"color": "#c00"}
            )
        return _run_inline(recipe, Path(output_root.root))


def _apply_param_edit(recipe: Any, ctx: Any) -> tuple[bool, str | None]:
    """Resolve a triggered param widget back into a recipe mutation.

    Decodes ``ctx.triggered_id["prefix"]`` (``"analysis-{kind}-{index}"``)
    to find the section being edited, builds a new analyzer instance with
    the updated kwarg, and persists via :meth:`RecipeState.save`.

    For multi-union widgets the tag and value live in two separate
    components; we read both via ``ctx.inputs`` and pack them as a tuple
    that :func:`parse_widget_value` understands.

    Returns:
        ``(applied, kind)`` where ``applied`` is ``True`` only when the
        edit was saved. ``kind`` is the section kind that changed
        (``"post"`` / ``"filter"`` / ``"model"``) so the caller can
        rebuild only the touched stack; ``None`` when nothing applied.
    """
    from phenotypic.gui._operation_registry import get_registry
    from phenotypic.gui._param_forms import parse_widget_value

    triggered = ctx.triggered_id
    prefix = triggered.get("prefix", "")
    name = triggered.get("name")
    widget_type = triggered.get("type")
    if not isinstance(prefix, str) or not isinstance(name, str):
        return False, None

    # ``analysis-{kind}-{index}``
    parts = prefix.split("-", 2)
    if len(parts) != 3 or parts[0] != "analysis":
        return False, None
    kind = parts[1]
    try:
        index = int(parts[2])
    except ValueError:
        return False, None

    pipeline = recipe.pipeline
    if kind == "post":
        section_dict = pipeline.get_post()
        items = list(section_dict.items())
    elif kind == "filter":
        section_dict = pipeline.get_filters()
        items = list(section_dict.items())
    elif kind == "model":
        model = pipeline.get_model()
        if model is None:
            return False, None
        section_dict = None
        items = [(type(model).__name__, model)]
        index = 0
    else:
        return False, None

    if not (0 <= index < len(items)):
        return False, None
    section_key, current_instance = items[index]

    info = get_registry().get(type(current_instance).__name__)
    if info is None:
        return False, None
    p = info.parameters.get(name)
    if p is None:
        return False, None

    # Multi-component widgets (multi-union, column-with-alt) spread state
    # across two ids; pack both values so ``parse_widget_value`` can
    # dispatch on the resulting tuple.
    def _pair(type_a: str, type_b: str) -> tuple[Any, Any]:
        return (
            ctx.inputs.get(_pattern_input_key(name, prefix, type_a)),
            ctx.inputs.get(_pattern_input_key(name, prefix, type_b)),
        )

    if widget_type in ("param-multi-tag", "param-multi-value"):
        raw: Any = _pair("param-multi-tag", "param-multi-value")
    elif (
        widget_type in ("param-column-scalar", "param-column-mode")
        and p.column_ref is not None
        and p.column_ref.with_alt
    ):
        # Scalar column-with-alt only. ``param-column-multi`` is
        # intentionally absent: v1 has no ``ColumnRefList | None`` param,
        # and ``_column_or_alt_widget`` raises ``NotImplementedError`` for
        # ``spec.multi=True``. Adding multi+alt requires updating both
        # sites in lockstep.
        raw = _pair("param-column-mode", "param-column-scalar")
    else:
        raw = ctx.triggered[0]["value"] if ctx.triggered else None

    coerced = parse_widget_value(raw, p)

    new_kwargs = {
        k: v for k, v in vars(current_instance).items() if not k.startswith("_")
    }
    new_kwargs[name] = coerced

    sig = _filter_kwargs_to_signature(type(current_instance), new_kwargs)
    try:
        new_instance = type(current_instance)(**sig)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Could not rebuild %s with new param %s=%r",
            type(current_instance).__name__,
            name,
            coerced,
            exc_info=True,
        )
        return False, None

    if kind == "post":
        section_dict[section_key] = new_instance  # type: ignore[index]
        pipeline.set_post(section_dict)  # type: ignore[arg-type]
    elif kind == "filter":
        section_dict[section_key] = new_instance  # type: ignore[index]
        pipeline.set_filters(section_dict)  # type: ignore[arg-type]
    else:
        pipeline.set_model(new_instance)

    recipe.save()
    return True, kind


def _pattern_input_key(name: str, prefix: str, type_: str) -> str:
    """Build a ``ctx.inputs`` lookup key for a Dash pattern-matching id.

    Dash serializes pattern-matching ids as JSON objects with keys sorted
    alphabetically — ``name``, ``prefix``, ``type`` — followed by the
    ``.value`` property. Centralizing the format keeps the ordering
    invariant in one place rather than scattered across f-strings.
    """
    return f'{{"name":"{name}","prefix":"{prefix}","type":"{type_}"}}.value'


def _filter_kwargs_to_signature(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only ``kwargs`` entries that ``cls.__init__`` accepts.

    Reuses :data:`SerializablePipeline._ANALYZER_INIT_ALIASES` so the
    GUI's per-edit reconstruction follows the same name-mapping
    (``n_jobs`` -> ``num_workers``) the JSON deserializer uses.
    """
    import inspect

    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )

    sig = inspect.signature(cls.__init__)
    accepted = {p for p in sig.parameters if p != "self"}
    aliases = SerializablePipeline._ANALYZER_INIT_ALIASES

    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in accepted:
            out[k] = v
        elif k in aliases and aliases[k] in accepted:
            out[aliases[k]] = v
    return out


def _run_inline(recipe: Any, output_dir: Path) -> Any:
    """Read measurements.parquet, run analyze, atomic-write outputs."""
    from phenotypic._cli._cli_output_manager import _emit_analysis_outputs

    measurements = output_dir / MEASUREMENTS_PARQUET
    if not measurements.exists():
        return html.Span(
            f"Curated measurements not found at {measurements}.",
            style={"color": "#c00"},
        )

    start = time.time()
    try:
        master_pl = pl.read_parquet(measurements)
    except Exception as exc:  # noqa: BLE001
        return html.Span(f"Read failed: {exc}", style={"color": "#c00"})

    result = _emit_analysis_outputs(output_dir, master_pl, recipe.pipeline)
    duration = time.time() - start

    if result is None:
        return html.Span(
            "Analysis run failed (see server logs).",
            style={"color": "#c00"},
        )

    written, n_rows = result
    return html.Span(
        f"Wrote {written.name} ({n_rows} rows · {duration:.1f}s)",
        style={"color": "#2e7d32"},
    )


# --- helpers ---------------------------------------------------------------


_KIND_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "post": _POST_DEFAULTS,
    "filter": _FILTER_DEFAULTS,
    "model": _MODEL_DEFAULTS,
}

_KIND_MODULES: dict[str, str] = {
    "post": "phenotypic.post",
    "filter": "phenotypic.analysis",
    "model": "phenotypic.analysis",
}


def _instantiate(kind: ids.InstantiationKind, class_name: str) -> Any:
    """Instantiate a class with v1 placeholder defaults, or ``None``."""
    defaults = _KIND_DEFAULTS.get(kind, {}).get(class_name)
    module_name = _KIND_MODULES.get(kind)
    if defaults is None or module_name is None:
        return None
    try:
        import importlib

        module = importlib.import_module(module_name)
    except ImportError:
        return None

    cls = getattr(module, class_name, None)
    if cls is None:
        return None
    try:
        return cls(**defaults)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to instantiate %s with defaults", class_name, exc_info=True
        )
        return None


def _unique_key(existing: dict, base: str) -> str:
    """Return ``base`` (or ``base_1``, ``base_2``, ...) avoiding collisions."""
    if base not in existing:
        return base
    n = 1
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def _pipeline_summary(recipe: Any) -> Any:
    """Rebuild the pipeline-header content (delegates to layout helper)."""
    return pipeline_header_children(recipe)


__all__ = ["register_callbacks"]
