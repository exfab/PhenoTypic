"""ipywidgets notebook shell for :class:`~phenotypic.abc_.plotting.PhtPlot`.

Invoked only when a provider's figures declare ``Control``s (control-free
providers render as a composed ``go.Figure`` instead — see
:meth:`PhtPlot.report`). The render-loop logic is factored into pure,
ipywidgets-free module-level helpers so it can be unit-tested without a kernel;
``ipywidgets`` is imported lazily inside :func:`build_notebook_dashboard`.

Controls are deduped **by identity** (``id(control)``): one widget per unique
``Control`` instance, and changing it re-renders exactly the figures that
reference that instance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from phenotypic.abc_.plotting import Control, FigureSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic.abc_.plotting import PhtPlot

__all__ = [
    "unique_controls",
    "control_owners",
    "initial_control_state",
    "spec_control_kwargs",
    "build_notebook_dashboard",
]


# -- pure helpers (no ipywidgets) -------------------------------------------


def unique_controls(specs: list[FigureSpec]) -> list[Control]:
    """Return the controls across ``specs``, deduped by identity, first-seen order.

    Two distinct ``Control`` instances are kept separate even if their fields are
    equal; the same instance shared by several specs collapses to one entry.
    """
    seen: dict[int, Control] = {}
    for spec in specs:
        for control in spec.controls.values():
            seen.setdefault(id(control), control)
    return list(seen.values())


def control_owners(specs: list[FigureSpec]) -> dict[int, list[FigureSpec]]:
    """Map ``id(control)`` → the specs whose figures reference that control.

    Used to recompute exactly the affected figures when a control changes.
    """
    owners: dict[int, list[FigureSpec]] = {}
    for spec in specs:
        for control in spec.controls.values():
            owners.setdefault(id(control), [])
            if spec not in owners[id(control)]:
                owners[id(control)].append(spec)
    return owners


def initial_control_state(specs: list[FigureSpec]) -> dict[int, Any]:
    """Map ``id(control)`` → its default value, for every unique control."""
    return {id(c): c.default for c in unique_controls(specs)}


def spec_control_kwargs(spec: FigureSpec, state: dict[int, Any]) -> dict[str, Any]:
    """Resolve ``spec``'s control kwargs to current values from ``state``.

    Args:
        spec: The figure spec.
        state: ``{id(control): value}`` current control values.

    Returns:
        ``{method-kwarg: value}`` ready to splat into the figure method.
    """
    return {kwarg: state[id(control)] for kwarg, control in spec.controls.items()}


# -- ipywidgets shell -------------------------------------------------------


def _require_ipywidgets() -> Any:
    """Import and return the ``ipywidgets`` module, or raise a helpful error."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The interactive notebook dashboard requires ipywidgets. Install the "
            "GUI extra: `uv sync --extra gui`."
        ) from exc
    return widgets


def _make_widget(control: Control, widgets: Any) -> Any:
    """Build the ipywidget for a single :class:`Control`."""
    if control.kind == "float":
        low, high = control.bounds  # type: ignore[misc]
        kwargs: dict[str, Any] = dict(
            value=control.default, min=low, max=high, description=control.label
        )
        if control.step is not None:
            kwargs["step"] = control.step
        return widgets.FloatSlider(**kwargs)
    if control.kind == "select":
        return widgets.Dropdown(
            value=control.default,
            options=list(control.options or ()),
            description=control.label,
        )
    if control.kind == "bool":
        return widgets.Checkbox(value=control.default, description=control.label)
    return widgets.Text(value=control.default, description=control.label)


def _plotly_display_payload(fig: Any) -> dict[str, Any]:
    """Return an Output-compatible display-data payload for a Plotly figure."""
    from plotly.io import to_json

    data = {
        "application/vnd.plotly.v1+json": json.loads(
            to_json(fig, validate=False)
        )
    }
    return {
        "output_type": "display_data",
        "data": data,
        "metadata": {},
    }


def build_notebook_dashboard(provider: "PhtPlot", subject: Any = None) -> Any:
    """Build an ipywidgets dashboard for ``provider``'s figures.

    One widget per unique control (deduped by identity); changing a control
    re-renders exactly the figures that reference it. Figures are grouped into
    collapsible ``Accordion`` cards by their ``section`` tag.

    Args:
        provider: The :class:`PhtPlot` to render.
        subject: Subject to bind (operations); ``None`` uses the held subject.

    Returns:
        An ``ipywidgets.Widget`` ready to display in a Jupyter cell.
    """
    widgets = _require_ipywidgets()
    try:
        __import__("IPython")
    except ImportError as exc:  # pragma: no cover - exercised only without IPython
        raise ImportError(
            "The interactive notebook dashboard requires IPython. Install the "
            "GUI extra: `uv sync --extra gui`."
        ) from exc

    specs = provider.iter_figures()
    bound = provider.figures(subject)
    state = initial_control_state(specs)
    owners = control_owners(specs)

    # One Output per figure (keyed by spec name) and a control-widget per unique
    # control instance (keyed by id(control)).
    outputs: dict[str, Any] = {spec.name: widgets.Output() for spec in specs}
    control_widgets: dict[int, Any] = {
        id(control): _make_widget(control, widgets)
        for control in unique_controls(specs)
    }

    def render_spec(spec: FigureSpec) -> None:
        fig = bound.render(spec, **spec_control_kwargs(spec, state))
        out = outputs[spec.name]
        out.outputs = (_plotly_display_payload(fig),)

    def on_change(control_id: int) -> Any:
        def handler(change: Any) -> None:
            state[control_id] = change["new"]
            for spec in owners.get(control_id, ()):
                render_spec(spec)

        return handler

    for control_id, widget in control_widgets.items():
        widget.observe(on_change(control_id), names="value")

    # Initial render of every figure.
    for spec in specs:
        render_spec(spec)

    # Group figures into collapsible cards by section, in definition order.
    sections: dict[str, list[FigureSpec]] = {}
    for spec in specs:
        sections.setdefault(spec.section, []).append(spec)

    accordion = widgets.Accordion(
        children=[
            widgets.VBox([outputs[spec.name] for spec in section_specs])
            for section_specs in sections.values()
        ]
    )
    for i, name in enumerate(sections):
        accordion.set_title(i, name)

    controls_box = widgets.VBox(list(control_widgets.values()))
    return widgets.VBox([controls_box, accordion])
