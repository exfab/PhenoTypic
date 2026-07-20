"""Renderer-neutral plotting capability and figure composition helpers.

This module deliberately imports only the Python standard library at runtime.
Plotly, the PhenoTypic theme, and notebook widgets are loaded only when a figure
is rendered. This keeps the plotting capability safe to mix into Pydantic
models without adding fields, constructors, or persistent render state.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import plotly.graph_objects as go

__all__ = [
    "BoundFigures",
    "Control",
    "FigureSpec",
    "PhtPlot",
    "figure",
]


_FIGURE_ORDER = itertools.count()

ControlKind = Literal["float", "select", "bool", "text"]


@dataclass(frozen=True)
class Control:
    """Renderer-neutral input bound to a figure method keyword argument.

    Controls are bound by identity. Reusing one ``Control`` instance across
    figure methods creates one shared control, while equal but distinct
    instances remain independent.

    Args:
        label: Human-readable control label.
        kind: Control renderer kind.
        default: Initial value for the control.
        bounds: Required ``(low, high)`` range for a float control.
        step: Optional step for a float control.
        options: Required allowed values for a select control.
        help: Optional help text.

    Raises:
        ValueError: If the default or kind-specific settings are invalid.
    """

    label: str
    kind: ControlKind
    default: Any
    bounds: tuple[float, float] | None = None
    step: float | None = None
    options: tuple[Any, ...] | None = None
    help: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "float":
            if self.bounds is None:
                raise ValueError(
                    f"Control({self.label!r}): float requires bounds"
                )
            low, high = self.bounds
            if not (low <= self.default <= high):
                raise ValueError(
                    f"Control({self.label!r}): default {self.default!r} outside "
                    f"bounds {self.bounds!r}"
                )
        elif self.kind == "select":
            if not self.options:
                raise ValueError(
                    f"Control({self.label!r}): select requires non-empty options"
                )
            if self.default not in self.options:
                raise ValueError(
                    f"Control({self.label!r}): default {self.default!r} not in "
                    f"options {self.options!r}"
                )
        elif self.kind == "bool":
            if not isinstance(self.default, bool):
                raise ValueError(
                    f"Control({self.label!r}): bool default must be a bool"
                )
        elif self.kind == "text":
            if not isinstance(self.default, str):
                raise ValueError(
                    f"Control({self.label!r}): text default must be a str"
                )
        else:  # pragma: no cover - defensive for untyped callers
            raise ValueError(
                f"Control({self.label!r}): unknown kind {self.kind!r}"
            )


@dataclass(frozen=True)
class FigureSpec:
    """Metadata attached to a method by :func:`figure`.

    Attributes:
        title: Human-readable figure title.
        section: Grouping tag used by report adapters.
        controls: Mapping from method keyword to its control.
        description: Optional renderer-neutral explanatory content.
        primary: Whether this is the default :meth:`PhtPlot.inspect` figure.
        name: Decorated method name.
        method: Wrapped, theme-applying method.
        wants_subject: Whether the method accepts a positional subject.
        subject_param: Positional subject parameter name, if present.
        order: Definition-order index.
    """

    title: str
    section: str
    controls: dict[str, Control]
    description: Any
    primary: bool
    name: str
    method: Callable[..., "go.Figure"]
    wants_subject: bool
    subject_param: str | None
    order: int


def figure(
    *,
    title: str,
    section: str = "default",
    controls: dict[str, Control] | None = None,
    description: Any = None,
    primary: bool = False,
) -> Callable[[Callable[..., "go.Figure"]], Callable[..., "go.Figure"]]:
    """Mark a method as a figure builder and lazily apply the house theme.

    A figure method may accept a subject as its first positional parameter.
    Any positional parameter not named by ``controls`` is treated as the
    subject. Plot-specific parameters that are not subjects must therefore be
    keyword-only.

    Args:
        title: Human-readable figure title.
        section: Grouping tag used by report adapters.
        controls: Mapping from method keyword to renderer-neutral control.
        description: Optional renderer-neutral explanatory content.
        primary: Whether this is the default figure returned by ``inspect``.

    Returns:
        A decorator for a Plotly figure-building method.

    Raises:
        ValueError: If a control key does not name a method parameter.
    """
    declared_controls = dict(controls) if controls else {}

    def decorator(
        fn: Callable[..., "go.Figure"],
    ) -> Callable[..., "go.Figure"]:
        signature = inspect.signature(fn)
        params = [
            parameter
            for name, parameter in signature.parameters.items()
            if name != "self"
        ]
        param_names = {parameter.name for parameter in params}
        for kwarg in declared_controls:
            if kwarg not in param_names:
                raise ValueError(
                    f"@figure({fn.__name__!r}): control key {kwarg!r} is not a "
                    "parameter of the method"
                )

        subject_param: str | None = None
        for parameter in params:
            is_positional = parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
            if is_positional and parameter.name not in declared_controls:
                subject_param = parameter.name
                break

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> "go.Figure":
            from phenotypic.sdk_.viz.figures._theme import apply_theme

            return apply_theme(fn(*args, **kwargs))

        wrapper.__figure_spec__ = FigureSpec(  # type: ignore[attr-defined]
            title=title,
            section=section,
            controls=declared_controls,
            description=description,
            primary=primary,
            name=fn.__name__,
            method=wrapper,
            wants_subject=subject_param is not None,
            subject_param=subject_param,
            order=next(_FIGURE_ORDER),
        )
        return wrapper

    return decorator


class BoundFigures:
    """Transient subject binding for a :class:`PhtPlot`.

    Image subjects are held weakly so a notebook widget cannot extend the
    lifetime of a complete plate image. Aggregate subjects such as measurement
    tables are held strongly because they are compact, reusable plot inputs.
    Figures are rendered on demand rather than cached across control values.

    Args:
        provider: Plot provider whose figures will be rendered.
        subject: Runtime subject bound to those figures.
    """

    def __init__(self, provider: PhtPlot, subject: Any) -> None:
        self._provider = provider
        self._subject: Any = None
        self._subject_ref: weakref.ReferenceType[Any] | None = None
        if provider._weakly_bind_subject and subject is not None:
            self._subject_ref = weakref.ref(subject)
        else:
            self._subject = subject

    @property
    def subject(self) -> Any:
        """Return the bound runtime subject."""
        if self._subject_ref is not None:
            subject = self._subject_ref()
            if subject is None:
                raise RuntimeError(
                    "The image bound to this plotting report has been released. "
                    "Keep the image alive while interacting with the report."
                )
            return subject
        return self._subject

    def specs(self) -> list[FigureSpec]:
        """Return provider figures in definition order."""
        return self._provider.iter_figures()

    def render(self, spec: FigureSpec, **control_values: Any) -> "go.Figure":
        """Render a figure for the supplied control values.

        Args:
            spec: Figure metadata to render.
            **control_values: Control values passed to the figure method.

        Returns:
            The themed Plotly figure.
        """
        return self._provider._render_spec(
            spec, self.subject, **control_values
        )


class PhtPlot:
    """Methods-only mixin for saveable figures and complete reports.

    ``PhtPlot`` has no fields, constructor, abstract methods, or persistent
    instance state. It can therefore be combined with an existing Pydantic
    model without changing that model's schema or serialization.

    Subject-taking figure methods receive a subject passed to ``inspect`` or
    ``report``. Helpers that already hold their subject may instead override
    :meth:`_figure_subject` and declare subject-free figure methods.
    """

    _weakly_bind_subject = False

    def _figure_subject(self) -> Any:
        """Return a held figure subject, if the provider owns one."""
        return None

    def _resolve_subject(self, subject: Any) -> Any:
        """Resolve a call-time subject before falling back to held state."""
        return subject if subject is not None else self._figure_subject()

    def iter_figures(self) -> list[FigureSpec]:
        """Return all visible figure specs in definition order.

        Figure discovery follows normal Python override rules across the MRO.
        An undecorated override removes an inherited figure, while a decorated
        override retains the inherited figure's position.
        """
        specs: dict[str, FigureSpec] = {}
        orders: dict[str, int] = {}
        shadowed: set[str] = set()
        for index, klass in enumerate(type(self).__mro__):
            for name, attr in vars(klass).items():
                if name in shadowed:
                    continue
                shadowed.add(name)
                spec = getattr(attr, "__figure_spec__", None)
                if spec is not None:
                    specs[name] = spec
                    orders[name] = self._inherited_figure_order(
                        name, spec.order, klass, index
                    )
        return sorted(specs.values(), key=lambda spec: orders[spec.name])

    def _inherited_figure_order(
        self,
        name: str,
        fallback: int,
        selected_class: type,
        selected_index: int,
    ) -> int:
        """Return the inherited definition slot for a selected override."""
        if selected_class is type(self):
            ancestors = type(self).__mro__[selected_index + 1 :]
        else:
            ancestors = selected_class.__mro__[1:]

        for klass in ancestors:
            if name not in vars(klass):
                continue
            ancestor_spec = getattr(vars(klass)[name], "__figure_spec__", None)
            return (
                ancestor_spec.order if ancestor_spec is not None else fallback
            )
        return fallback

    def _primary_spec(self) -> FigureSpec:
        """Return the explicit primary figure or the only declared figure."""
        specs = self.iter_figures()
        if not specs:
            raise RuntimeError(
                f"{type(self).__name__} declares no @figure methods"
            )
        primaries = [spec for spec in specs if spec.primary]
        if primaries:
            return primaries[0]
        if len(specs) == 1:
            return specs[0]
        raise RuntimeError(
            f"{type(self).__name__} has multiple @figure methods but none is "
            "marked primary=True; cannot pick an inspect() figure"
        )

    def _render_spec(
        self,
        spec: FigureSpec,
        subject: Any = None,
        **control_values: Any,
    ) -> "go.Figure":
        """Render one figure spec with its resolved subject and controls."""
        method = getattr(self, spec.name)
        if spec.wants_subject:
            return method(self._resolve_subject(subject), **control_values)
        return method(**control_values)

    def inspect(
        self,
        subject: Any = None,
        *,
        for_save: bool = False,
        **overrides: Any,
    ) -> Any:
        """Return the primary saveable figure.

        Args:
            subject: Runtime subject, or ``None`` to use held subject state.
            for_save: Forwarded when the selected figure accepts it.
            **overrides: Values overriding declared control defaults.

        Returns:
            The themed primary figure. Multi-page producers may override this
            method and return a runtime plotting output.

        Raises:
            ValueError: If an override is not a declared control.
        """
        spec = self._primary_spec()
        method = getattr(self, spec.name)
        valid_params = set(inspect.signature(method).parameters)
        unknown = set(overrides) - set(spec.controls)
        if unknown:
            raise ValueError(
                f"inspect(): unknown override(s) {sorted(unknown)} for figure "
                f"{spec.name!r}; valid controls: {sorted(spec.controls)}"
            )
        kwargs = {
            kwarg: control.default for kwarg, control in spec.controls.items()
        }
        kwargs.update(overrides)
        if "for_save" in valid_params:
            kwargs["for_save"] = for_save
        if spec.wants_subject:
            return method(self._resolve_subject(subject), **kwargs)
        return method(**kwargs)

    def report(self, subject: Any = None, **overrides: Any) -> Any:
        """Return the complete composed or interactive report.

        Args:
            subject: Runtime subject, or ``None`` to use held subject state.
            **overrides: Reserved for specialized report implementations.

        Returns:
            A composed Plotly figure when all figures are control-free, or an
            ipywidgets report when any figure declares controls.

        Raises:
            RuntimeError: If no figure methods are declared.
            ValueError: If the base report receives overrides. Concrete plots
                may override this method to expose report-specific parameters.
        """
        if overrides:
            raise ValueError(
                f"report(): override(s) {sorted(overrides)} require a "
                "plot-specific report implementation"
            )
        specs = self.iter_figures()
        if not specs:
            raise RuntimeError(
                f"{type(self).__name__} declares no @figure methods"
            )
        if any(spec.controls for spec in specs):
            from phenotypic.sdk_.viz.notebook._adapter import (
                build_notebook_dashboard,
            )

            return build_notebook_dashboard(self, subject)
        return self._compose_control_free_figure(subject)

    def figures(self, subject: Any = None) -> BoundFigures:
        """Bind a subject to figures for use by a renderer adapter."""
        return BoundFigures(self, subject)

    def _compose_control_free_figure(self, subject: Any = None) -> "go.Figure":
        """Compose all control-free figures into one Plotly figure.

        A single figure is returned without re-wrapping so its existing layout
        and faceting remain intact. Multiple figures are stacked vertically in
        definition order.
        """
        from plotly.subplots import make_subplots

        from phenotypic.sdk_.viz.figures._theme import apply_theme

        specs = self.iter_figures()
        if len(specs) == 1:
            return self._render_spec(specs[0], subject)
        composed = make_subplots(
            rows=len(specs),
            cols=1,
            subplot_titles=[spec.title for spec in specs],
        )
        for row, spec in enumerate(specs, start=1):
            rendered = self._render_spec(spec, subject)
            for trace in rendered.data:
                composed.add_trace(trace, row=row, col=1)
        return apply_theme(composed)
