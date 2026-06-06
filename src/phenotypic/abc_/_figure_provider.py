"""Reusable visualization protocol: ``Control``, ``FigureSpec``, ``@figure``,
``FigureProvider``.

This is the renderer-neutral *contract* layer of PhenoTypic's plotting stack. It
turns ``@figure``-decorated methods into a uniform ``.inspect()`` / ``.dash()`` /
dashboard surface that works on both plain helper classes and pydantic
``ImageOperation`` models.

Design notes
------------
* **Stdlib only at import time.** This module imports nothing heavier than the
  standard library. ``plotly`` is touched *lazily* (the ``@figure`` wrapper
  applies the house theme only when a figure is actually rendered), and the
  ipywidgets/Dash shells are imported only inside :meth:`FigureProvider.dash`.
  An import-rule test enforces that importing this module pulls in no UI toolkit.
* **Pydantic-safe.** :class:`FigureProvider` is a methods-only mixin — no fields,
  no ``__init__``, no class-level annotations — so pydantic's ``ModelMetaclass``
  adds nothing and ``model_fields`` / ``model_json_schema()`` / ``to_json()`` are
  unchanged. It mirrors the proven ``LazyWidgetMixin`` shape. Any transient
  per-render cache lives on the throwaway :class:`BoundFigures` returned by
  :meth:`FigureProvider.figures`, never on the model.
* **Opt-in.** Mix :class:`FigureProvider` into the specific classes that declare
  ``@figure`` methods. It is intentionally *not* placed on ``BaseOperation`` — the
  CLI discovers ``--save-inspect`` targets via ``hasattr(measurer, "inspect")``,
  so a base-level ``inspect`` would mis-trigger on every operation.

Controls are bound to a figure method's keyword argument **by identity**: the same
``Control`` instance referenced by several methods becomes one shared widget;
distinct instances (even with the same label) are independent widgets.
"""

from __future__ import annotations

import functools
import inspect
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import plotly.graph_objects as go

__all__ = [
    "Control",
    "FigureSpec",
    "figure",
    "FigureProvider",
    "BoundFigures",
]

# Monotonic counter giving each @figure its definition-order index. Methods in a
# class body are decorated top-to-bottom, so this yields the authoring order
# (``dir()`` / ``vars()`` iteration order is not reliable for this).
_FIGURE_ORDER = itertools.count()

ControlKind = Literal["float", "select", "bool", "text"]


@dataclass(frozen=True)
class Control:
    """Renderer-neutral input bound to a figure method's keyword argument.

    Bound **by identity**: the same instance referenced by several ``@figure``
    methods renders as one shared widget; distinct instances (even with the same
    ``label``) are independent widgets. There is no global name namespace, so two
    unrelated figures may each have a ``sigma`` kwarg without colliding.

    Args:
        label: Human-readable widget label.
        kind: One of ``"float"``, ``"select"``, ``"bool"``, ``"text"``.
        default: Initial value. Must be consistent with ``kind`` (within
            ``bounds`` for ``float``; a member of ``options`` for ``select``; a
            ``bool`` for ``bool``).
        bounds: ``(low, high)`` range — **required** for ``"float"``.
        step: Optional slider step for ``"float"``.
        options: Allowed values — **required** for ``"select"``.
        help: Optional tooltip text.

    Raises:
        ValueError: If the kind-specific requirements are not met.
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
        else:  # pragma: no cover - guarded by the Literal type, defensive only
            raise ValueError(
                f"Control({self.label!r}): unknown kind {self.kind!r}"
            )


@dataclass(frozen=True)
class FigureSpec:
    """Introspectable metadata attached to a ``@figure`` method.

    Author-supplied fields come from the decorator call; the rest are derived by
    the decorator from the wrapped method's signature.

    Attributes:
        title: Figure title (author).
        section: Flat grouping tag → one collapsible card (author).
        controls: ``{method-kwarg: Control}`` (author).
        description: Optional interpretive block, e.g. a ``PanelDescription``
            (author). Typed ``Any`` to keep this module dependency-free.
        primary: Marks the figure returned by :meth:`FigureProvider.inspect`
            (author).
        name: The method's ``__name__`` (decorator).
        method: The wrapped, auto-styled callable (decorator).
        wants_subject: Whether the method takes a subject as its first
            positional parameter (decorator).
        subject_param: Name of that subject parameter, or ``None`` (decorator).
        order: Definition-order index used to sort figures (decorator).
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
    """Mark a method as a figure builder and auto-apply the house Plotly theme.

    The decorated method returns a raw ``plotly.graph_objects.Figure``, accepts
    each control as a keyword argument, and **may** accept a subject as its first
    positional parameter (any positional-or-keyword parameter that is not itself
    a control). The decorator:

    * validates that every ``controls`` key names a real parameter,
    * detects ``wants_subject`` / ``subject_param`` from the signature,
    * stashes a :class:`FigureSpec` on ``fn.__figure_spec__`` (invisible to
      pydantic), and
    * wraps the method so its returned figure is restyled via the ``"phenotypic"``
      template (imported lazily, so this module stays UI-toolkit-free).

    Args:
        title: Figure title.
        section: Flat grouping tag for collapsible cards.
        controls: ``{method-kwarg: Control}`` recompute inputs.
        description: Optional interpretive block.
        primary: Mark this as the ``inspect()`` figure.

    Returns:
        A decorator that returns the wrapped, theme-applying method.

    Raises:
        ValueError: If a ``controls`` key is not a parameter of the method.
    """
    controls = dict(controls) if controls else {}

    def decorator(
        fn: Callable[..., "go.Figure"],
    ) -> Callable[..., "go.Figure"]:
        sig = inspect.signature(fn)
        params = [p for name, p in sig.parameters.items() if name != "self"]
        param_names = {p.name for p in params}
        for kwarg in controls:
            if kwarg not in param_names:
                raise ValueError(
                    f"@figure({fn.__name__!r}): control key {kwarg!r} is not a "
                    f"parameter of the method"
                )

        # Subject = first positional(-or-keyword) parameter that is not a control.
        subject_param: str | None = None
        for p in params:
            positional = p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
            if positional and p.name not in controls:
                subject_param = p.name
                break

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> "go.Figure":
            from phenotypic.viz.figures._theme import apply_theme

            return apply_theme(fn(*args, **kwargs))

        wrapper.__figure_spec__ = FigureSpec(  # type: ignore[attr-defined]
            title=title,
            section=section,
            controls=controls,
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
    """A subject bound to a provider's figures — the transient renderable the GUI
    Dash adapter consumes.

    The per-render cache lives **here**, not on the provider, so a pydantic model
    stays free of transient state. Discard after use.
    """

    def __init__(self, provider: "FigureProvider", subject: Any) -> None:
        self._provider = provider
        self._subject = subject
        self._cache: dict[tuple[Any, ...], "go.Figure"] = {}

    @property
    def subject(self) -> Any:
        """The bound subject (may be ``None`` for helpers that hold their own)."""
        return self._subject

    def specs(self) -> list[FigureSpec]:
        """The provider's figures, sorted by definition order."""
        return self._provider.iter_figures()

    def render(self, spec: FigureSpec, **control_values: Any) -> "go.Figure":
        """Render ``spec`` with the given control values, caching by (name, values).

        Args:
            spec: The figure to render.
            **control_values: Control keyword values (simple scalars).

        Returns:
            The themed ``plotly.graph_objects.Figure``.
        """
        key = (spec.name, *sorted(control_values.items()))
        if key not in self._cache:
            self._cache[key] = self._provider._render_spec(
                spec, self._subject, **control_values
            )
        return self._cache[key]


class FigureProvider:
    """Mixin turning ``@figure`` methods into ``.inspect()`` / ``.dash()`` / a
    dashboard.

    Methods only — no fields, no ``__init__``, no instance state — so it is safe
    to mix into any ``pydantic.BaseModel`` in any MRO position (mirrors
    ``LazyWidgetMixin``). Mix it into the concrete class that owns the ``@figure``
    methods::

        class DiagnosticsPlotter(BasePlotter, FigureProvider): ...
        class MeasureSymmetricZones(MeasureFeatures, FigureProvider): ...

    Subject binding:
        * **Helpers** hold their subject — override :meth:`_figure_subject` to
          return it; their ``@figure`` methods read ``self`` and take no subject.
        * **Operations** pass the subject at call time (``op.dash(image)``); their
          ``@figure`` methods take it as the first positional parameter.
    """

    # -- subject resolution -------------------------------------------------

    def _figure_subject(self) -> Any:
        """Subject for subject-taking ``@figure`` methods.

        Helpers override to return their held state; operations leave this
        ``None`` and pass the subject at call time.
        """
        return None

    def _resolve_subject(self, subject: Any) -> Any:
        """Return ``subject`` if given, else the held :meth:`_figure_subject`."""
        return subject if subject is not None else self._figure_subject()

    # -- introspection ------------------------------------------------------

    def iter_figures(self) -> list[FigureSpec]:
        """All ``@figure`` specs on this instance's class, in definition order.

        Walks the MRO so inherited figures are included; the most-derived
        definition of a given method name wins, but keeps the *earliest*
        definition's position so a subclass override stays in its original slot
        instead of jumping to the end of the list.
        """
        specs: dict[str, FigureSpec] = {}
        first_order: dict[str, int] = {}
        for klass in type(self).__mro__:  # MRO runs derived → base
            for name, attr in vars(klass).items():
                spec = getattr(attr, "__figure_spec__", None)
                if spec is None:
                    continue
                if name not in specs:  # first hit = most-derived definition
                    specs[name] = spec
                first_order[name] = min(
                    first_order.get(name, spec.order), spec.order
                )
        return sorted(specs.values(), key=lambda s: first_order[s.name])

    def _primary_spec(self) -> FigureSpec:
        """The figure ``inspect()`` returns: ``primary=True``, or the sole one."""
        specs = self.iter_figures()
        if not specs:
            raise RuntimeError(
                f"{type(self).__name__} declares no @figure methods"
            )
        primaries = [s for s in specs if s.primary]
        if primaries:
            return primaries[0]
        if len(specs) == 1:
            return specs[0]
        raise RuntimeError(
            f"{type(self).__name__} has multiple @figure methods but none is "
            f"marked primary=True; cannot pick an inspect() figure"
        )

    # -- rendering ----------------------------------------------------------

    def _render_spec(
        self, spec: FigureSpec, subject: Any = None, **control_values: Any
    ) -> "go.Figure":
        """Call ``spec``'s bound method, injecting the subject when required."""
        method = getattr(self, spec.name)
        if spec.wants_subject:
            return method(self._resolve_subject(subject), **control_values)
        return method(**control_values)

    def inspect(
        self, subject: Any = None, *, for_save: bool = False, **overrides: Any
    ) -> "go.Figure":
        """The primary saveable ``go.Figure``.

        This is the default ``inspect()`` contract used by control-free helpers
        and by the CLI's ``--save-inspect``. Classes that hand-write their own
        ``inspect`` (e.g. ``MeasureSymmetricZones``) override this naturally.

        Args:
            subject: Subject to render against (operations); ``None`` uses the
                held subject (helpers).
            for_save: Forwarded to the figure method if it accepts ``for_save``
                (so legend-only layers can be flattened for a static raster).
            **overrides: Override control defaults by keyword.

        Returns:
            The themed primary ``plotly.graph_objects.Figure``.
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
        kwargs: dict[str, Any] = {
            kw: c.default for kw, c in spec.controls.items()
        }
        kwargs.update(overrides)
        if "for_save" in valid_params:
            kwargs["for_save"] = for_save
        if spec.wants_subject:
            return method(self._resolve_subject(subject), **kwargs)
        return method(**kwargs)

    def dash(self, subject: Any = None) -> Any:
        """The interactive view.

        * No ``Control`` anywhere → a composed subplot ``go.Figure`` (preserving
          the repo-wide ``.dash() -> go.Figure`` contract).
        * Any ``Control`` present → the ipywidgets notebook dashboard.

        Args:
            subject: Subject to bind (operations); ``None`` uses the held subject.

        Returns:
            A ``go.Figure`` (control-free) or an ipywidgets widget (controls).
        """
        specs = self.iter_figures()
        if not specs:
            raise RuntimeError(
                f"{type(self).__name__} declares no @figure methods"
            )
        if any(s.controls for s in specs):
            from phenotypic.viz.notebook._adapter import (
                build_notebook_dashboard,
            )

            return build_notebook_dashboard(self, subject)
        return self._compose_control_free_figure(subject)

    def figures(self, subject: Any = None) -> BoundFigures:
        """Bind ``subject`` → a transient :class:`BoundFigures` the Dash adapter
        consumes. The per-render cache lives on the returned object, not here."""
        return BoundFigures(self, subject)

    def _compose_control_free_figure(self, subject: Any = None) -> "go.Figure":
        """Stack every control-free figure into one vertically-stacked subplot
        ``go.Figure`` (the default control-free ``dash()`` rendering).

        A provider that exposes a **single** control-free figure (e.g. a
        detect-modes faceted figure that is already a subplot grid) is returned
        as-is, with its own layout intact — no re-wrapping.

        Note:
            For multiple figures, only traces are carried over; per-subfigure
            layout (axis titles, ranges, colorbars, annotations) is not
            propagated. Dashboards needing richer multi-figure composition should
            override ``dash()``.
        """
        from plotly.subplots import make_subplots

        from phenotypic.viz.figures._theme import apply_theme

        specs = self.iter_figures()
        if len(specs) == 1:
            # single figure → return it directly so its layout/faceting survives
            return self._render_spec(specs[0], subject)
        titles = [s.title for s in specs]
        composed = make_subplots(
            rows=len(specs), cols=1, subplot_titles=titles
        )
        for row, spec in enumerate(specs, start=1):
            sub = self._render_spec(spec, subject)
            for trace in sub.data:
                composed.add_trace(trace, row=row, col=1)
        return apply_theme(composed)
