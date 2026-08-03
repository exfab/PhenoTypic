"""Runtime plot bindings and dynamically resolved input declarations.

This module is deliberately independent from the pipeline implementation.  A
pipeline supplies its object registry when bindings are normalized or restored,
which keeps the plotting package usable without importing ``phenotypic._core``.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PipelineObjectRef(BaseModel):
    """Reference one object owned by an :class:`ImagePipeline` slot.

    Args:
        slot: Pipeline collection containing the object.
        key: Collection key. The singleton ``model`` slot does not use a key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: Literal["ops", "meas", "post", "filters", "model", "qc"]
    key: str | None = None

    @model_validator(mode="after")
    def _validate_key(self) -> "PipelineObjectRef":
        if self.slot == "model" and self.key is not None:
            raise ValueError("the model plot reference must not define a key")
        if self.slot != "model" and not self.key:
            raise ValueError(f"the {self.slot!r} plot reference requires a key")
        return self


class MeasurementInput(BaseModel):
    """Select the current post-applied measurement mirror."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["measurements"] = "measurements"


class AnalysisInput(BaseModel):
    """Select one named analysis table.

    Args:
        analysis_id: Safe identifier published in ``analysis_manifest.json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["analysis"] = "analysis"
    analysis_id: str

    @field_validator("analysis_id")
    @classmethod
    def _validate_analysis_id(cls, value: str) -> str:
        from ._analysis_artifacts import validate_analysis_id

        return validate_analysis_id(value)


PlotInput = MeasurementInput | AnalysisInput


class PlotBinding(BaseModel):
    """Wire a plot-capable object to a pipeline lifecycle.

    ``plot`` is the resolved live object and is intentionally excluded from
    generic Pydantic serialization. Pipeline JSON uses
    :func:`serialize_plot_binding`, which records either :attr:`ref` or an
    importable inline class plus its validated settings.

    Args:
        id: Stable output and component identifier.
        plot: Live plot-capable object.
        ref: Optional reference to a pipeline-owned object.
        input: Dynamic aggregate input selection for analysis and QC plots.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: str
    plot: Any = Field(default=None, exclude=True)
    ref: PipelineObjectRef | None = None
    input: PlotInput | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("plot binding id must not be empty")
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError(
                "plot binding id must be one safe path component without separators"
            )
        return value

    @model_validator(mode="after")
    def _require_target(self) -> "PlotBinding":
        if self.plot is None and self.ref is None:
            raise ValueError("plot binding requires either plot or ref")
        return self


ObjectRegistry = dict[tuple[str, str | None], Any]


def normalize_plot_bindings(
    entries: list[Any] | None,
    registry: ObjectRegistry,
) -> list[PlotBinding]:
    """Normalize raw plot objects and explicit bindings against a registry.

    Identity, rather than equality, decides whether a raw object becomes a
    reference. This is load-bearing for Pydantic models because two distinct
    models can compare equal and serialize to identical parameter dictionaries.

    Args:
        entries: Raw plot-capable objects or explicit bindings.
        registry: Mapping from pipeline slot/key pairs to live objects.

    Returns:
        Ordered resolved bindings.

    Raises:
        TypeError: If an entry is not plot-capable.
        ValueError: If a reference cannot resolve or IDs/lifecycles are invalid.
    """
    from phenotypic.abc_.plotting import (
        PhtPlot,
        PlotAnalysis,
        PlotImage,
        PlotMeas,
        PlotQc,
    )

    normalized: list[PlotBinding] = []
    for raw in entries or []:
        is_qc_plot_ref = False
        if isinstance(raw, PlotBinding):
            binding = raw.model_copy(deep=False)
            if binding.ref is not None:
                key = (binding.ref.slot, binding.ref.key)
                if key not in registry:
                    raise ValueError(
                        f"plot {binding.id!r} references missing pipeline object "
                        f"{binding.ref.slot}.{binding.ref.key or ''}".rstrip(".")
                    )
                binding.plot = registry[key]
            elif binding.plot is not None:
                ref = _identity_ref(binding.plot, registry)
                if ref is not None:
                    binding.ref = ref
        else:
            if not isinstance(raw, PhtPlot):
                raise TypeError(
                    "plots entries must inherit PhtPlot or be PlotBinding instances; "
                    f"got {type(raw).__name__}"
                )
            ref = _identity_ref(raw, registry)
            default_id = (
                ref.key
                if ref is not None and ref.key is not None
                else type(raw).__name__
            )
            binding = PlotBinding(id=default_id, plot=raw, ref=ref)

        if not isinstance(binding.plot, PhtPlot):
            # QC recipe references point at configuration entries rather than
            # persistent check instances. They are resolved by the QC runner.
            if binding.ref is None or binding.ref.slot != "qc":
                raise TypeError(
                    f"plot {binding.id!r} resolved to non-PhtPlot "
                    f"{type(binding.plot).__name__}"
                )
            plot_class = getattr(binding.plot, "cls", None)
            if not isinstance(plot_class, type) or not issubclass(
                plot_class, PlotQc
            ):
                class_name = getattr(plot_class, "__name__", repr(plot_class))
                raise ValueError(
                    f"plot {binding.id!r} references QC class {class_name} "
                    "which does not implement PlotQc"
                )
            lifecycles = [
                lifecycle.__name__
                for lifecycle in (PlotImage, PlotMeas, PlotAnalysis, PlotQc)
                if issubclass(plot_class, lifecycle)
            ]
            if lifecycles != ["PlotQc"]:
                raise ValueError(
                    f"plot {binding.id!r} QC reference must implement exactly "
                    f"the PlotQc lifecycle; found {lifecycles}"
                )
            is_qc_plot_ref = True
        else:
            lifecycles = [
                lifecycle.__name__
                for lifecycle in (PlotImage, PlotMeas, PlotAnalysis, PlotQc)
                if isinstance(binding.plot, lifecycle)
            ]
            if len(lifecycles) != 1:
                raise ValueError(
                    f"plot {binding.id!r} must implement exactly one actionable "
                    "plot lifecycle (PlotImage, PlotMeas, PlotAnalysis, or "
                    f"PlotQc); found {lifecycles or ['none']}"
                )

        if is_qc_plot_ref or isinstance(binding.plot, (PlotAnalysis, PlotQc)):
            if binding.input is None:
                binding.input = MeasurementInput()
        elif binding.input is not None:
            lifecycle = type(binding.plot).__name__
            if isinstance(binding.plot, PlotMeas):
                detail = "PlotMeas always consumes the measurement mirror"
            else:
                detail = f"{lifecycle} does not accept an aggregate input"
            raise ValueError(f"plot {binding.id!r}: {detail}")

        normalized.append(binding)

    ids = [binding.id for binding in normalized]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate plot binding ids: {duplicates}")
    from ._writer import safe_path_component

    path_ids: dict[str, str] = {}
    collisions: set[tuple[str, str]] = set()
    for binding in normalized:
        component = safe_path_component(binding.id).casefold()
        previous = path_ids.get(component)
        if previous is not None and previous != binding.id:
            first, second = sorted((previous, binding.id))
            collisions.add((first, second))
        else:
            path_ids[component] = binding.id
    if collisions:
        formatted = [list(pair) for pair in sorted(collisions)]
        raise ValueError(
            "plot binding ids collide after filesystem sanitization: "
            f"{formatted}"
        )
    return normalized


def serialize_plot_binding(binding: PlotBinding) -> dict[str, Any]:
    """Serialize one resolved binding without copying its live plot object."""
    payload: dict[str, Any] = {"id": binding.id}
    if binding.ref is not None:
        payload["ref"] = binding.ref.model_dump(mode="json", exclude_none=True)
    else:
        plot = binding.plot
        qualname = type(plot).__qualname__
        if "<locals>" in qualname:
            raise ValueError(
                f"plot {binding.id!r} uses local class {qualname!r}; "
                "move it to an importable module"
            )
        if not hasattr(plot, "model_dump"):
            raise TypeError(
                f"inline plot {binding.id!r} must be a Pydantic model so its "
                "settings can be serialized"
            )
        payload["inline"] = {
            "module": type(plot).__module__,
            "qualname": qualname,
            "params": plot.model_dump(mode="json"),
        }
    if binding.input is not None:
        payload["input"] = binding.input.model_dump(mode="json")
    return payload


def deserialize_plot_bindings(
    entries: list[dict[str, Any]] | None,
    registry: ObjectRegistry,
    *,
    skipped_refs: frozenset[tuple[str, str | None]] = frozenset(),
    skipped_inline: list[tuple[str, str]] | None = None,
) -> list[PlotBinding]:
    """Restore bindings after all regular pipeline objects exist.

    Args:
        entries: Serialized plot binding entries.
        registry: Live pipeline object registry.
        skipped_refs: References known to target analyzer or QC entries that
            tolerant pipeline loading already skipped.
        skipped_inline: Optional sink for unresolved inline plot classes as
            ``(binding_id, qualified_class_name)`` pairs. When omitted,
            unresolved classes raise normally.
    """
    raw_bindings: list[PlotBinding] = []
    for entry in entries or []:
        input_payload = entry.get("input")
        input_ref: PlotInput | None = None
        if input_payload is not None:
            kind = input_payload.get("kind")
            if kind == "measurements":
                input_ref = MeasurementInput.model_validate(input_payload)
            elif kind == "analysis":
                input_ref = AnalysisInput.model_validate(input_payload)
            else:
                raise ValueError(
                    f"plot {entry.get('id', '?')!r} has unknown input kind {kind!r}"
                )

        if "ref" in entry:
            ref = PipelineObjectRef.model_validate(entry["ref"])
            if (ref.slot, ref.key) in skipped_refs:
                continue
            raw_bindings.append(
                PlotBinding(
                    id=entry["id"],
                    ref=ref,
                    input=input_ref,
                )
            )
            continue

        inline = entry.get("inline")
        if not isinstance(inline, dict):
            raise ValueError(
                f"plot {entry.get('id', '?')!r} requires ref or inline settings"
            )
        qualified_name = f"{inline['module']}.{inline['qualname']}"
        try:
            plot_class = _load_qualified_class(
                module_name=inline["module"],
                qualname=inline["qualname"],
            )
        except (ImportError, AttributeError):
            if skipped_inline is None:
                raise
            skipped_inline.append((entry["id"], qualified_name))
            continue
        if not hasattr(plot_class, "model_validate"):
            raise TypeError(
                f"inline plot class {inline['module']}.{inline['qualname']} "
                "is not a Pydantic model"
            )
        raw_bindings.append(
            PlotBinding(
                id=entry["id"],
                plot=plot_class.model_validate(inline.get("params", {})),
                input=input_ref,
            )
        )
    return normalize_plot_bindings(raw_bindings, registry)


def _identity_ref(
    plot: Any,
    registry: ObjectRegistry,
) -> PipelineObjectRef | None:
    for (slot, key), candidate in registry.items():
        if candidate is plot:
            return PipelineObjectRef(slot=slot, key=key)  # type: ignore[arg-type]
    return None


def _load_qualified_class(*, module_name: str, qualname: str) -> type[Any]:
    if "<locals>" in qualname:
        raise ValueError(f"local plot class {qualname!r} cannot be deserialized")
    module = importlib.import_module(module_name)
    target: Any = module
    for component in qualname.split("."):
        target = getattr(target, component)
    if not isinstance(target, type):
        raise TypeError(f"{module_name}.{qualname} does not resolve to a class")
    return target


__all__ = [
    "AnalysisInput",
    "MeasurementInput",
    "PipelineObjectRef",
    "PlotBinding",
    "PlotInput",
    "deserialize_plot_bindings",
    "normalize_plot_bindings",
    "serialize_plot_binding",
]
