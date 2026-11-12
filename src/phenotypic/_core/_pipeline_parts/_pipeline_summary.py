"""The table behind :meth:`ImagePipelineCore.summary`.

One row per step the pipeline runs, in run order, built from the pipeline
alone: no image is loaded and nothing is applied. Paths come from
:func:`phenotypic.sdk_._operation_tree.walk_operations`, so a step is spelled
exactly as provenance journals and the staged-GPU refusals spell it.
"""

from __future__ import annotations

import enum
import math
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from phenotypic.abc_ import (
    BaseOperation,
    ImageCorrector,
    ImageEnhancer,
    MeasureFeatures,
    ObjectDetector,
    ObjectRefiner,
)
from phenotypic.enhance._set_detect_mode import SetDetectMode
from phenotypic.sdk_._operation_tree import (
    _SLOT_SEPARATOR,
    pipeline_slot_of,
    walk_operations,
)

if TYPE_CHECKING:
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

SUMMARY_COLUMNS = ("step", "slot", "operation", "module", "writes", "changed")

#: The layers a step writes, decided by its base class; the first match wins.
#: An enhancer writes ``detect_mat``, and writing it clears any detected
#: objects (``objmask``/``objmap``), so an enhancer placed after a detector
#: erases them; PhenoTypic warns when a pipeline does that
#: (``DetectionClearedWarning``). ``SetDetectMode`` is the one shipped
#: operation that subclasses ``ImageOperation`` directly yet rewrites a layer
#: (it resets ``detect_mat`` to a fresh channel, leaving detected objects in
#: place), so it is named here rather than reported as writing nothing.
#: Anything else unlisted writes ``""``: not declared, not "none".
_WRITES_BY_BASE: tuple[tuple[type, str], ...] = (
    (ImageEnhancer, "detect_mat"),
    (SetDetectMode, "detect_mat"),
    (ObjectDetector, "objmask, objmap"),
    (ObjectRefiner, "objmask, objmap"),
    (ImageCorrector, "rgb, gray, detect_mat, objmask, objmap"),
    (MeasureFeatures, "table"),
)


def _writes(operation: Any) -> str:
    for base, layers in _WRITES_BY_BASE:
        if isinstance(operation, base):
            return layers
    return ""


def _module(operation: Any) -> str:
    """``enhance`` for ``phenotypic.enhance._blur_gauss``; a custom op's module in full."""
    module = type(operation).__module__
    parts = module.split(".")
    if parts[0] == "phenotypic" and len(parts) > 1:
        return parts[1]
    return module


#: Default-constructed instances, by class, for one ``summarize_pipeline`` call.
#: ``None`` marks a class that cannot be constructed without arguments.
_References = dict[type, Any]


def _reference(cls: type, references: _References) -> Any:
    """A default-constructed *cls*, or ``None`` when it needs arguments."""
    if cls not in references:
        try:
            references[cls] = cls()
        except Exception:  # noqa: BLE001 -- any refusal means "no reference"
            references[cls] = None
    return references[cls]


def _is_generated_name(operation: Any, name: str) -> bool:
    """True for a pipeline's ``name`` left at its ``uuid4`` default factory.

    Every pipeline gets a fresh one, so it would always read as changed; a
    name the user chose is shown.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    if name != "name" or not isinstance(operation, ImagePipelineCore):
        return False
    try:
        uuid.UUID(str(operation.name))
    except ValueError:
        return False
    return True


def _parameter_names(operation: BaseOperation) -> list[str]:
    """The serialized parameters of *operation*, minus a generated pipeline name."""
    return [
        name
        for name in operation.model_dump()
        if not name.startswith("_") and not _is_generated_name(operation, name)
    ]


def _comparable(value: Any, references: _References) -> Any:
    """A plain value equal to another exactly when the two parameters are."""
    if isinstance(value, BaseOperation):
        return (
            type(value).__name__,
            tuple(
                (name, _comparable(getattr(value, name), references))
                for name in _parameter_names(value)
            ),
        )
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(item, references) for item in value)
    if isinstance(value, dict):
        return tuple((key, _comparable(item, references)) for key, item in value.items())
    return value


def _changed_names(operation: BaseOperation, references: _References) -> list[str]:
    """Parameters that differ from a default-constructed instance of the class.

    Comparing against an instance rather than the declared defaults keeps a
    value that a validator fills in from the others (``FilamentousFungiDetector``
    derives eight of its parameters) from reading as a user change. A class
    that cannot be constructed without arguments falls back to the declared
    defaults (``model_dump(exclude_defaults=True)``).
    """
    reference = _reference(type(operation), references)
    if reference is None:
        declared = operation.model_dump(exclude_defaults=True)
        return [name for name in _parameter_names(operation) if name in declared]
    return [
        name
        for name in _parameter_names(operation)
        if _comparable(getattr(operation, name), references)
        != _comparable(getattr(reference, name), references)
    ]


def _format_value(value: Any, references: _References) -> str:
    """Spell *value* as it would be written in a constructor call."""
    if isinstance(value, BaseOperation):
        return f"{type(value).__name__}({_changed_parameters(value, references)})"
    if isinstance(value, enum.Enum):
        return repr(value.value)
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item, references) for item in value) + "]"
    if isinstance(value, tuple):
        trailing = "," if len(value) == 1 else ""
        inner = ", ".join(_format_value(item, references) for item in value)
        return "(" + inner + trailing + ")"
    if isinstance(value, dict):
        items = (f"{key!r}: {_format_value(item, references)}" for key, item in value.items())
        return "{" + ", ".join(items) + "}"
    return repr(value)


def _changed_parameters(operation: BaseOperation, references: _References) -> str:
    """``name=value`` for every parameter not at its default, comma-separated."""
    return ", ".join(
        f"{name}={_format_value(getattr(operation, name), references)}"
        for name in _changed_names(operation, references)
    )


def _row(step: str, slot: str, operation: Any, references: _References) -> dict[str, str]:
    return {
        "step": step,
        "slot": slot,
        "operation": type(operation).__name__,
        "module": _module(operation),
        "writes": _writes(operation),
        "changed": _changed_parameters(operation, references),
    }


def summarize_pipeline(pipeline: ImagePipelineCore) -> pd.DataFrame:
    """One row per step *pipeline* runs, in run order.

    The ``ops`` chain comes first, then the measurers in the order
    ``measure()`` runs them. A pipeline nested in ``ops`` is expanded into its
    own ``ops``: its ``apply`` runs nothing else, so its ``meas`` (and every
    pipeline's ``post``/``filters``/``model``, which transform tables rather
    than images) are not steps here. Any other operation is one row, however
    many operations it holds; a composite's children appear in its
    ``changed`` cell.

    The measurers come from ``_build_measurement_run_order``, not from
    ``meas`` alone, so the ``CenteredAutoGridFinder`` that ``measure()``
    prepends when the pipeline's ``nrows``/``ncols`` preset is set (and
    ``meas`` holds no ``GridFinder``) is listed too.

    Args:
        pipeline: The pipeline to describe. It is only read.

    Returns:
        pd.DataFrame: Columns ``step`` (the tree path, segments joined with
        ``/``; a measurer is ``meas:<key>``), ``slot`` (``ops`` or ``meas``),
        ``operation`` (class name), ``module`` (the ``phenotypic``
        subpackage), ``writes`` (the layers the step writes, or ``table`` for
        a measurer) and ``changed`` (parameters that differ from a
        default-constructed instance, as ``name=value``).
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    references: _References = {}
    nodes: dict[tuple[str, ...], Any] = {(): pipeline}
    # Pipelines whose ``ops`` actually run: the root and any pipeline reached
    # from it through ``ops`` alone.
    expanded: set[tuple[str, ...]] = {()}
    rows: list[dict[str, str]] = []
    for path, operation in walk_operations(pipeline):
        nodes[path] = operation
        parent_path = path[:-1]
        if parent_path not in expanded:
            continue
        if pipeline_slot_of(nodes[parent_path], path[-1]) is not None:
            continue  # a slot entry; the root's measurers are added below
        if isinstance(operation, ImagePipelineCore):
            expanded.add(path)
            continue
        rows.append(_row("/".join(path), "ops", operation, references))
    for key, measurer in pipeline._build_measurement_run_order().items():
        rows.append(_row(f"meas{_SLOT_SEPARATOR}{key}", "meas", measurer, references))
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
