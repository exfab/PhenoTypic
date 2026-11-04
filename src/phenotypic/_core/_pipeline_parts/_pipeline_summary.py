"""The table behind :meth:`ImagePipelineCore.summary`.

One row per step the pipeline runs, in run order, built from the pipeline
alone: no image is loaded and nothing is applied. Paths come from
:func:`phenotypic.sdk_._operation_tree.walk_operations`, so a step is spelled
exactly as provenance journals and the staged-GPU refusals spell it.
"""

from __future__ import annotations

import enum
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
from phenotypic.sdk_._operation_tree import pipeline_slot_of, walk_operations

if TYPE_CHECKING:
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

SUMMARY_COLUMNS = ("step", "slot", "operation", "module", "writes", "changed")

#: The layers a step writes, decided by its base class; the first match wins.
#: ``SetDetectMode`` is the one shipped operation that subclasses
#: ``ImageOperation`` directly yet rewrites a layer (it replaces ``detect_mat``
#: with a fresh channel), so it is named here rather than reported as writing
#: nothing. Anything else unlisted writes ``""``: not declared, not "none".
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


def _format_value(value: Any) -> str:
    """Spell *value* as it would be written in a constructor call."""
    if isinstance(value, BaseOperation):
        return f"{type(value).__name__}({_changed_parameters(value)})"
    if isinstance(value, enum.Enum):
        return repr(value.value)
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, tuple):
        trailing = "," if len(value) == 1 else ""
        return "(" + ", ".join(_format_value(item) for item in value) + trailing + ")"
    if isinstance(value, dict):
        items = (f"{key!r}: {_format_value(item)}" for key, item in value.items())
        return "{" + ", ".join(items) + "}"
    return repr(value)


def _changed_parameters(operation: BaseOperation) -> str:
    """``name=value`` for every parameter not at its default, comma-separated."""
    changed = operation.model_dump(exclude_defaults=True)
    return ", ".join(
        f"{name}={_format_value(getattr(operation, name))}"
        for name in changed
        if not name.startswith("_")
    )


def summarize_pipeline(pipeline: ImagePipelineCore) -> pd.DataFrame:
    """One row per step *pipeline* runs, in run order.

    The ``ops`` chain comes first, then the root's ``meas``. A pipeline nested
    in ``ops`` is expanded into its own ``ops``: its ``apply`` runs nothing
    else, so its ``meas`` (and every pipeline's ``post``/``filters``/``model``,
    which transform tables rather than images) are not steps here. Any other
    operation is one row, however many operations it holds; a composite's
    children appear in its ``changed`` cell.

    Args:
        pipeline: The pipeline to describe. It is only read.

    Returns:
        pd.DataFrame: Columns ``step`` (the tree path, segments joined with
        ``/``), ``slot`` (``ops`` or ``meas``), ``operation`` (class name),
        ``module`` (the ``phenotypic`` subpackage), ``writes`` (the layers the
        step writes, or ``table`` for a measurer) and ``changed`` (non-default
        parameters as ``name=value``).
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

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
        slot = pipeline_slot_of(nodes[parent_path], path[-1])
        if slot is None:
            slot = "ops"
        elif not (slot == "meas" and parent_path == ()):
            continue
        if isinstance(operation, ImagePipelineCore):
            if slot == "ops":
                expanded.add(path)
            continue
        rows.append(
            {
                "step": "/".join(path),
                "slot": slot,
                "operation": type(operation).__name__,
                "module": _module(operation),
                "writes": _writes(operation),
                "changed": _changed_parameters(operation),
            }
        )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
