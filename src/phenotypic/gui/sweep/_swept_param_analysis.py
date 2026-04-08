"""Swept parameter analysis for sweep pipeline configurations.

Analyzes sweep pipeline configs to detect which parameters vary across
pipelines, enabling the viewer to present swept parameters as interactive
controls (sliders for numeric, dropdowns for categorical).

This module is pure Python with **no** Qt or napari dependencies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from ._sweep_data_model import PipelineConfig


@dataclass(frozen=True)
class SweptParameter:
    """A single parameter that varies across sweep pipeline configurations.

    Args:
        operation_name: Instance name of the operation (e.g. ``"GaussianBlur_0"``).
        operation_class: Class name of the operation (e.g. ``"GaussianBlur"``).
        param_name: Name of the parameter that varies (e.g. ``"sigma"``).
        values: Sorted distinct values taken by this parameter across configs.
            Numeric values are sorted ascending; non-numeric values preserve
            insertion order.
        is_numeric_ordered: ``True`` when all values are numeric (int or float,
            excluding bool), indicating a slider is appropriate.  ``False``
            otherwise, indicating a dropdown is appropriate.
    """

    operation_name: str
    operation_class: str
    param_name: str
    values: tuple
    is_numeric_ordered: bool


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def detect_swept_parameters(
    configs: Dict[str, PipelineConfig],
) -> List[SweptParameter]:
    """Detect parameters that vary across sweep pipeline configurations.

    Args:
        configs: Mapping of pipeline name to :class:`PipelineConfig`, as
            stored in :pyattr:`SweepOutputData.pipeline_configs`.

    Returns:
        List of :class:`SweptParameter` sorted by
        ``(operation_name, param_name)``.

    Each operation instance is identified by its ``name`` key (e.g.
    ``"GaussianBlur_0"``).  A parameter is considered *swept* if it takes
    more than one distinct value across all pipeline configs that contain
    that operation.
    """
    if not configs:
        return []

    # Collect all values for each (op_name, op_class, param_name) triple.
    # value_lists preserves insertion order per key.
    value_lists: Dict[
        Tuple[str, str, str], list
    ] = defaultdict(list)

    for _pipe_name, cfg in configs.items():
        for op in cfg.operations:
            op_name = op["name"]
            op_class = op["class"]
            params = op.get("params", {})
            for param_name, param_value in params.items():
                key = (op_name, op_class, param_name)
                value_lists[key].append(param_value)

        for meas in cfg.measurements:
            op_name = meas["name"]
            op_class = meas["class"]
            params = meas.get("params", {})
            for param_name, param_value in params.items():
                key = (op_name, op_class, param_name)
                value_lists[key].append(param_value)

    swept: List[SweptParameter] = []

    for (op_name, op_class, param_name), raw_values in value_lists.items():
        distinct = _unique_ordered(raw_values)
        if len(distinct) <= 1:
            continue

        is_numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in distinct
        )

        if is_numeric:
            sorted_values = tuple(sorted(distinct))
        else:
            sorted_values = tuple(distinct)

        swept.append(
            SweptParameter(
                operation_name=op_name,
                operation_class=op_class,
                param_name=param_name,
                values=sorted_values,
                is_numeric_ordered=is_numeric,
            )
        )

    swept.sort(key=lambda sp: (sp.operation_name, sp.param_name))
    return swept


def build_param_to_pipeline_map(
    configs: Dict[str, PipelineConfig],
    swept_params: List[SweptParameter],
) -> Dict[tuple, str]:
    """Build a reverse lookup from canonical value tuple to pipeline name.

    Args:
        configs: Mapping of pipeline name to :class:`PipelineConfig`.
        swept_params: Swept parameters as returned by
            :func:`detect_swept_parameters`.

    Returns:
        Dict mapping a tuple of canonicalized parameter values (ordered by
        *swept_params*) to the pipeline name that matches those values.

    Values are canonicalized with :func:`json.dumps` (``sort_keys=True``)
    so that unhashable types (lists, dicts) can be used as dict keys.
    """
    if not swept_params:
        return {}

    lookup: Dict[tuple, str] = {}

    for pipe_name, cfg in configs.items():
        # Build a fast index of (op_name -> params dict) for this config.
        param_index = _build_param_index(cfg)

        canon_values: List[str] = []
        for sp in swept_params:
            raw_value = param_index.get(
                (sp.operation_name, sp.param_name),
            )
            canon_values.append(_canonicalize(raw_value))

        lookup[tuple(canon_values)] = pipe_name

    return lookup


def resolve_pipeline_name(
    selections: Dict[tuple, object],
    lookup: Dict[tuple, str],
    swept_params: List[SweptParameter],
) -> Optional[str]:
    """Resolve current widget selections to a pipeline name.

    Args:
        selections: Current widget values as
            ``{(operation_name, param_name): value}``.
        lookup: Reverse lookup as returned by
            :func:`build_param_to_pipeline_map`.
        swept_params: Swept parameters as returned by
            :func:`detect_swept_parameters`.

    Returns:
        Pipeline name matching the selections, or ``None`` if no pipeline
        matches the given combination.
    """
    canon_key: List[str] = []
    for sp in swept_params:
        raw_value = selections.get((sp.operation_name, sp.param_name))
        canon_key.append(_canonicalize(raw_value))

    return lookup.get(tuple(canon_key))


def get_swept_param_names(
    swept_params: List[SweptParameter],
) -> Set[Tuple[str, str]]:
    """Return the set of swept parameter identifiers.

    Args:
        swept_params: Swept parameters as returned by
            :func:`detect_swept_parameters`.

    Returns:
        Set of ``(operation_name, param_name)`` tuples for parameters that
        vary across pipelines.  Useful for the config bar to bold swept
        parameter labels.
    """
    return {
        (sp.operation_name, sp.param_name)
        for sp in swept_params
    }


def compute_structural_signature(
    cfg: PipelineConfig,
) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]:
    """Compute a structural signature for a pipeline configuration.

    Args:
        cfg: A single pipeline configuration.

    Returns:
        Tuple of ``(ops_tuple, meas_tuple)`` where each element is a tuple
        of ``(op_name, op_class)`` pairs.  Two configs with the same
        signature have the same pipeline structure (same operations in the
        same order), differing only in parameter values.
    """
    ops_tuple = tuple(
        (op["name"], op["class"]) for op in cfg.operations
    )
    meas_tuple = tuple(
        (meas["name"], meas["class"]) for meas in cfg.measurements
    )
    return (ops_tuple, meas_tuple)


def group_configs_by_structure(
    configs: Dict[str, PipelineConfig],
) -> Dict[
    Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]],
    Dict[str, PipelineConfig],
]:
    """Group pipeline configs by their structural signature.

    Args:
        configs: Mapping of pipeline name to :class:`PipelineConfig`.

    Returns:
        Dict mapping structural signature to the subset of configs that
        share that structure.  Each value is a dict of
        ``{pipeline_name: PipelineConfig}``.
    """
    groups: Dict[
        Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]],
        Dict[str, PipelineConfig],
    ] = defaultdict(dict)

    for pipe_name, cfg in configs.items():
        sig = compute_structural_signature(cfg)
        groups[sig][pipe_name] = cfg

    return dict(groups)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonicalize(value: object) -> str:
    """Canonicalize a parameter value to a JSON string for hashing.

    Args:
        value: Any parameter value (int, float, str, bool, list, dict,
            None, etc.).

    Returns:
        Deterministic JSON string representation.
    """
    return json.dumps(value, sort_keys=True, default=str)


def _unique_ordered(values: list) -> list:
    """Return unique values preserving first-seen insertion order.

    Args:
        values: List of parameter values (may contain unhashable types).

    Returns:
        List of distinct values in insertion order.

    Uses :func:`json.dumps` for equality comparison so that unhashable
    types (lists, dicts) are handled correctly.
    """
    seen: set = set()
    result: list = []
    for v in values:
        canon = _canonicalize(v)
        if canon not in seen:
            seen.add(canon)
            result.append(v)
    return result


def _build_param_index(
    cfg: PipelineConfig,
) -> Dict[Tuple[str, str], object]:
    """Index all parameters in a config by ``(op_name, param_name)``.

    Args:
        cfg: A single pipeline configuration.

    Returns:
        Dict mapping ``(operation_name, param_name)`` to the parameter
        value.
    """
    index: Dict[Tuple[str, str], object] = {}
    for op in cfg.operations:
        op_name = op["name"]
        for param_name, param_value in op.get("params", {}).items():
            index[(op_name, param_name)] = param_value
    for meas in cfg.measurements:
        op_name = meas["name"]
        for param_name, param_value in meas.get("params", {}).items():
            index[(op_name, param_name)] = param_value
    return index
