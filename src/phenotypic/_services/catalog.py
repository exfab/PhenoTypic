"""Agent-facing projection of the operation registry.

The MCP server hands an agent JSON, not Python objects. This module turns
an :class:`~phenotypic._services.registry.OperationInfo` into a
JSON-serializable descriptor: the verbatim ``model_json_schema()`` plus the
handful of facts that schema structurally cannot state.

Two of those gaps are why a raw schema dump is not enough on its own:

* :data:`~phenotypic.sdk_.typing_.OperationField` erases its core type to
  ``Any``, so a parameter that takes another operation reports an empty
  ``{}`` branch and looks untyped.
* :data:`~phenotypic.sdk_.typing_.NdArrayField` reports
  ``{"type": "array", "items": {}}`` — no shape, no dtype. Flagging it
  ``ndarray`` at least tells the agent it is not practically authorable.

No GUI dependencies, and no measurement execution — this is pure
introspection over classes the registry already discovered.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, get_args

import numpy as np

from phenotypic._services.registry import OperationInfo, ParamInfo, get_registry
from phenotypic.sdk_.typing_ import ImageTypeName

#: The two image classes a pipeline can be run against, taken from the
#: ``ImageTypeName`` literal the CLI already threads through every run so
#: the catalog cannot drift from it.
_IMAGE_TYPE_NAMES = frozenset(get_args(ImageTypeName))

#: JSON Schema keywords that describe the *shape* of a property rather than
#: constrain its value. Everything else a property declares — ``minimum``,
#: ``exclusiveMinimum``, ``maxLength``, ``multipleOf``, … — is passed
#: through as a constraint under its real JSON Schema spelling. Pydantic's
#: ``Field(gt=0.0)`` reports ``exclusiveMinimum``, and the projection does
#: not invent a ``gt`` spelling for it.
_NON_CONSTRAINT_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "default",
        "deprecated",
        "description",
        "discriminator",
        "enum",
        "examples",
        "items",
        "oneOf",
        "prefixItems",
        "properties",
        "required",
        "title",
        "type",
    }
)

#: Split on a period/question/exclamation mark followed by whitespace. The
#: trailing ``\s`` is what keeps ``Typical range: 0.5--5.0`` intact — a
#: decimal point has no space after it.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _first_sentence(text: Optional[str]) -> Optional[str]:
    """Return the first sentence of *text*, or *text* when it has only one.

    Args:
        text: A parameter or class description, possibly ``None``.

    Returns:
        The leading sentence with surrounding whitespace collapsed, or
        ``None`` when *text* is ``None`` or blank.
    """
    if not text:
        return None
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    return _SENTENCE_END.split(collapsed, maxsplit=1)[0]


def _describe(text: Optional[str], *, verbose: bool) -> Optional[str]:
    """First sentence of *text*, or all of it when *verbose*."""
    if not text:
        return None
    collapsed = " ".join(text.split())
    return collapsed if verbose else _first_sentence(collapsed)


def _annotation_holds_ndarray(hint: Any) -> bool:
    """Whether ``np.ndarray`` appears anywhere in an annotation tree.

    :data:`~phenotypic.sdk_.typing_.NdArrayField` is usually nested inside a
    ``Union`` with a ``Literal`` of named shapes, so the marker is not at the
    top level; this walks ``Annotated`` extras and every ``get_args`` branch,
    mirroring
    :func:`~phenotypic._services.registry._has_operation_field_marker`.

    Args:
        hint: A type annotation, possibly wrapped or nested.

    Returns:
        ``True`` if the annotation can carry a raw NumPy array.
    """
    if hint is np.ndarray:
        return True
    return any(_annotation_holds_ndarray(arg) for arg in get_args(hint))


def _property_branches(prop: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The property itself, or its ``anyOf`` / ``oneOf`` alternatives."""
    for key in ("anyOf", "oneOf"):
        branches = prop.get(key)
        if isinstance(branches, list):
            return [b for b in branches if isinstance(b, dict)]
    return [prop]


def _param_type(prop: Dict[str, Any], *, holds_ndarray: bool) -> Optional[str]:
    """A compact type string for one property.

    Reports the JSON Schema ``type`` when the property declares one, and a
    ``"|"``-joined union of its branch types otherwise. A branch typed
    ``array`` is reported as ``ndarray`` when the annotation can carry a
    raw NumPy array — the schema says ``{"type": "array", "items": {}}``
    for both a real list and an ``NdArrayField``.

    Args:
        prop: One entry from the schema's ``properties`` block.
        holds_ndarray: Whether the live annotation contains ``np.ndarray``.

    Returns:
        The type string, or ``None`` when the schema declares no type at
        all — which is what an operation-valued parameter looks like.
    """
    names: List[str] = []
    for branch in _property_branches(prop):
        declared = branch.get("type")
        if not isinstance(declared, str):
            continue
        if declared == "array" and holds_ndarray:
            declared = "ndarray"
        if declared not in names:
            names.append(declared)
    return "|".join(names) if names else None


def _param_choices(prop: Dict[str, Any]) -> Optional[List[Any]]:
    """Closed value set for one property, or ``None``.

    Reads ``enum`` from the property or from whichever ``anyOf`` branch
    declares one — a ``Literal[...] | None`` field puts the enum on a
    branch, not at the top level.
    """
    for branch in _property_branches(prop):
        values = branch.get("enum")
        if isinstance(values, list):
            return list(values)
    return None


def _param_constraints(prop: Dict[str, Any]) -> Dict[str, Any]:
    """Value constraints declared by one property, JSON Schema spelling.

    Reads the top level first, and falls back to the ``anyOf`` / ``oneOf``
    branches when the top level declares none — the same walk
    :func:`_param_type` and :func:`_param_choices` already do. An optional
    bounded field publishes its bound on a *branch*, not at the top level::

        float | None = Field(None, gt=0)
        -> {"anyOf": [{"type": "number", "exclusiveMinimum": 0},
                      {"type": "null"}], "default": None}

    Reading only the top level reports ``{}`` for every such parameter —
    telling an agent ``gat_scale_factor`` is an unconstrained number when
    it must be ``> 0``.

    The ``null`` branch is skipped: it carries no constraint keyword
    anyway, and nullability is already reported through ``is_optional``.

    Args:
        prop: One entry from the schema's ``properties`` block.

    Returns:
        The constraint keywords, merged across the non-``null`` branches.
    """
    merged = {k: v for k, v in prop.items() if k not in _NON_CONSTRAINT_KEYS}
    if merged:
        return merged
    for branch in _property_branches(prop):
        if branch.get("type") == "null":
            continue
        merged.update(
            {k: v for k, v in branch.items() if k not in _NON_CONSTRAINT_KEYS}
        )
    return merged


def _alias_keys(field: Any) -> List[str]:
    """Schema property names a pydantic field may be published under."""
    keys: List[str] = []
    for candidate in (
        getattr(field, "validation_alias", None),
        getattr(field, "serialization_alias", None),
        getattr(field, "alias", None),
    ):
        if isinstance(candidate, str):
            keys.append(candidate)
        else:  # ``AliasChoices`` and friends
            for choice in getattr(candidate, "choices", ()):
                if isinstance(choice, str):
                    keys.append(choice)
    return keys


def _property_for(
    cls: Any, param_name: str, properties: Dict[str, Any]
) -> Dict[str, Any]:
    """The schema property for *param_name*, resolved through any alias.

    An aliased field is published in ``model_json_schema()`` under its
    alias, not its Python name: ``RemoveGridOutliers.cutoff_multiplier``
    carries ``AliasChoices("stddev_multiplier", "cutoff_multiplier")`` and
    appears as ``stddev_multiplier``. A plain name lookup misses it, and the
    parameter would then be projected with no type, no default and no
    constraints — silently, since the key simply is not there.

    Args:
        cls: The operation class being described.
        param_name: The parameter's Python name, as the registry reports it.
        properties: The schema's ``properties`` block.

    Returns:
        The matching property dict, or ``{}`` when the schema publishes no
        entry for the parameter at all.
    """
    if param_name in properties:
        return properties[param_name]
    field = getattr(cls, "model_fields", {}).get(param_name)
    for key in _alias_keys(field):
        if key in properties:
            return properties[key]
    return {}


def _layers_modified_for_class(cls: type) -> List[str]:
    """Layers an operation of this class writes to.

    Class-level twin of
    :func:`~phenotypic._core._pipeline_parts._image_pipeline_core._layers_modified_by`,
    which dispatches on ``isinstance`` and so needs an instance the catalog
    does not have (many operations have required parameters). The two are
    pinned together by
    ``test_layers_modified_agrees_with_the_live_helper``.

    A measurer returns ``[]`` rather than ``None``: it populates the
    measurement table, not an image layer.

    The entry guard is ``BaseOperation``, not ``ImageOperation`` — the
    helper's own parameter type. ``MeasureFeatures``, ``PostMeasurement``
    and ``PrefabPipeline`` are ``BaseOperation`` subclasses that are *not*
    ``ImageOperation`` subclasses, so guarding on the narrower base would
    silently report ``[]`` for all three.

    Args:
        cls: A ``BaseOperation`` subclass, or any other registered class.

    Returns:
        Layer names in the helper's order, or ``[]`` for a read-only
        operation and for classes outside the operation hierarchy
        (scorers, strategies, analyzers).
    """
    from phenotypic.abc_ import (
        ImageCorrector,
        ImageEnhancer,
        MeasureFeatures,
        ObjectDetector,
        ObjectRefiner,
    )
    from phenotypic.abc_._base_operation import BaseOperation

    if not (isinstance(cls, type) and issubclass(cls, BaseOperation)):
        return []
    if issubclass(cls, MeasureFeatures):
        return []
    if issubclass(cls, ImageCorrector):
        return ["rgb", "gray", "detect_mat", "objmap"]
    if issubclass(cls, ImageEnhancer):
        return ["detect_mat"]
    if issubclass(cls, (ObjectDetector, ObjectRefiner)):
        return ["objmap"]
    return ["rgb", "gray", "detect_mat", "objmap"]


def _project_param(
    param: ParamInfo,
    prop: Dict[str, Any],
    *,
    verbose: bool,
) -> Dict[str, Any]:
    """Project one :class:`ParamInfo` plus its schema entry into a dict.

    Args:
        param: Registry metadata for the parameter.
        prop: The parameter's entry in ``model_json_schema()["properties"]``,
            or ``{}`` when the schema omits it.
        verbose: Emit the full description rather than its first sentence.

    Returns:
        A JSON-serializable parameter descriptor.
    """
    holds_ndarray = _annotation_holds_ndarray(param.type_hint)
    column_ref = param.column_ref
    return {
        "name": param.name,
        "type": _param_type(prop, holds_ndarray=holds_ndarray),
        "default": prop.get("default") if param.has_default else None,
        "required": not param.has_default,
        "description": _describe(param.description, verbose=verbose),
        "constraints": _param_constraints(prop),
        "is_operation": param.is_operation,
        "is_pipeline": param.is_pipeline,
        "is_list": param.is_list,
        "is_optional": param.is_optional,
        "choices": _param_choices(prop),
        "column_ref": (
            None
            if column_ref is None
            else {
                "source": column_ref.source,
                "multi": column_ref.multi,
                "with_alt": column_ref.with_alt,
            }
        ),
    }


def describe_operation(name: str, *, verbose: bool = False) -> Dict[str, Any]:
    """Project an ``OperationInfo`` into the agent-facing descriptor.

    Args:
        name: Operation, scorer, or strategy class name.
        verbose: Return each parameter's full docstring text instead of its
            first sentence. Descriptions are long — ``BlurGauss.sigma``'s
            runs four sentences — and a 20-parameter detector spends the
            agent's context on prose it did not ask for.

    Returns:
        A JSON-serializable dict carrying the verbatim
        ``model_json_schema()`` plus the two facts that schema cannot
        express — whether a parameter takes an operation, and whether it is
        a raw array.

    Raises:
        KeyError: If *name* is not a registered class.

    Example:
        Read one parameter's real JSON Schema constraint:

        >>> from phenotypic._services.catalog import describe_operation
        >>> desc = describe_operation("FlattenIllumination")
        >>> sigma = next(p for p in desc["params"] if p["name"] == "sigma")
        >>> sigma["constraints"]
        {'exclusiveMinimum': 0.0}
    """
    info = get_registry().get(name)
    if info is None:
        raise KeyError(f"Operation {name!r} not found in the catalog")

    schema = info.cls.model_json_schema()
    properties = schema.get("properties", {})

    return {
        "name": info.name,
        "category": info.category,
        "module": info.module,
        "doc": _describe(info.docstring, verbose=verbose),
        "json_schema": schema,
        "params": [
            _project_param(
                param,
                _property_for(info.cls, param.name, properties),
                verbose=verbose,
            )
            for param in info.parameters.values()
        ],
        "layers_modified": _layers_modified_for_class(info.cls),
    }


def _summary_row(info: OperationInfo) -> Dict[str, Any]:
    """One compact catalog row — no JSON schema, by design."""
    return {
        "name": info.name,
        "category": info.category,
        "summary": _first_sentence(info.docstring),
        "n_params": len(info.parameters),
        "has_nested_operations": any(
            param.is_operation or param.is_pipeline
            for param in info.parameters.values()
        ),
    }


def list_operations(
    category: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """List catalog rows, optionally filtered by category and free text.

    Rows are deliberately compact: full JSON schemas come from
    :func:`describe_operation`, one operation at a time, so that browsing
    the catalog cannot flood an agent's context.

    Args:
        category: Restrict to one registry category (``"Detector"``,
            ``"Scorer"``, ``"Prefab"``, …). ``None`` lists every category.
        query: Case-insensitive substring matched against the class name
            and the docstring summary.
        limit: Row cap. The response reports whether it truncated and how
            many rows matched in total.

    Returns:
        ``{"operations": [...], "total": int, "truncated": bool}``, where
        ``total`` counts every match, not just the returned rows.

    Example:
        >>> from phenotypic._services.catalog import list_operations
        >>> result = list_operations(category="Detector", limit=1)
        >>> result["truncated"]
        True
    """
    rows = [
        _summary_row(info)
        for info in get_registry().get_all().values()
        if category is None or info.category == category
    ]

    if query:
        needle = query.lower()
        rows = [
            row
            for row in rows
            if needle in row["name"].lower()
            or needle in (row["summary"] or "").lower()
        ]

    rows.sort(key=lambda row: (row["category"], row["name"]))
    return {
        "operations": rows[:limit],
        "total": len(rows),
        "truncated": len(rows) > limit,
    }


def _texture_scales(owner: Any) -> List[int]:
    """Scales a texture measurer will emit headers for.

    ``MeasureTexture.scale`` is instance state, and the header count depends
    on it: 13 members x (4 angles + 1 average) = 65 columns *per scale*.

    Args:
        owner: The measurement operation instance that declared the enum.

    Returns:
        The instance's scales as a list, or ``[]`` when it declares none.
    """
    scale = getattr(owner, "scale", None)
    if scale is None:
        return []
    if isinstance(scale, (int, np.integer)):
        return [int(scale)]
    return [int(value) for value in scale]


def measurement_headers(info_cls: Any, owner: Any) -> List[str]:
    """DataFrame headers one measurement enum will emit for *owner*.

    Dispatches on ``info_cls.header_scheme()`` rather than calling
    ``get_headers()`` blindly. A blanket call is wrong for two schemes and
    for one of them it raises::

        SIZE.header_scheme()    -> "static"   -> get_headers() -> ['Size_Area', ...]
        TEXTURE.header_scheme() -> "texture"  -> get_headers() -> TypeError:
                                     missing 1 required positional argument: 'scale'

    Args:
        info_cls: A :class:`~phenotypic.schema.MeasurementInfo` subclass.
        owner: The live instance that declared it — a measurement operation
            or a model fitter. Both schemes below read runtime state off it,
            which is why the class alone is not enough.

    Returns:
        Header strings in emission order. Empty when a metric-qualified
        enum's owner names no metric: there is no derivable header then,
        and inventing a placeholder would be worse than reporting none.
    """
    from phenotypic.schema import qualified_header

    scheme = info_cls.header_scheme()

    if scheme == "texture":
        matrix_name = getattr(owner, "matrix_name", None)
        headers: List[str] = []
        for scale in _texture_scales(owner):
            headers.extend(info_cls.get_headers(scale, matrix_name))
        return headers

    if scheme == "metric_qualified":
        # The ``<metric>`` segment is a runtime value: a ``ModelFitter``
        # derives it from the column it was fitted on (``self.on``).
        token = getattr(owner, "_metric_token", None)
        if not token:
            return []
        return [qualified_header(member, token) for member in info_cls]

    return list(info_cls.get_headers())


def _info_block_columns(image_type: ImageTypeName) -> List[str]:
    """The per-object info block ``measure()`` appends to every table.

    ``ImagePipeline.measure()`` closes with
    ``image.grid.info()`` / ``image.objects.info()`` regardless of what the
    ``meas`` slot holds, so these columns are in the output of *every* run,
    including a run whose pipeline declares no measurers at all.

    The bounds half is read off :class:`~phenotypic.measure.MeasureBounds`
    — the operation ``objects.info()`` actually runs — rather than
    hard-coded, so a new ``BBOX`` member arrives here on its own.

    Args:
        image_type: ``"GridImage"`` or ``"Image"``. A ``GridFinder`` refuses
            a plain :class:`~phenotypic.Image` (``GridOperation`` raises on
            a non-gridded target), so this is the only thing the ``Grid_*``
            half turns on.

    Returns:
        ``Object_Label``, the ``Bbox_*`` geometry columns, and — for a
        ``GridImage`` — the four ``Grid_*`` assignment columns, in emission
        order.
    """
    from phenotypic.measure import MeasureBounds
    from phenotypic.schema import GRID, OBJECT

    bounds = MeasureBounds()
    columns: List[str] = [str(member) for member in OBJECT.get_headers()]
    for info_cls in bounds.get_measurement_infoclasses():
        columns.extend(measurement_headers(info_cls, bounds))

    if image_type == "GridImage":
        # The four assignment columns ``_assemble_grid_info`` writes. ``GRID``
        # also declares four interval bounds (``Grid_RowIntervalStart``, …)
        # which the info block does not carry, so a blanket
        # ``GRID.get_headers()`` would over-report by four.
        columns.extend(
            str(member)
            for member in (
                GRID.ROW_NUM,
                GRID.COL_NUM,
                GRID.ROW_MAJOR_IDX,
                GRID.COL_MAJOR_IDX,
            )
        )

    return columns


def derive_columns(pipeline: Any, *, image_type: ImageTypeName) -> List[str]:
    """Columns a pipeline's ``measure()`` will produce, metadata aside.

    Two sources, both of which the real table always carries:

    1. **The measurers.** Each live instance in the ``meas`` slot is asked
       which measurement schemas it is emitting
       (``MeasureFeatures.get_measurement_infoclasses()`` — genuinely
       instance-dependent, since ``MeasureColor`` includes or excludes
       members based on ``include_XYZ`` / ``include_xy``), and each schema
       is expanded through :func:`measurement_headers`.
    2. **The info block.** ``measure()`` appends ``Object_Label``, the
       ``Bbox_*`` geometry and (on a ``GridImage``) the ``Grid_*``
       assignment columns unconditionally — see :func:`_info_block_columns`.
       Walking the ``meas`` slot alone under-reports a plain
       ``OtsuDetector`` + ``MeasureSize`` + ``MeasureShape`` run by 15
       columns, which is most of what the results viewer keys off.

    Only the ``meas`` slot is walked for (1). Post transforms rewrite the
    table rather than declaring a schema, and an analysis model's
    metric-qualified output lands in its own deliverable, not in the
    measurement table.

    **What is excluded, and why.** The ``Metadata_*`` block is not derivable
    from a pipeline: the framework provenance columns come off the image
    (``Metadata_ImageName``, ``Metadata_BitDepth``, …) and any experimental
    metadata comes off the run's ``--metadata`` CSV. Neither is knowable
    here, so the contract is *"every measurement column, plus the info
    block, and no metadata"* rather than a silently partial list.

    Args:
        pipeline: An :class:`~phenotypic.ImagePipeline`.
        image_type: The class the pipeline will be run against —
            ``"GridImage"`` or ``"Image"``. Required, and keyword-only,
            because it is not knowable from the pipeline: the same pipeline
            yields four extra ``Grid_*`` columns on a ``GridImage`` and a
            default would answer for the caller silently. Callers holding a
            CLI/run config already carry this as ``image_type``.

    Returns:
        Column names, de-duplicated — two measurers can legitimately
        declare the same schema. Measurement columns first in ``meas``
        order, then the info block, mirroring ``measure()``'s own
        ``[measurements] -> [Metadata_] -> [info block]`` ordering with the
        undeducible ``Metadata_`` block dropped.

    Raises:
        ValueError: If *image_type* is not one of the two image classes.

    Example:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.measure import MeasureSize
        >>> from phenotypic._services.catalog import derive_columns
        >>> cols = derive_columns(
        ...     ImagePipeline(meas=[MeasureSize()]), image_type="GridImage"
        ... )
        >>> ("Size_Area" in cols, "Grid_RowNum" in cols)
        (True, True)
    """
    if image_type not in _IMAGE_TYPE_NAMES:
        raise ValueError(
            f"image_type must be one of {sorted(_IMAGE_TYPE_NAMES)}, "
            f"got {image_type!r}"
        )

    columns: List[str] = []
    seen: set[str] = set()

    for measurer in getattr(pipeline, "_meas", {}).values():
        for info_cls in measurer.get_measurement_infoclasses():
            for header in measurement_headers(info_cls, measurer):
                if header not in seen:
                    seen.add(header)
                    columns.append(header)

    for header in _info_block_columns(image_type):
        if header not in seen:
            seen.add(header)
            columns.append(header)

    return columns
