from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Tuple

import numpy as np
from pydantic import (
    AfterValidator,
    BeforeValidator,
    PlainSerializer,
    WithJsonSchema,
)

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageOperation

FootprintShape = Literal["disk", "square", "diamond"]

DetectMode = Literal["gray", "red", "green", "blue", "MinRGB", "LabL", "LabA", "LabB", "HsvS", "HsvV", "InvS"]

#: Image layer a process-only CLI run (``--process-only``) exports. A closed
#: subset of the layers exposed as Image accessors; ``rgb``/``gray``/
#: ``detect_mat`` save as TIFF, ``objmap`` as a raw-label PNG.
ProcessOnlyLayer = Literal["rgb", "gray", "detect_mat", "objmap"]

GridSearchSaveData = List[
    Literal["rgb", "gray", "detect_mat", "objmap", "objmask", "map2rgb"]
]

GridSearchConfig = List[Tuple["ImageOperation", Dict[str, List[Any]]]]

# ---------------------------------------------------------------------------
# CLI / GUI closed value sets — single source of truth, imported by callers
# rather than re-spelled. Pair with phenotypic.tools_.constants_.IMAGE_TYPES
# (paired Enum + Literal alignment is asserted in tests/unit/tools_/test_io_constants.py).
# ---------------------------------------------------------------------------

#: Forward-run / recompile execution backend. Bare-string carrier for
#: serialization (``progress/job_metadata.json``); callers convert to/from the
#: dataclass field of the same name on ``ProcessingState`` / ``ExecutionResults``.
ExecutionMode = Literal["local", "slurm"]

#: String form of the ``IMAGE_TYPES`` Enum's ``BASE`` and ``GRID`` members,
#: which are the only two image-type values that cross CLI / GUI boundary
#: code today. Other ``IMAGE_TYPES`` members (CROP / OBJECT / GRID_SECTION)
#: are internal to the core library and don't need a Literal partner yet.
ImageTypeName = Literal["Image", "GridImage"]

#: Per-image processing-events log statuses. Used by ``_cli_update_state``
#: and consumed by the dashboard generator.
ProcessingStatus = Literal["started", "completed", "failed"]

#: ``--recompile-task`` flag values for the SLURM recompile worker. Mirrors
#: the bare-string ``TASK_*`` constants in ``_cli_recompile_slurm_scripts.py``.
RecompileTaskType = Literal["measurements", "overlay", "finalize"]

#: ``--checkpoint-type`` flag values for the SLURM sentinel handler.
CheckpointType = Literal["manifest", "finalize"]

#: Tag attached to each row of ``progress/failures.jsonl`` distinguishing
#: Python-side exceptions from SLURM sbatch failures.
FailureSource = Literal["python", "slurm"]

# ---------------------------------------------------------------------------
# Pydantic-friendly array field — reusable annotated type for operation
# parameters that carry a raw ``np.ndarray`` (kernels, footprints, masks,
# coordinate grids). pydantic cannot validate or serialize a bare
# ``np.ndarray``, so this bundles the three pieces it needs:
#   - ``BeforeValidator`` coerces list / nested-list input to an ndarray,
#   - ``PlainSerializer`` emits a JSON-native nested list on ``model_dump``,
#   - ``WithJsonSchema`` supplies an "array" entry for ``model_json_schema``.
# Host models must still set ``model_config`` with
# ``arbitrary_types_allowed=True`` because the underlying field type is
# ``np.ndarray`` (an arbitrary, non-pydantic type).
# ---------------------------------------------------------------------------


def _coerce_to_ndarray(value: Any) -> np.ndarray:
    """Coerce list / nested-list / ndarray input to an ``np.ndarray``.

    Args:
        value: A ``list`` (possibly nested), an existing ``np.ndarray``,
            or any other array-like accepted by ``np.asarray``.

    Returns:
        np.ndarray: ``value`` itself if already an ndarray, otherwise the
        result of ``np.asarray(value)``.
    """
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _ndarray_to_list(value: np.ndarray) -> list:
    """Serialize an ``np.ndarray`` to a JSON-native nested list.

    Args:
        value: The array to serialize.

    Returns:
        list: ``value.tolist()`` — a nested list of Python scalars.
    """
    return value.tolist()


#: Annotated ``np.ndarray`` usable as a pydantic field annotation.
#:
#: Accepts a ``list`` / nested list or an ``np.ndarray`` as input,
#: serializes to a JSON-native nested list, and reports an ``"array"``
#: JSON schema. The host model **must** declare ``model_config`` with
#: ``arbitrary_types_allowed=True`` (the field type is the arbitrary,
#: non-pydantic ``np.ndarray``).
#:
#: Example:
#:     >>> import numpy as np
#:     >>> from pydantic import BaseModel, ConfigDict
#:     >>> from phenotypic.tools_.typing_ import NdArrayField
#:     >>> class KernelOp(BaseModel):
#:     ...     model_config = ConfigDict(arbitrary_types_allowed=True)
#:     ...     kernel: NdArrayField
#:     >>> op = KernelOp(kernel=[[1, 0], [0, 1]])
#:     >>> isinstance(op.kernel, np.ndarray)
#:     True
#:     >>> op.model_dump(mode="json")["kernel"]
#:     [[1, 0], [0, 1]]
#:     >>> KernelOp.model_json_schema()["properties"]["kernel"]["type"]
#:     'array'
NdArrayField = Annotated[
    np.ndarray,
    BeforeValidator(_coerce_to_ndarray),
    PlainSerializer(_ndarray_to_list, return_type=list),
    WithJsonSchema({"type": "array", "items": {}}),
]


# ---------------------------------------------------------------------------
# Pydantic-friendly operation field — reusable annotated type for operation
# parameters that hold *another* operation (or a nested pipeline) whose
# concrete class must survive a JSON round-trip.
#
# A plain ``model_dump`` of a field typed ``ObjectDetector | ImagePipeline``
# would dump only the *base-class* fields, silently losing the concrete
# subclass identity (an ``OtsuDetector`` would dump as an empty
# ``ImageOperation``). ``OperationField`` bundles:
#   - a ``PlainSerializer`` that tags each value with its class so the
#     concrete type can be rebuilt — ``{"class", "params"}`` for an
#     operation, ``{"__type__": "pipeline", "config": ...}`` for a pipeline,
#   - a ``BeforeValidator`` that reconstructs the operation/pipeline from
#     that tagged dict via the ``phenotypic`` class registry, while passing
#     an already-live operation instance straight through.
#
# Used by ``CompositeDetector.detectors`` and
# ``FilamentousFungiDetector.inoculum_detector``. The serialized shape is
# byte-compatible with what ``SerializablePipeline._serialize_*`` emits, so
# the two code paths agree and nested operations round-trip losslessly.
#
# Host models must declare ``arbitrary_types_allowed=True`` (inherited from
# ``BaseOperation``) because the underlying field type is an arbitrary
# (non-pydantic-by-this-annotation) operation class.
# ---------------------------------------------------------------------------


def _serialize_operation_value(value: Any) -> Any:
    """Serialize one operation/pipeline value to a class-tagged dict.

    Args:
        value: An ``ImageOperation`` instance or an ``ImagePipeline``
            (which is itself an ``ImageOperation``).

    Returns:
        A JSON-native dict carrying the concrete class identity:
        ``{"__type__": "pipeline", "config": {...}}`` for a pipeline,
        otherwise ``{"class": <name>, "params": {...}}`` for an operation.
        Any other input is returned unchanged (pydantic reports the
        type error).
    """
    # Lazy import: the serializer module imports operation classes, so a
    # top-level import here would create a cycle through ``tools_``.
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )

    if isinstance(value, SerializablePipeline):
        return {
            "__type__": "pipeline",
            "config": SerializablePipeline._serialize_pipeline_config(value),
        }
    if hasattr(value, "model_dump"):
        return {
            "class": type(value).__name__,
            "params": value.model_dump(mode="json"),
        }
    return value


def _deserialize_operation_value(value: Any) -> Any:
    """Reconstruct one operation/pipeline value from a class-tagged dict.

    Args:
        value: Either an already-live operation/pipeline instance (passed
            straight through), or a class-tagged dict produced by
            :func:`_serialize_operation_value`.

    Returns:
        The reconstructed ``ImageOperation`` / ``ImagePipeline`` instance,
        or ``value`` unchanged when it is not a recognised tagged dict.

    Raises:
        AttributeError: If a tagged dict names a class that cannot be
            resolved in the ``phenotypic`` namespace.
    """
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )

    # Already a live operation/pipeline — nothing to reconstruct.
    if not isinstance(value, dict):
        return value

    # Nested pipeline entry.
    if value.get("__type__") == "pipeline":
        return SerializablePipeline._deserialize_pipeline_config(
            value["config"]
        )

    # Plain operation entry — ``{"class": ..., "params": {...}}``.
    if "class" in value:
        cls = SerializablePipeline._find_class_in_phenotypic(value["class"])
        if cls is None:
            raise AttributeError(
                f"Class '{value['class']}' not found in phenotypic "
                f"namespace. Make sure it's properly imported in "
                f"phenotypic.__init__.py"
            )
        return cls.model_validate(value.get("params", {}) or {})

    return value


def _require_operation_value(value: Any) -> Any:
    """Assert a reconstructed value is an operation or a pipeline.

    Runs after :func:`_deserialize_operation_value`. Because
    :data:`OperationField` carries an ``Any`` core (it cannot name the
    operation base classes without an import cycle through ``tools_``),
    this validator restores the type guard a plain
    ``ObjectDetector | ImagePipeline`` annotation would otherwise provide.

    Args:
        value: The (already reconstructed) field value.

    Returns:
        ``value`` unchanged when it is a ``BaseOperation`` instance
        (``ImagePipeline`` is itself a ``BaseOperation``).

    Raises:
        ValueError: If ``value`` is not an operation/pipeline instance.
            Raised as ``ValueError`` (not ``TypeError``) so pydantic wraps
            it into a :class:`pydantic.ValidationError`.
    """
    from phenotypic.abc_ import BaseOperation

    if not isinstance(value, BaseOperation):
        raise ValueError(
            f"expected an operation or pipeline instance, got "
            f"{type(value).__name__}"
        )
    return value


class _OperationFieldMarker:
    """Sentinel attached to :data:`OperationField`'s ``Annotated`` chain.

    :data:`OperationField` erases its core type to ``Any`` (it cannot
    name the operation base classes without an import cycle through
    ``tools_``). That erasure also hides the field from the GUI's
    ``OperationRegistry``, which detects operation-valued parameters by
    inspecting the annotation. This marker is the distinguishing token
    the registry scans for — analogous to
    :class:`~phenotypic.tools_._column_ref._ColumnRefMarker` — so a
    field typed ``OperationField`` (or ``list[OperationField]`` /
    ``OperationField | None``) is still recognised as accepting an
    operation **or** a nested pipeline.

    Singleton-like: all instances compare equal so a duplicate marker in
    an ``Annotated`` chain de-dupes.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "_OperationFieldMarker()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OperationFieldMarker)

    def __hash__(self) -> int:
        return hash("_OperationFieldMarker")


#: Annotated operation type usable as a pydantic field annotation for a
#: parameter that holds another operation or a nested pipeline.
#:
#: Serializes to a class-tagged dict so the concrete subclass survives a
#: JSON round-trip; deserializes by resolving the class through the
#: ``phenotypic`` registry. Use it directly (``OperationField``) or inside
#: a container — e.g. ``list[OperationField]`` — when the field must
#: accept several operation types and round-trip each losslessly.
#:
#: The core type is ``Any`` (naming the operation base classes here would
#: create an import cycle through ``tools_``); an ``AfterValidator``
#: restores the operation/pipeline type guard. The trailing
#: :class:`_OperationFieldMarker` lets the GUI ``OperationRegistry``
#: recognise the field despite the ``Any`` erasure.
OperationField = Annotated[
    Any,
    BeforeValidator(_deserialize_operation_value),
    AfterValidator(_require_operation_value),
    PlainSerializer(_serialize_operation_value),
    _OperationFieldMarker(),
]
