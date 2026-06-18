from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

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

#: Image layer a process-mode CLI run exports. A closed
#: subset of the layers exposed as Image accessors; ``rgb``/``gray``/
#: ``detect_mat`` save as TIFF, ``objmap`` as a raw-label PNG.
ProcessOnlyLayer = Literal["rgb", "gray", "detect_mat", "objmap"]

#: Image layer a GpuDetector consumes as model input. Single-channel layers
#: (gray/detect_mat) are stacked to (H, W, 3) by GpuDetector.preprocess.
GpuInputLayer = Literal["rgb", "gray", "detect_mat"]

#: Object output a GpuDetector produces. "instance" -> labeled objmap;
#: "semantic" -> binary objmask (auto-labels into objmap, like a threshold detector).
GpuOutputKind = Literal["instance", "semantic"]

#: DINO backbone generation for DinoSam2Detector. 2 = DINOv2 (Apache, ungated,
#: default); 3 = DINOv3 (gated, opt-in — routes through require_license_acceptance).
#: Type-only closed set (no Enum / documentation surface needed).
DinoVersion = Literal[2, 3]

#: DINO backbone size for DinoSam2Detector. Maps with DinoVersion to the HF
#: model id (e.g. (2, "base") -> "facebook/dinov2-base").
DinoSize = Literal["small", "base", "large"]

#: Public top-level CLI execution mode. ``full`` performs the normal
#: apply-and-measure run, ``measure`` reruns measurement from existing HDFs,
#: ``recompile`` refreshes aggregate outputs from an existing output root, and
#: ``process`` performs the apply-only single-layer export selected by
#: ``--layer``.
CliMode = Literal["full", "measure", "recompile", "process"]

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

#: The single-objective composite blend selector — the serialized value of
#: ``CompositeScorer.blend``. ``"tchebycheff"`` (default) is conjunctive
#: (worst-axis-dominant, augmented Tchebycheff over per-child cost);
#: ``"weighted_mean"`` is the compensatory opt-out. No Enum partner is needed —
#: this is a serialized field value with no separate documentation surface
#: (mirrors ``DetectMode`` / ``ExecutionMode``). The geometric-mean-of-cost
#: blend is intentionally NOT offered (it inverts the conjunctive property).
CompositeBlend = Literal["tchebycheff", "weighted_mean"]

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


def _make_require_value(base: "type | Callable[[], type]"):
    """Build an ``AfterValidator`` guard asserting a value is a ``base`` instance.

    ``base`` may be a concrete type or a zero-arg callable that returns the
    type (resolved lazily, so ``OperationField`` can name ``BaseOperation``
    without importing it at ``tools_`` load time — avoiding the import cycle).

    Args:
        base: The class the value must be an instance of, or a zero-arg
            callable returning that class (resolved on first validation).

    Returns:
        A validator ``_require(value)`` that returns ``value`` unchanged when
        it is a ``base`` instance, and raises ``ValueError`` otherwise. The
        error is a ``ValueError`` (not ``TypeError``) so pydantic wraps it
        into a :class:`pydantic.ValidationError`.
    """

    def _require(value: Any) -> Any:
        resolved = base if isinstance(base, type) else base()
        if not isinstance(value, resolved):
            raise ValueError(
                f"expected an instance of {resolved.__name__}, got "
                f"{type(value).__name__}"
            )
        return value

    return _require


def polymorphic_field(base: "type | Callable[[], type]", *, marker: Any = None):
    """A pydantic field for a polymorphic model subtree.

    The concrete subclass survives a JSON round-trip via the ``phenotypic``
    class registry: the field serializes to ``{"class": <name>, "params":
    {...}}`` (or the pipeline-tagged form for an ``ImagePipeline``) and
    reconstructs the concrete subclass on load.

    Args:
        base: Constrains the accepted/validated type — a class or a zero-arg
            callable returning the class (the lazy form avoids import cycles,
            e.g. ``OperationField`` naming ``BaseOperation``).
        marker: Optional sentinel attached to the ``Annotated`` chain (e.g.
            the GUI's :class:`_OperationFieldMarker`) so introspecting tools
            can recognise the field despite the ``Any`` core erasure.

    Returns:
        An ``Annotated`` type usable as a pydantic field annotation. Host
        models must set ``model_config`` with ``arbitrary_types_allowed=True``.
    """
    core = Annotated[
        Any,
        BeforeValidator(_deserialize_operation_value),
        AfterValidator(_make_require_value(base)),
        PlainSerializer(_serialize_operation_value),
    ]
    if marker is None:
        return core
    # Annotated flattens a nested Annotated (PEP 593): the marker joins the chain.
    return Annotated[core, marker]


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


def _lazy_base_operation() -> type:
    from phenotypic.abc_ import BaseOperation

    return BaseOperation


#: Annotated operation type usable as a pydantic field annotation for a
#: parameter that holds another operation or a nested pipeline.
#:
#: Back-compat alias — an operation/pipeline-valued field built by
#: :func:`polymorphic_field`. Serializes to a class-tagged dict so the
#: concrete subclass survives a JSON round-trip; deserializes by resolving
#: the class through the ``phenotypic`` registry. Use it directly
#: (``OperationField``) or inside a container — e.g. ``list[OperationField]``
#: — when the field must accept several operation types and round-trip each
#: losslessly.
#:
#: The core type is ``Any`` (naming the operation base classes here would
#: create an import cycle through ``tools_``); an ``AfterValidator``
#: restores the operation/pipeline type guard. The trailing
#: :class:`_OperationFieldMarker` lets the GUI ``OperationRegistry``
#: recognise the field despite the ``Any`` erasure.
OperationField = polymorphic_field(
    base=_lazy_base_operation, marker=_OperationFieldMarker()
)


class TuneSpec:
    """Per-field tuning-search metadata (Tier-1 inference override).

    ``TuneSpec`` mirrors the existing field-marker pattern (``_ColumnRefMarker``,
    ``_OperationFieldMarker``): a frozen, slotted, **non-pydantic** sentinel that
    is a complete no-op at runtime and is read *only* by ``infer_search_space``,
    via ``op.model_fields[name].metadata`` (where pydantic v2 stores
    ``Annotated`` extras). It rides in an ``Annotated[T, TuneSpec(...)]`` chain::

        class GaussianBlur(ImageEnhancer):
            sigma:    Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 2.0
            truncate: Annotated[float, TuneSpec(tunable=False)]      = 4.0

    At runtime ``sigma`` is still a plain ``float``; ``GaussianBlur(sigma=999.0)``
    constructs exactly as before. The marker is the **search** domain, never the
    **valid** domain — validity stays the job of pydantic ``Field(ge=, le=)``.

    It lives here (not in ``phenotypic.tune``) so operation modules can import it
    without dragging in the tune engine — the public re-export
    ``from phenotypic.tune import TuneSpec`` still works.

    Args:
        low: Inclusive lower search bound (positional). ``None`` leaves the
            bound to Tier-2 inference / the co-located ``Field`` constraint.
        high: Inclusive upper search bound (positional). ``None`` as ``low``.
        step: Discretization stride — integer step or quantized float.
            ``None`` means continuous (or unit-step for integers).
        log: Whether to sample on a logarithmic scale (default ``False``).
        categories: Override / subset the auto-derived categorical choices.
            A list is coerced to a tuple so the marker stays hashable.
        tunable: ``False`` excludes the field from tuning outright and
            short-circuits Tier-2 (default ``True``).

    Note:
        This is **not** a pydantic model — it is a plain slotted sentinel so it
        can sit in an ``Annotated`` chain without pydantic trying to validate
        it. It carries ``__eq__``/``__hash__`` over the full field tuple so a
        duplicate in an ``Annotated`` chain de-dupes (PEP 593 chain semantics).
    """

    __slots__ = ("low", "high", "step", "log", "categories", "tunable")

    low: Optional[float]
    high: Optional[float]
    step: Optional[float]
    log: bool
    categories: Optional[tuple]
    tunable: bool

    def __init__(
        self,
        low: Optional[float] = None,
        high: Optional[float] = None,
        *,
        step: Optional[float] = None,
        log: bool = False,
        categories: Optional[tuple] = None,
        tunable: bool = True,
    ) -> None:
        self.low = low
        self.high = high
        self.step = step
        self.log = log
        self.categories = tuple(categories) if categories is not None else None
        self.tunable = tunable

    def _as_tuple(self) -> tuple:
        return (
            self.low,
            self.high,
            self.step,
            self.log,
            self.categories,
            self.tunable,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TuneSpec):
            return NotImplemented
        return self._as_tuple() == other._as_tuple()

    def __hash__(self) -> int:
        return hash(self._as_tuple())

    def __repr__(self) -> str:
        return (
            f"TuneSpec(low={self.low!r}, high={self.high!r}, "
            f"step={self.step!r}, log={self.log!r}, "
            f"categories={self.categories!r}, tunable={self.tunable!r})"
        )
