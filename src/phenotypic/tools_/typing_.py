from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Tuple

import numpy as np
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageOperation

FootprintShape = Literal["disk", "square", "diamond"]

DetectMode = Literal["gray", "red", "green", "blue", "MinRGB", "LabL", "LabA", "LabB", "HsvS", "HsvV", "InvS"]

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
