from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

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
