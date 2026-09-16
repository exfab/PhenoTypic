"""
Pipeline validation for the PhenoTypic CLI.

This module provides validation functions to check pipeline configuration
before running large batch processing jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic.abc_ import GpuDetector

from phenotypic import ImagePipeline
from ._cli_types import ExecutionConfig


def validate_pipeline(
    pipeline_path: Path,
    skip_validation: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate that pipeline JSON can be loaded successfully.
    
    Args:
        pipeline_path: Path to pipeline JSON file
        skip_validation: If True, skip validation (for advanced users)
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if skip_validation:
        return True, None
    
    try:
        # Try to load pipeline
        pipeline = ImagePipeline.from_json(pipeline_path)
        
        # Check that pipeline has operations or measurements
        if not pipeline._ops and not pipeline._meas:
            return False, "Pipeline has no operations or measurements"
        
        return True, None
        
    except FileNotFoundError:
        return False, f"Pipeline file not found: {pipeline_path}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in pipeline file: {e}"
    except Exception as e:
        return False, f"Failed to load pipeline: {type(e).__name__}: {e}"


def validate_execution_config(
    config: ExecutionConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate execution configuration for obvious errors.
    
    Args:
        config: Execution configuration to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    # Check pipeline file exists
    if not config.pipeline_json.exists():
        return False, f"Pipeline file not found: {config.pipeline_json}"
    
    # Check input path exists
    if not config.input_path.exists():
        return False, f"Input path not found: {config.input_path}"
    
    # Check grid dimensions for GridImage. ``None`` means "no CLI override —
    # fall back to the pipeline preset / built-in default at resolve time"
    # and is valid here; only explicit non-positive values are rejected.
    if config.image_type == "GridImage":
        if config.nrows is not None and config.nrows <= 0:
            return False, f"Invalid nrows: {config.nrows} (must be positive)"
        if config.ncols is not None and config.ncols <= 0:
            return False, f"Invalid ncols: {config.ncols} (must be positive)"
    
    # Check n_jobs is valid
    if config.n_jobs == 0:
        return False, "n_jobs cannot be 0 (use -1 for all cores or positive integer)"
    
    # Check SLURM parameters if provided
    if config.slurm_args:
        # Warn about common missing parameters
        required_slurm_params = ["slurm_partition"]
        missing = [p for p in required_slurm_params if p not in config.slurm_args]
        if missing:
            # This is a warning, not an error - let SLURM handle it
            pass
    
    return True, None


def full_validation(
    config: ExecutionConfig,
) -> Tuple[bool, list[str]]:
    """
    Validate execution configuration and pipeline loading.

    Args:
        config: Execution configuration.

    Returns:
        Tuple of (is_valid, list_of_errors).
        If valid, list_of_errors is empty.
    """
    errors = []

    # Validate config
    config_valid, config_error = validate_execution_config(config)
    if not config_valid:
        errors.append(config_error)
        return False, errors

    # Validate pipeline can be loaded
    pipeline_valid, pipeline_error = validate_pipeline(
        config.pipeline_json,
        config.skip_validation
    )
    if not pipeline_valid:
        errors.append(pipeline_error)

    return len(errors) == 0, errors


class UnstageableGpuDetectorError(ValueError):
    """A GpuDetector sits somewhere the staged engine cannot drive it."""


#: Pipeline slots the staged engine runs on a CPU node in Stage 3. A
#: GpuDetector in any of them cannot be staged, so it is refused rather than
#: silently run on CPU.
_CPU_ONLY_SLOTS = ("meas", "post", "filters", "model")


#: What each container hands its children. Keyed by class, and NOT a
#: declaration on the operations themselves: for these types the semantics is
#: definitional rather than incidental. A CompositeDetector whose branches
#: chained would not be a composite -- it would be an ImagePipeline, which
#: already exists for that. So this table restates a type contract; it does not
#: cache an observation about today's ``_operate``.
#:
#: Coverage is asserted on the TABLE ITSELF, not by enumerating the tree: the
#: set is closed by rule, so a new container needs no entry and no decision --
#: it is refused by default, which is the correct answer for it. There is no
#: ``_UNSUPPORTED_CONTAINERS`` map.
#:
#: ``ImagePipeline`` is deliberately absent and handled by ``isinstance`` in
#: ``_child_contract``: it is matched by subclass, not by exact type, so a
#: class-keyed entry would refuse a subclass a user actually holds.
#:
#: Populated lazily by ``_populate_child_contract`` -- read it through that,
#: never directly, or a first reader sees an empty dict.
_CHILD_CONTRACT: dict[type, str] = {}


def _populate_child_contract() -> None:
    """Fill ``_CHILD_CONTRACT`` on first use.

    Deferred rather than done at import time because this module is imported
    by CLI argument validation that may never reach a GPU question, and
    ``phenotypic.detect`` / ``phenotypic.enhance`` are heavy subpackages.
    """
    if _CHILD_CONTRACT:
        return
    from phenotypic.detect import CompositeDetector
    from phenotypic.enhance import CompositeEnhance

    _CHILD_CONTRACT[CompositeDetector] = "parallel"
    _CHILD_CONTRACT[CompositeEnhance] = "parallel"


def _child_contract(container: Any) -> str:
    """``"sequence"`` or ``"parallel"``; raise for anything else.

    ONLY composition primitives may carry a staged GpuDetector. A domain
    detector is refused even when its current code would classify cleanly --
    ``FilamentousFungiDetector`` feeds ``inoculum_detector`` the container's
    own image today (``_filamentous_fungi_detector.py:395,398``) and so reads
    as ``"parallel"``, but that is incidental to an algorithm that also runs an
    inline ``ContrastStretching`` (``:413``) and a destructive
    ``_subtract_background``. Nothing about being a fungus detector constrains
    it to keep doing that, so the table's safety argument -- "this restates a
    type contract, it does not cache an observation" -- would not hold
    uniformly if it were admitted.
    """
    from phenotypic._core._image_pipeline import ImagePipeline as _ImagePipeline

    _populate_child_contract()
    if isinstance(container, _ImagePipeline):
        return "sequence"
    cls = type(container)
    if cls in _CHILD_CONTRACT:
        return _CHILD_CONTRACT[cls]
    raise UnstageableGpuDetectorError(
        f"a GpuDetector cannot be nested inside {cls.__name__}: only "
        "composition primitives (ImagePipeline, CompositeDetector, "
        "CompositeEnhance) may carry one. Lift the detector into a "
        "CompositeDetector branch, or into the top-level pipeline."
    )


def validate_ancestor_contracts(
    pipeline: ImagePipeline, path: tuple[str, ...]
) -> None:
    """Every container on the ancestor chain must declare a child contract.

    Called from ``find_gpu_detectors`` -- NOT from a prefix builder. The
    refusal is a property of *placement*, not of prefix computation, and
    ``pipeline_requires_gpu`` is the production entry point that must carry
    placement refusals. A refusal reachable only from the prefix builder fires
    after the run has already been routed, which is how this narrowing was
    silently lost once already.

    Args:
        pipeline: Root pipeline the path is addressed against.
        path: Tree path to a GpuDetector, as returned by
            ``find_gpu_detectors``.

    Raises:
        UnstageableGpuDetectorError: some container on the chain is not a
            composition primitive.
    """
    from phenotypic.sdk_._operation_tree import get_at_path

    for depth in range(len(path)):  # path[:0] is the root pipeline
        _child_contract(get_at_path(pipeline, path[:depth]))


def find_gpu_detectors(
    pipeline: ImagePipeline, *, strict: bool = False
) -> list[tuple[tuple[str, ...], "GpuDetector"]]:
    """Every ``(path, detector)`` GpuDetector in *pipeline*, tree-wide.

    Placement refusals -- an unstageable ancestor, or a CPU-only slot -- raise
    **regardless of** ``strict``. They have to fire on the production path
    (``pipeline_requires_gpu``), because the only other caller,
    ``split_pipeline_at_gpu``, is reached only once ``pipeline_requires_gpu``
    has already returned True. A refusal reachable only under ``strict=True``
    is a refusal production never performs, while a unit test calling the
    helper directly still passes.

    Args:
        pipeline: The pipeline to scan.
        strict: When True, additionally raise for MORE THAN ONE detector. The
            GUI (``gui/run_console/_callbacks.py:253``) calls the non-strict
            path, where a multi-detector pipeline should report True rather
            than raise.

            **``strict=True`` has no production caller yet** -- it is
            exercised only by tests. Task 5 adds the one caller, repointing
            ``split_pipeline_at_gpu`` onto this function; until then that
            function keeps its own top-level-only scan
            (``_cli_pipeline_split.py:34-43``), which catches two TOP-LEVEL
            detectors but not two nested inside a composite. So this branch is
            pending, not dead.

    Returns:
        ``(path, detector)`` pairs in depth-first order. Each ``path`` is a
        tuple of non-empty strings addressing the detector from the root
        pipeline; list entries are spelled ``"ops[0]"``.

    Raises:
        UnstageableGpuDetectorError: a detector sits under a container that is
            not a composition primitive, or in a CPU-only slot; or, under
            ``strict``, there is more than one detector.
    """
    from phenotypic.abc_ import GpuDetector
    from phenotypic.sdk_._operation_tree import find_operations, walk_operations

    hits = find_operations(pipeline, lambda op: isinstance(op, GpuDetector))

    if strict and len(hits) > 1:
        paths = ", ".join("/".join(p) for p, _ in hits)
        # Name EVERY offending path, not just the count: the message's job is
        # to tell the user which branches to split. Deferred feature, not a
        # limit of the design -- see spec section 13 for the intended N>1
        # execution model. Keep the literal "more than one GpuDetector": it is
        # the wording the existing suite already pins
        # (``test_cli_pipeline_split.py:33``). The path list follows it rather
        # than replacing it.
        raise UnstageableGpuDetectorError(
            "staged execution does not support more than one GpuDetector "
            f"per pipeline; found {len(hits)} at: {paths}"
        )

    # Placement refusals, ALL of them, live here -- not in the splitter. This
    # is the function `pipeline_requires_gpu` calls, so a refusal placed
    # anywhere else fires only after the run has already been routed.
    for path, _ in hits:
        validate_ancestor_contracts(pipeline, path)

    # Unconditional -- deliberately NOT gated on `strict`; see the docstring.
    #
    # ROOT ONLY, and that is a KNOWN GAP rather than a choice. These accessors
    # are called on `pipeline` itself, and `walk_operations` cannot reach a
    # nested pipeline's non-`ops` slots either (`_operation_tree.py:41-43`
    # short-circuits on ImagePipelineCore and yields only `get_ops()`). So a
    # GpuDetector in a NESTED pipeline's meas/post/filters/model is neither
    # staged nor refused -- measured: the run routes to LocalParallelStrategy
    # and performs per-image inference on a CPU node with nothing reported.
    # Pinned by an xfail in `test_gpu_detection_tree_wide.py`; closing it means
    # teaching the walker to descend those slots, not widening this loop.
    for slot in _CPU_ONLY_SLOTS:
        accessor = getattr(pipeline, f"get_{slot}", None)
        if accessor is None:
            continue
        container = accessor()
        if container is None:
            continue
        # get_model() returns Optional[ModelFitter], NOT a dict -- calling
        # .items() on it raises AttributeError.
        entries = (
            container.items()
            if isinstance(container, dict)
            else [(slot, container)]
        )
        for name, op in entries:
            for sub_path, sub_op in walk_operations(op):
                if isinstance(sub_op, GpuDetector):
                    raise UnstageableGpuDetectorError(
                        f"GpuDetector at {slot}/{name}/"
                        f"{'/'.join(sub_path)} cannot be staged: Stage 3 "
                        f"runs the {slot!r} slot on a CPU node"
                    )
    return hits


def pipeline_requires_gpu(pipeline_path: Path) -> bool:
    """Check whether a pipeline JSON contains any GpuDetector, at any depth.

    Scans the whole operation tree, not just the top level: a ``GpuDetector``
    nested inside a ``CompositeDetector`` is still a GPU pipeline, and missing
    it means the run silently completes on CPU with different numbers.

    NOTE the callers handle the refusal differently, and neither was designed:
    ``gui/run_console/_callbacks.py:246-255`` wraps this in
    ``except (OSError, ValueError, TypeError): return False``, and
    ``UnstageableGpuDetectorError`` IS a ``ValueError`` -- so the GUI silently
    reports "not a GPU pipeline" instead of surfacing the message. The CLI
    paths (``_cli_execution_strategies.py:344``, ``:906``, ``:1341``) do not
    catch it, so there the user gets a raw traceback. Both still want
    deciding: the GUI should surface the reason, and the CLI should print it
    rather than a traceback.

    Args:
        pipeline_path: Path to pipeline JSON file.

    Returns:
        True if the pipeline contains at least one GpuDetector operation,
        anywhere in the operation tree.

    Raises:
        UnstageableGpuDetectorError: a GpuDetector sits in a CPU-only slot, or
            under a container that is not a composition primitive. This MUST
            raise from here, not only from ``split_pipeline_at_gpu``: that
            function is reached only after this one returns True, so a refusal
            gated behind it can never fire in production -- the op would route
            to the non-staged strategy and silently run on CPU.
    """
    pipeline = ImagePipeline.from_json(pipeline_path)
    return bool(find_gpu_detectors(pipeline))
