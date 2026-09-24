"""
Pipeline validation for the PhenoTypic CLI.

This module provides validation functions to check pipeline configuration
before running large batch processing jobs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic.abc_ import GpuDetector

from phenotypic import ImagePipeline
from ._cli_types import ExecutionConfig

logger = logging.getLogger(__name__)

#: Plot-backend warnings already logged by this process. A process may validate
#: the same pipeline more than once, and the announcement is meant to appear
#: once.
_ANNOUNCED_PLOT_WARNINGS: set[str] = set()


def validate_pipeline(
    pipeline_path: Path,
    skip_validation: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate that pipeline JSON can be loaded successfully.

    A thin wrapper over :func:`read_pipeline_file` and
    :func:`check_loaded_pipeline`, kept for callers that only need a verdict.
    The CLI itself calls ``load_pipeline_for_validation``
    (``_cli_preflight.py``), which keeps the loaded pipeline for the run
    preflight instead of discarding it.

    Args:
        pipeline_path: Path to pipeline JSON file
        skip_validation: If True, skip validation (for advanced users)

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if skip_validation:
        return True, None

    pipeline, error = read_pipeline_file(pipeline_path)
    if pipeline is None:
        return False, error
    error = check_loaded_pipeline(pipeline)
    return error is None, error


def read_pipeline_file(
    pipeline_path: Path,
) -> Tuple[Optional[ImagePipeline], Optional[str]]:
    """Load a pipeline file, turning every load failure into a message.

    Args:
        pipeline_path: Path to pipeline JSON file.

    Returns:
        ``(pipeline, None)`` on success, else ``(None, message)`` with the
        wording ``validate_pipeline`` has always reported.
    """
    try:
        return ImagePipeline.from_json(pipeline_path), None
    except FileNotFoundError:
        return None, f"Pipeline file not found: {pipeline_path}"
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in pipeline file: {e}"
    except Exception as e:
        return None, f"Failed to load pipeline: {type(e).__name__}: {e}"


def check_loaded_pipeline(pipeline: ImagePipeline) -> Optional[str]:
    """The validation that needs only a loaded pipeline.

    Args:
        pipeline: A pipeline :func:`read_pipeline_file` returned.

    Returns:
        ``None`` when it passes, else the message ``validate_pipeline`` has
        always reported.
    """
    try:
        # Check that pipeline has operations or measurements
        if not pipeline._ops and not pipeline._meas:
            return "Pipeline has no operations or measurements"

        # Backends are checked here rather than per figure during the run: on
        # SLURM this is the submitting process, so a pipeline that will not
        # rasterise says so before the array is submitted. For a single-page
        # image plot this warning is the only record of why no PNG exists.
        # PlotBackendUnavailable is caught here rather than by the generic
        # handler below: the pipeline loaded fine, and "Failed to load
        # pipeline" would send the user to their JSON, not their environment.
        from phenotypic.plotting._pipeline._backends import (
            PlotBackendUnavailable,
            preflight_plot_backends,
        )

        try:
            warning_lines = preflight_plot_backends(pipeline)
        except PlotBackendUnavailable as e:
            return f"Plot backend unavailable: {e}"
        for line in warning_lines:
            if line not in _ANNOUNCED_PLOT_WARNINGS:
                _ANNOUNCED_PLOT_WARNINGS.add(line)
                logger.warning(line)
        return None
    except Exception as e:
        return f"Failed to load pipeline: {type(e).__name__}: {e}"


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


class UnstageableGpuDetectorError(ValueError):
    """A GpuDetector sits somewhere the staged engine cannot drive it."""


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
#: ``ImagePipeline`` is deliberately absent and handled separately in
#: ``_child_contract``.
#:
#: **Entries are matched by ``isinstance``, not by exact type**, so a subclass a
#: user actually holds -- ``class Plotted(CompositeDetector, PlotImage)`` -- is
#: accepted. An exact-type lookup refused every such subclass; the reasoning
#: that kept ``ImagePipeline`` out of this table applies here too and the table
#: did not follow it. The safety argument is preserved by refusing a subclass
#: that **overrides ``_operate``**: the contract was verified against the
#: base's ``_operate``, so a subclass replacing it has been verified against
#: nothing. See ``_child_contract``.
#:
#: Populated lazily by ``_populate_child_contract`` -- read it through that,
#: never directly, or a first reader sees an empty dict.
_CHILD_CONTRACT: dict[type, str] = {}


def _populate_child_contract() -> None:
    """Fill ``_CHILD_CONTRACT`` on first use.

    Deferred rather than done at import time because this module is imported
    by CLI argument validation that may never reach a GPU question, and
    ``phenotypic.detect`` / ``phenotypic.enhance`` are heavy subpackages.

    **Build the table fully, then install it in ONE operation.** The guard is
    ``if _CHILD_CONTRACT: return``, so a thread arriving while the table is
    half-populated sees a truthy one-key dict, returns at the guard, and then
    ``_child_contract`` refuses a ``CompositeEnhance`` -- a placement the
    design permits. Populating key-by-key made that window reachable. It
    matters because the GUI reaches here on threaded Werkzeug
    (``_gui/run_console/_callbacks.py:_staged_gpu_capability``), which turns
    this refusal into a red alert and a disabled Run button -- so the symptom
    there would be a spurious refusal of a pipeline the design permits. (The
    run itself is a separate ``python -m phenotypic`` process with its own
    table, so it does not inherit a wrong answer from this race.)

    The dict literal below is fully constructed before ``update`` is called,
    and ``dict.update`` from a dict is atomic under the GIL, so no thread can
    observe a partial table.
    """
    if _CHILD_CONTRACT:
        return
    from phenotypic.detect import CompositeDetector
    from phenotypic.enhance import CompositeEnhance

    _CHILD_CONTRACT.update(
        {CompositeDetector: "parallel", CompositeEnhance: "parallel"}
    )


def _child_contract(container: Any) -> str:
    """``"sequence"`` or ``"parallel"``; raise for anything else.

    ONLY composition primitives may carry a staged GpuDetector. A domain
    detector is refused even when its current code would classify cleanly --
    ``FilamentousFungiDetector`` feeds ``inoculum_detector`` the container's
    own image today (``_filamentous_fungi_detector.py:398-403``) and so reads
    as ``"parallel"``, but that is incidental to an algorithm that also runs an
    inline ``ContrastStretching`` (``:418``) and a destructive
    ``_subtract_background``. Nothing about being a fungus detector constrains
    it to keep doing that, so the table's safety argument -- "this restates a
    type contract, it does not cache an observation" -- would not hold
    uniformly if it were admitted.

    **Table entries match by subclass, but a subclass that overrides
    ``_operate`` is refused.** An exact-type lookup refused every subclass of
    ``CompositeDetector`` -- including one that only mixes in ``PlotImage`` --
    which is the same defect the note on ``_CHILD_CONTRACT`` already identifies
    for ``ImagePipeline`` ("matched by subclass, not by exact type, so a
    class-keyed entry would refuse a subclass a user actually holds"). A bare
    ``isinstance``, though, would silently admit a subclass whose ``_operate``
    chains its branches, reintroducing exactly the error this narrowing exists
    to prevent: the contract was verified against the *base's* ``_operate`` and
    against nothing else. So the rule is isinstance **plus** an unmodified
    ``_operate``.
    """
    _populate_child_contract()
    if isinstance(container, ImagePipeline):
        return "sequence"
    cls = type(container)
    # Snapshot before scanning. `_populate_child_contract` installs the table
    # in one `update`, but this loop is now a SCAN rather than a key lookup, so
    # a concurrent first-population overlapping this iteration would raise
    # "dictionary changed size during iteration". Cheap insurance, and the
    # table is two entries.
    for base, contract in tuple(_CHILD_CONTRACT.items()):
        if not isinstance(container, base):
            continue
        # `getattr` rather than attribute access: `cls` is `type(container)`
        # where `container` is `Any`, so mypy types it as bare `type` and
        # cannot see `_operate` on it. It also states the honest comparison --
        # a future table entry need not have an `_operate` at all, and two
        # absent ones compare equal, which is the right answer for a base with
        # no algorithm for a subclass to diverge from.
        if getattr(cls, "_operate", None) is not getattr(base, "_operate", None):
            raise UnstageableGpuDetectorError(
                f"a GpuDetector cannot be nested inside {cls.__name__}: it "
                f"subclasses {base.__name__} but overrides _operate, so the "
                f"{contract!r} child-input contract verified for "
                f"{base.__name__} does not carry over to it. Lift the "
                f"detector into a plain {base.__name__} branch, or into the "
                "top-level pipeline."
            )
        return contract
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


def refuse_cpu_only_slot(pipeline: ImagePipeline, path: tuple[str, ...]) -> None:
    """Refuse a GpuDetector reached through any pipeline's non-``ops`` slot.

    Applies at every depth -- the root pipeline's ``meas``/``post``/
    ``filters``/``model`` and those of any nested pipeline alike. Those slots
    run after the op chain (``.measure()`` runs after ``.apply()``), and in the
    staged engine the op chain is Stage 3: a detector there needs Stage 3's
    output as its input, and Stage 2, where GPU inference happens, has
    already finished. A measurement may also hand its nested detector a
    derived image, possibly once per object, while Stage 2 records one array
    per image per slot.

    **Not refusing is worse than not seeing.** An unseen detector routes the
    run to CPU -- slow but correct. A seen and unrefused one would be staged,
    run by Stage 2 on the stored image (the wrong input), and replayed by
    Stage 3: fast and silently wrong.

    Args:
        pipeline: Root pipeline the path is addressed against.
        path: Tree path to a GpuDetector, as returned by
            ``find_operations``.

    Raises:
        UnstageableGpuDetectorError: some segment of *path* is taken out of a
            pipeline's ``meas``/``post``/``filters``/``model`` slot.
    """
    from phenotypic.sdk_._operation_tree import get_at_path, pipeline_slot_of

    for depth, segment in enumerate(path):
        owner = get_at_path(pipeline, path[:depth])  # path[:0] is the root
        slot = pipeline_slot_of(owner, segment)
        if slot is None:
            continue
        where = (
            "the root pipeline"
            if depth == 0
            else f"the pipeline at {'/'.join(path[:depth])}"
        )
        raise UnstageableGpuDetectorError(
            f"GpuDetector at {'/'.join(path)} cannot be staged: it sits in "
            f"the {slot!r} slot of {where}, which runs after the op chain -- "
            "Stage 3 runs it on a CPU node, after GPU inference has "
            "finished. Move the detector into that pipeline's ops."
        )


def find_gpu_detectors(
    pipeline: ImagePipeline, *, strict: bool = False
) -> list[tuple[tuple[str, ...], "GpuDetector"]]:
    """Every ``(path, detector)`` GpuDetector in *pipeline*, tree-wide.

    Placement refusals -- a CPU-only slot of any pipeline at any depth, or an
    unstageable ancestor -- raise **regardless of** ``strict``. They have to
    fire on the production path (``pipeline_requires_gpu``), because the only
    other caller,
    ``split_pipeline_at_gpu``, is reached only once ``pipeline_requires_gpu``
    has already returned True. A refusal reachable only under ``strict=True``
    is a refusal production never performs, while a unit test calling the
    helper directly still passes.

    Args:
        pipeline: The pipeline to scan.
        strict: When True, additionally raise for MORE THAN ONE detector. The
            GUI (``_gui/run_console/_callbacks.py:_staged_gpu_capability``)
            calls the non-strict path, where a multi-detector pipeline should
            report True rather than raise.

            **``split_pipeline_at_gpu`` is the one production caller** and
            passes ``strict=True``. It previously carried its own
            top-level-only scan, which caught two TOP-LEVEL detectors but not
            two nested inside a composite; that scan is gone and this is now
            the only multi-detector refusal in the codebase.

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
    from phenotypic.sdk_._operation_tree import find_operations

    hits = find_operations(pipeline, lambda op: isinstance(op, GpuDetector))

    # Unconditional -- deliberately NOT gated on `strict`; see the docstring.
    # FIRST, before the multi-detector count and before the ancestor
    # contracts: for `meas:MeasureSymZones/center_detector` the ancestor check
    # would refuse too, but it would blame MeasureSymZones rather than the
    # slot, which is the actual reason.
    for path, _ in hits:
        refuse_cpu_only_slot(pipeline, path)

    if strict and len(hits) > 1:
        paths = ", ".join("/".join(p) for p, _ in hits)
        # Name EVERY offending path, not just the count: the message's job is
        # to tell the user which branches to split. Deferred feature, not a
        # limit of the design -- see spec section 13 for the intended N>1
        # execution model. Keep the literal "more than one GpuDetector": it is
        # the wording the existing suite already pins
        # (``test_cli_pipeline_split.py:42``). The path list follows it rather
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
    return hits


def pipeline_requires_gpu(pipeline_path: Path) -> bool:
    """Check whether a pipeline JSON contains any GpuDetector, at any depth.

    Scans the whole operation tree, not just the top level: a ``GpuDetector``
    nested inside a ``CompositeDetector`` is still a GPU pipeline, and missing
    it means the run silently completes on CPU with different numbers.

    How the two front ends surface the refusal:

    - **GUI** -- ``_gui/run_console/_callbacks.py:_staged_gpu_capability``
      catches ``UnstageableGpuDetectorError`` *before* its generic
      ``(OSError, ValueError, TypeError)`` handler; the refusal IS a
      ``ValueError``, so that clause order is load-bearing. It shows the
      message in a red alert, disables Run, and refuses Validate/Run at the
      launch seam before any generation is allocated. An unreadable pipeline
      still takes the generic path and shows no refusal.
    - **CLI** -- ``phenotypicCLI.py`` calls ``uses_staged_gpu_strategy``
      immediately after building ``ExecutionConfig`` and raises
      ``click.UsageError``. That preflight sits above ``--overwrite``
      clearing, run-identity minting and the ``--dry-run`` exit, so a refused
      pipeline can no longer delete a previous run's output, and Validate
      refuses what Run would refuse.

    An earlier version of this note said the GUI's swallow made a refused run
    "route to CPU". It never did: the swallowing helper's one consumer only
    toggled a form section, and the GUI launches runs as ``python -m
    phenotypic`` subprocesses whose routing call raised.

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
