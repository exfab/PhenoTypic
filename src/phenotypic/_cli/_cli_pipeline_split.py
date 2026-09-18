"""Split a GpuDetector pipeline at the detector boundary (CLI orchestration).

This is a CLI concern, NOT an ImagePipeline change: ImagePipeline stays a plain
ordered container. The splitter reads the public ordered ``pipeline.get_ops()``
and builds throwaway sub-pipelines the staged strategy runs per stage.
See Spec 1 §3 and the nested-staging spec §4.

The detector is addressed by its **tree path**, not by a top-level key, so a
``GpuDetector`` nested inside a ``CompositeDetector`` (or inside a branch
pipeline of one) is staged exactly like a top-level one. The cut is taken at
the detector's *top-level ancestor*: that ancestor heads ``post_pipeline``, and
Stage 3 substitutes a ``ReplayDetector`` at ``gpu_path`` inside it so the
enclosing operation runs normally with the recorded mask standing in for live
inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector
from phenotypic.sdk_._operation_tree import get_at_path

from ._cli_validation import _child_contract, find_gpu_detectors


@dataclass
class StagePlan:
    """Result of splitting a pipeline at its (single) GpuDetector.

    Attributes:
        pre_pipeline: Ops before the detector's top-level ancestor (Stage 1).
        gpu_path: Tree path to the detector, addressed from the root pipeline.
            List entries are spelled ``"ops[0]"``. Doubles as the detector's
            ``pipeline_step_path`` and as the Stage-2 signal's slot key.
        gpu_detector: The detector itself (Stage 2).
        stage2_prefix: Operations Stage 2 must apply to an in-memory copy
            before calling the detector, because they sit ahead of it *inside
            its own branch*. Empty for a bare leaf and for a top-level
            detector. Entries are ``ImageOperation``s or nested pipelines --
            anything with ``.apply()``.
        post_pipeline: The top-level ancestor onward, plus
            meas/post/filters/model/qc/plots (Stage 3).
    """

    pre_pipeline: ImagePipeline
    gpu_path: tuple[str, ...]
    gpu_detector: GpuDetector
    stage2_prefix: list[Any]
    post_pipeline: ImagePipeline


def _branch_prefix(pipeline: ImagePipeline, path: tuple[str, ...]) -> list[Any]:
    """Ops between the Stage-1 store and the GPU op's model input.

    Dispatches on :func:`_child_contract`, never on ``isinstance``. A
    ``"sequence"`` container contributes the ops preceding the step taken from
    it; a ``"parallel"`` container contributes nothing, because its children
    are parallel branches and none runs "before" another.

    Two failures this has had, in opposite directions, both of which looked
    right:

    - An earlier draft tested ``isinstance(container, ImagePipeline)`` and
      ``continue``d past everything else, so ``_child_contract`` was never
      called and the "composition primitives only" rule existed in prose, in a
      lookup table and in tests, but in **no code path**.
    - The spike's version lacked the root guard on its final block, so a
      TOP-LEVEL detector's prefix was every preceding top-level op -- ops
      Stage 1 had already applied and written to the store, which Stage 2 would
      then re-run on top of themselves.

    Args:
        pipeline: Root pipeline the path is addressed against.
        path: Tree path to the GpuDetector.

    Returns:
        Operations to apply, in order, before inference.
    """
    prefix: list[Any] = []
    for depth in range(len(path)):
        container = get_at_path(pipeline, path[:depth])  # path[:0] -> the root
        if _child_contract(container) == "parallel":
            continue
        if container is pipeline:
            # The root's own preceding ops are Stage 1's; they already ran and
            # are already in the store. This is the guard the spike lacked.
            continue
        ops = container.get_ops()
        keys = list(ops)
        for key in keys[: keys.index(path[depth])]:
            prefix.append(ops[key])
    return prefix


def split_pipeline_at_gpu(pipeline: ImagePipeline) -> StagePlan:
    """Partition *pipeline* around its single GpuDetector, at any depth.

    Args:
        pipeline: The configured pipeline to stage.

    Returns:
        The :class:`StagePlan` the staged engine drives.

    Raises:
        ValueError: Zero GpuDetectors in the pipeline.
        UnstageableGpuDetectorError: More than one GpuDetector, or one in a
            slot the staged engine cannot drive. (A ``ValueError`` subclass.)
    """
    # Tree-wide, and strict. The previous top-level-only scan here caught two
    # TOP-LEVEL detectors but not two nested inside a composite; `strict=True`
    # keeps the literal "more than one GpuDetector" wording the suite pins.
    hits = find_gpu_detectors(pipeline, strict=True)
    if not hits:
        raise ValueError(
            "no GpuDetector in pipeline; staged execution requires exactly one"
        )
    gpu_path, gpu_detector = hits[0]

    ops = pipeline.get_ops()
    keys = list(ops)
    cut = keys.index(gpu_path[0])
    pre_ops = {k: ops[k] for k in keys[:cut]}
    post_ops = {k: ops[k] for k in keys[cut:]}  # ANCESTOR INCLUDED

    for binding in pipeline.get_plots():
        ref = binding.ref
        if ref is None or ref.slot != "ops":
            continue
        # `ref.key in pre_ops` alone is NOT enough. When the detector is itself
        # top-level, `gpu_path[0]` IS the detector and now lives in post_ops --
        # but Stage 3 never runs the real detector, and the substituted
        # ReplayDetector is not plot-capable, so a plot bound to it must still
        # be refused. A plot bound to a *container* ancestor is legal: that
        # ancestor really does run in Stage 3.
        if ref.key in pre_ops or (len(gpu_path) == 1 and ref.key == gpu_path[0]):
            raise ValueError(
                f"plot {binding.id!r} references pre-GPU operation "
                f"{ref.key!r}; staged plotting supports only post-GPU "
                "operations, measurers, aggregate slots, and inline plots"
            )

    pre_pipeline = ImagePipeline(
        ops=pre_ops, nrows=pipeline.nrows, ncols=pipeline.ncols
    )
    post_pipeline = ImagePipeline(
        ops=post_ops,
        meas=pipeline.get_meas(),
        post=pipeline.get_post(),
        filters=pipeline.get_filters(),
        model=pipeline.get_model(),
        qc=pipeline.get_qc(),
        plots=pipeline.get_plots(),
        nrows=pipeline.nrows,
        ncols=pipeline.ncols,
    )
    return StagePlan(
        pre_pipeline=pre_pipeline,
        gpu_path=gpu_path,
        gpu_detector=gpu_detector,
        stage2_prefix=_branch_prefix(pipeline, gpu_path),
        post_pipeline=post_pipeline,
    )
