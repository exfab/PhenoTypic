"""Split a GpuDetector pipeline at the detector boundary (CLI orchestration).

This is a CLI concern, NOT an ImagePipeline change: ImagePipeline stays a plain
ordered container. The splitter reads the public ordered ``pipeline.get_ops()``
and builds throwaway sub-pipelines the staged strategy runs per stage.
See Spec 1 §3.
"""

from __future__ import annotations

from dataclasses import dataclass

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector


@dataclass
class StagePlan:
    """Result of splitting a pipeline at its (single) GpuDetector."""

    pre_pipeline: ImagePipeline      # ops before the detector (Stage 1)
    gpu_key: str                     # configured detector key (provenance path)
    gpu_detector: GpuDetector        # the detector itself (Stage 2)
    post_pipeline: ImagePipeline     # ops after + meas/post/filters/model/qc (Stage 3)


def split_pipeline_at_gpu(pipeline: ImagePipeline) -> StagePlan:
    """Partition *pipeline* at the first GpuDetector into pre/detector/post.

    Raises:
        ValueError: if the pipeline contains zero or more than one GpuDetector.
    """
    ops = pipeline.get_ops()  # ordered dict
    gpu_keys = [k for k, op in ops.items() if isinstance(op, GpuDetector)]
    if len(gpu_keys) == 0:
        raise ValueError(
            "no GpuDetector in pipeline; staged execution requires exactly one"
        )
    if len(gpu_keys) > 1:
        raise ValueError(
            "staged execution does not support more than one GpuDetector "
            f"per pipeline (found {len(gpu_keys)}: {gpu_keys})"
        )

    gpu_key = gpu_keys[0]
    gpu_detector = ops[gpu_key]
    assert isinstance(gpu_detector, GpuDetector)  # guaranteed by gpu_keys filter
    keys = list(ops.keys())
    cut = keys.index(gpu_key)
    pre_ops = {k: ops[k] for k in keys[:cut]}
    post_ops = {k: ops[k] for k in keys[cut + 1:]}

    for binding in pipeline.get_plots():
        ref = binding.ref
        if ref is None or ref.slot != "ops":
            continue
        if ref.key == gpu_key or ref.key in pre_ops:
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
        gpu_key=gpu_key,
        gpu_detector=gpu_detector,
        post_pipeline=post_pipeline,
    )
