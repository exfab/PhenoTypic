"""Warn when a pipeline runs an image enhancer after an object detector.

Writing ``detect_mat`` resets the object map, so an enhancer that runs after a
detector erases the colonies it found. The check reads the pipeline's step
order, never the image: ``load_synth_yeast_plate()`` ships with its colonies
already in the object map, so a per-image check would fire on the docs' first
example and on most doctests (spec 2026-10-31 docs reorganization, §10.11).

It runs once per outermost ``ImagePipelineCore.apply``. A pipeline applied while
another is already running (a nested ``ops`` pipeline, a detector's branch
pipeline, a ``GridApply`` sub-pipeline) is covered by the outer check and skips
its own, so one call warns at most once.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_CHECK_RUNNING: ContextVar[bool] = ContextVar(
    "phenotypic_pipeline_step_order_check_running", default=False
)


def _run_order(pipeline: Any, prefix: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(path, op)`` for the leaf steps of *pipeline*'s ``ops`` in run order.

    A pipeline nested in ``ops`` runs inline on the same image (the parent
    forces ``reset=False`` on it), so its steps are expanded in place. Any
    other operation is one step, whatever it holds in its own fields.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import ImagePipelineCore

    for key, op in pipeline.get_ops().items():
        if op is None:
            continue
        path = prefix + (key,)
        if isinstance(op, ImagePipelineCore):
            yield from _run_order(op, path)
        else:
            yield path, op


def _chain_roots(root: Any) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield every pipeline in *root*'s ``ops`` tree that runs its own step chain.

    That is *root* itself and any pipeline held in an operation's field (a
    ``CompositeDetector`` branch, a ``GridApply`` sub-pipeline), which runs on
    its own copy of the image. Pipelines under the ``meas``/``post``/
    ``filters``/``model`` slots are not ``ops`` and are skipped. A pipeline
    nested directly in ``ops`` is yielded too; its steps are a subset of its
    parent's chain, so it can only repeat a finding, and the first one wins.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import ImagePipelineCore
    from phenotypic.sdk_._operation_tree import PIPELINE_SLOTS, walk_operations

    yield (), root
    for path, node in walk_operations(root):
        if not isinstance(node, ImagePipelineCore):
            continue
        if any(seg.partition(":")[0] in PIPELINE_SLOTS and ":" in seg for seg in path):
            continue
        yield path, node


def find_enhancer_after_detector(
        pipeline: Any,
) -> tuple[tuple[str, ...], Any, tuple[str, ...], Any] | None:
    """Return the first enhancer that runs after a detector, with that detector.

    Args:
        pipeline: An ``ImagePipelineCore``.

    Returns:
        ``(enhancer_path, enhancer, detector_path, detector)``, where the
        detector is the last one to run before the enhancer, or ``None`` when
        every enhancer runs before every detector in each step chain.
    """
    from phenotypic.abc_ import ImageEnhancer, ObjectDetector

    for root_path, chain_root in _chain_roots(pipeline):
        last_detector: tuple[tuple[str, ...], Any] | None = None
        for path, op in _run_order(chain_root, root_path):
            if isinstance(op, ObjectDetector):
                last_detector = (path, op)
            elif isinstance(op, ImageEnhancer) and last_detector is not None:
                return path, op, last_detector[0], last_detector[1]
    return None


def warn_if_enhancer_after_detector(pipeline: Any, *, stacklevel: int) -> None:
    """Emit one ``DetectionClearedWarning`` if *pipeline* enhances after detecting."""
    from phenotypic.abc_ import DetectionClearedWarning

    found = find_enhancer_after_detector(pipeline)
    if found is None:
        return
    enh_path, enhancer, det_path, detector = found
    enh_name, det_name = type(enhancer).__name__, type(detector).__name__
    warnings.warn(
            f"Step '{'/'.join(enh_path)}' ({enh_name}) runs after the detector "
            f"at step '{'/'.join(det_path)}' ({det_name}). Enhancers write "
            f"detect_mat, and writing detect_mat clears the object map, so "
            f"{enh_name} erases the colonies {det_name} found. Put enhancers "
            f"before the detector.",
            DetectionClearedWarning,
            stacklevel=stacklevel + 1,
    )


@contextmanager
def step_order_checked(pipeline: Any, *, stacklevel: int) -> Iterator[None]:
    """Check *pipeline*'s step order unless an outer pipeline run already did.

    Args:
        pipeline: The pipeline about to run.
        stacklevel: Frames above the caller of this context manager that the
            warning should point at, as for :func:`warnings.warn`.
    """
    if _CHECK_RUNNING.get():
        yield
        return
    # +2: this generator frame and contextlib's __enter__ sit between the
    # warning and the caller of ``with step_order_checked(...)``.
    warn_if_enhancer_after_detector(pipeline, stacklevel=stacklevel + 2)
    token = _CHECK_RUNNING.set(True)
    try:
        yield
    finally:
        _CHECK_RUNNING.reset(token)
