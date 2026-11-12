"""Warn when a pipeline runs an image enhancer after its last object detector.

Writing ``detect_mat`` resets the object map, so an enhancer that runs after a
detector erases the colonies it found. The check reads the pipeline's step
order, never the image: ``load_synth_yeast_plate()`` ships with its colonies
already in the object map, so a per-image check would fire on the docs' first
example and on most doctests (spec 2026-10-31 docs reorganization, §10.11).

Only an enhancer that no later detector follows is reported. Detect, use the
objects (``GridAligner`` rotates by them), enhance, then detect again is a
documented pattern (``correction/_grid_aligner.py``), and it loses nothing.

It runs once per outermost pipeline run: ``apply``, ``apply_and_measure``,
``apply_with_intermediates`` and ``apply_napari``. A pipeline applied while
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
    its own copy of the image. A pipeline nested directly in ``ops`` is not
    one: it runs inline, so its steps are judged only within its parent's
    chain, where a later detector of the parent may re-detect.

    The walk is structural. From a pipeline it descends only through
    ``get_ops()``, never the ``meas``/``post``/``filters``/``model`` slots, so
    an ``ops`` key that happens to be spelled like a slot segment (``"meas:x"``)
    is still an ``ops`` step.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import ImagePipelineCore
    from phenotypic.sdk_._operation_tree import iter_child_operations

    def visit(
            node: Any, path: tuple[str, ...], *, inline: bool
    ) -> Iterator[tuple[tuple[str, ...], Any]]:
        if isinstance(node, ImagePipelineCore):
            if not inline:
                yield path, node
            for key, child in node.get_ops().items():
                if child is not None:
                    yield from visit(child, path + (key,), inline=True)
        else:
            for segment, child in iter_child_operations(node):
                yield from visit(child, path + (segment,), inline=False)

    yield from visit(root, (), inline=False)


def _is_detector(op: Any) -> bool:
    """True for a step that writes a fresh object map.

    A ``GridApply`` whose ``image_op`` is an ``ObjectDetector`` detects per
    grid section, so it counts too. Any other ``GridApply`` stays an ordinary
    step: one wrapping a sub-pipeline (``GridSectionPipeline``'s) is checked as
    its own chain instead.
    """
    from phenotypic.abc_ import ObjectDetector
    from phenotypic.grid import GridApply

    if isinstance(op, ObjectDetector):
        return True
    return isinstance(op, GridApply) and isinstance(op.image_op, ObjectDetector)


def find_enhancer_after_detector(
        pipeline: Any,
) -> tuple[tuple[str, ...], Any, tuple[str, ...], Any] | None:
    """Return the first enhancer that runs after the last detector of a chain.

    An enhancer between two detectors is not reported: the later detector
    re-detects, so nothing is lost.

    Args:
        pipeline: An ``ImagePipelineCore``.

    Returns:
        ``(enhancer_path, enhancer, detector_path, detector)``, where the
        detector is the last one in the enhancer's step chain, or ``None``
        when no chain runs an enhancer after its last detector.
    """
    from phenotypic.abc_ import ImageEnhancer

    for root_path, chain_root in _chain_roots(pipeline):
        steps = list(_run_order(chain_root, root_path))
        detector_at = [i for i, (_, op) in enumerate(steps) if _is_detector(op)]
        if not detector_at:
            continue
        last = detector_at[-1]
        for path, op in steps[last + 1:]:
            if isinstance(op, ImageEnhancer):
                det_path, detector = steps[last]
                return path, op, det_path, detector
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
            f"at step '{'/'.join(det_path)}' ({det_name}), and no detector runs "
            f"after it. Enhancers write detect_mat, and writing detect_mat "
            f"clears the object map, so {enh_name} erases the colonies "
            f"{det_name} found. Put enhancers before the detector.",
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
