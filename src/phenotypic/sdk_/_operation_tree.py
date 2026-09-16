"""The single traversal over operation-bearing children of a pipeline.

Three callers share this module and each wants something slightly different
(see the spec §4.1): the CLI wants *paths to GpuDetectors*, ``gui/`` wants
*marker presence on an annotation*, ``tune/`` wants *one-level list recursion
with its own depth rule*. The primitive here is the traversal; each caller
adapts it rather than reimplementing it.

Path segments are always **non-empty strings**, because a path is also a
``pipeline_step_path``, which ``_provenance.validate_provenance_journal``
rejects unless every segment is a non-empty string. A list entry is therefore
addressed ``"ops[0]"``, never ``("ops", 0)``.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator, Sequence

_INDEXED = re.compile(r"^(?P<field>[^\[\]]+)\[(?P<index>\d+)\]$")


def _is_operation(value: Any) -> bool:
    """True for anything that can carry further operations."""
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.abc_ import ImageOperation, MeasureFeatures

    return isinstance(value, (ImageOperation, MeasureFeatures, ImagePipeline))


def iter_child_operations(obj: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(segment, child)`` for each operation-bearing child of *obj*."""
    # Key on ImagePipelineCore, not ImagePipeline: `ops` is typed
    # Dict[str, Union[ImageOperation, "ImagePipelineCore"]]
    # (`_image_pipeline_core.py:202`), and ImagePipelineCore has a second
    # concrete subclass (`NapariPipelineViewer`) that is not an ImagePipeline.
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    if isinstance(obj, ImagePipelineCore):
        yield from obj.get_ops().items()
        return

    model_fields = getattr(type(obj), "model_fields", None)
    if not model_fields:
        return

    for field_name in model_fields:
        value = getattr(obj, field_name, None)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if _is_operation(item):
                    yield f"{field_name}[{index}]", item
        elif _is_operation(value):
            yield field_name, value


def walk_operations(pipeline: Any) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Depth-first walk yielding ``(path, operation)`` for every node.

    Does not yield the root itself (its path would be empty, and an empty
    ``pipeline_step_path`` is invalid).

    KNOWN LIMIT: for a **nested** ``ImagePipeline`` this descends only its
    ``ops``, not its own ``meas``/``post``/``filters``/``model``. A GpuDetector
    hidden in a nested pipeline's ``meas`` is therefore neither staged nor
    refused. That shape is not reachable from the GUI builder and has no known
    user, so it is out of scope here -- but it is a gap, not an invariant, and
    the CPU-only-slot refusal covers only the ROOT pipeline's slots.

    **Say the consequence, not just the gap.** "Neither staged nor refused"
    means the run routes to the CPU strategy and the detector performs
    per-image inference on a CPU node, with nothing reported -- the same silent
    failure the tree-wide scan exists to remove, surviving in a shape this
    walker cannot see. Measured for a nested pipeline's ``meas``:
    ``find_gpu_detectors`` returns ``[]``, ``pipeline_requires_gpu`` returns
    ``False``, ``uses_staged_gpu_strategy`` returns ``False``. Pinned by
    ``test_a_gpu_detector_in_a_NESTED_pipelines_meas_slot_is_refused``
    (xfail, non-strict), which XPASSes when this is fixed.
    """

    def visit(
        node: Any, path: tuple[str, ...]
    ) -> Iterator[tuple[tuple[str, ...], Any]]:
        if path:
            yield path, node
        for segment, child in iter_child_operations(node):
            yield from visit(child, path + (segment,))

    yield from visit(pipeline, ())


def find_operations(
    pipeline: Any, predicate: Callable[[Any], bool]
) -> list[tuple[tuple[str, ...], Any]]:
    """Every ``(path, operation)`` in *pipeline* satisfying *predicate*."""
    return [(path, op) for path, op in walk_operations(pipeline) if predicate(op)]


def _child(node: Any, segment: str) -> Any:
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    matched = _INDEXED.match(segment)
    if matched is not None:
        field = matched.group("field")
        index = int(matched.group("index"))
        sequence = getattr(node, field, None)
        if not isinstance(sequence, list) or index >= len(sequence):
            raise KeyError(segment)
        return sequence[index]
    if isinstance(node, ImagePipelineCore):
        ops = node.get_ops()
        if segment not in ops:
            raise KeyError(segment)
        return ops[segment]
    if not hasattr(node, segment):
        raise KeyError(segment)
    return getattr(node, segment)


def get_at_path(root: Any, path: Sequence[str]) -> Any:
    """Resolve *path* against *root*; raise ``KeyError`` if absent."""
    node = root
    for segment in path:
        node = _child(node, segment)
    return node


def substitute_at_path(root: Any, path: Sequence[str], replacement: Any) -> Any:
    """Return a copy of *root* with the node at *path* replaced.

    *root* is never mutated: each node on the path is copied on the way down.
    """
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    if not path:
        return replacement

    head, rest = path[0], tuple(path[1:])

    # `iter_child_operations` and `_child` both descend any ImagePipelineCore,
    # so a walk can hand us a path THROUGH a non-ImagePipeline core (today:
    # NapariPipelineViewer). The rebuild below is ImagePipeline-specific -- it
    # names `nrows`/`ncols`/`qc`/`plots` -- so refuse that node by name rather
    # than fall through to the generic branch, where `head` is an ops KEY and
    # not an attribute, and the failure would surface as a bare KeyError
    # naming the operation instead of the unsupported container.
    if isinstance(root, ImagePipelineCore) and not isinstance(root, ImagePipeline):
        raise TypeError(
            f"cannot substitute inside {type(root).__name__}: "
            "substitute_at_path rebuilds ImagePipeline only"
        )

    if isinstance(root, ImagePipeline):
        ops = dict(root.get_ops())
        if head not in ops:
            raise KeyError(head)
        ops[head] = (
            replacement
            if not rest
            else substitute_at_path(ops[head], rest, replacement)
        )
        # Rebuild carrying every slot a Stage-3 pipeline READS. An earlier
        # draft dropped `qc`, `plots`, `name` and `_provenance_pipeline` -- a
        # booby trap for whoever next reads plots off a substituted pipeline.
        # NOT carried, deliberately: `benchmark`, `verbose`, `reset`,
        # `desc_value` (`_image_pipeline_core.py:190-221`). Harmless for a
        # throwaway Stage-3 pipeline, but do not describe this as complete.
        rebuilt = ImagePipeline(
            ops=ops,
            meas=root.get_meas(),
            post=root.get_post(),
            filters=root.get_filters(),
            model=root.get_model(),
            qc=root.get_qc(),
            plots=root.get_plots(),
            nrows=root.nrows,
            ncols=root.ncols,
        )
        rebuilt.name = root.name
        rebuilt._provenance_pipeline = root._provenance_pipeline
        return rebuilt

    # SHALLOW, deliberately. model_copy(deep=True) would copy the entire
    # subtree -- including the real GpuDetector and whatever its PrivateAttr
    # holds -- only to overwrite one child of it. Measured: a deep copy does
    # carry PrivateAttr through and allocates a new child object, so a loaded
    # torch model in that subtree would be deep-copied and discarded. Latent
    # today (Stage 3 never loads a model) and free to avoid.
    #
    # Untouched siblings stay shared by reference, which is exactly what the
    # original pipeline does with them.
    node = root.model_copy(deep=False)
    matched = _INDEXED.match(head)
    if matched is not None:
        field = matched.group("field")
        index = int(matched.group("index"))
        sequence = list(getattr(node, field))
        if index >= len(sequence):
            raise KeyError(head)
        sequence[index] = (
            replacement
            if not rest
            else substitute_at_path(sequence[index], rest, replacement)
        )
        setattr(node, field, sequence)
        return node

    if not hasattr(node, head):
        raise KeyError(head)
    current = getattr(node, head)
    setattr(
        node,
        head,
        replacement if not rest else substitute_at_path(current, rest, replacement),
    )
    return node
