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

A pipeline's four non-``ops`` slots are addressed with the slot name as a
colon-separated namespace: ``"meas:<key>"``, ``"post:<key>"``,
``"filters:<key>"`` for the three dict slots, and ``"model:<ClassName>"`` for
the single ``model``. A bare ``"meas"`` would collide with an ``ops`` key a
user is free to choose, and the ``"ops[0]"`` bracket form admits only integer
indices.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator, Sequence

_INDEXED = re.compile(r"^(?P<field>[^\[\]]+)\[(?P<index>\d+)\]$")

#: The ``ImagePipelineCore`` slots that run after the op chain. The first three
#: are ``Dict[str, ...]``; ``model`` is ``Optional[ModelFitter]``.
PIPELINE_SLOTS = ("meas", "post", "filters", "model")
_DICT_SLOTS = ("meas", "post", "filters")
_SLOT_SEPARATOR = ":"


def _is_operation(value: Any) -> bool:
    """True for anything that can carry further operations.

    Admits any ``ImagePipelineCore``, not only ``ImagePipeline``:
    ``_provenance.apply_child`` pushes a ``pipeline_step`` segment for every
    child a container hands it, ungated by type, so a bare
    ``NapariPipelineViewer`` held in an operation-valued field must be yielded
    here too or its recorded path would be one this walker never produces.

    This predicate gates only operation-valued **fields**. A pipeline's slot
    children (``PostMeasurement``, ``SetAnalyzer``, ``ModelFitter``) are not
    admitted by it and do not need to be: ``iter_child_operations`` yields them
    because of the slot they sit in, not because of their type.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )
    from phenotypic.abc_ import ImageOperation, MeasureFeatures

    return isinstance(value, (ImageOperation, MeasureFeatures, ImagePipelineCore))


def _iter_slot_children(pipeline: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(segment, child)`` for every entry of *pipeline*'s four slots.

    Unconditional -- never filtered through ``_is_operation``, which rejects
    ``ModelFitter``, ``SetAnalyzer`` and ``PostMeasurement``. The slot's
    declared type is what makes these children operation-bearing.
    """
    for slot in _DICT_SLOTS:
        for key, child in getattr(pipeline, f"get_{slot}")().items():
            yield f"{slot}{_SLOT_SEPARATOR}{key}", child
    model = pipeline.get_model()
    if model is not None:
        yield f"model{_SLOT_SEPARATOR}{type(model).__name__}", model


def iter_child_operations(obj: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(segment, child)`` for each operation-bearing child of *obj*."""
    # Key on ImagePipelineCore, not ImagePipeline: `ops` is typed
    # Dict[str, Union[ImageOperation, "ImagePipelineCore"]]
    # (`_image_pipeline_core.py:202`), and a bare `NapariPipelineViewer` IS an
    # ImagePipelineCore without being an ImagePipeline. Note it is an
    # ANCESTOR of ImagePipeline, not a sibling -- the MRO is ImagePipeline ->
    # SerializablePipeline -> NapariPipelineViewer -> ImagePipelineCore -- so
    # `isinstance(some_pipeline, NapariPipelineViewer)` is True and cannot be
    # used to tell them apart. Only `not isinstance(x, ImagePipeline)` does.
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    if isinstance(obj, ImagePipelineCore):
        yield from obj.get_ops().items()
        yield from _iter_slot_children(obj)
        return

    model_fields = getattr(type(obj), "model_fields", None)
    if not model_fields:
        return

    for field_name in model_fields:
        value = getattr(obj, field_name, None)
        # Sequences are indexed; a dict is keyed with the same colon namespace
        # the pipeline slots use, because the bracket form admits only
        # integers. No shipped operation declares a dict or tuple of
        # operations -- every one is a single operation or a list -- but an
        # unwalked field is worse than an unsupported one: a GpuDetector there
        # would be invisible to `find_gpu_detectors`, so the run would report
        # "not a GPU pipeline" and go to the CPU strategy with nothing said.
        # A set is deliberately not walked: it has no stable segment to name
        # an entry by, so a path into one could not round-trip.
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if _is_operation(item):
                    yield f"{field_name}[{index}]", item
        elif isinstance(value, dict):
            for key, item in value.items():
                if _is_operation(item) and isinstance(key, str) and key:
                    yield f"{field_name}{_SLOT_SEPARATOR}{key}", item
        elif _is_operation(value):
            yield field_name, value


def walk_operations(pipeline: Any) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Depth-first walk yielding ``(path, operation)`` for every node.

    Does not yield the root itself (its path would be empty, and an empty
    ``pipeline_step_path`` is invalid).

    Every pipeline on the way down -- the root and any nested one -- is
    descended through its ``ops`` **and** its ``meas``/``post``/``filters``/
    ``model`` slots. Seeing the slots is what lets ``find_gpu_detectors`` refuse
    a GpuDetector placed in one: those slots run after the op chain, so the
    staged engine would run such a detector in Stage 3, on a CPU node, after
    GPU inference has finished. Invisible, it would instead route the whole
    run to the CPU strategy with nothing reported.
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


_ABSENT = object()


def _slot_child(pipeline: Any, segment: str) -> tuple[str | None, Any]:
    """``(slot, child)`` if *segment* addresses one of *pipeline*'s slots.

    Returns ``(None, _ABSENT)`` when *segment* is not spelled as a slot
    segment, and ``(slot, _ABSENT)`` when it is but names nothing there --
    including a ``"model:<ClassName>"`` whose class no longer matches, so a
    recorded path stops resolving once the model it addressed is replaced.
    """
    slot, separator, key = segment.partition(_SLOT_SEPARATOR)
    if not separator or slot not in PIPELINE_SLOTS:
        return None, _ABSENT
    if slot == "model":
        model = pipeline.get_model()
        if model is None or type(model).__name__ != key:
            return slot, _ABSENT
        return slot, model
    return slot, getattr(pipeline, f"get_{slot}")().get(key, _ABSENT)


def _pipeline_child(pipeline: Any, segment: str) -> tuple[str | None, Any]:
    """Resolve *segment* on a pipeline: ``(slot or None, child)``.

    Raises ``KeyError`` when nothing matches, and also when the segment is
    BOTH an ``ops`` key and a resolvable slot segment -- a user may key an op
    ``"meas:X"`` -- rather than silently picking one of the two.
    """
    ops = pipeline.get_ops()
    slot, child = _slot_child(pipeline, segment)
    if segment in ops:
        if child is not _ABSENT:
            raise KeyError(
                f"{segment!r} is ambiguous: it is both an ops key and a "
                f"{slot!r} slot entry"
            )
        return None, ops[segment]
    if child is _ABSENT:
        raise KeyError(segment)
    return slot, child


def pipeline_slot_of(node: Any, segment: str) -> str | None:
    """The slot (``"meas"``, ``"post"``, ...) *segment* takes out of *node*.

    ``None`` when *node* is not a pipeline or *segment* names one of its
    ``ops``. Decided against the live node, never by parsing the string alone:
    an ``ops`` key may itself contain a colon.

    Raises:
        KeyError: *segment* does not resolve on *node*.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    if not isinstance(node, ImagePipelineCore):
        return None
    slot, _ = _pipeline_child(node, segment)
    return slot


def _child(node: Any, segment: str) -> Any:
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    # Pipelines first: their segments are ops keys or slot segments, either of
    # which may contain brackets, and a pipeline has no list-valued child.
    if isinstance(node, ImagePipelineCore):
        _, child = _pipeline_child(node, segment)
        return child
    matched = _INDEXED.match(segment)
    if matched is not None:
        field = matched.group("field")
        index = int(matched.group("index"))
        sequence = getattr(node, field, None)
        if not isinstance(sequence, (list, tuple)) or index >= len(sequence):
            raise KeyError(segment)
        return sequence[index]
    # A dict-valued field's entry, spelled as `iter_child_operations` yields
    # it. Tried before the plain attribute lookup so a field whose NAME
    # contains a colon cannot shadow it; `hasattr` below still resolves such a
    # field when nothing keyed matches.
    field, separator, key = segment.partition(_SLOT_SEPARATOR)
    if separator:
        mapping = getattr(node, field, None)
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
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
        # Classify `head` against the live pipeline BEFORE the ops lookup, for
        # two reasons that a plain `head not in ops` check gets wrong:
        #
        # - A slot segment ("meas:<key>" ...) resolves on the pipeline but is
        #   not an ops key, so it used to raise a bare KeyError naming the
        #   segment -- the same confusing failure the TypeError guard above
        #   exists to prevent for an unsupported container. Substituting into a
        #   slot is unsupported: the staged engine refuses every slot path in
        #   find_gpu_detectors, so no StagePlan can carry one. Say so by name.
        # - A segment that is BOTH an ops key and a slot entry (a user may key
        #   an op "meas:X") is ambiguous. `get_at_path` refuses it; checking
        #   `head in ops` first would have silently substituted into the ops
        #   entry instead, so the two functions disagreed about the same path.
        #   `pipeline_slot_of` raises KeyError for that case, as `_child` does.
        slot = pipeline_slot_of(root, head)
        if slot is not None:
            raise TypeError(
                f"cannot substitute at {head!r}: substitute_at_path does not "
                f"descend a pipeline's {slot!r} slot"
            )
        ops = dict(root.get_ops())
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
