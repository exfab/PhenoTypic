"""Which ``meas`` entries ``--mode process`` runs, skips and refuses (#281 spec E4).

Process mode runs a pipeline's ``ops`` and, since #281 P5, the ``meas`` entries
a process store can hold, writing their tables into each store's own
transaction. User decision 2026-11-18: **run what fits, skip the rest.**

- An **embedder** whose ``requires_objects`` is false (a grid embedder)
  **runs**, and its table goes into the store's run folder. In v1 nothing
  else runs: a process store holds embedding tables only.
- Every other entry is **skipped**, as process mode always skipped ``meas``,
  and named in one warning line, never dropped silently:

  - one that needs objects (``requires_objects`` true): a process store holds
    no object map (``write_objmap=False``), so a table keyed on
    ``Object_Label`` would point at labels the store cannot resolve;
  - one that needs no objects but is not an embedder: ``measure()`` is never
    called in process mode, so it has no table to go into.
- A runnable entry with a flat output (``--process-format tiff``, ``--layer
  objmap``, ``--layer detect_mat``) is **refused**: its table would be lost.

The checks live here so their two callers cannot disagree: the CLI's
read-only half (a ``click.UsageError`` and a warning line), and
:func:`~phenotypic._cli._cli_process_only.process_single_apply_only_core`
(a :class:`ProcessMeasRefusedError`, a ``ValueError``, and a logged warning),
which AutoConvertRaw-GC calls directly without passing the CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phenotypic import Image, ImagePipeline
    from phenotypic.sdk_._embedding_tables import EmbeddingPayload

#: Why an object measurer is skipped, for the warning and its readers.
PROCESS_STORES_NO_OBJECTS = "process mode stores no objects"

#: Why a no-objects entry that is not an embedder is skipped (v1).
PROCESS_STORES_ONLY_EMBEDDINGS = "process mode stores only embedding tables"


class ProcessMeasRefusedError(ValueError):
    """A process run would lose a ``meas`` table: a runnable entry, a flat output.

    Raised by the per-image core before the image is read. A ``ValueError``,
    so a direct caller (AutoConvertRaw-GC) catches it as a configuration
    error, and never wrapped as a per-image scientific failure: the pipeline,
    not the image, is at fault.
    """


def _needs_objects(measurer: Any) -> bool:
    """``requires_objects``; an entry without the property is taken to need them."""
    return bool(getattr(measurer, "requires_objects", True))


def runs_in_process_mode(measurer: Any) -> bool:
    """Whether process mode runs this ``meas`` entry: a no-objects embedder.

    v1 runs only :class:`~phenotypic.abc_.MeasureEmbedding` entries that need
    no objects (a grid embedder). Every other entry is skipped and named by
    :func:`skipped_measurers_warning`.
    """
    from phenotypic.abc_ import MeasureEmbedding

    return isinstance(measurer, MeasureEmbedding) and not _needs_objects(measurer)


def object_measurers_in_process_mode(pipeline: "ImagePipeline") -> list[str]:
    """The ``meas`` entries process mode skips because they need objects.

    Only the root pipeline's ``meas`` runs in process mode: a pipeline nested
    in ``ops`` is applied, never measured.

    Args:
        pipeline: The loaded pipeline.

    Returns:
        ``meas:<key>`` for each entry whose ``requires_objects`` is true, in
        ``meas`` order.

    Examples:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.measure import MeasureSize
        >>> from phenotypic._cli._cli_process_meas import (
        ...     object_measurers_in_process_mode,
        ... )
        >>> object_measurers_in_process_mode(ImagePipeline(meas={"size": MeasureSize()}))
        ['meas:size']
    """
    return [
        f"meas:{key}"
        for key, measurer in pipeline.get_meas().items()
        if _needs_objects(measurer)
    ]


def process_mode_measurers(pipeline: "ImagePipeline") -> dict[str, Any]:
    """The ``meas`` entries process mode runs (no-objects embedders), keyed as in ``meas``."""
    return {
        key: measurer
        for key, measurer in pipeline.get_meas().items()
        if runs_in_process_mode(measurer)
    }


def skipped_measurers_warning(pipeline: "ImagePipeline") -> str | None:
    """The one warning line naming every ``meas`` entry process mode skips.

    Each entry not :func:`runs_in_process_mode` is named with its reason:
    :data:`PROCESS_STORES_NO_OBJECTS` when it needs objects,
    :data:`PROCESS_STORES_ONLY_EMBEDDINGS` when it needs none but is not an
    embedder. Both groups share the line, separated by ``"; "``.

    Returns:
        e.g. ``"meas:size, meas:shape skipped: process mode stores no
        objects"``, or ``None`` when nothing is skipped.

    Examples:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.measure import MeasureSize
        >>> from phenotypic._cli._cli_process_meas import skipped_measurers_warning
        >>> skipped_measurers_warning(ImagePipeline(meas={"size": MeasureSize()}))
        'meas:size skipped: process mode stores no objects'
    """
    groups: dict[str, list[str]] = {
        PROCESS_STORES_NO_OBJECTS: [],
        PROCESS_STORES_ONLY_EMBEDDINGS: [],
    }
    for key, measurer in pipeline.get_meas().items():
        if runs_in_process_mode(measurer):
            continue
        reason = (
            PROCESS_STORES_NO_OBJECTS
            if _needs_objects(measurer)
            else PROCESS_STORES_ONLY_EMBEDDINGS
        )
        groups[reason].append(f"meas:{key}")
    parts = [
        f"{', '.join(slots)} skipped: {reason}"
        for reason, slots in groups.items()
        if slots
    ]
    return "; ".join(parts) or None


def process_meas_refusal(
    pipeline: "ImagePipeline", *, layer: str, process_format: str
) -> str | None:
    """Why a process run of *pipeline* would lose a ``meas`` table, or ``None``.

    Refused: a ``meas`` entry process mode runs (:func:`process_mode_measurers`)
    when the output is not a store, i.e. anything but ``--layer rgb``/``gray``
    with ``--process-format zarr``. A skipped object measurer is never
    refused; :func:`skipped_measurers_warning` names it.

    Args:
        pipeline: The loaded pipeline.
        layer: The ``--layer`` being exported.
        process_format: The resolved ``--process-format``.

    Returns:
        One line naming the output and each runnable ``meas`` slot, or
        ``None`` when nothing would be lost.

    Examples:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.measure import MeasureSize
        >>> from phenotypic._cli._cli_process_meas import process_meas_refusal
        >>> pipeline = ImagePipeline(meas={"size": MeasureSize()})
        >>> process_meas_refusal(pipeline, layer="rgb", process_format="tiff") is None
        True
    """
    from ._cli_types import writes_process_store

    if writes_process_store(layer, process_format):
        return None
    runnable = process_mode_measurers(pipeline)
    if not runnable:
        return None
    slots = ", ".join(f"meas:{key}" for key in runnable)
    return (
        f"--mode process --layer {layer} --process-format {process_format} "
        f"writes a flat file, not a store, so the table of {slots} would be "
        "lost. Use --layer rgb or --layer gray with --process-format zarr, "
        "or remove it from meas."
    )


def embed_in_process_mode(
    pipeline: "ImagePipeline", image: "Image"
) -> dict[str, "EmbeddingPayload"]:
    """:meth:`ImagePipeline.embed` over only the entries process mode runs.

    An objects embedder needs objects, so it is skipped like any object
    measurer; ``embed`` itself would run it over an empty object map and
    return an empty table keyed on labels the store does not hold.

    Returns:
        ``{"<model>/<layer>/<region>": EmbeddingPayload}``; empty when the
        pipeline has no no-objects embedder.
    """
    from phenotypic import ImagePipeline

    embedders = process_mode_measurers(pipeline)
    if not embedders:
        return {}
    return ImagePipeline(meas=embedders).embed(image)
