"""Which ``meas`` entries ``--mode process`` runs, skips and refuses (#281 spec E4).

Process mode runs a pipeline's ``ops`` and, since #281 P5, the ``meas`` entries
a process store can hold, writing their tables into each store's own
transaction. User decision 2026-11-18: **run what fits, skip the rest.**

- An entry whose ``requires_objects`` is false (today only a grid embedder)
  **runs**, and its table goes into the store's run folder.
- An entry whose ``requires_objects`` is true is **skipped**, as process mode
  always skipped ``meas``, with one warning naming each slot: a process store
  holds no object map (``write_objmap=False``), so a table keyed on
  ``Object_Label`` would point at labels the store cannot resolve.
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


class ProcessMeasRefusedError(ValueError):
    """A process run would lose a ``meas`` table: a runnable entry, a flat output.

    Raised by the per-image core before the image is read. A ``ValueError``,
    so a direct caller (AutoConvertRaw-GC) catches it as a configuration
    error, and never wrapped as a per-image scientific failure: the pipeline,
    not the image, is at fault.
    """


def runs_in_process_mode(measurer: Any) -> bool:
    """Whether process mode runs this ``meas`` entry: it needs no objects.

    Duck-typed on ``requires_objects``; an entry without the property (no
    ``MeasureFeatures``) is treated as needing objects, so it is skipped.
    """
    return not getattr(measurer, "requires_objects", True)


def object_measurers_in_process_mode(pipeline: "ImagePipeline") -> list[str]:
    """The ``meas`` entries process mode skips, as slot paths.

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
        if not runs_in_process_mode(measurer)
    ]


def process_mode_measurers(pipeline: "ImagePipeline") -> dict[str, Any]:
    """The ``meas`` entries process mode runs, keyed as in ``meas``."""
    return {
        key: measurer
        for key, measurer in pipeline.get_meas().items()
        if runs_in_process_mode(measurer)
    }


def skipped_measurers_warning(pipeline: "ImagePipeline") -> str | None:
    """The one warning line naming the ``meas`` entries process mode skips.

    Returns:
        ``"meas:size, meas:shape skipped: process mode stores no objects"``,
        or ``None`` when nothing is skipped.

    Examples:
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.measure import MeasureSize
        >>> from phenotypic._cli._cli_process_meas import skipped_measurers_warning
        >>> skipped_measurers_warning(ImagePipeline(meas={"size": MeasureSize()}))
        'meas:size skipped: process mode stores no objects'
    """
    skipped = object_measurers_in_process_mode(pipeline)
    if not skipped:
        return None
    return f"{', '.join(skipped)} skipped: {PROCESS_STORES_NO_OBJECTS}"


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
    from ._cli_process_only import _ZARR_CAPABLE_LAYERS

    if process_format == "zarr" and layer in _ZARR_CAPABLE_LAYERS:
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

    A crop embedder needs objects, so it is skipped like any object
    measurer; ``embed`` itself would run it over an empty object map and
    return an empty table keyed on labels the store does not hold.

    Returns:
        ``{"<model>/<crops|grid>": EmbeddingPayload}``; empty when no
        runnable entry is an embedder.
    """
    from phenotypic import ImagePipeline
    from phenotypic.abc_ import MeasureEmbedding

    embedders = {
        key: measurer
        for key, measurer in process_mode_measurers(pipeline).items()
        if isinstance(measurer, MeasureEmbedding)
    }
    if not embedders:
        return {}
    return ImagePipeline(meas=embedders).embed(image)
