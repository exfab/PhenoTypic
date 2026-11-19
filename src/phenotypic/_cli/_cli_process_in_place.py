"""In-place process runs: ``--mode process`` with ``--output`` equal to ``--input`` (#281 spec E7).

An in-place run adds the pipeline's run folder to each input store where it
is, without rewriting a pixel. It reuses the measure rewrite's transaction
(:func:`~phenotypic.sdk_._measurement_tables._rewrite_store_tables`): hard-link
the store into a ``.part``, write the run folder (D6) and the run's figure
folder, append the process application to the journal, write the root last,
consolidate, check that the live root is still the one the write started
from, and swap. Pixel arrays stay hard links to the same inodes.

It is for pipelines that detect nothing, such as grid embeddings after
AutoConvertRaw-GC's calibration and denoising. Three things live here, so
their callers cannot disagree:

- :func:`in_place_refusal`, the read-only half's refusals, one line each;
- :func:`process_in_place_core`, the per-image core both process callers
  dispatch to in place;
- :func:`in_place_input_digest`, the input digest of an in-place work id,
  which leaves out the bytes the run writes.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

from phenotypic.sdk_._measurement_tables import (
    StoreRootChangedError,
    array_digest,
    write_run_entry,
)

if TYPE_CHECKING:
    import numpy as np

    from phenotypic import Image, ImagePipeline
    from phenotypic.sdk_ import CommitGuard
    from phenotypic.sdk_._embedding_tables import EmbeddingPayload
    from phenotypic.sdk_._image_figures import RunInitiation, StoredFigures
    from phenotypic.sdk_.typing_ import ImageTypeName

    from ._cli_reference import ReferencePin

logger = logging.getLogger(__name__)

#: Domain tag of :func:`in_place_input_digest`; never the copy digest's.
_IN_PLACE_TREE_DOMAIN = b"phenotypic-store-tree-in-place-v1\0"

#: Top-level store members an in-place run writes, left out of its work id.
_IN_PLACE_WRITTEN_GROUPS = frozenset({"tables", "figures"})


class InPlacePixelsChangedError(ValueError):
    """``apply()`` changed the pixels of the series an in-place run keeps.

    The exported layer's level-0 digest differs from the store's
    (:func:`level0_digest`). The image fails and nothing is written: an
    in-place run never rewrites pixels, so a table computed from changed
    pixels would describe an image the store does not hold. This catches a
    pixel-changing operation the ``ImageCorrector`` refusal misses.
    """


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def level0_digest(array: "np.ndarray") -> str:
    """sha256 over an array's dtype, shape and C-order bytes.

    The ``objmap_digest`` recipe (#281 P3 T2),
    :func:`~phenotypic.sdk_._measurement_tables.array_digest`, applied to a
    store's stored level-0 array and to the exported layer cast to the stored
    dtype. Two arrays digest alike exactly when they hold the same values in
    the same shape and dtype, whatever their memory layout.

    Examples:
        >>> import numpy as np
        >>> from phenotypic._cli._cli_process_in_place import level0_digest
        >>> a = np.arange(6, dtype=np.uint8).reshape(2, 3)
        >>> level0_digest(a) == level0_digest(np.asfortranarray(a))
        True
        >>> level0_digest(a) == level0_digest(a.astype(np.uint16))
        False
    """
    return array_digest(array)


def _written_by_in_place(relative: PurePosixPath) -> bool:
    """Whether an in-place run writes *relative* (the root, ``tables/``, ``figures/``)."""
    parts = relative.parts
    if parts == ("zarr.json",):
        return True
    return bool(parts) and parts[0] in _IN_PLACE_WRITTEN_GROUPS


def in_place_input_digest(store: Path) -> str:
    """The input digest of an in-place work id: the store tree minus what the run writes.

    The :func:`~._cli_failure_tracker.file_sha256` tree digest, with the root
    ``zarr.json``, ``tables/`` and ``figures/`` left out. An in-place run
    writes exactly those, so its own write never changes its work id, and a
    re-run of the same command finds the image it finished. Every pixel
    chunk and every series' metadata stays in, so a store whose pixels
    change (an AutoConvertRaw-GC reprocess) gets a new work id. The domain
    tag differs from the copy digest's, so the two never coincide.

    Args:
        store: A ``*.ome.zarr`` store directory.

    Returns:
        The hex digest.

    Raises:
        IsADirectoryError: If *store* is a directory that is not a store.
        OSError: On a symlink or a non-regular member.
    """
    from phenotypic.sdk_ import is_zarr_store_name

    from ._cli_failure_tracker import store_tree_sha256

    store = Path(store)
    if not (store.is_dir() and is_zarr_store_name(store)):
        raise IsADirectoryError(
            f"{store} is not an OME-Zarr store; an in-place run reads only stores"
        )
    return store_tree_sha256(
        store, domain=_IN_PLACE_TREE_DOMAIN, skip=_written_by_in_place
    )


def _root_sha256(store: Path) -> str:
    """sha256 of the store's root ``zarr.json`` bytes."""
    from phenotypic.sdk_ import ngff_

    return hashlib.sha256(
        (Path(store) / ngff_.STORE_ROOT_JSON).read_bytes()
    ).hexdigest()


def _stored_level0(store: Path, series: str) -> "np.ndarray":
    """The store's level-0 array of *series*, read with zarr-python."""
    import numpy as np
    import zarr

    from phenotypic.sdk_ import ngff_

    level0 = Path(store) / series / "0"
    return np.asarray(zarr.open_array(store=ngff_.long_path(level0), mode="r")[...])


# ---------------------------------------------------------------------------
# Refusals (the read-only half)
# ---------------------------------------------------------------------------


def _object_detector_path(pipeline: "ImagePipeline") -> str | None:
    """The tree path of the first ``ObjectDetector``, any slot, any depth."""
    from phenotypic.abc_ import ObjectDetector
    from phenotypic.sdk_._operation_tree import walk_operations

    for path, node in walk_operations(pipeline):
        if isinstance(node, ObjectDetector):
            return f"{'/'.join(path)} ({type(node).__name__})"
    return None


def _walk_ops_only(node: Any, path: tuple[str, ...]) -> Iterable[tuple[tuple[str, ...], Any]]:
    """Every operation reached through ``ops`` alone, at any depth.

    A pipeline contributes its ``ops`` entries and never its
    ``meas``/``post``/``filters``/``model`` slots, since only ``ops`` is
    applied to the stored pixels. Any other operation contributes every
    operation-valued field (a composite's children).
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )
    from phenotypic.sdk_._operation_tree import iter_child_operations

    children = (
        node.get_ops().items()
        if isinstance(node, ImagePipelineCore)
        else iter_child_operations(node)
    )
    for segment, child in children:
        child_path = path + (segment,)
        yield child_path, child
        yield from _walk_ops_only(child, child_path)


def _image_corrector_path(pipeline: "ImagePipeline") -> str | None:
    """The tree path of the first ``ImageCorrector`` in ``ops``, at any depth."""
    from phenotypic.abc_ import ImageCorrector

    for path, node in _walk_ops_only(pipeline, ()):
        if isinstance(node, ImageCorrector):
            return f"{'/'.join(path)} ({type(node).__name__})"
    return None


def _store_refusal(store: Path, *, layer: str, input_root: Path) -> str | None:
    """Checks 5-7 of :func:`in_place_refusal` for one input, reading one root."""
    from phenotypic.sdk_ import MIGRATION_REMEDY, is_zarr_store_name, ngff_
    from phenotypic.sdk_._table_runs import tables_schema_of_block

    not_a_store = (
        f"--mode process in place adds a run folder to each input store, and "
        f"{store} is not a PhenoTypic store (a raw image, or an OME-Zarr "
        "without attributes.phenotypic). Choose a different --output to "
        "write new stores."
    )
    if not (Path(store).is_dir() and is_zarr_store_name(Path(store))):
        return not_a_store
    try:
        block = ngff_.read_phenotypic_attributes(Path(store))
    except (OSError, KeyError, ValueError):
        return not_a_store
    if not isinstance(block, Mapping):
        return not_a_store
    if tables_schema_of_block(block) == 1:
        return (
            f"{store} still holds its tables in the v1 flat layout "
            "(tables/measurements/). Convert the tree first with:\n"
            f"  python -m phenotypic {MIGRATION_REMEDY} --output {input_root}"
        )
    series = block.get(ngff_.PhenotypicAttr.SERIES)
    try:
        primary = ngff_.primary_series(list(series or ()))
    except ValueError:
        return not_a_store
    if layer != primary:
        return (
            f"--layer {layer} is not the primary series of {store} "
            f"({primary}). In place keeps the store's pixels, so it exports "
            f"the series the store holds: use --layer {primary}."
        )
    return None


def in_place_refusal(
    *,
    pipeline: "ImagePipeline",
    input_root: Path,
    stores: Iterable[Path],
    layer: str,
    process_format: str,
    overwrite: bool,
) -> str | None:
    """One line for the first refusal an in-place process run meets, or ``None``.

    Checked in this order, each before anything is written (#281 spec E7):

    1. ``--overwrite``, which would delete inputs. ``--restart`` is admitted
       (user ruling 2026-11-18): it clears only the non-preserved
       ``.phenotypic/`` machine state under the input root, keeps the
       terminal-failure records and never touches a store, and it is the
       remedy when a different pipeline is run over the same stores;
    2. ``--process-format tiff``, or a layer that writes no store;
    3. an ``ObjectDetector`` anywhere in the pipeline, any slot, any depth. A
       store holds one label image, so a second detecting pipeline would
       leave the first one's table pointing at labels it was not computed
       from;
    4. an ``ImageCorrector`` anywhere in ``ops``, at any depth, which changes
       the pixels the store holds;
    5. ``--input`` naming one store or file rather than the directory holding
       stores (machine state would land inside it), or an input that is not a
       PhenoTypic store (no ``attributes.phenotypic``: a raw image, a
       third-party OME-Zarr);
    6. an input still in the v1 tables layout, pointing at ``--mode migrate``;
    7. a ``--layer`` that is not the store's primary series (``rgb`` when
       present, else ``gray``).

    Checks 5-7 read one root per store and stop at the first failure.

    Args:
        pipeline: The loaded pipeline.
        input_root: The ``--input`` (equal to ``--output``).
        stores: The inputs the scan found.
        layer: The ``--layer``.
        process_format: The resolved ``--process-format``.
        overwrite: Whether ``--overwrite`` was given.

    Returns:
        The refusal, one line (a remedy may follow on a second line), or
        ``None``.

    Examples:
        >>> from pathlib import Path
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic._cli._cli_process_in_place import in_place_refusal
        >>> line = in_place_refusal(
        ...     pipeline=ImagePipeline(), input_root=Path("tree"), stores=[],
        ...     layer="rgb", process_format="zarr", overwrite=True,
        ... )
        >>> line.startswith("--overwrite")
        True
    """
    from ._cli_process_only import _ZARR_CAPABLE_LAYERS

    if overwrite:
        return (
            "--overwrite is refused for an in-place process run (--output "
            "equal to --input): it would delete the run's own inputs. Run "
            "without it (--restart clears only the machine state), or choose "
            "a different --output."
        )
    if process_format != "zarr" or layer not in _ZARR_CAPABLE_LAYERS:
        return (
            f"--layer {layer} --process-format {process_format} writes no "
            "store, and an in-place run adds a run folder to each input store. "
            "Use --layer rgb or --layer gray with --process-format zarr."
        )
    detector = _object_detector_path(pipeline)
    if detector is not None:
        return (
            f"The pipeline holds an ObjectDetector at {detector}, and an "
            "in-place run is for pipelines that detect nothing: a store holds "
            "one label image, so a second detecting pipeline would leave the "
            "first one's table pointing at labels it was not computed from. "
            "Choose a different --output."
        )
    corrector = _image_corrector_path(pipeline)
    if corrector is not None:
        return (
            f"The pipeline's ops hold an ImageCorrector at {corrector}, which "
            "changes the pixels the store holds, and an in-place run never "
            "rewrites pixels. Choose a different --output."
        )
    root = Path(input_root)
    if not root.is_dir() or root.name.endswith(".ome.zarr"):
        return (
            f"--input {input_root} names one image or store; an in-place run "
            "needs the directory that holds the stores, since its machine "
            "state (.phenotypic/) is written under --output."
        )
    for store in stores:
        refusal = _store_refusal(Path(store), layer=layer, input_root=root)
        if refusal is not None:
            return refusal
    return None


# ---------------------------------------------------------------------------
# The per-image core
# ---------------------------------------------------------------------------


def _owned_run_is_current(block: Mapping[str, Any], pipeline_sha256: str) -> bool:
    """Whether *block* holds this pipeline's folder with every entry ``stale: false``."""
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._table_runs import (
        EMBEDDINGS_KEY,
        RUNS_KEY,
        owned_table_run_id,
        tables_schema_of_block,
    )

    if tables_schema_of_block(block) != 2:
        return False
    run_id = owned_table_run_id(block, pipeline_sha256=pipeline_sha256)
    if run_id is None:
        return False
    entry = block[ngff_.PhenotypicAttr.TABLES][RUNS_KEY][run_id]
    embeddings = entry.get(EMBEDDINGS_KEY) or {}
    return all(
        isinstance(value, Mapping) and value.get("stale") is False
        for value in embeddings.values()
    )


def _verify_pixels_unchanged(store: Path, image: "Image", layer: str) -> str:
    """Compare the exported layer's level-0 digest with the store's; return it.

    Raises:
        InPlacePixelsChangedError: When they differ.
    """
    stored = _stored_level0(store, layer)
    expected = level0_digest(stored)
    exported = image._store_series_array(layer)
    if exported.shape == stored.shape:
        exported = exported.astype(stored.dtype, copy=False)
    del stored
    if level0_digest(exported) != expected:
        raise InPlacePixelsChangedError(
            f"apply() changed the {layer!r} pixels of {store}, which an "
            "in-place run never rewrites; nothing was written. Remove the "
            "pixel-changing operation from ops, or choose a different --output."
        )
    return expected


def _in_place_plan(
    *,
    image: "Image",
    pipeline_sha256: str,
    embeddings: Mapping[str, "EmbeddingPayload"],
    figures: "StoredFigures | None",
    layer: str,
    layer_sha256: str,
    selected: dict[str, Any],
) -> Callable[[dict[str, Any]], Callable[[Path], None]]:
    """The ``plan`` an in-place write hands :func:`_rewrite_store_tables`.

    The plan replaces the root's journal with the image's (the store's own
    journal plus this process application, wall-clock fields dropped, as a
    process store publishes it) and returns the writer that adds this
    pipeline's run folder through :func:`write_run_entry` and the figure run.
    Every other run folder and the ``upstream`` block are carried as they
    are. *selected* receives ``"run_id"``, the folder written (``None`` when
    the pipeline embeds nothing).
    """
    from phenotypic._core._provenance import strip_non_reproducible_operation_fields
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._measurement_tables import apply_image_tables_attributes
    from phenotypic.sdk_._table_runs import (
        RUNS_KEY,
        TABLE_RUN_ID_RE,
        UPSTREAM_KEY,
        choose_table_run_id,
        tables_schema_of_block,
    )

    def _plan(phenotypic: dict[str, Any]) -> Callable[[Path], None]:
        schema = tables_schema_of_block(phenotypic)
        if schema == 1:
            raise ValueError(
                "an in-place run cannot write into a store still in the v1 "
                "tables layout; convert the tree with --mode migrate"
            )
        tables = phenotypic.get(ngff_.PhenotypicAttr.TABLES)
        tables = tables if isinstance(tables, Mapping) else {}
        run_id = (
            choose_table_run_id(
                phenotypic if schema == 2 else {}, pipeline_sha256=pipeline_sha256
            )
            if embeddings
            else None
        )
        selected["run_id"] = run_id
        kept = {
            key: dict(value)
            for key, value in (tables.get(RUNS_KEY) or {}).items()
            if key != run_id
        }
        upstream = dict(tables.get(UPSTREAM_KEY) or {})
        journal = deepcopy(image._metadata.provenance_journal)
        strip_non_reproducible_operation_fields(journal)
        phenotypic[ngff_.PhenotypicAttr.PROVENANCE] = journal

        def _populate(part: Path) -> None:
            if run_id is not None:
                match = TABLE_RUN_ID_RE.match(run_id)
                assert match is not None
                own = write_run_entry(
                    part,
                    run_id,
                    order=int(match["order"]),
                    pipeline_sha256=pipeline_sha256,
                    embeddings=embeddings,
                    array_sha256={layer: layer_sha256},
                )
                apply_image_tables_attributes(
                    phenotypic, {**kept, run_id: own}, upstream=upstream
                )
            if figures is not None:
                from phenotypic.sdk_._image_figures import (
                    apply_image_figures_attributes,
                    write_image_figures,
                )

                apply_image_figures_attributes(
                    phenotypic, write_image_figures(part, figures)
                )

        return _populate

    return _plan


def _known_figures(store: Path, figures: "StoredFigures | None") -> "StoredFigures | None":
    """*figures*, or ``None`` when the store's figures layout is one this writer does not know."""
    if figures is None:
        return None
    from phenotypic.sdk_._image_figures import (
        known_figures_schema,
        read_image_figures_descriptor,
    )

    descriptor = read_image_figures_descriptor(Path(store))
    if known_figures_schema(descriptor):
        return figures
    logger.warning(
        "Adding no figure run to %s: its figures schema_version %r is not one "
        "this writer knows",
        store,
        (descriptor or {}).get("schema_version"),
    )
    return None


def _in_place_attempt(
    pipeline_path: Path,
    pipeline: "ImagePipeline",
    pipeline_sha256: str,
    store: Path,
    input_root: Path,
    *,
    layer: str,
    image_type: "ImageTypeName",
    read_kwargs: Mapping[str, Any],
    cli_nrows: int | None,
    cli_ncols: int | None,
    commit_guard: "CommitGuard | None",
    run_initiation: "RunInitiation | None",
    dataset_name: str | None,
    reference_pin: "ReferencePin | None",
    durable: bool | None,
) -> None:
    """Steps 1-4 of :func:`process_in_place_core`, from the live root once.

    Raises:
        StoreRootChangedError: The root changed after step 1 (step 4).
        PerImageScientificError: The apply, the pixel check or the embed
            failed.
    """
    from phenotypic._core._provenance import (
        continuing_provenance_application,
        initialize_cli_provenance,
        set_provenance_status,
    )
    from phenotypic.sdk_._measurement_tables import _rewrite_store_tables

    from ._cli_failure_tracker import PerImageScientificError
    from ._cli_process_meas import embed_in_process_mode
    from ._cli_process_only import build_process_store_figures, read_process_input
    from ._cli_reference import ReferencePlanStaleError, worker_reference_context

    # 1. The root the write is built from, read before the pixels.
    root_sha256 = _root_sha256(store)
    image: Image | None = None
    try:
        # `imread` keeps the store's journal, so the process application
        # below is appended to the store's history, never a fresh one.
        image = read_process_input(
            pipeline,
            store,
            image_type=image_type,
            read_kwargs=read_kwargs,
            cli_nrows=cli_nrows,
            cli_ncols=cli_ncols,
        )
        initialize_cli_provenance(
            image,
            pipeline_path,
            kind="process",
            input_filename=store.name,
            basename_only=True,
        )
        # 2. apply(), then the level-0 check; 3. embed and figures.
        with continuing_provenance_application(image), worker_reference_context(
            input_root, dataset_name, pin=reference_pin
        ):
            pipeline.apply(image, inplace=True)
            layer_sha256 = _verify_pixels_unchanged(store, image, layer)
            embeddings = embed_in_process_mode(pipeline, image)
        figures = _known_figures(
            store,
            build_process_store_figures(
                pipeline, image, run_initiation=run_initiation, image_path=store
            ),
        )
        set_provenance_status(image, "complete")
    except (MemoryError, ReferencePlanStaleError):
        raise
    except Exception as exc:
        raise PerImageScientificError("process", exc) from exc

    if not embeddings and figures is None:
        logger.warning(
            "%s: the pipeline embeds nothing and draws no figure, so an "
            "in-place run has nothing to add; the store is unchanged",
            store,
        )
        return
    # 4. The transaction, checked against step 1's root before the swap.
    selected: dict[str, Any] = {}
    _rewrite_store_tables(
        store,
        plan=_in_place_plan(
            image=image,
            pipeline_sha256=pipeline_sha256,
            embeddings=embeddings,
            figures=figures,
            layer=layer,
            layer_sha256=layer_sha256,
            selected=selected,
        ),
        durable=durable,
        commit_guard=commit_guard,
        clear_figure_run=figures.run.run_id if figures is not None else None,
        clear_table_run=lambda: selected["run_id"],
        clear_tables=bool(embeddings),
        expected_root_sha256=root_sha256,
        consolidate=True,
    )


def process_in_place_core(
    pipeline_path: Path,
    store: Path,
    input_root: Path,
    *,
    layer: str,
    read_kwargs: Mapping[str, Any],
    image_type: "ImageTypeName" = "Image",
    cli_nrows: int | None = None,
    cli_ncols: int | None = None,
    commit_guard: "CommitGuard | None" = None,
    run_initiation: "RunInitiation | None" = None,
    dataset_name: str | None = None,
    reference_pin: "ReferencePin | None" = None,
    durable: bool | None = None,
) -> bool:
    """Add the pipeline's run folder to one input store, in place (#281 spec E7).

    1. Read the root's sha256, then ``imread`` the store, which keeps its
       journal.
    2. Open a ``process`` application and ``apply()``, then compare the
       exported layer's level-0 digest with the store's
       (:func:`level0_digest`). A mismatch fails the image without writing
       (:class:`InPlacePixelsChangedError`).
    3. Embed (:func:`~._cli_process_meas.embed_in_process_mode`: in v1, the
       no-objects embedders only; ``measure()`` is never called) and build
       the figures, as a copy run does.
    4. Write through the measure rewrite's transaction, with
       ``expected_root_sha256`` from step 1 and ``consolidate=True``: the run
       folder (:func:`write_run_entry`), the figure run and the image's
       journal go into the root, and every other folder is carried.
    5. On :class:`~phenotypic.sdk_._measurement_tables.StoreRootChangedError`,
       repeat steps 1-4 once from the new root; a second change is raised.

    **Continuation.** A store that already holds this pipeline's folder with
    every entry ``stale: false`` does no work and is not read. A pipeline
    that embeds nothing and draws no figure has nothing to add, and the store
    is left as it is.

    Args:
        pipeline_path: The pipeline file.
        store: The input store, written in place.
        input_root: The ``--input`` (equal to ``--output``): the reference
            manifest's run directory.
        layer: The ``--layer``, the store's primary series.
        read_kwargs: From ``input_read_settings``.
        image_type: ``"Image"`` or ``"GridImage"``.
        cli_nrows: ``--nrows``, or ``None``.
        cli_ncols: ``--ncols``, or ``None``.
        commit_guard: Publication guard, checked at the swap.
        run_initiation: The run's initial call; its date names the figure run.
        dataset_name: The image's dataset, for the reference manifest.
        reference_pin: The reference digest the work id was computed from.
        durable: ``fsync`` before the swap; ``None`` auto-detects SLURM.

    Returns:
        ``True``.

    Raises:
        PerImageScientificError: The apply, the pixel check or the embed
            failed; a terminal failure of this image.
        StoreRootChangedError: The root changed twice under the write. Never
            wrapped: the image is not at fault, and a plain re-run retries it.
        ProcessMeasRefusedError: Never in place, which the refusals restrict
            to store layers; kept for a direct caller.
    """
    from phenotypic import ImagePipeline

    from ._cli_process_meas import (
        ProcessMeasRefusedError,
        process_meas_refusal,
        skipped_measurers_warning,
    )

    store = Path(store)
    pipeline = ImagePipeline.from_json(pipeline_path)
    refusal = process_meas_refusal(pipeline, layer=layer, process_format="zarr")
    if refusal is not None:
        raise ProcessMeasRefusedError(refusal)
    skipped = skipped_measurers_warning(pipeline)
    if skipped is not None:
        logger.warning(skipped)
    pipeline_sha256 = hashlib.sha256(Path(pipeline_path).read_bytes()).hexdigest()

    from phenotypic.sdk_ import ngff_

    if _owned_run_is_current(ngff_.read_phenotypic_attributes(store), pipeline_sha256):
        logger.info("%s already holds this pipeline's run folder; nothing to do", store)
        return True

    for attempt in (1, 2):
        try:
            _in_place_attempt(
                pipeline_path,
                pipeline,
                pipeline_sha256,
                store,
                Path(input_root),
                layer=layer,
                image_type=image_type,
                read_kwargs=read_kwargs,
                cli_nrows=cli_nrows,
                cli_ncols=cli_ncols,
                commit_guard=commit_guard,
                run_initiation=run_initiation,
                dataset_name=dataset_name,
                reference_pin=reference_pin,
                durable=durable,
            )
            return True
        except StoreRootChangedError:
            if attempt == 2:
                raise
            logger.warning(
                "%s changed while its in-place write was built; rebuilding once "
                "from the new root",
                store,
            )
    raise AssertionError("unreachable")
