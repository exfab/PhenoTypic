"""Process-only CLI mode: run pipeline.apply() and export a single layer.

Used when the user wants PhenoTypic preprocessing/detection output without the
full measurement/analysis suite. See
docs/superpowers/specs/2026-06-03-cli-process-only-and-phenotypic-cache-design.md.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import click

from phenotypic import GridImage, Image, ImagePipeline
from phenotypic.sdk_ import (
    CommitGuard,
    atomic_write_with_writer,
    source_image_stem,
    upstream_entries_for_input,
)
from phenotypic._core._provenance import (
    continuing_provenance_application,
    initialize_cli_provenance,
    set_provenance_status,
)
from phenotypic.sdk_.typing_ import ImageTypeName, ProcessFormat, ProcessOnlyLayer
from ._cli_failure_tracker import PerImageScientificError
from ._cli_process_meas import (
    ProcessMeasRefusedError,
    embed_in_process_mode,
    process_meas_refusal,
    skipped_measurers_warning,
)
from ._cli_reference import (
    ReferencePin,
    ReferencePlanStaleError,
    worker_reference_context,
)

if TYPE_CHECKING:
    from phenotypic.sdk_._embedding_tables import EmbeddingPayload
    from phenotypic.sdk_._image_figures import RunInitiation, StoredFigures

logger = logging.getLogger(__name__)

#: Layers that have a single-series OME-Zarr form. ``objmap`` and ``detect_mat``
#: are absent for two different reasons, and :func:`write_process_only_layer`
#: refuses each with its own message saying which (spec §5.3).
_ZARR_CAPABLE_LAYERS: frozenset[str] = frozenset({"rgb", "gray"})

#: Why each non-store layer is refused. Two entries, two reasons, deliberately
#: not collapsed: ``objmap`` is refused by the FORMAT, ``detect_mat`` by US.
_NO_ZARR_FORM: dict[str, str] = {
    "objmap": (
        "--layer objmap has no single-series OME-Zarr form (NGFF 0.5 §2.6: "
        "a labels group is nested inside an image group and is not itself an "
        "image). Use --process-format tiff for the 16-bit raw-label PNG, or "
        "--layer rgb."
    ),
    "detect_mat": (
        "--layer detect_mat has no single-series OME-Zarr form: PhenoTypic's "
        "store writer requires a primary series (rgb or gray) and detect_mat "
        "is neither. Use --process-format tiff for the float TIFF, or "
        "--layer gray."
    ),
}


def resolve_process_format(
    layer: ProcessOnlyLayer, requested: ProcessFormat | None
) -> ProcessFormat:
    """Resolve ``--process-format``, whose default depends on ``--layer``.

    The default is not a single constant: ``rgb`` and ``gray`` default to
    ``zarr`` and ``detect_mat``/``objmap`` to ``tiff``, so every bare command
    keeps working and each layer gets the format that suits it. The rule lives
    here rather than in the option declaration so it has exactly one home --
    the user-facing CLI and the per-image worker both call it.

    The two refusals carry different reasons on purpose. ``objmap`` is refused
    by NGFF: 0.5 §2.6 nests a label image inside an image group and states
    that the labels group is not itself an image, so a standalone objmap store
    has no conformant single-series form. ``detect_mat`` is refused by
    PhenoTypic: ``_write_store_part`` calls ``ngff_.primary_series``
    unconditionally and that function accepts only ``rgb`` or ``gray``, so
    ``_save_store(series=("detect_mat",))`` raises ``no primary series among
    ['detect_mat']``. The first is a format rule and unfixable here; the second
    is ours, and widening ``primary_series`` is a change that belongs in its own
    design. A user reading the message deserves to know which they are hitting.

    Args:
        layer: The layer being exported.
        requested: The user's explicit ``--process-format``, or ``None``.

    Returns:
        The resolved format.

    Raises:
        click.UsageError: On an explicit ``zarr`` for a layer with no store
            form, naming the reason and the remedy.

    Examples:
        >>> from phenotypic._cli._cli_process_only import resolve_process_format
        >>> resolve_process_format("rgb", None)
        'zarr'
        >>> resolve_process_format("objmap", None)
        'tiff'
        >>> resolve_process_format("gray", "tiff")
        'tiff'
    """
    if requested is None:
        return "zarr" if layer in _ZARR_CAPABLE_LAYERS else "tiff"
    if requested == "zarr" and layer not in _ZARR_CAPABLE_LAYERS:
        raise click.UsageError(_NO_ZARR_FORM[layer])
    return requested


def process_only_output_path(
    output_dir: Path,
    image_path: Path,
    input_root: Path,
    layer: ProcessOnlyLayer,
    fmt: ProcessFormat = "tiff",
) -> Path:
    """Mirror ``image_path`` (relative to ``input_root``) under ``output_dir``.

    Args:
        output_dir: Run output root.
        image_path: The input image -- a flat file, or a ``*.ome.zarr`` store
            directory once a tree of process-mode output is used as input.
        input_root: The ``--input`` root the mirror is relative to.
        layer: The layer being exported.
        fmt: ``"zarr"`` names a ``<stem>.ome.zarr`` store directory;
            ``"tiff"`` names ``<stem>.png`` for ``objmap`` and
            ``<stem>.tiff`` otherwise.

    Returns:
        The output path. Bounded by the 1-level dataset scanner (D12).
    """
    from phenotypic.sdk_ import STORE_SUFFIX

    if fmt == "zarr":
        ext = STORE_SUFFIX
    else:
        ext = ".png" if layer == "objmap" else ".tiff"
    try:
        rel = image_path.relative_to(input_root)
    except ValueError:
        rel = Path(image_path.name)
    if rel == Path("."):
        # `--input` names the image itself, so `relative_to` yields `.` and
        # `Path(".").stem` is `""` -- the run would write `<out>/.ome.zarr`.
        # Pre-existing on the flat-file path (`--input <one tiff>` writes
        # `<out>/.tiff` today, verified); fixed here because a single store
        # input is exactly what spec §7 makes routine.
        rel = Path(image_path.name)
    # `store_stem`, never `Path.stem`, on a store input: `.ome.zarr` is a
    # double suffix, so `.stem` yields `p01.ome` and a zarr run would write
    # `p01.ome.ome.zarr`. `store_stem` RAISES on a non-store path
    # (_io_constants.py:1554), so the suffix test is required, not defensive.
    stem = source_image_stem(rel)
    return output_dir / rel.parent / f"{stem}{ext}"


def write_process_only_layer(
    image: Any,
    layer: ProcessOnlyLayer,
    out_path: Path,
    *,
    fmt: ProcessFormat = "tiff",
    commit_guard: CommitGuard | None = None,
    figures: StoredFigures | None = None,
    upstream: Mapping[str, Mapping] | None = None,
    embeddings: Mapping[str, EmbeddingPayload] | None = None,
) -> None:
    """Write one image layer, as a flat file or as a single-series store.

    The ``tiff`` branch delegates to the accessor's ``imsave`` through
    ``atomic_write_with_writer``, reusing the single, golden-tested writer
    (``_accessor_io_handler.imsave`` / the multichannel and objmap ``imsave``
    overrides): each layer is written at its **native dtype** with the
    PhenoTypic metadata embedded — ``rgb`` as an integer TIFF at the source bit
    depth, ``gray``/``detect_mat`` as float TIFFs (full precision preserved),
    and ``objmap`` as a 16-bit raw-label PNG (D10). No quantization is
    performed here.

    The ``zarr`` branch delegates to ``Image._save_store``, whose
    ``.part``-then-rename promote is atomic by construction — a store either
    has its root ``zarr.json`` or does not exist, so a kill mid-write cannot
    leave a truncated artifact at the final path.

    The store carries **only** *layer* (plus the *embeddings* run folder and
    the *figures*, when given): no objmap, no other series, and no
    ``image_class`` (which is what makes ``Image.load_zarr`` refuse it and
    point at ``Image.imread``).

    For ``objmap`` with no detected objects (e.g. a pipeline without a detector),
    emits the D9 warning and still writes the (all-zero) map; the run does not
    fail.

    *figures* are the per-image figures to write inside the store's
    transaction (spec 2026-09-22 §3); only the ``zarr`` branch writes them.

    *upstream* is the input's table runs
    (:func:`~phenotypic.sdk_.upstream_entries_for_input`), recorded under the
    store's ``tables.upstream`` (#281 spec D8); only the ``zarr`` branch
    records them, since a flat file has no descriptor. The entries are a pure
    function of the input's root bytes and its input-relative path, so a
    re-derived store stays byte-identical.

    *embeddings* are the pipeline's embedding tables for this image
    (:meth:`~phenotypic.ImagePipeline.embed`), written as the pipeline's run
    folder ``tables/<n>-<sha12>/embeddings/<model>/<region>/`` inside the
    store's ``.part``, before the root and before consolidation, so the store
    stays write-once (#281 spec E4). Only the ``zarr`` branch has somewhere
    to put them: a flat export given any raises ``ValueError`` rather than
    dropping them.

    Raises:
        ValueError: On a ``zarr`` export of a layer with no store form, or
            *embeddings* given to a flat export.
    """
    if fmt != "zarr" and embeddings:
        raise ValueError(
            f"a flat {fmt!r} export of layer {layer!r} has nowhere to put "
            f"embedding tables ({', '.join(sorted(embeddings))}); write a "
            "store (fmt='zarr', layer 'rgb' or 'gray')"
        )
    if fmt == "zarr":
        from phenotypic.sdk_ import ngff_

        # Two refusals, not one, and the difference is the point (spec §5.3):
        # `objmap` is refused by an NGFF structural rule, `detect_mat` by
        # PhenoTypic's own writer requirement. Left to `_save_store` both
        # would surface as `no primary series among [...]` -- true, but about
        # internal series naming rather than about what the caller asked for,
        # and identical for two unrelated causes. The guard lives here, not
        # only in the CLI, because `write_process_only_layer` is importable
        # and called directly by the staged strategy.
        if layer == "objmap":
            raise ValueError(
                "layer 'objmap' has no single-series OME-Zarr form: NGFF 0.5 "
                "§2.6 nests a labels group inside an image group and states "
                "that the labels group is not itself an image. Write it with "
                "fmt='tiff' for the 16-bit raw-label PNG, or export 'rgb'."
            )
        if layer not in _ZARR_CAPABLE_LAYERS:
            raise ValueError(
                f"layer {layer!r} has no single-series OME-Zarr form: "
                f"PhenoTypic's store writer requires a primary series (rgb or "
                f"gray) and {layer!r} is neither. Write it with fmt='tiff' "
                f"for the float TIFF, or export 'gray'."
            )
        # `image.shape[:2]` and not `image.gray[:].shape[:2]`: the same two
        # numbers, without routing through a layer an rgb store never writes.
        height, width = image.shape[:2]
        image._save_store(
            out_path,
            series=(layer,),
            write_objmap=False,
            levels=ngff_.pyramid_level_count(height, width),
            work_id=None,
            durable=None,
            commit_guard=commit_guard,
            write_image_class=False,
            # The published artifact drops the journal's two wall-clock
            # readings, and nothing else (spec 2.3.3). They are the entire
            # difference between two runs of one image through one pipeline,
            # so without this the store is a different object every time it is
            # regenerated -- which defeats content-addressed storage and
            # server-side dedup, and destabilises the whole-tree `file_sha256`
            # that continuation derives a work ID from. The image's own
            # journal and the bundle store keep them.
            reproducible_provenance=True,
            # Consolidated INSIDE the .part, before the promote -- see
            # `_consolidate_store_part`. A process-mode store is written once
            # and never mutated, so the consolidated view cannot drift from
            # the tree it describes; do not lift this onto a store that is
            # rewritten in place.
            consolidate=True,
            figures=figures,
            upstream=upstream,
            embeddings=embeddings or None,
        )
        return

    # Figures are never built for flat exports (spec 2026-09-22 non-goal).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accessor = getattr(image, layer)
    if layer == "objmap" and image.num_objects == 0:
        warnings.warn(
            f"pipeline produced no objects; writing empty object map to {out_path}"
        )
    atomic_write_with_writer(
        out_path,
        lambda temporary: accessor.imsave(filepath=Path(temporary)),
        commit_guard=commit_guard,
        temp_suffix=f".tmp{out_path.suffix}",
    )


def read_process_input(
    pipeline: ImagePipeline,
    image_path: Path,
    *,
    image_type: ImageTypeName,
    read_kwargs: Mapping[str, Any],
    cli_nrows: Optional[int] = None,
    cli_ncols: Optional[int] = None,
) -> Image:
    """Read one process-mode input as every process core reads it.

    The caller's ``input_read_settings`` kwargs, completed by
    :func:`~._cli_utils.image_read_kwargs` (a ``GridImage``'s shape from the
    flags, then the pipeline's preset), then ``imread``; a ``detect_mode``
    other than ``gray`` is applied after the read. Shared by the copy core
    and the in-place core (#281 spec E7), so the two read an input alike.

    Args:
        pipeline: The loaded pipeline (for a ``GridImage`` preset).
        image_path: A flat image file or a ``*.ome.zarr`` store.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: From ``input_read_settings``.
        cli_nrows: ``--nrows``, or ``None``.
        cli_ncols: ``--ncols``, or ``None``.

    Returns:
        The image.
    """
    from ._cli_utils import image_read_kwargs

    image_cls = GridImage if image_type == "GridImage" else Image
    kwargs = image_read_kwargs(
        read_kwargs,
        image_type=image_type,
        cli_nrows=cli_nrows,
        cli_ncols=cli_ncols,
        pipeline=pipeline,
    )
    detect_mode = kwargs.pop("detect_mode", "gray")
    image = image_cls.imread(image_path, **kwargs)
    if detect_mode != "gray":
        image.set_detect_mode(detect_mode)
    return image


def build_process_store_figures(
    pipeline: ImagePipeline,
    image: Image,
    *,
    run_initiation: RunInitiation | None,
    image_path: Path,
) -> StoredFigures | None:
    """The per-image figures a process store holds, or ``None``.

    The run folder is dated by *run_initiation*'s date only: a process store
    omits the call's timestamp and pid, like the journal's wall-clock times
    (spec §1a). There is no deliverables tree, so a run that cannot be named
    is only logged. Measurer-backed bindings recompute inside ``inspect()``,
    because ``apply()`` never filled their cache (spec §3, process mode).

    Args:
        pipeline: The loaded pipeline.
        image: The image after ``apply()``.
        run_initiation: The run's initial call, or ``None`` (today, UTC).
        image_path: The input, whose name keys the run's failure lines.
    """
    from phenotypic.plotting._pipeline._store_figures import (
        build_image_figures,
        name_figure_run,
    )

    run = name_figure_run(
        pipeline.get_plots(),
        image,
        date=run_initiation.date if run_initiation is not None else None,
        image_stem=image_path.name,
    )
    if run is None:
        return None
    return build_image_figures(pipeline, image, run=run)


def process_single_apply_only_core(
    pipeline_path: Path,
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    layer: ProcessOnlyLayer,
    read_kwargs: Dict[str, Any],
    cli_nrows: Optional[int] = None,
    cli_ncols: Optional[int] = None,
    commit_guard: CommitGuard | None = None,
    process_format: ProcessFormat = "tiff",
    run_initiation: RunInitiation | None = None,
    dataset_name: str | None = None,
    reference_pin: ReferencePin | None = None,
) -> bool:
    """Apply the pipeline to one image, embed it, and export ``layer``.

    Raises on failure (caller logs/handles), mirroring
    :func:`process_single_image_core`.

    **The ``meas`` entries a process store can hold run too** (#281 spec E4,
    user decision 2026-11-18: run what fits, skip the rest). An embedder whose
    ``requires_objects`` is false (a grid embedder) runs after ``apply()``,
    through :func:`~._cli_process_meas.embed_in_process_mode`, and its table
    goes into the store's own transaction as the pipeline's run folder. In v1
    nothing else runs: every other entry is skipped, as process mode always
    skipped ``meas``, and named in one logged warning, either because it
    needs objects (a process store holds no object map) or because it is not
    an embedder (a process store holds only embedding tables). ``measure()``
    is never called. A runnable entry with a flat output would lose its table, so it
    raises :class:`~._cli_process_meas.ProcessMeasRefusedError` (a
    ``ValueError``) before the image is read. The core checks this itself,
    with the check the CLI runs in its read-only half
    (:func:`~._cli_process_meas.process_meas_refusal`), because
    AutoConvertRaw-GC calls it directly and never passes the CLI.

    ``run_initiation`` is the run's initial CLI call (figures spec §1a); only
    its date is used, as the figure-folder ``{date}`` (``None``: today in
    UTC). Its timestamp and pid are deliberately left out of the store, so
    same-day process stores stay byte-identical. Process mode has no
    ``OutputManager`` to carry it, so it is passed here.

    ``dataset_name`` selects the image's entry in the run's reference
    manifest; it is required when the run has one (a pipeline that reads
    reference metadata) and unused otherwise. ``reference_pin`` is the
    reference digest the caller's work-id was computed from; the apply
    refuses a manifest that has since re-planned the image.
    """
    image: Image | None = None
    provenance_application_opened = False
    figures: StoredFigures | None = None
    upstream: dict[str, dict] = {}
    embeddings: dict[str, EmbeddingPayload] = {}

    def _mark_provenance_failed() -> None:
        if image is None or not provenance_application_opened:
            return
        try:
            set_provenance_status(image, "failed")
        except Exception:
            logger.exception(
                "Failed to mark process-only provenance application failed"
            )

    try:
        pipeline = ImagePipeline.from_json(pipeline_path)
        # Before the image is read: the pipeline, not the image, is at fault.
        refusal = process_meas_refusal(
            pipeline, layer=layer, process_format=process_format
        )
        if refusal is not None:
            raise ProcessMeasRefusedError(refusal)
        skipped = skipped_measurers_warning(pipeline)
        if skipped is not None:
            logger.warning(skipped)
        image = read_process_input(
            pipeline,
            image_path,
            image_type=image_type,
            read_kwargs=read_kwargs,
            cli_nrows=cli_nrows,
            cli_ncols=cli_ncols,
        )
        # The input's table runs, referenced rather than copied (spec D8). A
        # flat export has no descriptor to record them in.
        if process_format == "zarr":
            upstream = upstream_entries_for_input(image_path, input_root)

        # BEFORE apply, and that ordering is load-bearing:
        # `initialize_cli_provenance` opens with `new_provenance_journal()`
        # (_provenance.py:294), which discards `operations[]`. Called after
        # apply it would erase the very records it exists to contextualise,
        # and the store would still carry a `pipeline` key -- so it would
        # look right.
        #
        # What it adds is the `pipeline` identity. The operations themselves
        # are recorded either way: `wrap_image_operation_apply` appends one
        # per operation unconditionally (_provenance.py:227).
        #
        # `basename_only` keeps the publishing artifact free of cluster
        # filesystem layout: a process-mode store goes to a NAS and then to
        # object storage, where an absolute path would carry the username and
        # project directory names. sha256 still pins the pipeline exactly.
        initialize_cli_provenance(
            image,
            pipeline_path,
            kind="process",
            input_filename=image_path.name,
            basename_only=True,
        )
        provenance_application_opened = True
        with continuing_provenance_application(image), worker_reference_context(
            output_dir, dataset_name, pin=reference_pin
        ):
            pipeline.apply(image, inplace=True)
            # Beside apply, as the full writer embeds beside measure (spec
            # E2): the tables go into this store's run folder. Only the
            # embedders that need no objects; empty when there are none, and
            # always so for a flat export, which the refusal above admits
            # only without one.
            embeddings = embed_in_process_mode(pipeline, image)
        # Figures only when there is a store to hold them (spec §3 by mode),
        # and before the status is closed, as in full mode. Measurer-backed
        # bindings recompute inside inspect() here, because apply() never
        # filled their cache -- accepted (spec §3 process mode).
        if process_format == "zarr":
            figures = build_process_store_figures(
                pipeline, image, run_initiation=run_initiation, image_path=image_path
            )
        # BEFORE the write below, or the store records the stale default.
        # `initialize_cli_provenance` opens at `"in_progress"`
        # (_provenance.py:305), and every sibling path closes it --
        # `_cli_process_single.py:311`, `_cli_staged_workers.py:493`. This one
        # did not, so every store this mode published said it was still being
        # written. `status` is the field that says whether an artifact is
        # trustworthy: `_cli_staged_resume.py:101` already gates on
        # `status in {"staged", "complete"}`, so a consumer following that
        # convention would reject every store we publish (spec 2.3.4).
        set_provenance_status(image, "complete")
    except (MemoryError, ProcessMeasRefusedError):
        raise
    except ReferencePlanStaleError:
        # Not this image's fault: never a terminal scientific failure.
        _mark_provenance_failed()
        raise
    except Exception as exc:
        _mark_provenance_failed()
        raise PerImageScientificError("process", exc) from exc

    out_path = process_only_output_path(
        output_dir, image_path, input_root, layer, fmt=process_format
    )
    write_process_only_layer(
        image,
        layer,
        out_path,
        fmt=process_format,
        commit_guard=commit_guard,
        figures=figures,
        upstream=upstream,
        embeddings=embeddings,
    )
    return True
