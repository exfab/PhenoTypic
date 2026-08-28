"""Process-only CLI mode: run pipeline.apply() and export a single layer.

Used when the user wants PhenoTypic preprocessing/detection output without the
full measurement/analysis suite. See
docs/superpowers/specs/2026-06-03-cli-process-only-and-phenotypic-cache-design.md.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from phenotypic import GridImage, Image, ImagePipeline
from phenotypic.sdk_ import CommitGuard, atomic_write_with_writer
from phenotypic._core._provenance import initialize_cli_provenance
from phenotypic.sdk_.typing_ import ImageTypeName, ProcessFormat, ProcessOnlyLayer
from ._cli_failure_tracker import PerImageScientificError

logger = logging.getLogger(__name__)

#: Layers that have a single-series OME-Zarr form. ``objmap`` and ``detect_mat``
#: are absent for two different reasons, both spelled out in
#: :func:`resolve_process_format`'s refusals.
_ZARR_CAPABLE_LAYERS: frozenset[str] = frozenset({"rgb", "gray"})


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
    from phenotypic.sdk_ import STORE_SUFFIX, store_stem

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
    stem = store_stem(rel) if rel.name.endswith(STORE_SUFFIX) else rel.stem
    return output_dir / rel.parent / f"{stem}{ext}"


def write_process_only_layer(
    image: Any,
    layer: ProcessOnlyLayer,
    out_path: Path,
    *,
    fmt: ProcessFormat = "tiff",
    commit_guard: CommitGuard | None = None,
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

    The store carries **only** *layer*: no objmap, no other series, and no
    ``image_class`` (which is what makes ``Image.load_zarr`` refuse it and
    point at ``Image.imread``).

    For ``objmap`` with no detected objects (e.g. a pipeline without a detector),
    emits the D9 warning and still writes the (all-zero) map; the run does not
    fail.
    """
    if fmt == "zarr":
        from phenotypic.sdk_ import ngff_

        if layer not in _ZARR_CAPABLE_LAYERS:
            # The CLI refuses this earlier and with a better message
            # (`resolve_process_format`). This guard exists because
            # `write_process_only_layer` is importable and called directly by
            # the staged strategy, and because `_save_store` would otherwise
            # fail for `detect_mat` with `no primary series among
            # ['detect_mat']` -- true, but about internal series naming rather
            # than about what the caller asked for.
            raise ValueError(
                f"layer {layer!r} has no single-series OME-Zarr form; write "
                f"it with fmt='tiff'"
            )
        height, width = image.gray[:].shape[:2]
        image._save_store(
            out_path,
            series=(layer,),
            write_objmap=False,
            levels=ngff_.pyramid_level_count(height, width),
            work_id=None,
            durable=None,
            commit_guard=commit_guard,
            write_image_class=False,
        )
        return

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
) -> bool:
    """Apply the pipeline to one image and export ``layer``. No measurement.

    Raises on failure (caller logs/handles), mirroring
    :func:`process_single_image_core`.
    """
    try:
        pipeline = ImagePipeline.from_json(pipeline_path)
        image_cls = GridImage if image_type == "GridImage" else Image

        read_kwargs = dict(read_kwargs)
        if image_type == "GridImage":
            from ._cli_utils import resolve_grid_shape

            nrows, ncols = resolve_grid_shape(
                cli_nrows=cli_nrows,
                cli_ncols=cli_ncols,
                pipeline_nrows=pipeline.nrows,
                pipeline_ncols=pipeline.ncols,
            )
            read_kwargs["nrows"] = nrows
            read_kwargs["ncols"] = ncols

        detect_mode = read_kwargs.pop("detect_mode", "gray")
        image = image_cls.imread(image_path, **read_kwargs)
        if detect_mode != "gray":
            image.set_detect_mode(detect_mode)

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
        initialize_cli_provenance(image, pipeline_path, basename_only=True)
        pipeline.apply(image, inplace=True)
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError("process", exc) from exc

    out_path = process_only_output_path(
        output_dir, image_path, input_root, layer, fmt=process_format
    )
    write_process_only_layer(
        image, layer, out_path, fmt=process_format, commit_guard=commit_guard
    )
    return True
