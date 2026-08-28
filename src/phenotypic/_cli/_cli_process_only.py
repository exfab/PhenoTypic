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

import click

from phenotypic import GridImage, Image, ImagePipeline
from phenotypic.sdk_ import CommitGuard, atomic_write_with_writer
from phenotypic._core._provenance import initialize_cli_provenance
from phenotypic.sdk_.typing_ import ImageTypeName, ProcessFormat, ProcessOnlyLayer
from ._cli_failure_tracker import PerImageScientificError

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
            # Consolidated INSIDE the .part, before the promote -- see
            # `_consolidate_store_part`. A process-mode store is written once
            # and never mutated, so the consolidated view cannot drift from
            # the tree it describes; do not lift this onto a store that is
            # rewritten in place.
            consolidate=True,
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
