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
from phenotypic.sdk_.typing_ import ImageTypeName, ProcessOnlyLayer

logger = logging.getLogger(__name__)


def process_only_output_path(
    output_dir: Path, image_path: Path, input_root: Path, layer: ProcessOnlyLayer
) -> Path:
    """Mirror ``image_path`` (relative to ``input_root``) under ``output_dir``.

    Names the file ``<stem>_<layer>.<ext>`` (``.png`` for objmap, else
    ``.tiff``). Bounded by the 1-level dataset scanner (D12).
    """
    ext = ".png" if layer == "objmap" else ".tiff"
    try:
        rel = image_path.relative_to(input_root)
    except ValueError:
        rel = Path(image_path.name)
    return output_dir / rel.parent / f"{rel.stem}_{layer}{ext}"


def write_process_only_layer(
    image: Any, layer: ProcessOnlyLayer, out_path: Path
) -> None:
    """Write one image layer by delegating to the accessor's ``imsave``.

    Reuses the single, golden-tested writer (``_accessor_io_handler.imsave`` /
    the multichannel and objmap ``imsave`` overrides): each layer is written at
    its **native dtype** with the PhenoTypic metadata embedded — ``rgb`` as an
    integer TIFF at the source bit depth, ``gray``/``detect_mat`` as float TIFFs
    (full precision preserved), and ``objmap`` as a 16-bit raw-label PNG (D10).
    No quantization is performed here.

    For ``objmap`` with no detected objects (e.g. a pipeline without a detector),
    emits the D9 warning and still writes the (all-zero) map; the run does not
    fail.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accessor = getattr(image, layer)
    if layer == "objmap" and image.num_objects == 0:
        warnings.warn(
            f"pipeline produced no objects; writing empty object map to {out_path}"
        )
    accessor.imsave(filepath=out_path)  # native dtype + embedded metadata


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
) -> bool:
    """Apply the pipeline to one image and export ``layer``. No measurement.

    Raises on failure (caller logs/handles), mirroring
    :func:`process_single_image_core`.
    """
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

    pipeline.apply(image, inplace=True)

    out_path = process_only_output_path(output_dir, image_path, input_root, layer)
    write_process_only_layer(image, layer, out_path)
    return True
