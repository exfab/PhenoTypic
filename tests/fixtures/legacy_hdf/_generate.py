"""Generate the six golden legacy-HDF fixtures ``--mode migrate`` reads.

Run as a script to rebuild every ``.h5`` under this directory::

    uv run python tests/fixtures/legacy_hdf/_generate.py

**Hand-rolled on ``HDF.save_array2hdf5``, deliberately.** ``Image.save2hdf5``
and ``Image._save_image2hdfgroup`` are deleted in Phase 6, so a generator built
on them could not rebuild these fixtures afterwards. The cost of hand-rolling
is that every fixture is an *approximation* of what production wrote --
:func:`tests.unit.sdk_.test_hdf_to_zarr.test_the_generator_matches_the_real_writer`
closes that gap while the real writer still exists, and self-documents once it
is gone.

The six fixtures, and what each pins:

======================  ====================================================
``v1_flat``             the legacy flat layout, with legacy per-topic
                        metadata headers
``v2_grouped``          the current grouped (``schema_version=2``) layout
``v2_enh_gray``         the pre-rename ``enh_gray`` layer
``v2_grid``             a ``GridImage`` -- ``nrows``/``ncols``/``grid_finder``
``v2_image_type``       a non-default ``Metadata_ImageType`` (ledger MIG-2)
``v2_work_id``          a root ``phenotypic_work_id`` attr (ledger FLOW-1)
======================  ====================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import phenotypic
from phenotypic import GridImage, Image
from phenotypic.sdk_.hdf_ import HDF

FIXTURE_ROOT = Path(__file__).resolve().parent

#: Layouts this module writes, in the order the tests parametrize them.
LAYOUTS: tuple[str, ...] = (
    "v1_flat",
    "v2_grouped",
    "v2_enh_gray",
    "v2_grid",
    "v2_image_type",
    "v2_work_id",
)

#: Written into ``v2_image_type``'s protected metadata. The test asserts
#: against THIS constant, not against a second load of the same file -- both
#: sides of a load-vs-load comparison go through the same lossy reader, so it
#: certifies the bug instead of catching it (ledger MIG-2).
V2_IMAGE_TYPE_AUTHORED: str = "GridSection"

#: Written as ``v2_work_id``'s root ``phenotypic_work_id`` attribute.
V2_WORK_ID_AUTHORED: str = "w-fixture"

#: The grid geometry ``v2_grid`` carries. Non-default on both axes, so a
#: loader that silently falls back to the ``GridImage`` defaults (8x12) fails.
V2_GRID_SHAPE: tuple[int, int] = (16, 24)

#: ``v2_grid``'s grid finder class. Deliberately NOT the ``GridImage``
#: default (``CenteredAutoGridFinder``): the constructor mints a default
#: finder whenever none is supplied, so ``grid_finder is not None`` holds
#: even when the stored one was dropped entirely. Only a differing class --
#: or differing params -- can tell the two apart.
V2_GRID_FINDER_CLASS: str = "AutoGridFinder"

#: A non-default parameter on that finder, so dropping the serialized params
#: while keeping the class is also visible.
V2_GRID_FINDER_RESIDUAL_FRACTION: float = 0.4

#: A legacy per-topic metadata header and the flat header it resolves to.
#: ``MetadataGenetic_Strain`` is a real legacy spelling -- verified against
#: ``ensure_metadata_prefix``, which maps it to ``Metadata_Strain``. Spellings
#: the registry does NOT know (``MetadataPlate_Strain``, say) are preserved
#: verbatim by ``_remap_legacy_metadata_key`` and would never canonicalize.
V1_LEGACY_PUBLIC_HEADER: str = "MetadataGenetic_Strain"
V1_CANONICAL_PUBLIC_HEADER: str = "Metadata_Strain"
V1_PUBLIC_VALUE: str = "BY4741"

_SCHEMA_VERSION = 2
_METADATA_SCHEMA_VERSION = 2
_COMPRESSION: dict[str, Any] = {"compression": "gzip", "compression_opts": 4}


# ---------------------------------------------------------------------------
# The subject image
# ---------------------------------------------------------------------------


def build_fixture_image() -> Image:
    """Return the deterministic ``Image`` every fixture is written from.

    Small (64x64) and seeded, so the committed ``.h5`` files stay tiny and
    rebuilding them is byte-reproducible. Every layer is *populated* --
    a conversion that wrote a correctly-shaped zero ``detect_mat`` must fail
    the content comparison, which it cannot do against an all-zero fixture.

    Returns:
        An ``Image`` with rgb, gray, detect_mat, objmap and metadata set.
    """
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    objmap = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.ogrid[:64, :64]
    for label, (cy, cx) in enumerate(((18, 18), (18, 46), (46, 18), (46, 46)), 1):
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= 8**2
        rgb[disc] = (200 + 10 * label, 180, 160)
        objmap[disc] = label

    image = Image(rgb)
    image.detect_mat[:] = image.gray[:].astype(np.float32)
    image.objmap[:] = objmap
    image._metadata.protected[str(_image_name_header())] = "img"
    image._metadata.public["Metadata_Strain"] = V1_PUBLIC_VALUE
    image._metadata.imported["Metadata_PlateNum"] = 3
    return image


def build_fixture_grid_image() -> GridImage:
    """Return the ``GridImage`` ``v2_grid`` is written from."""
    from phenotypic.grid import AutoGridFinder

    base = build_fixture_image()
    nrows, ncols = V2_GRID_SHAPE
    grid = GridImage(
        base.rgb[:],
        grid_finder=AutoGridFinder(
            nrows=nrows,
            ncols=ncols,
            residual_fraction=V2_GRID_FINDER_RESIDUAL_FRACTION,
        ),
    )
    grid.detect_mat[:] = base.detect_mat[:]
    grid.objmap[:] = base.objmap[:]
    grid._metadata.protected.update(base._metadata.protected)
    grid._metadata.public.update(base._metadata.public)
    grid._metadata.imported.update(base._metadata.imported)
    return grid


def _image_name_header() -> str:
    from phenotypic.schema import IMAGE

    return str(IMAGE.IMAGE_NAME)


def _image_type_header() -> str:
    from phenotypic.schema import IMAGE

    return str(IMAGE.IMAGE_TYPE)


def _encode(value: Any) -> str:
    """JSON-encode one metadata value exactly as the production writer does."""
    return json.dumps(value, default=str)


# ---------------------------------------------------------------------------
# schema_version = 2, grouped
# ---------------------------------------------------------------------------


def _write_root_attrs(handle: h5py.File, image: Image) -> None:
    handle.attrs["version"] = phenotypic.__version__
    handle.attrs["schema_version"] = _SCHEMA_VERSION
    handle.attrs["metadata_schema_version"] = _METADATA_SCHEMA_VERSION
    handle.attrs["phenotypic_class"] = type(image).__name__
    if image.bit_depth is not None:
        handle.attrs["bit_depth"] = int(image.bit_depth)
    if image.illuminant is not None:
        handle.attrs["illuminant"] = str(image.illuminant)
    if image.gamma is not None:
        handle.attrs["gamma"] = (
            image.gamma.name
            if hasattr(image.gamma, "name")
            else str(image.gamma)
        )


def _write_layers(
    handle: h5py.File, image: Image, *, detect_mat_name: str = "detect_mat"
) -> None:
    layers = handle.require_group("layers")
    if not image.rgb.isempty():
        rgb = image.rgb[:]
        HDF.save_array2hdf5(
            group=layers, array=rgb, name="rgb", dtype=rgb.dtype, **_COMPRESSION
        )
    gray = image.gray[:]
    HDF.save_array2hdf5(
        group=layers, array=gray, name="gray", dtype=gray.dtype, **_COMPRESSION
    )
    detect_mat = image.detect_mat[:]
    HDF.save_array2hdf5(
        group=layers,
        array=detect_mat,
        name=detect_mat_name,
        dtype=detect_mat.dtype,
        **_COMPRESSION,
    )
    if detect_mat_name == "detect_mat":
        # The pre-rename `enh_gray` layer carried no detect_mode attr; that is
        # exactly why the fallback hard-codes "gray".
        layers[detect_mat_name].attrs["detect_mode"] = image._data.detect_mode
    objmap = image.objmap[:]
    HDF.save_array2hdf5(
        group=layers,
        array=objmap,
        name="objmap",
        dtype=objmap.dtype,
        **_COMPRESSION,
    )


def _write_metadata(handle: h5py.File, sections: dict[str, dict]) -> None:
    meta = handle.require_group("metadata")
    for name, section in sections.items():
        sub = meta.require_group(name)
        for key, value in section.items():
            sub.attrs[str(key)] = _encode(value)


def _sections_of(image: Image) -> dict[str, dict]:
    return {
        "protected": {str(k): v for k, v in image._metadata.protected.items()},
        "public": {str(k): v for k, v in image._metadata.public.items()},
        "imported": {str(k): v for k, v in image._metadata.imported.items()},
    }


def write_v2_grouped(path: Path, image: Image) -> Path:
    """Write *image* in the current grouped ``schema_version=2`` layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode="w") as handle:
        _write_root_attrs(handle, image)
        _write_layers(handle, image)
        _write_metadata(handle, _sections_of(image))
    return path


def write_v2_grid(path: Path, image: GridImage) -> Path:
    """Write a ``GridImage`` in the grouped layout, including ``/grid/``."""
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode="w") as handle:
        _write_root_attrs(handle, image)
        _write_layers(handle, image)
        _write_metadata(handle, _sections_of(image))
        grid = handle.require_group("grid")
        grid.attrs["nrows"] = int(image.nrows)
        grid.attrs["ncols"] = int(image.ncols)
        if image.grid_finder is not None:
            payload = {
                "class": type(image.grid_finder).__name__,
                "params": SerializablePipeline._serialize_single_operation(
                    image.grid_finder
                ),
            }
            grid.create_dataset(
                "grid_finder_json",
                data=json.dumps(payload),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
    return path


def write_v2_enh_gray(path: Path, image: Image) -> Path:
    """Write the grouped layout with the PRE-RENAME ``enh_gray`` layer name.

    ``valid_staged_hdf`` accepted ``enh_gray`` at ``schema_version >= 2``, so
    the code believes such files exist in the wild; ``_load_v2_grouped`` did a
    bare ``layers["detect_mat"]`` and would ``KeyError`` on one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode="w") as handle:
        _write_root_attrs(handle, image)
        _write_layers(handle, image, detect_mat_name="enh_gray")
        _write_metadata(handle, _sections_of(image))
    return path


def write_v2_image_type(path: Path, image: Image) -> Path:
    """Write the grouped layout carrying a NON-DEFAULT ``Metadata_ImageType``.

    The ``Image`` constructor sets that key to ``"Image"``, and
    ``_load_v2_grouped``'s restore loop skips any key the constructor already
    populated -- so the stored ``"GridSection"`` was silently replaced by the
    default and a key-set comparison saw no difference at all (ledger MIG-2).
    """
    sections = _sections_of(image)
    sections["protected"][_image_type_header()] = V2_IMAGE_TYPE_AUTHORED
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode="w") as handle:
        _write_root_attrs(handle, image)
        _write_layers(handle, image)
        _write_metadata(handle, sections)
    return path


def write_v2_work_id(path: Path, image: Image) -> Path:
    """Write the grouped layout plus a root ``phenotypic_work_id`` attribute.

    The CLI patched that attribute in after the write; it lives in no image
    field, so ``load_image_from_hdf`` does not carry it and ``save2zarr``
    would never see it. Without it every migrated image reclassifies
    ``"stage1"`` (ledger FLOW-1).
    """
    path = write_v2_grouped(path, image)
    with h5py.File(path, mode="a") as handle:
        handle.attrs["phenotypic_work_id"] = V2_WORK_ID_AUTHORED
    return path


# ---------------------------------------------------------------------------
# schema_version = 1, flat
# ---------------------------------------------------------------------------


def write_v1_flat(path: Path, image: Image) -> Path:
    """Write the legacy flat layout, with legacy per-topic metadata headers.

    Layers sit at the group root and metadata lives in ``protected_metadata``
    / ``public_metadata`` attribute subgroups holding **bare strings** -- the
    legacy loader's ``int(v) if v.isdigit() else v`` coercion is what reads
    them back. ``imported`` is absent because legacy files never stored it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode="w") as handle:
        handle.attrs["version"] = phenotypic.__version__
        handle.attrs["schema_version"] = 1
        handle.attrs["phenotypic_class"] = type(image).__name__
        handle.attrs["bit_depth"] = int(image.bit_depth)

        rgb = image.rgb[:]
        HDF.save_array2hdf5(
            group=handle, array=rgb, name="rgb", dtype=rgb.dtype, **_COMPRESSION
        )
        gray = image.gray[:]
        HDF.save_array2hdf5(
            group=handle,
            array=gray,
            name="gray",
            dtype=gray.dtype,
            **_COMPRESSION,
        )
        detect_mat = image.detect_mat[:]
        HDF.save_array2hdf5(
            group=handle,
            array=detect_mat,
            name="detect_mat",
            dtype=detect_mat.dtype,
            **_COMPRESSION,
        )
        handle["detect_mat"].attrs["detect_mode"] = image._data.detect_mode
        objmap = image.objmap[:]
        HDF.save_array2hdf5(
            group=handle,
            array=objmap,
            name="objmap",
            dtype=objmap.dtype,
            **_COMPRESSION,
        )

        protected = handle.require_group("protected_metadata")
        protected.attrs["MetadataImage_ImageName"] = "img"
        protected.attrs["MetadataImage_BitDepth"] = str(int(image.bit_depth))
        public = handle.require_group("public_metadata")
        public.attrs[V1_LEGACY_PUBLIC_HEADER] = V1_PUBLIC_VALUE
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def regenerate(root: Path = FIXTURE_ROOT) -> dict[str, Path]:
    """Rebuild every fixture under *root*.

    Args:
        root: Directory holding one subdirectory per layout.

    Returns:
        ``{layout: path-to-img.h5}``.
    """
    image = build_fixture_image()
    written: dict[str, Path] = {}
    written["v1_flat"] = write_v1_flat(root / "v1_flat" / "img.h5", image)
    written["v2_grouped"] = write_v2_grouped(
        root / "v2_grouped" / "img.h5", image
    )
    written["v2_enh_gray"] = write_v2_enh_gray(
        root / "v2_enh_gray" / "img.h5", image
    )
    written["v2_grid"] = write_v2_grid(
        root / "v2_grid" / "img.h5", build_fixture_grid_image()
    )
    written["v2_image_type"] = write_v2_image_type(
        root / "v2_image_type" / "img.h5", image
    )
    written["v2_work_id"] = write_v2_work_id(
        root / "v2_work_id" / "img.h5", image
    )
    return written


if __name__ == "__main__":  # pragma: no cover - developer entry point
    for layout, target in regenerate().items():
        print(f"{layout}: {target}")
