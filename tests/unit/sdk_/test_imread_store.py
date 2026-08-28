"""imread reads a store as plain pixels -- as if it were a TIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import ngff_


def _processed_rgb_store(tmp_path: Path) -> tuple[Path, Image]:
    img = Image(load_synth_yeast_plate())
    store = img._save_store(
        tmp_path / "IMG_4471.ome.zarr",
        series=("rgb",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.rgb[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )
    return store, img


def test_imread_round_trips_rgb_bit_exactly(tmp_path: Path) -> None:
    store, img = _processed_rgb_store(tmp_path)
    loaded = Image.imread(store)
    assert np.array_equal(loaded.rgb[:], img.rgb[:])


def test_imread_uses_the_store_stem_not_path_stem(tmp_path: Path) -> None:
    """Path('IMG_4471.ome.zarr').stem is 'IMG_4471.ome' -- a wrong name."""
    store, _ = _processed_rgb_store(tmp_path)
    assert Image.imread(store).name == "IMG_4471"


def test_imread_records_the_store_suffix(tmp_path: Path) -> None:
    store, _ = _processed_rgb_store(tmp_path)
    assert Image.imread(store).metadata[IMAGE.SUFFIX] == ngff_.STORE_SUFFIX


def test_imread_carries_provenance_across(tmp_path: Path) -> None:
    """The operations that produced the pixels survive the round trip."""
    store, _ = _processed_rgb_store(tmp_path)
    loaded = Image.imread(store)
    assert loaded._metadata.provenance_journal is not None


def test_imported_tags_land_in_the_imported_section(tmp_path: Path) -> None:
    """Not in `public`, which `image.metadata[key] = value` would give.

    MetadataAccessor.__setitem__ routes any key it does not already know into
    `_public_metadata` and raises ValueError on a non-scalar value -- so the
    obvious assignment loop would both put the tags in the wrong section and
    blow up on a structured TIFF tag. The store branch writes through
    `_metadata.imported.update(...)`, matching the TIFF branch.
    """
    img = Image(load_synth_yeast_plate())
    img._metadata.imported.update({"Metadata_Make": "Canon"})
    store = img._save_store(
        tmp_path / "tagged.ome.zarr",
        series=("rgb",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.rgb[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )
    loaded = Image.imread(store)
    assert loaded._metadata.imported["Metadata_Make"] == "Canon"
    assert "Metadata_Make" not in loaded._metadata.public


def test_imread_does_not_carry_run_state_across(tmp_path: Path) -> None:
    """`protected` and `public` are run state. That is the line (spec 4.5).

    Carrying them would make imread a partial load_zarr, which is precisely
    the distinction the two verbs exist to keep.
    """
    img = Image(load_synth_yeast_plate())
    img.metadata["operator_note"] = "run 3, plate B"  # -> public
    store = img._save_store(
        tmp_path / "stateful.ome.zarr",
        series=("rgb",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.rgb[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )
    assert "operator_note" not in Image.imread(store)._metadata.public


def test_imread_yields_no_objects_from_a_processed_store(tmp_path: Path) -> None:
    """Pixels only. A process store has no objmap and imread invents none."""
    store, _ = _processed_rgb_store(tmp_path)
    assert Image.imread(store).num_objects == 0


def test_imread_reads_a_bundle_store_as_pixels(tmp_path: Path) -> None:
    """Documented behaviour: the verb decides, not the file (spec 3.2)."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    assert np.array_equal(Image.imread(store).rgb[:], img.rgb[:])


def test_a_non_store_directory_is_still_unsupported(tmp_path: Path) -> None:
    from phenotypic.sdk_.exceptions_ import UnsupportedFileTypeError

    plain = tmp_path / "just_a_folder"
    plain.mkdir()
    with pytest.raises((UnsupportedFileTypeError, ValueError, IsADirectoryError)):
        Image.imread(plain)
