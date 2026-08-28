"""imread reads a store as plain pixels -- as if it were a TIFF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import ngff_
from tests._process_stores import write_process_store


def _processed_rgb_store(tmp_path: Path) -> tuple[Path, Image]:
    img = Image(load_synth_yeast_plate())
    store = write_process_store(
        tmp_path / "IMG_4471.ome.zarr", img, series="rgb"
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


def _pipeline_file(tmp_path: Path) -> Path:
    nested = tmp_path / "config" / "deep"
    nested.mkdir(parents=True, exist_ok=True)
    path = nested / "preprocess_pipeline.json.pht-pipe"
    path.write_text('{"name": "acr_preprocess"}', encoding="utf-8")
    return path


def _provenanced_store(
    tmp_path: Path, *, basename_only: bool = False
) -> tuple[Path, Image]:
    """A store whose journal actually carries a pipeline identity and an op.

    The empty journal every `Image` is born with (`_image_data_manager` fills
    it from `new_provenance_journal()` unconditionally) makes a bare
    `is not None` assertion unfalsifiable -- deleting the copy in
    `_load_from_store`'s imread sibling leaves it green. So populate the
    source journal first and compare recovered VALUES.
    """
    from phenotypic._core._provenance import (
        append_operation_provenance,
        initialize_cli_provenance,
    )
    from phenotypic.enhance import BlurGauss

    img = Image(load_synth_yeast_plate())
    initialize_cli_provenance(
        img, _pipeline_file(tmp_path), basename_only=basename_only
    )
    append_operation_provenance(
        img,
        BlurGauss(),
        duration_seconds=0.125,
        pipeline_step_path=["preprocess", "0"],
    )
    store = write_process_store(
        tmp_path / "provenanced.ome.zarr", img, series="rgb"
    )
    return store, img


def test_imread_carries_provenance_across(tmp_path: Path) -> None:
    """The operations that produced the pixels survive the round trip.

    Asserted on VALUES, not on presence: every Image has a journal, so
    `is not None` passes with the copy deleted.
    """
    store, img = _provenanced_store(tmp_path)
    source = img._metadata.provenance_journal
    assert source["pipeline"]["sha256"]  # the fixture really populated it
    assert source["operations"], "fixture must record at least one operation"

    recovered = Image.imread(store)._metadata.provenance_journal
    assert recovered["pipeline"]["sha256"] == source["pipeline"]["sha256"]
    assert (
        recovered["operations"][0]["operation_name"]
        == source["operations"][0]["operation_name"]
    )
    assert recovered["operations"][0]["operation_name"] == "BlurGauss"


def test_imread_carries_a_basename_only_pipeline_path(tmp_path: Path) -> None:
    """`basename_only=True` is what keeps a cluster path out of a published
    store; this is its only end-to-end coverage through a store."""
    store, img = _provenanced_store(tmp_path, basename_only=True)
    recovered = Image.imread(store)._metadata.provenance_journal
    source_path = recovered["pipeline"]["source_path"]
    assert source_path == "preprocess_pipeline.json.pht-pipe"
    assert "/" not in source_path


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
    store = write_process_store(tmp_path / "tagged.ome.zarr", img, series="rgb")
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
    img._metadata.protected["Metadata_ImageType"] = "GridSection"  # -> protected
    store = write_process_store(
        tmp_path / "stateful.ome.zarr", img, series="rgb"
    )
    loaded = Image.imread(store)
    assert "operator_note" not in loaded._metadata.public
    # `protected` is the other half of the line, and the store DOES carry it
    # (`phenotypic.metadata.protected`) -- so this is a real read, not an
    # absence that was never written.
    stored = ngff_.read_phenotypic_attributes(store)
    assert stored["metadata"]["protected"]["Metadata_ImageType"] == "GridSection"
    assert loaded._metadata.protected["Metadata_ImageType"] != "GridSection"


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


def test_imread_reads_a_store_with_no_phenotypic_block(tmp_path: Path) -> None:
    """The "never calls require_readable_store" constraint, proven at the
    PUBLIC verb. It was pinned only at the resolver, so a guard reintroduced
    anywhere between `Image.imread` and `read_ngff_image_spec` would have gone
    unnoticed -- and a napari / QuPath / bioformats2raw store has no
    `phenotypic` block at all (spec 4, case C).
    """
    import zarr

    store = tmp_path / "foreign.ome.zarr"
    group = zarr.create_group(store=str(store), zarr_format=3)
    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    pixels = np.arange(16 * 12, dtype=np.uint16).reshape(16, 12)
    arr = group.create_array(
        "0", shape=pixels.shape, chunks=pixels.shape, dtype="uint16",
        dimension_names=["y", "x"],
    )
    arr[:] = pixels
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "name": "foreign",
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]}
                        ],
                    }
                ],
            }
        ],
    }

    assert "phenotypic" not in ngff_.read_root_attributes(store)
    loaded = Image.imread(store)
    assert np.array_equal(loaded.gray[:], pixels)
    assert loaded.name == "foreign"


def test_imread_reads_a_consolidated_store(tmp_path: Path) -> None:
    """The actual on-disk shape of a `--mode process` store: consolidated
    metadata and no `image_class`. Never read back until now."""
    img = Image(load_synth_yeast_plate())
    store = write_process_store(
        tmp_path / "consolidated.ome.zarr", img, series="rgb", consolidate=True
    )
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert "consolidated_metadata" in root, "fixture must really be consolidated"

    loaded = Image.imread(store)
    assert np.array_equal(loaded.rgb[:], img.rgb[:])


def test_imread_round_trips_a_gray_float_store_bit_exactly(tmp_path: Path) -> None:
    """`gray` is the default layer for `--mode process --layer gray`, and it is
    float32 -- so `metadata.protected[Metadata_BitDepth]` is the ONLY bit-depth
    source; dtype inference has no answer for a float array."""
    img = Image(load_synth_yeast_plate())
    store = write_process_store(
        tmp_path / "grayscale.ome.zarr", img, levels=1, consolidate=True
    )
    stored = ngff_.read_phenotypic_attributes(store)
    assert stored["metadata"]["protected"][IMAGE.BIT_DEPTH] == img.bit_depth

    loaded = Image.imread(store)
    assert loaded.bit_depth == img.bit_depth
    assert np.array_equal(loaded.gray[:], img.gray[:])
