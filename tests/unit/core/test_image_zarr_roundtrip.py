"""Image -> store -> Image must be bit-exact in layers and equal in metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes


_NGFF_ODD_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "phenotypic"
    / "ngff_multiscales_odd.json"
)


@pytest.fixture
def plate() -> Image:
    return Image(load_synth_yeast_plate())


@pytest.fixture
def blank(plate: Image) -> Image:
    """A plate with no detections.

    The synthetic fixture already carries 96 labelled colonies, so the
    "nothing detected" case needs a fresh image built from its pixels.
    """
    return Image(np.asarray(plate.rgb[:]))


@pytest.fixture
def odd_singleton_rgb() -> Image:
    """A three-level pyramid with one odd and one singleton spatial extent."""
    pixels = np.empty((1025, 1, 3), dtype=np.uint8)
    pixels[..., 0] = 17
    pixels[..., 1] = 91
    pixels[..., 2] = 203
    image = Image(pixels)
    image._metadata.protected["Metadata_ImageName"] = "odd_plate"
    return image


def _read_array_json(store: Path, member: str, level: int) -> dict:
    return json.loads(
        (store / member / str(level) / "zarr.json").read_text(encoding="utf-8")
    )


def test_layers_round_trip_bit_exact(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    np.testing.assert_array_equal(back.rgb[:], plate.rgb[:])
    np.testing.assert_array_equal(back.gray[:], plate.gray[:])
    np.testing.assert_array_equal(back.detect_mat[:], plate.detect_mat[:])
    np.testing.assert_array_equal(back.objmap[:], plate.objmap[:])


def test_layer_dtypes_survive_the_round_trip(plate: Image, tmp_path: Path) -> None:
    """A silent upcast keeps every value equal and still breaks the store."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert back.rgb[:].dtype == plate.rgb[:].dtype
    assert back.gray[:].dtype == plate.gray[:].dtype
    assert back.detect_mat[:].dtype == plate.detect_mat[:].dtype
    assert back.objmap[:].dtype == plate.objmap[:].dtype


def test_objmap_labels_survive_by_value_not_just_by_shape(
    plate: Image, tmp_path: Path
) -> None:
    assert plate.num_objects == 96
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert back.num_objects == plate.num_objects
    assert set(np.unique(back.objmap[:])) == set(np.unique(plate.objmap[:]))


def test_rgb_is_stored_channel_first_on_disk(plate: Image, tmp_path: Path) -> None:
    """NGFF axes are (c, y, x); a transposed write reads back transposed."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    height, width, channels = plate.rgb[:].shape
    payload = _read_array_json(store, "rgb", 0)
    assert tuple(payload["shape"]) == (channels, height, width)
    assert payload["dimension_names"] == ["c", "y", "x"]


def test_two_dimensional_series_declare_yx_dimension_names(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    members = ["gray", "detect_mat", block[PhenotypicAttr.LABELS]["objmap"]]
    for member in members:
        payload = _read_array_json(store, member, 0)
        assert payload["dimension_names"] == ["y", "x"], member
        assert tuple(payload["shape"]) == plate.gray[:].shape, member


def test_metadata_sections_round_trip(plate: Image, tmp_path: Path) -> None:
    plate._metadata.public["Metadata_Strain"] = "BY4741"
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert dict(back._metadata.public) == dict(plate._metadata.public)
    assert dict(back._metadata.protected) == dict(plate._metadata.protected)
    assert dict(back._metadata.imported) == dict(plate._metadata.imported)
    assert back._metadata.public["Metadata_Strain"] == "BY4741"


def test_stored_metadata_overwrites_constructor_defaults(
    plate: Image, tmp_path: Path
) -> None:
    """The HDF loader's setdefault merge DROPS these; the store must not.

    ``Metadata_ImageName`` is set by the constructor before the loader runs, so
    a merge that skips already-populated keys restores a fresh UUID name
    instead of the stored one.
    """
    plate._metadata.protected["Metadata_ImageName"] = "plate_A01"
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert back._metadata.protected["Metadata_ImageName"] == "plate_A01"


def test_objmap_is_written_even_when_nothing_is_detected(
    blank: Image, tmp_path: Path
) -> None:
    """Stage 1 relies on this: valid_staged_store requires objmap to exist."""
    assert blank.num_objects == 0
    store = blank.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    objmap = block[PhenotypicAttr.LABELS]["objmap"]
    assert (store / objmap / "0").is_dir()
    assert (Image.load_zarr(store).objmap[:] == 0).all()


def test_a_zeros_objmap_store_is_accepted_by_valid_staged_store(
    blank: Image, tmp_path: Path
) -> None:
    from phenotypic.sdk_.ngff_ import valid_staged_store

    assert valid_staged_store(blank.save2zarr(tmp_path / "p.ome.zarr"))


def test_rgb_is_omitted_entirely_when_empty(tmp_path: Path) -> None:
    # Image(<2-D ndarray>) yields rgb.isempty() is True -- verified by
    # execution. There is no rgb.clear(); the accessor has no such method.
    gray_only = Image(np.asarray(Image(load_synth_yeast_plate()).gray[:]))
    assert gray_only.rgb.isempty()
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    assert not (store / "rgb").exists()
    block = read_phenotypic_attributes(store)
    assert "rgb" not in block[PhenotypicAttr.SERIES]
    assert block[PhenotypicAttr.LABELS]["objmap"] == "gray/labels/objmap"


def test_an_rgb_less_store_round_trips_through_gray(tmp_path: Path) -> None:
    source = Image(load_synth_yeast_plate())
    gray_only = Image(np.asarray(source.gray[:]))
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    back = Image.load_zarr(store)
    assert back.rgb.isempty()
    np.testing.assert_array_equal(back.gray[:], gray_only.gray[:])
    np.testing.assert_array_equal(back.detect_mat[:], gray_only.detect_mat[:])


def test_primary_series_is_first_in_the_ome_series_list(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    ome = json.loads((store / "OME" / "zarr.json").read_text(encoding="utf-8"))
    assert ome["attributes"]["ome"]["series"][0] == "rgb"


def test_ome_xml_is_written_with_one_image_per_series(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    xml = (store / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    height, width, channels = plate.rgb[:].shape
    assert xml.count("<Image ") == 3
    assert f'SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="{channels}"' in xml
    assert "<MetadataOnly/>" in xml


def test_root_zarr_json_is_written_last(
    plate: Image, tmp_path: Path, monkeypatch
) -> None:
    """An interrupted store has no valid root and must read as absent.

    Records the ACTUAL write order. Collecting ``rglob("zarr.json")`` and
    sorting instead would assert nothing: ``"zarr.json"`` sorts after every
    nested path whatever order they were written in.
    """
    from phenotypic._core._image_parts._image_io_handler import ImageIOHandler
    from phenotypic.sdk_ import ngff_

    order: list[str] = []
    chunks_present_at_root_write: list[bool] = []
    part_dir: list[Path] = []
    real_write = ImageIOHandler._write_group_json
    real_promote = ngff_.promote_store

    def _record(group_dir, attributes):
        order.append(str(group_dir))
        if PhenotypicAttr.ROOT in attributes:
            chunks_present_at_root_write.append(
                any(Path(group_dir).glob("gray/0/c/*"))
                or any(Path(group_dir).glob("gray/0/*"))
            )
        return real_write(group_dir, attributes)

    def _capture_part(part: Path, final: Path, *, fsync: bool):
        part_dir.append(Path(part))
        return real_promote(part, final, fsync=fsync)

    monkeypatch.setattr(
        ImageIOHandler, "_write_group_json", staticmethod(_record)
    )
    monkeypatch.setattr(ngff_, "promote_store", _capture_part)
    plate.save2zarr(tmp_path / "p.ome.zarr")

    part = str(part_dir[0])
    assert order[-1] == part, order
    assert order[-2] == str(Path(part) / "OME"), order
    assert order.count(part) == 1
    assert chunks_present_at_root_write == [True]


def test_failed_store_write_removes_only_the_part_owned_by_that_attempt(
    plate: Image, tmp_path: Path, monkeypatch
) -> None:
    """A failed writer must not delete a same-target writer's live part."""
    from phenotypic._core._image_parts._image_io_handler import ImageIOHandler
    from phenotypic.sdk_ import ngff_

    final = tmp_path / "p.ome.zarr"
    live_part = ngff_.new_part_path(final)
    live_part.mkdir()
    allocated: list[Path] = []
    real_new_part_path = ngff_.new_part_path

    def _capture_part(path: Path) -> Path:
        part = real_new_part_path(path)
        allocated.append(part)
        return part

    def _fail_after_creating_part(*args, **kwargs) -> None:
        allocated[0].mkdir(parents=True, exist_ok=True)
        raise OSError("write failed")

    monkeypatch.setattr(ngff_, "new_part_path", _capture_part)
    monkeypatch.setattr(ImageIOHandler, "_write_series", _fail_after_creating_part)

    with pytest.raises(OSError, match="write failed"):
        plate.save2zarr(final)

    assert len(allocated) == 1
    assert not allocated[0].exists(), "the failed call must remove its own part"
    assert live_part.is_dir(), "a concurrent same-target writer owns this part"


def test_a_store_whose_root_is_missing_reads_as_absent(
    plate: Image, tmp_path: Path
) -> None:
    from phenotypic.sdk_.ngff_ import valid_staged_store

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    (store / "zarr.json").unlink()
    assert valid_staged_store(store) is False
    with pytest.raises(FileNotFoundError):
        Image.load_zarr(store)


def test_work_id_is_written_into_the_block_not_patched(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr", work_id="w-42")
    assert read_phenotypic_attributes(store)[PhenotypicAttr.WORK_ID] == "w-42"


def test_work_id_is_absent_when_not_supplied(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert PhenotypicAttr.WORK_ID not in read_phenotypic_attributes(store)


def test_store_schema_version_is_recorded(plate: Image, tmp_path: Path) -> None:
    from phenotypic.sdk_.ngff_ import STORE_SCHEMA_VERSION

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == STORE_SCHEMA_VERSION


def test_load_zarr_raises_on_a_newer_store_schema_version(
    plate: Image, tmp_path: Path
) -> None:
    """User ruling 2026-08-19: the loader is the path a user invokes.

    ``valid_staged_store`` returns False on a mismatch; the loader must say
    why, naming both versions.
    """
    from phenotypic.sdk_.ngff_ import STORE_SCHEMA_VERSION

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    root = store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"][PhenotypicAttr.STORE_SCHEMA_VERSION] = (
        STORE_SCHEMA_VERSION + 1
    )
    root.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="newer PhenoTypic") as excinfo:
        Image.load_zarr(store)
    message = str(excinfo.value)
    assert str(STORE_SCHEMA_VERSION) in message
    assert str(STORE_SCHEMA_VERSION + 1) in message


def test_load_layer_zarr_raises_on_a_newer_store_schema_version(
    plate: Image, tmp_path: Path
) -> None:
    """The layer reader decodes pixels, so it carries the same gate.

    Not covered by any plan task: the 2026-08-19 ruling landed after Phase 2's
    text was written and says "the loader", and ``load_layer_zarr`` is a second
    entry point into the same bytes -- the GUI tile server's, called per tile.
    Ungated, a v4 store renders as v3-decoded pixels with nothing raised.

    The gate costs no I/O: ``require_readable_store`` is the same read of the
    root ``zarr.json`` this method already performed to resolve the layer.
    """
    from phenotypic.sdk_.ngff_ import STORE_SCHEMA_VERSION

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    root = store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"][PhenotypicAttr.STORE_SCHEMA_VERSION] = (
        STORE_SCHEMA_VERSION + 1
    )
    root.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="newer PhenoTypic") as excinfo:
        Image.load_layer_zarr(store, "gray")
    message = str(excinfo.value)
    assert str(STORE_SCHEMA_VERSION) in message
    assert str(STORE_SCHEMA_VERSION + 1) in message


def test_load_zarr_raises_when_the_version_marker_is_missing(
    plate: Image, tmp_path: Path
) -> None:
    """Gated by VALUE, not presence -- absence is a mismatch too."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    root = store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    del payload["attributes"]["phenotypic"][PhenotypicAttr.STORE_SCHEMA_VERSION]
    root.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        Image.load_zarr(store)


def test_metadata_schema_version_is_not_written(plate: Image, tmp_path: Path) -> None:
    """Task 1.3 dropped it; nothing may reintroduce it."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert "metadata_schema_version" not in read_phenotypic_attributes(store)


def test_pyramid_depth_is_uniform_across_every_series(
    plate: Image, tmp_path: Path
) -> None:
    """NGFF requires a label image to carry its parent's level count."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    levels = block[PhenotypicAttr.PYRAMID]["levels"]
    assert levels > 1
    members = [
        *block[PhenotypicAttr.SERIES].values(),
        # `.get`: a label-less store OMITS the key (Task 1.3, ledger C3).
        *block.get(PhenotypicAttr.LABELS, {}).values(),
    ]
    assert len(members) == 4
    for member in members:
        found = sorted(p.name for p in (store / member).iterdir() if p.name.isdigit())
        assert found == [str(i) for i in range(levels)], member


def test_pyramid_depth_is_a_pure_function_of_shape(plate: Image, tmp_path: Path) -> None:
    """Fixed, not tunable: save2zarr takes no pyramid argument at all (P3)."""
    import inspect

    from phenotypic.sdk_ import ngff_

    assert "pyramid_levels" not in inspect.signature(Image.save2zarr).parameters
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    gray = plate.gray[:]
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == (
        ngff_.pyramid_level_count(gray.shape[0], gray.shape[1])
    )


def test_the_objmap_pyramid_is_nearest_neighbour_not_mean(
    plate: Image, tmp_path: Path
) -> None:
    """A mean-downsampled label map fabricates labels at no level-0 pixel."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    level0 = Image.load_layer_zarr(store, "objmap", level=0)
    level1 = Image.load_layer_zarr(store, "objmap", level=1)
    np.testing.assert_array_equal(level1, level0[::2, ::2])
    assert set(np.unique(level1)) <= set(np.unique(level0))


def test_the_image_pyramid_is_a_block_mean_not_a_subsample(
    plate: Image, tmp_path: Path
) -> None:
    from phenotypic.sdk_ import ngff_

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    level0 = Image.load_layer_zarr(store, "gray", level=0)
    level1 = Image.load_layer_zarr(store, "gray", level=1)
    np.testing.assert_array_equal(level1, ngff_.downsample_image(level0))


def test_omero_is_emitted_for_rgb_and_omitted_for_the_float_series(
    plate: Image, tmp_path: Path
) -> None:
    """Keyed on dtype: a bit-depth window over float data renders black."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")

    def _group(member: str) -> dict:
        return json.loads(
            (store / member / "zarr.json").read_text(encoding="utf-8")
        )["attributes"]["ome"]

    rgb = _group("rgb")
    assert [channel["label"] for channel in rgb["omero"]["channels"]] == [
        "R",
        "G",
        "B",
    ]
    assert rgb["omero"]["channels"][0]["window"] == {
        "min": 0,
        "max": 255,
        "start": 0,
        "end": 255,
    }
    assert "omero" not in _group("gray")
    assert "omero" not in _group("detect_mat")


def test_image_label_block_is_emitted_on_the_label_group(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    objmap = read_phenotypic_attributes(store)[PhenotypicAttr.LABELS]["objmap"]
    block = json.loads(
        (store / objmap / "zarr.json").read_text(encoding="utf-8")
    )["attributes"]["ome"]
    assert block["image-label"]["colors"] == [
        {"label-value": 0, "rgba": [0, 0, 0, 0]}
    ]
    assert "omero" not in block
    assert block["multiscales"][0]["type"] == "nearest"


def test_labels_group_lists_the_objmap(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = json.loads(
        (store / "rgb" / "labels" / "zarr.json").read_text(encoding="utf-8")
    )
    assert block["attributes"]["ome"]["labels"] == ["objmap"]


def test_load_layer_zarr_reads_one_layer_without_a_full_image(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    np.testing.assert_array_equal(Image.load_layer_zarr(store, "gray"), plate.gray[:])


def test_load_layer_zarr_returns_rgb_channel_last(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    np.testing.assert_array_equal(
        Image.load_layer_zarr(store, "rgb"), plate.rgb[:]
    )


def test_load_layer_zarr_resolves_objmap_via_the_labels_key(
    tmp_path: Path,
) -> None:
    """rgb-less stores put the label under gray; a hard-coded path would 404."""
    # Image(<2-D ndarray>) yields rgb.isempty() is True -- verified by
    # execution. There is no rgb.clear(); the accessor has no such method.
    gray_only = Image(np.asarray(Image(load_synth_yeast_plate()).gray[:]))
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    assert Image.load_layer_zarr(store, "objmap").shape == gray_only.gray[:].shape


def test_load_layer_zarr_can_read_a_pyramid_level(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    full = Image.load_layer_zarr(store, "gray", level=0)
    half = Image.load_layer_zarr(store, "gray", level=1)
    assert half.shape == ((full.shape[0] + 1) // 2, (full.shape[1] + 1) // 2)


def test_load_layer_zarr_raises_keyerror_for_an_unknown_layer(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    with pytest.raises(KeyError):
        Image.load_layer_zarr(store, "not_a_layer")


def test_load_layer_zarr_raises_keyerror_for_rgb_in_an_rgb_less_store(
    tmp_path: Path,
) -> None:
    gray_only = Image(np.asarray(Image(load_synth_yeast_plate()).gray[:]))
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    with pytest.raises(KeyError):
        Image.load_layer_zarr(store, "rgb")


def test_image_load_zarr_on_a_gridimage_store_warns_without_upcasting(
    tmp_path: Path,
) -> None:
    grid = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    with pytest.warns(UserWarning, match="GridImage"):
        back = Image.load_zarr(store)
    assert type(back) is Image


def test_image_load_zarr_on_an_image_store_does_not_warn(
    plate: Image, tmp_path: Path
) -> None:
    import warnings as _warnings

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        assert type(Image.load_zarr(store)) is Image


def test_image_class_and_image_type_are_independent(tmp_path: Path) -> None:
    plain = Image(load_synth_yeast_plate())
    plain._metadata.protected["Metadata_ImageType"] = "GridSection"
    store = plain.save2zarr(tmp_path / "s.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "Image"
    assert (
        block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"]
        == "GridSection"
    )
    back = Image.load_zarr(store)
    assert type(back) is Image
    assert back._metadata.protected["Metadata_ImageType"] == "GridSection"


def test_load_image_from_store_dispatches_on_image_class(tmp_path: Path) -> None:
    from phenotypic.sdk_ import load_image_from_store

    plain = Image(load_synth_yeast_plate())
    plain._metadata.protected["Metadata_ImageType"] = "GridSection"
    store = plain.save2zarr(tmp_path / "s.ome.zarr")
    assert type(load_image_from_store(store)) is Image


def test_detect_mode_illuminant_and_gamma_survive(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert back.illuminant == plate.illuminant
    assert str(back.gamma) == str(plate.gamma)
    assert back._data.detect_mode == plate._data.detect_mode


def test_a_non_default_detect_mode_survives(plate: Image, tmp_path: Path) -> None:
    """The default is "gray", so a dropped value looks correct on the fixture."""
    plate._data.detect_mode = "red"
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert read_phenotypic_attributes(store)[PhenotypicAttr.DETECT_MODE] == "red"
    assert Image.load_zarr(store)._data.detect_mode == "red"


def test_a_non_default_illuminant_survives(tmp_path: Path) -> None:
    plate = Image(load_synth_yeast_plate(), illuminant="D50")
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert read_phenotypic_attributes(store)[PhenotypicAttr.ILLUMINANT] == "D50"
    assert Image.load_zarr(store).illuminant == "D50"


def test_explicit_kwargs_take_priority_over_the_stored_state(
    tmp_path: Path,
) -> None:
    plate = Image(load_synth_yeast_plate(), illuminant="D50")
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert Image.load_zarr(store, illuminant="D65").illuminant == "D65"


def test_a_failed_ome_xml_build_aborts_the_write(
    plate: Image, tmp_path: Path, monkeypatch
) -> None:
    """Fatal, not degraded (ALGO-3). Dropping `series` while keeping named
    groups satisfied neither arm of §2.2.3, so the old fallback shipped a store
    LESS conformant than either option."""
    from phenotypic.sdk_ import ngff_

    def _boom(**kwargs):
        raise RuntimeError("synthetic OME-XML failure")

    monkeypatch.setattr(ngff_, "build_ome_xml", _boom)
    with pytest.raises(RuntimeError):
        plate.save2zarr(tmp_path / "p.ome.zarr")
    assert not (tmp_path / "p.ome.zarr").exists(), (
        "a failed write must leave no store -- the .part is never promoted"
    )


def test_saving_over_an_existing_store_replaces_it(
    plate: Image, blank: Image, tmp_path: Path
) -> None:
    """Promote is a move-aside; os.replace onto a non-empty dir is ENOTEMPTY."""
    target = tmp_path / "p.ome.zarr"
    plate.save2zarr(target)
    blank.save2zarr(target)
    assert Image.load_zarr(target).num_objects == 0
    assert not list(tmp_path.glob(".*.part"))
    assert not list(tmp_path.glob(".*.trash"))


def test_multiscales_datasets_describe_every_level_that_was_written(
    plate: Image, tmp_path: Path
) -> None:
    """The public pyramid record must match the arrays on disk.

    ``attributes.phenotypic.pyramid.levels`` and the level directories can
    agree while ``ome.multiscales[].datasets`` lists fewer -- an external
    viewer then sees a pyramid missing its lower levels, and nothing in
    PhenoTypic notices because PhenoTypic never reads the OME projection.
    """
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    levels = block[PhenotypicAttr.PYRAMID]["levels"]
    members = [
        *block[PhenotypicAttr.SERIES].values(),
        *block.get(PhenotypicAttr.LABELS, {}).values(),
    ]
    for member in members:
        payload = json.loads(
            (store / member / "zarr.json").read_text(encoding="utf-8")
        )["attributes"]["ome"]
        datasets = payload["multiscales"][0]["datasets"]
        assert [d["path"] for d in datasets] == [str(i) for i in range(levels)], (
            member
        )
        for dataset in datasets:
            assert (store / member / dataset["path"] / "zarr.json").is_file()


def test_odd_singleton_multiscales_match_the_on_disk_golden(
    odd_singleton_rgb: Image, tmp_path: Path
) -> None:
    """Pin every image/label multiscales output a third-party reader sees."""
    store = odd_singleton_rgb.save2zarr(tmp_path / "odd.ome.zarr")
    block = read_phenotypic_attributes(store)
    members = {
        **block[PhenotypicAttr.SERIES],
        **block[PhenotypicAttr.LABELS],
    }
    observed = {
        logical_name: json.loads(
            (store / member / "zarr.json").read_text(encoding="utf-8")
        )["attributes"]["ome"]["multiscales"]
        for logical_name, member in members.items()
    }
    expected = json.loads(_NGFF_ODD_GOLDEN.read_text(encoding="utf-8"))
    assert observed == expected


def test_odd_singleton_sampling_transforms_follow_the_arrays_on_disk(
    odd_singleton_rgb: Image, tmp_path: Path
) -> None:
    """Exercise shapes, channel semantics, ordering, and label co-registration."""
    store = odd_singleton_rgb.save2zarr(tmp_path / "odd.ome.zarr")
    block = read_phenotypic_attributes(store)
    members = {
        **block[PhenotypicAttr.SERIES],
        **block[PhenotypicAttr.LABELS],
    }

    expected_2d_shapes = [(1025, 1), (513, 1), (257, 1)]
    expected_3d_shapes = [(3, 1025, 1), (3, 513, 1), (3, 257, 1)]
    expected_2d_transforms = [
        [{"type": "scale", "scale": [1.0, 1.0]}],
        [
            {"type": "scale", "scale": [2.0, 1.0]},
            {"type": "translation", "translation": [0.5, 0.0]},
        ],
        [
            {"type": "scale", "scale": [4.0, 1.0]},
            {"type": "translation", "translation": [1.5, 0.0]},
        ],
    ]
    expected_3d_transforms = [
        [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
        [
            {"type": "scale", "scale": [1.0, 2.0, 1.0]},
            {"type": "translation", "translation": [0.0, 0.5, 0.0]},
        ],
        [
            {"type": "scale", "scale": [1.0, 4.0, 1.0]},
            {"type": "translation", "translation": [0.0, 1.5, 0.0]},
        ],
    ]

    observed_transforms: dict[str, list[list[dict]]] = {}
    for logical_name, member in members.items():
        group = json.loads(
            (store / member / "zarr.json").read_text(encoding="utf-8")
        )["attributes"]["ome"]["multiscales"][0]
        observed_transforms[logical_name] = [
            dataset["coordinateTransformations"] for dataset in group["datasets"]
        ]
        on_disk_shapes = [
            tuple(_read_array_json(store, member, level)["shape"])
            for level in range(3)
        ]
        expected_shapes = (
            expected_3d_shapes if logical_name == "rgb" else expected_2d_shapes
        )
        assert on_disk_shapes == expected_shapes, logical_name

    assert observed_transforms["rgb"] == expected_3d_transforms
    for logical_name in ("gray", "detect_mat", "objmap"):
        assert observed_transforms[logical_name] == expected_2d_transforms
    for rgb_level, label_level in zip(
        observed_transforms["rgb"], observed_transforms["objmap"], strict=True
    ):
        assert [item["type"] for item in rgb_level] == [
            item["type"] for item in label_level
        ]
        for rgb_transform, label_transform in zip(
            rgb_level, label_level, strict=True
        ):
            vector_name = rgb_transform["type"]
            assert rgb_transform[vector_name][1:] == label_transform[vector_name]


# ---------------------------------------------------------------------------
# The layers on disk -- a lazily-derived layer makes a naive round-trip vacuous
# ---------------------------------------------------------------------------


def _open_store_array(store: Path, member: str, level: int = 0) -> np.ndarray:
    """Read one member's level array straight off disk, bypassing the loader.

    Every other assertion here goes through ``load_zarr``, which cannot tell a
    correctly-written store from one whose loader re-derives what it failed to
    read.
    """
    import zarr

    from phenotypic.sdk_ import ngff_

    return zarr.open_array(
        store=ngff_.long_path(store / member / str(level)), mode="r"
    )[...]


@pytest.fixture
def enhanced(plate: Image) -> Image:
    """A plate whose ``detect_mat`` no longer equals its ``gray``.

    On the raw fixture ``detect_mat`` IS ``gray`` -- verified by execution,
    ``np.array_equal(img.gray[:], img.detect_mat[:])`` is True -- because
    ``detect_mat`` is derived from ``gray`` on first access and nothing has
    enhanced it. Any assertion about ``detect_mat`` on the raw fixture
    therefore passes whether the loader reads the stored array or silently
    re-derives it, and whether the writer stored ``detect_mat`` or ``gray``.
    ``.apply`` returns a copy; it does not mutate *plate*.
    """
    from phenotypic.enhance import BlurGauss

    return BlurGauss(sigma=6.0).apply(plate)


def test_a_stored_detect_mat_is_read_back_not_re_derived(
    enhanced: Image, tmp_path: Path
) -> None:
    """Dropping the ``detect_mat`` read in the loader must not pass."""
    assert not np.array_equal(enhanced.detect_mat[:], enhanced.gray[:])
    store = enhanced.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    np.testing.assert_array_equal(back.detect_mat[:], enhanced.detect_mat[:])
    assert not np.array_equal(back.detect_mat[:], back.gray[:])


def test_a_stored_gray_is_read_back_not_re_derived_from_rgb(
    plate: Image, tmp_path: Path
) -> None:
    """``gray`` is derived from ``rgb`` on first access, so a loader that never
    assigns the stored array still returns the right pixels for an untouched
    image. Store a ``gray`` that the derivation cannot reproduce."""
    derived = np.array(plate.gray[:])
    plate.gray[:] = (derived * 0.25).astype(derived.dtype)
    assert not np.array_equal(plate.gray[:], derived)

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    np.testing.assert_array_equal(back.rgb[:], plate.rgb[:])
    np.testing.assert_array_equal(back.gray[:], plate.gray[:])
    assert not np.array_equal(back.gray[:], derived)


def test_every_series_holds_its_own_pixels_on_disk(
    enhanced: Image, tmp_path: Path
) -> None:
    """Read the arrays with zarr directly: a writer that fed ``detect_mat``
    into the ``gray`` group produces a fully conformant store that PhenoTypic
    round-trips, and only an external viewer would ever see it."""
    store = enhanced.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)

    np.testing.assert_array_equal(
        _open_store_array(store, "gray"), enhanced.gray[:]
    )
    np.testing.assert_array_equal(
        _open_store_array(store, "detect_mat"), enhanced.detect_mat[:]
    )
    np.testing.assert_array_equal(
        _open_store_array(store, "rgb"),
        np.moveaxis(enhanced.rgb[:], -1, 0),
    )
    np.testing.assert_array_equal(
        _open_store_array(store, block[PhenotypicAttr.LABELS]["objmap"]),
        enhanced.objmap[:],
    )


# ---------------------------------------------------------------------------
# durable= must reach promote_store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("durable", [True, False])
def test_the_durable_flag_reaches_promote_store(
    plate: Image, tmp_path: Path, monkeypatch, durable: bool
) -> None:
    """The fsync-before-promote guarantee for SLURM Stage-1 writes.

    Dropping the argument loses stores on node loss and is invisible to every
    other assertion in this file -- an unsynced store reads back correctly on a
    machine that did not crash.
    """
    from phenotypic.sdk_ import ngff_

    recorded: list[bool] = []
    real_promote = ngff_.promote_store

    def _capture(part: Path, final: Path, *, fsync: bool):
        recorded.append(fsync)
        return real_promote(part, final, fsync=fsync)

    monkeypatch.setattr(ngff_, "promote_store", _capture)
    plate.save2zarr(tmp_path / "p.ome.zarr", durable=durable)
    assert recorded == [durable]


# ---------------------------------------------------------------------------
# omero window, gamma, and the metadata sections
# ---------------------------------------------------------------------------


@pytest.fixture
def sixteen_bit(plate: Image) -> Image:
    """A 16-bit plate. ``257 * uint8`` maps 255 -> 65535 exactly."""
    return Image(np.asarray(plate.rgb[:]).astype(np.uint16) * 257, bit_depth=16)


def test_the_omero_window_follows_the_source_bit_depth(
    sixteen_bit: Image, tmp_path: Path
) -> None:
    """Hard-coding 8 puts a ``0..255`` window over 16-bit data and every viewer
    honouring ``omero`` renders the plate blown out. The only other fixture in
    this file is 8-bit, where the hard-coded value is indistinguishable."""
    assert sixteen_bit.bit_depth == 16
    store = sixteen_bit.save2zarr(tmp_path / "p.ome.zarr")
    channels = json.loads(
        (store / "rgb" / "zarr.json").read_text(encoding="utf-8")
    )["attributes"]["ome"]["omero"]["channels"]
    for channel in channels:
        assert channel["window"] == {
            "min": 0,
            "max": 65535,
            "start": 0,
            "end": 65535,
        }
    assert Image.load_zarr(store).bit_depth == 16


def test_a_non_default_gamma_survives(plate: Image, tmp_path: Path) -> None:
    """The default is sRGB, so a dropped gamma looks correct on every other
    fixture here."""
    from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

    linear = Image(np.asarray(plate.rgb[:]), gamma=GAMMA_ENCODINGS.LINEAR)
    store = linear.save2zarr(tmp_path / "p.ome.zarr")
    assert read_phenotypic_attributes(store)[PhenotypicAttr.GAMMA] == "LINEAR"
    assert Image.load_zarr(store).gamma is GAMMA_ENCODINGS.LINEAR


def test_the_imported_metadata_section_round_trips(
    plate: Image, tmp_path: Path
) -> None:
    """``imported`` is empty on the fixture, so ``{} == {}`` passes for a
    writer that drops the section entirely."""
    plate._metadata.imported["Metadata_SourceFile"] = "plate_A01.tif"
    plate._metadata.imported["Metadata_Exposure"] = 250
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    stored = read_phenotypic_attributes(store)[PhenotypicAttr.METADATA]
    assert stored[PhenotypicAttr.IMPORTED] == {
        "Metadata_SourceFile": "plate_A01.tif",
        "Metadata_Exposure": 250,
    }
    back = Image.load_zarr(store)
    assert dict(back._metadata.imported) == dict(plate._metadata.imported)


def test_a_non_json_native_metadata_value_serialises_instead_of_aborting(
    plate: Image, tmp_path: Path
) -> None:
    """Exactly the case ``_write_group_json``'s ``default=str`` exists for: an
    ``np.datetime64`` read off a TIFF. Metadata values are stored verbatim and
    unvalidated, so without the hook the whole write raises ``TypeError``."""
    plate._metadata.public["Metadata_AcquiredAt"] = np.datetime64(
        "2026-08-19T10:30"
    )
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    stored = read_phenotypic_attributes(store)[PhenotypicAttr.METADATA]
    assert stored[PhenotypicAttr.PUBLIC]["Metadata_AcquiredAt"] == (
        "2026-08-19T10:30"
    )
    assert (
        Image.load_zarr(store)._metadata.public["Metadata_AcquiredAt"]
        == "2026-08-19T10:30"
    )


@pytest.mark.parametrize("section", ["protected", "public", "imported"])
def test_a_metadata_key_collision_names_the_section_it_was_found_in(
    plate: Image, tmp_path: Path, section: str
) -> None:
    """The loader normalizes each section under its own name. Passing a
    constant section name instead leaves every collision blaming ``public``,
    and a real one in ``protected`` or ``imported`` is unfindable."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    root = store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    block = payload["attributes"]["phenotypic"][PhenotypicAttr.METADATA]
    # 'ImageName' is a known bare label and 'Metadata_ImageName' its header;
    # both normalize to 'Metadata_ImageName'.
    block[section]["ImageName"] = "from_the_bare_label"
    block[section]["Metadata_ImageName"] = "from_the_header"
    root.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Conflicting stored metadata keys") as exc:
        Image.load_zarr(store)
    assert f"section {section!r}" in str(exc.value)


# ---------------------------------------------------------------------------
# load_image_from_store -- dispatch is on image_class, never Metadata_ImageType
# ---------------------------------------------------------------------------


def test_load_image_from_store_ignores_a_grid_image_type_on_a_plain_image(
    tmp_path: Path,
) -> None:
    """The discriminating direction: ``Metadata_ImageType`` says ``GridImage``
    and ``image_class`` says ``Image``. Dispatching on the metadata field --
    or on an enum that merely happens to spell the same string -- returns a
    GridImage here."""
    from phenotypic.sdk_ import load_image_from_store

    plain = Image(load_synth_yeast_plate())
    plain._metadata.protected["Metadata_ImageType"] = "GridImage"
    store = plain.save2zarr(tmp_path / "s.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "Image"
    assert block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"] == (
        "GridImage"
    )
    back = load_image_from_store(store)
    assert type(back) is Image


def test_load_image_from_store_falls_back_when_image_class_is_absent(
    tmp_path: Path,
) -> None:
    """The ``fallback`` parameter has to select the class, not decorate the
    signature: a store with no ``image_class`` is what the parameter is for."""
    from phenotypic.sdk_ import load_image_from_store

    grid = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root = store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    del payload["attributes"]["phenotypic"][PhenotypicAttr.IMAGE_CLASS]
    root.write_text(json.dumps(payload), encoding="utf-8")

    assert PhenotypicAttr.IMAGE_CLASS not in read_phenotypic_attributes(store)
    assert type(load_image_from_store(store)) is Image
    back = load_image_from_store(store, fallback="GridImage")
    assert type(back) is GridImage
    assert (back.nrows, back.ncols) == (8, 12)
