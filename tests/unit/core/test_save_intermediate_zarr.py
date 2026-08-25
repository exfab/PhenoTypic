"""``save_intermediate_zarr``: single-level preview stores for the GUI builder.

Every node preview the builder caches goes through here, so the shapes asserted
below are the ones the DAG actually produces: ``_layers_modified_by`` returns
``("detect_mat",)`` for every ``ImageEnhancer`` and ``("objmap",)`` for every
``ObjectDetector``/``ObjectRefiner``. Both are exercised end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes


@pytest.fixture(scope="module")
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def _series(store: Path) -> dict:
    return read_phenotypic_attributes(store)[PhenotypicAttr.SERIES]


def test_writes_only_the_requested_layers(plate: Image, tmp_path: Path) -> None:
    store = plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    assert set(_series(store)) == {"gray"}
    # Not merely absent from the block -- absent from disk.
    assert not (store / "detect_mat").exists()
    assert not (store / "rgb").exists()


def test_is_single_level_by_design(plate: Image, tmp_path: Path) -> None:
    """Node previews are transient; pyramiding them multiplies cache inodes."""
    store = plate.save_intermediate_zarr(
            tmp_path / "n.ome.zarr", layers=("gray", "detect_mat")
    )
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == 1
    assert not (store / "gray" / "1").exists()
    assert not (store / "detect_mat" / "1").exists()
    # The OME projection must agree with the arrays actually written: one
    # dataset per series, not the full-resolution store's level count.
    ome = json.loads((store / "gray" / "zarr.json").read_text())["attributes"]["ome"]
    assert len(ome["multiscales"][0]["datasets"]) == 1


def test_round_trips_through_load_layer_zarr(plate: Image, tmp_path: Path) -> None:
    store = plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    np.testing.assert_array_equal(Image.load_layer_zarr(store, "gray"), plate.gray[:])


def test_unknown_layer_names_raise(plate: Image, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown layer"):
        plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("nope",))
    assert not (tmp_path / "n.ome.zarr").exists()


def test_uses_the_promote_primitive(plate: Image, tmp_path: Path, monkeypatch) -> None:
    """Dash callbacks write these concurrently; a half-written dir is live risk."""
    from phenotypic.sdk_ import ngff_

    calls: list[str] = []
    real = ngff_.promote_store
    monkeypatch.setattr(
            ngff_,
            "promote_store",
            lambda part, final, *, fsync, commit_guard=None: (
                calls.append(final.name),
                real(
                    part, final, fsync=fsync, commit_guard=commit_guard
                ),
            )[1],
    )
    plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    assert calls == ["n.ome.zarr"]
    # Nothing is left behind for the orphan sweep to find.
    assert [p.name for p in tmp_path.iterdir()] == ["n.ome.zarr"]


# ---------------------------------------------------------------------------
# The two shapes the real DAG produces. Both were dead under the draft that
# passed `series=tuple(layers)` straight through.
# ---------------------------------------------------------------------------


def test_enhancer_shape_is_loadable(plate: Image, tmp_path: Path) -> None:
    """``_layers_modified_by`` returns ``("detect_mat",)`` for EVERY enhancer."""
    store = plate.save_intermediate_zarr(
            tmp_path / "n.ome.zarr", layers=("detect_mat",)
    )
    block = read_phenotypic_attributes(store)
    assert set(block[PhenotypicAttr.SERIES]) == {"gray", "detect_mat"}
    np.testing.assert_array_equal(
            Image.load_layer_zarr(store, "detect_mat"), plate.detect_mat[:]
    )
    # No label was requested, so the key is ABSENT -- not an empty mapping.
    assert PhenotypicAttr.LABELS not in block
    assert not (store / "gray" / "labels").exists()


def test_detector_shape_is_loadable(plate: Image, tmp_path: Path) -> None:
    """``_layers_modified_by`` returns ``("objmap",)`` for EVERY detector."""
    store = plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("objmap",))
    block = read_phenotypic_attributes(store)
    # `objmap` is the LABEL; it is never a member of `series`.
    assert set(block[PhenotypicAttr.SERIES]) == {"gray"}
    assert block[PhenotypicAttr.LABELS] == {"objmap": "gray/labels/objmap"}
    objmap = Image.load_layer_zarr(store, "objmap")
    np.testing.assert_array_equal(objmap, plate.objmap[:])
    # A value assertion, not a shape one: the fixture plate has colonies, so a
    # store that wrote zeros (or a fresh Image's empty objmap) fails here.
    assert int(objmap.max()) == int(plate.objmap[:].max()) > 0


def test_a_preview_store_always_has_a_primary_series(
        plate: Image, tmp_path: Path
) -> None:
    """User ruling 2026-08-19: `gray` is co-written whatever the node changed.

    Without it there is no anchor for ``objmap_path``, the ``labels`` group, or
    the OME projection, and ``primary_series`` raises.
    """
    from phenotypic.sdk_ import ngff_

    for layers in (("detect_mat",), ("objmap",), ("gray",)):
        store = plate.save_intermediate_zarr(
                tmp_path / f"{'_'.join(layers)}.ome.zarr", layers=layers
        )
        names = list(read_phenotypic_attributes(store)[PhenotypicAttr.SERIES])
        assert ngff_.primary_series(names) == "gray"
        # Co-written by VALUE, not merely declared: a `gray` group holding a
        # zeros array would satisfy a presence-only assertion.
        np.testing.assert_array_equal(
                Image.load_layer_zarr(store, "gray"), plate.gray[:]
        )


def test_corrector_shape_round_trips_every_layer(
        plate: Image, tmp_path: Path
) -> None:
    """``ImageCorrector`` reports all four layers -- the widest preview shape."""
    store = plate.save_intermediate_zarr(
            tmp_path / "n.ome.zarr", layers=("rgb", "gray", "detect_mat", "objmap")
    )
    block = read_phenotypic_attributes(store)
    assert list(block[PhenotypicAttr.SERIES]) == ["rgb", "gray", "detect_mat"]
    # rgb is present, so the label moves under it.
    assert block[PhenotypicAttr.LABELS] == {"objmap": "rgb/labels/objmap"}
    for layer in ("rgb", "gray", "detect_mat", "objmap"):
        np.testing.assert_array_equal(
                Image.load_layer_zarr(store, layer), getattr(plate, layer)[:]
        )


def test_an_rgb_less_image_never_writes_an_rgb_series(tmp_path: Path) -> None:
    """``self.rgb[:]`` RAISES ``NoArrayError`` on a 2-D image.

    A corrector reports ``("rgb", ...)`` unconditionally, so a grayscale-loaded
    preview would abort mid-write if `rgb` were taken from `layers` verbatim.
    """
    flat = Image(arr=np.arange(48 * 64, dtype=np.uint8).reshape(48, 64))
    store = flat.save_intermediate_zarr(
            tmp_path / "n.ome.zarr", layers=("rgb", "gray", "detect_mat", "objmap")
    )
    assert set(read_phenotypic_attributes(store)[PhenotypicAttr.SERIES]) == {
        "gray", "detect_mat",
    }
    assert not (store / "rgb").exists()


def test_grid_state_survives_a_preview_store(tmp_path: Path) -> None:
    """The builder caches ``GridImage`` previews; the loader dispatches on it."""
    from phenotypic import GridImage

    grid = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    store = grid.save_intermediate_zarr(
            tmp_path / "n.ome.zarr", layers=("detect_mat",)
    )
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage"
    assert block[PhenotypicAttr.GRID]["nrows"] == 8
    assert block[PhenotypicAttr.GRID]["ncols"] == 12
