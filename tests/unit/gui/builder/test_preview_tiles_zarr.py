"""Builder node previews render from OME-Zarr node stores.

Two contracts here, and the second is the reason the describe helper had to
be lifted out of ``_build_manifest``:

* ``stage_channel_png`` renders a channel straight out of a node store.
* Building a manifest entry reads **metadata** -- ``phenotypic.series`` plus
  the level-0 array shapes -- and never opens a full :class:`Image`. As a
  closure inside ``_build_manifest`` that invariant was unassertable:
  ``_preview_cache._describe`` was not a module attribute and never was.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate


def test_channel_png_renders_from_a_node_store(tmp_path: Path) -> None:
    from phenotypic.gui.builder._preview_tiles import stage_channel_png

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr",
        layers=("gray", "detect_mat", "objmap"),
    )
    png = stage_channel_png(tmp_path, "block-1", "gray", store)
    assert png.is_file() and png.stat().st_size > 0


def test_channel_png_refreshes_when_the_store_is_republished(
    tmp_path: Path,
) -> None:
    """Freshness is measured against the root ``zarr.json``, not the directory.

    A store DIRECTORY's ``st_mtime_ns`` does not move when a nested chunk is
    rewritten, so a cache keyed on it serves the previous publish's pixels
    with nothing failing. The root is the file every promote writes last.

    This claim was carried in ``FEATURES.md`` against a results-viewer DZI
    test that only ever covered the DZI half; the results Plate has no
    rendered-PNG cache any more, so the builder half is asserted here
    directly.
    """
    from phenotypic.gui.builder._preview_tiles import stage_channel_png

    scope = tmp_path / "scope"
    target = tmp_path / "00_base.ome.zarr"
    first = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        target, layers=("gray",)
    )
    png = stage_channel_png(scope, "00", "gray", first)
    before = png.read_bytes()

    inverted = Image(load_synth_yeast_plate())
    # ``gray`` is float in [0, 1]; the accessor asserts the range on write.
    inverted.gray[:] = 1.0 - inverted.gray[:]
    republished = inverted.save_intermediate_zarr(target, layers=("gray",))

    assert stage_channel_png(scope, "00", "gray", republished).read_bytes() != (
        before
    )


def test_channel_png_reuses_the_cache_when_nothing_was_republished(
    tmp_path: Path,
) -> None:
    """The other half of the same key: an unchanged store must not re-render."""
    from phenotypic.gui.builder._preview_tiles import stage_channel_png

    scope = tmp_path / "scope"
    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray",)
    )
    png = stage_channel_png(scope, "00", "gray", store)
    stamp = png.stat().st_mtime_ns
    assert stage_channel_png(scope, "00", "gray", store).stat().st_mtime_ns == (
        stamp
    )


def test_channel_png_renders_the_requested_channel(tmp_path: Path) -> None:
    """Not merely "a PNG": the objmap and gray renders must differ.

    ``_channel_to_rgb_uint8`` dispatches per channel (``label2rgb`` for the
    objmap, contrast-normalised greyscale otherwise), so a staging function
    that ignored ``channel`` would still produce a valid PNG.
    """
    from phenotypic.gui.builder._preview_tiles import stage_channel_png

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr",
        layers=("gray", "detect_mat", "objmap"),
    )
    gray = stage_channel_png(tmp_path, "block-1", "gray", store).read_bytes()
    objmap = stage_channel_png(
        tmp_path, "block-1", "objmap", store
    ).read_bytes()
    assert gray != objmap


def test_manifest_describe_does_not_load_a_full_image(
    tmp_path: Path, monkeypatch
) -> None:
    """Reads store metadata only; a manifest entry must not cost a full decode."""
    from phenotypic.gui.builder import _preview_cache

    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray",)
    )
    monkeypatch.setattr(
        Image,
        "load_zarr",
        lambda *a, **k: pytest.fail("manifest must not load an Image"),
    )
    node = _preview_cache._describe_store_node(store)
    assert node is not None
    assert node["layers"] == ["gray"]


def test_manifest_describe_reports_the_level_zero_shape(tmp_path: Path) -> None:
    from phenotypic.gui.builder import _preview_cache

    image = Image(load_synth_yeast_plate())
    store = image.save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray",)
    )
    node = _preview_cache._describe_store_node(store)
    assert node is not None
    assert node["shape"] == list(image.gray[:].shape[:2])


def test_manifest_describe_counts_objects_when_a_label_exists(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.builder import _preview_cache

    image = Image(arr=np.zeros((32, 32, 3), dtype=np.uint8))
    labeled = np.zeros((32, 32), dtype=np.int32)
    labeled[2:8, 2:8] = 1
    labeled[12:18, 12:18] = 2
    image.objmap[:] = labeled
    store = image.save_intermediate_zarr(
        tmp_path / "00_base.ome.zarr", layers=("gray", "objmap")
    )
    node = _preview_cache._describe_store_node(store)
    assert node is not None
    assert node["num_objects"] == 2
    assert "objmap" in node["layers"]


def test_manifest_describe_treats_a_rootless_store_as_absent(
    tmp_path: Path,
) -> None:
    """An interrupted write leaves no root, and reads as ABSENT, not partial."""
    from phenotypic.gui.builder import _preview_cache

    partial = tmp_path / "00_base.ome.zarr"
    (partial / "gray" / "0").mkdir(parents=True)
    assert _preview_cache._describe_store_node(partial) is None


def test_describe_is_reached_through_compute_scope(
    tmp_path: Path, monkeypatch
) -> None:
    """The lifted helper must still be what the manifest is built from.

    Lifting a closure to module scope makes the invariant assertable, and
    immediately makes it possible for the lifted copy to go unused. This is
    the other half: ``compute_scope`` must route through it.

    Harness copied from ``tests/gui/builder/test_preview_compute_scope.py``,
    the only real entry point into ``_build_manifest``.
    """
    from phenotypic.gui.builder import _preview_cache as pc
    from phenotypic.gui.builder._state import (
        BlockNode,
        Edge,
        _DagBuilderScope,
        _DagBuilderState,
        _new_block_id,
    )

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")

    seen: list[Path] = []
    real_describe = pc._describe_store_node

    def _recording(store_path: Path):
        seen.append(Path(store_path))
        return real_describe(store_path)

    monkeypatch.setattr(pc, "_describe_store_node", _recording)

    blur = BlockNode(
        block_id=_new_block_id(), class_name="BlurGauss", params={"sigma": 1}
    )
    scope = _DagBuilderScope()  # __post_init__ seeds InputImage at index 0
    scope.blocks.append(blur)
    scope.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=scope.blocks[0].block_id,
            source_port="out",
            target_block_id=blur.block_id,
            target_port="in",
            kind="image",
        )
    )
    state = _DagBuilderState(root=scope)

    manifest = pc.compute_scope("sess1", state, [], None, None, None)

    assert manifest["error"] is None, manifest["error"]
    node = manifest["nodes"][blur.block_id]
    assert node["store"].endswith(".ome.zarr")
    assert "gray" in node["layers"]
    assert seen, "compute_scope did not route through _describe_store_node"


def test_class_dispatch_uses_image_class_not_h5py(tmp_path: Path) -> None:
    from phenotypic import GridImage
    from phenotypic.sdk_ import load_image_from_store

    store = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12).save2zarr(
        tmp_path / "g.ome.zarr"
    )
    assert type(load_image_from_store(store)).__name__ == "GridImage"
