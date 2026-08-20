"""Cache invalidation across the mtime/fingerprint traps.

Three facts govern this module, each verified rather than assumed:

1. ``file_fingerprint`` opens its argument as a file, so it raises
   ``IsADirectoryError`` on a store -- which is why the tile route had to
   change helpers, not just rename a variable.
2. ``paths_fingerprint`` reduces a **directory** to its name plus one sentinel
   byte and does not recurse, so ``paths_fingerprint([store])`` is a constant
   function of the path and would freeze the cache permanently.
3. A re-promote whose metadata did not change writes a **byte-identical** root
   ``zarr.json``. So the content token cannot be its bytes alone: the decoded
   -array LRU would return the previous publish's pixels. It must also carry
   the root's ``st_mtime_ns``, which a promote always moves.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic import Image
from phenotypic.gui.results_viewer import _tile_routes
from phenotypic.sdk_ import file_fingerprint, paths_fingerprint
from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON


@pytest.fixture(scope="module")
def promoted_store(tmp_path_factory) -> Path:
    """One promoted store shared by the read-only pin tests."""
    root = tmp_path_factory.mktemp("promoted")
    return Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(
        root / "p.ome.zarr"
    )


# ---------------------------------------------------------------------------
# Pinned facts (these pass before the port; they are why it is shaped this way)
# ---------------------------------------------------------------------------


def test_file_fingerprint_raises_on_a_store_directory(
    promoted_store: Path,
) -> None:
    """Pins the exact reason the tile route must switch helpers."""
    with pytest.raises(IsADirectoryError):
        file_fingerprint(promoted_store)


def test_paths_fingerprint_of_a_store_directory_is_a_constant(
    promoted_store: Path,
) -> None:
    """The trap the spec's "handles directories" wording invites.

    It does not raise -- it emits one sentinel byte and does not recurse. So
    the fingerprint of a store directory never moves, whatever happens
    inside it, and a cache keyed on it never invalidates again.
    """
    before = paths_fingerprint([promoted_store])
    chunk = next(
        p for p in (promoted_store / "rgb" / "0").rglob("*") if p.is_file()
    )
    chunk.write_bytes(b"completely different bytes")
    assert paths_fingerprint([promoted_store]) == before


def test_paths_fingerprint_keys_on_the_root_json(promoted_store: Path) -> None:
    assert paths_fingerprint(
        [promoted_store / STORE_ROOT_JSON]
    ).startswith("sha256:")


def test_store_directory_mtime_does_not_change_when_a_chunk_is_rewritten(
    tmp_path: Path,
) -> None:
    """The verified fact the whole task exists for.

    Demonstrated by writing a chunk file directly, since nothing in the
    design opens a promoted store for writing any more. The fact still
    governs: a nested chunk rewrite leaves the store directory's own mtime
    untouched, which is why no staleness check may key on the directory.
    """
    store = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(
        tmp_path / "p.ome.zarr"
    )
    before = os.stat(store).st_mtime_ns
    chunk = next(p for p in (store / "rgb" / "0").rglob("*") if p.is_file())
    chunk.write_bytes(chunk.read_bytes())
    assert os.stat(store).st_mtime_ns == before


def test_a_republish_with_unchanged_metadata_is_byte_identical(
    tmp_path: Path,
) -> None:
    """The premise of the content-token fix (OPEN-QUESTIONS B7/P17).

    Pixels are not summarised anywhere in the root, so two publishes of the
    same-shaped image with the same metadata produce the same bytes -- while
    the root's mtime moves, because the promote replaces the file.
    """
    target = tmp_path / "p.ome.zarr"
    store = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(target)
    root = store / STORE_ROOT_JSON
    before_bytes = root.read_bytes()
    before_mtime = root.stat().st_mtime_ns

    Image(arr=np.full((64, 64, 3), 255, dtype=np.uint8)).save2zarr(target)

    assert root.read_bytes() == before_bytes
    assert root.stat().st_mtime_ns != before_mtime


# ---------------------------------------------------------------------------
# The tile route's source PNG
# ---------------------------------------------------------------------------


def test_source_png_is_generated_and_stamped_from_the_root(
    tmp_path: Path,
) -> None:
    """``os.utime`` must copy the ROOT's mtime, not the directory's."""
    store = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(
        tmp_path / "p.ome.zarr"
    )
    png = tmp_path / "out" / "p.png"
    png.parent.mkdir()
    _tile_routes._ensure_store_layer_source_png(store, "rgb", png)
    assert png.is_file()
    assert png.stat().st_mtime_ns == (store / STORE_ROOT_JSON).stat().st_mtime_ns


def test_a_fresh_source_png_is_not_regenerated(tmp_path: Path) -> None:
    store = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(
        tmp_path / "p.ome.zarr"
    )
    png = tmp_path / "out" / "p.png"
    png.parent.mkdir()
    _tile_routes._ensure_store_layer_source_png(store, "rgb", png)
    first = png.stat().st_mtime_ns
    png.write_bytes(b"sentinel: not a PNG at all")
    os.utime(png, ns=(first, first))
    _tile_routes._ensure_store_layer_source_png(store, "rgb", png)
    assert png.read_bytes() == b"sentinel: not a PNG at all"


def test_a_byte_identical_republish_still_refreshes_the_source_png(
    tmp_path: Path,
) -> None:
    """The B7/P17 defect, end to end.

    Two publishes, identical metadata, different pixels. The root
    ``zarr.json`` is byte-identical, so a token that is only its bytes does
    not move -- and ``_load_zarr_level_rgb`` is LRU-cached on
    ``(path, token, layer, level)``, so the "regenerated" PNG is written
    from the PREVIOUS publish's decoded array. The token therefore has to
    carry the root's mtime as well, which a promote always moves.
    """
    target = tmp_path / "p.ome.zarr"
    store = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(target)
    first = tmp_path / "a" / "p.png"
    second = tmp_path / "b" / "p.png"
    first.parent.mkdir()
    second.parent.mkdir()

    _tile_routes._ensure_store_layer_source_png(store, "rgb", first)
    with PILImage.open(first) as handle:
        assert handle.getpixel((0, 0)) == (0, 0, 0)

    root_bytes = (store / STORE_ROOT_JSON).read_bytes()
    Image(arr=np.full((64, 64, 3), 255, dtype=np.uint8)).save2zarr(target)
    assert (store / STORE_ROOT_JSON).read_bytes() == root_bytes, (
        "premise broken: the republish changed the root's bytes, so this "
        "test would pass on a bytes-only token and prove nothing"
    )

    _tile_routes._ensure_store_layer_source_png(store, "rgb", second)
    with PILImage.open(second) as handle:
        assert handle.getpixel((0, 0)) == (255, 255, 255)


def test_the_content_token_moves_on_a_metadata_edit_at_a_fixed_mtime(
    tmp_path: Path,
) -> None:
    """The BYTES half of the token, isolated from the mtime half.

    ``paths_fingerprint([store])`` -- the store DIRECTORY -- reduces to one
    sentinel byte and is a constant function of the path. Paired with the
    mtime that mutation is invisible, because a promote always moves the
    mtime. Pinning the mtime is what separates them.
    """
    import json

    store = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(
        tmp_path / "p.ome.zarr"
    )
    root = store / STORE_ROOT_JSON
    fixed = root.stat().st_mtime_ns
    before = _tile_routes._store_content_token(store)

    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"]["metadata"]["public"][
        "Metadata_Strain"
    ] = "BY4742"
    root.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(root, ns=(fixed, fixed))
    assert root.stat().st_mtime_ns == fixed

    assert _tile_routes._store_content_token(store) != before


def test_the_content_token_moves_on_every_promote(tmp_path: Path) -> None:
    """Directly on the token, so the mtime element cannot be dropped."""
    target = tmp_path / "p.ome.zarr"
    store = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(target)
    before = _tile_routes._store_content_token(store)
    Image(arr=np.full((64, 64, 3), 255, dtype=np.uint8)).save2zarr(target)
    assert _tile_routes._store_content_token(store) != before


# ---------------------------------------------------------------------------
# The served tile, through the real route
# ---------------------------------------------------------------------------


def test_a_promote_stales_the_bound_viewer(live_viewer) -> None:
    """A republish under a bound viewer is a 409, not a silently new tile.

    This is the contract the source token exists to enforce, and it is the
    first thing a republish must do -- serving fresh pixels against a stale
    binding would let curation write against measurements that no longer
    describe the image.
    """
    live_viewer.get_tile("d1", "img001", layer="objmap")
    live_viewer.republish_with_objmap("d1", "img001", value=7)
    assert (
        live_viewer.request_tile("d1", "img001", layer="objmap").status_code
        == 409
    )


def test_served_tile_changes_after_a_promote_and_refresh(live_viewer) -> None:
    """End-to-end: republish, Refresh, and assert the served PIXELS changed.

    Asserted on the source PNG the pyramid was tiled from, not on the
    ``.dzi`` manifest -- the manifest is identical whatever the pixels are,
    so a manifest assertion would pass against a stale tile. This is the
    test the whole content-token fix exists for: the cache dir, the source
    token file, and the source PNG all survive the republish, so every
    layer of the cache has to notice on its own.
    """
    first = live_viewer.get_tile("d1", "img001", layer="objmap")
    live_viewer.republish_with_objmap("d1", "img001", value=7)
    live_viewer.rebind()
    second = live_viewer.get_tile("d1", "img001", layer="objmap")
    assert first != second


def test_served_tile_is_reused_when_nothing_was_republished(
    live_viewer,
) -> None:
    """The other half: invalidation that fires every time is not a cache."""
    first = live_viewer.get_tile("d1", "img001", layer="objmap")
    assert live_viewer.get_tile("d1", "img001", layer="objmap") == first


def test_a_refresh_alone_does_not_change_the_served_tile(live_viewer) -> None:
    """Rebinding without a republish must be a no-op for the pixels.

    Without this, ``test_served_tile_changes_after_a_promote_and_refresh``
    would pass against an implementation that simply regenerates on every
    Refresh, proving nothing about the token.
    """
    first = live_viewer.get_tile("d1", "img001", layer="objmap")
    live_viewer.rebind()
    assert live_viewer.get_tile("d1", "img001", layer="objmap") == first


def test_an_unreadable_store_answers_422_with_its_own_message(
    live_viewer,
) -> None:
    """A ``store_schema_version`` mismatch must explain itself, not 500.

    Rebound after the corruption, so the staleness gate (409) is satisfied
    and the request reaches the store read -- which is the code path under
    test.
    """
    live_viewer.corrupt_schema_version("d1", "img001")
    live_viewer.rebind()
    response = live_viewer.request_tile("d1", "img001", layer="rgb")
    assert response.status_code == 422
    body = response.get_json()["error"]
    assert "store_schema_version" in body
    assert "upgrade" in body
