"""Builder node-preview PNGs invalidate on a store republish.

``stage_channel_png`` compared the cached PNG's mtime against the store
DIRECTORY's. That happens to work for a promote -- ``os.replace`` installs a
new directory, so its mtime moves -- but the directory is not a sound
staleness key: its ``st_mtime_ns`` does not move when a nested chunk is
rewritten (verified in
``tests/unit/gui/results_viewer/test_tile_cache_invalidation.py``). The key is
the root ``zarr.json``, which the promote writes last on every publish.
"""

from __future__ import annotations

import os

from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON


def test_preview_png_is_staged_on_first_request(builder_preview) -> None:
    assert builder_preview.png_bytes("block-1", "gray")


def test_preview_png_is_reused_when_nothing_was_republished(
    builder_preview,
) -> None:
    """Invalidation that fires every time is not a cache.

    Asserted on the file's IDENTITY, not its bytes: a staging function that
    re-rendered on every request would produce byte-identical output and
    satisfy a content comparison while doing all the work the cache exists
    to avoid. ``stage_channel_png`` publishes through ``Path.replace``, so a
    rewrite is a new inode.
    """
    builder_preview.png_bytes("block-1", "gray")
    png = builder_preview.scope_dir / "tiles_src" / "block-1__gray.png"
    before = (png.stat().st_ino, png.stat().st_mtime_ns)
    builder_preview.png_bytes("block-1", "gray")
    assert (png.stat().st_ino, png.stat().st_mtime_ns) == before


def test_preview_png_invalidates_on_a_store_republish(builder_preview) -> None:
    first = builder_preview.png_bytes("block-1", "gray")
    builder_preview.rewrite_node_store("block-1", level=5)
    assert builder_preview.png_bytes("block-1", "gray") != first


def test_preview_freshness_is_measured_against_the_root_json(
    builder_preview,
) -> None:
    """Not against the store DIRECTORY.

    Separated by pinning the two apart: the directory is stamped OLDER than
    the cached PNG while the root stays newer. A directory-keyed compare
    calls the PNG fresh and skips; only a root-keyed one regenerates. On a
    promote the two agree, which is why nothing else here can tell them
    apart.
    """
    builder_preview.png_bytes("block-1", "gray")
    store = builder_preview.store_for("block-1")
    root = store / STORE_ROOT_JSON
    png = builder_preview.scope_dir / "tiles_src" / "block-1__gray.png"

    root_mtime = root.stat().st_mtime_ns
    old_dir = root_mtime - 10**9
    os.utime(store, ns=(old_dir, old_dir))
    between = root_mtime - 10**8
    os.utime(png, ns=(between, between))
    assert os.stat(store).st_mtime_ns < png.stat().st_mtime_ns < root_mtime

    builder_preview.png_bytes("block-1", "gray")
    assert png.stat().st_mtime_ns != between


def test_preview_freshness_compares_at_nanosecond_resolution(
    builder_preview,
) -> None:
    """``st_mtime`` is a float of seconds; ``st_mtime_ns`` is exact.

    At present-day epoch magnitudes a float64 second carries roughly
    quarter-microsecond resolution, so two stats that differ by less than
    that compare EQUAL as floats and a republished store reads as fresh.
    """
    builder_preview.png_bytes("block-1", "gray")
    root = builder_preview.store_for("block-1") / STORE_ROOT_JSON
    png = builder_preview.scope_dir / "tiles_src" / "block-1__gray.png"

    root_mtime = root.stat().st_mtime_ns
    just_older = root_mtime - 1
    os.utime(png, ns=(just_older, just_older))
    assert png.stat().st_mtime < root.stat().st_mtime or True  # float ties
    assert png.stat().st_mtime_ns < root.stat().st_mtime_ns

    builder_preview.png_bytes("block-1", "gray")
    assert png.stat().st_mtime_ns != just_older
