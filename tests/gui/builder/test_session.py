"""Tests for :mod:`phenotypic.gui.builder._session`.

Covers the bounded LRU/FIFO behavior of :class:`IntermediatesCache` and the
mixed payload-type contract introduced with the pre-baked PNG cache:
:class:`bytes` (ops nodes), :class:`pandas.DataFrame` (measurement / post),
and :class:`PreviewRenderError` (rendering failure marker).
"""

from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.gui.builder._session import (
    IntermediatesCache,
    PreviewRenderError,
    SessionData,
    get_cache,
)


# ---------------------------------------------------------------------------
# Construction / lifecycle
# ---------------------------------------------------------------------------


def test_session_data_default_fields():
    sd = SessionData()
    assert sd.image is None
    assert sd.image_path is None
    assert len(sd.intermediates) == 0


def test_get_cache_returns_singleton():
    a = get_cache()
    b = get_cache()
    assert a is b


# ---------------------------------------------------------------------------
# set_image / get_image
# ---------------------------------------------------------------------------


def test_set_and_get_image_round_trip():
    cache = IntermediatesCache()
    cache.set_image("s1", object(), "/tmp/foo.png")  # type: ignore[arg-type]
    img, path = cache.get_image("s1")
    assert img is not None
    assert path == "/tmp/foo.png"


def test_get_image_unknown_session_returns_none():
    cache = IntermediatesCache()
    assert cache.get_image("nope") == (None, None)


def test_clearing_image_clears_intermediates():
    """Setting image=None must drop derived intermediates (node ids stale)."""
    cache = IntermediatesCache()
    cache.set_image("s1", object(), "/tmp/x.png")  # type: ignore[arg-type]
    cache.set_intermediate("s1", "node-a", b"\x89PNGfake")
    cache.set_image("s1", None, None)
    assert cache.get_intermediate("s1", "node-a") is None


# ---------------------------------------------------------------------------
# Mixed payload contract
# ---------------------------------------------------------------------------


def test_intermediate_accepts_bytes_payload():
    cache = IntermediatesCache()
    cache.set_intermediate("s1", "node-a", b"\x89PNGfake")
    assert cache.get_intermediate("s1", "node-a") == b"\x89PNGfake"


def test_intermediate_accepts_dataframe_payload():
    cache = IntermediatesCache()
    df = pd.DataFrame({"colony": [1, 2], "Size_Area": [100, 200]})
    cache.set_intermediate("s1", "meas-node", df)
    got = cache.get_intermediate("s1", "meas-node")
    assert isinstance(got, pd.DataFrame)
    pd.testing.assert_frame_equal(got, df)


def test_intermediate_accepts_preview_render_error():
    cache = IntermediatesCache()
    err = PreviewRenderError("boom")
    cache.set_intermediate("s1", "broken-node", err)
    got = cache.get_intermediate("s1", "broken-node")
    assert got is err
    assert isinstance(got, PreviewRenderError)
    assert got.message == "boom"


def test_mixed_payload_types_coexist():
    cache = IntermediatesCache()
    cache.set_intermediate("s1", "ops-node", b"\x89PNGfake")
    cache.set_intermediate("s1", "meas-node", pd.DataFrame({"x": [1]}))
    cache.set_intermediate("s1", "broken-node", PreviewRenderError("nope"))
    assert isinstance(cache.get_intermediate("s1", "ops-node"), bytes)
    assert isinstance(
        cache.get_intermediate("s1", "meas-node"), pd.DataFrame
    )
    assert isinstance(
        cache.get_intermediate("s1", "broken-node"), PreviewRenderError
    )


def test_preview_render_error_is_frozen():
    err = PreviewRenderError("immutable")
    with pytest.raises(Exception):
        err.message = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LRU eviction within a session
# ---------------------------------------------------------------------------


def test_lru_eviction_within_session_drops_oldest():
    cache = IntermediatesCache(max_sessions=2, max_per_session=3)
    for i in range(4):
        cache.set_intermediate("s1", f"node-{i}", f"png-{i}".encode())
    assert cache.get_intermediate("s1", "node-0") is None
    for i in range(1, 4):
        assert cache.get_intermediate("s1", f"node-{i}") == f"png-{i}".encode()


def test_lru_access_promotes_to_most_recently_used():
    cache = IntermediatesCache(max_sessions=2, max_per_session=2)
    cache.set_intermediate("s1", "a", b"a")
    cache.set_intermediate("s1", "b", b"b")
    # access 'a' to bump it to MRU
    assert cache.get_intermediate("s1", "a") == b"a"
    # now insert 'c' — 'b' (the LRU) should be evicted, 'a' survives
    cache.set_intermediate("s1", "c", b"c")
    assert cache.get_intermediate("s1", "a") == b"a"
    assert cache.get_intermediate("s1", "b") is None
    assert cache.get_intermediate("s1", "c") == b"c"


def test_set_existing_node_id_moves_to_mru():
    """Re-setting the same node_id should NOT count as a new entry."""
    cache = IntermediatesCache(max_sessions=2, max_per_session=2)
    cache.set_intermediate("s1", "a", b"v1")
    cache.set_intermediate("s1", "b", b"b")
    cache.set_intermediate("s1", "a", b"v2")  # update; should not evict 'b'
    assert cache.get_intermediate("s1", "a") == b"v2"
    assert cache.get_intermediate("s1", "b") == b"b"


# ---------------------------------------------------------------------------
# FIFO eviction across sessions
# ---------------------------------------------------------------------------


def test_fifo_eviction_across_sessions_drops_oldest_session():
    cache = IntermediatesCache(max_sessions=2, max_per_session=4)
    cache.set_intermediate("s1", "n", b"1")
    cache.set_intermediate("s2", "n", b"2")
    cache.set_intermediate("s3", "n", b"3")
    assert cache.get_intermediate("s1", "n") is None
    assert cache.get_intermediate("s2", "n") == b"2"
    assert cache.get_intermediate("s3", "n") == b"3"


def test_clear_drops_session_entirely():
    cache = IntermediatesCache()
    cache.set_intermediate("s1", "a", b"a")
    cache.clear("s1")
    assert cache.get_intermediate("s1", "a") is None
    assert cache.get_image("s1") == (None, None)


# ---------------------------------------------------------------------------
# Snapshot / known keys
# ---------------------------------------------------------------------------


def test_known_intermediate_keys_returns_insertion_order():
    cache = IntermediatesCache()
    cache.set_intermediate("s1", "a", b"a")
    cache.set_intermediate("s1", "b", b"b")
    cache.set_intermediate("s1", "c", b"c")
    assert cache.known_intermediate_keys("s1") == ["a", "b", "c"]


def test_known_intermediate_keys_unknown_session():
    cache = IntermediatesCache()
    assert cache.known_intermediate_keys("nope") == []


# ---------------------------------------------------------------------------
# Atomic preview generations
# ---------------------------------------------------------------------------


def test_preview_generation_is_invisible_until_atomic_publish() -> None:
    cache = IntermediatesCache()
    first = cache.begin_preview_generation("s1", "revision-1")
    first.set_intermediate("s1", "node-a", b"complete-1")
    generation_1 = cache.publish_preview_generation(first)
    key_1 = ("s1", "node-a", "revision-1", generation_1)

    second = cache.begin_preview_generation("s1", "revision-2")
    second.set_intermediate("s1", "node-a", b"partial-2")

    assert cache.preview_descriptor("s1") == ("revision-1", generation_1)
    assert cache.get_preview(key_1) == b"complete-1"
    assert cache.get_preview(("s1", "node-a", "revision-2", generation_1 + 1)) is None

    generation_2 = cache.publish_preview_generation(second)
    assert generation_2 == generation_1 + 1
    assert cache.get_preview(key_1) is None
    assert (
        cache.get_preview(("s1", "node-a", "revision-2", generation_2))
        == b"partial-2"
    )


def test_abandoned_preview_writer_cannot_mix_with_live_generation() -> None:
    cache = IntermediatesCache()
    published = cache.begin_preview_generation("s1", "revision-1")
    published.set_intermediate("s1", "node-a", b"a1")
    published.set_intermediate("s1", "node-b", b"b1")
    generation = cache.publish_preview_generation(published)

    failed = cache.begin_preview_generation("s1", "revision-2")
    failed.set_intermediate("s1", "node-a", b"a2")
    # Simulate a bake failure before node-b is staged: no publish occurs.

    assert cache.preview_descriptor("s1") == ("revision-1", generation)
    assert (
        cache.get_preview(("s1", "node-a", "revision-1", generation)) == b"a1"
    )
    assert (
        cache.get_preview(("s1", "node-b", "revision-1", generation)) == b"b1"
    )
