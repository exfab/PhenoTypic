"""Chunk/shard/codec policy. Divisibility is claim C3 of the validation script."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.sdk_ import ngff_


def test_rgb_chunk_is_one_channel_by_1024_square() -> None:
    assert ngff_.chunk_shape_for((3, 4000, 3000)) == (1, 1024, 1024)


def test_two_d_chunk_is_1024_square() -> None:
    assert ngff_.chunk_shape_for((4000, 3000)) == (1024, 1024)


def test_rgb_shard_spans_the_full_channel_axis() -> None:
    assert ngff_.shard_shape_for((3, 4000, 3000)) == (3, 4096, 4096)


def test_two_d_shard() -> None:
    assert ngff_.shard_shape_for((4000, 3000)) == (4096, 4096)


@pytest.mark.parametrize(
    "shape", [(3, 4000, 3000), (4000, 3000), (3, 2048, 2048), (6000, 4000), (257, 2)]
)
def test_shard_is_an_exact_multiple_of_chunk_in_every_dimension(shape) -> None:
    chunk = ngff_.chunk_shape_for(shape)
    shard = ngff_.shard_shape_for(shape)
    assert len(chunk) == len(shard) == len(shape)
    for c, s in zip(chunk, shard, strict=True):
        assert s % c == 0, (shape, chunk, shard)


def test_small_level_clamps_chunk_and_shard_to_its_own_shape() -> None:
    """A 257x2 pyramid level must not carry a 1024x1024 chunk."""
    assert ngff_.chunk_shape_for((257, 2)) == (257, 2)
    assert ngff_.shard_shape_for((257, 2)) == (257, 2)


def test_create_kwargs_carry_dimension_names_matching_axes() -> None:
    kwargs = ngff_.array_create_kwargs((3, 4000, 3000), np.dtype("uint8"), "rgb")
    assert tuple(kwargs["dimension_names"]) == ("c", "y", "x")
    kwargs2d = ngff_.array_create_kwargs((4000, 3000), np.dtype("float64"), "detect_mat")
    assert tuple(kwargs2d["dimension_names"]) == ("y", "x")


def test_create_kwargs_use_the_dot_chunk_key_separator() -> None:
    """A Windows MAX_PATH measure; must be uniform store-wide."""
    kwargs = ngff_.array_create_kwargs((4000, 3000), np.dtype("uint16"), "objmap")
    encoding = kwargs["chunk_key_encoding"]
    assert encoding["configuration"]["separator"] == "."


def test_create_kwargs_use_zstd() -> None:
    kwargs = ngff_.array_create_kwargs((4000, 3000), np.dtype("uint16"), "gray")
    assert "zstd" in repr(kwargs["compressors"]).lower()


def test_shard_write_buffer_is_bounded_and_documented() -> None:
    """96 MB for rgb uint16, 128 MB for a float64 detect_mat (spec 1.4).

    Asserts the documented BYTE bounds. The earlier form recomputed the shard
    shape and compared it against itself -- neither of the two figures in this
    docstring appeared in an assertion, so the test could not fail for any
    shard-policy change that kept divisibility.
    """
    rgb = int(np.prod(ngff_.shard_shape_for((3, 4000, 3000)))) * 2
    detect = int(np.prod(ngff_.shard_shape_for((4000, 3000)))) * 8
    assert rgb == 96 * 1024**2
    assert detect == 128 * 1024**2


def test_create_kwargs_carry_both_the_chunk_and_the_shard_policy() -> None:
    """Kills the dropped-``shards`` mutant (S4).

    ``shard_shape_for`` was tested only in isolation; nothing checked its value
    reached the kwargs. Without ``shards`` a 4000x3000 rgb plate lands as 132
    files instead of 40 (claim C4 of the committed logic-validation script) --
    a 3.3x inode blowup on Lustre with no other symptom.
    """
    kwargs = ngff_.array_create_kwargs((3, 4000, 3000), np.dtype("uint8"), "rgb")
    assert kwargs["shape"] == (3, 4000, 3000)
    assert kwargs["dtype"] == np.dtype("uint8")
    assert kwargs["chunks"] == (1, 1024, 1024)
    assert kwargs["shards"] == (3, 4096, 4096)


def test_a_plate_sized_array_is_really_created_with_that_chunk_and_shard(
    tmp_path,
) -> None:
    """The policy, observed on a real store rather than in the kwargs mapping.

    Every other real-zarr fixture in this suite is 64x48, where
    ``chunk == shard == extent`` -- so the 1024/4096 policy this module exists
    to configure was never actually created by any test.
    """
    import zarr

    array = zarr.create_array(
        store=str(tmp_path / "rgb" / "0"),
        **ngff_.array_create_kwargs((3, 4000, 3000), np.dtype("uint8"), "rgb"),
    )
    assert array.chunks == (1, 1024, 1024)
    assert array.shards == (3, 4096, 4096)
