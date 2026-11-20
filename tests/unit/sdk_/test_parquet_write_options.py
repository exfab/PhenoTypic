"""Parquet writer metadata is pinned (#281 spec C11).

Deliverable tables are written by polars with ``PARQUET_WRITE_OPTIONS``. Their
bytes must not depend on how the frame was assembled (a fan-out merge leaves
one chunk per shard) or carry a writer version, host, path or time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from phenotypic.sdk_ import PARQUET_WRITE_OPTIONS

ROWS = 300_000  # more than two pinned row groups


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "Size_Area": rng.random(ROWS),
        "Object_Label": rng.integers(0, 100, ROWS),
        "Metadata_ImageName": [f"plate_{i % 97}" for i in range(ROWS)],
    })


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_row_group_size_is_pinned() -> None:
    assert PARQUET_WRITE_OPTIONS["row_group_size"] == 122_880


def test_chunk_layout_does_not_reach_the_bytes(tmp_path: Path) -> None:
    whole = _frame()
    shards = pl.concat(
        [whole.slice(start, 7_001) for start in range(0, ROWS, 7_001)], rechunk=False
    )
    assert shards.n_chunks() > 1
    a, b = tmp_path / "whole.parquet", tmp_path / "shards.parquet"
    whole.write_parquet(a, **PARQUET_WRITE_OPTIONS)
    shards.write_parquet(b, **PARQUET_WRITE_OPTIONS)
    assert _digest(a) == _digest(b)
    metadata = pq.ParquetFile(a).metadata
    sizes = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    assert sizes == [122_880, 122_880, ROWS - 2 * 122_880]


def test_polars_writes_no_version_host_or_time(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    _frame().head(10).write_parquet(path, **PARQUET_WRITE_OPTIONS)
    metadata = pq.ParquetFile(path).metadata
    assert metadata.created_by == "Polars"
    assert set(metadata.metadata or {}) == {b"ARROW:schema"}


def test_pyarrow_and_pandas_accept_the_options(tmp_path: Path) -> None:
    """The store-table writer (pyarrow) and ``atomic_write_parquet`` (pandas) share them."""
    small = _frame().head(10)
    pq.write_table(small.to_arrow(), tmp_path / "a.parquet", **PARQUET_WRITE_OPTIONS)
    small.to_pandas().to_parquet(tmp_path / "p.parquet", index=False, **PARQUET_WRITE_OPTIONS)
    plain = tmp_path / "plain.parquet"
    pq.write_table(
        small.to_arrow(), plain, compression="zstd", compression_level=3
    )
    # One row group either way, so the pin leaves a small store table's bytes as they were.
    assert _digest(tmp_path / "a.parquet") == _digest(plain)
    assert pa.parquet.read_table(tmp_path / "p.parquet").num_rows == 10
    assert pd.read_parquet(tmp_path / "p.parquet").shape == (10, 3)
