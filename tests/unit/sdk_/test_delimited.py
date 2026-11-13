"""The shared delimiter rule and TSV writer (spec B1, B4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.sdk_ import (
    CSV_SUFFIX,
    INPUT_TABLE_SUFFIXES,
    TSV_SUFFIX,
    delimiter_for,
    read_delimited,
    write_tsv,
)


def make_frame(lib: str, data: dict):
    """Build a polars or pandas frame from ``data``."""
    if lib == "polars":
        import polars as pl

        return pl.DataFrame(data)
    import pandas as pd

    return pd.DataFrame(data)


def test_suffix_constants():
    assert (TSV_SUFFIX, CSV_SUFFIX) == (".tsv", ".csv")
    assert INPUT_TABLE_SUFFIXES == frozenset({".csv", ".tsv"})


def test_delimiter_for():
    assert delimiter_for(Path("m.tsv")) == "\t"
    assert delimiter_for(Path("m.CSV")) == ","
    with pytest.raises(ValueError, match=r"\.csv or \.tsv"):
        delimiter_for(Path("m.xlsx"))
    with pytest.raises(ValueError, match=r"\.csv or \.tsv"):
        delimiter_for(Path("metadata"))


@pytest.mark.parametrize("lib", ["polars", "pandas"])
def test_write_tsv_quotes_tabs_and_newlines(tmp_path, lib):
    frame = make_frame(lib, {"a": ["x\ty", "line1\nline2", 'q"q'], "b": [1, 2, 3]})
    out = tmp_path / "t.tsv"
    write_tsv(frame, out)
    back = read_delimited(out)
    assert back["a"].tolist() == ["x\ty", "line1\nline2", 'q"q']
    assert back["b"].tolist() == [1, 2, 3]
    assert out.read_bytes().splitlines()[0] == b"a\tb"


@pytest.mark.parametrize("lib", ["polars", "pandas"])
def test_write_tsv_bytes_are_utf8_lf_and_one_header(tmp_path, lib):
    out = tmp_path / "t.tsv"
    write_tsv(make_frame(lib, {"strain": ["Δura3", "wt"], "n": [1, 2]}), out)
    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.decode("utf-8") == "strain\tn\nΔura3\t1\nwt\t2\n"


def test_write_tsv_leaves_no_temp_and_creates_parent(tmp_path):
    out = tmp_path / "nested" / "t.tsv"
    write_tsv(make_frame("polars", {"a": [1]}), out)
    assert sorted(p.name for p in out.parent.iterdir()) == ["t.tsv"]


def test_write_tsv_failure_keeps_the_old_file(tmp_path):
    out = tmp_path / "t.tsv"
    out.write_bytes(b"old\n")
    with pytest.raises(TypeError):
        write_tsv(object(), out)  # type: ignore[arg-type]
    assert out.read_bytes() == b"old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.tsv"]


def test_read_delimited_reads_csv_by_suffix(tmp_path):
    src = tmp_path / "meta.csv"
    src.write_text("a,b\n1,x\n", encoding="utf-8")
    assert read_delimited(src).to_dict("list") == {"a": [1], "b": ["x"]}


def test_read_delimited_refuses_an_explicit_separator(tmp_path):
    src = tmp_path / "meta.csv"
    src.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="sep="):
        read_delimited(src, sep=";")
