"""Tests for the DuckDB-backed QC artifact writer."""


def test_duckdb_importable():
    import duckdb

    con = duckdb.connect(":memory:")
    assert con.execute("SELECT 42").fetchone()[0] == 42
    con.close()
