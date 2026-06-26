"""Tests for the DuckDB-backed QC artifact writer."""


def test_duckdb_importable():
    import duckdb

    con = duckdb.connect(":memory:")
    assert con.execute("SELECT 42").fetchone()[0] == 42
    con.close()


def test_safe_table_name_is_deterministic_and_identifier_safe():
    from phenotypic.sdk_._qc_recipe._runner import _safe_table_name

    name = _safe_table_name("qc-SE-1a2b3c4d")
    assert name == _safe_table_name("qc-SE-1a2b3c4d")  # deterministic
    assert name[0].isalpha()
    assert all(c.isalnum() or c == "_" for c in name)
    # Distinct ids → distinct names.
    assert _safe_table_name("qc-SE-1a2b3c4d") != _safe_table_name("qc-SE-99999999")
