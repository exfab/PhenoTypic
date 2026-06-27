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
    assert _safe_table_name("qc-SE-1a2b3c4d") != _safe_table_name(
        "qc-SE-99999999"
    )


def test_qc_temp_db_path_is_unique_per_writer(tmp_path):
    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import _qc_temp_db_path

    db = qc_duckdb_path(tmp_path)
    fixed_tmp = db.with_suffix(db.suffix + ".tmp")

    first = _qc_temp_db_path(db)
    second = _qc_temp_db_path(db)

    assert first != second
    assert first.parent == db.parent
    assert second.parent == db.parent
    assert first != fixed_tmp
    assert second != fixed_tmp
    assert first.name.startswith(f"{db.name}.")
    assert first.name.endswith(".tmp")


def _two_check_pipeline():
    from phenotypic import ImagePipeline
    from phenotypic.analysis.qc import MaxModifiedZScore, RelativeMAD

    pipe = ImagePipeline()
    pipe.set_qc(
        [
            _entry(
                MaxModifiedZScore, {"on": "Size_Area", "groupby": ["Plate"]}
            ),
            _entry(RelativeMAD, {"on": "Size_Area", "groupby": ["Plate"]}),
        ]
    )
    return pipe


def _entry(cls, params):
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry

    name = cls.name
    return QcRecipeEntry(
        cls=cls, params=params, instance_id=f"qc-{name}-00000001", enabled=True
    )


def _frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "Metadata_ImageFile": ["a.png"] * 6,
            "Object_Label": [1, 2, 3, 4, 5, 6],
            "Plate": ["P1"] * 6,
            "Size_Area": [10.0, 11.0, 12.0, 10.5, 11.5, 200.0],
        }
    )


def test_run_qc_writes_per_module_tables_and_catalog(tmp_path):
    import duckdb

    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    # NOTE: instance_ids must be unique per module; fix the second one.
    pipe = _two_check_pipeline()
    qc = pipe.get_qc()
    qc[1] = qc[1].__class__(
        cls=qc[1].cls,
        params=qc[1].params,
        instance_id="qc-MAD-00000002",
        enabled=True,
    )
    pipe.set_qc(qc)

    run_qc(_frame(), pipe, tmp_path)

    db = qc_duckdb_path(tmp_path)
    assert db.is_file()
    con = duckdb.connect(str(db), read_only=True)
    try:
        cat = con.execute(
            "SELECT instance_id, table_name, summary_table, "
            "supports_object_curation FROM qc_modules ORDER BY ordinal"
        ).fetchall()
        assert [r[0] for r in cat] == ["qc-ZMax-00000001", "qc-MAD-00000002"]
        # Each module's data + summary tables exist and the metric column is kept.
        for _iid, tname, stname, _curation in cat:
            cols = [
                c[0] for c in con.execute(f'DESCRIBE "{tname}"').fetchall()
            ]
            assert any(
                c.startswith("QC_") and c.endswith("_Metric") for c in cols
            )
            scols = [
                c[0] for c in con.execute(f'DESCRIBE "{stname}"').fetchall()
            ]
            assert {
                "metric",
                "status",
                "rank",
                "n_members",
                "n_flagged",
            } <= set(scols)
    finally:
        con.close()


def test_run_qc_cleans_staging_tmp_on_success(tmp_path):
    """A successful run_qc leaves the canonical db and no ``.tmp`` staging file.

    The writer builds into ``qc.duckdb.tmp`` then atomically ``os.replace``-s
    it over ``qc.duckdb``; after a clean build the staging file must be gone
    and only the canonical file present.
    """
    from phenotypic import ImagePipeline
    from phenotypic.analysis.qc import MaxModifiedZScore
    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    pipe = ImagePipeline()
    pipe.set_qc(
        [_entry(MaxModifiedZScore, {"on": "Size_Area", "groupby": ["Plate"]})]
    )

    run_qc(_frame(), pipe, tmp_path)

    db = qc_duckdb_path(tmp_path)
    tmp = db.with_suffix(db.suffix + ".tmp")
    assert db.is_file()
    assert not tmp.exists()


def test_run_qc_no_enabled_checks_is_noop(tmp_path):
    from phenotypic import ImagePipeline
    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    run_qc(_frame(), ImagePipeline(), tmp_path)
    assert not qc_duckdb_path(tmp_path).exists()


def test_run_qc_all_disabled_is_noop(tmp_path):
    from phenotypic import ImagePipeline
    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry
    from phenotypic.sdk_._qc_recipe._runner import run_qc
    from phenotypic.analysis.qc import MaxModifiedZScore

    pipe = ImagePipeline()
    pipe.set_qc(
        [
            QcRecipeEntry(
                cls=MaxModifiedZScore,
                params={"on": "Size_Area", "groupby": ["Plate"]},
                instance_id="qc-ZMax-00000001",
                enabled=False,
            )
        ]
    )
    run_qc(_frame(), pipe, tmp_path)
    assert not qc_duckdb_path(tmp_path).exists()
