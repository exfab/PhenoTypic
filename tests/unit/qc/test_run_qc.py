"""Unit tests for :func:`phenotypic.sdk_._qc_recipe._runner.run_qc`.

``run_qc`` is the atomic full-rebuild writer of ``deliverables/qc/qc.duckdb``.
These tests verify, against the rebuilt database:

* the ``qc_modules`` catalog: one row per enabled instance, in recipe order,
  with the column roles and a real (non-empty) ``params`` snapshot;
* the per-module data table (``QualityCheck.to_table()``) and the per-module
  ``<table>__summary`` worklist (worst-first ``rank``, NaN-last);
* disabled entries are excluded;
* purity: ``run_qc`` never writes ``review_state.json`` or
  ``measurements.parquet``;
* tolerance: an entry that fails to build is skipped, not fatal;
* no-op cases (no entries / all disabled / all-fail) write no database, so a
  stale prior artifact is never left behind half-written.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount, ReplicateAgreement
from phenotypic.sdk_ import measurements_parquet_path, qc_duckdb_path
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc


def _measurements() -> pd.DataFrame:
    """Two plates: p1 tight replicates (pass), p2 wildly scattered (fail)."""
    rows = []
    for plate, areas in [
        ("p1.png", [100, 101, 102, 100, 101, 102]),
        ("p2.png", [50, 500, 80, 300, 90, 400]),
    ]:
        for i, area in enumerate(areas, start=1):
            rows.append(
                {
                    "Metadata_ImageFile": plate,
                    "Object_Label": i,
                    "Size_Area": float(area),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def layout_csv(tmp_path: Path) -> Path:
    """Layout where p1 expects 6 wells and p2 expects 8 (count mismatch)."""
    md = pd.DataFrame(
        {
            "Metadata_ImageFile": ["p1.png"] * 6 + ["p2.png"] * 8,
            "Object_Label": list(range(1, 7)) + list(range(1, 9)),
        }
    )
    path = tmp_path / "layout.csv"
    md.to_csv(path, index=False)
    return path


def _pipeline(layout_csv: Path) -> ImagePipeline:
    return ImagePipeline(
        qc=[
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={"on": "Size_Area", "groupby": ["Metadata_ImageFile"]},
                instance_id="qc-SE-111",
                enabled=True,
            ),
            QcRecipeEntry(
                cls=ExpectedVsDetectedCount,
                params={
                    "metadata": str(layout_csv),
                    "groupby": ["Metadata_ImageFile"],
                },
                instance_id="qc-Count-222",
                enabled=True,
            ),
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={"on": "Size_Area", "groupby": ["Metadata_ImageFile"]},
                instance_id="qc-SE-disabled",
                enabled=False,
            ),
        ]
    )


def _connect(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    db = qc_duckdb_path(tmp_path)
    assert db.is_file()
    return duckdb.connect(str(db), read_only=True)


def _seed_stale_qc_db(tmp_path: Path) -> Path:
    db = qc_duckdb_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"stale qc database from an earlier run")
    return db


class TestCatalog:
    """``qc_modules`` catalog rows, ordering, roles, and params snapshot."""

    def test_one_catalog_row_per_enabled_instance_in_order(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        con = _connect(tmp_path)
        try:
            ids = [
                r[0]
                for r in con.execute(
                    "SELECT instance_id FROM qc_modules ORDER BY ordinal"
                ).fetchall()
            ]
        finally:
            con.close()
        assert ids == ["qc-SE-111", "qc-Count-222"]

    def test_disabled_entry_excluded(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        con = _connect(tmp_path)
        try:
            ids = {
                r[0]
                for r in con.execute(
                    "SELECT instance_id FROM qc_modules"
                ).fetchall()
            }
        finally:
            con.close()
        assert "qc-SE-disabled" not in ids

    def test_catalog_records_column_roles(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        con = _connect(tmp_path)
        try:
            row = con.execute(
                "SELECT class, name, groupby_cols, metric_col, status_col, "
                "supports_object_curation FROM qc_modules "
                "WHERE instance_id = 'qc-SE-111'"
            ).fetchone()
        finally:
            con.close()
        cls_name, name, groupby_cols, metric_col, status_col, curation = row
        assert cls_name == "ReplicateAgreement"
        assert json.loads(groupby_cols) == ["Metadata_ImageFile"]
        assert metric_col == ReplicateAgreement.metric_col()
        assert status_col == ReplicateAgreement.status_col()
        assert bool(curation) is True

    def test_catalog_params_snapshot_is_real(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        # The params column must carry the actual entry params, never ``{}``.
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        con = _connect(tmp_path)
        try:
            params_json = con.execute(
                "SELECT params FROM qc_modules WHERE instance_id = 'qc-SE-111'"
            ).fetchone()[0]
        finally:
            con.close()
        params = json.loads(params_json)
        assert params  # non-empty
        assert params.get("on") == "Size_Area"
        assert params.get("groupby") == ["Metadata_ImageFile"]


class TestPerModuleTables:
    """Each module's data table + ``__summary`` worklist."""

    def test_data_table_is_self_describing(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        con = _connect(tmp_path)
        try:
            tname = con.execute(
                "SELECT table_name FROM qc_modules "
                "WHERE instance_id = 'qc-SE-111'"
            ).fetchone()[0]
            cols = [
                c[0] for c in con.execute(f'DESCRIBE "{tname}"').fetchall()
            ]
            n_rows = con.execute(f'SELECT count(*) FROM "{tname}"').fetchone()[
                0
            ]
        finally:
            con.close()
        # Member-level: one row per measured object (12 here), with the
        # check's own QC_<name>_* columns.
        assert n_rows == 12
        assert any(c.startswith("QC_") and c.endswith("_Metric") for c in cols)
        assert "Metadata_ImageFile" in cols

    def test_summary_schema_and_worst_first_rank(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        con = _connect(tmp_path)
        try:
            stname = con.execute(
                "SELECT summary_table FROM qc_modules "
                "WHERE instance_id = 'qc-SE-111'"
            ).fetchone()[0]
            scols = [
                c[0] for c in con.execute(f'DESCRIBE "{stname}"').fetchall()
            ]
            summ = con.execute(f'SELECT * FROM "{stname}" ORDER BY rank').pl()
        finally:
            con.close()

        assert {
            "metric",
            "status",
            "flag",
            "n_members",
            "n_flagged",
            "rank",
        } <= set(scols)
        # Worst-first: rank 0 is the failing plate p2.
        worst = summ.row(0, named=True)
        assert worst["Metadata_ImageFile"] == "p2.png"
        assert worst["status"] == "fail"
        assert worst["flag"] is True


class TestPurity:
    """``run_qc`` writes only the QC database — never review/mirror state."""

    def test_does_not_write_review_state(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        assert not (
            tmp_path / "deliverables" / "qc" / "review_state.json"
        ).exists()

    def test_preserves_existing_review_state(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        (tmp_path / "deliverables" / "qc").mkdir(parents=True)
        review = tmp_path / "deliverables" / "qc" / "review_state.json"
        review.write_text('{"qc-SE-111": {"reviewed": ["p1.png"]}}')

        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)

        # run_qc must leave a pre-existing review_state untouched.
        assert "p1.png" in review.read_text()

    def test_does_not_write_measurements_parquet(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        assert not measurements_parquet_path(tmp_path).exists()


class TestTolerance:
    """A bad entry is skipped, not fatal; empty/no-op cases write nothing."""

    def test_unbuildable_check_is_skipped(self, tmp_path: Path) -> None:
        # A Count check whose metadata path does not exist fails to build;
        # it must be skipped while the good SE check still produces output.
        pipe = ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=ReplicateAgreement,
                    params={
                        "on": "Size_Area",
                        "groupby": ["Metadata_ImageFile"],
                    },
                    instance_id="qc-SE-ok",
                    enabled=True,
                ),
                QcRecipeEntry(
                    cls=ExpectedVsDetectedCount,
                    params={
                        "metadata": "/nonexistent/layout.csv",
                        "groupby": ["Metadata_ImageFile"],
                    },
                    instance_id="qc-Count-broken",
                    enabled=True,
                ),
            ]
        )
        run_qc(_measurements(), pipe, tmp_path)
        con = _connect(tmp_path)
        try:
            ids = {
                r[0]
                for r in con.execute(
                    "SELECT instance_id FROM qc_modules"
                ).fetchall()
            }
        finally:
            con.close()
        assert ids == {"qc-SE-ok"}

    def test_no_qc_entries_is_noop(self, tmp_path: Path) -> None:
        run_qc(_measurements(), ImagePipeline(), tmp_path)
        assert not qc_duckdb_path(tmp_path).exists()

    def test_no_qc_entries_removes_stale_database(
        self, tmp_path: Path
    ) -> None:
        stale = _seed_stale_qc_db(tmp_path)

        run_qc(_measurements(), ImagePipeline(), tmp_path)

        assert not stale.exists()

    def test_all_entries_disabled_is_noop(self, tmp_path: Path) -> None:
        pipe = ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=ReplicateAgreement,
                    params={
                        "on": "Size_Area",
                        "groupby": ["Metadata_ImageFile"],
                    },
                    instance_id="qc-SE-off",
                    enabled=False,
                )
            ]
        )
        run_qc(_measurements(), pipe, tmp_path)
        assert not qc_duckdb_path(tmp_path).exists()

    def test_all_entries_disabled_removes_stale_database(
        self, tmp_path: Path
    ) -> None:
        stale = _seed_stale_qc_db(tmp_path)
        pipe = ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=ReplicateAgreement,
                    params={
                        "on": "Size_Area",
                        "groupby": ["Metadata_ImageFile"],
                    },
                    instance_id="qc-SE-off",
                    enabled=False,
                )
            ]
        )

        run_qc(_measurements(), pipe, tmp_path)

        assert not stale.exists()

    def test_all_entries_fail_writes_no_database(self, tmp_path: Path) -> None:
        pipe = ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=ExpectedVsDetectedCount,
                    params={
                        "metadata": "/nonexistent/layout.csv",
                        "groupby": ["Metadata_ImageFile"],
                    },
                    instance_id="qc-Count-broken",
                    enabled=True,
                )
            ]
        )
        run_qc(_measurements(), pipe, tmp_path)
        assert not qc_duckdb_path(tmp_path).exists()

    def test_all_entries_fail_removes_stale_database(
        self, tmp_path: Path
    ) -> None:
        stale = _seed_stale_qc_db(tmp_path)
        pipe = ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=ExpectedVsDetectedCount,
                    params={
                        "metadata": "/nonexistent/layout.csv",
                        "groupby": ["Metadata_ImageFile"],
                    },
                    instance_id="qc-Count-broken",
                    enabled=True,
                )
            ]
        )

        run_qc(_measurements(), pipe, tmp_path)

        assert not stale.exists()


class TestRankNaNLast:
    """NaN metrics sort last in the worst-first rank."""

    def test_nan_metric_group_ranks_after_finite(self, tmp_path: Path) -> None:
        # An under-powered group (single replicate) yields NaN rel-SE; it must
        # rank after a real failing group.
        rows = [
            # fail group: scattered
            {
                "Metadata_ImageFile": "bad.png",
                "Object_Label": 1,
                "Size_Area": 10.0,
            },
            {
                "Metadata_ImageFile": "bad.png",
                "Object_Label": 2,
                "Size_Area": 1000.0,
            },
            # under-powered: single member -> NaN metric
            {
                "Metadata_ImageFile": "thin.png",
                "Object_Label": 1,
                "Size_Area": 100.0,
            },
        ]
        pipe = ImagePipeline(
            qc=[
                QcRecipeEntry(
                    cls=ReplicateAgreement,
                    params={
                        "on": "Size_Area",
                        "groupby": ["Metadata_ImageFile"],
                    },
                    instance_id="qc-SE-nan",
                    enabled=True,
                )
            ]
        )
        run_qc(pd.DataFrame(rows), pipe, tmp_path)
        con = _connect(tmp_path)
        try:
            stname = con.execute(
                "SELECT summary_table FROM qc_modules "
                "WHERE instance_id = 'qc-SE-nan'"
            ).fetchone()[0]
            summ = con.execute(f'SELECT * FROM "{stname}"').pl().to_pandas()
        finally:
            con.close()

        nan_row = summ[summ["Metadata_ImageFile"] == "thin.png"].iloc[0]
        bad_row = summ[summ["Metadata_ImageFile"] == "bad.png"].iloc[0]
        assert np.isnan(nan_row["metric"])
        # NaN-metric group ranks after (higher rank number than) the finite one.
        assert nan_row["rank"] > bad_row["rank"]
