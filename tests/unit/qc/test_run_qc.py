"""Unit tests for :func:`phenotypic.qc._runner.run_qc` artifact schema/content.

Verifies the compact ``qc/`` artifact:

* ``qc_summary.parquet`` schema + the ``summary()``→artifact column mapping
  (``qc_worst_metric``→``metric`` etc.) + group-level ``flag`` + worst-first
  ``rank`` (NaN-last), one row per ``(instance_id, group)``;
* ``qc_members.parquet`` schema (no duplicate ``Metadata_ImageFile`` column
  when it is also a ``groupby`` column);
* ``qc_config.json`` snapshot of the entries that produced the artifact;
* disabled entries are excluded;
* purity: ``run_qc`` never writes ``review_state.json`` or
  ``measurements.parquet``;
* tolerance: an entry that fails to instantiate is skipped, not fatal;
* the empty case still writes schema-correct (empty) parquets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount, ReplicateAgreement
from phenotypic.tools_._qc_recipe import QcRecipeEntry
from phenotypic.tools_._qc_recipe._runner import run_qc
from phenotypic.tools_ import measurements_parquet_path


def _measurements() -> pd.DataFrame:
    """Two plates: p1 tight replicates (pass), p2 wildly scattered (fail)."""
    rows = []
    for plate, areas in [
        ("p1.png", [100, 101, 102, 100, 101, 102]),
        ("p2.png", [50, 500, 80, 300, 90, 400]),
    ]:
        for i, area in enumerate(areas, start=1):
            rows.append({
                "Metadata_ImageFile": plate,
                "Object_Label": i,
                "Size_Area": float(area),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def layout_csv(tmp_path: Path) -> Path:
    """Layout where p1 expects 6 wells and p2 expects 8 (count mismatch)."""
    md = pd.DataFrame({
        "Metadata_ImageFile": ["p1.png"] * 6 + ["p2.png"] * 8,
        "Object_Label": list(range(1, 7)) + list(range(1, 9)),
    })
    path = tmp_path / "layout.csv"
    md.to_csv(path, index=False)
    return path


def _pipeline(layout_csv: Path) -> ImagePipeline:
    return ImagePipeline(qc=[
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
    ])


class TestSummaryArtifact:
    """``qc_summary.parquet`` schema, mapping, and ranking."""

    def test_summary_schema(self, tmp_path: Path, layout_csv: Path) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")

        assert list(summ.columns) == [
            "instance_id", "class", "Metadata_ImageFile",
            "metric", "status", "flag", "n_members", "n_flagged", "rank",
        ]

    def test_one_row_per_instance_and_group(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")

        # 2 enabled instances x 2 plates = 4 rows.
        assert len(summ) == 4
        assert set(summ["instance_id"]) == {"qc-SE-111", "qc-Count-222"}

    def test_disabled_entry_excluded(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")
        assert "qc-SE-disabled" not in set(summ["instance_id"])

    def test_rank_is_worst_first_within_instance(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")

        for _, grp in summ.groupby("instance_id"):
            worst = grp.loc[grp["rank"].idxmin()]
            best = grp.loc[grp["rank"].idxmax()]
            # The worst-ranked (rank 0) row is the failing plate p2.
            assert worst["Metadata_ImageFile"] == "p2.png"
            assert worst["status"] == "fail"
            assert best["Metadata_ImageFile"] == "p1.png"

    def test_group_flag_true_when_members_flagged(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")

        fails = summ[summ["status"] == "fail"]
        assert (fails["flag"]).all()
        assert (fails["n_flagged"] > 0).all()


class TestMembersArtifact:
    """``qc_members.parquet`` schema + no duplicate group-key column."""

    def test_members_schema_no_duplicate_imagefile_column(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        mem = pd.read_parquet(tmp_path / "qc" / "qc_members.parquet")

        # Metadata_ImageFile is a groupby column AND the curation key — it
        # must appear exactly once.
        assert list(mem.columns).count("Metadata_ImageFile") == 1
        assert list(mem.columns) == [
            "instance_id", "Metadata_ImageFile", "Object_Label", "member_value",
        ]

    def test_member_value_carries_on_column(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        mem = pd.read_parquet(tmp_path / "qc" / "qc_members.parquet")

        se_members = mem[mem["instance_id"] == "qc-SE-111"]
        # 12 measurement rows -> 12 SE members; member_value is Size_Area.
        assert len(se_members) == 12
        assert set(se_members["member_value"]) == set(
            _measurements()["Size_Area"]
        )

    def test_extra_groupby_column_is_spliced(self, tmp_path: Path) -> None:
        # When groupby has a non-curation-key column it must be a real
        # column in the members frame.
        meas = _measurements()
        meas["Strain"] = "A"
        pipe = ImagePipeline(qc=[
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": ["Strain", "Metadata_ImageFile"],
                },
                instance_id="qc-SE-strain",
                enabled=True,
            )
        ])
        run_qc(meas, pipe, tmp_path)
        mem = pd.read_parquet(tmp_path / "qc" / "qc_members.parquet")

        assert list(mem.columns) == [
            "instance_id", "Strain",
            "Metadata_ImageFile", "Object_Label", "member_value",
        ]
        assert (mem["Strain"] == "A").all()


class TestConfigSnapshot:
    """``qc_config.json`` records the entries that produced the artifact."""

    def test_config_lists_enabled_entries(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        import json

        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        cfg = json.loads((tmp_path / "qc" / "qc_config.json").read_text())

        ids = [e["instance_id"] for e in cfg["qc"]]
        assert ids == ["qc-SE-111", "qc-Count-222"]


class TestPurity:
    """``run_qc`` writes only the qc/ artifact — never review/mirror state."""

    def test_does_not_write_review_state(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        run_qc(_measurements(), _pipeline(layout_csv), tmp_path)
        assert not (tmp_path / "qc" / "review_state.json").exists()

    def test_preserves_existing_review_state(
        self, tmp_path: Path, layout_csv: Path
    ) -> None:
        (tmp_path / "qc").mkdir()
        review = tmp_path / "qc" / "review_state.json"
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
    """A bad entry is skipped, not fatal; empty/no-op cases are safe."""

    def test_unbuildable_check_is_skipped(self, tmp_path: Path) -> None:
        # A Count check whose metadata path does not exist fails to build;
        # it must be skipped while the good SE check still produces output.
        pipe = ImagePipeline(qc=[
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={"on": "Size_Area", "groupby": ["Metadata_ImageFile"]},
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
        ])
        run_qc(_measurements(), pipe, tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")

        assert set(summ["instance_id"]) == {"qc-SE-ok"}

    def test_no_qc_entries_is_noop(self, tmp_path: Path) -> None:
        run_qc(_measurements(), ImagePipeline(), tmp_path)
        assert not (tmp_path / "qc").exists()

    def test_all_entries_fail_writes_empty_schema(self, tmp_path: Path) -> None:
        pipe = ImagePipeline(qc=[
            QcRecipeEntry(
                cls=ExpectedVsDetectedCount,
                params={
                    "metadata": "/nonexistent/layout.csv",
                    "groupby": ["Metadata_ImageFile"],
                },
                instance_id="qc-Count-broken",
                enabled=True,
            )
        ])
        run_qc(_measurements(), pipe, tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")

        assert summ.empty
        assert "instance_id" in summ.columns


class TestRankNaNLast:
    """NaN metrics sort last in the worst-first rank."""

    def test_nan_metric_group_ranks_after_finite(self, tmp_path: Path) -> None:
        # An under-powered group (single replicate) yields NaN rel-SE; it must
        # rank after a real failing group.
        rows = [
            # fail group: scattered
            {"Metadata_ImageFile": "bad.png", "Object_Label": 1, "Size_Area": 10.0},
            {"Metadata_ImageFile": "bad.png", "Object_Label": 2, "Size_Area": 1000.0},
            # under-powered: single member -> NaN metric
            {"Metadata_ImageFile": "thin.png", "Object_Label": 1, "Size_Area": 100.0},
        ]
        pipe = ImagePipeline(qc=[
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={"on": "Size_Area", "groupby": ["Metadata_ImageFile"]},
                instance_id="qc-SE-nan",
                enabled=True,
            )
        ])
        run_qc(pd.DataFrame(rows), pipe, tmp_path)
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")

        nan_row = summ[summ["Metadata_ImageFile"] == "thin.png"].iloc[0]
        bad_row = summ[summ["Metadata_ImageFile"] == "bad.png"].iloc[0]
        assert np.isnan(nan_row["metric"])
        # NaN-metric group ranks after (higher rank number than) the finite one.
        assert nan_row["rank"] > bad_row["rank"]
