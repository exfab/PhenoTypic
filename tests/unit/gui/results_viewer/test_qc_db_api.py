"""Tests for the catalog-driven DuckDB read API (``review/_db.py``).

Seeds a real ``qc.duckdb`` via :func:`run_qc` against a tmp dir, then reads
it back through the short-lived ``read_only`` connections the GUI Review +
Error tabs use. A tiny fake ``output_root`` exposing ``.layout`` stands in
for the real :class:`OutputRoot`.
"""

from __future__ import annotations

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis.qc import MaxModifiedZScore
from phenotypic.sdk_ import BundleLayout
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc


class _Root:
    def __init__(self, layout):
        self.layout = layout


def _layout(tmp_path):
    """Full-run-style layout rooted at ``tmp_path`` (deliverables under it)."""
    return BundleLayout(
        deliverables_base=tmp_path / "deliverables", output_root=tmp_path
    )


def _seed_db(tmp_path):
    pipe = ImagePipeline()
    pipe.set_qc(
        [
            QcRecipeEntry(
                cls=MaxModifiedZScore,
                params={"on": "Size_Area", "groupby": ["Plate"]},
                instance_id="qc-ZMax-00000001",
                enabled=True,
            )
        ]
    )
    df = pd.DataFrame(
        {
            "Metadata_ImageFile": ["a.png"] * 4,
            "Object_Label": [1, 2, 3, 4],
            "Plate": ["P1"] * 4,
            "Size_Area": [10.0, 11.0, 12.0, 99.0],
        }
    )
    run_qc(df, pipe, tmp_path)
    return _Root(_layout(tmp_path))


def test_list_modules_reads_catalog(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    root = _seed_db(tmp_path)
    mods = _db.list_modules(root)
    assert [m.instance_id for m in mods] == ["qc-ZMax-00000001"]
    assert mods[0].groupby_cols == ["Plate"]
    assert mods[0].supports_object_curation is True
    assert mods[0].metric_col == "QC_ZMax_Metric"
    assert mods[0].cls_name == "MaxModifiedZScore"


def test_module_summary_and_members(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    root = _seed_db(tmp_path)
    summ = _db.module_summary(root, "qc-ZMax-00000001")
    assert "rank" in summ.columns and summ.height == 1
    members = _db.module_members(root, "qc-ZMax-00000001", ("P1",))
    assert members.height == 4


def test_module_members_empty_group_returns_all(tmp_path):
    """An empty group-key tuple applies no filter → the full data table."""
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    root = _seed_db(tmp_path)
    members = _db.module_members(root, "qc-ZMax-00000001", ())
    assert members.height == 4


def test_summary_stats_from_module_summary(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    root = _seed_db(tmp_path)
    stats = _db.summary_stats(_db.module_summary(root, "qc-ZMax-00000001"))
    assert stats["total"] == 1
    assert stats["colonies_removed"] == 0


def test_open_qc_db_missing_returns_none(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    assert _db.open_qc_db(_Root(_layout(tmp_path))) is None


def test_list_modules_missing_db_is_empty(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    assert _db.list_modules(_Root(_layout(tmp_path))) == []
