"""Integration tests for QC compute inside ``finalize_post_master_outputs``.

Exercises the Phase B finalize seam directly (the canonical entry point the
forward CLI, ``--measure``, and ``--recompile`` all funnel through):

* a non-empty ``pipeline.qc`` causes ``finalize`` to write the ``qc/``
  artifact from the post-applied frame;
* ``no_qc=True`` skips QC compute entirely;
* a rerun clears a stale ``qc/review_state.json`` (reset-on-rerun) even
  when ``no_qc=True``;
* QC compute failures never break the authoritative master/mirror outputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ReplicateAgreement
from phenotypic.tools_._qc_recipe import QcRecipeEntry
from phenotypic.tools_ import measurements_parquet_path
from phenotypic._cli._cli_output_manager import finalize_post_master_outputs

from tests._output_layout import write_master


def _master() -> pl.DataFrame:
    """A small clean master with two plates of replicate areas."""
    rows = []
    for plate, areas in [
        ("p1.png", [100, 101, 102]),
        ("p2.png", [50, 500, 90]),
    ]:
        for i, area in enumerate(areas, start=1):
            rows.append({
                "Metadata_ImageFile": plate,
                "Object_Label": i,
                "Size_Area": float(area),
            })
    return pl.from_pandas(pd.DataFrame(rows))


def _qc_pipeline() -> ImagePipeline:
    return ImagePipeline(qc=[QcRecipeEntry(
        cls=ReplicateAgreement,
        params={"on": "Size_Area", "groupby": ["Metadata_ImageFile"]},
        instance_id="qc-SE-fin",
        enabled=True,
    )])


def _write_master_files(output_dir: Path, master: pl.DataFrame) -> None:
    """finalize assumes the master files already exist on disk.

    The master archive lives under ``<output>/deliverables/`` — route
    through the production path-builders via the shared helper so this
    fixture auto-tracks the layout.
    """
    write_master(output_dir, master)


class TestFinalizeWritesQc:
    def test_qc_written_when_pipeline_has_entries(
        self, tmp_path: Path
    ) -> None:
        _write_master_files(tmp_path, _master())
        finalize_post_master_outputs(tmp_path, _master(), _qc_pipeline())

        assert (tmp_path / "qc" / "qc_summary.parquet").exists()
        assert (tmp_path / "qc" / "qc_members.parquet").exists()
        assert (tmp_path / "qc" / "qc_config.json").exists()
        summ = pd.read_parquet(tmp_path / "qc" / "qc_summary.parquet")
        assert set(summ["instance_id"]) == {"qc-SE-fin"}

    def test_no_qc_when_pipeline_has_no_entries(self, tmp_path: Path) -> None:
        _write_master_files(tmp_path, _master())
        finalize_post_master_outputs(tmp_path, _master(), ImagePipeline())
        assert not (tmp_path / "qc").exists()


class TestNoQcFlag:
    def test_no_qc_true_skips_compute(self, tmp_path: Path) -> None:
        _write_master_files(tmp_path, _master())
        finalize_post_master_outputs(
            tmp_path, _master(), _qc_pipeline(), no_qc=True
        )
        assert not (tmp_path / "qc" / "qc_summary.parquet").exists()

    def test_no_qc_true_still_writes_mirror(self, tmp_path: Path) -> None:
        # The authoritative measurements mirror must still be seeded.
        _write_master_files(tmp_path, _master())
        finalize_post_master_outputs(
            tmp_path, _master(), _qc_pipeline(), no_qc=True
        )
        assert measurements_parquet_path(tmp_path).exists()


class TestResetOnRerun:
    def test_rerun_clears_stale_review_state(self, tmp_path: Path) -> None:
        _write_master_files(tmp_path, _master())
        # Simulate prior GUI review progress.
        (tmp_path / "qc").mkdir()
        review = tmp_path / "qc" / "review_state.json"
        review.write_text('{"qc-SE-fin": {"reviewed": ["p1.png"]}}')

        finalize_post_master_outputs(tmp_path, _master(), _qc_pipeline())

        # A fresh CLI run resets review progress.
        assert not review.exists()
        # ...but the freshly-computed qc artifact is present.
        assert (tmp_path / "qc" / "qc_summary.parquet").exists()

    def test_review_state_reset_even_with_no_qc(self, tmp_path: Path) -> None:
        _write_master_files(tmp_path, _master())
        (tmp_path / "qc").mkdir()
        review = tmp_path / "qc" / "review_state.json"
        review.write_text('{"qc-SE-fin": {"reviewed": ["p1.png"]}}')

        finalize_post_master_outputs(
            tmp_path, _master(), _qc_pipeline(), no_qc=True
        )

        # Reset happens regardless of whether QC then recomputes.
        assert not review.exists()


class TestFailureIsolation:
    def test_master_mirror_survives_qc_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_master_files(tmp_path, _master())

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated QC failure")

        monkeypatch.setattr("phenotypic.tools_._qc_recipe._runner.run_qc", _boom)

        # Must not raise — finalize swallows QC failures.
        finalize_post_master_outputs(tmp_path, _master(), _qc_pipeline())

        # The authoritative mirror is still written.
        assert measurements_parquet_path(tmp_path).exists()
