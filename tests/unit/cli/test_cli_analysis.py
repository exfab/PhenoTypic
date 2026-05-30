"""Unit tests for the CLI analysis emission helpers introduced for the
analysis GUI: ``_persist_pipeline_to_output_dir`` writes the canonical
``pipeline.json`` next to the master measurements, and
``_emit_analysis_outputs`` runs ``pipeline.analyze`` and writes
``analysis.{csv,parquet}`` whenever the pipeline has a ``model``
configured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import LogGrowthModel, TukeyOutlierRemover
from phenotypic.tools_ import (
    analysis_csv_path,
    analysis_parquet_path,
    pipeline_json_path,
)
from phenotypic._cli._cli_output_manager import (
    _emit_analysis_outputs,
    _load_pipeline_from_output_dir,
    _persist_pipeline_to_output_dir,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CLI output manager uses POSIX atomic writes",
)


def _synthetic_growth_master() -> pl.DataFrame:
    """Build a tiny logistic-growth master covering enough timepoints for
    :class:`LogGrowthModel` to converge on each strain group.
    """
    import math

    rows: list[dict[str, object]] = []
    for strain in ("CBS-A", "CBS-B"):
        for t in (0, 6, 12, 24, 36, 48):
            for rep in range(3):
                # Logistic growth K=1000, N0=100, r=0.15 with mild offset.
                n = 100 + 800 / (1 + (1000 - 100) / 100 * math.exp(-0.15 * t))
                rows.append({
                    "Metadata_Strain": strain,
                    "Metadata_Time": float(t),
                    "Object_Label": rep,
                    "Shape_Area": float(n + (rep - 1) * 5),
                })
    return pl.DataFrame(rows)


class TestPersistPipelineJson:
    """``_persist_pipeline_to_output_dir`` writes a canonical pipeline.json."""

    def test_writes_pipeline_json_to_output_dir(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(name="canonical-pipeline")
        target = _persist_pipeline_to_output_dir(tmp_path, pipeline)
        assert target == pipeline_json_path(tmp_path)
        assert target.exists()

    def test_round_trip_via_load_pipeline_from_output_dir(
        self, tmp_path: Path
    ) -> None:
        model = LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            time_label="Metadata_Time",
            n_jobs=1,
        )
        pipeline = ImagePipeline(model=model, name="canonical-pipeline")
        _persist_pipeline_to_output_dir(tmp_path, pipeline)

        loaded = _load_pipeline_from_output_dir(tmp_path)
        assert loaded is not None
        assert isinstance(loaded.get_model(), LogGrowthModel)


class TestEmitAnalysisOutputs:
    """``_emit_analysis_outputs`` is gated on ``pipeline.get_model()``."""

    def test_no_model_is_no_op(self, tmp_path: Path) -> None:
        master = _synthetic_growth_master()
        result = _emit_analysis_outputs(tmp_path, master, ImagePipeline())
        assert result is None
        assert not analysis_csv_path(tmp_path).exists()
        assert not analysis_parquet_path(tmp_path).exists()

    def test_writes_csv_and_parquet_when_model_configured(
        self, tmp_path: Path
    ) -> None:
        master = _synthetic_growth_master()
        pipeline = ImagePipeline(
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        result = _emit_analysis_outputs(tmp_path, master, pipeline)
        assert result is not None
        path, n_rows = result
        assert path == analysis_parquet_path(tmp_path)
        assert n_rows > 0
        assert analysis_csv_path(tmp_path).exists()
        assert analysis_parquet_path(tmp_path).exists()
        assert pl.read_parquet(path).height == n_rows

    def test_filter_chain_runs_before_model(self, tmp_path: Path) -> None:
        master = _synthetic_growth_master()
        pipeline = ImagePipeline(
            filters=[
                TukeyOutlierRemover(
                    on="Shape_Area",
                    groupby=["Metadata_Strain"],
                    k=3.0,
                ),
            ],
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        result = _emit_analysis_outputs(tmp_path, master, pipeline)
        assert result is not None
        loaded = pl.read_parquet(analysis_parquet_path(tmp_path))
        # One fit per strain group.
        assert loaded.height <= master["Metadata_Strain"].n_unique()
        assert loaded.height > 0

    def test_analysis_failure_is_non_fatal(self, tmp_path: Path) -> None:
        # Master frame missing the column the model needs.
        master = pl.DataFrame({"Metadata_Strain": ["A"], "Object_Label": [1]})
        pipeline = ImagePipeline(
            model=LogGrowthModel(
                on="Shape_Area",  # not present in the frame
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        result = _emit_analysis_outputs(tmp_path, master, pipeline)
        assert result is None
        assert not analysis_csv_path(tmp_path).exists()
        assert not analysis_parquet_path(tmp_path).exists()


class TestLoadPipelinePrefersCanonical:
    """``_load_pipeline_from_output_dir`` prefers ``pipeline.json``."""

    def test_canonical_pipeline_json_wins(self, tmp_path: Path) -> None:
        canonical = ImagePipeline(name="canonical")
        legacy = ImagePipeline(name="legacy")

        # Stage canonical pipeline.json
        _persist_pipeline_to_output_dir(tmp_path, canonical)

        # Stage a legacy state file pointing at a different copy
        legacy_path = tmp_path / "legacy_pipeline.json"
        legacy_path.write_text(legacy.to_json() or "", encoding="utf-8")
        (tmp_path / "processing_state.json").write_text(
            f'{{"pipeline_path": "{legacy_path}"}}', encoding="utf-8"
        )

        loaded = _load_pipeline_from_output_dir(tmp_path)
        assert loaded is not None
        assert loaded.name == "canonical"
