"""Unit tests for the CLI analysis emission helpers introduced for the
analysis GUI: ``_persist_pipeline_to_output_dir`` writes the canonical
``pipeline.json`` next to the master measurements, and
``_emit_analysis_outputs`` runs ``pipeline.analyze`` and writes
class-named CSV/Parquet artifacts whenever the pipeline has a ``model``
configured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import LogGrowthModel, TukeyOutlierRemover
from phenotypic.sdk_ import deliverables_dir, pipeline_json_path
from phenotypic.plotting import (
    analysis_manifest_path,
    named_analysis_csv_path,
    named_analysis_parquet_path,
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
        base = deliverables_dir(tmp_path)
        assert not named_analysis_csv_path(base, "LogGrowthModel").exists()
        assert not named_analysis_parquet_path(base, "LogGrowthModel").exists()

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
        base = deliverables_dir(tmp_path)
        path = named_analysis_parquet_path(base, "LogGrowthModel")
        assert result.artifacts is not None
        assert result.artifacts.parquet == path
        assert len(result.table) > 0
        assert named_analysis_csv_path(base, "LogGrowthModel").exists()
        assert path.exists()
        assert analysis_manifest_path(base).exists()
        assert pl.read_parquet(path).height == len(result.table)

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
        loaded = pl.read_parquet(
            named_analysis_parquet_path(
                deliverables_dir(tmp_path), "LogGrowthModel"
            )
        )
        # One fit per strain group.
        assert loaded.height <= master["Metadata_Strain"].n_unique()
        assert loaded.height > 0

    def test_deliverables_base_override_writes_into_bundle(
        self, tmp_path: Path
    ) -> None:
        """A ``deliverables_base`` override writes ``analysis.*`` there directly,
        bypassing ``deliverables_dir(output_dir)`` — the standalone-bundle path
        used by the analysis sub-app so ``deliverables/`` is never double-joined.
        """
        master = _synthetic_growth_master()
        pipeline = ImagePipeline(
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        bundle = tmp_path / "my_export"  # renamed standalone deliverables folder
        bundle.mkdir()
        result = _emit_analysis_outputs(
            tmp_path, master, pipeline, deliverables_base=bundle
        )
        assert result is not None
        assert result.artifacts is not None
        path = result.artifacts.parquet
        # Written into the bundle directly, NOT under deliverables_dir(tmp_path).
        assert path == bundle / "LogGrowthModel.parquet"
        assert (bundle / "LogGrowthModel.csv").exists()
        assert (bundle / "LogGrowthModel.parquet").exists()
        assert not named_analysis_parquet_path(
            deliverables_dir(tmp_path), "LogGrowthModel"
        ).exists()

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
        base = deliverables_dir(tmp_path)
        assert not named_analysis_csv_path(base, "LogGrowthModel").exists()
        assert not named_analysis_parquet_path(base, "LogGrowthModel").exists()

    def test_publication_guard_blocks_before_artifact_mutation(
        self,
        tmp_path: Path,
    ) -> None:
        """GUI snapshot/owner guard is rechecked inside the artifact lock."""
        master = _synthetic_growth_master()
        pipeline = ImagePipeline(
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )

        result = _emit_analysis_outputs(
            tmp_path,
            master,
            pipeline,
            publication_guard=lambda: False,
        )

        base = deliverables_dir(tmp_path)
        assert result is None
        assert not named_analysis_csv_path(base, "LogGrowthModel").exists()
        assert not named_analysis_parquet_path(base, "LogGrowthModel").exists()
        assert not analysis_manifest_path(base).exists()

    def test_publication_guard_change_before_replace_rolls_back(
        self,
        tmp_path: Path,
    ) -> None:
        """A source change after staging leaves no partial generation."""
        master = _synthetic_growth_master()
        pipeline = ImagePipeline(
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        checks = 0

        def _guard() -> bool:
            nonlocal checks
            checks += 1
            return checks == 1

        result = _emit_analysis_outputs(
            tmp_path,
            master,
            pipeline,
            publication_guard=_guard,
        )

        base = deliverables_dir(tmp_path)
        assert result is None
        assert not named_analysis_csv_path(base, "LogGrowthModel").exists()
        assert not named_analysis_parquet_path(base, "LogGrowthModel").exists()
        assert not analysis_manifest_path(base).exists()

    def test_publication_guard_change_after_manifest_rolls_back(
        self,
        tmp_path: Path,
    ) -> None:
        """A commit-boundary source change restores artifacts and manifest."""
        master = _synthetic_growth_master()
        pipeline = ImagePipeline(
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        assert _emit_analysis_outputs(tmp_path, master, pipeline) is not None
        base = deliverables_dir(tmp_path)
        csv_path = named_analysis_csv_path(base, "LogGrowthModel")
        parquet_path = named_analysis_parquet_path(base, "LogGrowthModel")
        manifest_path = analysis_manifest_path(base)
        previous = (
            csv_path.read_bytes(),
            parquet_path.read_bytes(),
            manifest_path.read_bytes(),
        )
        changed = master.with_columns(
            (pl.col("Shape_Area") * 1.5).alias("Shape_Area")
        )
        checks = 0

        def _guard() -> bool:
            nonlocal checks
            checks += 1
            return checks < 3

        result = _emit_analysis_outputs(
            tmp_path,
            changed,
            pipeline,
            publication_guard=_guard,
        )

        assert checks == 3
        assert result is None
        assert (
            csv_path.read_bytes(),
            parquet_path.read_bytes(),
            manifest_path.read_bytes(),
        ) == previous

    def test_manifest_failure_restores_previous_artifact_generation(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import phenotypic.plotting as plotting

        master = _synthetic_growth_master()
        pipeline = ImagePipeline(
            model=LogGrowthModel(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
                time_label="Metadata_Time",
                n_jobs=1,
            ),
        )
        assert _emit_analysis_outputs(tmp_path, master, pipeline) is not None
        base = deliverables_dir(tmp_path)
        csv_path = named_analysis_csv_path(base, "LogGrowthModel")
        parquet_path = named_analysis_parquet_path(base, "LogGrowthModel")
        manifest_path = analysis_manifest_path(base)
        previous = (
            csv_path.read_bytes(),
            parquet_path.read_bytes(),
            manifest_path.read_bytes(),
        )

        def fail_manifest(*_args, **_kwargs):
            raise OSError("manifest unavailable")

        monkeypatch.setattr(
            plotting, "publish_analysis_manifest_entry", fail_manifest
        )
        changed = master.with_columns(
            (pl.col("Shape_Area") * 1.5).alias("Shape_Area")
        )

        assert _emit_analysis_outputs(tmp_path, changed, pipeline) is None
        assert (
            csv_path.read_bytes(),
            parquet_path.read_bytes(),
            manifest_path.read_bytes(),
        ) == previous


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
