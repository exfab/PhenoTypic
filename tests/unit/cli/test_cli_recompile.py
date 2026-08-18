"""Unit tests for the CLI recompile mode (``_handle_recompile``).

Verifies recompile routing, aggregation, overlay regeneration, manifest
publication, and progress-dashboard regeneration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_utils import resolve_local_worker_count
from phenotypic.sdk_ import (
    MetadataMigrationResult,
    master_measurements_csv_path,
)
from phenotypic.phenotypicCLI import (
    _handle_recompile,
    _regenerate_missing_overlays,
    phenotypic_cli,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CLI recompile path has non-Windows dependencies",
)


def _make_fake_results(tmp_path: Path) -> Path:
    """Create a minimal output dir with one dataset and a stub parquet."""
    output_dir = tmp_path / "out"
    ds_meas_dir = output_dir / "results" / "ds1" / "measurements"
    ds_meas_dir.mkdir(parents=True)
    pd.DataFrame({"Size_Area": [1.0]}).to_parquet(
        ds_meas_dir / "img1.parquet", index=False
    )
    return output_dir


def _migration_result(
    *,
    status: Literal["compatible", "applied", "blocked", "failed"] = "compatible",
    migrated: tuple[str, ...] = (),
    skipped: tuple[str, ...] = (),
    blocked: tuple[str, ...] = (),
    receipt_path: Path | None = None,
    conflicts: tuple[str, ...] = (),
) -> MetadataMigrationResult:
    """Build a focused migration result for local recompile tests."""
    return MetadataMigrationResult(
        status=status,
        source="/output",
        source_fingerprint="sha256:source",
        resulting_fingerprint=(
            "sha256:result" if status in {"compatible", "applied"} else None
        ),
        plan_fingerprint="sha256:plan",
        receipt_path=receipt_path,
        migrated_targets=migrated,
        skipped_targets=skipped,
        blocked_targets=blocked,
        conflicts=conflicts,
    )


class TestRecompileCliRouting:
    """Top-level recompile mode selects local or SLURM explicitly."""

    def test_recompile_without_slurm_uses_local_handler(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)

        with (
            patch("phenotypic.phenotypicCLI._handle_recompile") as mock_local,
            patch("phenotypic.phenotypicCLI._handle_recompile_slurm") as mock_slurm,
        ):
            result = CliRunner().invoke(
                phenotypic_cli,
                ["--mode", "recompile", "--output", str(output_dir)],
            )

        assert result.exit_code == 0, result.output
        mock_local.assert_called_once_with(
            output_dir, None, True, 0.3, -1, no_qc=False
        )
        mock_slurm.assert_not_called()

    def test_recompile_with_slurm_uses_slurm_handler(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)

        with (
            patch("phenotypic.phenotypicCLI._handle_recompile") as mock_local,
            patch("phenotypic.phenotypicCLI._handle_recompile_slurm") as mock_slurm,
        ):
            result = CliRunner().invoke(
                phenotypic_cli,
                [
                    "--mode",
                    "recompile",
                    "--output",
                    str(output_dir),
                    "--slurm",
                    "slurm_partition=compute",
                    "--slurm",
                    "time=30",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_local.assert_not_called()
        mock_slurm.assert_called_once_with(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={"slurm_partition": "compute", "time": "00:30:00"},
            wait=False,
            no_qc=False,
        )

    def test_recompile_force_local_overrides_slurm(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)

        with (
            patch("phenotypic.phenotypicCLI._handle_recompile") as mock_local,
            patch("phenotypic.phenotypicCLI._handle_recompile_slurm") as mock_slurm,
        ):
            result = CliRunner().invoke(
                phenotypic_cli,
                [
                    "--mode",
                    "recompile",
                    "--output",
                    str(output_dir),
                    "--slurm",
                    "slurm_partition=compute",
                    "--force-local",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_local.assert_called_once_with(
            output_dir, None, True, 0.3, -1, no_qc=False
        )
        mock_slurm.assert_not_called()


class TestHandleRecompile:
    """``_handle_recompile`` republishes current supported artifacts."""

    def test_rebuilds_manifest_and_progress_dashboard(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)

        def _fake_aggregate(**_kwargs: object) -> Path:
            master = master_measurements_csv_path(output_dir)
            master.parent.mkdir(parents=True, exist_ok=True)
            master.write_text("col_a\n1\n", encoding="utf-8")
            return master

        with (
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                side_effect=_fake_aggregate,
            ),
            patch(
                "phenotypic._cli._dashboard._manifest_builder.build_manifest"
            ) as mock_manifest,
            patch(
                "phenotypic._cli._dashboard._generator.generate_dashboard"
            ) as mock_dashboard,
        ):
            _handle_recompile(
                output_dir=output_dir,
                metadata_csv=None,
                include_dataset_column=True,
                overlay_alpha=0.3,
                n_jobs=1,
            )

        mock_manifest.assert_called_once()
        mock_dashboard.assert_called_once()

    def test_migrates_after_discovery_and_before_aggregation(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)
        events: list[str] = []

        def _migrate(_output_dir: Path) -> MetadataMigrationResult:
            events.append("migration")
            return _migration_result(status="compatible")

        def _aggregate(**_kwargs: object) -> None:
            events.append("aggregate")

        with (
            patch(
                "phenotypic._cli._cli_recompile_metadata_migration."
                "migrate_metadata_schema_for_recompile",
                side_effect=_migrate,
            ),
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                side_effect=_aggregate,
            ),
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        assert events == ["migration", "aggregate"]

    @pytest.mark.parametrize("status", ["blocked", "failed"])
    def test_unsafe_migration_aborts_before_any_recompile_publication(
        self,
        tmp_path: Path,
        status: Literal["blocked", "failed"],
    ) -> None:
        from phenotypic._cli._cli_recompile_metadata_migration import (
            RecompileMetadataMigrationError,
        )

        output_dir = _make_fake_results(tmp_path)
        result = _migration_result(
            status=status,
            blocked=(str(output_dir / "legacy.h5"),),
            conflicts=("legacy and canonical values disagree",),
        )

        with (
            patch(
                "phenotypic._cli._cli_recompile_metadata_migration."
                "migrate_metadata_schema_for_recompile",
                side_effect=RecompileMetadataMigrationError(result),
            ),
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements"
            ) as aggregate,
            patch(
                "phenotypic.phenotypicCLI._regenerate_missing_overlays"
            ) as overlays,
            patch(
                "phenotypic._cli._dashboard.regenerate_dashboard_artifacts"
            ) as dashboard,
            pytest.raises(SystemExit) as exc_info,
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        assert exc_info.value.code == 1
        aggregate.assert_not_called()
        overlays.assert_not_called()
        dashboard.assert_not_called()

    def test_reports_migration_counts_and_receipt(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output_dir = _make_fake_results(tmp_path)
        receipt = output_dir / ".phenotypic" / "metadata_migration" / "receipt.json"
        result = _migration_result(
            status="applied",
            migrated=("one.h5", "two.h5"),
            skipped=("pipeline.json",),
            receipt_path=receipt,
        )

        with (
            patch(
                "phenotypic._cli._cli_recompile_metadata_migration."
                "migrate_metadata_schema_for_recompile",
                return_value=result,
            ),
            patch("phenotypic._cli._cli_output_manager.aggregate_measurements"),
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        output = capsys.readouterr().out
        assert "migrated=2, skipped=1, blocked=0" in output
        compact_output = "".join(output.split())
        assert f"Migrationreceipt:{receipt}" in compact_output

    def test_external_metadata_is_not_a_migration_target_or_mutated(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)
        metadata_csv = tmp_path / "external-metadata.csv"
        original = b"MetadataImage_ImageName,MetadataSample_Strain\nplate-a,WT\n"
        metadata_csv.write_bytes(original)

        with (
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements"
            ) as aggregate,
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, metadata_csv, True, 0.3, 1)

        assert aggregate.call_args.kwargs["metadata_csv"] == metadata_csv
        assert metadata_csv.read_bytes() == original

    def test_hdf_only_dataset_migrates_before_safe_empty_aggregation(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        hdf_dir = output_dir / "results" / "hdf-only" / "hdf"
        hdf_dir.mkdir(parents=True)
        (hdf_dir / "plateA.h5").write_bytes(b"migration authority")
        compatible = _migration_result(status="compatible")

        with (
            patch(
                "phenotypic._cli._cli_recompile_metadata_migration."
                "migrate_metadata_schema_for_recompile",
                return_value=compatible,
            ) as migrate,
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        migrate.assert_called_once_with(output_dir)
        assert not (output_dir / "deliverables").exists()


class TestRecompileMetadataMigrationFacade:
    """The local migration seam is idempotent for canonical bundles."""

    def test_legacy_hdf_is_migrated_once_then_becomes_no_op(
        self, tmp_path: Path
    ) -> None:
        import h5py

        from phenotypic._cli._cli_recompile_metadata_migration import (
            migrate_metadata_schema_for_recompile,
        )

        output_dir = _make_fake_results(tmp_path)
        hdf_path = output_dir / "results" / "ds1" / "hdf" / "plateA.h5"
        hdf_path.parent.mkdir(parents=True)
        with h5py.File(hdf_path, "w") as handle:
            handle.attrs["schema_version"] = 1
            public = handle.create_group("public_metadata")
            public.attrs["MetadataGenetic_Strain"] = "S288C"

        first = migrate_metadata_schema_for_recompile(output_dir)
        with h5py.File(hdf_path, "r") as handle:
            assert handle["public_metadata"].attrs["Metadata_Strain"] == "S288C"
            assert "MetadataGenetic_Strain" not in handle["public_metadata"].attrs
            assert int(handle.attrs["metadata_schema_version"]) == 2
        second = migrate_metadata_schema_for_recompile(output_dir)

        assert first.status == "applied"
        assert first.migrated_targets == (str(hdf_path.resolve()),)
        assert first.receipt_path is not None
        assert second.status == "compatible"
        assert second.migrated_targets == ()
        assert second.receipt_path is None

    def test_canonical_bundle_is_repeatable_preflight_no_op(
        self, tmp_path: Path
    ) -> None:
        from phenotypic._cli._cli_recompile_metadata_migration import (
            migrate_metadata_schema_for_recompile,
        )

        output_dir = _make_fake_results(tmp_path)

        first = migrate_metadata_schema_for_recompile(output_dir)
        second = migrate_metadata_schema_for_recompile(output_dir)

        assert first.status == second.status == "compatible"
        assert first.migrated_targets == second.migrated_targets == ()
        assert first.receipt_path is second.receipt_path is None
        assert not (output_dir / ".phenotypic" / "metadata_migration").exists()


class TestResolveLocalWorkerCount:
    """Resolve local worker counts with optional SLURM allocation caps."""

    def test_all_jobs_uses_host_cpus_without_slurm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

        with patch("phenotypic._cli._cli_utils.os.cpu_count", return_value=12):
            assert resolve_local_worker_count(n_jobs=-1, work_items=20) == 12

    def test_all_jobs_uses_slurm_allocation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

        assert resolve_local_worker_count(n_jobs=-1, work_items=20) == 8

    def test_explicit_jobs_are_capped_by_slurm_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

        assert resolve_local_worker_count(n_jobs=64, work_items=20) == 8

    def test_explicit_jobs_below_slurm_allocation_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

        assert resolve_local_worker_count(n_jobs=4, work_items=20) == 4

    def test_invalid_slurm_allocation_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-an-int")

        with patch("phenotypic._cli._cli_utils.os.cpu_count", return_value=12):
            assert resolve_local_worker_count(n_jobs=-1, work_items=20) == 12

    def test_work_items_cap_worker_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

        assert resolve_local_worker_count(n_jobs=-1, work_items=3) == 3


class TestRegenerateMissingOverlays:
    """Overlay regeneration uses the shared local worker resolver."""

    def test_passes_slurm_capped_workers_to_thread_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_dir = tmp_path / "out"
        hdf_paths = [
            output_dir / "results" / "ds1" / "hdf" / f"img{i}.h5"
            for i in range(10)
        ]
        datasets = [SimpleNamespace(name="ds1", images=hdf_paths)]
        submitted = []
        max_workers_seen = []

        class FakeFuture:
            def result(self) -> None:
                return None

        class FakeThreadPoolExecutor:
            def __init__(self, max_workers: int) -> None:
                max_workers_seen.append(max_workers)

            def __enter__(self) -> "FakeThreadPoolExecutor":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def submit(self, _fn: object, dataset_name: str, hdf_path: Path) -> FakeFuture:
                submitted.append((dataset_name, hdf_path))
                return FakeFuture()

        class FakeOutputManager:
            def get_output_path(self, dataset_name: str, kind: str, stem: str) -> Path:
                return output_dir / "results" / dataset_name / kind / f"{stem}.png"

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        with (
            patch("phenotypic.phenotypicCLI.scan_hdf_outputs", return_value=datasets),
            patch(
                "phenotypic.phenotypicCLI.OutputManager.from_config",
                return_value=FakeOutputManager(),
            ),
            patch(
                "concurrent.futures.ThreadPoolExecutor",
                FakeThreadPoolExecutor,
            ),
            patch("concurrent.futures.as_completed", side_effect=lambda futures: futures),
        ):
            _regenerate_missing_overlays(output_dir, overlay_alpha=0.3, n_jobs=64)

        assert max_workers_seen == [8]
        assert len(submitted) == len(hdf_paths)
