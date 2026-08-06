"""Unit tests for the CLI recompile mode (``_handle_recompile``).

Verifies recompile routing, aggregation, overlay regeneration, manifest
publication, and progress-dashboard regeneration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_utils import resolve_local_worker_count
from phenotypic.sdk_ import (
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
    (ds_meas_dir / "img1.parquet").write_text("")  # presence only
    return output_dir


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
