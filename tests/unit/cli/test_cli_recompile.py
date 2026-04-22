"""Unit tests for the CLI ``--recompile`` path (``_handle_recompile``).

Verifies that recompile mirrors the SLURM finalizer's post-aggregation
pattern: it reads the freshly-written ``master_measurements.csv`` and
dispatches analysis plugins via ``_run_analysis_plugins`` (not the
DuckDB-based ``write_analysis_sidecar``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CLI recompile path has non-Windows dependencies",
)

from phenotypic.phenotypicCLI import _handle_recompile


def _make_fake_results(tmp_path: Path) -> Path:
    """Create a minimal output dir with one dataset and a stub parquet."""
    output_dir = tmp_path / "out"
    ds_meas_dir = output_dir / "results" / "ds1" / "measurements"
    ds_meas_dir.mkdir(parents=True)
    (ds_meas_dir / "img1.parquet").write_text("")  # presence only
    return output_dir


class TestHandleRecompile:
    """``_handle_recompile`` uses the finalizer analysis-plugin pattern."""

    def test_dispatches_run_analysis_plugins_from_master_csv(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)

        def _fake_aggregate(**_kwargs: object) -> Path:
            master = output_dir / "master_measurements.csv"
            master.write_text("col_a\n1\n", encoding="utf-8")
            return master

        with (
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                side_effect=_fake_aggregate,
            ),
            patch(
                "phenotypic._cli._cli_chunk_writer._run_analysis_plugins"
            ) as mock_plugins,
            patch(
                "phenotypic._cli._dashboard.build_manifest"
            ) as mock_manifest,
            patch(
                "phenotypic._cli._dashboard.generate_dashboard"
            ) as mock_dashboard,
            patch(
                "phenotypic._cli._dashboard._analysis_data.write_analysis_sidecar"
            ) as mock_sidecar,
        ):
            _handle_recompile(
                output_dir=output_dir,
                metadata_csv=None,
                include_dataset_column=True,
            )

        mock_sidecar.assert_not_called()

        mock_plugins.assert_called_once()
        args = mock_plugins.call_args.args
        assert args[0] == output_dir
        assert args[1] == output_dir / "progress"
        merged_df = args[2]
        assert merged_df is not None
        assert merged_df.height == 1
        assert "col_a" in merged_df.columns

        mock_manifest.assert_called_once()
        mock_dashboard.assert_called_once()

    def test_plugin_dispatch_failure_does_not_crash(self, tmp_path: Path) -> None:
        output_dir = _make_fake_results(tmp_path)

        def _fake_aggregate(**_kwargs: object) -> Path:
            master = output_dir / "master_measurements.csv"
            master.write_text("col_a\n1\n", encoding="utf-8")
            return master

        with (
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                side_effect=_fake_aggregate,
            ),
            patch(
                "phenotypic._cli._cli_chunk_writer._run_analysis_plugins",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "phenotypic._cli._dashboard.build_manifest"
            ) as mock_manifest,
            patch(
                "phenotypic._cli._dashboard.generate_dashboard"
            ) as mock_dashboard,
        ):
            _handle_recompile(
                output_dir=output_dir,
                metadata_csv=None,
                include_dataset_column=True,
            )

        mock_manifest.assert_called_once()
        mock_dashboard.assert_called_once()
