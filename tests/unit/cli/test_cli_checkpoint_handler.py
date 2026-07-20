"""Unit tests for the SLURM checkpoint handler.

Focused on schema coercion: ``job_metadata["datasets"]`` is written with the
nested ``{name: {total, images}}`` shape but ``build_manifest`` (and
downstream code) expect the flat ``{name: int}`` shape.  The handler must
coerce at the boundary, mirroring :mod:`phenotypic._cli._cli_sentinel`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="SLURM checkpoint handler only runs on POSIX",
)

from phenotypic._cli._cli_checkpoint_handler import (  # noqa: E402
    _run_finalize,
    _run_manifest,
)


NESTED_DATASETS = {
    "ds1": {"total": 3, "images": ["a.tif", "b.tif", "c.tif"]},
    "ds2": {"total": 2, "images": ["x.tif", "y.tif"]},
}
FLAT_DATASETS = {"ds1": 3, "ds2": 2}
EXPECTED_TOTALS = {"ds1": 3, "ds2": 2}


def _write_job_metadata(
    progress_dir: Path,
    datasets_field: dict,
    *,
    no_qc: bool = False,
) -> None:
    """Write a minimal job_metadata.json with the given datasets shape."""
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "job_metadata.json").write_text(
        json.dumps(
            {
                "start_time": "2026-04-21T00:00:00.000",
                "execution_mode": "slurm",
                "datasets": datasets_field,
                "chunk_job_ids": {"0": "12345"},
                "chunk_scripts": [],
                "image_task_mapping": {},
                "include_dataset_column": True,
                "metadata_csv": None,
                "no_qc": no_qc,
                "input_path": "fake",
            }
        ),
        encoding="utf-8",
    )


class TestRunManifestCoercion:
    """``_run_manifest`` must pass ``{name: int}`` to ``build_manifest``."""

    @pytest.mark.parametrize(
        "datasets_field",
        [NESTED_DATASETS, FLAT_DATASETS],
        ids=["nested_schema", "legacy_flat_int"],
    )
    def test_build_manifest_receives_flat_int_datasets(
        self, tmp_path: Path, datasets_field: dict
    ) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        _write_job_metadata(progress_dir, datasets_field)

        with patch(
            "phenotypic._cli._dashboard._manifest_builder.build_manifest"
        ) as mock_build_manifest:
            _run_manifest(output_dir, progress_dir)

        mock_build_manifest.assert_called_once()
        kwargs = mock_build_manifest.call_args.kwargs
        assert kwargs["datasets"] == EXPECTED_TOTALS
        assert all(isinstance(v, int) for v in kwargs["datasets"].values())

    def test_no_metadata_file_early_exits(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        progress_dir.mkdir(parents=True)

        with patch(
            "phenotypic._cli._dashboard._manifest_builder.build_manifest"
        ) as mock_build_manifest:
            _run_manifest(output_dir, progress_dir)

        mock_build_manifest.assert_not_called()

    def test_finalize_propagates_no_qc_to_aggregation(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        _write_job_metadata(progress_dir, NESTED_DATASETS, no_qc=True)

        with (
            patch(
                "phenotypic._cli._cli_checkpoint_handler._wait_for_completion"
            ),
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements"
            ) as mock_aggregate,
            patch(
                "phenotypic._cli._dashboard._manifest_builder.build_manifest"
            ),
            patch(
                "phenotypic._cli._cli_chunk_writer._run_analysis_plugins"
            ),
            patch(
                "phenotypic._cli._dashboard._generator.generate_dashboard"
            ),
        ):
            _run_finalize(output_dir, progress_dir)

        assert mock_aggregate.call_args.kwargs["no_qc"] is True


class TestRunFinalizeCoercion:
    """``_run_finalize`` must coerce before sum(), aggregate, and manifest."""

    @pytest.mark.parametrize(
        "datasets_field",
        [NESTED_DATASETS, FLAT_DATASETS],
        ids=["nested_schema", "legacy_flat_int"],
    )
    def test_finalize_coerces_and_propagates_flat_totals(
        self, tmp_path: Path, datasets_field: dict
    ) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        _write_job_metadata(progress_dir, datasets_field)

        with (
            patch(
                "phenotypic._cli._cli_checkpoint_handler._wait_for_completion"
            ) as mock_wait,
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements"
            ) as mock_aggregate,
            patch(
                "phenotypic._cli._dashboard._manifest_builder.build_manifest"
            ) as mock_build_manifest,
            patch(
                "phenotypic._cli._cli_chunk_writer._run_analysis_plugins"
            ) as mock_plugins,
            patch(
                "phenotypic._cli._dashboard._generator.generate_dashboard"
            ) as mock_dashboard,
        ):
            _run_finalize(output_dir, progress_dir)

        mock_wait.assert_called_once()
        wait_total = mock_wait.call_args.args[1]
        assert wait_total == sum(EXPECTED_TOTALS.values())

        mock_aggregate.assert_called_once()
        aggregate_kwargs = mock_aggregate.call_args.kwargs
        assert aggregate_kwargs["dataset_names"] == list(EXPECTED_TOTALS.keys())

        mock_build_manifest.assert_called_once()
        manifest_kwargs = mock_build_manifest.call_args.kwargs
        assert manifest_kwargs["datasets"] == EXPECTED_TOTALS
        assert all(isinstance(v, int) for v in manifest_kwargs["datasets"].values())

        mock_plugins.assert_called_once()
        mock_dashboard.assert_called_once()

    def test_no_metadata_file_early_exits(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        progress_dir.mkdir(parents=True)

        with (
            patch(
                "phenotypic._cli._cli_checkpoint_handler._wait_for_completion"
            ) as mock_wait,
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements"
            ) as mock_aggregate,
            patch(
                "phenotypic._cli._dashboard._manifest_builder.build_manifest"
            ) as mock_build_manifest,
        ):
            _run_finalize(output_dir, progress_dir)

        mock_wait.assert_not_called()
        mock_aggregate.assert_not_called()
        mock_build_manifest.assert_not_called()
