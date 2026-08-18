"""Unit tests for the SLURM checkpoint handler.

Focused on schema coercion: ``job_metadata["datasets"]`` is written with the
nested ``{name: {total, images}}`` shape but ``build_manifest`` (and
downstream code) expect the flat ``{name: int}`` shape.  The handler must
coerce at the boundary, mirroring :mod:`phenotypic._cli._cli_sentinel`.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="SLURM checkpoint handler only runs on POSIX",
)

from phenotypic._cli._cli_checkpoint_handler import (  # noqa: E402
    _publish_run_completion_marker,
    _run_finalize,
    _run_manifest,
    main,
)
from phenotypic._cli._cli_file_locking import FileLockTimeout  # noqa: E402
from phenotypic._cli._cli_staged_orchestration import (  # noqa: E402
    initialize_orchestration,
    staged_completion_path,
)
from phenotypic.sdk_ import (  # noqa: E402
    atomic_write_json,
    progress_dir as progress_dir_helper,
    resolve_manifest_json_path,
    run_completion_marker_path,
)
from phenotypic._cli._cli_slurm_lifecycle import (  # noqa: E402
    deactivate_generation,
    initialize_slurm_lifecycle,
    lifecycle_state_path,
    load_slurm_lifecycle,
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
    slurm_generation: str | None = None,
) -> None:
    """Write a minimal job_metadata.json with the given datasets shape."""
    progress_dir.mkdir(parents=True, exist_ok=True)
    payload = {
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
    if slurm_generation is not None:
        payload["slurm_generation"] = slurm_generation
    (progress_dir / "job_metadata.json").write_text(
        json.dumps(payload),
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

    def test_process_export_manifest_publishes_generation_marker_last(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        generation = "fedcba9876543210fedcba9876543210"
        _write_job_metadata(
            progress_dir,
            {"ds1": {"total": 1, "images": ["a.tif"]}},
            slurm_generation=generation,
        )
        initialize_slurm_lifecycle(
            output_dir,
            generation=generation,
            mode="ordinary",
        )
        marker_path = run_completion_marker_path(output_dir)

        def publish_manifest(**_kwargs: object) -> None:
            assert not marker_path.exists()
            atomic_write_json(
                resolve_manifest_json_path(output_dir),
                {
                    "is_complete": True,
                    "completed": 1,
                    "failed": 0,
                    "total_images": 1,
                },
            )

        with (
            patch(
                "phenotypic._cli._dashboard._manifest_builder.build_manifest",
                side_effect=publish_manifest,
            ),
            patch(
                "phenotypic._cli._cli_state_management.load_processing_state",
                return_value=SimpleNamespace(
                    config={"process_only_layer": "rgb"}
                ),
            ),
        ):
            _run_manifest(output_dir, progress_dir)

        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["generation"] == generation
        assert marker["status"] == "complete"
        assert marker["finalizer_succeeded"] is True

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
                "phenotypic._cli._dashboard._generator.generate_dashboard"
            ) as mock_dashboard,
        ):
            _run_finalize(output_dir, progress_dir)

        mock_wait.assert_called_once()
        wait_total = mock_wait.call_args.kwargs["total_expected"]
        assert wait_total == sum(EXPECTED_TOTALS.values())
        if datasets_field is NESTED_DATASETS:
            assert mock_wait.call_args.kwargs["inventory"] == {
                "ds1": frozenset({"a.tif", "b.tif", "c.tif"}),
                "ds2": frozenset({"x.tif", "y.tif"}),
            }
        else:
            assert mock_wait.call_args.kwargs["inventory"] is None

        mock_aggregate.assert_called_once()
        aggregate_kwargs = mock_aggregate.call_args.kwargs
        assert aggregate_kwargs["dataset_names"] == list(EXPECTED_TOTALS.keys())

        mock_build_manifest.assert_called_once()
        manifest_kwargs = mock_build_manifest.call_args.kwargs
        assert manifest_kwargs["datasets"] == EXPECTED_TOTALS
        assert all(isinstance(v, int) for v in manifest_kwargs["datasets"].values())

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

    def test_ordinary_finalizer_publishes_generation_marker_last(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        generation = "0123456789abcdef0123456789abcdef"
        _write_job_metadata(
            progress_dir,
            {"ds1": {"total": 1, "images": ["a.tif"]}},
            slurm_generation=generation,
        )
        initialize_slurm_lifecycle(
            output_dir,
            generation=generation,
            mode="ordinary",
        )
        marker_path = run_completion_marker_path(output_dir)

        def publish_manifest(**_kwargs: object) -> None:
            assert not marker_path.exists()
            atomic_write_json(
                resolve_manifest_json_path(output_dir),
                {
                    "is_complete": True,
                    "completed": 1,
                    "failed": 0,
                    "total_images": 1,
                },
            )

        with (
            patch(
                "phenotypic._cli._cli_checkpoint_handler._wait_for_completion"
            ),
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                return_value=output_dir / "deliverables" / "measurements.parquet",
            ),
            patch(
                "phenotypic._cli._dashboard._manifest_builder.build_manifest",
                side_effect=publish_manifest,
            ),
            patch(
                "phenotypic._cli._dashboard._generator.generate_dashboard"
            ),
        ):
            _run_finalize(output_dir, progress_dir)

        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["generation"] == generation
        assert marker["status"] == "complete"
        assert marker["finalizer_succeeded"] is True
        lifecycle = load_slurm_lifecycle(output_dir)
        assert lifecycle is not None
        assert lifecycle["active"] is False

    def test_ordinary_finalizer_refuses_marker_for_failed_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "out"
        progress_dir = output_dir / "progress"
        _write_job_metadata(
            progress_dir,
            {"ds1": {"total": 1, "images": ["a.tif"]}},
            slurm_generation="0123456789abcdef0123456789abcdef",
        )
        initialize_slurm_lifecycle(
            output_dir,
            generation="0123456789abcdef0123456789abcdef",
            mode="ordinary",
        )

        def publish_failed_manifest(**_kwargs: object) -> None:
            atomic_write_json(
                resolve_manifest_json_path(output_dir),
                {
                    "is_complete": True,
                    "completed": 0,
                    "failed": 1,
                    "total_images": 1,
                },
            )

        with (
            patch(
                "phenotypic._cli._cli_checkpoint_handler._wait_for_completion"
            ),
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                return_value=output_dir / "deliverables" / "measurements.parquet",
            ),
            patch(
                "phenotypic._cli._dashboard._manifest_builder.build_manifest",
                side_effect=publish_failed_manifest,
            ),
            patch(
                "phenotypic._cli._dashboard._generator.generate_dashboard"
            ),
            pytest.raises(RuntimeError, match="incomplete or failed manifest"),
        ):
            _run_finalize(output_dir, progress_dir)

        assert not run_completion_marker_path(output_dir).exists()

    def test_completion_marker_is_idempotent_for_same_finished_generation(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "out"
        generation = "1123456789abcdef0123456789abcdef"
        initialize_slurm_lifecycle(
            output_dir,
            generation=generation,
            mode="ordinary",
        )
        atomic_write_json(
            resolve_manifest_json_path(output_dir),
            {
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            },
        )

        _publish_run_completion_marker(output_dir, generation)
        _publish_run_completion_marker(output_dir, generation)

        marker = json.loads(
            run_completion_marker_path(output_dir).read_text(encoding="utf-8")
        )
        assert marker["generation"] == generation
        assert load_slurm_lifecycle(output_dir)["active"] is False  # type: ignore[index]

    def test_completion_marker_does_not_reacquire_lifecycle_lock(
        self,
        tmp_path: Path,
    ) -> None:
        """Already-locked completion uses the lock-required fence helper."""
        output_dir = tmp_path / "out"
        generation = "1223456789abcdef0123456789abcdef"
        initialize_slurm_lifecycle(
            output_dir,
            generation=generation,
            mode="ordinary",
        )
        atomic_write_json(
            resolve_manifest_json_path(output_dir),
            {
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            },
        )
        depth = 0
        acquisitions = 0

        @contextmanager
        def fail_on_nested_lock(*_args: object, **_kwargs: object):
            nonlocal acquisitions, depth
            if depth:
                raise AssertionError("nested lifecycle lock acquisition")
            depth += 1
            acquisitions += 1
            try:
                yield
            finally:
                depth -= 1

        with (
            patch(
                "phenotypic._cli._cli_checkpoint_handler.exclusive_path_lock",
                fail_on_nested_lock,
            ),
            patch(
                "phenotypic._cli._cli_slurm_lifecycle.exclusive_path_lock",
                fail_on_nested_lock,
            ),
        ):
            _publish_run_completion_marker(output_dir, generation)

        assert acquisitions == 1
        lifecycle = load_slurm_lifecycle(output_dir)
        assert lifecycle is not None
        assert lifecycle["active"] is False
        assert run_completion_marker_path(output_dir).is_file()

    def test_old_finalizer_cannot_publish_after_new_generation(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "out"
        old_generation = "2123456789abcdef0123456789abcdef"
        new_generation = "3123456789abcdef0123456789abcdef"
        initialize_slurm_lifecycle(
            output_dir,
            generation=old_generation,
            mode="ordinary",
        )
        state = load_slurm_lifecycle(output_dir)
        assert state is not None
        state["active"] = False
        atomic_write_json(lifecycle_state_path(output_dir), state)
        initialize_slurm_lifecycle(
            output_dir,
            generation=new_generation,
            mode="ordinary",
        )
        atomic_write_json(
            resolve_manifest_json_path(output_dir),
            {
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            },
        )

        with pytest.raises(RuntimeError, match="stale SLURM generation"):
            _publish_run_completion_marker(output_dir, old_generation)

        assert not run_completion_marker_path(output_dir).exists()
        lifecycle = load_slurm_lifecycle(output_dir)
        assert lifecycle is not None
        assert lifecycle["generation"] == new_generation
        assert lifecycle["active"] is True

    def test_cancelled_generation_cannot_publish_completion_marker(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "out"
        generation = "4123456789abcdef0123456789abcdef"
        initialize_slurm_lifecycle(
            output_dir,
            generation=generation,
            mode="ordinary",
        )
        assert deactivate_generation(output_dir, generation) is True
        atomic_write_json(
            resolve_manifest_json_path(output_dir),
            {
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            },
        )

        with pytest.raises(RuntimeError, match="cancelled or superseded"):
            _publish_run_completion_marker(output_dir, generation)

        assert not run_completion_marker_path(output_dir).exists()

    def test_staged_finalize_fails_when_no_current_measurements(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        progress_dir = progress_dir_helper(output_dir)
        _write_job_metadata(progress_dir, NESTED_DATASETS)
        initialize_orchestration(
            output_dir,
            epoch="epoch-1",
            mode="restart",
            controller_config_path=progress_dir / "controller.json",
        )

        with (
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                return_value=None,
            ),
            pytest.raises(
                RuntimeError,
                match="No current-epoch measurements",
            ),
        ):
            _run_finalize(output_dir, progress_dir, epoch="epoch-1")

    def test_duplicate_finalizer_waits_for_matching_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        progress_dir = progress_dir_helper(output_dir)
        initialize_orchestration(
            output_dir,
            epoch="epoch-1",
            mode="fresh",
            controller_config_path=progress_dir / "controller.json",
        )
        attempts = 0

        @contextmanager
        def fake_file_lock(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise FileLockTimeout("winner still publishing")
            atomic_write_json(
                staged_completion_path(output_dir), {"epoch": "epoch-1"}
            )
            yield

        monkeypatch.setattr(
            "phenotypic._cli._cli_checkpoint_handler.file_lock", fake_file_lock
        )

        result = CliRunner().invoke(
            main,
            [
                "--output-dir",
                str(output_dir),
                "--checkpoint-type",
                "finalize",
                "--epoch",
                "epoch-1",
            ],
        )

        assert result.exit_code == 0, result.output
        assert attempts == 2
