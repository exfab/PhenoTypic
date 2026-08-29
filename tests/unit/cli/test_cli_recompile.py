"""Unit tests for the CLI recompile mode (``_handle_recompile``).

Verifies recompile routing, aggregation, overlay regeneration, manifest
publication, and progress-dashboard regeneration.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_utils import resolve_local_worker_count
from phenotypic.sdk_ import (
    master_measurements_csv_path,
    store_stem,
    zarr_store_path,
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


class TestRecompileCliRouting:
    """Top-level recompile mode selects local or SLURM explicitly."""

    def test_recompile_without_slurm_uses_local_handler(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)

        with (
            patch("phenotypic.phenotypicCLI._handle_recompile") as mock_local,
            patch(
                "phenotypic.phenotypicCLI._handle_recompile_slurm"
            ) as mock_slurm,
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
            patch(
                "phenotypic.phenotypicCLI._handle_recompile_slurm"
            ) as mock_slurm,
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
            patch(
                "phenotypic.phenotypicCLI._handle_recompile_slurm"
            ) as mock_slurm,
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

    def test_skips_metadata_preflight_and_its_migrate_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Only explicit migrate owns metadata migration preflight."""
        import phenotypic.sdk_ as sdk

        output_dir = _make_fake_results(tmp_path)
        preflight_calls: list[object] = []

        def _unexpected_preflight(*args: object, **_kwargs: object):
            preflight_calls.extend(args)
            return SimpleNamespace(
                targets=(SimpleNamespace(status="migratable"),)
            )

        monkeypatch.setattr(
            sdk, "preflight_metadata_schema", _unexpected_preflight
        )
        with (
            patch("phenotypic._cli._cli_output_manager.aggregate_measurements"),
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        output = capsys.readouterr().out
        assert preflight_calls == []
        assert "Metadata schema" not in output
        assert "--mode migrate" not in output

    def test_external_metadata_is_not_a_migration_target_or_mutated(
        self, tmp_path: Path
    ) -> None:
        output_dir = _make_fake_results(tmp_path)
        metadata_csv = tmp_path / "external-metadata.csv"
        original = (
            b"MetadataImage_ImageName,MetadataSample_Strain\nplate-a,WT\n"
        )
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

    def test_store_only_dataset_is_discovered_and_aggregates_safely(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        # A dataset with image stores but no measurements must still be found.
        store = zarr_store_path(output_dir, "store-only", "plateA")
        store.mkdir(parents=True)
        (store / "zarr.json").write_bytes(b"migration authority")

        with (
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        assert not (output_dir / "deliverables").exists()


class TestResolveLocalWorkerCount:
    """Resolve local worker counts with optional SLURM allocation caps."""

    def test_all_jobs_uses_host_cpus_without_slurm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

        with patch("phenotypic._cli._cli_utils.os.cpu_count", return_value=12):
            assert resolve_local_worker_count(n_jobs=-1, work_items=20) == 12

    def test_all_jobs_uses_slurm_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_work_items_cap_worker_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

        assert resolve_local_worker_count(n_jobs=-1, work_items=3) == 3


class TestRegenerateMissingOverlays:
    """Overlay regeneration uses the shared local worker resolver."""

    def test_passes_slurm_capped_workers_to_thread_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_dir = tmp_path / "out"
        # ``_regenerate_missing_overlays`` now derives the stem with
        # ``store_stem``, which RAISES on anything that is not a
        # ``*.ome.zarr`` path -- so a `.h5` fixture no longer stands in.
        store_paths = [
            zarr_store_path(output_dir, "ds1", f"img{i}") for i in range(10)
        ]
        datasets = [SimpleNamespace(name="ds1", images=store_paths)]
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

            def submit(
                self,
                _fn: object,
                dataset_name: str,
                store_path: Path,
                _requires_marker_restore: bool,
            ) -> FakeFuture:
                submitted.append((dataset_name, store_path))
                return FakeFuture()

        class FakeOutputManager:
            def get_output_path(
                self, dataset_name: str, kind: str, stem: str
            ) -> Path:
                return (
                    output_dir
                    / "results"
                    / dataset_name
                    / kind
                    / f"{stem}.png"
                )

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        with (
            patch(
                "phenotypic.phenotypicCLI.scan_store_outputs",
                return_value=datasets,
            ),
            patch(
                "phenotypic.phenotypicCLI.OutputManager.from_config",
                return_value=FakeOutputManager(),
            ),
            patch(
                "concurrent.futures.ThreadPoolExecutor",
                FakeThreadPoolExecutor,
            ),
            patch(
                "concurrent.futures.as_completed",
                side_effect=lambda futures: futures,
            ),
        ):
            _regenerate_missing_overlays(
                output_dir, overlay_alpha=0.3, n_jobs=64
            )

        assert max_workers_seen == [8]
        assert len(submitted) == len(store_paths)
        # The bare stem, not `imgN.ome`: the overlay probe is what a
        # `.stem` regression would silently corrupt.
        assert sorted(store_stem(p) for _, p in submitted) == sorted(
            f"img{i}" for i in range(10)
        )


def test_local_recompile_restores_deleted_overlay_marker_and_master(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A missing marker-bound overlay is repaired before table aggregation."""
    import polars as pl

    from phenotypic._cli._cli_completion import (
        authorized_measurement_sources,
        valid_image_success,
    )
    from phenotypic.schema import IMAGE
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
        master_measurements_parquet_path,
    )
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stems = run_stems(output_dir)
    tables = {
        zarr_store_path(output_dir, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
        for stem in stems
    }
    expected_rows = sum(pl.read_parquet(table).height for table in tables)
    missing_stem = stems[0]
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{missing_stem}.png"
    overlay.unlink()
    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=missing_stem,
        work_id=run_work_id(output_dir, missing_stem),
    )

    _handle_recompile(
        output_dir,
        metadata_csv=None,
        include_dataset_column=True,
        overlay_alpha=0.6,
        n_jobs=1,
        no_qc=True,
    )

    assert overlay.is_file()
    assert valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=missing_stem,
        work_id=run_work_id(output_dir, missing_stem),
    )
    authorized = authorized_measurement_sources(output_dir)
    assert authorized is not None
    assert set(authorized) == tables
    master = pl.read_parquet(master_measurements_parquet_path(output_dir))
    assert master.height == expected_rows
    assert set(master[str(IMAGE.IMAGE_NAME)].unique()) == set(stems)


@pytest.mark.parametrize("refresh_outcome", ["false", "raise"])
def test_local_recompile_aborts_when_required_overlay_marker_refresh_fails(
    _completed_run_two: Path,
    tmp_path: Path,
    refresh_outcome: str,
) -> None:
    """A marker-bound overlay repair cannot degrade to a partial master."""
    from phenotypic.sdk_ import dataset_overlays_dir
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    missing_stem = run_stems(output_dir)[0]
    (
        dataset_overlays_dir(output_dir, DATASET) / f"{missing_stem}.png"
    ).unlink()
    refresh_kwargs = (
        {"side_effect": RuntimeError("simulated marker refresh failure")}
        if refresh_outcome == "raise"
        else {"return_value": False}
    )

    with (
        patch(
            "phenotypic._cli._cli_recompile_slurm_scripts."
            "repair_overlay_marker_authority",
            **refresh_kwargs,
        ),
        patch(
            "phenotypic._cli._cli_output_manager.aggregate_measurements"
        ) as aggregate,
        pytest.raises(RuntimeError, match="marker authority"),
    ):
        _handle_recompile(
            output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.6,
            n_jobs=1,
            no_qc=True,
        )

    aggregate.assert_not_called()


def test_local_overlay_recovery_rejects_post_discovery_table_corruption(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Recovery compares non-overlay artifacts again immediately before publish."""
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
    )
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.unlink()
    original_save = OutputManager.save_overlay

    def _save_then_corrupt(
        manager: OutputManager,
        image: object,
        dataset_name: str,
        image_stem: str,
    ) -> object:
        result = original_save(
            manager,
            image,
            dataset_name,
            image_stem,  # type: ignore[arg-type]
        )
        table.write_bytes(b"not a parquet file")
        return result

    with (
        patch.object(OutputManager, "save_overlay", _save_then_corrupt),
        pytest.raises(RuntimeError, match="marker authority"),
    ):
        _regenerate_missing_overlays(output_dir, overlay_alpha=0.6, n_jobs=1)

    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )


def test_local_overlay_recovery_rejects_publish_window_table_change(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """Recovery cannot bless table bytes changed after descriptor validation."""
    import polars as pl

    from phenotypic._cli import _cli_completion
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
    )
    from tests.unit.sdk_._migration_fixtures import (
        DATASET,
        run_stems,
        run_work_id,
    )

    output_dir = tmp_path / "completed"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.unlink()
    original_publish = _cli_completion.publish_image_success

    def _change_table_then_publish(*args: object, **kwargs: object) -> Path:
        changed = pl.read_parquet(table).with_columns(
            pl.lit("changed").alias("Metadata_PublishWindowProbe")
        )
        changed.write_parquet(table)
        return original_publish(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(
            _cli_completion,
            "publish_image_success",
            _change_table_then_publish,
        ),
        pytest.raises(RuntimeError, match="marker authority"),
    ):
        _regenerate_missing_overlays(output_dir, overlay_alpha=0.6, n_jobs=1)

    assert not valid_image_success(
        output_dir,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(output_dir, stem),
    )


def test_local_process_only_missing_overlay_remains_best_effort(
    _completed_run_two: Path,
    tmp_path: Path,
) -> None:
    """A marker without measurement authority does not make overlay repair fatal."""
    import json

    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_overlays_dir,
        image_completion_marker_path,
    )
    from tests.unit.sdk_._migration_fixtures import DATASET, run_stems

    output_dir = tmp_path / "process-only"
    shutil.copytree(_completed_run_two, output_dir)
    stem = run_stems(output_dir)[0]
    store = zarr_store_path(output_dir, DATASET, stem)
    marker_path = image_completion_marker_path(output_dir, DATASET, stem)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["mode"] = "process"
    marker["artifacts"].pop("measurements")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (store / MEASUREMENT_TABLE_RELATIVE_PATH).unlink()
    overlay = dataset_overlays_dir(output_dir, DATASET) / f"{stem}.png"
    overlay.unlink()

    _regenerate_missing_overlays(output_dir, overlay_alpha=0.6, n_jobs=1)

    assert overlay.is_file()
