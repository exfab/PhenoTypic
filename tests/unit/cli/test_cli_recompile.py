"""Unit tests for the CLI recompile mode (``_handle_recompile``).

Verifies recompile routing, aggregation, overlay regeneration, manifest
publication, and progress-dashboard regeneration.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
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

    def test_reports_the_schema_before_aggregating_and_never_rewrites_it(
        self, tmp_path: Path
    ) -> None:
        """Recompile INSPECTS the metadata schema; it does not migrate it.

        The rewrite moved to ``--mode migrate`` (Phase 5 Task 5.4, superseding
        flat-metadata decision #1). Decision #3 is untouched -- the read path
        canonicalizes legacy headers in memory -- so the inspection still runs
        first, ahead of aggregation, purely so the user is told.
        """
        output_dir = _make_fake_results(tmp_path)
        events: list[str] = []

        def _report(_output_dir: Path):
            events.append("report")
            return SimpleNamespace(targets=())

        def _aggregate(**_kwargs: object) -> None:
            events.append("aggregate")

        with (
            patch(
                "phenotypic._cli._cli_recompile_metadata_migration."
                "report_metadata_schema_for_recompile",
                side_effect=_report,
            ),
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements",
                side_effect=_aggregate,
            ),
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        assert events == ["report", "aggregate"]

    def test_no_rewriting_entry_point_survives_on_the_recompile_seam(
        self,
    ) -> None:
        """The seam is read-only by construction, not merely by call site.

        A recompile that still *could* rewrite is one refactor away from
        doing so again, so the mutating names are asserted absent rather
        than merely unused.
        """
        from phenotypic._cli import _cli_recompile_metadata_migration as seam

        for name in (
            "migrate_metadata_schema_for_recompile",
            "RecompileMetadataMigrationError",
        ):
            assert not hasattr(seam, name), name

    def test_reports_the_legacy_target_count_and_names_the_remedy(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output_dir = _make_fake_results(tmp_path)
        report = SimpleNamespace(
            targets=(
                SimpleNamespace(status="migratable"),
                SimpleNamespace(status="migratable"),
                SimpleNamespace(status="compatible"),
            )
        )

        with (
            patch(
                "phenotypic._cli._cli_recompile_metadata_migration."
                "report_metadata_schema_for_recompile",
                return_value=report,
            ),
            patch(
                "phenotypic._cli._cli_output_manager.aggregate_measurements"
            ),
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        output = capsys.readouterr().out
        compact = "".join(output.split())
        assert "2target(s)" in compact
        assert "--modemigrate" in compact
        assert not (output_dir / ".phenotypic" / "metadata_migration").exists()

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
        # A dataset with image stores but no measurements: dataset
        # discovery must still find it, or the schema report it authorizes
        # never runs.
        store = zarr_store_path(output_dir, "store-only", "plateA")
        store.mkdir(parents=True)
        (store / "zarr.json").write_bytes(b"migration authority")

        with (
            patch(
                "phenotypic._cli._cli_recompile_metadata_migration."
                "report_metadata_schema_for_recompile",
                return_value=SimpleNamespace(targets=()),
            ) as report,
            patch("phenotypic.phenotypicCLI._regenerate_missing_overlays"),
            patch("phenotypic._cli._dashboard.regenerate_dashboard_artifacts"),
        ):
            _handle_recompile(output_dir, None, True, 0.3, 1)

        report.assert_called_once_with(output_dir)
        assert not (output_dir / "deliverables").exists()


class TestRecompileMetadataSchemaSeam:
    """The local metadata seam INSPECTS; it never rewrites."""

    def test_a_legacy_hdf_is_reported_and_left_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """The MIG-25 guard, from the recompile side.

        Recompile used to byte-copy and rewrite every legacy ``.h5`` through
        ``_migrate_hdf_copy``. Nothing does that now: ``--mode migrate``
        excludes ``.h5`` from its metadata pass unconditionally, and recompile
        does not migrate at all. So the originals survive both.
        """
        import hashlib

        import h5py

        from phenotypic._cli._cli_recompile_metadata_migration import (
            legacy_header_target_count,
            report_metadata_schema_for_recompile,
        )

        output_dir = _make_fake_results(tmp_path)
        hdf_path = output_dir / "results" / "ds1" / "hdf" / "plateA.h5"
        hdf_path.parent.mkdir(parents=True)
        with h5py.File(hdf_path, "w") as handle:
            handle.attrs["schema_version"] = 1
            public = handle.create_group("public_metadata")
            public.attrs["MetadataGenetic_Strain"] = "S288C"
        before = hashlib.sha256(hdf_path.read_bytes()).hexdigest()

        report = report_metadata_schema_for_recompile(output_dir)

        assert legacy_header_target_count(report) >= 1
        assert hashlib.sha256(hdf_path.read_bytes()).hexdigest() == before
        with h5py.File(hdf_path, "r") as handle:
            assert (
                handle["public_metadata"].attrs["MetadataGenetic_Strain"]
                == "S288C"
            )
        assert not (output_dir / ".phenotypic" / "metadata_migration").exists()

    def test_a_canonical_bundle_is_a_repeatable_no_op(
        self, tmp_path: Path
    ) -> None:
        from phenotypic._cli._cli_recompile_metadata_migration import (
            report_metadata_schema_for_recompile,
        )

        output_dir = _make_fake_results(tmp_path)

        first = report_metadata_schema_for_recompile(output_dir)
        second = report_metadata_schema_for_recompile(output_dir)

        assert first.status == second.status == "compatible"
        assert not (output_dir / ".phenotypic" / "metadata_migration").exists()

    def test_marker_authorized_measurements_are_left_untouched(
        self, tmp_path: Path
    ) -> None:
        """Recompile cannot invalidate per-image authority it never rewrites.

        The old version of this test asserted that a metadata rewrite
        PRESERVED that authority through a receipt bridge. The rewrite is
        gone from recompile, so the stronger property now holds: the bytes
        the markers bind are not touched at all. The migrate-side equivalent
        is ``test_every_image_still_validates_after_migration``.
        """
        import h5py

        from phenotypic._cli._cli_completion import (
            authorized_measurement_sources,
            publish_image_success,
            valid_image_success,
        )
        from phenotypic._cli._cli_recompile_metadata_migration import (
            report_metadata_schema_for_recompile,
        )
        from phenotypic._cli._cli_state_management import save_processing_state
        from phenotypic._cli._cli_types import DatasetState, ProcessingState

        output_dir = tmp_path / "out"
        measurement = (
            output_dir / "results" / "ds1" / "measurements" / "img1.parquet"
        )
        measurement.parent.mkdir(parents=True)
        pd.DataFrame(
            {"MetadataGenetic_Strain": ["S288C"], "Size_Area": [1.0]}
        ).to_parquet(measurement, index=False)
        hdf_path = output_dir / "results" / "ds1" / "hdf" / "img1.h5"
        hdf_path.parent.mkdir(parents=True)
        with h5py.File(hdf_path, "w") as handle:
            handle.attrs["schema_version"] = 1
            public = handle.create_group("public_metadata")
            public.attrs["MetadataGenetic_Strain"] = "S288C"

        now = datetime.now()
        save_processing_state(
            ProcessingState(
                version="3.0.0",
                pipeline_path=output_dir / "pipeline.json",
                input_path=tmp_path / "input",
                output_dir=output_dir,
                timestamp=now,
                execution_mode="local",
                last_updated=now,
                datasets={"ds1": DatasetState(initial_images={"img1.tif"})},
                config={
                    "success_markers_required": True,
                    "work_ids": {"ds1": {"img1.tif": "work-1"}},
                    "pipeline_sha256": "pipeline",
                },
            ),
            output_dir,
        )
        publish_image_success(
            output_dir,
            work_id="work-1",
            dataset="ds1",
            relative_image_path="ds1/img1.tif",
            image_stem="img1",
            mode="full",
            attempt_id="attempt",
            lifecycle_epoch="local",
            artifacts={"measurements": measurement, "hdf": hdf_path},
        )
        measurement_bytes = measurement.read_bytes()

        report_metadata_schema_for_recompile(output_dir)

        assert measurement.read_bytes() == measurement_bytes
        assert valid_image_success(
            output_dir,
            dataset="ds1",
            image_stem="img1",
            work_id="work-1",
        )
        assert authorized_measurement_sources(output_dir) == {
            measurement.resolve(): "ds1"
        }
        assert list(pd.read_parquet(measurement).columns) == [
            "MetadataGenetic_Strain",
            "Size_Area",
        ]


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
                self, _fn: object, dataset_name: str, store_path: Path
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
