"""Legacy external Parquets migrate into OME-Zarr table authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from click.testing import CliRunner
import polars as pl
import pytest

from phenotypic._cli._cli_completion import valid_image_success
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    aggregate_publication_marker_path,
    PhenotypicAttr,
    image_completion_marker_path,
    read_phenotypic_attributes,
    zarr_store_path,
)
from tests.unit.sdk_._migration_fixtures import (
    DATASET,
    LEGACY_MEASUREMENT_COLUMN,
    run_stems,
    run_work_id,
)


def _file_inventory(root: Path) -> dict[str, str]:
    """Return path-to-digest inventory for every regular file below *root*."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_migration_embeds_parquet_preserves_then_safely_deletes_source(
    legacy_run: Path,
) -> None:
    """Default preservation, idempotence, and explicit source deletion."""
    source = legacy_run / "results" / "ds" / "measurements" / "img.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "Object_Label": [1],
            "Size_Area": [25.0],
            "Metadata_ImageName": ["img"],
        }
    ).write_parquet(source)

    first = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run)],
    )
    assert first.exit_code == 0, first.output
    table = (
        zarr_store_path(legacy_run, "ds", "img")
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    assert table.is_file(), "migration left measurement authority external"
    assert source.is_file(), "sources must be preserved by default"
    first_bytes = table.read_bytes()

    second = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run)],
    )
    assert second.exit_code == 0, second.output
    assert table.read_bytes() == first_bytes

    deleting = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--delete-sources",
        ],
    )
    assert deleting.exit_code == 0, deleting.output
    assert not source.exists()
    assert table.is_file()


def test_migration_repairs_payload_only_store_and_preserves_header_order(
    legacy_headers_run: Path,
) -> None:
    """A stranded payload gains validated groups and a root-last descriptor."""
    stem = run_stems(legacy_headers_run)[0]
    source = (
        legacy_headers_run
        / "results"
        / DATASET
        / "measurements"
        / f"{stem}.parquet"
    )
    legacy_columns = pl.read_parquet(source).columns
    legacy_index = legacy_columns.index(LEGACY_MEASUREMENT_COLUMN)
    store = zarr_store_path(legacy_headers_run, DATASET, stem)
    payload = store / MEASUREMENT_TABLE_RELATIVE_PATH
    payload.parent.mkdir(parents=True)
    shutil.copy2(source, payload)
    assert not (store / "tables" / "zarr.json").exists()
    assert PhenotypicAttr.TABLES not in read_phenotypic_attributes(store)

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )

    assert result.exit_code == 0, result.output
    for group in (store / "tables", store / "tables" / "measurements"):
        document = json.loads(
            (group / "zarr.json").read_text(encoding="utf-8")
        )
        assert document["zarr_format"] == 3
        assert document["node_type"] == "group"
    descriptor = read_phenotypic_attributes(store)[PhenotypicAttr.TABLES][
        "measurements"
    ]
    expected_columns = list(legacy_columns)
    expected_columns[legacy_index] = "Metadata_Strain"
    assert descriptor["measurement_columns"] == expected_columns
    assert (
        pl.read_parquet(payload).columns[: len(expected_columns)]
        == expected_columns
    )
    assert source.is_file(), (
        "repair must preserve the legacy source by default"
    )
    assert valid_image_success(
        legacy_headers_run,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(legacy_headers_run, stem),
    )


def test_migration_repairs_corrupt_measurement_group_documents(
    legacy_headers_run: Path,
) -> None:
    """A readable payload is not enough when its Zarr group nodes are invalid."""
    first = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )
    assert first.exit_code == 0, first.output
    stem = run_stems(legacy_headers_run)[0]
    store = zarr_store_path(legacy_headers_run, DATASET, stem)
    group_root = store / "tables" / "zarr.json"
    group_root.write_text('{"zarr_format": 2}', encoding="utf-8")

    second = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )

    assert second.exit_code == 0, second.output
    repaired = json.loads(group_root.read_text(encoding="utf-8"))
    assert repaired == {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
    }


def test_embedded_table_migration_dry_run_is_fully_nonmutating(
    legacy_headers_run: Path,
) -> None:
    """Dry run leaves existing stores, external tables, and provenance byte-identical."""
    before = _file_inventory(legacy_headers_run)

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_headers_run),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _file_inventory(legacy_headers_run) == before


def test_embedded_table_migration_retries_after_marker_interruption(
    legacy_headers_run: Path, monkeypatch
) -> None:
    """A table committed before marker publication is authorized on retry."""
    from phenotypic._cli import _cli_completion

    original_publish = _cli_completion.publish_image_success
    interrupted = False

    def fail_first_publish(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError(
                "simulated interruption before marker publication"
            )
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        _cli_completion, "publish_image_success", fail_first_publish
    )
    first = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )
    assert first.exit_code != 0
    stem = run_stems(legacy_headers_run)[0]
    table = (
        zarr_store_path(legacy_headers_run, DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    assert table.is_file(), (
        "interruption must leave the committed table retryable"
    )

    monkeypatch.setattr(
        _cli_completion, "publish_image_success", original_publish
    )
    second = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )
    assert second.exit_code == 0, second.output
    marker = image_completion_marker_path(legacy_headers_run, DATASET, stem)
    assert marker.is_file()
    assert valid_image_success(
        legacy_headers_run,
        dataset=DATASET,
        image_stem=stem,
        work_id=run_work_id(legacy_headers_run, stem),
    )


def test_migration_reconstructs_authority_without_machine_state(
    legacy_headers_run: Path,
) -> None:
    """A state-free archive gains marker and aggregate authority."""
    from phenotypic._cli._cli_completion import (
        authorized_measurement_sources,
        current_aggregate_is_current,
    )
    from phenotypic._cli._cli_state_management import load_processing_state
    from phenotypic.sdk_ import phenotypic_cache_dir

    shutil.rmtree(phenotypic_cache_dir(legacy_headers_run))
    assert load_processing_state(legacy_headers_run) is None

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )

    assert result.exit_code == 0, result.output
    state = load_processing_state(legacy_headers_run)
    assert state is not None
    assert state.config["success_markers_required"] is True
    sources = authorized_measurement_sources(legacy_headers_run)
    assert sources is not None
    assert len(sources) == len(run_stems(legacy_headers_run))
    assert current_aggregate_is_current(legacy_headers_run) is True


def test_hdf_only_migration_keeps_store_measurement_free(
    legacy_run: Path,
) -> None:
    """An image without a legacy table converts safely without inventing one."""
    source = legacy_run / "results" / DATASET / "measurements" / "img.parquet"
    source.unlink()

    with patch(
        "phenotypic._cli._cli_migrate.republish_aggregate"
    ) as republish:
        result = CliRunner().invoke(
            phenotypic_cli,
            ["--mode", "migrate", "--output", str(legacy_run)],
        )

    assert result.exit_code == 0, result.output
    republish.assert_not_called()
    store = zarr_store_path(legacy_run, DATASET, "img")
    assert store.is_dir()
    assert not (store / MEASUREMENT_TABLE_RELATIVE_PATH).exists()
    assert PhenotypicAttr.TABLES not in read_phenotypic_attributes(store)
    assert not aggregate_publication_marker_path(legacy_run).exists()


@pytest.mark.parametrize("aggregate_outcome", ["raise", "none"])
def test_migration_does_not_certify_stale_outputs_after_aggregate_failure(
    legacy_headers_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_outcome: str,
) -> None:
    """A failed or empty rebuild leaves no aggregate authority behind."""
    first = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )
    assert first.exit_code == 0, first.output
    marker = aggregate_publication_marker_path(legacy_headers_run)
    assert marker.is_file()

    def fail_or_return_none(**_kwargs: object) -> None:
        if aggregate_outcome == "raise":
            raise RuntimeError("simulated aggregate failure")
        return None

    monkeypatch.setattr(
        "phenotypic._cli._cli_output_manager.aggregate_measurements",
        fail_or_return_none,
    )
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_headers_run)],
    )

    assert result.exit_code != 0
    assert "aggregate publication failed" in result.output
    assert not marker.exists(), (
        "stale aggregate outputs retained fresh authority"
    )
