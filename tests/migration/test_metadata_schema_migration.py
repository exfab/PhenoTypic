"""Durable migration coverage for the flat metadata namespace."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from phenotypic.sdk_ import (
    BundleLayout,
    file_fingerprint,
    migrate_metadata_bundle,
    migrate_metadata_file,
    preflight_metadata_schema,
    rollback_metadata_migration,
)

LEGACY_IMAGE = "MetadataImage_ImageName"
CANONICAL_IMAGE = "Metadata_ImageName"
LEGACY_STRAIN = "MetadataGenetic_Strain"
CANONICAL_STRAIN = "Metadata_Strain"
CANONICAL_VALUE = "Metadata_value"
BARE_IMAGE = "ImageName"
BARE_STRAIN = "Strain"


def _write_v1_hdf(path: Path, *, duplicate_conflict: bool = False) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["sentinel"] = "keep"
        handle.create_dataset("gray", data=np.arange(12, dtype=np.uint16).reshape(3, 4))
        protected = handle.create_group("protected_metadata")
        protected.attrs[BARE_IMAGE] = "plate.tif"
        public = handle.create_group("public_metadata")
        public.attrs[BARE_STRAIN] = "BY4741"
        if duplicate_conflict:
            public.attrs[CANONICAL_STRAIN] = "mutant"


def _write_v2_hdf(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 2
        handle.attrs["phenotypic_class"] = "Image"
        layers = handle.create_group("layers")
        layers.create_dataset("gray", data=np.arange(6, dtype=np.float32).reshape(2, 3))
        layers.create_dataset("detect_mat", data=np.ones((2, 3), dtype=np.float32))
        layers.create_dataset("objmap", data=np.zeros((2, 3), dtype=np.uint16))
        public = handle.create_group("metadata/public")
        public.attrs[LEGACY_STRAIN] = json.dumps("BY4741")
        imported = handle.create_group("metadata/imported")
        imported.attrs["camera"] = json.dumps("scope")


def _hdf_dataset_bytes(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with h5py.File(path, "r") as handle:
        def collect(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                result[name] = np.asarray(obj[()]).tobytes()

        handle.visititems(collect)
    return result


def test_frame_preflight_is_immutable_and_reports_header_map() -> None:
    frame = pd.DataFrame(
        {LEGACY_STRAIN: ["BY4741", None], CANONICAL_STRAIN: [None, "mutant"]}
    )
    original = frame.copy(deep=True)

    report = preflight_metadata_schema(frame)

    assert report.status == "migratable"
    assert report.targets[0].proposed_header_map == (
        (LEGACY_STRAIN, CANONICAL_STRAIN),
    )
    pd.testing.assert_frame_equal(frame, original)


def test_polars_frame_preflight_is_immutable() -> None:
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({LEGACY_STRAIN: ["BY4741"], "Batch": [1]})
    original = frame.clone()

    report = preflight_metadata_schema(frame)

    assert report.status == "migratable"
    assert frame.equals(original)


def test_csv_migration_coalesces_and_rolls_back_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        f"{LEGACY_STRAIN},{CANONICAL_STRAIN},value\nBY4741,,1\n,mutant,2\n",
        encoding="utf-8",
    )
    original = path.read_bytes()
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert result.status == "applied"
    migrated = pd.read_csv(path)
    assert list(migrated.columns) == [CANONICAL_STRAIN, CANONICAL_VALUE]
    assert migrated[CANONICAL_STRAIN].tolist() == ["BY4741", "mutant"]
    assert result.receipt_path is not None

    rollback = rollback_metadata_migration(result.receipt_path)
    assert rollback.status == "rolled_back"
    assert path.read_bytes() == original


def test_duplicate_conflict_blocks_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {LEGACY_STRAIN: ["BY4741"], CANONICAL_STRAIN: ["mutant"]}
    ).to_parquet(path, index=False)
    before = file_fingerprint(path)
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert report.status == "blocked"
    assert result.status == "blocked"
    assert result.conflicts
    assert file_fingerprint(path) == before


def test_parquet_migration_preserves_rows_and_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "metadata.parquet"
    frame = pd.DataFrame(
        {
            LEGACY_STRAIN: pd.Series([1, None], dtype="Int32"),
            CANONICAL_STRAIN: pd.Series([None, 2], dtype="Int64"),
            "value": pd.Series([2.5, 1.5], dtype="Float32"),
        }
    )
    frame.to_parquet(path, index=False)
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    migrated = pd.read_parquet(path)

    assert result.status == "applied"
    assert list(migrated.columns) == [CANONICAL_STRAIN, CANONICAL_VALUE]
    assert migrated[CANONICAL_STRAIN].dtype == pd.Int64Dtype()
    assert migrated[CANONICAL_STRAIN].tolist() == [1, 2]
    assert migrated[CANONICAL_VALUE].dtype == pd.Float32Dtype()
    assert migrated[CANONICAL_VALUE].tolist() == [2.5, 1.5]


def test_typed_pipeline_json_migrates_only_real_column_reference_fields(
    tmp_path: Path,
) -> None:
    from phenotypic import ImagePipeline
    from phenotypic.analysis import TukeyOutlierRemover
    from phenotypic.post import AppendString, MergeMetadata

    path = tmp_path / "pipeline.json.pht-pipe"
    pipeline = ImagePipeline(
        name=LEGACY_IMAGE,
        desc=LEGACY_STRAIN,
        post=[
            AppendString(column="Strain", value=LEGACY_IMAGE),
            MergeMetadata(columns=["Strain", "ImageName"], label="SampleID"),
        ],
        filters=[TukeyOutlierRemover(on="Size_Area", groupby=["Strain"])],
    )
    payload = json.loads(pipeline.to_json() or "{}")
    append_params = payload["post"]["AppendString"]["params"]
    append_params["column"] = LEGACY_STRAIN
    append_params["value"] = LEGACY_IMAGE
    merge_params = payload["post"]["MergeMetadata"]["params"]
    merge_params["columns"] = [LEGACY_STRAIN, LEGACY_IMAGE]
    merge_params["label"] = LEGACY_STRAIN
    filter_params = payload["filters"]["TukeyOutlierRemover"]["params"]
    filter_params["groupby"] = [LEGACY_STRAIN]
    filter_params["on"] = LEGACY_IMAGE
    payload["custom"] = {LEGACY_IMAGE: LEGACY_STRAIN, "value": LEGACY_IMAGE}
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    migrated = json.loads(path.read_text(encoding="utf-8"))

    assert result.status == "applied"
    assert migrated["name"] == LEGACY_IMAGE
    assert migrated["desc"] == LEGACY_STRAIN
    assert migrated["post"]["AppendString"]["params"] == {
        "column": CANONICAL_STRAIN,
        "value": LEGACY_IMAGE,
    }
    assert migrated["post"]["MergeMetadata"]["params"]["columns"] == [
        CANONICAL_STRAIN,
        CANONICAL_IMAGE,
    ]
    assert migrated["post"]["MergeMetadata"]["params"]["label"] == CANONICAL_STRAIN
    assert migrated["filters"]["TukeyOutlierRemover"]["params"]["groupby"] == [
        CANONICAL_STRAIN
    ]
    assert migrated["filters"]["TukeyOutlierRemover"]["params"]["on"] == CANONICAL_IMAGE
    assert migrated["custom"] == payload["custom"]


def test_unknown_custom_operation_params_are_opaque_but_known_nested_ops_migrate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custom.json.pht-pipe"
    custom_params = {
        "column": LEGACY_STRAIN,
        "groupby": [LEGACY_IMAGE],
        "description": LEGACY_STRAIN,
        "nested": {
            "__type__": "operation",
            "class": "AppendString",
            "params": {"column": LEGACY_STRAIN, "value": LEGACY_IMAGE},
        },
    }
    path.write_text(
        json.dumps(
            {
                "pipe_cfgs": {},
                "meas": {},
                "post": {"CustomOp": {"class": "CustomOp", "params": custom_params}},
                "filters": {},
                "model": None,
            }
        ),
        encoding="utf-8",
    )
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    migrated = json.loads(path.read_text(encoding="utf-8"))
    params = migrated["post"]["CustomOp"]["params"]

    assert result.status == "applied"
    assert params["column"] == LEGACY_STRAIN
    assert params["groupby"] == [LEGACY_IMAGE]
    assert params["description"] == LEGACY_STRAIN
    assert params["nested"]["params"] == {
        "column": CANONICAL_STRAIN,
        "value": LEGACY_IMAGE,
    }


def test_real_inline_plot_column_references_migrate(tmp_path: Path) -> None:
    from phenotypic import ImagePipeline
    from phenotypic.plotting import PlotColonyMetricOverTime

    path = tmp_path / "plot.json.pht-pipe"
    payload = json.loads(
        ImagePipeline(
            plots=[PlotColonyMetricOverTime(on="Size_Area")]
        ).to_json()
        or "{}"
    )
    params = payload["plots"][0]["inline"]["params"]
    params.update(
        {
            "on": LEGACY_IMAGE,
            "strain_label": LEGACY_STRAIN,
            "groupby": [LEGACY_IMAGE],
            "replicate_label": LEGACY_STRAIN,
            "time": LEGACY_IMAGE,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    migrated = json.loads(path.read_text(encoding="utf-8"))
    migrated_params = migrated["plots"][0]["inline"]["params"]

    assert result.status == "applied"
    assert migrated_params["on"] == CANONICAL_IMAGE
    assert migrated_params["strain_label"] == CANONICAL_STRAIN
    assert migrated_params["groupby"] == [CANONICAL_IMAGE]
    assert migrated_params["replicate_label"] == CANONICAL_STRAIN
    assert migrated_params["time"] == CANONICAL_IMAGE
    assert migrated_params["connect"] is True


def test_unknown_inline_plot_is_fully_opaque(tmp_path: Path) -> None:
    path = tmp_path / "unknown-plot.json.pht-pipe"
    payload = {
        "pipe_cfgs": {},
        "meas": {},
        "post": {},
        "filters": {},
        "model": None,
        "plots": [
            {
                "id": "custom",
                "inline": {
                    "module": "custom_plots",
                    "qualname": "CustomPlot",
                    "params": {
                        "on": LEGACY_IMAGE,
                        "groupby": [LEGACY_STRAIN],
                    },
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert report.status == "compatible"
    assert result.status == "compatible"
    assert path.read_bytes() == before


@pytest.mark.parametrize("writer", [_write_v1_hdf, _write_v2_hdf])
def test_hdf_migration_preserves_layout_arrays_and_non_target_attrs(
    tmp_path: Path, writer
) -> None:
    path = tmp_path / "image.h5"
    writer(path)
    datasets_before = _hdf_dataset_bytes(path)
    with h5py.File(path, "r") as handle:
        layout_version = int(handle.attrs["schema_version"])
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert result.status == "applied"
    assert _hdf_dataset_bytes(path) == datasets_before
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["schema_version"]) == layout_version
        assert int(handle.attrs["metadata_schema_version"]) == 2
        if layout_version == 1:
            assert CANONICAL_STRAIN in handle["public_metadata"].attrs
            assert BARE_STRAIN not in handle["public_metadata"].attrs
            assert handle.attrs["sentinel"] == "keep"
        else:
            assert CANONICAL_STRAIN in handle["metadata/public"].attrs
            assert LEGACY_STRAIN not in handle["metadata/public"].attrs

    assert result.receipt_path is not None
    rollback = rollback_metadata_migration(result.receipt_path)
    assert rollback.status == "rolled_back"
    assert _hdf_dataset_bytes(path) == datasets_before
    with h5py.File(path, "r") as handle:
        assert "metadata_schema_version" not in handle.attrs
        if layout_version == 1:
            assert handle.attrs["sentinel"] == "keep"
        group = handle["public_metadata"] if layout_version == 1 else handle["metadata/public"]
        source_header = BARE_STRAIN if layout_version == 1 else LEGACY_STRAIN
        assert source_header in group.attrs
        assert CANONICAL_STRAIN not in group.attrs


def test_hdf_conflict_blocks_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "image.h5"
    _write_v1_hdf(path, duplicate_conflict=True)
    before = file_fingerprint(path)
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert report.status == "blocked"
    assert result.status == "blocked"
    assert file_fingerprint(path) == before


def test_hdf_bare_and_legacy_alias_conflict_without_canonical_blocks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.h5"
    _write_v1_hdf(path)
    with h5py.File(path, "r+") as handle:
        handle["public_metadata"].attrs[LEGACY_STRAIN] = "mutant"
    before = file_fingerprint(path)

    report = preflight_metadata_schema(path)
    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert report.status == "blocked"
    assert result.status == "blocked"
    assert any("converging" in conflict for conflict in result.conflicts)
    assert file_fingerprint(path) == before


def test_hdf_equal_bare_and_canonical_attrs_coalesce(tmp_path: Path) -> None:
    path = tmp_path / "image.h5"
    _write_v1_hdf(path)
    with h5py.File(path, "r+") as handle:
        handle["public_metadata"].attrs[CANONICAL_STRAIN] = "BY4741"
    report = preflight_metadata_schema(path)

    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert result.status == "applied"
    with h5py.File(path, "r") as handle:
        attrs = handle["public_metadata"].attrs
        assert attrs[CANONICAL_STRAIN] == "BY4741"
        assert BARE_STRAIN not in attrs


def test_hdf_outdated_marker_alone_is_migratable(tmp_path: Path) -> None:
    path = tmp_path / "image.h5"
    _write_v2_hdf(path)
    with h5py.File(path, "r+") as handle:
        public = handle["metadata/public"]
        public.attrs[CANONICAL_STRAIN] = public.attrs[LEGACY_STRAIN]
        del public.attrs[LEGACY_STRAIN]
        handle.attrs["metadata_schema_version"] = 1
    report = preflight_metadata_schema(path)

    assert report.status == "migratable"
    assert report.targets[0].proposed_header_map == ()
    assert report.targets[0].needs_metadata_marker is True
    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert result.status == "applied"
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["schema_version"]) == 2
        assert int(handle.attrs["metadata_schema_version"]) == 2


def test_pre_replace_failure_leaves_original_and_resume_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    path = tmp_path / "metadata.csv"
    path.write_text(f"{LEGACY_STRAIN}\nBY4741\n", encoding="utf-8")
    before = path.read_bytes()
    report = preflight_metadata_schema(path)
    real_publish = migration._publish_temp

    def fail_publish(_temp: Path, _target: Path) -> None:
        raise OSError("injected before replace")

    monkeypatch.setattr(migration, "_publish_temp", fail_publish)
    failed = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert failed.status == "failed"
    assert path.read_bytes() == before

    monkeypatch.setattr(migration, "_publish_temp", real_publish)
    assert failed.receipt_path is not None
    resumed = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert resumed.status == "applied"
    assert pd.read_csv(path).columns.tolist() == [CANONICAL_STRAIN]


def test_post_replace_crash_resumes_by_original_path_and_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    path = tmp_path / "metadata.csv"
    path.write_text(f"{LEGACY_STRAIN}\nBY4741\n", encoding="utf-8")
    report = preflight_metadata_schema(path)
    original_fingerprint = report.targets[0].source_fingerprint
    real_write_receipt = migration._write_receipt

    def crash_after_replace(receipt_path: Path, receipt: dict[str, object]) -> None:
        targets = receipt.get("targets", [])
        if isinstance(targets, list) and targets and targets[0].get("state") == "applied":
            raise KeyboardInterrupt("injected process death after replace")
        real_write_receipt(receipt_path, receipt)

    monkeypatch.setattr(migration, "_write_receipt", crash_after_replace)
    with pytest.raises(KeyboardInterrupt, match="after replace"):
        migrate_metadata_file(path, expected_source_fingerprint=original_fingerprint)
    assert pd.read_csv(path).columns.tolist() == [CANONICAL_STRAIN]

    monkeypatch.setattr(migration, "_write_receipt", real_write_receipt)
    resumed = migrate_metadata_file(
        path, expected_source_fingerprint=original_fingerprint
    )
    assert resumed.status == "applied"
    assert resumed.migrated_targets == (str(path),)


def test_crafted_receipt_cannot_redirect_rollback_to_external_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(f"{LEGACY_STRAIN}\nBY4741\n", encoding="utf-8")
    report = preflight_metadata_schema(path)
    applied = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert applied.receipt_path is not None
    victim = tmp_path / "victim.csv"
    victim.write_text("do-not-touch\n", encoding="utf-8")
    victim_before = victim.read_bytes()
    source_before = path.read_bytes()
    receipt = json.loads(applied.receipt_path.read_text(encoding="utf-8"))
    receipt["targets"][0]["path"] = str(victim)
    applied.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    crafted_before = applied.receipt_path.read_bytes()

    result = rollback_metadata_migration(applied.receipt_path)

    assert result.status == "failed"
    assert victim.read_bytes() == victim_before
    assert path.read_bytes() == source_before
    assert applied.receipt_path.read_bytes() == crafted_before


def test_existing_corrupt_backup_blocks_without_publication(tmp_path: Path) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    path = tmp_path / "metadata.csv"
    path.write_text(f"{LEGACY_STRAIN}\nBY4741\n", encoding="utf-8")
    before = path.read_bytes()
    report = preflight_metadata_schema(path)
    receipt_path = migration._receipt_path(path, report.plan_fingerprint, bundle=False)
    backup_dir = receipt_path.parent / "backups"
    backup_dir.mkdir(parents=True)
    source_fp = report.targets[0].source_fingerprint
    backup = backup_dir / f"{path.name}.{source_fp.removeprefix('sha256:')[:16]}.bak"
    backup.write_bytes(b"corrupt")

    result = migrate_metadata_file(path, expected_source_fingerprint=source_fp)

    assert result.status == "failed"
    assert "wrong fingerprint" in result.conflicts[0]
    assert path.read_bytes() == before
    assert backup.read_bytes() == b"corrupt"


def test_corrupt_prepared_backup_fails_before_replace_and_remains_rollback_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    path = tmp_path / "metadata.csv"
    path.write_text(f"{LEGACY_STRAIN}\nBY4741\n", encoding="utf-8")
    before = path.read_bytes()
    report = preflight_metadata_schema(path)

    def corrupt_copy(_source: Path, destination: Path) -> Path:
        Path(destination).write_bytes(b"corrupt backup")
        return Path(destination)

    monkeypatch.setattr(migration.shutil, "copy2", corrupt_copy)
    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert result.status == "failed"
    assert "Prepared migration backup has wrong fingerprint" in result.conflicts[0]
    assert path.read_bytes() == before
    assert result.receipt_path is not None
    assert not list((result.receipt_path.parent / "backups").glob("*.bak"))
    rollback = rollback_metadata_migration(result.receipt_path)
    assert rollback.status == "rolled_back"
    assert path.read_bytes() == before


def test_backup_publication_uses_sibling_temp_fsync_and_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    source = tmp_path / "metadata.csv"
    source.write_text(f"{LEGACY_STRAIN}\nWT\n", encoding="utf-8")
    receipt = tmp_path / ".metadata_migration" / "receipt.json"
    source_fp = file_fingerprint(source)
    replacements: list[tuple[Path, Path]] = []
    synced: list[Path] = []
    real_replace = migration.os.replace
    real_fsync_file = migration._fsync_file

    def record_replace(old: Path, new: Path) -> None:
        replacements.append((Path(old), Path(new)))
        real_replace(old, new)

    def record_fsync(path: Path) -> None:
        synced.append(Path(path))
        real_fsync_file(path)

    monkeypatch.setattr(migration.os, "replace", record_replace)
    monkeypatch.setattr(migration, "_fsync_file", record_fsync)
    backup = migration._copy_backup(
        source, receipt, source_fingerprint=source_fp
    )

    assert replacements == [(replacements[0][0], backup)]
    assert replacements[0][0].parent == backup.parent
    assert replacements[0][0].name.endswith(".tmp")
    assert replacements[0][0] in synced
    assert file_fingerprint(backup) == source_fp


def test_durable_directory_creation_fsyncs_each_entry_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    synced: list[Path] = []
    monkeypatch.setattr(
        migration, "_fsync_directory", lambda path: synced.append(Path(path))
    )
    first = tmp_path / "journal"
    second = first / "backups"

    migration._ensure_directory_durable(second)

    assert first.is_dir() and second.is_dir()
    assert synced == [first, tmp_path, second, first]


def test_full_bundle_migrates_hdf_and_pipeline_but_not_external_copy_or_derived(
    tmp_path: Path,
) -> None:
    deliverables = tmp_path / "deliverables"
    deliverables.mkdir()
    master = deliverables / "master_measurements.parquet"
    pd.DataFrame({LEGACY_STRAIN: ["legacy"]}).to_parquet(master, index=False)
    external_copy = deliverables / "metadata.csv"
    external_copy.write_text(f"{LEGACY_STRAIN}\nlegacy\n", encoding="utf-8")
    from phenotypic import ImagePipeline
    from phenotypic.post import AppendString

    pipeline = deliverables / "pipeline.json.pht-pipe"
    pipeline_payload = json.loads(
        ImagePipeline(post=[AppendString(column="Strain", value="-x")]).to_json() or "{}"
    )
    pipeline_payload["post"]["AppendString"]["params"]["column"] = LEGACY_STRAIN
    pipeline.write_text(json.dumps(pipeline_payload), encoding="utf-8")
    hdf = tmp_path / "results" / "dataset" / "hdf" / "plate.h5"
    hdf.parent.mkdir(parents=True)
    _write_v2_hdf(hdf)
    master_before = file_fingerprint(master)
    external_before = file_fingerprint(external_copy)
    report = preflight_metadata_schema(tmp_path)

    result = migrate_metadata_bundle(
        tmp_path, expected_plan_fingerprint=report.plan_fingerprint
    )

    assert result.status == "applied"
    assert file_fingerprint(master) == master_before
    assert file_fingerprint(external_copy) == external_before
    assert json.loads(pipeline.read_text())["post"]["AppendString"]["params"]["column"] == CANONICAL_STRAIN
    with h5py.File(hdf, "r") as handle:
        assert CANONICAL_STRAIN in handle["metadata/public"].attrs


@pytest.mark.skipif(os.name != "posix", reason="symlink security contract")
def test_bundle_pipeline_symlink_to_external_file_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "out"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    (output / "results").mkdir()
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    victim = tmp_path / "external.json.pht-pipe"
    victim.write_text(json.dumps({"groupby": [LEGACY_STRAIN]}), encoding="utf-8")
    before = file_fingerprint(victim)
    (deliverables / "pipeline.json.pht-pipe").symlink_to(victim)

    with pytest.raises(ValueError, match="pipeline cannot be a symlink"):
        preflight_metadata_schema(output)

    assert file_fingerprint(victim) == before


@pytest.mark.skipif(os.name != "posix", reason="symlink security contract")
def test_bundle_hdf_symlink_to_external_file_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "out"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    hdf_dir = output / "results" / "dataset" / "hdf"
    hdf_dir.mkdir(parents=True)
    victim = tmp_path / "external.h5"
    _write_v2_hdf(victim)
    before = file_fingerprint(victim)
    (hdf_dir / "plate.h5").symlink_to(victim)

    with pytest.raises(ValueError, match="HDF cannot be a symlink"):
        preflight_metadata_schema(output)

    assert file_fingerprint(victim) == before


@pytest.mark.skipif(os.name != "posix", reason="symlink security contract")
def test_standalone_master_symlink_to_external_file_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    victim = tmp_path / "external.parquet"
    pd.DataFrame({LEGACY_STRAIN: ["WT"]}).to_parquet(victim, index=False)
    before = file_fingerprint(victim)
    (bundle / "master_measurements.parquet").symlink_to(victim)

    with pytest.raises(ValueError, match="master cannot be a symlink"):
        preflight_metadata_schema(bundle)

    assert file_fingerprint(victim) == before


def test_standalone_bundle_migrates_clean_master_only(tmp_path: Path) -> None:
    master = tmp_path / "master_measurements.parquet"
    mirror = tmp_path / "measurements.parquet"
    pd.DataFrame({LEGACY_STRAIN: ["legacy"]}).to_parquet(master, index=False)
    pd.DataFrame({LEGACY_STRAIN: ["curated"]}).to_parquet(mirror, index=False)
    mirror_before = file_fingerprint(mirror)
    report = preflight_metadata_schema(tmp_path)

    result = migrate_metadata_bundle(
        tmp_path, expected_plan_fingerprint=report.plan_fingerprint
    )

    assert result.status == "applied"
    assert pd.read_parquet(master).columns.tolist() == [CANONICAL_STRAIN]
    assert file_fingerprint(mirror) == mirror_before


def test_hdf_only_full_run_receipt_validates_without_master_table(tmp_path: Path) -> None:
    hdf = tmp_path / "results" / "dataset" / "hdf" / "plate.h5"
    hdf.parent.mkdir(parents=True)
    _write_v2_hdf(hdf)
    layout = BundleLayout(
        deliverables_base=tmp_path / "deliverables", output_root=tmp_path
    )
    report = preflight_metadata_schema(layout)

    result = migrate_metadata_bundle(
        layout, expected_plan_fingerprint=report.plan_fingerprint
    )

    assert result.status == "applied"
    assert result.receipt_path is not None
    with h5py.File(hdf, "r") as handle:
        assert CANONICAL_STRAIN in handle["metadata/public"].attrs


def test_hdf_rollback_rejects_snapshot_keys_outside_planned_map(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.h5"
    _write_v1_hdf(path)
    report = preflight_metadata_schema(path)
    applied = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert applied.receipt_path is not None
    receipt = json.loads(applied.receipt_path.read_text(encoding="utf-8"))
    snapshot = receipt["targets"][0]["hdf_snapshot"]
    attributes = next(record for record in snapshot if "attributes" in record)
    attributes["affected"].append("sentinel")
    attributes["affected"].sort()
    attributes["attributes"]["sentinel"] = snapshot[-1]["marker_value"] or {
        "encoding": "list",
        "dtype": "<U4",
        "shape": [],
        "value": "keep",
    }
    applied.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    migrated_before = file_fingerprint(path)

    rollback = rollback_metadata_migration(applied.receipt_path)

    assert rollback.status == "failed"
    assert "preflight binding" in rollback.conflicts[0]
    assert file_fingerprint(path) == migrated_before
    with h5py.File(path, "r") as handle:
        assert handle.attrs["sentinel"] == "keep"
        assert CANONICAL_STRAIN in handle["public_metadata"].attrs


@pytest.mark.parametrize(
    "tamper",
    ["nonmetadata_group", "duplicate_marker", "missing_marker"],
)
def test_hdf_rollback_rejects_incomplete_or_duplicate_snapshot_topology(
    tmp_path: Path,
    tamper: str,
) -> None:
    path = tmp_path / "image.h5"
    _write_v2_hdf(path)
    report = preflight_metadata_schema(path)
    applied = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert applied.receipt_path is not None
    receipt = json.loads(applied.receipt_path.read_text(encoding="utf-8"))
    snapshot = receipt["targets"][0]["hdf_snapshot"]
    marker = next(record for record in snapshot if record.get("marker"))
    if tamper == "nonmetadata_group":
        next(record for record in snapshot if "attributes" in record)["group"] = "/"
    elif tamper == "duplicate_marker":
        snapshot.append(dict(marker))
    else:
        snapshot.remove(marker)
    applied.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    migrated_before = file_fingerprint(path)

    rollback = rollback_metadata_migration(applied.receipt_path)

    assert rollback.status == "failed"
    assert file_fingerprint(path) == migrated_before
    with h5py.File(path, "r") as handle:
        assert CANONICAL_STRAIN in handle["metadata/public"].attrs
        assert LEGACY_STRAIN not in handle["metadata/public"].attrs


@pytest.mark.parametrize("tamper", ["delete_group", "omit_equal_canonical"])
def test_hdf_snapshot_is_bound_to_exact_preflight_group_state(
    tmp_path: Path,
    tamper: str,
) -> None:
    path = tmp_path / "image.h5"
    _write_v2_hdf(path)
    with h5py.File(path, "r+") as handle:
        public = handle["metadata/public"]
        if tamper == "delete_group":
            protected = handle.create_group("metadata/protected")
            protected.attrs[LEGACY_STRAIN] = public.attrs[LEGACY_STRAIN]
        else:
            public.attrs[CANONICAL_STRAIN] = public.attrs[LEGACY_STRAIN]
    report = preflight_metadata_schema(path)
    applied = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert applied.receipt_path is not None
    receipt = json.loads(applied.receipt_path.read_text(encoding="utf-8"))
    snapshot = receipt["targets"][0]["hdf_snapshot"]
    group_records = [record for record in snapshot if "attributes" in record]
    if tamper == "delete_group":
        snapshot.remove(group_records[-1])
    else:
        group_records[0]["attributes"].pop(CANONICAL_STRAIN)
    applied.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    migrated_before = file_fingerprint(path)

    rollback = rollback_metadata_migration(applied.receipt_path)

    assert rollback.status == "failed"
    assert "preflight binding" in rollback.conflicts[0]
    assert file_fingerprint(path) == migrated_before


@pytest.mark.skipif(os.name != "posix", reason="symlink security contract")
@pytest.mark.parametrize("directory_name", ["hdf", "measurements"])
@pytest.mark.parametrize("broken", [False, True])
def test_bundle_rejects_symlinked_authoritative_dataset_directories(
    tmp_path: Path,
    directory_name: str,
    broken: bool,
) -> None:
    output = tmp_path / "out"
    deliverables = output / "deliverables"
    dataset = output / "results" / "dataset"
    deliverables.mkdir(parents=True)
    dataset.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    external = tmp_path / f"external-{directory_name}"
    if not broken:
        external.mkdir()
        if directory_name == "hdf":
            _write_v2_hdf(external / "plate.h5")
        else:
            pd.DataFrame({LEGACY_STRAIN: ["WT"]}).to_parquet(
                external / "plate.parquet", index=False
            )
    (dataset / directory_name).symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match=f"(?i){directory_name}.*symlink"):
        preflight_metadata_schema(output)


def test_prepared_unpublished_bundle_rollback_clears_temp_and_reapplies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    output = tmp_path / "out"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    measurement = output / "results" / "dataset" / "measurements" / "plate.parquet"
    measurement.parent.mkdir(parents=True)
    pd.DataFrame({LEGACY_STRAIN: ["WT"], "Size_Area": [2.0]}).to_parquet(
        measurement, index=False
    )
    report = preflight_metadata_schema(output)
    real_publish = migration._publish_temp

    def fail_before_replace(_temp: Path, _target: Path) -> None:
        raise OSError("injected before target replace")

    monkeypatch.setattr(migration, "_publish_temp", fail_before_replace)
    failed = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert failed.status == "failed"
    assert failed.receipt_path is not None
    prepared = json.loads(failed.receipt_path.read_text(encoding="utf-8"))
    assert prepared["targets"][0]["state"] == "prepared"
    assert prepared["targets"][0]["temp_path"] is not None

    rolled_back = rollback_metadata_migration(failed.receipt_path)

    assert rolled_back.status == "rolled_back"
    journal = json.loads(failed.receipt_path.read_text(encoding="utf-8"))
    assert journal["targets"][0]["state"] == "rolled_back"
    assert journal["targets"][0]["temp_path"] is None
    assert LEGACY_STRAIN in pd.read_parquet(measurement).columns

    monkeypatch.setattr(migration, "_publish_temp", real_publish)
    reapplied = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert reapplied.status == "applied", reapplied.conflicts
    assert CANONICAL_STRAIN in pd.read_parquet(measurement).columns


def test_hdf_bundle_rollback_is_idempotent_reapplicable_and_change_guarded(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    hdf = output / "results" / "dataset" / "hdf" / "plate.h5"
    hdf.parent.mkdir(parents=True)
    _write_v2_hdf(hdf)
    report = preflight_metadata_schema(output)
    applied = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert applied.status == "applied"
    assert applied.receipt_path is not None

    first_rollback = rollback_metadata_migration(applied.receipt_path)
    assert first_rollback.status == "rolled_back"
    receipt = json.loads(applied.receipt_path.read_text(encoding="utf-8"))
    target = receipt["targets"][0]
    assert target["state"] == "rolled_back"
    assert target["rollback_fingerprint"] == file_fingerprint(hdf)
    assert target["hdf_snapshot_fingerprint"] is not None

    second_rollback = rollback_metadata_migration(applied.receipt_path)
    assert second_rollback.status == "rolled_back"
    reapplied = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert reapplied.status == "applied", reapplied
    with h5py.File(hdf, "r") as handle:
        assert CANONICAL_STRAIN in handle["metadata/public"].attrs
        assert LEGACY_STRAIN not in handle["metadata/public"].attrs

    rerolled = rollback_metadata_migration(applied.receipt_path)
    assert rerolled.status == "rolled_back"
    with h5py.File(hdf, "r+") as handle:
        handle.attrs["external_change"] = "reject"
    changed = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert changed.status == "failed"
    assert any("fingerprint changed" in conflict for conflict in changed.conflicts)


def test_version_two_receipt_is_explicitly_rejected_without_hdf_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.h5"
    _write_v2_hdf(path)
    report = preflight_metadata_schema(path)
    applied = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    assert applied.receipt_path is not None
    receipt = json.loads(applied.receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = 2
    applied.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    before = file_fingerprint(path)

    rollback = rollback_metadata_migration(applied.receipt_path)

    assert rollback.status == "failed"
    assert "Unsupported metadata migration receipt schema" in rollback.conflicts
    assert file_fingerprint(path) == before


def test_direct_hdf_rollback_is_idempotent_reused_and_change_guarded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.h5"
    _write_v2_hdf(path)
    report = preflight_metadata_schema(path)
    original_fingerprint = report.targets[0].source_fingerprint
    applied = migrate_metadata_file(
        path, expected_source_fingerprint=original_fingerprint
    )
    assert applied.status == "applied"
    assert applied.receipt_path is not None

    first_rollback = rollback_metadata_migration(applied.receipt_path)
    assert first_rollback.status == "rolled_back"
    second_rollback = rollback_metadata_migration(applied.receipt_path)
    assert second_rollback.status == "rolled_back"
    reapplied = migrate_metadata_file(
        path, expected_source_fingerprint=original_fingerprint
    )
    assert reapplied.status == "applied", reapplied.conflicts
    assert reapplied.receipt_path == applied.receipt_path

    rerolled = rollback_metadata_migration(applied.receipt_path)
    assert rerolled.status == "rolled_back"
    with h5py.File(path, "r+") as handle:
        handle.attrs["external_change"] = "reject"
    changed_fingerprint = file_fingerprint(path)
    blocked = migrate_metadata_file(
        path, expected_source_fingerprint=original_fingerprint
    )
    assert blocked.status in {"blocked", "failed"}
    assert file_fingerprint(path) == changed_fingerprint


@pytest.mark.skipif(os.name != "posix", reason="symlink security contract")
def test_bundle_journal_symlink_is_rejected_before_external_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    measurement = output / "results" / "dataset" / "measurements" / "plate.parquet"
    measurement.parent.mkdir(parents=True)
    pd.DataFrame({LEGACY_STRAIN: ["WT"], "Size_Area": [2.0]}).to_parquet(
        measurement, index=False
    )
    external = tmp_path / "external-state"
    external.mkdir()
    victim = external / "victim.txt"
    victim.write_text("do-not-touch", encoding="utf-8")
    (output / ".phenotypic").symlink_to(external, target_is_directory=True)
    report = preflight_metadata_schema(output)

    with pytest.raises(ValueError, match="symlink component"):
        migrate_metadata_bundle(
            output, expected_plan_fingerprint=report.plan_fingerprint
        )

    assert victim.read_text(encoding="utf-8") == "do-not-touch"
    assert not (external / "metadata_migration").exists()
    assert pd.read_parquet(measurement).columns.tolist() == [
        LEGACY_STRAIN,
        "Size_Area",
    ]


def test_full_bundle_migrates_individual_measurements_and_root_pipeline(
    tmp_path: Path,
) -> None:
    from phenotypic import ImagePipeline
    from phenotypic.post import AppendString

    output = tmp_path / "out"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    master = deliverables / "master_measurements.parquet"
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(master, index=False)
    measurement_dir = output / "results" / "dataset" / "measurements"
    measurement_dir.mkdir(parents=True)
    individual = measurement_dir / "plate.parquet"
    aggregate = measurement_dir / "_dataset_aggregated.parquet"
    pd.DataFrame(
        {LEGACY_STRAIN: ["WT"], "Size_Area": [2.0], "Object_Label": [1]}
    ).to_parquet(individual, index=False)
    pd.DataFrame({LEGACY_STRAIN: ["derived"]}).to_parquet(aggregate, index=False)
    aggregate_before = file_fingerprint(aggregate)
    external = deliverables / "metadata.csv"
    external.write_text(f"{LEGACY_STRAIN}\nexternal\n", encoding="utf-8")
    external_before = file_fingerprint(external)
    pipeline = output / "historical_pipeline.json"
    payload = json.loads(
        ImagePipeline(post=[AppendString(column="Strain", value="-x")]).to_json()
        or "{}"
    )
    payload["post"]["AppendString"]["params"]["column"] = LEGACY_STRAIN
    pipeline.write_text(json.dumps(payload), encoding="utf-8")
    (output / "processing_state.json").write_text(
        json.dumps({"pipeline_path": "/old/location/historical_pipeline.json"}),
        encoding="utf-8",
    )
    report = preflight_metadata_schema(output)
    target_paths = {Path(target.path) for target in report.targets}

    assert individual in target_paths
    assert pipeline in target_paths
    assert aggregate not in target_paths
    result = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )

    assert result.status == "applied"
    assert pd.read_parquet(individual).columns.tolist() == [
        CANONICAL_STRAIN,
        "Size_Area",
        "Object_Label",
    ]
    assert file_fingerprint(aggregate) == aggregate_before
    assert file_fingerprint(external) == external_before
    assert (
        json.loads(pipeline.read_text(encoding="utf-8"))["post"]["AppendString"][
            "params"
        ]["column"]
        == CANONICAL_STRAIN
    )
    assert result.receipt_path is not None
    rollback = rollback_metadata_migration(result.receipt_path)
    assert rollback.status == "rolled_back"
    assert LEGACY_STRAIN in pd.read_parquet(individual).columns
    assert (
        json.loads(pipeline.read_text(encoding="utf-8"))["post"]["AppendString"][
            "params"
        ]["column"]
        == LEGACY_STRAIN
    )
    reapplied = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert reapplied.status == "applied"
    assert CANONICAL_STRAIN in pd.read_parquet(individual).columns


def test_full_bundle_migrates_sole_aggregate_measurement_source(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    aggregate = (
        output
        / "results"
        / "dataset"
        / "measurements"
        / "_dataset_aggregated.parquet"
    )
    aggregate.parent.mkdir(parents=True)
    pd.DataFrame({LEGACY_STRAIN: ["WT"], "Size_Area": [2.0]}).to_parquet(
        aggregate, index=False
    )
    report = preflight_metadata_schema(output)

    assert Path(report.targets[0].path) == aggregate
    result = migrate_metadata_bundle(
        output, expected_plan_fingerprint=report.plan_fingerprint
    )

    assert result.status == "applied"
    assert pd.read_parquet(aggregate).columns.tolist() == [
        CANONICAL_STRAIN,
        "Size_Area",
    ]
    repeated = preflight_metadata_schema(output)
    assert repeated.status == "compatible"
    assert (
        migrate_metadata_bundle(
            output, expected_plan_fingerprint=repeated.plan_fingerprint
        ).status
        == "compatible"
    )


def test_standalone_mixed_master_preserves_nonmetadata_columns(tmp_path: Path) -> None:
    master = tmp_path / "master_measurements.parquet"
    pd.DataFrame(
        {
            LEGACY_STRAIN: ["legacy"],
            BARE_IMAGE: ["plate.tif"],
            "Size_Area": [12.5],
            "Object_Label": [1],
            "CustomScore": [0.75],
        }
    ).to_parquet(master, index=False)
    report = preflight_metadata_schema(tmp_path)

    result = migrate_metadata_bundle(
        tmp_path, expected_plan_fingerprint=report.plan_fingerprint
    )

    assert result.status == "applied"
    assert pd.read_parquet(master).columns.tolist() == [
        CANONICAL_STRAIN,
        CANONICAL_IMAGE,
        "Size_Area",
        "Object_Label",
        "CustomScore",
    ]


def test_resume_revalidates_compatible_skipped_bundle_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    deliverables = tmp_path / "deliverables"
    deliverables.mkdir()
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        deliverables / "master_measurements.parquet", index=False
    )
    from phenotypic import ImagePipeline

    pipeline = deliverables / "pipeline.json.pht-pipe"
    pipeline.write_text(ImagePipeline(name="before").to_json() or "{}", encoding="utf-8")
    hdf = tmp_path / "results" / "dataset" / "hdf" / "plate.h5"
    hdf.parent.mkdir(parents=True)
    _write_v2_hdf(hdf)
    report = preflight_metadata_schema(tmp_path)
    real_publish = migration._publish_temp

    def fail_publish(_temp: Path, _target: Path) -> None:
        raise OSError("injected before replace")

    monkeypatch.setattr(migration, "_publish_temp", fail_publish)
    failed = migrate_metadata_bundle(
        tmp_path, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert failed.status == "failed"
    pipeline.write_text(ImagePipeline(name="after").to_json() or "{}", encoding="utf-8")

    monkeypatch.setattr(migration, "_publish_temp", real_publish)
    resumed = migrate_metadata_bundle(
        tmp_path, expected_plan_fingerprint=report.plan_fingerprint
    )
    assert resumed.status == "failed"
    assert any("fingerprint changed" in item for item in resumed.conflicts)


@pytest.mark.skipif(os.name != "posix", reason="POSIX stat and directory fsync contract")
def test_table_publication_preserves_stat_and_fsyncs_all_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    path = tmp_path / "metadata.csv"
    path.write_text(f"{LEGACY_STRAIN}\nBY4741\n", encoding="utf-8")
    path.chmod(0o640)
    timestamp_ns = 1_700_000_000_123_456_789
    os.utime(path, ns=(timestamp_ns, timestamp_ns))
    before_stat = path.stat()
    synced: list[Path] = []
    real_fsync_directory = migration._fsync_directory

    def record_fsync(directory: Path) -> None:
        synced.append(Path(directory))
        real_fsync_directory(directory)

    monkeypatch.setattr(migration, "_fsync_directory", record_fsync)
    report = preflight_metadata_schema(path)
    result = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )

    assert result.status == "applied"
    after_stat = path.stat()
    assert stat.S_IMODE(after_stat.st_mode) == stat.S_IMODE(before_stat.st_mode)
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert result.receipt_path is not None
    backup_dir = result.receipt_path.parent / "backups"
    assert path.parent in synced
    assert result.receipt_path.parent in synced
    assert backup_dir in synced

    sync_count = len(synced)
    rollback = rollback_metadata_migration(result.receipt_path)
    assert rollback.status == "rolled_back"
    assert len(synced) > sync_count
    assert path.parent in synced[sync_count:]


def test_canonical_file_is_idempotent_noop(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    path.write_text(f"{CANONICAL_STRAIN}\nBY4741\n", encoding="utf-8")
    before = file_fingerprint(path)
    report = preflight_metadata_schema(path)

    first = migrate_metadata_file(
        path, expected_source_fingerprint=report.targets[0].source_fingerprint
    )
    second_report = preflight_metadata_schema(path)
    second = migrate_metadata_file(
        path, expected_source_fingerprint=second_report.targets[0].source_fingerprint
    )

    assert first.status == second.status == "compatible"
    assert file_fingerprint(path) == before
    assert first.receipt_path is None
