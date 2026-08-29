"""``--mode migrate`` CLI contract.

Three defects in an earlier draft of this block, all corrected here:

1. ``main`` is not a symbol in :mod:`phenotypic.phenotypicCLI`; the click
   command is ``phenotypic_cli``.
2. There is no ``cli_runner`` fixture -- every existing CLI test constructs
   ``CliRunner()`` inline, and this follows that.
3. ``--pipeline`` and ``--input`` are both ``click.Path(exists=True)``, so a
   test handing click a non-existent path exits **2 during parsing** and the
   mode guard under test never runs. The paths below exist on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli


def test_migrate_is_an_accepted_mode() -> None:
    """``--help`` alone proves nothing -- it is an EAGER click option.

    ``["--mode", "migrate", "--help"]`` exits 0 before ``--mode`` is ever
    validated, so that invocation passed while ``migrate`` was still absent
    from the ``Choice``. The listing below is what actually moves.
    """
    result = CliRunner().invoke(phenotypic_cli, ["--help"])
    assert result.exit_code == 0
    assert "migrate" in result.output

    from phenotypic.phenotypicCLI import phenotypic_cli as command

    mode = next(p for p in command.params if p.name == "mode")
    assert "migrate" in mode.type.choices


def test_the_mode_help_states_both_passes() -> None:
    """The two-pass shape is a user-facing contract (user ruling, MIG-7b)."""
    from phenotypic.phenotypicCLI import phenotypic_cli as command

    mode = next(p for p in command.params if p.name == "mode")
    help_text = (mode.help or "").lower()
    assert "two passes" in help_text or "two-pass" in help_text, help_text


def test_migrate_rejects_pipeline_and_input(tmp_path: Path, legacy_run) -> None:
    """Same validation as recompile: the tree is named by --output alone.

    Both flags are ``click.Path(exists=True)``, so the arguments must exist on
    disk or click exits 2 while parsing and the mode guard never runs.
    """
    pipeline = tmp_path / "p.json"
    pipeline.write_text("{}", encoding="utf-8")
    images = tmp_path / "imgs"
    images.mkdir()

    for flag, value in (("--pipeline", pipeline), ("--input", images)):
        result = CliRunner().invoke(
            phenotypic_cli,
            ["--mode", "migrate", "--output", str(legacy_run), flag, str(value)],
        )
        assert result.exit_code != 0
        # The exit CODE cannot tell the two apart: the mode guard raises
        # click.UsageError, which also exits 2. The guard's own message is the
        # discriminator, and click's parse error for a missing path
        # ("does not exist") never contains it.
        assert f"--mode migrate does not accept {flag}" in result.output, (
            result.output
        )


def test_migrate_accepts_dry_run(legacy_run) -> None:
    """recompile REJECTS --dry-run; migrate must be exempt from that guard.

    ``--dry-run`` is a required part of migrate -- the spec's interface line,
    ``migrate_run_hdf_to_zarr(dry_run=...)``, and a phase exit criterion all
    depend on it. Folding migrate into recompile's guard rejects it (FLOW-34).
    """
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"]
    )
    assert result.exit_code == 0
    assert "--dry-run cannot be combined" not in result.output


def test_migration_is_in_place(legacy_run) -> None:
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import valid_staged_store

    assert (
        CliRunner()
        .invoke(phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)])
        .exit_code
        == 0
    )
    assert valid_staged_store(zarr_store_path(legacy_run, "ds", "img"))


def test_dot_prefixed_hdfs_are_ignored_and_never_deleted(legacy_run) -> None:
    """AppleDouble and other dotfiles are not image migration inputs."""
    from phenotypic.sdk_ import datasets_needing_migration
    from phenotypic.sdk_._hdf_to_zarr import iter_legacy_hdfs

    hdf_dir = legacy_run / "results" / "ds" / "hdf"
    sidecar = hdf_dir / "._img.h5"
    sidecar.write_bytes(b"AppleDouble metadata, not HDF5")
    hidden = hdf_dir / ".hidden.h5"
    hidden.write_bytes(b"hidden metadata, not HDF5")

    dry_run = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "would convert 1, skipped 0" in dry_run.output
    assert [path.name for _, path in iter_legacy_hdfs(legacy_run)] == [
        "img.h5"
    ]

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--delete-sources",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sidecar.read_bytes() == b"AppleDouble metadata, not HDF5"
    assert hidden.read_bytes() == b"hidden metadata, not HDF5"
    assert datasets_needing_migration(legacy_run) == []


def test_migration_leaves_the_rest_of_the_tree_where_it_was(legacy_run) -> None:
    """In place means in place: only ``zarr/`` and the derived view appear."""
    before = {
        path.relative_to(legacy_run).as_posix()
        for path in legacy_run.rglob("*")
        if path.is_file()
    }
    CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    )
    after = {
        path.relative_to(legacy_run).as_posix()
        for path in legacy_run.rglob("*")
        if path.is_file()
    }
    assert not (before - after), f"migration removed {sorted(before - after)}"
    added = {path.split("/")[0] for path in after - before}
    assert added <= {"results", "deliverables", ".phenotypic"}, sorted(added)


def test_sources_are_retained_unless_delete_sources_is_passed(legacy_run) -> None:
    """MIG-9: --delete-sources is the only path to keep_source=False."""
    hdf = legacy_run / "results" / "ds" / "hdf"

    assert (
        CliRunner()
        .invoke(phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)])
        .exit_code
        == 0
    )
    assert list(hdf.glob("*.h5")), "retained by default"

    assert (
        CliRunner()
        .invoke(
            phenotypic_cli,
            ["--mode", "migrate", "--output", str(legacy_run), "--delete-sources"],
        )
        .exit_code
        == 0
    )
    assert not list(hdf.glob("*.h5"))


def test_delete_sources_refuses_when_the_re_read_diverges(
    legacy_run, monkeypatch
) -> None:
    """MIG-20: a lossy conversion can still be structurally valid, so the
    precondition for the one irreversible step must re-read and compare."""
    from phenotypic.sdk_ import _hdf_to_zarr

    monkeypatch.setattr(_hdf_to_zarr, "_conversion_is_faithful", lambda *a, **k: False)
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--delete-sources"],
    )
    assert result.exit_code != 0
    assert list((legacy_run / "results" / "ds" / "hdf").glob("*.h5")), "nothing unlinked"


def test_the_faithfulness_gate_catches_an_altered_metadata_VALUE(
    legacy_run,
) -> None:
    """The MIG-2 shape: a key that is PRESENT and carries the wrong value.

    A key-set comparison sees two identical key sets and returns ``True``; the
    ``.h5`` is then unlinked and the correct value is gone permanently. This
    mutates the store's metadata after conversion and asserts the gate refuses.
    """
    import json

    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_._hdf_to_zarr import (
        _conversion_is_faithful,
        migrate_run_hdf_to_zarr,
    )
    from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON, PhenotypicAttr

    migrate_run_hdf_to_zarr(legacy_run)
    source = legacy_run / "results" / "ds" / "hdf" / "img.h5"
    store = zarr_store_path(legacy_run, "ds", "img")
    assert _conversion_is_faithful(source, store) is True

    root = store / STORE_ROOT_JSON
    payload = json.loads(root.read_text(encoding="utf-8"))
    block = payload["attributes"][PhenotypicAttr.ROOT]
    block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"] = "Crop"
    root.write_text(json.dumps(payload), encoding="utf-8")

    assert _conversion_is_faithful(source, store) is False


def test_the_faithfulness_gate_catches_altered_PIXELS(legacy_run) -> None:
    """Shapes and dtypes are blind to content: a zeroed layer has both."""
    import numpy as np

    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_._hdf_to_zarr import (
        _conversion_is_faithful,
        migrate_run_hdf_to_zarr,
    )
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    import zarr

    migrate_run_hdf_to_zarr(legacy_run)
    source = legacy_run / "results" / "ds" / "hdf" / "img.h5"
    store = zarr_store_path(legacy_run, "ds", "img")
    assert _conversion_is_faithful(source, store) is True

    member = read_phenotypic_attributes(store)[PhenotypicAttr.SERIES]["gray"]
    handle = zarr.open_array(store=str(store / member / "0"), mode="r+")
    handle[...] = np.zeros_like(handle[...])

    assert _conversion_is_faithful(source, store) is False


def test_migrate_converts_a_legacy_tree(legacy_run) -> None:
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import valid_staged_store

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    )
    assert result.exit_code == 0
    assert valid_staged_store(zarr_store_path(legacy_run, "ds", "img"))


def test_migrate_never_submits_a_slurm_job(legacy_run, monkeypatch) -> None:
    """One-time, resumable work does not justify another scheduler surface."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("migrate must not shell out")
    )
    assert (
        CliRunner()
        .invoke(phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)])
        .exit_code
        == 0
    )


def test_a_legacy_only_output_fails_with_a_pointer(legacy_format_run) -> None:
    """Conversion rewrites the whole results tree; it must be typed deliberately."""
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "recompile", "--output", str(legacy_format_run)],
    )
    assert result.exit_code != 0
    assert "--mode migrate" in result.output


def test_dry_run_reports_without_writing(legacy_run) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"],
    )
    assert result.exit_code == 0
    assert not dataset_zarr_dir(legacy_run, "ds").exists()


def test_dry_run_reports_a_count_for_EACH_pass(legacy_run) -> None:
    """Phase 5 exit criterion 2. A summary naming only pass 2 gates nothing."""
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "pass 1" in result.output.lower()
    assert "pass 2" in result.output.lower()


def test_dry_run_writes_no_receipt(legacy_run) -> None:
    """Pass 1's dry run is free because ``preflight_metadata_schema`` writes
    nothing -- not incidental, so it is asserted."""
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"],
    )
    # Without this the test passes on a CLI that never ran at all.
    assert result.exit_code == 0, result.output
    receipts = legacy_run / ".phenotypic" / "metadata_migration"
    assert not receipts.exists() or not list(receipts.glob("*.json"))


# ---------------------------------------------------------------------------
# Pass 1's scope
# ---------------------------------------------------------------------------


def test_metadata_pass_reuses_preflight_and_revalidates_before_first_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Semantic parsing is once-only; fresh path/byte checks guard mutation."""
    import phenotypic.sdk_._metadata_migration as migration
    from phenotypic._cli._cli_migrate import run_metadata_pass

    output = tmp_path / "output"
    (output / "deliverables").mkdir(parents=True)
    measurements = output / "results" / "dataset" / "measurements"
    measurements.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame({"MetadataGenetic_Strain": ["WT"]}).to_parquet(
        measurements / "_dataset_aggregated.parquet", index=False
    )
    events: list[tuple[str, str | None]] = []
    real_preflight_file = migration._preflight_file
    real_discover = migration._discover_bundle_targets
    real_fingerprint = migration.file_fingerprint
    real_publish_anchored_journal_json = migration._publish_anchored_journal_json

    def record_preflight(path: Path, *, mixed_table: bool = False):
        events.append(("semantic", str(path)))
        return real_preflight_file(path, mixed_table=mixed_table)

    def record_discovery(*args: object, **kwargs: object):
        events.append(("discover", None))
        return real_discover(*args, **kwargs)

    def record_fingerprint(path: Path) -> str:
        events.append(("fingerprint", str(path)))
        return real_fingerprint(path)

    def record_first_durable_publication(*args: object, **kwargs: object):
        events.append(("mutation", str(args[0])))
        return real_publish_anchored_journal_json(*args, **kwargs)

    monkeypatch.setattr(migration, "_preflight_file", record_preflight)
    monkeypatch.setattr(migration, "_discover_bundle_targets", record_discovery)
    monkeypatch.setattr(migration, "file_fingerprint", record_fingerprint)
    monkeypatch.setattr(
        migration,
        "_publish_anchored_journal_json",
        record_first_durable_publication,
    )

    result = run_metadata_pass(output, dry_run=False)

    assert result.authority is not None
    first_mutation = next(
        index for index, event in enumerate(events) if event[0] == "mutation"
    )
    pre_mutation = events[:first_mutation]
    semantic_paths = [value for name, value in pre_mutation if name == "semantic"]
    assert semantic_paths
    assert len(semantic_paths) == len(set(semantic_paths))
    discovery_indices = [
        index for index, event in enumerate(pre_mutation) if event[0] == "discover"
    ]
    assert len(discovery_indices) == 2
    authoritative = pre_mutation[discovery_indices[-1] + 1 :]
    assert not [event for event in authoritative if event[0] == "semantic"]
    assert [value for name, value in authoritative if name == "fingerprint"] == semantic_paths


def test_dry_metadata_pass_returns_count_without_authority_or_publication(
    tmp_path: Path,
) -> None:
    import pandas as pd

    from phenotypic._cli._cli_migrate import run_metadata_pass

    output = tmp_path / "output"
    (output / "deliverables").mkdir(parents=True)
    measurements = output / "results" / "dataset" / "measurements"
    measurements.mkdir(parents=True)
    pd.DataFrame({"MetadataGenetic_Strain": ["WT"]}).to_parquet(
        measurements / "_dataset_aggregated.parquet", index=False
    )

    result = run_metadata_pass(output, dry_run=True)

    assert result.headers_migrated == 1
    assert result.failures == ()
    assert result.authority is None
    assert not (output / ".phenotypic").exists()


def test_pass_one_targets_contain_no_hdf(legacy_run) -> None:
    """MIG-25/FLOW-35: pass 1 must not touch a ``.h5`` at all."""
    from phenotypic.sdk_ import BundleLayout, deliverables_dir
    from phenotypic.sdk_._metadata_migration import (
        NON_IMAGE_KINDS,
        _discover_bundle_targets,
        _discover_legacy_bundle_targets,
    )

    layout = BundleLayout(
        deliverables_base=deliverables_dir(legacy_run), output_root=legacy_run
    )
    unfiltered = _discover_legacy_bundle_targets(layout)
    assert any(path.suffix == ".h5" for path in unfiltered), (
        "fixture must contain a .h5 or this proves nothing"
    )
    filtered = _discover_bundle_targets(layout, kinds=NON_IMAGE_KINDS)
    assert not any(path.suffix == ".h5" for path in filtered), filtered
    assert filtered, "pass 1 must still have non-image targets"


def test_bundle_durable_pass_never_scans_per_image_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass 1 owns the named aggregate, never Task-1 image Parquets."""
    import pandas as pd

    from phenotypic.sdk_ import (
        DATASET_AGGREGATED_PARQUET,
        BundleLayout,
        deliverables_dir,
    )
    from phenotypic.sdk_._metadata_migration import (
        BUNDLE_DURABLE_TARGET_ROLE,
        NON_IMAGE_KINDS,
        preflight_metadata_schema,
    )

    output = tmp_path / "output"
    deliverables = deliverables_dir(output)
    deliverables.mkdir(parents=True)
    (deliverables / "pipeline.json").write_text(
        '{"MetadataGenetic_Strain":"WT"}', encoding="utf-8"
    )
    measurements = output / "results" / "dataset" / "measurements"
    measurements.mkdir(parents=True)
    image_source = measurements / "plate.parquet"
    aggregate = measurements / DATASET_AGGREGATED_PARQUET
    pd.DataFrame({"MetadataGenetic_Strain": ["image"]}).to_parquet(
        image_source, index=False
    )
    pd.DataFrame({"MetadataGenetic_Strain": ["aggregate"]}).to_parquet(
        aggregate, index=False
    )
    real_iterdir = Path.iterdir
    real_glob = Path.glob

    def refuse_measurement_iterdir(path: Path):
        if path == measurements:
            raise AssertionError("per-image measurement directory was entered")
        return real_iterdir(path)

    def refuse_measurement_glob(path: Path, pattern: str):
        if path == measurements:
            raise AssertionError("per-image measurement directory was opened")
        return real_glob(path, pattern)

    monkeypatch.setattr(Path, "iterdir", refuse_measurement_iterdir)
    monkeypatch.setattr(Path, "glob", refuse_measurement_glob)
    report = preflight_metadata_schema(
        BundleLayout(deliverables_base=deliverables, output_root=output),
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )

    paths = {Path(target.path) for target in report.targets}
    assert aggregate in paths
    assert image_source not in paths
    assert report.target_role == BUNDLE_DURABLE_TARGET_ROLE


def test_new_receipt_persists_bundle_durable_role_and_excludes_image_sources(
    tmp_path: Path,
) -> None:
    import json

    import pandas as pd

    from phenotypic.sdk_ import BundleLayout, deliverables_dir
    from phenotypic.sdk_._metadata_migration import (
        BUNDLE_DURABLE_TARGET_ROLE,
        NON_IMAGE_KINDS,
        migrate_preflighted_metadata_bundle,
        preflight_metadata_schema,
    )

    output = tmp_path / "output"
    deliverables = deliverables_dir(output)
    deliverables.mkdir(parents=True)
    measurements = output / "results" / "dataset" / "measurements"
    measurements.mkdir(parents=True)
    image_source = measurements / "plate.parquet"
    pd.DataFrame({"MetadataGenetic_Strain": ["image"]}).to_parquet(
        image_source, index=False
    )
    (deliverables / "pipeline.json").write_text(
        '{"MetadataGenetic_Strain":"WT"}', encoding="utf-8"
    )
    layout = BundleLayout(deliverables_base=deliverables, output_root=output)
    report = preflight_metadata_schema(
        layout,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )

    result = migrate_preflighted_metadata_bundle(
        layout, report=report, kinds=NON_IMAGE_KINDS
    )

    assert result.receipt_path is not None
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 4
    assert receipt["target_role"] == BUNDLE_DURABLE_TARGET_ROLE
    assert image_source not in {Path(item["path"]) for item in receipt["targets"]}


def test_a_filtered_receipt_survives_validation_on_a_tree_with_hdfs(
    legacy_run,
) -> None:
    """The test that would have caught MIG-26/C10.

    ``_validate_receipt`` re-derives the target set to prove the receipt is
    authoritative. Re-deriving the UNFILTERED set against a scoped receipt
    makes it raise on every tree holding a single ``.h5`` -- which is every
    tree being migrated.
    """
    from phenotypic.sdk_ import BundleLayout, deliverables_dir
    from phenotypic.sdk_._metadata_migration import (
        NON_IMAGE_KINDS,
        migrate_metadata_bundle,
        preflight_metadata_schema,
    )

    assert list((legacy_run / "results").rglob("*.h5")), "fixture has no .h5"
    layout = BundleLayout(
        deliverables_base=deliverables_dir(legacy_run), output_root=legacy_run
    )
    report = preflight_metadata_schema(layout, kinds=NON_IMAGE_KINDS)
    result = migrate_metadata_bundle(
        layout,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )
    assert result.status in {"compatible", "applied"}, result.status


def test_migration_never_rewrites_a_source_hdf(legacy_run) -> None:
    """The rollback story: after migration the ``.h5`` files are the ORIGINALS.

    Pass 1's apply path for a ``.h5`` is ``_migrate_hdf_copy`` -- a full
    ``shutil.copy2`` followed by an attribute rewrite and a rename. If pass 1
    ever stops excluding them, every ``.h5`` in the archive is rewritten before
    a single store exists.
    """
    import hashlib

    hdf_dir = legacy_run / "results" / "ds" / "hdf"
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(hdf_dir.glob("*.h5"))
    }
    assert before, "fixture has no .h5"
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    )
    # Without this the test passes on a CLI that never ran at all.
    assert result.exit_code == 0, result.output
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(hdf_dir.glob("*.h5"))
    }
    assert after == before


def test_running_migrate_twice_converts_zero_and_migrates_zero_headers(
    legacy_run,
) -> None:
    """A per-image source embeds once without a durable header rewrite."""
    import polars as pl

    from phenotypic._cli._cli_migrate import run_migrate
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, zarr_store_path

    source = legacy_run / "results" / "ds" / "measurements" / "img.parquet"
    source_bytes = source.read_bytes()
    assert "MetadataGenetic_Strain" in pl.read_parquet(source).columns

    first = run_migrate(legacy_run, njobs=1, dry_run=False, delete_sources=False)
    second = run_migrate(legacy_run, njobs=1, dry_run=False, delete_sources=False)
    assert first.ok
    assert first.converted > 0
    assert first.headers_migrated == 0
    assert first.tables_migrated > 0
    embedded_columns = pl.read_parquet(
        zarr_store_path(legacy_run, "ds", "img")
        / MEASUREMENT_TABLE_RELATIVE_PATH
    ).columns
    assert "MetadataGenetic_Strain" not in embedded_columns
    assert "Metadata_Strain" in embedded_columns
    assert source.read_bytes() == source_bytes
    assert second.converted == 0
    assert second.headers_migrated == 0
    assert second.tables_migrated == 0
    assert second.overlays_created == 0
    assert second.ok
    assert source.read_bytes() == source_bytes


def test_pass_three_embeds_canonical_measurement_headers_without_rewriting_source(
    legacy_run,
) -> None:
    """Pass 3 embeds canonical headers without rewriting retained sources."""
    import polars as pl

    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_measurements_dir,
        zarr_store_path,
    )
    from phenotypic._cli._cli_migrate import run_migrate

    measurements = dataset_measurements_dir(legacy_run, "ds")
    parquets = sorted(measurements.glob("*.parquet"))
    assert parquets, "fixture has no measurements"
    source = next(path for path in parquets if not path.name.startswith("_"))
    source_bytes = source.read_bytes()
    source_columns = set(pl.read_parquet(source).columns)
    assert "MetadataGenetic_Strain" in source_columns, (
        "fixture is already canonical, so pass 1 would be a no-op"
    )

    run_migrate(legacy_run, njobs=1, dry_run=False, delete_sources=False)

    assert source.read_bytes() == source_bytes
    embedded_columns = set(
        pl.read_parquet(
            zarr_store_path(legacy_run, "ds", source.stem)
            / MEASUREMENT_TABLE_RELATIVE_PATH
        ).columns
    )
    assert "MetadataGenetic_Strain" not in embedded_columns
    assert "Metadata_Strain" in embedded_columns


def test_a_blocked_preflight_aborts_before_anything_is_written(
    legacy_run, monkeypatch
) -> None:
    """Conflicts a human must resolve stop the run, not get migrated past.

    This coverage moves here from ``test_cli_recompile.py``, which asserted
    it against recompile's own rewrite -- the rewrite Task 5.4 removed.
    """
    from types import SimpleNamespace

    from phenotypic._cli import _cli_migrate
    from phenotypic.sdk_ import dataset_zarr_dir

    monkeypatch.setattr(
        _cli_migrate,
        "preflight_metadata_schema",
        lambda *a, **k: SimpleNamespace(
            status="blocked",
            conflicts=("legacy and canonical values disagree",),
            targets=(),
            plan_fingerprint="sha256:0",
        ),
    )
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    )
    assert result.exit_code != 0
    assert "legacy and canonical values disagree" in result.output
    # Pass 1 aborts, so pass 2 never runs and no store is written.
    assert not dataset_zarr_dir(legacy_run, "ds").exists()


def test_a_blocked_preflight_aborts_a_dry_run_too(
    legacy_run, monkeypatch
) -> None:
    """A dry run that reported "would convert N" on a blocked tree would be a
    plan the real run cannot execute."""
    from types import SimpleNamespace

    from phenotypic._cli import _cli_migrate

    monkeypatch.setattr(
        _cli_migrate,
        "preflight_metadata_schema",
        lambda *a, **k: SimpleNamespace(
            status="blocked",
            conflicts=("legacy and canonical values disagree",),
            targets=(),
            plan_fingerprint="sha256:0",
        ),
    )
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# The predicate, applied to every consumer (Task 5.7)
# ---------------------------------------------------------------------------


def test_full_mode_refuses_a_half_migrated_tree(half_migrated_run, tmp_path) -> None:
    """Without this, --mode full silently reprocesses from source.

    ``--pipeline`` and ``--input`` are ``click.Path(exists=True)``, so both
    must exist on disk or click exits 2 while parsing and the migration guard
    never runs. The exit CODE cannot separate the two -- the guard raises
    ``click.UsageError``, which also exits 2 -- so the guard's own text is the
    discriminator.
    """
    pipeline = tmp_path / "p.json"
    pipeline.write_text("{}", encoding="utf-8")
    images = tmp_path / "imgs"
    images.mkdir()

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "full",
            "--output",
            str(half_migrated_run),
            "--pipeline",
            str(pipeline),
            "--input",
            str(images),
        ],
    )
    assert result.exit_code != 0
    assert "--mode migrate" in result.output


@pytest.mark.parametrize("mode", ["full", "measure", "recompile", "process"])
def test_every_consuming_mode_refuses_a_half_migrated_tree(
    mode: str, half_migrated_run, tmp_path
) -> None:
    """Applied to every mode that WRITES or REPROCESSES, not recompile alone.

    After Phase 6 the forward path genuinely cannot read those images.
    """
    pipeline = tmp_path / "p.json"
    pipeline.write_text("{}", encoding="utf-8")
    images = tmp_path / "imgs"
    images.mkdir()

    args = ["--mode", mode, "--output", str(half_migrated_run)]
    if mode in {"full", "process"}:
        args += ["--pipeline", str(pipeline), "--input", str(images)]
    if mode == "measure":
        args += ["--pipeline", str(pipeline)]
    if mode == "process":
        args += ["--layer", "objmap"]

    result = CliRunner().invoke(phenotypic_cli, args)
    assert result.exit_code != 0
    assert "--mode migrate" in result.output, result.output


def test_migrate_itself_is_exempt(half_migrated_run) -> None:
    """MIG-19: it is the remedy. Guarding it makes the tree unmigratable."""
    from phenotypic.sdk_ import datasets_needing_migration

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(half_migrated_run)]
    )
    assert result.exit_code == 0, result.output
    assert datasets_needing_migration(half_migrated_run) == []


def test_the_viewer_surfaces_it(half_migrated_run) -> None:
    """Reported through the EXISTING consistency surface, not a new one.

    Same predicate, same reason text, **different severity**: a mode that
    writes refuses, while the viewer is informational. A half-migrated
    tree's deliverables, measurements and dashboards are all still readable,
    and the images that are missing are precisely the ones it would
    otherwise render empty.
    """
    from phenotypic.gui.results_viewer._output_consistency import (
        inspect_output_consistency,
    )
    from phenotypic.sdk_ import BundleLayout, deliverables_dir

    report = inspect_output_consistency(
        BundleLayout(
            deliverables_base=deliverables_dir(half_migrated_run),
            output_root=half_migrated_run,
        )
    )
    assert any("--mode migrate" in reason for reason in report.reasons), (
        report.reasons
    )


def test_the_viewer_says_nothing_about_a_fully_migrated_tree(
    migrated_run,
) -> None:
    from phenotypic.gui.results_viewer._output_consistency import (
        inspect_output_consistency,
    )
    from phenotypic.sdk_ import BundleLayout, deliverables_dir

    report = inspect_output_consistency(
        BundleLayout(
            deliverables_base=deliverables_dir(migrated_run),
            output_root=migrated_run,
        )
    )
    assert not any("--mode migrate" in reason for reason in report.reasons)
