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


def test_pass_one_targets_contain_no_hdf(legacy_run) -> None:
    """MIG-25/FLOW-35: pass 1 must not touch a ``.h5`` at all."""
    from phenotypic.sdk_ import BundleLayout, deliverables_dir
    from phenotypic.sdk_._metadata_migration import (
        NON_IMAGE_KINDS,
        _discover_bundle_targets,
    )

    layout = BundleLayout(
        deliverables_base=deliverables_dir(legacy_run), output_root=legacy_run
    )
    unfiltered = _discover_bundle_targets(layout)
    assert any(path.suffix == ".h5" for path in unfiltered), (
        "fixture must contain a .h5 or this proves nothing"
    )
    filtered = _discover_bundle_targets(layout, kinds=NON_IMAGE_KINDS)
    assert not any(path.suffix == ".h5" for path in filtered), filtered
    assert filtered, "pass 1 must still have non-image targets"


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
    """Phase 5 exit criterion 3.

    The header half holds because an already-canonical bundle short-circuits
    to ``_compatible_result``, **not** because of any per-image skip -- a
    criterion that would pass either way gates nothing (ledger MIG-31).
    """
    from phenotypic._cli._cli_migrate import run_migrate

    first = run_migrate(legacy_run, njobs=1, dry_run=False, delete_sources=False)
    second = run_migrate(legacy_run, njobs=1, dry_run=False, delete_sources=False)
    assert first.converted > 0
    # Without this the criterion passes on a run that never migrated a header
    # at all -- verified: skipping pass 1 entirely survived until it was here.
    assert first.headers_migrated > 0
    assert second.converted == 0
    assert second.headers_migrated == 0


def test_pass_one_canonicalizes_a_measurement_header(legacy_run) -> None:
    """Pass 1's observable effect, asserted on the bytes it rewrites.

    A count is not an effect: a driver that skipped pass 1 and reported zero
    would satisfy every count-shaped assertion in this file.
    """
    import polars as pl

    from phenotypic.sdk_ import dataset_measurements_dir
    from phenotypic._cli._cli_migrate import run_migrate

    measurements = dataset_measurements_dir(legacy_run, "ds")
    parquets = sorted(measurements.glob("*.parquet"))
    assert parquets, "fixture has no measurements"
    before = set(pl.read_parquet(parquets[0]).columns)
    assert "MetadataGenetic_Strain" in before, (
        "fixture is already canonical, so pass 1 would be a no-op"
    )

    run_migrate(legacy_run, njobs=1, dry_run=False, delete_sources=False)

    after = set(pl.read_parquet(parquets[0]).columns)
    assert "MetadataGenetic_Strain" not in after
    assert "Metadata_Strain" in after
