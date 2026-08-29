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

import hashlib
from pathlib import Path
from types import SimpleNamespace

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


def test_migration_help_names_all_reclaimed_sources_and_durability_rejection() -> None:
    """Destructive and inapplicable option help must match migration behavior."""
    from phenotypic.phenotypicCLI import phenotypic_cli as command

    delete_sources = next(param for param in command.params if param.name == "delete_sources")
    durable_writes = next(param for param in command.params if param.name == "durable_writes")

    assert "hdf" in (delete_sources.help or "").lower()
    assert "parquet" in (delete_sources.help or "").lower()
    assert "migrate" in (durable_writes.help or "").lower()


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


def test_migrate_without_slurm_uses_the_local_runner(
    legacy_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting --slurm must retain the established local migration path."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic.sdk_._hdf_to_zarr import MigrationReport

    calls: list[dict[str, object]] = []

    def _local(output_dir: Path, **kwargs: object) -> MigrationReport:
        calls.append({"output_dir": output_dir, **kwargs})
        return MigrationReport()

    monkeypatch.setattr(migrate, "run_migrate", _local)

    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_run)]
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["output_dir"] == legacy_run


def test_migrate_with_repeated_slurm_plans_and_submits_once(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SLURM flags select one planned scheduler attempt, never local science."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli._cli_migrate_slurm import MigrationSlurmPlan

    control = tmp_path / "control"
    control.mkdir()
    manifest = control / "migration_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    script = control / "metadata.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    finalizer = control / "finalize.sh"
    finalizer.write_text("#!/bin/bash\n", encoding="utf-8")
    plan = MigrationSlurmPlan(
        generation="attempt-1",
        control_root=control,
        manifest_path=manifest,
        flat_scripts=(script,),
        finalizer_script=finalizer,
        task_count=1,
    )
    planned: list[dict[str, object]] = []
    submitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        migrate,
        "run_migrate",
        lambda *_a, **_k: pytest.fail("SLURM dispatch ran local migration"),
    )
    monkeypatch.setattr(
        migrate,
        "new_slurm_generation",
        lambda: "attempt-1",
    )
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda output_dir, **kwargs: (
            planned.append({"output_dir": output_dir, **kwargs}) or plan
        ),
    )
    monkeypatch.setattr(
        migrate,
        "submit_migration_slurm_plan",
        lambda submitted_plan, **kwargs: (
            submitted.append({"plan": submitted_plan, **kwargs})
            or SimpleNamespace(job_ids=["101", "102"])
        ),
    )

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
            "--slurm",
            "time=30",
        ],
    )

    assert result.exit_code == 0, result.output
    assert planned == [
        {
            "output_dir": legacy_run,
            "slurm_args": {"slurm_partition": "short", "time": "00:30:00"},
            "overlay_alpha": 0.3,
            "delete_sources": False,
            "dry_run": False,
            "generation": "attempt-1",
        }
    ]
    assert submitted[0]["plan"] is plan
    assert "attempt-1" in result.output
    assert str(control) in result.output


@pytest.mark.parametrize("njobs", [-1, 1])
def test_explicit_njobs_is_rejected_before_slurm_migration_writes(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, njobs: int
) -> None:
    """Click provenance, not the numeric value, guards SLURM worker ownership."""
    from phenotypic._cli import _cli_migrate as migrate

    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda *_a, **_k: pytest.fail("invalid options wrote a plan"),
    )
    before = _tree_snapshot(legacy_run)

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
            "--njobs",
            str(njobs),
        ],
    )

    assert result.exit_code != 0
    assert "--njobs" in result.output
    assert _tree_snapshot(legacy_run) == before
    assert not cache.exists()


def test_slurm_migration_dry_run_preserves_every_output_byte(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduler preview owns only an external control tree, never science."""
    from phenotypic._cli import _cli_migrate_slurm as slurm

    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setattr(slurm, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(slurm, "get_slurm_max_submit_jobs", lambda: 100)
    before = _tree_snapshot(legacy_run)

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _tree_snapshot(legacy_run) == before
    controls = list(cache.rglob("migration_manifest.json"))
    assert len(controls) == 1
    control_root = controls[0].parent
    assert not control_root.is_relative_to(legacy_run)
    assert all(
        path.is_relative_to(control_root)
        for path in [
            controls[0],
            control_root / "migration_config.json",
            *(control_root / "scripts").glob("*.sh"),
            *(control_root / "logs").glob("*"),
        ]
    )


def test_waited_slurm_failure_is_reported_as_a_click_error_after_closure(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable failed terminal status must not be hidden as an unexpected error."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic.sdk_._hdf_to_zarr import MigrationReport

    plan = _migration_plan(tmp_path, "attempt-1")
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(
        migrate, "generate_migration_slurm_plan", lambda *_a, **_k: plan
    )

    def _submit(*_args, **_kwargs):
        report = MigrationReport(
            failed=((Path("/source.h5"), "conversion failed"),)
        )
        migrate.publish_migration_terminal_status(
            legacy_run,
            generation="attempt-1",
            succeeded=False,
            failure_category="image",
            reason="conversion failed",
            report=report,
            control_root=plan.control_root,
        )
        assert migrate.mark_generation_failed(
            legacy_run, "attempt-1", "conversion failed"
        ) is True
        return SimpleNamespace(job_ids=["101"])

    monkeypatch.setattr(migrate, "submit_migration_slurm_plan", _submit)

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
            "--wait",
        ],
    )

    assert result.exit_code == 1
    assert "Error: SLURM migration failed after lifecycle closure" in result.output
    assert "Unexpected error" not in result.output


def test_wait_rejects_missing_lifecycle_authority_without_polling(
    tmp_path: Path,
) -> None:
    """A vanished attempt fence is terminal evidence failure, not an endless wait."""
    import click

    from phenotypic._cli._cli_migrate import _wait_for_migration_terminal_status

    with pytest.raises(click.ClickException, match="lifecycle authority is missing"):
        _wait_for_migration_terminal_status(
            tmp_path / "output",
            control_root=tmp_path / "control",
            generation="attempt-1",
            poll_interval=0.0,
            timeout=0.0,
        )


def test_wait_rereads_terminal_status_after_observing_lifecycle_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finalizer publish between the two probes cannot become false absence."""
    from phenotypic._cli import _cli_migrate as migrate

    terminal = _terminal_status_payload(generation="attempt-1")
    reads = iter((None, terminal))
    monkeypatch.setattr(migrate, "_read_migration_terminal_status", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(
        migrate,
        "load_slurm_lifecycle",
        lambda *_a: {"generation": "attempt-1", "active": False},
    )

    assert migrate._wait_for_migration_terminal_status(
        tmp_path / "output",
        control_root=tmp_path / "control",
        generation="attempt-1",
        poll_interval=0.0,
    ) == terminal


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("report", {"converted": "one"}),
        ("failure_category", "unknown"),
        ("completed_at", "not-a-timestamp"),
    ],
)
def test_terminal_authority_rejects_malformed_typed_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    """Wait authority must reject malformed reports, categories, and timestamps."""
    import json

    from phenotypic._cli._cli_migrate import _read_migration_terminal_status

    status = _terminal_status_payload(generation="attempt-1", failed=True)
    status[field] = value
    path = tmp_path / "terminal_status.json"
    path.write_text(json.dumps(status), encoding="utf-8")

    assert _read_migration_terminal_status(path, generation="attempt-1") is None


def test_succeeded_terminal_rejects_any_failure_rows(tmp_path: Path) -> None:
    """A success authority cannot carry contradictory failed-work evidence."""
    import json

    from phenotypic._cli._cli_migrate import _read_migration_terminal_status

    status = _terminal_status_payload(generation="attempt-1")
    report = status["report"]
    assert isinstance(report, dict)
    report["overlay_failures"] = [
        {"path": "/overlay.png", "reason": "render failed"}
    ]
    path = tmp_path / "terminal_status.json"
    path.write_text(json.dumps(status), encoding="utf-8")

    assert _read_migration_terminal_status(path, generation="attempt-1") is None


def test_waited_success_prints_typed_migration_summary(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real waiter reports final pass counts only after lifecycle closure."""
    from phenotypic._cli import _cli_migrate as migrate

    plan = _migration_plan(tmp_path, "attempt-1")
    terminal = _terminal_status_payload(generation="attempt-1")
    terminal_report = terminal["report"]
    assert isinstance(terminal_report, dict)
    terminal_report["failed"] = []
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(migrate, "generate_migration_slurm_plan", lambda *_a, **_k: plan)

    def _submit(*_args, **_kwargs):
        migrate.publish_migration_terminal_status(
            legacy_run,
            generation="attempt-1",
            succeeded=True,
            failure_category=None,
            reason=None,
            report=_migration_report_from_payload(terminal_report),
            control_root=plan.control_root,
        )
        migrate.deactivate_generation(legacy_run, "attempt-1")
        return SimpleNamespace(job_ids=["101"])

    monkeypatch.setattr(migrate, "submit_migration_slurm_plan", _submit)
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
            "--wait",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pass 1" in result.output
    assert "converted 2, skipped 1" in result.output


def test_no_wait_reports_job_ids_without_scientific_publication(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submission output names durable IDs while scientific files stay untouched."""
    from phenotypic._cli import _cli_migrate as migrate

    plan = _migration_plan(tmp_path, "attempt-1")
    before = _scientific_tree_snapshot(legacy_run)
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(migrate, "generate_migration_slurm_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        migrate,
        "submit_migration_slurm_plan",
        lambda *_a, **_k: SimpleNamespace(job_ids=["101", "102"]),
    )

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Initial job IDs: 101, 102" in result.output
    assert _scientific_tree_snapshot(legacy_run) == before


@pytest.mark.parametrize("job_ids", [None, [], ["not-a-slurm-id"]])
def test_public_submission_rejects_missing_or_invalid_job_ids(
    legacy_run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_ids: object,
) -> None:
    """No-wait success is permitted only after durable numeric scheduler IDs."""
    from phenotypic._cli import _cli_migrate as migrate

    plan = _migration_plan(tmp_path, "attempt-1")
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(migrate, "generate_migration_slurm_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        migrate,
        "submit_migration_slurm_plan",
        lambda *_a, **_k: SimpleNamespace(job_ids=job_ids),
    )

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
        ],
    )

    assert result.exit_code == 1
    assert "invalid job IDs" in result.output or "no job IDs" in result.output


def test_local_migration_dry_run_preserves_every_directory_and_file_byte(
    legacy_run,
) -> None:
    """The local preview cannot add, remove, or rewrite even empty directories."""
    before = _tree_snapshot(legacy_run)

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert _tree_snapshot(legacy_run) == before


def test_mocked_slurm_dry_run_preserves_every_directory_and_file_byte(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public dry SLURM dispatch cannot alter science when planner is mocked."""
    from phenotypic._cli import _cli_migrate as migrate

    plan = _migration_plan(tmp_path, "attempt-1")
    before = _tree_snapshot(legacy_run)
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(migrate, "generate_migration_slurm_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        migrate,
        "submit_migration_slurm_plan",
        lambda *_a, **_k: pytest.fail("dry-run submitted work"),
    )

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _tree_snapshot(legacy_run) == before


def test_wait_rejects_closed_lifecycle_without_terminal_status(
    tmp_path: Path,
) -> None:
    """Closure without terminal authority is reported immediately, not polled."""
    import click

    from phenotypic._cli._cli_migrate import _wait_for_migration_terminal_status
    from phenotypic._cli._cli_slurm_lifecycle import (
        deactivate_generation,
        initialize_slurm_lifecycle,
    )

    initialize_slurm_lifecycle(tmp_path, generation="attempt-1", mode="migrate")
    assert deactivate_generation(tmp_path, "attempt-1") is True

    with pytest.raises(click.ClickException, match="closed without a valid terminal"):
        _wait_for_migration_terminal_status(
            tmp_path,
            control_root=tmp_path / "control",
            generation="attempt-1",
            poll_interval=0.0,
        )


@pytest.mark.parametrize("dry_run", [False, True])
def test_local_wait_combinations_reject_before_any_artifact_write(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """Local wait has no terminal scheduler authority and is invalid before work."""
    from phenotypic._cli import _cli_migrate as migrate

    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda *_a, **_k: pytest.fail("local wait planned scheduler work"),
    )
    before = _tree_snapshot(legacy_run)
    args = ["--mode", "migrate", "--output", str(legacy_run), "--wait"]
    if dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(phenotypic_cli, args)

    assert result.exit_code != 0
    assert "--wait requires --slurm" in result.output
    assert _tree_snapshot(legacy_run) == before
    assert not cache.exists()


def test_slurm_wait_dry_run_rejects_before_scientific_or_control_writes(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parsed SLURM dry-run+wait branch creates neither plan nor cache tree."""
    from phenotypic._cli import _cli_migrate as migrate

    cache = tmp_path / "external-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda *_a, **_k: pytest.fail("incompatible dry-run wait planned work"),
    )
    before_science = _tree_snapshot(legacy_run)
    before_control = _tree_snapshot(cache)

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
            "--wait",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "--wait cannot be combined with --dry-run" in result.output
    assert _tree_snapshot(legacy_run) == before_science
    assert _tree_snapshot(cache) == before_control


def test_public_rerun_after_waited_terminal_failure_uses_new_attempt(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a real waited finalizer failure, a new public call owns a new generation."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic.sdk_._hdf_to_zarr import MigrationReport

    generations = iter(("attempt-1", "attempt-2"))
    planned: list[str] = []
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: next(generations))
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda _output, **kwargs: (
            planned.append(kwargs["generation"])
            or _migration_plan(tmp_path, kwargs["generation"])
        ),
    )

    def _submit(plan, **_kwargs):
        failed = plan.generation == "attempt-1"
        report = (
            MigrationReport(failed=((Path("/source.h5"), "conversion failed"),))
            if failed
            else MigrationReport()
        )
        migrate.publish_migration_terminal_status(
            legacy_run,
            generation=plan.generation,
            succeeded=not failed,
            failure_category="image" if failed else None,
            reason="conversion failed" if failed else None,
            report=report,
            control_root=plan.control_root,
        )
        if failed:
            migrate.mark_generation_failed(legacy_run, plan.generation, "conversion failed")
        else:
            migrate.deactivate_generation(legacy_run, plan.generation)
        return SimpleNamespace(job_ids=["101"])

    monkeypatch.setattr(migrate, "submit_migration_slurm_plan", _submit)
    args = [
        "--mode",
        "migrate",
        "--output",
        str(legacy_run),
        "--slurm",
        "slurm_partition=short",
        "--wait",
    ]

    first = CliRunner().invoke(phenotypic_cli, args)
    second = CliRunner().invoke(phenotypic_cli, args)

    assert first.exit_code == 1
    assert second.exit_code == 0, second.output
    assert planned == ["attempt-1", "attempt-2"]


@pytest.mark.parametrize(
    "error,expected",
    [
        (RuntimeError("Output already has an active SLURM generation 'active'"), "click"),
        (RuntimeError("programming defect"), "raise"),
    ],
)
def test_initialization_only_normalizes_active_lifecycle_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: RuntimeError,
    expected: str,
) -> None:
    """Only the known active-attempt guard is a user-facing configuration error."""
    import click

    from phenotypic._cli import _cli_migrate as migrate

    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda *_a, **_k: _migration_plan(tmp_path, "attempt-1"),
    )
    monkeypatch.setattr(
        migrate,
        "initialize_slurm_lifecycle",
        lambda *_a, **_k: (_ for _ in ()).throw(error),
    )

    if expected == "click":
        with pytest.raises(click.ClickException, match="active SLURM generation"):
            migrate.handle_migrate_mode(
                tmp_path, slurm_args={"slurm_partition": "short"}
            )
    else:
        with pytest.raises(RuntimeError, match="programming defect"):
            migrate.handle_migrate_mode(
                tmp_path, slurm_args={"slurm_partition": "short"}
            )


def test_public_active_attempt_conflict_is_a_click_error(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public command reports an already-active migration attempt cleanly."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    initialize_slurm_lifecycle(legacy_run, generation="active", mode="migrate")
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda *_a, **_k: _migration_plan(tmp_path, "attempt-1"),
    )
    monkeypatch.setattr(
        migrate,
        "submit_migration_slurm_plan",
        lambda *_a, **_k: pytest.fail("active lifecycle submitted work"),
    )

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--slurm",
            "slurm_partition=short",
        ],
    )

    assert result.exit_code == 1
    assert "Error: Could not initialize SLURM migration attempt" in result.output
    assert "Unexpected error" not in result.output


def test_public_rerun_after_failed_submission_uses_a_fresh_attempt(
    legacy_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed attempt cannot strand the public CLI on its old generation."""
    from phenotypic._cli import _cli_migrate as migrate

    generations = iter(("attempt-1", "attempt-2"))
    submitted = iter((SimpleNamespace(job_ids=[]), SimpleNamespace(job_ids=["102"])))
    planned: list[str] = []
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: next(generations))
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda _output, **kwargs: (
            planned.append(kwargs["generation"])
            or _migration_plan(tmp_path, kwargs["generation"])
        ),
    )
    monkeypatch.setattr(migrate, "submit_migration_slurm_plan", lambda *_a, **_k: next(submitted))

    first = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--slurm", "slurm_partition=short"],
    )
    second = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run), "--slurm", "slurm_partition=short"],
    )

    assert first.exit_code == 1
    assert second.exit_code == 0, second.output
    assert planned == ["attempt-1", "attempt-2"]
    assert "attempt-2" in second.output


@pytest.mark.parametrize("error", [ValueError("bad plan"), AssertionError("bug")])
def test_planner_only_normalizes_expected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Configuration errors become Click errors; programming defects remain visible."""
    import click

    from phenotypic._cli import _cli_migrate as migrate

    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda *_a, **_k: (_ for _ in ()).throw(error),
    )

    if isinstance(error, ValueError):
        with pytest.raises(click.ClickException, match="Could not plan"):
            migrate.handle_migrate_mode(
                tmp_path, slurm_args={"slurm_partition": "short"}
            )
    else:
        with pytest.raises(AssertionError, match="bug"):
            migrate.handle_migrate_mode(
                tmp_path, slurm_args={"slurm_partition": "short"}
            )


@pytest.mark.parametrize("error", [RuntimeError("sbatch unavailable"), AssertionError("bug")])
def test_submitter_only_normalizes_expected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Expected scheduler failure closes an attempt; an assertion does not hide."""
    import click

    from phenotypic._cli import _cli_migrate as migrate

    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "attempt-1")
    monkeypatch.setattr(
        migrate,
        "generate_migration_slurm_plan",
        lambda *_a, **_k: _migration_plan(tmp_path, "attempt-1"),
    )
    monkeypatch.setattr(
        migrate,
        "submit_migration_slurm_plan",
        lambda *_a, **_k: (_ for _ in ()).throw(error),
    )

    if isinstance(error, RuntimeError):
        with pytest.raises(click.ClickException, match="Could not submit"):
            migrate.handle_migrate_mode(
                tmp_path, slurm_args={"slurm_partition": "short"}
            )
    else:
        with pytest.raises(AssertionError, match="bug"):
            migrate.handle_migrate_mode(
                tmp_path, slurm_args={"slurm_partition": "short"}
            )


def _migration_plan(tmp_path: Path, generation: str):
    """Build the smallest durable control-plan fixture for public CLI tests."""
    from phenotypic._cli._cli_migrate_slurm import MigrationSlurmPlan

    control = tmp_path / generation
    control.mkdir(exist_ok=True)
    manifest = control / "migration_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    script = control / "metadata.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    finalizer = control / "finalize.sh"
    finalizer.write_text("#!/bin/bash\n", encoding="utf-8")
    return MigrationSlurmPlan(
        generation=generation,
        control_root=control,
        manifest_path=manifest,
        flat_scripts=(script,),
        finalizer_script=finalizer,
        task_count=1,
    )


def _terminal_status_payload(*, generation: str, failed: bool = False) -> dict[str, object]:
    """Return a literal terminal document matching the worker's public schema."""
    return {
        "schema_version": 1,
        "generation": generation,
        "status": "failed" if failed else "succeeded",
        "failure_category": "image" if failed else None,
        "reason": "conversion failed" if failed else None,
        "report": {
            "converted": 2,
            "skipped": 1,
            "headers_migrated": 3,
            "tables_migrated": 4,
            "tables_skipped": 5,
            "overlays_created": 6,
            "overlays_skipped": 7,
            "failed": (
                [{"path": "/source.h5", "reason": "conversion failed"}]
                if failed
                else []
            ),
            "header_failures": [],
            "table_failures": [],
            "overlay_failures": [],
            "publication_failures": [],
        },
        "completed_at": "2026-08-29T12:34:56.789+00:00",
    }


def _migration_report_from_payload(payload: object):
    """Use a hand-authored payload to construct the worker's report type."""
    from phenotypic.sdk_._hdf_to_zarr import MigrationReport

    assert isinstance(payload, dict)
    return MigrationReport(
        converted=payload["converted"],
        skipped=payload["skipped"],
        headers_migrated=payload["headers_migrated"],
        tables_migrated=payload["tables_migrated"],
        tables_skipped=payload["tables_skipped"],
        overlays_created=payload["overlays_created"],
        overlays_skipped=payload["overlays_skipped"],
        failed=tuple((Path(item["path"]), item["reason"]) for item in payload["failed"]),
    )


def _scientific_tree_snapshot(root: Path) -> dict[str, str]:
    """Snapshot scientific directory topology plus file bytes, excluding control state."""
    return {
        path.relative_to(root).as_posix() + ("/" if path.is_dir() else ""): (
            "directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(root.rglob("*"))
        if ".phenotypic" not in path.relative_to(root).parts
    }


def _tree_snapshot(root: Path) -> dict[str, str]:
    """Return a directory- and byte-sensitive snapshot without migration helpers."""
    return {
        path.relative_to(root).as_posix() + ("/" if path.is_dir() else ""): (
            "directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(root.rglob("*"))
        if path.is_dir() or path.is_file()
    }


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
