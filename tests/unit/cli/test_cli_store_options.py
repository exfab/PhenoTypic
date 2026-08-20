"""``--durable-writes``: tri-state, transported, and load-bearing.

Spec §3.7 / Phase 3 Task 3.7. The flag exists because the same command carries
different durability guarantees in different places -- a genuinely surprising
thing to debug -- so the resolved mode is logged and the detection is
overridable. A flag that *parses* but never reaches a writer restores exactly
the surprise it was added to remove, so every test here pins a **resolved
value at a boundary it crosses**, never the fact that click accepts the
spelling.

Three states, not two:

===================  ==============================================
value                meaning
===================  ==============================================
``None`` (unset)     auto-detect: on under SLURM, off locally
``True``             ``--durable-writes``  -- fsync regardless of env
``False``            ``--no-durable-writes`` -- never fsync
===================  ==============================================

``None`` is resolved in exactly one place,
:func:`phenotypic.sdk_.ngff_._resolve_durability`, which both
``durable_writes_enabled`` and ``describe_durability`` share so the flag and
the sentence describing it cannot drift. Everything between the CLI and that
function must carry ``None`` **as** ``None``; a layer that eagerly resolves it
to a bool would freeze the submitting process's environment into a value the
worker then re-uses on a different node.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.phenotypicCLI import phenotypic_cli


# ---------------------------------------------------------------------------
# The option surface
# ---------------------------------------------------------------------------


def test_help_offers_both_spellings() -> None:
    result = CliRunner().invoke(phenotypic_cli, ["-h"])

    assert result.exit_code == 0, result.output
    assert "--durable-writes" in result.output
    assert "--no-durable-writes" in result.output


def test_no_pyramid_levels_option_exists() -> None:
    """Descoped: the pyramid depth is a pure function of shape (P3)."""
    result = CliRunner().invoke(phenotypic_cli, ["-h"])

    assert result.exit_code == 0, result.output
    assert "--pyramid-levels" not in result.output


def _captured_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    extra: list[str],
) -> Any:
    """Run a dry-run to the point of config construction and return the config."""
    seen: dict[str, Any] = {}

    def _spy(config, datasets, output_dir) -> None:  # noqa: ANN001
        seen["config"] = config

    monkeypatch.setattr("phenotypic.phenotypicCLI.execute_dry_run", _spy)
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "full",
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
            "--force-local",
            *extra,
        ],
    )
    assert "config" in seen, result.output
    return seen["config"]


def test_unset_reaches_the_config_as_none_not_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
) -> None:
    """A plain ``is_flag=True`` would make this ``False`` and lose SLURM detection."""
    config = _captured_config(
        monkeypatch, tmp_path, simple_pipeline_json, synth_one_level_input, []
    )

    assert config.durable_writes is None


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--durable-writes", True), ("--no-durable-writes", False)],
)
def test_explicit_flags_reach_the_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    flag: str,
    expected: bool,
) -> None:
    config = _captured_config(
        monkeypatch,
        tmp_path,
        simple_pipeline_json,
        synth_one_level_input,
        [flag],
    )

    assert config.durable_writes is expected


def test_rejected_on_recompile(tmp_path: Path) -> None:
    """recompile writes no image store from a pipeline, so the flag is a lie there."""
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "recompile",
            "--output",
            str(tmp_path / "out"),
            "--durable-writes",
        ],
    )

    assert result.exit_code != 0
    # NOT a bare ``"--durable-writes" in result.output``: before the option
    # existed, click's own "No such option: --durable-writes" satisfied that
    # and the test passed green against nothing.
    assert "--durable-writes is not accepted with --mode recompile" in (
        result.output
    )


def test_rejection_set_already_names_migrate() -> None:
    """``--mode migrate`` lands in Phase 5; the guard must not need editing then.

    Asserting the guard through the CLI is impossible today -- click's
    ``Choice`` rejects ``migrate`` before the guard runs -- so the set itself
    is the contract.
    """
    from phenotypic.phenotypicCLI import DURABLE_WRITES_REJECTED_MODES

    assert DURABLE_WRITES_REJECTED_MODES == frozenset({"recompile", "migrate"})


@pytest.mark.parametrize("mode", ["full", "measure", "process"])
def test_accepted_on_the_modes_that_write_stores(mode: str) -> None:
    """Acceptance is proven by the failure being about something else."""
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", mode, "--durable-writes"]
    )

    assert "No such option" not in result.output
    assert "--durable-writes is not accepted" not in result.output


# ---------------------------------------------------------------------------
# The start-of-run log line -- the mitigation the flag exists to feed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "slurm", "sentence"),
    [
        (None, False, "durable writes: off (local)"),
        (None, True, "durable writes: on (SLURM)"),
        (True, False, "durable writes: on (--durable-writes)"),
        (False, True, "durable writes: off (--no-durable-writes)"),
    ],
)
def test_create_execution_strategy_logs_the_resolved_mode(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
    make_output_manager: Callable[..., OutputManager],
    override: bool | None,
    slurm: bool,
    sentence: str,
) -> None:
    """The last row is the one that matters: an override must beat the env.

    It is also the mutation detector for the ``create_execution_strategy``
    call site -- dropping ``durable_writes=config.durable_writes`` there makes
    the run log ``on (SLURM)`` while every store is written without fsync.
    """
    from phenotypic._cli._cli_execution_strategies import (
        create_execution_strategy,
    )

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    if slurm:
        monkeypatch.setenv("SLURM_JOB_ID", "12345")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=output_dir,
        force_local=True,
        durable_writes=override,
    )
    with caplog.at_level(logging.INFO):
        create_execution_strategy(config, make_output_manager(output_dir))

    assert any(sentence in record.message for record in caplog.records), (
        f"{sentence!r} not in {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Continuation must survive a durability change
# ---------------------------------------------------------------------------


def test_durability_is_not_part_of_the_work_id(
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    """Durability is a storage guarantee, not a scientific parameter.

    Folding it into the digest would make ``--no-durable-writes`` restart a
    finished run from zero -- a far worse outcome than the fsync it saves.
    """
    from phenotypic._cli._cli_failure_tracker import (
        processing_configuration_digest,
    )

    digests = {
        processing_configuration_digest(
            make_exec_config(
                pipeline_json=simple_pipeline_json,
                input_path=synth_one_level_input,
                durable_writes=value,
            )
        )
        for value in (None, True, False)
    }

    assert len(digests) == 1


# ---------------------------------------------------------------------------
# OutputManager carries the tri-state to every write site
# ---------------------------------------------------------------------------


def test_output_manager_defaults_to_auto_detect(tmp_path: Path) -> None:
    assert OutputManager.from_config(tmp_path, ".tiff").durable_writes is None


@pytest.mark.parametrize("value", [None, True, False])
def test_the_cli_attaches_the_flag_to_the_manager_it_hands_the_strategy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    value: bool | None,
) -> None:
    """The one manager every local write goes through.

    Every ``save_image_store`` site reads its durability off the manager, so
    this single ``from_config`` call is where a local run's flag would be lost
    -- silently, with the start-of-run log still reporting the right mode
    because that log is fed from the config rather than from the manager.
    """

    class _Stop(Exception):
        pass

    seen: dict[str, Any] = {}

    def _spy(config, output_manager):  # noqa: ANN001
        seen["manager"] = output_manager
        raise _Stop()

    monkeypatch.setattr(
        "phenotypic.phenotypicCLI.create_execution_strategy", _spy
    )
    flag = {None: [], True: ["--durable-writes"], False: ["--no-durable-writes"]}
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "full",
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "out"),
            "--force-local",
            "--skip-validation",
            *flag[value],
        ],
    )

    assert "manager" in seen, result.output
    assert seen["manager"].durable_writes is value


@pytest.mark.parametrize("value", [True, False, None])
def test_from_config_carries_the_tri_state(
    tmp_path: Path, value: bool | None
) -> None:
    manager = OutputManager.from_config(
        tmp_path, ".tiff", durable_writes=value
    )

    assert manager.durable_writes is value
