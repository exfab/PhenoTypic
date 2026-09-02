"""Full-forward CLI provenance, original retention, and flag transport."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_failure_tracker import processing_configuration_digest
from phenotypic._cli._cli_state_management import validate_resume_compatibility
from phenotypic._cli._cli_types import Dataset
from phenotypic.phenotypicCLI import phenotypic_cli


def _captured_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    extra: list[str],
) -> Any:
    seen: dict[str, Any] = {}
    def _spy(config: Any, datasets: Any, output_dir: Any) -> None:
        del datasets, output_dir
        seen["config"] = config

    monkeypatch.setattr("phenotypic.phenotypicCLI.execute_dry_run", _spy)
    result = CliRunner().invoke(
        phenotypic_cli,
        [
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


def test_drop_originals_defaults_false_and_reaches_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
) -> None:
    default = _captured_config(
        monkeypatch, tmp_path, simple_pipeline_json, synth_one_level_input, []
    )
    dropped = _captured_config(
        monkeypatch,
        tmp_path,
        simple_pipeline_json,
        synth_one_level_input,
        ["--drop-originals"],
    )

    expected_identity = {
        "source_path": str(simple_pipeline_json.resolve()),
        "sha256": hashlib.sha256(simple_pipeline_json.read_bytes()).hexdigest(),
    }
    assert default.drop_originals is False
    assert dropped.drop_originals is True
    assert default.pipeline_identity == expected_identity
    assert dropped.pipeline_identity == expected_identity


@pytest.mark.parametrize("mode", ["measure", "recompile", "process", "migrate"])
def test_drop_originals_is_rejected_outside_full_mode(
    tmp_path: Path, mode: str
) -> None:
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            mode,
            "--output",
            str(tmp_path / "out"),
            "--drop-originals",
        ],
    )

    assert result.exit_code != 0
    assert f"--drop-originals is not accepted with --mode {mode}" in result.output


def test_drop_originals_changes_processing_fingerprint_and_resume_compatibility(
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    keep = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        drop_originals=False,
    )
    drop = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        drop_originals=True,
    )

    assert processing_configuration_digest(keep) != processing_configuration_digest(drop)

    state = SimpleNamespace(
        pipeline_path=simple_pipeline_json,
        input_path=synth_one_level_input,
        config={
            "image_type": keep.image_type,
            "pipeline_sha256": None,
            "process_only_layer": None,
            "drop_originals": False,
        },
    )
    compatible, error = validate_resume_compatibility(state, drop)
    assert compatible is False
    assert error == "drop_originals mismatch: saved=False, current=True"


def test_local_strategy_hands_drop_originals_to_forward_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
    make_output_manager: Callable[..., Any],
) -> None:
    from phenotypic._cli import _cli_execution_strategies as strategies

    image = next(synth_one_level_input.rglob("*.tif"))
    output_dir = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=output_dir,
        drop_originals=True,
    )
    strategy = strategies.LocalParallelStrategy(
        config, make_output_manager(output_dir, save_overlays=False)
    )
    dataset = Dataset("ds", [image], synth_one_level_input, output_dir)
    seen: dict[str, Any] = {}
    identity_calls = 0
    real_work_id_for_image = strategies.work_id_for_image
    expected_identity = real_work_id_for_image(config, "ds", image)

    def _work_identity(*args: Any, **kwargs: Any) -> tuple[str, str]:
        nonlocal identity_calls
        identity_calls += 1
        return real_work_id_for_image(*args, **kwargs)


    def _spy(**kwargs: Any) -> bool:
        seen.update(kwargs)
        return True

    published: dict[str, Any] = {}

    def _publish(*args: Any, **kwargs: Any) -> None:
        del args
        published.update(kwargs)

    monkeypatch.setattr(strategies, "work_id_for_image", _work_identity)
    monkeypatch.setattr(strategies, "process_single_image_core", _spy)
    monkeypatch.setattr(strategies, "_publish_local_image_success", _publish)

    result = strategy._process_single_local(
        dataset, image, output_dir, tmp_path / "events.jsonl"
    )

    assert result[2] is True
    assert seen["drop_originals"] is True
    assert seen["work_id"] == expected_identity[0]
    assert published["work_identity"] == expected_identity
    assert identity_calls == 1


def test_local_strategy_reuses_preflight_identity_for_failure_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
    make_output_manager: Callable[..., Any],
) -> None:
    from phenotypic._cli import _cli_execution_strategies as strategies
    from phenotypic._cli._cli_failure_tracker import PerImageScientificError

    image = next(synth_one_level_input.rglob("*.tif"))
    output_dir = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=output_dir,
    )
    strategy = strategies.LocalParallelStrategy(
        config, make_output_manager(output_dir, save_overlays=False)
    )
    dataset = Dataset("ds", [image], synth_one_level_input, output_dir)
    real_work_id_for_image = strategies.work_id_for_image
    expected_identity = real_work_id_for_image(config, "ds", image)
    identity_calls = 0
    recorded: dict[str, Any] = {}

    def _work_identity(*args: Any, **kwargs: Any) -> tuple[str, str]:
        nonlocal identity_calls
        identity_calls += 1
        return real_work_id_for_image(*args, **kwargs)

    def _fail(**kwargs: Any) -> None:
        assert kwargs["work_id"] == expected_identity[0]
        raise PerImageScientificError("pipeline", RuntimeError("boom"))

    def _record(*args: Any, **kwargs: Any) -> bool:
        del args
        recorded.update(kwargs)
        return True

    monkeypatch.setattr(strategies, "work_id_for_image", _work_identity)
    monkeypatch.setattr(strategies, "process_single_image_core", _fail)
    monkeypatch.setattr(strategies, "_record_local_terminal_failure", _record)

    result = strategy._process_single_local(
        dataset, image, output_dir, tmp_path / "events.jsonl"
    )

    assert result[2] is False
    assert recorded["work_identity"] == expected_identity
    assert identity_calls == 1


def test_ordinary_worker_and_slurm_script_transport_drop_originals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    from phenotypic._cli import _cli_process_single
    from phenotypic._cli._cli_slurm_array_scripts import generate_array_job_script

    image = next(synth_one_level_input.rglob("*.tif"))
    output_dir = tmp_path / "out"
    pipeline_identity = {
        "source_path": str(simple_pipeline_json.resolve()),
        "sha256": hashlib.sha256(simple_pipeline_json.read_bytes()).hexdigest(),
    }
    seen: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> bool:
        seen.update(kwargs)
        raise RuntimeError("stop after transport")

    monkeypatch.setattr(_cli_process_single, "process_single_image_core", _spy)
    result = CliRunner().invoke(
        _cli_process_single.main,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "ds",
            "--image-type",
            "Image",
            "--provenance-pipeline-source-path",
            pipeline_identity["source_path"],
            "--provenance-pipeline-sha256",
            pipeline_identity["sha256"],
            "--drop-originals",
        ],
    )
    assert seen.get("drop_originals") is True, result.output
    assert seen.get("pipeline_identity") == pipeline_identity, result.output

    output_dir.mkdir(exist_ok=True)
    snapshot = tmp_path / "submission" / "pipeline.json"
    snapshot.parent.mkdir()
    snapshot.write_bytes(simple_pipeline_json.read_bytes())
    config = make_exec_config(
        pipeline_json=snapshot,
        input_path=synth_one_level_input,
        output_dir=output_dir,
        force_local=False,
        slurm_args={"slurm_partition": "short"},
        drop_originals=True,
        pipeline_identity=pipeline_identity,
    )
    dataset = Dataset("ds", [image], synth_one_level_input, output_dir)
    script = generate_array_job_script(dataset, (0, 1), config, output_dir).read_text()
    assert "--drop-originals" in script
    assert (
        f"--pipeline \\\n    {snapshot.resolve()}"
        in script
    )
    assert "--provenance-pipeline-source-path" in script
    assert pipeline_identity["source_path"] in script
    assert pipeline_identity["sha256"] in script
