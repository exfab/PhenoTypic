"""--process-format reaches the command a user actually runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_failure_tracker import (
    processing_configuration_digest_from_values,
)
from phenotypic._cli._cli_types import ExecutionConfig
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import image_record_path


def _digest(**overrides) -> str:
    base = dict(
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=16,
        detect_mode="gray",
        process_only_layer="rgb",
        ext="tiff",
        process_format="tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
    )
    base.update(overrides)
    return processing_configuration_digest_from_values(**base)


def _help_text() -> str:
    """`--help`, with click's column wrapping collapsed to single spaces.

    Click hard-wraps option help into a narrow right-hand column, so a phrase
    of more than two or three words is split across lines and a raw substring
    assertion against it passes no matter what the help says. That is a test
    that cannot fail: the plan's `"TIFF for rgb/gray/detect_mat" not in
    output` assertion was already green against the unedited help, which
    contains exactly that sentence -- as `"TIFF for\n  rgb/gray/detect_mat"`.
    """
    result = CliRunner().invoke(phenotypic_cli, ["--help"])
    assert result.exit_code == 0
    return " ".join(result.output.split())


def test_the_option_exists_and_states_its_default() -> None:
    text = _help_text()
    assert "--process-format" in text
    assert "zarr for rgb/gray" in text


def test_the_layer_help_no_longer_claims_tiff_for_everything() -> None:
    """It said "TIFF for rgb/gray/detect_mat". That stops being true here."""
    assert "TIFF for rgb/gray/detect_mat" not in _help_text()


def test_process_format_is_rejected_outside_process_mode(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    """Mirrors how --layer already behaves (phenotypicCLI.py:1336-1339)."""
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "full",
            "--process-format",
            "zarr",
        ],
    )
    assert result.exit_code != 0
    assert "--process-format can only be used with --mode process" in (
        " ".join(result.output.split())
    )


def test_an_impossible_layer_and_format_pair_is_refused(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "process",
            "--layer",
            "objmap",
            "--process-format",
            "zarr",
        ],
    )
    assert result.exit_code != 0
    assert "labels group" in result.output


def test_the_config_carries_a_resolved_format_not_none() -> None:
    """`ExecutionConfig` never holds the raw option; it holds the answer."""
    assert (
        ExecutionConfig.__dataclass_fields__["process_format"].default == "tiff"
    )


def test_the_format_joins_the_continuation_identity() -> None:
    """Switching format must invalidate continuation, not reuse the other kind."""
    assert _digest(process_format="tiff") != _digest(process_format="zarr")


def test_the_format_does_not_disturb_a_non_process_run() -> None:
    """A full run's digest must not change, or every existing run resumes cold.

    `process_format` joins the payload only inside the process-only branch,
    beside `ext`, exactly as `process_only_layer` does.
    """
    full = dict(
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=16,
        detect_mode="gray",
        process_only_layer=None,
        ext="tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
    )
    assert processing_configuration_digest_from_values(
        **full, process_format="tiff"
    ) == processing_configuration_digest_from_values(
        **full, process_format="zarr"
    )


def test_the_dry_run_plan_shows_the_resolved_format(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    """One of the seven `process_only_output_path` sites, seen end to end.

    `_print_process_only_dry_run_plan` computes the mirrored sample paths. If
    it does not receive the resolved format it prints `.tiff` for a run that
    writes `.ome.zarr` -- silently, because the parameter defaults to tiff.
    """
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "process",
            "--layer",
            "rgb",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "plateA.ome.zarr" in result.output
    assert "plateA.tiff" not in result.output


def test_the_slurm_worker_command_line_carries_the_format(
    tmp_path: Path,
) -> None:
    """Without this every array task re-resolves its own default.

    Right today by coincidence, and wrong the moment a user asks for
    `--process-format tiff` on rgb.
    """
    from phenotypic._cli._cli_slurm_array_scripts import (
        generate_array_job_script,
    )
    from phenotypic._cli._cli_types import Dataset

    images = []
    for index in range(2):
        image_path = tmp_path / f"image_{index}.tif"
        image_path.touch()
        images.append(image_path)
    pipeline_json = tmp_path / "pipeline.json"
    pipeline_json.write_text('{"operations": []}')
    config = ExecutionConfig(
        pipeline_json=pipeline_json,
        input_path=tmp_path,
        output_dir=tmp_path / "output",
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=-1,
        slurm_args={"slurm_partition": "short", "mem_gb": 16, "time": 60},
        force_local=False,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.3,
        include_dataset_column=False,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=False,
        process_only_layer="rgb",
        process_format="zarr",
    )
    script = generate_array_job_script(
        dataset=Dataset(
            name="ds",
            images=images,
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
        ),
        array_indices=(0, 2),
        config=config,
        output_dir=tmp_path / "output",
    )
    content = script.read_text(encoding="utf-8")
    assert "--process-format" in content
    assert "zarr" in content


def test_a_local_run_writes_a_store_and_continues_off_it(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    """The two remaining call sites, proved by running the thing.

    `_cli_execution_strategies.py`'s core call decides what gets written, and
    its completion-marker site decides what continuation then looks for. Both
    default to tiff, so a miss at either is silent: the first run writes a
    store and the marker names a `.tiff` that was never written, and the
    second run reprocesses every image forever.
    """
    import json

    args = [
        "--pipeline",
        str(simple_pipeline_json),
        "--input",
        str(synth_one_level_input),
        "--output",
        str(tmp_path / "out"),
        "--mode",
        "process",
        "--layer",
        "rgb",
        "--force-local",
        "--njobs",
        "1",
    ]
    first = CliRunner().invoke(phenotypic_cli, args)
    assert first.exit_code == 0, first.output

    store = tmp_path / "out" / "day1" / "plateA.ome.zarr"
    assert store.is_dir()
    assert not (tmp_path / "out" / "day1" / "plateA.tiff").exists()

    # Resolved through the helper, not hand-joined. The hand-joined form is
    # what made this site invisible to P3's repointing sweep, which grepped
    # for `image_completion_marker_path` and could not see a path spelled out
    # segment by segment -- the reason CLAUDE.md says to resolve through the
    # `sdk_` helpers and never hand-join names.
    marker = json.loads(
        image_record_path(tmp_path / "out", "day1", "plateA").read_text(
            encoding="utf-8"
        )
    )
    artifact = marker["artifacts"]["process_output"]
    assert artifact["path"] == "day1/plateA.ome.zarr"
    assert artifact["kind"] == "store"

    # Continuation must recognise its own output rather than rewrite it.
    root_mtime = (store / "zarr.json").stat().st_mtime_ns
    second = CliRunner().invoke(phenotypic_cli, args)
    assert second.exit_code == 0, second.output
    assert (store / "zarr.json").stat().st_mtime_ns == root_mtime


@pytest.mark.parametrize(
    "shape", ["same", "output_parent", "output_child", "symlink_alias"]
)
def test_process_refuses_canonical_input_output_overlap_before_mutation(
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    shape: str,
) -> None:
    """No aliasing/nesting shape may let process mutate its source tree."""
    source = synth_one_level_input
    source_bytes = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    if shape == "same":
        output = source
    elif shape == "output_parent":
        output = source.parent
    elif shape == "output_child":
        output = source / "process-output"
    else:
        output = tmp_path / "source-alias"
        try:
            output.symlink_to(source, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("directory symlinks are unavailable")

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(source),
            "--output",
            str(output),
            "--mode",
            "process",
            "--layer",
            "rgb",
            "--overwrite",
            "--force-local",
            "--njobs",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "must not overlap" in " ".join(result.output.split()).lower()
    assert {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == source_bytes


def test_the_worker_and_the_top_level_cli_agree_on_the_work_id(
    tmp_path: Path, simple_pipeline_json: Path, synth_one_level_input: Path
) -> None:
    """The seventh call site, and the identity the whole threading exists for.

    `_cli_process_single.py` publishes its own marker under SLURM, computing
    the work ID from its own `--process-format` rather than from the config.
    If the format reached one digest and not the other, every SLURM task would
    certify an ID the submitter never selected and the array would rerun for
    ever. Nothing local exercises this path, so it is asserted directly.
    """
    import json

    from phenotypic._cli._cli_process_single import main as worker

    image = synth_one_level_input / "day1" / "plateA.tif"
    common = [
        "--pipeline",
        str(simple_pipeline_json),
        "--input",
        str(synth_one_level_input),
        "--output",
        str(tmp_path / "local"),
        "--mode",
        "process",
        "--layer",
        "rgb",
        "--force-local",
        "--njobs",
        "1",
    ]
    local = CliRunner().invoke(phenotypic_cli, common)
    assert local.exit_code == 0, local.output

    remote = CliRunner().invoke(
        worker,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--image",
            str(image),
            "--output-dir",
            str(tmp_path / "worker"),
            "--dataset-name",
            "day1",
            "--input-root",
            str(synth_one_level_input),
            "--mode",
            "process",
            "--layer",
            "rgb",
            "--process-format",
            "zarr",
        ],
    )
    assert remote.exit_code == 0, remote.output
    assert (tmp_path / "worker" / "day1" / "plateA.ome.zarr").is_dir()

    def _marker(root: Path) -> dict:
        # Helper-resolved, not hand-joined -- see the note at the sibling
        # read-back above.
        return json.loads(
            image_record_path(root, "day1", "plateA").read_text(
                encoding="utf-8"
            )
        )

    worker_marker = _marker(tmp_path / "worker")
    assert worker_marker["artifacts"]["process_output"] == {
        "kind": "store",
        "path": "day1/plateA.ome.zarr",
        "sha256": worker_marker["artifacts"]["process_output"]["sha256"],
    }
    assert worker_marker["work_id"] == _marker(tmp_path / "local")["work_id"]
