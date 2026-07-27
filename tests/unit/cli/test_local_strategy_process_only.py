import json
from uuid import uuid4

import pytest

from phenotypic._cli._cli_gui_lifecycle import (
    publish_local_gui_completion,
)
from phenotypic._cli._cli_execution_strategies import LocalParallelStrategy
from phenotypic.gui.shell._runs_registry import RunRegistry
from phenotypic.sdk_ import (
    deliverables_dir,
    manifest_json_path,
    run_completion_marker_path,
)
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR


def test_local_process_only_writes_layers_and_manifest_no_deliverables(
    tmp_path,
    synth_one_level_input,
    simple_pipeline_json,
    make_exec_config,
    make_output_manager,
):
    out = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=out,
        n_jobs=1,
        force_local=True,
        process_only_layer="detect_mat",
    )
    strat = LocalParallelStrategy(config, make_output_manager(out))
    # datasets discovered by the CLI scanner; build them here for the unit test
    from phenotypic._cli._cli_directory_scanner import (
        organize_by_dataset,
        scan_directory_structure,
    )

    datasets = organize_by_dataset(
        scan_directory_structure(synth_one_level_input), out
    )
    strat.execute(datasets, out)

    # mirrored layer files exist
    tiffs = list(out.rglob("*.tiff"))
    assert tiffs, "no mirrored detect_mat tiffs written"
    assert not list(out.rglob("*_detect_mat.tiff"))
    # progress manifest written (run console visibility), but no deliverables/dashboard
    assert manifest_json_path(out).is_file()
    assert not deliverables_dir(out).exists()


@pytest.mark.parametrize("failed_publication", ["manifest", "dashboard"])
def test_failed_current_publication_cannot_bless_preexisting_manifest(
    tmp_path,
    synth_one_level_input,
    simple_pipeline_json,
    make_exec_config,
    make_output_manager,
    monkeypatch,
    failed_publication,
):
    """A swallowed build failure leaves no current-generation completion."""
    out = tmp_path / "out"
    manifest = manifest_json_path(out)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    prior_generation = uuid4()
    manifest.write_text(
        json.dumps(
            {
                "execution_mode": "local",
                "gui_record_generation": str(prior_generation),
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            }
        ),
        encoding="utf-8",
    )
    registry = RunRegistry()
    record = registry.allocate(
        mode="local",
        output_dir=out,
        rel_path="out",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    monkeypatch.setenv(
        GUI_RECORD_GENERATION_ENV_VAR,
        str(record.generation),
    )

    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=out,
        n_jobs=1,
        force_local=True,
        process_only_layer=(
            "detect_mat" if failed_publication == "manifest" else None
        ),
    )
    from phenotypic._cli._cli_directory_scanner import (
        organize_by_dataset,
        scan_directory_structure,
    )

    datasets = organize_by_dataset(
        scan_directory_structure(synth_one_level_input), out
    )
    output_manager = make_output_manager(out)
    if failed_publication == "dashboard":
        output_manager.create_structure(datasets)
    strategy = LocalParallelStrategy(config, output_manager)
    if failed_publication == "manifest":
        monkeypatch.setattr(
            "phenotypic._cli._dashboard._manifest_builder.build_manifest",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("simulated current manifest build failure")
            ),
        )
    else:
        monkeypatch.setattr(
            "phenotypic._cli._dashboard._generator.generate_dashboard",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("simulated current dashboard build failure")
            ),
        )

    strategy.execute(datasets, out)

    with pytest.raises(RuntimeError, match="stale-generation"):
        publish_local_gui_completion(out)
    assert not run_completion_marker_path(out).exists()
    assert registry.observe_local_exit("out", record.generation, 0)
    updated = registry.get("out")
    assert updated is not None
    assert updated.status == "failed"
    assert "manifest belongs to a different launch generation" in (
        updated.status_detail or ""
    )
