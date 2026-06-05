from phenotypic._cli._cli_execution_strategies import LocalParallelStrategy
from phenotypic.tools_ import deliverables_dir, manifest_json_path


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
    tiffs = list(out.rglob("*_detect_mat.tiff"))
    assert tiffs, "no mirrored detect_mat tiffs written"
    # progress manifest written (run console visibility), but no deliverables/dashboard
    assert manifest_json_path(out).is_file()
    assert not deliverables_dir(out).exists()
