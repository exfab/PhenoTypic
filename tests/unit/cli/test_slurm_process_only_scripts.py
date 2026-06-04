from phenotypic._cli._cli_directory_scanner import (
    organize_by_dataset,
    scan_directory_structure,
)
from phenotypic._cli._cli_slurm_array_scripts import generate_all_array_job_scripts


def test_array_script_threads_process_only_and_omits_aggregation(
    tmp_path, simple_pipeline_json, synth_one_level_input, make_exec_config
):
    out = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=out,
        force_local=False,
        process_only_layer="rgb",
        slurm_args={"slurm_partition": "compute"},
    )
    datasets = organize_by_dataset(
        scan_directory_structure(synth_one_level_input), out
    )
    # Real signature: (datasets, config, output_dir, array_limit) -> Dict[str, List[Path]]
    scripts_by_ds = generate_all_array_job_scripts(
        datasets, config, out, array_limit=1000
    )
    script_paths = [p for paths in scripts_by_ds.values() for p in paths]
    blob = "\n".join(p.read_text() for p in script_paths)

    # The per-image command is line-broken with ``\`` continuations, so the
    # flag and its value land on separate lines — assert each token is threaded
    # into the per-image worker invocation.
    assert "--process-only" in blob
    assert "phenotypic._cli._cli_process_single" in blob
    # The layer value appears on its own continuation line after --process-only.
    assert "\n    rgb \\" in blob
    assert "--input-root" in blob
    # No measurement aggregation / checkpoint / finalizer chain in process-only
    assert "_cli_chunk_writer" not in blob
    assert "--checkpoint-type finalize" not in blob
    assert "--checkpoint-type manifest" not in blob
    # Process-only never threads the forward-run overlay flag
    assert "--save-overlays" not in blob


def test_process_only_finalize_script_is_manifest_only(
    tmp_path, simple_pipeline_json, synth_one_level_input, make_exec_config
):
    from phenotypic._cli._cli_slurm_array_scripts import (
        generate_process_only_finalize_script,
    )

    out = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=out,
        force_local=False,
        process_only_layer="detect_mat",
        slurm_args={"slurm_partition": "compute"},
    )
    script = generate_process_only_finalize_script(config, out)
    text = script.read_text()
    # Manifest-only: rebuilds manifest.json, never aggregates / finalizes.
    assert "--checkpoint-type manifest" in text
    assert "--checkpoint-type finalize" not in text
    assert "aggregate_measurements" not in text
    assert "generate_dashboard" not in text
