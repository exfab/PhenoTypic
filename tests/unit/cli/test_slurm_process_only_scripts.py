from phenotypic._cli._cli_directory_scanner import (
    organize_by_dataset,
    scan_directory_structure,
)
from phenotypic._cli._cli_slurm_array_scripts import (
    generate_all_array_job_scripts,
    generate_process_finalizer_script,
)
from phenotypic.sdk_ import logs_dir, slurm_scripts_dir


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

    # Process-only SLURM machine-state must remain hidden, just like full-mode
    # SLURM runs. Exported layer files themselves are intentionally outside the
    # cache, mirrored under ``out`` by the worker.
    dataset = datasets[0]
    assert all(path.is_relative_to(slurm_scripts_dir(out)) for path in script_paths)
    expected_log = logs_dir(out) / "slurm" / dataset.name / f"{dataset.name}_%A_%a.log"
    assert expected_log.as_posix() in blob

    # The per-image command is line-broken with ``\`` continuations, so the
    # flag and its value land on separate lines — assert each token is threaded
    # into the per-image worker invocation.
    assert "--mode" in blob
    assert "\n    process \\" in blob
    assert "--layer" in blob
    assert "phenotypic._cli._cli_process_single" in blob
    # The layer value appears on its own continuation line after --layer.
    assert "\n    rgb \\" in blob
    assert "--input-root" in blob
    # No measurement aggregation / full finalizer chain in process-only.
    assert "_cli_chunk_writer" not in blob
    assert "--checkpoint-type finalize" not in blob
    # Completion is not embedded in the concurrent image array.
    assert "__PHENOTYPIC_MANIFEST__" not in blob
    assert "--checkpoint-type manifest" not in blob
    finalizer = generate_process_finalizer_script(config, out).read_text()
    assert "--checkpoint-type manifest" in finalizer
    assert "--checkpoint-type finalize" not in finalizer
    # Process-only never threads the forward-run overlay flag
    assert "--save-overlays" not in blob


def test_array_script_mode_rewrite_does_not_mutate_dataset_named_full(
    tmp_path, simple_pipeline_json, synth_one_level_input, make_exec_config
):
    from phenotypic._cli._cli_slurm_array_scripts import generate_array_job_script

    out = tmp_path / "out"
    datasets = organize_by_dataset(
        scan_directory_structure(synth_one_level_input), out
    )
    dataset = datasets[0]
    dataset.name = "full"

    process_config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=out,
        force_local=False,
        process_only_layer="rgb",
        slurm_args={"slurm_partition": "compute"},
    )
    process_script = generate_array_job_script(
        dataset, (0, len(dataset.images)), process_config, out, chunk_id=0
    ).read_text()
    assert "--dataset-name \\\n    full \\" in process_script
    assert "--mode \\\n    process \\" in process_script
    assert "--layer \\\n    rgb \\" in process_script

    measure_config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=out,
        force_local=False,
        measure_only=True,
        slurm_args={"slurm_partition": "compute"},
    )
    measure_script = generate_array_job_script(
        dataset, (0, len(dataset.images)), measure_config, out, chunk_id=1
    ).read_text()
    assert "--dataset-name \\\n    full \\" in measure_script
    assert "--mode \\\n    measure \\" in measure_script


def test_process_only_never_embeds_concurrent_manifest_sentinel(
    tmp_path, simple_pipeline_json, make_exec_config
):
    """Every process-only chunk contains only image work."""
    from phenotypic._cli._cli_slurm_array_scripts import (
        _MANIFEST_SENTINEL,
        generate_array_job_script,
    )

    # Flat input with several images (dummy TIFFs — script generation only uses
    # the paths, never reads pixels).
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(4):
        (in_dir / f"plate{i}.tif").write_bytes(b"II*\x00")
    out = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=in_dir,
        output_dir=out,
        force_local=False,
        process_only_layer="rgb",
        slurm_args={"slurm_partition": "compute"},
    )
    datasets = organize_by_dataset(scan_directory_structure(in_dir), out)
    dataset = datasets[0]
    assert len(dataset.images) >= 4

    # Split into two chunks by hand and generate each.
    non_last = generate_array_job_script(
        dataset, (0, 2), config, out, chunk_id=0, is_last_chunk=False
    )
    last = generate_array_job_script(
        dataset, (2, len(dataset.images)), config, out, chunk_id=1, is_last_chunk=True
    )
    import re

    def image_list_entries(script_text: str) -> list[str]:
        m = re.search(r"IMAGE_LIST=\(\n(.*?)\n\)", script_text, re.DOTALL)
        assert m, "IMAGE_LIST block not found"
        return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]

    non_last_entries = image_list_entries(non_last.read_text())
    last_entries = image_list_entries(last.read_text())

    assert _MANIFEST_SENTINEL not in non_last_entries
    assert _MANIFEST_SENTINEL not in last_entries
    assert len(non_last_entries) == 2
    assert len(last_entries) == len(dataset.images) - 2
    last_text = last.read_text()
    assert "__PHENOTYPIC_FINALIZER__" not in last_entries
    assert "--checkpoint-type finalize" not in last_text
    assert "_cli_chunk_writer" not in last_text
