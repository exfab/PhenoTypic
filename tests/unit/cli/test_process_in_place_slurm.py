"""The SLURM path of an in-place process run (#281 P5 review I4, spec E7).

Nothing local exercises it: the worklist's expected input digests, and the
per-image worker's in-place derivation, identity, expected-input check and
dispatch. Torch-free: the toy grid embedder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from click.testing import CliRunner

from phenotypic.sdk_ import image_record_path, list_table_runs
from tests._tiny_run import toy_grid_process_pipeline


def _in_place_config(make_exec_config, tree: Path, pipeline: Path):
    return make_exec_config(
        pipeline_json=pipeline,
        input_path=tree,
        output_dir=tree,
        image_type="Image",
        force_local=False,
        slurm_args={"slurm_partition": "short"},
        process_only_layer="rgb",
        process_format="zarr",
    )


def _stores(tree: Path) -> list[Path]:
    return sorted(tree.rglob("*.ome.zarr"))


def test_the_worklist_carries_the_in_place_input_digest(
    toy_process_store_tree, tmp_path, make_exec_config
):
    """Each row's expected input digest is the one the worker will compute."""
    from phenotypic._cli._cli_process_in_place import in_place_input_digest
    from phenotypic._cli._cli_slurm_array_scripts import generate_array_job_script
    from phenotypic._cli._cli_types import Dataset

    tree = toy_process_store_tree
    pipeline = toy_grid_process_pipeline(tmp_path, token="toy-v2")
    config = _in_place_config(make_exec_config, tree, pipeline)
    assert config.in_place
    stores = _stores(tree)
    dataset = Dataset("ds", stores, tree, tree)
    script = generate_array_job_script(dataset, (0, len(stores)), config, tree).read_text()
    block = re.search(r"EXPECTED_INPUT_SHA256S=\(\n(.*?)\n\)", script, re.S)
    assert block is not None, script
    rows = [line.strip().strip("'") for line in block.group(1).splitlines()]
    assert rows == [in_place_input_digest(store) for store in stores]


def test_the_worker_writes_folder_2_and_publishes_its_record(
    toy_process_store_tree, tmp_path, make_exec_config, registered_toy_embedder
):
    from phenotypic._cli._cli_failure_tracker import file_sha256, work_id_for_image
    from phenotypic._cli._cli_process_in_place import in_place_input_digest
    from phenotypic._cli._cli_process_single import main as worker

    tree = toy_process_store_tree
    pipeline = toy_grid_process_pipeline(tmp_path, token="toy-v2")
    config = _in_place_config(make_exec_config, tree, pipeline)
    store = _stores(tree)[0]
    work_id, _ = work_id_for_image(config, "ds", store)
    result = CliRunner().invoke(
        worker,
        [
            "--pipeline", str(pipeline),
            "--image", str(store),
            "--output-dir", str(tree),
            "--dataset-name", "ds",
            "--input-root", str(tree),
            "--image-type", "Image",
            "--mode", "process",
            "--layer", "rgb",
            "--process-format", "zarr",
            "--expected-work-id", work_id,
            "--expected-input-sha256", in_place_input_digest(store),
            "--expected-pipeline-sha256", file_sha256(pipeline),
            "--attempt-id", "a" * 32,
        ],
    )
    assert result.exit_code == 0, result.output
    assert [run.order for run in list_table_runs(store)] == [1, 2]
    assert set(list_table_runs(store)[-1].descriptor["embeddings"]) == {"toy-v2/grid"}
    record = json.loads(
        image_record_path(tree, "ds", "plate_1").read_text(encoding="utf-8")
    )
    assert record["work_id"] == work_id
    assert record["artifacts"]["process_output"]["path"] == "ds/plate_1.ome.zarr"
    # The worker's own write leaves its work id where it was.
    assert work_id_for_image(config, "ds", store)[0] == work_id
