"""End-to-end full-forward provenance and decoded-original retention."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image as PILImage

from phenotypic import GridImage, Image, ImagePipeline
from phenotypic.enhance import BlurGauss
from phenotypic._cli._cli_state_management import load_processing_state
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import zarr_store_path
from tests._ngff_conformance import assert_store_conforms


@pytest.mark.parametrize(
    ("drop_originals", "expected_original"),
    [(False, True), (True, False)],
)
def test_full_local_cli_publishes_complete_provenance_and_original_policy(
    tmp_path: Path,
    synth_plate_dir: Path,
    simple_pipeline_json: Path,
    drop_originals: bool,
    expected_original: bool,
) -> None:
    output_dir = tmp_path / "out"
    args = [
        "--pipeline",
        str(simple_pipeline_json),
        "--input",
        str(synth_plate_dir),
        "--output",
        str(output_dir),
        "--njobs",
        "1",
        "--skip-validation",
        "--force-local",
    ]
    if drop_originals:
        args.append("--drop-originals")

    result = CliRunner().invoke(phenotypic_cli, args)
    assert result.exit_code == 0, result.output

    store = zarr_store_path(output_dir, "plates", "plate_001")
    assert_store_conforms(store)
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    journal = root["attributes"]["phenotypic"]["provenance"]
    expected_ops = ImagePipeline.from_json(simple_pipeline_json).get_ops()
    application = journal["applications"][-1]
    assert journal["schema_version"] == 2
    assert journal["status"] == "complete"
    assert journal["original_filename"] == "plate_001.png"
    assert application["status"] == "complete"
    assert application["kind"] == "full"
    assert application["input_filename"] == "plate_001.png"
    assert application["pipeline"] == {
        "source_path": simple_pipeline_json.name,
        "sha256": hashlib.sha256(simple_pipeline_json.read_bytes()).hexdigest(),
    }
    assert application["retry_base_length"] == 0
    assert [entry["sequence"] for entry in application["operations"]] == [1, 2]
    assert [entry["operation_name"] for entry in application["operations"]] == [
        type(operation).__name__ for operation in expected_ops.values()
    ]
    assert [
        entry["pipeline_step_path"] for entry in application["operations"]
    ] == [[key] for key in expected_ops]
    assert all(
        entry["duration_seconds"] >= 0 for entry in application["operations"]
    )
    assert "provenance" not in root["attributes"].get("ome", {})

    ome_payload = json.loads(
        (store / "OME" / "zarr.json").read_text(encoding="utf-8")
    )
    series = ome_payload["attributes"]["ome"]["series"]
    assert "provenance" not in ome_payload["attributes"]["ome"]
    ome_xml = (store / "OME" / "METADATA.ome.xml").read_text(
        encoding="utf-8"
    )
    assert "provenance" not in ome_xml

    loaded = GridImage.load_zarr(store)
    if expected_original:
        assert series[-1] == "original"
        assert (store / "original" / "zarr.json").is_file()
        original_metadata = json.loads(
            (store / "original" / "zarr.json").read_text(encoding="utf-8")
        )
        datasets = original_metadata["attributes"]["ome"]["multiscales"][0][
            "datasets"
        ]
        assert len(datasets) > 1
        assert all(
            (store / "original" / dataset["path"] / "zarr.json").is_file()
            for dataset in datasets
        )
        assert loaded._original is not None
        input_image = next(synth_plate_dir.glob("*.png"))
        with PILImage.open(input_image) as source:
            decoded = np.asarray(source.convert("RGB"))
        np.testing.assert_array_equal(loaded._original, decoded)
        assert 'Name="original"' in ome_xml
    else:
        assert "original" not in series
        assert not (store / "original").exists()
        assert loaded._original is None
        assert 'Name="original"' not in ome_xml

        source_input = next(synth_plate_dir.glob("*.png"))
        shutil.copyfile(source_input, synth_plate_dir / "plate_002.png")
        continued = CliRunner().invoke(phenotypic_cli, args)
        assert continued.exit_code == 0, continued.output
        state = load_processing_state(output_dir)
        assert state is not None
        assert state.config["drop_originals"] is True
        second_store = zarr_store_path(
            output_dir, "plates", "plate_002"
        )
        assert second_store.is_dir()
        assert not (second_store / "original").exists()


def test_process_store_into_full_cli_preserves_both_applications(
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
) -> None:
    source_image = next(synth_one_level_input.rglob("*.tif"))
    process_pipeline = tmp_path / "process.json.pht-pipe"
    ImagePipeline(ops={"process-blur": BlurGauss(sigma=0.75)}).to_json(
        process_pipeline
    )
    process_output = tmp_path / "process-output"

    processed = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(process_pipeline),
            "--input",
            str(source_image),
            "--output",
            str(process_output),
            "--mode",
            "process",
            "--layer",
            "rgb",
            "--process-format",
            "zarr",
            "--njobs",
            "1",
            "--skip-validation",
            "--force-local",
        ],
    )
    assert processed.exit_code == 0, processed.output
    process_stores = list(process_output.rglob("*.ome.zarr"))
    assert len(process_stores) == 1
    process_store = process_stores[0]
    assert Image.imread(process_store)._metadata.provenance_journal[
        "applications"
    ][0]["kind"] == "process"
    from phenotypic.gui.results_viewer._store_source import build_source_spec

    source_spec = build_source_spec(process_store, "/browse/process.ome.zarr")
    assert source_spec["seriesPath"] == "rgb"
    assert source_spec["labelPath"] is None

    full_output = tmp_path / "full-output"
    completed = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(process_store),
            "--output",
            str(full_output),
            "--njobs",
            "1",
            "--skip-validation",
            "--force-local",
        ],
    )
    assert completed.exit_code == 0, completed.output
    full_stores = list(full_output.rglob("*.ome.zarr"))
    assert len(full_stores) == 1
    full_store = full_stores[0]

    root = json.loads((full_store / "zarr.json").read_text(encoding="utf-8"))
    journal = root["attributes"]["phenotypic"]["provenance"]
    applications = journal["applications"]
    assert journal["original_filename"] == source_image.name
    assert [application["kind"] for application in applications] == [
        "process",
        "full",
    ]
    assert [application["input_filename"] for application in applications] == [
        source_image.name,
        process_store.name,
    ]
    assert [application["pipeline"]["source_path"] for application in applications] == [
        process_pipeline.name,
        simple_pipeline_json.name,
    ]
    assert [application["pipeline"]["sha256"] for application in applications] == [
        hashlib.sha256(process_pipeline.read_bytes()).hexdigest(),
        hashlib.sha256(simple_pipeline_json.read_bytes()).hexdigest(),
    ]
    operation_sequences = [
        operation["sequence"]
        for application in applications
        for operation in application["operations"]
    ]
    assert operation_sequences == list(range(1, len(operation_sequences) + 1))
    assert GridImage.load_zarr(full_store)._metadata.provenance_journal == journal


def test_ordinary_worker_preserves_explicit_user_pipeline_identity(
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
) -> None:
    from phenotypic._cli._cli_process_single import main as process_single

    snapshot = tmp_path / "submission" / "pipeline.snapshot.json"
    snapshot.parent.mkdir()
    snapshot.write_bytes(simple_pipeline_json.read_bytes())
    image = next(synth_one_level_input.rglob("*.tif"))
    output_dir = tmp_path / "out"
    explicit_identity = {
        "source_path": str(simple_pipeline_json.resolve()),
        "sha256": hashlib.sha256(simple_pipeline_json.read_bytes()).hexdigest(),
    }

    result = CliRunner().invoke(
        process_single,
        [
            "--pipeline",
            str(snapshot),
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "ds",
            "--image-type",
            "Image",
            "--no-save-overlays",
            "--provenance-pipeline-source-path",
            explicit_identity["source_path"],
            "--provenance-pipeline-sha256",
            explicit_identity["sha256"],
        ],
    )

    assert result.exit_code == 0, result.output
    store = zarr_store_path(output_dir, "ds", image.stem)
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    journal = root["attributes"]["phenotypic"]["provenance"]
    application = journal["applications"][-1]
    assert journal["status"] == "complete"
    assert application["pipeline"] == {
        **explicit_identity,
        "source_path": simple_pipeline_json.name,
    }
    assert application["pipeline"]["source_path"] != snapshot.name
