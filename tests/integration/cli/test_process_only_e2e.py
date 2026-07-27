"""End-to-end process-mode CLI run: mirrored layer files + manifest, no suite."""

import json
from uuid import uuid4

import numpy as np
import tifffile
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    deliverables_dir,
    manifest_json_path,
    phenotypic_cache_dir,
    processing_state_path,
    rembi_manifest_path,
    results_dir,
    run_completion_marker_path,
)
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR


def test_process_only_end_to_end(
    tmp_path,
    synth_one_level_input,
    simple_pipeline_json,
    monkeypatch,
):
    out = tmp_path / "out"
    generation = uuid4()
    monkeypatch.setenv(GUI_RECORD_GENERATION_ENV_VAR, str(generation))
    r = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline", str(simple_pipeline_json),
            "--input", str(synth_one_level_input),
            "--output", str(out),
            "--mode", "process",
            "--layer", "detect_mat",
            "--force-local", "--njobs", "1",
        ],
    )
    assert r.exit_code == 0, r.output
    tiffs = [
        path
        for path in out.rglob("*.tiff")
        if not path.is_relative_to(phenotypic_cache_dir(out))
    ]
    assert tiffs, "no mirrored tiffs"
    assert not list(out.rglob("*_detect_mat.tiff"))
    # detect_mat float TIFF (imsave, full precision)
    assert np.issubdtype(tifffile.imread(tiffs[0]).dtype, np.floating)
    assert manifest_json_path(out).is_file()  # run-console visibility
    completion = json.loads(
        run_completion_marker_path(out).read_text(encoding="utf-8")
    )
    assert completion["generation"] == str(generation)
    assert completion["mode"] == "local"
    assert phenotypic_cache_dir(out).is_dir()
    state = json.loads(processing_state_path(out).read_text(encoding="utf-8"))
    assert state["config"]["process_only_layer"] == "detect_mat"
    assert not deliverables_dir(out).exists()  # no analysis suite
    assert not rembi_manifest_path(out).exists()  # no REMBI manifest in process mode
    assert not results_dir(out).exists()
    assert not (out / "logs").exists()
    assert not (out / "slurm_scripts").exists()
