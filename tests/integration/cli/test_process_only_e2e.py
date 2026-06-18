"""End-to-end process-mode CLI run: mirrored layer files + manifest, no suite."""

import numpy as np
import tifffile
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    deliverables_dir,
    manifest_json_path,
    phenotypic_cache_dir,
    results_dir,
)


def test_process_only_end_to_end(tmp_path, synth_one_level_input, simple_pipeline_json):
    out = tmp_path / "out"
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
    tiffs = list(out.rglob("*_detect_mat.tiff"))
    assert tiffs, "no mirrored tiffs"
    # detect_mat float TIFF (imsave, full precision)
    assert np.issubdtype(tifffile.imread(tiffs[0]).dtype, np.floating)
    assert manifest_json_path(out).is_file()  # run-console visibility
    assert phenotypic_cache_dir(out).is_dir()
    assert not deliverables_dir(out).exists()  # no analysis suite
    assert not results_dir(out).exists()
