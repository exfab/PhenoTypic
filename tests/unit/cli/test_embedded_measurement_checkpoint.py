"""Checkpoint aggregation treats embedded tables as marker-authorized input."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from skimage.io import imsave

from phenotypic import ImagePipeline
from phenotypic._cli._cli_chunk_writer import flush_unchunked_measurements
from phenotypic._cli._cli_process_single import main
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.sdk_ import (
    chunk_state_path,
    master_measurements_parquet_path,
    progress_dir,
)


# The strict xfail that stood here is REMOVED, not reworded. Its mechanism was
# right and its attribution was inverted: it called the failure "PRE-EXISTING,
# and not P3's", when two long-standing preconditions plus one new one is a
# regression caused by the new one. The new one was P3 repointing only arm 2 of
# `authorized_measurement_sources`, leaving arm 1 -- the arm this stateless
# worker takes -- globbing `image_complete/`, which the publisher had just
# stopped writing. Arm 1 now scans both shapes, so the cause is gone.
#
# A strict xfail with a wrong cause is worse than none, because it becomes the
# record: it pointed a future reader at the chunk writer, where there was
# nothing to fix, and assigned it to "whoever next revisits" it -- an owner who
# does not exist in any phase.
def test_checkpoint_reads_authorized_embedded_table_into_hidden_cache(
    tmp_path: Path,
) -> None:
    """Reserved checkpoint triggers must not recreate visible split authority."""
    input_root = tmp_path / "input"
    input_root.mkdir()
    image_path = input_root / "plate.tiff"
    imsave(
        str(image_path), load_synth_yeast_plate().rgb[:], check_contrast=False
    )
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "--pipeline",
            str(pipeline),
            "--image",
            str(image_path),
            "--output-dir",
            str(output),
            "--dataset-name",
            "input",
            "--input-root",
            str(input_root),
            "--no-save-overlays",
        ],
    )
    assert result.exit_code == 0, result.output

    flush_unchunked_measurements(output)

    assert chunk_state_path(output).is_file()
    assert list((progress_dir(output) / "chunks").glob("*.parquet"))
    assert not master_measurements_parquet_path(output).exists()
    assert not (output / "results" / "input" / "measurements").exists()
