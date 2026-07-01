"""End-to-end: a ``--metadata`` CLI run copies the source CSV to
``deliverables/metadata.csv`` (spec §8 / D6).

Phase 5 unit-tested the copy by calling ``finalize_post_master_outputs``
directly; this asserts the REAL wiring CLI → aggregate_measurements →
finalize threads ``--metadata`` through and produces the co-located copy.
Mirrors the smallest real run in ``test_phenotypic_cache_layout.py``
(``--force-local --skip-validation --njobs 1`` over the ``synth_plate_dir``
fixture), then asserts the byte-for-byte deliverable.
"""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.schema import METADATA
from phenotypic.sdk_ import metadata_csv_deliverable_path


def test_metadata_run_copies_source_csv_to_deliverables(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    out = tmp_path / "out"
    # A real, readable metadata CSV. ``Metadata_ImageName`` is a column the
    # measurement frame carries, so this also exercises a normal inner-join
    # onto the post-applied MIRROR (deliverables/measurements.parquet) — the
    # master_measurements.* archive stays metadata-free (spec §8.2). A non-ASCII
    # cell guards against an accidental text-mode re-encode in a future
    # refactor (a byte-for-byte copy preserves the UTF-8 bytes).
    source = tmp_path / "meta.csv"
    source.write_text(
        str(METADATA.IMAGE_NAME)
        + ",Metadata_Strain\nplate_001,Säccharomyces\n",
        encoding="utf-8",
    )

    res = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_plate_dir),
            "--output",
            str(out),
            "--metadata",
            str(source),
            "--force-local",
            "--skip-validation",
            "--njobs",
            "1",
        ],
    )

    assert res.exit_code == 0, res.output
    copied = metadata_csv_deliverable_path(out)
    assert copied.exists(), f"expected {copied} to exist after a --metadata run"
    assert copied.read_bytes() == source.read_bytes()
