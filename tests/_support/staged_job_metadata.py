"""The ``job_metadata.json`` a staged SLURM submitter leaves for its workers.

The staged SLURM Stage-1 worker reads ``--nrows``/``--ncols``/``--bit-depth``/
``--detect-mode`` from that file and refuses to run without it, because the
submitter always writes it before any worker starts. Tests that drive
``run_stage1_step`` directly, without the submitter, write it with this.

It writes only the keys the worker reads. A test of the writer itself goes
through the real ``_write_staged_job_metadata`` instead
(``tests/unit/cli/test_staged_stage1_reads_like_single_pass.py``).
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.sdk_ import JobMetadataKey, atomic_write_json, job_metadata_path


def write_staged_job_metadata(
    output_dir: Path,
    *,
    nrows: int | None = None,
    ncols: int | None = None,
    bit_depth: int | None = None,
    detect_mode: str = "gray",
) -> Path:
    """Write the reader flags a staged SLURM run records, defaults unset."""
    path = job_metadata_path(Path(output_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            JobMetadataKey.NROWS: nrows,
            JobMetadataKey.NCOLS: ncols,
            JobMetadataKey.BIT_DEPTH: bit_depth,
            JobMetadataKey.DETECT_MODE: detect_mode,
        },
    )
    return path
