"""Byte-stable deliverables (spec C11, #281 P2 Task 4).

DVC re-runs every downstream stage when an upstream out's bytes change, so two
runs over identical inputs, pipeline and PhenoTypic version must publish
byte-identical tables, ``metrics.json`` and ``schema.json``. Reports,
dashboards and ``.phenotypic/`` are exempt (C11).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from phenotypic._cli._cli_failure_tracker import file_sha256
from tests._tiny_run import run_cli

#: Named deliverables C11 pins, beside every table found by suffix.
STABLE = ("metrics.json", "schema.json")
#: Every table under ``deliverables/`` (spec C11: "Parquet and TSV").
#: ``.csv`` covers the curated mirror twin ``measurements.csv`` (spec B9).
TABLE_SUFFIXES = (".parquet", ".tsv", ".csv")
#: Folders whose every file is pinned, whatever its suffix.
STABLE_DIRS = ("embeddings",)


def _digests(out: Path) -> dict[str, str]:
    d = out / "deliverables"
    files = {d / name for name in STABLE if (d / name).exists()}
    files |= {p for p in d.rglob("*") if p.is_file() and p.suffix in TABLE_SUFFIXES}
    for sub in STABLE_DIRS:
        files |= {p for p in (d / sub).rglob("*") if p.is_file()}
    return {str(p.relative_to(d)): file_sha256(p) for p in sorted(files)}


def _run_in_fresh_process(
    inputs: Path,
    out: Path,
    pipeline: Path,
    *,
    hash_seed: int,
    njobs: int,
    extra: Sequence[str] = (),
) -> None:
    """One ``python -m phenotypic`` run in its own interpreter.

    In-process runs share a pid, a hash seed and module-level caches, so a
    set-ordered column list or a pid in a table would pass unseen; separate
    interpreters with distinct ``PYTHONHASHSEED`` values do not share them.
    """
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [
            sys.executable, "-m", "phenotypic",
            "--input", str(inputs), "--output", str(out), "--mode", "full",
            "--pipeline", str(pipeline), "--njobs", str(njobs), *extra,
        ],
        env=env, capture_output=True, text=True, timeout=900, check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_identical_runs_give_identical_deliverables(
    tiny_inputs: Path, tiny_pipeline_file: Path, tmp_path: Path
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    # One in-process run (pytest's own hash seed, --njobs 1) against one in a
    # fresh interpreter (PYTHONHASHSEED=1, --njobs 2): they share no pid, no
    # hash seed, no module cache and no worker layout.
    result = run_cli(tiny_inputs, a, pipeline=tiny_pipeline_file)
    assert result.exit_code == 0, result.output
    _run_in_fresh_process(tiny_inputs, b, tiny_pipeline_file, hash_seed=1, njobs=2)
    _assert_identical_deliverables(a, b)


def test_identical_runs_with_metadata_and_two_datasets_give_identical_deliverables(
    tiny_inputs: Path, tiny_pipeline_file: Path, tmp_path: Path
) -> None:
    """#281 P2 review: the join, the phantom append and the dataset order are pinned too.

    The plain test runs one dataset with no ``--metadata``, so the mirror's
    join, its metadata-only phantom rows and the dataset order of the master's
    concatenation were never compared -- the configuration a DVC user runs.
    """
    import polars as pl

    second = tiny_inputs / "ds2"
    second.mkdir()
    for index, plate in enumerate(sorted((tiny_inputs / "ds").glob("*.tif")), start=3):
        (second / f"plate_{index}.tif").write_bytes(plate.read_bytes())
    meta = tmp_path / "meta.csv"
    rows = [f"plate_{i},strain_{i}" for i in range(1, 5)]
    rows += ["plate_ghost_a,strain_ghost", "plate_ghost_b,strain_ghost"]  # phantoms
    meta.write_text(
        "\n".join(["Metadata_ImageName,Metadata_Strain", *rows]) + "\n", encoding="utf-8"
    )
    a, b = tmp_path / "a", tmp_path / "b"
    extra = ["--metadata", str(meta)]
    result = run_cli(tiny_inputs, a, pipeline=tiny_pipeline_file, extra=extra)
    assert result.exit_code == 0, result.output
    _run_in_fresh_process(
        tiny_inputs, b, tiny_pipeline_file, hash_seed=2, njobs=2, extra=extra
    )
    # The premise: two datasets were measured and both phantoms were appended.
    mirror = pl.read_parquet(a / "deliverables" / "measurements.parquet")
    assert set(mirror["Metadata_Dataset"].drop_nulls()) == {"ds", "ds2"}
    assert int(mirror["QC_MetadataOnly"].sum()) == 2
    _assert_identical_deliverables(a, b)


def _assert_identical_deliverables(a: Path, b: Path) -> None:
    digests_a, digests_b = _digests(a), _digests(b)
    # The comparison only means something if it covers the C11 set.
    for required in (
        "measurements.parquet",
        "measurements.tsv",
        "master_measurements.parquet",
        "metrics.json",
        "schema.json",
    ):
        assert required in digests_a, sorted(digests_a)
    assert any(k.startswith("measurements_by_feature/") for k in digests_a)
    assert any(k.startswith("measurements_by_category/") for k in digests_a)
    differing = sorted(
        k for k in digests_a.keys() | digests_b.keys() if digests_a.get(k) != digests_b.get(k)
    )
    assert not differing, differing
