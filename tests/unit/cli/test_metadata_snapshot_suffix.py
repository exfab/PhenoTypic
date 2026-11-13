"""User tables may be ``.csv`` or ``.tsv``; the snapshots keep their bytes and suffix.

Spec Part B4/B5 (#281 P1 Task 3). The provenance invariant under test is the
root CLAUDE.md "Metadata startup snapshot" rule: ``deliverables/metadata.<ext>``
and ``.phenotypic/reference_metadata.<ext>`` are byte-for-byte copies of the
user's input, and **no** operation rewrites them. Every path that touches a
snapshot is checked as bytes-before == bytes-after.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import tifffile

from phenotypic.sdk_ import (
    metadata_csv_deliverable_path,
    metadata_snapshot_path,
    reference_snapshot_path,
    resolve_metadata_snapshot_path,
    resolve_reference_metadata_snapshot_path,
)
from tests._tiny_run import run_cli

EXTS = (".csv", ".tsv")


def _write_source(tmp_path: Path, ext: str, metadata_bytes: dict[str, bytes]) -> Path:
    source = tmp_path / f"meta{ext}"
    source.write_bytes(metadata_bytes[ext])
    return source


# --------------------------------------------------------------------------- #
# The plan's four tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ext", EXTS)
def test_snapshot_keeps_bytes_and_extension(tmp_path, tiny_inputs, ext, metadata_bytes):
    src = _write_source(tmp_path, ext, metadata_bytes)
    result = run_cli(tiny_inputs, tmp_path / "out", extra=["--metadata", str(src)])
    assert result.exit_code == 0, result.output
    snap = tmp_path / "out" / "deliverables" / f"metadata{ext}"
    assert snap.read_bytes() == src.read_bytes()
    assert resolve_metadata_snapshot_path(tmp_path / "out") == snap
    other = ".csv" if ext == ".tsv" else ".tsv"
    assert not (tmp_path / "out" / "deliverables" / f"metadata{other}").exists()


def test_metadata_with_other_suffix_is_refused(tmp_path, tiny_inputs):
    bad = tmp_path / "meta.xlsx"
    bad.write_bytes(b"x")
    result = run_cli(tiny_inputs, tmp_path / "out", extra=["--metadata", str(bad)])
    assert result.exit_code == 2, result.output
    assert ".csv or .tsv" in result.output
    assert not (tmp_path / "out").exists()


def test_tsv_metadata_joins_like_csv(tmp_path, tiny_inputs, metadata_bytes):
    for ext in EXTS:
        src = _write_source(tmp_path, ext, metadata_bytes)
        result = run_cli(tiny_inputs, tmp_path / f"out{ext}", extra=["--metadata", str(src)])
        assert result.exit_code == 0, result.output

    def read(ext: str) -> pl.DataFrame:
        return pl.read_parquet(tmp_path / f"out{ext}" / "deliverables" / "measurements.parquet")

    tsv, csv = read(".tsv"), read(".csv")
    assert tsv.equals(csv)
    # The comma inside the quoted CSV field survived both readers intact.
    assert set(tsv["Metadata_Strain"].to_list()) == {"S288C, MATa", "BY4741"}


def test_one_argument_snapshot_path_is_unchanged(tmp_path):
    assert metadata_csv_deliverable_path(tmp_path) == tmp_path / "deliverables" / "metadata.csv"


# --------------------------------------------------------------------------- #
# Path functions and resolvers
# --------------------------------------------------------------------------- #


def test_suffixed_paths_normalise_case_and_refuse_other_suffixes(tmp_path):
    assert metadata_snapshot_path(tmp_path, ".TSV") == tmp_path / "deliverables" / "metadata.tsv"
    assert (
        reference_snapshot_path(tmp_path, ".csv")
        == tmp_path / ".phenotypic" / "reference_metadata.csv"
    )
    with pytest.raises(ValueError, match=r"\.csv or \.tsv"):
        metadata_snapshot_path(tmp_path, ".parquet")
    with pytest.raises(ValueError, match=r"\.csv or \.tsv"):
        reference_snapshot_path(tmp_path, ".xlsx")


@pytest.mark.parametrize(
    ("resolve", "build"),
    [
        (resolve_metadata_snapshot_path, metadata_snapshot_path),
        (resolve_reference_metadata_snapshot_path, reference_snapshot_path),
    ],
)
def test_resolvers_find_either_suffix_and_refuse_both(tmp_path, resolve, build):
    assert resolve(tmp_path) is None
    csv = build(tmp_path, ".csv")
    csv.parent.mkdir(parents=True)
    csv.write_bytes(b"a\n1\n")
    assert resolve(tmp_path) == csv  # an existing tree keeps metadata.csv
    tsv = build(tmp_path, ".tsv")
    tsv.write_bytes(b"a\n1\n")
    with pytest.raises(ValueError, match="Both"):
        resolve(tmp_path)
    csv.unlink()
    assert resolve(tmp_path) == tsv


# --------------------------------------------------------------------------- #
# The startup snapshot writer: bytes in == bytes on disk, on every path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ext", EXTS)
def test_startup_snapshot_writer_keeps_bytes_and_reuses_without_source(
    tmp_path, ext, metadata_bytes
):
    from phenotypic.phenotypicCLI import _snapshot_metadata_csv

    src = _write_source(tmp_path, ext, metadata_bytes)
    out = tmp_path / "out"
    snap = _snapshot_metadata_csv(out, src)
    assert snap == out / "deliverables" / f"metadata{ext}"
    before = snap.read_bytes()
    assert before == src.read_bytes()
    # A continuation without --metadata reuses the snapshot and writes nothing.
    assert _snapshot_metadata_csv(out, None) == snap
    assert snap.read_bytes() == before


def test_switching_suffix_replaces_the_snapshot_rather_than_adding_one(
    tmp_path, metadata_bytes
):
    from phenotypic.phenotypicCLI import _snapshot_metadata_csv

    out = tmp_path / "out"
    _snapshot_metadata_csv(out, _write_source(tmp_path, ".csv", metadata_bytes))
    tsv_src = _write_source(tmp_path, ".tsv", metadata_bytes)
    snap = _snapshot_metadata_csv(out, tsv_src)
    assert snap == out / "deliverables" / "metadata.tsv"
    assert snap.read_bytes() == tsv_src.read_bytes()
    assert not (out / "deliverables" / "metadata.csv").exists()
    assert resolve_metadata_snapshot_path(out) == snap


def _fail_on_write(*_args, **_kwargs):
    raise OSError("simulated quota failure")


def test_a_failed_suffix_switch_keeps_the_old_snapshot(
    tmp_path, metadata_bytes, monkeypatch
):
    """Spec B5 / review I1: publish the new suffix first, retire the old after.

    A write that fails mid-switch must leave the old snapshot in place, so a
    later continuation without ``--metadata`` still joins it rather than
    silently running with no metadata at all.
    """
    import phenotypic.phenotypicCLI as cli

    out = tmp_path / "out"
    old = cli._snapshot_metadata_csv(out, _write_source(tmp_path, ".csv", metadata_bytes))
    before = old.read_bytes()
    monkeypatch.setattr(cli, "atomic_write_bytes", _fail_on_write)

    with pytest.raises(OSError, match="simulated"):
        cli._snapshot_metadata_csv(out, _write_source(tmp_path, ".tsv", metadata_bytes))

    assert old.read_bytes() == before
    assert resolve_metadata_snapshot_path(out) == old


def test_a_failed_reference_suffix_switch_keeps_the_old_snapshot(
    tmp_path, monkeypatch
):
    """The process-mode snapshot obeys the same publish-then-retire order."""
    import phenotypic.sdk_._atomic_io as atomic_io
    from phenotypic._cli._cli_reference import snapshot_reference_metadata

    out = tmp_path / "out"
    csv_src = tmp_path / "blanks.csv"
    csv_src.write_bytes(b"ImageName,BlankImage\nt01,blank\n")
    tsv_src = tmp_path / "blanks.tsv"
    tsv_src.write_bytes(b"ImageName\tBlankImage\nt01\tblank\n")
    old = snapshot_reference_metadata(out, csv_src)
    before = old.read_bytes()
    monkeypatch.setattr(atomic_io, "atomic_write_bytes", _fail_on_write)

    with pytest.raises(OSError, match="simulated"):
        snapshot_reference_metadata(out, tsv_src)

    assert old.read_bytes() == before
    assert resolve_reference_metadata_snapshot_path(out) == old


def test_a_reference_suffix_switch_replaces_the_snapshot(tmp_path):
    from phenotypic._cli._cli_reference import snapshot_reference_metadata

    out = tmp_path / "out"
    csv_src = tmp_path / "blanks.csv"
    csv_src.write_bytes(b"ImageName,BlankImage\nt01,blank\n")
    tsv_src = tmp_path / "blanks.tsv"
    tsv_src.write_bytes(b"ImageName\tBlankImage\nt01\tblank\n")
    snapshot_reference_metadata(out, csv_src)
    snap = snapshot_reference_metadata(out, tsv_src)
    assert snap == out / ".phenotypic" / "reference_metadata.tsv"
    assert snap.read_bytes() == tsv_src.read_bytes()
    assert not (out / ".phenotypic" / "reference_metadata.csv").exists()


def test_an_unparseable_tsv_never_replaces_an_existing_snapshot(tmp_path, metadata_bytes):
    from phenotypic.phenotypicCLI import _snapshot_metadata_csv

    out = tmp_path / "out"
    snap = _snapshot_metadata_csv(out, _write_source(tmp_path, ".csv", metadata_bytes))
    before = snap.read_bytes()
    bad = tmp_path / "bad.tsv"
    bad.write_bytes(b'a\tb\n"unterminated\t1\n')
    with pytest.raises(ValueError):
        _snapshot_metadata_csv(out, bad)
    assert snap.read_bytes() == before
    assert not (out / "deliverables" / "metadata.tsv").exists()


@pytest.mark.parametrize("ext", EXTS)
def test_continuation_and_recompile_leave_the_snapshot_byte_identical(
    tmp_path, tiny_inputs, ext, metadata_bytes
):
    src = _write_source(tmp_path, ext, metadata_bytes)
    out = tmp_path / "out"
    assert run_cli(tiny_inputs, out, extra=["--metadata", str(src)]).exit_code == 0
    snap = out / "deliverables" / f"metadata{ext}"
    before = snap.read_bytes()

    continued = run_cli(tiny_inputs, out)
    assert continued.exit_code == 0, continued.output
    assert snap.read_bytes() == before

    recompiled = run_cli(None, out, mode="recompile")
    assert recompiled.exit_code == 0, recompiled.output
    assert snap.read_bytes() == before
    # The bare recompile still joined the snapshot (the fallback found it).
    mirror = pl.read_parquet(out / "deliverables" / "measurements.parquet")
    assert "Metadata_Strain" in mirror.columns


@pytest.mark.parametrize("ext", EXTS)
def test_canonical_view_is_tsv_and_leaves_the_snapshot_byte_identical(tmp_path, ext):
    from phenotypic.sdk_._hdf_to_zarr import emit_canonical_metadata_view

    snap = metadata_snapshot_path(tmp_path, ext)
    snap.parent.mkdir(parents=True)
    sep = "\t" if ext == ".tsv" else ","
    snap.write_bytes(f"Strain{sep}ImageName\nS288C{sep}007\n".encode())
    before = snap.read_bytes()

    view = emit_canonical_metadata_view(tmp_path)

    assert view == tmp_path / "deliverables" / "metadata.canonical.tsv"
    assert snap.read_bytes() == before
    frame = pd.read_csv(view, sep="\t", dtype=str)
    assert "Metadata_Strain" in frame.columns
    assert frame.iloc[0].tolist() == ["S288C", "007"]  # values are kept verbatim


# --------------------------------------------------------------------------- #
# The input readers choose the delimiter by suffix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ext", EXTS)
def test_metadata_payload_accepts_both_suffixes(tmp_path, ext, metadata_bytes):
    from phenotypic._cli._cli_reference import metadata_csv_payload

    src = _write_source(tmp_path, ext, metadata_bytes)
    assert metadata_csv_payload(src) == metadata_bytes[ext]


def test_metadata_payload_refuses_other_suffixes(tmp_path):
    from phenotypic._cli._cli_reference import metadata_csv_payload

    bad = tmp_path / "meta.parquet"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"\.csv or \.tsv"):
        metadata_csv_payload(bad)


@pytest.mark.parametrize("ext", EXTS)
def test_read_metadata_csv_uses_the_suffix_delimiter(tmp_path, ext, metadata_bytes):
    from phenotypic._cli._metadata_join import read_metadata_csv

    frame = read_metadata_csv(_write_source(tmp_path, ext, metadata_bytes))
    assert frame.columns == ["Metadata_ImageName", "Metadata_Strain"]
    assert frame["Metadata_Strain"].to_list() == ["S288C, MATa", "BY4741"]


@pytest.mark.parametrize("ext", EXTS)
def test_reference_context_reads_both_suffixes(tmp_path, ext):
    from phenotypic._core._reference_context import ReferenceContext

    sep = "\t" if ext == ".tsv" else ","
    table = tmp_path / f"blanks{ext}"
    table.write_text(f"ImageName{sep}BlankImage\n000123{sep}blank\n", encoding="utf-8")
    context = ReferenceContext(table)
    assert context.table_sha256 is not None


@pytest.mark.parametrize("ext", EXTS)
def test_metadata_migration_reads_and_writes_by_suffix(tmp_path, ext):
    from phenotypic.sdk_._metadata_migration import (
        migrate_metadata_file,
        preflight_metadata_schema,
    )

    sep = "\t" if ext == ".tsv" else ","
    table = tmp_path / f"user_table{ext}"
    table.write_text(
        f'Strain{sep}Note\nS288C{sep}"a, b"\n', encoding="utf-8"
    )
    report = preflight_metadata_schema(table)
    assert report.status != "blocked", report.conflicts
    result = migrate_metadata_file(
        table, expected_source_fingerprint=report.source_fingerprint
    )
    assert result.status == "applied", result
    frame = pd.read_csv(table, sep=sep, dtype=str)
    assert frame.columns.tolist() == ["Metadata_Strain", "Metadata_Note"]
    assert frame.iloc[0].tolist() == ["S288C", "a, b"]


@pytest.mark.parametrize("ext", [".csv", ".tsv", ".parquet"])
def test_tune_count_ground_truth_accepts_tsv(tmp_path, ext):
    from phenotypic.tune.score import GroundTruthMasks

    counts = tmp_path / f"counts{ext}"
    counts.write_bytes(b"x")
    assert GroundTruthMasks(gt_masks_source=counts).modality() == "count"


# --------------------------------------------------------------------------- #
# The process-mode reference snapshot, through the real CLI
# --------------------------------------------------------------------------- #


@pytest.fixture
def reference_run(tmp_path: Path):
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import SubtractBlank
    from phenotypic.measure import MeasureSize

    root = tmp_path / "images" / "plate1"
    root.mkdir(parents=True)
    frame = np.full((32, 32, 3), 20, dtype=np.uint8)
    tifffile.imwrite(root / "blank.tiff", frame)
    colony = frame.copy()
    colony[10:16, 10:16] = 230
    tifffile.imwrite(root / "t01.tiff", colony)
    manifest = tmp_path / "frames.txt"
    manifest.write_text("plate1/t01.tiff\n", encoding="utf-8")
    pipeline = tmp_path / "pipeline.pht-pipe.json"
    pipeline.write_text(
        ImagePipeline(
            ops={"sb": SubtractBlank(), "det": OtsuDetector()}, meas={"size": MeasureSize()}
        ).to_json(),
        encoding="utf-8",
    )
    return tmp_path, manifest, pipeline


def _process(base: Path, manifest: Path, pipeline: Path, *extra: str):
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    args = [
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--output", str(base / "out"), "--image-manifest", str(manifest),
        "--mode", "process", "--layer", "detect_mat", "--njobs", "1", *extra,
    ]  # fmt: skip
    return CliRunner().invoke(phenotypic_cli, args)


def test_process_mode_reference_snapshot_keeps_tsv_bytes_and_suffix(reference_run):
    base, manifest, pipeline = reference_run
    table = base / "blank_map.tsv"
    table.write_bytes(b"ImageName\tBlankImage\nt01\tblank\n")

    result = _process(base, manifest, pipeline, "--metadata", str(table))
    assert result.exit_code == 0, result.output
    out = base / "out"
    snap = out / ".phenotypic" / "reference_metadata.tsv"
    assert snap.read_bytes() == table.read_bytes()
    assert not (out / ".phenotypic" / "reference_metadata.csv").exists()
    assert resolve_reference_metadata_snapshot_path(out) == snap

    # A continuation without --metadata falls back to the .tsv snapshot and
    # leaves it byte-identical; --restart preserves it.
    before = snap.read_bytes()
    continued = _process(base, manifest, pipeline)
    assert continued.exit_code == 0, continued.output
    assert snap.read_bytes() == before
    restarted = _process(base, manifest, pipeline, "--restart")
    assert restarted.exit_code == 0, restarted.output
    assert snap.read_bytes() == before


def test_a_continuation_over_two_snapshots_is_refused_before_any_write(
    tmp_path, tiny_inputs, metadata_bytes
):
    """Both suffixes present is ambiguous provenance: refuse, change nothing."""
    out = tmp_path / "out"
    src = _write_source(tmp_path, ".csv", metadata_bytes)
    assert run_cli(tiny_inputs, out, extra=["--metadata", str(src)]).exit_code == 0
    stray = out / "deliverables" / "metadata.tsv"
    stray.write_bytes(metadata_bytes[".tsv"])
    state = out / ".phenotypic" / "processing_state.json"
    watched = (out / "deliverables" / "metadata.csv", stray, state)
    before = {path: path.read_bytes() for path in watched}

    result = run_cli(tiny_inputs, out, extra=["--restart"])

    assert result.exit_code == 2, result.output
    assert "Both metadata.tsv and metadata.csv" in result.output
    assert {path: path.read_bytes() for path in watched} == before
