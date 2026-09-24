"""Output-location checks of the run preflight (spec 2026-09-24-cli-preflight §9).

Findings F25, F29; review R21, R30. The node-local check reads the filesystem
type of each path's mount from ``/proc/self/mounts``; these tests replace that
read with a scripted mount table so they run the same on any host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_preflight
from phenotypic._cli._cli_preflight import check_output_location
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.post import JoinMetadata
from tests.unit.cli._preflight_support import make_context, make_datasets


def _pipeline(**post) -> ImagePipeline:
    return ImagePipeline(ops={"d": OtsuDetector()}, meas={"s": MeasureSize()}, post=post or None)


def _by_code(findings) -> dict[str, object]:
    return {f.code: f for f in findings}


# --- writability ----------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses permissions")
def test_an_unwritable_ancestor_is_an_error(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        (finding,) = check_output_location(
            make_context(_pipeline(), output_dir=locked / "run" / "out")
        )
    finally:
        locked.chmod(0o755)

    assert finding.code == "PF-OUTPUT-UNWRITABLE" and finding.severity == "error"
    assert str(locked) in finding.message


def test_a_writable_output_that_does_not_exist_yet_is_fine(tmp_path: Path) -> None:
    assert check_output_location(make_context(_pipeline(), output_dir=tmp_path / "new" / "out")) == []


# --- free space -----------------------------------------------------------------


def _big_input(tmp_path: Path, size: int = 4096) -> tuple:
    image = tmp_path / "images" / "img001.tiff"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\0" * size)
    return make_datasets(image)


def _short_on_space(monkeypatch, free: int = 1024) -> None:
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10**9, 10**9 - free, free))


def test_less_space_than_the_inputs_warns_as_a_heuristic(tmp_path: Path, monkeypatch) -> None:
    datasets = _big_input(tmp_path)
    _short_on_space(monkeypatch)

    (finding,) = check_output_location(
        make_context(_pipeline(), datasets=datasets, output_dir=tmp_path / "out")
    )

    assert finding.code == "PF-OUTPUT-SPACE" and finding.severity == "warning"
    assert "heuristic" in finding.message


@pytest.mark.parametrize("mode", ["process", "measure"])
def test_the_space_heuristic_runs_in_full_mode_only(tmp_path: Path, monkeypatch, mode: str) -> None:
    datasets = _big_input(tmp_path)
    _short_on_space(monkeypatch)

    assert check_output_location(
        make_context(_pipeline(), mode, datasets, output_dir=tmp_path / "out")
    ) == []


def test_enough_space_is_fine(tmp_path: Path) -> None:
    datasets = _big_input(tmp_path)

    assert check_output_location(
        make_context(_pipeline(), datasets=datasets, output_dir=tmp_path / "out")
    ) == []


# --- node-local storage (F29, R30) ------------------------------------------------


MOUNTS = "\n".join(
    [
        "rootfs / ext4 rw 0 0",
        "tmpfs /tmp tmpfs rw 0 0",
        "scratch /scratch xfs rw 0 0",
        "gpfs0 /bigdata gpfs rw 0 0",
        "lustre /lus lustre rw 0 0",
        "home:/export /rhome nfs4 rw 0 0",
        "tmpfs /bigdata/with\\040space tmpfs rw 0 0",
    ]
) + "\n"


@pytest.fixture
def mounts(monkeypatch) -> None:
    monkeypatch.setattr(_cli_preflight, "_mounts_text", lambda: MOUNTS)


def _slurm_context(pipeline=None, **paths):
    values = dict(
        output_dir=Path("/bigdata/run/out"),
        input_path=Path("/bigdata/images"),
        pipeline_json=Path("/rhome/me/pipeline.json"),
        metadata_csv=None,
    )
    values.update(paths)
    return make_context(
        pipeline or _pipeline(),
        slurm_args={"slurm_partition": "short"},
        force_local=False,
        **values,
    )


def _node_local(findings):
    return _by_code(findings).get("PF-NODE-LOCAL")


def test_shared_filesystems_produce_nothing(mounts) -> None:
    assert _node_local(check_output_location(_slurm_context(
        input_path=Path("/lus/project/images"),
        metadata_csv=Path("/rhome/me/meta.csv"),
    ))) is None


@pytest.mark.parametrize(
    ("option", "value", "fs"),
    [
        ("output_dir", Path("/tmp/run/out"), "tmpfs"),
        ("output_dir", Path("/scratch/job/out"), "xfs"),
        ("input_path", Path("/home/me/images"), "ext4"),
        ("pipeline_json", Path("/tmp/pipeline.json"), "tmpfs"),
        ("metadata_csv", Path("/scratch/meta.csv"), "xfs"),
        ("input_path", Path("/bigdata/with space/images"), "tmpfs"),
    ],
)
def test_each_node_local_path_is_named(mounts, option: str, value: Path, fs: str) -> None:
    finding = _node_local(check_output_location(_slurm_context(**{option: value})))

    assert finding is not None and finding.severity == "warning"
    (subject,) = finding.subjects
    assert str(value) in subject and f"({fs})" in subject


def test_a_join_metadata_table_on_node_local_storage_is_named(tmp_path: Path, monkeypatch) -> None:
    # JoinMetadata reads its table when constructed, so the file must exist;
    # the scripted mount table declares tmp_path node-local.
    table = tmp_path / "plate_map.csv"
    table.write_text("ImageName,Strain\nimg001,WT\n", encoding="utf-8")
    monkeypatch.setattr(
        _cli_preflight, "_mounts_text", lambda: MOUNTS + f"tmpfs {tmp_path.resolve()} tmpfs rw 0 0\n"
    )
    pipeline = _pipeline(join=JoinMetadata(metadata=table, on=["ImageName"]))

    finding = _node_local(check_output_location(_slurm_context(pipeline)))

    assert finding is not None
    assert any("plate_map.csv" in subject for subject in finding.subjects)


def test_a_local_run_does_not_check_mounts(mounts) -> None:
    context = make_context(_pipeline(), output_dir=Path("/tmp/run/out"), input_path=Path("/tmp/in"))

    assert _node_local(check_output_location(context)) is None


def test_no_mount_table_means_no_finding(monkeypatch) -> None:
    """Off Linux there is no /proc/self/mounts; the check skips rather than guess."""
    monkeypatch.setattr(_cli_preflight, "_mounts_text", lambda: None)

    assert _node_local(check_output_location(_slurm_context(output_dir=Path("/tmp/out")))) is None


def test_the_longest_mount_wins() -> None:
    fs = _cli_preflight._filesystem_type(Path("/bigdata/with space/x"), MOUNTS)

    assert fs == "tmpfs"
    assert _cli_preflight._filesystem_type(Path("/bigdata/x"), MOUNTS) == "gpfs"
