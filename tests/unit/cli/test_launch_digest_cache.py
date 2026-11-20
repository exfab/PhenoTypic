"""One digest per unchanged input file per CLI launch (``launch_digest_cache``).

A forward SLURM launch derived each image's SHA-256 three-plus times, each a
full read: ~4 TB and ~10 h before the first worker on a 10,476-photo run. The
memo removes the repeats without weakening content-change detection.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from phenotypic._cli import _cli_failure_tracker as ft
from phenotypic._cli._cli_failure_tracker import file_sha256, launch_digest_cache


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every real (streaming) digest."""
    seen: list[Path] = []
    real = ft._stream_sha256

    def _counting(target: Path) -> str:
        seen.append(Path(target))
        return real(target)

    monkeypatch.setattr(ft, "_stream_sha256", _counting)
    return seen


def _file(tmp_path: Path, data: bytes = b"a" * 4096) -> Path:
    path = tmp_path / "plate.tif"
    path.write_bytes(data)
    return path


def test_outside_a_launch_every_call_reads_the_file(tmp_path, reads) -> None:
    path = _file(tmp_path)
    file_sha256(path)
    file_sha256(path)
    assert len(reads) == 2


def test_inside_a_launch_an_unchanged_file_is_read_once(tmp_path, reads) -> None:
    path = _file(tmp_path)
    expected = ft._stream_sha256(path)
    reads.clear()
    with launch_digest_cache():
        assert file_sha256(path) == expected
        assert file_sha256(path) == expected
        assert file_sha256(tmp_path / "." / path.name) == expected  # same resolved file
    assert reads == [path]


def test_the_memo_closes_with_the_launch(tmp_path, reads) -> None:
    path = _file(tmp_path)
    with launch_digest_cache():
        file_sha256(path)
    file_sha256(path)
    assert len(reads) == 2
    assert ft._launch_digest_memo() is None


def _wait_for_a_new_ctime_tick(path: Path) -> None:
    """Block until a metadata change on *path* would carry a later ctime.

    Timestamps are coarse-grained (the kernel's coarse clock, ~1-10 ms), so an
    edit inside the tick of the last hash keeps every stat field -- the
    documented limit of the memo. Probe a scratch file rather than sleeping a
    guessed interval.
    """
    probe = path.with_name(".tick-probe")
    probe.write_bytes(b"")
    start = probe.stat().st_ctime_ns
    for _ in range(400):
        os.utime(probe)
        if probe.stat().st_ctime_ns != start:
            return
        time.sleep(0.005)
    pytest.fail("filesystem ctime did not advance within 2 s")


def test_a_same_size_rewrite_with_restored_mtime_is_rehashed(tmp_path, reads) -> None:
    """ctime is in the key: restoring mtime with utime cannot hide an edit
    made in a later timestamp tick than the hash it would replace."""
    path = _file(tmp_path, b"a" * 4096)
    with launch_digest_cache():
        before = file_sha256(path)
        st = path.stat()
        _wait_for_a_new_ctime_tick(path)
        path.write_bytes(b"b" * 4096)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert path.stat().st_mtime_ns == st.st_mtime_ns
        after = file_sha256(path)
    assert before != after
    assert after == ft._stream_sha256(path)
    assert len(reads) == 3  # two in the launch, one direct


def test_a_store_directory_is_never_memoized(tmp_path, reads) -> None:
    store = tmp_path / "p01.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    with launch_digest_cache():
        file_sha256(store)
        file_sha256(store)
    assert len(reads) == 2


def test_a_forked_child_ignores_the_inherited_memo(tmp_path, reads, monkeypatch) -> None:
    """The memo is bound to its pid, so a fork that inherits it re-reads."""
    path = _file(tmp_path)
    with launch_digest_cache():
        file_sha256(path)
        parent = os.getpid()
        monkeypatch.setattr(ft.os, "getpid", lambda: parent + 1)
        file_sha256(path)
    assert len(reads) == 2


def test_nesting_keeps_the_outer_memo(tmp_path, reads) -> None:
    path = _file(tmp_path)
    with launch_digest_cache():
        file_sha256(path)
        with launch_digest_cache():
            file_sha256(path)
        file_sha256(path)
    assert reads == [path]


def test_the_cli_runs_inside_the_launch_memo_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
) -> None:
    """phenotypic_cli opens the memo before execution and releases it after."""
    from phenotypic.phenotypicCLI import phenotypic_cli

    class _Stop(Exception):
        pass

    seen: dict[str, Any] = {}

    def _spy(config, output_manager):  # noqa: ANN001
        seen["memo"] = ft._launch_digest_memo()
        raise _Stop()

    monkeypatch.setattr("phenotypic.phenotypicCLI.create_execution_strategy", _spy)
    CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode", "full",
            "--pipeline", str(simple_pipeline_json),
            "--input", str(synth_one_level_input),
            "--output", str(tmp_path / "out"),
            "--force-local",
        ],
    )
    assert "memo" in seen, "execution strategy was never reached"
    assert seen["memo"] is not None
    assert ft._launch_digest_memo() is None
