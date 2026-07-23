"""Tests for the CLI's exact reserved GUI-log freshness exception."""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.phenotypicCLI import is_safe_gui_log_entry
from phenotypic.sdk_ import RUN_LOG_DIRNAME, STDOUT_LOG


def test_stdout_only_gui_log_directory_is_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()
    (log_dir / STDOUT_LOG).write_text("", encoding="utf-8")

    assert is_safe_gui_log_entry(log_dir)


def test_empty_real_gui_log_directory_is_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()

    assert is_safe_gui_log_entry(log_dir)


@pytest.mark.parametrize("child_name", ["stderr.log", "notes.txt"])
def test_unrecognized_gui_log_file_is_not_safe(
    tmp_path: Path, child_name: str
) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()
    (log_dir / child_name).write_text("", encoding="utf-8")

    assert not is_safe_gui_log_entry(log_dir)


def test_nested_gui_log_directory_is_not_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    (log_dir / "nested").mkdir(parents=True)

    assert not is_safe_gui_log_entry(log_dir)


def test_wrong_directory_name_is_not_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / ".other-log"
    log_dir.mkdir()
    (log_dir / STDOUT_LOG).write_text("", encoding="utf-8")

    assert not is_safe_gui_log_entry(log_dir)


def test_symlinked_gui_log_directory_is_not_safe(tmp_path: Path) -> None:
    target = tmp_path / "real-log"
    target.mkdir()
    (target / STDOUT_LOG).write_text("", encoding="utf-8")
    link = tmp_path / RUN_LOG_DIRNAME
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")

    assert not is_safe_gui_log_entry(link)


def test_symlinked_allowed_log_file_is_not_safe(tmp_path: Path) -> None:
    target = tmp_path / "real.log"
    target.write_text("", encoding="utf-8")
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()
    link = log_dir / STDOUT_LOG
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are unavailable")

    assert not is_safe_gui_log_entry(log_dir)
