"""``PF-DVC-LINKED-CACHE``: a DVC project whose cache links files (#281 spec C6).

DVC makes linked (``hardlink``/``symlink``) files read-only, and PhenoTypic
appends in place to files under ``.phenotypic/``, so a linked cache over the
run's outs is unsupported. The check reads only the project's ``.dvc/config``
and ``.dvc/config.local``, with the stdlib, and never imports DVC.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from phenotypic._cli._cli_preflight import (
    CHECKS,
    HINTS,
    _check_dvc_cache_link_type,
    find_dvc_cache_type,
)
from tests._tiny_run import make_dvc_repo, run_preflight

CODE = "PF-DVC-LINKED-CACHE"


@pytest.mark.parametrize(
    ("value", "warns"),
    [
        ("hardlink", True),
        ("symlink,copy", True),
        ("reflink,hardlink,copy", True),
        ("reflink,copy", False),
        ("copy", False),
        (None, False),
    ],
)
def test_linked_cache_warns(tmp_path: Path, value: str | None, warns: bool) -> None:
    make_dvc_repo(tmp_path, cache_type=value)
    findings = run_preflight(output_dir=tmp_path / "out")
    hits = [f for f in findings if f.code == CODE]
    assert bool(hits) is warns
    assert all(f.severity == "warning" for f in hits)
    assert not [f for f in findings if f.code == "PF-CHECK-CRASHED"]


def test_the_message_names_the_file_the_append_risk_and_the_unread_configs(
    tmp_path: Path,
) -> None:
    make_dvc_repo(tmp_path, cache_type="hardlink")
    (finding,) = [f for f in run_preflight(output_dir=tmp_path / "out") if f.code == CODE]
    assert str(tmp_path / ".dvc" / "config") in finding.message
    assert ".phenotypic/" in finding.message and "append" in finding.message
    assert "global" in finding.message and "system" in finding.message
    assert finding.hint == HINTS[CODE]


def test_config_local_overrides_config(tmp_path: Path) -> None:
    make_dvc_repo(tmp_path, cache_type="reflink,copy", local_type="hardlink")
    assert find_dvc_cache_type(tmp_path / "out") == "hardlink"
    (finding,) = [f for f in run_preflight(output_dir=tmp_path / "out") if f.code == CODE]
    assert str(tmp_path / ".dvc" / "config.local") in finding.message


def test_config_local_can_clear_a_linked_config(tmp_path: Path) -> None:
    make_dvc_repo(tmp_path, cache_type="symlink", local_type="reflink,copy")
    assert find_dvc_cache_type(tmp_path) == "reflink,copy"
    assert not [f for f in run_preflight(output_dir=tmp_path / "out") if f.code == CODE]


def test_the_nearest_project_wins_and_the_walk_starts_at_a_missing_output(
    tmp_path: Path,
) -> None:
    make_dvc_repo(tmp_path, cache_type="hardlink")
    inner = tmp_path / "sub" / "project"
    make_dvc_repo(inner, cache_type="reflink,copy")
    assert find_dvc_cache_type(inner / "not" / "yet" / "out") == "reflink,copy"
    assert find_dvc_cache_type(tmp_path / "sub" / "out") == "hardlink"


def test_outside_a_dvc_project_there_is_nothing_to_read(tmp_path: Path) -> None:
    assert find_dvc_cache_type(tmp_path / "out") is None


def test_quoted_and_mixed_case_values_are_read(tmp_path: Path) -> None:
    dvc = tmp_path / ".dvc"
    dvc.mkdir()
    (dvc / "config").write_text(
        "[core]\n    remote = store\n['remote \"store\"']\n    url = /x\n"
        "[cache]\n    type = \"Reflink, HardLink\"\n",
        encoding="utf-8",
    )
    assert find_dvc_cache_type(tmp_path) == "Reflink, HardLink"
    assert [f for f in run_preflight(output_dir=tmp_path / "out") if f.code == CODE]


def test_the_check_is_registered() -> None:
    assert _check_dvc_cache_link_type in CHECKS


def test_the_check_never_imports_dvc(tmp_path: Path) -> None:
    make_dvc_repo(tmp_path, cache_type="hardlink")
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from phenotypic._cli._cli_preflight import find_dvc_cache_type\n"
        f"assert find_dvc_cache_type(Path({str(tmp_path)!r})) == 'hardlink'\n"
        "assert not any(m == 'dvc' or m.startswith('dvc.') for m in sys.modules), "
        "sorted(m for m in sys.modules if m.startswith('dvc'))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
