"""Every bundled sample-data file is matched by a ``package-data`` glob.

``phenotypic.data`` loaders read files from ``src/phenotypic/data/`` at run
time -- ``load_synth_yeast_plate`` reads ``synthetic_test_plate/``,
``load_yeast_plate`` reads ``snp-imager-samples/``, ``load_meas`` reads
``meas/``. setuptools copies a non-Python file into the wheel only when a
``[tool.setuptools.package-data]`` glob matches it, so a folder the globs miss
works from a source checkout and raises ``FileNotFoundError`` after
``pip install``. Those three folders were missed until this test existed.

The check expands the configured globs the way setuptools' ``build_py`` does
(``glob(os.path.join(package_dir, pattern), recursive=True)``, keeping files
only), so it needs no wheel build. The file list comes from ``git ls-files``,
so a data folder added later is checked without anyone editing this test.
``tests/integration/packaging/test_package_contents.py`` is the slow
counterpart that builds the wheel and reads its contents.
"""

from __future__ import annotations

import glob
import os
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "src" / "phenotypic"
DATA_DIR = PACKAGE_DIR / "data"

#: A file a loader reads, so an empty or mis-rooted file list cannot pass.
KNOWN_DATA_FILE = "data/synthetic_test_plate/yeast_plate_rgb.png"


def _package_data_config() -> dict:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return pyproject["tool"]["setuptools"]


def _tracked_data_files() -> list[str]:
    """Non-Python files under ``data/``, relative to the package, POSIX-style.

    Uses ``git ls-files`` so untracked local scratch files cannot fail the
    test. Outside a git checkout (an unpacked sdist), falls back to walking
    the directory, which still fails loudly rather than skipping.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", str(DATA_DIR.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        paths = [
            REPO_ROOT / name
            for name in result.stdout.decode().split("\0")
            if name
        ]
    else:
        paths = [
            path
            for path in DATA_DIR.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
    return sorted(
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in paths
        if path.suffix not in {".py", ".pyc"}
    )


def _files_matched_by_package_data() -> set[str]:
    """Package-relative paths the ``phenotypic`` package-data globs select."""
    patterns = _package_data_config()["package-data"]["phenotypic"]
    matched: set[str] = set()
    for pattern in patterns:
        for hit in glob.glob(
            os.path.join(PACKAGE_DIR, pattern), recursive=True
        ):
            if os.path.isfile(hit):
                matched.add(Path(hit).relative_to(PACKAGE_DIR).as_posix())
    return matched


def test_every_tracked_data_file_is_matched_by_a_package_data_glob() -> None:
    tracked = _tracked_data_files()
    assert KNOWN_DATA_FILE in tracked, f"file list is mis-wired: {tracked}"

    missing = sorted(set(tracked) - _files_matched_by_package_data())
    assert not missing, (
        f"{len(missing)} bundled data file(s) would be left out of the wheel; "
        f"add a [tool.setuptools.package-data] glob for them: {missing}"
    )


def test_no_exclude_package_data_rule_is_configured() -> None:
    """The emulation above ignores ``exclude-package-data``; keep it absent.

    If an exclusion is ever needed, teach
    :func:`_files_matched_by_package_data` to subtract it first, or this test
    would report files as shipped that setuptools drops.
    """
    assert "exclude-package-data" not in _package_data_config()
