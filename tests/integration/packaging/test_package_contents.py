"""Layer 1 — build-artifact packaging tests.

These build a wheel + sdist from the project and assert that the GUI
sub-app static assets (CSS/JS + vendored OpenSeadragon control icons) are
actually shipped. This is the regression guard for the ``pip install``
deployment bug where the ``phenotypic-gui`` entry point crashed on import
because ``gui/shell/_assets/shell.css`` — read at import time — was missing
from the wheel.

Two design choices matter:

* **Build from an isolated copy, not the repo root in place.** A stale
  ``src/phenotypic.egg-info/SOURCES.txt`` left by the editable dev install
  (``uv sync``) lists every tracked file and is reused by setuptools' sdist
  builder. Building in place therefore ships the assets *even when the
  ``package-data`` config is broken*, silently masking the very bug this
  test exists to catch. The helper copies the minimal build inputs into a
  tmp dir, explicitly excluding ``*.egg-info``/``build``/``dist``, so the
  artifact reflects ``pyproject.toml`` alone.
* **Derive the expected asset set from the source tree.** Walking
  ``src/phenotypic/gui`` for ``*.css``/``*.js``/``*.png`` means any future
  sub-app's assets are covered automatically — a newly added asset that
  isn't packaged fails this test without anyone editing it.

Marked ``slow`` (shells out to ``uv build``); excluded from the default
``-m 'not slow'`` PR run.
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUI_DIR = REPO_ROOT / "src" / "phenotypic" / "gui"
ASSET_SUFFIXES = (".css", ".js", ".png")

# Representative assets that MUST ship — one per GUI sub-app surface, plus a
# couple of pre-existing globbed assets so a regression in the original
# ``data``/``_assets`` patterns is caught here too.
REQUIRED_WHEEL_PATHS = [
    "phenotypic/gui/shell/_assets/shell.css",
    "phenotypic/gui/builder/assets/builder.css",
    "phenotypic/gui/builder/assets/builder.js",
    "phenotypic/gui/results_viewer/_assets/results_viewer.css",
    "phenotypic/gui/results_viewer/_assets/openseadragon/images/zoomin_rest.png",
    "phenotypic/gui/browse/_assets/browse.js",
    "phenotypic/gui/browse/_assets/openseadragon/images/zoomin_rest.png",
    "phenotypic/gui/run_console/_assets/run_console.css",
    "phenotypic/gui/analysis/_assets/analysis.css",
    "phenotypic/gui/tune/_assets/tune.css",
    # Pre-existing globbed assets (regression guard for the original config).
    "phenotypic/_assets/vendor/plotly.min.js",
    "phenotypic/data/early_colony.png",
]


def _expected_gui_asset_wheel_paths() -> list[str]:
    """Every CSS/JS/PNG under ``src/phenotypic/gui`` as its in-wheel path."""
    paths = []
    for path in GUI_DIR.rglob("*"):
        if path.is_file() and path.suffix in ASSET_SUFFIXES:
            rel = path.relative_to(REPO_ROOT / "src")
            paths.append(rel.as_posix())
    return sorted(paths)


def _build_dists(dest: Path) -> Path:
    """Build wheel + sdist from an egg-info-free copy of the project.

    Returns the directory containing the built artifacts.
    """
    src_copy = dest / "project"
    src_copy.mkdir(parents=True)
    # Minimal build inputs. Copying only these (rather than the whole repo)
    # keeps the build fast and sidesteps docs/test fixtures entirely.
    shutil.copy2(REPO_ROOT / "pyproject.toml", src_copy / "pyproject.toml")
    shutil.copy2(REPO_ROOT / "README.md", src_copy / "README.md")
    if (REPO_ROOT / "LICENSE").exists():
        shutil.copy2(REPO_ROOT / "LICENSE", src_copy / "LICENSE")
    shutil.copytree(
        REPO_ROOT / "src",
        src_copy / "src",
        ignore=shutil.ignore_patterns(
            "*.egg-info", "build", "dist", "__pycache__", "*.pyc"
        ),
    )
    out_dir = dest / "dist"
    subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=src_copy,
        check=True,
        capture_output=True,
        text=True,
    )
    return out_dir


@pytest.fixture(scope="module")
def built_dists(tmp_path_factory) -> dict[str, Path]:
    out_dir = _build_dists(tmp_path_factory.mktemp("pkgbuild"))
    wheels = list(out_dir.glob("*.whl"))
    sdists = list(out_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected 1 wheel, got {wheels}"
    assert len(sdists) == 1, f"expected 1 sdist, got {sdists}"
    return {"wheel": wheels[0], "sdist": sdists[0]}


@pytest.mark.slow
def test_wheel_contains_every_gui_asset(built_dists):
    """Every CSS/JS/PNG under ``gui/`` must be present in the wheel."""
    with zipfile.ZipFile(built_dists["wheel"]) as zf:
        names = set(zf.namelist())
    expected = _expected_gui_asset_wheel_paths()
    assert expected, "no GUI assets found on disk — test wiring is wrong"
    missing = [p for p in expected if p not in names]
    assert not missing, (
        f"{len(missing)} GUI asset(s) missing from wheel "
        f"(package-data globs do not cover them): {missing[:10]}"
    )


@pytest.mark.slow
def test_wheel_contains_required_paths(built_dists):
    """Representative per-sub-app + pre-existing assets ship in the wheel."""
    with zipfile.ZipFile(built_dists["wheel"]) as zf:
        names = set(zf.namelist())
    missing = [p for p in REQUIRED_WHEEL_PATHS if p not in names]
    assert not missing, f"missing from wheel: {missing}"


@pytest.mark.slow
def test_sdist_contains_required_paths(built_dists):
    """The sdist carries the GUI assets too (wheels build from the sdist)."""
    with tarfile.open(built_dists["sdist"]) as tf:
        # sdist entries are prefixed by ``<name>-<version>/src/`` — flatten
        # to the ``phenotypic/...`` form used by the wheel for comparison.
        flat = set()
        for name in tf.getnames():
            parts = name.split("/")
            if "src" in parts:
                idx = parts.index("src")
                flat.add("/".join(parts[idx + 1 :]))
    missing = [p for p in REQUIRED_WHEEL_PATHS if p not in flat]
    assert not missing, f"missing from sdist: {missing}"
