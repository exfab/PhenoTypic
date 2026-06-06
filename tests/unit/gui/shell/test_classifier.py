"""Unit tests for ``phenotypic.gui.shell._classifier.classify``.

Covers each branch of :class:`Capabilities`:

    * empty directory               → all-False
    * images-only directory         → ``is_image_dir`` + ``image_count``
    * CLI-output directory          → ``is_cli_output``
    * mixed (images + dashboard)    → both flags set
    * permission-denied directory   → ``bad_perms``
    * pipeline JSON file            → ``has_pipeline_json``

Also verifies the LRU cache invalidates on mtime change (the cache key is
``(path, mtime_ns)``).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from phenotypic.gui._config import DELIVERABLES_DIRNAME
from phenotypic.gui.shell._classifier import (
    Capabilities,
    classify,
    invalidate_cache,
)


def _seed_deliverables_cli_output(root: Path, *, dashboard: bool = False) -> None:
    """Write the NESTED CLI-output layout under ``root``.

    ``master_measurements.parquet`` (and optionally ``dashboard.html``) go
    INTO ``root/deliverables/``; ``results/`` stays at the root. This is the
    only layout the classifier recognizes after the hard cutover.
    """
    deliverables = root / DELIVERABLES_DIRNAME
    deliverables.mkdir(parents=True, exist_ok=True)
    (deliverables / "master_measurements.parquet").write_bytes(b"")
    if dashboard:
        (deliverables / "dashboard.html").write_text("<html/>")
    (root / "results").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def _flush_classifier_cache() -> None:
    """Each test starts with an empty cache."""
    invalidate_cache()


# ---------------------------------------------------------------------------
# Directory classification
# ---------------------------------------------------------------------------

def test_empty_directory(tmp_path: Path) -> None:
    caps = classify(tmp_path)
    assert caps == Capabilities(
        is_image_dir=False,
        has_pipeline_json=False,
        is_cli_output=False,
        has_dashboard=False,
        is_process_only_output=False,
        is_tune_output=False,
        image_count=None,
        bad_perms=False,
    )


def test_images_only(tmp_path: Path) -> None:
    (tmp_path / "plate1.tif").write_bytes(b"")
    (tmp_path / "plate2.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignored")
    caps = classify(tmp_path)
    assert caps.is_image_dir is True
    assert caps.image_count == 2
    assert caps.is_cli_output is False
    assert caps.has_pipeline_json is False
    assert caps.bad_perms is False


def test_image_extension_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "P1.TIF").write_bytes(b"")
    (tmp_path / "P2.JpG").write_bytes(b"")
    caps = classify(tmp_path)
    assert caps.is_image_dir is True
    assert caps.image_count == 2


def test_cli_output_directory(tmp_path: Path) -> None:
    """NESTED layout: master parquet under deliverables/, results/ at root."""
    _seed_deliverables_cli_output(tmp_path, dashboard=True)
    caps = classify(tmp_path)
    assert caps.is_cli_output is True
    assert caps.has_dashboard is True
    assert caps.is_image_dir is False


def test_cli_output_requires_both_markers(tmp_path: Path) -> None:
    """deliverables/master without results/, or vice versa, is not a CLI output."""
    deliverables = tmp_path / DELIVERABLES_DIRNAME
    deliverables.mkdir()
    (deliverables / "master_measurements.parquet").write_bytes(b"")
    caps = classify(tmp_path)
    assert caps.is_cli_output is False

    invalidate_cache()
    (deliverables / "master_measurements.parquet").unlink()
    (tmp_path / "results").mkdir()
    caps = classify(tmp_path)
    assert caps.is_cli_output is False


def test_legacy_root_layout_not_recognized(tmp_path: Path) -> None:
    """Hard cutover: master + dashboard at the ROOT (legacy) is NOT a CLI output.

    After deliverables/ became the only recognized location, a directory
    that still has ``master_measurements.parquet`` + ``dashboard.html`` at
    the output root (plus ``results/``) must classify as neither a CLI
    output nor as having a dashboard.
    """
    (tmp_path / "master_measurements.parquet").write_bytes(b"")
    (tmp_path / "dashboard.html").write_text("<html/>")
    (tmp_path / "results").mkdir()
    caps = classify(tmp_path)
    assert caps.is_cli_output is False
    assert caps.has_dashboard is False


def test_dashboard_flagged(tmp_path: Path) -> None:
    """``has_dashboard`` reads ``deliverables/dashboard.html``."""
    deliverables = tmp_path / DELIVERABLES_DIRNAME
    deliverables.mkdir()
    (deliverables / "dashboard.html").write_text("<html/>")
    caps = classify(tmp_path)
    assert caps.has_dashboard is True


def test_mixed_images_and_dashboard(tmp_path: Path) -> None:
    (tmp_path / "plate.tif").write_bytes(b"")
    deliverables = tmp_path / DELIVERABLES_DIRNAME
    deliverables.mkdir()
    (deliverables / "dashboard.html").write_text("<html/>")
    caps = classify(tmp_path)
    assert caps.is_image_dir is True
    assert caps.image_count == 1
    assert caps.has_dashboard is True


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

def test_pipeline_json_detected(tmp_path: Path) -> None:
    p = tmp_path / "my_pipeline.json"
    p.write_text(json.dumps(
        {"name": "demo", "operations": [{"type": "GrayscaleEnhancer"}]}
    ))
    caps = classify(p)
    assert caps.has_pipeline_json is True
    assert caps.is_image_dir is False


def test_arbitrary_json_not_a_pipeline(tmp_path: Path) -> None:
    p = tmp_path / "not_a_pipeline.json"
    p.write_text(json.dumps({"foo": "bar", "baz": [1, 2, 3]}))
    caps = classify(p)
    assert caps.has_pipeline_json is False


def test_image_file_alone(tmp_path: Path) -> None:
    """A single image file (not a directory) is not flagged ``is_image_dir``.

    The classifier semantics: ``is_image_dir`` is a *directory* property.
    """
    p = tmp_path / "plate.tif"
    p.write_bytes(b"")
    caps = classify(p)
    assert caps.is_image_dir is False
    assert caps.has_pipeline_json is False


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_missing_path_returns_empty_no_raise(tmp_path: Path) -> None:
    caps = classify(tmp_path / "does_not_exist")
    assert caps.bad_perms is False
    assert caps.is_image_dir is False


def test_broken_symlink_returns_empty(tmp_path: Path) -> None:
    target = tmp_path / "vanished"
    link = tmp_path / "link"
    target.write_text("x")
    link.symlink_to(target)
    target.unlink()
    caps = classify(link)
    assert caps.bad_perms is False
    assert caps.is_image_dir is False


@pytest.mark.skipif(
    sys.platform == "win32" or os.getuid() == 0,
    reason="chmod-based permission test is POSIX-specific and meaningless as root",
)
def test_permission_denied_directory(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "plate.tif").write_bytes(b"")
    os.chmod(locked, 0o000)
    try:
        caps = classify(locked)
        assert caps.bad_perms is True
        assert caps.is_image_dir is False
    finally:
        os.chmod(locked, 0o755)


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

def test_cache_invalidates_on_mtime_change(tmp_path: Path) -> None:
    """Adding a new image bumps the directory's mtime → cache miss."""
    caps_empty = classify(tmp_path)
    assert caps_empty.is_image_dir is False

    # Force an mtime bump. ``os.utime`` is the most portable way to make
    # the test deterministic.
    (tmp_path / "plate.tif").write_bytes(b"")
    new_mtime = tmp_path.stat().st_mtime_ns + 1_000_000_000
    os.utime(tmp_path, ns=(new_mtime, new_mtime))

    caps_filled = classify(tmp_path)
    assert caps_filled.is_image_dir is True
    assert caps_filled.image_count == 1


def test_invalidate_cache_clears_state(tmp_path: Path) -> None:
    classify(tmp_path)
    invalidate_cache()
    # Second call after invalidation still works (no exception).
    classify(tmp_path)


def test_cache_hits_on_repeat_call(tmp_path: Path) -> None:
    """Repeat ``classify`` with unchanged mtime is a cache hit.

    Without the LRU layer, the sidebar would re-stat/re-iterdir on every
    poll; the cache is not optional. We assert the hit count via the
    underlying ``_classify_cached.cache_info()`` so a regression that
    bypasses the cache layer (e.g. a future "always recompute") is caught
    at unit-test time, not under load.
    """
    from phenotypic.gui.shell._classifier import _classify_cached

    info_before = _classify_cached.cache_info()
    classify(tmp_path)
    info_after_first = _classify_cached.cache_info()
    classify(tmp_path)  # same mtime → must hit
    info_after_second = _classify_cached.cache_info()

    new_misses = info_after_first.misses - info_before.misses
    new_hits = info_after_second.hits - info_after_first.hits
    assert new_misses == 1
    assert new_hits == 1
