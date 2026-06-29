"""Unit tests for the results-viewer header mode badge.

:func:`phenotypic.gui.results_viewer._layout.build_mode_badge` reads only
``output_root.has_results`` and renders a pill reading **Full run** (per-image
``results/`` present) or **Standalone bundle** (deliverables-only). The badge
distinguishes a portable bundle — where the per-image pixel-layer toggle is
unavailable — from a full ``python -m phenotypic`` run.
"""
from __future__ import annotations

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._layout import build_mode_badge


class _Full:
    """Minimal stand-in for a full-run output root."""

    has_results = True


class _Bundle:
    """Minimal stand-in for a standalone deliverables bundle."""

    has_results = False


def test_mode_badge_text_by_capability() -> None:
    """The badge text flips on ``has_results`` (full vs standalone)."""
    full = str(build_mode_badge(_Full()))
    bundle = str(build_mode_badge(_Bundle()))
    assert "Full run" in full
    assert "Standalone" not in full
    assert "Standalone" in bundle
    assert "Full run" not in bundle


def test_mode_badge_carries_stable_id() -> None:
    """The badge id is the shared header constant so callbacks/tests can target it."""
    assert build_mode_badge(_Full()).id == ids.HEADER_MODE_BADGE_ID
    assert build_mode_badge(_Bundle()).id == ids.HEADER_MODE_BADGE_ID


def test_mode_badge_reads_only_has_results() -> None:
    """A bare duck-typed object exposing only ``has_results`` is sufficient."""

    class _OnlyFlag:
        has_results = False

    badge = build_mode_badge(_OnlyFlag())
    assert "Standalone" in str(badge)
