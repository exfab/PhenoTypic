"""Tests for the Curate difference toggle + write-winner path (Task B5).

* The pure :func:`curate_mode` helper resolves the Side-by-side ↔ Difference
  toggle to a mode string.
* :func:`write_winner` builds ``build_pipeline(base, winner.params)`` and writes
  it atomically (temp file + ``os.replace``) to the run's
  ``deliverables/best_pipeline.json`` for both single- and multi-objective runs;
  the result round-trips via ``ImagePipeline.from_json``.
* A ``PermissionError`` on the atomic replace (HPCC read-only output dir) is
  re-raised by the helper so the callback can surface it in a toast.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.gui.tune._run_root import TuneRunRoot
from phenotypic.tune._study_store import Trial


def _root(tmp_path: Path) -> TuneRunRoot:
    from phenotypic.sdk_ import best_pipeline_path

    return TuneRunRoot(
        path=tmp_path,
        trials_path=None,
        storage_url=None,
        study_name="tune",
        directions=None,
        images_dir=None,
        best_pipeline_path=best_pipeline_path(tmp_path),
    )


def _base() -> ImagePipeline:
    return ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()])


# ---------------------------------------------------------------------------
# Difference toggle helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("trigger", "expected"),
    [("side", "side"), ("difference", "difference"), (None, "side"), ("bogus", "side")],
)
def test_curate_mode_resolves(trigger: str | None, expected: str) -> None:
    from phenotypic.gui.tune._callbacks import curate_mode

    assert curate_mode(trigger) == expected


# ---------------------------------------------------------------------------
# write_winner — atomic best_pipeline.json
# ---------------------------------------------------------------------------

def test_write_winner_writes_round_trippable_pipeline(tmp_path: Path) -> None:
    from phenotypic.gui.tune._winner import write_winner

    root = _root(tmp_path)
    base = _base()
    # The winner overlays sigma=3.0 onto the base's first op.
    winner = Trial(number=2, params={"0.sigma": 3.0}, score=0.9, terms={}, n_images=3)

    written = write_winner(root, base, winner)
    assert written == root.best_pipeline_path
    assert written.exists()

    restored = ImagePipeline.from_json(written.read_text())
    ops = list(restored.get_ops().values())
    # The override landed: the first op's sigma is the winner's 3.0, not 1.0.
    assert ops[0].sigma == 3.0


def test_write_winner_overwrites_atomically(tmp_path: Path) -> None:
    """A second write replaces the prior winner (atomic os.replace, no append)."""
    from phenotypic.gui.tune._winner import write_winner

    root = _root(tmp_path)
    base = _base()
    write_winner(root, base, Trial(number=0, params={"0.sigma": 2.0}, score=0.5, terms={}, n_images=1))
    written = write_winner(root, base, Trial(number=1, params={"0.sigma": 4.0}, score=0.7, terms={}, n_images=1))

    restored = ImagePipeline.from_json(written.read_text())
    ops = list(restored.get_ops().values())
    assert ops[0].sigma == 4.0


def test_write_winner_reraises_permission_error(tmp_path: Path, monkeypatch) -> None:
    """A read-only output dir (PermissionError on replace) is re-raised."""
    import os

    from phenotypic.gui.tune._winner import write_winner

    root = _root(tmp_path)
    base = _base()
    winner = Trial(number=0, params={"0.sigma": 2.0}, score=0.5, terms={}, n_images=1)

    def _boom(src: object, dst: object) -> None:
        raise PermissionError("read-only output dir")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(PermissionError):
        write_winner(root, base, winner)
