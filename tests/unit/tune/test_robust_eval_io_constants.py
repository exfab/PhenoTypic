"""4.5p1 A4 — robust-eval split + generalization artifact paths.

The held-out split assignment is a **machine-state sidecar** (it must survive a
fresh-master rewrite and gate resume), so it lives at the output-dir ROOT under
``splits/`` — a sibling to ``trials.parquet`` / ``study.db``. The generalization
report is a **user-facing deliverable**, so it lands under ``deliverables/``.
"""
from __future__ import annotations

from pathlib import Path

from phenotypic.tools_ import _io_constants as io


def test_splits_dir_at_root():
    out = Path("/tmp/run")
    assert io.splits_dir(out) == out / "splits"


def test_split_assignment_path():
    out = Path("/tmp/run")
    assert io.split_assignment_path(out) == out / "splits" / "split.json"


def test_generalization_path_in_deliverables():
    out = Path("/tmp/run")
    assert io.generalization_path(out) == io.deliverables_dir(out) / "generalization.json"
