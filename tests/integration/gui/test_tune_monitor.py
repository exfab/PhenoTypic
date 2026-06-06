"""Integration tests for the tune Monitor view (3s poll + figure slots).

Builds the loaded tune app over a parquet-journal-only run and confirms the
Monitor view carries the 3-second poll interval, the objective / importance
figure slots, the gap badge, the trials table, and the degrade note. Also
exercises the poll callback's pure read path against a finished trials.parquet
(no live study) and confirms it degrades to the journal without raising.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.tune import create_app
from phenotypic.gui.tune._run_root import TuneRunRoot
from phenotypic.tune._study_store import JournalStudyStore, Trial


def _journal_run(path: Path) -> TuneRunRoot:
    """Write a trials.parquet under ``path`` and discover it as a legacy root."""
    from phenotypic.tools_ import trials_parquet_path

    store = JournalStudyStore(
        trials=[
            Trial(number=0, params={"thresh": 0.1}, score=0.3, terms={}, n_images=3),
            Trial(number=1, params={"thresh": 0.2}, score=0.6, terms={}, n_images=3),
        ]
    )
    parquet = trials_parquet_path(path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(parquet)
    return TuneRunRoot.discover(path)


def test_monitor_view_has_poll_and_figure_slots(tmp_path: Path) -> None:
    root = _journal_run(tmp_path)
    app = create_app(root=root, url_prefix="/tune/")
    layout = str(app.layout)
    for component_id in (
        "tune-study-poll",
        "tune-objective-figure",
        "tune-importance-figure",
        "tune-gap-badge",
        "tune-trials-table",
        "tune-monitor-note",
    ):
        assert component_id in layout


def test_poll_interval_is_three_seconds(tmp_path: Path) -> None:
    root = _journal_run(tmp_path)
    app = create_app(root=root, url_prefix="/tune/")

    def _find_interval(node: object) -> object | None:
        children = getattr(node, "children", None)
        if getattr(node, "id", None) == "tune-study-poll":
            return node
        if isinstance(children, (list, tuple)):
            for child in children:
                found = _find_interval(child)
                if found is not None:
                    return found
        elif children is not None:
            return _find_interval(children)
        return None

    interval = _find_interval(app.layout)
    assert interval is not None
    assert interval.interval == 3000


def test_read_study_for_monitor_degrades_to_journal(tmp_path: Path) -> None:
    """With no live storage URL, the read falls back to the finished parquet."""
    from phenotypic.gui.tune._callbacks import read_study_for_monitor

    root = _journal_run(tmp_path)
    store, note = read_study_for_monitor(root)
    assert store is not None
    assert [t.score for t in store.trials] == [0.3, 0.6]
    # A parquet-only run yields no degrade banner (it never tried a live read).
    assert note == ""


@pytest.mark.parametrize("missing_path", ["/nonexistent/tune/run"])
def test_read_study_for_monitor_missing_parquet_is_safe(missing_path: str) -> None:
    """A run whose parquet doesn't exist yet returns no store, no raise."""
    from phenotypic.gui.tune._callbacks import read_study_for_monitor

    root = TuneRunRoot(
        path=Path(missing_path),
        trials_path=None,
        storage_url=None,
        study_name="tune",
        directions=None,
        images_dir=None,
        best_pipeline_path=Path(missing_path) / "best_pipeline.json",
    )
    store, note = read_study_for_monitor(root)
    assert store is None
