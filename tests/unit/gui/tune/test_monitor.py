from types import SimpleNamespace

import pytest

from phenotypic.gui.tune._monitor import (
    cancel_prompt,
    live_view_kind,
    run_switcher_items,
)


def _rec(run_id, mode, status):
    return SimpleNamespace(run_id=run_id, mode=mode, status=status)


def test_switcher_marks_active_and_killable():
    recs = [
        _rec("a", "local", "running"),
        _rec("b", "slurm", "running"),
        _rec("c", "local", "complete"),
    ]
    items = run_switcher_items(recs, active_id="b")
    by_id = {item.run_id: item for item in items}
    assert by_id["b"].active is True
    assert by_id["a"].active is False
    assert by_id["a"].killable is True
    assert by_id["b"].killable is False
    assert by_id["c"].killable is False


def test_live_view_kind_selects_local_vs_slurm_and_degrades():
    assert live_view_kind("local", store_reachable=True) == "local-log"
    assert live_view_kind("slurm", store_reachable=True) == "slurm-fleet"
    assert live_view_kind("slurm", store_reachable=False) == "slurm-detached"


def test_cancel_prompt_is_local_only():
    local = cancel_prompt("yeast_qc_tpe", "local")
    assert "SIGTERM" in local and "resumed" in local

    with pytest.raises(ValueError, match="SLURM"):
        cancel_prompt("slurm_run", "slurm")


def test_legacy_study_degrade_note_is_friendly(tmp_path):
    """A pre-cutover run (study_name='tune') degrades with a re-run message,
    not the generic 'couldn't reach the live study' note."""
    from phenotypic.gui.tune._callbacks import _monitor_degrade_note
    from phenotypic.gui.tune._run_root import TuneRunRoot

    legacy = TuneRunRoot(
        path=tmp_path, trials_path=None, storage_url="sqlite:///x.db",
        study_name="tune", directions=None, images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )
    note = _monitor_degrade_note(legacy, RuntimeError("study not found"))
    assert "re-run" in note.lower() or "pre-cutover" in note.lower()


def test_current_study_degrade_note_is_generic(tmp_path):
    """A current-convention run keeps the generic unreachable note."""
    from phenotypic.gui.tune._callbacks import (
        _NOTE_LIVE_UNREACHABLE, _monitor_degrade_note,
    )
    from phenotypic.gui.tune._run_root import TuneRunRoot

    current = TuneRunRoot(
        path=tmp_path, trials_path=None, storage_url="sqlite:///x.db",
        study_name="tune_cost_v1", directions=None, images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )
    assert _monitor_degrade_note(current, RuntimeError("timeout")) == _NOTE_LIVE_UNREACHABLE
