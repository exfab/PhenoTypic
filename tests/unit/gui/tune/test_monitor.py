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
