import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from phenotypic.gui.tune._monitor import (
    cancel_prompt,
    live_view_kind,
    run_switcher_items,
)


def _rec(run_id, mode, status):
    return SimpleNamespace(
        run_id=run_id,
        generation=uuid4(),
        mode=mode,
        status=status,
    )


def test_switcher_marks_active_and_killable():
    recs = [
        _rec("a", "local", "running"),
        _rec("b", "slurm", "running"),
        _rec("c", "local", "complete"),
    ]
    items = run_switcher_items(
        recs,
        active_receipt={
            "run_id": "b",
            "generation": str(recs[1].generation),
        },
    )
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


def _run_root(tmp_path, storage_url, study_name="tune_cost_v1"):
    from phenotypic.gui.tune._run_root import TuneRunRoot

    return TuneRunRoot(
        path=tmp_path, trials_path=None, storage_url=storage_url,
        study_name=study_name, directions=None, images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )


def test_a_server_backed_study_keeps_the_network_note(tmp_path):
    """A Postgres URL that failed to open is the one case ~/.pgpass helps with."""
    from phenotypic.gui.tune._callbacks import (
        _NOTE_LIVE_UNREACHABLE, _monitor_degrade_note,
    )

    current = _run_root(tmp_path, "postgresql+psycopg://user@host:5432/db")
    assert (
        _monitor_degrade_note(current, RuntimeError("connection refused"))
        == _NOTE_LIVE_UNREACHABLE
    )


@pytest.mark.parametrize(
    "storage_url",
    ["journal:///runs/out/.pht-tune-cache/journal.log", "sqlite:///runs/out/study.db"],
    ids=["journal", "sqlite"],
)
def test_a_file_backed_study_never_blames_the_network(tmp_path, storage_url):
    """The ``--slurm`` default has no server, so the note must not name one.

    Telling a user to check ``~/.pgpass`` for a local ``journal.log`` sends them
    after a problem that does not exist in their configuration.
    """
    from phenotypic.gui.tune._callbacks import (
        _NOTE_LIVE_UNREADABLE, _monitor_degrade_note,
    )

    note = _monitor_degrade_note(_run_root(tmp_path, storage_url), OSError("boom"))
    assert note == _NOTE_LIVE_UNREADABLE
    assert "pgpass" not in note
    assert "network" not in note


@pytest.mark.parametrize(
    "storage_url",
    ["journal:///runs/out/journal.log", "sqlite:///runs/out/study.db"],
    ids=["journal", "sqlite"],
)
def test_an_overrun_file_read_says_so_instead_of_unreachable(tmp_path, storage_url):
    """A local file that was read and merely ran long is not an unreachable study."""
    from concurrent.futures import TimeoutError as FutureTimeout

    from phenotypic.gui.tune._callbacks import (
        _NOTE_LIVE_TIMEOUT, _monitor_degrade_note,
    )

    note = _monitor_degrade_note(_run_root(tmp_path, storage_url), FutureTimeout())
    assert note == _NOTE_LIVE_TIMEOUT
    assert "pgpass" not in note


def test_a_postgres_timeout_still_points_at_the_connection(tmp_path):
    """An overrun against a server is (almost always) libpq still connecting.

    So the credential/network hint stays where it is actually actionable — the
    split is about not showing it for a local file, not about retiring it.
    """
    from concurrent.futures import TimeoutError as FutureTimeout

    from phenotypic.gui.tune._callbacks import (
        _NOTE_LIVE_UNREACHABLE, _monitor_degrade_note,
    )

    root = _run_root(tmp_path, "postgresql+psycopg://user@host:5432/db")
    assert _monitor_degrade_note(root, FutureTimeout()) == _NOTE_LIVE_UNREACHABLE


def test_the_connect_bound_fits_inside_the_read_bound():
    """NB-6: the connect nests inside the read, so it cannot consume all of it.

    Both were 3.0 s, which meant a slow Postgres connect could spend the entire
    read budget and leave nothing for the reads it precedes.
    """
    from phenotypic.gui.tune._callbacks import (
        _LIVE_CONNECT_TIMEOUT_S, _LIVE_READ_TIMEOUT_S,
    )

    assert _LIVE_CONNECT_TIMEOUT_S < _LIVE_READ_TIMEOUT_S
    # libpq clamps anything under 2 s up to 2 s, so a smaller value would
    # document a bound the driver does not honor.
    assert _LIVE_CONNECT_TIMEOUT_S >= 2.0


# ---------------------------------------------------------------------------
# BLOCK-2 — the importance cache keeps fANOVA out of the poll's read budget
# ---------------------------------------------------------------------------


def _cache():
    from phenotypic.gui.tune._callbacks import _ImportanceCache

    return _ImportanceCache()


def test_a_finished_refresh_is_served_and_not_recomputed():
    """One fANOVA per trial count, however many ticks read it."""
    cache = _cache()
    calls = []

    def _compute():
        calls.append(1)
        return {"thresh": 1.0}

    key = ("journal:///x/journal.log", "tune_cost_v1")
    assert cache.read(key, 3, _compute) is None  # first tick: nothing cached yet
    assert cache.wait_for_refresh(10.0)
    assert cache.read(key, 3, _compute) == {"thresh": 1.0}
    assert cache.read(key, 3, _compute) == {"thresh": 1.0}
    assert calls == [1]


def test_only_one_importance_refresh_is_ever_in_flight():
    """The backlog that a shared single worker accumulated cannot form.

    At 400 trials fANOVA drains slower than the 3 s poll enqueues, so an
    unguarded schedule grows without bound — every abandoned job still running a
    full fANOVA on the GUI host.
    """
    cache = _cache()
    calls = []
    started = threading.Event()
    gate = threading.Event()

    def _slow():
        calls.append(1)
        started.set()
        gate.wait(timeout=30.0)
        return {"thresh": 1.0}

    key = ("journal:///x/journal.log", "tune_cost_v1")
    try:
        assert cache.read(key, 0, _slow) is None
        assert started.wait(10.0), "the refresh never started"
        for n_trials in range(1, 6):  # five more ticks, five new trial counts
            assert cache.read(key, n_trials, _slow) is None
        assert calls == [1], "a second fANOVA was queued behind the first"
    finally:
        gate.set()
        cache.wait_for_refresh(10.0)


def test_a_failing_refresh_degrades_to_no_figure_not_to_no_read():
    """An exception inside the model is absorbed; the cache stays usable."""
    cache = _cache()

    def _broken():
        raise RuntimeError("fANOVA is degenerate")

    key = ("journal:///x/journal.log", "tune_cost_v1")
    assert cache.read(key, 2, _broken) is None
    assert cache.wait_for_refresh(10.0)
    assert cache.read(key, 2, _broken) is None


def test_switching_runs_discards_the_other_runs_importances():
    """Two runs must never see each other's model."""
    cache = _cache()
    first = ("journal:///a/journal.log", "tune_cost_v1")
    second = ("journal:///b/journal.log", "tune_cost_v1")

    assert cache.read(first, 2, lambda: {"thresh": 1.0}) is None
    assert cache.wait_for_refresh(10.0)
    assert cache.read(first, 2, lambda: {"thresh": 1.0}) == {"thresh": 1.0}

    assert cache.read(second, 2, lambda: {"radius": 1.0}) is None
    cache.wait_for_refresh(10.0)
    assert cache.read(second, 2, lambda: {"radius": 1.0}) == {"radius": 1.0}
