from __future__ import annotations

import sys
import types


def test_manual_heartbeat_records_initial_beat_without_optuna_extra():
    from phenotypic.tune._strategies._optuna import _start_heartbeat

    recorded: list[int] = []

    class _Storage:
        def get_heartbeat_interval(self):
            return 3600

        def record_heartbeat(self, trial_id):
            recorded.append(trial_id)

    study = types.SimpleNamespace(_storage=_Storage())
    trial = types.SimpleNamespace(_trial_id=17, number=3)

    handle = _start_heartbeat(study, trial)
    assert handle is not None
    try:
        assert recorded == [17]
    finally:
        handle.stop()


def test_manual_heartbeat_is_disabled_without_storage_interval():
    from phenotypic.tune._strategies._optuna import _start_heartbeat

    class _Storage:
        def get_heartbeat_interval(self):
            return None

        def record_heartbeat(self, trial_id):
            raise AssertionError("heartbeat should be disabled")

    study = types.SimpleNamespace(_storage=_Storage())
    trial = types.SimpleNamespace(_trial_id=17, number=3)

    assert _start_heartbeat(study, trial) is None


def test_fail_stale_trials_delegates_to_optuna_when_available(monkeypatch):
    from phenotypic.tune._strategies._optuna import _fail_stale_trials

    called = {}
    study = object()

    def _fake_fail_stale(arg):
        called["study"] = arg

    fake_optuna = types.SimpleNamespace(
        storages=types.SimpleNamespace(fail_stale_trials=_fake_fail_stale)
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    _fail_stale_trials(study)

    assert called["study"] is study
