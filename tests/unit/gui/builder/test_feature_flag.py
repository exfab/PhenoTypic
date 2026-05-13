"""Feature-flag invariants for the Pipeline Builder DAG redesign.

The flag (spec §5.7a) is read **once at module import** of
``phenotypic.gui.builder._state``.  Subsequent mutations of
``os.environ["PHENOTYPIC_GUI_DAG"]`` are deliberately ignored — the
process must restart to flip the flag.

These tests sit in the unit-test lane (no Dash, no images) and import
``phenotypic.gui.builder._state`` to assert the value the module
captured.  Reload behaviour is exercised separately via
``importlib.reload``.
"""

from __future__ import annotations

import importlib
import os
from typing import Iterator

import pytest

from phenotypic.gui.builder import _state as state_module


@pytest.fixture
def restore_state_module() -> Iterator[None]:
    """Reload the ``_state`` module after the test so others see the default.

    Several tests in this file flip the env var and reload the module
    to probe the import-time read.  This fixture guarantees we leave
    the module in a clean state by reloading it once more with the
    *current* (test-runner) value of ``PHENOTYPIC_GUI_DAG`` at the
    end.
    """

    yield
    importlib.reload(state_module)


def test_flag_default_off() -> None:
    """Without the env var set to ``"1"``, the flag is ``False``."""

    # The test harness doesn't set PHENOTYPIC_GUI_DAG by default, so the
    # captured flag should be False. (We don't *force* the var to be
    # unset here because the test process may legitimately have it
    # set; instead, the assertion mirrors the import-time read.)
    expected = os.environ.get("PHENOTYPIC_GUI_DAG", "0") == "1"
    assert state_module.PHENOTYPIC_GUI_DAG is expected


def test_flag_read_once_at_import(
    monkeypatch: pytest.MonkeyPatch,
    restore_state_module: None,
) -> None:
    """Mutating the env var after import does NOT update the flag value.

    ``importlib.reload`` is the only path that re-runs the import-time
    read; this is documented in :mod:`phenotypic.gui.builder._state`.
    """

    # Snapshot the value the module captured at import time.
    captured = state_module.PHENOTYPIC_GUI_DAG

    # Flip the env var to the opposite literal value.
    new_env_value = "0" if captured else "1"
    monkeypatch.setenv("PHENOTYPIC_GUI_DAG", new_env_value)

    # Without a reload, the captured flag must NOT have changed.
    assert state_module.PHENOTYPIC_GUI_DAG is captured


def test_flag_reload_picks_up_new_env_value(
    monkeypatch: pytest.MonkeyPatch,
    restore_state_module: None,
) -> None:
    """``importlib.reload`` with the env var set to ``"1"`` flips to True."""

    monkeypatch.setenv("PHENOTYPIC_GUI_DAG", "1")
    reloaded = importlib.reload(state_module)
    assert reloaded.PHENOTYPIC_GUI_DAG is True


def test_flag_reload_with_env_unset_is_false(
    monkeypatch: pytest.MonkeyPatch,
    restore_state_module: None,
) -> None:
    """``importlib.reload`` with the env var deleted yields ``False``."""

    monkeypatch.delenv("PHENOTYPIC_GUI_DAG", raising=False)
    reloaded = importlib.reload(state_module)
    assert reloaded.PHENOTYPIC_GUI_DAG is False


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("1", True),
        ("0", False),
        ("true", False),
        ("True", False),
        ("yes", False),
        ("on", False),
        ("", False),
        ("01", False),  # only the literal "1" string counts.
    ],
)
def test_flag_truthy_only_for_literal_1(
    monkeypatch: pytest.MonkeyPatch,
    restore_state_module: None,
    env_value: str,
    expected: bool,
) -> None:
    """Only the literal ``"1"`` value enables the flag — not other truthies.

    This is intentional: a misspelled value ("yes", "True", "1 ", etc.)
    should *not* silently enable the redesign.  The flag is a hard
    on/off switch, not a fuzzy truthy gate.
    """

    monkeypatch.setenv("PHENOTYPIC_GUI_DAG", env_value)
    reloaded = importlib.reload(state_module)
    assert reloaded.PHENOTYPIC_GUI_DAG is expected
