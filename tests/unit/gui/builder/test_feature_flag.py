"""Feature-flag invariants for the Pipeline Builder DAG redesign.

The flag (spec §5.7a) is read **once at module import** of
``phenotypic.gui.builder._state``.  Subsequent mutations of
``os.environ["PHENOTYPIC_GUI_DAG"]`` are deliberately ignored — the
process must restart to flip the flag.

Reload-based assertions run in **subprocess isolation** so reloading
``_state`` cannot corrupt the parent test process's class identities
(``isinstance`` checks against pre-reload class references would
otherwise spuriously fail in sibling tests that imported the DAG
dataclasses before the reload).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import pytest

from phenotypic.gui.builder import _state as state_module


def _run_flag_subprocess(env_value: Optional[str]) -> bool:
    """Spawn a fresh interpreter, set the env var, import ``_state``, and
    print the captured flag value.

    Returns the captured ``PHENOTYPIC_GUI_DAG`` value as a Python ``bool``.
    Subprocess isolation guarantees the parent process's class identities
    survive intact (no ``importlib.reload`` here).
    """

    env = {**os.environ}
    if env_value is None:
        env.pop("PHENOTYPIC_GUI_DAG", None)
    else:
        env["PHENOTYPIC_GUI_DAG"] = env_value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from phenotypic.gui.builder._state import PHENOTYPIC_GUI_DAG\n"
            "print(PHENOTYPIC_GUI_DAG)\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout.strip()
    if out == "True":
        return True
    if out == "False":
        return False
    raise AssertionError(f"unexpected subprocess stdout: {out!r}")


def test_flag_default_off() -> None:
    """Without the env var set to ``"1"``, the flag is ``False``."""

    expected = os.environ.get("PHENOTYPIC_GUI_DAG", "0") == "1"
    assert state_module.PHENOTYPIC_GUI_DAG is expected


def test_flag_read_once_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating the env var after import does NOT update the flag value."""

    captured = state_module.PHENOTYPIC_GUI_DAG
    new_env_value = "0" if captured else "1"
    monkeypatch.setenv("PHENOTYPIC_GUI_DAG", new_env_value)
    assert state_module.PHENOTYPIC_GUI_DAG is captured


def test_flag_subprocess_picks_up_new_env_value() -> None:
    """A fresh interpreter import with ``PHENOTYPIC_GUI_DAG="1"`` flips to True."""

    assert _run_flag_subprocess("1") is True


def test_flag_subprocess_with_env_unset_is_false() -> None:
    """A fresh interpreter import without the env var yields False."""

    assert _run_flag_subprocess(None) is False


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
def test_flag_truthy_only_for_literal_1(env_value: str, expected: bool) -> None:
    """Only the literal ``"1"`` value enables the flag — not other truthies.

    Misspelled values ("yes", "True", "1 ", etc.) must NOT silently enable
    the redesign.  The flag is a hard on/off switch, not a fuzzy truthy gate.
    """

    assert _run_flag_subprocess(env_value) is expected
