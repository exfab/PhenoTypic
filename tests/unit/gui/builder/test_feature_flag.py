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

To keep the suite fast, the eight parametrised truthiness cases are
collected into a single batched subprocess call (one interpreter spawn,
one ``_state`` import, one stdout block) instead of ten separate
``subprocess.run`` invocations. The pre-batch shape took ~35 s; the
batched shape takes <5 s because Python's interpreter + ``phenotypic``
import cost (~3 s each) is paid once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import pytest

from phenotypic.gui.builder import _state as state_module


# Eight parametrised cases for the truthiness gate (see
# ``test_flag_truthy_only_for_literal_1`` below) plus the two simpler
# cases that previously had their own subprocess each.
_BATCH_CASES: List[Tuple[Optional[str], bool]] = [
    ("1", True),
    ("0", False),
    ("true", False),
    ("True", False),
    ("yes", False),
    ("on", False),
    ("", False),
    ("01", False),  # only the literal "1" string counts.
    (None, False),  # env var unset → False.
]


# Script run inside the batched subprocess. Reads a JSON list of env
# values on stdin, imports ``_state`` once per env value via
# ``subprocess.run(..., env=...)`` is not possible (env locked at
# spawn), so we use ``importlib.reload`` instead — but ONLY in the
# subprocess. The parent test process's class identities are never
# touched.
_BATCH_SCRIPT = (
    "import json, os, sys, importlib\n"
    "cases = json.load(sys.stdin)\n"
    "from phenotypic.gui.builder import _state\n"
    "out = []\n"
    "for env_value in cases:\n"
    "    if env_value is None:\n"
    "        os.environ.pop('PHENOTYPIC_GUI_DAG', None)\n"
    "    else:\n"
    "        os.environ['PHENOTYPIC_GUI_DAG'] = env_value\n"
    "    importlib.reload(_state)\n"
    "    out.append(_state.PHENOTYPIC_GUI_DAG)\n"
    "json.dump(out, sys.stdout)\n"
)


@pytest.fixture(scope="module")
def batched_flag_results() -> Dict[Optional[str], bool]:
    """Run all subprocess cases in ONE interpreter spawn; cache the result.

    The batched script reloads ``_state`` once per env value (cheap;
    just re-runs the module body) so the expensive Python interpreter
    + ``phenotypic`` import cost is paid once. Result is cached at
    module scope so every per-case test reads from the same dict.

    Returns:
        Dict mapping each env value (``None`` for "unset") to the
        captured ``PHENOTYPIC_GUI_DAG`` flag value.
    """

    cases = [env_value for env_value, _ in _BATCH_CASES]
    proc = subprocess.run(
        [sys.executable, "-c", _BATCH_SCRIPT],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    flags = json.loads(proc.stdout)
    assert len(flags) == len(cases), (
        f"batched subprocess returned {len(flags)} results for "
        f"{len(cases)} cases; stderr={proc.stderr!r}"
    )
    return dict(zip(cases, flags))


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


def test_flag_subprocess_picks_up_new_env_value(
    batched_flag_results: Dict[Optional[str], bool],
) -> None:
    """A fresh interpreter import with ``PHENOTYPIC_GUI_DAG="1"`` flips to True."""

    assert batched_flag_results["1"] is True


def test_flag_subprocess_with_env_unset_is_false(
    batched_flag_results: Dict[Optional[str], bool],
) -> None:
    """A fresh interpreter import without the env var yields False."""

    assert batched_flag_results[None] is False


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
    env_value: str,
    expected: bool,
    batched_flag_results: Dict[Optional[str], bool],
) -> None:
    """Only the literal ``"1"`` value enables the flag — not other truthies.

    Misspelled values ("yes", "True", "1 ", etc.) must NOT silently enable
    the redesign.  The flag is a hard on/off switch, not a fuzzy truthy gate.
    """

    assert batched_flag_results[env_value] is expected
