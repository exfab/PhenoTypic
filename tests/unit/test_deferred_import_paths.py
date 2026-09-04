"""Run the deferred-import exercise script under pytest, so CI executes it.

The script at ``docs/superpowers/plans/2026-09-02-import-laziness/`` is the only
artifact that proves a deferred import site still *runs* — every other guard in
this suite is static analysis or a ``sys.modules`` snapshot. Left beside the plan
it was a bare-assert file run by hand, which is the wrong home for the one check
that would notice a ``NameError`` on a colour-conversion path.

It stays a standalone script rather than being rewritten as test functions: its
assertions are about ``sys.modules`` state in a *fresh* interpreter — each
library absent until the exact access that should load it, present immediately
after — and pytest has already imported most of the library before any test
body runs. Subprocessing the real script keeps one source of truth.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-09-02-import-laziness"
    / "exercise_deferred_paths.py"
)


def test_the_exercise_script_exists_where_the_plan_says_it_does():
    """A moved or deleted script must fail loudly, not skip silently."""
    assert SCRIPT.is_file(), f"missing deferred-path exercise script: {SCRIPT}"


def test_every_deferred_path_still_runs_in_a_fresh_interpreter():
    """Drive all deferred import sites; assert the laziness in both directions."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=SCRIPT.parents[3],
    )
    assert completed.returncode == 0, (
        f"exercise script failed (exit {completed.returncode}):\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    assert "11/11 deferred paths exercised successfully" in completed.stdout


@pytest.mark.parametrize(
    "detector_name",
    ["FilamentousFungiDetector", "TwoKFilamentousDetector"],
)
def test_the_detectors_that_deferred_numba_still_construct_and_bind(detector_name):
    """The two ``_operate`` bodies where seven names each moved out of module scope.

    The exercise script reaches ``sdk_.reconnect`` directly, which proves the
    package imports but not that either detector's method body resolves the
    names it now imports locally. Running a full detection here would be slow
    and is already covered in ``tests/unit/detect``; what is checked instead is
    the specific thing the refactor could have broken — that the deferred names
    are importable from inside the method's own namespace.
    """
    import inspect

    from phenotypic import detect

    detector_cls = getattr(detect, detector_name)
    source = inspect.getsource(detector_cls._operate)
    assert "from phenotypic.sdk_.reconnect import" in source, (
        f"{detector_name}._operate no longer imports reconnect locally; if the "
        "import moved back to module scope, numba is eager again"
    )

    # Every name the method body imports must actually exist on the package.
    import phenotypic.sdk_.reconnect as reconnect

    block = source.split("from phenotypic.sdk_.reconnect import", 1)[1]
    block = block.split(")", 1)[0].lstrip("( \n")
    names = [n.strip().rstrip(",") for n in block.splitlines() if n.strip()]
    missing = [n for n in names if n and not hasattr(reconnect, n)]
    assert not missing, f"{detector_name}._operate imports missing names: {missing}"
