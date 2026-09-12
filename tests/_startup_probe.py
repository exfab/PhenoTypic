"""Run a snippet in a fresh interpreter and read back the report it prints.

A fresh process is the only honest place to ask what an import loads: other tests
in the same xdist worker have already filled ``sys.modules``, so an in-process check
passes or fails by accident. A probe that crashes, hangs or prints no report fails
the calling test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Prefix of the one stdout line carrying the probe's JSON report.
_REPORT_MARKER = "__PHENOTYPIC_STARTUP_PROBE__="

#: Seconds before a probe is killed; a hang must fail, not stall the suite. Generous
#: because a cold import of numba, cv2, plotly and mahotas on shared storage is slow,
#: and a timeout here is a hard failure rather than a retry.
PROBE_TIMEOUT_SECONDS = 180


def run_startup_probe(body: str) -> dict[str, Any]:
    """Execute ``body`` in a fresh interpreter and return the ``report`` it binds.

    Args:
        body: Python source run after ``import json, sys``. It must bind a
            JSON-serialisable name ``report``.

    Returns:
        The decoded ``report``.

    Raises:
        AssertionError: If the interpreter exits non-zero or prints no report.
        subprocess.TimeoutExpired: If the probe runs past
            :data:`PROBE_TIMEOUT_SECONDS`.
    """
    program = f"import json, sys\n{body}\nprint({_REPORT_MARKER!r} + json.dumps(report))\n"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg"}
    env.pop("PHENOTYPIC_DOCS_BUILD", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
        env=env,
        cwd=REPO_ROOT,
    )
    # Match anywhere in the line: a C extension (cv2, numba, mahotas) can write to fd 1
    # without a trailing newline, which would leave the marker mid-line.
    reports = [line.split(_REPORT_MARKER, 1)[1] for line in result.stdout.splitlines() if _REPORT_MARKER in line]
    if result.returncode != 0 or not reports:
        raise AssertionError(
            f"startup probe failed with exit {result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    return json.loads(reports[-1])
