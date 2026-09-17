"""Best-of-N wall-clock time for the startup paths in the lazy-startup spec.

The numbers are reported, never asserted: timing on shared machines is noisy, and
the tests guard the module set instead. Each path runs in a fresh interpreter, so
a module one path imported cannot make another path look fast.

Run it from the checkout whose code you want to time. The interpreter running this
script is the one it measures:

    uv run python docs/superpowers/plans/2026-09-11-lazy-startup/measure_startup.py \
        --label after --out docs/superpowers/reports/2026-09-11-lazy-startup/startup-after.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_CLI_HELP = """\
import contextlib, io, runpy, sys
sys.argv = ["phenotypic", "--help"]
with contextlib.redirect_stdout(io.StringIO()):
    try:
        runpy.run_module("phenotypic", run_name="__main__")
    except SystemExit:
        pass
"""

_GUI_HELP = """\
import contextlib, io
from phenotypic._gui.shell._launcher import main
with contextlib.redirect_stdout(io.StringIO()):
    try:
        main(["--help"])
    except SystemExit:
        pass
"""

_HUB = """\
import tempfile
from phenotypic._gui.shell._app import create_app
from phenotypic._gui.shell._sandbox import SandboxRoot
app = create_app(
    SandboxRoot.from_path(tempfile.mkdtemp()),
    start_idle_thread=False,
    start_slurm_observer=False,
)
"""

#: Label -> program run as ``python -c``. The order is the report's row order.
STARTUP_PATHS: dict[str, str] = {
    "bare interpreter": "pass",
    "import phenotypic": "import phenotypic",
    "from phenotypic import Image": "from phenotypic import Image",
    "python -m phenotypic --help": _CLI_HELP,
    "phenotypic-gui --help": _GUI_HELP,
    "hub to servable (no request)": _HUB,
    "hub + first /builder/ request": _HUB + 'assert app.server.test_client().get("/builder/").status_code == 200\n',
}


def time_program(program: str, *, repeats: int) -> tuple[float, list[float]]:
    """Return the best and all wall-clock times for ``program`` in fresh interpreters.

    One discarded warm-up run fills the bytecode cache first, so the first measured
    run is not paying for compilation the others skip.
    """
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg"}
    env.pop("PHENOTYPIC_DOCS_BUILD", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    command = [sys.executable, "-c", program]
    subprocess.run(command, check=True, capture_output=True, env=env)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        subprocess.run(command, check=True, capture_output=True, env=env)
        samples.append(time.perf_counter() - started)
    return min(samples), samples


def measure_all(repeats: int) -> dict[str, dict[str, object]]:
    """Time every startup path, in declaration order."""
    results: dict[str, dict[str, object]] = {}
    for label, program in STARTUP_PATHS.items():
        best, samples = time_program(program, repeats=repeats)
        results[label] = {"best_seconds": round(best, 3), "samples_seconds": [round(s, 3) for s in samples]}
        print(f"{label:34s} best {best:6.3f} s", flush=True)
    return results


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Best-of-N wall-clock time for the lazy-startup paths.")
    parser.add_argument("--label", required=True, help="Name of this measurement, e.g. before or after.")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured runs per path (default 5).")
    return parser.parse_args(argv)


def write_measurement(argv: list[str] | None = None) -> int:
    """Measure the startup paths and write the labelled JSON record."""
    args = parse_arguments(argv)
    record = {
        "label": args.label,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "repeats": args.repeats,
        "paths": measure_all(args.repeats),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(write_measurement())
