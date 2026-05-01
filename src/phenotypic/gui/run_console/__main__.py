"""Standalone launcher for the Run console (debugging parity).

Mirrors :mod:`phenotypic.gui.builder.__main__` and
:mod:`phenotypic.gui.results_viewer.__main__` so contributors can spin
up just the Run console while iterating on its form / callbacks. The
unified hub entry point is ``python -m phenotypic.gui``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from phenotypic.gui.run_console._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)


def launch_run_console(
    root: Path | str = Path.cwd(),
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Boot the standalone Run console."""
    sandbox = SandboxRoot.from_path(root)
    app = create_app(sandbox)
    print(f"PhenoTypic Run console — http://{host}:{port}/  (sandbox: {sandbox.root})")
    app.run(host=host, port=port, debug=debug)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.gui.run_console",
        description="Run the standalone Run console for debugging.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        launch_run_console(
            root=args.root, host=args.host, port=args.port, debug=args.debug
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        print(f"phenotypic.gui.run_console: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
