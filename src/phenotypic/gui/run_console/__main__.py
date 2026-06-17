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

from phenotypic.gui._config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_URL_PREFIX,
    TITLE_RUN,
    add_launcher_args,
    configure_launcher_logging,
    print_launcher_banner,
)
from phenotypic.gui.run_console._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)


def launch_run_console(
    root: Path | str = Path.cwd(),
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    debug: bool = False,
    url_prefix: str = DEFAULT_URL_PREFIX,
) -> None:
    """Boot the standalone Run console."""
    sandbox = SandboxRoot.from_path(root)
    app = create_app(
        sandbox,
        url_prefix=url_prefix,
        server_url_prefix=url_prefix,
    )
    print_launcher_banner(
        title=TITLE_RUN,
        host=host,
        port=port,
        root=sandbox.root,
        url_prefix=url_prefix,
    )
    app.run(host=host, port=port, debug=debug)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.gui.run_console",
        description="Run the standalone Run console for debugging.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    configure_launcher_logging(debug=args.debug)
    try:
        launch_run_console(
            root=args.root,
            host=args.host,
            port=args.port,
            debug=args.debug,
            url_prefix=args.url_prefix,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        print(f"phenotypic.gui.run_console: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
