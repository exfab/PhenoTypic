"""Shell launcher + ``phenotypic-gui`` console-script entry point.

This module ships two callables:

    * :func:`launch_gui` — programmatic boot used by the ``__main__`` module
      and downstream tests. Mirrors
      :func:`phenotypic.gui.results_viewer.launch_results_viewer` for
      consistency with the existing standalone tools.
    * :func:`main` — argparse front-end wired into ``[project.scripts]``
      (``phenotypic-gui = phenotypic.gui.shell._launcher:main``) and into
      ``python -m phenotypic.gui``.

The launcher refuses to boot if ``--root`` does not exist or is not a
directory; we catch :class:`ValueError`/``FileNotFoundError``/etc. from
:meth:`SandboxRoot.from_path` and surface a clean error rather than a
stack trace.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from phenotypic.gui.shell._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["launch_gui", "main"]


def launch_gui(
    root: Path | str = Path.cwd(),
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Boot the unified PhenoTypic GUI shell.

    Args:
        root: Sandbox root. Strings accepted for ergonomics. Resolved to an
            absolute :class:`pathlib.Path` and frozen for the lifetime of
            the process. Defaults to the current working directory.
        host: Interface to bind. ``127.0.0.1`` (default) keeps the server
            loopback-only — pair with SSH port forwarding for remote
            access. ``0.0.0.0`` exposes it on the network (not recommended
            without authentication; cloud mode is a non-goal in v1).
        port: TCP port. Defaults to ``8050``.
        debug: Run Dash in debug mode (auto-reload + verbose tracebacks).
            Defaults to ``False``.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` exists but is not a directory.
        RuntimeError: On a symlink loop encountered while resolving root.
    """
    sandbox = SandboxRoot.from_path(root)
    app = create_app(sandbox)
    _print_banner(host, port, sandbox.root)
    app.run(host=host, port=port, debug=debug)


def _print_banner(host: str, port: int, root: Path) -> None:
    """Print a one-shot startup banner with SSH-tunnel + cache-nuke hints."""
    print()
    print("PhenoTypic GUI")
    print(f"  root  : {root}")
    print(f"  url   : http://{host}:{port}/")
    print()
    print(f"  SSH tunnel from local: ssh -L {port}:localhost:{port} <cluster>")
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phenotypic-gui",
        description=(
            "Launch the unified PhenoTypic GUI hub (pipeline builder, "
            "results viewer, and run console under one URL). "
            "Reach the UI by SSH-tunnelling the chosen port "
            "(`ssh -L 8050:localhost:8050 user@cluster`)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Sandbox root. The GUI's file browser is restricted to this "
            "directory. Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Interface to bind. Default 127.0.0.1 keeps the server "
            "loopback-only — pair with SSH port forwarding for remote "
            "access."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="TCP port. Default 8050.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point.

    Returns:
        Process exit code (0 = clean shutdown, non-zero = startup failure).
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        launch_gui(
            root=args.root,
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        print(f"phenotypic-gui: {exc}", file=sys.stderr)
        return 2
    return 0
