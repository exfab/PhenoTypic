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

CLI defaults (host, port, debug flag) and the logging format come from
:mod:`phenotypic.gui._config` so they stay in lock-step with the
standalone debug launchers.
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
    SSH_TUNNEL_HINT,
    TITLE_HUB,
    add_launcher_args,
    configure_launcher_logging,
    print_launcher_banner,
)
from phenotypic.gui.shell._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._startup import StartupReporter, should_use_rich

logger = logging.getLogger(__name__)

__all__ = ["launch_gui", "main"]

#: Number of bar segments the launcher advances: core import (already done),
#: sandbox resolution, hub composition. See :class:`StartupReporter`.
_STARTUP_STEPS = 3


def _core_import_elapsed() -> float | None:
    """Seconds spent importing the ``phenotypic`` library before ``main()``.

    Measured against :data:`phenotypic._startup_perf.IMPORT_STARTED_AT`,
    which is stamped the moment the (heavy) package import begins — the only
    point early enough to capture it, since the console-script import chain
    runs before any launcher code. Returns ``None`` if the stamp is absent
    (e.g. a stubbed import in tests).
    """
    import time

    try:
        import phenotypic

        return time.perf_counter() - phenotypic._IMPORT_STARTED_AT
    except Exception:  # pragma: no cover - defensive
        return None


def launch_gui(
    root: Path | str = Path.cwd(),
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    debug: bool = False,
    url_prefix: str = DEFAULT_URL_PREFIX,
    *,
    reporter: StartupReporter | None = None,
) -> None:
    """Boot the unified PhenoTypic GUI shell.

    Args:
        root: Sandbox root. Strings accepted for ergonomics. Resolved to an
            absolute :class:`pathlib.Path` and frozen for the lifetime of
            the process. Defaults to the current working directory.
        host: Interface to bind. :data:`DEFAULT_HOST` keeps the server
            loopback-only — pair with SSH port forwarding for remote
            access. ``0.0.0.0`` exposes it on the network (not recommended
            without authentication; cloud mode is a non-goal in v1).
        port: TCP port. Defaults to :data:`DEFAULT_PORT`.
        debug: Run Dash in debug mode (auto-reload + verbose tracebacks).
            Defaults to ``False``.
        url_prefix: Browser-visible path prefix for reverse proxies such as
            Open OnDemand ``/node`` or ``/rnode``. Defaults to ``"/"``.
        reporter: Optional :class:`StartupReporter` driving staged progress
            feedback (sandbox resolution + hub composition). ``None``
            (default — the programmatic boot path) skips progress reporting
            entirely and just resolves, composes, and serves.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` exists but is not a directory.
        RuntimeError: On a symlink loop encountered while resolving root.
    """
    if reporter is None:
        sandbox = SandboxRoot.from_path(root)
        app = create_app(sandbox, url_prefix=url_prefix)
    else:
        with reporter:
            reporter.record_done(
                "Core library loaded", reporter.import_elapsed
            )
            with reporter.stage("Resolving sandbox root"):
                sandbox = SandboxRoot.from_path(root)
            with reporter.stage("Composing GUI hub"):
                app = create_app(
                    sandbox,
                    url_prefix=url_prefix,
                    progress=reporter.detail,
                )
        # ``sandbox``/``app`` are always bound here: the ``with`` bodies run
        # unless ``reporter`` raises, in which case we never reach this line.
    print_launcher_banner(
        title=TITLE_HUB,
        host=host,
        port=port,
        root=sandbox.root,
        url_prefix=url_prefix,
    )
    app.run(host=host, port=port, debug=debug)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phenotypic-gui",
        description=(
            "Launch the unified PhenoTypic GUI hub (pipeline builder, "
            "results viewer, and run console under one URL). "
            f"Reach the UI by SSH-tunnelling the chosen port (`{SSH_TUNNEL_HINT}`)."
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
    add_launcher_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point.

    Returns:
        Process exit code (0 = clean shutdown, non-zero = startup failure).
    """
    args = _build_parser().parse_args(argv)
    configure_launcher_logging(debug=args.debug)
    reporter = StartupReporter(
        total_steps=_STARTUP_STEPS,
        use_rich=should_use_rich(debug=args.debug),
        import_elapsed=_core_import_elapsed(),
    )
    try:
        launch_gui(
            root=args.root,
            host=args.host,
            port=args.port,
            debug=args.debug,
            url_prefix=args.url_prefix,
            reporter=reporter,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        print(f"phenotypic-gui: {exc}", file=sys.stderr)
        return 2
    return 0
