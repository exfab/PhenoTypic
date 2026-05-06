"""CLI launcher for the PhenoTypic results viewer.

Boots a Dash server hosting an OpenSeadragon-backed interactive viewer
over a CLI output directory produced by ``python -m phenotypic``. The
typical workflow is to run this on a cluster login or compute node and
SSH-tunnel the chosen port back to a workstation::

    ssh -L 8050:localhost:8050 user@cluster
    # on the cluster:
    cd <output-root>
    uv run python -m phenotypic.gui.results_viewer

Then point a local browser at ``http://localhost:8050/``.

Examples:
    Default port, current working directory as the output root::

        uv run python -m phenotypic.gui.results_viewer

    Explicit output root + port + Dash debug mode::

        uv run python -m phenotypic.gui.results_viewer \\
            --output-root /scratch/$USER/run-2026-01 \\
            --port 9000 \\
            --debug
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from phenotypic.gui._config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    SSH_TUNNEL_HINT,
    TITLE_VIEWER,
    VIEWER_CACHE_DIRNAME,
    add_launcher_args,
    configure_launcher_logging,
    print_launcher_banner,
)
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


def launch_results_viewer(
    output_root: Path | str = Path.cwd(),
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    debug: bool = False,
) -> None:
    """Boot the Dash results viewer against an output root.

    Validates the directory layout via
    :meth:`OutputRoot.discover`, builds the Dash app via
    :func:`phenotypic.gui.results_viewer._app.create_app`, prints a
    one-shot startup banner with SSH-tunnel and cache-nuke hints, and
    finally hands control to ``app.run`` (which blocks until the user
    interrupts).

    Args:
        output_root: Path to a CLI output directory (the one containing
            ``master_measurements.parquet`` and ``results/``). Strings
            are accepted for ergonomics; both are resolved to an
            absolute :class:`pathlib.Path`. Defaults to the current
            working directory.
        host: Interface to bind. :data:`DEFAULT_HOST` keeps the server
            loopback-only -- pair with SSH port forwarding for remote
            access. ``0.0.0.0`` exposes the app on the network.
        port: TCP port to bind. Defaults to :data:`DEFAULT_PORT`.
        debug: Run Dash in debug mode (auto-reload + verbose
            tracebacks). Defaults to ``False``.

    Raises:
        FileNotFoundError: If *output_root* does not contain a valid CLI
            output layout.
        ValueError: If the master measurements parquet is missing
            required ``Metadata_Dataset`` / ``Metadata_ImageFile``
            columns.
    """
    root = Path(output_root).resolve()
    output = OutputRoot.discover(root)
    app = create_app(output)
    cache_dir = root / VIEWER_CACHE_DIRNAME
    print_launcher_banner(
        title=TITLE_VIEWER,
        host=host,
        port=port,
        root=root,
        extra_lines=(f"Clear tile cache    : rm -rf {cache_dir}",),
    )
    app.run(host=host, port=port, debug=debug)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser used by :func:`main`.

    Returns:
        A configured :class:`argparse.ArgumentParser` with
        ``--output-root``, ``--host``, ``--port``, and ``--debug``
        flags. Defaults match :func:`launch_results_viewer`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.gui.results_viewer",
        description=(
            "Launch the PhenoTypic results viewer (Dash + OpenSeadragon) "
            "over a CLI output directory. Browse, filter, and pixel-zoom "
            "per-image overlays produced by `python -m phenotypic`. "
            f"Reach the UI by SSH-tunneling the chosen port (`{SSH_TUNNEL_HINT}`)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Path to the CLI output directory containing "
            "`master_measurements.parquet` and `results/<dataset>/overlays/`. "
            "Defaults to the current working directory."
        ),
    )
    add_launcher_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Module-level entry point invoked by ``python -m``.

    Args:
        argv: Optional argument vector for testing. ``None`` (the
            default) lets argparse read from ``sys.argv[1:]``.
    """
    args = _build_parser().parse_args(argv)
    configure_launcher_logging(debug=args.debug)
    launch_results_viewer(
        output_root=args.output_root,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
