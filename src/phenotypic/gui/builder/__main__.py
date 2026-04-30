"""CLI entry point for the Dash + dash-cytoscape pipeline builder.

Boots a Dash server hosting an interactive node-graph editor for
constructing :class:`phenotypic.ImagePipeline` objects. Designed to run on
an HPCC head node and be reached from a workstation via SSH port
forwarding (``ssh -L 8050:localhost:8050 user@hpcc``).

Examples:
    Default port + bind to all interfaces (so an SSH tunnel to a
    cluster login node sees the server on the forwarded port)::

        uv run python -m phenotypic.gui.builder

    Serve from a project scratch directory so the in-app file picker
    can browse the user's images::

        uv run python -m phenotypic.gui.builder \\
            --image-root /scratch/$USER/images \\
            --port 8050
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser used by :func:`main`."""
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.gui.builder",
        description=(
            "Launch the Dash + dash-cytoscape ImagePipeline builder. "
            "Reach the UI by SSH-tunneling the chosen port "
            "('ssh -L 8050:localhost:8050 user@hpcc')."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind. Default 127.0.0.1 keeps the server "
            "loopback-only — pair with SSH port forwarding for remote "
            "access. Use 0.0.0.0 to expose on the network (not "
            "recommended without authentication)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="TCP port to bind. Default 8050.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Directory rooting the in-app file picker. Omit to disable "
            "the tree (the user can still type a path or use the "
            "synthetic plate fallback)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode (auto-reload, verbose tracebacks).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and run the builder Dash server.

    Args:
        argv: Optional argument vector for testing. ``None`` uses
            ``sys.argv[1:]``.
    """
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.image_root is not None and not args.image_root.is_dir():
        raise SystemExit(
            f"--image-root {args.image_root} is not an existing directory"
        )

    from phenotypic.gui.builder._app import create_app

    app = create_app(image_root=args.image_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
