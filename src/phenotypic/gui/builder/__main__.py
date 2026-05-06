"""CLI entry point for the Dash + dash-cytoscape pipeline builder.

Boots a Dash server hosting an interactive node-graph editor for
constructing :class:`phenotypic.ImagePipeline` objects. Designed to run on
an HPCC head node and be reached from a workstation via SSH port
forwarding (see :data:`phenotypic.gui._config.SSH_TUNNEL_HINT`).

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
from pathlib import Path
from typing import Optional, Sequence

from phenotypic.gui._config import (
    SSH_TUNNEL_HINT,
    add_launcher_args,
    configure_launcher_logging,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser used by :func:`main`."""
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.gui.builder",
        description=(
            "Launch the Dash + dash-cytoscape ImagePipeline builder. "
            f"Reach the UI by SSH-tunneling the chosen port ('{SSH_TUNNEL_HINT}')."
        ),
    )
    add_launcher_args(parser)
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and run the builder Dash server.

    Args:
        argv: Optional argument vector for testing. ``None`` uses
            ``sys.argv[1:]``.
    """
    args = _build_parser().parse_args(argv)

    configure_launcher_logging(debug=args.debug)

    if args.image_root is not None and not args.image_root.is_dir():
        raise SystemExit(
            f"--image-root {args.image_root} is not an existing directory"
        )

    from phenotypic.gui.builder._app import create_app

    app = create_app(image_root=args.image_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
