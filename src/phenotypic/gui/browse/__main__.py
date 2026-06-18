"""Standalone launcher: ``python -m phenotypic.gui.browse --root ./images``."""
from __future__ import annotations

import argparse

from phenotypic.gui._config import (
    TITLE_BROWSE,
    add_launcher_args,
    configure_launcher_logging,
    print_launcher_banner,
)
from phenotypic.gui.browse._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot


def main() -> None:
    parser = argparse.ArgumentParser(description="PhenoTypic Source Browser")
    parser.add_argument("--root", default=".", help="Sandbox / source image root")
    add_launcher_args(parser)
    args = parser.parse_args()
    configure_launcher_logging(debug=args.debug)
    sandbox = SandboxRoot.from_path(args.root)
    app = create_app(sandbox, url_prefix=args.url_prefix)
    print_launcher_banner(
        title=TITLE_BROWSE,
        host=args.host,
        port=args.port,
        root=sandbox.root,
        url_prefix=args.url_prefix,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
