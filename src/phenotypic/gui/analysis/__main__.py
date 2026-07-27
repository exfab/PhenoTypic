"""Standalone entry point for the analysis sub-app.

Use ``python -m phenotypic.gui.analysis --root <output-dir>`` to launch
the sub-app outside the unified hub. Mirrors the launcher convention
used by ``phenotypic.gui.results_viewer.__main__``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from phenotypic.gui._config import (
    TITLE_ANALYSIS,
    add_launcher_args,
    configure_launcher_logging,
    print_launcher_banner,
)
from phenotypic.gui.analysis import create_app
from phenotypic.gui.results_viewer._output_root import (
    OutputRoot,
    user_viewer_cache_root,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.gui.analysis",
        description=(
            "PhenoTypic analysis sub-app. Reads pipeline.json from the "
            "given CLI output root, lets you compose filters / model, "
            "and emits analysis.{csv,parquet}."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to a CLI output root (must contain "
        "master_measurements.parquet).",
    )
    add_launcher_args(parser)
    args = parser.parse_args(argv)

    configure_launcher_logging(debug=args.debug)

    output_root = OutputRoot.discover(
        args.root,
        cache_root=user_viewer_cache_root(),
    )
    app = create_app(output_root=output_root, url_prefix=args.url_prefix)

    print_launcher_banner(
        title=TITLE_ANALYSIS,
        host=args.host,
        port=args.port,
        root=args.root,
        url_prefix=args.url_prefix,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
