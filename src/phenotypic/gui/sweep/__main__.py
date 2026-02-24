"""Entry point for ``python -m phenotypic.gui.sweep [path]``."""

from __future__ import annotations

import sys as _sys
from pathlib import Path


def launch_napari_sweep_viewer() -> None:
    """Launch the napari sweep results viewer."""
    import logging

    sweep_logger = logging.getLogger("phenotypic.gui.sweep")
    sweep_logger.setLevel(logging.DEBUG)
    if not sweep_logger.handlers:
        handler = logging.StreamHandler(_sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(name)s %(levelname)s: %(message)s")
        )
        sweep_logger.addHandler(handler)

    sweep_dir = (
        Path(_sys.argv[1]) if len(_sys.argv) > 1 else Path.cwd()
    )
    sweep_logger.info("Sweep viewer starting — root: %s", sweep_dir)

    # Remote display setup BEFORE any Qt/napari imports
    from ._remote_display import (
        configure_remote_display,
        detect_remote_session,
        ensure_display_available,
    )

    ensure_display_available()
    if detect_remote_session():
        configure_remote_display()

    # Now safe to import napari/Qt
    from ._napari_sweep_viewer import launch_sweep_viewer

    launch_sweep_viewer(sweep_dir)


if __name__ == "__main__":
    launch_napari_sweep_viewer()
