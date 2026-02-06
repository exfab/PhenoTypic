"""Entry point for ``python -m phenotypic.gui.sweep [path]``."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the napari sweep results viewer."""
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

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
    main()
