"""Remote display detection and configuration for X11 forwarding.

Must be imported and called **before** any Qt or napari imports so that
environment variables take effect before ``QApplication`` is created.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def detect_remote_session() -> bool:
    """Check if running over SSH.

    Returns:
        ``True`` if ``SSH_CONNECTION`` or ``SSH_CLIENT`` is set.
    """
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


def configure_remote_display() -> None:
    """Configure Qt and OpenGL for remote X11 forwarding.

    Sets environment variables that force software rendering so that napari
    works over ``ssh -X``.  Must be called **before** any Qt/napari imports.
    """
    os.environ["QT_OPENGL"] = "software"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ.pop("LIBGL_ALWAYS_INDIRECT", None)
    logger.info(
        "Remote session detected — enabled software OpenGL rendering "
        "(QT_OPENGL=software, LIBGL_ALWAYS_SOFTWARE=1)"
    )


def ensure_display_available() -> None:
    """Verify that a display server is reachable.

    On non-macOS/Windows systems a ``DISPLAY`` variable is required for X11.

    Raises:
        RuntimeError: If ``DISPLAY`` is not set on a platform that needs it.
    """
    import sys

    if sys.platform in ("darwin", "win32"):
        return  # macOS and Windows don't need DISPLAY

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "No DISPLAY environment variable set. "
            "If connecting via SSH, use 'ssh -X' or 'ssh -Y' to enable "
            "X11 forwarding, or set DISPLAY manually."
        )
