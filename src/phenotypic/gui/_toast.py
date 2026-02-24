"""Toast notification helper for Panel applications.

Provides simple interface for displaying success/error/warning/info messages
using Panel's notification system.
"""

from __future__ import annotations


class ToastNotification:
    """Toast notification helper for Panel apps.

    Provides simple interface for showing success/error/warning messages
    using Panel's global notification system.

    Examples:
        >>> import panel as pn
        >>> from phenotypic.gui._toast import ToastNotification
        >>>
        >>> # Initialize (app parameter not used, kept for consistency)
        >>> toast = ToastNotification(app=None)
        >>>
        >>> # Show success message
        >>> toast.success("Pipeline saved successfully")
        >>>
        >>> # Show error message
        >>> toast.error("Pipeline execution failed")
        >>>
        >>> # Show warning
        >>> toast.warning("Parameter type mismatch detected")
        >>>
        >>> # Show info
        >>> toast.info("Loading preview image...")
    """

    def __init__(self, app=None):
        """Initialize ToastNotification.

        Args:
            app: Panel app/layout (not currently used, reserved for future)
        """
        self._app = app

    def success(self, message: str, duration: int = 3000) -> None:
        """Show success toast notification.

        Args:
            message: Success message to display
            duration: Display duration in milliseconds (default 3000)
        """
        import panel as pn

        # Only show if in Panel server context
        if pn.state.notifications is not None:
            pn.state.notifications.success(message, duration=duration)

    def error(self, message: str, duration: int = 5000) -> None:
        """Show error toast notification.

        Args:
            message: Error message to display
            duration: Display duration in milliseconds (default 5000)
        """
        import panel as pn

        # Only show if in Panel server context
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=duration)

    def warning(self, message: str, duration: int = 4000) -> None:
        """Show warning toast notification.

        Args:
            message: Warning message to display
            duration: Display duration in milliseconds (default 4000)
        """
        import panel as pn

        # Only show if in Panel server context
        if pn.state.notifications is not None:
            pn.state.notifications.warning(message, duration=duration)

    def info(self, message: str, duration: int = 3000) -> None:
        """Show info toast notification.

        Args:
            message: Info message to display
            duration: Display duration in milliseconds (default 3000)
        """
        import panel as pn

        # Only show if in Panel server context
        if pn.state.notifications is not None:
            pn.state.notifications.info(message, duration=duration)
