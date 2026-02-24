"""Global session management for PhenoTypic GUI.

Provides automatic initialization of Panel extension and a persistent
InstanceManager, similar to napari's global viewer pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Global state
_panel_initialized = False
_global_instance_manager: Optional["InstanceManager"] = None


def _ensure_panel_initialized() -> None:
    """Ensure Panel extension is initialized (Jupyter only).

    Automatically calls pn.extension() on first use in Jupyter notebooks.
    Safe to call multiple times - only initializes once.
    """
    global _panel_initialized

    if _panel_initialized:
        return

    try:
        # Check if we're in a Jupyter environment
        get_ipython()  # type: ignore  # noqa: F821
        in_jupyter = True
    except NameError:
        in_jupyter = False

    if in_jupyter:
        import importlib.util

        if importlib.util.find_spec("panel") is not None:
            import panel as pn

            pn.extension()
            _panel_initialized = True


def get_global_manager(workspace: Optional[Path] = None) -> "InstanceManager":
    """Get or create the global InstanceManager instance.

    Similar to napari's global viewer pattern, this provides a persistent
    InstanceManager that can be reused across multiple GUI sessions.

    Args:
        workspace: Optional custom workspace directory. If provided on first call,
            sets the global manager's workspace. Subsequent calls ignore this
            parameter and return the existing manager. Default is ./pipelines/

    Returns:
        The global InstanceManager instance

    Examples:
        >>> from phenotypic.gui import get_global_manager
        >>>
        >>> # First call creates manager with default workspace
        >>> manager = get_global_manager()
        >>>
        >>> # Subsequent calls return the same instance
        >>> same_manager = get_global_manager()
        >>> assert manager is same_manager
        >>>
        >>> # Can specify custom workspace on first call
        >>> from pathlib import Path
        >>> manager = get_global_manager(workspace=Path("./my_pipelines"))
    """
    global _global_instance_manager

    if _global_instance_manager is None:
        from ._instance_manager import InstanceManager

        _global_instance_manager = InstanceManager(
            workspace=workspace, auto_cleanup=False
        )

    return _global_instance_manager


def reset_global_manager(workspace: Optional[Path] = None) -> "InstanceManager":
    """Reset the global InstanceManager with a new workspace.

    Useful for switching to a different pipeline workspace without
    restarting the kernel.

    Args:
        workspace: New workspace directory. None = ./pipelines/

    Returns:
        New global InstanceManager instance

    Examples:
        >>> from phenotypic.gui import reset_global_manager
        >>> from pathlib import Path
        >>>
        >>> # Switch to a different workspace
        >>> manager = reset_global_manager(workspace=Path("./experiment2"))
    """
    global _global_instance_manager

    from ._instance_manager import InstanceManager

    _global_instance_manager = InstanceManager(
        workspace=workspace, auto_cleanup=False
    )

    return _global_instance_manager
