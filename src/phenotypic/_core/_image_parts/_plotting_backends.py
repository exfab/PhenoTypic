"""Backend abstraction for image plotting with matplotlib and plotly support.

This module provides utilities for unified image visualization across matplotlib
and plotly backends, enabling interactive and static plotting with a consistent
interface.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Type aliases
MatplotlibReturn = Tuple[Figure, Axes]
PlotlyReturn = Any  # Will be go.Figure at runtime
PlotReturn = Union[MatplotlibReturn, PlotlyReturn]

# Backend availability cache
_PLOTLY_AVAILABLE: bool | None = None


def check_plotly_available() -> bool:
    """Check if plotly is available for import.

    Uses cached result to avoid repeated import attempts.

    Returns:
        bool: True if plotly is available, False otherwise.
    """
    global _PLOTLY_AVAILABLE

    if _PLOTLY_AVAILABLE is not None:
        return _PLOTLY_AVAILABLE

    try:
        import plotly.graph_objects as go  # noqa: F401
        _PLOTLY_AVAILABLE = True
        return True
    except ImportError:
        _PLOTLY_AVAILABLE = False
        return False


def validate_backend(backend: str) -> Literal["matplotlib", "plotly"]:
    """Validate and normalize backend string.

    Args:
        backend: Backend name ("matplotlib" or "plotly")

    Returns:
        Normalized backend name

    Raises:
        ValueError: If backend is not supported
        ImportError: If plotly requested but unavailable
    """
    backend = backend.lower().strip()

    if backend not in ("matplotlib", "plotly"):
        raise ValueError(
            f"Unsupported backend: '{backend}'. "
            "Must be 'matplotlib' or 'plotly'."
        )

    if backend == "plotly" and not check_plotly_available():
        raise ImportError(
            "Plotly backend requested but plotly is not installed.\n"
            "Install with: pip install plotly>=6.5.0\n"
            "Or: pip install 'phenotypic[plotly]'\n"
            "See: https://plotly.com/python/getting-started/"
        )

    return backend  # type: ignore[return-value]


def translate_colormap(mpl_cmap: str, backend: Literal["matplotlib", "plotly"]) -> str:
    """Translate matplotlib colormap name to plotly equivalent.

    Args:
        mpl_cmap: Matplotlib colormap name
        backend: Target backend

    Returns:
        Colormap name for the target backend
    """
    if backend == "matplotlib":
        return mpl_cmap

    # Plotly colormap translation table
    MPL_TO_PLOTLY = {
        # Grayscale
        "gray": "gray",
        "grey": "gray",
        "bone": "Greys",
        # Sequential
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "cividis": "Cividis",
        # Diverging
        "coolwarm": "RdBu",
        "bwr": "RdBu",
        "seismic": "RdBu",
        "rdbu": "RdBu",
        "rdylbu": "RdYlBu",
        # Rainbow/Jet
        "jet": "Jet",
        "hsb": "HSV",
        "hsv": "HSV",
        "rainbow": "Jet",
        # Thermal
        "hot": "Hot",
        "copper": "Oranges",
        # Nature
        "spring": "Reds",
        "summer": "Greens",
        "autumn": "YlOrRd",
        "winter": "Blues",
        "cool": "Blues",
        # Categorical
        "tab10": "Plotly",
        "tab20": "Plotly",
        "set1": "Set1",
        "set2": "Set2",
        "set3": "Set3",
        "paired": "Paired",
        # Special
        "nipy_spectral": "Portland",
    }

    plotly_name = MPL_TO_PLOTLY.get(mpl_cmap.lower())

    if plotly_name is None:
        warnings.warn(
            f"No plotly equivalent for matplotlib colormap '{mpl_cmap}'. "
            f"Using 'Viridis' instead.",
            UserWarning,
            stacklevel=3,
        )
        return "Viridis"

    return plotly_name


def plot_image_matplotlib(
    arr: np.ndarray,
    figsize: Tuple[int, int],
    title: str | None,
    cmap: str,
    ax: Axes | None,
    mpl_settings: dict | None,
) -> MatplotlibReturn:
    """Plot image using matplotlib backend.

    This contains the matplotlib-specific plotting logic extracted from
    ImageAccessorBase._plot().

    Args:
        arr: Image array to plot (2D or 3D)
        figsize: Figure size in inches
        title: Plot title
        cmap: Matplotlib colormap name
        ax: Existing axes or None
        mpl_settings: Additional imshow kwargs

    Returns:
        Tuple of (Figure, Axes)
    """
    from phenotypic.tools.funcs_ import normalize_rgb_bitdepth

    fig, ax = (ax.get_figure(), ax) if ax else plt.subplots(figsize=figsize)

    mpl_settings = mpl_settings if mpl_settings else {}
    cmap = mpl_settings.pop("cmap", cmap)

    # Handle bit depth normalization
    plot_arr = normalize_rgb_bitdepth(arr) if arr.ndim == 3 else arr

    if np.issubdtype(plot_arr.dtype, np.integer):
        vmax = np.iinfo(plot_arr.dtype).max
    elif np.issubdtype(plot_arr.dtype, np.floating):
        vmax = 1.0
    else:
        vmax = 1
    vmax = mpl_settings.pop("vmax", vmax)

    # Plot based on dimensionality
    if plot_arr.ndim == 2:
        ax.imshow(plot_arr, cmap=cmap, **mpl_settings)
    else:
        ax.imshow(plot_arr, vmax=vmax, **mpl_settings)

    ax.grid(False)

    if title:
        ax.set_title(title)

    return fig, ax


def plot_image_plotly(
    arr: np.ndarray,
    figsize: Tuple[int, int],
    title: str | None,
    cmap: str,
    plotly_settings: dict | None,
) -> PlotlyReturn:
    """Plot image using plotly backend.

    Args:
        arr: Image array to plot (2D or 3D)
        figsize: Figure size in inches (converted to pixels)
        title: Plot title
        cmap: Matplotlib colormap name (will be translated)
        plotly_settings: Additional trace kwargs

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from phenotypic.tools.funcs_ import normalize_rgb_bitdepth

    plotly_settings = plotly_settings if plotly_settings else {}

    # Convert figsize from inches to pixels (assume 100 DPI)
    width_px = int(figsize[0] * 100)
    height_px = int(figsize[1] * 100)

    # Handle bit depth normalization
    plot_arr = normalize_rgb_bitdepth(arr) if arr.ndim == 3 else arr

    if plot_arr.ndim == 2:
        # Grayscale image
        colorscale = translate_colormap(cmap, "plotly")

        fig = go.Figure(
            data=go.Heatmap(z=plot_arr, colorscale=colorscale, **plotly_settings)
        )
    else:
        # RGB image
        # Plotly expects RGB in 0-255 range for imshow
        if plot_arr.dtype in (np.float64, np.float32):
            plot_arr = (plot_arr * 255).astype(np.uint8)

        fig = go.Figure(data=go.Image(z=plot_arr, **plotly_settings))

    # Configure layout
    fig.update_layout(
        title=title,
        width=width_px,
        height=height_px,
        xaxis=dict(showgrid=False, showticklabels=True),
        yaxis=dict(showgrid=False, showticklabels=True, scaleanchor="x"),
    )

    return fig


def plot_overlay_plotly(
    arr: np.ndarray,
    objmap: np.ndarray,
    figsize: Tuple[int, int],
    title: str | None,
    overlay_alpha: float,
    plotly_settings: dict | None,
) -> PlotlyReturn:
    """Plot image with object overlay using plotly.

    Args:
        arr: Base image array
        objmap: Object label map
        figsize: Figure size in inches
        title: Plot title
        overlay_alpha: Opacity for overlay (0-1)
        plotly_settings: Additional trace kwargs

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import skimage as ski
    from phenotypic.tools.funcs_ import normalize_rgb_bitdepth

    # Create overlay using skimage
    overlay_arr = ski.color.label2rgb(
        label=objmap, image=arr, bg_label=0, alpha=overlay_alpha
    )

    # Convert to proper format for plotly
    plot_arr = normalize_rgb_bitdepth(overlay_arr)
    if plot_arr.dtype in (np.float32, np.float64):
        plot_arr = (plot_arr * 255).astype(np.uint8)

    # Create figure
    width_px = int(figsize[0] * 100)
    height_px = int(figsize[1] * 100)

    plotly_settings = plotly_settings if plotly_settings else {}

    fig = go.Figure(data=go.Image(z=plot_arr, **plotly_settings))

    fig.update_layout(
        title=title,
        width=width_px,
        height=height_px,
        xaxis=dict(showgrid=False, showticklabels=True),
        yaxis=dict(showgrid=False, showticklabels=True, scaleanchor="x"),
    )

    return fig


def add_scatter_annotations_plotly(
    fig: Any,
    labels: np.ndarray,
    centroids: list[Tuple[float, float]],
    color: str = "white",
    size: int = 12,
) -> Any:
    """Add text annotations to plotly figure for object labels.

    Equivalent to matplotlib's ax.text() for marking object centroids.

    Args:
        fig: Plotly figure
        labels: Object label values
        centroids: List of (row, col) centroid positions
        color: Text color
        size: Font size

    Returns:
        Modified plotly figure
    """
    annotations = []
    for label, (row, col) in zip(labels, centroids):
        annotations.append(
            dict(
                x=col,
                y=row,
                text=str(label),
                showarrow=False,
                font=dict(color=color, size=size),
                bgcolor="rgba(255, 0, 0, 0.6)",
                borderpad=4,
            )
        )

    fig.update_layout(annotations=annotations)
    return fig
