"""Abstract base class for interactive image analysis tools using Dash."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np


class InteractiveImageAnalyzer(ABC):
    """Abstract base class for interactive Dash-based image analysis tools.
    
    This class provides common infrastructure for building interactive image
    analysis applications using Dash and Plotly. Subclasses should implement
    specific analysis functionality by overriding abstract methods.
    
    The analyzer handles:
    - Dash application lifecycle
    - Jupyter notebook integration
    - Image display with Plotly
    - Parameter controls
    - Measurement callbacks
    
    Args:
        image: The phenotypic.Image instance to analyze.
        port: Port number for the Dash server. Defaults to 8050.
        height: Height of the image display in pixels. Defaults to 800.
        mode: Display mode - 'inline', 'external', or 'jupyterlab'. Defaults to 'external'.
        
    Attributes:
        image: The image being analyzed.
        app: The Dash application instance.
        port: Server port number.
        height: Display height.
        mode: Display mode.
    """
    
    def __init__(
        self,
        image: Image,
        port: int = 8050,
        height: int = 800,
        mode: Literal['inline', 'external', 'jupyterlab'] = 'external'
    ):
        self.image = image
        self.port = port
        self.height = height
        self.mode = mode
        self.app = None
        self._validate_dependencies()
    
    @staticmethod
    def _validate_dependencies():
        """Validate that required optional dependencies are installed."""
        try:
            import dash
            import jupyter_dash
        except ImportError as e:
            raise ImportError(
                "Interactive features require optional dependencies. "
                "Install with: pip install phenotypic[interactive]"
            ) from e
    
    @abstractmethod
    def setup_layout(self):
        """Set up the Dash application layout.
        
        This method should define the complete layout structure including:
        - Image display components
        - Control widgets (sliders, buttons, etc.)
        - Results display areas
        
        Returns:
            dash.html.Div: The complete application layout.
        """
        pass
    
    @abstractmethod
    def create_callbacks(self):
        """Create Dash callbacks for interactivity.
        
        This method should register all callback functions that handle:
        - Parameter updates
        - Click events on images
        - Measurement calculations
        - Display updates
        """
        pass
    
    @abstractmethod
    def update_image(self, *args, **kwargs):
        """Update the displayed image based on current parameters.
        
        This method should generate the Plotly figure with:
        - Base image display
        - Object overlays
        - Interactive elements
        
        Returns:
            plotly.graph_objects.Figure: The updated figure.
        """
        pass
    
    def _create_app(self):
        """Initialize the Dash application."""
        # Use JupyterDash for notebook environments, regular Dash otherwise
        try:
            if self.mode in ['inline', 'jupyterlab']:
                from jupyter_dash import JupyterDash
                self.app = JupyterDash(__name__)
            else:
                import dash
                self.app = dash.Dash(__name__)
        except ImportError as e:
            # Fall back to regular Dash if jupyter-dash has issues
            import dash
            self.app = dash.Dash(__name__)
            if self.mode in ['inline', 'jupyterlab']:
                import warnings
                warnings.warn(
                    f"jupyter-dash import failed, falling back to external mode. "
                    f"Error: {e}"
                )
                self.mode = 'external'
        
        self.app.layout = self.setup_layout()
        self.create_callbacks()
    
    def _convert_image_to_plotly(self, image_array: np.ndarray) -> np.ndarray:
        """Convert image array to format suitable for Plotly display.
        
        Args:
            image_array: The image array (grayscale or RGB).
            
        Returns:
            np.ndarray: Array in format suitable for Plotly (0-255 uint8).
        """
        # Normalize to 0-255 range if needed
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        elif image_array.dtype == np.uint16:
            # Scale 16-bit to 8-bit
            image_array = (image_array / 256).astype(np.uint8)
        elif image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        return image_array
    
    def _create_overlay_image(self, base_image: np.ndarray, objmap: np.ndarray, 
                             alpha: float = 0.3) -> np.ndarray:
        """Create an overlay of objects on the base image.
        
        Args:
            base_image: Base grayscale or RGB image.
            objmap: Object map with labeled regions.
            alpha: Transparency of the overlay (0-1).
            
        Returns:
            np.ndarray: RGB image with overlay.
        """
        import skimage as ski
        
        # Convert grayscale to RGB if needed
        if base_image.ndim == 2:
            base_rgb = np.stack([base_image] * 3, axis=-1)
        else:
            base_rgb = base_image.copy()
        
        # Create overlay using skimage
        overlay = ski.color.label2rgb(
            label=objmap,
            image=base_rgb,
            bg_label=0,
            alpha=alpha
        )
        
        return self._convert_image_to_plotly(overlay)
    
    def run(self):
        """Launch the interactive application.
        
        This method creates the Dash app and starts the server based on the
        specified mode (inline, external, or jupyterlab).
        """
        self._create_app()
        
        # Run the app with appropriate parameters
        try:
            if self.mode in ['inline', 'jupyterlab']:
                # Jupyter modes - use jupyter-dash specific parameters
                self.app.run(
                    mode=self.mode,
                    port=self.port,
                    height=self.height,
                    debug=False
                )
            else:
                # External mode - open in browser
                # Check if this is a JupyterDash instance
                if hasattr(self.app, 'run_server') and 'mode' in self.app.run_server.__code__.co_varnames:
                    self.app.run(
                        port=self.port,
                        debug=False,
                        mode='external'
                    )
                else:
                    # Regular Dash app
                    self.app.run_server(
                        port=self.port,
                        debug=False,
                        dev_tools_hot_reload=False
                    )
        except AttributeError as e:
            # Handle cases where run_server might not be available
            print(f"Error starting server: {e}")
            print("Attempting to use standard Dash server...")
            self.app.run_server(
                port=self.port,
                debug=False
            )

