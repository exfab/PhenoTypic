from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING, Any, Optional, get_args, get_origin, Literal

if TYPE_CHECKING:
    from phenotypic import Image
    from ipywidgets import Widget


class LazyWidgetMixin:
    """Mixin providing a lazy ipywidget interface.

    This mixin allows ImageOperation classes to automatically generate a Jupyter
    widget interface for parameter tuning and visualization.
    """

    _ui: Optional[Widget] = None
    _param_widgets: dict[str, Widget]
    _view_dropdown: Optional[Widget] = None
    _update_button: Optional[Widget] = None
    _output_widget: Optional[Widget] = None
    _image_ref: Optional[Image] = None

    def widget(self, image: Optional[Image] = None) -> Widget:
        """Return (and optionally display) the root widget.

        Args:
            image (Image | None): Optional image to visualize. If provided,
                visualization controls will be added to the widget.

        Returns:
            ipywidgets.Widget: The root widget.

        Raises:
            ImportError: If ipywidgets or IPython are not installed.
        """
        try:
            import ipywidgets
            from IPython.display import display
        except ImportError as e:
            raise ImportError(
                "The 'ipywidgets' and 'IPython' packages are required for the widget interface. "
                "Please install the 'jupyter' optional dependency group: "
                "pip install 'phenotypic[jupyter]'"
            ) from e

        # Store image reference for visualization
        if image is not None:
            self._image_ref = image

        if self._ui is None:
            self._create_widgets()

        # If we have an image and the visualization parts weren't created (e.g. widget() called before without image),
        # we might need to recreate or append?
        # For simplicity, if _ui exists, we assume it's good. 
        # But if image was added later, we might want to add viz widgets.
        # Let's just recreate if image is new and we don't have viz widgets yet?
        # Or simpler: if _image_ref is set, _create_widgets will include viz widgets.
        # If _ui exists but no output widget and now we have image, we should probably rebuild.
        
        has_viz = getattr(self, '_output_widget', None) is not None
        if image is not None and not has_viz:
            self._create_widgets() # Re-create to include viz
            
        display(self._ui)
        return self._ui

    def _create_widgets(self) -> None:
        """Create and assign the root widget to self._ui."""
        import ipywidgets as widgets
        
        self._param_widgets = {}
        controls = []

        # 1. Introspect __init__ parameters
        sig = inspect.signature(self.__init__)
        hints = typing.get_type_hints(self.__init__)
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'args' or param_name == 'kwargs':
                continue
                
            # Get current value from instance
            if hasattr(self, param_name):
                current_val = getattr(self, param_name)
            else:
                # Fallback to default if attribute missing (shouldn't happen for well-behaved ops)
                current_val = param.default if param.default is not inspect.Parameter.empty else None

            # Skip if we can't determine value or it's private
            if current_val is None and param.default is inspect.Parameter.empty:
                continue

            # Determine widget type
            widget = self._create_widget_for_param(param_name, hints.get(param_name, Any), current_val)
            if widget:
                self._param_widgets[param_name] = widget
                controls.append(widget)
                
                # Bind change
                widget.observe(self._on_param_change, names='value')

        # 2. Visualization controls (if image provided)
        if self._image_ref is not None:
            viz_controls = self._create_viz_widgets()
            # Combine
            left_panel = widgets.VBox(controls, layout=widgets.Layout(width='40%'))
            right_panel = widgets.VBox(viz_controls, layout=widgets.Layout(width='60%'))
            self._ui = widgets.HBox([left_panel, right_panel])
        else:
            self._ui = widgets.VBox(controls)

    def _create_widget_for_param(self, name: str, type_hint: Any, value: Any) -> Optional[Widget]:
        import ipywidgets as widgets
        
        # Handle Literal
        if get_origin(type_hint) is Literal:
            options = get_args(type_hint)
            return widgets.Dropdown(
                options=options,
                value=value,
                description=name,
                style={'description_width': 'initial'}
            )
            
        # Basic types
        if isinstance(value, bool):
             return widgets.Checkbox(value=value, description=name)
        elif isinstance(value, int):
             return widgets.IntText(value=value, description=name, style={'description_width': 'initial'})
        elif isinstance(value, float):
             return widgets.FloatText(value=value, description=name, style={'description_width': 'initial'})
        elif isinstance(value, str):
             return widgets.Text(value=value, description=name, style={'description_width': 'initial'})
             
        # Fallback for known types if value is None but hint exists?
        # For now, skip complex types
        return None

    def _create_viz_widgets(self) -> list[Widget]:
        import ipywidgets as widgets
        
        self._view_dropdown = widgets.Dropdown(
            options=['overlay', 'rgb', 'gray', 'enh_gray', 'objmap', 'objmask'],
            value='overlay',
            description='View:',
        )
        
        self._update_button = widgets.Button(
            description='Update View',
            button_style='info',
            icon='refresh'
        )
        self._update_button.on_click(self._on_update_view_click)
        
        self._output_widget = widgets.Output()
        
        return [
            widgets.HBox([self._view_dropdown, self._update_button]),
            self._output_widget
        ]

    def _on_param_change(self, change):
        if change['type'] != 'change' or change['name'] != 'value':
            return
            
        # Find which parameter this widget belongs to
        owner = change['owner']
        param_name = None
        for name, widget in self._param_widgets.items():
            if widget == owner:
                param_name = name
                break
                
        if param_name:
            setattr(self, param_name, change['new'])

    def _on_update_view_click(self, b):
        import matplotlib.pyplot as plt
        
        if self._output_widget is None or self._image_ref is None:
            return
            
        self._output_widget.clear_output(wait=True)
        
        with self._output_widget:
            try:
                # Create copy and apply
                # Note: self.apply is expected to exist on the subclass (ImageOperation)
                img_copy = self._image_ref.copy()
                
                # Check if we are in an ImageOperation
                if hasattr(self, 'apply'):
                     self.apply(img_copy, inplace=True)
                else:
                     print("Error: Mixin used on class without apply()")
                     return
                
                # Show
                view = self._view_dropdown.value
                
                # Handle closing previous figures to save memory?
                # plt.close('all') # Too aggressive?
                
                if view == 'overlay':
                    img_copy.show_overlay()
                elif view == 'rgb':
                    if not img_copy.rgb.isempty():
                        img_copy.rgb.show()
                    else:
                        print("No RGB data available.")
                elif view == 'gray':
                    img_copy.gray.show()
                elif view == 'enh_gray':
                    img_copy.enh_gray.show()
                elif view == 'objmap':
                    img_copy.objmap.show()
                elif view == 'objmask':
                    img_copy.objmask.show()
                    
                plt.show()
                
            except Exception as e:
                print(f"Error during visualization: {e}")
                import traceback
                traceback.print_exc()

    def sync_widgets_from_state(self) -> None:
        """Push internal state into widgets."""
        if not hasattr(self, '_param_widgets'):
            return
            
        for name, widget in self._param_widgets.items():
            val = getattr(self, name, None)
            if val is not None:
                widget.value = val

    def dispose_widgets(self) -> None:
        """Drop references to the UI widgets."""
        self._ui = None
        self._param_widgets = {}
        self._view_dropdown = None
        self._update_button = None
        self._output_widget = None

