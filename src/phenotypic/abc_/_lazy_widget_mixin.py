from __future__ import annotations

import inspect
import re
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

    def widget(self, image: Optional[Image] = None, show: bool = False) -> Widget:
        """Return (and optionally display) the root widget.

        Args:
            image (Image | None): Optional image to visualize. If provided,
                visualization controls will be added to the widget.
            show (bool): Whether to display the widget immediately. Defaults to False.

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

        has_viz = getattr(self, '_output_widget', None) is not None
        if image is not None and not has_viz:
            self._create_widgets() # Re-create to include viz
            
        if show:
            display(self._ui)
        return self._ui

    def _create_widgets(self) -> None:
        """Create and assign the root widget to self._ui."""
        import ipywidgets as widgets
        import phenotypic as pht
        
        self._param_widgets = {}
        controls = []

        # 1. Introspect __init__ parameters
        # Skip for ImagePipeline 
        
        if not isinstance(self, pht.ImagePipeline):
            sig = inspect.signature(self.__init__)
            hints = typing.get_type_hints(self.__init__)
            
            # Parse docstring for parameter descriptions
            doc_params = self._parse_docstring()
            
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
                    
                    # Check for help text from docstring
                    if param_name in doc_params:
                        help_text = doc_params[param_name]
                        # Create help label
                        help_label = widgets.HTML(
                            value=f"<span style='color: #666; font-size: 0.85em; font-style: italic; margin-left: 2px;'>{help_text}</span>"
                        )
                        # Wrap widget and help in VBox
                        control_group = widgets.VBox([widget, help_label], layout=widgets.Layout(margin='0px 0px 8px 0px'))
                        controls.append(control_group)
                    else:
                        controls.append(widget)
                    
                    # Bind change
                    widget.observe(self._on_param_change, names='value')
        
        # 2. Recursive Inspection for _ops (Pipelines)
        if hasattr(self, '_ops') and isinstance(self._ops, dict):
            op_accordions = []
            for op_name, op in self._ops.items():
                # Check if the operation supports widgets (inherits LazyWidgetMixin or similar)
                if hasattr(op, 'widget'):
                    # Recursively create widget tree for the child op
                    # We don't pass image down recursively for viz yet, main pipeline handles viz
                    # We also don't show it immediately
                    child_widget = op.widget(image=None, show=False)
                    
                    # Create accordion/group
                    acc = widgets.Accordion(children=[child_widget])
                    acc.set_title(0, f"Operation: {op_name}")
                    op_accordions.append(acc)
            
            if op_accordions:
                # Add operations section to controls
                ops_container = widgets.VBox(op_accordions, layout=widgets.Layout(margin='10px 0px 0px 0px'))
                controls.append(ops_container)

        # 3. Visualization controls (if image provided)
        if self._image_ref is not None:
            viz_controls = self._create_viz_widgets()
            # Combine
            left_panel = widgets.VBox(controls, layout=widgets.Layout(width='40%'))
            right_panel = widgets.VBox(viz_controls, layout=widgets.Layout(width='60%'))
            self._ui = widgets.HBox([left_panel, right_panel])
        else:
            self._ui = widgets.VBox(controls)

    def _create_widget_for_param(self, name: str, type_hint: Any, value: Any) -> Optional[Widget]:
        import types
        from typing import Union
        
        origin = get_origin(type_hint)
        if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
             return self._create_union_widget(name, type_hint, value)

        return self._create_simple_widget(name, type_hint, value)

    def _create_union_widget(self, name: str, type_hint: Any, value: Any) -> Optional[Widget]:
        import ipywidgets as widgets
        import traitlets
        
        args = get_args(type_hint)
        type_map = {}
        
        for t in args:
            if t is type(None):
                type_map['None'] = t
            else:
                if hasattr(t, '__name__'):
                    t_name = t.__name__
                else:
                    t_name = str(t)
                type_map[t_name] = t
        
        if not type_map:
            return None
            
        class UnionWidget(widgets.VBox):
            value = traitlets.Any()

            def __init__(self, name, type_map, initial_value, mixin_ref):
                self.type_map = type_map
                self.mixin_ref = mixin_ref
                self.name = name
                self._ignore_updates = False
                
                self.selector = widgets.Dropdown(
                    options=list(type_map.keys()),
                    description=f'{name} type:',
                    style={'description_width': 'initial'}
                )
                
                self.inner_container = widgets.VBox()
                self.inner_container.layout = widgets.Layout(margin='0px 0px 0px 20px')
                self.inner_widget = None
                
                super().__init__([self.selector, self.inner_container])
                
                self.value = initial_value
                self._set_ui_from_value(initial_value)
                
                self.selector.observe(self._on_type_change, names='value')
                
            def _on_type_change(self, change):
                if self._ignore_updates: return
                
                selected_type_name = change['new']
                selected_type = self.type_map[selected_type_name]
                
                self._create_inner_widget(selected_type)
                self._update_value_from_inner()
                
            def _create_inner_widget(self, type_cls, value=None):
                self.inner_container.children = []
                
                if type_cls is type(None):
                    self.inner_widget = None
                    self.inner_container.children = [widgets.Label(value="Value: None")]
                    return

                if value is None:
                    if type_cls is int: value = 0
                    elif type_cls is float: value = 0.0
                    elif type_cls is str: value = ""
                    elif type_cls is bool: value = False
                    elif get_origin(type_cls) is Literal:
                        opts = get_args(type_cls)
                        value = opts[0] if opts else None

                self.inner_widget = self.mixin_ref._create_simple_widget(self.name, type_cls, value)
                
                if self.inner_widget:
                    self.inner_container.children = [self.inner_widget]
                    self.inner_widget.observe(self._on_inner_change, names='value')
                else:
                    self.inner_container.children = [widgets.Label(f"No widget for {type_cls}")]
                    self.inner_widget = None
                    
            def _on_inner_change(self, change):
                 if self._ignore_updates: return
                 self.value = change['new']
                 
            def _update_value_from_inner(self):
                if self.inner_widget:
                    self.value = self.inner_widget.value
                else:
                    selected_type = self.type_map[self.selector.value]
                    if selected_type is type(None):
                        self.value = None

            @traitlets.observe('value')
            def _on_value_change(self, change):
                new_val = change['new']
                self._set_ui_from_value(new_val)
                
            def _set_ui_from_value(self, val):
                self._ignore_updates = True
                try:
                    target_type_name = None
                    target_type = None
                    
                    if val is None:
                        if 'None' in self.type_map:
                            target_type_name = 'None'
                            target_type = type(None)
                    else:
                        for name, t in self.type_map.items():
                            if t is type(None): continue
                            if type(val) == t:
                                target_type_name = name
                                target_type = t
                                break
                        
                        if not target_type_name:
                             for name, t in self.type_map.items():
                                 if t is type(None): continue
                                 
                                 # Handle generics for isinstance check
                                 check_type = t
                                 origin = get_origin(t)
                                 
                                 if origin is Literal:
                                     # For Literal, we check if the value is one of the args
                                     lit_args = get_args(t)
                                     if val in lit_args:
                                         target_type_name = name
                                         target_type = t
                                         break
                                     continue
                                     
                                 if origin is not None:
                                     check_type = origin
                                     
                                 if isinstance(val, check_type): 
                                     target_type_name = name
                                     target_type = t
                                     break
                    
                    if target_type_name:
                        if self.selector.value != target_type_name:
                             self.selector.value = target_type_name
                             self._create_inner_widget(target_type, val)
                        else:
                             if self.inner_widget is None and target_type is not type(None):
                                 self._create_inner_widget(target_type, val)
                                 
                             if self.inner_widget:
                                 if self.inner_widget.value != val:
                                     self.inner_widget.value = val
                             elif target_type is type(None):
                                 pass 
                finally:
                    self._ignore_updates = False
        
        return UnionWidget(name, type_map, value, self)

    def _create_simple_widget(self, name: str, type_hint: Any, value: Any) -> Optional[Widget]:
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

    def _parse_docstring(self) -> dict[str, str]:
        """Parse the class docstring to extract parameter descriptions.
        
        Supports Google, NumPy, and Sphinx/ReST styles.
        
        Returns:
            dict: A dictionary mapping parameter names to their description strings.
        """
        doc = self.__doc__
        if not doc:
            return {}
            
        params = {}
        lines = doc.split('\n')
        
        # State machine variables
        in_param_section = False
        current_param = None
        current_desc = []
        
        # Regex patterns
        # Google: "Args:", "Attributes:", "Parameters:"
        section_header_re = re.compile(r'^\s*(Args|Attributes|Parameters)\s*:\s*$', re.IGNORECASE)
        # NumPy: "Parameters\n----------"
        numpy_header_re = re.compile(r'^\s*Parameters\s*$', re.IGNORECASE)
        numpy_underline_re = re.compile(r'^\s*-+\s*$')
        
        # Param patterns
        # Google: "name (type): description" or "name: description"
        google_param_re = re.compile(r'^\s*(\w+)\s*(\(.*?\))?\s*:\s*(.+)$')
        # NumPy: "name : type"
        numpy_param_re = re.compile(r'^\s*(\w+)\s*:\s*(.*)$')
        # Sphinx: ":param name: description" (can appear anywhere)
        sphinx_param_re = re.compile(r'^\s*:param\s+(\w+)\s*:\s*(.+)$')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            
            # Check for Sphinx style (line by line, no section needed)
            sphinx_match = sphinx_param_re.match(line)
            if sphinx_match:
                # Save previous param if exists
                if current_param:
                    params[current_param] = ' '.join(current_desc).strip()
                
                current_param = sphinx_match.group(1)
                current_desc = [sphinx_match.group(2)]
                in_param_section = False # Sphinx doesn't strictly enforce sections
                i += 1
                continue
                
            # Check for section headers
            if section_header_re.match(stripped_line):
                in_param_section = True
                # If previous param was pending, save it
                if current_param:
                    params[current_param] = ' '.join(current_desc).strip()
                    current_param = None
                    current_desc = []
                i += 1
                continue
                
            if numpy_header_re.match(stripped_line):
                # Check next line for underline
                if i + 1 < len(lines) and numpy_underline_re.match(lines[i+1]):
                    in_param_section = True
                    i += 2
                    continue
            
            if in_param_section:
                # End of section detection
                # Heuristic: if line is unindented and not empty, and doesn't match param pattern, section might be over
                if line and not line[0].isspace() and not google_param_re.match(line) and not numpy_param_re.match(line):
                     in_param_section = False
                     if current_param:
                        params[current_param] = ' '.join(current_desc).strip()
                        current_param = None
                        current_desc = []
                     i += 1
                     continue

                # Check for new parameter definition
                # Google style
                g_match = google_param_re.match(line)
                if g_match:
                    if current_param:
                        params[current_param] = ' '.join(current_desc).strip()
                    
                    current_param = g_match.group(1)
                    current_desc = [g_match.group(3)]
                    i += 1
                    continue
                    
                # NumPy style
                n_match = numpy_param_re.match(line)
                if n_match:
                    if current_param:
                        params[current_param] = ' '.join(current_desc).strip()
                    
                    current_param = n_match.group(1)
                    current_desc = [] # NumPy desc starts on next line
                    i += 1
                    continue
                
                # Continuation of description
                if current_param and stripped_line:
                    current_desc.append(stripped_line)
            
            i += 1
            
        # End of loop, save last param
        if current_param:
            params[current_param] = ' '.join(current_desc).strip()
            
        return params

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
                     # Use inplace=True for efficiency on the copy
                     # Check signature of apply to be safe (some might not support inplace?)
                     # Standard ImageOperation.apply supports inplace.
                     self.apply(img_copy, inplace=True)
                else:
                     print("Error: Mixin used on class without apply()")
                     return
                
                # Show
                view = self._view_dropdown.value
                
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
        self._image_ref = None

        # Propagate to children in _ops if they exist (e.g. for ImagePipeline)
        if hasattr(self, '_ops') and isinstance(self._ops, dict):
            for op in self._ops.values():
                if hasattr(op, 'dispose_widgets') and callable(op.dispose_widgets):
                    op.dispose_widgets()

    def __getstate__(self):
        """
        Prepare the object for pickling by disposing of any widgets.
        
        This ensures that UI components (which may contain unpickleable objects like
        input functions or thread locks) are cleaned up before serialization.
        
        Note:
            This method modifies the object state by calling dispose_widgets().
            Any active widgets will be detached from the object.
        """
        self.dispose_widgets()
        return self.__dict__
