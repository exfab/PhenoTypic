"""Interactive measurement tool for area calculation and object selection."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
import pandas as pd

from ._interactive_image_analyzer import InteractiveImageAnalyzer


class InteractiveMeasurementAnalyzer(InteractiveImageAnalyzer):
    """Interactive Dash application for measuring object areas.
    
    This tool provides an interactive interface for:
    - Visualizing detected objects with overlays
    - Clicking to select individual objects
    - Calculating and displaying pixel areas
    - Adjusting detection parameters in real-time
    - Exporting measurements as pandas DataFrame
    
    The interface displays the image with object overlays and provides
    parameter controls for threshold adjustment. Clicking on objects
    calculates and displays their area.
    
    Args:
        image: The phenotypic.Image instance with detected objects.
        port: Port number for the Dash server. Defaults to 8050.
        height: Height of the image display in pixels. Defaults to 800.
        mode: Display mode - 'inline', 'external', or 'jupyterlab'. Defaults to 'external'.
        detector_type: Type of detector to use for parameter tuning ('otsu', 'watershed', etc.).
        
    Example:
        >>> import phenotypic as pht
        >>> image = pht.Image.imread('colony_plate.jpg')
        >>> image.detect_objects()  # Apply detection
        >>> analyzer = InteractiveMeasurementAnalyzer(image)
        >>> analyzer.run()
    """
    
    def __init__(
        self,
        image: Image,
        port: int = 8050,
        height: int = 800,
        mode: str = 'external',
        detector_type: str = 'otsu'
    ):
        super().__init__(image, port, height, mode)
        self.detector_type = detector_type
        self.selected_objects = set()
        self.measurements = pd.DataFrame()
        
    def setup_layout(self):
        """Create the Dash layout with image, controls, and measurement display."""
        from dash import html, dcc
        import plotly.graph_objects as go
        
        # Create initial figure
        initial_fig = self._create_figure()
        
        layout = html.Div([
            html.H1("Interactive Object Measurement Tool", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            html.Div([
                # Left panel: Image display
                html.Div([
                    dcc.Graph(
                        id='image-display',
                        figure=initial_fig,
                        style={'height': f'{self.height}px'}
                    )
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right panel: Controls and measurements
                html.Div([
                    html.H3("Controls"),
                    html.Label("Show Overlay:"),
                    dcc.Checklist(
                        id='overlay-toggle',
                        options=[{'label': 'Show Objects', 'value': 'show'}],
                        value=['show'],
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Overlay Transparency:"),
                    dcc.Slider(
                        id='alpha-slider',
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=0.3,
                        marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    
                    html.Hr(),
                    
                    html.H3("Measurements"),
                    html.Div(id='measurement-display', 
                            style={'maxHeight': '400px', 'overflowY': 'scroll'}),
                    
                    html.Hr(),
                    
                    html.Button('Clear Selection', id='clear-button', 
                               style={'marginTop': '10px', 'width': '100%'}),
                    html.Button('Export Measurements', id='export-button',
                               style={'marginTop': '10px', 'width': '100%'}),
                    html.Div(id='export-status', style={'marginTop': '10px'})
                    
                ], style={
                    'width': '28%', 
                    'display': 'inline-block', 
                    'verticalAlign': 'top',
                    'padding': '20px',
                    'marginLeft': '2%',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '5px'
                })
            ]),
            
            # Hidden div to store selected objects
            html.Div(id='selected-objects', style={'display': 'none'})
        ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
        
        return layout
    
    def create_callbacks(self):
        """Register Dash callbacks for interactivity."""
        from dash import Input, Output, State
        from dash.exceptions import PreventUpdate
        
        @self.app.callback(
            [Output('image-display', 'figure'),
             Output('measurement-display', 'children'),
             Output('selected-objects', 'children')],
            [Input('image-display', 'clickData'),
             Input('overlay-toggle', 'value'),
             Input('alpha-slider', 'value'),
             Input('clear-button', 'n_clicks')],
            [State('selected-objects', 'children')]
        )
        def update_display(click_data, overlay_toggle, alpha, clear_clicks, selected_str):
            """Update image and measurements based on user interactions."""
            from dash import callback_context
            from dash import html
            
            # Parse selected objects
            if selected_str:
                selected = set(map(int, selected_str.split(','))) if selected_str != '' else set()
            else:
                selected = set()
            
            # Check which input triggered the callback
            ctx = callback_context
            if not ctx.triggered:
                trigger_id = None
            else:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Handle clear button
            if trigger_id == 'clear-button' and clear_clicks:
                selected = set()
            
            # Handle click on image
            elif trigger_id == 'image-display' and click_data:
                # Get click coordinates
                point = click_data['points'][0]
                x = int(point['x'])
                y = int(point['y'])
                
                # Find which object was clicked
                objmap = self.image.objmap[:]
                if y < objmap.shape[0] and x < objmap.shape[1]:
                    label = objmap[y, x]
                    if label > 0:
                        if label in selected:
                            selected.remove(label)
                        else:
                            selected.add(label)
            
            # Create figure with current settings
            show_overlay = 'show' in overlay_toggle
            fig = self._create_figure(alpha=alpha, show_overlay=show_overlay, 
                                     highlighted_objects=selected)
            
            # Calculate measurements for selected objects
            measurement_div = self._create_measurement_display(selected)
            
            # Convert selected set to string for storage
            selected_str = ','.join(map(str, sorted(selected))) if selected else ''
            
            return fig, measurement_div, selected_str
        
        @self.app.callback(
            Output('export-status', 'children'),
            [Input('export-button', 'n_clicks')],
            [State('selected-objects', 'children')]
        )
        def export_measurements(n_clicks, selected_str):
            """Export measurements to CSV file."""
            if not n_clicks:
                raise PreventUpdate
            
            if not selected_str or selected_str == '':
                return html.Div("No objects selected", style={'color': 'red'})
            
            selected = set(map(int, selected_str.split(',')))
            df = self._calculate_measurements(selected)
            
            # Save to CSV
            filename = f'measurements_{self.image.name}.csv'
            df.to_csv(filename, index=False)
            
            return html.Div(f"Exported to {filename}", style={'color': 'green'})
    
    def _create_figure(self, alpha: float = 0.3, show_overlay: bool = True,
                      highlighted_objects: set = None):
        """Create Plotly figure with image and optional overlay."""
        import plotly.graph_objects as go
        import skimage as ski
        
        # Get base image (prefer gray for analysis)
        if self.image.gray.isempty():
            raise ValueError("Image has no grayscale data")
        
        base_image = self.image.gray[:]
        
        # Create overlay if requested
        if show_overlay and self.image.num_objects > 0:
            objmap = self.image.objmap[:]
            
            # Highlight selected objects
            if highlighted_objects:
                highlight_map = np.zeros_like(objmap)
                for label in highlighted_objects:
                    highlight_map[objmap == label] = label
                objmap = highlight_map
            
            display_image = self._create_overlay_image(base_image, objmap, alpha)
        else:
            display_image = self._convert_image_to_plotly(base_image)
        
        # Create figure
        fig = go.Figure()
        
        # Add image
        if display_image.ndim == 2:
            fig.add_trace(go.Heatmap(
                z=display_image,
                colorscale='gray',
                showscale=False,
                hovertemplate='x: %{x}<br>y: %{y}<br>intensity: %{z}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Image(z=display_image))
        
        # Update layout
        fig.update_layout(
            title=f"{self.image.name} - {self.image.num_objects} objects detected",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, scaleanchor='x'),
            hovermode='closest',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def _calculate_measurements(self, selected_labels: set) -> pd.DataFrame:
        """Calculate area measurements for selected objects."""
        if not selected_labels:
            return pd.DataFrame()
        
        # Get object map and mask
        objmap = self.image.objmap[:]
        objmask = self.image.objmask[:]
        
        measurements = []
        for label in sorted(selected_labels):
            # Calculate area (number of pixels in object)
            object_mask = (objmap == label)
            area = np.sum(object_mask)
            
            # Get centroid from regionprops
            props = self.image.objects.props
            obj_props = [p for p in props if p.label == label]
            if obj_props:
                centroid = obj_props[0].centroid
                centroid_r, centroid_c = centroid
            else:
                centroid_r, centroid_c = np.nan, np.nan
            
            measurements.append({
                'Object_Label': label,
                'Area_pixels': area,
                'Centroid_Row': centroid_r,
                'Centroid_Col': centroid_c
            })
        
        return pd.DataFrame(measurements)
    
    def _create_measurement_display(self, selected_labels: set):
        """Create HTML display of measurements."""
        from dash import html
        
        if not selected_labels:
            return html.Div("Click on objects to measure them", 
                          style={'fontStyle': 'italic', 'color': '#6c757d'})
        
        df = self._calculate_measurements(selected_labels)
        
        # Create table
        table_header = html.Thead(html.Tr([
            html.Th('Label'),
            html.Th('Area (px)'),
            html.Th('Centroid (r,c)')
        ]))
        
        table_rows = []
        for _, row in df.iterrows():
            table_rows.append(html.Tr([
                html.Td(int(row['Object_Label'])),
                html.Td(int(row['Area_pixels'])),
                html.Td(f"({row['Centroid_Row']:.1f}, {row['Centroid_Col']:.1f})")
            ]))
        
        table_body = html.Tbody(table_rows)
        
        table = html.Table(
            [table_header, table_body],
            style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'fontSize': '14px'
            }
        )
        
        # Add summary
        total_area = df['Area_pixels'].sum()
        summary = html.Div([
            html.Hr(),
            html.P(f"Total selected: {len(selected_labels)} objects"),
            html.P(f"Total area: {int(total_area)} pixels")
        ], style={'marginTop': '10px', 'fontWeight': 'bold'})
        
        return html.Div([table, summary])
    
    def update_image(self, *args, **kwargs):
        """Update the displayed image - delegates to _create_figure."""
        return self._create_figure(*args, **kwargs)

