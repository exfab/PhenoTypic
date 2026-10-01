# Custom Dashboard

Create interactive Plotly dashboards for plate analysis.

## Dashboard Pattern

PhenoTypic's interactive visualizations use Plotly `go.Figure` objects.
Custom dashboards follow the same pattern — build a figure, add traces
and layout, return it.

## Example: Colony Size Heatmap

```python
import plotly.graph_objects as go
import numpy as np

def colony_size_heatmap(image, nrows=8, ncols=12):
    """Create a heatmap of colony sizes across the grid."""
    info = image.grid.info()
    grid = np.zeros((nrows, ncols))

    for _, row in info.iterrows():
        r, c = int(row["RowNum"]), int(row["ColNum"])
        if 0 <= r < nrows and 0 <= c < ncols:
            grid[r, c] += 1  # or use Area, MeanIntensity, etc.

    fig = go.Figure(data=go.Heatmap(z=grid, colorscale="Viridis"))
    fig.update_layout(
        title="Colony count per grid section",
        xaxis_title="Column",
        yaxis_title="Row",
    )
    return fig
```

Follow the Okabe-Ito color palette defined in
`docs/style_guide/dashboards/CLAUDE.md` for consistency with
PhenoTypic's built-in dashboards.
