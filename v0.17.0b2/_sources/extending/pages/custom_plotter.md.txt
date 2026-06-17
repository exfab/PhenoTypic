# Custom Plotter

Create a custom visualization that registers with PhenoTypic's plot
accessor system.

## Plot Accessor Pattern

PhenoTypic's plot system uses registered plotters accessible via
`image.plot.<method_name>()`. Custom plotters integrate into this
system so they appear alongside built-in methods.

## Basic Pattern

```python
import matplotlib.pyplot as plt

def plot_intensity_profile(image, row=None, figsize=(10, 4)):
    """Plot the horizontal intensity profile across the plate center."""
    gray = image.gray[:]
    center_row = row or gray.shape[0] // 2
    profile = gray[center_row, :]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(profile)
    ax.set_xlabel("Column (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Intensity profile at row {center_row}")
    return fig, ax
```

Custom plotters should return `(fig, ax)` tuples for consistency with
the built-in methods. The caller is responsible for calling
`plt.close(fig)` to free memory.
