from __future__ import annotations

from pathlib import Path

import numpy as np

from neurospora_orientation_samples import reproduce_notebook_segmentation
from phenotypic.measure import MeasureOrientationZones


OUTPUT_DIR = Path(
    "/Users/alex/.codex/visualizations/2026/07/15/"
    "019f6340-b68c-7a81-b738-983ed6ea1a27/orientation-real-image"
)
COLONIES = (("A", 1116, 35), ("B", 626, 18))


def render_nematic_bend_samples() -> None:
    """Render balanced multiscale bend panels for both reference colonies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segmented, _stats = reproduce_notebook_segmentation()
    for name, label, section_index in COLONIES:
        section = segmented.grid[section_index]
        objmap = section.objmap[:]
        section.objmap[:] = np.where(objmap == label, label, 0)
        operation = MeasureOrientationZones(quiver_block=24)
        operation.measure(section)
        figure = operation.fiber_bend_overlay(
            section,
            base_layer="detect_mat",
            scale_set="balanced",
        )
        output = OUTPUT_DIR / f"multiscale_fiber_bend_colony_{name.lower()}.png"
        figure.write_image(str(output), width=1800, height=900, scale=1)
        print(output, flush=True)


if __name__ == "__main__":
    render_nematic_bend_samples()
