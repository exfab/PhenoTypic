from __future__ import annotations

import numpy as np
from skimage.measure import regionprops

import neurospora_orientation_samples as notebook
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure._zone_segmentation import (
    compute_zone_segmentation,
    distance_from_point,
)
from phenotypic.util._orientation_field import orientation_field


def verify_real_notebook_colonies() -> None:
    """Print implemented radial-relative metrics for two real colonies."""
    segmented, _ = notebook.reproduce_notebook_segmentation()
    props = {
        int(prop.label): prop
        for prop in regionprops(
            segmented.objmap[:],
            intensity_image=segmented.gray[:].astype(np.float64, copy=False),
        )
    }
    operation = MeasureOrientationZones()
    for label, section_index in ((1116, 35), (626, 18)):
        prop = props[label]
        segmentation = compute_zone_segmentation(
            segmented,
            prop,
            params=operation._zone_params(),
        )
        tile, object_mask, centre = operation._resolve_tile(
            segmented,
            segmentation,
            prop,
            {label: section_index},
        )
        phi, coherence, gradient = orientation_field(
            tile,
            operation.sigma_d,
            operation.sigma_i,
        )
        distance = distance_from_point(tile.shape, centre)
        row = {}
        operation._fill_metrics(
            row,
            segmentation,
            object_mask,
            phi,
            coherence,
            gradient,
            distance,
            centre,
        )
        print(f"label={label} section={section_index}")
        for zone in ("Overall", "Dense", "Sparse"):
            tilt = row[f"OrientZones_RadialTilt-Mask-{zone}"]
            turning = row[f"OrientZones_OutwardTurning-Mask-{zone}"]
            support = row[f"OrientZones_RadialSectorSupport-Mask-{zone}"]
            print(
                f"  {zone}: tilt={np.degrees(tilt):.2f} deg "
                f"({tilt:.4f} rad), outward_turning={turning:.5f} rad/px, "
                f"sector_support={support:.3f}"
            )


if __name__ == "__main__":
    verify_real_notebook_colonies()
