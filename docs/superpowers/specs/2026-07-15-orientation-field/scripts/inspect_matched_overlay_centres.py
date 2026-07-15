from __future__ import annotations

import sys

import numpy as np
from skimage.measure import regionprops

sys.path.insert(0, "/private/tmp")

from render_matched_ring_comparison import extract_profiles, label_centroid
from render_twok_reconnected_orientation import (
    isolated_global_crop,
    load_twok_detection,
)
from phenotypic.measure import MeasureOrientationZones


def inspect_matched_overlay_centres() -> None:
    detected, _old = load_twok_detection()
    for colony, label in (("R3C4", 24), ("R4C6", 36)):
        full_centroid = label_centroid(detected, label)
        rows, cols = np.nonzero(np.asarray(detected.objmap[:]) == label)
        section = isolated_global_crop(detected, label, full_centroid)
        section_mask = np.asarray(section.objmap[:]) == label
        section_rows, section_cols = np.nonzero(section_mask)
        prop = regionprops(section_mask.astype(np.uint8))[0]
        profiles = extract_profiles(
            section,
            MeasureOrientationZones(
                radial_ring_width=8.0,
                long_range_lag=32.0,
                quiver_block=24,
            ),
        )
        print(
            {
                "colony": colony,
                "label": label,
                "full_shape": detected.shape,
                "full_centroid": full_centroid,
                "full_bounds": (
                    int(rows.min()),
                    int(rows.max()),
                    int(cols.min()),
                    int(cols.max()),
                ),
                "section_shape": section.shape,
                "section_geometric_centroid": tuple(prop.centroid),
                "section_bounds": (
                    int(section_rows.min()),
                    int(section_rows.max()),
                    int(section_cols.min()),
                    int(section_cols.max()),
                ),
                "inferred_inoculum_centre": profiles["centre"],
                "outer_radius": profiles["outer_radius"],
            }
        )


if __name__ == "__main__":
    inspect_matched_overlay_centres()
