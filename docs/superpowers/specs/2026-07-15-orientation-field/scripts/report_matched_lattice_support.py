from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/private/tmp")

from render_matched_ring_comparison import (
    extract_profiles,
    label_centroid,
)
from render_twok_reconnected_orientation import (
    isolated_global_crop,
    load_twok_detection,
)
from phenotypic.measure import MeasureOrientationZones


def report_matched_lattice_support() -> None:
    detected, _old = load_twok_detection()
    for colony, label in (("R3C4", 24), ("R4C6", 36)):
        section = isolated_global_crop(
            detected,
            label,
            label_centroid(detected, label),
        )
        profiles = extract_profiles(
            section,
            MeasureOrientationZones(
                radial_ring_width=8.0,
                long_range_lag=32.0,
                quiver_block=24,
            ),
        )
        reliable = np.isfinite(profiles["fiber_resultant"])
        matched = np.isfinite(profiles["matched_fiber"])
        total = reliable.size
        print(
            colony,
            {
                "rings_x_sectors": reliable.shape,
                "total_cells": total,
                "underlying_reliable": int(reliable.sum()),
                "underlying_reliable_fraction": float(reliable.mean()),
                "matched": int(matched.sum()),
                "matched_fraction": float(matched.mean()),
                "reliable_but_unmatched": int((reliable & ~matched).sum()),
                "unsupported_input": int((~reliable).sum()),
            },
        )


if __name__ == "__main__":
    report_matched_lattice_support()
