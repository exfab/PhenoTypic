from render_twok_reconnected_orientation import (
    isolated_global_crop,
    load_twok_detection,
)
from phenotypic.measure import MeasureOrientationZones
from skimage.measure import regionprops
import numpy as np

detected, _ = load_twok_detection()
props = {
    prop.label: prop for prop in regionprops(np.asarray(detected.objmap[:]))
}
for label in (37, 24, 10):
    centroid = tuple(float(value) for value in props[label].centroid)
    section = isolated_global_crop(detected, label, centroid)
    operation = MeasureOrientationZones()
    measurements = operation.measure(section)
    rasters, peaks, stride = operation._bend_scale_rasters(
        section,
        section.shape[:2],
        (4.0, 8.0, 16.0),
    )
    print(
        label,
        section.shape,
        int(np.count_nonzero(section.objmap[:])),
        measurements[["Object_Label"]].to_dict("records"),
        operation._cache.get(label, {}).get("radii"),
        peaks,
        [int(np.isfinite(raster).sum()) for raster in rasters],
    )
