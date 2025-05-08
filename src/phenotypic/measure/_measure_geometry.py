from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np

from phenotypic.abstract import FeatureMeasure

from phenotypic.util.constants_ import GEOM_LABELS as C

# TODO: Add more measurements
class MeasureGeometry(FeatureMeasure):
    """Calculates various geometric measures of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the geometric measures of the objects in the image.

    Notes:
        Area: The sum of the individual pixel's in the object's footprint
        Perimeter: The length of the object's boundary
        Circularity: Calculated as :math:`\frac{4\pi*Area}{Perimeter^2}
        ConvexArea: The area of the convex hull of the object

    References:
        1. D. R. Stirling, M. J. Swain-Bowden, A. M. Lucas, A. E. Carpenter, B. A. Cimini, and A. Goodman,
            “CellProfiler 4: improvements in speed, utility and usability,” BMC Bioinformatics, vol. 22, no. 1, p. 433, Sep. 2021, doi: 10.1186/s12859-021-04344-9.
        2. “Shape factor (image analysis and microscopy),” Wikipedia. Oct. 09, 2021. Accessed: Apr. 08, 2025. [Online]. Available: https://en.wikipedia.org/w/index.php?title=Shape_factor_(image_analysis_and_microscopy)&oldid=1048998776

    """

    def _operate(self, image: Image) -> pd.DataFrame:
        measurements = {
            str(C.AREA): [],
            str(C.PERIMETER): [],
            str(C.CIRCULARITY): [],
            str(C.CONVEX_AREA): [],
        }
        for obj_props in image.objects.props:
            measurements[str(C.AREA)].append(obj_props.area)
            measurements[str(C.PERIMETER)].append(obj_props.perimeter)

            circularity = (4*np.pi*obj_props.area) / (obj_props.perimeter**2)
            measurements[str(C.CIRCULARITY)].append(circularity)

            convex_hull = ConvexHull(obj_props.coords)
            measurements[str(C.CONVEX_AREA)].append(convex_hull.area)

        return pd.DataFrame(measurements, index=image.objects.get_labels_series())
