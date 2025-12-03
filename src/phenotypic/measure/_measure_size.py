from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic.tools.constants_ import OBJECT

if TYPE_CHECKING:
    from phenotypic import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import MeasurementInfo, MeasureFeatures


class SIZE(MeasurementInfo):
    """The labels and descriptions of the size measurements."""

    @classmethod
    def category(cls):
        return "Size"

    AREA = (
        "Area",
        "Total number of pixels occupied by the microbial colony."
        "Larger areas typically indicate more robust growth or longer incubation times.",
    )
    INTEGRATED_INTENSITY = (
        "IntegratedIntensity",
        r"The sum of the object\'s grayscale pixels. Calculated as"
        r"$\sum{pixel values}*area$",
    )


class MeasureSize(MeasureFeatures):
    """Measure basic size metrics of detected microbial colonies.

    This class extracts two fundamental size metrics: Area (colony pixel count) and Integrated Intensity
    (total grayscale brightness). These measurements are extracted from shape and intensity analyses but
    are provided as a lightweight convenience class for quick size assessment without full morphological
    or intensity statistical analysis.

    **Intuition:** Colony size and brightness are primary phenotypic traits in high-throughput screening.
    Area directly reflects biomass and growth extent on the agar surface. Integrated Intensity approximates
    optical density without requiring cell suspension, making it useful for time-course growth tracking.
    Together, these metrics enable rapid size-based sorting and quality filtering.

    **Use cases (agar plates):**
    - Quickly assess colony size distribution on a plate for quality control.
    - Track colony growth kinetics over time via area measurements at multiple time points.
    - Estimate total biomass via integrated intensity (sum of all pixel values).
    - Filter colonies by minimum size threshold to exclude aborted or contaminating growth.
    - Rank colonies by size for automated picking or downstream processing.

    **Caveats:**
    - Area is measured in pixels; it must be scaled by pixel-to-micron conversion factors if physical
      dimensions are needed.
    - Integrated intensity is sensitive to imaging conditions (lighting, exposure, camera gain); absolute
      values are not comparable across different imaging setups without normalization.
    - Both metrics depend on accurate segmentation; over- or under-segmentation distorts size estimates.
    - For size-normalized intensity analysis (e.g., intensity per unit area), use MeasureIntensity.mean
      instead of integrated intensity, or divide integrated intensity by area.
    - Area does not distinguish between a single solid colony and multiple touching fragments; use
      MeasureGridSpread or object count metrics for multi-object detection in grid sections.

    Returns:
        pd.DataFrame: Object-level size measurements with columns:
            - Label: Unique object identifier.
            - Area: Number of pixels occupied by the colony.
            - IntegratedIntensity: Sum of grayscale pixel values in the colony (proxy for biomass/OD).

    Examples:
        .. dropdown:: Quick size assessment and filtering

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import OtsuDetector
                from phenotypic.measure import MeasureSize

                # Load and detect colonies
                image = Image.from_image_path("colony_plate.jpg")
                detector = OtsuDetector()
                image = detector.operate(image)

                # Measure size
                sizer = MeasureSize()
                sizes = sizer.operate(image)

                # Filter colonies by size (exclude very small or very large)
                min_area, max_area = 50, 5000  # pixels
                good_colonies = sizes[
                    (sizes['Size_Area'] >= min_area) &
                    (sizes['Size_Area'] <= max_area)
                ]
                print(f"Colonies within size range: {len(good_colonies)}/{len(sizes)}")

        .. dropdown:: Track colony growth over time

            .. code-block:: python

                # Load images from multiple time points
                import pandas as pd
                from phenotypic import Image
                from phenotypic.detect import OtsuDetector
                from phenotypic.measure import MeasureSize

                detector = OtsuDetector()
                sizer = MeasureSize()

                # Simulate time-series measurements
                growth_data = []
                for timepoint_h in [0, 6, 12, 24, 48]:
                    img_path = f"plate_t{timepoint_h}h.jpg"
                    image = Image.from_image_path(img_path)
                    image = detector.operate(image)
                    sizes = sizer.operate(image)
                    sizes['TimePoint_h'] = timepoint_h
                    growth_data.append(sizes)

                # Combine and analyze
                growth_df = pd.concat(growth_data, ignore_index=True)
                # Track individual colonies and compute growth rate (simplified)
                avg_area = growth_df.groupby('TimePoint_h')['Size_Area'].mean()
                print("Average colony area over time:")
                print(avg_area)
    """

    def _operate(self, image: Image) -> pd.DataFrame:
        # Create empty numpy arrays to store measurements
        measurements = {
            str(feature): np.zeros(shape=image.num_objects)
            for feature in SIZE
            if feature != SIZE.CATEGORY
        }

        # Calculate integrated intensity using the sum calculation method from base class
        intensity_matrix = image.gray[:].copy()
        objmap = image.objmap[:].copy()

        measurements[SIZE.AREA] = self._calculate_sum(
            array=image.objmask[:], objmap=objmap
        )
        measurements[SIZE.INTEGRATED_INTENSITY] = self._calculate_sum(
            array=intensity_matrix, objmap=objmap
        )

        measurements = pd.DataFrame(measurements)
        measurements.insert(
            loc=0, column=OBJECT.LABEL, value=image.objects.labels2series()
        )
        return measurements


MeasureSize.__doc__ = SIZE.append_rst_to_doc(MeasureSize)
