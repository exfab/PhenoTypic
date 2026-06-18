"""The labels and descriptions of the size measurements."""

from ._measurement_info import Entry
from ._tiers import DirectPhenotype


class SIZE(DirectPhenotype):
    """Measure colony area and integrated intensity as lightweight size proxies.

    Extract two fundamental size metrics per detected colony: pixel area
    (biomass extent) and integrated grayscale intensity (total brightness,
    a proxy for optical density). This is a convenience class for rapid
    size assessment without the overhead of full shape or intensity
    statistical analysis.
    """

    @classmethod
    def category(cls):
        return "Size"

    AREA = Entry(
        "Area",
        "Total number of pixels occupied by the microbial colony."
        "Larger areas typically indicate more robust growth or longer incubation times.",
    )
    INTEGRATED_INTENSITY = Entry(
        "IntegratedIntensity",
        r"The sum of the object's grayscale pixels. Calculated as "
        r":math:`\sum{\text{pixel values}} \times \text{area}`.",
    )
