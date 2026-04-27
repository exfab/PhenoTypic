"""The labels and descriptions of the size measurements."""

from phenotypic.abc_._measurement_info import MeasurementInfo


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
