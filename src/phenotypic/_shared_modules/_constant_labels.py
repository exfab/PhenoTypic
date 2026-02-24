from ._measurement_info import MeasurementInfo


class ConstantLabels(MeasurementInfo):
    """Base class for constant labels in phenotypic. This class is to distinguish
    between ConstantLabels and MeasurementInfo usages."""

    def __init__(self, label: str, value: float):
        super().__init__(label, value)
