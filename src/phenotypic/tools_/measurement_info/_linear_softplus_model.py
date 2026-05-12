"""Fitted parameters for the linear-softplus growth model (no saturation)."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class LINEAR_SOFTPLUS_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "LinearSoftplus"

    v = ("v", "The post-lag phase growth rate.")
    s0 = ("s0", "The initial size")
    lam = ("lambda", "The duration of the lag phase")
    alpha = ("alpha", "lag phase transition sharpness")
