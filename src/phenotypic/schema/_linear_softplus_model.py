"""Fitted parameters for the linear-softplus growth model (no saturation)."""

from ._measurement_info import Entry, MeasurementInfo


class LINEAR_SOFTPLUS_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "LinearSoftplus"

    v = Entry("v", "The post-lag phase growth rate.")
    s0 = Entry("s0", "The initial size")
    lam = Entry("lambda", "The duration of the lag phase")
    alpha = Entry("alpha", "lag phase transition sharpness")
