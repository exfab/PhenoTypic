"""Fitted parameters for the double-softplus growth model (with saturation ceiling)."""

from ._measurement_info import Entry, MeasurementInfo


class DOUBLE_SOFTPLUS_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "DoubleSoftplus"

    v = Entry("v", "The post-lag phase growth rate.")
    s0 = Entry("s0", "The initial size")
    lam = Entry("lambda", "The duration of the lag phase")
    alpha = Entry("alpha", "lag phase transition sharpness")
    smax = Entry(
        "smax",
        "Carrying capacity used by the model. Either the user-provided "
        "scalar or the per-group observed maximum.",
    )
    beta = Entry(
        "beta",
        "Saturation transition sharpness. Fitted per-group when a "
        "saturation shoulder is detected and ``beta`` is ``None`` at "
        "construction; held at the user-provided scalar (or the "
        "module default) when no shoulder is present.",
    )
    mode = Entry(
        "mode",
        "Fit variant selected per-group: 'fixed_beta' (beta held at "
        "the user-provided or module-default value) or 'fitted_beta' "
        "(beta fitted as a 5th free parameter when a saturation "
        "shoulder is detected).",
    )
