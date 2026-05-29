"""Fitted parameters for the double-softplus growth model (with saturation ceiling)."""

from ._measurement_info import MeasurementInfo


class DOUBLE_SOFTPLUS_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "DoubleSoftplus"

    v = ("v", "The post-lag phase growth rate.")
    s0 = ("s0", "The initial size")
    lam = ("lambda", "The duration of the lag phase")
    alpha = ("alpha", "lag phase transition sharpness")
    smax = (
        "smax",
        "Carrying capacity used by the model. Either the user-provided "
        "scalar or the per-group observed maximum.",
    )
    beta = (
        "beta",
        "Saturation transition sharpness. Fitted per-group when a "
        "saturation shoulder is detected and ``beta`` is ``None`` at "
        "construction; held at the user-provided scalar (or the "
        "module default) when no shoulder is present.",
    )
    mode = (
        "mode",
        "Fit variant selected per-group: 'fixed_beta' (beta held at "
        "the user-provided or module-default value) or 'fitted_beta' "
        "(beta fitted as a 5th free parameter when a saturation "
        "shoulder is detected).",
    )
