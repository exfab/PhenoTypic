"""Fitted parameters for the linear-softplus growth model."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class LINEAR_SOFTPLUS_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "LinearSoftplusModel"

    v = ("v", "The post-lag phase growth rate.")
    s0 = ("s0", "The initial size")
    lam = ("lambda", "The duration of the lag phase")
    alpha = ("alpha", "lag phase transition sharpness")
    smax = (
        "smax",
        "Carrying capacity used by the model. Provided, per-group "
        "observed max, or NaN when the group fit as unclamped.",
    )
    beta = (
        "beta",
        "Saturation transition sharpness. Fitted per-group when a "
        "saturation shoulder is detected and ``beta`` is ``None`` at "
        "construction; held at the user-provided scalar (or the "
        "module default) when no shoulder is present. NaN in "
        "unclamped mode.",
    )
    mode = (
        "mode",
        "Fit variant selected per-group: 'unclamped' (no saturation "
        "term), 'fixed_beta' (clamped with user-provided or default "
        "beta), or 'fitted_beta' (clamped with beta fitted as a 5th "
        "free parameter).",
    )
