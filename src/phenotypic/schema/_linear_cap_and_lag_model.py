"""Fitted parameters for the double-softplus growth model (with saturation ceiling)."""

from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class LINEAR_CAP_AND_LAG_MODEL(DerivedMeasure):
    @classmethod
    def category(cls) -> str:
        return "LinearCapAndLagModel"

    @classmethod
    def header_scheme(cls) -> str:
        return "metric_qualified"

    v = Entry("v", "The post-lag phase growth rate.",
              tier=1, derivation_type="parameterization", derives_from="SIZE")
    s0 = Entry("s0", "The initial size",
               tier=1, derivation_type="parameterization", derives_from="SIZE")
    lam = Entry("lambda", "The duration of the lag phase",
                tier=1, derivation_type="parameterization", derives_from="SIZE")
    alpha = Entry("alpha", "lag phase transition sharpness",
                  tier=2, derivation_type="parameterization", derives_from="SIZE")
    smax = Entry(
        "smax",
        "Carrying capacity used by the model. Either the user-provided "
        "scalar or the per-group observed maximum.",
        tier=1, derivation_type="parameterization", derives_from="SIZE",
    )
    beta = Entry(
        "beta",
        "Saturation transition sharpness. Fitted per-group when a "
        "saturation shoulder is detected and ``beta`` is ``None`` at "
        "construction; held at the user-provided scalar (or the "
        "module default) when no shoulder is present.",
        tier=2, derivation_type="parameterization", derives_from="SIZE",
    )
    mode = Entry(
        "mode",
        "Fit variant selected per-group: 'fixed_beta' (beta held at "
        "the user-provided or module-default value) or 'fitted_beta' "
        "(beta fitted as a 5th free parameter when a saturation "
        "shoulder is detected).",
        derivation_type="diagnostic",
    )
