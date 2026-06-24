"""Fitted parameters for the linear-softplus growth model (no saturation)."""

from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class LINEAR_LAG_MODEL(DerivedMeasure):
    @classmethod
    def category(cls) -> str:
        return "LinearLagModel"

    v = Entry("v", "The post-lag phase growth rate.",
              bio_desc="The post-lag phase growth rate "
                       "using the target metric (usually radius)",
              tier=1, derivation_type="parameterization", derives_from="SIZE")
    s0 = Entry("s0", "The initial value of the target metric",
               bio_desc="The initial size",
               tier=1, derivation_type="parameterization", derives_from="SIZE")
    lam = Entry("lambda", "The duration of the lag phase",
                tier=1, derivation_type="parameterization", derives_from="SIZE")
    alpha = Entry("alpha", "lag phase transition sharpness",
                  tier=2, derivation_type="parameterization", derives_from="SIZE")
