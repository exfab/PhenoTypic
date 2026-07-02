"""Fitted parameters for the linear-softplus growth model (no saturation)."""

from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class LINEAR_LAG_MODEL(DerivedMeasure):
    """Fitted parameters of the linear-softplus lag model (no saturation).

    Output columns are **metric-qualified**: each header is
    ``LinearLagModel_<metric>_<parameter>``, where ``<metric>`` records the
    measurement the model was fit on (``self.on`` with its category prefix
    stripped, e.g. ``Shape_Area`` → ``Area``). For example, fitting on
    ``Shape_Area`` emits ``LinearLagModel_Area_v`` (post-lag growth rate) and
    ``LinearLagModel_Area_s0`` (initial size). The labels below are the
    ``<parameter>`` segment; the ``<metric>`` infix is filled in at fit time.
    """

    @classmethod
    def category(cls) -> str:
        return "LinearLagModel"

    @classmethod
    def header_scheme(cls) -> str:
        return "metric_qualified"

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
