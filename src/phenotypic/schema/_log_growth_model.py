"""Fitted parameters and bounds for the logistic growth model."""

from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class LOG_GROWTH_MODEL(DerivedMeasure):
    @classmethod
    def category(cls) -> str:
        return "LogGrowthModel"

    @classmethod
    def header_scheme(cls) -> str:
        return "metric_qualified"

    R_FIT = Entry("r", "The intrinsic growth rate",
                  tier=1, derivation_type="parameterization", derives_from="SIZE")
    K_FIT = Entry("K", "The carrying capacity",
                  tier=1, derivation_type="parameterization", derives_from="SIZE")
    N0_FIT = Entry("N0", "The initial number of the colony size metric being fitted",
                   tier=1, derivation_type="parameterization", derives_from="SIZE")
    LAM = Entry(
        "lambda",
        "The regularization factor applied to the max specific growth rate "
        "and initial population size",
        derivation_type="diagnostic",
    )
    BETA = Entry(
        "beta",
        (
            "The penalty factor applied to relative difference of "
            "the carrying capacity from the largest measurement"
        ),
        derivation_type="diagnostic",
    )
    GROWTH_RATE = Entry("µmax", "The growth rate of the colony calculated as (K*r)/4",
                        tier=1, derivation_type="parameterization", derives_from="SIZE")
    K_MAX = Entry("Kmax", "The upper bound of the carrying capacity for model fitting",
                  derivation_type="diagnostic")
