"""Fitted parameters and bounds for the logistic growth model."""

from ._measurement_info import Entry, MeasurementInfo


class LOG_GROWTH_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "LogGrowthModel"

    R_FIT = Entry("r", "The intrinsic growth rate")
    K_FIT = Entry("K", "The carrying capacity")
    N0_FIT = Entry("N0", "The initial number of the colony size metric being fitted")
    LAM = Entry(
        "lambda",
        "The regularization factor applied to the max specific growth rate "
        "and initial population size",
    )
    BETA = Entry(
        "beta",
        (
            "The penalty factor applied to relative difference of "
            "the carrying capacity from the largest measurement"
        ),
    )
    GROWTH_RATE = Entry("µmax", "The growth rate of the colony calculated as (K*r)/4")
    K_MAX = Entry("Kmax", "The upper bound of the carrying capacity for model fitting")
