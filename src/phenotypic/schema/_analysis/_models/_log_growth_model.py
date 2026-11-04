"""Fitted parameters and bounds for the logistic growth model."""

from ..._base._measurement_info import Entry
from ..._base._units import UNITS
from ..._base._tiers import DerivedMeasure


class LOG_GROWTH_MODEL(DerivedMeasure):
    """Fitted parameters and bounds of the logistic growth model.

    Output columns are **metric-qualified**: each header is
    ``LogGrowthModel_<metric>_<parameter>``, where ``<metric>`` records the
    measurement the model was fit on (``self.on`` with its metric-family prefix
    stripped, e.g. ``Size_Area`` → ``Area``). For example, fitting on
    ``Size_Area`` emits ``LogGrowthModel_Area_r`` (intrinsic growth rate) and
    ``LogGrowthModel_Area_µmax`` (maximum absolute growth rate). The labels
    below are the ``<parameter>`` segment; the ``<metric>`` infix is filled in
    at fit time.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "LogGrowthModel"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.OF_INPUT

    @classmethod
    def header_scheme(cls) -> str:
        return "metric_qualified"

    R_FIT = Entry("r", "The intrinsic growth rate",
                  tier=1, derivation_type="parameterization", derives_from="SIZE", unit=UNITS.PER_TIME)
    K_FIT = Entry("K", "The carrying capacity",
                  tier=1, derivation_type="parameterization", derives_from="SIZE")
    N0_FIT = Entry("N0", "The initial number of the colony size metric being fitted",
                   tier=1, derivation_type="parameterization", derives_from="SIZE")
    LAM = Entry(
        "lambda",
        "Regularization weight λ that penalizes a large maximum growth rate "
        "(µmax = K·r/4) and a large initial value N0 during fitting.",
        derivation_type="diagnostic",
        unit=UNITS.UNITLESS,
    )
    BETA = Entry(
        "beta",
        (
            "Penalty weight β on how far the fitted carrying capacity K lies "
            "from the value measured at the last time point, relative to that "
            "value."
        ),
        derivation_type="diagnostic",
        unit=UNITS.UNITLESS,
    )
    GROWTH_RATE = Entry(
        "µmax",
        "Maximum growth rate of the fitted logistic curve: its slope where "
        "the curve reaches half the carrying capacity K. It is calculated as "
        "(K*r)/4.",
        tier=1,
        derivation_type="parameterization",
        derives_from="SIZE",
        unit=UNITS.OF_INPUT_PER_TIME,
    )
    K_MAX = Entry("Kmax", "The upper bound of the carrying capacity for model fitting",
                  derivation_type="diagnostic")
