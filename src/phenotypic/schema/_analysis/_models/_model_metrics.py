"""Generic fit-quality metrics shared by all ModelFitter subclasses."""

from ..._base._measurement_info import Entry
from ..._base._units import UNITS
from ..._base._tiers import QualityInfo


class MODEL_METRICS(QualityInfo):
    """Generic fit-quality metrics and diagnostics shared by all ModelFitter subclasses.

    These columns are produced by any model fitter that wraps
    :func:`scipy.optimize.least_squares`, independent of the specific
    mathematical model. Subclass-specific fitted parameters live in the
    subclass's own MeasurementInfo class (e.g., ``LOG_GROWTH_MODEL``).
    """

    @classmethod
    def metric_family(cls) -> str:
        return "ModelMetrics"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.OF_INPUT

    @classmethod
    def header_scheme(cls) -> str:
        return "metric_qualified"

    # fit-quality metrics
    MAE = Entry("MAE", "The mean absolute error")
    MSE = Entry("MSE", "The mean squared error", unit=UNITS.OF_INPUT_SQUARED)
    RMSE = Entry("RMSE", "The root mean squared error")
    R2 = Entry("R2", "The coefficient of determination", unit=UNITS.UNITLESS)

    # fit diagnostics
    NUM_SAMPLES = Entry("NumSamples", "The number of samples used for model fitting", unit=UNITS.UNITLESS)
    LOSS = Entry("OptimizerLoss", "The loss of model fitting", unit=UNITS.NOT_A_QUANTITY)
    STATUS = Entry("OptimizerStatus", "The output of the optimizer status", unit=UNITS.NOT_A_QUANTITY)
