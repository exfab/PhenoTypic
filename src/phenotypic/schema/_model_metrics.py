"""Generic fit-quality metrics shared by all ModelFitter subclasses."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class MODEL_METRICS(QualityInfo):
    """Generic fit-quality metrics and diagnostics shared by all ModelFitter subclasses.

    These columns are produced by any model fitter that wraps
    :func:`scipy.optimize.least_squares`, independent of the specific
    mathematical model. Subclass-specific fitted parameters live in the
    subclass's own MeasurementInfo class (e.g., ``LOG_GROWTH_MODEL``).
    """

    @classmethod
    def category(cls) -> str:
        return "ModelMetrics"

    @classmethod
    def header_scheme(cls) -> str:
        return "metric_qualified"

    # fit-quality metrics
    MAE = Entry("MAE", "The mean absolute error")
    MSE = Entry("MSE", "The mean squared error")
    RMSE = Entry("RMSE", "The root mean squared error")
    R2 = Entry("R2", "The coefficient of determination")

    # fit diagnostics
    NUM_SAMPLES = Entry("NumSamples", "The number of samples used for model fitting")
    LOSS = Entry("OptimizerLoss", "The loss of model fitting")
    STATUS = Entry("OptimizerStatus", "The output of the optimizer status")
