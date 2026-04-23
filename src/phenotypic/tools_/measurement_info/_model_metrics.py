"""Generic fit-quality metrics shared by all ModelFitter subclasses."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class MODEL_METRICS(MeasurementInfo):
    """Generic fit-quality metrics and diagnostics shared by all ModelFitter subclasses.

    These columns are produced by any model fitter that wraps
    :func:`scipy.optimize.least_squares`, independent of the specific
    mathematical model. Subclass-specific fitted parameters live in the
    subclass's own MeasurementInfo class (e.g., ``LOG_GROWTH_MODEL``).
    """

    @classmethod
    def category(cls) -> str:
        return "ModelMetrics"

    # fit-quality metrics
    MAE = "MAE", "The mean absolute error"
    MSE = "MSE", "The mean squared error"
    RMSE = "RMSE", "The root mean squared error"
    R2 = "R2", "The coefficient of determination"

    # fit diagnostics
    NUM_SAMPLES = "NumSamples", "The number of samples used for model fitting"
    LOSS = "OptimizerLoss", "The loss of model fitting"
    STATUS = "OptimizerStatus", "The output of the optimizer status"
