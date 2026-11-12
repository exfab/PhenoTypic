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
    MAE = Entry(
        "MAE",
        "Mean absolute difference between the fitted curve and the data "
        "points it was fitted to.",
    )
    MSE = Entry(
        "MSE",
        "Mean squared difference between the fitted curve and the data points "
        "it was fitted to.",
        unit=UNITS.OF_INPUT_SQUARED,
    )
    RMSE = Entry(
        "RMSE",
        "Typical distance between the fitted curve and the data points it was "
        "fitted to, as the square root of the mean squared difference.",
    )
    R2 = Entry(
        "R2",
        "Share of the variation in the fitted data points that the curve "
        "explains: 1 is a perfect fit, and 0 or below is no better than the "
        "points' mean. It is the coefficient of determination, and NaN for a "
        "group with two or fewer points.",
        unit=UNITS.UNITLESS,
    )

    # fit diagnostics
    NUM_SAMPLES = Entry(
        "NumSamples",
        "Number of data points the model was fitted to, after any points the "
        "fitter drops (such as post-saturation time points).",
        unit=UNITS.UNITLESS,
    )
    # SEM-weighted (analysis/abc_/_linear_softplus_base.py:362), so its units
    # vary by group and no conversion is ever valid: never OF_INPUT_SQUARED.
    LOSS = Entry(
        "OptimizerLoss",
        "Final value of the cost that the fitting optimizer minimized. It is "
        "the cost of scipy's least_squares: half the sum of the robust loss ρ "
        "applied to each squared residual, including any weighting or "
        "regularization terms the model adds.",
        unit=UNITS.NOT_A_QUANTITY,
    )
    STATUS = Entry(
        "OptimizerStatus",
        "Integer code from scipy's least_squares saying why the fit stopped: "
        "0 when it reached the evaluation limit and 1 to 4 when a convergence "
        "tolerance was met. It is NaN when the fit failed.",
        unit=UNITS.NOT_A_QUANTITY,
    )
