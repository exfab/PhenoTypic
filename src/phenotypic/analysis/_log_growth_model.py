from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from phenotypic.analysis.abc_ import ModelFitter
from phenotypic.analysis.abc_._model_fitter import LossKind
from phenotypic.tools_ import ColumnRef, ColumnRefList
from phenotypic.tools_.measurement_info import LOG_GROWTH_MODEL, MODEL_METRICS


class LogGrowthModel(ModelFitter):
    r"""Logistic-growth model fitter with regularized least-squares objective.

    Logistic Kinetics Model:

        .. math::

           N(t) = \frac{K}{1 + \frac{K - N_0}{N_0} e^{-rt}}

        :math:`N_t`: population size at time :math:`t`

        :math:`N_0`: initial population size at time :math:`t`

        :math:`r`: growth rate

        :math:`K`: carrying capacity (maximum population size)

        From this we derive:

        .. math::

           \mu_{\max} = \frac{K r}{4}

        :math:`\mu_{\max}`: maximum specific growth rate


    Loss Function:

        To solve for the parameters, we use the following loss function with the
        SciPy linear least-squares solver:

        .. math::

           J(K, N_0, r) =
           \frac{1}{n}\sum_{i=1}^{n}
           \frac{1}{2}\left(f_{K,N_0,r}(t^{(i)}) - N_t^{(i)}\right)^2
           + \lambda\left(\left(\frac{dN}{dt}\right)^2 + N_0^2\right)
           + \beta \frac{\lvert K - \max(N_t) \rvert}{N_t}

        :math:`\lambda`: regularization term for growth rate and initial population size

        :math:`\beta`: penalty term for deviations in carrying capacity relative to
            the largest measurement


    Attributes:
        lam (float): The penalty factor applied to growth rates.
        beta (float): The maximum penalty factor applied to the carrying
            capacity.
        Kmax_label (str | None): The column name for the maximum carrying capacity
            values, if provided.
    """

    _measurement_infoclass = LOG_GROWTH_MODEL

    def __init__(
            self,
            on: ColumnRef,
            groupby: ColumnRefList,
            time_label: ColumnRef = "Metadata_Time",
            agg_func: Callable | str | list | dict | None = "mean",
            lam: float = 1.2,
            beta: float = 2,
            Kmax_label: ColumnRef | None = None,
            loss: LossKind = "huber",
            f_scale: float = 1.0,
            verbose: bool = False,
            n_jobs: int = 1,
    ):
        """Initialize the log-growth fitter.

        Args:
            on: Target column (population-size measurement) to fit.
            groupby: Columns defining the per-fit grouping structure.
            time_label: Column name representing time. Defaults to
                ``"Metadata_Time"``.
            agg_func: Aggregation function fed to ``DataFrame.groupby.agg()``.
                Defaults to ``"mean"``.
            lam: Regularization factor applied to the maximum specific
                growth rate and initial population size. Defaults to 1.2.
            beta: Penalty factor applied to the relative difference
                between ``K`` and the largest observed measurement.
                Defaults to 2.
            Kmax_label: Optional column providing a per-group upper bound
                on ``K``. When omitted, the observed maximum of ``on`` is
                used.
            loss: Loss method passed through to
                :func:`scipy.optimize.least_squares`. One of
                ``"linear"``, ``"soft_l1"``, ``"huber"``, ``"cauchy"``,
                ``"arctan"``. Defaults to ``"huber"`` — standard
                least-squares on inliers, linear downweighting past
                ``f_scale`` for outlier timepoints. Pass ``"linear"``
                for the classical sum-of-squared-residuals loss.
            f_scale: Soft margin between inlier and outlier residuals
                handed to :func:`scipy.optimize.least_squares`. No
                effect when ``loss="linear"``. Must be positive;
                defaults to ``1.0``.
            verbose: If ``True``, enables the optimizer's verbose output.
            n_jobs: Number of parallel workers for per-group fits.
        """
        super().__init__(
            on=on,
            groupby=groupby,
            time_label=time_label,
            agg_func=agg_func,
            n_jobs=n_jobs,
            loss=loss,
            f_scale=f_scale,
            verbose=verbose,
        )
        self.lam = lam
        self.beta = beta
        self.Kmax_label = Kmax_label

    # ------------------------------------------------------------------ #
    # Model math
    # ------------------------------------------------------------------ #
    @staticmethod
    def model_func(t: np.ndarray | float, r: float, K: float, N0: float):
        r"""Logistic growth model evaluated at ``t``.

        .. math:: N(t) = K / \left(1 + \frac{K - N_0}{N_0} e^{-rt}\right)

        Args:
            t: Time at which the population is evaluated (scalar or array).
            r: Growth rate.
            K: Carrying capacity.
            N0: Initial population size at ``t = 0``.

        Returns:
            Population size at ``t``. Scalar when ``t`` is scalar,
            otherwise an array.
        """
        a = (K - N0) / N0
        return K / (1 + a * np.exp(-r * t))

    @staticmethod
    def _loss_func(params, t, y, lam, beta):  # type: ignore[override]
        r"""Regularized residuals for :func:`scipy.optimize.least_squares`.

        .. math::

           J(K, N_0, r) =
           \frac{1}{n}\sum_{i=1}^{n}
           \left(f_{K,N_0,r}(t^{(i)}) - N_t^{(i)}\right)^2
           + \lambda\left(\left(\frac{dN}{dt}\right)^2 + N_0^2\right)
           + \beta \frac{\lvert K - N_{n} \rvert}{N_{n}}

        Args:
            params: ``[r, K, N0]`` — candidate parameter vector.
            t: Time points for the observations.
            y: Observed population sizes (list, ndarray, or pandas Series).
            lam: Regularization weight for ``dN/dt`` and ``N0``.
            beta: Scaling factor for the ``K`` vs. observed-max penalty.

        Returns:
            Flat residual vector consumed by
            :func:`scipy.optimize.least_squares`.
        """
        r, K, N0 = params

        residuals = y - LogGrowthModel.model_func(t=t, r=r, K=K, N0=N0)

        dN_dt = r * K / 4
        reg_term = np.sqrt(lam) * np.array([dN_dt, N0])

        if hasattr(y, "values"):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Last-timepoint value as the best proxy for carrying capacity.
        y_max_observed = y_array[np.argmax(t)]

        epsilon = 1e-8 * np.median(np.abs(y_array[y_array > 0]))
        if epsilon == 0 or np.isnan(epsilon):
            epsilon = 1e-8

        K_penalty_weight = np.sqrt(beta)
        K_penalty = (
                K_penalty_weight * np.abs(K - y_max_observed)
                / (y_max_observed + epsilon)
        )

        return np.hstack([residuals, reg_term, [K_penalty]])

    # ------------------------------------------------------------------ #
    # Fit hooks
    # ------------------------------------------------------------------ #
    def _kmax_for(self, group: pd.DataFrame) -> float:
        if self.Kmax_label is None:
            return group[self.on].max()
        k_max = group[self.Kmax_label].max()
        if pd.isna(k_max):
            return group[self.on].max() + 1
        return k_max

    def _initial_guess(self, group: pd.DataFrame) -> List[float]:
        return [1e-5, group[self.on].max(), 0]

    def _bounds(
            self, group: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        size_data = group[self.on]

        r_min, r_max = 1e-5, np.inf
        N0_min, N0_max = 0, size_data.min()
        # Ensure upper > lower per least_squares contract.
        if N0_max <= N0_min:
            N0_max = N0_min + 1

        K_min = 0
        K_max = self._kmax_for(group)

        return [r_min, K_min, N0_min], [r_max, K_max, N0_max]

    def _unpack_params(
            self, x: np.ndarray, group: pd.DataFrame
    ) -> Dict[Any, float]:
        r, K, N0 = float(x[0]), float(x[1]), float(x[2])
        return {
            LOG_GROWTH_MODEL.R_FIT: r,
            LOG_GROWTH_MODEL.K_FIT: K,
            LOG_GROWTH_MODEL.N0_FIT: N0,
            LOG_GROWTH_MODEL.GROWTH_RATE: (r * K) / 4,
            LOG_GROWTH_MODEL.K_MAX: self._kmax_for(group),
        }

    def _predict_kwargs(self, row) -> Dict[str, float]:
        return {
            "r": row[LOG_GROWTH_MODEL.R_FIT],
            "K": row[LOG_GROWTH_MODEL.K_FIT],
            "N0": row[LOG_GROWTH_MODEL.N0_FIT],
        }

    def _hyperparam_kwargs(self) -> Dict[str, float]:
        return {"lam": self.lam, "beta": self.beta}

    def _post_fit_columns(self) -> Dict[Any, float]:
        return {
            LOG_GROWTH_MODEL.LAM: self.lam,
            LOG_GROWTH_MODEL.BETA: self.beta,
        }

    def _extra_agg_columns(self) -> Dict[str, Any]:
        if self.Kmax_label is None:
            return {}
        return {self.Kmax_label: "max"}

    def _hover_fields(self) -> List[Tuple[str, Any, str]]:
        return [
            ("r", LOG_GROWTH_MODEL.R_FIT, ".4f"),
            ("K", LOG_GROWTH_MODEL.K_FIT, ".2f"),
            ("N₀", LOG_GROWTH_MODEL.N0_FIT, ".2f"),
            ("µmax", LOG_GROWTH_MODEL.GROWTH_RATE, ".4f"),
            ("RMSE", MODEL_METRICS.RMSE, ".4f"),
        ]
