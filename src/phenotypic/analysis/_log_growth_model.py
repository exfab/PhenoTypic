from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from phenotypic.analysis.abc_ import ModelFitter
from phenotypic.sdk_ import ColumnRef
from phenotypic.schema import LOG_GROWTH_MODEL, MODEL_METRICS


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

        :math:`\mu_{\max}`: maximum absolute growth rate, the slope of the curve
            where :math:`N_t = K/2` (the maximum specific growth rate is :math:`r`)


    Loss Function:

        The parameters minimize the cost of SciPy's nonlinear least-squares
        solver, :func:`scipy.optimize.least_squares` with ``method="trf"``, over
        the residual vector :meth:`_loss_func` returns. That vector holds the
        :math:`n` data residuals and three penalty residuals:

        .. math::

           z = \left(N_t^{(1)} - f_{K,N_0,r}(t^{(1)}),\ \dots,\;
                N_t^{(n)} - f_{K,N_0,r}(t^{(n)}),\;
                \sqrt{\lambda}\,\frac{rK}{4},\ \sqrt{\lambda}\,N_0,\;
                \sqrt{\beta}\,\frac{\lvert K - N_{n} \rvert}{N_{n} + \varepsilon}\right)

        and the cost is

        .. math::

           F(r, K, N_0) = \frac{1}{2} \sum_{j} C^2\,
           \rho\!\left(\frac{z_j^2}{C^2}\right)

        where :math:`C` is ``f_scale`` (default 1.0) and :math:`\rho` is the
        ``loss`` (default ``"huber"``: :math:`\rho(s) = s` for :math:`s \le 1`
        and :math:`2\sqrt{s} - 1` above). The robust loss applies to the
        penalty residuals as well as the data. With ``loss="linear"``,
        :math:`\rho(s) = s` and the cost is a sum, not a mean, with a squared
        :math:`\beta` term:

        .. math::

           F = \frac{1}{2}\left[\sum_{i=1}^{n}
           \left(f_{K,N_0,r}(t^{(i)}) - N_t^{(i)}\right)^2
           + \lambda\left(\left(\frac{rK}{4}\right)^2 + N_0^2\right)
           + \beta\left(\frac{K - N_{n}}{N_{n} + \varepsilon}\right)^2\right]

        :math:`\lambda`: regularization weight on the maximum growth rate
            :math:`rK/4` and on :math:`N_0`

        :math:`\beta`: weight of the penalty on the distance of :math:`K` from
            :math:`N_{n}`, the measurement at the last time point (the first row
            at the latest time), relative to :math:`N_{n}`

        :math:`\varepsilon`: :math:`10^{-8}` times the median positive
            measurement (or :math:`10^{-8}`), which keeps the division finite

    Output column naming:
        ``analyze`` emits metric-qualified columns
        ``LogGrowthModel_<metric>_<parameter>`` (e.g. fitting
        ``on="Size_Area"`` yields ``LogGrowthModel_Area_r`` and
        ``LogGrowthModel_Area_µmax``), plus qualified fit-quality columns
        ``ModelMetrics_<metric>_<label>``. The ``<metric>`` segment is
        ``self.on`` with a recognized metric-family prefix stripped.

    Attributes:
        lam (float): Regularization weight on the maximum growth rate
            r·K/4 and on N0; larger values pull both toward 0.
        beta (float): Weight of the penalty on the distance of the fitted
            carrying capacity K from the value measured at the last time
            point, relative to that value.
        Kmax_label (str | None): The column name for the maximum carrying capacity
            values, if provided.
    """

    _measurement_infoclass = LOG_GROWTH_MODEL

    lam: float = 1.2
    beta: int | float = 2
    Kmax_label: ColumnRef | None = None

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

        Returns the vector

        .. math::

           z = \left(N_t^{(i)} - f_{K,N_0,r}(t^{(i)})\ \text{for each } i,\;
                \sqrt{\lambda}\,\frac{rK}{4},\ \sqrt{\lambda}\,N_0,\;
                \sqrt{\beta}\,\frac{\lvert K - N_{n} \rvert}{N_{n} + \varepsilon}\right)

        with :math:`N_{n}` the measurement at the last time point. The solver
        minimizes :math:`\frac{1}{2} \sum_j C^2 \rho(z_j^2 / C^2)` over it (see
        the class docstring), which for ``loss="linear"`` is
        :math:`\frac{1}{2} \sum_j z_j^2`.

        Args:
            params: ``[r, K, N0]`` — candidate parameter vector.
            t: Time points for the observations.
            y: Observed population sizes (list, ndarray, or pandas Series).
            lam: Regularization weight for ``dN/dt`` and ``N0``.
            beta: Scaling factor for the penalty on ``K`` versus the value
                measured at the last time point.

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
            LOG_GROWTH_MODEL.R_FIT      : r,
            LOG_GROWTH_MODEL.K_FIT      : K,
            LOG_GROWTH_MODEL.N0_FIT     : N0,
            LOG_GROWTH_MODEL.GROWTH_RATE: (r * K) / 4,
            LOG_GROWTH_MODEL.K_MAX      : self._kmax_for(group),
        }

    def _predict_kwargs(self, row) -> Dict[str, float]:
        return {
            "r" : row[LOG_GROWTH_MODEL.R_FIT],
            "K" : row[LOG_GROWTH_MODEL.K_FIT],
            "N0": row[LOG_GROWTH_MODEL.N0_FIT],
        }

    def _hyperparam_kwargs(self) -> Dict[str, float]:
        return {"lam": self.lam, "beta": self.beta}

    def _post_fit_columns(self) -> Dict[Any, float]:
        return {
            LOG_GROWTH_MODEL.LAM : self.lam,
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


LogGrowthModel.__doc__ = LOG_GROWTH_MODEL.append_rst_to_doc(LogGrowthModel)
