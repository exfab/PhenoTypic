from typing import Callable, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from phenotypic.analysis.abstract import ModelFitter


# TODO
class LogGrowthModel(ModelFitter):

    def __init__(self, on: str, groupby: List[str],
                 agg_func: Callable | str | list | dict | None = 'mean',
                 reg_factor=1.2, cap_penalty=5, loss='linear', verbose: bool = False, num_workers: int = 1):
        super().__init__(on=on, groupby=groupby, agg_func=agg_func, num_workers=num_workers)
        self.reg_factor = reg_factor
        self.cap_penalty = cap_penalty
        self.loss = loss
        self.verbose = verbose

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        self._latest_measurements = data

        pass

    def show(self) -> Tuple[plt.Figure, plt.Axes]:
        pass

    def results(self) -> pd.DataFrame:
        return self._latest_measurements

    @staticmethod
    def _model_func(t, r, K, N0):
        """
        Computes the population size at time `t` based on a logistic growth model.

        The logistic growth model is commonly used to describe population growth
        where the growth rate decreases as the population size approaches the
        carrying capacity (K). This model is governed by an exponential component
        that reflects growth while considering the limits imposed by the carrying
        capacity.

        Args:
            t (float): Time at which the population size is being calculated.
            r (float): Intrinsic growth rate of the population.
            K (float): Carrying capacity of the population, defining the maximum
                sustainable population size.
            N0 (float): Initial population size at time `t=0`.

        Returns:
            float: Population size at time `t` based on the logistic growth model.
        """
        a = (K - N0)/N0
        return K/(1 + a*np.exp(-r*t))

    @staticmethod
    def _loss_func(params, t, y, lam, alpha):
        r"""
        Computes a combined loss which includes both the residuals from the predicted
        values using a logarithmic growth model, a regularization term, and a penalty
        for deviations in the carrying capacity (K).

        :math:`J(K,N_0,r) = \frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}(f_{K,N0,r}(t^{(i)} - N_t^{(i)})^2)+ \lambda(\frac{dN}{dt}^2 + N_0^2) + \alphaK`

        The function calculates the residuals (difference between actual and predicted
        values), a regularization term based on biological parameters, and applies a
        penalty proportional to the deviation of K from the observed maximum value
        within the data. A small epsilon is used to ensure numerical stability during
        penalty calculation.

        Note:
            This function is meant to be used in conjunction with the scipy least squares optimization method

        Args:
            params (List[float]): A list containing the parameters [r, K, N0], where:
                r: Growth rate.
                K: Carrying capacity.
                N0: Initial population size.
            t (Union[List[float], np.ndarray]): Time points for the observations.
            y (Union[List[float], np.ndarray, pd.Series]): Observed population size
                corresponding to the time points t. Can be a list, numpy array, or
                pandas.Series object.
            lam (float): Regularization parameter for the biological parameters.
            alpha (float): Scaling parameter for the K-based penalty.

        Returns:
            np.ndarray: A combined loss array consisting of the residuals, regularization
            terms, and the K penalty. The array includes:
                - Residuals: Difference between observed and model-predicted values.
                - Regularization terms: Regularization applied to dN/dt and N0.
                - K Penalty: Penalty term applied based on the deviation of K.
        """
        r, K, N0 = params

        # Original cost function (residuals)
        cost_func = y - log_growth_model(t=t, r=r, K=K, N0=N0)

        # Original regularization term
        dN_dt = r*K/4
        reg_term = np.sqrt(lam)*np.array([dN_dt, N0])

        # K-based penalty
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)

        y_max_observed = np.max(y_array)

        # Numerical stability epsilon
        epsilon = 1e-8*np.median(np.abs(y_array[y_array > 0]))
        if epsilon == 0 or np.isnan(epsilon):
            epsilon = 1e-8

        # Relative K penalty
        K_penalty_weight = np.sqrt(lam*alpha)
        K_penalty = K_penalty_weight*(K - y_max_observed)/(y_max_observed + epsilon)

        return np.hstack([cost_func, reg_term, [K_penalty]])

    @staticmethod
    def _apply2group_func(group: pd.DataFrame):
        pass
