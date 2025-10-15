import itertools
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from joblib import delayed, Parallel
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

from phenotypic.analysis.abstract import ModelFitter
from phenotypic.tools.constants_ import MeasurementInfo


class LOG_GROWTH_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return 'LogGrowthModel'

    R_FIT = 'r', 'The intrinsic growth rate'
    K_FIT = 'K', "The carrying capacity"
    N0_FIT = "N0", "The initial number of the colony size metric being fitted"
    GROWTH_RATE = "d(N)/dt", "The growth rate of the colony calculated as (K*r)/4"
    K_MAX = "Kmax", "The upper bound of the carrying capacity for model fitting"
    NUM_SAMPLES = "NumSamples", "The number of samples used for model fitting"
    LOSS = "OptimizerLoss", "The loss of model fitting"
    STATUS = "OptimizerStatus", "The output of the optimizer status"
    MAE = "MAE", "The mean absolute error"
    MSE = "MSE", "The mean squared error"
    RMSE = "RMSE", "The root mean squared error"


class LogGrowthModel(ModelFitter):
    """
    A model for analyzing and fitting logarithmic growth data.

    This class provides tools for fitting a logarithmic growth model to data,
    analyzing grouped data, visualizing model results, and accessing summarized
    results. It extends the functionalities of the `ModelFitter` base class,
    offering parameter optimization, aggregation methods, and flexibility in
    penalty and loss functions to fine-tune model behavior.

    Attributes:
        reg_factor (float): Regularization factor applied to growth rate during
            optimization.
        kmax_penalty (float): Penalty factor influencing the carrying capacity
            during optimization.
        loss (str): Specifies the loss function to be used. Currently supports
            "linear" as the default option.
        verbose (bool): Indicates whether detailed information should be displayed
            during fitting and optimization.
        time_label (str): Column name in the data representing time measurements.
        Kmax_label (str | None): Column name for the carrying capacity if provided.
            Defaults to None.
    """

    def __init__(self, on: str,
                 groupby: List[str],
                 agg_func: Callable | str | list | dict | None = 'mean',
                 time_label: str = 'Metadata_Time',
                 Kmax_label: str | None = None,
                 growth_rate_penalty=1.2,
                 cap_penalty=5,
                 loss: Literal["linear"] = "linear",
                 verbose: bool = False,
                 n_jobs: int = 1):
        super().__init__(on=on, groupby=groupby, agg_func=agg_func, num_workers=n_jobs)
        self.reg_factor = growth_rate_penalty
        self.kmax_penalty = cap_penalty
        self.loss = loss
        self.verbose = verbose

        self.time_label = time_label
        self.Kmax_label = Kmax_label

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        self._latest_measurements = data

        # aggregate so that only one sample per timepoint
        agg_dict = {self.on: self.agg_func}
        if self.Kmax_label is not None:
            agg_dict[self.Kmax_label] = 'max'  # Use max for Kmax as it's a carrying capacity
        agg_data = data.groupby(by=self.groupby + [self.time_label], as_index=False).agg(agg_dict)

        fitting_func = partial(self._apply2group_func,
                               groupby_names=self.groupby,
                               model=self._model_func,
                               time_label=self.time_label,
                               size_label=self.on,
                               Kmax_label=self.Kmax_label,
                               lam=self.reg_factor,
                               alpha=self.kmax_penalty,
                               loss=self.loss,
                               verbose=self.verbose,
                               )

        grouped = agg_data.groupby(by=self.groupby, as_index=True)
        if self.n_jobs == 1:
            model_res = []
            for key, group in grouped:
                model_res.append(fitting_func(key, group))
        else:
            model_res = Parallel(n_jobs=self.n_jobs)(
                    delayed(fitting_func)(key, group)
                    for key, group in grouped
            )
        self._latest_model_scores = pd.concat(model_res, axis=0).reset_index(drop=False)
        return self._latest_model_scores

    def show(self,
             criteria: Dict[str, Union[Any, List[Any]]] | None = None,
             figsize=(6, 4), cmap: str = 'tab20',
             legend=True, ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        if criteria is not None:
            filtered_model_scores = self._filter_by(df=self._latest_model_scores, criteria=criteria, copy=True)
            filtered_measurements = self._filter_by(df=self._latest_measurements, criteria=criteria, copy=True)
        else:
            filtered_model_scores = self._latest_model_scores
            filtered_measurements = self._latest_measurements
        model_groups = {model_keys: model_groups for model_keys, model_groups in
                        filtered_model_scores.groupby(by=self.groupby)}
        meas_groups = {meas_keys: meas_groups for meas_keys, meas_groups in
                       filtered_measurements.groupby(by=self.groupby)}
        tmax = filtered_measurements.loc[:, self.time_label].max()
        t = np.arange(stop=tmax, step=1)
        cmap = cm.get_cmap(cmap)
        color_iter = itertools.cycle(cmap(np.linspace(0, 1, 256)))
        next(color_iter)
        for model_key, model_group in model_groups.items():
            curr_meas = meas_groups[model_key]
            curr_color = next(color_iter)
            y_pred = self._model_func(t=t,
                                      r=model_group[LOG_GROWTH_MODEL.R_FIT].iloc[0],
                                      K=model_group[LOG_GROWTH_MODEL.K_FIT].iloc[0],
                                      N0=model_group[LOG_GROWTH_MODEL.N0_FIT].iloc[0],
                                      )
            ax.plot(t, y_pred, label=model_key, color=curr_color)
            ax.scatter(
                    x=curr_meas.loc[:, self.time_label],
                    y=curr_meas.loc[:, self.on],
                    color=curr_color, label=model_key,
            )
        return fig, ax

    def results(self) -> pd.DataFrame:
        return self._latest_model_scores

    @staticmethod
    def _model_func(t: np.ndarray[float] | float, r: float, K: float, N0: float):
        """
        Computes the value of the logistic growth model for a given time point or array
        of time points and parameters. The logistic model describes growth that
        initially increases exponentially but levels off as the population reaches
        a carrying capacity.

        This static method uses the formula:
            N(t) = K / (1 + [(K - N0) / N0] * exp(-r * t))

        Where:
            t: Time (independent variable, can be scalar or array).
            r: Growth rate.
            K: Carrying capacity (maximum population size).
            N0: Initial population size.

        Args:
            t (np.ndarray[float] | float): Time at which the population is calculated.
                Can be a single value or an array of values.
            r (float): Growth rate of the population.
            K (float): Carrying capacity or the maximum population size.
            N0 (float): Initial population size at time t=0.

        Returns:
            float | np.ndarray[float]: The computed population size at the given time
            or array of times based on the logistic growth model.
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
        cost_func = y - LogGrowthModel._model_func(t=t, r=r, K=K, N0=N0)

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
    def _apply2group_func(group_key: tuple,
                          group: pd.DataFrame,
                          groupby_names: tuple,
                          model: Callable,
                          time_label: str,
                          size_label: str,
                          Kmax_label: str | None,
                          lam: float,
                          alpha: float,
                          loss: Literal['linear'],
                          verbose: bool):
        t_data = group[time_label]
        size_data = group[size_label]

        i_min = 0
        n_samples = len(t_data)

        r_min, r_max = 1e-5, np.inf

        N0_min, N0_max = 0, size_data.min()
        if N0_max <= N0_min: N0_max = N0_min + 1  # Safety check since max bound must be higher than min bound

        if Kmax_label is None:
            K_max = size_data.max()
        else:
            K_max = group[Kmax_label].max()

        if K_max == np.nan:
            K_max = size_data.max() + 1

        K_min = i_min

        try:
            out = optimize.least_squares(LogGrowthModel._loss_func,
                                         x0=[1e-5, size_data.max(), 0],
                                         bounds=(
                                             [r_min, K_min, N0_min],
                                             [r_max, K_max, N0_max],
                                         ),
                                         kwargs=dict(
                                                 t=t_data,
                                                 y=size_data,
                                                 lam=lam,
                                                 alpha=alpha,
                                         ),
                                         verbose=verbose,
                                         method='trf',
                                         loss=loss,
                                         )
            x = out.x
            fitted_values = {
                LOG_GROWTH_MODEL.R_FIT      : x[0],
                LOG_GROWTH_MODEL.K_FIT      : x[1],
                LOG_GROWTH_MODEL.N0_FIT     : x[2],
                LOG_GROWTH_MODEL.GROWTH_RATE: (x[0]*x[1])/4,
            }

            y_pred = LogGrowthModel._model_func(t=t_data, r=x[0], K=x[1], N0=x[2])
            model_stats = {
                LOG_GROWTH_MODEL.K_MAX      : K_max,
                LOG_GROWTH_MODEL.NUM_SAMPLES: n_samples,
                LOG_GROWTH_MODEL.LOSS       : out.cost,
                LOG_GROWTH_MODEL.STATUS     : out.status,
                LOG_GROWTH_MODEL.MAE        : mean_absolute_error(size_data, y_pred),
                LOG_GROWTH_MODEL.MSE        : mean_squared_error(size_data, y_pred),
                LOG_GROWTH_MODEL.RMSE       : root_mean_squared_error(size_data, y_pred),
            }
        except ValueError:
            fitted_values = {
                LOG_GROWTH_MODEL.R_FIT      : np.nan,
                LOG_GROWTH_MODEL.K_FIT      : np.nan,
                LOG_GROWTH_MODEL.N0_FIT     : np.nan,
                LOG_GROWTH_MODEL.GROWTH_RATE: np.nan,
            }

            model_stats = {
                LOG_GROWTH_MODEL.K_MAX      : np.nan,
                LOG_GROWTH_MODEL.NUM_SAMPLES: np.nan,
                LOG_GROWTH_MODEL.LOSS       : np.nan,
                LOG_GROWTH_MODEL.STATUS     : np.nan,
                LOG_GROWTH_MODEL.MAE        : np.nan,
                LOG_GROWTH_MODEL.MSE        : np.nan,
                LOG_GROWTH_MODEL.RMSE       : np.nan,
            }

        return pd.DataFrame(data={**fitted_values, **model_stats},
                            index=pd.MultiIndex.from_tuples(
                                    tuples=[group_key], names=groupby_names
                            ))
