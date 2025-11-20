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

from phenotypic.analysis.abc_ import ModelFitter
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
    Represents a log growth model fitter.

    This class defines methods and attributes to configure and fit logarithmic
    growth models to grouped data. It provides functionality for analyzing and
    visualizing the fitted models as well as exposing the results for further
    processing.

    Loss Function Calculation:
         :math:`J(K,N_0,r) = \frac{1}{n}\sum_{i=1}^{n}\frac{1}{2}(f_{K,N0,r}(t^{(i)}) - N_t^{(i)})^2)
            + \lambda(\frac{dN}{dt}^2 + N_0^2)
            + \alpha \frac{\lvert K-\max({N_t})\rvert}{N_t}`

        :math:`N_t`: population size at time t
        :math:`N_0`: initial population size at time t
        :math:`r`: growth rate
        :math:`K`: Carrying capacity (maximum population size).
        :math:`\lambda`: regularization term for growth rate and initial population size
        :math:`\alpha`: penalty term for deviations in the carrying capacity relative to the largest measurement


    Attributes:
        lam (float): The penalty factor applied to growth rates.
        alpha (float): The maximum penalty factor applied to the carrying
            capacity.
        loss (Literal["linear"]): The loss calculation method used for fitting.
        verbose (bool): A flag to enable or disable detailed logging.
        time_label (str): The column name representing the time dimension
            in the input data.
        Kmax_label (str | None): The column name for the maximum carrying capacity
            values, if provided.
    """

    def __init__(self,
                 on: str,
                 groupby: List[str],
                 time_label: str = 'Metadata_Time',
                 agg_func: Callable | str | list | dict | None = 'mean',
                 lam=1.2,
                 alpha=2,
                 Kmax_label: str | None = None,
                 loss: Literal["linear"] = "linear",
                 verbose: bool = False,
                 n_jobs: int = 1):
        """
        This class initializes parameters for a data processing or modeling procedure.
        It takes configuration arguments for handling data grouping, time management,
        aggregation, penalties, loss calculation, and verbosity.

        Args:
            on (str): The target variable or column to process.
            groupby (List[str]): The columns that define the grouping structure.
            time_label (str): Column name that represents time in the data. Defaults to 'Metadata_Time'.
            agg_func (Callable | str | list | dict | None): Aggregation function(s) to
                apply to grouped data. Parameter is fed to `pandas.DataFrame.groupby.agg()`.
                Defaults to 'mean'.
            lam: The penalty factor applied to growth rates. Defaults to 1.2.
            alpha: The maximum penalty factor applied to the carrying capacity. Defaults to 2.
            Kmax_label (str | None): Column name that provides maximum K value for processing. Defaults to None.
            loss (Literal["linear"]): Loss calculation method to apply. Defaults to "linear".
            verbose (bool): If True, enables detailed logging for process execution. Defaults to False.
            n_jobs (int): Number of parallel jobs to execute. Defaults to 1.
        """
        super().__init__(on=on, groupby=groupby, agg_func=agg_func, num_workers=n_jobs)
        self.lam = lam
        self.alpha = alpha
        self.loss = loss
        self.verbose = verbose

        self.time_label = time_label
        self.Kmax_label = Kmax_label

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        data: pd.DataFrame = data.copy(deep=True)
        data.loc[:, self.time_label] = self._ensure_float_array(data.loc[:, self.time_label])
        self._latest_measurements = data

        apply2group_kwargs = dict(
                groupby_names=self.groupby,
                model=self.__class__.model_func,
                time_label=self.time_label,
                size_label=self.on,
                Kmax_label=self.Kmax_label,
                lam=self.lam,
                alpha=self.alpha,
                loss=self.loss,
                verbose=self.verbose,
        )

        # aggregate so that only one sample per timepoint
        agg_dict = {self.on: self.agg_func}
        if self.Kmax_label is not None:
            agg_dict[self.Kmax_label] = 'max'  # Use max for Kmax as it's a carrying capacity

        agg_data = data.groupby(by=self.groupby + [self.time_label], as_index=False).agg(agg_dict)

        # Create groups
        grouped = agg_data.groupby(by=self.groupby, as_index=True)
        if self.n_jobs == 1:

            model_res = []
            for key, group in grouped:
                model_res.append(self.__class__._apply2group_func(key, group, **apply2group_kwargs))
        else:
            model_res = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.__class__._apply2group_func)(key, group, **apply2group_kwargs)
                    for key, group in grouped
            )
        self._latest_model_scores = pd.concat(model_res, axis=0).reset_index(drop=False)
        return self._latest_model_scores

    def show(self,
             tmax: int | float | None = None,
             criteria: Dict[str, Union[Any, List[Any]]] | None = None,
             figsize=(6, 4),
             cmap: str = 'tab20',
             legend=True,
             ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualizes model predictions alongside measurements, allowing optional
        filtering by specified criteria and plotting configuration.

        Args:
            tmax (int | float | None, optional): The maximum time value for plotting. If
                set to None, the maximum time value will be determined from the data
                automatically.
            criteria (Dict[str, Union[Any, List[Any]]] | None, optional): A dictionary
                specifying filtering criteria for data selection. When provided, only
                data matching the criteria will be used for plotting.
            figsize (tuple, optional): A tuple specifying the size of the figure.
                Defaults to (6, 4).
            cmap (str, optional): A string representing the colormap to be used for
                plotting model predictions. Defaults to 'tab20'.
            legend (bool, optional): A boolean that controls whether a legend is
                displayed on the plot. Defaults to True.
            ax (plt.Axes, optional): A matplotlib Axes object on which to plot. If not
                provided, a new figure and axes object will be created.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and
                Axes objects used for plotting.

        Raises:
            KeyError: If the group keys for model results and measurements do not
                align, or if specified columns are missing from the input data.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # apply filtering of data table
        if criteria is not None:
            filtered_model_scores = self._filter_by(df=self._latest_model_scores, criteria=criteria, copy=True)
            filtered_measurements = self._filter_by(df=self._latest_measurements, criteria=criteria, copy=True)
        else:
            filtered_model_scores = self._latest_model_scores.copy()
            filtered_measurements = self._latest_measurements.copy()

        if filtered_measurements.empty:
            import warnings
            warnings.warn("No data found matching the criteria. Returning empty plot.")
            return fig, ax

        model_groups = {model_keys: model_groups for model_keys, model_groups in
                        filtered_model_scores.groupby(by=self.groupby)}
        meas_groups = {meas_keys: meas_groups for meas_keys, meas_groups in
                       filtered_measurements.groupby(by=self.groupby)}

        filtered_measurements.loc[:, self.time_label] = \
            self._ensure_float_array(filtered_measurements.loc[:, self.time_label])

        timepoints = pd.Series(filtered_measurements.loc[:, self.time_label].unique())

        step = np.mean(
                timepoints
                .sort_values()
                .diff()
                .dropna()
        )

        if np.isnan(step) or step <= 0:
            step = 1.0

        tmax = timepoints.max() if tmax is None else tmax

        t = np.arange(stop=tmax + step, step=step)

        cmap = cm.get_cmap(cmap)
        color_iter = itertools.cycle(cmap(np.linspace(start=0, stop=1, num=len(model_groups), endpoint=False)))
        for model_key, model_group in model_groups.items():
            curr_meas = meas_groups[model_key]
            curr_color = next(color_iter)
            y_pred = self.model_func(t=t,
                                     r=model_group[LOG_GROWTH_MODEL.R_FIT].iloc[0],
                                     K=model_group[LOG_GROWTH_MODEL.K_FIT].iloc[0],
                                     N0=model_group[LOG_GROWTH_MODEL.N0_FIT].iloc[0],
                                     )
            ax.plot(t, y_pred, color=curr_color)

            curr_time_groups = curr_meas.groupby(by=self.time_label)
            curr_mean = curr_time_groups[self.on].mean()
            curr_stddev = curr_time_groups[self.on].std()
            curr_stderr = curr_stddev/np.sqrt(curr_time_groups[self.on].count())

            ax.errorbar(
                    x=curr_mean.index.values,
                    y=curr_mean.values,
                    yerr=curr_stderr,
                    fmt='o',
                    color=curr_color,
                    ecolor=curr_color,
                    elinewidth=1,
                    capsize=2,
                    label=f'{model_key[0]}',
            )
        if legend:
            # Create legend and check if it fits within the axes
            legend_obj = ax.legend()

            # Draw to ensure bounding boxes are available
            fig.canvas.draw()

            # Get bounding boxes in display coordinates
            legend_bbox = legend_obj.get_window_extent()
            axes_bbox = ax.get_window_extent()

            # Check if legend is larger than axes (with small tolerance)
            if legend_bbox.width > axes_bbox.width*0.95 or legend_bbox.height > axes_bbox.height*0.95:
                legend_obj.remove()

        ax.set_title('mean±SE')
        return fig, ax

    def results(self) -> pd.DataFrame:
        return self._latest_model_scores

    @staticmethod
    def model_func(t: np.ndarray[float] | float, r: float, K: float, N0: float):
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

        :math:`J(K,N_0,r) = \frac{1}{n}\sum_{i=1}^{n}\frac{1}{2}(f_{K,N0,r}(t^{(i)}) - N_t^{(i)})^2)
            + \lambda(µ_{max}^2 + N_0^2)
            + \alpha \frac{\lvert K-\max({N_t})\rvert}{N_t}`

        :math:`N_t`: population size at time t.
        :math:`N_0`: initial population size at time t.
        :math:`r`: intrinsic growth rate.
        :math:`K`: Carrying capacity (maximum population size).
        :math:`µ_{max}`: Specific growth rate.
        :math:`\lambda`: regularization term for the specific growth rate and initial population size.
        :math:`\alpha`: penalty term for deviations in the carrying capacity relative to the largest measurement.

        The function calculates the residuals (difference between actual and predicted
        values), a regularization term based on biological parameters, and applies a
        penalty proportional to the deviation of K from the observed maximum value
        within the data. A small epsilon is used to ensure numerical stability during
        penalty calculation.

        Note:
            This function is used in conjunction with the `scipy.optimize.least_squares` solver to
            find the parameters

        Args:
            params (List[float]): A list containing the parameters [r, K, N0], where:
                r: Growth rate.
                K: Carrying capacity.
                N0: Initial population size.
            t (Union[List[float], np.ndarray]): Time points for the observations.
            y (Union[List[float], np.ndarray, pd.Series]): Observed population size
                corresponding to the time points t. Can be a list, numpy array, or
                pandas.Series object.
            lam (float): Regularization parameter for the specific growth rate and initial population size.
            alpha (float): Scaling parameter for the K-based penalty.

        Returns:
            np.ndarray: A combined loss array consisting of the residuals, regularization
            terms, and the K penalty. The array includes:
                - Residuals: Difference between observed and model-predicted values.
                - Regularization terms: Regularization applied to dN/dt and N0.
                - K Penalty: Penalty term applied based on the deviation of K.
        """
        r, K, N0 = params

        # cost function (residuals)
        cost_func = y - LogGrowthModel.model_func(t=t, r=r, K=K, N0=N0)

        # regularization term
        dN_dt = r*K/4
        reg_term = np.sqrt(lam)*np.array([dN_dt, N0])

        # K-based penalty
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Get the value of the last time point
        # assumes thats the best indicator for carrying capacity
        y_max_observed = y_array[np.argmax(t)]

        # Numerical stability epsilon
        epsilon = 1e-8*np.median(np.abs(y_array[y_array > 0]))
        if epsilon == 0 or np.isnan(epsilon):
            epsilon = 1e-8

        # Relative K penalty
        K_penalty_weight = np.sqrt(lam*alpha)
        K_penalty = K_penalty_weight*np.abs(K - y_max_observed)/(y_max_observed + epsilon)

        return np.hstack([cost_func, reg_term, [K_penalty]])

    @staticmethod
    def _apply2group_func(
            group_key: tuple,
            group: pd.DataFrame,
            groupby_names: tuple,
            model: Callable,
            time_label: str,
            size_label: str,
            Kmax_label: str | None,
            lam: float,
            alpha: float,
            loss: Literal['linear'],
            verbose: bool
    ):
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

            y_pred = LogGrowthModel.model_func(t=t_data, r=x[0], K=x[1], N0=x[2])
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
