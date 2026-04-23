from __future__ import annotations

import abc
import itertools
from abc import ABC
from typing import Any, Callable, Dict, List, Literal, Tuple, TYPE_CHECKING, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from joblib import Parallel, delayed
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from phenotypic.tools_.measurement_info_ import MODEL_METRICS

from ._set_analyzer import SetAnalyzer

if TYPE_CHECKING:
    import plotly.graph_objects as go


class ModelFitter(SetAnalyzer, ABC):
    """Template base class for grouped least-squares model fitting.

    Subclasses provide the mathematical model (`model_func`), the loss
    function (`_loss_func`), an initial parameter guess, parameter bounds,
    and a small set of hooks that let the base class drive ``analyze``,
    ``show``, ``dash``, and ``results`` without hard-coding parameter
    names.

    Fit-quality metrics (MAE, MSE, RMSE, R²) and optimizer diagnostics
    (loss, status, sample count) are emitted under the shared
    :class:`MODEL_METRICS` category so that every ``ModelFitter`` subclass
    produces a consistent set of diagnostic columns.

    Attributes:
        time_label (str): Column name representing the independent
            variable (typically time).
        loss (Literal["linear"]): Loss calculation method passed through
            to :func:`scipy.optimize.least_squares`.
        verbose (bool): Whether to print detailed optimizer output.
    """

    _measurement_info_class: type

    def __init__(
        self,
        on: str,
        groupby: List[str],
        time_label: str = "Metadata_Time",
        agg_func: Callable | str | list | dict | None = "mean",
        *,
        num_workers: int = 1,
        loss: Literal["linear"] = "linear",
        verbose: bool = False,
    ):
        super().__init__(
            on=on, groupby=groupby, agg_func=agg_func, num_workers=num_workers
        )
        self._latest_model_scores: pd.DataFrame = pd.DataFrame()
        self.time_label = time_label
        self.loss = loss
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Abstract model math — subclasses must implement
    # ------------------------------------------------------------------ #
    @staticmethod
    @abc.abstractmethod
    def model_func(*args, **kwargs):
        """Mathematical model. First positional argument is the independent variable."""
        pass

    def _params_to_model_kwargs(self, params) -> Dict[str, float]:
        """Map the raw optimizer vector to kwargs for ``model_func``.

        Required only when the subclass relies on the default MSE
        :meth:`_loss_func`. Subclasses that override ``_loss_func`` with a
        fully custom residual (e.g. regularized) do not need to implement
        this hook.
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses the default MSE loss but does "
            f"not implement `_params_to_model_kwargs`. Either implement "
            f"it to map the optimizer vector to `model_func` kwargs, or "
            f"override `_loss_func` with a fully custom residual."
        )

    def _loss_func(self, params, t, y, **_):
        """Default residual vector — plain MSE against the observations.

        ``scipy.optimize.least_squares`` minimizes half the sum of
        squared residuals, so returning ``y - model_func(t, …)`` here
        yields the standard MSE fit. Subclasses may override this to add
        regularization, penalties, or robust loss terms (see
        :class:`LogGrowthModel` for an example).

        Extra keyword arguments are accepted but ignored so that any
        hyperparameters supplied by :meth:`_hyperparam_kwargs` can be
        safely forwarded without needing to match this default
        signature.
        """
        return y - self.model_func(t, **self._params_to_model_kwargs(params))

    # ------------------------------------------------------------------ #
    # Abstract fit hooks — subclasses must implement
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _initial_guess(self, group: pd.DataFrame) -> list[float]:
        """Initial guess ``x0`` for :func:`scipy.optimize.least_squares`."""
        pass

    @abc.abstractmethod
    def _bounds(
        self, group: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """Return ``(lower, upper)`` bounds for the fitted parameters."""
        pass

    @abc.abstractmethod
    def _unpack_params(
        self, x: np.ndarray, group: pd.DataFrame
    ) -> Dict[Any, Any]:
        """Map optimizer output to a dict keyed by MeasurementInfo members.

        Must include every fitted/derived parameter column produced by this
        model (e.g. ``r``, ``K``, ``N0``, ``µmax`` for log-growth). May also
        include per-group bounds that should be preserved in results
        (e.g. ``K_max``), or string-valued diagnostic fields (e.g. the
        per-group fit mode emitted by :class:`LinearSoftplusModel`).
        """
        pass

    @abc.abstractmethod
    def _predict_kwargs(self, row) -> Dict[str, Any]:
        """Build kwargs for ``model_func(t, **kwargs)`` from a results row.

        ``row`` is any mapping keyed by MeasurementInfo members — a dict
        produced by ``_unpack_params`` at fit time, or a ``pd.Series``
        drawn from the results DataFrame at plot time. Values may be
        ``None`` for optional model kwargs (e.g. ``smax=None`` to
        disable the saturation ceiling in
        :class:`LinearSoftplusModel`).
        """
        pass

    # ------------------------------------------------------------------ #
    # Optional hooks — subclasses may override
    # ------------------------------------------------------------------ #
    def _hyperparam_kwargs(self) -> Dict[str, float]:
        """Extra kwargs forwarded to ``_loss_func`` (e.g. regularization)."""
        return {}

    def _prepare_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Preprocess one group before ``t``/``y`` extraction.

        Default returns the group unchanged. Subclasses override to
        filter rows (e.g. saturation pruning) or synthesize helper
        columns needed by the loss function.
        """
        return group

    def _extra_loss_kwargs(self, group: pd.DataFrame) -> Dict[str, Any]:
        """Per-group kwargs forwarded to :meth:`_loss_func`.

        Default is an empty dict. Subclasses override to inject
        per-group arrays (e.g. weight vectors, per-group bounds)
        that ``_hyperparam_kwargs`` cannot express because it is
        group-agnostic. Merged *after* ``_hyperparam_kwargs`` so
        per-group entries win on collision.
        """
        return {}

    def _post_fit_columns(self) -> Dict[Any, float]:
        """Scalar metadata columns appended to the results DataFrame.

        Useful for recording hyperparameters (e.g. ``lam``, ``beta``) that
        are constant across every fitted group.
        """
        return {}

    def _extra_agg_columns(self) -> Dict[str, Any]:
        """Additional ``{column: agg_func}`` entries for per-timepoint aggregation."""
        return {}

    def _hover_fields(self) -> List[Tuple[str, Any, str]]:
        """``(label, column_key, format_spec)`` entries for the Plotly hover tooltip."""
        return []

    # ------------------------------------------------------------------ #
    # Shared metrics / NaN templates
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_metrics(y_true, y_pred) -> Dict[Any, float]:
        """Compute the generic fit-quality metrics shared by all subclasses."""
        return {
            MODEL_METRICS.MAE: mean_absolute_error(y_true, y_pred),
            MODEL_METRICS.MSE: mean_squared_error(y_true, y_pred),
            MODEL_METRICS.RMSE: root_mean_squared_error(y_true, y_pred),
            MODEL_METRICS.R2: r2_score(y_true, y_pred),
        }

    def _nan_fit_columns(self) -> Dict[Any, float]:
        """NaN-filled row used when ``least_squares`` raises ``ValueError``.

        Covers every MeasurementInfo member declared on the subclass plus
        the full :class:`MODEL_METRICS` set (metrics + diagnostics). Any
        column supplied by ``_post_fit_columns`` is excluded so the
        constant hyperparameter value is preserved rather than overwritten
        with NaN.
        """
        mi = self._measurement_info_class
        post_keys = set(self._post_fit_columns().keys())
        model_cols = {
            member: np.nan
            for member in mi.__members__.values()
            if member not in post_keys
        }
        metric_cols = {
            member: np.nan for member in MODEL_METRICS.__members__.values()
        }
        return {**model_cols, **metric_cols}

    # ------------------------------------------------------------------ #
    # Per-group fit — replaces the old `_apply2group_func` boilerplate
    # ------------------------------------------------------------------ #
    def _apply2group_func(self, group_key, group: pd.DataFrame) -> pd.DataFrame:
        """Fit one group and return a single-row DataFrame of results."""
        group = self._prepare_group(group)
        t_data = group[self.time_label]
        y_data = group[self.on]

        loss_kwargs: Dict[str, Any] = {
            **self._hyperparam_kwargs(),
            **self._extra_loss_kwargs(group),
        }

        try:
            out = optimize.least_squares(
                self._loss_func,
                x0=self._initial_guess(group),
                bounds=self._bounds(group),
                kwargs=dict(t=t_data, y=y_data, **loss_kwargs),
                verbose=int(self.verbose),
                method="trf",
                loss=self.loss,
            )
            fitted = self._unpack_params(out.x, group)
            y_pred = self.model_func(t=t_data, **self._predict_kwargs(fitted))
            row: Dict[Any, Any] = {
                **fitted,
                **self._compute_metrics(y_data, y_pred),
                MODEL_METRICS.LOSS: out.cost,
                MODEL_METRICS.STATUS: out.status,
                MODEL_METRICS.NUM_SAMPLES: len(t_data),
            }
        except ValueError:
            row = self._nan_fit_columns()

        return pd.DataFrame(
            data=row,
            index=pd.MultiIndex.from_tuples(
                tuples=[group_key], names=self.groupby
            ),
        )

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model to every group of ``data`` and return the results.

        Standard template: copy, float-coerce the time column, aggregate
        to one sample per timepoint, dispatch per-group fits (serial or
        parallel via :class:`joblib.Parallel`), concatenate, and append
        constant hyperparameter columns from ``_post_fit_columns``.
        """
        data = data.copy(deep=True)
        data.loc[:, self.time_label] = self._ensure_float_array(
            data.loc[:, self.time_label]
        )
        self._latest_measurements = data

        agg_dict: Dict[str, Any] = {self.on: self.agg_func}
        agg_dict.update(self._extra_agg_columns())

        agg_data = data.groupby(
            by=self.groupby + [self.time_label], as_index=False
        ).agg(agg_dict)

        grouped = agg_data.groupby(by=self.groupby, as_index=True)
        if self.n_jobs == 1:
            model_res = [
                self._apply2group_func(key, group) for key, group in grouped
            ]
        else:
            model_res = Parallel(n_jobs=self.n_jobs)(
                delayed(self._apply2group_func)(key, group)
                for key, group in grouped
            )

        results = pd.concat(model_res, axis=0).reset_index(drop=False)

        for col_key, val in self._post_fit_columns().items():
            results.insert(loc=len(results.columns), column=col_key, value=val)

        self._latest_model_scores = results
        return self._latest_model_scores

    def results(self) -> pd.DataFrame:
        """Return the most recent fit results produced by :meth:`analyze`."""
        return self._latest_model_scores

    # ------------------------------------------------------------------ #
    # Internal helpers for plotting
    # ------------------------------------------------------------------ #
    def _filter_for_plot(
        self, criteria: Dict[str, Union[Any, List[Any]]] | None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply `criteria` (if any) to both the model-scores and measurements frames."""
        if criteria is not None:
            model_scores = self._filter_by(
                df=self._latest_model_scores, criteria=criteria, copy=True
            )
            measurements = self._filter_by(
                df=self._latest_measurements, criteria=criteria, copy=True
            )
        else:
            model_scores = self._latest_model_scores.copy()
            measurements = self._latest_measurements.copy()
        return model_scores, measurements

    def _time_axis(
        self, timepoints: pd.Series, tmax: int | float | None
    ) -> Tuple[np.ndarray, float]:
        """Derive a dense, uniform time axis for plotting prediction curves.

        Samples ``[0, upper]`` with ``2 * upper`` points (floor 2) so
        that sharp model transitions (e.g. the softplus lag/saturation
        in :class:`LinearSoftplusModel`) render smoothly rather than as
        a polyline sampled at the observed timepoints.
        """
        upper = float(timepoints.max() if tmax is None else tmax)
        if not np.isfinite(upper) or upper <= 0:
            upper = 1.0
        num = max(2, int(2 * upper))
        t = np.linspace(0.0, upper, num=num)
        step = upper / (num - 1)
        return t, step

    def _format_hover(self, row) -> str:
        """Join `_hover_fields` into a Plotly ``<extra>`` payload."""
        parts = []
        for label, col_key, fmt in self._hover_fields():
            val = row[col_key]
            parts.append(f"{label} = {format(val, fmt)}")
        return "<br>".join(parts)

    # ------------------------------------------------------------------ #
    # Matplotlib visualization
    # ------------------------------------------------------------------ #
    def show(
        self,
        tmax: int | float | None = None,
        criteria: Dict[str, Union[Any, List[Any]]] | None = None,
        figsize=(6, 4),
        cmap: str | None = "tab20",
        legend: bool = True,
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot model predictions alongside measurements with optional filtering.

        Args:
            tmax: Upper bound of the prediction curve. If ``None``, uses
                the maximum observed time.
            criteria: Column/value filter applied to both fitted results
                and raw measurements before plotting.
            figsize: Matplotlib figure size. Used only when ``ax`` is None.
            cmap: Matplotlib colormap name, a single color string, or
                ``None`` for matplotlib's default color cycle.
            legend: Whether to render a legend (auto-removed if larger
                than the axes).
            ax: Existing axes to draw into. A new figure is created when
                omitted.
            **kwargs: Styling overrides — ``dpi``, ``facecolor``,
                ``edgecolor``, ``line_width``, ``marker_size``,
                ``elinewidth``, ``capsize``, ``legend_loc``,
                ``legend_fontsize``, ``label``.

        Returns:
            A ``(Figure, Axes)`` pair.
        """
        fig_kwargs = {
            k: v for k, v in kwargs.items() if k in ("dpi", "facecolor", "edgecolor")
        }
        line_width = kwargs.get("line_width", None)
        marker_size = kwargs.get("marker_size", None)
        elinewidth = kwargs.get("elinewidth", 1)
        capsize = kwargs.get("capsize", 2)
        legend_loc = kwargs.get("legend_loc", "best")
        legend_fontsize = kwargs.get("legend_fontsize", None)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **fig_kwargs)
        else:
            fig = ax.get_figure()

        model_scores, measurements = self._filter_for_plot(criteria)

        if measurements.empty:
            import warnings

            warnings.warn("No data found matching the criteria. Returning empty plot.")
            return fig, ax

        measurements.loc[:, self.time_label] = self._ensure_float_array(
            measurements.loc[:, self.time_label]
        )

        model_groups = {
            keys: grp for keys, grp in model_scores.groupby(by=self.groupby)
        }
        meas_groups = {
            keys: grp for keys, grp in measurements.groupby(by=self.groupby)
        }

        timepoints = pd.Series(measurements.loc[:, self.time_label].unique())
        t, _ = self._time_axis(timepoints, tmax)

        if cmap is not None:
            try:
                cmap_obj = (
                    matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
                )
                color_iter = itertools.cycle(
                    cmap_obj(
                        np.linspace(
                            start=0, stop=1, num=len(model_groups), endpoint=False
                        )
                    )
                )
            except (ValueError, AttributeError):
                color_iter = itertools.cycle([cmap])
        else:
            color_iter = itertools.cycle([None] * len(model_groups))

        for model_key, model_group in model_groups.items():
            curr_meas = meas_groups[model_key]
            curr_color = next(color_iter)
            row = model_group.iloc[0]
            y_pred = self.model_func(t=t, **self._predict_kwargs(row))

            plot_kwargs: Dict[str, Any] = {}
            if curr_color is not None:
                plot_kwargs["color"] = curr_color
            if line_width is not None:
                plot_kwargs["linewidth"] = line_width
            ax.plot(t, y_pred, **plot_kwargs)

            curr_time_groups = curr_meas.groupby(by=self.time_label)
            curr_mean = curr_time_groups[self.on].mean()
            curr_stddev = curr_time_groups[self.on].std()
            curr_stderr = curr_stddev / np.sqrt(curr_time_groups[self.on].count())

            # noinspection PyUnresolvedReferences
            errorbar_kwargs: Dict[str, Any] = {
                "x": curr_mean.index.values,
                "y": curr_mean.values,
                "yerr": curr_stderr,
                "fmt": "o",
                "elinewidth": elinewidth,
                "capsize": capsize,
                "label": kwargs.get("label", f"{model_key[0]}"),
            }
            if curr_color is not None:
                errorbar_kwargs["color"] = curr_color
                errorbar_kwargs["ecolor"] = curr_color
            if marker_size is not None:
                errorbar_kwargs["markersize"] = marker_size
            ax.errorbar(**errorbar_kwargs)

        if legend:
            legend_kwargs: Dict[str, Any] = {"loc": legend_loc}
            if legend_fontsize is not None:
                legend_kwargs["fontsize"] = legend_fontsize
            legend_obj = ax.legend(**legend_kwargs)

            fig.canvas.draw()
            legend_bbox = legend_obj.get_window_extent()
            axes_bbox = ax.get_window_extent()
            if (
                legend_bbox.width > axes_bbox.width * 0.95
                or legend_bbox.height > axes_bbox.height * 0.95
            ):
                legend_obj.remove()

        ax.set_title("mean±SE")
        return fig, ax

    # ------------------------------------------------------------------ #
    # Plotly visualization
    # ------------------------------------------------------------------ #
    def dash(
        self,
        tmax: int | float | None = None,
        criteria: Dict[str, Union[Any, List[Any]]] | None = None,
        figsize=(6, 4),
        cmap: str | None = "tab20",
        legend: bool = True,
        **kwargs,
    ) -> "go.Figure":
        """Interactive Plotly version of :meth:`show`.

        Hover tooltips are populated from ``_hover_fields`` so subclasses
        can expose whichever fitted parameters and metrics are most
        meaningful for their model.

        Raises:
            ImportError: If ``plotly`` is not installed.
        """
        from phenotypic.tools_._plotly_helpers import _require_plotly

        _require_plotly()
        import plotly.graph_objects as go

        model_scores, measurements = self._filter_for_plot(criteria)

        if measurements.empty:
            import warnings

            warnings.warn(
                "No data found matching the criteria. Returning empty figure."
            )
            return go.Figure()

        measurements.loc[:, self.time_label] = self._ensure_float_array(
            measurements.loc[:, self.time_label]
        )

        model_groups = {
            keys: grp for keys, grp in model_scores.groupby(by=self.groupby)
        }
        meas_groups = {
            keys: grp for keys, grp in measurements.groupby(by=self.groupby)
        }

        timepoints = pd.Series(measurements.loc[:, self.time_label].unique())
        t, _ = self._time_axis(timepoints, tmax)

        _OKABE_ITO = [
            "#003660", "#E69F00", "#56B4E9",
            "#009E73", "#0072B2", "#CC79A7",
        ]
        if cmap is not None:
            try:
                cmap_obj = matplotlib.colormaps[cmap]
                colors = [
                    f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})"
                    for c in cmap_obj(
                        np.linspace(0, 1, max(len(model_groups), 1), endpoint=False)
                    )
                ]
                color_iter = itertools.cycle(colors)
            except (ValueError, KeyError):
                color_iter = itertools.cycle([cmap])
        else:
            color_iter = itertools.cycle(_OKABE_ITO)

        fig = go.Figure()

        for model_key, model_group in model_groups.items():
            curr_meas = meas_groups[model_key]
            curr_color = next(color_iter)
            row = model_group.iloc[0]

            y_pred = self.model_func(t=t, **self._predict_kwargs(row))

            if isinstance(model_key, tuple):
                label = ", ".join(str(k) for k in model_key)
            else:
                label = str(model_key)

            hover_extra = self._format_hover(row)

            fig.add_trace(go.Scatter(
                x=t,
                y=y_pred,
                mode="lines",
                name=label,
                line=dict(color=curr_color, width=2),
                legendgroup=label,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Time: %{x:.1f}<br>"
                    "Predicted: %{y:.2f}<br>"
                    f"<extra>{hover_extra}</extra>"
                ),
            ))

            curr_time_groups = curr_meas.groupby(by=self.time_label)
            curr_mean = curr_time_groups[self.on].mean()
            curr_stddev = curr_time_groups[self.on].std()
            curr_count = curr_time_groups[self.on].count()
            curr_stderr = curr_stddev / np.sqrt(curr_count)

            time_vals = curr_mean.index.values.astype(float)
            mean_vals = curr_mean.values
            stderr_vals = np.nan_to_num(curr_stderr.values, nan=0.0)

            fig.add_trace(go.Scatter(
                x=time_vals,
                y=mean_vals,
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=False,
                marker=dict(
                    color=curr_color,
                    size=7,
                    line=dict(color=curr_color, width=1),
                ),
                error_y=dict(
                    type="data",
                    array=stderr_vals,
                    visible=True,
                    color=curr_color,
                    thickness=1,
                ),
                customdata=np.column_stack([time_vals, mean_vals, stderr_vals]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Time: %{customdata[0]:.1f}<br>"
                    "Mean: %{customdata[1]:.2f}<br>"
                    "SE: %{customdata[2]:.4f}<br>"
                    "<extra></extra>"
                ),
            ))

        height_px = figsize[1] * 100

        fig.update_layout(
            autosize=True,
            height=height_px,
            title=dict(
                text=kwargs.get("title", "mean±SE"),
                font=dict(
                    family="DM Sans, system-ui, sans-serif",
                    size=13,
                    color="#003660",
                ),
            ),
            xaxis=dict(
                title=dict(
                    text=kwargs.get("xlabel", self.time_label),
                    font=dict(
                        family="DM Mono, Courier New, monospace",
                        size=9,
                        color="#2e3a4e",
                    ),
                ),
                tickfont=dict(
                    family="DM Mono, Courier New, monospace",
                    size=8,
                    color="#8892a4",
                ),
                gridcolor="#e8ecf2",
                gridwidth=1,
                linecolor="#dde3ed",
                linewidth=1.5,
                showline=True,
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(
                    text=kwargs.get("ylabel", self.on),
                    font=dict(
                        family="DM Mono, Courier New, monospace",
                        size=9,
                        color="#2e3a4e",
                    ),
                ),
                tickfont=dict(
                    family="DM Mono, Courier New, monospace",
                    size=8,
                    color="#8892a4",
                ),
                gridcolor="#e8ecf2",
                gridwidth=1,
                linecolor="#dde3ed",
                linewidth=1.5,
                showline=True,
                zeroline=False,
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f5f7fa",
            showlegend=legend,
            legend=dict(
                font=dict(
                    family="DM Sans, system-ui, sans-serif",
                    size=11,
                    color="#2e3a4e",
                ),
            ),
            hovermode="closest",
        )

        return fig
