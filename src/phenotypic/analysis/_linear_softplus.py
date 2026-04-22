from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from phenotypic.analysis.abc_ import ModelFitter
from phenotypic.tools_.measurement_info_ import (
    LINEAR_SOFTPLUS_MODEL,
    MODEL_METRICS,
)


class LinearSoftplusModel(ModelFitter):
    r"""Linear-with-softplus lag-phase growth fitter.

    The model combines a linear post-lag growth phase with a softplus lag
    transition and an optional softplus saturation ceiling:

    .. math::

        s(t) = \frac{v}{\alpha}\, \ln\!\bigl(1 + e^{\alpha(t-\lambda)}\bigr) + s_0

    When ``smax`` is provided (or inferred per-group as the observed
    maximum), a second softplus clamps the curve to the saturation
    ceiling:

    .. math::

        s(t) = s_{\max}
               - \frac{1}{\beta}\,\ln\!\bigl(1 + e^{\beta(s_{\max} - s_{\text{unclamped}}(t))}\bigr)

    Attributes:
        smax (float | None): Fixed carrying capacity. ``None`` falls back
            to per-group observed maximum.
        beta (float): Saturation transition sharpness.
        stderr_label (str | None): Column providing per-timepoint standard
            errors used as weights in the fit. When ``None``, the fit
            auto-derives a replicate-SE column during aggregation.
        inoc_size_label (str | None): Column of per-row inoculum size
            measurements. When supplied, per-group mean and sample std
            are computed automatically and used as an informative
            Gaussian prior on ``s0``; when ``None``, no prior is applied
            (appropriate for yeast where inoculum size is not imaged).
        prune_saturated (bool): Whether to drop post-saturation timepoints
            before fitting.
        saturation_threshold (float): Fraction of peak ``ds/dt`` below
            which the curve is considered saturated.
        saturation_buffer (int): Extra rows past the saturation index kept
            so the fit still sees some plateau evidence.
        v_upper (float): Upper bound on ``v`` in the optimizer.
    """

    _measurement_info_class = LINEAR_SOFTPLUS_MODEL

    def __init__(
        self,
        on: str,
        groupby: List[str],
        time_label: str = "Metadata_Time",
        agg_func: Callable | str | list | dict | None = "mean",
        *,
        smax: float | None = None,
        beta: float = 10,
        stderr_label: str | None = None,
        inoc_size_label: str | None = None,
        prune_saturated: bool = True,
        saturation_threshold: float = 0.05,
        saturation_buffer: int = 2,
        v_upper: float = 50.0,
        num_workers: int = 1,
        loss: Literal["linear"] = "linear",
        verbose: bool = False,
    ):
        """Initialize the linear-softplus fitter.

        Args:
            on: Target column (size measurement) to fit.
            groupby: Columns defining the per-fit grouping structure.
            time_label: Column name representing time. Defaults to
                ``"Metadata_Time"``.
            agg_func: Aggregation function for the ``on`` column when
                ``stderr_label`` is provided. Ignored when
                ``stderr_label is None`` because the fitter uses pandas
                named aggregation to derive mean and SE together.
                Defaults to ``"mean"``.
            smax: Fixed carrying capacity for every group. When
                ``None``, each group uses its own post-pruning observed
                maximum of ``on``.
            beta: Saturation transition sharpness.
            stderr_label: Column providing per-timepoint standard
                errors used as weights. When ``None``, replicate SE is
                computed automatically during aggregation.
            inoc_size_label: Optional column providing per-row inoculum
                size measurements (typically available for filamentous
                fungi from the t=0 image, absent for yeast). When
                supplied, per-group mean and sample standard deviation
                of the column are computed automatically via
                ``groupby.transform`` and fed into a single virtual
                Gaussian residual on ``s0`` (spec §5.3). When ``None``
                (the default, appropriate for yeast), the inoculum
                prior is omitted from the residual vector entirely.
            prune_saturated: Whether to drop post-saturation timepoints
                before fitting.
            saturation_threshold: Fraction of peak ``ds/dt`` below which
                the curve is considered saturated.
            saturation_buffer: Extra rows past the saturation index
                retained so the fit still sees plateau evidence.
            v_upper: Upper bound on ``v``.
            num_workers: Number of parallel workers for per-group fits.
            loss: Loss method passed through to
                :func:`scipy.optimize.least_squares`. Defaults to
                ``"linear"``.
            verbose: If ``True``, enables optimizer verbose output.
        """
        super().__init__(
            on=on,
            groupby=groupby,
            time_label=time_label,
            agg_func=agg_func,
            num_workers=num_workers,
            loss=loss,
            verbose=verbose,
        )
        self.smax = smax
        self.beta = beta
        self.stderr_label = stderr_label
        self.inoc_size_label = inoc_size_label
        self.prune_saturated = prune_saturated
        self.saturation_threshold = saturation_threshold
        self.saturation_buffer = saturation_buffer
        self.v_upper = v_upper

    # ------------------------------------------------------------------ #
    # Model math
    # ------------------------------------------------------------------ #
    @staticmethod
    def model_func(
        t: np.ndarray | float,
        v: float,
        s0: float,
        lam: float,
        alpha: float,
        smax: float | None = None,
        beta: float = 10,
    ) -> float | np.ndarray:
        r"""Linear-softplus growth curve with optional saturation ceiling.

        Args:
            t: Time (scalar or array).
            v: Post-lag growth rate.
            s0: Initial size.
            lam: Lag duration.
            alpha: Lag transition sharpness.
            smax: Optional carrying capacity. When ``None``, the curve
                grows linearly forever past the lag.
            beta: Saturation transition sharpness.

        Returns:
            Predicted size at ``t``; scalar when ``t`` is scalar,
            otherwise an array.
        """
        t_arr = np.asarray(t, dtype=float)
        # `logaddexp` is the numerically stable form of log(1 + exp(x)).
        softplus_lag = np.logaddexp(0.0, alpha * (t_arr - lam)) / alpha
        s_unclamped = v * softplus_lag + s0

        if smax is None:
            return s_unclamped

        softplus_sat = np.logaddexp(0.0, beta * (smax - s_unclamped)) / beta
        s_clamped = smax - softplus_sat
        return s_clamped

    # ------------------------------------------------------------------ #
    # Loss function (instance method so we can reach self.beta)
    # ------------------------------------------------------------------ #
    def _loss_func(
        self,
        params,
        t,
        y,
        sigma=None,
        smax: float | None = None,
        sGMM_mean: float | None = None,
        sGMM_sigma: float | None = None,
        **_,
    ):
        r"""Weighted residuals with optional inoculum prior.

        Residuals are the measurement-vs-model differences, optionally
        divided by per-timepoint ``sigma`` (spec §6.3). When
        ``sGMM_mean`` and ``sGMM_sigma`` are supplied (resolved
        per-group from the ``inoc_size_label`` column by
        :meth:`_extra_loss_kwargs`), a single virtual residual
        ``(s0 - sGMM_mean) / sGMM_sigma`` is appended (spec §5.3) to
        implement the Bayesian MAP-equivalent Gaussian prior on ``s0``.
        When no inoculum column is provided, the prior is omitted
        entirely — the residual vector contains only data residuals.

        Args:
            params: Optimizer vector ``[v, s0, lam, alpha]``.
            t: Time points.
            y: Observed sizes.
            sigma: Optional per-timepoint standard errors; ``None``
                yields unweighted residuals.
            smax: Per-group carrying capacity.
            sGMM_mean: Per-group inoculum-size mean (optional).
            sGMM_sigma: Per-group inoculum-size sample std (optional).

        Returns:
            Flat residual vector consumed by
            :func:`scipy.optimize.least_squares`.
        """
        v, s0, lam, alpha = params
        y_pred = self.model_func(
            t=t,
            v=v,
            s0=s0,
            lam=lam,
            alpha=alpha,
            smax=smax,
            beta=self.beta,
        )
        residuals = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
        if sigma is not None:
            residuals = residuals / np.asarray(sigma, dtype=float)

        if sGMM_mean is not None and sGMM_sigma is not None:
            prior_residual = (s0 - sGMM_mean) / sGMM_sigma
            return np.concatenate([residuals, [prior_residual]])
        return residuals

    # ------------------------------------------------------------------ #
    # Saturation pruning
    # ------------------------------------------------------------------ #
    def _prepare_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Drop post-saturation timepoints via a robust hybrid heuristic.

        The pruning rule requires **both** an amplitude criterion
        (``y >= 90% * (max - min) + min``) *and* a sustained
        sub-threshold derivative run (3 consecutive points below
        ``saturation_threshold * peak_slope``) to agree before trimming.
        The amplitude gate is structurally immune to lag-phase noise
        because the lag phase sits near ``s0``, far below the amplitude
        target. The sustained-run gate rejects transient mid-growth
        dips. A tail-growth guard short-circuits on curves that never
        saturate within the observation window.
        """
        if not self.prune_saturated or len(group) < 6:
            return group

        g = group.sort_values(self.time_label).reset_index(drop=True)
        y = g[self.on].to_numpy(dtype=float)
        t = g[self.time_label].to_numpy(dtype=float)

        window = min(5, max(3, len(y) // 4))
        dy_dt = np.gradient(y, t)
        smoothed = np.convolve(dy_dt, np.ones(window) / window, mode="same")
        peak_slope = float(smoothed.max())

        # Guard against NaN-propagated smoothed arrays (e.g. all-NaN y)
        # that would silently fall through both gates below.
        if not np.isfinite(peak_slope) or peak_slope <= 0:
            return g

        tail_window = smoothed[-window:]
        if tail_window.size == 0 or not np.any(np.isfinite(tail_window)):
            return g
        tail_slope = float(np.nanmean(tail_window))
        if tail_slope > self.saturation_threshold * peak_slope:
            return g

        peak_idx = int(np.argmax(smoothed))

        amp_target = y.min() + 0.90 * (y.max() - y.min())
        amp_mask = y >= amp_target
        amp_idx = int(np.argmax(amp_mask)) if amp_mask.any() else len(y)

        threshold = self.saturation_threshold * peak_slope
        below = smoothed[peak_idx:] < threshold
        sustained = 3
        deriv_idx: int | None = None
        run = 0
        for i, is_below in enumerate(below):
            run = run + 1 if is_below else 0
            if run >= sustained:
                deriv_idx = peak_idx + i - (sustained - 1)
                break
        if deriv_idx is None:
            return g

        sat_idx = max(amp_idx, deriv_idx)
        keep_through = min(sat_idx + self.saturation_buffer, len(g) - 1)
        return g.iloc[: keep_through + 1].copy()

    # ------------------------------------------------------------------ #
    # Per-group loss kwargs (smax, sigma, inoculum prior)
    # ------------------------------------------------------------------ #
    _INOC_MEAN_SUFFIX = "_group_mean"
    _INOC_SIGMA_SUFFIX = "_group_sigma"

    def _extra_loss_kwargs(self, group: pd.DataFrame) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"smax": self._smax_for(group)}
        sigma = self._resolve_sigma(group)
        if sigma is not None:
            kw["sigma"] = sigma
        stats = self._inoc_stats(group)
        if stats is not None:
            kw["sGMM_mean"], kw["sGMM_sigma"] = stats
        return kw

    def _inoc_stats(self, group: pd.DataFrame) -> Tuple[float, float] | None:
        """Resolve per-group inoculum mean and std, or ``None``.

        Reads the pre-computed per-group aggregate columns injected by
        :meth:`analyze`. Returns ``None`` whenever the column is not
        configured, the aggregate columns are missing/invalid, or the
        sample std is zero/non-finite — all of which would make the
        Gaussian prior undefined.
        """
        if self.inoc_size_label is None:
            return None
        mean_col = f"{self.inoc_size_label}{self._INOC_MEAN_SUFFIX}"
        sigma_col = f"{self.inoc_size_label}{self._INOC_SIGMA_SUFFIX}"
        if mean_col not in group.columns or sigma_col not in group.columns:
            return None
        mean_series = group[mean_col].dropna()
        sigma_series = group[sigma_col].dropna()
        if mean_series.empty or sigma_series.empty:
            return None
        mean = float(mean_series.iloc[0])
        sigma = float(sigma_series.iloc[0])
        if (
            not np.isfinite(mean)
            or not np.isfinite(sigma)
            or sigma <= 0
        ):
            return None
        return mean, sigma

    def _smax_for(self, group: pd.DataFrame) -> float:
        if self.smax is not None:
            return float(self.smax)
        return float(group[self.on].max())

    def _resolve_sigma(self, group: pd.DataFrame) -> np.ndarray | None:
        """Build a per-timepoint sigma vector, aligned with group rows.

        Priority order:
          1. User-supplied ``stderr_label`` column.
          2. Auto-derived replicate-SE column ``f"{on}_stderr"`` emitted
             by :meth:`_extra_agg_columns` when ``stderr_label is None``.
          3. No weights (return ``None``).
        """
        if self.stderr_label is not None and self.stderr_label in group.columns:
            raw = group[self.stderr_label].to_numpy(dtype=float)
        elif f"{self.on}_stderr" in group.columns:
            raw = group[f"{self.on}_stderr"].to_numpy(dtype=float)
        else:
            return None

        # Replace zero/NaN sigmas with a small epsilon so the weighted
        # residuals stay finite. Scale epsilon to the median positive
        # sigma so it stays commensurate with the data.
        positive = raw[(raw > 0) & np.isfinite(raw)]
        if positive.size > 0:
            eps = 1e-8 * float(np.nanmedian(np.abs(positive)))
            if eps <= 0 or not np.isfinite(eps):
                eps = 1e-8
        else:
            eps = 1e-8
        return np.where((raw > 0) & np.isfinite(raw), raw, eps)

    def _extra_agg_columns(self) -> Dict[str, Any]:
        """Carry per-timepoint stderr and per-group inoculum stats.

        - ``stderr_label`` or the auto-computed ``f"{on}_stderr"`` goes
          through with a mean aggregation so the weighted loss can
          read one SE per timepoint.
        - When ``inoc_size_label`` is configured, the two per-group
          aggregate columns injected by :meth:`analyze` are pulled
          through with ``"first"`` — they are already constant within
          each group so any reducer gives the same answer.
        """
        extras: Dict[str, Any] = {}
        if self.stderr_label is not None:
            extras[self.stderr_label] = "mean"
        else:
            extras[f"{self.on}_stderr"] = "mean"
        if self.inoc_size_label is not None:
            extras[f"{self.inoc_size_label}{self._INOC_MEAN_SUFFIX}"] = "first"
            extras[f"{self.inoc_size_label}{self._INOC_SIGMA_SUFFIX}"] = "first"
        return extras

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model to every group of ``data``.

        Pre-computes two broadcasted helper columns on the raw data
        before delegating to the base-class aggregation pipeline:

        - When ``stderr_label`` is ``None``, a replicate-SEM column
          derived via ``groupby.transform("sem")`` so the weighted
          loss can downweight noisy timepoints automatically.
        - When ``inoc_size_label`` is set, per-group mean and sample
          std of the inoculum-size column via ``groupby.transform``
          — these feed the optional Gaussian prior on ``s0``
          (:meth:`_inoc_stats`).

        Each helper is constant within its group, so the base-class
        dict-style aggregation carries it through as a flat column
        without MultiIndex juggling.
        """
        needs_copy = (
            self.stderr_label is None or self.inoc_size_label is not None
        )
        if needs_copy:
            data = data.copy(deep=True)

        if self.stderr_label is None:
            se_col = f"{self.on}_stderr"
            data[se_col] = data.groupby(
                self.groupby + [self.time_label]
            )[self.on].transform("sem")

        if self.inoc_size_label is not None:
            mean_col = f"{self.inoc_size_label}{self._INOC_MEAN_SUFFIX}"
            sigma_col = f"{self.inoc_size_label}{self._INOC_SIGMA_SUFFIX}"
            grouped = data.groupby(self.groupby)[self.inoc_size_label]
            data[mean_col] = grouped.transform("mean")
            data[sigma_col] = grouped.transform("std")

        return super().analyze(data)

    # ------------------------------------------------------------------ #
    # Required fit hooks
    # ------------------------------------------------------------------ #
    def _initial_guess(self, group: pd.DataFrame) -> list[float]:
        """Heuristic initial guess for ``[v, s0, lam, alpha]``."""
        g = group.sort_values(self.time_label)
        y = g[self.on].to_numpy(dtype=float)
        t = g[self.time_label].to_numpy(dtype=float)

        stats = self._inoc_stats(group)
        if stats is not None:
            s0_init = stats[0]
        else:
            s0_init = float(np.median(y[: max(2, len(y) // 4)]))

        cut = max(2, int(0.4 * len(y)))
        if len(y) - cut >= 2 and np.ptp(t[cut:]) > 0:
            slope = float(np.polyfit(t[cut:], y[cut:], 1)[0])
        else:
            slope = 1.0
        v_init = float(np.clip(slope, 1e-4, self.v_upper))

        y_range = max(y.max() - s0_init, 1e-6)
        crossing_mask = y > s0_init + 0.1 * y_range
        if crossing_mask.any():
            lam_init = float(t[np.argmax(crossing_mask)])
        else:
            lam_init = float(t[0])

        alpha_init = 10.0
        return [v_init, s0_init, lam_init, alpha_init]

    def _bounds(self, group: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Parameter bounds ``(lower, upper)`` for ``[v, s0, lam, alpha]``."""
        t_max = float(group[self.time_label].max())
        if t_max <= 0:
            t_max = 1.0

        stats = self._inoc_stats(group)
        if stats is not None:
            s0_upper = float(max(3.0 * stats[0], 1e-6))
        else:
            s0_upper = float(group[self.on].max()) or 1.0
            if s0_upper <= 0:
                s0_upper = 1.0

        lower = [0.0, 0.0, 0.0, 2.0]
        upper = [self.v_upper, s0_upper, t_max, 50.0]
        return lower, upper

    def _unpack_params(
        self, x: np.ndarray, group: pd.DataFrame
    ) -> Dict[Any, float]:
        v, s0, lam, alpha = (float(x[i]) for i in range(4))
        return {
            LINEAR_SOFTPLUS_MODEL.v: v,
            LINEAR_SOFTPLUS_MODEL.s0: s0,
            LINEAR_SOFTPLUS_MODEL.lam: lam,
            LINEAR_SOFTPLUS_MODEL.alpha: alpha,
            LINEAR_SOFTPLUS_MODEL.smax: self._smax_for(group),
            LINEAR_SOFTPLUS_MODEL.beta: float(self.beta),
        }

    def _predict_kwargs(self, row) -> Dict[str, float]:
        return {
            "v": row[LINEAR_SOFTPLUS_MODEL.v],
            "s0": row[LINEAR_SOFTPLUS_MODEL.s0],
            "lam": row[LINEAR_SOFTPLUS_MODEL.lam],
            "alpha": row[LINEAR_SOFTPLUS_MODEL.alpha],
            "smax": row[LINEAR_SOFTPLUS_MODEL.smax],
            "beta": row[LINEAR_SOFTPLUS_MODEL.beta],
        }

    def _hover_fields(self) -> List[Tuple[str, Any, str]]:
        return [
            ("v", LINEAR_SOFTPLUS_MODEL.v, ".4f"),
            ("s0", LINEAR_SOFTPLUS_MODEL.s0, ".3f"),
            ("lambda", LINEAR_SOFTPLUS_MODEL.lam, ".3f"),
            ("alpha", LINEAR_SOFTPLUS_MODEL.alpha, ".2f"),
            ("smax", LINEAR_SOFTPLUS_MODEL.smax, ".3f"),
            ("RMSE", MODEL_METRICS.RMSE, ".4f"),
        ]
