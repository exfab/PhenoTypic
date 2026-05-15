from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from phenotypic.analysis.abc_._linear_softplus_base import (
    _DEFAULT_BETA,
    _LinearSoftplusBase,
)
from phenotypic.analysis.abc_._model_fitter import LossKind
from phenotypic.tools_ import ColumnRef, ColumnRefList
from phenotypic.tools_.measurement_info import (
    DOUBLE_SOFTPLUS_MODEL,
    MODEL_METRICS,
)

_MODE_FIXED_BETA = "fixed_beta"
_MODE_FITTED_BETA = "fitted_beta"


class DoubleSoftplus(_LinearSoftplusBase):
    r"""Linear-softplus growth fitter with a softplus saturation ceiling.

    Fits a linear post-lag growth phase with a softplus lag transition
    *and* a softplus saturation ceiling:

    .. math::

        s_{\text{unclamped}}(t) =
            \frac{v}{\alpha}\, \ln\!\bigl(1 + e^{\alpha(t-\lambda)}\bigr) + s_0

    .. math::

        s(t) = s_{\max}
               - \frac{1}{\beta}\,\ln\!\bigl(1 + e^{\beta(s_{\max} - s_{\text{unclamped}}(t))}\bigr)

    Use this class when colonies show a clear carrying-capacity plateau
    in the observation window. For pre-saturation linear growth, use
    :class:`LinearSoftplus` instead.

    Per-group mode dispatch:
        The fit picks one of two variants per fit group, recorded in
        ``DOUBLE_SOFTPLUS_MODEL.mode``:

        - ``"fitted_beta"`` — 5-parameter fit. Triggered when ``beta``
          is ``None`` *and* a saturation shoulder is detected in the
          group (smoothed tail slope flattens below
          ``shoulder_slope_ratio`` times the peak slope).
        - ``"fixed_beta"`` — 4-parameter fit with ``beta`` held
          constant. Triggered when the user supplied an explicit
          scalar ``beta``, *or* when no shoulder is detected. The
          effective ``beta`` is ``self.beta`` when set, else the
          module default (``10.0``).

    Pruning is intentionally not exposed on this class — the saturation
    plateau IS the model, so dropping the tail would defeat the fit.

    Attributes:
        smax (float | None): Fixed carrying capacity. ``None`` falls back
            to the per-group observed max.
        beta (float | None): Saturation transition sharpness. ``None``
            (default) opts into per-group mode dispatch — fit when a
            shoulder is present, otherwise held at the module default.
            Set a positive scalar to force ``"fixed_beta"`` mode
            unconditionally.
        stderr_label (str | None): Column providing per-timepoint standard
            errors used as weights. Same semantics as :class:`LinearSoftplus`.
        s0_prior (bool | float | str | None): Unified Gaussian-prior
            source for ``s0``. Same dispatch as :class:`LinearSoftplus`.
        s0_prior_cv (float | None): CV coefficient for the prior σ
            (``σ = cv × µ``). Mutually exclusive with
            ``s0_prior_sigma``. Defaults to ``None``; if neither knob
            is set and the prior is engaged, CV=0.05 is applied.
        s0_prior_sigma (float | None): Absolute σ for the prior.
            Mutually exclusive with ``s0_prior_cv``.
        s0_prior_groupby (List[str] | None): Optional coarser grouping
            for empirical-Bayes pooling of the per-group prior ``µ``.
        shoulder_slope_ratio (float): Fraction of peak ``ds/dt`` below
            which the tail slope counts as a saturation shoulder for
            mode dispatch. Defaults to ``0.05``.
    """

    _measurement_infoclass = DOUBLE_SOFTPLUS_MODEL

    def __init__(
            self,
            on: ColumnRef,
            groupby: ColumnRefList,
            time_label: ColumnRef = "Metadata_Time",
            *,
            smax: float | None = None,
            beta: float | None = None,
            stderr_label: str | None = None,
            s0_prior: bool | float | int | str | None = None,
            s0_prior_cv: float | None = None,
            s0_prior_sigma: float | None = None,
            s0_prior_groupby: List[str] | None = None,
            shoulder_slope_ratio: float = 0.05,
            n_jobs: int = 1,
            loss: LossKind = "huber",
            f_scale: float = 1.0,
            verbose: bool = False,
    ):
        """Initialize the double-softplus fitter.

        Args:
            on: Target column (size measurement) to fit.
            groupby: Columns defining the per-fit grouping structure.
            time_label: Column name representing time. Defaults to
                ``"Metadata_Time"``.
            smax: Fixed carrying capacity for every group. When
                ``None``, the model uses the per-group observed
                maximum.
            beta: Saturation transition sharpness. ``None`` (default)
                enables per-group mode dispatch — the fitter picks
                between fitted-beta and fixed-beta-at-default based on
                whether a shoulder is detected. A positive scalar
                forces ``"fixed_beta"`` mode with that value across
                every group. Non-positive scalars raise
                :class:`ValueError`.
            stderr_label: Column providing per-timepoint standard
                errors used as weights. When ``None``, replicate SE is
                computed automatically during aggregation.
            s0_prior: Unified prior-mean source, dispatched by type.
                See :class:`LinearSoftplus` for the dispatch table.
            s0_prior_cv: CV coefficient for the prior σ
                (``σ = cv × µ``). Mutually exclusive with
                ``s0_prior_sigma``. Defaults to ``None``; if neither
                knob is set and the prior is engaged, CV=0.05 is
                applied.
            s0_prior_sigma: Absolute σ for the prior. Mutually exclusive
                with ``s0_prior_cv``.
            s0_prior_groupby: Coarser grouping (subset of ``groupby``)
                for empirical-Bayes pooling of the per-group ``µ``.
            shoulder_slope_ratio: Fraction of peak ``ds/dt`` below
                which the curve is considered to show a saturation
                shoulder for mode dispatch. Defaults to ``0.05``.
            n_jobs: Number of parallel workers for per-group fits.
            loss: Loss method for :func:`scipy.optimize.least_squares`.
            f_scale: Soft margin between inlier and outlier residuals.
            verbose: If ``True``, enables optimizer verbose output.
        """
        super().__init__(
                on=on,
                groupby=groupby,
                time_label=time_label,
                stderr_label=stderr_label,
                s0_prior=s0_prior,
                s0_prior_cv=s0_prior_cv,
                s0_prior_sigma=s0_prior_sigma,
                s0_prior_groupby=s0_prior_groupby,
                n_jobs=n_jobs,
                loss=loss,
                f_scale=f_scale,
                verbose=verbose,
        )
        if beta is not None:
            if not np.isfinite(beta) or beta <= 0:
                raise ValueError(
                        f"beta must be None or a positive finite number, "
                        f"got {beta!r}."
                )
        if (
                not np.isfinite(shoulder_slope_ratio)
                or shoulder_slope_ratio <= 0
                or shoulder_slope_ratio >= 1
        ):
            raise ValueError(
                    f"shoulder_slope_ratio must be in (0, 1), "
                    f"got {shoulder_slope_ratio!r}."
            )
        self.smax = smax
        self.beta = beta
        self.shoulder_slope_ratio = shoulder_slope_ratio

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
            smax: float,
            beta: float = _DEFAULT_BETA,
    ) -> float | np.ndarray:
        r"""Linear-softplus growth curve with softplus saturation ceiling.

        Args:
            t: Time (scalar or array).
            v: Post-lag growth rate.
            s0: Initial size.
            lam: Lag duration.
            alpha: Lag transition sharpness.
            smax: Carrying capacity (saturation ceiling).
            beta: Saturation transition sharpness.

        Returns:
            Predicted size at ``t``; scalar when ``t`` is scalar,
            otherwise an array.
        """
        unclamped = _LinearSoftplusBase._lag_softplus(
                t=t, v=v, s0=s0, lam=lam, alpha=alpha
        )
        softplus_sat = np.logaddexp(0.0, beta * (smax - unclamped)) / beta
        return smax - softplus_sat

    # ------------------------------------------------------------------ #
    # Saturation-shoulder detection and per-group mode dispatch
    # ------------------------------------------------------------------ #
    def _has_saturation_shoulder(self, group: pd.DataFrame) -> bool:
        """Return ``True`` iff ``group`` shows a saturation shoulder.

        Compares a robust peak slope (smoothed ``dy/dt`` maximum) to
        the minimum of the last few raw gradients. A shoulder is
        declared when the curve's terminal slope fell to
        ``self.shoulder_slope_ratio`` times the peak or below — a
        criterion that still fires on curves with a short
        post-saturation tail (as few as 2 plateau points), where
        tail-mean statistics would be diluted by transitional points.
        A dynamic-range guard rejects flat-noise groups that would
        otherwise trivially satisfy the ratio test.
        """
        if len(group) < 6:
            return False

        g = group.sort_values(self.time_label).reset_index(drop=True)
        y = g[self.on].to_numpy(dtype=float)
        t = g[self.time_label].to_numpy(dtype=float)

        y_finite = y[np.isfinite(y)]
        if y_finite.size == 0:
            return False
        med_abs = abs(float(np.median(y_finite))) + 1e-9
        if float(y_finite.max() - y_finite.min()) <= 0.05 * med_abs:
            return False

        dy_dt = np.gradient(y, t)

        # Peak slope: smoothed max — robust to single-point noise spikes.
        window = min(5, max(3, len(y) // 4))
        smoothed = np.convolve(dy_dt, np.ones(window) / window, mode="same")
        peak_slope = float(np.nanmax(smoothed))
        if not np.isfinite(peak_slope) or peak_slope <= 0:
            return False

        # Terminal slope: min of the last few raw gradients. Using the
        # min (not mean) keeps the criterion sensitive to short plateaus
        # where the final 2-3 points have collapsed to near-zero slope
        # but earlier tail points are still in the transition.
        tail_k = max(2, min(3, len(dy_dt) // 5))
        tail_min = float(np.nanmin(dy_dt[-tail_k:]))
        if not np.isfinite(tail_min):
            return False
        return tail_min <= self.shoulder_slope_ratio * peak_slope

    def _mode_for(self, group: pd.DataFrame) -> str:
        """Pick the fit variant for ``group``.

        Two-way dispatch:

        1. User-explicit scalar ``beta`` → ``"fixed_beta"``.
        2. Shoulder detected (``beta is None``) → ``"fitted_beta"``.
        3. Otherwise (``beta is None``, no shoulder) → ``"fixed_beta"``
           with the module-default ``β``.
        """
        if self.beta is not None:
            return _MODE_FIXED_BETA
        if self._has_saturation_shoulder(group):
            return _MODE_FITTED_BETA
        return _MODE_FIXED_BETA

    # ------------------------------------------------------------------ #
    # Per-group loss kwargs (smax, beta on top of the base σ + prior).
    # ------------------------------------------------------------------ #
    def _extra_loss_kwargs(self, group: pd.DataFrame) -> Dict[str, Any]:
        kw = super()._extra_loss_kwargs(group)
        kw["smax"] = self._smax_for(group)
        if self._mode_for(group) == _MODE_FIXED_BETA:
            kw["beta"] = (
                float(self.beta) if self.beta is not None else _DEFAULT_BETA
            )
        # In fitted-beta mode the 5th optimizer entry supplies beta
        # directly; no kwarg needed.
        return kw

    def _smax_for(self, group: pd.DataFrame) -> float:
        if self.smax is not None:
            return float(self.smax)
        return float(group[self.on].max())

    # ------------------------------------------------------------------ #
    # Required fit hooks — 4-or-5 vector depending on mode.
    # ------------------------------------------------------------------ #
    def _initial_guess(self, group: pd.DataFrame) -> list[float]:
        """Heuristic initial guess for the per-group optimizer vector.

        Returns ``[v, s0, lam, alpha]`` in fixed-beta mode, or
        ``[v, s0, lam, alpha, beta]`` in fitted-beta mode. The fifth
        entry seeds at the module-default ``beta``.
        """
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
        v_init = float(np.clip(slope, 1e-4, self._V_UPPER))

        y_range = max(y.max() - s0_init, 1e-6)
        crossing_mask = y > s0_init + 0.1 * y_range
        if crossing_mask.any():
            lam_init = float(t[np.argmax(crossing_mask)])
        else:
            lam_init = float(t[0])

        alpha_init = 10.0
        guess = [v_init, s0_init, lam_init, alpha_init]
        if self._mode_for(group) == _MODE_FITTED_BETA:
            guess.append(_DEFAULT_BETA)
        return guess

    def _bounds(self, group: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Parameter bounds ``(lower, upper)``.

        Four entries in fixed-beta mode; a fifth ``(2.0, 50.0)`` pair
        appended in fitted-beta mode, matching the ``alpha`` range
        convention (softplus is ~linear below 2 and effectively a
        hard step above 50).
        """
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
        upper = [self._V_UPPER, s0_upper, t_max, 50.0]
        if self._mode_for(group) == _MODE_FITTED_BETA:
            lower.append(2.0)
            upper.append(50.0)
        return lower, upper

    def _unpack_params(
            self, x: np.ndarray, group: pd.DataFrame
    ) -> Dict[Any, Any]:
        v, s0, lam, alpha = (float(x[i]) for i in range(4))
        smax_val = self._smax_for(group)
        # Derive mode from the optimizer-vector length. The vector
        # length was set by ``_initial_guess`` (which called
        # ``_mode_for``); reading it back here is both faster than
        # re-running shoulder detection and structurally consistent
        # with whatever mode set the bounds and x0 — no risk of the
        # detector flipping between calls if a subclass mutates the
        # group in-place.
        if len(x) >= 5:
            mode = _MODE_FITTED_BETA
            beta_val = float(x[4])
        else:
            mode = _MODE_FIXED_BETA
            beta_val = (
                float(self.beta) if self.beta is not None else _DEFAULT_BETA
            )

        return {
            DOUBLE_SOFTPLUS_MODEL.v    : v,
            DOUBLE_SOFTPLUS_MODEL.s0   : s0,
            DOUBLE_SOFTPLUS_MODEL.lam  : lam,
            DOUBLE_SOFTPLUS_MODEL.alpha: alpha,
            DOUBLE_SOFTPLUS_MODEL.smax : smax_val,
            DOUBLE_SOFTPLUS_MODEL.beta : beta_val,
            DOUBLE_SOFTPLUS_MODEL.mode : mode,
        }

    def _predict_kwargs(self, row) -> Dict[str, Any]:
        smax_val = row[DOUBLE_SOFTPLUS_MODEL.smax]
        beta_val = row[DOUBLE_SOFTPLUS_MODEL.beta]
        return {
            "v"    : float(row[DOUBLE_SOFTPLUS_MODEL.v]),
            "s0"   : float(row[DOUBLE_SOFTPLUS_MODEL.s0]),
            "lam"  : float(row[DOUBLE_SOFTPLUS_MODEL.lam]),
            "alpha": float(row[DOUBLE_SOFTPLUS_MODEL.alpha]),
            "smax" : float(smax_val),
            "beta" : _DEFAULT_BETA if pd.isna(beta_val) else float(beta_val),
        }

    def _hover_fields(self) -> List[Tuple[str, Any, str]]:
        return [
            ("v", DOUBLE_SOFTPLUS_MODEL.v, ".4f"),
            ("s0", DOUBLE_SOFTPLUS_MODEL.s0, ".3f"),
            ("lambda", DOUBLE_SOFTPLUS_MODEL.lam, ".3f"),
            ("alpha", DOUBLE_SOFTPLUS_MODEL.alpha, ".2f"),
            ("smax", DOUBLE_SOFTPLUS_MODEL.smax, ".3f"),
            ("beta", DOUBLE_SOFTPLUS_MODEL.beta, ".2f"),
            ("mode", DOUBLE_SOFTPLUS_MODEL.mode, ""),
            ("RMSE", MODEL_METRICS.RMSE, ".4f"),
        ]


DoubleSoftplus.__doc__ = DOUBLE_SOFTPLUS_MODEL.append_rst_to_doc(DoubleSoftplus)
