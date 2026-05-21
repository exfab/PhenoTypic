from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pandas as pd

from phenotypic.analysis.abc_._linear_softplus_base import _LinearSoftplusBase
from phenotypic.tools_.measurement_info import (
    LINEAR_SOFTPLUS_MODEL,
    MODEL_METRICS,
)


class LinearSoftplus(_LinearSoftplusBase):
    r"""Linear-with-softplus lag-phase growth fitter (no saturation).

    Fits a 4-parameter linear post-lag growth model with a softplus lag
    transition:

    .. math::

        s(t) = \frac{v}{\alpha}\, \ln\!\bigl(1 + e^{\alpha(t-\lambda)}\bigr) + s_0

    Use this class when colonies are still in the linear-growth regime
    or when you want the saturation tail discarded as observation noise.
    For data with a clear carrying-capacity plateau, use
    :class:`DoubleSoftplus` instead.

    Pruning is ON by default — post-saturation timepoints are dropped
    from the fit so the linear regime is recovered cleanly. Disable with
    ``prune_saturated=False`` if your data is fully pre-saturation.

    Attributes:
        stderr_label (str | None): Column providing per-timepoint standard
            errors used as weights in the fit. When ``None``, the fit
            auto-derives a replicate-SE column during aggregation *and*
            a per-fit-group pooled point-level std (median across the
            n≥2 timepoints' stds) that fills σ for any n=1 timepoints
            in the group. This keeps single-replicate rows from
            dominating the 1/σ² weighting — they get σ ≈ typical point
            noise instead of ε.
        s0_prior (bool | float | str | None): Unified Gaussian-prior
            source for ``s0``. Dispatch (by type):

            - ``None`` or ``False`` → no prior (default).
            - ``True`` → ground on data: ``µ`` = median of ``self.on``
              at the earliest observed timepoint within the effective
              group.
            - ``str`` → ground on named column: ``µ`` = median of
              ``data[s0_prior]`` at the earliest timepoint within the
              effective group.
            - positive ``float`` / ``int`` → scalar prior mean applied
              uniformly to every fit group.
        s0_prior_cv (float | None): CV coefficient for the prior σ
            (``σ = cv × µ``). Mutually exclusive with
            ``s0_prior_sigma``. Defaults to ``None``; if both
            ``s0_prior_cv`` and ``s0_prior_sigma`` are ``None`` and the
            prior is engaged, the helper applies CV=0.05 as a moderately
            informative default.
        s0_prior_sigma (float | None): Absolute σ for the prior.
            Mutually exclusive with ``s0_prior_cv``. Use when the data
            scale makes a CV-based σ awkward (e.g. fractional /
            normalized data where ``µ < 1``).
        s0_prior_groupby (List[str] | None): Optional coarser grouping
            (must be a subset of ``groupby``) used for the per-group
            ``µ`` estimation on column-backed priors. When supplied,
            ``µ`` is pooled across replicate fits within each coarser
            group — an empirical-Bayes move appropriate when
            inoculation spread varies across conditions (e.g. per
            media). Only meaningful when ``s0_prior`` is ``True`` or a
            string.
        prune_saturated (bool): Whether to drop post-saturation timepoints
            before fitting. Defaults to ``True``.
    """

    _measurement_infoclass = LINEAR_SOFTPLUS_MODEL

    _PRUNE_SLOPE_RATIO: ClassVar[float] = 0.05
    """Fraction of peak ``ds/dt`` below which a tail point counts as
    plateau for pruning. Subclass override to change pruning sensitivity."""

    _PRUNE_BUFFER: ClassVar[int] = 2
    """Extra rows past the saturation index kept so the fit still sees
    some plateau evidence. Subclass override to tune."""

    prune_saturated: bool = True

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
    ) -> float | np.ndarray:
        r"""Linear-softplus growth curve, no saturation ceiling.

        .. math::
            s(t) = \frac{v}{\alpha}\, \ln\!\bigl(1 + e^{\alpha(t-\lambda)}\bigr) + s_0

        Args:
            t: Time (scalar or array).
            v: Post-lag growth rate.
            s0: Initial size.
            lam: Lag duration.
            alpha: Lag transition sharpness.

        Returns:
            Predicted size at ``t``; scalar when ``t`` is scalar,
            otherwise an array.
        """
        return _LinearSoftplusBase._lag_softplus(
                t=t, v=v, s0=s0, lam=lam, alpha=alpha
        )

    # ------------------------------------------------------------------ #
    # Saturation pruning — drops the post-saturation tail before fitting.
    # ------------------------------------------------------------------ #
    def _prepare_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Drop post-saturation timepoints via a robust hybrid heuristic.

        The pruning rule requires **both** an amplitude criterion
        (``y >= 90% * (max - min) + min``) *and* a sustained
        sub-threshold derivative run (3 consecutive points below
        ``_PRUNE_SLOPE_RATIO * peak_slope``) to agree before trimming.
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
        if tail_slope > self._PRUNE_SLOPE_RATIO * peak_slope:
            return g

        peak_idx = int(np.argmax(smoothed))

        amp_target = y.min() + 0.90 * (y.max() - y.min())
        amp_mask = y >= amp_target
        amp_idx = int(np.argmax(amp_mask)) if amp_mask.any() else len(y)

        threshold = self._PRUNE_SLOPE_RATIO * peak_slope
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
        keep_through = min(sat_idx + self._PRUNE_BUFFER, len(g) - 1)
        return g.iloc[: keep_through + 1].copy()

    # ------------------------------------------------------------------ #
    # Required fit hooks — 4-vector for v/s0/λ/α only.
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
        v_init = float(np.clip(slope, 1e-4, self._V_UPPER))

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
        upper = [self._V_UPPER, s0_upper, t_max, 50.0]
        return lower, upper

    def _unpack_params(
            self, x: np.ndarray, group: pd.DataFrame
    ) -> Dict[Any, Any]:
        v, s0, lam, alpha = (float(x[i]) for i in range(4))
        return {
            LINEAR_SOFTPLUS_MODEL.v    : v,
            LINEAR_SOFTPLUS_MODEL.s0   : s0,
            LINEAR_SOFTPLUS_MODEL.lam  : lam,
            LINEAR_SOFTPLUS_MODEL.alpha: alpha,
        }

    def _predict_kwargs(self, row) -> Dict[str, Any]:
        return {
            "v"    : float(row[LINEAR_SOFTPLUS_MODEL.v]),
            "s0"   : float(row[LINEAR_SOFTPLUS_MODEL.s0]),
            "lam"  : float(row[LINEAR_SOFTPLUS_MODEL.lam]),
            "alpha": float(row[LINEAR_SOFTPLUS_MODEL.alpha]),
        }

    def _hover_fields(self) -> List[Tuple[str, Any, str]]:
        return [
            ("v", LINEAR_SOFTPLUS_MODEL.v, ".4f"),
            ("s0", LINEAR_SOFTPLUS_MODEL.s0, ".3f"),
            ("lambda", LINEAR_SOFTPLUS_MODEL.lam, ".3f"),
            ("alpha", LINEAR_SOFTPLUS_MODEL.alpha, ".2f"),
            ("RMSE", MODEL_METRICS.RMSE, ".4f"),
        ]


LinearSoftplus.__doc__ = LINEAR_SOFTPLUS_MODEL.append_rst_to_doc(LinearSoftplus)
