from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from phenotypic.analysis.abc_ import ModelFitter
from phenotypic.analysis.abc_._model_fitter import LossKind
from phenotypic.tools_.measurement_info import (
    LINEAR_SOFTPLUS_MODEL,
    MODEL_METRICS,
)

_DEFAULT_BETA = 10.0
_MODE_UNCLAMPED = "unclamped"
_MODE_FIXED_BETA = "fixed_beta"
_MODE_FITTED_BETA = "fitted_beta"


class _InoculumPrior:
    """Gaussian prior on ``s0`` for :class:`LinearSoftplusModel`.

    Resolves ``(µ, σ)`` per fit group. The public model dispatches
    user input (the polymorphic ``s0_prior`` and the numeric
    ``s0_prior_factor``) into this helper's internal fields:

    - ``label`` — column from which per-group ``µ`` is derived.
      ``s0_prior=True`` resolves to ``on_column``; ``s0_prior="<str>"``
      resolves to that column name.
    - ``direct_mean`` — scalar ``µ`` applied uniformly
      (``s0_prior=<float>`` path).
    - ``direct_sigma`` — absolute ``σ`` override
      (``s0_prior_factor > 1`` path).
    - ``cv`` — CV coefficient for ``σ = cv × µ``
      (``s0_prior_factor ≤ 1`` path). Exactly one of ``direct_sigma``
      and ``cv`` is set per instance.
    - ``groupby`` — coarser grouping for empirical-Bayes ``µ``
      estimation on column-backed priors.

    See :class:`LinearSoftplusModel` for the user-facing dispatch
    tables and usage.
    """

    _MEAN_SUFFIX = "_group_mean"

    def __init__(
            self,
            *,
            s0_prior: bool | float | int | str | None,
            s0_prior_factor: float,
            s0_prior_groupby: List[str] | None,
            on_column: str,
    ):
        label: str | None = None
        direct_mean: float | None = None

        # Type-dispatch the prior source. Order matters — ``bool`` is a
        # subclass of ``int``, so the ``isinstance(s0_prior, bool)``
        # check must come before the numeric branch.
        if s0_prior is None:
            pass  # disabled
        elif isinstance(s0_prior, bool):
            if s0_prior:
                label = on_column
            # False: disabled (same as None).
        elif isinstance(s0_prior, str):
            label = s0_prior
        elif isinstance(s0_prior, (int, float)):
            if not np.isfinite(s0_prior) or s0_prior <= 0:
                raise ValueError(
                        f"s0_prior scalar must be a positive finite "
                        f"number, got {s0_prior!r}."
                )
            direct_mean = float(s0_prior)
        else:
            raise TypeError(
                    f"s0_prior must be bool, float, int, str, or None; "
                    f"got {type(s0_prior).__name__}."
            )

        # Value-dispatch the σ/CV knob.
        if not np.isfinite(s0_prior_factor) or s0_prior_factor <= 0:
            raise ValueError(
                    f"s0_prior_factor must be a positive finite number, "
                    f"got {s0_prior_factor!r}."
            )
        direct_sigma: float | None
        cv: float | None
        if s0_prior_factor > 1.0:
            direct_sigma = float(s0_prior_factor)
            cv = None
        else:
            direct_sigma = None
            cv = float(s0_prior_factor)

        # Validate the optional coarser grouping.
        if s0_prior_groupby is not None:
            if label is None:
                raise ValueError(
                        "s0_prior_groupby requires a column-backed prior "
                        "— set s0_prior=True or s0_prior=<column_name>."
                )
            if len(s0_prior_groupby) == 0:
                raise ValueError(
                        "s0_prior_groupby must not be an empty list — "
                        "pass ``None`` to fall back to the fit groupby."
                )

        self.label = label
        self.direct_mean = direct_mean
        self.direct_sigma = direct_sigma
        self.cv = cv
        self.groupby = (
            list(s0_prior_groupby) if s0_prior_groupby is not None else None
        )

    @property
    def is_configured(self) -> bool:
        """``True`` when the prior will attempt to engage at fit time."""
        return self.label is not None or self.direct_mean is not None

    @property
    def needs_broadcast(self) -> bool:
        """``True`` when :meth:`prepare` will mutate the dataframe."""
        return self.label is not None and self.direct_mean is None

    @property
    def mean_col(self) -> str | None:
        """Broadcast column name for column-based priors, else ``None``."""
        if self.label is None:
            return None
        return f"{self.label}{self._MEAN_SUFFIX}"

    def validate(
            self,
            data_columns: Iterable[str],
            fit_groupby: List[str],
    ) -> None:
        """Raise :class:`ValueError` on data-dependent misconfiguration.

        Called once at ``analyze`` time, before the prepare step, to
        fail loud rather than silently skip the prior.
        """
        cols = set(data_columns)
        if self.label is not None and self.label not in cols:
            raise ValueError(
                    f"inoc_size_label {self.label!r} not present in data columns."
            )
        if self.groupby is not None:
            if not set(self.groupby).issubset(set(fit_groupby)):
                raise ValueError(
                        f"inoc_groupby {self.groupby!r} must be a subset of "
                        f"groupby {list(fit_groupby)!r} — the prior group "
                        f"must be equal or coarser than the fit group."
                )
            missing = [c for c in self.groupby if c not in cols]
            if missing:
                raise ValueError(
                        f"inoc_groupby columns not in data: {missing!r}."
                )

    def prepare(
            self,
            data: pd.DataFrame,
            fit_groupby: List[str],
            time_label: str,
    ) -> pd.DataFrame:
        """Broadcast the per-group prior mean onto ``data`` as a new column.

        For each effective group (``self.groupby`` if set, otherwise
        ``fit_groupby``), computes the median of ``label`` values at
        ``time_label == min(time_label within group)`` and broadcasts
        the scalar median onto every row of the group under
        :attr:`mean_col`.

        Mutates ``data`` in place (via column assignment) when the
        prior is column-configured, and returns the same object.
        No-op when the prior is not configured or uses a scalar
        ``direct_mean`` (the same ``data`` reference is returned
        unchanged). Callers who need an un-mutated view should copy
        ``data`` before calling — see :attr:`needs_broadcast` to gate
        the copy.
        """
        if not self.needs_broadcast:
            return data

        effective = (
            list(self.groupby)
            if self.groupby is not None
            else list(fit_groupby)
        )
        mean_col = self.mean_col
        assert mean_col is not None  # label is not None above

        t_min = data.groupby(effective)[time_label].transform("min")
        masked_label = data[self.label].where(data[time_label] == t_min)
        data[mean_col] = masked_label.groupby(
                [data[c] for c in effective]
        ).transform("median")
        return data

    def stats_for(
            self, group: pd.DataFrame
    ) -> Tuple[float, float] | None:
        """Resolve ``(µ, σ)`` for one fit group, or ``None``.

        Returns ``None`` when the prior is not configured, when
        column-based ``µ`` is unavailable for the group, or when the
        resulting ``σ`` is non-finite or non-positive.
        """
        if self.direct_mean is not None:
            mu = float(self.direct_mean)
        elif self.label is not None:
            mean_col = self.mean_col
            assert mean_col is not None
            if mean_col not in group.columns:
                return None
            series = group[mean_col].dropna()
            if series.empty:
                return None
            mu = float(series.iloc[0])
        else:
            return None

        if not np.isfinite(mu):
            return None
        sigma = self._sigma_for_mean(mu)
        if sigma is None:
            return None
        return mu, sigma

    def _sigma_for_mean(self, mu: float) -> float | None:
        """Apply the direct-σ override or fall back to ``cv × µ``.

        Exactly one of ``self.direct_sigma`` and ``self.cv`` is set
        after construction, so reaching the ``else`` branch implies
        ``self.cv is not None``.
        """
        if self.direct_sigma is not None:
            sigma = self.direct_sigma
        else:
            assert self.cv is not None
            sigma = self.cv * mu
        if not np.isfinite(sigma) or sigma <= 0:
            return None
        return sigma

    def extra_agg_columns(self) -> Dict[str, str]:
        """Per-timepoint aggregation entries for the broadcast column."""
        if self.label is None or self.direct_mean is not None:
            return {}
        mean_col = self.mean_col
        assert mean_col is not None
        return {mean_col: "first"}


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

    Per-group mode dispatch:
        The fit picks one of three variants per fit group, recorded in
        ``LINEAR_SOFTPLUS_MODEL.mode``:

        - ``"fitted_beta"`` — 5-parameter fit. Triggered when ``beta`` is
          ``None`` *and* a saturation shoulder is detected in the group
          (smoothed tail slope flattens below ``saturation_threshold``
          times the peak slope). ``smax`` is the user-provided value or
          the per-group observed max.
        - ``"fixed_beta"`` — 4-parameter fit with ``beta`` held constant.
          Triggered when the user supplied an explicit scalar ``beta``,
          *or* when no shoulder is detected but ``smax`` is provided.
          The effective ``beta`` is ``self.beta`` when set, else the
          module default (``10.0``).
        - ``"unclamped"`` — 4-parameter fit with no saturation term.
          Triggered when ``beta`` is ``None``, no shoulder is detected,
          and ``smax`` is ``None``. ``smax`` and ``beta`` are reported
          as NaN. Chosen to avoid silently pinning ``smax`` to an
          observed max that does not reflect a real carrying capacity.

    Attributes:
        smax (float | None): Fixed carrying capacity. ``None`` falls back
            to per-group observed max in clamped modes, or triggers the
            unclamped variant when combined with ``beta=None`` and no
            detected shoulder.
        beta (float | None): Saturation transition sharpness. ``None``
            (default) opts into per-group mode dispatch — fit when a
            shoulder is present, otherwise held at the module default.
            Set a positive scalar to force ``"fixed_beta"`` mode
            unconditionally.
        stderr_label (str | None): Column providing per-timepoint standard
            errors used as weights in the fit. When ``None``, the fit
            auto-derives a replicate-SE column during aggregation.
        stderr_floor_quantile (float | None): Lower bound on the
            per-timepoint σ used in the weighted loss, expressed as a
            quantile in ``(0, 1]`` of the group's finite, positive σ
            values. Defaults to ``0.25`` — σ values below the 25th
            percentile of the group are lifted up to it before the
            residuals are divided through, capping the 1/σ² weight
            ratio between the softest and stiffest points at 16×.
            Pass ``None`` to disable the floor entirely and recover
            raw inverse-variance weighting (useful when the supplied
            ``stderr_label`` is already known to be trustworthy across
            its full range). The floor guards against the main
            pathology of replicate-SEM weighting: coincidentally-
            agreeing replicates whose SEM collapses toward zero by
            chance, pinning the fit to a handful of "lucky" points
            with no real precision advantage. Groups with *no* finite
            positive σ at all (e.g. singleton replicates where SEM is
            NaN everywhere, or noise-free synthetic data where SEM is
            zero everywhere) bypass the weighting pathway entirely
            and fall back to an unweighted residual — the floor is
            inapplicable in that case. Applies to both user-supplied
            ``stderr_label`` columns and the auto-derived
            replicate-SEM column.
        s0_prior (bool | float | str | None): Unified Gaussian-prior
            source for ``s0``. The prior is engaged when this is
            ``True``, a positive scalar, or a column name; disabled
            when ``None`` or ``False``. Dispatch (by type):

            - ``None`` or ``False`` → no prior (default).
            - ``True`` → ground on data: ``µ`` = median of ``self.on``
              at the earliest observed timepoint within the effective
              group.
            - ``str`` → ground on named column: ``µ`` = median of
              ``data[s0_prior]`` at the earliest timepoint within the
              effective group.
            - positive ``float`` / ``int`` → scalar prior mean applied
              uniformly to every fit group.
        s0_prior_factor (float): Unified σ/CV knob for the prior.
            Values ``> 1`` are treated as absolute σ (``σ = factor``);
            values ``≤ 1`` are treated as CV coefficients
            (``σ = factor × µ``). Defaults to ``0.05`` — a moderately
            informative "tight zone" prior that keeps ``s0`` within
            roughly ±5% of ``µ``.

            The dispatch assumes inoculum magnitudes are ``≫ 1`` (pixel
            counts, areas) and CV coefficients are ``≪ 1``. For
            normalized/fractional data where ``µ < 1``, a value
            intended as absolute σ may be silently reinterpreted as a
            CV — specify ``factor`` above ``1`` (the domain convention)
            to force the σ branch, or rescale the data.
        s0_prior_groupby (List[str] | None): Optional coarser grouping
            (must be a subset of ``groupby``) used for the per-group
            ``µ`` estimation on column-backed priors. When supplied,
            ``µ`` is pooled across replicate fits within each coarser
            group — an empirical-Bayes move appropriate when
            inoculation spread varies across conditions (e.g. per
            media). Only meaningful when ``s0_prior`` is ``True`` or a
            string.
        prune_saturated (bool): Whether to drop post-saturation timepoints
            before fitting.
        saturation_threshold (float): Fraction of peak ``ds/dt`` below
            which the curve is considered saturated.
        saturation_buffer (int): Extra rows past the saturation index kept
            so the fit still sees some plateau evidence.
        v_upper (float): Upper bound on ``v`` in the optimizer.
    """

    _measurement_infoclass = LINEAR_SOFTPLUS_MODEL

    def __init__(
            self,
            on: str,
            groupby: List[str],
            time_label: str = "Metadata_Time",
            agg_func: Callable | str | list | dict | None = "mean",
            *,
            smax: float | None = None,
            beta: float | None = None,
            stderr_label: str | None = None,
            stderr_floor_quantile: float | None = 0.25,
            s0_prior: bool | float | int | str | None = None,
            s0_prior_factor: float = 0.05,
            s0_prior_groupby: List[str] | None = None,
            prune_saturated: bool = True,
            saturation_threshold: float = 0.05,
            saturation_buffer: int = 2,
            v_upper: float = 50.0,
            num_workers: int = 1,
            loss: LossKind = "huber",
            f_scale: float = 1.0,
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
                ``None``, the model either uses the per-group observed
                maximum (clamped modes) or drops the saturation term
                entirely when no shoulder is detected (unclamped mode —
                see class docstring).
            beta: Saturation transition sharpness. ``None`` (default)
                enables per-group mode dispatch — the fitter picks
                between fitted-beta, fixed-beta (at ``10.0``), and
                unclamped variants based on the presence of a
                saturation shoulder and whether ``smax`` is provided.
                A positive scalar forces ``"fixed_beta"`` mode with
                that value across every group. Non-positive scalars
                raise :class:`ValueError`.
            stderr_label: Column providing per-timepoint standard
                errors used as weights. When ``None``, replicate SE is
                computed automatically during aggregation.
            stderr_floor_quantile: Quantile in ``(0, 1]`` used to
                floor the per-timepoint σ before the weighted loss
                divides through. Defaults to ``0.25``, which caps the
                weight ratio between the softest and stiffest points
                at 16× and neutralizes coincidentally-tiny SEM entries
                (see class docstring for the all-NaN / all-zero σ
                fallback behavior). Pass ``None`` to disable the floor
                and recover raw inverse-variance weighting. Raises
                :class:`ValueError` on values outside ``(0, 1]``.
            s0_prior: Unified prior-mean source, dispatched by type:

                - ``None`` / ``False``: no prior (default).
                - ``True``: ground on data — ``µ`` is the median of
                  ``self.on`` at the earliest observed timepoint
                  within the effective group.
                - ``str``: ground on the named column — ``µ`` is the
                  median of ``data[s0_prior]`` at the earliest
                  timepoint.
                - positive ``int`` / ``float``: scalar ``µ`` applied
                  uniformly.

                Raises :class:`TypeError` on anything else, and
                :class:`ValueError` on non-positive scalars.
            s0_prior_factor: σ/CV knob, dispatched by value. Values
                ``> 1`` are treated as absolute σ (``σ = factor``);
                values ``≤ 1`` are treated as CV coefficients
                (``σ = factor × µ``). Must be positive and finite.
                Defaults to ``0.05``. See class docstring for the
                small-``µ`` caveat.
            s0_prior_groupby: Coarser grouping (subset of ``groupby``)
                for empirical-Bayes pooling of the per-group ``µ``.
                Only meaningful when ``s0_prior`` is ``True`` or a
                string. Subset-of-``groupby`` constraint is verified
                at ``analyze`` time.
            prune_saturated: Whether to drop post-saturation timepoints
                before fitting.
            saturation_threshold: Fraction of peak ``ds/dt`` below which
                the curve is considered saturated.
            saturation_buffer: Extra rows past the saturation index
                retained so the fit still sees plateau evidence.
            v_upper: Upper bound on ``v``.
            num_workers: Number of parallel workers for per-group fits.
            loss: Loss method passed through to
                :func:`scipy.optimize.least_squares`. One of
                ``"linear"``, ``"soft_l1"``, ``"huber"``, ``"cauchy"``,
                ``"arctan"``. Defaults to ``"huber"`` — behaves like
                standard least-squares on inlier residuals (below
                ``f_scale``) but downweights large residuals from
                outlier timepoints (bubble artifacts, contamination
                spikes, mis-segmented frames), so a handful of bad
                points can't drag the fit. Pass ``"linear"`` for the
                classical sum-of-squared-residuals loss, or
                ``"soft_l1"`` / ``"cauchy"`` / ``"arctan"`` for
                progressively more aggressive outlier suppression.
            f_scale: Soft margin between inlier and outlier residuals.
                Residuals well below ``f_scale`` behave like the
                standard squared loss; residuals well above it are
                downweighted according to the chosen robust ``loss``.
                No effect when ``loss="linear"``. Must be positive;
                defaults to ``1.0``. For weighted fits where residuals
                are already pre-scaled by ``y_stderr``, ``f_scale`` is
                a multiple of a "typical standard error" — e.g.
                ``f_scale=3`` treats points more than ~3 σ off the
                curve as outliers.
            verbose: If ``True``, enables optimizer verbose output.
        """
        super().__init__(
                on=on,
                groupby=groupby,
                time_label=time_label,
                agg_func=agg_func,
                num_workers=num_workers,
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
        if stderr_floor_quantile is not None:
            if (
                not np.isfinite(stderr_floor_quantile)
                or stderr_floor_quantile <= 0
                or stderr_floor_quantile > 1
            ):
                raise ValueError(
                    f"stderr_floor_quantile must be None or in (0, 1], "
                    f"got {stderr_floor_quantile!r}."
                )
        self.smax = smax
        self.beta = beta
        self.stderr_label = stderr_label
        self.stderr_floor_quantile = stderr_floor_quantile
        self.s0_prior = s0_prior
        self.s0_prior_factor = s0_prior_factor
        self.s0_prior_groupby = s0_prior_groupby
        self.prune_saturated = prune_saturated
        self.saturation_threshold = saturation_threshold
        self.saturation_buffer = saturation_buffer
        self.v_upper = v_upper

        self._prior = _InoculumPrior(
                s0_prior=s0_prior,
                s0_prior_factor=s0_prior_factor,
                s0_prior_groupby=s0_prior_groupby,
                on_column=on,
        )

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
    # Loss function (instance method — per-group mode dispatch)
    # ------------------------------------------------------------------ #
    def _loss_func(
            self,
            params,
            t,
            y,
            y_stderr=None,
            smax: float | None = None,
            beta: float | None = None,
            s0_prior_mean: float | None = None,
            s0_prior_sigma: float | None = None,
            **_,
    ):
        r"""Weighted residuals with optional inoculum prior on ``s0``.

        Residuals are the measurement-vs-model differences, optionally
        divided by per-timepoint ``y_stderr``. When ``s0_prior_mean`` and
        ``s0_prior_sigma`` are supplied (resolved per-group by
        :meth:`_extra_loss_kwargs`), a virtual residual
        ``(s0 - s0_prior_mean) / s0_prior_sigma`` is appended to
        implement the MAP-equivalent Gaussian prior on ``s0``.

        The optimizer vector is 4 or 5 elements depending on the
        per-group mode (see :meth:`_mode_for`). In ``"fitted_beta"``
        mode the 5th element is ``beta`` and overrides whatever ``beta``
        kwarg was injected by :meth:`_extra_loss_kwargs`. In the other
        two modes the kwarg carries the effective ``beta`` (unused when
        ``smax`` is ``None``, i.e. unclamped mode).

        Args:
            params: Optimizer vector ``[v, s0, lam, alpha]`` or
                ``[v, s0, lam, alpha, beta]``.
            t: Time points.
            y: Observed sizes.
            y_stderr: Optional per-timepoint standard errors on the
                observations; ``None`` yields unweighted residuals.
            smax: Per-group carrying capacity, or ``None`` for
                unclamped.
            beta: Fixed ``beta`` for clamped-fixed mode; ignored when
                ``params`` supplies a 5th entry; ignored when ``smax``
                is ``None``.
            s0_prior_mean: Mean of the Gaussian prior on ``s0``
                (optional).
            s0_prior_sigma: Standard deviation of the Gaussian prior
                on ``s0`` (optional).

        Returns:
            Flat residual vector consumed by
            :func:`scipy.optimize.least_squares`.
        """
        if len(params) >= 5:
            v, s0, lam, alpha = (float(params[i]) for i in range(4))
            beta_eff: float = float(params[4])
        else:
            v, s0, lam, alpha = (float(params[i]) for i in range(4))
            beta_eff = float(beta) if beta is not None else _DEFAULT_BETA
        y_pred = self.model_func(
                t=t,
                v=v,
                s0=s0,
                lam=lam,
                alpha=alpha,
                smax=smax,
                beta=beta_eff,
        )
        residuals = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
        if y_stderr is not None:
            residuals = residuals / np.asarray(y_stderr, dtype=float)

        if s0_prior_mean is not None and s0_prior_sigma is not None:
            prior_residual = (s0 - s0_prior_mean) / s0_prior_sigma
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
    # Saturation-shoulder detection and per-group mode dispatch
    # ------------------------------------------------------------------ #
    def _has_saturation_shoulder(self, group: pd.DataFrame) -> bool:
        """Return ``True`` iff ``group`` shows a saturation shoulder.

        Compares a robust peak slope (smoothed ``dy/dt`` maximum) to
        the minimum of the last few raw gradients. A shoulder is
        declared when the curve's terminal slope fell to
        ``saturation_threshold`` times the peak or below — a criterion
        that still fires on curves with a short post-saturation tail
        (as few as 2 plateau points), where tail-mean statistics would
        be diluted by transitional points. A dynamic-range guard
        rejects flat-noise groups that would otherwise trivially
        satisfy the ratio test.
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
        return tail_min <= self.saturation_threshold * peak_slope

    def _mode_for(self, group: pd.DataFrame) -> str:
        """Pick the fit variant for ``group``.

        Priority:
          1. User-explicit scalar ``beta`` → ``"fixed_beta"``.
          2. Shoulder detected → ``"fitted_beta"``.
          3. ``smax`` provided (no shoulder) → ``"fixed_beta"`` with the
             module default ``beta``.
          4. Otherwise → ``"unclamped"``.
        """
        if self.beta is not None:
            return _MODE_FIXED_BETA
        if self._has_saturation_shoulder(group):
            return _MODE_FITTED_BETA
        if self.smax is not None:
            return _MODE_FIXED_BETA
        return _MODE_UNCLAMPED

    # ------------------------------------------------------------------ #
    # Per-group loss kwargs (smax, beta, y_stderr, s0 prior)
    # ------------------------------------------------------------------ #
    def _extra_loss_kwargs(self, group: pd.DataFrame) -> Dict[str, Any]:
        mode = self._mode_for(group)
        if mode == _MODE_UNCLAMPED:
            kw: Dict[str, Any] = {"smax": None}
        else:
            kw = {"smax": self._smax_for(group)}
            if mode == _MODE_FIXED_BETA:
                kw["beta"] = (
                    float(self.beta) if self.beta is not None else _DEFAULT_BETA
                )
            # In fitted-beta mode the 5th optimizer entry supplies beta
            # directly; no kwarg needed.
        y_stderr = self._resolve_y_stderr(group)
        if y_stderr is not None:
            kw["y_stderr"] = y_stderr
        stats = self._inoc_stats(group)
        if stats is not None:
            kw["s0_prior_mean"], kw["s0_prior_sigma"] = stats
        return kw

    def _inoc_stats(self, group: pd.DataFrame) -> Tuple[float, float] | None:
        """Resolve the ``s0`` prior ``(µ, σ)`` for one group, or ``None``.

        Thin delegate to :class:`_InoculumPrior.stats_for`. Kept as a
        method on the model so existing callers (``_initial_guess``,
        ``_bounds``) and tests that probe via ``m._inoc_stats(group)``
        continue to work unchanged.
        """
        return self._prior.stats_for(group)

    def _smax_for(self, group: pd.DataFrame) -> float:
        if self.smax is not None:
            return float(self.smax)
        return float(group[self.on].max())

    def _resolve_y_stderr(self, group: pd.DataFrame) -> np.ndarray | None:
        """Build a per-timepoint ``y_stderr`` vector aligned with group rows.

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

        # When no positive σ exist (e.g. noise-free synthetic fixtures
        # where all replicates agree and SEM is 0, or singleton-replicate
        # groups where SEM is NaN), skip the weighting pathway entirely.
        # Faking σ with an ε-fill would rescale residuals by ~1/ε and
        # break the conditioning of robust losses (huber, soft_l1, etc.)
        # whose ``f_scale`` threshold is interpreted in residual units.
        positive = raw[(raw > 0) & np.isfinite(raw)]
        if positive.size == 0:
            return None

        # Replace the remaining zero/NaN entries with a small epsilon
        # scaled to the median positive stderr so the weighted residuals
        # stay finite and commensurate with the data.
        eps = 1e-8 * float(np.nanmedian(np.abs(positive)))
        if eps <= 0 or not np.isfinite(eps):
            eps = 1e-8
        sigma = np.where((raw > 0) & np.isfinite(raw), raw, eps)

        # Optional quantile floor: neutralizes coincidentally-tiny σ
        # that would otherwise dominate the 1/σ² weighting. Floor is
        # computed from finite, >0 entries so ε-filled positions do
        # not skew it; they still get lifted to the floor via np.maximum.
        if self.stderr_floor_quantile is not None:
            floor = float(np.quantile(positive, self.stderr_floor_quantile))
            if np.isfinite(floor) and floor > 0:
                sigma = np.maximum(sigma, floor)
        return sigma

    def _extra_agg_columns(self) -> Dict[str, Any]:
        """Carry per-timepoint stderr and per-group prior-mean columns.

        - ``stderr_label`` or the auto-computed ``f"{on}_stderr"`` goes
          through with a mean aggregation so the weighted loss can
          read one SE per timepoint.
        - The prior helper contributes its own entries (see
          :meth:`_InoculumPrior.extra_agg_columns`) — a single
          broadcast ``_group_mean`` column when the prior is
          column-configured, empty otherwise.
        """
        extras: Dict[str, Any] = {}
        if self.stderr_label is not None:
            extras[self.stderr_label] = "mean"
        else:
            extras[f"{self.on}_stderr"] = "mean"
        extras.update(self._prior.extra_agg_columns())
        return extras

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model to every group of ``data``.

        Pre-computes broadcasted helper columns on the raw data before
        delegating to the base-class aggregation pipeline:

        - When ``stderr_label`` is ``None``, a replicate-SEM column
          derived via ``groupby.transform("sem")`` so the weighted
          loss can downweight noisy timepoints automatically.
        - When the inoculum prior is column-based, a per-group median
          of ``inoc_size_label`` at the earliest observed timepoint is
          broadcast into a ``f"{label}_group_mean"`` column — the
          source of ``µ`` for the Gaussian prior on ``s0``
          (:class:`_InoculumPrior`).

        Each helper is constant within its effective group, so the
        base-class dict-style aggregation carries it through as a flat
        column without MultiIndex juggling.

        Raises:
            ValueError: If the inoculum prior is configured with an
                ``inoc_groupby`` that is not a subset of ``self.groupby``,
                or references columns absent from ``data``.
        """
        if self._prior.is_configured:
            self._prior.validate(data.columns, self.groupby)

        needs_copy = (
                self.stderr_label is None or self._prior.needs_broadcast
        )
        if needs_copy:
            data = data.copy(deep=True)

        if self.stderr_label is None:
            se_col = f"{self.on}_stderr"
            data[se_col] = data.groupby(
                    self.groupby + [self.time_label]
            )[self.on].transform("sem")

        self._prior.prepare(data, self.groupby, self.time_label)

        return super().analyze(data)

    # ------------------------------------------------------------------ #
    # Required fit hooks
    # ------------------------------------------------------------------ #
    def _initial_guess(self, group: pd.DataFrame) -> list[float]:
        """Heuristic initial guess for the per-group optimizer vector.

        Returns ``[v, s0, lam, alpha]`` in clamped-fixed and unclamped
        modes, or ``[v, s0, lam, alpha, beta]`` in fitted-beta mode
        (see :meth:`_mode_for`). The fifth entry seeds at the module
        default ``beta`` so the optimizer starts from the same point
        the previous fixed-beta build always used.
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
        v_init = float(np.clip(slope, 1e-4, self.v_upper))

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

        Four entries in clamped-fixed and unclamped modes; a fifth
        ``(2.0, 50.0)`` pair appended in fitted-beta mode, matching the
        ``alpha`` range convention (softplus is ~linear below 2 and
        effectively a hard step above 50).
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
        upper = [self.v_upper, s0_upper, t_max, 50.0]
        if self._mode_for(group) == _MODE_FITTED_BETA:
            lower.append(2.0)
            upper.append(50.0)
        return lower, upper

    def _unpack_params(
            self, x: np.ndarray, group: pd.DataFrame
    ) -> Dict[Any, Any]:
        mode = self._mode_for(group)
        v, s0, lam, alpha = (float(x[i]) for i in range(4))

        if mode == _MODE_UNCLAMPED:
            smax_val: float = float("nan")
            beta_val: float = float("nan")
        elif mode == _MODE_FITTED_BETA:
            smax_val = self._smax_for(group)
            beta_val = float(x[4])
        else:  # _MODE_FIXED_BETA
            smax_val = self._smax_for(group)
            beta_val = (
                float(self.beta) if self.beta is not None else _DEFAULT_BETA
            )

        return {
            LINEAR_SOFTPLUS_MODEL.v    : v,
            LINEAR_SOFTPLUS_MODEL.s0   : s0,
            LINEAR_SOFTPLUS_MODEL.lam  : lam,
            LINEAR_SOFTPLUS_MODEL.alpha: alpha,
            LINEAR_SOFTPLUS_MODEL.smax : smax_val,
            LINEAR_SOFTPLUS_MODEL.beta : beta_val,
            LINEAR_SOFTPLUS_MODEL.mode : mode,
        }

    def _predict_kwargs(self, row) -> Dict[str, Any]:
        smax_val = row[LINEAR_SOFTPLUS_MODEL.smax]
        beta_val = row[LINEAR_SOFTPLUS_MODEL.beta]
        return {
            "v"    : float(row[LINEAR_SOFTPLUS_MODEL.v]),
            "s0"   : float(row[LINEAR_SOFTPLUS_MODEL.s0]),
            "lam"  : float(row[LINEAR_SOFTPLUS_MODEL.lam]),
            "alpha": float(row[LINEAR_SOFTPLUS_MODEL.alpha]),
            "smax" : None if pd.isna(smax_val) else float(smax_val),
            "beta" : _DEFAULT_BETA if pd.isna(beta_val) else float(beta_val),
        }

    def _hover_fields(self) -> List[Tuple[str, Any, str]]:
        return [
            ("v", LINEAR_SOFTPLUS_MODEL.v, ".4f"),
            ("s0", LINEAR_SOFTPLUS_MODEL.s0, ".3f"),
            ("lambda", LINEAR_SOFTPLUS_MODEL.lam, ".3f"),
            ("alpha", LINEAR_SOFTPLUS_MODEL.alpha, ".2f"),
            ("smax", LINEAR_SOFTPLUS_MODEL.smax, ".3f"),
            ("beta", LINEAR_SOFTPLUS_MODEL.beta, ".2f"),
            ("mode", LINEAR_SOFTPLUS_MODEL.mode, ""),
            ("RMSE", MODEL_METRICS.RMSE, ".4f"),
        ]
