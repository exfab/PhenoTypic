"""Shared base class for linear-softplus growth fitters.

Holds the σ-resolution, inoculum-prior, and aggregation-broadcasting
machinery used by both :class:`LinearSoftplus` (single softplus, no
saturation ceiling) and :class:`DoubleSoftplus` (with a softplus
saturation ceiling). Subclasses provide their own ``model_func``
geometry, parameter unpacking, and (in DoubleSoftplus) per-group mode
dispatch — everything else lives here.

Class-level constants ``_V_UPPER`` and ``_STDERR_FLOOR_QUANTILE`` are
intentionally not user-tunable kwargs on either subclass; they are
power-user knobs that move on subclass override.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

import numpy as np
import pandas as pd
from pydantic import PrivateAttr

from phenotypic.analysis._inoculum_prior import _InoculumPrior
from phenotypic.analysis.abc_._model_fitter import ModelFitter


_DEFAULT_BETA = 10.0
"""Module-level default for the saturation transition sharpness ``β``.

Used as the seed for fitted-β optimization in :class:`DoubleSoftplus`
and as the fallback when the prediction row carries ``NaN`` for the
``beta`` field.
"""


class _LinearSoftplusBase(ModelFitter):
    """Shared infrastructure for :class:`LinearSoftplus` / :class:`DoubleSoftplus`.

    Subclasses must implement ``model_func``, ``_initial_guess``,
    ``_bounds``, ``_unpack_params``, ``_predict_kwargs``, and
    ``_hover_fields`` (i.e. everything that depends on whether a
    saturation ceiling is part of the model). Everything related to
    σ resolution, the s0 prior, and pre-aggregation broadcasting is
    inherited from this base.
    """

    _V_UPPER: ClassVar[float] = 50.0
    """Upper bound on ``v`` in the optimizer.

    Subclass override if the units of ``on`` make the default 50/[time]
    too tight; tuning this from a kwarg is intentionally not exposed."""

    _STDERR_FLOOR_QUANTILE: ClassVar[float | None] = 0.25
    """Lower bound on per-timepoint σ as a quantile of the group's σ.

    Sub-quantile σ entries are lifted to the q-th percentile of the
    group's finite, positive σ values. With the default ``0.25`` and
    a typical empirical σ distribution (where σ_max is within ~4× of
    σ_median), this caps the 1/σ² weight ratio between the softest
    and stiffest points around 16×. On heavily-skewed σ inputs the
    post-floor ratio is larger but still orders of magnitude below
    the unfloored ratio. Set to ``None`` on a subclass to disable the
    floor and recover raw inverse-variance weighting."""

    stderr_label: str | None = None
    # ``s0_prior`` accepts bool / float / int / str / None. It is typed
    # ``Any`` (not a union) so pydantic stores the value verbatim without
    # coercion or rejection — type-dispatch and validation are the job of
    # ``_InoculumPrior.__init__`` below, which raises ``TypeError`` for
    # unsupported types and ``ValueError`` for non-positive scalars.
    s0_prior: Any = None
    s0_prior_cv: float | None = None
    s0_prior_sigma: float | None = None
    s0_prior_groupby: list[str] | None = None

    _prior: _InoculumPrior = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Build the inoculum-prior helper from the resolved fields.

        Runs after pydantic has validated every field. Constructing
        :class:`_InoculumPrior` here preserves the original
        ``__init__``-time validation: it raises ``TypeError`` for an
        unsupported ``s0_prior`` type and ``ValueError`` for a
        non-positive scalar, a mutually-exclusive σ pair, or an empty
        ``s0_prior_groupby`` list.

        Args:
            __context: Pydantic post-init context (unused).
        """
        self._prior = _InoculumPrior(
                s0_prior=self.s0_prior,
                s0_prior_cv=self.s0_prior_cv,
                s0_prior_sigma=self.s0_prior_sigma,
                s0_prior_groupby=self.s0_prior_groupby,
                on_column=self.on,
        )

    # ------------------------------------------------------------------ #
    # Shared model math — the linear+lag softplus term used by both
    # subclasses. DoubleSoftplus wraps this with a saturation ceiling
    # in its own ``model_func``.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lag_softplus(
            t: np.ndarray | float,
            v: float,
            s0: float,
            lam: float,
            alpha: float,
    ) -> np.ndarray | float:
        r"""Linear-with-softplus-lag growth, no saturation.

        .. math::

            s(t) = \frac{v}{\alpha}\, \ln\!\bigl(1 + e^{\alpha(t-\lambda)}\bigr) + s_0
        """
        t_arr = np.asarray(t, dtype=float)
        # ``logaddexp`` is the numerically stable form of log(1 + exp(x)).
        return v * np.logaddexp(0.0, alpha * (t_arr - lam)) / alpha + s0

    # ------------------------------------------------------------------ #
    # Loss function — handles 4-or-5 param vectors and the s0 prior.
    # The 5-param case is only used by DoubleSoftplus's fitted_beta mode.
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
        :meth:`_extra_loss_kwargs` on the subclass), a virtual residual
        ``(s0 - s0_prior_mean) / s0_prior_sigma`` is appended to
        implement the MAP-equivalent Gaussian prior on ``s0``.

        The optimizer vector is 4 or 5 elements depending on the
        per-group mode chosen by the subclass. In the 5-element case
        (DoubleSoftplus fitted-beta) the 5th entry is ``beta`` and
        overrides the kwarg. In the 4-element case the kwargs carry
        ``smax`` and ``beta``, with ``smax=None`` indicating the
        no-saturation path used by :class:`LinearSoftplus`.
        """
        if len(params) >= 5:
            v, s0, lam, alpha = (float(params[i]) for i in range(4))
            beta_eff: float = float(params[4])
        else:
            v, s0, lam, alpha = (float(params[i]) for i in range(4))
            beta_eff = float(beta) if beta is not None else _DEFAULT_BETA

        if smax is None:
            y_pred = self._lag_softplus(t=t, v=v, s0=s0, lam=lam, alpha=alpha)
        else:
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
    # Per-group helpers — σ resolution and prior stat lookup.
    # ------------------------------------------------------------------ #
    def _inoc_stats(self, group: pd.DataFrame) -> Tuple[float, float] | None:
        """Resolve the ``s0`` prior ``(µ, σ)`` for one group, or ``None``.

        Thin delegate to :meth:`_InoculumPrior.stats_for`; kept as a
        method on the base so subclasses' ``_initial_guess`` / ``_bounds``
        and tests that probe via ``m._inoc_stats(group)`` continue to
        work unchanged.
        """
        return self._prior.stats_for(group)

    def _resolve_y_stderr(self, group: pd.DataFrame) -> np.ndarray | None:
        """Build a per-timepoint ``y_stderr`` vector aligned with group rows.

        Priority order:
          1. User-supplied ``stderr_label`` column (no pool fallback —
             the user has opted into their own σ semantics).
          2. Auto-derived replicate-SE column ``f"{on}_stderr"``. NaN /
             zero σ entries (n=1 timepoints, or coincidentally-identical
             replicates) are filled with the broadcast pooled point-level
             std ``f"{on}_std_pool"``. When the pool is itself NaN
             (fully-singleton fit group), the auto-σ path degrades to
             the ε-fill behavior used by the user-supplied branch.
          3. No weights (return ``None``).
        """
        pool_value: float | None = None
        if self.stderr_label is not None and self.stderr_label in group.columns:
            raw = group[self.stderr_label].to_numpy(dtype=float)
        elif f"{self.on}_stderr" in group.columns:
            raw = group[f"{self.on}_stderr"].to_numpy(dtype=float)
            pool_col = f"{self.on}_std_pool"
            if pool_col in group.columns:
                pool_series = group[pool_col].dropna()
                if not pool_series.empty:
                    cand = float(pool_series.iloc[0])
                    if np.isfinite(cand) and cand > 0:
                        pool_value = cand
        else:
            return None

        # When no positive σ exist AND no pool-std is available (noise-
        # free synthetic fixtures where replicates agree and SEM is 0,
        # or fully-singleton-replicate fit groups where SEM is NaN),
        # skip the weighting pathway entirely. Faking σ with an ε-fill
        # would rescale residuals by ~1/ε and break the conditioning of
        # robust losses (huber, soft_l1, etc.) whose ``f_scale`` threshold
        # is interpreted in residual units.
        positive = raw[(raw > 0) & np.isfinite(raw)]
        if positive.size == 0 and pool_value is None:
            return None

        # Fill NaN / zero σ entries. When auto-deriving σ and the fit
        # group has at least one multi-replicate timepoint, use the
        # broadcast pooled point-level std — this gives n=1 rows σ ≈
        # typical point noise, so their 1/σ² weight is commensurate
        # with (rather than dominating) the SEM-weighted multi-replicate
        # rows. Fall back to the original ε-scaled fill on the user-
        # supplied ``stderr_label`` path or when no pool is available.
        if pool_value is not None:
            fill = pool_value
        else:
            eps = 1e-8 * float(np.nanmedian(np.abs(positive)))
            if eps <= 0 or not np.isfinite(eps):
                eps = 1e-8
            fill = eps
        sigma = np.where((raw > 0) & np.isfinite(raw), raw, fill)

        # Optional quantile floor: neutralizes coincidentally-tiny σ
        # that would otherwise dominate the 1/σ² weighting. Floor is
        # computed from finite, >0 entries only so pool-/ε-filled
        # positions do not skew it; they still get lifted to the floor
        # via np.maximum.
        if self._STDERR_FLOOR_QUANTILE is not None and positive.size > 0:
            floor = float(np.quantile(positive, self._STDERR_FLOOR_QUANTILE))
            if np.isfinite(floor) and floor > 0:
                sigma = np.maximum(sigma, floor)
        return sigma

    # ------------------------------------------------------------------ #
    # Aggregation hooks — pre-broadcast the SEM and prior-mean columns.
    # ------------------------------------------------------------------ #
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
            # Pool column is constant within each fit group — "first"
            # carries the broadcast value through the per-timepoint
            # aggregation without change.
            extras[f"{self.on}_std_pool"] = "first"
        extras.update(self._prior.extra_agg_columns())
        return extras

    def _extra_loss_kwargs(self, group: pd.DataFrame) -> Dict[str, Any]:
        """Per-group kwargs forwarded to :meth:`_loss_func`.

        Base implementation only attaches ``y_stderr`` and the s0 prior
        stats. Subclasses that need ``smax`` / ``beta`` per-group
        (i.e. :class:`DoubleSoftplus`) must override this and call
        ``super()._extra_loss_kwargs(group)`` to pick up the σ + prior
        entries first.
        """
        kw: Dict[str, Any] = {}
        y_stderr = self._resolve_y_stderr(group)
        if y_stderr is not None:
            kw["y_stderr"] = y_stderr
        stats = self._inoc_stats(group)
        if stats is not None:
            kw["s0_prior_mean"], kw["s0_prior_sigma"] = stats
        return kw

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-broadcast helper columns, then delegate to the base pipeline.

        - When ``stderr_label`` is ``None``, a replicate-SEM column
          derived via ``groupby.transform("sem")`` so the weighted
          loss can downweight noisy timepoints automatically, plus a
          per-fit-group pooled point-level std column
          (``f"{on}_std_pool"``) computed as the median of per-
          timepoint stds across the group's n≥2 timepoints. The pool
          gives :meth:`_resolve_y_stderr` a principled fallback σ for
          n=1 timepoints (σ ≈ typical point noise) instead of the
          vanishingly-small ε fill. Fit groups with zero multi-
          replicate timepoints produce NaN here and inherit the
          unweighted-residual fallback.
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

            # Pooled point-level std, broadcast per fit group. Used by
            # ``_resolve_y_stderr`` as the σ fallback for n=1 timepoints.
            # Fully-singleton fit groups yield NaN here → the resolver
            # returns ``None`` and the fit runs unweighted (preserving
            # the existing singleton-fallback behavior).
            std_pool_col = f"{self.on}_std_pool"
            std_pt = data.groupby(
                    self.groupby + [self.time_label]
            )[self.on].transform("std")
            data[std_pool_col] = std_pt.groupby(
                    [data[c] for c in self.groupby]
            ).transform("median")

        self._prior.prepare(data, self.groupby, self.time_label)

        return super().analyze(data)
