"""Gaussian prior on the inoculum size ``s0`` for softplus-family models.

Used internally by :class:`LinearLagModel` and :class:`LinearCapAndLagModel` (and
their shared base) to attach a per-group Gaussian prior on the initial size.
The user-facing dispatch lives on each model class; this module owns the
internal field shape and the prior-engagement logic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


class _InoculumPrior:
    """Gaussian prior on ``s0`` for softplus-family fitters.

    Resolves ``(µ, σ)`` per fit group. The public model dispatches user
    input (the polymorphic ``s0_prior`` and the explicit
    ``s0_prior_cv`` / ``s0_prior_sigma`` knobs) into this helper's
    internal fields:

    - ``label`` — column from which per-group ``µ`` is derived.
      ``s0_prior=True`` resolves to ``on_column``; ``s0_prior="<str>"``
      resolves to that column name.
    - ``direct_mean`` — scalar ``µ`` applied uniformly
      (``s0_prior=<float>`` path).
    - ``direct_sigma`` — absolute ``σ`` (``s0_prior_sigma`` path).
    - ``cv`` — CV coefficient for ``σ = cv × µ``
      (``s0_prior_cv`` path). Exactly one of ``direct_sigma`` and
      ``cv`` is set per instance.
    - ``groupby`` — coarser grouping for empirical-Bayes ``µ``
      estimation on column-backed priors.

    See :class:`LinearLagModel` / :class:`LinearCapAndLagModel` for the
    user-facing dispatch tables and usage.
    """

    _MEAN_SUFFIX = "_group_mean"

    def __init__(
            self,
            *,
            s0_prior: bool | float | int | str | None,
            s0_prior_cv: float | None,
            s0_prior_sigma: float | None,
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

        # XOR validation on the σ knobs: at most one is non-None.
        if s0_prior_cv is not None and s0_prior_sigma is not None:
            raise ValueError(
                    "Pass at most one of s0_prior_cv and s0_prior_sigma — "
                    "they are mutually exclusive σ specifications."
            )
        for name, val in (
                ("s0_prior_cv", s0_prior_cv),
                ("s0_prior_sigma", s0_prior_sigma),
        ):
            if val is not None and (not np.isfinite(val) or val <= 0):
                raise ValueError(
                        f"{name} must be positive and finite, got {val!r}."
                )

        # Default σ behaviour: if neither was set, fall back to CV=0.05
        # (same default as the legacy ``s0_prior_factor=0.05``). This
        # makes the prior usable with just ``s0_prior=True`` without
        # also requiring an explicit σ knob.
        if s0_prior_cv is None and s0_prior_sigma is None:
            s0_prior_cv = 0.05

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
        self.direct_sigma = s0_prior_sigma
        self.cv = s0_prior_cv
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
