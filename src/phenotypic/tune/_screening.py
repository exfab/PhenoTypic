"""Parameter importance — fANOVA-vs-RF dispatch on store capability (Phase 2).

The screening layer ranks tuned parameters by their contribution to the
objective. It dispatches **on store capability**, never on type: a backend that
exposes ``param_importances()`` (an Optuna study → fANOVA, whose variance
decomposition attributes interaction effects to each parameter) drives the
``"fanova"`` path; any backend returning ``None`` (the journal, or an Optuna
study with no native sampler dimensions) falls back to the homegrown
``RandomForest`` + permutation estimate (``"rf-permutation"``). The
``interactions_estimated`` honesty flag records whether the chosen method
accounts for interactions (fANOVA does; RF-permutation is main-effect only) so a
downstream freeze decision can be conservative when interactions are unverified
(screening-importance.md §1, §7).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from ._study._protocol import StudyStore

#: The importance estimators, a closed set (never a bare ``str``). ``"fanova"``
#: is the native Optuna variance decomposition (main + interaction); the
#: ``"rf-permutation"`` fallback is the homegrown RandomForest + permutation
#: estimate (main effects only).
ImportanceMethod = Literal["fanova", "rf-permutation"]


class ImportanceReport(BaseModel):
    """A ranked importance estimate plus its method + interaction honesty flag.

    Args:
        importances: ``{param_key: importance}`` sorted descending (empty when
            there is nothing to fit).
        method: Which estimator produced ``importances`` (a closed
            :data:`ImportanceMethod` set).
        interactions_estimated: Whether ``method`` accounts for interaction
            effects. ``True`` for ``"fanova"`` (variance decomposition); ``False``
            for ``"rf-permutation"`` (main-effect permutation only). A freeze
            decision should stay conservative when this is ``False`` — a
            low-main/high-interaction parameter may look unimportant
            (screening-importance.md §7).
    """

    model_config = ConfigDict(frozen=True)

    importances: dict[str, float]
    method: ImportanceMethod
    interactions_estimated: bool


def compute_param_importance(
    store: StudyStore, *, random_state: int = 0, objective: str | None = None
) -> dict[str, float]:
    """Rank tuned parameters by importance against the objective (the dict view).

    The back-compat, dict-returning façade over
    :func:`compute_param_importance_report`: it dispatches the same way (native
    fANOVA when the store offers it, RandomForest + permutation otherwise) but
    discards the method / honesty metadata.

    Args:
        store: The study store of completed trials.
        random_state: Seed for the forest + permutation (reproducibility); only
            used on the RandomForest fallback path.
        objective: The named multi-objective to rank against (plan §0a sidecar).
            ``None`` (default) ranks against ``Trial.score`` — the unchanged
            single-objective path. A name ranks against
            ``Trial.objectives[name]``, skipping trials that lack the objective.

    Returns:
        ``{param_key: importance}`` sorted descending. Empty when fewer than two
        usable trials (nothing to fit).
    """
    return compute_param_importance_report(
        store, random_state=random_state, objective=objective
    ).importances


def compute_param_importance_report(
    store: StudyStore, *, random_state: int = 0, objective: str | None = None
) -> ImportanceReport:
    """Rank parameters and record the method + interaction honesty flag.

    First asks the store for a native importance model via
    ``store.param_importances()`` (capability dispatch — never an ``isinstance``
    check). A non-empty result is the fANOVA path
    (``interactions_estimated=True``); ``None`` falls back to the homegrown
    RandomForest + permutation estimate (``interactions_estimated=False``). A
    **per-objective** request (``objective`` given) always takes the RF path: the
    native importance models rank against the optimizer's scalar, not a named
    objective from the multi-objective sidecar.

    Args:
        store: The study store of completed trials. Any object exposing
            ``param_importances()`` and ``trials`` satisfies the contract.
        random_state: Seed for the forest + permutation (RandomForest path).
        objective: The named multi-objective to rank against (plan §0a sidecar).
            ``None`` ranks against ``Trial.score`` and may use the native fANOVA
            model; a name forces the RF path against ``Trial.objectives[name]``.

    Returns:
        An :class:`ImportanceReport` carrying the ranked importances, the
        ``method``, and the ``interactions_estimated`` flag.
    """
    # A per-objective request cannot use the native (scalar-targeted) model.
    if objective is None:
        native = store.param_importances()
        if native:
            ranked = dict(
                sorted(native.items(), key=lambda kv: kv[1], reverse=True)
            )
            return ImportanceReport(
                importances=ranked,
                method="fanova",
                interactions_estimated=True,
            )
    return ImportanceReport(
        importances=_rf_permutation_importance(
            store, random_state=random_state, objective=objective
        ),
        method="rf-permutation",
        interactions_estimated=False,
    )


def _rf_permutation_importance(
    store: StudyStore, *, random_state: int = 0, objective: str | None = None
) -> dict[str, float]:
    """The homegrown fallback: RandomForest + permutation importance.

    Fits a ``RandomForestRegressor`` on the trials' (encoded) params → target and
    runs ``permutation_importance``. The target is ``Trial.score`` when
    ``objective`` is ``None`` (single-objective), else ``Trial.objectives[name]``
    over the trials that carry it (others are dropped). Non-numeric params are
    one-hot encoded (per-key prefix) and the encoded importances summed back to
    the original key; absent conditional params fill to ``0``. Imports no optuna —
    the lazy-import boundary holds on this path (screening-importance.md §1).

    Args:
        store: The journal of completed trials.
        random_state: Seed for the forest + permutation (reproducibility).
        objective: The named multi-objective to target, or ``None`` for the
            scalar ``Trial.score`` (plan §0a sidecar).

    Returns:
        ``{param_key: importance}`` sorted descending. Empty when fewer than two
        usable trials (nothing to fit).
    """
    trials = [t for t in store.trials if not t.failed]
    if objective is not None:
        # Per-objective: keep only trials that carry the named objective.
        trials = [
            t
            for t in trials
            if t.objectives is not None and objective in t.objectives
        ]
    if len(trials) < 2:
        return {}

    raw = pd.DataFrame([t.params for t in trials])
    if objective is None:
        y = np.asarray([t.score for t in trials], dtype=float)
    else:
        y = np.asarray(
            [t.objectives[objective] for t in trials],  # type: ignore[index]
            dtype=float,
        )
    original_keys = list(raw.columns)

    numeric = raw.select_dtypes(include="number")
    non_numeric = raw.drop(columns=list(numeric.columns))

    parts: list[pd.DataFrame] = []
    col_to_key: dict[str, str] = {}

    for col in numeric.columns:
        series = numeric[col].astype(float)
        fill = float(series.median()) if series.notna().any() else 0.0
        parts.append(series.fillna(fill).to_frame(name=col))
        col_to_key[col] = col

    if not non_numeric.empty:
        dummies = pd.get_dummies(
            non_numeric.astype("object"), prefix_sep="=", dummy_na=False
        )
        for col in dummies.columns:
            col_to_key[col] = col.split("=", 1)[0]
        parts.append(dummies)

    features = pd.concat(parts, axis=1).fillna(0.0)
    if features.shape[1] == 0:
        return {}

    forest = RandomForestRegressor(n_estimators=200, random_state=random_state)
    forest.fit(features.to_numpy(), y)
    perm = permutation_importance(
        forest, features.to_numpy(), y, n_repeats=10, random_state=random_state
    )

    importances: dict[str, float] = {key: 0.0 for key in original_keys}
    for col, value in zip(features.columns, perm.importances_mean):
        importances[col_to_key[col]] += float(value)

    return dict(
        sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    )
