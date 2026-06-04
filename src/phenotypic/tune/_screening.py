"""Parameter importance — Phase-1 RF + permutation fallback (fANOVA is Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from ._study_store import StudyStore


def compute_param_importance(
    store: StudyStore, *, random_state: int = 0
) -> dict[str, float]:
    """Rank tuned parameters by permutation importance against the objective.

    Fits a ``RandomForestRegressor`` on the trials' (encoded) params → score and
    runs ``permutation_importance``. Non-numeric params are one-hot encoded
    (per-key prefix) and the encoded importances summed back to the original key;
    absent conditional params fill to ``0``.

    Args:
        store: The journal of completed trials.
        random_state: Seed for the forest + permutation (reproducibility).

    Returns:
        ``{param_key: importance}`` sorted descending. Empty when fewer than two
        non-failed trials (nothing to fit).
    """
    trials = [t for t in store.trials if not t.failed]
    if len(trials) < 2:
        return {}

    raw = pd.DataFrame([t.params for t in trials])
    y = np.asarray([t.score for t in trials], dtype=float)
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
