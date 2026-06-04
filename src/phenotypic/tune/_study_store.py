"""The trial journal — Phase-1 homegrown persistence (Optuna SQLite is Phase 2).

A ``StudyStore`` accumulates ``Trial`` records, reports the ``best`` (max score
among non-failed trials), and round-trips through ``trials.parquet`` (params and
terms persisted as JSON columns — lossless across heterogeneous/conditional
param sets). Reloading a store powers CLI resume (``_engine`` fast-forwards a
deterministic strategy past the recorded trials).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class Trial(BaseModel):
    """One evaluated candidate: its params, score, per-term scores, and status.

    Args:
        number: The zero-based trial index in journaling order.
        params: The sampled combo (``{root-relative-key: value}``).
        score: The finalized scalar objective (higher = better).
        terms: The robust-aggregated per-term scores backing ``score``.
        n_images: Number of calibration images evaluated.
        failed: ``True`` when the candidate raised and scored the failure floor.
        pruned: ``True`` when the rung ladder early-stopped this candidate.
            Distinct from ``failed``: pruned trials ran cleanly on a partial set
            and still count against the budget (failed trials do not).
    """

    model_config = ConfigDict(frozen=True)

    number: int
    params: dict[str, Any]
    score: float
    terms: dict[str, float]
    n_images: int
    failed: bool = False
    pruned: bool = False


class StudyStore:
    """An append-only journal of trials with best-tracking + parquet I/O."""

    def __init__(self, trials: Optional[list[Trial]] = None) -> None:
        """Initialize the journal.

        Args:
            trials: Optional seed trials (e.g. a resumed run's prior journal).
        """
        self._trials: list[Trial] = list(trials or [])

    def append(self, trial: Trial) -> None:
        """Record one completed ``trial``."""
        self._trials.append(trial)

    @property
    def trials(self) -> list[Trial]:
        """A copy of the journaled trials in order."""
        return list(self._trials)

    def __len__(self) -> int:
        return len(self._trials)

    def best(self) -> Optional[Trial]:
        """The non-failed trial with the highest score, or ``None``."""
        valid = [t for t in self._trials if not t.failed]
        if not valid:
            return None
        return max(valid, key=lambda t: t.score)

    #: Stable column order for the trials frame (explicit so an empty store
    #: still writes a valid parquet schema rather than a zero-column frame).
    _COLUMNS = [
        "number", "score", "n_images", "failed", "pruned",
        "params_json", "terms_json",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per trial; ``params``/``terms`` serialized as JSON strings."""
        rows = [
            {
                "number": t.number,
                "score": t.score,
                "n_images": t.n_images,
                "failed": t.failed,
                "pruned": t.pruned,
                "params_json": json.dumps(t.params, sort_keys=True),
                "terms_json": json.dumps(t.terms, sort_keys=True),
            }
            for t in self._trials
        ]
        return pd.DataFrame(rows, columns=self._COLUMNS)

    def to_parquet(self, path: Path) -> None:
        """Write the journal to ``path`` (creating parent dirs)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: Path) -> "StudyStore":
        """Reload a journal previously written by :meth:`to_parquet`."""
        df = pd.read_parquet(path)
        trials = [
            Trial(
                number=int(row["number"]),
                params=json.loads(str(row["params_json"])),
                score=float(row["score"]),
                terms=json.loads(str(row["terms_json"])),
                n_images=int(row["n_images"]),
                failed=bool(row["failed"]),
                # Tolerate pre-pruned-column journals (default to not-pruned).
                pruned=bool(row.get("pruned", False)),
            )
            for row in df.to_dict(orient="records")
        ]
        return cls(trials)
