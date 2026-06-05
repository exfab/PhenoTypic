"""The trial journal — Phase-1 homegrown persistence (Optuna SQLite is Phase 2).

A ``JournalStudyStore`` accumulates ``Trial`` records, reports the ``best`` (max
score among non-failed trials), and round-trips through ``trials.parquet``
(params and terms persisted as JSON columns — lossless across heterogeneous/
conditional param sets). Reloading a store powers CLI resume (``_engine``
fast-forwards a deterministic strategy past the recorded trials). It is the
concrete Phase-1 implementation of the ``StudyStore`` Protocol
(``_study/_protocol.py``); ``StudyStore`` remains an exported back-compat alias
for the concrete journal.
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
        score: The finalized scalar objective (higher = better). For a
            multi-objective trial this is the scalar projection of
            ``objectives`` (``mean(objectives.values())``).
        terms: The robust-aggregated per-term scores backing ``score``.
        n_images: Number of calibration images evaluated.
        objectives: The named multi-objective values (plan §0a sidecar), or
            ``None`` for a single-objective trial. Carried from
            ``EvaluationResult.objectives``; persisted as the ``objectives_json``
            journal column. ``None`` for every legacy (pre-sidecar) trial.
        failed: ``True`` when the candidate raised and scored the failure floor.
        pruned: ``True`` when the rung ladder early-stopped this candidate.
            Distinct from ``failed``: pruned trials ran cleanly on a partial set
            and still count against the budget (failed trials do not).
        gap: The trial's relative across-plate dispersion of the primary term —
            a cheap instability / overfit-risk flag carried from
            ``EvaluationResult.gap``, persisted as the nullable-float ``gap``
            journal column. **Not** a held-out generalization gap. ``None`` when
            the signal was unavailable (and for every legacy pre-4.5p1 trial).
        suspicious: ``True`` when the trial matched the qc §5 under-detection
            gaming signature; carried from ``EvaluationResult.suspicious`` and
            persisted as the ``suspicious`` bool journal column. ``False`` for
            every legacy (pre-4.5p1) trial.
    """

    model_config = ConfigDict(frozen=True)

    number: int
    params: dict[str, Any]
    score: float
    terms: dict[str, float]
    n_images: int
    objectives: Optional[dict[str, float]] = None
    failed: bool = False
    pruned: bool = False
    gap: Optional[float] = None
    suspicious: bool = False


class JournalStudyStore:
    """An append-only journal of trials with best-tracking + parquet I/O.

    The Phase-1 concrete :class:`~phenotypic.tune._study._protocol.StudyStore`
    backend. It resumes by **replay** (the engine fast-forwards the deterministic
    strategy past the recorded trials), so :meth:`is_resumable_in_place` is
    ``False`` — distinguishing it from a future Optuna ``RDBStorage`` backend
    whose own storage reconstructs the sampler state.
    """

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

    def is_resumable_in_place(self) -> bool:
        """Always ``False``: the journal resumes by deterministic replay."""
        return False

    def completed_count(self) -> int:
        """The number of completed (non-failed) trials; pruned counts as done."""
        return sum(1 for t in self._trials if not t.failed)

    def param_importances(self) -> Optional[dict[str, float]]:
        """Always ``None``: the journal owns no native importance model.

        The screening layer falls back to its RandomForest + permutation
        estimate over the journaled trials (screening-importance.md §1).
        """
        return None

    def pareto_front(self) -> list[Trial]:
        """The non-dominated trials by their ``objectives`` sidecar (plan §0a).

        Delegates to the store-agnostic :func:`pareto_front_of` over the
        journaled trials. A single-objective journal (no trial carries
        ``objectives``) returns ``[]`` while scalar :meth:`best` still works.
        """
        from ._study._pareto import pareto_front_of

        return pareto_front_of(self._trials)

    def knee_point(self, front: list[Trial]) -> Optional[Trial]:
        """The ``front`` trial at max perpendicular distance to the chord.

        Delegates to the store-agnostic :func:`knee_point_of`; ``None`` for an
        empty front.
        """
        from ._study._pareto import knee_point_of

        return knee_point_of(front)

    #: Stable column order for the trials frame (explicit so an empty store
    #: still writes a valid parquet schema rather than a zero-column frame).
    #: ``objectives_json`` is the multi-objective sidecar column (plan §0a):
    #: ``null`` for single-objective trials, a JSON dict for multi-objective ones.
    #: ``gap`` (nullable float) + ``suspicious`` (bool) are the 4.5p1 robust-eval
    #: signals, appended **last** so a legacy parquet without them still loads.
    _COLUMNS = [
        "number", "score", "n_images", "failed", "pruned",
        "params_json", "terms_json", "objectives_json",
        "gap", "suspicious",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per trial; ``params``/``terms``/``objectives`` as JSON strings.

        ``objectives_json`` is ``None`` for single-objective trials (the column
        holds ``null``) and ``json.dumps(t.objectives)`` for multi-objective ones.
        """
        rows = [
            {
                "number": t.number,
                "score": t.score,
                "n_images": t.n_images,
                "failed": t.failed,
                "pruned": t.pruned,
                "params_json": json.dumps(t.params, sort_keys=True),
                "terms_json": json.dumps(t.terms, sort_keys=True),
                "objectives_json": (
                    json.dumps(t.objectives, sort_keys=True)
                    if t.objectives
                    else None
                ),
                "gap": None if t.gap is None else float(t.gap),
                "suspicious": bool(t.suspicious),
            }
            for t in self._trials
        ]
        return pd.DataFrame(rows, columns=self._COLUMNS)

    def to_parquet(self, path: Path) -> None:
        """Write the journal to ``path`` (creating parent dirs)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: Path) -> "JournalStudyStore":
        """Reload a journal previously written by :meth:`to_parquet`.

        Reads ``objectives_json`` defensively: a legacy Phase-1/2 parquet
        predating the multi-objective sidecar (plan §0a) has no such column, and a
        single-objective trial stores ``null`` — both resolve to
        ``objectives=None`` so older journals still load.
        """
        df = pd.read_parquet(path)
        trials = [
            Trial(
                number=int(row["number"]),
                params=json.loads(str(row["params_json"])),
                score=float(row["score"]),
                terms=json.loads(str(row["terms_json"])),
                n_images=int(row["n_images"]),
                objectives=cls._parse_objectives(row.get("objectives_json")),
                failed=bool(row["failed"]),
                # Tolerate pre-pruned-column journals (default to not-pruned).
                pruned=bool(row.get("pruned", False)),
                # Tolerate pre-4.5p1 journals (no gap/suspicious → neutral).
                gap=cls._parse_optional_float(row.get("gap")),
                suspicious=bool(row.get("suspicious", False)),
            )
            for row in df.to_dict(orient="records")
        ]
        return cls(trials)

    @staticmethod
    def _is_null_cell(raw: Any) -> bool:
        """Whether a journal cell is null — ``None`` or a pandas ``NaN`` float.

        The shared null/NaN preamble of :meth:`_parse_objectives` and
        :meth:`_parse_optional_float`: a missing column reads as ``None`` (via
        ``row.get``) and an empty cell reads as a ``float`` ``NaN``. Catching the
        ``NaN`` here, before any ``float``/``json`` coercion, keeps both parsers'
        null handling identical.

        Args:
            raw: The raw cell value.

        Returns:
            ``True`` when ``raw`` is ``None`` or a ``NaN`` float.
        """
        return raw is None or (isinstance(raw, float) and pd.isna(raw))

    @classmethod
    def _parse_objectives(cls, raw: Any) -> Optional[dict[str, float]]:
        """Decode an ``objectives_json`` cell into a dict, or ``None``.

        Tolerates the three back-compat shapes: a missing column (``raw`` is
        ``None`` via ``row.get``), a ``null``/``NaN`` cell (single-objective
        trial), and a JSON dict string (multi-objective trial).

        Args:
            raw: The raw ``objectives_json`` cell — ``None``, a pandas ``NaN``, or
                a JSON dict string.

        Returns:
            The decoded ``{objective: value}`` dict, or ``None`` when there is no
            multi-objective payload.
        """
        return None if cls._is_null_cell(raw) else json.loads(str(raw))

    @classmethod
    def _parse_optional_float(cls, raw: Any) -> Optional[float]:
        """Decode a nullable-float journal cell (e.g. ``gap``) into ``float``.

        Mirrors :meth:`_parse_objectives` for the 4.5p1 ``gap`` column: tolerates
        the three back-compat shapes — a missing column (``raw`` is ``None`` via
        ``row.get``), a ``null``/``NaN`` cell (the signal was unavailable), and a
        real numeric value. The shared :meth:`_is_null_cell` guard catches the
        ``NaN`` before the ``float`` coercion (strictly safe for the float/null
        ``gap`` column).

        Args:
            raw: The raw cell — ``None``, a pandas ``NaN``, or a number.

        Returns:
            The decoded ``float``, or ``None`` when there is no value.
        """
        return None if cls._is_null_cell(raw) else float(raw)


#: Back-compat alias: ``StudyStore`` historically named the concrete journal.
#: The name now also denotes the Protocol in ``_study/_protocol.py``; the public
#: ``phenotypic.tune.StudyStore`` export resolves to this concrete journal so all
#: Phase-1 imports/constructions (``StudyStore()``) keep working.
StudyStore = JournalStudyStore
