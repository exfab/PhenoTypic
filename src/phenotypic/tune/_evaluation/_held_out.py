"""The held-out evaluation config + group-key inference (robust-eval part 1).

``HeldOutConfig`` is the run-level policy block governing how plates are reserved
for the (part 2) generalization pass: the held-out fraction, the optional
grouping column, the data-poor floor, and the pass/fail gap margins. It rides the
``TuningSpec`` as a back-compat-defaulted field so a legacy ``tuning_spec.json``
(predating robust-eval) still validates — every missing key falls to its default.

``infer_group_key`` resolves the grouping column from the count scorer's own
``check.groupby[0]`` so a run defaults its held-out grouping to the same unit the
QC objective already compares plates by; an explicit
:attr:`HeldOutConfig.group_key` override takes precedence at the engine boundary.

optuna-free; pydantic v2 conventions (frozen, annotated fields, no ``__init__``).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class HeldOutConfig(BaseModel):
    """Run-level held-out / generalization policy (robust-eval).

    All numeric defaults are **conservative** and flagged for a pending
    literature fact-check (the ``# TODO: review`` comments); they should not be
    treated as validated thresholds yet.

    Args:
        held_out_fraction: The target fraction of plates reserved for the
            within-group hold-out tier (``round(fraction * n_plates)``).
        group_key: An explicit grouping column override; ``None`` defers to
            :func:`infer_group_key` (the count scorer's ``groupby[0]``).
        min_heldout_plates: The data-poor floor — below this many plates, no
            hold-out is reserved (all calibration).
        gap_margin_relative: The relative slack on the calibration→held-out score
            drop the (part 2) generalization gate tolerates before flagging
            overfit.
        gap_margin_absolute: The absolute slack on that drop (the gate fails only
            when **both** the relative and absolute margins are exceeded).
    """

    model_config = ConfigDict(frozen=True)

    held_out_fraction: float = 0.2  # TODO: review (unverified vs literature)
    group_key: Optional[str] = None  # TODO: review (unverified vs literature)
    min_heldout_plates: int = 6  # TODO: review (unverified vs literature)
    gap_margin_relative: float = 0.15  # TODO: review (unverified vs literature)
    gap_margin_absolute: float = 0.05  # TODO: review (unverified vs literature)


def infer_group_key(scorer: Any) -> Optional[str]:
    """Infer the held-out grouping column from a count ``scorer``.

    Reads ``scorer.check.groupby[0]`` — the first column the QC count objective
    groups plates by — so a run's held-out grouping defaults to the same unit the
    objective already compares. Returns ``None`` when the scorer has no resolvable
    ``check.groupby`` (e.g. a reference-free or composite scorer), letting the
    caller fall back to the within-group / data-poor tiers.

    Args:
        scorer: A tuning scorer; only ``scorer.check.groupby`` is read.

    Returns:
        The first ``groupby`` column name, or ``None`` when unavailable.
    """
    check = getattr(scorer, "check", None)
    groupby = getattr(check, "groupby", None)
    if groupby:
        return str(groupby[0])
    return None
