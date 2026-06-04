"""The tuning_spec.json model — one self-contained, round-trippable recipe."""
from __future__ import annotations

import json
from typing import Any, Optional, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
)

from phenotypic import ImagePipeline
from phenotypic.tools_.typing_ import polymorphic_field

from ._evaluation import Evaluator
from ._scoring import Scorer
from ._search_space import SearchSpace
from ._strategies._config import StrategyConfigUnion

#: A ``Scorer``-valued field that round-trips any subclass via the registry
#: (Phase-0 ``polymorphic_field`` + ``_find_class_in_phenotypic`` += ``phenotypic.tune``).
#: Typed ``TypeAlias`` so mypy accepts the ``Annotated`` core (erased to ``Any``)
#: as a field annotation — the ``AfterValidator`` restores the runtime guard.
ScorerField: TypeAlias = Any  # = polymorphic_field(base=Scorer); see below
ScorerField = polymorphic_field(base=Scorer)  # type: ignore[misc]


class Budget(BaseModel):
    """Stopping criteria (engine-arch §5). Phase 1: trial count + failure cap.

    Args:
        n_trials: Engine-level cap on the number of trials; ``None`` runs until
            the strategy exhausts (grid → until the product is covered).
        max_failures: Abort after this many failed candidates; ``None`` → never.
    """

    model_config = ConfigDict(frozen=True)

    n_trials: Optional[int] = None
    max_failures: Optional[int] = None


class TuningSpec(BaseModel):
    """A complete tuning run: base pipeline + space + scorer + strategy + budget.

    The base ``pipeline`` is embedded (engine-arch §6). A plain pydantic field
    cannot round-trip a pipeline (its polymorphic ops fail to reconstruct
    against the abstract ``ImageOperation``), so the field uses a custom
    serializer/validator delegating to the pipeline's own ``to_json``/
    ``from_json``. ``scorer`` is a ``ScorerField`` so any ``Scorer`` subclass
    round-trips through the registry; ``strategy`` is the Phase-1b grid/random
    discriminated union.

    Args:
        pipeline: The base pipeline being tuned (embedded).
        search_space: The hand-authored or migrated search space.
        scorer: The tuning objective (any ``Scorer`` subclass).
        evaluator: The candidate-evaluation policy.
        strategy: The optimizer config (``GridConfig`` / ``RandomConfig``).
        budget: The stopping criteria.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: ImagePipeline
    search_space: SearchSpace
    scorer: ScorerField
    evaluator: Evaluator
    strategy: StrategyConfigUnion
    budget: Budget

    @field_validator("pipeline", mode="before")
    @classmethod
    def _coerce_pipeline(cls, value: object) -> ImagePipeline:
        """Accept a live pipeline, its JSON string, or its embedded dict."""
        if isinstance(value, ImagePipeline):
            return value
        if isinstance(value, str):
            return ImagePipeline.from_json(value)
        if isinstance(value, dict):
            return ImagePipeline.from_json(json.dumps(value))
        raise TypeError(
            f"pipeline must be an ImagePipeline, JSON string, or dict; "
            f"got {type(value).__name__}"
        )

    @field_serializer("pipeline")
    def _dump_pipeline(self, value: ImagePipeline) -> dict:
        payload = value.to_json()
        if payload is None:  # pragma: no cover - to_json always returns a string
            raise ValueError("ImagePipeline.to_json() returned None")
        return json.loads(payload)
