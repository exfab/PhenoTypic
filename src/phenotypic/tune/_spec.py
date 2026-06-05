"""The tuning_spec.json model — one self-contained, round-trippable recipe."""
from __future__ import annotations

import json
from typing import Any, Optional, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from phenotypic import ImagePipeline
from phenotypic.tools_.typing_ import polymorphic_field

from ._evaluation import Evaluator
from ._multi_objective import reject_grid_random_multi_objective
from ._scoring import Scorer
from ._search_space import SearchSpace
from ._strategies._config import StrategyConfig, StrategyConfigUnion

#: A ``Scorer``-valued field that round-trips any subclass via the registry
#: (Phase-0 ``polymorphic_field`` + ``_find_class_in_phenotypic`` += ``phenotypic.tune``).
#: Typed ``TypeAlias`` so mypy accepts the ``Annotated`` core (erased to ``Any``)
#: as a field annotation — the ``AfterValidator`` restores the runtime guard.
ScorerField: TypeAlias = Any  # = polymorphic_field(base=Scorer); see below
ScorerField = polymorphic_field(base=Scorer)  # type: ignore[misc]

#: A ``StrategyConfig``-valued field that round-trips **any** subclass via the
#: class registry — Phase-1's built-in ``GridConfig``/``RandomConfig`` *and* the
#: Phase-2 ``OptunaConfig`` (which lives outside the closed ``StrategyConfigUnion``
#: discriminated union, so it could not round-trip through the union alone). The
#: same ``polymorphic_field`` machinery as ``ScorerField``; ``StrategyConfigUnion``
#: stays exported for callers that want the narrow built-in union. A frozen
#: Phase-1 ``tuning_spec.json`` whose ``strategy`` block is the original
#: discriminator form (``{"seed": 0, "kind": "grid"}`` — no ``"class"`` wrapper)
#: is reconstructed by ``_coerce_strategy`` below.
StrategyConfigField: TypeAlias = Any  # = polymorphic_field(base=StrategyConfig)
StrategyConfigField = polymorphic_field(base=StrategyConfig)  # type: ignore[misc]

#: TypeAdapter over the built-in discriminated union, reused to reconstruct a
#: legacy discriminator-tagged strategy dict (no ``"class"`` wrapper). Annotated
#: as the common ``StrategyConfig`` base (the union's ``Annotated`` form is not a
#: type mypy can infer the adapter's generic from).
_STRATEGY_UNION_ADAPTER: TypeAdapter[StrategyConfig] = TypeAdapter(
    StrategyConfigUnion
)


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
    round-trips through the registry; ``strategy`` is a ``StrategyConfigField``
    so any ``StrategyConfig`` subclass — the built-in ``GridConfig``/
    ``RandomConfig`` *and* the Phase-2 ``OptunaConfig`` — round-trips through the
    registry. A frozen Phase-1 ``tuning_spec.json`` (whose ``strategy`` block is
    the original ``{"seed": ..., "kind": ...}`` discriminator form, with no
    ``"class"`` wrapper) is still accepted via :meth:`_coerce_strategy`.

    Args:
        pipeline: The base pipeline being tuned (embedded).
        search_space: The hand-authored or migrated search space.
        scorer: The tuning objective (any ``Scorer`` subclass).
        evaluator: The candidate-evaluation policy.
        strategy: The optimizer config (``GridConfig`` / ``RandomConfig`` /
            ``OptunaConfig``, or any ``StrategyConfig`` subclass).
        budget: The stopping criteria.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: ImagePipeline
    search_space: SearchSpace
    scorer: ScorerField
    evaluator: Evaluator
    strategy: StrategyConfigField
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

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, value: object) -> object:
        """Reconstruct a legacy discriminator-tagged strategy dict.

        The widened ``strategy`` field is a ``polymorphic_field`` whose tagged
        form is ``{"class": ..., "params": {...}}``. A frozen Phase-1
        ``tuning_spec.json`` instead carries the original discriminated-union
        form ``{"seed": ..., "kind": "grid"}`` (no ``"class"`` wrapper). When we
        see such a bare dict — a mapping with a ``"kind"`` discriminator and no
        ``"class"`` key — route it through the built-in union adapter so the
        concrete ``GridConfig``/``RandomConfig`` is rebuilt before the
        polymorphic field's validators run. Live instances and new tagged dicts
        pass through untouched.
        """
        if isinstance(value, dict) and "kind" in value and "class" not in value:
            return _STRATEGY_UNION_ADAPTER.validate_python(value)
        return value

    @field_serializer("pipeline")
    def _dump_pipeline(self, value: ImagePipeline) -> dict:
        payload = value.to_json()
        if payload is None:  # pragma: no cover - to_json always returns a string
            raise ValueError("ImagePipeline.to_json() returned None")
        return json.loads(payload)

    @model_validator(mode="after")
    def _reject_multi_objective_without_optuna(self) -> "TuningSpec":
        """Reject a multi-objective scorer paired with a grid/random strategy.

        Multi-objective (Pareto) search needs a multi-objective optimizer — the
        Optuna NSGA-II sampler. The exhaustive grid and the seeded-random
        strategies are single-objective only (they have no notion of a
        non-dominated set), so pairing one with a
        ``CompositeScorer(multi_objective=True)`` is a configuration error caught
        here, at construction, with an actionable message (the same guard the
        ``run_tuning`` ``--strategy`` override re-asserts).

        Returns:
            ``self`` (unchanged) for a valid pairing.

        Raises:
            ValueError: When the scorer is multi-objective but the strategy is
                not an Optuna strategy.
        """
        reject_grid_random_multi_objective(self.scorer, self.strategy)
        return self
