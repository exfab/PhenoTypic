"""The tuning_spec.json model — one self-contained, round-trippable recipe."""
from __future__ import annotations

import difflib
import json
import warnings
from pathlib import Path
from typing import Any, Optional, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_serializer,
    field_validator,
    Field,
    model_validator,
)

from phenotypic import ImagePipeline
from phenotypic.sdk_._io_constants import (
    CONFIG_SUFFIX_TUNING,
    ensure_typed_json_suffix,
)
from phenotypic.sdk_.typing_ import polymorphic_field

from ._evaluation import Evaluator, HeldOutConfig
from ._multi_objective import reject_grid_random_multi_objective
from ._scoring import ScorerField
from ._search_space import SearchSpace
from ._search_space._targets import KnobTarget, Nested, Param
from ._strategies._config import StrategyConfig, StrategyConfigUnion

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


def _current_phenotypic_version() -> str:
    """Return the package version used to stamp serialized tuning specs."""
    import phenotypic

    return phenotypic.__version__


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


def _check_field(op_obj: Any, field: str, op: int, cls_name: str) -> None:
    """Assert ``field`` exists on ``op_obj``; raise with a did-you-mean otherwise."""
    if field in type(op_obj).model_fields:
        return
    available = sorted(type(op_obj).model_fields)
    suggestion = difflib.get_close_matches(field, available, n=1)
    hint = f" — did you mean {suggestion[0]!r}?" if suggestion else ""
    raise ValueError(
        f"op {op} ({cls_name}) has no field {field!r}{hint}; available: {available}"
    )


def _validate_target(target: KnobTarget, ordered_ops: list) -> None:
    """Validate one knob target against the live pipeline ops (cross-check)."""
    op = target.op
    if not 0 <= op < len(ordered_ops):
        raise ValueError(
            f"knob target {target.key!r} addresses op {op}, but the pipeline has "
            f"{len(ordered_ops)} op(s)"
        )
    actual = ordered_ops[op]
    actual_cls = type(actual).__name__
    if target.op_class is not None and target.op_class != actual_cls:
        raise ValueError(
            f"knob target {target.key!r} names class {target.op_class!r}, but op "
            f"{op} is a {actual_cls!r}"
        )
    if isinstance(target, Param):
        _check_field(actual, target.field, op, actual_cls)
    elif isinstance(target, Nested):
        nested = getattr(actual, target.field, None)
        if not isinstance(nested, list):
            raise ValueError(
                f"nested target {target.key!r}: {actual_cls}.{target.field} is not "
                "a list-of-ops field"
            )
        if not 0 <= target.index < len(nested):
            raise ValueError(
                f"nested target {target.key!r}: index {target.index} out of range "
                f"({len(nested)} slot(s))"
            )
        slot = nested[target.index]
        if slot is None:
            raise ValueError(
                f"nested target {target.key!r}: slot {target.index} is empty (None)"
            )
        _check_field(slot, target.leaf, op, type(slot).__name__)
    # Presence: op-range + op_class already checked above.


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
        held_out: The robust-eval held-out / generalization policy. Defaults to a
            conservative :class:`HeldOutConfig`; a frozen pre-4.5p1
            ``tuning_spec.json`` with **no** ``held_out`` block still validates
            (the field defaults), so the addition is back-compatible.
        phenotypic_version: Package version that wrote the spec. Missing or
            mismatched values warn during load but do not reject legacy specs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    phenotypic_version: str = Field(default_factory=_current_phenotypic_version)
    pipeline: ImagePipeline
    search_space: SearchSpace
    scorer: ScorerField
    evaluator: Evaluator
    strategy: StrategyConfigField
    budget: Budget
    held_out: HeldOutConfig = HeldOutConfig()

    @model_validator(mode="before")
    @classmethod
    def _warn_version_provenance(cls, value: object) -> object:
        """Warn when loading legacy or older/newer tuning spec payloads."""
        if not isinstance(value, dict):
            return value
        if isinstance(value.get("pipeline"), ImagePipeline):
            return value
        saved = value.get("phenotypic_version")
        current = _current_phenotypic_version()
        if saved is None:
            warnings.warn(
                "tuning spec is missing phenotypic_version; assuming current "
                f"version {current}",
                UserWarning,
                stacklevel=2,
            )
        elif saved != current:
            warnings.warn(
                "tuning spec phenotypic_version "
                f"{saved!r} differs from current version {current!r}",
                UserWarning,
                stacklevel=2,
            )
        return value

    def to_json(
        self,
        filepath: str | Path | None = None,
        *,
        indent: int | None = 2,
    ) -> str | None:
        """Serialize this tuning spec to JSON.

        Args:
            filepath: Optional path to write. When provided, legacy ``.json``
                names are normalized to the typed tuning config suffix.
            indent: Indentation passed to
                :meth:`~pydantic.BaseModel.model_dump_json`. Defaults to 2.

        Returns:
            The JSON string when ``filepath`` is None, otherwise None.
        """
        payload = self.model_dump_json(indent=indent)
        if filepath is not None:
            ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_TUNING).write_text(
                payload
            )
            return None
        return payload

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

    @model_validator(mode="after")
    def _validate_targets_against_pipeline(self) -> "TuningSpec":
        """Cross-check every knob target (and conditional parent) vs the pipeline.

        Catches targeting mistakes — out-of-range op, ``op_class`` mismatch,
        missing field/leaf (with a did-you-mean), unresolvable nesting — at spec
        construction (where an MCP submits), rather than deep in
        ``build_pipeline`` at evaluation time. Complements the apply-time ``⊆``
        backstop, which still catches validator-enforced value bounds.

        Returns:
            ``self`` when every target resolves.

        Raises:
            ValueError: With an actionable message naming the offending target.
        """
        ordered_ops = list(self.pipeline.get_ops().values())
        for knob in self.search_space.knobs:
            _validate_target(knob.target, ordered_ops)
            for parent, _ in (knob.conditional_on or ()):
                _validate_target(parent, ordered_ops)
        return self
