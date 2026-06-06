"""Unit tests for the Space pure helpers (Task C2).

Two pure functions back the Space view:

* :func:`~phenotypic.gui.tune._space._knob_form` — maps one inferred ``Knob`` to
  a ``dbc.Row`` editor (FloatRange → low/high + log; IntRange → low/high int;
  Categorical → checklist), plus a per-knob ``tunable`` toggle. A ``Nested`` knob
  renders read-only / disabled (depth-1 nested leaves are not v1-editable).
* :func:`~phenotypic.gui.tune._space.space_to_spec` — the OQ8 config-preserving
  builder: from an existing ``TuningSpec`` it replaces only ``search_space`` and
  keeps the run's scorer / strategy / budget / evaluator; from a bare pipeline it
  defaults the scorer (``QCScorer``) / strategy / budget with a "review in Launch"
  note.

Both are optuna-free: importing the module must not pull ``optuna``.
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import GaussianBlur


def _synth_runnable_pipeline() -> ImagePipeline:
    """A ``load_synth_yeast_plate()``-runnable flat pipeline (blur + Otsu)."""
    return ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])


def _nested_pipeline() -> ImagePipeline:
    """A pipeline whose CompositeDetector yields depth-1 ``Nested`` knobs."""
    return ImagePipeline(
        ops=[GaussianBlur(sigma=2.0), CompositeDetector(detectors=[OtsuDetector()])]
    )


def _existing_spec(tmp_path):  # type: ignore[no-untyped-def]
    """A round-trippable existing ``TuningSpec`` (path-based QC scorer)."""
    from phenotypic.analysis import ExpectedVsDetectedCount
    from phenotypic.tune import (
        Budget,
        Categorical,
        Evaluator,
        Knob,
        QCScorer,
        RandomConfig,
        SearchSpace,
        TuningSpec,
    )

    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["plate1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=_synth_runnable_pipeline(),
        search_space=SearchSpace(
            knobs=(Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),)
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=RandomConfig(n_trials=17),
        budget=Budget(n_trials=23),
    )


# ---------------------------------------------------------------------------
# space_to_spec — knob matching + round-trip (fresh-from-spec path)
# ---------------------------------------------------------------------------

def test_space_to_spec_matches_inferred_flat_and_presence_targets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from phenotypic.gui.tune._space import space_to_spec
    from phenotypic.tune import infer_search_space

    spec_in = _existing_spec(tmp_path)
    inferred = infer_search_space(spec_in.pipeline)
    # The editable (flat + presence) targets, dropping depth-1 nested leaves.
    expected_keys = {
        k.key for k in inferred.knobs if type(k.target).__name__ != "Nested"
    }

    result = space_to_spec(spec_in, edits={})
    assert {k.key for k in result.search_space.knobs} == expected_keys


def test_space_to_spec_round_trips_model_dump_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from phenotypic.tune import TuningSpec

    from phenotypic.gui.tune._space import space_to_spec

    spec_in = _existing_spec(tmp_path)
    result = space_to_spec(spec_in, edits={})
    dumped = result.model_dump_json()
    restored = TuningSpec.model_validate_json(dumped)
    assert {k.key for k in restored.search_space.knobs} == {
        k.key for k in result.search_space.knobs
    }


def test_space_to_spec_drops_nested_knobs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from phenotypic.gui.tune._space import space_to_spec
    from phenotypic.tune import infer_search_space

    spec_in = _existing_spec(tmp_path).model_copy(
        update={"pipeline": _nested_pipeline()}
    )
    inferred = infer_search_space(spec_in.pipeline)
    assert any(type(k.target).__name__ == "Nested" for k in inferred.knobs)

    result = space_to_spec(spec_in, edits={})
    assert all(
        type(k.target).__name__ != "Nested" for k in result.search_space.knobs
    )


# ---------------------------------------------------------------------------
# OQ8 — preserve scorer / strategy / budget from an existing spec
# ---------------------------------------------------------------------------

def test_space_to_spec_preserves_existing_scorer_strategy_budget(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from phenotypic.gui.tune._space import space_to_spec

    spec_in = _existing_spec(tmp_path)
    result = space_to_spec(spec_in, edits={})

    # Scorer / strategy / budget / evaluator are carried verbatim; only the
    # search space changes.
    assert type(result.scorer).__name__ == "QCScorer"
    assert result.scorer.check.metadata_source == spec_in.scorer.check.metadata_source
    assert type(result.strategy).__name__ == "RandomConfig"
    assert result.strategy.n_trials == 17
    assert result.budget.n_trials == 23


# ---------------------------------------------------------------------------
# Fresh-from-pipeline path — default scorer / strategy / budget + review note
# ---------------------------------------------------------------------------

def test_space_to_spec_from_bare_pipeline_defaults_scorer_and_strategy() -> None:
    from phenotypic.gui.tune._space import space_to_spec

    result = space_to_spec(_synth_runnable_pipeline(), edits={})
    assert type(result.scorer).__name__ == "QCScorer"
    # A fresh default scorer is unconfigured (no layout) — the "review in Launch"
    # signal: the user must point it at a metadata layout before tuning.
    assert result.scorer.availability() is False
    assert type(result.strategy).__name__ in {"GridConfig", "RandomConfig"}
    assert result.budget is not None


# ---------------------------------------------------------------------------
# _knob_form — per-domain editor + tunable toggle
# ---------------------------------------------------------------------------

def test_knob_form_floatrange_renders_low_high_and_log() -> None:
    from phenotypic.gui.tune._space import _knob_form
    from phenotypic.tune import infer_search_space

    inferred = infer_search_space(_synth_runnable_pipeline())
    sigma = next(k for k in inferred.knobs if k.key == "0.sigma")
    assert sigma.domain.kind == "float_range"
    row = _knob_form(sigma)
    rendered = str(row)
    # Two numeric inputs (low / high) + a log toggle + a tunable toggle.
    assert "0.sigma" in rendered
    assert rendered.count("Input") >= 2
    assert "low" in rendered.lower() and "high" in rendered.lower()


def test_knob_form_categorical_renders_checklist() -> None:
    from phenotypic.gui.tune._space import _knob_form
    from phenotypic.tune import infer_search_space

    inferred = infer_search_space(_synth_runnable_pipeline())
    ignore = next(k for k in inferred.knobs if k.key == "1.ignore_zeros")
    assert ignore.domain.kind == "categorical"
    row = _knob_form(ignore)
    assert "Checklist" in str(row)


def _collect_disabled_flags(component) -> list[bool]:  # type: ignore[no-untyped-def]
    """Gather every ``disabled`` flag (top-level attr + per-option) in a tree."""
    flags: list[bool] = []
    top = getattr(component, "disabled", None)
    if isinstance(top, bool):
        flags.append(top)
    for option in getattr(component, "options", None) or []:
        if isinstance(option, dict) and "disabled" in option:
            flags.append(bool(option["disabled"]))
    children = getattr(component, "children", None)
    if children is None:
        return flags
    if not isinstance(children, list):
        children = [children]
    for child in children:
        if child is None or isinstance(child, str):
            continue
        flags.extend(_collect_disabled_flags(child))
    return flags


def test_knob_form_nested_is_disabled() -> None:
    from phenotypic.gui.tune._space import _knob_form
    from phenotypic.tune import infer_search_space

    inferred = infer_search_space(_nested_pipeline())
    nested = next(
        k for k in inferred.knobs if type(k.target).__name__ == "Nested"
    )
    row = _knob_form(nested)
    flags = _collect_disabled_flags(row)
    # The nested (read-only) row's interactive widgets are all disabled.
    assert flags and all(flags)


def test_knob_form_flat_is_not_disabled() -> None:
    from phenotypic.gui.tune._space import _knob_form
    from phenotypic.tune import infer_search_space

    inferred = infer_search_space(_synth_runnable_pipeline())
    sigma = next(k for k in inferred.knobs if k.key == "0.sigma")
    row = _knob_form(sigma)
    flags = _collect_disabled_flags(row)
    # A flat knob's widgets are editable (no disabled flag is set to True).
    assert not any(flags)


def test_space_module_does_not_import_optuna() -> None:
    sys.modules.pop("optuna", None)
    import importlib

    importlib.import_module("phenotypic.gui.tune._space")
    assert "optuna" not in sys.modules


@pytest.mark.parametrize(
    "factory", [_synth_runnable_pipeline, _nested_pipeline]
)
def test_space_to_spec_validates_against_its_pipeline(factory) -> None:  # type: ignore[no-untyped-def]
    """Every produced spec's knob targets resolve against its own pipeline."""
    from phenotypic.gui.tune._space import space_to_spec

    # A fresh-from-pipeline build must construct without a target-validation error.
    result = space_to_spec(factory(), edits={})
    # Re-validate the produced spec through JSON model_validate on the structural
    # search space (the scorer is unconfigured fresh, so we only assert the knobs
    # all addressed valid ops — construction already ran the target validator).
    assert len(result.search_space.knobs) >= 1
