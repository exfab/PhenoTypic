"""Round-trip every legacy pipeline JSON through the DAG conversion path.

Spec §5.4 / §5.7 — the DAG redesign keeps ``ImagePipeline.to_json`` /
``from_json`` untouched.  Loading a pre-redesign ``pipeline.json`` flows
through :func:`from_pipeline_dag` (DAG conversion) which is the only
migration path.  This test exercises the round-trip for every shipped
prefab pipeline (which is what the user's saved ``pipeline.json`` files
were generated from in the pre-redesign era) and any JSON fixtures
under ``tests/fixtures/pipelines/`` (currently none).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest

from phenotypic.gui.builder._conversion_dag import (
    from_pipeline_dag,
    to_pipeline_dag,
)
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
)


PIPELINE_JSON_DIR = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "pipelines"


def _load_prefab_pipelines() -> List[Any]:
    """Return prefab pipeline instances under round-trip test."""

    # Each prefab is a parameterless callable yielding an ImagePipeline.
    # Importing inside the function keeps test discovery fast.
    from phenotypic.prefab import (
        FilamentousFungiPipeline,
        HeavyOtsuPipeline,
        HeavyRoundPeaksPipeline,
        HeavyWatershedPipeline,
        RoundPeaksPipeline,
    )

    return [
        ("HeavyOtsuPipeline", HeavyOtsuPipeline()),
        ("HeavyRoundPeaksPipeline", HeavyRoundPeaksPipeline()),
        ("HeavyWatershedPipeline", HeavyWatershedPipeline()),
        ("RoundPeaksPipeline", RoundPeaksPipeline()),
        ("FilamentousFungiPipeline", FilamentousFungiPipeline()),
    ]


_PREFAB_CASES = _load_prefab_pipelines()


@pytest.mark.parametrize(
    "name, pipeline",
    _PREFAB_CASES,
    ids=[name for name, _ in _PREFAB_CASES],
)
def test_prefab_pipeline_round_trips_through_dag(name: str, pipeline: Any) -> None:
    """from_pipeline_dag → to_pipeline_dag preserves a prefab's content."""

    # Convert into a DAG state, then convert back.
    state = from_pipeline_dag(pipeline)
    assert any(
        b.class_name == INPUT_IMAGE_CLASS_NAME for b in state.root.blocks
    ), f"{name}: from_pipeline_dag should auto-seed an InputImage block"
    rebuilt = to_pipeline_dag(state)
    # Validate the round-trip preserved ops/meas/post cardinality.
    assert len(rebuilt.get_ops()) == len(pipeline.get_ops()), (
        f"{name}: ops count drift "
        f"({len(rebuilt.get_ops())} vs {len(pipeline.get_ops())})"
    )
    assert len(rebuilt.get_meas()) == len(pipeline.get_meas()), (
        f"{name}: meas count drift "
        f"({len(rebuilt.get_meas())} vs {len(pipeline.get_meas())})"
    )
    assert len(rebuilt.get_post()) == len(pipeline.get_post()), (
        f"{name}: post count drift "
        f"({len(rebuilt.get_post())} vs {len(pipeline.get_post())})"
    )


@pytest.mark.parametrize(
    "name, pipeline",
    _PREFAB_CASES,
    ids=[name for name, _ in _PREFAB_CASES],
)
def test_prefab_pipeline_class_names_preserved(name: str, pipeline: Any) -> None:
    """The class names in ops/meas/post survive the DAG round-trip."""

    original_ops = [type(op).__name__ for op in pipeline.get_ops().values()]
    original_meas = [type(op).__name__ for op in pipeline.get_meas().values()]
    original_post = [type(op).__name__ for op in pipeline.get_post().values()]

    state = from_pipeline_dag(pipeline)
    rebuilt = to_pipeline_dag(state)

    rebuilt_ops = [type(op).__name__ for op in rebuilt.get_ops().values()]
    rebuilt_meas = [type(op).__name__ for op in rebuilt.get_meas().values()]
    rebuilt_post = [type(op).__name__ for op in rebuilt.get_post().values()]

    assert sorted(rebuilt_ops) == sorted(original_ops)
    assert sorted(rebuilt_meas) == sorted(original_meas)
    assert sorted(rebuilt_post) == sorted(original_post)


def _shipped_pipeline_jsons() -> List[Path]:
    """Return any ``tests/fixtures/pipelines/*.json`` that exist on disk."""

    if not PIPELINE_JSON_DIR.exists():
        return []
    return sorted(PIPELINE_JSON_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "json_path",
    _shipped_pipeline_jsons(),
    ids=lambda p: p.name,
)
def test_shipped_pipeline_json_round_trips(json_path: Path) -> None:
    """If ``tests/fixtures/pipelines/`` ships JSON, round-trip each file."""

    from phenotypic import ImagePipeline

    pipeline = ImagePipeline.from_json(json.loads(json_path.read_text()))
    state = from_pipeline_dag(pipeline)
    rebuilt = to_pipeline_dag(state)
    assert len(rebuilt.get_ops()) == len(pipeline.get_ops())
    assert len(rebuilt.get_meas()) == len(pipeline.get_meas())
    assert len(rebuilt.get_post()) == len(pipeline.get_post())


def test_round_trip_idempotent_when_repeated() -> None:
    """Repeated DAG↔pipeline cycles converge (no drift across iterations)."""

    from phenotypic.prefab import HeavyOtsuPipeline

    pipeline = HeavyOtsuPipeline()
    state = from_pipeline_dag(pipeline)
    once = to_pipeline_dag(state)
    twice = to_pipeline_dag(from_pipeline_dag(once))
    assert len(twice.get_ops()) == len(once.get_ops())
    assert len(twice.get_meas()) == len(once.get_meas())
    assert len(twice.get_post()) == len(once.get_post())
