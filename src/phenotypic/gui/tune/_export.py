"""Export tuned pipelines from winning or Pareto-front parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phenotypic import ImagePipeline
from phenotypic.sdk_ import (
    best_params_path,
    best_pipeline_path,
    atomic_write_text,
    pareto_best_pipeline_path,
    resolve_tuning_spec_path,
)
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune._spec import TuningSpec


@dataclass(frozen=True)
class PreparedPipelineExport:
    """Complete in-memory pipeline export awaiting atomic publication."""

    path: Path
    payload: str


def export_winning_pipeline(
    base: ImagePipeline,
    params: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write the single-objective tuned winner pipeline."""
    pipeline = build_pipeline(base, params)
    path = best_pipeline_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.to_json(path)
    return path


def export_pareto_pipeline(
    base: ImagePipeline,
    params: dict[str, Any],
    output_dir: Path,
    *,
    objective: str,
) -> Path:
    """Write a per-objective Pareto tuned pipeline."""
    pipeline = build_pipeline(base, params)
    path = pareto_best_pipeline_path(output_dir, objective)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.to_json(path)
    return path


def _params_from_best_params_payload(payload: object) -> dict[str, Any]:
    """Extract flat knob params from canonical or legacy best-params payloads."""
    if not isinstance(payload, dict):
        raise ValueError("best params must be a JSON object")
    wrapped = payload.get("params")
    if isinstance(wrapped, dict):
        return wrapped
    if "params" in payload:
        raise ValueError("best params 'params' must be a JSON object")
    return payload


def prepare_best_from_run(output_dir: Path) -> PreparedPipelineExport:
    """Read winner inputs and build the complete pipeline payload in memory."""
    spec_path = resolve_tuning_spec_path(output_dir)
    if not spec_path.is_file():
        raise FileNotFoundError(f"tuning spec not found: {spec_path}")

    params_path = best_params_path(output_dir)
    if not params_path.is_file():
        raise FileNotFoundError(f"best params not found: {params_path}")

    spec = TuningSpec.model_validate_json(spec_path.read_text())
    params = _params_from_best_params_payload(json.loads(params_path.read_text()))
    pipeline = build_pipeline(spec.pipeline, params)
    payload = pipeline.to_json()
    if not isinstance(payload, str):  # pragma: no cover
        raise RuntimeError("in-memory pipeline serialization returned no payload")
    return PreparedPipelineExport(
        path=best_pipeline_path(output_dir),
        payload=payload,
    )


def publish_prepared_export(prepared: PreparedPipelineExport) -> Path:
    """Atomically publish one fully prepared pipeline payload."""
    atomic_write_text(prepared.path, prepared.payload)
    return prepared.path


def export_best_from_run(output_dir: Path) -> Path:
    """Read a completed run's winner params and atomically write its pipeline."""
    return publish_prepared_export(prepare_best_from_run(output_dir))


__all__ = [
    "export_best_from_run",
    "export_pareto_pipeline",
    "export_winning_pipeline",
    "prepare_best_from_run",
    "publish_prepared_export",
    "PreparedPipelineExport",
]
