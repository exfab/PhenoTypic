"""Shared GUI lifecycle hooks for configured pipeline plots."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import pandas as pd

from phenotypic.plotting._pipeline import (
    AnalysisInput,
    AnalysisRegistry,
    MeasurementInput,
    PlotCoordinator,
)

if TYPE_CHECKING:
    from phenotypic.plotting._pipeline import AnalysisResult
    from phenotypic.sdk_ import BundleLayout
    from phenotypic.sdk_._qc_recipe._runner import SuccessfulQcModule


def refresh_measurement_plots(
    pipeline: Any,
    layout: "BundleLayout",
    measurements: pd.DataFrame,
    *,
    publication_guard: Callable[[], bool] | None = None,
) -> None:
    """Refresh every configured ``PlotMeas`` from a GUI mirror update."""
    coordinator = _coordinator(
        pipeline,
        layout,
        publication_guard=publication_guard,
    )
    coordinator.emit_measurements(measurements)
    registry = AnalysisRegistry(layout.deliverables_base)
    refreshed_analysis_ids = coordinator.emit_analyses(
        measurements,
        registry,
        updated_input=MeasurementInput(),
        refresh_producers=True,
    ) or ()
    coordinator.emit_dependent_qc(
        measurements,
        registry,
        updated_input=MeasurementInput(),
    )
    for analysis_id in refreshed_analysis_ids:
        updated_analysis = AnalysisInput(analysis_id=analysis_id)
        coordinator.emit_analyses(
            measurements,
            registry,
            updated_input=updated_analysis,
        )
        coordinator.emit_dependent_qc(
            measurements,
            registry,
            updated_input=updated_analysis,
        )


def refresh_analysis_plots(
    pipeline: Any,
    layout: "BundleLayout",
    measurements: pd.DataFrame,
    result: "AnalysisResult",
    *,
    publication_guard: Callable[[], bool] | None = None,
) -> None:
    """Refresh every configured ``PlotAnalysis`` after GUI analysis output."""
    registry = AnalysisRegistry(layout.deliverables_base)
    registry.register(
        result.analysis_id,
        result.table,
        producer=result.producer,
        artifacts=result.artifacts,
        manifest_entry=result.manifest_entry,
    )
    coordinator = _coordinator(
        pipeline,
        layout,
        publication_guard=publication_guard,
    )
    coordinator.emit_analyses(
        measurements,
        registry,
        updated_input=AnalysisInput(analysis_id=result.analysis_id),
    )
    coordinator.emit_dependent_qc(
        measurements,
        registry,
        updated_input=AnalysisInput(analysis_id=result.analysis_id),
    )


def refresh_qc_plots(
    pipeline: Any,
    layout: "BundleLayout",
    measurements: pd.DataFrame,
    successful_modules: Iterable["SuccessfulQcModule"],
    *,
    publication_guard: Callable[[], bool] | None = None,
) -> None:
    """Refresh every configured ``PlotQc`` after a GUI QC rebuild."""
    modules = {module.instance_id: module for module in successful_modules}
    _coordinator(
        pipeline,
        layout,
        publication_guard=publication_guard,
    ).emit_qc(
        measurements,
        AnalysisRegistry(layout.deliverables_base),
        successful_modules=modules,
        qc_database=layout.qc_duckdb if modules else None,
        review_state=_read_review_state(layout.qc_review_state_path),
    )


def _coordinator(
    pipeline: Any,
    layout: "BundleLayout",
    *,
    publication_guard: Callable[[], bool] | None = None,
) -> PlotCoordinator:
    output_dir = layout.output_root or layout.deliverables_base
    return PlotCoordinator(
        pipeline,
        output_dir,
        plots_base=layout.plots_dir,
        publication_guard=publication_guard,
    )


def _read_review_state(path: Path) -> dict[str, Any]:
    """Return a defensive JSON snapshot for ``QcPlotSubject``."""
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = [
    "refresh_analysis_plots",
    "refresh_measurement_plots",
    "refresh_qc_plots",
]
