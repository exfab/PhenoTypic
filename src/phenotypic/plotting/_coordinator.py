"""Lifecycle-aware plot dispatch for CLI and GUI refresh seams."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from phenotypic.abc_.plotting import PlotAnalysis, PlotImage, PlotMeas, PlotQc
from phenotypic.sdk_ import plots_dir

from ._adapter import FigureAdapter
from ._analysis_registry import AnalysisRegistry
from ._bindings import AnalysisInput, MeasurementInput, PlotBinding, PlotInput
from ._output import normalize_plot_output
from ._writer import publish_plot_output, safe_path_component

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QcPlotSubject:
    """Runtime-only payload passed to :class:`PlotQc` consumers."""

    input_table: pd.DataFrame
    input_ref: MeasurementInput | AnalysisInput
    qc_instance_id: str | None = None
    analyzed_check: Any = None
    qc_database: Path | None = None
    review_state: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


class PlotCoordinator:
    """Dispatch normalized pipeline bindings at their declared lifecycles.

    Args:
        pipeline: Pipeline containing normalized plot bindings.
        output_dir: Full CLI output root. Used with :func:`plots_dir` unless
            ``plots_base`` is supplied.
        plots_base: Explicit resolved plots directory. GUI callers use their
            :class:`BundleLayout` path so standalone deliverables bundles do
            not acquire a second ``deliverables`` segment.
    """

    def __init__(
        self,
        pipeline: Any,
        output_dir: Path,
        *,
        plots_base: Path | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._plots_base = (
            Path(plots_base)
            if plots_base is not None
            else plots_dir(Path(output_dir))
        )

    def emit_image(
        self,
        image: Any,
        *,
        dataset: str,
        image_stem: str,
    ) -> None:
        """Emit every ``PlotImage`` after one image has been measured."""
        for binding in self._bindings(PlotImage):
            try:
                value = binding.plot.inspect(image, for_save=True)
                self._publish_image_value(
                    binding,
                    value,
                    dataset=dataset,
                    image_stem=image_stem,
                )
            except Exception:  # noqa: BLE001 - plot output is best-effort
                logger.warning(
                    "Plot %s failed during image inspect", binding.id,
                    exc_info=True,
                )

    def emit_measurements(self, measurements: pd.DataFrame) -> None:
        """Emit every ``PlotMeas`` from the current measurement mirror."""
        for binding in self._bindings(PlotMeas):
            self._emit_aggregate(binding, measurements, lifecycle="measurements")

    def emit_analyses(
        self,
        measurements: pd.DataFrame,
        registry: AnalysisRegistry,
        *,
        updated_input: PlotInput | None = None,
        refresh_producers: bool = False,
    ) -> tuple[str, ...]:
        """Emit matching ``PlotAnalysis`` bindings from resolved inputs.

        Args:
            measurements: Current post-applied measurement mirror.
            registry: Dynamically resolved named analysis tables.
            updated_input: Optional dependency filter. ``None`` emits all.
            refresh_producers: Re-run measurement-consuming analyzer producers
                before plotting, as required after GUI measurement edits.
        """
        refreshed_analysis_ids: list[str] = []
        for binding in self._bindings(PlotAnalysis):
            try:
                input_ref = binding.input or MeasurementInput()
                updated_result = (
                    registry.get(updated_input.analysis_id)
                    if isinstance(updated_input, AnalysisInput)
                    else None
                )
                if updated_input is not None and input_ref != updated_input:
                    if (
                        updated_result is None
                        or updated_result.producer is not binding.plot
                    ):
                        continue
                if (
                    refresh_producers
                    and isinstance(input_ref, MeasurementInput)
                ):
                    refreshed_id = self._refresh_analysis_producer(
                        binding, measurements, registry
                    )
                    if refreshed_id is not None:
                        refreshed_analysis_ids.append(refreshed_id)
                if isinstance(input_ref, AnalysisInput):
                    result = registry.get(input_ref.analysis_id)
                    reused = result is not None and result.producer is binding.plot
                    subject: Any = registry.resolve(input_ref)
                else:
                    result = (
                        updated_result
                        if updated_result is not None
                        and updated_result.producer is binding.plot
                        else registry.get(type(binding.plot).__name__)
                    )
                    reused = result is not None and result.producer is binding.plot
                    subject = None if reused else measurements
                value = (
                    binding.plot.inspect(for_save=True)
                    if reused
                    else binding.plot.inspect(subject, for_save=True)
                )
                self._publish_aggregate(binding, value)
            except Exception:  # noqa: BLE001 - plot output is best-effort
                logger.warning(
                    "Plot %s failed during analysis inspect", binding.id,
                    exc_info=True,
                )
        return tuple(dict.fromkeys(refreshed_analysis_ids))

    def _refresh_analysis_producer(
        self,
        binding: PlotBinding,
        measurements: pd.DataFrame,
        registry: AnalysisRegistry,
    ) -> str | None:
        """Refresh fitted state for a measurement-consuming analyzer plot."""
        analyzer = binding.plot
        pipeline_model = self._pipeline.get_model()
        if analyzer is pipeline_model:
            table = self._pipeline.analyze(measurements)
        else:
            analyze = getattr(analyzer, "analyze", None)
            if not callable(analyze):
                return None
            table = analyze(measurements)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"analysis plot {binding.id!r} producer returned "
                f"{type(table).__name__}, expected pandas.DataFrame"
            )
        analysis_id = type(analyzer).__name__
        registry.register(
            analysis_id,
            table,
            producer=analyzer,
        )
        return analysis_id

    def emit_qc(
        self,
        measurements: pd.DataFrame,
        registry: AnalysisRegistry,
        *,
        successful_modules: Mapping[str, Any] | None = None,
        qc_database: Path | None = None,
        review_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit every ``PlotQc`` with a fresh input and analyzed-check context."""
        modules = successful_modules or {}
        immutable_state = MappingProxyType(dict(review_state or {}))
        for configured in self._pipeline.get_plots():
            try:
                ref = configured.ref
                is_qc_ref = ref is not None and ref.slot == "qc"
                module_key = configured.id
                if is_qc_ref:
                    assert ref is not None and ref.key is not None
                    module_key = ref.key
                module = modules.get(module_key)
                plot = configured.plot
                if is_qc_ref and module is not None:
                    plot = module.check
                if not isinstance(plot, PlotQc):
                    continue
                binding = configured.model_copy(update={"plot": plot})
                input_ref = binding.input or MeasurementInput()
                table = (
                    registry.resolve(input_ref)
                    if isinstance(input_ref, AnalysisInput)
                    else measurements
                )
                subject = QcPlotSubject(
                    input_table=table,
                    input_ref=input_ref,
                    qc_instance_id=(
                        getattr(module, "instance_id", module_key)
                        if module is not None
                        else None
                    ),
                    analyzed_check=getattr(module, "check", None),
                    qc_database=qc_database,
                    review_state=immutable_state,
                )
                self._emit_aggregate(binding, subject, lifecycle="qc")
            except Exception:  # noqa: BLE001 - plot output is best-effort
                logger.warning(
                    "Plot %s failed during QC inspect", binding.id,
                    exc_info=True,
                )

    def emit_dependent_qc(
        self,
        measurements: pd.DataFrame,
        registry: AnalysisRegistry,
        *,
        updated_input: PlotInput,
    ) -> None:
        """Refresh standalone ``PlotQc`` consumers of one updated table.

        QC-recipe references are excluded because they require a freshly
        analyzed check instance and are emitted only after a QC rebuild.

        Args:
            measurements: Current post-applied measurement mirror.
            registry: Dynamically resolved named analysis tables.
            updated_input: Input whose generation just changed.
        """
        immutable_state: Mapping[str, Any] = MappingProxyType({})
        for binding in self._pipeline.get_plots():
            ref = binding.ref
            if ref is not None and ref.slot == "qc":
                continue
            if not isinstance(binding.plot, PlotQc):
                continue
            input_ref = binding.input or MeasurementInput()
            if input_ref != updated_input:
                continue
            try:
                table = (
                    registry.resolve(input_ref)
                    if isinstance(input_ref, AnalysisInput)
                    else measurements
                )
                subject = QcPlotSubject(
                    input_table=table,
                    input_ref=input_ref,
                    review_state=immutable_state,
                )
                self._emit_aggregate(binding, subject, lifecycle="qc dependency")
            except Exception:  # noqa: BLE001 - plot output is best-effort
                logger.warning(
                    "Plot %s failed during dependent QC inspect",
                    binding.id,
                    exc_info=True,
                )

    def _bindings(self, lifecycle: type[Any]) -> list[PlotBinding]:
        return [
            binding
            for binding in self._pipeline.get_plots()
            if isinstance(binding.plot, lifecycle)
        ]

    def _emit_aggregate(
        self,
        binding: PlotBinding,
        subject: Any,
        *,
        lifecycle: str,
    ) -> None:
        try:
            value = binding.plot.inspect(subject, for_save=True)
            self._publish_aggregate(binding, value)
        except Exception:  # noqa: BLE001 - plot output is best-effort
            logger.warning(
                "Plot %s failed during %s inspect", binding.id, lifecycle,
                exc_info=True,
            )

    def _publish_aggregate(self, binding: PlotBinding, value: Any) -> None:
        directory = self._plots_base / safe_path_component(binding.id)
        publish_plot_output(
            value,
            directory,
            plot_id=binding.id,
            plot_class=type(binding.plot).__name__,
        )

    def _publish_image_value(
        self,
        binding: PlotBinding,
        value: Any,
        *,
        dataset: str,
        image_stem: str,
    ) -> None:
        output = normalize_plot_output(value)
        output_stem = _image_output_stem(dataset, image_stem)
        base = (
            self._plots_base
            / safe_path_component(binding.id)
            / safe_path_component(dataset)
        )
        if len(output.pages) != 1 or output.pages[0].key != "default":
            publish_plot_output(
                output,
                base / output_stem,
                plot_id=binding.id,
                plot_class=type(binding.plot).__name__,
            )
            return
        destination = base / f"{output_stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            FigureAdapter.save_png(output.pages[0].figure, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def _image_output_stem(dataset: str, image_stem: str) -> str:
    """Return a stable filename stem unique to the original image identity.

    Sanitization is intentionally many-to-one and filesystems may compare names
    case-insensitively. Hashing the unsanitized dataset/stem pair prevents two
    source images from overwriting each other while keeping rerun paths stable.
    """
    identity = dataset.encode("utf-8") + b"\0" + image_stem.encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:12]
    return f"{safe_path_component(image_stem)}-{digest}"


__all__ = ["PlotCoordinator", "QcPlotSubject"]
