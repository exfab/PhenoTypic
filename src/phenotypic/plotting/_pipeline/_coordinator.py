"""Lifecycle-aware plot dispatch for CLI and GUI refresh seams."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from collections.abc import Callable
from typing import Any, Mapping

import pandas as pd

from phenotypic.abc_.plotting import PlotAnalysis, PlotImage, PlotMeas, PlotQc
from phenotypic.sdk_ import CommitGuard, plots_dir

from ._adapter import FigureAdapter
from ._analysis_registry import AnalysisRegistry
from ._bindings import AnalysisInput, MeasurementInput, PlotBinding, PlotInput
from ._failures import _format_error, record_plot_failure
from ._output import normalize_plot_output
from ._writer import (
    PlotPublicationBlocked,
    _enter_commit,
    _render_page,
    publish_plot_output,
    safe_path_component,
)

logger = logging.getLogger(__name__)

#: ``plot_class`` recorded when a failure precedes binding resolution, so the
#: class that would have emitted is unknown. For a QC reference the configured
#: object is the recipe entry, not the check substituted for it, so naming its
#: class would record a plausible-looking wrong answer instead of an absence.
_UNRESOLVED_PLOT_CLASS = "<unresolved>"


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
        publication_guard: Optional GUI compare-and-set predicate forwarded
            to the transactional plot writer. CLI callers omit it.
    """

    def __init__(
        self,
        pipeline: Any,
        output_dir: Path,
        *,
        plots_base: Path | None = None,
        publication_guard: Callable[[], bool] | None = None,
        commit_guard: CommitGuard | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._plots_base = (
            Path(plots_base)
            if plots_base is not None
            else plots_dir(Path(output_dir))
        )
        self._publication_guard = publication_guard
        self._commit_guard = commit_guard

    def emit_image(
        self,
        image: Any,
        *,
        dataset: str,
        image_stem: str,
        strict: bool = False,
    ) -> None:
        """Emit every ``PlotImage`` after one image has been measured.

        Args:
            image: Image passed to each image plot binding.
            dataset: Dataset name used in output paths.
            image_stem: Image stem used in output paths.
            strict: Re-raise publication failures instead of logging them.
        """
        for binding in self._bindings(PlotImage):
            try:
                value = binding.plot.inspect(image, for_save=True)
                self._publish_image_value(
                    binding,
                    value,
                    dataset=dataset,
                    image_stem=image_stem,
                )
            except PlotPublicationBlocked:
                # A refused guard or fence voids the whole refresh: this
                # process may no longer own the output. It is not a plot
                # failure, so it is never recorded -- recording would write
                # into the tree the guard just refused. Same in every handler.
                raise
            except Exception as exc:  # noqa: BLE001 - plot output is best-effort
                if strict:
                    raise
                self._record_failure(
                    binding,
                    exc,
                    lifecycle="image",
                    dataset=dataset,
                    image_stem=image_stem,
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
            except PlotPublicationBlocked:
                raise
            except Exception as exc:  # noqa: BLE001 - plot output is best-effort
                self._record_failure(binding, exc, lifecycle="analysis")
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
            # Reset EVERY iteration, before the prelude. Assigned only after
            # the prelude resolves the plot, so a prelude failure must not see
            # the previous iteration's binding -- it would record the failure
            # against the wrong plot. The prelude stays inside the `try` by
            # decision: one malformed binding must not stop the QC bindings
            # after it (spec DEFERRED.md records the cost).
            binding: PlotBinding | None = None
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
            except PlotPublicationBlocked:
                raise
            except Exception as exc:  # noqa: BLE001 - plot output is best-effort
                if binding is not None:
                    self._record_failure(binding, exc, lifecycle="qc")
                else:
                    # `configured.id` IS the binding id -- `model_copy` only
                    # replaces `plot`. The class is what cannot be known here.
                    self._record_failure_by_name(
                        binding_id=configured.id,
                        plot_class=_UNRESOLVED_PLOT_CLASS,
                        error=exc,
                        lifecycle="qc",
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
                # "qc", not "qc dependency": `lifecycle` is a field of the
                # durable record and belongs to a closed set. The dependency
                # refresh is the same emit point; the distinction lives only
                # in the log line.
                self._emit_aggregate(
                    binding, subject, lifecycle="qc", log_label="dependent QC"
                )
            except PlotPublicationBlocked:
                raise
            except Exception as exc:  # noqa: BLE001 - plot output is best-effort
                self._record_failure(
                    binding, exc, lifecycle="qc", log_label="dependent QC"
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
        log_label: str | None = None,
    ) -> None:
        try:
            value = binding.plot.inspect(subject, for_save=True)
            self._publish_aggregate(binding, value)
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - plot output is best-effort
            self._record_failure(
                binding, exc, lifecycle=lifecycle, log_label=log_label
            )

    def _record_failure(
        self,
        binding: PlotBinding,
        error: BaseException,
        *,
        lifecycle: str,
        log_label: str | None = None,
        dataset: str | None = None,
        image_stem: str | None = None,
    ) -> None:
        """Log and durably record one swallowed failure of a resolved binding."""
        self._record_failure_by_name(
            binding_id=binding.id,
            plot_class=type(binding.plot).__name__,
            error=error,
            lifecycle=lifecycle,
            log_label=log_label,
            dataset=dataset,
            image_stem=image_stem,
        )

    def _record_failure_by_name(
        self,
        *,
        binding_id: str,
        plot_class: str,
        error: BaseException,
        lifecycle: str,
        log_label: str | None = None,
        dataset: str | None = None,
        image_stem: str | None = None,
    ) -> None:
        """Log and durably record one swallowed plot failure.

        Takes the identity directly for the one caller that has no resolved
        binding to read it from. ``record_plot_failure`` never raises, so this
        is safe to call from inside an ``except`` block.

        Args:
            binding_id: Stable plot binding id.
            plot_class: Producer class name, or ``_UNRESOLVED_PLOT_CLASS``.
            error: The swallowed exception.
            lifecycle: Emit point; one of the closed set documented on
                :func:`record_plot_failure`.
            log_label: Log-line wording when it differs from ``lifecycle``.
            dataset: Dataset name, for the image lifecycle only.
            image_stem: Image stem, for the image lifecycle only.
        """
        logger.warning(
            "Plot %s failed during %s inspect",
            binding_id,
            log_label or lifecycle,
            exc_info=error,
        )
        record_plot_failure(
            self._plots_base,
            binding_id=binding_id,
            plot_class=plot_class,
            lifecycle=lifecycle,
            error=error,
            dataset=dataset,
            image_stem=image_stem,
        )

    def _publish_aggregate(self, binding: PlotBinding, value: Any) -> None:
        directory = self._plots_base / safe_path_component(binding.id)
        publish_plot_output(
            value,
            directory,
            plot_id=binding.id,
            plot_class=type(binding.plot).__name__,
            plots_base=self._plots_base,
            publication_guard=self._publication_guard,
            commit_guard=self._commit_guard,
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
                plots_base=self._plots_base,
                publication_guard=self._publication_guard,
                commit_guard=self._commit_guard,
            )
            return
        # The flat single-page path: every bare figure lands here, as
        # `<stem>-<hash>.{html,png}` directly under `base`. Routing it through
        # `publish_plot_output` per image, as the multi-page branch above does,
        # would work, but costs a directory and a `manifest.json` per image and
        # changes the documented layout to `<stem>-<hash>/default.*`. No
        # manifest is written here, so a Chrome-less machine leaves HTML and no
        # PNG with nothing recording why: that is a missing capability, not a
        # failure, and the CLI preflight announces it.
        figure = output.pages[0].figure
        try:
            self._require_publication()
            base.mkdir(parents=True, exist_ok=True)
            files, errors, backend = _render_page(
                figure,
                base,
                output_stem,
                plots_base=self._plots_base,
                plot_id=binding.id,
                publication_guard=self._publication_guard,
                commit_guard=self._commit_guard,
            )
        finally:
            FigureAdapter.close(figure)
        self._remove_stale_sibling(base, output_stem, backend, files)
        # Recorded as raised: the class is the diagnostic, so never re-wrap.
        for error in errors:
            record_plot_failure(
                self._plots_base,
                binding_id=binding.id,
                plot_class=type(binding.plot).__name__,
                lifecycle="image",
                error=error,
                dataset=dataset,
                image_stem=image_stem,
            )
        if not files:
            # Recorded a second time by `emit_image`'s handler, deliberately:
            # the entry above names the renderer, this one the image. It also
            # keeps `strict=True` meaningful for this path, and `from` keeps
            # the renderer's exception reachable for a strict caller.
            raise RuntimeError(
                f"plot {binding.id!r} produced no file for "
                f"{dataset}/{image_stem}: "
                + (_format_error(errors[0]) if errors else "no renderer")
            ) from (errors[0] if errors else None)

    def _remove_stale_sibling(
        self,
        base: Path,
        output_stem: str,
        backend: str | None,
        files: Mapping[str, str],
    ) -> None:
        """Remove the rendering this generation did not write.

        With no manifest on this path, a file surviving from an earlier run
        beside a fresh one reads as this run's: a PNG from a node that had
        Chrome next to HTML from one that does not, or HTML left behind by a
        plot that has since switched to matplotlib.
        """
        stale: list[str] = []
        if backend == "plotly" and "png" not in files:
            stale.append(f"{output_stem}.png")
        if backend == "mpl":
            stale.append(f"{output_stem}.html")
        for name in stale:
            path = base / name
            if not path.exists():
                continue
            # `_enter_commit`, not `publication_commit`: a fenced removal must
            # surface as PlotPublicationBlocked like every other commit here,
            # or the handler would record the fence as a plot failure.
            with _enter_commit(self._commit_guard):
                self._require_publication()
                path.unlink(missing_ok=True)

    def _require_publication(self) -> None:
        """Fail closed immediately before a custom image-plot write."""
        if (
            self._publication_guard is not None
            and not self._publication_guard()
        ):
            raise PlotPublicationBlocked(
                "Plot publication blocked because its output snapshot changed."
            )


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
