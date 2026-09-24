"""The run preflight: read-only checks that refuse an incompatible run early.

Spec: ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §0, §2.

``phenotypic_cli`` calls :func:`run_preflight` in the read-only half of its
body, after the pipeline has loaded and before the ``--dry-run`` exit, so a
refusal happens before anything under ``--output`` changes. Every check reads
only the pipeline, the options, the environment, the cluster configuration and
file headers; none runs an operation on an image. A sample run would conflate
"this configuration is incompatible" with "this one image is bad", which is
the failure mode this module exists to avoid.

Checks return findings instead of raising, so one launch reports every
problem. A finding that makes every image fail is an ``error`` and refuses the
run; one that affects only some inputs is a ``warning``, because the CLI
already isolates per-image failures.

"Preflight" also names ``preflight_plot_backends``, the GUI's
``build_metadata_preflight`` and the GPU placement refusal; user-facing text
calls this one the **run preflight**.

This module imports nothing heavy at module level (``tests/unit/ci``'s
startup-import guards): checks import what they need inside their bodies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

if TYPE_CHECKING:
    from phenotypic import ImagePipeline

    from ._cli_types import Dataset, ExecutionConfig

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning"]

RunMode = Literal["full", "measure", "process"]

#: Every stable finding identifier. A closed set: ``HINTS`` must cover it, and
#: the user docs list every value (spec §2). Later tasks extend it together
#: with ``HINTS`` as they add the checks that emit each code.
FindingCode = Literal[
    "PF-CHECK-CRASHED",
    "PF-PIPELINE-LOAD",
    "PF-GRID-IMAGE",
    "PF-GRID-PRESET",
    "PF-NO-DETECTOR",
]

#: One remedy per finding code, shown under the finding's message.
HINTS: dict[str, str] = {
    "PF-CHECK-CRASHED": (
        "This is a defect in the preflight, not in your run, and it did not "
        "block the run. Please report it with the message above."
    ),
    "PF-PIPELINE-LOAD": (
        "Fix the pipeline file named above, then run again; nothing under "
        "--output was changed."
    ),
    "PF-GRID-IMAGE": (
        "Run with --image-type GridImage (and --nrows/--ncols for your plate "
        "layout), or remove the grid operations listed above."
    ),
    "PF-GRID-PRESET": (
        "The pipeline's nrows/ncols preset makes measure() add a "
        "CenteredAutoGridFinder, which needs a GridImage. Run with "
        "--image-type GridImage, or remove nrows and ncols from the pipeline."
    ),
    "PF-NO-DETECTOR": (
        "Add an object detector (for example OtsuDetector) to the pipeline's "
        "ops. If a custom operation writes the object map itself, rerun with "
        "--skip-validation."
    ),
}

#: How many affected subjects a finding displays before summarizing the rest.
SUBJECT_DISPLAY_LIMIT = 20

#: The pipeline slots each mode executes at the ROOT pipeline (spec §0,
#: "Checks are scoped to what the mode runs"). ``full`` applies ``ops`` and
#: then measures; ``process`` only applies ``ops``; ``measure`` only measures a
#: stored image. A pipeline nested in ``ops`` is applied, not measured, so only
#: its own ``ops`` ever run -- see :func:`operations_in_scope`.
MODE_SLOTS: dict[str, frozenset[str]] = {
    "full": frozenset({"ops", "meas", "post", "filters", "model"}),
    "process": frozenset({"ops"}),
    "measure": frozenset({"meas", "post", "filters", "model"}),
}


@dataclass(frozen=True)
class PreflightFinding:
    """One problem the preflight found.

    Attributes:
        code: Stable identifier; one of :data:`FindingCode`.
        severity: ``"error"`` refuses the run; ``"warning"`` is reported and
            the run continues.
        message: What is wrong, naming the operation path, file or option.
        subjects: Affected paths or keys, all of them; display is capped at
            :data:`SUBJECT_DISPLAY_LIMIT`.
    """

    code: str
    severity: Severity
    message: str
    subjects: tuple[str, ...] = ()

    @property
    def hint(self) -> str:
        """The remedy registered for this finding's code."""
        return HINTS[self.code]


@dataclass(frozen=True)
class PreflightReport:
    """Every finding from one preflight, in the order the checks ran."""

    findings: tuple[PreflightFinding, ...]

    @property
    def errors(self) -> tuple[PreflightFinding, ...]:
        """Findings that refuse the run."""
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[PreflightFinding, ...]:
        """Findings that are reported while the run continues."""
        return tuple(f for f in self.findings if f.severity == "warning")

    def render_lines(self) -> list[str]:
        """The report as plain text lines, errors first.

        Returns:
            Lines ready to print; empty when there are no findings.
        """
        lines: list[str] = []
        for finding in (*self.errors, *self.warnings):
            label = "✗ Error" if finding.severity == "error" else "! Warning"
            lines.append(f"{label} [{finding.code}]: {finding.message}")
            shown = finding.subjects[:SUBJECT_DISPLAY_LIMIT]
            lines.extend(f"    - {subject}" for subject in shown)
            hidden = len(finding.subjects) - len(shown)
            if hidden > 0:
                lines.append(f"    … and {hidden} more")
            lines.append(f"    → {finding.hint}")
        return lines


@dataclass(frozen=True)
class PreflightContext:
    """Everything a check may read. Nothing in it may be written.

    Attributes:
        config: The invocation's execution configuration.
        pipeline: The loaded pipeline.
        datasets: The scanned inputs (stores, in ``measure`` mode).
        mode: Which part of the pipeline this run executes.
    """

    config: "ExecutionConfig"
    pipeline: "ImagePipeline"
    datasets: Sequence["Dataset"]
    mode: RunMode


Check = Callable[[PreflightContext], "list[PreflightFinding]"]

def _image_class_by_input(context: PreflightContext) -> dict[str, str]:
    """``"Image"`` or ``"GridImage"`` for every input, as the run will load it.

    ``full`` and ``process`` build every image as ``--image-type``. ``measure``
    loads each store as its recorded ``phenotypic.image_class``, with
    ``--image-type`` only as the fallback the worker uses
    (``load_image_from_store``), so the answer can differ per store.
    """
    fallback = str(context.config.image_type)
    inputs = [str(path) for dataset in context.datasets for path in dataset.images]
    if context.mode != "measure":
        return {path: fallback for path in inputs}

    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    classes: dict[str, str] = {}
    for path in inputs:
        try:
            block = read_phenotypic_attributes(Path(path))
        except (OSError, KeyError, ValueError):
            classes[path] = fallback
            continue
        recorded = block.get(PhenotypicAttr.IMAGE_CLASS, fallback)
        classes[path] = "GridImage" if recorded == "GridImage" else "Image"
    return classes


def _plain_image_reach(
    context: PreflightContext,
) -> "tuple[list[str], Severity] | None":
    """Which inputs the run loads as a plain ``Image``, and the finding severity.

    Returns:
        ``None`` when no input is a plain ``Image``. Otherwise the affected
        inputs and ``"error"`` when every input is affected, ``"warning"`` when
        only some are. With no inputs scanned the answer is ``--image-type``
        alone, and a plain ``Image`` then counts as every input.
    """
    classes = _image_class_by_input(context)
    if not classes:
        if str(context.config.image_type) == "Image":
            return [], "error"
        return None
    plain = [path for path, cls in classes.items() if cls == "Image"]
    if not plain:
        return None
    return plain, "error" if len(plain) == len(classes) else "warning"


def check_grid_image(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-GRID-IMAGE``: grid operations that will meet a plain ``Image``.

    Spec §4, F4. Every such operation derives from one of the four ABCs that
    raise ``GridImageInputError`` on a plain Image, which is what
    ``preflight_requirements().grid_image`` reports.
    """
    grid_ops = [
        "/".join(path)
        for path, operation in operations_in_scope(context)
        if operation.preflight_requirements().grid_image
    ]
    if not grid_ops:
        return []
    reach = _plain_image_reach(context)
    if reach is None:
        return []
    plain, severity = reach
    where = (
        "--image-type Image"
        if context.mode != "measure"
        else f"{len(plain)} store(s) recorded as a plain Image"
    )
    return [
        PreflightFinding(
            code="PF-GRID-IMAGE",
            severity=severity,
            message=(
                f"grid operation(s) {', '.join(grid_ops)} require a GridImage, "
                f"but the run uses {where}; each such image would fail with "
                "GridImageInputError"
            ),
            subjects=tuple(plain) if context.mode == "measure" else (),
        )
    ]


def check_grid_preset(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-GRID-PRESET``: a preset that injects a grid finder into a plain ``Image``.

    Spec §4, F5 (review R5). ``measure()`` injects ``CenteredAutoGridFinder``
    only when the pipeline carries **both** ``nrows`` and ``ncols`` and ``meas``
    holds no ``GridFinder`` (``_image_pipeline_core.py:1305-1316``). Under
    ``--image-type Image`` the CLI never applies ``--nrows``/``--ncols`` to the
    pipeline, so only the preset matters. ``process`` never measures.
    """
    if context.mode == "process":
        return []
    pipeline = context.pipeline
    if pipeline._nrows is None or pipeline._ncols is None:
        return []
    from phenotypic.abc_ import GridFinder

    if any(isinstance(m, GridFinder) for m in pipeline.get_meas().values()):
        return []
    reach = _plain_image_reach(context)
    if reach is None:
        return []
    plain, severity = reach
    return [
        PreflightFinding(
            code="PF-GRID-PRESET",
            severity=severity,
            message=(
                f"the pipeline presets nrows={pipeline._nrows}, "
                f"ncols={pipeline._ncols}, so measure() adds a "
                "CenteredAutoGridFinder, which fails on a plain Image"
            ),
            subjects=tuple(plain) if context.mode == "measure" else (),
        )
    ]


def check_detector_present(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-NO-DETECTOR``: a forward run with nothing that produces objects.

    Spec §4, F6. A freshly read image has an empty object map; ``measure()``
    then raises ``NoObjectsError`` from ``_get_image_info`` (plain Image) or from
    the default ``CenteredAutoGridFinder`` (GridImage), for every image. Only
    ``full`` mode applies ``ops`` and then measures: ``measure`` mode measures
    stored object maps and ``process`` mode never measures.
    """
    if context.mode != "full":
        return []
    from phenotypic.abc_ import ObjectDetector
    from phenotypic.sdk_._operation_tree import pipeline_slot_of

    for path, operation in operations_in_scope(context):
        if pipeline_slot_of(context.pipeline, path[0]) is not None:
            continue  # reached through meas/post/...: runs after the op chain
        if isinstance(operation, ObjectDetector):
            return []
    return [
        PreflightFinding(
            code="PF-NO-DETECTOR",
            severity="error",
            message=(
                "the pipeline's ops contain no object detector, so every "
                "image reaches measure() with no objects and fails with "
                "NoObjectsError"
            ),
        )
    ]


#: The checks :func:`run_preflight` runs, in order: pipeline, environment,
#: cluster, inputs, metadata, output. Later tasks register theirs here.
CHECKS: tuple[Check, ...] = (
    check_grid_image,
    check_grid_preset,
    check_detector_present,
)


def run_mode_of(config: "ExecutionConfig") -> RunMode:
    """The :data:`RunMode` an execution configuration selects."""
    if config.measure_only:
        return "measure"
    if config.process_only_layer is not None:
        return "process"
    return "full"


def operations_in_scope(
    context: PreflightContext,
) -> list[tuple[tuple[str, ...], Any]]:
    """``(path, operation)`` for every operation this run will execute.

    Walks the whole tree with ``walk_operations`` and keeps a node only when
    every pipeline slot on its path executes in this mode: at the root, the
    slot must be in ``MODE_SLOTS[context.mode]``; in a pipeline nested in
    ``ops``, only ``ops`` runs, because the parent applies it
    (``operation.apply``) and never measures it. Fields of a non-pipeline
    operation (a composite's branches, a measurer's nested detector) run
    whenever their owner does.

    Every requirement check iterates this, never the whole tree (spec D10):
    walking everything would refuse a ``process`` run over a measurement it
    never takes.

    Args:
        context: The preflight context.

    Returns:
        In-scope ``(path, operation)`` pairs in depth-first order.
    """
    from phenotypic.sdk_._operation_tree import (
        get_at_path,
        pipeline_slot_of,
        walk_operations,
    )

    root_slots = MODE_SLOTS[context.mode]
    in_scope: list[tuple[tuple[str, ...], Any]] = []
    for path, operation in walk_operations(context.pipeline):
        if all(
            _segment_runs(
                get_at_path(context.pipeline, path[:depth]),
                segment,
                allowed=root_slots if depth == 0 else frozenset({"ops"}),
                pipeline_slot_of=pipeline_slot_of,
            )
            for depth, segment in enumerate(path)
        ):
            in_scope.append((path, operation))
    return in_scope


def _segment_runs(
    owner: Any,
    segment: str,
    *,
    allowed: frozenset[str],
    pipeline_slot_of: Callable[[Any, str], "str | None"],
) -> bool:
    """Whether the child *segment* of *owner* executes, given *allowed* slots."""
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        ImagePipelineCore,
    )

    if not isinstance(owner, ImagePipelineCore):
        return True
    return (pipeline_slot_of(owner, segment) or "ops") in allowed


def run_preflight(
    context: PreflightContext,
    checks: Sequence[Check] | None = None,
) -> PreflightReport:
    """Run every check and collect its findings.

    A check that raises does not stop the others and never refuses the run: it
    becomes a ``PF-CHECK-CRASHED`` warning naming the check (spec §0).

    Args:
        context: What the checks may read.
        checks: The checks to run; defaults to :data:`CHECKS`.

    Returns:
        The report, in check order.
    """
    findings: list[PreflightFinding] = []
    for check in CHECKS if checks is None else checks:
        try:
            findings.extend(check(context))
        except Exception as exc:  # noqa: BLE001 -- a checker defect must not block a run
            name = getattr(check, "__qualname__", repr(check))
            logger.debug("preflight check %s crashed", name, exc_info=True)
            findings.append(
                PreflightFinding(
                    code="PF-CHECK-CRASHED",
                    severity="warning",
                    message=(
                        f"the preflight check {name} could not run: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
    return PreflightReport(tuple(findings))


def load_pipeline_for_validation(
    pipeline_path: Path,
) -> "tuple[ImagePipeline | None, PreflightFinding | None]":
    """Load a pipeline and run the checks that need nothing but the file.

    The single loader behind CLI validation: it replaces the load that
    ``validate_pipeline`` performed and then discarded, so the preflight can
    check the same object (spec §2, review R19). ``validate_pipeline`` keeps
    its ``(bool, str)`` contract as a wrapper over this.

    Args:
        pipeline_path: The ``--pipeline`` file.

    Returns:
        ``(pipeline, None)`` when the file loads and passes, else
        ``(None, finding)`` with a ``PF-PIPELINE-LOAD`` error whose message
        is the text ``validate_pipeline`` has always reported.
    """
    from ._cli_validation import check_loaded_pipeline, read_pipeline_file

    pipeline, error = read_pipeline_file(pipeline_path)
    if pipeline is None:
        return None, _load_error(error)
    error = check_loaded_pipeline(pipeline)
    if error is not None:
        return None, _load_error(error)
    return pipeline, None


def _load_error(message: "str | None") -> PreflightFinding:
    return PreflightFinding(
        code="PF-PIPELINE-LOAD",
        severity="error",
        message=message or "the pipeline could not be loaded",
    )
