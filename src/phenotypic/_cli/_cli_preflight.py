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
import os
from dataclasses import dataclass, field
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
    "PF-CUSTOM-OP",
    "PF-GRID-IMAGE",
    "PF-GRID-PRESET",
    "PF-NO-DETECTOR",
    "PF-MISSING-MODULE",
    "PF-LICENSE",
    "PF-WEIGHTS-UNCACHED",
    "PF-SBATCH-REJECTED",
    "PF-SBATCH-UNAVAILABLE",
    "PF-TIME-OVER-PARTITION",
    "PF-SLURM-LIMIT",
    "PF-GPU-PARTITION",
    "PF-HEADER-UNREADABLE",
    "PF-CHANNELS",
    "PF-DETECT-MODE-GRAY",
    "PF-RGB-OP-GRAY",
    "PF-BIT-DEPTH",
    "PF-RAW-NO-RAWPY",
    "PF-STEM-COLLISION",
    "PF-META-PARSE",
    "PF-META-ALIAS",
    "PF-META-NO-KEYS",
    "PF-META-DUP-KEYS",
    "PF-META-UNMATCHED",
    "PF-META-ORPHANS",
    "PF-META-UNVERIFIED",
    "PF-OUTPUT-UNWRITABLE",
    "PF-OUTPUT-SPACE",
    "PF-NODE-LOCAL",
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
    "PF-CUSTOM-OP": (
        "List a module in PHENOTYPIC_PRELOAD_MODULES whose import attaches the "
        "class to the phenotypic namespace (for example "
        "`phenotypic.MyDetector = MyDetector`). A module that only defines "
        "the class is not enough: pipeline JSON records bare class names."
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
    "PF-MISSING-MODULE": (
        "Install the package in the environment the run uses, e.g. "
        "`uv sync --extra <extra>` for the extra named above; micro-sam is "
        "installed with conda (`conda install -c conda-forge micro_sam`)."
    ),
    "PF-LICENSE": (
        "Read the model's license, then accept it for this run, e.g. "
        "`export PHENOTYPIC_ACCEPT_MODEL_LICENSE=<name>` (comma-separate "
        "several). Gated Hugging Face models also need `uv run hf auth login`."
    ),
    "PF-WEIGHTS-UNCACHED": (
        "The weights will be downloaded by the first worker that needs them, "
        "which requires network access on the compute node. Pre-download them "
        "on a node with network access: `uv run python -m phenotypic.detect.nn "
        "download --help`."
    ),
    "PF-HEADER-UNREADABLE": (
        "Replace or remove the files listed; they cannot be opened as images. "
        "A truncated file whose header still parses is not caught here."
    ),
    "PF-CHANNELS": (
        "Convert the listed files to grayscale or RGB(A); Image.imread refuses "
        "2-channel and 5-or-more-channel images."
    ),
    "PF-DETECT-MODE-GRAY": (
        "Use --detect-mode gray for grayscale images, or supply RGB images."
    ),
    "PF-RGB-OP-GRAY": (
        "Supply RGB images, or remove the colour operations listed above for "
        "grayscale runs."
    ),
    "PF-BIT-DEPTH": (
        "Drop --bit-depth to let each image report its own depth, or set it to "
        "the depth the files actually store."
    ),
    "PF-RAW-NO-RAWPY": (
        "Install rawpy (`uv sync`; it is a core dependency except on Windows), "
        "or convert the RAW files to TIFF first."
    ),
    "PF-STEM-COLLISION": (
        "Rename or move one file of each pair listed: within a dataset, each "
        "input's name without its extension must be unique."
    ),
    "PF-META-PARSE": (
        "Fix the CSV so it parses with full-file type inference; check for a "
        "value that does not fit its column (e.g. text in a numeric column)."
    ),
    "PF-META-ALIAS": (
        "The CSV carries both a legacy and a current spelling of one metadata "
        "column with conflicting values; keep one spelling."
    ),
    "PF-META-NO-KEYS": (
        "Add a column that identifies each image, e.g. ImageName (the file "
        "name without its extension), optionally with Dataset."
    ),
    "PF-META-DUP-KEYS": (
        "Make each join key unique in the CSV; duplicated keys copy every "
        "measured row once per duplicate."
    ),
    "PF-META-UNMATCHED": (
        "Add rows for the listed images, or check that ImageName values match "
        "file names without extensions; unmatched images are dropped from "
        "measurements.csv (the master table keeps them)."
    ),
    "PF-META-ORPHANS": (
        "Expected when the CSV describes wells or strains that grew nothing; "
        "they appear as metadata-only rows (QC_MetadataOnly)."
    ),
    "PF-META-UNVERIFIED": (
        "These columns can only be matched against measurements, so the "
        "preflight cannot check them; the production join uses any of them "
        "that the measurements emit."
    ),
    "PF-SBATCH-REJECTED": (
        "Fix the --slurm/--gpu-slurm option sbatch names above (a partition, "
        "account, QoS, time, memory or GPU request the cluster refuses, or a "
        "misspelled key); nothing was submitted."
    ),
    "PF-SBATCH-UNAVAILABLE": (
        "The profile could not be confirmed. If sbatch's message names drained or "
        "unavailable nodes, the job will queue until they return; otherwise the "
        "run still reports a real submission failure when it submits."
    ),
    "PF-TIME-OVER-PARTITION": (
        "Lower the requested time to the partition's MaxTime, or choose a "
        "partition that allows it (unless your QOS overrides the limit)."
    ),
    "PF-SLURM-LIMIT": (
        "Reduce --gpu-shards, or ask for a higher MaxSubmitJobs; the staged "
        "GPU engine needs 3 submission slots and one GPU array chunk."
    ),
    "PF-GPU-PARTITION": (
        "Point the GPU job at a partition with GPUs: "
        "--gpu-slurm slurm_partition=<gpu-partition> for a staged run, or "
        "--slurm slurm_partition=<gpu-partition> for --mode process."
    ),
    "PF-OUTPUT-UNWRITABLE": (
        "Choose an --output you can write to, or fix the directory's permissions."
    ),
    "PF-OUTPUT-SPACE": (
        "Free space or choose another --output. This compares free space with "
        "the inputs' total size, a heuristic, and cannot see GPFS user quotas "
        "(check those with your site's quota command)."
    ),
    "PF-NODE-LOCAL": (
        "Put these paths on shared storage (e.g. /bigdata or a home "
        "directory): compute nodes cannot see another node's local disk or "
        "tmpfs. On a login node that itself exports the path over NFS, the "
        "warning can be spurious: compute nodes then see it as nfs."
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
    #: Per-preflight memo shared by the checks (e.g. input headers, read once).
    #: Excluded from equality and repr; the context itself stays frozen.
    scratch: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


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


def _requirements_in_scope(context: PreflightContext) -> list[tuple[str, Any]]:
    """``("path/to/op", requirements)`` for every in-scope operation."""
    return [
        ("/".join(path), operation.preflight_requirements())
        for path, operation in operations_in_scope(context)
    ]


def check_optional_modules(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-MISSING-MODULE``: an operation's lazily imported package is absent.

    Spec §5, F9. ``find_spec`` locates a package without importing it, so this
    costs no ``torch`` import in the submitting process.
    """
    import importlib.util

    # One finding per package, however many operations and extras need it
    # (Sam2 names the 'torch' extra, Sam3 the 'foundation' extra; both pull
    # torch).
    missing: dict[str, tuple[list[str], list[str]]] = {}
    for path, requirements in _requirements_in_scope(context):
        for module in requirements.modules:
            try:
                present = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                present = False
            if not present:
                paths, extras = missing.setdefault(module, ([], []))
                paths.append(path)
                if requirements.extra and requirements.extra not in extras:
                    extras.append(requirements.extra)
    return [
        PreflightFinding(
            code="PF-MISSING-MODULE",
            severity="error",
            message=(
                f"the package {module!r} is not installed, but "
                f"{', '.join(paths)} {'import' if len(paths) > 1 else 'imports'} "
                "it at run time"
                + (
                    f" (provided by the {' or '.join(repr(e) for e in extras)} "
                    f"extra{'s' if len(extras) > 1 else ''})"
                    if extras
                    else ""
                )
            ),
        )
        for module, (paths, extras) in missing.items()
    ]


def _weights_in_scope(context: PreflightContext) -> list[tuple[str, Any]]:
    return [
        (path, weight)
        for path, requirements in _requirements_in_scope(context)
        for weight in requirements.weights
    ]


def check_model_licenses(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-LICENSE``: gated weights whose license this run has not accepted.

    Spec §5, §10.3. No runtime path prompts any more, so an unaccepted license
    fails every image. ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` is parsed by
    ``accepted_model_licenses``, the same function the runtime gate uses.
    """
    from phenotypic.detect.nn._helper._checkpoint_manager import (
        accepted_model_licenses,
    )

    accepted = accepted_model_licenses()
    findings = []
    for path, weight in _weights_in_scope(context):
        if weight.license_key and weight.license_key.lower() not in accepted:
            findings.append(
                PreflightFinding(
                    code="PF-LICENSE",
                    severity="error",
                    message=(
                        f"{path} loads {weight.model}, whose license has not "
                        "been accepted: PHENOTYPIC_ACCEPT_MODEL_LICENSE does not "
                        f"include {weight.license_key!r}"
                    ),
                )
            )
    return findings


def check_model_weights_cached(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-WEIGHTS-UNCACHED``: weights that a worker would have to download.

    Spec §5, F12. A warning, not an error: some clusters give compute nodes
    network access. A probe that cannot tell (``None``) reports nothing.
    """
    findings = []
    for path, weight in _weights_in_scope(context):
        if weight.is_cached() is False:
            findings.append(
                PreflightFinding(
                    code="PF-WEIGHTS-UNCACHED",
                    severity="warning",
                    message=f"{path} loads {weight.model}, which is not in the local cache",
                )
            )
    return findings


def _severity_for(affected: int, total: int) -> Severity:
    """``error`` when every input is affected, else ``warning`` (spec §0)."""
    return "error" if total and affected >= total else "warning"


#: ``sbatch`` stderr fragments that mean the controller could not be reached,
#: which says nothing about the configuration (review R15).
SBATCH_COMMUNICATION_PATTERNS: tuple[str, ...] = (
    "Socket timed out",
    "Unable to contact slurm controller",
    "Slurm backup controller in standby mode",
    "Transport endpoint is not connected",
)

#: ``sbatch --test-only`` failures that name a configuration fault a real
#: submission rejects in the same way, matched case-insensitively. Only these
#: are errors (Phase E review E1). The controller strings are Slurm's
#: ``slurm_errno.c`` (SchedMD/slurm ``master``, read 2026-09-25, lines 221,
#: 415, 427, 521); the last two are ``sbatch``'s own option parsing, which
#: reads the same script either way.
SBATCH_REJECTION_PATTERNS: tuple[str, ...] = (
    "invalid partition name specified",
    "invalid account or account/partition combination specified",
    "invalid qos specification",
    "invalid generic resource (gres) specification",
    "unrecognized option",
    "invalid option",
)


def _scheduler_available(name: str) -> bool:
    """Whether the Slurm client *name* is on ``PATH``."""
    import shutil

    return shutil.which(name) is not None


def _run_scheduler_command(
    command: Sequence[str],
    *,
    input: "str | None" = None,
    timeout: float,
    env: "dict[str, str] | None" = None,
) -> Any:
    """Run one read-only Slurm query; the one seam every cluster check uses.

    Every call carries an explicit timeout (30 s for ``sbatch``, 10 s for
    ``scontrol`` and ``sinfo``). Nothing it runs submits a job.
    """
    import subprocess

    return subprocess.run(
        list(command), input=input, capture_output=True, text=True,
        timeout=timeout, env=env,
    )


def _staged_slurm_run(context: PreflightContext) -> bool:
    """Whether the run takes the staged GPU SLURM path (``StagedSlurmStrategy``)."""
    if context.mode != "full" or not context.config.is_slurm_mode():
        return False
    from ._cli_validation import find_gpu_detectors

    try:
        return bool(find_gpu_detectors(context.pipeline))
    except ValueError:
        return False  # an unstageable placement is refused before this runs


def _autonomous_gpu_run(context: PreflightContext) -> bool:
    """A GPU pipeline that ``AutonomousSLURMStrategy`` runs (not staged, not ``measure``).

    That strategy submits the ``--slurm`` profile with a GPU request added
    (``with_default_gpu_request``), so the preflight tests that profile, not
    the bare one (Phase E review E2). ``--mode process`` is this case.
    """
    config = context.config
    if context.mode == "measure" or not config.is_slurm_mode() or _staged_slurm_run(context):
        return False
    from ._cli_validation import find_gpu_detectors

    try:
        return bool(find_gpu_detectors(context.pipeline))
    except ValueError:
        return False


def _slurm_profiles(context: PreflightContext) -> list[tuple[str, dict[str, Any]]]:
    """``(label, args)`` for every SBATCH profile the run will submit."""
    if not context.config.is_slurm_mode():
        return []
    if _autonomous_gpu_run(context):
        from phenotypic.sdk_.slurm import with_default_gpu_request

        return [(
            "profile (--slurm, with the GPU request the run adds)",
            with_default_gpu_request(context.config.slurm_args),
        )]
    profiles = [("CPU profile (--slurm)", dict(context.config.slurm_args))]
    if _staged_slurm_run(context):
        from ._cli_staged_slurm import resolve_stage_slurm_args

        profiles.append((
            "GPU profile (--gpu-slurm over --slurm)",
            resolve_stage_slurm_args(
                context.config.gpu_slurm_args, context.config.slurm_args
            ),
        ))
    return profiles


def render_test_script(profile: dict[str, Any]) -> str:
    """A minimal batch script for ``sbatch --test-only`` (review R15).

    ``format_sbatch_directives`` emits no shebang, and ``sbatch`` rejects a
    script whose first line is not ``#!``; logs go to ``/dev/null`` because
    nothing runs.
    """
    from phenotypic.sdk_.slurm import format_sbatch_directives

    directives = format_sbatch_directives(
        job_name="phenotypic-preflight",
        slurm_args=profile,
        output_log=Path("/dev/null"),
        error_log=Path("/dev/null"),
    )
    return f"#!/bin/bash\n{directives.rstrip()}\ntrue\n"


def check_slurm_profiles(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-SBATCH-REJECTED``: ``sbatch --test-only`` refuses a profile.

    Spec §6, F17. Tests each profile the run submits in one call, without
    submitting a job and without writing a file (the script goes on stdin).
    The environment is the one ``submit_script`` uses, so ``SBATCH_*``
    variables apply identically.

    Only a failure in ``SBATCH_REJECTION_PATTERNS`` is an error. Every other
    failure is the ``PF-SBATCH-UNAVAILABLE`` warning, carrying sbatch's
    message, because ``--test-only`` fails where a real submission queues:
    its will-run test ignores DOWN and DRAINED nodes ("Requested node
    configuration is not available" during a maintenance drain), it does not
    retry a full queue or a busy controller as ``sbatch`` does, and it reports
    controller and authentication faults that say nothing about the profile
    (Phase E review E1, from Slurm's ``job_scheduler.c``, ``job_mgr.c`` and
    ``sbatch.c``).
    """
    profiles = _slurm_profiles(context)
    if not profiles:
        return []

    def unavailable(detail: str) -> PreflightFinding:
        return PreflightFinding(
            "PF-SBATCH-UNAVAILABLE", "warning",
            f"the SLURM profile could not be checked: {detail}",
        )

    if not _scheduler_available("sbatch"):
        return [unavailable("sbatch is not on PATH")]
    import subprocess

    from phenotypic.sdk_.slurm import sbatch_submission_environment

    findings: list[PreflightFinding] = []
    for label, profile in profiles:
        try:
            result = _run_scheduler_command(
                ["sbatch", "--test-only"],
                input=render_test_script(profile),
                timeout=30,
                env=sbatch_submission_environment(),
            )
        except subprocess.TimeoutExpired:
            findings.append(unavailable(f"sbatch --test-only timed out for the {label}"))
            continue
        except OSError as exc:
            findings.append(unavailable(f"{type(exc).__name__}: {exc}"))
            continue
        if result.returncode == 0:
            continue
        detail = (result.stderr or result.stdout or "").strip()
        if any(pattern in detail.lower() for pattern in SBATCH_REJECTION_PATTERNS):
            findings.append(PreflightFinding(
                "PF-SBATCH-REJECTED", "error",
                f"sbatch --test-only rejected the {label}: {detail}",
            ))
        elif any(pattern in detail for pattern in SBATCH_COMMUNICATION_PATTERNS):
            findings.append(unavailable(detail))
        else:
            findings.append(unavailable(
                f"sbatch --test-only did not accept the {label}, for a reason a "
                f"real submission may queue through (e.g. drained nodes): {detail}"
            ))
    return findings


def parse_slurm_duration_minutes(value: str) -> "int | None":
    """Minutes in a Slurm duration, or ``None`` for an unlimited one.

    Accepts the forms Slurm prints and accepts: ``UNLIMITED``/``infinite``,
    ``D-HH:MM:SS``, ``D-HH:MM``, ``D-HH``, ``HH:MM:SS``, ``MM:SS`` and ``MM``.
    Seconds round up to the next minute.
    """
    text = str(value).strip()
    if text.lower() in {"unlimited", "infinite", "none", ""}:
        return None
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
        parts = [int(p) for p in text.split(":")] + [0, 0]
        hours, minutes, seconds = parts[0], parts[1], parts[2]
    else:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, (minutes, seconds) = 0, parts
        else:
            hours, minutes, seconds = 0, parts[0], 0
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def _partition_blocks(text: str) -> list[dict[str, str]]:
    """``scontrol show partition`` output as one ``{field: value}`` per partition."""
    blocks: list[dict[str, str]] = []
    for token in text.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if key == "PartitionName":
            blocks.append({})
        if blocks:
            blocks[-1][key] = value
    return blocks


def _enforce_part_limits() -> "str | None":
    import re

    try:
        result = _run_scheduler_command(["scontrol", "show", "config"], timeout=10)
    except Exception:  # noqa: BLE001 -- unknown is a valid answer here
        return None
    match = re.search(r"EnforcePartLimits\s*=\s*(\S+)", result.stdout or "")
    return match.group(1).upper() if match else None


def _effective_option(profile: dict[str, Any], option: str, env_var: str) -> "str | None":
    """What ``sbatch`` will use for ``--<option>``: its ``SBATCH_*`` variable, else the script.

    ``sbatch`` lets input environment variables override options set in the
    batch script (``sbatch.1``, INPUT ENVIRONMENT VARIABLES, NOTE), and the
    preflight hands ``sbatch`` the submission environment, so the variable
    comes first. The script's value is read back from the rendered
    directives, so either key spelling counts (Phase E review E3, E7).
    """
    from phenotypic.sdk_.slurm import effective_sbatch_option, sbatch_submission_environment

    return (
        sbatch_submission_environment().get(env_var)
        or effective_sbatch_option(profile, option)
    )


def check_partition_time(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-TIME-OVER-PARTITION``: a time limit above the partition's ``MaxTime``.

    Spec §6, F19 (review R11). A warning, not an error: a QOS with
    ``Flags=PartitionTimeLimit`` may override the partition limit, and the
    preflight does not resolve QOS flags.

    For a partition list the rule is ``EnforcePartLimits``'s (``slurm.conf.5``):
    under ``ALL`` the job must satisfy every requested partition, so the
    tightest ``MaxTime`` decides and the job is rejected; under ``ANY`` it is
    accepted if it satisfies one, and under ``NO`` it stays queued only when it
    exceeds all of them, so the loosest decides (Phase E review E7).
    """
    profiles = _slurm_profiles(context)
    if not profiles or not _scheduler_available("scontrol"):
        return []
    findings: list[PreflightFinding] = []
    enforce: "str | None" = None
    for label, profile in profiles:
        requested = _effective_option(profile, "time", "SBATCH_TIMELIMIT")
        if requested is None:
            continue
        minutes = parse_slurm_duration_minutes(requested)
        if minutes is None:
            continue
        partition = _effective_option(profile, "partition", "SBATCH_PARTITION")
        names = [n for n in str(partition or "").split(",") if n]
        if names:
            blocks = []
            for name in names:
                result = _run_scheduler_command(
                    ["scontrol", "show", "partition", name], timeout=10
                )
                blocks += _partition_blocks(result.stdout or "")
        else:
            result = _run_scheduler_command(["scontrol", "show", "partition"], timeout=10)
            blocks = [b for b in _partition_blocks(result.stdout or "") if b.get("Default") == "YES"]
        limits = [
            (b.get("PartitionName", "?"), b.get("MaxTime", ""), parse_slurm_duration_minutes(b.get("MaxTime", "")))
            for b in blocks
        ]
        if not limits:
            continue
        # An UNLIMITED partition satisfies any time, and so does any list holding one.
        bounded = [entry for entry in limits if entry[2] is not None]
        tightest = min(bounded, key=lambda entry: entry[2]) if bounded else None
        if tightest is None or minutes <= tightest[2]:
            continue
        if enforce is None:
            enforce = _enforce_part_limits() or "UNKNOWN"
        if enforce == "ALL":
            name, max_time, _ = tightest
        else:
            if len(bounded) < len(limits):
                continue
            name, max_time, loosest = max(bounded, key=lambda entry: entry[2])
            if minutes <= loosest:
                continue
        consequence = {
            "NO": "the job would be accepted and then pend indefinitely (EnforcePartLimits=NO)",
            "UNKNOWN": "the job would pend or be rejected (EnforcePartLimits could not be read)",
        }.get(enforce, f"the job would be rejected at submission (EnforcePartLimits={enforce})")
        findings.append(PreflightFinding(
            "PF-TIME-OVER-PARTITION", "warning",
            f"the {label} requests {requested}, above MaxTime {max_time} of "
            f"partition {name}; {consequence}",
        ))
    return findings


def _slurm_submission_limits() -> "tuple[int | None, int]":
    """``(MaxSubmitJobs, MaxArraySize)`` as the staged strategy reads them."""
    from phenotypic.sdk_.slurm import get_slurm_array_limit, get_slurm_max_submit_jobs

    return get_slurm_max_submit_jobs(), get_slurm_array_limit()


def check_staged_slurm_limits(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-SLURM-LIMIT``: limits the staged strategy would refuse after hashing.

    Spec §6, F18. The strategy checks these only after building its manifest,
    which hashes every input; the same function answers here first.
    """
    if not _staged_slurm_run(context):
        return []
    from ._cli_staged_slurm import staged_slurm_limit_errors

    max_submit, array_limit = _slurm_submission_limits()
    return [
        PreflightFinding("PF-SLURM-LIMIT", "error", message)
        for message in staged_slurm_limit_errors(
            max_submit, array_limit, context.config.gpu_shards
        )
    ]


def check_gpu_partition(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-GPU-PARTITION``: the GPU profile's partition lists no GPU.

    Spec §6, F18. The staged GPU stage's profile, or, for a GPU pipeline the
    non-staged strategy runs (``--mode process``), the profile with the GPU
    request that strategy adds (Phase E review E2). The partition is the one
    ``sbatch`` will use (review E3), and ``partition_gres_error`` answers only
    when ``sinfo`` positively lists that partition's GRES (review E5, E6).
    """
    if not _scheduler_available("sinfo"):
        return []
    if _staged_slurm_run(context):
        from ._cli_staged_slurm import resolve_stage_slurm_args

        profile = resolve_stage_slurm_args(
            context.config.gpu_slurm_args, context.config.slurm_args
        )
        which = "the GPU stage's"
    elif _autonomous_gpu_run(context):
        (_, profile), = _slurm_profiles(context)
        which = "the GPU pipeline's"
    else:
        return []
    from phenotypic.sdk_.slurm import effective_sbatch_option
    from phenotypic.sdk_.slurm._config import partition_gres_error

    if effective_sbatch_option(profile, "gpus-per-node") in (None, "0"):
        return []  # no GPU requested (an explicit slurm_gpus_per_node=0)
    partition = _effective_option(profile, "partition", "SBATCH_PARTITION")
    if not partition:
        return []
    message = partition_gres_error(
        str(partition),
        run=lambda command, timeout: _run_scheduler_command(command, timeout=timeout),
    )
    if message is None:
        return []
    return [PreflightFinding("PF-GPU-PARTITION", "error", f"{which} {message}")]


def _input_paths(context: PreflightContext) -> list[str]:
    return [str(path) for dataset in context.datasets for path in dataset.images]


def _input_headers(context: PreflightContext) -> list[Any]:
    """Every input's header, read once per preflight and shared by the checks.

    ``full`` and ``process`` read ``--input``; ``measure`` reads the stores it
    will re-measure. Headers only: see ``_cli_input_headers``.
    """
    if "headers" not in context.scratch:
        from ._cli_input_headers import read_input_headers

        context.scratch["headers"] = read_input_headers(_input_paths(context))
    return context.scratch["headers"]


def _reach_finding(
    code: str, context: PreflightContext, affected: list[str], message: str
) -> list[PreflightFinding]:
    if not affected:
        return []
    total = len(_input_paths(context))
    return [
        PreflightFinding(
            code=code,
            severity=_severity_for(len(affected), total),
            message=f"{message} ({len(affected)} of {total} input(s))",
            subjects=tuple(affected),
        )
    ]


def check_input_headers_readable(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-HEADER-UNREADABLE``: inputs whose header cannot be parsed (spec §7).

    A zero-byte or unidentifiable file. A truncated file whose header parses is
    not caught here; only a decode finds it (header-behavior.md).
    """
    if context.mode == "measure":
        return []
    affected = [h.path for h in _input_headers(context) if h.error]
    return _reach_finding(
        "PF-HEADER-UNREADABLE", context, affected,
        "these inputs have an unreadable header and will fail to load",
    )


def check_input_channels(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-CHANNELS``: channel counts ``imread`` refuses (2, or 5 and more)."""
    if context.mode == "measure":
        return []
    affected = [h.path for h in _input_headers(context) if h.raw_channels is not None]
    return _reach_finding(
        "PF-CHANNELS", context, affected,
        "these inputs store a channel count Image.imread refuses "
        "(\"Image with N channels (unknown format)\")",
    )


def _gray_inputs(context: PreflightContext) -> list[str]:
    return [h.path for h in _input_headers(context) if h.channels == 1]


def check_detect_mode_on_gray(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-DETECT-MODE-GRAY``: a colour ``--detect-mode`` over grayscale inputs.

    Spec §7, F7. Every forward and process worker calls ``set_detect_mode``
    after reading, which raises on an image without RGB; ``measure`` never
    applies ``--detect-mode``.
    """
    if context.mode == "measure" or context.config.detect_mode == "gray":
        return []
    from phenotypic._core._image_parts.detection_modes import get_detection_mode

    if not get_detection_mode(context.config.detect_mode).requires_rgb:
        return []
    return _reach_finding(
        "PF-DETECT-MODE-GRAY", context, _gray_inputs(context),
        f"--detect-mode {context.config.detect_mode} needs RGB, but these inputs "
        "are grayscale and would each fail with \"image has no RGB data\"",
    )


def check_rgb_ops_on_gray(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-RGB-OP-GRAY``: an in-scope RGB-reading operation over grayscale inputs.

    Spec §7, F8. In ``measure`` mode the inputs are stores, whose recorded
    series say whether they hold RGB.
    """
    readers = [
        "/".join(path)
        for path, operation in operations_in_scope(context)
        if operation.preflight_requirements().rgb_input
    ]
    if not readers:
        return []
    return _reach_finding(
        "PF-RGB-OP-GRAY", context, _gray_inputs(context),
        f"{', '.join(readers)} read(s) RGB, but these inputs are grayscale",
    )


def check_bit_depth(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-BIT-DEPTH``: ``--bit-depth`` that contradicts the stored dtype.

    ``imread`` accepts the contradiction silently and records the wrong bit
    depth for the data (header-behavior.md), so every intensity normalized by
    it is off by a factor of 256 in one direction or the other.

    JPEG is exempt: ``imread`` sets ``bit_depth = 8`` for a JPEG whatever the
    flag says (``_image_io_handler.py``), so nothing is mislabelled (review
    D3).
    """
    from phenotypic.sdk_.constants_ import IO

    bit_depth = context.config.bit_depth
    if context.mode == "measure" or bit_depth is None:
        return []
    jpeg = {suffix.lower() for suffix in IO.JPEG_FILE_EXTENSIONS}
    affected = [
        h.path for h in _input_headers(context)
        if h.bits is not None
        and h.bits != int(bit_depth)
        and Path(h.path).suffix.lower() not in jpeg
    ]
    return _reach_finding(
        "PF-BIT-DEPTH", context, affected,
        f"--bit-depth {bit_depth} contradicts these inputs' stored sample depth, "
        "which Image.imread would silently mislabel",
    )


def check_raw_needs_rawpy(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-RAW-NO-RAWPY``: camera-RAW inputs without ``rawpy`` (spec §7, §12)."""
    if context.mode == "measure":
        return []
    import importlib.util

    from phenotypic.sdk_.constants_ import IO

    raw_suffixes = {s.lower() for s in IO.RAW_FILE_EXTENSIONS}
    raw = [p for p in _input_paths(context) if Path(p).suffix.lower() in raw_suffixes]
    if not raw or importlib.util.find_spec("rawpy") is not None:
        return []
    return _reach_finding(
        "PF-RAW-NO-RAWPY", context, raw,
        "these camera-RAW inputs need the optional package 'rawpy', which is not installed",
    )


def check_stem_collisions(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-STEM-COLLISION``: two inputs of one dataset share a stem (F28).

    They map to one store and one metadata key; a probe showed both written to
    one ``<stem>.ome.zarr`` and the run failing only at publication
    (header-behavior.md). An error: it corrupts outputs, it does not fail one image.
    """
    if context.mode == "measure":
        return []
    from phenotypic.sdk_ import source_image_stem

    groups: dict[tuple[str, str], list[str]] = {}
    for dataset in context.datasets:
        for path in dataset.images:
            groups.setdefault((dataset.name, source_image_stem(Path(path))), []).append(str(path))
    clashes = {key: paths for key, paths in groups.items() if len(paths) > 1}
    if not clashes:
        return []
    names = ", ".join(f"{ds}/{stem}" for ds, stem in clashes)
    return [
        PreflightFinding(
            code="PF-STEM-COLLISION",
            severity="error",
            message=(
                f"inputs share a stem within a dataset ({names}); each group "
                "would be written to one store and one metadata key"
            ),
            subjects=tuple(p for paths in clashes.values() for p in paths),
        )
    ]


def check_metadata_join(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-META-*``: how ``--metadata`` would join onto the scanned images.

    Spec §9, F21 (review R1). Only ``full`` mode joins metadata: ``process``
    ignores ``--metadata`` and ``measure`` joins nothing new. The CSV is read
    with the one shared reader and normalized exactly as the production join
    normalizes it (``prepare_metadata_join_keys``).

    The key findings are errors only when the join is fully knowable here.
    A per-well plate map keyed on ``ImageName + Grid_RowNum + Grid_ColNum``
    looks duplicated against the source-only key frame, and a layout keyed on
    grid position alone looks keyless, yet both join correctly against the
    measurement frame; with such a measurement-level column present the same
    findings are warnings. They are warnings too when the run's metadata set
    is not known (``_metadata_set_is_complete``): inputs that restore
    PhenoTypic metadata, or a custom operation, can carry the very columns the
    CSV is keyed on (review D2).

    A qualified CSV column counts as a possible measurement key only when it
    is a known schema header (``Grid_RowNum``), which is when the production
    join keeps its raw spelling (``external_metadata_preserved_columns``);
    any other, such as ``Strain_ID``, is joined as an attribute. With a custom
    operation in scope any qualified column may be one it emits (review D4).
    """
    metadata_csv = context.config.metadata_csv
    if context.mode != "full" or metadata_csv is None:
        return []
    from ._metadata_join import read_metadata_csv
    from ._metadata_preflight import analyze_metadata_join

    try:
        frame = read_metadata_csv(metadata_csv)
    except Exception as exc:  # noqa: BLE001 -- any parse failure is the finding
        return [PreflightFinding("PF-META-PARSE", "error", f"{metadata_csv} does not parse: {exc}")]
    images = [(d.name, Path(p)) for d in context.datasets for p in d.images]
    try:
        analysis = analyze_metadata_join(images, frame)
    except ValueError as exc:
        return [PreflightFinding("PF-META-ALIAS", "error", str(exc))]

    unverified = analysis.unverified_join_columns
    if not _custom_operation_in_scope(context):
        import phenotypic.schema as _schema

        known_headers = _schema.header_to_module()
        unverified = tuple(c for c in unverified if c in known_headers)
    complete = _metadata_set_is_complete(context)
    key_severity: Severity = "error" if complete and not unverified else "warning"
    if unverified:
        caveat = "; it may still join on the measurement-level columns below"
    elif not complete:
        caveat = (
            "; the inputs restore PhenoTypic metadata or a custom operation "
            "may set it, so a key may still be carried by the images"
        )
    else:
        caveat = ", so without a measurement-level key nothing would join"
    findings: list[PreflightFinding] = []
    if not analysis.join_columns:
        findings.append(PreflightFinding(
            "PF-META-NO-KEYS", key_severity,
            f"{metadata_csv} shares no column with the images' ImageName, "
            "FileSuffix or Dataset" + caveat,
        ))
    if analysis.duplicate_key_count:
        findings.append(PreflightFinding(
            "PF-META-DUP-KEYS", key_severity,
            f"{analysis.duplicate_key_count} metadata row(s) repeat a key on "
            f"{', '.join(analysis.join_columns)}"
            + ("" if key_severity == "error" else " (they may be distinguished by "
               "columns only the measurements carry)"),
        ))
    if analysis.unmatched_images:
        findings.append(PreflightFinding(
            "PF-META-UNMATCHED", "warning",
            f"{len(analysis.unmatched_images)} of {analysis.source_count} image(s) "
            f"have no metadata row on {', '.join(analysis.join_columns)}",
            subjects=analysis.unmatched_images,
        ))
    if analysis.metadata_only_count and analysis.join_columns:
        findings.append(PreflightFinding(
            "PF-META-ORPHANS", "warning",
            f"{analysis.metadata_only_count} of {analysis.metadata_row_count} "
            "metadata row(s) match no image",
        ))
    if unverified:
        findings.append(PreflightFinding(
            "PF-META-UNVERIFIED", "warning",
            f"the CSV joins on measurement-level column(s) {', '.join(unverified)}, "
            "which cannot be checked before measuring",
        ))
    return findings


def _metadata_set_is_complete(context: PreflightContext) -> bool:
    """Whether every metadata column the run can carry is known (spec §9, R6).

    Not when an in-scope operation is a class from outside ``phenotypic`` (it
    may set ``image.metadata``), and not when an input restores PhenoTypic
    metadata on read (a store, or a file carrying the ``phenotypic`` key).
    """
    if context.mode == "measure" or _custom_operation_in_scope(context):
        return False
    return not any(h.carries_phenotypic_metadata for h in _input_headers(context))


def _custom_operation_in_scope(context: PreflightContext) -> bool:
    """Whether an operation this run executes is a class from outside ``phenotypic``."""
    return any(
        not type(operation).__module__.startswith("phenotypic.")
        for _, operation in operations_in_scope(context)
    )


#: Filesystem types that live on one node. Shared types (gpfs, lustre, nfs,
#: nfs4, beegfs, cifs, cephfs, ...) and anything unrecognized produce nothing.
#: ``overlay`` is deliberately absent: it is the root of a container image,
#: and a path baked into the image is visible to workers running the same
#: image (Phase E review E8).
_NODE_LOCAL_FILESYSTEMS = frozenset(
    {"tmpfs", "ramfs", "devtmpfs", "ext2", "ext3", "ext4", "xfs", "btrfs"}
)
_MOUNTS = Path("/proc/self/mounts")


def _mounts_text() -> "str | None":
    """``/proc/self/mounts``, or ``None`` off Linux (the check then skips)."""
    try:
        return _MOUNTS.read_text(encoding="utf-8")
    except OSError:
        return None


def _unescape_mount_field(field: str) -> str:
    """Decode ``/proc/self/mounts``' octal escapes (``\\040`` space, ``\\011`` tab, ...)."""
    import re

    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), field)


def _filesystem_type(path: Path, mounts: str) -> "str | None":
    """The filesystem type of the mount that contains *path*.

    The longest containing mount point wins, and of two entries for the same
    point the later one, because a later mount covers an earlier one there
    (review E8).
    """
    resolved = Path(os.path.abspath(path)).resolve(strict=False)
    best: "tuple[int, str] | None" = None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point = Path(_unescape_mount_field(fields[1]))
        if resolved == mount_point or resolved.is_relative_to(mount_point):
            depth = len(mount_point.parts)
            if best is None or depth >= best[0]:
                best = (depth, fields[2])
    return best[1] if best else None


def _nearest_existing_ancestor(path: Path) -> Path:
    current = Path(os.path.abspath(path))
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def check_output_writable(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-OUTPUT-UNWRITABLE``: ``--output`` cannot be created or written.

    Spec §9, F25. The directory may not exist yet, so the probe is
    ``os.access`` on its nearest existing ancestor; ``access(2)`` consults
    ACLs and, on NFS, the server. The three output-location checks are
    separate so that a fault in one cannot hide another's finding (review
    E13).
    """
    if context.config.output_dir is None:
        return []
    anchor = _nearest_existing_ancestor(Path(context.config.output_dir))
    if os.access(anchor, os.W_OK | os.X_OK):
        return []
    return [PreflightFinding(
        "PF-OUTPUT-UNWRITABLE", "error",
        f"{anchor} (the nearest existing directory of --output) is not writable",
    )]


def check_output_space(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-OUTPUT-SPACE``: less free space than the inputs' size (``full`` only).

    Spec §9, R21: a heuristic, labelled so, and blind to GPFS quotas.
    """
    import shutil

    if context.mode != "full" or context.config.output_dir is None:
        return []
    anchor = _nearest_existing_ancestor(Path(context.config.output_dir))
    if not os.access(anchor, os.W_OK | os.X_OK):
        return []  # check_output_writable reports this
    total = 0
    for path in _input_paths(context):
        try:
            if Path(path).is_file():
                total += Path(path).stat().st_size
        except OSError:
            continue
    free = shutil.disk_usage(anchor).free
    if not total or free >= total:
        return []
    return [PreflightFinding(
        "PF-OUTPUT-SPACE", "warning",
        f"{anchor} has {free / 2**30:.1f} GiB free, less than the inputs' "
        f"{total / 2**30:.1f} GiB (a heuristic, not an estimate of the "
        "run's footprint)",
    )]


def check_node_local_paths(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-NODE-LOCAL``: a SLURM run's paths on storage other nodes cannot see.

    Spec §9, F29 (review R30): ``--output``, ``--input``, ``--pipeline``,
    ``--metadata`` and every in-scope ``JoinMetadata`` table, judged by the
    filesystem type of its mount rather than by its name.
    """
    config = context.config
    if not config.is_slurm_mode():
        return []
    mounts = _mounts_text()
    if mounts is None:
        return []
    candidates: list[tuple[str, Path]] = [
        (option, Path(value))
        for option, value in (
            ("--output", config.output_dir),
            ("--input", config.input_path),
            ("--pipeline", config.pipeline_json),
            ("--metadata", config.metadata_csv),
        )
        if value is not None
    ]
    from phenotypic.post import JoinMetadata

    for path, operation in operations_in_scope(context):
        if isinstance(operation, JoinMetadata):
            candidates.append((f"{'/'.join(path)} table", Path(operation.metadata)))
    local = [
        f"{label} {candidate} ({fs})"
        for label, candidate in candidates
        if (fs := _filesystem_type(candidate, mounts)) in _NODE_LOCAL_FILESYSTEMS
    ]
    if not local:
        return []
    return [PreflightFinding(
        "PF-NODE-LOCAL", "warning",
        "on a SLURM run these paths are on node-local storage, which "
        "workers on other nodes cannot see",
        subjects=tuple(local),
    )]


#: The checks :func:`run_preflight` runs, in order: pipeline, environment,
#: cluster, inputs, metadata, output. Later tasks register theirs here.
CHECKS: tuple[Check, ...] = (
    check_grid_image,
    check_grid_preset,
    check_detector_present,
    check_optional_modules,
    check_model_licenses,
    check_model_weights_cached,
    check_slurm_profiles,
    check_partition_time,
    check_staged_slurm_limits,
    check_gpu_partition,
    check_input_headers_readable,
    check_input_channels,
    check_stem_collisions,
    check_raw_needs_rawpy,
    check_detect_mode_on_gray,
    check_rgb_ops_on_gray,
    check_bit_depth,
    check_metadata_join,
    check_output_writable,
    check_output_space,
    check_node_local_paths,
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
        ``(None, finding)``: ``PF-CUSTOM-OP`` when an operation class cannot be
        resolved, otherwise ``PF-PIPELINE-LOAD``, each carrying the text
        ``validate_pipeline`` has always reported.
    """
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        UnknownOperationClassError,
    )

    from ._cli_validation import check_loaded_pipeline, read_pipeline_file

    pipeline, error, exc = read_pipeline_file(pipeline_path)
    if pipeline is None:
        if isinstance(exc, UnknownOperationClassError):
            return None, PreflightFinding(
                code="PF-CUSTOM-OP",
                severity="error",
                message=error or str(exc),
            )
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
