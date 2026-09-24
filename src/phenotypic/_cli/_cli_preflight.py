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
    "PF-POST-COLUMN",
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
    "PF-POST-COLUMN": (
        "Correct the column name in the post operation, or add the column "
        "(e.g. through --metadata). At finalization a failing post operation "
        "is logged and ALL post-operation output is discarded."
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

    missing: dict[tuple[str, "str | None"], list[str]] = {}
    for path, requirements in _requirements_in_scope(context):
        for module in requirements.modules:
            try:
                present = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                present = False
            if not present:
                missing.setdefault((module, requirements.extra), []).append(path)
    return [
        PreflightFinding(
            code="PF-MISSING-MODULE",
            severity="error",
            message=(
                f"the package {module!r} is not installed, but "
                f"{', '.join(paths)} imports it at run time"
                + (f" (provided by the {extra!r} extra)" if extra else "")
            ),
        )
        for (module, extra), paths in missing.items()
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
    fails every image; ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` is read the same way
    ``require_license_acceptance`` reads it.
    """
    import os

    accepted = {
        name.strip().lower()
        for name in os.environ.get("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "").split(",")
        if name.strip()
    }
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
    """
    bit_depth = context.config.bit_depth
    if context.mode == "measure" or bit_depth is None:
        return []
    affected = [
        h.path for h in _input_headers(context)
        if h.bits is not None and h.bits != int(bit_depth)
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

    The key findings are errors only when the CSV has no measurement-level key
    column. A per-well plate map keyed on ``ImageName + Grid_RowNum +
    Grid_ColNum`` looks duplicated against the source-only key frame, and a
    layout keyed on grid position alone looks keyless, yet both join correctly
    against the measurement frame; with an unverifiable column present the
    same findings are warnings.
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
    key_severity: Severity = "warning" if unverified else "error"
    findings: list[PreflightFinding] = []
    if not analysis.join_columns:
        findings.append(PreflightFinding(
            "PF-META-NO-KEYS", key_severity,
            f"{metadata_csv} shares no column with the images' ImageName, "
            "FileSuffix or Dataset"
            + (", so without a measurement-level key nothing would join" if not unverified
               else "; it may still join on the measurement-level columns below"),
        ))
    if analysis.duplicate_key_count:
        findings.append(PreflightFinding(
            "PF-META-DUP-KEYS", key_severity,
            f"{analysis.duplicate_key_count} metadata row(s) repeat a key on "
            f"{', '.join(analysis.join_columns)}"
            + (" (possibly distinguished by the measurement-level columns below)" if unverified else ""),
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


def intrinsic_metadata_headers(image_type: str) -> tuple[str, ...]:
    """The metadata columns ``measure()`` inserts for a freshly read image.

    Read back from ``insert_metadata`` on one tiny in-memory image of the run's
    class, carrying the file suffix ``Image.imread`` records, rather than from
    a hand-kept list that could drift from it. A test pins it against a real
    ``imread`` + ``apply_and_measure``.

    Args:
        image_type: ``"Image"`` or ``"GridImage"``.

    Returns:
        The inserted metadata headers.
    """
    import numpy as np
    import pandas as pd

    from phenotypic import GridImage, Image
    from phenotypic.schema import IMAGE

    image_cls = GridImage if image_type == "GridImage" else Image
    image = image_cls(np.zeros((8, 8), dtype=np.uint8), name="preflight")
    image.metadata[IMAGE.SUFFIX] = ".tiff"
    frame = image.metadata.insert_metadata(pd.DataFrame())
    return tuple(str(column) for column in frame.columns)


def _measurement_headers(context: PreflightContext) -> set[str]:
    """Headers the in-scope measurers declare; incomplete by design (spec §8)."""
    from phenotypic.abc_ import MeasureFeatures

    headers: set[str] = set()
    for _, operation in operations_in_scope(context):
        if not isinstance(operation, MeasureFeatures):
            continue
        for info in operation.get_measurement_infoclasses():
            try:
                headers.update(str(h) for h in info.get_headers())
            except TypeError:
                continue  # e.g. TEXTURE.get_headers needs a scale argument
    return headers


def _metadata_csv_headers(context: PreflightContext) -> set[str]:
    """``--metadata`` headers as the production join normalizes them."""
    metadata_csv = context.config.metadata_csv
    if context.mode != "full" or metadata_csv is None:
        return set()
    from ._metadata_join import prepare_metadata_join_keys, read_metadata_csv
    from ._metadata_preflight import source_join_key_frame

    images = [(d.name, Path(p)) for d in context.datasets for p in d.images]
    try:
        prepared = prepare_metadata_join_keys(
            source_join_key_frame(images), read_metadata_csv(metadata_csv)
        )
    except Exception:  # noqa: BLE001 -- check_metadata_join reports it
        return set()
    return {str(column) for column in prepared.metadata.columns}


def _metadata_set_is_complete(context: PreflightContext) -> bool:
    """Whether every metadata column the run can carry is known (spec §8, R6).

    Not when an in-scope operation is a class from outside ``phenotypic`` (it
    may set ``image.metadata``), and not when an input restores PhenoTypic
    metadata on read (a store, or a file carrying the ``phenotypic`` key).
    """
    if context.mode == "measure":
        return False
    for _, operation in operations_in_scope(context):
        if not type(operation).__module__.startswith("phenotypic."):
            return False
    return not any(h.carries_phenotypic_metadata for h in _input_headers(context))


def check_post_columns(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-POST-COLUMN``: a post op naming a column the master will not carry.

    Spec §8, F23 (review R6). Post runs once, at finalization, on the joined
    master, and a raise there is logged at WARNING and discards the output of
    EVERY post op. The chain is walked in order, each op asked through its own
    ``preflight_columns`` (its own resolution rules) and credited with the
    columns it adds. A metadata-reading op's miss is an error only when the
    metadata set is provably complete; any other miss is a warning.
    """
    if "post" not in MODE_SLOTS[context.mode]:
        return []
    post_ops = list(context.pipeline.get_post().items())
    if not post_ops:
        return []
    from phenotypic.schema import EXPERIMENT

    known = set(intrinsic_metadata_headers(str(context.config.image_type)))
    known |= _measurement_headers(context)
    known |= _metadata_csv_headers(context)
    if context.config.include_dataset_column:
        known.add(str(EXPERIMENT.DATASET))
    complete = _metadata_set_is_complete(context)

    findings: list[PreflightFinding] = []
    for key, operation in post_ops:
        missing, produced = operation.preflight_columns(sorted(known))
        if missing:
            certain = complete and operation._preflight_reads_metadata_only
            findings.append(
                PreflightFinding(
                    code="PF-POST-COLUMN",
                    severity="error" if certain else "warning",
                    message=(
                        f"post:{key} ({type(operation).__name__}) reads "
                        f"{', '.join(missing)}, which the measurement table "
                        + ("will not carry" if certain else "may not carry")
                    ),
                )
            )
        known.update(produced)
    return findings


#: The checks :func:`run_preflight` runs, in order: pipeline, environment,
#: cluster, inputs, metadata, output. Later tasks register theirs here.
CHECKS: tuple[Check, ...] = (
    check_grid_image,
    check_grid_preset,
    check_detector_present,
    check_optional_modules,
    check_model_licenses,
    check_model_weights_cached,
    check_input_headers_readable,
    check_input_channels,
    check_stem_collisions,
    check_raw_needs_rawpy,
    check_detect_mode_on_gray,
    check_rgb_ops_on_gray,
    check_bit_depth,
    check_metadata_join,
    check_post_columns,
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
