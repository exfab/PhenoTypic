"""Pure path resolution and atomic Setup authoring for Tune."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, TypedDict

from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.gui._config import tune_presets_dir
from phenotypic.gui.shell._metadata_context import resolve_metadata_csv
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import sandbox_fingerprint
from phenotypic.gui.tune._space import apply_space_edits, space_to_spec
from phenotypic.schema import METADATA
from phenotypic.sdk_ import (
    CONFIG_SUFFIX_TUNING,
    PIPELINE_CONFIG_SUFFIXES,
    atomic_write_text,
    matches_any_suffix,
)
from phenotypic.tune import TuningSpec
from phenotypic.tune.score import QCScorer

SetupPathKind = Literal["pipeline", "metadata"]
SetupPathSource = Literal["typed", "picker", "shared", "unset"]

_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
_METADATA_SUFFIXES = frozenset({".csv", ".parquet"})
_PIPELINE_SUFFIXES = PIPELINE_CONFIG_SUFFIXES | frozenset({CONFIG_SUFFIX_TUNING})


class SetupPathPayload(TypedDict):
    """Versioned, sandbox-bound current-session picker payload."""

    version: int
    kind: SetupPathKind
    relative_path: str
    absolute_path_at_selection: str
    sandbox_fingerprint: str
    selected_at: str


@dataclass(frozen=True)
class SetupPathResolution:
    """Resolved Setup path plus its precedence source and all issues."""

    path: Path | None
    source: SetupPathSource
    issues: tuple[str, ...] = ()

    def to_store(self) -> dict[str, object]:
        """Return a JSON-serializable Dash-store payload."""
        return {
            "path": str(self.path) if self.path is not None else None,
            "source": self.source,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class SetupAuthoringResult:
    """In-memory authored spec or its complete validation issue list."""

    spec: TuningSpec | None
    source_is_spec: bool
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Whether the full authored spec validated."""
        return self.spec is not None and not self.issues


def _safe_stem(path: Path) -> str:
    """Return a filesystem-safe stem for a GUI-authored spec."""
    stem = _SAFE_STEM.sub("-", path.stem).strip(".-")
    return stem or "tuning-spec"


def load_pipeline_or_spec(path: Path) -> ImagePipeline | TuningSpec:
    """Load a selected pipeline or existing tuning spec from disk."""
    text = path.read_text(encoding="utf-8")
    if matches_any_suffix(path, (CONFIG_SUFFIX_TUNING,)):
        return TuningSpec.model_validate_json(text)
    return ImagePipeline.from_json(text)


def setup_path_payload(
    sandbox: SandboxRoot,
    path: str | Path,
    *,
    kind: SetupPathKind,
) -> SetupPathPayload | None:
    """Build a fresh picker payload after validating one selected file."""
    try:
        resolved = sandbox.resolve(path)
    except ValueError:
        return None
    suffixes = _PIPELINE_SUFFIXES if kind == "pipeline" else _METADATA_SUFFIXES
    if not resolved.is_file() or not matches_any_suffix(resolved, suffixes):
        return None
    return {
        "version": 2,
        "kind": kind,
        "relative_path": str(resolved.relative_to(sandbox.root)) or ".",
        "absolute_path_at_selection": str(resolved),
        "sandbox_fingerprint": sandbox_fingerprint(sandbox),
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }


def resolve_picker_payload(
    sandbox: SandboxRoot,
    payload: object,
    *,
    kind: SetupPathKind,
) -> Path | None:
    """Resolve a current-session picker payload against the current sandbox."""
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("version") != 2
        or payload.get("kind") != kind
        or payload.get("sandbox_fingerprint") != sandbox_fingerprint(sandbox)
    ):
        return None
    relative = payload.get("relative_path")
    if not isinstance(relative, str) or not relative:
        return None
    candidate = setup_path_payload(sandbox, relative, kind=kind)
    return (
        Path(candidate["absolute_path_at_selection"])
        if candidate is not None
        else None
    )


def _candidate_path(
    sandbox: SandboxRoot,
    candidate: str,
    *,
    kind: SetupPathKind,
    source: SetupPathSource,
) -> SetupPathResolution:
    """Validate one precedence-selected path without falling through."""
    try:
        path = sandbox.resolve(candidate.strip())
    except ValueError:
        return SetupPathResolution(
            None,
            source,
            (f"{kind.capitalize()} path escapes the GUI sandbox.",),
        )
    suffixes = _PIPELINE_SUFFIXES if kind == "pipeline" else _METADATA_SUFFIXES
    issues = []
    if not path.is_file():
        issues.append(f"{kind.capitalize()} path is not an existing file: {path}")
    if not matches_any_suffix(path, suffixes):
        suffix_text = ", ".join(sorted(suffixes))
        issues.append(
            f"{kind.capitalize()} path must use one of: {suffix_text}"
        )
    return SetupPathResolution(path, source, tuple(issues))


def resolve_setup_path(
    *,
    sandbox: SandboxRoot,
    kind: SetupPathKind,
    typed_path: str | None,
    picker_payload: object,
    shared_payload: object,
) -> SetupPathResolution:
    """Resolve Setup input using typed, picker, shared, then unset precedence."""
    if typed_path and typed_path.strip():
        return _candidate_path(
            sandbox, typed_path, kind=kind, source="typed"
        )

    picked = resolve_picker_payload(sandbox, picker_payload, kind=kind)
    if picked is not None:
        return SetupPathResolution(picked, "picker")

    if kind == "metadata":
        shared = resolve_metadata_csv(sandbox, shared_payload)
        if shared is not None:
            return SetupPathResolution(shared, "shared")
    else:
        candidates: list[str] = []
        if isinstance(shared_payload, str):
            candidates.append(shared_payload)
        elif isinstance(shared_payload, dict):
            for key in ("relative_path", "path", "abs_path"):
                value = shared_payload.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
                    break
        for candidate in candidates:
            resolution = _candidate_path(
                sandbox, candidate, kind=kind, source="shared"
            )
            if not resolution.issues:
                return resolution

    return SetupPathResolution(None, "unset")


def authored_setup_spec_path(*, sandbox_root: Path, source_path: Path) -> Path:
    """Return the GUI preset path for a spec authored from ``source_path``."""
    return (
        tune_presets_dir(sandbox_root)
        / f"{_safe_stem(source_path)}.setup.json.pht-tune"
    )


def _validation_messages(exc: ValidationError) -> list[str]:
    """Format every pydantic issue with its precise model location."""
    messages = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        prefix = f"{location}: " if location else ""
        messages.append(prefix + str(error.get("msg", "invalid value")))
    return messages


def build_authored_setup_spec(
    *,
    pipeline_or_spec_path: Path,
    metadata_path: Path | None,
    edits: Mapping[str, Mapping[str, object]] | None = None,
    replace_scorer: bool = False,
    metadata_groupby: list[str] | None = None,
) -> SetupAuthoringResult:
    """Construct and fully validate a Setup-authored spec in memory."""
    issues: list[str] = []
    if not pipeline_or_spec_path.is_file():
        return SetupAuthoringResult(
            None,
            False,
            (f"Pipeline/spec file does not exist: {pipeline_or_spec_path}",),
        )
    try:
        source = load_pipeline_or_spec(pipeline_or_spec_path)
    except (OSError, ValueError, ValidationError) as exc:
        return SetupAuthoringResult(
            None,
            False,
            (f"Could not load pipeline/spec: {exc}",),
        )

    source_is_spec = isinstance(source, TuningSpec)
    needs_metadata = not source_is_spec or replace_scorer
    if needs_metadata:
        if metadata_path is None:
            issues.append(
                "Metadata is required for the metadata-backed QC scorer."
            )
        elif not metadata_path.is_file():
            issues.append(f"Metadata file does not exist: {metadata_path}")
        elif not matches_any_suffix(metadata_path, _METADATA_SUFFIXES):
            issues.append("Metadata must be a CSV or Parquet file.")
    elif metadata_path is not None and not metadata_path.is_file():
        issues.append(f"Selected metadata file does not exist: {metadata_path}")

    try:
        if isinstance(source, TuningSpec):
            spec = source
            if edits:
                space = apply_space_edits(source.search_space, edits)
                spec = spec.model_copy(update={"search_space": space})
        else:
            spec = space_to_spec(source, edits=edits or {})

        if needs_metadata and metadata_path is not None and metadata_path.is_file():
            groupby = metadata_groupby or [str(METADATA.IMAGE_NAME)]
            scorer = QCScorer(
                check=ExpectedVsDetectedCount(
                    metadata=str(metadata_path),
                    groupby=groupby,
                )
            )
            spec = spec.model_copy(update={"scorer": scorer})

        if not spec.search_space.knobs:
            issues.append("No active knobs to tune.")
        validated = TuningSpec.model_validate_json(spec.model_dump_json())
    except ValidationError as exc:
        issues.extend(_validation_messages(exc))
        validated = None
    except (TypeError, ValueError) as exc:
        issues.append(str(exc))
        validated = None

    if issues:
        return SetupAuthoringResult(None, source_is_spec, tuple(issues))
    return SetupAuthoringResult(validated, source_is_spec, ())


def write_authored_setup_spec(
    *,
    sandbox_root: Path,
    pipeline_or_spec_path: Path,
    metadata_path: Path | None = None,
    edits: Mapping[str, Mapping[str, object]] | None = None,
    replace_scorer: bool = False,
    metadata_groupby: list[str] | None = None,
) -> Path:
    """Atomically validate and write one GUI-authored tuning spec."""
    if not pipeline_or_spec_path.is_file():
        raise FileNotFoundError(pipeline_or_spec_path)
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    result = build_authored_setup_spec(
        pipeline_or_spec_path=pipeline_or_spec_path,
        metadata_path=metadata_path,
        edits=edits,
        replace_scorer=replace_scorer,
        metadata_groupby=metadata_groupby,
    )
    if not result.is_valid or result.spec is None:
        raise ValueError("\n".join(result.issues))
    target = authored_setup_spec_path(
        sandbox_root=sandbox_root,
        source_path=pipeline_or_spec_path,
    )
    atomic_write_text(target, result.spec.model_dump_json(indent=2))
    return target


__all__ = [
    "SetupAuthoringResult",
    "SetupPathPayload",
    "SetupPathResolution",
    "authored_setup_spec_path",
    "build_authored_setup_spec",
    "load_pipeline_or_spec",
    "resolve_picker_payload",
    "resolve_setup_path",
    "setup_path_payload",
    "write_authored_setup_spec",
]
