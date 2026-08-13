"""Pure path resolution and atomic Setup authoring for Tune."""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Literal, Mapping, TypedDict

from pydantic import ValidationError
from typing_extensions import NotRequired

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.gui._config import tune_presets_dir
from phenotypic.gui.shell._metadata_context import resolve_metadata_csv
from phenotypic.gui.shell._sandbox import (
    SandboxRoot,
    _is_safe_relative_path,
    _v1_selection_matches_sandbox,
)
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

SETUP_DRAFT_VERSION = 2
_SETUP_DRAFT_CACHE_SIZE = 256
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
    selection_id: NotRequired[str]


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


def setup_path_resolution_from_store(
    value: object,
    *,
    sandbox: SandboxRoot | None = None,
    kind: SetupPathKind | None = None,
) -> SetupPathResolution:
    """Parse a path-resolution store, optionally rechecking its sandbox path."""
    if not isinstance(value, dict):
        return SetupPathResolution(None, "unset")
    path = value.get("path")
    source = value.get("source")
    issues = value.get("issues")
    if (
        (path is not None and not isinstance(path, str))
        or source not in {"typed", "picker", "shared", "unset"}
        or not isinstance(issues, list)
    ):
        return SetupPathResolution(
            None,
            "unset",
            ("Setup path state is invalid; select the file again.",),
        )
    resolution = SetupPathResolution(
        Path(path) if path else None,
        source,
        tuple(str(issue) for issue in issues),
    )
    if sandbox is None or kind is None or resolution.path is None:
        return resolution
    checked = _candidate_path(
        sandbox,
        str(resolution.path),
        kind=kind,
        source=resolution.source,
    )
    return SetupPathResolution(
        checked.path,
        checked.source,
        tuple(dict.fromkeys((*resolution.issues, *checked.issues))),
    )


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


@dataclass(frozen=True)
class SetupDraft:
    """One revisioned, validated interpretation of all Setup controls.

    ``revision`` binds the resolved paths, current source bytes, search-space
    edits, scorer choice, validation issues, and validated spec JSON. The full
    object remains server-side because an existing spec may contain credentials.
    """

    revision: str
    source_revision: str
    pipeline_path: str | None
    pipeline_source: SetupPathSource
    metadata_path: str | None
    metadata_source: SetupPathSource
    replace_scorer: bool
    source_is_spec: bool
    edits: dict[str, dict[str, object]]
    source_fingerprint: str
    metadata_fingerprint: str
    scorer_name: str | None
    issues: tuple[str, ...]
    spec_json: str | None

    @property
    def is_valid(self) -> bool:
        """Whether this draft contains a fully validated tuning spec."""
        return self.spec_json is not None and not self.issues

    def to_store(self) -> dict[str, object]:
        """Return a redacted summary that never includes authored spec content.

        This summary is diagnostic only. Browser callbacks use
        :class:`SetupDraftCache.publish`, which adds an unguessable server-cache
        handle while retaining only this revision in client transport.
        """
        return {
            "version": SETUP_DRAFT_VERSION,
            "revision": self.revision,
        }


class SetupDraftCache:
    """Bounded per-app server cache for credential-bearing Setup drafts."""

    def __init__(self, *, max_entries: int = _SETUP_DRAFT_CACHE_SIZE) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._drafts: OrderedDict[str, SetupDraft] = OrderedDict()
        self._handles_by_revision: dict[str, str] = {}
        self._lock = RLock()

    def publish(self, draft: SetupDraft) -> dict[str, object]:
        """Cache ``draft`` and return its credential-free browser receipt."""
        with self._lock:
            handle = self._handles_by_revision.get(draft.revision)
            if handle is None or handle not in self._drafts:
                handle = secrets.token_urlsafe(32)
                self._handles_by_revision[draft.revision] = handle
            self._drafts[handle] = draft
            self._drafts.move_to_end(handle)
            while len(self._drafts) > self._max_entries:
                evicted_handle, evicted = self._drafts.popitem(last=False)
                if self._handles_by_revision.get(evicted.revision) == evicted_handle:
                    self._handles_by_revision.pop(evicted.revision, None)
        return {
            "version": SETUP_DRAFT_VERSION,
            "handle": handle,
            "revision": draft.revision,
        }

    def resolve(self, value: object) -> SetupDraft | None:
        """Resolve an unmodified browser receipt to its server-side draft."""
        if not isinstance(value, dict) or value.get("version") != SETUP_DRAFT_VERSION:
            return None
        if set(value) != {"version", "handle", "revision"}:
            return None
        handle = value.get("handle")
        revision = value.get("revision")
        if (
            not isinstance(handle, str)
            or not handle
            or not isinstance(revision, str)
            or not revision
        ):
            return None
        with self._lock:
            draft = self._drafts.get(handle)
            if draft is None or draft.revision != revision:
                return None
            self._drafts.move_to_end(handle)
            return draft


@dataclass(frozen=True)
class SetupWriteReceipt:
    """Immutable provenance for the exact authored bytes written by Continue."""

    path: Path
    draft_revision: str
    source_fingerprint: str
    metadata_fingerprint: str
    authored_fingerprint: str


def _safe_stem(path: Path) -> str:
    """Return a filesystem-safe stem for a GUI-authored spec."""
    stem = _SAFE_STEM.sub("-", path.stem).strip(".-")
    return stem or "tuning-spec"


def path_content_fingerprint(path: Path | None) -> str:
    """Return a canonical path-and-content identity without exposing content.

    Args:
        path: File to identify, or ``None`` for an unset optional input.

    Returns:
        A SHA-256 digest over the canonical path and current file bytes. Missing
        and unreadable paths receive stable sentinel identities.
    """
    if path is None:
        return hashlib.sha256(b"unset").hexdigest()
    try:
        canonical = path.expanduser().resolve(strict=False)
    except OSError:
        canonical = path.expanduser().absolute()
    try:
        content = canonical.read_bytes()
        state = b"file"
    except OSError:
        content = b""
        state = b"unavailable"
    digest = hashlib.sha256()
    digest.update(state)
    digest.update(b"\0")
    digest.update(str(canonical).encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    digest.update(content)
    return digest.hexdigest()


def authored_content_fingerprint(path: Path) -> str:
    """Return the SHA-256 digest of one authored spec's current bytes."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _canonical_edits(
    edits: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    """Return a detached, JSON-safe edit mapping in deterministic key order."""
    encoded = json.dumps(
        edits or {},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in sorted(decoded.items())
        if isinstance(value, dict)
    }


def _content_revision(payload: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 revision for JSON-safe content."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _setup_source_revision(
    *,
    pipeline_path: str | None,
    pipeline_source: SetupPathSource,
    source_fingerprint: str,
) -> str:
    """Return the revision controlling source-dependent editor rendering."""
    return _content_revision(
        {
            "pipeline_path": pipeline_path,
            "pipeline_source": pipeline_source,
            "source_fingerprint": source_fingerprint,
        }
    )


def setup_draft_from_store(
    value: object,
    *,
    cache: SetupDraftCache,
) -> SetupDraft | None:
    """Resolve a redacted browser receipt through its per-app server cache."""
    return cache.resolve(value)


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
        "relative_path": resolved.relative_to(sandbox.root).as_posix() or ".",
        "absolute_path_at_selection": str(resolved),
        "sandbox_fingerprint": sandbox_fingerprint(sandbox),
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "selection_id": secrets.token_hex(16),
    }


def resolve_picker_payload(
    sandbox: SandboxRoot,
    payload: object,
    *,
    kind: SetupPathKind,
) -> Path | None:
    """Resolve a selected V1/V2 descriptor against the current sandbox.

    V2 is fingerprint-bound. V1 remains readable only when its absolute and
    sandbox-relative mirrors still identify the same current-sandbox file.
    """
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    if version == 2:
        if (
            payload.get("kind") != kind
            or payload.get("sandbox_fingerprint")
            != sandbox_fingerprint(sandbox)
        ):
            return None
        relative = payload.get("relative_path")
        if not isinstance(relative, str) or not _is_safe_relative_path(relative):
            return None
    elif version == 1:
        raw_path = payload.get("abs_path", payload.get("path"))
        relative = payload.get("rel_path", payload.get("relative_path"))
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or not isinstance(relative, str)
            or not _is_safe_relative_path(relative)
            or not _v1_selection_matches_sandbox(
                sandbox,
                raw_path=raw_path,
                relative_path=relative,
            )
        ):
            return None
    else:
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
        shared = resolve_picker_payload(sandbox, shared_payload, kind="pipeline")
        if shared is not None:
            return SetupPathResolution(shared, "shared")
        legacy_candidates: list[str] = []
        if isinstance(shared_payload, str):
            legacy_candidates.append(shared_payload)
        elif isinstance(shared_payload, dict) and shared_payload.get("version") is None:
            for key in ("relative_path", "rel_path", "path", "abs_path"):
                candidate = shared_payload.get(key)
                if isinstance(candidate, str) and candidate:
                    legacy_candidates.append(candidate)
                    break
        for candidate in legacy_candidates:
            resolution = _candidate_path(
                sandbox,
                candidate,
                kind="pipeline",
                source="shared",
            )
            if not resolution.issues:
                return resolution

    return SetupPathResolution(None, "unset")


def authored_setup_spec_path(
    *,
    sandbox_root: Path,
    source_path: Path,
    metadata_path: Path | None = None,
    authored_content: str = "",
) -> Path:
    """Return a collision-safe GUI preset path for one authored spec.

    The readable source stem is only a label. The digest also binds the
    canonical source and metadata identities plus the exact authored content,
    so equal stems in different directories and changed inputs cannot alias.
    """
    identity = {
        "source": path_content_fingerprint(source_path),
        "metadata": path_content_fingerprint(metadata_path),
        "authored": hashlib.sha256(authored_content.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    suffix = hashlib.sha256(encoded).hexdigest()[:20]
    return (
        tune_presets_dir(sandbox_root)
        / f"{_safe_stem(source_path)}-{suffix}.setup.json.pht-tune"
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
    except (OSError, ValueError, ValidationError):
        return SetupAuthoringResult(
            None,
            False,
            ("Could not load pipeline/spec; review the server log.",),
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
    except (TypeError, ValueError):
        issues.append("Could not apply Setup edits; review the server log.")
        validated = None

    if issues:
        return SetupAuthoringResult(None, source_is_spec, tuple(issues))
    return SetupAuthoringResult(validated, source_is_spec, ())


def build_setup_draft(
    *,
    pipeline: SetupPathResolution,
    metadata: SetupPathResolution,
    edits: Mapping[str, Mapping[str, object]] | None = None,
    replace_scorer: bool = False,
) -> SetupDraft:
    """Build the sole revisioned Setup state from resolved controls.

    Args:
        pipeline: Precedence-resolved pipeline or tuning-spec path.
        metadata: Precedence-resolved optional metadata path.
        edits: Raw search-space editor values keyed by knob.
        replace_scorer: Whether an existing scorer should be explicitly replaced.

    Returns:
        A self-consistent draft carrying the validated spec JSON or all issues.
    """
    canonical_edits = _canonical_edits(edits)
    pipeline_path = str(pipeline.path) if pipeline.path is not None else None
    metadata_path = str(metadata.path) if metadata.path is not None else None
    source_fingerprint = path_content_fingerprint(pipeline.path)
    metadata_fingerprint = path_content_fingerprint(metadata.path)
    issues = [*pipeline.issues, *metadata.issues]
    source_is_spec = False
    scorer_name: str | None = None
    spec_json: str | None = None
    if pipeline.path is None:
        if not issues:
            issues.append("Choose a pipeline or existing tuning spec.")
    elif not pipeline.issues:
        result = build_authored_setup_spec(
            pipeline_or_spec_path=pipeline.path,
            metadata_path=metadata.path,
            edits=canonical_edits,
            replace_scorer=replace_scorer,
        )
        source_is_spec = result.source_is_spec
        issues.extend(result.issues)
        if result.is_valid and result.spec is not None and not issues:
            scorer_name = type(result.spec.scorer).__name__
            spec_json = result.spec.model_dump_json(indent=2)

    source_revision = _setup_source_revision(
        pipeline_path=pipeline_path,
        pipeline_source=pipeline.source,
        source_fingerprint=source_fingerprint,
    )
    normalized_issues = tuple(dict.fromkeys(issues))
    revision_payload = {
        "source_revision": source_revision,
        "pipeline_path": pipeline_path,
        "pipeline_source": pipeline.source,
        "metadata_path": metadata_path,
        "metadata_source": metadata.source,
        "replace_scorer": replace_scorer,
        "source_is_spec": source_is_spec,
        "edits": canonical_edits,
        "source_fingerprint": source_fingerprint,
        "metadata_fingerprint": metadata_fingerprint,
        "scorer_name": scorer_name,
        "issues": normalized_issues,
        "spec_json": spec_json,
    }
    return SetupDraft(
        revision=_content_revision(revision_payload),
        source_revision=source_revision,
        pipeline_path=pipeline_path,
        pipeline_source=pipeline.source,
        metadata_path=metadata_path,
        metadata_source=metadata.source,
        replace_scorer=replace_scorer,
        source_is_spec=source_is_spec,
        edits=canonical_edits,
        source_fingerprint=source_fingerprint,
        metadata_fingerprint=metadata_fingerprint,
        scorer_name=scorer_name,
        issues=normalized_issues,
        spec_json=spec_json,
    )


def write_setup_draft_receipt(
    *,
    sandbox_root: Path,
    draft: SetupDraft,
) -> SetupWriteReceipt:
    """Atomically write ``draft`` and return its immutable provenance.

    The source and optional metadata fingerprints are rechecked immediately
    before writing so Continue cannot publish a draft whose files changed after
    validation.
    """
    if not draft.is_valid or draft.spec_json is None or draft.pipeline_path is None:
        raise ValueError("\n".join(draft.issues) or "Setup draft is invalid.")
    sandbox = SandboxRoot.from_path(sandbox_root)
    source_path = sandbox.resolve(draft.pipeline_path)
    metadata_path = (
        sandbox.resolve(draft.metadata_path) if draft.metadata_path else None
    )
    if path_content_fingerprint(source_path) != draft.source_fingerprint:
        raise ValueError("Pipeline/spec changed after Setup validation.")
    if path_content_fingerprint(metadata_path) != draft.metadata_fingerprint:
        raise ValueError("Metadata changed after Setup validation.")
    validated = TuningSpec.model_validate_json(draft.spec_json)
    authored_content = validated.model_dump_json(indent=2)
    target = authored_setup_spec_path(
        sandbox_root=sandbox_root,
        source_path=source_path,
        metadata_path=metadata_path,
        authored_content=authored_content,
    )
    atomic_write_text(target, authored_content)
    return SetupWriteReceipt(
        path=target,
        draft_revision=draft.revision,
        source_fingerprint=draft.source_fingerprint,
        metadata_fingerprint=draft.metadata_fingerprint,
        authored_fingerprint=hashlib.sha256(
            authored_content.encode("utf-8")
        ).hexdigest(),
    )


def write_setup_draft(*, sandbox_root: Path, draft: SetupDraft) -> Path:
    """Compatibility wrapper returning only the authored spec path."""
    return write_setup_draft_receipt(
        sandbox_root=sandbox_root,
        draft=draft,
    ).path


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
    authored_content = result.spec.model_dump_json(indent=2)
    target = authored_setup_spec_path(
        sandbox_root=sandbox_root,
        source_path=pipeline_or_spec_path,
        metadata_path=metadata_path,
        authored_content=authored_content,
    )
    atomic_write_text(target, authored_content)
    return target


__all__ = [
    "SETUP_DRAFT_VERSION",
    "SetupAuthoringResult",
    "SetupDraft",
    "SetupDraftCache",
    "SetupPathPayload",
    "SetupPathResolution",
    "SetupWriteReceipt",
    "authored_content_fingerprint",
    "authored_setup_spec_path",
    "build_authored_setup_spec",
    "build_setup_draft",
    "load_pipeline_or_spec",
    "path_content_fingerprint",
    "resolve_picker_payload",
    "resolve_setup_path",
    "setup_draft_from_store",
    "setup_path_resolution_from_store",
    "setup_path_payload",
    "write_authored_setup_spec",
    "write_setup_draft",
    "write_setup_draft_receipt",
]
