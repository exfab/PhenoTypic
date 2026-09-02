"""Context-local operation provenance for :class:`phenotypic.Image`."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path, PureWindowsPath
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, TypeVar, cast

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic.abc_._image_operation import ImageOperation
    from phenotypic.sdk_ import CommitGuard


PROVENANCE_SCHEMA_VERSION = 2

_APPLICATION_KINDS = frozenset({"process", "full", "programmatic", "legacy"})
_APPLICATION_STATUSES = frozenset(
    {"complete", "failed", "in_progress", "staged"}
)
_JOURNAL_KEYS = frozenset(
    {"schema_version", "status", "original_filename", "applications"}
)
_APPLICATION_KEYS = frozenset(
    {
        "sequence",
        "kind",
        "phenotypic_version",
        "input_filename",
        "status",
        "pipeline",
        "retry_base_length",
        "operations",
    }
)
_OPERATION_REQUIRED_KEYS = frozenset(
    {
        "sequence",
        "operation_name",
        "operation_class",
        "phenotypic_version",
        "parameters",
        "pipeline_step_path",
    }
)
_OPERATION_TIMING_KEYS = frozenset({"applied_at_utc", "duration_seconds"})

_Apply = TypeVar("_Apply", bound=Callable[..., "Image"])


@dataclass
class _OperationApplyFrame:
    """One distinct public operation invocation and its completed children."""

    operation: object
    owner: type
    nested_records: list[dict[str, Any]] = field(default_factory=list)
    nested_sink_published: bool = False


_operation_apply_stack: ContextVar[tuple[_OperationApplyFrame, ...]] = ContextVar(
    "phenotypic_operation_apply_stack", default=()
)
_pipeline_step_path: ContextVar[tuple[str, ...] | None] = ContextVar(
    "phenotypic_pipeline_step_path", default=None
)
_success_sink: ContextVar[Callable[["Image"], object] | None] = ContextVar(
    "phenotypic_provenance_success_sink", default=None
)
_application_owner_depth: ContextVar[int] = ContextVar(
    "phenotypic_provenance_application_owner_depth", default=0
)


class _ReadOnlyList(list[Any]):
    """List-shaped immutable view so JSON arrays still compare as lists."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("provenance is read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def provenance_basename(value: str | Path | None) -> str | None:
    """Return a portable final component for POSIX or Windows path syntax."""
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return PureWindowsPath(text).name or None


def _basename(value: str | Path | None) -> str | None:
    """Return the exact final path component without normalizing its spelling."""
    return provenance_basename(value)


def new_provenance_journal(
    original_filename: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fresh canonical version-2 journal owned by one image."""
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "complete",
        "original_filename": _basename(original_filename),
        "applications": [],
    }


def _validate_filename(value: Any, *, field_name: str, nullable: bool) -> None:
    if value is None and nullable:
        return
    if (
        not isinstance(value, str)
        or not value
        or provenance_basename(value) != value
    ):
        raise ValueError(
            f"malformed provenance schema v2: {field_name} must be a basename"
        )


def _validate_pipeline(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"source_path", "sha256"}:
        raise ValueError("malformed provenance schema v2 pipeline identity")
    source_path = value["source_path"]
    if (
        not isinstance(source_path, str)
        or not source_path
        or provenance_basename(source_path) != source_path
    ):
        raise ValueError(
            "malformed provenance schema v2 pipeline source_path must be a basename"
        )
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("malformed provenance schema v2 pipeline sha256")


def validate_provenance_journal(journal: Mapping[str, Any]) -> None:
    """Validate a canonical v2 journal without changing it.

    Version-1 journals remain readable through :func:`readonly_operations`,
    but ordinary mutation must stop and direct the caller to explicit migrate
    mode rather than silently replacing their history.
    """
    version = journal.get("schema_version")
    if version == 1:
        raise ValueError(
            "provenance schema v1 requires explicit migration before mutation"
        )
    if version != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported provenance schema version {version!r}; "
            f"expected {PROVENANCE_SCHEMA_VERSION}"
        )
    if set(journal) != _JOURNAL_KEYS:
        raise ValueError("malformed provenance schema v2 journal fields")
    status = journal["status"]
    if status not in _APPLICATION_STATUSES:
        raise ValueError("malformed provenance schema v2 journal status")
    _validate_filename(
        journal["original_filename"],
        field_name="original_filename",
        nullable=True,
    )
    applications = journal["applications"]
    if not isinstance(applications, list):
        raise ValueError("malformed provenance schema v2 applications")

    expected_operation_sequence = 1
    for expected_application_sequence, application in enumerate(
        applications, start=1
    ):
        if not isinstance(application, dict) or set(application) != _APPLICATION_KEYS:
            raise ValueError("malformed provenance schema v2 application fields")
        if application["sequence"] != expected_application_sequence:
            raise ValueError("malformed provenance schema v2 application sequence")
        kind = application["kind"]
        if kind not in _APPLICATION_KINDS:
            raise ValueError("malformed provenance schema v2 application kind")
        application_status = application["status"]
        if application_status not in _APPLICATION_STATUSES:
            raise ValueError("malformed provenance schema v2 application status")
        version_value = application["phenotypic_version"]
        if kind == "legacy":
            if version_value is not None and (
                not isinstance(version_value, str) or not version_value
            ):
                raise ValueError("malformed legacy phenotypic_version")
        elif not isinstance(version_value, str) or not version_value:
            raise ValueError(
                "malformed provenance schema v2: new applications require "
                "a non-empty phenotypic_version"
            )
        _validate_filename(
            application["input_filename"],
            field_name="input_filename",
            nullable=kind in {"legacy", "programmatic"},
        )
        _validate_pipeline(application["pipeline"])
        operations = application["operations"]
        if not isinstance(operations, list):
            raise ValueError("malformed provenance schema v2 operations")
        retry_base = application["retry_base_length"]
        if (
            not isinstance(retry_base, int)
            or isinstance(retry_base, bool)
            or retry_base < 0
            or retry_base > len(operations)
        ):
            raise ValueError("malformed provenance schema v2 retry_base_length")
        for operation in operations:
            operation_keys = set(operation) if isinstance(operation, dict) else set()
            if operation_keys not in {
                _OPERATION_REQUIRED_KEYS,
                _OPERATION_REQUIRED_KEYS | _OPERATION_TIMING_KEYS,
            }:
                raise ValueError("malformed provenance schema v2 operation fields")
            if operation["sequence"] != expected_operation_sequence:
                raise ValueError("malformed provenance schema v2 operation sequence")
            for field_name in (
                "operation_name",
                "operation_class",
                "phenotypic_version",
            ):
                if not isinstance(operation[field_name], str) or not operation[field_name]:
                    raise ValueError(
                        f"malformed provenance schema v2 operation {field_name}"
                    )
            if not isinstance(operation["parameters"], dict):
                raise ValueError("malformed provenance schema v2 operation parameters")
            if _OPERATION_TIMING_KEYS <= operation_keys:
                applied_at = operation["applied_at_utc"]
                if not isinstance(applied_at, str) or not applied_at.endswith("Z"):
                    raise ValueError(
                        "malformed provenance schema v2 operation applied_at_utc"
                    )
                duration = operation["duration_seconds"]
                if (
                    not isinstance(duration, (int, float))
                    or isinstance(duration, bool)
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    raise ValueError("malformed provenance schema v2 operation duration")
            step_path = operation["pipeline_step_path"]
            if step_path is not None and (
                not isinstance(step_path, list)
                or not step_path
                or any(not isinstance(step, str) or not step for step in step_path)
            ):
                raise ValueError("malformed provenance schema v2 pipeline step path")
            expected_operation_sequence += 1

        if (
            expected_application_sequence < len(applications)
            and application_status not in {"complete", "failed"}
        ):
            raise ValueError(
                "malformed provenance schema v2: prior applications must be terminal"
            )

    expected_status = applications[-1]["status"] if applications else "complete"
    if status != expected_status:
        raise ValueError(
            "malformed provenance schema v2: root status must mirror the last "
            "application"
        )


def _operations(journal: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return flattened operations for v1 reads or validated v2 internals."""
    if journal.get("schema_version") == 1:
        operations = journal.get("operations", [])
        return operations if isinstance(operations, list) else []
    validate_provenance_journal(journal)
    return [
        operation
        for application in journal["applications"]
        for operation in application["operations"]
    ]


def _current_application(journal: Mapping[str, Any]) -> dict[str, Any]:
    validate_provenance_journal(journal)
    applications = journal["applications"]
    if not applications:
        raise ValueError("provenance journal has no application to mutate")
    return applications[-1]


def readonly_operations(journal: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return a deeply detached, immutable view of journal operations."""

    def _freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return MappingProxyType(
                {key: _freeze(item) for key, item in value.items()}
            )
        if isinstance(value, list):
            return _ReadOnlyList(_freeze(item) for item in value)
        return value

    return tuple(_freeze(deepcopy(entry)) for entry in _operations(journal))


def _application_input_filename(
    image: "Image", value: str | Path | None
) -> str | None:
    """Resolve one immediate input basename without inventing a path."""
    if value is not None:
        return _basename(value)
    original = image._metadata.provenance_journal.get("original_filename")
    if isinstance(original, str) and original:
        return original
    return _basename(getattr(image, "name", None))


def _append_application(
    journal: dict[str, Any],
    *,
    kind: str,
    phenotypic_version: str | None,
    input_filename: str | None,
    pipeline: Mapping[str, str] | None,
    status: str,
    retry_base_length: int = 0,
) -> dict[str, Any]:
    """Append one canonical application to a validated journal."""
    validate_provenance_journal(journal)
    applications = journal["applications"]
    if applications and applications[-1]["status"] not in {"complete", "failed"}:
        raise ValueError("cannot start a new provenance application before the last ends")
    normalized_pipeline = deepcopy(dict(pipeline)) if pipeline is not None else None
    if normalized_pipeline is not None:
        normalized_pipeline["source_path"] = provenance_basename(
            normalized_pipeline["source_path"]
        )
    application = {
        "sequence": len(applications) + 1,
        "kind": kind,
        "phenotypic_version": phenotypic_version,
        "input_filename": _basename(input_filename),
        "status": status,
        "pipeline": normalized_pipeline,
        "retry_base_length": int(retry_base_length),
        "operations": [],
    }
    applications.append(application)
    journal["status"] = status
    try:
        validate_provenance_journal(journal)
    except BaseException:
        applications.pop()
        journal["status"] = applications[-1]["status"] if applications else "complete"
        raise
    return application


def _set_journal_status(journal: dict[str, Any], status: str) -> None:
    """Set the active application and root status as one validated change."""
    application = _current_application(journal)
    previous_application_status = application["status"]
    previous_root_status = journal["status"]
    application["status"] = status
    journal["status"] = status
    try:
        validate_provenance_journal(journal)
    except BaseException:
        application["status"] = previous_application_status
        journal["status"] = previous_root_status
        raise


@contextmanager
def provenance_application(
    image: "Image",
    *,
    kind: str = "programmatic",
    pipeline: Mapping[str, str] | None = None,
    input_filename: str | Path | None = None,
) -> Iterator[None]:
    """Own one outer programmatic application or join an active CLI owner."""
    journal = image._metadata.provenance_journal
    validate_provenance_journal(journal)
    depth = _application_owner_depth.get()
    owns_application = depth == 0
    if owns_application:
        if journal["original_filename"] is None:
            journal["original_filename"] = _application_input_filename(
                image, input_filename
            )
        _append_application(
            journal,
            kind=kind,
            phenotypic_version=_installed_phenotypic_version(),
            input_filename=_application_input_filename(image, input_filename),
            pipeline=pipeline,
            status="in_progress",
        )
    token = _application_owner_depth.set(depth + 1)
    try:
        yield
    except BaseException:
        if owns_application:
            _set_journal_status(
                image._metadata.provenance_journal, "failed"
            )
        raise
    else:
        if owns_application:
            _set_journal_status(
                image._metadata.provenance_journal, "complete"
            )
    finally:
        _application_owner_depth.reset(token)


@contextmanager
def continuing_provenance_application(image: "Image") -> Iterator[None]:
    """Mark an already-open CLI application as owned by this execution scope."""
    journal = image._metadata.provenance_journal
    application = _current_application(journal)
    if application["status"] not in {"in_progress", "staged"}:
        raise ValueError("provenance application is not open for continuation")
    depth = _application_owner_depth.get()
    token = _application_owner_depth.set(depth + 1)
    try:
        yield
    finally:
        _application_owner_depth.reset(token)


def _journal_with_operations(
    source_journal: Mapping[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy a journal and append entries it does not already contain."""
    merged: dict[str, Any] = deepcopy(dict(source_journal))
    operations = _current_application(merged)["operations"]
    for entry in entries:
        if entry in _operations(merged):
            continue
        carried_entry = deepcopy(entry)
        carried_entry["sequence"] = len(_operations(merged)) + 1
        operations.append(carried_entry)
    return merged


def _carry_logical_image_state(
    result: "Image",
    source_journal: Mapping[str, Any],
    source_original: Any,
    nested_operations: list[dict[str, Any]],
) -> None:
    """Carry image-owned provenance/source state across replacement operations."""
    returned_operations = deepcopy(_operations(result._metadata.provenance_journal))
    source_operations = _operations(source_journal)
    common_prefix = 0
    for source_entry, returned_entry in zip(
        source_operations, returned_operations, strict=False
    ):
        if source_entry != returned_entry:
            break
        common_prefix += 1

    merged = _journal_with_operations(source_journal, nested_operations)
    operations = _current_application(merged)["operations"]
    unmatched_nested = deepcopy(nested_operations)
    for returned_entry in returned_operations[common_prefix:]:
        match_index = next(
            (
                index
                for index, nested_entry in enumerate(unmatched_nested)
                if nested_entry == returned_entry
            ),
            None,
        )
        if match_index is not None:
            del unmatched_nested[match_index]
            continue
        carried_entry = deepcopy(returned_entry)
        carried_entry["sequence"] = len(_operations(merged)) + 1
        operations.append(carried_entry)

    result._metadata.provenance_journal = merged
    # The retained pixels are an immutable decoded-source snapshot. Reusing
    # the reference avoids another full-image allocation at every operation.
    result._original = source_original


@contextmanager
def pipeline_step(key: str) -> Iterator[None]:
    """Append one configured operation key to the current nested step path."""
    parent = _pipeline_step_path.get()
    token = _pipeline_step_path.set((*parent, key) if parent else (key,))
    try:
        yield
    finally:
        _pipeline_step_path.reset(token)


@contextmanager
def provenance_success_sink(
    sink: Callable[["Image"], object],
) -> Iterator[None]:
    """Install a context-local sink called after each successful leaf operation."""
    token = _success_sink.set(sink)
    try:
        yield
    finally:
        _success_sink.reset(token)


def wrap_image_operation_apply(apply_method: _Apply, owner: type) -> _Apply:
    """Wrap one subclass's resolved public ``apply`` at its outer success edge."""
    if getattr(apply_method, "__phenotypic_provenance_owner__", None) is owner:
        return apply_method

    @wraps(apply_method)
    def _recording_apply(
        self: "ImageOperation", *args: Any, **kwargs: Any
    ) -> "Image":
        stack = _operation_apply_stack.get()
        if (
            stack
            and stack[-1].operation is self
            and stack[-1].owner is not owner
            and issubclass(stack[-1].owner, owner)
        ):
            return apply_method(self, *args, **kwargs)

        parent_frame = stack[-1] if stack else None
        frame = _OperationApplyFrame(operation=self, owner=owner)
        token = _operation_apply_stack.set((*stack, frame))
        logical_input = cast(
            "Image", kwargs.get("image", args[0] if args else None)
        )
        input_journal = deepcopy(logical_input._metadata.provenance_journal)
        source_original = logical_input._original
        source_journal = deepcopy(input_journal)
        owns_application = False
        try:
            validate_provenance_journal(source_journal)
            owns_application = (
                parent_frame is None and _application_owner_depth.get() == 0
            )
            if owns_application:
                if source_journal["original_filename"] is None:
                    source_journal["original_filename"] = (
                        _application_input_filename(logical_input, None)
                    )
                _append_application(
                    source_journal,
                    kind="programmatic",
                    phenotypic_version=_installed_phenotypic_version(),
                    input_filename=_application_input_filename(logical_input, None),
                    pipeline=None,
                    status="in_progress",
                )
                logical_input._metadata.provenance_journal = deepcopy(source_journal)
            source_journal = _journal_with_operations(
                source_journal,
                parent_frame.nested_records if parent_frame is not None else [],
            )
            source_length = len(_operations(source_journal))
        except BaseException:
            _operation_apply_stack.reset(token)
            raise
        started = perf_counter()
        try:
            result = apply_method(self, *args, **kwargs)
            duration = perf_counter() - started
            _carry_logical_image_state(
                result,
                source_journal,
                source_original,
                frame.nested_records,
            )
            operations = _current_application(
                result._metadata.provenance_journal
            )["operations"]
            prior_length = len(operations)
            step_path = _pipeline_step_path.get()
            append_operation_provenance(
                result,
                self,
                duration_seconds=duration,
                pipeline_step_path=list(step_path) if step_path else None,
            )
            try:
                sink = _success_sink.get()
                if sink is not None:
                    sink(result)
                    frame.nested_sink_published = True
            except BaseException:
                del operations[prior_length:]
                raise
            produced_records = deepcopy(
                _operations(result._metadata.provenance_journal)[source_length:]
            )
            if parent_frame is not None:
                parent_frame.nested_records.extend(produced_records)
                parent_frame.nested_sink_published = (
                    parent_frame.nested_sink_published
                    or frame.nested_sink_published
                )
            if owns_application:
                _set_journal_status(result._metadata.provenance_journal, "complete")
            if result is not logical_input:
                logical_input._metadata.provenance_journal = input_journal
                logical_input._original = source_original
            return result
        except BaseException:
            sink = _success_sink.get()
            try:
                if frame.nested_sink_published and sink is not None:
                    logical_input._metadata.provenance_journal = deepcopy(
                        source_journal
                    )
                    logical_input._original = source_original
                    sink(logical_input)
            finally:
                if owns_application:
                    logical_input._metadata.provenance_journal = deepcopy(source_journal)
                    _set_journal_status(
                        logical_input._metadata.provenance_journal, "failed"
                    )
                else:
                    logical_input._metadata.provenance_journal = input_journal
                logical_input._original = source_original
            raise
        finally:
            _operation_apply_stack.reset(token)

    setattr(_recording_apply, "__phenotypic_provenance_owner__", owner)
    return cast(_Apply, _recording_apply)

def _installed_phenotypic_version() -> str:
    """Resolve the installed package version without an import cycle at module load."""
    import phenotypic

    return phenotypic.__version__


def pipeline_source_identity(
    path: str | Path, *, basename_only: bool = False
) -> dict[str, str]:
    """Return the pipeline's recorded source and SHA-256 content identity.

    Args:
        path: The pipeline file.
        basename_only: Record only the file's name rather than its resolved
            absolute path. ``True`` for artifacts that leave the run directory
            -- a ``--mode process`` store is published to a NAS and then to
            object storage, and an absolute path there would carry cluster
            filesystem layout, the username, and project directory names.
            ``sha256`` is unchanged, so identity is not weakened.

    Returns:
        ``{"source_path": …, "sha256": …}``.
    """
    source = Path(path).resolve()
    return {
        "source_path": source.name if basename_only else str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def start_provenance_application(
    image: "Image",
    *,
    kind: str,
    input_filename: str | Path | None,
    pipeline_identity: Mapping[str, str] | None = None,
    status: str = "in_progress",
    retry_base_length: int = 0,
) -> None:
    """Append one typed application without discarding earlier history."""
    journal = image._metadata.provenance_journal
    validate_provenance_journal(journal)
    resolved_input = _application_input_filename(image, input_filename)
    if journal["original_filename"] is None:
        journal["original_filename"] = resolved_input
    _append_application(
        journal,
        kind=kind,
        phenotypic_version=_installed_phenotypic_version(),
        input_filename=resolved_input,
        pipeline=pipeline_identity,
        status=status,
        retry_base_length=retry_base_length,
    )


def initialize_cli_provenance(
    image: "Image",
    pipeline_path: str | Path,
    *,
    kind: str = "full",
    input_filename: str | Path | None = None,
    pipeline_identity: Mapping[str, str] | None = None,
    status: str = "in_progress",
    retry_base_length: int = 0,
    basename_only: bool = False,
) -> None:
    """Append one CLI-owned application to an image's existing journal."""
    identity = (
        pipeline_source_identity(pipeline_path, basename_only=basename_only)
        if pipeline_identity is None
        else deepcopy(dict(pipeline_identity))
    )
    start_provenance_application(
        image,
        kind=kind,
        input_filename=input_filename,
        pipeline_identity=identity,
        status=status,
        retry_base_length=retry_base_length,
    )


def set_provenance_status(image: "Image", status: str) -> None:
    """Set the active application's lifecycle status and mirrored root status."""
    _set_journal_status(image._metadata.provenance_journal, status)


def resume_provenance_application(
    image: "Image",
    checkpoint_journal: Mapping[str, Any],
    *,
    kind: str,
    input_filename: str | Path,
    pipeline_identity: Mapping[str, str] | None,
    expected_work_id: str | None,
    checkpoint_work_id: object,
) -> bool:
    """Resume one exact unfinished checkpoint without inventing a new application.

    The freshly decoded image is the authority for all history preceding the
    interrupted application.  A same-stem checkpoint is adopted only when that
    prefix, the original filename, the immediate input, the application kind,
    and the pipeline content identity all agree.  This prevents a direct worker
    invocation from attaching history belonging to a different work item.

    Returns:
        ``True`` when an unfinished application was restored, or ``False`` when
        the checkpoint is terminal or contains no application.

    Raises:
        ValueError: If an unfinished checkpoint does not match the input image
            and requested application exactly, or uses an unsupported schema.
    """
    current = image._metadata.provenance_journal
    validate_provenance_journal(current)
    validate_provenance_journal(checkpoint_journal)
    applications = checkpoint_journal["applications"]
    if not applications:
        return False
    checkpoint_application = applications[-1]
    if checkpoint_application["status"] not in {"failed", "in_progress"}:
        return False
    if (
        not isinstance(expected_work_id, str)
        or not expected_work_id
        or checkpoint_work_id != expected_work_id
    ):
        raise ValueError(
            "unfinished provenance checkpoint work identity does not match; "
            "refusing to overwrite it"
        )

    normalized_pipeline = (
        deepcopy(dict(pipeline_identity))
        if pipeline_identity is not None
        else None
    )
    if normalized_pipeline is not None:
        normalized_pipeline["source_path"] = Path(
            normalized_pipeline["source_path"]
        ).name
    expected_input = _basename(input_filename)
    matches = (
        checkpoint_journal["original_filename"]
        == current["original_filename"]
        and applications[:-1] == current["applications"]
        and checkpoint_application["kind"] == kind
        and checkpoint_application["input_filename"] == expected_input
        and checkpoint_application["pipeline"] == normalized_pipeline
    )
    if not matches:
        raise ValueError(
            "unfinished provenance checkpoint does not match this input and "
            "pipeline; refusing to overwrite it"
        )

    image._metadata.provenance_journal = deepcopy(dict(checkpoint_journal))
    _set_journal_status(image._metadata.provenance_journal, "in_progress")
    truncate_provenance_to_retry_base(image)
    return True


def current_application_operations(image: "Image") -> list[dict[str, Any]]:
    """Return the mutable operation list for the current internal application."""
    return _current_application(image._metadata.provenance_journal)["operations"]


def set_retry_base_length(image: "Image", length: int) -> None:
    """Persist the current application's Stage-1 retry prefix length."""
    journal = image._metadata.provenance_journal
    application = _current_application(journal)
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length < 0
        or length > len(application["operations"])
    ):
        raise ValueError("retry base length exceeds current application operations")
    application["retry_base_length"] = length
    validate_provenance_journal(journal)


def truncate_provenance_to_retry_base(image: "Image") -> None:
    """Discard only current-application entries after its durable prefix."""
    journal = image._metadata.provenance_journal
    application = _current_application(journal)
    base = application["retry_base_length"]
    del application["operations"][base:]
    validate_provenance_journal(journal)

def append_operation_provenance(
    image: "Image",
    operation: "ImageOperation",
    *,
    duration_seconds: float,
    pipeline_step_path: list[str] | None,
) -> None:
    """Append one successful leaf record, including staged detector merges."""
    journal = image._metadata.provenance_journal
    operations = _current_application(journal)["operations"]
    parameters = json.loads(
        json.dumps(operation.model_dump(mode="json"), ensure_ascii=False)
    )
    operations.append(
        {
            "sequence": len(_operations(journal)) + 1,
            "operation_name": type(operation).__name__,
            "operation_class": (
                f"{type(operation).__module__}.{type(operation).__qualname__}"
            ),
            "phenotypic_version": _installed_phenotypic_version(),
            "parameters": parameters,
            "applied_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "duration_seconds": float(duration_seconds),
            "pipeline_step_path": pipeline_step_path,
        }
    )


#: The operation-entry fields that are readings of the wall clock rather than
#: functions of the inputs, and therefore the entire source of
#: non-reproducibility in a store. Both are written by
#: :func:`append_operation_provenance`; keep this tuple beside it.
NON_REPRODUCIBLE_OPERATION_FIELDS: tuple[str, ...] = (
    "applied_at_utc",
    "duration_seconds",
)


def strip_non_reproducible_operation_fields(
    journal: dict[str, Any],
) -> dict[str, Any]:
    """Drop the wall-clock fields from every entry in ``operations[]``, in place.

    Measured across two runs of one image through one pipeline, these two
    fields are the only bytes that move: everything else in the journal --
    ``operation_name``, ``operation_class``, ``phenotypic_version``, the
    resolved ``parameters``, ``pipeline_step_path``, and the ``pipeline``
    digest -- is a pure function of the inputs. Removing them makes a
    published ``--mode process`` store byte-identical across identical runs,
    which is what content-addressed storage, server-side dedup, and the
    whole-tree ``file_sha256`` of spec 7.3 all require (spec 2.3.3).

    Only ``operations[]`` entries are touched. The surrounding
    ``schema_version``, ``status``, ``pipeline`` and ``retry_base_length`` are
    the store's provenance and stay; an artifact made reproducible by saying
    nothing is not the goal.

    Call this on a *copy*. The image's own journal keeps both fields, and so
    does the bundle store, which never leaves the run directory.

    Args:
        journal: A mutable journal copy.

    Returns:
        The same mapping, for call-site chaining.
    """
    if journal.get("schema_version") == 1:
        operation_groups = (journal.get("operations", ()),)
    else:
        validate_provenance_journal(journal)
        operation_groups = tuple(
            application["operations"] for application in journal["applications"]
        )
        for application in journal["applications"]:
            pipeline_identity = application.get("pipeline")
            if pipeline_identity is not None:
                pipeline_identity["source_path"] = Path(
                    pipeline_identity["source_path"]
                ).name
    for operations in operation_groups:
        for entry in operations:
            for name in NON_REPRODUCIBLE_OPERATION_FIELDS:
                entry.pop(name, None)
    return journal


def write_provenance_checkpoint(
    store: str | Path,
    image: "Image",
    *,
    journal_only: bool = False,
    work_id: str | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically replace only root provenance, or create a journal-only root."""
    from phenotypic.sdk_ import atomic_write_json

    validate_provenance_journal(image._metadata.provenance_journal)
    root = Path(store) / "zarr.json"
    if root.is_file():
        payload = json.loads(root.read_text(encoding="utf-8"))
    elif journal_only:
        root.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"phenotypic": {}},
        }
    else:
        raise FileNotFoundError(root)
    attributes = payload.setdefault("attributes", {})
    phenotypic = attributes.setdefault("phenotypic", {})
    phenotypic["provenance"] = deepcopy(image._metadata.provenance_journal)
    if work_id is not None:
        phenotypic["work_id"] = work_id
    atomic_write_json(
        root, payload, sort_keys=False, commit_guard=commit_guard
    )
    return root.parent
