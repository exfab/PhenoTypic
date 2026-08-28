"""Context-local operation provenance for :class:`phenotypic.Image`."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, TypeVar, cast

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic.abc_._image_operation import ImageOperation
    from phenotypic.sdk_ import CommitGuard


PROVENANCE_SCHEMA_VERSION = 1

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


def new_provenance_journal() -> dict[str, Any]:
    """Return a fresh version-1 journal owned by one image."""
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    }


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

    return tuple(_freeze(deepcopy(entry)) for entry in journal["operations"])


def _journal_with_operations(
    source_journal: Mapping[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy a journal and append entries it does not already contain."""
    merged: dict[str, Any] = deepcopy(dict(source_journal))
    operations = merged["operations"]
    for entry in entries:
        if entry in operations:
            continue
        carried_entry = deepcopy(entry)
        carried_entry["sequence"] = len(operations) + 1
        operations.append(carried_entry)
    return merged


def _carry_logical_image_state(
    result: "Image",
    source_journal: Mapping[str, Any],
    source_original: Any,
    nested_operations: list[dict[str, Any]],
) -> None:
    """Carry image-owned provenance/source state across replacement operations."""
    returned_operations = deepcopy(
        result._metadata.provenance_journal.get("operations", [])
    )
    source_operations = source_journal["operations"]
    common_prefix = 0
    for source_entry, returned_entry in zip(
        source_operations, returned_operations, strict=False
    ):
        if source_entry != returned_entry:
            break
        common_prefix += 1

    merged = _journal_with_operations(source_journal, nested_operations)
    operations = merged["operations"]
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
        carried_entry["sequence"] = len(operations) + 1
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
            # ``super().apply`` is another wrapper layer around the same public
            # invocation. A distinct operation instance starts its own frame.
            return apply_method(self, *args, **kwargs)

        parent_frame = stack[-1] if stack else None
        frame = _OperationApplyFrame(operation=self, owner=owner)
        token = _operation_apply_stack.set((*stack, frame))
        logical_input = cast(
            "Image", kwargs.get("image", args[0] if args else None)
        )
        input_journal = deepcopy(logical_input._metadata.provenance_journal)
        source_original = logical_input._original
        source_journal = _journal_with_operations(
            input_journal,
            parent_frame.nested_records if parent_frame is not None else [],
        )
        source_length = len(source_journal["operations"])
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
            operations = result._metadata.provenance_journal["operations"]
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
            produced_records = deepcopy(operations[source_length:])
            if parent_frame is not None:
                parent_frame.nested_records.extend(produced_records)
                parent_frame.nested_sink_published = (
                    parent_frame.nested_sink_published
                    or frame.nested_sink_published
                )
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


def initialize_cli_provenance(
    image: "Image",
    pipeline_path: str | Path,
    *,
    pipeline_identity: Mapping[str, str] | None = None,
    status: str = "in_progress",
    retry_base_length: int = 0,
    basename_only: bool = False,
) -> None:
    """Reset a decoded image to a fresh CLI journal for this pipeline attempt.

    Args:
        image: The image whose provenance journal is reset.
        pipeline_path: The pipeline file this attempt is running.
        pipeline_identity: Precomputed identity to record instead of deriving
            one from *pipeline_path*.
        status: Initial journal status.
        retry_base_length: Durable Stage-1 operation-count prefix to seed.
        basename_only: Forwarded to :func:`pipeline_source_identity`. Ignored
            when *pipeline_identity* is supplied, since the caller has then
            already decided what to record.
    """
    journal = new_provenance_journal()
    journal.update(
        {
            "status": status,
            "pipeline": (
                pipeline_source_identity(
                    pipeline_path, basename_only=basename_only
                )
                if pipeline_identity is None
                else deepcopy(dict(pipeline_identity))
            ),
            "retry_base_length": int(retry_base_length),
        }
    )
    image._metadata.provenance_journal = journal


def set_provenance_status(image: "Image", status: str) -> None:
    """Set the lifecycle status on an image-owned journal."""
    image._metadata.provenance_journal["status"] = status


def set_retry_base_length(image: "Image", length: int) -> None:
    """Persist the Stage-1 prefix length used to make Stage-3 retries idempotent."""
    image._metadata.provenance_journal["retry_base_length"] = int(length)


def truncate_provenance_to_retry_base(image: "Image") -> None:
    """Discard entries written after the durable Stage-1 prefix."""
    journal = image._metadata.provenance_journal
    base = int(journal.get("retry_base_length", 0))
    del journal["operations"][base:]


def append_operation_provenance(
    image: "Image",
    operation: "ImageOperation",
    *,
    duration_seconds: float,
    pipeline_step_path: list[str] | None,
) -> None:
    """Append one successful leaf record, including staged detector merges."""
    operations = image._metadata.provenance_journal["operations"]
    parameters = json.loads(
        json.dumps(operation.model_dump(mode="json"), ensure_ascii=False)
    )
    operations.append(
        {
            "sequence": len(operations) + 1,
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


def write_provenance_checkpoint(
    store: str | Path,
    image: "Image",
    *,
    journal_only: bool = False,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically replace only root provenance, or create a journal-only root."""
    from phenotypic.sdk_ import atomic_write_json

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
    atomic_write_json(
        root, payload, sort_keys=False, commit_guard=commit_guard
    )
    return root.parent
