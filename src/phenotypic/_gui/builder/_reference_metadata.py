"""Builder-side reference metadata: preview context and RefColumn dropdown choices.

The builder never stores a table on an operation. The table picked for the
session (``_DagBuilderState.reference_metadata_path``) is the preview's ambient
ReferenceContext, as ``--metadata`` is for the CLI, and supplies the dropdown
choices for ``RefColumn`` parameters.

Imports nothing heavy at module level: the layout modules import it, and the
ReferenceContext machinery (polars, pandas) loads only when a table is set.
"""

from __future__ import annotations

import functools
import hashlib
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

REFERENCE_SOURCE = "reference_metadata"

#: Appended to a "no active ReferenceContext" preview error: that message
#: tells a Python or CLI user what to do, and the builder's answer differs.
_PICK_A_TABLE_HINT = (
    "Pick a table in the builder's Reference metadata field "
    "(under Image source) and run the preview again."
)

#: One or more ``ImagePipeline`` step wrappers, outermost first, as
#: ``_image_pipeline_core.py`` formats them: ``[Op] (step i/n, key='k'): ``.
_STEP_PREFIXES = re.compile(r"(?:\[\w+\] \(step \d+/\d+, key='[^\n]*?'\): )+")


def describe_reference_table(path: Optional[str]) -> tuple[str, str]:
    """Validate a picked table: ``(value_to_store, status_message)``.

    Args:
        path: The path typed into the picker, or ``None``/``""`` to clear it.

    Returns:
        The resolved path to store (``""`` when cleared or invalid) and a
        one-line status: rows and columns, or why the table was refused.
    """
    if not path:
        return "", ""
    from phenotypic._core._reference_context import ReferenceContext, ReferenceTableError

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return "", f"Not found: {candidate}"
    try:
        context = ReferenceContext(candidate)
    except ReferenceTableError as exc:
        return "", str(exc)
    return (
        str(candidate.resolve()),
        f"{candidate.name} · {context.table.height} rows · {len(context.columns)} columns",
    )


@functools.lru_cache(maxsize=8)
def _columns_for(path: str, mtime_ns: int) -> tuple[str, ...]:
    from phenotypic._core._reference_context import ReferenceContext

    return ReferenceContext(path).columns


def reference_columns_provider(path: Optional[str]) -> Optional[Callable[[str], list[str]]]:
    """A ``columns_provider`` for the shared param form, or ``None`` without a table.

    Cached on the file's mtime: the inspector re-renders on every edit.
    """
    if not path:
        return None
    from phenotypic._core._reference_context import ReferenceTableError

    try:
        columns = _columns_for(path, Path(path).stat().st_mtime_ns)
    except (OSError, ReferenceTableError):
        return None

    def provide(source: str) -> list[str]:
        return list(columns) if source == REFERENCE_SOURCE else []

    return provide


@functools.lru_cache(maxsize=8)
def _file_sha256(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_identity(path: Optional[str]) -> str:
    """Content identity of the picked table for preview fingerprints and revisions.

    Cached on ``(path, mtime, size)``: the preview status is re-derived on
    every builder render, and the table is re-read only when it changes.
    """
    if not path:
        return ""
    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError:
        return f"missing:{candidate}"
    if not candidate.is_file():
        return f"missing:{candidate}"
    return _file_sha256(str(candidate), stat.st_mtime_ns, stat.st_size)


@contextmanager
def preview_reference_context(
    path: Optional[str], image_path: Optional[str]
) -> Iterator[object]:
    """Activate the picked table around a preview.

    Reference images resolve beside the preview image. When the table has a
    ``Metadata_Dataset`` column naming the image's directory, lookups narrow
    to that dataset, as the CLI narrows each dataset by its input directory's
    name. With no table this activates nothing, so a reference operation
    fails exactly as a bare ``apply`` would.
    """
    if not path:
        yield None
        return
    from phenotypic._core._reference_context import ReferenceContext
    from phenotypic._gui.builder._directory_browser import SYNTHETIC_SENTINEL
    from phenotypic.schema import EXPERIMENT

    # The synthetic plate has no directory; its sentinel's parent is the cwd.
    root = Path(image_path).parent if image_path and image_path != SYNTHETIC_SENTINEL else None
    context = ReferenceContext(path, image_root=root)
    dataset_column = str(EXPERIMENT.DATASET)
    if (
        root is not None
        and dataset_column in context.columns
        and root.name in context.table.get_column(dataset_column).to_list()
    ):
        context = context.narrow(dataset=root.name)
    with context:
        yield context


def reference_error_message(exc: BaseException) -> Optional[str]:
    """The innermost reference-metadata failure behind *exc*, for display.

    ``ImagePipeline`` wraps an operation's error in a ``RuntimeError`` (once
    per nesting level), so the message a user can act on sits at the bottom
    of the ``__cause__`` chain. The wrappers' ``[Op] (step i/n, key='k'): ``
    prefixes are kept in front of it: with two reference operations, the
    bare message does not say which one failed.

    Returns:
        ``"<step prefixes><ErrorType>: <message>"`` for the innermost
        ``ReferenceContextError``, or ``None`` when the chain holds none.
    """
    from phenotypic._core._reference_context import (
        RefMetadataUnavailableError,
        ReferenceContextError,
    )

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        # The traceback module's rule: ``raise ... from None`` hides the context.
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    found = next(
        (e for e in reversed(chain) if isinstance(e, ReferenceContextError)), None
    )
    if found is None:
        return None
    message = f"{type(found).__name__}: {found}"
    detail = str(found)
    for wrapper in chain:
        text = str(wrapper)
        if wrapper is not found and text.endswith(detail):
            prefix = text[: len(text) - len(detail)]
            if prefix and _STEP_PREFIXES.fullmatch(prefix):
                message = prefix + message
                break
    if isinstance(found, RefMetadataUnavailableError):
        message = f"{message}\n{_PICK_A_TABLE_HINT}"
    return message
