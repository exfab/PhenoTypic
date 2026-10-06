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


#: The whole status for a refused path. It says nothing about the file, so a
#: path outside the image root cannot be probed for existence or contents.
REFUSED_TABLE_MESSAGE = (
    "Refused: the reference table must be a .csv or .parquet file under the image root."
)

_TABLE_SUFFIXES = (".csv", ".parquet")


def confined_reference_path(path: object, image_root: object) -> Optional[Path]:
    """Resolve a reference-table path the way the builder's other pickers are confined.

    The one rule for every use of the table path: the picker, and each
    consumer of the (client-writable) builder state, which re-checks at use.

    Args:
        path: The typed or stored path. A relative path resolves against
            ``image_root``, not the server's working directory.
        image_root: The builder's ``--image-root``; ``None`` refuses every path.

    Returns:
        The resolved path when it lies under ``image_root`` after following
        symlinks (``SandboxRoot.resolve``) and ends in ``.csv`` or
        ``.parquet``; otherwise ``None``.
    """
    if not isinstance(path, str) or not path or image_root is None:
        return None
    from phenotypic._gui.shell._sandbox import SandboxRoot

    try:
        sandbox = SandboxRoot.from_path(image_root)
        resolved = sandbox.resolve(Path(path).expanduser())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if resolved.suffix.lower() not in _TABLE_SUFFIXES:
        return None
    return resolved


def _confined(path: object) -> Optional[Path]:
    """:func:`confined_reference_path` against the running builder's image root.

    Outside an app context there is no image root, so every path is refused.
    """
    if not path:
        return None
    from flask import current_app, has_app_context

    from phenotypic._gui._config import CFG_IMAGE_ROOT

    if not has_app_context():
        return None
    return confined_reference_path(path, current_app.config.get(CFG_IMAGE_ROOT))


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

    candidate = _confined(path)
    if candidate is None:
        return "", REFUSED_TABLE_MESSAGE
    if not candidate.is_file():
        return "", f"Not found: {candidate}"
    try:
        context = ReferenceContext(candidate)
    except ReferenceTableError as exc:
        return "", str(exc)
    return (
        str(candidate),
        f"{candidate.name} · {context.table.height} rows · {len(context.columns)} columns",
    )


@functools.lru_cache(maxsize=8)
def _columns_for(path: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    from phenotypic._core._reference_context import ReferenceContext

    return ReferenceContext(path).columns


def reference_columns_provider(path: Optional[str]) -> Optional[Callable[[str], list[str]]]:
    """A ``columns_provider`` for the shared param form, or ``None`` without a table.

    Cached on the file's ``(mtime, size)``: the inspector re-renders on every edit.
    A path outside the image root is treated as no table.
    """
    confined = _confined(path)
    if confined is None:
        return None
    from phenotypic._core._reference_context import ReferenceTableError

    try:
        stat = confined.stat()
        columns = _columns_for(str(confined), stat.st_mtime_ns, stat.st_size)
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
    A path outside the image root is never read, and identifies as no table.
    """
    candidate = _confined(path)
    if candidate is None:
        return ""
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
    name. With no table, or one outside the image root, this activates
    nothing, so a reference operation fails exactly as a bare ``apply`` would.
    """
    confined = _confined(path)
    if confined is None:
        yield None
        return
    from phenotypic._core._reference_context import ReferenceContext
    from phenotypic._gui.builder._directory_browser import SYNTHETIC_SENTINEL
    from phenotypic.schema import EXPERIMENT

    # The synthetic plate has no directory; its sentinel's parent is the cwd.
    root = Path(image_path).parent if image_path and image_path != SYNTHETIC_SENTINEL else None
    context = ReferenceContext(confined, image_root=root)
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
