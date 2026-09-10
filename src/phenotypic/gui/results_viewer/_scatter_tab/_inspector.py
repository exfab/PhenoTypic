"""Resolve a clicked point back to the colony it represents.

The producer (:func:`index_frame`) and the consumer (:func:`resolve_click`)
live in one module on purpose. Gate 0 found the plan building the index
against the *filtered* frame in one task while resolving it against
``master_df`` in another: every click would have opened a real but wrong
colony -- a real crop, a plausible result, nothing raising. Splitting a
contract across two modules is how that survived a careful read, so both
halves are here and any caller that reaches for ``with_row_index`` itself
has reintroduced the defect.

Three properties below are correctness requirements, and each is a
measurement rather than a judgement:

* **The index is stamped before filtering, never after.** The viewer
  filters through ``FilterSpec.apply_to``, which is *not* a bare
  ``.filter()`` -- it runs ``normalize_viewer_frame`` first, renaming every
  non-metadata column to a shield name and back again. Measured against
  the verification fixture: the column survives that round trip with its
  values intact, and every carried index still resolves to the same
  colony.
* **A stale index is refused, not resolved.** ``master_df`` is stable
  within one binding but not across a Refresh, and the mirror is
  session-mutable, so the caller carries the ``OutputSnapshotDescriptor``
  fingerprint captured when the figure was drawn.
* **Every guard here fails closed.** A refusal returns ``None`` and the
  inspector simply does not open. That is the only safe direction: the
  alternative to refusing is not an error, it is the wrong colony
  rendered convincingly.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import SupportsIndex

import polars as pl

from phenotypic.gui.results_viewer._filtered_state import (
    KEY_DATASET,
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
)
from phenotypic.gui.results_viewer._scatter_tab._figure import CUSTOMDATA_COL


@dataclass(frozen=True)
class ColonyRef:
    """The key the crop route, the Viv stage and curation all take.

    Args:
        dataset: The dataset (plate / condition group) the colony sits in.
        stem: The source image's stem.
        label: The colony's ``Object_Label`` within that image.
    """

    dataset: str
    stem: str
    label: int


def index_frame(master_df: pl.DataFrame) -> pl.DataFrame:
    """Stamp each row with its positional index into ``master_df``.

    **Call this before filtering, never after.** The index written here is
    positional into the frame passed in, and :func:`resolve_click` reads it
    positionally out of the same frame. Indexing a filtered frame instead
    produces indices that address the wrong rows of ``master_df`` -- and
    because every one of them is a real colony with a real crop, nothing
    errors and the inspector simply shows the wrong colony.

    A frame that already carries the column is returned **unchanged**,
    rather than re-stamped or raised on. Both alternatives are worse.
    ``with_row_index`` raises ``DuplicateError`` on a re-call, which would
    turn a defensive call in a callback into a 500; and catching that to
    re-stamp would renumber an already-filtered frame, which is precisely
    the defect this module exists to prevent -- manufactured by the guard
    meant to prevent a crash. An index already present is master-anchored
    and therefore already correct, so handing it back is both the safe
    answer and the right one.

    Args:
        master_df: The frozen run frame from ``OutputRoot``.

    Returns:
        ``master_df`` with a ``CUSTOMDATA_COL`` index column prepended, or
        unchanged when it already carries one.
    """
    if CUSTOMDATA_COL in master_df.columns:
        return master_df
    return master_df.with_row_index(CUSTOMDATA_COL)


def _row_position(index: object) -> int | None:
    """Read a click payload as a row position, or refuse it.

    The test is "does it implement ``__index__``", never
    ``isinstance(index, int)``, and the difference is load-bearing in both
    directions:

    * ``numpy.int64`` is **not** a subclass of Python ``int``, but polars
      accepts it as a row index perfectly well. A type whitelist would
      turn a working click into a dead one, silently, because a refusal
      opens nothing.
    * ``bool`` *is* an ``int`` subclass and ``True.__index__()`` is ``1``,
      so a truthy flag arriving where an index is expected would resolve
      to colony 1. It is checked first, before anything else can accept
      it.

    ``float``, ``str`` and ``None`` implement no ``__index__`` and are
    refused. A float in particular must not be coerced:
    ``DataFrame.row(1.0)`` raises, so a producer emitting floats is a
    defect to surface, not to paper over.

    ``SupportsIndex`` is a runtime-checkable protocol, so the one
    ``isinstance`` both narrows the argument for the type checker and does
    the real work -- no ``type: ignore``, and one guard rather than a
    check plus a ``try/except`` that would shadow each other.

    Args:
        index: The raw customdata value carried back from the click.

    Returns:
        The row position, or None when the value cannot be one.
    """
    if isinstance(index, bool):
        return None
    if not isinstance(index, SupportsIndex):
        return None
    return operator.index(index)


def resolve_click(
    master_df: pl.DataFrame,
    index: int,
    fingerprint: str,
    expected_fingerprint: str,
) -> ColonyRef | None:
    """Resolve a point's row index into a colony, or refuse it.

    The index is positional into ``master_df``, which ``OutputRoot``
    freezes at ``discover()`` -- not into the filtered frame, which is
    re-derived on every filter and sort change. A positional index into a
    moving frame has a race with no error path: a click on a stale figure
    resolves against the new frame and opens the wrong colony, silently
    and plausibly.

    ``master_df`` is stable within one binding but not across a refresh,
    and curation can be written while the tab is open, so the caller
    passes the fingerprint captured when the figure was drawn. A mismatch
    is refused.

    Every other guard exists because the value it rejects would otherwise
    produce a *plausible* answer rather than an error. A negative index
    resolves Python-style to the last row (measured). A null dataset or
    image name would stringify to ``"None"``, a well-formed reference to
    a dataset that does not exist. And ``master_df`` is the mirror, so it
    carries metadata-only phantoms -- 121 of the verification fixture's
    844 rows, 117,415 of the full run's 231,229 -- which have no colony to
    open at all.

    Args:
        master_df: The frozen run frame.
        index: Positional row index carried as the point's customdata.
        fingerprint: Snapshot fingerprint captured with the figure.
        expected_fingerprint: The binding's current fingerprint.

    Returns:
        The colony, or ``None`` when the index is stale, unreadable, out
        of range, or lands on a row that cannot name a colony.
    """
    if fingerprint != expected_fingerprint:
        return None

    position = _row_position(index)
    if position is None or position < 0 or position >= master_df.height:
        return None

    # `.get` rather than `[...]`: a frame missing one of these columns is
    # a caller error, and refusing beats a KeyError that 500s the tab.
    row = master_df.row(position, named=True)
    dataset = row.get(KEY_DATASET)
    stem = row.get(KEY_IMAGE_FILE)
    if dataset is None or stem is None:
        return None
    try:
        # ONE guard for "this row has no usable label", deliberately not
        # two. A phantom's label is null and `int(None)` raises TypeError;
        # were the column ever Float64 a phantom would be NaN, which is
        # not None and which `int()` rejects with ValueError. An explicit
        # `label is None` check in front of this would be shadowed by it
        # -- both would have to be removed to change any behaviour, so
        # neither would be pinned by a test, and a redundant guard reads
        # as a tested one.
        label = int(row.get(KEY_OBJECT_LABEL))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return ColonyRef(dataset=str(dataset), stem=str(stem), label=label)
