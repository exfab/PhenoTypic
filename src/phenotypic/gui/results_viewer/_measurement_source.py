"""Which measurement columns a run offers, and one column's values for a grid.

The Colony grid draws from many images at once, so both questions it asks --
"what can I display?" and "what is this colony's value?" -- span every store
behind the frame in view. This module answers them, and owns the two
properties that make doing so affordable.

**The column list costs no Parquet read.** Each store enumerates its own
columns in ``attributes.phenotypic.tables.measurements.measurement_columns``,
so populating a picker is a small JSON parse per image and never touches the
~130-column payload.

**Values are memoized on the payload's identity.** A grid re-render fires on
every filter, axis, tile-size and dim change; re-reading 32 Parquet files each
time would cost ~150 ms of pure repetition. The cache is keyed on the payload
file's inode, size and both timestamps -- the same construction
``_zarr_routes`` uses for a store generation, and for the same reason: a
rewrite that lands inside one timestamp tick still moves the inode or the
size, and a cache that keyed on mtime alone would go stale silently.

**A store with no ``tables`` descriptor contributes nothing, and that is
normal.** A ``--mode process`` run never measures, and a migrated store may
carry none. Such an image simply offers no columns and no values; its cards
render exactly as they do today. It must never be reported as "measurement
pending" -- see the retraction note in ``_store_source.py``.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    embedded_measurement_columns,
    read_embedded_measurement_column,
)

logger = logging.getLogger(__name__)

#: How many (payload, generation, column) projections to keep. A Colony grid
#: spans at most a few dozen images and the user changes column rarely, so a
#: few hundred entries covers a working session without holding a run's whole
#: table in memory -- each entry is one column, not 130.
_PROJECTION_CACHE_SIZE = 512

#: How many (store, generation) descriptors to keep. Read on every picker
#: refresh, once per image.
_DESCRIPTOR_CACHE_SIZE = 512


def _payload_identity(payload: Path) -> tuple[int, int, int, int]:
    """Identify one generation of a store's Parquet payload.

    Args:
        payload: Path to ``tables/measurements/table.parquet``.

    Returns:
        Inode, size and both timestamps.

    Raises:
        OSError: If the payload does not exist.
    """
    stat = os.stat(payload)
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@functools.lru_cache(maxsize=_DESCRIPTOR_CACHE_SIZE)
def _columns_for_store(
    store: str, identity: tuple[int, int, int, int]
) -> tuple[str, ...]:
    """Read one store's declared column list. Memoized on *identity*."""
    return embedded_measurement_columns(Path(store))


@functools.lru_cache(maxsize=_PROJECTION_CACHE_SIZE)
def _column_for_store(
    store: str, identity: tuple[int, int, int, int], column: str
) -> tuple[tuple[int, float | None], ...]:
    """Project one column of one store. Memoized on *identity*.

    Returns a tuple of pairs rather than a dict because
    :func:`functools.lru_cache` holds its return value directly and a mutable
    one would let a caller corrupt every later hit.
    """
    values = read_embedded_measurement_column(Path(store), column)
    return tuple(sorted(values.items()))


def _store_and_identity(
    output_root: OutputRoot, dataset: str, stem: str
) -> tuple[Path, tuple[int, int, int, int]] | None:
    """Resolve one image's store and its payload generation, or ``None``.

    ``None`` covers every routine absence in one place: a standalone bundle
    with no per-image stores, a store that carries no embedded table, and a
    promote in flight.
    """
    store = output_root.store_path(dataset, stem)
    if store is None or not store.is_dir():
        return None
    try:
        return store, _payload_identity(
            store / MEASUREMENT_TABLE_RELATIVE_PATH
        )
    except OSError:
        return None


def displayable_measurement_columns(
    output_root: OutputRoot,
    pairs: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Return the columns the Colony picker should offer, sorted.

    Two filters apply, in this order.

    The store's own ``measurement_columns`` supplies the candidates, so the
    picker can never offer a column that would 400 at the route. Reading it
    opens no Parquet.

    Then only columns that are **numeric** survive, because the display
    scales them onto a continuous ramp and there is no ramp over
    ``ColorLab_MedoidColorHex``'s ``#a08866``. Numeric-ness is asked of
    :meth:`OutputRoot.is_numeric_column`, which already answers it for the
    filter sidebar -- so a numeric-valued *string* column like ``Grid_RowNum``
    is correctly offered, and a column absent from the viewer's frame
    entirely is not.

    Args:
        output_root: Validated handle on the CLI output directory.
        pairs: ``(dataset, image_file)`` pairs currently in view.

    Returns:
        The offerable column names, sorted. **Empty is a normal answer**: a
        standalone bundle ships no stores, and a ``--mode process`` run
        measured nothing. Empty means "nothing to display", never "pending".
    """
    candidates: set[str] = set()
    for dataset, stem in pairs:
        resolved = _store_and_identity(output_root, dataset, stem)
        if resolved is None:
            continue
        store, identity = resolved
        try:
            candidates.update(_columns_for_store(str(store), identity))
        except (OSError, KeyError, RuntimeError):
            # No root, no ``phenotypic`` block, no descriptor, or a store
            # this build cannot decode. One unreadable image must not empty
            # the picker for the rest of the run.
            logger.debug(
                "No measurement columns for %s/%s", dataset, stem, exc_info=True
            )
    return tuple(
        sorted(
            column
            for column in candidates
            if output_root.is_numeric_column(column)
        )
    )


def measurement_values_for(
    output_root: OutputRoot,
    pairs: Iterable[tuple[str, str]],
    column: str,
) -> dict[tuple[str, int], float | None]:
    """Read one column across several images, keyed for the Colony grid.

    The grid identifies a colony by ``(image_file, label)`` and the store
    joins on the ``Object_Label`` its descriptor names, so this returns the
    grid's key directly and no call site re-derives the join.

    Args:
        output_root: Validated handle on the CLI output directory.
        pairs: ``(dataset, image_file)`` pairs to read.
        column: The measurement column to project.

    Returns:
        ``{(image_file, label): value}``. An image whose store is absent,
        undecodable, table-less, or which does not declare *column*
        contributes no keys -- its cards render untinted, which is the
        correct rendering for "no value here".
    """
    values: dict[tuple[str, int], float | None] = {}
    for dataset, stem in pairs:
        resolved = _store_and_identity(output_root, dataset, stem)
        if resolved is None:
            continue
        store, identity = resolved
        try:
            projected = _column_for_store(str(store), identity, column)
        except (OSError, KeyError, ValueError, TypeError, RuntimeError):
            # Absent store or descriptor, a column this image does not
            # declare, a non-numeric column, or a store this build cannot
            # decode. Every one of them means "no value for these cards".
            logger.debug(
                "No %s values for %s/%s", column, dataset, stem, exc_info=True
            )
            continue
        for label, value in projected:
            values[(stem, int(label))] = value
    return values
