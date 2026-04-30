"""Curation layer for the results viewer.

This module backs the viewer's "remove colony" feature. The pipeline writes
``master_measurements.parquet`` once and never touches it again; users curate
that frame by marking ``(Metadata_ImageFile, ObjectLabel)`` keys as removed.
The curated view is mirrored to two sibling files in the output root:

- ``<root>/filtered_measurements.parquet`` — source of truth for the curated
  frame, used to restore state across viewer sessions.
- ``<root>/filtered_measurements.csv`` — human-readable mirror so users can
  hand-inspect or share the curated subset without booting polars.

The :class:`FilteredMeasurements` dataclass holds the in-memory removal set,
loads existing curation files, and atomically rewrites both mirrors whenever
the user mutates the set. All public mutators serialise on a per-instance
re-entrant lock so concurrent Dash callbacks cannot interleave reads and
writes — and so a future caller that wraps an external mutation in
``with state._lock:`` cannot deadlock against the lock-acquiring
:meth:`save`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import polars as pl

logger = logging.getLogger(__name__)

#: Column name that identifies the source image of a colony.
KEY_IMAGE_FILE: str = "Metadata_ImageFile"

#: Column name that identifies a colony within its source image.
KEY_OBJECT_LABEL: str = "ObjectLabel"

#: Tuple form of the curation key columns. Importable from a single
#: source so callers (filter panel, viewer card, colony grid, crop
#: route) don't drift apart on string literals.
KEY_COLUMNS: tuple[str, str] = (KEY_IMAGE_FILE, KEY_OBJECT_LABEL)

# Backwards-compat alias for code that already grew up reading the
# leading-underscore name; do not introduce new uses.
_KEY_COLUMNS = KEY_COLUMNS


def decode_removed_keys_payload(
    payload: object,
) -> list[tuple[str, int]]:
    """Coerce a ``STORE_REMOVED_KEYS`` / selection payload into typed keys.

    Dash stores marshal data as JSON, so what arrives is a list of two-
    element lists with possibly stringified ints. This helper round-trips
    each entry to ``(str, int)`` and silently drops anything malformed
    (logged at DEBUG so the coercion is observable but not noisy).

    Args:
        payload: Whatever the Dash store returned. Expected shape is
            ``[[image_file, label], ...]`` but anything else is tolerated.

    Returns:
        A list of ``(image_file, object_label)`` tuples in the order
        they appeared in the input. Use ``set(...)`` if you need a
        hash-set instead.
    """
    if not isinstance(payload, list):
        return []
    out: list[tuple[str, int]] = []
    for entry in payload:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            out.append((str(entry[0]), int(entry[1])))
        except (TypeError, ValueError):
            logger.debug("Dropping malformed removed-keys entry %r", entry)
            continue
    return out


def _extract_keys(df: pl.DataFrame) -> set[tuple[str, int]]:
    """Pull ``(Metadata_ImageFile, ObjectLabel)`` keys out of ``df``.

    Args:
        df: A polars frame that must expose both key columns.

    Returns:
        A set of ``(image_file, object_label)`` tuples. ``image_file`` is
        coerced to ``str`` and ``object_label`` to ``int`` so the set is safe
        to compare across frames that may differ in dtype (e.g. parquet vs.
        Dash JSON round-trip).

    Raises:
        ValueError: If either key column is missing from ``df``.
    """
    missing = [col for col in _KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "FilteredMeasurements requires the master frame to expose "
            f"{list(_KEY_COLUMNS)} but the following are missing: {missing}. "
            "Re-run the pipeline or pass a frame that exposes these columns."
        )
    image_files = df.get_column(_KEY_COLUMNS[0]).to_list()
    object_labels = df.get_column(_KEY_COLUMNS[1]).to_list()
    return {(str(f), int(label)) for f, label in zip(image_files, object_labels)}


@dataclass
class FilteredMeasurements:
    """In-memory curation state plus on-disk mirrors.

    Removals are keyed by ``(Metadata_ImageFile, ObjectLabel)``. The class
    never mutates ``master_measurements.parquet``; instead it writes a
    curated copy to :attr:`parquet_path` and :attr:`csv_path`. Public
    mutators (:meth:`remove`, :meth:`restore`, :meth:`remove_many`,
    :meth:`restore_many`) acquire :attr:`_lock` for the duration of the
    mutation and the subsequent save so concurrent callbacks cannot race.

    The :attr:`_master_df` reference is captured at :meth:`load` time and
    reused on every save, so mutations don't pay a parquet re-read per
    click. This also keeps the in-memory and on-disk views of the master
    frame in sync — if some external process replaced the parquet under
    the running viewer, callers must explicitly re-:meth:`load` to pick
    up the change.

    Attributes:
        root: Output root directory (the parent that holds
            ``master_measurements.parquet``).
        parquet_path: Destination for the curated parquet mirror,
            conventionally ``<root>/filtered_measurements.parquet``.
        csv_path: Destination for the curated CSV mirror, conventionally
            ``<root>/filtered_measurements.csv``.
        removed_keys: Set of removed ``(image_file, object_label)`` tuples.
            Mutating this set directly bypasses the lock and the on-disk
            mirrors — prefer the public mutators.
        _master_df: Cached reference to the master frame supplied at
            :meth:`load` time. Used by every internal save so mutations
            don't pay a parquet re-read on the click hot path.
        _lock: Per-instance re-entrant mutex protecting concurrent
            mutations and saves. Re-entrant so callers (or future code)
            can hold the lock across a mutation + save without
            deadlocking on the lock that :meth:`save` itself takes.
            Excluded from the dataclass repr.
    """

    root: Path
    parquet_path: Path
    csv_path: Path
    removed_keys: set[tuple[str, int]]
    _master_df: pl.DataFrame = field(repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def load(cls, root: Path, master_df: pl.DataFrame) -> "FilteredMeasurements":
        """Build a :class:`FilteredMeasurements` from disk state.

        Reads ``<root>/filtered_measurements.parquet`` if present and
        computes :attr:`removed_keys` as the master keys minus the curated
        keys. Keys that appear in the curated parquet but not in the master
        frame are logged at WARNING and dropped — this happens when the
        master parquet has been re-run with a different set of objects.

        If the curated parquet is absent, an empty removal set is returned
        and **no files are written**. The on-disk mirrors are only created
        the first time the user actually removes something.

        Args:
            root: Output root directory.
            master_df: The full master measurements frame, used to compute
                the removal set as ``master_keys − filtered_keys`` and
                cached on the instance for subsequent saves. Must expose
                both ``Metadata_ImageFile`` and ``ObjectLabel``.

        Returns:
            A new :class:`FilteredMeasurements` whose :attr:`removed_keys`
            reflects the on-disk curation state (or an empty set if the
            mirrors do not exist yet).

        Raises:
            ValueError: If the master frame is missing one of the key
                columns. Re-raised from :func:`_extract_keys` with a
                friendly message rather than letting polars' own
                ``ColumnNotFoundError`` bubble up at viewer-boot time.
        """
        parquet_path = root / "filtered_measurements.parquet"
        csv_path = root / "filtered_measurements.csv"

        # Validate up front so a bad master surfaces at boot, not on the
        # first user click.
        master_keys = _extract_keys(master_df)

        if not parquet_path.exists():
            return cls(
                root=root,
                parquet_path=parquet_path,
                csv_path=csv_path,
                removed_keys=set(),
                _master_df=master_df,
            )

        filtered_df = pl.read_parquet(parquet_path)
        filtered_keys = _extract_keys(filtered_df)

        unknown = filtered_keys - master_keys
        if unknown:
            logger.warning(
                "Curated parquet at %s contains %d key(s) not present in the master "
                "frame; dropping them from the removal set.",
                parquet_path,
                len(unknown),
            )
            filtered_keys = filtered_keys & master_keys

        removed_keys = master_keys - filtered_keys
        return cls(
            root=root,
            parquet_path=parquet_path,
            csv_path=csv_path,
            removed_keys=removed_keys,
            _master_df=master_df,
        )

    # ------------------------------------------------------------------ #
    # Read-only queries (no lock required).
    # ------------------------------------------------------------------ #

    def is_removed(self, image_file: str, object_label: int) -> bool:
        """Return whether ``(image_file, object_label)`` is currently removed.

        Args:
            image_file: Value of ``Metadata_ImageFile`` for the colony.
            object_label: Value of ``ObjectLabel`` for the colony.

        Returns:
            ``True`` if the key is in :attr:`removed_keys`, else ``False``.
        """
        return (image_file, object_label) in self.removed_keys

    def removed_count_in(self, df: pl.DataFrame) -> int:
        """Count rows of ``df`` whose key is currently removed.

        Args:
            df: Any polars frame exposing both key columns. Typically a
                filtered or paginated view of the master frame.

        Returns:
            The number of rows in ``df`` whose
            ``(Metadata_ImageFile, ObjectLabel)`` is in
            :attr:`removed_keys`.
        """
        if df.is_empty() or not self.removed_keys:
            return 0
        df_keys = _extract_keys(df)
        return len(df_keys & self.removed_keys)

    def filtered_df(self, master_df: pl.DataFrame) -> pl.DataFrame:
        """Return ``master_df`` with all removed rows dropped.

        Implemented as an anti-join on a synthetic two-column polars frame
        built from :attr:`removed_keys`. The cast to ``str`` / ``Int64``
        mirrors the coercion done in :func:`_extract_keys`, so frames whose
        ``Metadata_ImageFile`` is e.g. ``Categorical`` still match.

        Args:
            master_df: The full master measurements frame.

        Returns:
            A new frame containing only rows whose key is not in
            :attr:`removed_keys`. If :attr:`removed_keys` is empty, the
            input frame is returned unchanged.
        """
        if not self.removed_keys:
            return master_df

        removed_frame = pl.DataFrame(
            {
                _KEY_COLUMNS[0]: [k[0] for k in self.removed_keys],
                _KEY_COLUMNS[1]: [k[1] for k in self.removed_keys],
            },
            schema={_KEY_COLUMNS[0]: pl.String, _KEY_COLUMNS[1]: pl.Int64},
        )

        # Cast master keys to matching dtypes so the anti-join lines up
        # even if the source frame stores them as Categorical / UInt32.
        keyed_master = master_df.with_columns(
            pl.col(_KEY_COLUMNS[0]).cast(pl.String),
            pl.col(_KEY_COLUMNS[1]).cast(pl.Int64),
        )
        return keyed_master.join(
            removed_frame,
            on=list(_KEY_COLUMNS),
            how="anti",
        )

    def removed_keys_payload(self) -> list[list]:
        """Serialise :attr:`removed_keys` for ``STORE_REMOVED_KEYS``.

        Dash stores marshal data as JSON, so tuples become lists. The output
        is sorted (image file ascending, then label ascending) to give
        callbacks a deterministic ordering for diffing.

        Returns:
            A list of ``[image_file, object_label]`` pairs ready to be
            placed in ``dcc.Store``.
        """
        sorted_keys = sorted(self.removed_keys, key=lambda k: (k[0], k[1]))
        return [[image_file, object_label] for image_file, object_label in sorted_keys]

    # ------------------------------------------------------------------ #
    # Mutators (each acquires the lock and persists to disk).
    # ------------------------------------------------------------------ #

    def remove(self, image_file: str, object_label: int) -> None:
        """Mark a single colony as removed and persist the change.

        Idempotent: removing an already-removed key is a no-op (the lock is
        still acquired, but no save is performed). Otherwise the key is
        added to :attr:`removed_keys` and both on-disk mirrors are
        rewritten.

        Args:
            image_file: Value of ``Metadata_ImageFile`` for the colony.
            object_label: Value of ``ObjectLabel`` for the colony.
        """
        key = (image_file, object_label)
        with self._lock:
            if key in self.removed_keys:
                return
            self.removed_keys.add(key)
            self._save_locked()

    def restore(self, image_file: str, object_label: int) -> None:
        """Restore a single colony and persist the change.

        Idempotent: restoring a key that was never removed is a no-op.
        Otherwise the key is dropped from :attr:`removed_keys` and both
        on-disk mirrors are rewritten.

        Args:
            image_file: Value of ``Metadata_ImageFile`` for the colony.
            object_label: Value of ``ObjectLabel`` for the colony.
        """
        key = (image_file, object_label)
        with self._lock:
            if key not in self.removed_keys:
                return
            self.removed_keys.discard(key)
            self._save_locked()

    def remove_many(self, keys: Iterable[tuple[str, int]]) -> None:
        """Mark a batch of colonies as removed in a single save.

        Acquires the lock once, applies every key, and writes the on-disk
        mirrors at most once. If no key in ``keys`` is new, no save is
        performed.

        Args:
            keys: Iterable of ``(image_file, object_label)`` tuples.
        """
        new_keys = set(keys)
        with self._lock:
            additions = new_keys - self.removed_keys
            if not additions:
                return
            self.removed_keys |= additions
            self._save_locked()

    def restore_many(self, keys: Iterable[tuple[str, int]]) -> None:
        """Restore a batch of colonies in a single save.

        Acquires the lock once, applies every key, and writes the on-disk
        mirrors at most once. If no key in ``keys`` was actually removed,
        no save is performed.

        Args:
            keys: Iterable of ``(image_file, object_label)`` tuples.
        """
        target_keys = set(keys)
        with self._lock:
            removals = target_keys & self.removed_keys
            if not removals:
                return
            self.removed_keys -= removals
            self._save_locked()

    def toggle(self, image_file: str, object_label: int) -> None:
        """Flip the curation state for a single colony, lock-guarded.

        Equivalent to ``restore`` if the key was removed, ``remove`` if
        it wasn't. Implemented as a single critical section so callers
        don't have to choose-then-mutate (which would race) and so the
        save fires exactly once.

        Args:
            image_file: Value of ``Metadata_ImageFile`` for the colony.
            object_label: Value of ``ObjectLabel`` for the colony.
        """
        key = (image_file, object_label)
        with self._lock:
            if key in self.removed_keys:
                self.removed_keys.discard(key)
            else:
                self.removed_keys.add(key)
            self._save_locked()

    def mutate_and_payload(
        self, action: Callable[["FilteredMeasurements"], None]
    ) -> list[list]:
        """Apply ``action`` and return the new payload, all under the lock.

        Callers that write ``STORE_REMOVED_KEYS`` after a mutation should
        prefer this helper over a separate ``mutate`` + ``removed_keys_payload``
        pair: the second call would happen after the lock had been released,
        so a concurrent mutator could change the payload between the two.
        Holding the (re-entrant) lock across both gives Dash a consistent
        snapshot.

        Args:
            action: A callable that performs whatever mutation is desired.
                Receives this :class:`FilteredMeasurements` instance and
                may call any of the public mutators (which will re-enter
                the lock harmlessly thanks to :class:`threading.RLock`).

        Returns:
            The updated :meth:`removed_keys_payload` after ``action`` ran.
        """
        with self._lock:
            action(self)
            return self.removed_keys_payload()

    # ------------------------------------------------------------------ #
    # Public save (used by callers that already have the master frame).
    # ------------------------------------------------------------------ #

    def save(self, master_df: pl.DataFrame | None = None) -> None:
        """Atomically rewrite both on-disk mirrors.

        Acquires :attr:`_lock` for the duration of the write. The parquet
        and CSV are each written to a ``.tmp`` sidecar and then
        ``os.replace``-d into place, which is atomic on every platform we
        support.

        If :attr:`removed_keys` is empty **and** neither mirror exists yet,
        this is a no-op — empty curation files are not created until the
        user actually removes something. If the mirrors already exist (the
        user removed and then restored everything), the full master frame
        is written so the on-disk view stays in parity with what the
        viewer shows.

        Args:
            master_df: Optional override of the master frame to write. If
                omitted (the common case from the public mutators) the
                cached :attr:`_master_df` reference is used. Pass an
                explicit frame only when the master has been refreshed
                out-of-band.
        """
        with self._lock:
            self._save_locked(master_df)

    # ------------------------------------------------------------------ #
    # Internal helpers.
    # ------------------------------------------------------------------ #

    def _save_locked(self, master_df: pl.DataFrame | None = None) -> None:
        """Write both mirrors, assuming :attr:`_lock` is already held.

        Args:
            master_df: Optional override; defaults to the cached
                :attr:`_master_df` reference captured at :meth:`load` time.
        """
        df = master_df if master_df is not None else self._master_df
        if not self.removed_keys and not self.parquet_path.exists() and not self.csv_path.exists():
            # Never been curated; don't create empty mirror files.
            return

        filtered = self.filtered_df(df)

        parquet_tmp = self.parquet_path.with_suffix(self.parquet_path.suffix + ".tmp")
        csv_tmp = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")

        filtered.write_parquet(parquet_tmp)
        os.replace(parquet_tmp, self.parquet_path)

        try:
            filtered.write_csv(csv_tmp)
            os.replace(csv_tmp, self.csv_path)
        except Exception:
            # Parquet has succeeded; the CSV is best-effort and
            # regenerated from parquet on next load. Log loudly so
            # operators notice if non-CSV-encodable columns were added
            # upstream.
            logger.exception(
                "Failed to write curation CSV mirror at %s; parquet write "
                "succeeded so curation state is preserved.",
                self.csv_path,
            )
