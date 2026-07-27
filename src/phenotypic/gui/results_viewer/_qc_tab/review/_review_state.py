"""GUI-owned per-module review progress for the QC Review tab.

``<output>/qc/review_state.json`` records, **per QC module** (keyed by
check ``instance_id``), which groups the user has marked reviewed and
which group they last visited:

```json
{
  "qc-SE-1a2b3c4d": {"reviewed": ["P1", "P2"], "last": "P3"},
  "qc-ICC-9f8e7d6c": {"reviewed": [], "last": null}
}
```

Ownership rules (spec §D.6):

* **Written only by the GUI** — :func:`phenotypic.sdk_._qc_recipe._runner.run_qc`
  never touches this file, so an in-session recompute preserves review
  progress.
* **Reset by the CLI** — ``finalize_post_master_outputs`` clears it on
  every recompile/remeasure, so a fresh run starts the queue over.

A group key is a tuple of the module's ``groupby`` values (e.g.
``("plate1", "A")``). JSON has no tuples, so keys are encoded as a
JSON-string of the value list (``'["plate1", "A"]'``) on the way out and
decoded back to a tuple on the way in — this keeps multi-column group
keys round-trippable through the on-disk store.

Full-file writes are serialized with an interprocess lock and use an exact
content-fingerprint compare-and-swap guard. A viewer session therefore never
overwrites progress written by another process after this state was loaded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic.sdk_ import BundleLayout

logger = logging.getLogger(__name__)

_MISSING_FINGERPRINT = "missing"


class ReviewStateConflictError(RuntimeError):
    """Raised when review state changed after the current session loaded it."""


def review_state_lock_path(path: Path) -> Path:
    """Return the canonical lock shared by every review-state writer."""
    state_path = Path(path)
    return state_path.with_name(f".{state_path.name}.lock")


#: A group key — the tuple of ``groupby`` column values identifying one
#: QC group. Single-column checks still use a 1-tuple.
GroupKey = tuple[object, ...]


def encode_group_key(key: GroupKey) -> str:
    """Encode a group-key tuple as a stable JSON-string store key.

    Args:
        key: The ``groupby`` value tuple (already JSON-native scalars).

    Returns:
        A canonical JSON-array string (e.g. ``'["plate1", "A"]'``) usable
        as a dict key in ``review_state.json``.
    """
    return json.dumps([_jsonable(v) for v in key], separators=(",", ":"))


def decode_group_key(encoded: str) -> GroupKey:
    """Decode a stored JSON-string key back into a group-key tuple.

    Args:
        encoded: A value previously produced by :func:`encode_group_key`.

    Returns:
        The decoded tuple. A malformed entry yields a 1-tuple of the raw
        string so a corrupt key never raises mid-render.
    """
    try:
        decoded = json.loads(encoded)
    except (json.JSONDecodeError, TypeError):
        return (encoded,)
    if isinstance(decoded, list):
        return tuple(decoded)
    return (decoded,)


def _jsonable(value: object) -> object:
    """Coerce a group-key element to a JSON-native scalar.

    polars / numpy scalars that slip through (e.g. ``numpy.int64``) are
    not JSON-serializable; stringify anything exotic so encoding never
    raises. Plain ``str``/``int``/``float``/``bool``/``None`` pass through
    unchanged so the common case round-trips exactly.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class ModuleProgress:
    """Per-module review progress: the reviewed set + last-visited group.

    Attributes:
        reviewed: Encoded group keys (see :func:`encode_group_key`) the
            user has marked reviewed.
        last: Encoded key of the last group the user opened, or ``None``.
    """

    reviewed: set[str] = field(default_factory=set)
    last: str | None = None


@dataclass
class ReviewState:
    """In-memory mirror of ``qc/review_state.json`` with atomic persistence.

    Loaded once per viewer session via :meth:`load`; mutated in place by
    the Review tab's mark-reviewed / advance callbacks and re-persisted on
    each change. All state is keyed by check ``instance_id`` so each QC
    module owns an independent worklist + progress (spec §D.6).

    Attributes:
        path: Absolute path to ``<output>/qc/review_state.json``.
        modules: Mapping ``instance_id -> ModuleProgress``.
    """

    path: Path
    modules: dict[str, ModuleProgress] = field(default_factory=dict)
    source_fingerprint: str = _MISSING_FINGERPRINT
    source_readable: bool = True
    original_payload: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, layout: "BundleLayout") -> "ReviewState":
        """Load review progress from the bundle's ``qc/review_state.json``.

        A missing or corrupt file yields an empty state (the file is
        created lazily on the first :meth:`save`). Corruption is logged at
        WARNING and the on-disk file is left untouched so it can be
        recovered by hand.

        Args:
            layout: Resolved bundle topology; the review-state path is
                ``layout.qc_review_state_path`` so a standalone deliverables
                bundle reads/writes inside the bundle.

        Returns:
            A :class:`ReviewState` ready for in-place mutation.
        """
        path = layout.qc_review_state_path
        if not path.exists():
            return cls(path=path)

        try:
            source_bytes = path.read_bytes()
            payload = json.loads(source_bytes)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "qc/review_state.json at %s could not be read; starting with "
                "empty review progress. On-disk file left untouched. Error: %s",
                path,
                exc,
            )
            from phenotypic.sdk_ import file_fingerprint

            try:
                fingerprint = file_fingerprint(path)
            except OSError:
                fingerprint = _MISSING_FINGERPRINT
            return cls(
                path=path,
                source_fingerprint=fingerprint,
                source_readable=False,
            )

        from phenotypic.sdk_ import bytes_fingerprint

        fingerprint = bytes_fingerprint(source_bytes)
        if not isinstance(payload, dict):
            logger.warning(
                "qc/review_state.json at %s is not a JSON object; refusing "
                "future writes from this session.",
                path,
            )
            return cls(
                path=path,
                source_fingerprint=fingerprint,
                source_readable=False,
            )

        modules: dict[str, ModuleProgress] = {}
        for instance_id, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            reviewed = raw.get("reviewed", [])
            last = raw.get("last")
            modules[str(instance_id)] = ModuleProgress(
                reviewed={str(k) for k in reviewed}
                if isinstance(reviewed, list)
                else set(),
                last=str(last) if isinstance(last, str) else None,
            )

        return cls(
            path=path,
            modules=modules,
            source_fingerprint=fingerprint,
            original_payload=dict(payload),
        )

    def progress_for(self, instance_id: str) -> ModuleProgress:
        """Return (creating if absent) the progress record for ``instance_id``."""
        return self.modules.setdefault(instance_id, ModuleProgress())

    def is_reviewed(self, instance_id: str, key: GroupKey) -> bool:
        """Return ``True`` if group ``key`` is marked reviewed for the module."""
        progress = self.modules.get(instance_id)
        if progress is None:
            return False
        return encode_group_key(key) in progress.reviewed

    def reviewed_count(self, instance_id: str) -> int:
        """Return how many groups are marked reviewed for ``instance_id``."""
        progress = self.modules.get(instance_id)
        return 0 if progress is None else len(progress.reviewed)

    def mark_reviewed(self, instance_id: str, key: GroupKey) -> bool:
        """Mark group ``key`` reviewed for the module and persist."""
        progress = self.progress_for(instance_id)
        encoded = encode_group_key(key)
        was_present = encoded in progress.reviewed
        progress.reviewed.add(encoded)
        if self.save():
            return True
        if not was_present:
            progress.reviewed.discard(encoded)
        return False

    def unmark_reviewed(self, instance_id: str, key: GroupKey) -> bool:
        """Drop group ``key`` from the module's reviewed set and persist."""
        progress = self.progress_for(instance_id)
        encoded = encode_group_key(key)
        was_present = encoded in progress.reviewed
        progress.reviewed.discard(encoded)
        if self.save():
            return True
        if was_present:
            progress.reviewed.add(encoded)
        return False

    def reconcile_to_summary(
        self, instance_id: str, present_keys: set[str]
    ) -> None:
        """Drop reviewed encoded keys whose group no longer exists.

        After a full rebuild the module's group keys may change (a settings
        edit altered ``groupby`` or thresholds, dropping or renaming groups).
        Prune any reviewed key not in ``present_keys`` so the worklist's
        reviewed counter + the verified-good set never reference a vanished
        group. Persists only when something changed.

        Args:
            instance_id: The recomputed module.
            present_keys: Encoded group keys present in the new summary.
        """
        progress = self.modules.get(instance_id)
        if progress is None:
            return
        stale = progress.reviewed - present_keys
        if stale:
            progress.reviewed -= stale
            if progress.last in stale:
                progress.last = None
            self.save()

    def set_last(self, instance_id: str, key: GroupKey | None) -> bool:
        """Record the last-visited group for the module and persist.

        Args:
            instance_id: The QC module.
            key: The group key just opened, or ``None`` to clear.
        """
        progress = self.progress_for(instance_id)
        previous = progress.last
        progress.last = None if key is None else encode_group_key(key)
        if self.save():
            return True
        progress.last = previous
        return False

    def to_payload(self) -> dict[str, object]:
        """Serialize the in-memory state to the on-disk JSON shape."""
        payload = dict(self.original_payload)
        for instance_id, progress in self.modules.items():
            existing = payload.get(instance_id)
            module_payload = dict(existing) if isinstance(existing, dict) else {}
            module_payload.update({
                "reviewed": sorted(progress.reviewed),
                "last": progress.last,
            })
            payload[instance_id] = module_payload
        return payload

    def _current_fingerprint(self) -> str:
        """Return the exact current generation without modifying the file."""
        from phenotypic.sdk_ import file_fingerprint

        try:
            return file_fingerprint(self.path)
        except FileNotFoundError:
            return _MISSING_FINGERPRINT

    def save(self) -> bool:
        """Atomically write the current state to :attr:`path`.

        The complete read/check/write transaction holds a sibling
        interprocess lock. A source fingerprint mismatch or an unreadable
        loaded source refuses the write and preserves external edits.

        Returns:
            ``True`` when persisted, otherwise ``False``.
        """
        from phenotypic.sdk_ import atomic_write_json, file_fingerprint
        from phenotypic.sdk_._file_locking import exclusive_path_lock

        if not self.source_readable:
            logger.warning(
                "Refusing to overwrite unreadable qc/review_state.json at %s",
                self.path,
            )
            return False
        try:
            lock_path = review_state_lock_path(self.path)
            with exclusive_path_lock(lock_path):
                current = self._current_fingerprint()
                if current != self.source_fingerprint:
                    raise ReviewStateConflictError(
                        "review_state.json changed after this session loaded it"
                    )
                atomic_write_json(self.path, self.to_payload(), sort_keys=False)
                self.source_fingerprint = file_fingerprint(self.path)
                self.original_payload = self.to_payload()
            return True
        except (OSError, ReviewStateConflictError):
            logger.warning(
                "Refused or failed to write qc/review_state.json at %s",
                self.path,
                exc_info=True,
            )
            return False


__all__ = [
    "GroupKey",
    "ModuleProgress",
    "ReviewState",
    "ReviewStateConflictError",
    "review_state_lock_path",
    "encode_group_key",
    "decode_group_key",
]
