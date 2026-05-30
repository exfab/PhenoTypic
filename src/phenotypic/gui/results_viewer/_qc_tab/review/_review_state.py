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

* **Written only by the GUI** — :func:`phenotypic.qc._runner.run_qc`
  never touches this file, so an in-session recompute preserves review
  progress.
* **Reset by the CLI** — ``finalize_post_master_outputs`` clears it on
  every recompile/remeasure, so a fresh run starts the queue over.

A group key is a tuple of the module's ``groupby`` values (e.g.
``("plate1", "A")``). JSON has no tuples, so keys are encoded as a
JSON-string of the value list (``'["plate1", "A"]'``) on the way out and
decoded back to a tuple on the way in — this keeps multi-column group
keys round-trippable through the on-disk store.

The store is written atomically (temp file + ``os.replace``) so a crash
mid-write never leaves a half-JSON file. Concurrency is intentionally
light: ``review_state.json`` is single-viewer-session state, not a
CLI↔GUI handoff like ``pipeline.json``, so there is no mtime guard.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from phenotypic.tools_ import qc_review_state_path

logger = logging.getLogger(__name__)

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

    @classmethod
    def load(cls, output_root_path: Path) -> "ReviewState":
        """Load review progress from ``<output_root_path>/qc/review_state.json``.

        A missing or corrupt file yields an empty state (the file is
        created lazily on the first :meth:`save`). Corruption is logged at
        WARNING and the on-disk file is left untouched so it can be
        recovered by hand.

        Args:
            output_root_path: The results-viewer output root.

        Returns:
            A :class:`ReviewState` ready for in-place mutation.
        """
        path = qc_review_state_path(Path(output_root_path))
        if not path.exists():
            return cls(path=path)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "qc/review_state.json at %s could not be read; starting with "
                "empty review progress. On-disk file left untouched. Error: %s",
                path,
                exc,
            )
            return cls(path=path)

        modules: dict[str, ModuleProgress] = {}
        if isinstance(payload, dict):
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
        return cls(path=path, modules=modules)

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

    def mark_reviewed(self, instance_id: str, key: GroupKey) -> None:
        """Mark group ``key`` reviewed for the module and persist."""
        progress = self.progress_for(instance_id)
        progress.reviewed.add(encode_group_key(key))
        self.save()

    def unmark_reviewed(self, instance_id: str, key: GroupKey) -> None:
        """Drop group ``key`` from the module's reviewed set and persist."""
        progress = self.progress_for(instance_id)
        progress.reviewed.discard(encode_group_key(key))
        self.save()

    def set_last(self, instance_id: str, key: GroupKey | None) -> None:
        """Record the last-visited group for the module and persist.

        Args:
            instance_id: The QC module.
            key: The group key just opened, or ``None`` to clear.
        """
        progress = self.progress_for(instance_id)
        progress.last = None if key is None else encode_group_key(key)
        self.save()

    def to_payload(self) -> dict[str, dict[str, object]]:
        """Serialize the in-memory state to the on-disk JSON shape."""
        return {
            instance_id: {
                "reviewed": sorted(progress.reviewed),
                "last": progress.last,
            }
            for instance_id, progress in self.modules.items()
        }

    def save(self) -> None:
        """Atomically write the current state to :attr:`path`.

        Writes to a sibling temp file then ``os.replace``-s it into place
        (atomic on every supported platform). Failures are logged at
        WARNING and swallowed — losing a review-progress write is a
        recoverable annoyance, never a reason to crash a curation
        callback.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_payload(), indent=2)
        tmp_path: str | None = None
        try:
            handle = tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=f".{self.path.stem}_",
                suffix=".tmp",
                delete=False,
            )
            tmp_path = handle.name
            handle.close()
            Path(tmp_path).write_text(payload, encoding="utf-8")
            os.replace(tmp_path, self.path)
        except OSError:
            logger.warning(
                "Failed to write qc/review_state.json at %s", self.path,
                exc_info=True,
            )
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


__all__ = [
    "GroupKey",
    "ModuleProgress",
    "ReviewState",
    "encode_group_key",
    "decode_group_key",
]
