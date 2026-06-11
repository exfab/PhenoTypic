"""Durable, categorized curation-labels store for the results viewer.

Generalizes the remove-only ``FilteredMeasurements``: every removed object
carries an :class:`ErrorCategory` bare label (or a registered custom token),
and the assignment set is the single source of truth from which the curated
``deliverables/measurements.parquet`` mirror and the per-category
``deliverables/errors/<category>.parquet`` files are derived.

The labels parquet at ``<root>/qc/curation_labels.parquet`` is **never wiped by
the CLI**. On load it is re-keyed onto the current master frame using each
object's centroid fingerprint (Task 5), so a re-detection that renumbers
``Object_Label`` re-attaches labels correctly or drops ambiguous ones; the
kept/re-keyed/dropped tallies feed the viewer's stale banner.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from phenotypic.schema import CURATION, OBJECT, ErrorCategory
from phenotypic.tools_ import (
    curation_labels_parquet_path,
    custom_categories_json_path,
    measurements_csv_path,
    measurements_parquet_path,
)

logger = logging.getLogger(__name__)

KEY_IMAGE_FILE: str = "Metadata_ImageFile"
KEY_OBJECT_LABEL: str = str(OBJECT.LABEL)
KEY_DATASET: str = "Metadata_Dataset"
KEY_CATEGORY: str = str(CURATION.ERROR_CATEGORY)  # "Curation_Category"
KEY_CENTER_RR: str = "Bbox_CenterRR"
KEY_CENTER_CC: str = "Bbox_CenterCC"
KEY_COLUMNS: tuple[str, str] = (KEY_IMAGE_FILE, KEY_OBJECT_LABEL)

#: The reserved reasonless category (= today's plain "remove").
OTHER_CATEGORY: str = ErrorCategory.OTHER.label

#: Max centroid drift (px, Euclidean) tolerated when validating/re-keying.
FINGERPRINT_TOL_PX: float = 2.0

_UNSAFE_CHARS = re.compile(r"[^a-z0-9._-]+")

#: (image_file, object_label) curation key.
LabelKey = tuple[str, int]


def sanitize_category(name: str) -> str:
    """Coerce a free-text category name to a filename-safe bare token.

    Lowercases, replaces any run of non ``[a-z0-9._-]`` characters with a
    single underscore, and strips leading/trailing separators. Returns ``""``
    for input that has no usable characters (the caller must reject empties).

    Args:
        name: User-entered category name.

    Returns:
        A sanitized token, or ``""`` if nothing usable remained.
    """
    cleaned = _UNSAFE_CHARS.sub("_", name.strip().lower())
    return cleaned.strip("._-")


@dataclass(frozen=True)
class RekeyReport:
    """Tally of how stored labels re-attached to the current master frame.

    Attributes:
        kept: Labels whose exact key matched and passed fingerprint validation.
        rekeyed: Labels re-attached to a renumbered object by fingerprint.
        dropped: Labels with no confident match in the current master (dropped).
        migrated: Legacy removals inferred from a pre-existing
            ``measurements.parquet`` (no prior labels store) and imported as
            ``other`` — counted separately from ``kept`` so the stale banner is
            accurate.
    """

    kept: int = 0
    rekeyed: int = 0
    dropped: int = 0
    migrated: int = 0

    @property
    def total(self) -> int:
        return self.kept + self.rekeyed + self.dropped + self.migrated


@dataclass
class CurationLabels:
    """In-memory categorized curation state plus its durable on-disk mirrors.

    Attributes:
        root: Output root directory.
        labels: Mapping ``(image_file, object_label) -> category token``.
        fingerprints: Mapping key -> ``(center_rr, center_cc)`` captured at mark
            time, used to re-key across re-detections.
        custom_categories: Ordered list of registered custom category tokens.
        rekey_report: Result of the most recent load's re-keying pass.
        _master_df: Master frame captured at load (all objects + measurements).
        _lock: Re-entrant mutation/save mutex.
    """

    root: Path
    labels: dict[LabelKey, str]
    fingerprints: dict[LabelKey, tuple[float, float]]
    custom_categories: list[str]
    rekey_report: RekeyReport
    _master_df: pl.DataFrame = field(repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # -- paths ---------------------------------------------------------------
    @property
    def labels_path(self) -> Path:
        return curation_labels_parquet_path(self.root)

    @property
    def custom_path(self) -> Path:
        return custom_categories_json_path(self.root)

    @property
    def measurements_parquet(self) -> Path:
        return measurements_parquet_path(self.root)

    @property
    def measurements_csv(self) -> Path:
        return measurements_csv_path(self.root)

    # -- vocabulary ----------------------------------------------------------
    def categories(self) -> list[str]:
        """Return all known category tokens: core enum labels then custom."""
        return [*ErrorCategory.labels(), *self.custom_categories]

    def is_valid_category(self, category: str) -> bool:
        """Return whether ``category`` is a core or registered custom token."""
        return category in set(self.categories())

    def register_custom_category(self, name: str) -> str:
        """Register (idempotently) a custom category and persist the registry.

        Args:
            name: Free-text category name (sanitized to a bare token).

        Returns:
            The sanitized token.

        Raises:
            ValueError: If the name sanitizes to empty or collides with a core
                ``ErrorCategory`` token.
        """
        token = sanitize_category(name)
        if not token:
            raise ValueError(f"Category name {name!r} sanitizes to empty.")
        if token in set(ErrorCategory.labels()):
            raise ValueError(f"{token!r} collides with a core category.")
        with self._lock:
            if token not in self.custom_categories:
                self.custom_categories.append(token)
                self._save_custom_registry()
        return token

    # -- load ----------------------------------------------------------------
    @classmethod
    def load(cls, root: Path, master_df: pl.DataFrame) -> "CurationLabels":
        """Build the store from disk, re-keyed onto ``master_df``.

        Reads the custom-category registry and (Task 5) the labels parquet,
        re-attaching each stored label to the current master via fingerprint.
        A missing labels parquet yields an empty label set (migration from a
        legacy ``measurements.parquet`` is added in Task 5).

        Args:
            root: Output root directory.
            master_df: Full master measurements frame (all objects).

        Returns:
            A ready-to-mutate :class:`CurationLabels`.
        """
        custom = cls._read_custom_registry(custom_categories_json_path(root))
        labels: dict[LabelKey, str] = {}
        fingerprints: dict[LabelKey, tuple[float, float]] = {}
        report = RekeyReport()

        labels_path = curation_labels_parquet_path(root)
        if labels_path.exists():
            stored = cls._read_labels_parquet(labels_path)
            labels, fingerprints, report = cls._rekey(stored, master_df)

        return cls(
            root=root,
            labels=labels,
            fingerprints=fingerprints,
            custom_categories=custom,
            rekey_report=report,
            _master_df=master_df,
        )

    # -- registry IO ---------------------------------------------------------
    @staticmethod
    def _read_custom_registry(path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read custom-category registry at %s", path)
            return []
        cats = payload.get("categories") if isinstance(payload, dict) else None
        if not isinstance(cats, list):
            return []
        out: list[str] = []
        for c in cats:
            token = sanitize_category(str(c))
            if token and token not in out and token not in set(ErrorCategory.labels()):
                out.append(token)
        return out

    def _save_custom_registry(self) -> None:
        """Atomically persist the custom-category registry (lock held)."""
        self.custom_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"categories": self.custom_categories}, indent=2)
        tmp = self.custom_path.with_suffix(self.custom_path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.custom_path)

    # -- labels parquet IO (re-keying added in Task 5) -----------------------
    @staticmethod
    def _read_labels_parquet(path: Path) -> list[tuple[str, int, str, float, float]]:
        """Read the labels parquet into raw tuples (no re-keying)."""
        df = pl.read_parquet(path)
        rows: list[tuple[str, int, str, float, float]] = []
        for row in df.iter_rows(named=True):
            rows.append(
                (
                    str(row[KEY_IMAGE_FILE]),
                    int(row[KEY_OBJECT_LABEL]),
                    str(row[KEY_CATEGORY]),
                    float(row[KEY_CENTER_RR]),
                    float(row[KEY_CENTER_CC]),
                )
            )
        return rows

    @staticmethod
    def _rekey(
        stored: list[tuple[str, int, str, float, float]],
        master_df: pl.DataFrame,
    ) -> tuple[dict[LabelKey, str], dict[LabelKey, tuple[float, float]], RekeyReport]:
        """Exact-key re-key (Task 3 stub).

        Task 5 replaces this with fingerprint validation + renumber recovery.
        For now: keep a stored label iff its exact key exists in master.
        """
        master_keys = {
            (str(img), int(lbl))
            for img, lbl in zip(
                master_df.get_column(KEY_IMAGE_FILE).to_list(),
                master_df.get_column(KEY_OBJECT_LABEL).to_list(),
            )
        }
        labels: dict[LabelKey, str] = {}
        fingerprints: dict[LabelKey, tuple[float, float]] = {}
        kept = dropped = 0
        for image_file, label, category, rr, cc in stored:
            key = (image_file, label)
            if key in master_keys:
                labels[key] = category
                fingerprints[key] = (rr, cc)
                kept += 1
            else:
                dropped += 1
        return labels, fingerprints, RekeyReport(kept=kept, dropped=dropped)
