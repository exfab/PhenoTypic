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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from phenotypic.schema import CURATION, OBJECT, ErrorCategory
from phenotypic.tools_ import (
    curation_labels_parquet_path,
    custom_categories_json_path,
    error_category_parquet_path,
    errors_dir,
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
        elif measurements_parquet_path(root).exists():
            labels, fingerprints = cls._migrate_legacy(root, master_df)
            report = RekeyReport(migrated=len(labels))

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

    # -- labels parquet IO + re-keying --------------------------------------
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
    def _master_index(
        master_df: pl.DataFrame,
    ) -> tuple[dict[LabelKey, tuple[float, float]], dict[str, list[tuple[int, float, float]]]]:
        """Build (exact-key -> centroid) and (image -> [(label, rr, cc)]) indexes."""
        exact: dict[LabelKey, tuple[float, float]] = {}
        per_image: dict[str, list[tuple[int, float, float]]] = {}
        has_fp = KEY_CENTER_RR in master_df.columns and KEY_CENTER_CC in master_df.columns
        cols = [KEY_IMAGE_FILE, KEY_OBJECT_LABEL]
        if has_fp:
            cols += [KEY_CENTER_RR, KEY_CENTER_CC]
        for row in master_df.select(cols).iter_rows(named=True):
            image_file = str(row[KEY_IMAGE_FILE])
            label = int(row[KEY_OBJECT_LABEL])
            rr = float(row[KEY_CENTER_RR]) if has_fp else float("nan")
            cc = float(row[KEY_CENTER_CC]) if has_fp else float("nan")
            exact[(image_file, label)] = (rr, cc)
            per_image.setdefault(image_file, []).append((label, rr, cc))
        return exact, per_image

    @staticmethod
    def _nearest_unique(
        candidates: list[tuple[int, float, float]], rr: float, cc: float, tol: float
    ) -> tuple[int, float, float] | None:
        """Return the single candidate within ``tol`` of (rr, cc), else None."""
        within = [
            (label, crr, ccc)
            for (label, crr, ccc) in candidates
            if ((crr - rr) ** 2 + (ccc - cc) ** 2) ** 0.5 <= tol
        ]
        return within[0] if len(within) == 1 else None

    @classmethod
    def _rekey(
        cls,
        stored: list[tuple[str, int, str, float, float]],
        master_df: pl.DataFrame,
        tol: float = FINGERPRINT_TOL_PX,
    ) -> tuple[dict[LabelKey, str], dict[LabelKey, tuple[float, float]], RekeyReport]:
        """Re-attach stored labels to the current master.

        Policy (resolved during plan review):

        * **No Bbox columns** (``Bbox_CenterRR/CC`` absent — e.g. a pipeline
          without ``MeasureBounds``): fingerprint validation is impossible, so
          *degrade gracefully* — keep every stored label whose exact
          ``(image_file, object_label)`` still exists in master, drop the rest,
          and log a single WARNING. Renumber-recovery is unavailable here.
        * **Bbox present**: if the exact key exists AND its centroid is within
          ``tol`` → keep. If the exact key exists but the centroid moved beyond
          ``tol`` → **drop immediately** (ambiguous identity; do NOT search
          neighbours, which could mis-attach to an adjacent colony). If the exact
          key is absent → re-key only when exactly one object in the same image
          is within ``tol`` of the stored centroid, else drop.
        """
        has_fp = (
            KEY_CENTER_RR in master_df.columns and KEY_CENTER_CC in master_df.columns
        )
        exact, per_image = cls._master_index(master_df)
        labels: dict[LabelKey, str] = {}
        fingerprints: dict[LabelKey, tuple[float, float]] = {}
        kept = rekeyed = dropped = 0

        if not has_fp:
            logger.warning(
                "Master frame lacks %s/%s; fingerprint re-keying disabled "
                "(exact-key only). Add MeasureBounds to enable renumber recovery.",
                KEY_CENTER_RR,
                KEY_CENTER_CC,
            )
            for image_file, label, category, _rr, _cc in stored:
                key = (image_file, label)
                if key in exact:
                    labels[key] = category  # no fingerprint available to store
                    kept += 1
                else:
                    dropped += 1
            return labels, fingerprints, RekeyReport(kept=kept, dropped=dropped)

        for image_file, label, category, rr, cc in stored:
            key = (image_file, label)
            mfp = exact.get(key)
            if mfp is not None:
                # Exact key survived: trust it only while the centroid is stable.
                if ((mfp[0] - rr) ** 2 + (mfp[1] - cc) ** 2) ** 0.5 <= tol:
                    labels[key] = category
                    fingerprints[key] = mfp
                    kept += 1
                else:
                    dropped += 1  # moved too far — drop, never risk a neighbour
                continue
            # Exact key gone (renumbered?): recover only on a unique fingerprint.
            match = cls._nearest_unique(per_image.get(image_file, []), rr, cc, tol)
            if match is not None:
                new_label, mrr, mcc = match
                nkey = (image_file, new_label)
                labels[nkey] = category
                fingerprints[nkey] = (mrr, mcc)
                rekeyed += 1
            else:
                dropped += 1
        return labels, fingerprints, RekeyReport(kept=kept, rekeyed=rekeyed, dropped=dropped)

    @classmethod
    def _migrate_legacy(
        cls, root: Path, master_df: pl.DataFrame
    ) -> tuple[dict[LabelKey, str], dict[LabelKey, tuple[float, float]]]:
        """Import a legacy ``measurements.parquet`` mirror as ``other`` labels.

        Removed objects are ``master_keys - curated_keys``; each is labeled
        ``other`` with its fingerprint taken from the master.
        """
        curated = pl.read_parquet(measurements_parquet_path(root))
        exact, _ = cls._master_index(master_df)
        curated_keys = {
            (str(img), int(lbl))
            for img, lbl in zip(
                curated.get_column(KEY_IMAGE_FILE).to_list(),
                curated.get_column(KEY_OBJECT_LABEL).to_list(),
            )
        }
        labels: dict[LabelKey, str] = {}
        fingerprints: dict[LabelKey, tuple[float, float]] = {}
        for key, fp in exact.items():
            if key not in curated_keys:
                labels[key] = OTHER_CATEGORY
                fingerprints[key] = fp
        return labels, fingerprints

    # -- queries -------------------------------------------------------------
    def filtered_df(self, master_df: pl.DataFrame) -> pl.DataFrame:
        """Return ``master_df`` with all labeled (removed) rows dropped."""
        if not self.labels:
            return master_df
        removed = pl.DataFrame(
            {
                KEY_COLUMNS[0]: [k[0] for k in self.labels],
                KEY_COLUMNS[1]: [k[1] for k in self.labels],
            },
            schema={KEY_COLUMNS[0]: pl.String, KEY_COLUMNS[1]: pl.Int64},
        )
        keyed = master_df.with_columns(
            pl.col(KEY_COLUMNS[0]).cast(pl.String),
            pl.col(KEY_COLUMNS[1]).cast(pl.Int64),
        )
        return keyed.join(removed, on=list(KEY_COLUMNS), how="anti")

    def _fingerprint_of(self, image_file: str, label: int) -> tuple[float, float] | None:
        """Look up an object's centroid in the cached master, or ``None``."""
        if KEY_CENTER_RR not in self._master_df.columns:
            return None
        row = (
            self._master_df.filter(
                (pl.col(KEY_IMAGE_FILE).cast(pl.String) == image_file)
                & (pl.col(KEY_OBJECT_LABEL).cast(pl.Int64) == label)
            )
            .select(KEY_CENTER_RR, KEY_CENTER_CC)
            .head(1)
        )
        if row.is_empty():
            return None
        return (float(row.get_column(KEY_CENTER_RR)[0]), float(row.get_column(KEY_CENTER_CC)[0]))

    # -- mutators ------------------------------------------------------------
    def mark(self, image_file: str, label: int, category: str) -> None:
        """Assign ``category`` to one object and persist all derived outputs.

        Raises:
            ValueError: If ``category`` is neither a core nor registered token.
        """
        if not self.is_valid_category(category):
            raise ValueError(f"Unknown category {category!r}.")
        key = (image_file, label)
        with self._lock:
            self.labels[key] = category
            fp = self._fingerprint_of(image_file, label)
            if fp is not None:
                self.fingerprints[key] = fp
            self._save_locked()

    def unmark(self, image_file: str, label: int) -> None:
        """Remove any label for one object and persist."""
        key = (image_file, label)
        with self._lock:
            if key not in self.labels:
                return
            self.labels.pop(key, None)
            self.fingerprints.pop(key, None)
            self._save_locked()

    def mark_many(self, keys: Iterable[LabelKey], category: str) -> None:
        """Assign ``category`` to a batch in one save."""
        if not self.is_valid_category(category):
            raise ValueError(f"Unknown category {category!r}.")
        with self._lock:
            changed = False
            for image_file, label in keys:
                key = (image_file, label)
                self.labels[key] = category
                fp = self._fingerprint_of(image_file, label)
                if fp is not None:
                    self.fingerprints[key] = fp
                changed = True
            if changed:
                self._save_locked()

    def unmark_many(self, keys: Iterable[LabelKey]) -> None:
        """Remove labels for a batch in one save."""
        with self._lock:
            removed = False
            for key in keys:
                if key in self.labels:
                    self.labels.pop(key, None)
                    self.fingerprints.pop(key, None)
                    removed = True
            if removed:
                self._save_locked()

    # -- persistence ---------------------------------------------------------
    def save(self) -> None:
        """Persist all derived outputs under the lock (public entry)."""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Write labels parquet + curated mirror + per-category files (lock held)."""
        self._write_labels_parquet()
        self._write_curated_mirror()
        self._write_category_parquets()

    def _write_labels_parquet(self) -> None:
        path = self.labels_path
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = {
            KEY_IMAGE_FILE: [k[0] for k in self.labels],
            KEY_OBJECT_LABEL: [k[1] for k in self.labels],
            KEY_CATEGORY: [self.labels[k] for k in self.labels],
            KEY_CENTER_RR: [self.fingerprints.get(k, (float("nan"), float("nan")))[0] for k in self.labels],
            KEY_CENTER_CC: [self.fingerprints.get(k, (float("nan"), float("nan")))[1] for k in self.labels],
        }
        df = pl.DataFrame(
            rows,
            schema={
                KEY_IMAGE_FILE: pl.String,
                KEY_OBJECT_LABEL: pl.Int64,
                KEY_CATEGORY: pl.String,
                KEY_CENTER_RR: pl.Float64,
                KEY_CENTER_CC: pl.Float64,
            },
        )
        _atomic_write_parquet(df, path)

    def _write_curated_mirror(self) -> None:
        curated = self.filtered_df(self._master_df)
        self.measurements_parquet.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(curated, self.measurements_parquet)
        try:
            _atomic_write_csv(curated, self.measurements_csv)
        except Exception:
            logger.exception("Failed to write curated CSV mirror at %s", self.measurements_csv)

    def _write_category_parquets(self) -> None:
        errs_dir = errors_dir(self.root)
        errs_dir.mkdir(parents=True, exist_ok=True)
        # Group keys by category.
        by_cat: dict[str, list[LabelKey]] = {}
        for key, cat in self.labels.items():
            by_cat.setdefault(cat, []).append(key)
        # Write present categories; remove files for categories no longer present.
        present = set()
        for cat, keys in by_cat.items():
            token = sanitize_category(cat)
            if not token:
                logger.warning("Skipping category with no filename-safe token: %r", cat)
                continue
            present.add(token)
            sub = pl.DataFrame(
                {
                    KEY_COLUMNS[0]: [k[0] for k in keys],
                    KEY_COLUMNS[1]: [k[1] for k in keys],
                },
                schema={KEY_COLUMNS[0]: pl.String, KEY_COLUMNS[1]: pl.Int64},
            )
            keyed = self._master_df.with_columns(
                pl.col(KEY_COLUMNS[0]).cast(pl.String),
                pl.col(KEY_COLUMNS[1]).cast(pl.Int64),
            )
            frame = keyed.join(sub, on=list(KEY_COLUMNS), how="semi").with_columns(
                pl.lit(cat).alias(KEY_CATEGORY)
            )
            _atomic_write_parquet(frame, error_category_parquet_path(self.root, token))
        # Clean up stale category files.
        for existing in errs_dir.glob("*.parquet"):
            if existing.stem not in present:
                try:
                    existing.unlink()
                except OSError:
                    logger.warning("Could not remove stale category file %s", existing)


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a parquet via a sibling temp file + ``os.replace`` (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _atomic_write_csv(df: pl.DataFrame, path: Path) -> None:
    """Write a CSV via a sibling temp file + ``os.replace`` (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_csv(tmp)
    os.replace(tmp, path)
