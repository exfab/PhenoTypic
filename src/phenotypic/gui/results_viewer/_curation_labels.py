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

import collections
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

from phenotypic.schema import CURATION, METADATA, OBJECT, ErrorCategory
from phenotypic.sdk_ import BundleLayout, paths_fingerprint
from phenotypic.sdk_._file_locking import ArtifactLockTimeout, exclusive_path_lock

logger = logging.getLogger(__name__)

KEY_IMAGE_FILE: str = str(METADATA.IMAGE_NAME)
KEY_OBJECT_LABEL: str = str(OBJECT.LABEL)
KEY_CATEGORY: str = str(CURATION.ERROR_CATEGORY)  # "Curation_Category"
KEY_CENTER_RR: str = "Bbox_CenterRR"
KEY_CENTER_CC: str = "Bbox_CenterCC"
KEY_COLUMNS: tuple[str, str] = (KEY_IMAGE_FILE, KEY_OBJECT_LABEL)

#: The reserved reasonless category (= today's plain "remove").
OTHER_CATEGORY: str = ErrorCategory.OTHER.label

#: Max centroid drift (px, Euclidean) tolerated when validating/re-keying.
FINGERPRINT_TOL_PX: float = 2.0

#: Sentinel NaN fingerprint used when no centroid is available.
_NAN_FP: tuple[float, float] = (float("nan"), float("nan"))

_UNSAFE_CHARS = re.compile(r"[^a-z0-9._-]+")

#: (image_file, object_label) curation key.
LabelKey = tuple[str, int]

#: Retired ad-hoc image-stem column that older ``curation_labels.parquet``
#: files were keyed on before the consolidation onto the canonical
#: ``Metadata_ImageName``. Kept only so :func:`_migrate_legacy_imagefile`
#: can recognize and rename it on load.
_LEGACY_IMAGE_FILE = "Metadata_ImageFile"


def _migrate_legacy_imagefile(df: pl.DataFrame) -> pl.DataFrame:
    """Rename a legacy image-stem column to the canonical image-name column.

    Old ``curation_labels.parquet`` files were keyed on the retired ad-hoc
    column tracked by :data:`_LEGACY_IMAGE_FILE`. When such a frame is read,
    rename that column to :data:`~phenotypic.schema.METADATA.IMAGE_NAME` so the
    durable curation state survives the consolidation. A frame that already
    carries the canonical column (or lacks the legacy one) is returned
    unchanged.

    Args:
        df: The frame just read from ``curation_labels.parquet``.

    Returns:
        The frame with the legacy column renamed when present, else *df*.
    """
    canonical = str(METADATA.IMAGE_NAME)
    if _LEGACY_IMAGE_FILE in df.columns and canonical not in df.columns:
        return df.rename({_LEGACY_IMAGE_FILE: canonical})
    return df


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


def _within_tol(rr0: float, cc0: float, rr1: float, cc1: float, tol: float) -> bool:
    """Whether two centroids are within ``tol`` px (Euclidean)."""
    return ((rr0 - rr1) ** 2 + (cc0 - cc1) ** 2) ** 0.5 <= tol


def _key_frame(keys: Iterable[LabelKey]) -> pl.DataFrame:
    """Build the synthetic 2-column (String, Int64) join frame from keys."""
    key_list = list(keys)
    return pl.DataFrame(
        {KEY_COLUMNS[0]: [k[0] for k in key_list], KEY_COLUMNS[1]: [k[1] for k in key_list]},
        schema={KEY_COLUMNS[0]: pl.String, KEY_COLUMNS[1]: pl.Int64},
    )


def _join_on_keys(
    master_df: pl.DataFrame,
    keys: Iterable[LabelKey],
    how: Literal["anti", "semi"],
) -> pl.DataFrame:
    """Cast master key columns to (String, Int64) and {anti,semi}-join against the key frame."""
    keyed = master_df.with_columns(
        pl.col(KEY_COLUMNS[0]).cast(pl.String),
        pl.col(KEY_COLUMNS[1]).cast(pl.Int64),
    )
    return keyed.join(_key_frame(keys), on=list(KEY_COLUMNS), how=how)


def _keys_of(df: pl.DataFrame) -> set[tuple[str, int]]:
    """Extract the (image_file, object_label) key set from a frame.

    Rows whose ``Object_Label`` is null are skipped: the post-applied
    ``measurements.parquet`` mirror is built with a **left** join against the
    ``--metadata`` table, so it carries "phantom" rows — metadata for strains
    that were never detected, with a null label and null measurements. A
    phantom is not a curatable object (there is nothing on the plate to mark),
    so it can never contribute a curation key, and ``int(None)`` would raise.
    The phantom rows themselves are deliberately left in the mirror — the
    viewer must show the user which strains went undetected.

    Args:
        df: Any frame exposing both key columns (typically a filtered view of
            the mirror).

    Returns:
        The ``(image_file, object_label)`` key set of the real (detected) rows.
    """
    keyed = df.filter(pl.col(KEY_COLUMNS[1]).is_not_null())
    return {
        (str(f), int(lbl))
        for f, lbl in zip(
            keyed.get_column(KEY_COLUMNS[0]).to_list(),
            keyed.get_column(KEY_COLUMNS[1]).to_list(),
        )
    }


@dataclass(frozen=True)
class RekeyReport:
    """Tally of how stored labels re-attached to the current master frame.

    Attributes:
        kept: Labels whose exact key matched and passed fingerprint validation.
        rekeyed: Labels re-attached to a renumbered object by fingerprint.
        dropped: Labels with no confident match in the current master (dropped).
        migrated: Reserved (always ``0``). Legacy auto-migration of a curated
            ``measurements.parquet`` into ``other`` labels was removed once
            re-keying moved to the clean master: the mirror's missing rows can
            no longer be distinguished from post-op / ``--metadata`` row drops,
            so importing them would fabricate spurious removals.
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
        _layout: Resolved :class:`~phenotypic.sdk_.BundleLayout` topology;
            all durable paths (labels parquet, custom-category registry,
            curated mirror, per-category error parquets) resolve from it, so
            a standalone deliverables bundle writes inside the bundle.
        labels: Mapping ``(image_file, object_label) -> category token``.
        fingerprints: Mapping key -> ``(center_rr, center_cc)`` captured at mark
            time, used to re-key across re-detections.
        custom_categories: Ordered list of registered custom category tokens.
        rekey_report: Result of the most recent load's re-keying pass.
        _master_df: Master frame captured at load (all objects + measurements).
        _lock: Re-entrant mutation/save mutex.
        _seed_mtime_ns: Nanosecond mtime of ``measurements.parquet`` as last
            observed by this instance.  ``None`` means the mirror has never
            existed from this instance's perspective.  Used to detect external
            rewrites (CLI measure / recompile mode against a directory
            whose viewer session is still open) so we don't clobber a freshly
            seeded master with stale curation derived from an older
            ``_master_df``.
        _expected_source_fingerprint: Combined content fingerprint of every
            curation source and derived artifact observed at load or after this
            instance's last successful publication. It protects labels,
            custom categories, mirrors, and category partitions from
            concurrent or external writers.
        _stale: Set to ``True`` once a write refusal has fired.  Exposed as
            the read-only :attr:`stale` property; external callers may not
            clear it.
    """

    _layout: BundleLayout
    labels: dict[LabelKey, str]
    fingerprints: dict[LabelKey, tuple[float, float]]
    custom_categories: list[str]
    rekey_report: RekeyReport
    #: The CLEAN (pre-post) master — all objects, including those a prior
    #: curation removed from the mirror. Re-keying, fingerprinting, and the
    #: per-category error partitions all read this so labels survive a viewer
    #: reload (the curated mirror no longer contains the labeled objects).
    _master_df: pl.DataFrame = field(repr=False)
    #: The post-applied measurements mirror, written back (minus labeled rows)
    #: as ``measurements.parquet`` so the viewer's filter sidebar keeps its
    #: post columns. ``None`` (legacy/direct construction) falls back to
    #: ``_master_df`` for the curated-mirror write.
    _mirror_df: pl.DataFrame | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _seed_mtime_ns: int | None = field(default=None, repr=False)
    _expected_source_fingerprint: str = field(default="", repr=False)
    _stale: bool = field(default=False, repr=False)

    @property
    def stale(self) -> bool:
        """Whether an external mtime change blocked a write in this session."""
        return self._stale

    # -- paths ---------------------------------------------------------------
    @property
    def labels_path(self) -> Path:
        return self._layout.curation_labels_parquet

    @property
    def custom_path(self) -> Path:
        return self._layout.custom_categories_json

    @property
    def measurements_parquet(self) -> Path:
        return self._layout.mirror_parquet

    @property
    def measurements_csv(self) -> Path:
        return self._layout.mirror_csv

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
                if not self._save_custom_registry():
                    self.custom_categories.remove(token)
        return token

    # -- load ----------------------------------------------------------------
    @classmethod
    def load(cls, layout: BundleLayout, master_df: pl.DataFrame) -> "CurationLabels":
        """Build the store from disk, re-keyed onto ``master_df``.

        Reads the custom-category registry and the labels parquet,
        re-attaching each stored label to the current master via fingerprint.
        A missing labels parquet yields an empty label set.

        Args:
            layout: Resolved bundle topology. Durable paths (labels parquet,
                custom registry, curated mirror, per-category error parquets)
                all resolve from it, so a standalone deliverables bundle
                curates inside the bundle. The GUI passes
                ``output_root.layout``; the CLI passes
                ``BundleLayout.detect(output_dir)``.
            master_df: The viewer's display frame — the **post-applied
                measurements mirror** (``OutputRoot.master_df``). Re-keying is
                done against the CLEAN ``master_measurements.parquet`` read here
                (it still contains objects the curated mirror dropped, so labels
                survive a reload); ``master_df`` is retained only as the frame
                the curated mirror is written back from.

        Returns:
            A ready-to-mutate :class:`CurationLabels`.
        """
        custom = cls._read_custom_registry(layout.custom_categories_json)
        labels: dict[LabelKey, str] = {}
        fingerprints: dict[LabelKey, tuple[float, float]] = {}
        report = RekeyReport()

        # Re-key against the CLEAN master (all objects), never the curated
        # mirror — a labeled object is absent from the mirror, so re-keying
        # against the mirror would drop every label on reload.
        clean_master = cls._read_clean_master(layout, master_df)

        labels_path = layout.curation_labels_parquet
        if labels_path.exists():
            stored = cls._read_labels_parquet(labels_path)
            labels, fingerprints, report = cls._rekey(stored, clean_master)
        # No legacy migration from ``measurements.parquet``: now that re-keying
        # uses the CLEAN master, the rows the curated mirror is missing are
        # ambiguous — they may be old curation removals OR objects a post op
        # (e.g. outlier removal) or an external ``--metadata`` inner-join
        # legitimately dropped from the post-applied mirror. Importing the
        # latter as ``other`` would silently fabricate error labels and delete
        # valid objects on first GUI load, so we start empty (matching the CLI
        # finalize decision). A genuine durable store re-keys above.

        # Capture the mtime of the curated mirror so _save_locked can detect
        # an external re-seed (e.g. CLI measure mode) while the session is open.
        mirror = layout.mirror_parquet
        seed_mtime: int | None = mirror.stat().st_mtime_ns if mirror.exists() else None

        return cls(
            _layout=layout,
            labels=labels,
            fingerprints=fingerprints,
            custom_categories=custom,
            rekey_report=report,
            _master_df=clean_master,
            _mirror_df=master_df,
            _seed_mtime_ns=seed_mtime,
            _expected_source_fingerprint=cls._source_fingerprint(layout),
        )

    @staticmethod
    def _read_clean_master(layout: BundleLayout, fallback: pl.DataFrame) -> pl.DataFrame:
        """Return the clean (pre-post) master frame for re-keying.

        Reads ``deliverables/master_measurements.parquet`` — the full,
        archival object set that the curated ``measurements.parquet`` mirror
        is derived from (and which it removes labeled rows from). Falls back to
        ``fallback`` (the passed mirror) only mid-run / on legacy outputs where
        the clean master is absent.
        """
        path = layout.master_parquet
        if path.is_file():
            try:
                return pl.read_parquet(path)
            except Exception:  # noqa: BLE001 - a corrupt master is non-fatal here
                logger.warning(
                    "Could not read clean master at %s; re-keying against the "
                    "mirror (labels for curated-out objects may be dropped).",
                    path,
                )
        return fallback

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

    def _write_custom_registry(self) -> None:
        """Atomically persist the custom-category registry (locks held)."""
        self.custom_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"categories": self.custom_categories}, indent=2)
        tmp = self.custom_path.with_suffix(self.custom_path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.custom_path)

    def _save_custom_registry(self) -> bool:
        """Persist the custom-category registry under the shared CAS lock."""
        return self._publish_if_current(
            self._write_custom_registry,
            context="custom-category registry",
        )

    # -- labels parquet IO + re-keying --------------------------------------
    @staticmethod
    def _read_labels_parquet(path: Path) -> list[tuple[str, int, str, float, float]]:
        """Read the labels parquet into raw tuples (no re-keying)."""
        df = pl.read_parquet(path)
        # Rename-on-load: legacy parquets keyed on the retired ad-hoc column
        # must be migrated to the canonical key before any lookup/join.
        df = _migrate_legacy_imagefile(df)
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
        """Build (exact-key -> centroid) and (image -> [(label, rr, cc)]) indexes.

        Rows with a null ``Object_Label`` are skipped. ``_read_clean_master``
        normally supplies the CLEAN master (which has no such rows), but it
        falls back to the post-applied mirror when the master parquet is
        missing or corrupt — and the mirror's ``--metadata`` left join carries
        phantom rows for undetected strains, whose null label is not an
        indexable object identity (and would raise on ``int(None)``).
        """
        exact: dict[LabelKey, tuple[float, float]] = {}
        per_image: dict[str, list[tuple[int, float, float]]] = {}
        has_fp = KEY_CENTER_RR in master_df.columns and KEY_CENTER_CC in master_df.columns
        cols = [KEY_IMAGE_FILE, KEY_OBJECT_LABEL]
        if has_fp:
            cols += [KEY_CENTER_RR, KEY_CENTER_CC]
        indexable = master_df.filter(pl.col(KEY_OBJECT_LABEL).is_not_null())
        for row in indexable.select(cols).iter_rows(named=True):
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
            if _within_tol(crr, ccc, rr, cc, tol)
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
        * **Two-pass conflict resolution**: when two stored labels both
          fingerprint-match the same surviving object, both are dropped (last-wins
          is not correct behaviour; the collision signals ambiguity).
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

        # PASS 1: classify each stored entry as a direct drop or a candidate.
        # candidate: (source_stored_key, target_key, category, fingerprint, kind)
        # kind: "kept" | "rekeyed"
        direct_drops = 0
        candidates: list[tuple[LabelKey, LabelKey, str, tuple[float, float], str]] = []

        for image_file, label, category, rr, cc in stored:
            key = (image_file, label)
            mfp = exact.get(key)
            if mfp is not None:
                # Exact key survived: trust it only while the centroid is stable.
                if _within_tol(mfp[0], mfp[1], rr, cc, tol):
                    candidates.append((key, key, category, mfp, "kept"))
                else:
                    direct_drops += 1  # moved too far — drop, never risk a neighbour
                continue
            # Exact key gone (renumbered?): recover only on a unique fingerprint.
            match = cls._nearest_unique(per_image.get(image_file, []), rr, cc, tol)
            if match is not None:
                new_label, mrr, mcc = match
                nkey = (image_file, new_label)
                candidates.append((key, nkey, category, (mrr, mcc), "rekeyed"))
            else:
                direct_drops += 1

        # PASS 2: commit only candidates whose target is claimed exactly once.
        target_counts: collections.Counter[LabelKey] = collections.Counter(
            c[1] for c in candidates
        )
        for _src, target, category, fp, kind in candidates:
            if target_counts[target] == 1:
                labels[target] = category
                fingerprints[target] = fp
                if kind == "kept":
                    kept += 1
                else:
                    rekeyed += 1
            else:
                dropped += 1

        dropped += direct_drops
        return labels, fingerprints, RekeyReport(kept=kept, rekeyed=rekeyed, dropped=dropped)

    # -- queries -------------------------------------------------------------
    def filtered_df(self, master_df: pl.DataFrame) -> pl.DataFrame:
        """Return ``master_df`` with all labeled (removed) rows dropped."""
        if not self.labels:
            return master_df
        return _join_on_keys(master_df, self.labels, "anti")

    def _fingerprint_of(self, image_file: str, label: int) -> tuple[float, float] | None:
        """Look up an object's centroid in the cached master, or ``None``."""
        if KEY_CENTER_RR not in self._master_df.columns or KEY_CENTER_CC not in self._master_df.columns:
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

    def write_error_partitions(self) -> None:
        """Write the per-category error parquets + the (re-keyed) labels parquet.

        Deliberately does **not** rewrite the curated ``measurements.parquet``
        mirror — used by CLI finalize to re-emit the durable error deliverables
        headlessly while leaving the post-applied measurements seed untouched
        (curation of the mirror stays the GUI's live responsibility, re-derived
        on the next viewer load from the re-keyed labels). Dash-free.
        """
        with self._lock:
            self._publish_if_current(
                self._write_error_partitions_locked,
                context="error partitions",
            )

    def _write_error_partitions_locked(self) -> None:
        """Write error partitions and labels while publication locks are held."""
        self._write_category_parquets()
        self._write_labels_parquet()

    def _save_locked(self) -> None:
        """Write curated mirror + per-category files, then labels parquet (lock held).

        The labels parquet is written LAST so that a crash mid-save leaves the
        durable store consistent with the previously written derived outputs,
        not ahead of them.

        Refuses to write when any curation source changed after this store
        loaded or last published successfully.
        """
        self._publish_if_current(
            self._write_all_derived_outputs_locked,
            context="curation outputs",
        )

    def _write_all_derived_outputs_locked(self) -> None:
        """Write every curation output while publication locks are held."""
        self._write_curated_mirror()
        self._write_category_parquets()
        self._write_labels_parquet()

    @property
    def _publication_lock_path(self) -> Path:
        """Return the stable interprocess lock shared by curation writers."""
        return self.labels_path.with_suffix(".lock")

    @staticmethod
    def _source_fingerprint(layout: BundleLayout) -> str:
        """Fingerprint curation inputs and every currently published partition."""
        paths = [
            layout.mirror_parquet,
            layout.mirror_csv,
            layout.curation_labels_parquet,
            layout.custom_categories_json,
        ]
        if layout.errors_dir.is_dir():
            paths.extend(sorted(layout.errors_dir.glob("*.parquet")))
        return paths_fingerprint(paths, root=layout.deliverables_base)

    def _publish_if_current(
        self,
        writer: Callable[[], None],
        *,
        context: str,
    ) -> bool:
        """Compare-and-publish one curation mutation under a shared file lock."""
        try:
            with exclusive_path_lock(self._publication_lock_path):
                mirror = self.measurements_parquet
                current_fingerprint = self._source_fingerprint(self._layout)
                current_mtime = (
                    mirror.stat().st_mtime_ns if mirror.exists() else None
                )
                if (
                    current_fingerprint != self._expected_source_fingerprint
                    or current_mtime != self._seed_mtime_ns
                ):
                    logger.warning(
                        "Refusing to overwrite %s because curation sources "
                        "changed since this viewer session loaded them. Reload "
                        "the viewer before curating again.",
                        context,
                    )
                    self._stale = True
                    return False
                writer()
                self._expected_source_fingerprint = self._source_fingerprint(
                    self._layout
                )
                self._seed_mtime_ns = (
                    mirror.stat().st_mtime_ns if mirror.exists() else None
                )
                return True
        except ArtifactLockTimeout:
            logger.warning(
                "Refusing to overwrite %s because another curation writer "
                "did not release the shared publication lock.",
                context,
            )
            self._stale = True
            return False

    def _write_labels_parquet(self) -> None:
        path = self.labels_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = [self.fingerprints.get(k, _NAN_FP) for k in self.labels]
        rows = {
            KEY_IMAGE_FILE: [k[0] for k in self.labels],
            KEY_OBJECT_LABEL: [k[1] for k in self.labels],
            KEY_CATEGORY: [self.labels[k] for k in self.labels],
            KEY_CENTER_RR: [fp[0] for fp in fps],
            KEY_CENTER_CC: [fp[1] for fp in fps],
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
        # Curate the POST-APPLIED mirror (keeps its post columns), not the
        # clean master — falls back to the clean master only when no mirror
        # frame was supplied (direct construction / legacy).
        mirror_source = self._mirror_df if self._mirror_df is not None else self._master_df
        curated = self.filtered_df(mirror_source)
        self.measurements_parquet.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(curated, self.measurements_parquet)
        try:
            _atomic_write_csv(curated, self.measurements_csv)
        except Exception:
            logger.exception("Failed to write curated CSV mirror at %s", self.measurements_csv)

    def _write_category_parquets(self) -> None:
        errs_dir = self._layout.errors_dir
        errs_dir.mkdir(parents=True, exist_ok=True)
        # Group keys by category.
        by_cat: dict[str, list[LabelKey]] = {}
        for key, cat in self.labels.items():
            by_cat.setdefault(cat, []).append(key)
        # Write present categories; collect the known-category tokens to sweep.
        present: set[str] = set()
        for cat, keys in by_cat.items():
            token = sanitize_category(cat)
            if not token:
                logger.warning("Skipping category with no filename-safe token: %r", cat)
                continue
            frame = _join_on_keys(self._master_df, keys, "semi").with_columns(
                pl.lit(cat).alias(KEY_CATEGORY)
            )
            # S1: skip writing 0-row category parquets; don't add to present so
            # any prior stale file for this token gets pruned below.
            if frame.is_empty():
                continue
            present.add(token)
            _atomic_write_parquet(frame, self._layout.error_category_parquet(token))
        # Clean up stale category files — restrict sweep to the KNOWN vocabulary
        # only, so unrelated files in errors/ are never touched (FIX-2).
        known_filenames = {f"{sanitize_category(c)}.parquet" for c in self.categories()}
        for fname in known_filenames:
            stem = fname[: -len(".parquet")]
            if stem not in present:
                candidate = errs_dir / fname
                if candidate.exists():
                    try:
                        candidate.unlink()
                    except OSError:
                        logger.warning("Could not remove stale category file %s", candidate)

    # -- FilteredMeasurements-compatible surface -----------------------------
    @property
    def removed_keys(self) -> set[LabelKey]:
        """Snapshot copy of all labeled keys (any category) — the removal set.

        Returns a fresh ``set`` each call. Unlike the old ``FilteredMeasurements``
        field, mutating the returned set does **not** change stored state —
        mutate via :meth:`mark`/:meth:`unmark`/:meth:`remove`/:meth:`restore`.
        """
        return set(self.labels.keys())

    def is_removed(self, image_file: str, object_label: int) -> bool:
        """Return whether the object carries any label."""
        return (image_file, object_label) in self.labels

    def removed_count_in(self, df: pl.DataFrame) -> int:
        """Count rows of ``df`` whose key is currently labeled."""
        if df.is_empty() or not self.labels:
            return 0
        df_keys = _keys_of(df)
        return len(df_keys & set(self.labels.keys()))

    def remove(self, image_file: str, object_label: int) -> None:
        """Mark as the reasonless ``other`` category (legacy remove)."""
        self.mark(image_file, object_label, OTHER_CATEGORY)

    def restore(self, image_file: str, object_label: int) -> None:
        """Clear any label (legacy restore)."""
        self.unmark(image_file, object_label)

    def remove_many(self, keys: Iterable[LabelKey]) -> None:
        """Mark a batch as ``other`` in one save."""
        self.mark_many(keys, OTHER_CATEGORY)

    def restore_many(self, keys: Iterable[LabelKey]) -> None:
        """Remove labels for a batch in one save."""
        self.unmark_many(keys)

    def toggle(self, image_file: str, object_label: int) -> None:
        """Flip label state for one object (clears if labeled, else ``other``)."""
        key = (image_file, object_label)
        with self._lock:
            if key in self.labels:
                self.unmark(image_file, object_label)
            else:
                self.mark(image_file, object_label, OTHER_CATEGORY)

    def removed_keys_payload(self) -> list[list]:
        """``[[image_file, object_label], ...]`` sorted, for the dcc.Store."""
        return [[f, lbl] for f, lbl in sorted(self.labels.keys(), key=lambda k: (k[0], k[1]))]

    def labels_payload(self) -> list[list]:
        """``[[image_file, object_label, category], ...]`` sorted, category-aware."""
        return [
            [f, lbl, self.labels[(f, lbl)]]
            for f, lbl in sorted(self.labels.keys(), key=lambda k: (k[0], k[1]))
        ]

    def mutate_and_payload(self, action: Callable[["CurationLabels"], None]) -> list[list]:
        """Apply ``action`` and return the removed-keys payload, all under the lock."""
        with self._lock:
            action(self)
            return self.removed_keys_payload()


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
