# Error-Triage Cutoffs — Phase 1: Data-Model Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and unit-test the standalone data-model foundation for error-category triage — the `ErrorCategory`/`CURATION` schema enums, the `deliverables/errors/` + `qc/curation_labels.parquet` path helpers, and the durable `CurationLabels` store (categorized labels, fingerprint re-keying, derived curated + per-category outputs) — without yet touching the live GUI.

**Architecture:** A single durable labels store keyed by `(Metadata_ImageFile, Object_Label) → category` is the source of truth; the curated `deliverables/measurements.parquet` mirror and the per-category `deliverables/errors/<category>.parquet` files are derived from it on every mutation. The store generalizes the existing remove-only `FilteredMeasurements` and is built API-compatible with it so Phase 2 can swap it into the running viewer with minimal churn. This phase produces fully unit-tested modules; **no app wiring happens here** (that is Phase 2, Task 1).

**Tech Stack:** Python 3.12, pydantic-free `dataclasses`, `polars` (frames + parquet), `str/Enum` (`MeasurementInfo`), `pytest`, `uv` runner.

**Spec:** `docs/superpowers/specs/2026-06-10-error-category-triage-cutoff-finder-design.md` (§5 Data model).

---

## Conventions for this plan

- Run everything through `uv run` (never bare `python`/`pytest`).
- Run a single test: `uv run pytest <path>::<test> -v`. Qt is **not** needed for any Phase 1 test (these are pure data/schema modules), so no `QT_QPA_PLATFORM` is required.
- Commit after each task with the message shown in its final step. Commit scoped to the named paths (`git add <paths>` then `git commit`), never `git add -A`.
- Working dir is the worktree: `/Users/alex/Projects/PhenoTypic/.claude/worktrees/error-triage-cutoffs`. Use worktree-relative paths.

## File structure (created/modified in Phase 1)

- Create: `src/phenotypic/schema/_error_category.py` — `ErrorCategory` value enum.
- Create: `src/phenotypic/schema/_curation.py` — `CURATION` column-name enum.
- Modify: `src/phenotypic/schema/__init__.py` — re-export both.
- Modify: `src/phenotypic/tools_/_io_constants.py` — `DIR_ERRORS`, filename constants, path helpers, `__all__`.
- Modify: `src/phenotypic/tools_/__init__.py` — re-export the new helpers.
- Create: `src/phenotypic/gui/results_viewer/_curation_labels.py` — `CurationLabels` store + `RekeyReport`.
- Create: `tests/unit/schema/test_error_category.py`
- Create: `tests/unit/tools_/test_error_paths.py`
- Create: `tests/unit/gui/results_viewer/test_curation_labels.py`

**Explicitly NOT touched in Phase 1:** `results_viewer/_app.py`, `_filtered_state.py`, any `_callbacks.py`, `_layout.py`, `gui/_config.py`. Integration is Phase 2.

---

### Task 1: `ErrorCategory` + `CURATION` schema enums

**Files:**
- Create: `src/phenotypic/schema/_error_category.py`
- Create: `src/phenotypic/schema/_curation.py`
- Modify: `src/phenotypic/schema/__init__.py`
- Test: `tests/unit/schema/test_error_category.py`

**Why:** The core taxonomy must be a closed, documented value set. Per project convention (`CLAUDE.md` "closed value sets needing user-visible documentation: prefer `MeasurementInfo`/`ConstantLabels`"), these are `MeasurementInfo` subclasses. `schema/` may import only stdlib + its sibling base, so they subclass `MeasurementInfo` directly (not `ConstantLabels`, which lives in `tools_`). `ErrorCategory` values are persisted as the **bare `.label`** (e.g. `"oversegmented"`), filename-safe by construction. `CURATION.CATEGORY` names the `Curation_Category` column.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schema/test_error_category.py`:

```python
"""Tests for the ErrorCategory + CURATION schema enums."""

import pytest

from phenotypic.schema import CURATION, ErrorCategory


CORE_LABELS = [
    "oversegmented",
    "undersegmented",
    "merged",
    "background_noise",
    "debris",
    "other",
]


def test_error_category_labels_are_the_core_taxonomy():
    assert ErrorCategory.labels() == CORE_LABELS


def test_error_category_label_is_bare_and_filename_safe():
    # The persisted token is the bare label, not the prefixed value.
    assert ErrorCategory.OVERSEGMENTED.label == "oversegmented"
    assert str(ErrorCategory.OVERSEGMENTED) == "ErrorCategory_oversegmented"
    for member in ErrorCategory:
        assert member.label.replace("_", "").isalnum(), member.label


def test_error_category_descriptions_present():
    for member in ErrorCategory:
        assert member.desc, f"{member.label} missing a description"


def test_other_is_the_reserved_reasonless_bucket():
    assert ErrorCategory.OTHER.label == "other"


def test_from_label_round_trips_and_rejects_unknown():
    assert ErrorCategory.from_label("debris") is ErrorCategory.DEBRIS
    assert ErrorCategory.from_label("not_a_category") is None


def test_curation_category_column_name():
    assert str(CURATION.CATEGORY) == "Curation_Category"
    assert CURATION.CATEGORY.label == "Category"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_error_category.py -v`
Expected: FAIL — `ImportError: cannot import name 'ErrorCategory' from 'phenotypic.schema'`.

- [ ] **Step 3: Create `ErrorCategory`**

Create `src/phenotypic/schema/_error_category.py`:

```python
"""The closed taxonomy of colony-detection error categories.

``ErrorCategory`` is the vocabulary a user assigns when triaging a detected
object as bad (rather than merely removing it). Members are
``MeasurementInfo``-style ``(label, description)`` tuples so the description is
available to the GUI radial-menu tooltips and the generated docs. The
**bare label** (``member.label``, e.g. ``"oversegmented"``) is the canonical
token persisted in the ``Curation_Category`` column and used as the
per-category parquet filename, so every label is filename-safe.

``OTHER`` is the reserved catch-all: marking an object ``other`` is the
"remove without a specified reason" path. Categories beyond this enum are
user-defined custom categories, registered at runtime (not enum members).
"""

from __future__ import annotations

from ._measurement_info import MeasurementInfo


class ErrorCategory(MeasurementInfo):
    """Closed taxonomy of detection-error categories for object triage.

    The enum *value* is the category-prefixed header (``ErrorCategory_<label>``)
    per the ``MeasurementInfo`` convention, but callers persist and compare on
    the bare :attr:`~MeasurementInfo.label` (e.g. ``"debris"``). Use
    :meth:`from_label` to resolve a stored token back to a member.
    """

    @classmethod
    def category(cls) -> str:
        return "ErrorCategory"

    @classmethod
    def labels(cls) -> list[str]:
        """Return the bare category tokens in declaration order.

        Returns:
            The ``.label`` of every member, e.g.
            ``["oversegmented", ..., "other"]``.
        """
        return cls.get_labels()

    @classmethod
    def from_label(cls, label: str) -> "ErrorCategory | None":
        """Resolve a bare category token to its member, or ``None``.

        Args:
            label: A bare category token (e.g. ``"merged"``).

        Returns:
            The matching member, or ``None`` if ``label`` is not a core
            category (e.g. a custom category or a typo).
        """
        for member in cls:
            if member.label == label:
                return member
        return None

    OVERSEGMENTED = (
        "oversegmented",
        "One colony split into multiple detections.",
    )
    UNDERSEGMENTED = (
        "undersegmented",
        "A single colony detected as too small or fragmented low.",
    )
    MERGED = (
        "merged",
        "Multiple touching colonies detected as one object.",
    )
    BACKGROUND_NOISE = (
        "background_noise",
        "Not a colony — agar texture, reflection, or vignette.",
    )
    DEBRIS = (
        "debris",
        "Dust, scratch, bubble, or other plate artifact.",
    )
    OTHER = (
        "other",
        "Removed without a specified reason (the catch-all bucket).",
    )
```

- [ ] **Step 4: Create `CURATION`**

Create `src/phenotypic/schema/_curation.py`:

```python
"""Column names for GUI curation state written into derived frames."""

from __future__ import annotations

from ._measurement_info import MeasurementInfo


class CURATION(MeasurementInfo):
    """Curation-state columns attached to derived measurement frames.

    ``Curation_Category`` carries the :class:`ErrorCategory` bare label (or a
    custom category token) for each removed object in the per-category error
    parquets.
    """

    @classmethod
    def category(cls) -> str:
        return "Curation"

    CATEGORY = (
        "Category",
        "Error-category token assigned to a removed/triaged object.",
    )
```

- [ ] **Step 5: Re-export from the schema package**

In `src/phenotypic/schema/__init__.py`, add the imports next to the other enum imports (e.g. after the `from ._object import OBJECT` line):

```python
from ._curation import CURATION
from ._error_category import ErrorCategory
```

And add both names to `__all__` (keep the list alphabetized within its existing grouping):

```python
    "CURATION",
    "ErrorCategory",
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_error_category.py -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/schema/_error_category.py src/phenotypic/schema/_curation.py src/phenotypic/schema/__init__.py tests/unit/schema/test_error_category.py
git commit -m "feat(schema): add ErrorCategory taxonomy + CURATION column enum"
```

---

### Task 2: io_constants paths for the labels store + error deliverables

**Files:**
- Modify: `src/phenotypic/tools_/_io_constants.py`
- Modify: `src/phenotypic/tools_/__init__.py`
- Test: `tests/unit/tools_/test_error_paths.py`

**Why:** Every artifact filename/dir is single-sourced in `_io_constants.py` and re-exported from `tools_`. The store and (later) the CLI/GUI must join these via helpers, never by hand.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/tools_/test_error_paths.py`:

```python
"""Path-helper tests for the curation-labels store + error deliverables."""

from pathlib import Path

from phenotypic.tools_ import (
    curation_labels_parquet_path,
    custom_categories_json_path,
    deliverables_dir,
    error_analysis_csv_path,
    error_analysis_html_path,
    error_analysis_parquet_path,
    error_category_parquet_path,
    errors_dir,
    qc_dir,
)


def test_errors_dir_under_deliverables():
    out = Path("/tmp/run")
    assert errors_dir(out) == deliverables_dir(out) / "errors"


def test_error_category_parquet_path_uses_bare_token():
    out = Path("/tmp/run")
    assert (
        error_category_parquet_path(out, "background_noise")
        == errors_dir(out) / "background_noise.parquet"
    )


def test_error_analysis_paths_under_deliverables():
    out = Path("/tmp/run")
    assert error_analysis_parquet_path(out) == deliverables_dir(out) / "error_analysis.parquet"
    assert error_analysis_csv_path(out) == deliverables_dir(out) / "error_analysis.csv"
    assert error_analysis_html_path(out) == deliverables_dir(out) / "error_analysis.html"


def test_curation_store_paths_under_qc():
    out = Path("/tmp/run")
    assert curation_labels_parquet_path(out) == qc_dir(out) / "curation_labels.parquet"
    assert custom_categories_json_path(out) == qc_dir(out) / "custom_categories.json"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/tools_/test_error_paths.py -v`
Expected: FAIL — `ImportError: cannot import name 'errors_dir' from 'phenotypic.tools_'`.

- [ ] **Step 3: Add the constants**

In `src/phenotypic/tools_/_io_constants.py`, add next to `DIR_DELIVERABLES` (after line ~429):

```python
#: Per-category error-object parquet subdirectory under deliverables:
#: ``<output>/deliverables/errors/<category>.parquet``. Holds the master rows
#: for each triaged error category (GUI-written live, CLI-re-emitted).
DIR_ERRORS: Final[str] = "errors"

#: Durable curation-labels store: ``<output>/qc/curation_labels.parquet``.
#: The source of truth for categorized removals; the CLI re-keys but never
#: wipes it (contrast :data:`QC_REVIEW_STATE_JSON`).
CURATION_LABELS_PARQUET: Final[str] = "curation_labels.parquet"

#: Ordered custom-category registry sidecar: ``<output>/qc/custom_categories.json``.
CUSTOM_CATEGORIES_JSON: Final[str] = "custom_categories.json"

#: Ranked error-cutoff analysis deliverables.
ERROR_ANALYSIS_PARQUET: Final[str] = "error_analysis.parquet"
ERROR_ANALYSIS_CSV: Final[str] = "error_analysis.csv"
ERROR_ANALYSIS_HTML: Final[str] = "error_analysis.html"
```

- [ ] **Step 4: Add the path helpers**

In `src/phenotypic/tools_/_io_constants.py`, add near the deliverables/qc helpers (after `qc_review_state_path`, ~line 1242):

```python
def errors_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/errors/`` (per-category error parquets)."""
    return deliverables_dir(output_dir) / DIR_ERRORS


def error_category_parquet_path(output_dir: Path, category: str) -> Path:
    """Return ``<output>/deliverables/errors/<category>.parquet``.

    Args:
        output_dir: Run output directory.
        category: A bare, already-sanitized category token (e.g.
            ``"background_noise"``). The caller is responsible for sanitization.
    """
    return errors_dir(output_dir) / f"{category}.parquet"


def error_analysis_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/error_analysis.parquet``."""
    return deliverables_dir(output_dir) / ERROR_ANALYSIS_PARQUET


def error_analysis_csv_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/error_analysis.csv``."""
    return deliverables_dir(output_dir) / ERROR_ANALYSIS_CSV


def error_analysis_html_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/error_analysis.html``."""
    return deliverables_dir(output_dir) / ERROR_ANALYSIS_HTML


def curation_labels_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/curation_labels.parquet`` (durable labels store)."""
    return qc_dir(output_dir) / CURATION_LABELS_PARQUET


def custom_categories_json_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/custom_categories.json`` (custom-category registry)."""
    return qc_dir(output_dir) / CUSTOM_CATEGORIES_JSON
```

- [ ] **Step 5: Re-export from `tools_/__init__.py`**

(Note: `_io_constants.py` has **no** `__all__` — re-export and `__all__` registration happen only in `tools_/__init__.py`, below.)

In `src/phenotypic/tools_/__init__.py`, add the new names to the existing `from ._io_constants import (...)` block (next to `deliverables_dir`, `qc_dir`, `measurements_parquet_path`) **and** to the package `__all__` list:

```python
    curation_labels_parquet_path,
    custom_categories_json_path,
    error_analysis_csv_path,
    error_analysis_html_path,
    error_analysis_parquet_path,
    error_category_parquet_path,
    errors_dir,
```

(and the matching `"..."` string entries in `__all__`.)

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/unit/tools_/test_error_paths.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/tools_/_io_constants.py src/phenotypic/tools_/__init__.py tests/unit/tools_/test_error_paths.py
git commit -m "feat(io): add error-deliverables + curation-labels path helpers"
```

---

### Task 3: `CurationLabels` store — skeleton, custom registry, exact-key load

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_curation_labels.py`
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py`

**Why:** The store is the durable source of truth. This task builds the dataclass, the category vocabulary helpers, the custom-category registry, and an exact-key `load` (fingerprint validation and migration come in Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/results_viewer/test_curation_labels.py`:

```python
"""Tests for the durable CurationLabels store."""

from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui.results_viewer._curation_labels import (
    CurationLabels,
    sanitize_category,
)
from phenotypic.schema import ErrorCategory


def _master(n: int = 4) -> pl.DataFrame:
    """A minimal master frame: n objects in one image, distinct centroids."""
    return pl.DataFrame(
        {
            "Metadata_ImageFile": ["plateA"] * n,
            "Metadata_Dataset": ["ds1"] * n,
            "Object_Label": list(range(1, n + 1)),
            "Bbox_CenterRR": [10.0 * i for i in range(1, n + 1)],
            "Bbox_CenterCC": [20.0 * i for i in range(1, n + 1)],
            "Size_Area": [100.0 * i for i in range(1, n + 1)],
        }
    )


def test_sanitize_category():
    assert sanitize_category("  Halo Effect! ") == "halo_effect"
    assert sanitize_category("../etc") == "etc"
    assert sanitize_category("###") == ""


def test_load_empty_when_nothing_on_disk(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {}
    assert store.categories()[: len(ErrorCategory.labels())] == ErrorCategory.labels()
    assert store.rekey_report.total == 0


def test_register_custom_category_persists_and_dedupes(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    token = store.register_custom_category("Halo Effect")
    assert token == "halo_effect"
    assert "halo_effect" in store.categories()
    # idempotent
    assert store.register_custom_category("halo_effect") == "halo_effect"
    assert store.custom_categories.count("halo_effect") == 1
    # reloads from disk
    reloaded = CurationLabels.load(tmp_path, _master())
    assert "halo_effect" in reloaded.custom_categories


def test_register_rejects_core_collision_and_empty(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    with pytest.raises(ValueError):
        store.register_custom_category("debris")  # core token
    with pytest.raises(ValueError):
        store.register_custom_category("###")  # sanitizes to empty
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` for `_curation_labels`.

- [ ] **Step 3: Create the module skeleton + load + registry**

Create `src/phenotypic/gui/results_viewer/_curation_labels.py`:

```python
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
from typing import Callable, Iterable

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
KEY_CATEGORY: str = str(CURATION.CATEGORY)  # "Curation_Category"
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
            (str(f), int(l))
            for f, l in zip(
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -v`
Expected: PASS (4 tests: sanitize, empty-load, register-persist, register-reject).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_curation_labels.py tests/unit/gui/results_viewer/test_curation_labels.py
git commit -m "feat(viewer): CurationLabels skeleton — custom registry + exact-key load"
```

---

### Task 4: Mutators + derived outputs (curated mirror + per-category parquets)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py`
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py`

**Why:** Marking an object must (a) record its category, (b) capture its centroid fingerprint, and (c) live-write all three derived artifacts (labels parquet, curated mirror, per-category errors) atomically, mirroring `FilteredMeasurements`' lock + `.tmp`/`os.replace` discipline.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/gui/results_viewer/test_curation_labels.py`:

```python
import phenotypic.tools_ as tools_


def test_mark_writes_all_derived_outputs(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 2, "background_noise")

    # label recorded + fingerprint captured from master
    assert store.labels[("plateA", 2)] == "background_noise"
    assert store.fingerprints[("plateA", 2)] == (20.0, 40.0)

    # curated mirror drops the marked object
    curated = pl.read_parquet(tools_.measurements_parquet_path(tmp_path))
    assert curated.height == 3
    assert 2 not in curated.get_column("Object_Label").to_list()

    # per-category parquet contains exactly the marked object
    errs = pl.read_parquet(
        tools_.error_category_parquet_path(tmp_path, "background_noise")
    )
    assert errs.get_column("Object_Label").to_list() == [2]
    assert errs.get_column("Curation_Category").to_list() == ["background_noise"]

    # labels store round-trips on reload
    reloaded = CurationLabels.load(tmp_path, _master())
    assert reloaded.labels == {("plateA", 2): "background_noise"}


def test_unmark_restores_and_clears_category_file(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 2, "debris")
    store.unmark("plateA", 2)
    assert store.labels == {}
    curated = pl.read_parquet(tools_.measurements_parquet_path(tmp_path))
    assert curated.height == 4
    # the now-empty category file is removed
    assert not tools_.error_category_parquet_path(tmp_path, "debris").exists()


def test_mark_rejects_unknown_category(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    with pytest.raises(ValueError):
        store.mark("plateA", 1, "not_registered")


def test_mark_many_single_save(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark_many([("plateA", 1), ("plateA", 3)], "oversegmented")
    errs = pl.read_parquet(
        tools_.error_category_parquet_path(tmp_path, "oversegmented")
    )
    assert sorted(errs.get_column("Object_Label").to_list()) == [1, 3]


def test_unmark_one_of_two_categories_keeps_other(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 1, "debris")
    store.mark("plateA", 2, "merged")
    store.unmark("plateA", 1)
    # the emptied category file is removed; the other survives intact
    assert not tools_.error_category_parquet_path(tmp_path, "debris").exists()
    merged = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "merged"))
    assert merged.get_column("Object_Label").to_list() == [2]


def test_mark_absent_key_degrades_to_nan_fingerprint(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 999, "debris")  # object 999 is not in master
    assert store.labels[("plateA", 999)] == "debris"
    assert ("plateA", 999) not in store.fingerprints  # no centroid to capture
    # persisted with NaN fingerprint -> dropped on the next re-key load
    reloaded = CurationLabels.load(tmp_path, _master())
    assert ("plateA", 999) not in reloaded.labels
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -k "mark or unmark" -v`
Expected: FAIL — `AttributeError: 'CurationLabels' object has no attribute 'mark'`.

- [ ] **Step 3: Add mutators + derived-output writers**

Add these methods to the `CurationLabels` class in `_curation_labels.py`:

```python
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
```

And add these module-level helpers at the bottom of the file (above any `__all__`):

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -v`
Expected: PASS (all Task 3 + Task 4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_curation_labels.py tests/unit/gui/results_viewer/test_curation_labels.py
git commit -m "feat(viewer): CurationLabels mutators + live derived outputs"
```

---

### Task 5: Fingerprint re-keying + legacy migration

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py`
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py`

**Why:** The confirmed durability behavior: labels survive CLI re-runs. Since re-detection can renumber `Object_Label`, the load must validate each stored label against its centroid fingerprint and recover renumbered objects, dropping ambiguous ones (never silently moving). It must also migrate a legacy `deliverables/measurements.parquet`-only directory into the store as `other`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/gui/results_viewer/test_curation_labels.py`:

```python
def _write_store_with_label(tmp_path, image_file, label, category, rr, cc):
    """Helper: seed a labels parquet directly (simulating a prior session)."""
    df = pl.DataFrame(
        {
            "Metadata_ImageFile": [image_file],
            "Object_Label": [label],
            "Curation_Category": [category],
            "Bbox_CenterRR": [rr],
            "Bbox_CenterCC": [cc],
        }
    )
    path = tools_.curation_labels_parquet_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_rekey_keeps_exact_match(tmp_path: Path):
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 20.0, 40.0)
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {("plateA", 2): "debris"}
    assert store.rekey_report.kept == 1


def test_rekey_recovers_renumbered_object(tmp_path: Path):
    # Stored label was object 2 at (20,40). New master renumbered it to 99 but
    # the centroid is unchanged -> re-key to 99.
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 20.0, 40.0)
    master = _master().with_columns(
        pl.when(pl.col("Object_Label") == 2)
        .then(99)
        .otherwise(pl.col("Object_Label"))
        .alias("Object_Label")
    )
    store = CurationLabels.load(tmp_path, master)
    assert store.labels == {("plateA", 99): "debris"}
    assert store.rekey_report.rekeyed == 1


def test_rekey_drops_when_object_gone(tmp_path: Path):
    # Stored centroid (500,500) matches nothing in master -> dropped.
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 500.0, 500.0)
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {}
    assert store.rekey_report.dropped == 1


def test_migrates_legacy_measurements_parquet_as_other(tmp_path: Path):
    # A legacy curated mirror missing object 3 -> object 3 imported as "other".
    master = _master()
    curated = master.filter(pl.col("Object_Label") != 3)
    legacy = tools_.measurements_parquet_path(tmp_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    curated.write_parquet(legacy)

    store = CurationLabels.load(tmp_path, master)
    assert store.labels == {("plateA", 3): "other"}


def test_rekey_drops_rather_than_attaching_to_neighbor(tmp_path: Path):
    # Object 2 still exists (exact key present at (20,40)) but the STORED centroid
    # is (10,20) == object 1's position. The label must DROP, never silently
    # re-key onto the neighbour at the stored centroid.
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 10.0, 20.0)
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {}
    assert store.rekey_report.dropped == 1


def test_rekey_degrades_without_bbox(tmp_path: Path):
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 20.0, 40.0)
    master = _master().drop(["Bbox_CenterRR", "Bbox_CenterCC"])
    store = CurationLabels.load(tmp_path, master)
    # exact key (plateA, 2) still exists -> kept on the exact-key-only fallback
    assert store.labels == {("plateA", 2): "debris"}
    assert store.rekey_report.kept == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -k "rekey or migrate" -v`
Expected: FAIL — the Task 3 stub keeps exact keys unconditionally (no fingerprint), so: the renumber test expects key `99` (stub drops it), the migration test expects a non-empty store, and `test_rekey_drops_rather_than_attaching_to_neighbor` expects an empty store (the stub keeps object 2 instead of dropping it). `test_rekey_degrades_without_bbox` may already pass under the stub — it is a regression guard for the no-Bbox branch.

- [ ] **Step 3: Replace `_rekey` and add migration to `load`**

Replace the Task-3 `_rekey` stub with the fingerprint version, and add a `_migrate_legacy` helper. Also wire migration into `load` (when the labels parquet is absent but a legacy mirror exists):

```python
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
            (str(f), int(l))
            for f, l in zip(
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
```

Then update `load` so its labels-source branch reads:

```python
        labels_path = curation_labels_parquet_path(root)
        if labels_path.exists():
            stored = cls._read_labels_parquet(labels_path)
            labels, fingerprints, report = cls._rekey(stored, master_df)
        elif measurements_parquet_path(root).exists():
            labels, fingerprints = cls._migrate_legacy(root, master_df)
            report = RekeyReport(migrated=len(labels))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -v`
Expected: PASS (all tests, including rekey + migration).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_curation_labels.py tests/unit/gui/results_viewer/test_curation_labels.py
git commit -m "feat(viewer): fingerprint re-keying + legacy removal migration"
```

---

### Task 6: `FilteredMeasurements`-compatible API surface

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py`
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py`

**Why:** Phase 2 swaps `CurationLabels` in for `FilteredMeasurements` in the running app. To keep that swap a near-rename, the store must expose the same public methods the existing callbacks call (`remove`, `restore`, `toggle`, `is_removed`, `removed_count_in`, `removed_keys`, `removed_keys_payload`, `remove_many`, `restore_many`, `mutate_and_payload`), with `remove`/`toggle` defaulting to the `other` category. It also adds `labels_payload()` (the category-aware store payload Phase 2's tiles read).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/gui/results_viewer/test_curation_labels.py`:

```python
def test_filtered_measurements_compat_surface(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())

    # remove == mark as "other"
    store.remove("plateA", 1)
    assert store.is_removed("plateA", 1)
    assert store.labels[("plateA", 1)] == "other"
    assert store.removed_keys == {("plateA", 1)}

    # toggle off / on
    store.toggle("plateA", 1)
    assert not store.is_removed("plateA", 1)
    store.toggle("plateA", 3)
    assert store.is_removed("plateA", 3)

    # restore
    store.restore("plateA", 3)
    assert store.removed_keys == set()

    # payloads
    store.mark("plateA", 2, "debris")
    assert store.removed_keys_payload() == [["plateA", 2]]
    assert store.labels_payload() == [["plateA", 2, "debris"]]

    # removed_count_in
    assert store.removed_count_in(_master()) == 1


def test_mutate_and_payload_runs_under_lock(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    payload = store.mutate_and_payload(lambda s: s.mark("plateA", 1, "merged"))
    assert payload == [["plateA", 1]]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -k "compat or mutate_and_payload" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'remove'`.

- [ ] **Step 3: Add the compatibility surface**

Add these methods to `CurationLabels`:

```python
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
        df_keys = {
            (str(f), int(l))
            for f, l in zip(
                df.get_column(KEY_COLUMNS[0]).to_list(),
                df.get_column(KEY_COLUMNS[1]).to_list(),
            )
        }
        return len(df_keys & set(self.labels.keys()))

    def remove(self, image_file: str, object_label: int) -> None:
        """Mark as the reasonless ``other`` category (legacy remove)."""
        self.mark(image_file, object_label, OTHER_CATEGORY)

    def restore(self, image_file: str, object_label: int) -> None:
        """Clear any label (legacy restore)."""
        self.unmark(image_file, object_label)

    def remove_many(self, keys: Iterable[LabelKey]) -> None:
        self.mark_many(keys, OTHER_CATEGORY)

    def restore_many(self, keys: Iterable[LabelKey]) -> None:
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
        return [[f, l] for f, l in sorted(self.labels.keys(), key=lambda k: (k[0], k[1]))]

    def labels_payload(self) -> list[list]:
        """``[[image_file, object_label, category], ...]`` sorted, category-aware."""
        return [
            [f, l, self.labels[(f, l)]]
            for f, l in sorted(self.labels.keys(), key=lambda k: (k[0], k[1]))
        ]

    def mutate_and_payload(self, action: Callable[["CurationLabels"], None]) -> list[list]:
        """Apply ``action`` and return the removed-keys payload, all under the lock."""
        with self._lock:
            action(self)
            return self.removed_keys_payload()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -v`
Expected: PASS (entire file).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_curation_labels.py tests/unit/gui/results_viewer/test_curation_labels.py
git commit -m "feat(viewer): FilteredMeasurements-compatible surface on CurationLabels"
```

---

### Task 7: Import smoke + phase boundary

**Files:**
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py`

**Why:** Confirm the new module imports cleanly alongside the existing viewer package (no import cycles) and lock in the explicit boundary: Phase 1 does **not** rewire the live app.

- [ ] **Step 1: Add an import-smoke test**

Append to `tests/unit/gui/results_viewer/test_curation_labels.py`:

```python
def test_imports_alongside_existing_viewer_modules():
    # The new store must not introduce an import cycle with the viewer package.
    import phenotypic.gui.results_viewer._curation_labels as cl
    import phenotypic.gui.results_viewer._filtered_state as fs  # still present

    assert hasattr(cl.CurationLabels, "load")
    # Compat surface matches the methods the app currently calls on the old store.
    for name in ("remove", "restore", "toggle", "is_removed", "removed_keys_payload"):
        assert hasattr(cl.CurationLabels, name)
        assert hasattr(fs.FilteredMeasurements, name)
```

- [ ] **Step 2: Run the full Phase 1 test set**

Run:
```bash
uv run pytest tests/unit/schema/test_error_category.py tests/unit/tools_/test_error_paths.py tests/unit/gui/results_viewer/test_curation_labels.py -v
```
Expected: PASS (all Phase 1 tests, ~20).

- [ ] **Step 3: Type-check the new modules**

Run: `uv run mypy src/phenotypic/schema/_error_category.py src/phenotypic/schema/_curation.py src/phenotypic/gui/results_viewer/_curation_labels.py`
Expected: no errors (fix any surfaced before committing).

- [ ] **Step 4: Lint**

Run: `uv run ruff check --fix src/phenotypic/gui/results_viewer/_curation_labels.py src/phenotypic/schema/_error_category.py src/phenotypic/schema/_curation.py src/phenotypic/tools_/_io_constants.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/gui/results_viewer/test_curation_labels.py
git commit -m "test(viewer): import-smoke + phase-1 boundary for CurationLabels"
```

---

## Phase 1 self-review (against spec §5)

- **§5.1 ErrorCategory enum** → Task 1 (six members incl. `OTHER`, bare-label tokens, descriptions). ✅
- **§5.2 Custom registry** → Task 3 (sanitization, dedupe, core-collision reject, JSON persistence). ✅
- **§5.3 Durable store + derived outputs** → Tasks 3–4 (labels parquet, curated mirror, per-category `errors/<cat>.parquet`, atomic writes, lock). ✅
- **§5.4 Migration + fingerprint re-keying** → Task 5 (legacy→`other`, kept/rekeyed/dropped report). ✅
- **Behavior change (plain remove durable) / API compat** → Task 6 (`remove`/`toggle` default to `other`; store is the source of truth). ✅
- **io_constants paths** → Task 2. ✅

Not in Phase 1 (correctly deferred): app wiring, the `Curation_Category` value actually surfaced in the UI, colors, the ANOVA engine, the tab, CLI finalize, docs ledgers.

## Review decisions applied (2026-06-10)

Independent plan review + user decisions folded in:

- **Bbox absent → degrade + warn** (Q1): `_rekey` keeps exact keys only and logs a WARNING when `Bbox_CenterRR/CC` are missing. (`Bbox_*` already a de-facto requirement of the colony/QC crop tiles.)
- **Exact key + centroid moved beyond tol → drop immediately** (Q2): never fall through to a neighbour search; `test_rekey_drops_rather_than_attaching_to_neighbor` guards it.
- **Stale-session mtime guard → Phase 2** (Q3): added to the Phase 2 roadmap item.
- **Fixes:** deleted the bogus `_io_constants.py` `__all__` step; removed the `sanitize_category(cat) or cat` path-injection fallback; added a `RekeyReport.migrated` field (migrations no longer mis-counted as `kept`); documented `removed_keys` as a snapshot copy; added multi-category-unmark, absent-key-NaN, neighbour-drop, and no-Bbox tests.
- **Phase 5/6 docs:** add a one-line `CLAUDE.md` note that `deliverables/errors/*` and `deliverables/error_analysis.*` are GUI-written-live **and** CLI-re-emitted at finalize (the documented "finalize owns deliverables" exception, alongside the existing curated-mirror precedent).

---

## Roadmap — Phases 2–6 (expanded into their own plans as each lands)

Each subsequent phase binds to the concrete `CurationLabels` API and `io_constants` paths created here, so its detailed plan is authored once Phase 1 is merged and those APIs are real (keeps every code block accurate, not speculative).

- **Phase 2 — Tile UI + app integration.** Swap `FilteredMeasurements.load` → `CurationLabels.load` in `results_viewer/_app.py` (drop-in via the compat surface; plain removals now durable). **Add an mtime guard** on `measurements.parquet` (restoring the protection `FilteredMeasurements._seed_mtime_ns` gave): a viewer session open across a CLI re-run must refuse to clobber the freshly-seeded measurements file and trigger a reload instead — re-keying alone only runs at load, so a long-lived session needs this guard. Build the shared nested **radial menu** component in `gui/_shared/` (core wheel + `Other` + `Custom ▸` folder + `＋ Add custom`), the per-tile category badge, and the **bulk "Mark N selected as ▾"** bar, wired on both colony-view and QC tiles through `build_tile_cell`. Category→`OI_*` color map in `_design.py`. Pattern-matched per-wedge callbacks. `FEATURES.md` rows.
- **Phase 3 — Cutoff engine.** `analysis/_error_cutoffs.py`: per-measurement one-way ANOVA (`scipy.stats.f_oneway`, good vs. category), effect size + AUC, ROC/Youden cutoff with recall/precision, BH-FDR; min-n guard; exported from `analysis/__init__.py`. Pure, fully unit-testable.
- **Phase 4 — Error-analysis tab.** `gui/results_viewer/_error_tab/`: category switcher + counts, ranked table, good-vs-error boxplot with draggable cutoff, recall/precision readout, copy-filter-spec; debounced live recompute on label change; stale banner from `rekey_report`. `FEATURES.md` + `WORKFLOWS.md` rows.
- **Phase 5 — CLI finalize wiring.** `finalize_post_master_outputs` re-emits `deliverables/errors/*.parquet` + `deliverables/error_analysis.{parquet,csv,html}` and **preserves + re-keys** the labels store (no wipe); chunk writers untouched.
- **Phase 6 — Docs/ledgers/screenshots.** `WORKFLOWS.md` tutorial round-trip (`_capture_<id>` + walkthrough page), `gui/CLAUDE.md` + io_constants docstrings, regenerate the full GUI screenshot set.
