# Reference-Metadata Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an `ImageOperation` read per-image values from an experiment metadata table during `apply()` (via a public `phenotypic.ReferenceContext`), and ship `SubtractBlank`, which subtracts a time series' media-blank frame from `detect_mat`, through the Python API, the CLI, and the GUI.

**Architecture:** A `contextvars`-backed `ReferenceContext` (in `_core`) carries the table, the reference-image resolver and the reader settings; operations mixing in the fieldless `RefMetadata` (in `abc_`) declare their columns as `RefColumn`/`RefImageColumn` fields and read the active context inside `_operate`. The CLI never threads arguments to workers: at startup it plans every image's references once, writes a run-level manifest under `.phenotypic/`, and each worker core enters a context built from that manifest around its existing apply call; a per-image reference digest enters the work-id. The GUI builder sets a session-level context around its previews; the run console refuses Run when a needed table is missing.

**Tech Stack:** Python 3.12, pydantic v2, polars 1.41, numpy, click (CLI), Dash/dbc (GUI), pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-ref-metadata-ops/design.md`

## Global Constraints

- `uv` is the only runner: `uv run pytest …`, `uv run ruff check --fix <explicit paths>` (never bare `ruff check --fix`).
- Operations are pydantic models: typed class-level fields, **no hand-written `__init__`**, keyword-only construction.
- `_operate` is an instance method. Operations return the image; enhancers change `detect_mat` only (`rgb`/`gray` untouched).
- Lazy imports: `import phenotypic` and `phenotypic.ReferenceContext` attribute access must not load polars, pandas or anything in `HEAVY_STARTUP_MODULES`. Import those inside functions.
- Google-style docstrings; doctests must run with `load_synth_yeast_plate()`; microbiology context in examples.
- Metadata semantics through schema helpers (`ensure_metadata_prefix`, `normalize_metadata_columns`, `str(IMAGE.IMAGE_NAME)`, `str(EXPERIMENT.DATASET)`), never `startswith("Metadata_")`.
- `deliverables/metadata.csv` is never rewritten by anything in this plan.
- `bio_desc` is never authored (no new `MeasurementInfo` members here anyway).
- Test instrument per stage (root `CLAUDE.md`): per step the step's own test; per task the touched test files; per phase the affected surface once; the full sharded suite **once**, at the end, as a Slurm job.
- GUI tests: prefix with `QT_QPA_PLATFORM=offscreen`.
- Commits end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01QLNM2smfudbm5MhK2hgzT1
  ```

## Refinements to the spec (already applied to the spec)

The plan was written against the code and found these corrections; the spec was edited to match before this plan was reviewed.

- **S1 — identity.** Spec §5.2 says full mode "already folds the metadata digest into the work-id". **False:** `_cli_identity.py:333` puts `metadata_sha256` into the *finalization* digest only, so a blank edit would not re-run images. The plan adds a **per-image reference digest** to `compute_work_id` (Task 6), in every mode, present only when the pipeline needs references — so existing work-ids are untouched and only images whose blank assignment or blank file changed re-run.
- **S2 — how workers get the context.** Spec §5.2 had each worker call `narrow(dataset, image_root=ds.input_dir)`. Stage-3 and SLURM workers know only the run root and dataset name. Instead, startup writes `.phenotypic/reference_manifest.json` (table path + digest, reader kwargs, per-dataset resolved image paths, per-image digests) from the same planner the preflight uses; workers build their context from it (Task 4, Task 6).
- **S3 — column markers.** Reuse the existing `_ColumnRefMarker` (`sdk_/_column_ref.py`, already rendered as a dropdown by the GUI registry) with a new source `"reference_metadata"`, instead of a new marker class. Names: `RefColumn` (a value) and `RefImageColumn` (a value that names an image; adds `_ReferenceImageMarker` so the CLI knows which columns to resolve to files). Replaces the spec's `RefColumnField`.
- **S4 — strings.** The reference table is read with `infer_schema=False`, so digit-only stems (`000123`) keep their zeros. The spec's relocation of `read_metadata_csv` to `sdk_` is dropped (not needed).
- **S5 — `narrow` signature** is `narrow(*, dataset=None, image_root=None, images=None)`; `None` keeps the parent's value.
- **S6 — process-mode snapshot** `.phenotypic/reference_metadata.csv` is added to the restart-preserved set, matching full mode, whose `deliverables/metadata.csv` survives `--restart`.
- **S7 — preflight** gains `PF-REF-TABLE` (table unreadable/invalid) beside the spec's six codes, and never hashes image files (its module contract is "headers only"); hashing happens at startup.

## Review Focus

1. **Digit-only image stems** (`000123`) in the CSV must match the file `000123.tif` — reading with type inference would turn them into `123`. Pinned in Task 1.
2. **A blank named with its extension** (`t00.tif`) instead of its stem must resolve to that file. Pinned in Task 1.
3. **Bare headers** (`ImageName`, `BlankImage`) must behave like `Metadata_ImageName`/`Metadata_BlankImage`, as the CLI join treats them. Pinned in Task 1.
4. **The same stem in two datasets with no `Metadata_Dataset` column** and different blanks must raise *ambiguous*, never silently take the first row. Pinned in Task 1.
5. **A `GridImage` target** must subtract like a plain `Image`. Pinned in Task 3.

---

## Phase 1 — Core (Python API)

### Task 0: Worktree environment

- [ ] **Step 1: Sync the environment**

Run: `uv sync --group dev --group test-qt --group docs --extra gui --extra napari`
Then: `uv run python -c "import phenotypic, polars; print(phenotypic.__name__, polars.__version__)"`
Expected: `phenotypic 1.41.2`

---

### Task 1: `ReferenceContext`

**Files:**
- Create: `src/phenotypic/_core/_reference_context.py`
- Modify: `src/phenotypic/__init__.py` (`_LAZY_CLASSES`, `TYPE_CHECKING` import block near line 92, `__all__`)
- Modify: `src/phenotypic/sdk_/__init__.py` (`_LAZY_ATTRS`, `__all__`)
- Test: `tests/unit/core/test_reference_context.py`

**Interfaces:**
- Produces (module `phenotypic._core._reference_context`, public as `phenotypic.ReferenceContext`; errors also as `phenotypic.sdk_.<Name>`):
  - `class ReferenceContextError(ValueError)`
  - `class RefMetadataUnavailableError(ReferenceContextError)`
  - `class ReferenceTableError(ReferenceContextError)`
  - `class ReferenceLookupError(ReferenceContextError)` with attributes `reason: Literal["unmatched","null","ambiguous","self"]`, `image_name: str`, `column: str | None`; constructor `ReferenceLookupError(message, *, reason, image_name, column=None)`
  - `class ReferenceImageError(ReferenceContextError)`
  - `class ReferenceContext`:
    - `__init__(self, metadata: str | Path | pd.DataFrame | pl.DataFrame, *, image_root: str | Path | None = None, images: Mapping[str, Image | str | Path] | None = None, dataset: str | None = None, read_kwargs: Mapping[str, Any] | None = None)`
    - `__enter__() -> ReferenceContext`, `__exit__(*exc) -> None`, `@classmethod current() -> ReferenceContext | None`
    - `lookup(image: Image | str, columns: Sequence[str]) -> dict[str, str]` (keys are the names as requested)
    - `has_column(name: str) -> bool`
    - `resolve_image(name: str) -> Path | Image`
    - `load_image(name: str) -> Image` (shared cached instance; callers must not modify it)
    - `reference_image_digest(name: str) -> str | None`
    - `narrow(*, dataset: str | None = None, image_root: str | Path | None = None, images: Mapping | None = None) -> ReferenceContext`
    - properties `table: pl.DataFrame`, `columns: tuple[str, ...]`, `table_sha256: str | None`; attributes `dataset`, `image_root`, `images`, `read_kwargs`
  - module functions `_read_image(path: Path, read_kwargs: Mapping[str, Any]) -> Image`, `_clear_image_cache() -> None`

- [ ] **Step 1: Write the failing tests**

```python
"""ReferenceContext: table validation, lookup grain, resolution, activation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import tifffile

from phenotypic import Image, ReferenceContext
from phenotypic._core import _reference_context as rc
from phenotypic._core._reference_context import (
    ReferenceImageError,
    ReferenceLookupError,
    ReferenceTableError,
)


@pytest.fixture(autouse=True)
def _empty_image_cache():
    rc._clear_image_cache()
    yield
    rc._clear_image_cache()


def _table(tmp_path: Path, rows: dict, name: str = "layout.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_missing_file_fails_at_construction(tmp_path):
    with pytest.raises(ReferenceTableError, match="not found"):
        ReferenceContext(tmp_path / "absent.csv")


def test_table_must_carry_image_name(tmp_path):
    path = _table(tmp_path, {"Metadata_BlankImage": ["t00"]})
    with pytest.raises(ReferenceTableError, match="Metadata_ImageName"):
        ReferenceContext(path)


def test_per_colony_rows_collapse_to_one_value(tmp_path):
    path = _table(tmp_path, {
        "Metadata_ImageName": ["t04", "t04", "t04"],
        "Grid_RowNum": [1, 2, 3],
        "Metadata_BlankImage": ["t00", "t00", "t00"],
    })
    ctx = ReferenceContext(path)
    assert ctx.lookup("t04", ["Metadata_BlankImage"]) == {"Metadata_BlankImage": "t00"}


def test_disagreeing_rows_raise_ambiguous_and_list_values(tmp_path):
    path = _table(tmp_path, {
        "Metadata_ImageName": ["t04", "t04"],
        "Metadata_BlankImage": ["t00", "t01"],
    })
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "ambiguous"
    assert "'t00'" in str(info.value) and "'t01'" in str(info.value)


def test_missing_row_raises_unmatched(tmp_path):
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).lookup("t05", ["Metadata_BlankImage"])
    assert info.value.reason == "unmatched"
    assert info.value.image_name == "t05"


def test_empty_cell_raises_null(tmp_path):
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": [None]})
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "null"


def test_unknown_column_is_a_table_error(tmp_path):
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
    ctx = ReferenceContext(path)
    assert not ctx.has_column("Metadata_FlatField")
    with pytest.raises(ReferenceTableError, match="Metadata_FlatField"):
        ctx.lookup("t04", ["Metadata_FlatField"])


# Review Focus 1
def test_digit_only_stems_keep_leading_zeros(tmp_path):
    path = tmp_path / "layout.csv"
    path.write_text("Metadata_ImageName,Metadata_BlankImage\n000123,000100\n", encoding="utf-8")
    assert ReferenceContext(path).lookup("000123", ["Metadata_BlankImage"]) == {
        "Metadata_BlankImage": "000100"
    }


# Review Focus 3
def test_bare_headers_resolve_like_prefixed_ones(tmp_path):
    path = _table(tmp_path, {"ImageName": ["t04"], "BlankImage": ["t00"]})
    ctx = ReferenceContext(path)
    assert ctx.lookup("t04", ["BlankImage"]) == {"BlankImage": "t00"}
    assert ctx.lookup("t04", ["Metadata_BlankImage"]) == {"Metadata_BlankImage": "t00"}


def test_dataset_narrows_lookup_when_table_has_dataset(tmp_path):
    path = _table(tmp_path, {
        "Metadata_Dataset": ["A", "B"],
        "Metadata_ImageName": ["t04", "t04"],
        "Metadata_BlankImage": ["a0", "b0"],
    })
    ctx = ReferenceContext(path)
    assert ctx.narrow(dataset="B").lookup("t04", ["Metadata_BlankImage"]) == {
        "Metadata_BlankImage": "b0"
    }


# Review Focus 4
def test_same_stem_two_datasets_without_dataset_column_is_ambiguous(tmp_path):
    path = _table(tmp_path, {
        "Metadata_ImageName": ["t04", "t04"],
        "Metadata_BlankImage": ["a0", "b0"],
    })
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).narrow(dataset="A").lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "ambiguous"


def test_accepts_pandas_and_polars_frames():
    rows = {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]}
    for frame in (pd.DataFrame(rows), pl.DataFrame(rows)):
        ctx = ReferenceContext(frame)
        assert ctx.table_sha256 is None
        assert ctx.lookup("t04", ["Metadata_BlankImage"]) == {"Metadata_BlankImage": "t00"}


def test_table_sha256_is_file_digest(tmp_path):
    import hashlib

    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
    assert ReferenceContext(path).table_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


# Review Focus 2
def test_resolve_by_stem_and_by_filename(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    (root / "t00.tif").write_bytes(b"x")
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"]})
    ctx = ReferenceContext(path, image_root=root)
    assert ctx.resolve_image("t00") == root / "t00.tif"
    assert ctx.resolve_image("t00.tif") == root / "t00.tif"


def test_resolve_refuses_two_files_with_one_stem(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    (root / "t00.tif").write_bytes(b"x")
    (root / "t00.png").write_bytes(b"x")
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    with pytest.raises(ReferenceImageError, match="2 files"):
        ctx.resolve_image("t00")


def test_resolve_without_root_or_mapping_raises(tmp_path):
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}))
    with pytest.raises(ReferenceImageError, match="image_root"):
        ctx.resolve_image("t00")


def test_images_mapping_takes_precedence(tmp_path):
    blank = Image(arr=np.zeros((4, 4), dtype=np.float32), name="t00")
    ctx = ReferenceContext(
        _table(tmp_path, {"Metadata_ImageName": ["t04"]}),
        image_root=tmp_path,
        images={"t00": blank},
    )
    assert ctx.resolve_image("t00") is blank
    assert ctx.load_image("t00") is blank
    assert ctx.reference_image_digest("t00") is None


def test_load_image_reads_each_file_once_and_rereads_on_change(tmp_path, monkeypatch):
    root = tmp_path / "imgs"
    root.mkdir()
    blank_path = root / "t00.tif"
    tifffile.imwrite(blank_path, np.full((8, 8), 40, dtype=np.uint8))
    calls: list[Path] = []
    real = rc._read_image

    def counting(path, read_kwargs):
        calls.append(path)
        return real(path, read_kwargs)

    monkeypatch.setattr(rc, "_read_image", counting)
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    first = ctx.load_image("t00")
    assert ctx.load_image("t00") is first
    assert len(calls) == 1
    tifffile.imwrite(blank_path, np.full((8, 8), 41, dtype=np.uint8))
    stat = blank_path.stat()
    os.utime(blank_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    ctx.load_image("t00")
    assert len(calls) == 2
    assert ctx.reference_image_digest("t00") is not None


def test_current_is_none_outside_any_context():
    assert ReferenceContext.current() is None


def test_nesting_replaces_then_restores(tmp_path):
    outer = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}))
    inner = outer.narrow(dataset="B")
    with outer:
        assert ReferenceContext.current() is outer
        with inner:
            assert ReferenceContext.current() is inner
            assert inner.table is outer.table
        assert ReferenceContext.current() is outer
    assert ReferenceContext.current() is None


def test_context_restored_after_exception(tmp_path):
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}))
    with pytest.raises(RuntimeError):
        with ctx:
            raise RuntimeError("boom")
    assert ReferenceContext.current() is None


def test_attribute_access_does_not_import_polars_or_pandas():
    code = (
        "import sys, phenotypic; phenotypic.ReferenceContext; "
        "print(','.join(m for m in ('polars', 'pandas') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/core/test_reference_context.py -q`
Expected: collection error — `ImportError: cannot import name 'ReferenceContext' from 'phenotypic'`

- [ ] **Step 3: Implement `src/phenotypic/_core/_reference_context.py`**

```python
"""Per-image reference data for operations that read experiment metadata.

A :class:`ReferenceContext` holds one metadata table and the means to resolve
the reference images it names. Activating it (``with ctx:``) makes it visible
to every :class:`~phenotypic.abc_.RefMetadata` operation that runs inside the
block, at any nesting depth, without threading an argument through
``_operate``. The mechanism is a :class:`contextvars.ContextVar`, as in
``phenotypic._core._provenance``.

This module imports only the standard library at module level; polars and
pandas are imported inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from contextvars import ContextVar, Token
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl

    from phenotypic._core._image import Image

__all__ = [
    "RefMetadataUnavailableError",
    "ReferenceContext",
    "ReferenceContextError",
    "ReferenceImageError",
    "ReferenceLookupError",
    "ReferenceTableError",
]

LookupReason = Literal["unmatched", "null", "ambiguous", "self"]


class ReferenceContextError(ValueError):
    """Base class for every reference-metadata failure."""


class RefMetadataUnavailableError(ReferenceContextError):
    """A reference-metadata operation ran with no active ReferenceContext."""


class ReferenceTableError(ReferenceContextError):
    """The reference table is missing, unreadable, or lacks a needed column."""


class ReferenceLookupError(ReferenceContextError):
    """An image's rows cannot supply exactly one value for a column.

    Attributes:
        reason: ``"unmatched"`` (no row), ``"null"`` (only empty values),
            ``"ambiguous"`` (rows disagree), or ``"self"`` (an image names
            itself as its own reference).
        image_name: The image whose lookup failed.
        column: The column that failed, when one did.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: LookupReason,
        image_name: str,
        column: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: LookupReason = reason
        self.image_name = image_name
        self.column = column


class ReferenceImageError(ReferenceContextError):
    """A reference image cannot be resolved, read, or matched to its target."""


_ACTIVE: ContextVar["ReferenceContext | None"] = ContextVar(
    "phenotypic_reference_context", default=None
)

#: Loaded reference images, most recently used last. A worker processes many
#: frames of one plate against one blank, so a small cache removes almost all
#: re-reads while bounding memory (each entry is one full image).
_IMAGE_CACHE: "OrderedDict[tuple, tuple[Image, str]]" = OrderedDict()
_IMAGE_CACHE_SIZE = 4


def _clear_image_cache() -> None:
    """Drop every cached reference image (tests and long-lived sessions)."""
    _IMAGE_CACHE.clear()


def _read_image(path: Path, read_kwargs: Mapping[str, Any]) -> "Image":
    """Read one reference image from disk (the cache's single read path)."""
    from phenotypic._core._image import Image

    return Image.imread(path, **dict(read_kwargs))


def _image_name_header() -> str:
    from phenotypic.schema import IMAGE

    return str(IMAGE.IMAGE_NAME)


def _dataset_header() -> str:
    from phenotypic.schema import EXPERIMENT

    return str(EXPERIMENT.DATASET)


def _read_table(source: Any) -> "tuple[pl.DataFrame, str | None]":
    """Read *source* as an all-string polars frame with canonical headers."""
    import pandas as pd
    import polars as pl

    from phenotypic.sdk_._metadata_helpers import normalize_metadata_columns

    digest: str | None = None
    if isinstance(source, pl.DataFrame):
        frame = source
    elif isinstance(source, pd.DataFrame):
        frame = pl.from_pandas(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise ReferenceTableError(f"Reference metadata table not found: {path}")
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                # infer_schema=False: every value is a name, and inference would
                # turn the stem "000123" into the integer 123 (Review Focus 1).
                frame = pl.read_csv(path, infer_schema=False)
            elif suffix == ".parquet":
                frame = pl.read_parquet(path)
            else:
                raise ReferenceTableError(
                    f"Reference metadata must be .csv or .parquet, got {path.name!r}"
                )
        except ReferenceTableError:
            raise
        except Exception as exc:  # noqa: BLE001 -- any parse failure is the error
            raise ReferenceTableError(f"Cannot read reference metadata {path}: {exc}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        frame = normalize_metadata_columns(frame)
    except ValueError as exc:
        raise ReferenceTableError(f"Reference metadata headers conflict: {exc}") from exc
    frame = frame.with_columns(pl.all().cast(pl.String))
    if _image_name_header() not in frame.columns:
        raise ReferenceTableError(
            f"Reference metadata needs a {_image_name_header()} column "
            f"(the image file name without its extension); columns are {frame.columns}"
        )
    return frame, digest


class _SharedTable:
    """The parsed table and lookup indexes shared by a context and its narrowings."""

    __slots__ = ("table", "sha256", "indexes")

    def __init__(self, table: "pl.DataFrame", sha256: str | None) -> None:
        self.table = table
        self.sha256 = sha256
        self.indexes: dict[tuple, dict[tuple, dict[str, list[str]]]] = {}


class ReferenceContext:
    """Per-image reference data that RefMetadata operations read during apply().

    Activate it around a pipeline call. Every operation inside the block that
    mixes in :class:`~phenotypic.abc_.RefMetadata` looks up its own columns
    for the image being processed, and may load the reference images those
    columns name (a media-blank frame, say).

    Args:
        metadata: A ``.csv``/``.parquet`` path, or a pandas/polars frame. It must
            carry ``Metadata_ImageName`` (or the bare ``ImageName``). Values are
            read as strings.
        image_root: Directory that reference-image names resolve against. A bare
            stem must match exactly one file there; a full file name matches
            that file.
        images: Mapping of name to an in-memory ``Image`` or a path. Checked
            before ``image_root``; lets a notebook or doctest prototype without
            files on disk.
        dataset: Narrows lookups to ``Metadata_Dataset == dataset`` when the
            table has that column.
        read_kwargs: Keyword arguments for ``Image.imread`` when loading
            reference images (the CLI passes its input reader settings).

    Raises:
        ReferenceTableError: The table is missing, unreadable, has conflicting
            header spellings, or lacks ``Metadata_ImageName``. Raised here, at
            construction, so a bad table fails before the first image.

    Examples:
        Ask what an operation would see, without running it:

        >>> import pandas as pd
        >>> from phenotypic import ReferenceContext
        >>> layout = pd.DataFrame({
        ...     "Metadata_ImageName": ["plate1_t04", "plate1_t04"],
        ...     "Grid_RowNum": [1, 2],
        ...     "Metadata_BlankImage": ["plate1_t00", "plate1_t00"],
        ... })
        >>> ReferenceContext(layout).lookup("plate1_t04", ["Metadata_BlankImage"])
        {'Metadata_BlankImage': 'plate1_t00'}
    """

    def __init__(
        self,
        metadata: "str | Path | pd.DataFrame | pl.DataFrame",
        *,
        image_root: str | Path | None = None,
        images: Mapping[str, "Image | str | Path"] | None = None,
        dataset: str | None = None,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        table, digest = _read_table(metadata)
        self._shared = _SharedTable(table, digest)
        self.image_root: Path | None = Path(image_root) if image_root is not None else None
        self.images: dict[str, Any] = dict(images) if images is not None else {}
        self.dataset = dataset
        self.read_kwargs: dict[str, Any] = dict(read_kwargs or {})
        self._tokens: list[Token] = []

    # ------------------------------------------------------------- activation
    def __enter__(self) -> "ReferenceContext":
        self._tokens.append(_ACTIVE.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        _ACTIVE.reset(self._tokens.pop())

    @classmethod
    def current(cls) -> "ReferenceContext | None":
        """Return the innermost active context, or ``None`` outside any."""
        return _ACTIVE.get()

    # -------------------------------------------------------------- the table
    @property
    def table(self) -> "pl.DataFrame":
        """The parsed, header-normalized, all-string table (shared; do not mutate)."""
        return self._shared.table

    @property
    def columns(self) -> tuple[str, ...]:
        """The table's normalized column names."""
        return tuple(self._shared.table.columns)

    @property
    def table_sha256(self) -> str | None:
        """SHA-256 of the table file's bytes, or ``None`` for an in-memory frame."""
        return self._shared.sha256

    def _resolve_column(self, name: str) -> str | None:
        from phenotypic.sdk_._metadata_helpers import ensure_metadata_prefix

        available = self._shared.table.columns
        if name in available:
            return name
        prefixed = ensure_metadata_prefix(name)
        return prefixed if prefixed in available else None

    def has_column(self, name: str) -> bool:
        """Whether *name* (literal first, then its ``Metadata_`` spelling) is a column."""
        return self._resolve_column(name) is not None

    def _key_columns(self) -> tuple[str, ...]:
        if self.dataset is not None and _dataset_header() in self._shared.table.columns:
            return (_dataset_header(), _image_name_header())
        return (_image_name_header(),)

    def _index(self, keys: tuple[str, ...], columns: tuple[str, ...]) -> dict:
        cache_key = (keys, columns)
        index = self._shared.indexes.get(cache_key)
        if index is None:
            import polars as pl

            grouped = self._shared.table.group_by(list(keys)).agg(
                [pl.col(c).drop_nulls().unique(maintain_order=True) for c in columns]
            )
            index = {
                tuple(row[k] for k in keys): {c: list(row[c]) for c in columns}
                for row in grouped.iter_rows(named=True)
            }
            self._shared.indexes[cache_key] = index
        return index

    def lookup(self, image: "Image | str", columns: Sequence[str]) -> dict[str, str]:
        """Return one value per requested column for *image*.

        Args:
            image: The image (its ``name`` is used) or an image name.
            columns: Column names; each resolves literally first, then by its
                ``Metadata_`` spelling.

        Returns:
            ``{requested_name: value}``.

        Raises:
            ReferenceTableError: A requested column is not in the table.
            ReferenceLookupError: No row matches (``reason="unmatched"``), a
                column holds only empty values (``"null"``), or its rows
                disagree (``"ambiguous"``).
        """
        name = image if isinstance(image, str) else image.name
        resolved: list[str] = []
        for requested in columns:
            column = self._resolve_column(requested)
            if column is None:
                raise ReferenceTableError(
                    f"Reference metadata has no column {requested!r}; "
                    f"columns are {list(self.columns)}"
                )
            resolved.append(column)
        keys = self._key_columns()
        key = (self.dataset, name) if len(keys) == 2 else (name,)
        entry = self._index(keys, tuple(resolved)).get(key)
        where = f" in dataset {self.dataset!r}" if len(keys) == 2 else ""
        if entry is None:
            raise ReferenceLookupError(
                f"No row in the reference metadata for image {name!r}{where}",
                reason="unmatched",
                image_name=name,
            )
        values: dict[str, str] = {}
        for requested, column in zip(columns, resolved):
            found = entry[column]
            if not found:
                raise ReferenceLookupError(
                    f"{column} is empty for image {name!r}{where}",
                    reason="null",
                    image_name=name,
                    column=column,
                )
            if len(found) > 1:
                raise ReferenceLookupError(
                    f"{column} disagrees across the rows for image {name!r}{where}: "
                    f"{sorted(found)!r}",
                    reason="ambiguous",
                    image_name=name,
                    column=column,
                )
            values[requested] = found[0]
        return values

    # ---------------------------------------------------- reference images
    def resolve_image(self, name: str) -> "Path | Image":
        """Return the in-memory image or the file that *name* refers to.

        Raises:
            ReferenceImageError: No ``images`` entry and no ``image_root``, or
                the name matches zero or several files in ``image_root``.
        """
        if name in self.images:
            target = self.images[name]
            return Path(target) if isinstance(target, (str, Path)) else target
        if self.image_root is None:
            raise ReferenceImageError(
                f"Cannot resolve reference image {name!r}: the ReferenceContext has "
                f"no image_root and no images entry for it"
            )
        root = self.image_root
        exact = root / name
        if exact.is_file():
            return exact
        candidates = sorted(p for p in root.iterdir() if p.is_file() and p.stem == name)
        if len(candidates) != 1:
            raise ReferenceImageError(
                f"Reference image {name!r} matches {len(candidates)} files in {root}: "
                f"{[c.name for c in candidates]}"
            )
        return candidates[0]

    def _load(self, name: str) -> "tuple[Image, str | None]":
        target = self.resolve_image(name)
        if not isinstance(target, Path):
            return target, None
        stat = target.stat()
        key = (
            str(target.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            json.dumps(self.read_kwargs, sort_keys=True, default=str),
        )
        cached = _IMAGE_CACHE.get(key)
        if cached is None:
            try:
                image = _read_image(target, self.read_kwargs)
            except Exception as exc:  # noqa: BLE001 -- surface as the reference failure
                raise ReferenceImageError(f"Cannot read reference image {target}: {exc}") from exc
            cached = (image, hashlib.sha256(target.read_bytes()).hexdigest())
            _IMAGE_CACHE[key] = cached
            while len(_IMAGE_CACHE) > _IMAGE_CACHE_SIZE:
                _IMAGE_CACHE.popitem(last=False)
        else:
            _IMAGE_CACHE.move_to_end(key)
        return cached

    def load_image(self, name: str) -> "Image":
        """Load the reference image *name* (cached per process; do not modify it)."""
        return self._load(name)[0]

    def reference_image_digest(self, name: str) -> str | None:
        """SHA-256 of the reference image file, or ``None`` for an in-memory image."""
        return self._load(name)[1]

    # ------------------------------------------------------------ derivation
    def narrow(
        self,
        *,
        dataset: str | None = None,
        image_root: str | Path | None = None,
        images: Mapping[str, Any] | None = None,
    ) -> "ReferenceContext":
        """Return a context sharing this table and its indexes.

        Each argument left ``None`` keeps this context's value.
        """
        clone = object.__new__(ReferenceContext)
        clone._shared = self._shared
        clone.dataset = dataset if dataset is not None else self.dataset
        clone.image_root = Path(image_root) if image_root is not None else self.image_root
        clone.images = dict(images) if images is not None else dict(self.images)
        clone.read_kwargs = dict(self.read_kwargs)
        clone._tokens = []
        return clone

    def __repr__(self) -> str:
        return (
            f"ReferenceContext(rows={self._shared.table.height}, dataset={self.dataset!r}, "
            f"image_root={str(self.image_root) if self.image_root else None!r})"
        )
```

Note: `ReferenceContext` is documented as single-threaded per instance — entering one instance concurrently from two threads would interleave its token stack. Say so in the class docstring's last paragraph:

```text
    A context is per-process and per-thread: worker processes build their own,
    and one instance must not be entered concurrently from two threads.
```

- [ ] **Step 4: Export it**

In `src/phenotypic/__init__.py` add to `_LAZY_CLASSES`:

```python
    "ReferenceContext": "._core._reference_context",
```

add `from ._core._reference_context import ReferenceContext` beside the `TYPE_CHECKING`-guarded imports of `GridImage`/`ImagePipeline` (line ~92, same block), and `"ReferenceContext",` to `__all__`.

In `src/phenotypic/sdk_/__init__.py` add to `_LAZY_ATTRS`:

```python
    "RefMetadataUnavailableError": "phenotypic._core._reference_context",
    "ReferenceContextError": "phenotypic._core._reference_context",
    "ReferenceImageError": "phenotypic._core._reference_context",
    "ReferenceLookupError": "phenotypic._core._reference_context",
    "ReferenceTableError": "phenotypic._core._reference_context",
```

and the same five names to its `__all__`. (Verify `sdk_.__getattr__` resolves an absolute module name: it calls `importlib.import_module(module_name, __name__)`; an absolute name ignores the package argument. If it instead concatenates, use `".._core._reference_context"`.)

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/unit/core/test_reference_context.py -q`
Expected: all pass. If `test_per_colony_rows_collapse_to_one_value` fails because `normalize_metadata_columns` renames or rejects a known non-metadata header such as `Grid_RowNum`, shield those headers the way the CLI join does (`_normalize_selected_metadata_columns`, `_cli/_metadata_join.py:47`, using `external_metadata_preserved_columns`) — move that shielding into `sdk_` if `_core` would otherwise import `_cli`. If `test_bare_headers_resolve_like_prefixed_ones` fails because `normalize_metadata_columns` leaves `BlankImage` bare, keep the test and make `_resolve_column` try `ensure_metadata_prefix` on the *table* side too (match `ensure_metadata_prefix(col) == ensure_metadata_prefix(name)`); do not weaken the test.

Also run the doctest: `uv run pytest --doctest-modules src/phenotypic/_core/_reference_context.py -q` — Expected: pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_reference_context.py src/phenotypic/__init__.py src/phenotypic/sdk_/__init__.py tests/unit/core/test_reference_context.py
git add src/phenotypic/_core/_reference_context.py src/phenotypic/__init__.py src/phenotypic/sdk_/__init__.py tests/unit/core/test_reference_context.py
git commit -m "feat(core): public ReferenceContext for per-image reference metadata"
```

---

### Task 2: `RefColumn` markers, `RefMetadata` mixin, `ImagePipeline.reference_columns()`, tune refusal

**Files:**
- Modify: `src/phenotypic/sdk_/_column_ref.py`
- Modify: `src/phenotypic/sdk_/__init__.py` (export `RefColumn`, `RefImageColumn`)
- Create: `src/phenotypic/abc_/_ref_metadata.py`
- Modify: `src/phenotypic/abc_/__init__.py` (eager import + `__all__`)
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` (new method)
- Modify: `src/phenotypic/tune/_engine.py` (`TuningEngine.__init__`)
- Test: `tests/unit/abc_/test_ref_metadata.py`, `tests/unit/tune/test_reference_refusal.py`

**Interfaces:**
- Consumes: Task 1 (`ReferenceContext.current`, `.lookup`, `.load_image`, `.reference_image_digest`, `.table_sha256`, `RefMetadataUnavailableError`).
- Produces:
  - `phenotypic.sdk_.RefColumn = Annotated[str, _ColumnRefMarker("reference_metadata")]`
  - `phenotypic.sdk_.RefImageColumn = Annotated[str, _ColumnRefMarker("reference_metadata"), _ReferenceImageMarker()]`
  - `ColumnSource` gains `"reference_metadata"`
  - `phenotypic.abc_.RefMetadata` with `_ref_columns() -> tuple[str, ...]`, `_ref_image_columns() -> tuple[str, ...]`, `_ref_values(image) -> dict[str, str]`, `_ref_image(name: str) -> Image`, `provenance_parameters() -> dict` (adds `"_references"`)
  - `ImagePipeline.reference_columns(*, images_only: bool = False) -> dict[str, tuple[str, ...]]` keyed by `"/".join(tree_path)`
  - `phenotypic.tune._engine._refuse_reference_metadata(pipeline) -> None` (raises `ValueError`)

- [ ] **Step 1: Write the failing tests** (`tests/unit/abc_/test_ref_metadata.py`)

```python
"""RefMetadata: column discovery, context requirement, provenance, tree walk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic import Image, ImagePipeline, ReferenceContext
from phenotypic._core._provenance import current_application_operations
from phenotypic._core._reference_context import RefMetadataUnavailableError
from phenotypic.abc_ import ImageEnhancer, RefMetadata
from phenotypic.enhance import CompositeEnhance
from phenotypic.sdk_ import RefColumn, RefImageColumn

SEEN: list[dict] = []


class _ReadsStrain(ImageEnhancer, RefMetadata):
    """Test op: records what it read."""

    strain_column: RefColumn = "Metadata_Strain"
    blank_column: RefImageColumn = "Metadata_BlankImage"

    def _operate(self, image):
        SEEN.append(self._ref_values(image))
        return image


def _image(name: str = "t04") -> Image:
    return Image(arr=np.full((6, 6), 0.5, dtype=np.float32), name=name)


def _layout() -> pd.DataFrame:
    return pd.DataFrame({
        "Metadata_ImageName": ["t04"],
        "Metadata_Strain": ["WT"],
        "Metadata_BlankImage": ["t00"],
    })


def test_columns_are_discovered_from_marked_fields():
    op = _ReadsStrain(strain_column="Metadata_Strain")
    assert op._ref_columns() == ("Metadata_Strain", "Metadata_BlankImage")
    assert op._ref_image_columns() == ("Metadata_BlankImage",)


def test_without_context_the_error_names_both_fixes():
    with pytest.raises(RefMetadataUnavailableError) as info:
        _ReadsStrain().apply(_image())
    message = str(info.value)
    assert "_ReadsStrain" in message
    assert "with phenotypic.ReferenceContext(" in message
    assert "--metadata" in message


def test_with_context_the_op_reads_its_own_columns():
    SEEN.clear()
    with ReferenceContext(_layout()):
        _ReadsStrain().apply(_image())
    assert SEEN == [{"Metadata_Strain": "WT", "Metadata_BlankImage": "t00"}]


def test_provenance_records_the_resolved_values():
    with ReferenceContext(_layout()):
        out = ImagePipeline(ops={"reads": _ReadsStrain()}).apply(_image())
    records = [
        r for r in current_application_operations(out)
        if r["operation_name"] == "_ReadsStrain"
    ]
    assert records[-1]["parameters"]["_references"]["values"] == {
        "Metadata_Strain": "WT",
        "Metadata_BlankImage": "t00",
    }
    assert records[-1]["parameters"]["blank_column"] == "Metadata_BlankImage"


def test_only_image_operations_may_mix_it_in():
    with pytest.raises(TypeError, match="ImageOperation"):
        type("Bad", (RefMetadata,), {})


def test_pipeline_reports_reference_columns_tree_wide():
    pipe = ImagePipeline(ops={"comp": CompositeEnhance(ops=[_ReadsStrain()])})
    found = pipe.reference_columns()
    assert list(found.values()) == [("Metadata_Strain", "Metadata_BlankImage")]
    assert next(iter(found)).startswith("comp")
    assert list(pipe.reference_columns(images_only=True).values()) == [("Metadata_BlankImage",)]
    assert ImagePipeline(ops={}).reference_columns() == {}
```

And `tests/unit/tune/test_reference_refusal.py`:

```python
from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune._engine import _refuse_reference_metadata
from tests.unit.abc_.test_ref_metadata import _ReadsStrain


def test_tune_refuses_reference_metadata_pipelines():
    with pytest.raises(ValueError, match="reference metadata"):
        _refuse_reference_metadata(ImagePipeline(ops={"r": _ReadsStrain(), "d": OtsuDetector()}))


def test_tune_accepts_ordinary_pipelines():
    _refuse_reference_metadata(ImagePipeline(ops={"b": BlurGauss(), "d": OtsuDetector()}))
```

(If `tests.unit.abc_` is not importable as a package, copy the `_ReadsStrain` class into this file instead of importing it.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/abc_/test_ref_metadata.py tests/unit/tune/test_reference_refusal.py -q`
Expected: `ImportError: cannot import name 'RefMetadata'`

- [ ] **Step 3: Markers** — in `src/phenotypic/sdk_/_column_ref.py`:

```python
ColumnSource = Literal["measurements", "master_measurements", "reference_metadata"]


class _ReferenceImageMarker:
    """Sentinel: this reference column's value names an image to load."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_ReferenceImageMarker()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReferenceImageMarker)

    def __hash__(self) -> int:
        return hash("_ReferenceImageMarker")


#: A column of the run's reference metadata table (``ReferenceContext``),
#: read per image by a :class:`~phenotypic.abc_.RefMetadata` operation.
#: The GUI renders it as a dropdown of the active table's headers.
RefColumn = Annotated[str, _ColumnRefMarker("reference_metadata")]

#: A reference column whose value names another image (e.g. a media blank).
#: The CLI resolves these to files at startup and in its preflight.
RefImageColumn = Annotated[
    str, _ColumnRefMarker("reference_metadata"), _ReferenceImageMarker()
]
```

Add `"RefColumn"`, `"RefImageColumn"`, `"_ReferenceImageMarker"` to that module's `__all__`, and re-export `RefColumn`, `RefImageColumn` from `phenotypic.sdk_` next to the existing `ColumnRef` export (same import line style; add both to `sdk_.__all__`).

Check `src/phenotypic/_gui/_schema_cache.py:45` `_FILES_BY_SOURCE`: if it is typed `dict[ColumnSource, …]` and a test asserts it covers every `ColumnSource`, exclude `"reference_metadata"` explicitly there with a comment ("not a measurements file; the builder supplies its own provider"). Run `uv run pytest tests/unit/gui/test_schema_cache.py -q` to confirm.

- [ ] **Step 4: Mixin** — create `src/phenotypic/abc_/_ref_metadata.py`:

```python
"""Capability mixin: an operation that reads per-image reference metadata."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic._core._image import Image

__all__ = ["RefMetadata"]

#: What each op resolved during its current apply, keyed by ``id(op)``, read
#: once by ``provenance_parameters`` right after the op succeeds.
_RESOLVED: ContextVar[dict[int, dict[str, Any]] | None] = ContextVar(
    "phenotypic_ref_metadata_resolved", default=None
)


def _resolved() -> dict[int, dict[str, Any]]:
    store = _RESOLVED.get()
    if store is None:
        store = {}
        _RESOLVED.set(store)
    return store


def _marked_fields(cls: type, marker_type: type, *, source: str | None = None) -> tuple[str, ...]:
    names: list[str] = []
    for name, info in cls.model_fields.items():  # type: ignore[attr-defined]
        for marker in info.metadata:
            if isinstance(marker, marker_type) and (
                source is None or getattr(marker, "source", None) == source
            ):
                names.append(name)
                break
    return tuple(names)


class RefMetadata:
    """Mark an operation that reads per-image values from a ReferenceContext.

    Fieldless, like :class:`~phenotypic.abc_.plotting.PlotImage`. A subclass
    declares its columns as ordinary fields typed
    :data:`~phenotypic.sdk_.RefColumn` (a value) or
    :data:`~phenotypic.sdk_.RefImageColumn` (a value naming another image),
    then calls :meth:`_ref_values` and :meth:`_ref_image` inside ``_operate``.

    The operation never holds a table path: the table is supplied when the
    pipeline runs — ``with phenotypic.ReferenceContext(...)`` in Python, the
    CLI's ``--metadata``, or the GUI's reference-metadata picker — so using
    metadata is always a deliberate act at run time.

    Raises:
        TypeError: At class definition, when the subclass is not an
            ``ImageOperation``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        from phenotypic.abc_._image_operation import ImageOperation

        if not issubclass(cls, ImageOperation):
            raise TypeError(
                f"{cls.__name__}: RefMetadata may only be mixed into an ImageOperation"
            )

    def _ref_columns(self) -> tuple[str, ...]:
        """Every column this op reads, in field declaration order."""
        from phenotypic.sdk_._column_ref import _ColumnRefMarker

        fields = _marked_fields(type(self), _ColumnRefMarker, source="reference_metadata")
        return tuple(getattr(self, name) for name in fields)

    def _ref_image_columns(self) -> tuple[str, ...]:
        """The subset of :meth:`_ref_columns` whose values name images."""
        from phenotypic.sdk_._column_ref import _ReferenceImageMarker

        fields = _marked_fields(type(self), _ReferenceImageMarker)
        return tuple(getattr(self, name) for name in fields)

    def _require_context(self):
        from phenotypic._core._reference_context import (
            ReferenceContext,
            RefMetadataUnavailableError,
        )

        ctx = ReferenceContext.current()
        if ctx is None:
            raise RefMetadataUnavailableError(
                f"{type(self).__name__} reads {self._ref_columns()} from a "
                f"ReferenceContext, but none is active.\n"
                f"  Python: with phenotypic.ReferenceContext('layout.csv', "
                f"image_root='images/'): pipe.apply(img)\n"
                f"  CLI:    pass --metadata layout.csv"
            )
        return ctx

    def _ref_values(self, image: "Image") -> dict[str, str]:
        """Look up this op's columns for *image* in the active context."""
        ctx = self._require_context()
        values = ctx.lookup(image, self._ref_columns())
        _resolved()[id(self)] = {
            "table_sha256": ctx.table_sha256,
            "values": dict(values),
            "images": {},
        }
        return values

    def _ref_image(self, name: str) -> "Image":
        """Load the reference image *name* through the active context."""
        ctx = self._require_context()
        image = ctx.load_image(name)
        record = _resolved().setdefault(
            id(self), {"table_sha256": ctx.table_sha256, "values": {}, "images": {}}
        )
        record["images"][name] = {"sha256": ctx.reference_image_digest(name)}
        return image

    def provenance_parameters(self) -> dict[str, Any]:
        """Parameters for the provenance journal, plus what this apply resolved."""
        params = self.model_dump(mode="json")  # type: ignore[attr-defined]
        record = _resolved().pop(id(self), None)
        if record is not None:
            params["_references"] = record
        return params
```

In `src/phenotypic/abc_/__init__.py` add `from ._ref_metadata import RefMetadata  # noqa: E402` with the other eager imports and `"RefMetadata",` to `__all__`.

- [ ] **Step 5: Pipeline method** — in `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`, on the class that defines `apply` (line ~947), add:

```python
    def reference_columns(self, *, images_only: bool = False) -> dict[str, tuple[str, ...]]:
        """Metadata columns each reference-metadata operation reads, tree-wide.

        Args:
            images_only: Return only columns whose values name images.

        Returns:
            ``{tree_path: columns}``, the path spelled as ``find_operations``
            spells it and joined with ``/``. Empty when the pipeline needs no
            reference metadata.
        """
        from phenotypic.abc_._ref_metadata import RefMetadata
        from phenotypic.sdk_._operation_tree import find_operations

        found: dict[str, tuple[str, ...]] = {}
        for path, op in find_operations(self, lambda o: isinstance(o, RefMetadata)):
            columns = op._ref_image_columns() if images_only else op._ref_columns()
            if columns:
                found["/".join(path)] = columns
        return found
```

- [ ] **Step 6: Tune refusal** — in `src/phenotypic/tune/_engine.py`:

```python
def _refuse_reference_metadata(pipeline: "ImagePipeline") -> None:
    """Refuse a pipeline whose operations read reference metadata.

    The tune evaluator does not enter a ReferenceContext yet, so such an op
    would fail on every trial; refusing up front names the cause once.
    """
    needs = pipeline.reference_columns()
    if needs:
        where = ", ".join(f"{path} {cols}" for path, cols in needs.items())
        raise ValueError(
            "phenotypic-tune cannot tune a pipeline whose operations read reference "
            f"metadata yet: {where}"
        )
```

and call `_refuse_reference_metadata(spec.pipeline)` as the first line of `TuningEngine.__init__`.

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/unit/abc_/test_ref_metadata.py tests/unit/tune/test_reference_refusal.py tests/unit/gui/test_schema_cache.py -q`
Expected: pass.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/_column_ref.py src/phenotypic/sdk_/__init__.py src/phenotypic/abc_/_ref_metadata.py src/phenotypic/abc_/__init__.py src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py src/phenotypic/tune/_engine.py tests/unit/abc_/test_ref_metadata.py tests/unit/tune/test_reference_refusal.py
git add -A src/phenotypic/sdk_ src/phenotypic/abc_ src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py src/phenotypic/tune/_engine.py tests/unit/abc_/test_ref_metadata.py tests/unit/tune/test_reference_refusal.py
git commit -m "feat(abc): RefMetadata mixin, RefColumn markers, pipeline.reference_columns"
```

---

### Task 3: `SubtractBlank`

**Files:**
- Create: `src/phenotypic/enhance/_subtract_blank.py`
- Modify: `src/phenotypic/enhance/__init__.py` (export `SubtractBlank`)
- Modify: `src/phenotypic/sdk_/__init__.py` (`_LAZY_ATTRS["StaleDetectMatError"] = "phenotypic.enhance._subtract_blank"`, `__all__`)
- Test: `tests/unit/enhance/test_subtract_blank.py`

**Interfaces:**
- Consumes: Task 1 (`ReferenceContext`, `ReferenceImageError`, `ReferenceLookupError`), Task 2 (`RefMetadata`, `RefImageColumn`).
- Produces: `phenotypic.enhance.SubtractBlank(blank_column: RefImageColumn = "Metadata_BlankImage", polarity: Literal["brighter","darker","both"] = "brighter")`; `phenotypic.enhance._subtract_blank.StaleDetectMatError(ValueError)`.

- [ ] **Step 1: Write the failing tests**

```python
"""SubtractBlank: per-polarity arithmetic, detect-mode matching, guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic import GridImage, Image, ImagePipeline, ReferenceContext
from phenotypic._core._image_parts.detection_modes import get_detection_mode
from phenotypic._core._reference_context import ReferenceImageError, ReferenceLookupError
from phenotypic.abc_ import ImageCorrector
from phenotypic.enhance import BlurGauss, SetDetectMode, SubtractBlank
from phenotypic.enhance._subtract_blank import StaleDetectMatError


def _gray(values: np.ndarray, name: str) -> Image:
    return Image(arr=values.astype(np.float32), name=name)


def _pair():
    blank = np.full((8, 8), 0.40, dtype=np.float32)
    frame = blank.copy()
    frame[1:3, 1:3] = 0.85   # white mycelium
    frame[5:7, 5:7] = 0.15   # dark pigmented colony
    return _gray(frame, "t04"), _gray(blank, "t00")


def _ctx(target_name="t04", blank: Image | None = None, blank_name="t00"):
    layout = pd.DataFrame({"Metadata_ImageName": [target_name], "Metadata_BlankImage": [blank_name]})
    return ReferenceContext(layout, images={blank_name: blank} if blank is not None else None)


@pytest.mark.parametrize(
    "polarity,bright,dark",
    [("brighter", 0.45, 0.0), ("darker", 0.0, 0.25), ("both", 0.45, 0.25)],
)
def test_polarity_arithmetic(polarity, bright, dark):
    target, blank = _pair()
    with _ctx(blank=blank):
        out = SubtractBlank(polarity=polarity).apply(target)
    dm = out.detect_mat[:]
    assert dm[1, 1] == pytest.approx(bright, abs=1e-6)
    assert dm[5, 5] == pytest.approx(dark, abs=1e-6)
    assert dm[0, 7] == pytest.approx(0.0, abs=1e-6)   # bare agar cancels


def test_rgb_and_gray_are_untouched():
    target, blank = _pair()
    gray_before = target.gray[:].copy()
    with _ctx(blank=blank):
        out = SubtractBlank().apply(target)
    np.testing.assert_array_equal(out.gray[:], gray_before)


def test_blank_is_taken_in_the_targets_detect_mode():
    rng = np.random.default_rng(0)
    blank_rgb = rng.integers(0, 120, size=(8, 8, 3), dtype=np.uint8)
    frame_rgb = blank_rgb.copy()
    frame_rgb[2:5, 2:5] = 230
    target = Image(arr=frame_rgb, name="t04")
    blank = Image(arr=blank_rgb, name="t00")
    target.set_detect_mode("LabL")
    mode = get_detection_mode("LabL")
    expected = np.clip(mode.compute(target) - mode.compute(blank), 0.0, 1.0)
    with _ctx(blank=blank):
        out = SubtractBlank().apply(target)
    np.testing.assert_allclose(out.detect_mat[:], expected, atol=1e-6)


def test_runs_after_set_detect_mode_following_an_enhancer():
    target, blank = _pair()
    pipe = ImagePipeline(ops={"blur": BlurGauss(sigma=1.0), "mode": SetDetectMode(mode="gray"), "sb": SubtractBlank()})
    with _ctx(blank=blank):
        pipe.apply(target)


def test_refuses_after_an_enhancer():
    target, blank = _pair()
    pipe = ImagePipeline(ops={"blur": BlurGauss(sigma=1.0), "sb": SubtractBlank()})
    with _ctx(blank=blank), pytest.raises(StaleDetectMatError, match="SetDetectMode"):
        pipe.apply(target)


class _NoopCorrector(ImageCorrector):
    """A corrector that changes nothing; its presence alone must trip the guard."""

    def _operate(self, image):
        return image


def test_refuses_after_a_corrector():
    target, blank = _pair()
    pipe = ImagePipeline(ops={"fix": _NoopCorrector(), "sb": SubtractBlank()})
    with _ctx(blank=blank), pytest.raises(StaleDetectMatError, match="_NoopCorrector"):
        pipe.apply(target)


def test_refuses_self_reference():
    target, _ = _pair()
    with _ctx(blank_name="t04", blank=target), pytest.raises(ReferenceLookupError) as info:
        SubtractBlank().apply(target)
    assert info.value.reason == "self"


def test_refuses_shape_mismatch():
    target, _ = _pair()
    small = _gray(np.zeros((4, 4)), "t00")
    with _ctx(blank=small), pytest.raises(ReferenceImageError, match="shape"):
        SubtractBlank().apply(target)


# Review Focus 5
def test_grid_image_target():
    rgb = np.full((48, 48, 3), 30, dtype=np.uint8)
    frame = rgb.copy()
    frame[10:14, 10:14] = 220
    target = GridImage(arr=frame, name="t04")
    blank = Image(arr=rgb, name="t00")
    with _ctx(blank=blank):
        out = SubtractBlank().apply(target)
    assert out.detect_mat[:][11, 11] > 0.5
    assert out.detect_mat[:][40, 40] == pytest.approx(0.0, abs=1e-6)


def test_round_trips_through_json():
    pipe = ImagePipeline(ops={"sb": SubtractBlank(blank_column="Metadata_Frame0", polarity="darker")})
    again = ImagePipeline.from_json(pipe.to_json())
    op = again.get_ops()["sb"]
    assert (op.blank_column, op.polarity) == ("Metadata_Frame0", "darker")
```

(Check how `ImagePipeline` exposes its ops — `get_ops()` returns the dict, as `get_meas`/`get_post` do per `_operation_tree._slot_child`. If the accessor differs, use the one `test_cli_preflight_ordering.py` fixtures rely on.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/enhance/test_subtract_blank.py -q`
Expected: `ImportError: cannot import name 'SubtractBlank'`

- [ ] **Step 3: Implement `src/phenotypic/enhance/_subtract_blank.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from phenotypic.abc_ import RefMetadata
from phenotypic.abc_._enhance_markers._background_subtraction import BackgroundSubtraction
from phenotypic.sdk_ import RefImageColumn

if TYPE_CHECKING:
    from phenotypic import Image


class StaleDetectMatError(ValueError):
    """SubtractBlank ran on a detect_mat or image that its raw blank does not match."""


def _corrector_class_names() -> set[str]:
    from phenotypic.abc_ import ImageCorrector

    seen: set[type] = set()
    stack: list[type] = [ImageCorrector]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return {f"{c.__module__}.{c.__qualname__}" for c in seen}


class SubtractBlank(BackgroundSubtraction, RefMetadata):
    """Subtract a time series' media-blank frame from ``detect_mat``.

    Each image names its blank — typically the plate's frame 0, imaged before
    inoculation — in a metadata column. The blank is read raw, taken in the
    target's detection mode, and subtracted pixel by pixel, so the agar,
    lid glare and scanner vignetting shared by every frame of that plate
    cancel and only growth since frame 0 remains.

    The metadata table is supplied at run time, never stored on the
    operation: ``with phenotypic.ReferenceContext(table, image_root=...)`` in
    Python, ``--metadata`` on the CLI, or the GUI's reference-metadata picker.

    The blank is raw, so ``detect_mat`` must be raw too: place SubtractBlank
    before any enhancer, or directly after a ``SetDetectMode``, and after no
    ``ImageCorrector``. It refuses otherwise. A later ``SetDetectMode``
    discards the subtraction, like any enhancement.

    Best For:
        - Time-lapse plates with a media-only frame taken before growth.
        - Filamentous colonies whose faint mycelium is lost against uneven
          agar when the background is estimated from the image itself.

    Consider Also:
        - :class:`SubtractGaussian` or :class:`SubtractRollingBall` when no
          blank frame exists and the background must be estimated from the
          image alone.

    Args:
        blank_column: Metadata column holding each image's blank, as a file
            stem or file name in the same input directory. Default
            ``"Metadata_BlankImage"``.
        polarity: Which change from the blank counts as colony.
            ``"brighter"`` keeps pixels brighter than the blank (white
            mycelium on darker agar); ``"darker"`` keeps pixels darker than
            the blank and flips them bright (pigmented colonies);
            ``"both"`` keeps the absolute difference.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the clipped
        difference in ``[0, 1]``. ``rgb`` and ``gray`` are unchanged.

    Raises:
        RefMetadataUnavailableError: No ReferenceContext is active.
        ReferenceLookupError: The image's blank is missing, empty, ambiguous,
            or the image itself.
        ReferenceImageError: The blank cannot be resolved or read, or its
            shape or bit depth differs from the target's.
        StaleDetectMatError: ``detect_mat`` was already enhanced, or an
            ``ImageCorrector`` ran earlier in this pipeline.

    Examples:
        A frame identical to its blank cancels to zero:

        >>> import pandas as pd
        >>> from phenotypic import ReferenceContext
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import SubtractBlank
        >>> plate = load_synth_yeast_plate()
        >>> plate.name = "plate1_t04"
        >>> blank = load_synth_yeast_plate()   # stands in for the media-only frame
        >>> layout = pd.DataFrame({"Metadata_ImageName": ["plate1_t04"],
        ...                        "Metadata_BlankImage": ["plate1_t00"]})
        >>> with ReferenceContext(layout, images={"plate1_t00": blank}):
        ...     out = SubtractBlank().apply(plate)
        >>> float(out.detect_mat[:].max())
        0.0
    """

    blank_column: RefImageColumn = "Metadata_BlankImage"
    polarity: Literal["brighter", "darker", "both"] = "brighter"

    def _operate(self, image: "Image") -> "Image":
        from phenotypic._core._image_parts.detection_modes import get_detection_mode
        from phenotypic._core._reference_context import (
            ReferenceImageError,
            ReferenceLookupError,
        )

        name = self._ref_values(image)[self.blank_column]
        if name == image.name:
            raise ReferenceLookupError(
                f"Image {image.name!r} names itself as its blank in {self.blank_column}; "
                f"leave blank frames out of the input",
                reason="self",
                image_name=image.name,
                column=self.blank_column,
            )
        mode = get_detection_mode(image.detect_mode)
        self._require_raw_target(image, mode)
        blank = self._ref_image(name)
        if blank.gray.shape != image.gray.shape:
            raise ReferenceImageError(
                f"Blank {name!r} has shape {blank.gray.shape}, image {image.name!r} "
                f"has {image.gray.shape}; frames must be pixel-aligned"
            )
        if blank.bit_depth != image.bit_depth:
            raise ReferenceImageError(
                f"Blank {name!r} is {blank.bit_depth}-bit, image {image.name!r} "
                f"is {image.bit_depth}-bit"
            )
        background = mode.compute(blank)
        target = image.detect_mat[:]
        if self.polarity == "brighter":
            result = np.clip(target - background, 0.0, 1.0)
        elif self.polarity == "darker":
            result = np.clip(background - target, 0.0, 1.0)
        else:
            result = np.abs(target - background)
        image.detect_mat[:] = result.astype(target.dtype, copy=False)
        return image

    @staticmethod
    def _require_raw_target(image: "Image", mode) -> None:
        journal = image._metadata.provenance_journal
        applications = journal.get("applications") or []
        operations = applications[-1]["operations"] if applications else []
        correctors = _corrector_class_names()
        for record in operations:
            if record.get("operation_class") in correctors:
                raise StaleDetectMatError(
                    f"SubtractBlank follows {record.get('operation_name')}, an "
                    f"ImageCorrector; the raw blank does not share its correction. "
                    f"Place SubtractBlank before any corrector."
                )
        if not np.array_equal(image.detect_mat[:], mode.compute(image)):
            raise StaleDetectMatError(
                "detect_mat was already enhanced; the raw blank cannot be subtracted "
                "from it. Place SubtractBlank before any enhancer, or directly after "
                "SetDetectMode."
            )
```

Note the limitation in a comment above `_require_raw_target`: records of a corrector that ran inside the *same* composite branch are appended only when the composite finishes, so that case is caught by neither check unless it also changed `detect_mat`; correctors do not live inside composites today.

Export: in `src/phenotypic/enhance/__init__.py` add `from ._subtract_blank import SubtractBlank` beside `SubtractGaussian` and `"SubtractBlank"` to `__all__`. Add the `StaleDetectMatError` lazy entry to `sdk_` as listed above.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/enhance/test_subtract_blank.py -q && uv run pytest --doctest-modules src/phenotypic/enhance/_subtract_blank.py -q`
Expected: pass. If `test_refuses_after_a_corrector` fails because a bare `ImageCorrector` subclass cannot be instantiated (an abstract hook), implement that hook minimally in `_NoopCorrector`.

- [ ] **Step 5: Affected-surface check for the new operation**

Run: `uv run pytest tests/unit/tune/test_annotation_coverage.py tests/unit/gui/test_operation_registry.py tests/unit/test_pickleable.py -q`
Expected: pass (the new op's only numeric-free fields need no `TuneSpec`; the registry must discover `SubtractBlank`; the op must pickle for loky workers).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/enhance/_subtract_blank.py src/phenotypic/enhance/__init__.py src/phenotypic/sdk_/__init__.py tests/unit/enhance/test_subtract_blank.py
git add src/phenotypic/enhance/_subtract_blank.py src/phenotypic/enhance/__init__.py src/phenotypic/sdk_/__init__.py tests/unit/enhance/test_subtract_blank.py
git commit -m "feat(enhance): SubtractBlank subtracts a time series' media-blank frame"
```

**Phase 1 gate:** `uv run pytest tests/unit/core tests/unit/abc_ tests/unit/enhance tests/unit/tune/test_reference_refusal.py tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py -q` once. Expected: pass.

---

## Phase 2 — CLI

### Task 4: Reference planner, manifest, path helpers, worker context

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`
- Create: `src/phenotypic/_cli/_cli_reference.py`
- Test: `tests/unit/cli/test_cli_reference.py`

**Interfaces:**
- Consumes: Task 1, Task 2 (`ImagePipeline.reference_columns`), `Dataset` (`_cli_types.py:27`: `name`, `images`, `input_dir`, `output_dir`), `file_sha256` (`_cli_failure_tracker.py:86`), `canonical_digest` (`sdk_/_digests.py:46`), `atomic_write_bytes`/`atomic_write_json` (`sdk_/_atomic_io.py`).
- Produces (all in `phenotypic._cli._cli_reference` unless noted):
  - `phenotypic.sdk_._io_constants`: `REFERENCE_METADATA_CSV = "reference_metadata.csv"`, `REFERENCE_MANIFEST_JSON = "reference_manifest.json"`, `reference_metadata_snapshot_path(output_dir) -> Path`, `reference_manifest_path(output_dir) -> Path`
  - `@dataclass(frozen=True) class ReferencePlan: total_images: int; unmatched: tuple[str, ...]; ambiguous: tuple[str, ...]; self_referenced: tuple[str, ...]; unresolved: tuple[str, ...]; images_by_dataset: dict[str, dict[str, str]]; digests: dict[str, dict[str, str]]` (labels are `"<dataset>/<stem>"`)
  - `plan_references(context, pipeline, datasets, *, hash_images: bool) -> ReferencePlan`
  - `resolve_reference_table_path(config, output_dir: Path | None) -> Path | None`
  - `input_read_kwargs(config) -> dict[str, Any]`
  - `snapshot_reference_metadata(output_dir: Path, source: Path | None) -> Path | None`
  - `write_reference_manifest(output_dir, *, plan, table_path: Path, table_sha256: str, read_kwargs: dict) -> Path`
  - `read_reference_manifest(output_dir: Path) -> dict | None`
  - `remove_reference_manifest(output_dir: Path) -> None`
  - `worker_reference_context(output_dir: Path, dataset_name: str | None)` — context manager yielding `ReferenceContext | None`
  - `reference_digest_for(output_dir: Path | None, dataset_name: str, image_stem: str) -> str | None`

- [ ] **Step 1: Write the failing tests**

```python
"""Reference planning, the run manifest, and the worker context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from phenotypic import ImagePipeline, ReferenceContext
from phenotypic._cli import _cli_reference as ref
from phenotypic._cli._cli_types import Dataset
from phenotypic._core._reference_context import ReferenceTableError
from phenotypic.enhance import SubtractBlank
from phenotypic.sdk_._io_constants import reference_manifest_path


def _write(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.full((8, 8), value, dtype=np.uint8))
    return path


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "in" / "plate1"
    for stem, value in (("blank", 20), ("t01", 60), ("t02", 90), ("t03", 90), ("orphan", 5)):
        _write(root / f"{stem}.tif", value)
    table = tmp_path / "layout.csv"
    pd.DataFrame({
        "Metadata_ImageName": ["t01", "t02", "t02", "t03"],
        "Metadata_BlankImage": ["blank", "blank", "other", "t03"],
    }).to_csv(table, index=False)
    dataset = Dataset(
        name="plate1",
        images=[root / f"{s}.tif" for s in ("t01", "t02", "t03", "orphan")],
        input_dir=root,
        output_dir=tmp_path / "out" / "plate1",
    )
    return table, dataset


def _pipe():
    return ImagePipeline(ops={"sb": SubtractBlank()})


def test_plan_classifies_every_image(tree):
    table, dataset = tree
    plan = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=True)
    assert plan.total_images == 4
    assert plan.unmatched == ("plate1/orphan",)
    assert plan.ambiguous == ("plate1/t02",)
    assert plan.self_referenced == ("plate1/t03",)
    assert plan.unresolved == ()
    assert plan.images_by_dataset["plate1"]["blank"].endswith("plate1/blank.tif")
    assert set(plan.digests["plate1"]) == {"t01"}


def test_plan_without_hashing_still_resolves(tree):
    table, dataset = tree
    plan = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=False)
    assert "blank" in plan.images_by_dataset["plate1"]
    assert plan.digests == {"plate1": {}}


def test_digest_changes_only_for_the_image_whose_blank_changed(tree, tmp_path):
    table, dataset = tree
    root = dataset.input_dir
    _write(root / "blank2.tif", 25)
    good = tmp_path / "good.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank", "blank"]}).to_csv(good, index=False)
    edited = tmp_path / "edited.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank2", "blank"]}).to_csv(edited, index=False)
    a = ref.plan_references(ReferenceContext(good), _pipe(), [dataset], hash_images=True).digests["plate1"]
    b = ref.plan_references(ReferenceContext(edited), _pipe(), [dataset], hash_images=True).digests["plate1"]
    assert a["t01"] != b["t01"]
    assert a["t02"] == b["t02"]


def test_manifest_round_trip_and_worker_context(tree, tmp_path):
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=True)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    assert reference_manifest_path(out).is_file()
    with ref.worker_reference_context(out, "plate1") as active:
        assert ReferenceContext.current() is active
        assert active.dataset == "plate1"
        assert active.load_image("blank").gray.shape == (8, 8)
    assert ReferenceContext.current() is None
    assert ref.reference_digest_for(out, "plate1", "t01") == plan.digests["plate1"]["t01"]
    assert ref.reference_digest_for(out, "plate1", "orphan") == "unplanned"


def test_no_manifest_means_no_context_and_no_digest(tmp_path):
    with ref.worker_reference_context(tmp_path, "plate1") as active:
        assert active is None
    assert ref.reference_digest_for(tmp_path, "plate1", "t01") is None
    assert ref.reference_digest_for(None, "plate1", "t01") is None


def test_worker_refuses_a_table_changed_after_planning(tree, tmp_path):
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=False)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    table.write_text(table.read_text() + "t09,blank\n", encoding="utf-8")
    ref._manifest_base_context.cache_clear()
    with pytest.raises(ReferenceTableError, match="changed"):
        with ref.worker_reference_context(out, "plate1"):
            pass


def test_snapshot_copies_bytes_and_reuses_existing(tmp_path, tree):
    table, _ = tree
    out = tmp_path / "out"
    snap = ref.snapshot_reference_metadata(out, table)
    assert snap.read_bytes() == table.read_bytes()
    assert ref.snapshot_reference_metadata(out, None) == snap
    assert ref.snapshot_reference_metadata(tmp_path / "fresh", None) is None


def test_snapshot_is_preserved_on_restart():
    from phenotypic.sdk_._io_constants import REFERENCE_METADATA_CSV, preserved_on_restart_names

    assert REFERENCE_METADATA_CSV in preserved_on_restart_names()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/cli/test_cli_reference.py -q`
Expected: `ModuleNotFoundError: No module named 'phenotypic._cli._cli_reference'`

- [ ] **Step 3: Path helpers** — in `src/phenotypic/sdk_/_io_constants.py`, beside `RESTART_EPOCH_JSON` (line ~719):

```python
#: ``<output>/.phenotypic/reference_metadata.csv`` -- byte-exact snapshot of the
#: ``--metadata`` table for a ``--mode process`` run whose pipeline reads
#: reference metadata (full mode reads ``deliverables/metadata.csv`` instead).
#: Preserved across ``--restart``, as ``deliverables/metadata.csv`` is.
REFERENCE_METADATA_CSV: Final[str] = "reference_metadata.csv"

#: ``<output>/.phenotypic/reference_manifest.json`` -- the run's resolved
#: reference plan (table, reader settings, reference-image paths, per-image
#: digests), rewritten at every startup. Workers build their ReferenceContext
#: from it.
REFERENCE_MANIFEST_JSON: Final[str] = "reference_manifest.json"
```

add `REFERENCE_METADATA_CSV` to the `_PRESERVED_ON_RESTART` set, and beside `restart_epoch_path`:

```python
def reference_metadata_snapshot_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/reference_metadata.csv``."""
    return phenotypic_cache_dir(output_dir) / REFERENCE_METADATA_CSV


def reference_manifest_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/reference_manifest.json``."""
    return phenotypic_cache_dir(output_dir) / REFERENCE_MANIFEST_JSON
```

Add both helpers to the module's `__all__` if it has one.

- [ ] **Step 4: Implement `src/phenotypic/_cli/_cli_reference.py`**

```python
"""Reference metadata for CLI runs: plan once at startup, read in every worker.

The main process resolves every input image's references (Task-5 preflight
calls the same planner without hashing). Startup writes the result to
``.phenotypic/reference_manifest.json``; each worker core enters
:func:`worker_reference_context` around its apply call, so no worker needs a
new argument and SLURM workers need nothing but the run root.
"""

from __future__ import annotations

import functools
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from phenotypic import ImagePipeline, ReferenceContext

    from ._cli_types import Dataset, ExecutionConfig

MANIFEST_SCHEMA_VERSION = 1
UNPLANNED_DIGEST = "unplanned"


@dataclass(frozen=True)
class ReferencePlan:
    """Every input image's reference resolution, classified."""

    total_images: int
    unmatched: tuple[str, ...]
    ambiguous: tuple[str, ...]
    self_referenced: tuple[str, ...]
    unresolved: tuple[str, ...]
    images_by_dataset: dict[str, dict[str, str]]
    digests: dict[str, dict[str, str]]


def _union(columns_by_path: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for columns in columns_by_path.values():
        for column in columns:
            seen.setdefault(column, None)
    return tuple(seen)


def plan_references(
    context: "ReferenceContext",
    pipeline: "ImagePipeline",
    datasets: Sequence["Dataset"],
    *,
    hash_images: bool,
) -> ReferencePlan:
    """Resolve every input image's reference values and reference images.

    Args:
        context: The run's reference table (dataset and root are set per dataset).
        pipeline: The run's pipeline; its ``reference_columns()`` decide what
            is looked up and which values are resolved to files.
        datasets: The scanned inputs.
        hash_images: Hash reference-image files for the per-image digests.
            The preflight passes ``False`` (headers-only contract); startup
            passes ``True``.

    Returns:
        The classified plan. An image appears in at most one failure list.
    """
    from phenotypic._core._reference_context import ReferenceImageError, ReferenceLookupError
    from phenotypic.sdk_._digests import canonical_digest

    from ._cli_failure_tracker import file_sha256

    value_columns = _union(pipeline.reference_columns())
    image_columns = set(_union(pipeline.reference_columns(images_only=True)))
    unmatched: list[str] = []
    ambiguous: list[str] = []
    self_referenced: list[str] = []
    unresolved: list[str] = []
    images_by_dataset: dict[str, dict[str, str]] = {}
    digests: dict[str, dict[str, str]] = {}
    sha_cache: dict[str, str] = {}
    total = 0
    for dataset in datasets:
        scoped = context.narrow(dataset=dataset.name, image_root=dataset.input_dir)
        resolved = images_by_dataset.setdefault(dataset.name, {})
        dataset_digests = digests.setdefault(dataset.name, {})
        for image_path in dataset.images:
            total += 1
            stem = Path(image_path).stem
            label = f"{dataset.name}/{stem}"
            try:
                values = scoped.lookup(stem, value_columns)
            except ReferenceLookupError as exc:
                (unmatched if exc.reason == "unmatched" else ambiguous).append(label)
                continue
            image_shas: dict[str, str] = {}
            failure: list[str] | None = None
            for column in value_columns:
                if column not in image_columns:
                    continue
                name = values[column]
                if name == stem:
                    failure = self_referenced
                    break
                try:
                    target = scoped.resolve_image(name)
                except ReferenceImageError:
                    failure = unresolved
                    break
                key = str(Path(target).resolve())
                resolved[name] = key
                if hash_images:
                    if key not in sha_cache:
                        sha_cache[key] = file_sha256(Path(key))
                    image_shas[name] = sha_cache[key]
            if failure is not None:
                failure.append(label)
                continue
            if hash_images:
                dataset_digests[stem] = canonical_digest({"values": values, "images": image_shas})
    return ReferencePlan(
        total_images=total,
        unmatched=tuple(unmatched),
        ambiguous=tuple(ambiguous),
        self_referenced=tuple(self_referenced),
        unresolved=tuple(unresolved),
        images_by_dataset=images_by_dataset,
        digests=digests,
    )


def resolve_reference_table_path(config: "ExecutionConfig", output_dir: Path | None) -> Path | None:
    """The table a run's reference ops read: ``--metadata``, else the run's snapshot."""
    from phenotypic.sdk_._io_constants import (
        metadata_csv_deliverable_path,
        reference_metadata_snapshot_path,
    )

    if config.metadata_csv is not None:
        return Path(config.metadata_csv)
    if output_dir is None or config.measure_only:
        return None
    snapshot = (
        reference_metadata_snapshot_path(output_dir)
        if config.process_only_layer is not None
        else metadata_csv_deliverable_path(output_dir)
    )
    return snapshot if snapshot.is_file() else None


def input_read_kwargs(config: "ExecutionConfig") -> dict[str, Any]:
    """``Image.imread`` kwargs for reference images: the inputs' reader settings."""
    return {"bit_depth": config.bit_depth} if config.bit_depth else {}


def snapshot_reference_metadata(output_dir: Path, source: Path | None) -> Path | None:
    """Byte-copy *source* to the process-mode snapshot; reuse it when *source* is None."""
    import io

    import pandas as pd

    from phenotypic.sdk_._atomic_io import atomic_write_bytes
    from phenotypic.sdk_._io_constants import reference_metadata_snapshot_path

    destination = reference_metadata_snapshot_path(output_dir)
    if source is None:
        return destination if destination.is_file() else None
    payload = Path(source).read_bytes()
    pd.read_csv(io.BytesIO(payload))  # never replace a valid snapshot with unparseable bytes
    if destination.is_file() and destination.read_bytes() == payload:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, payload)
    if destination.read_bytes() != payload:
        raise OSError(f"Reference metadata snapshot verification failed: {destination}")
    return destination


def write_reference_manifest(
    output_dir: Path,
    *,
    plan: ReferencePlan,
    table_path: Path,
    table_sha256: str,
    read_kwargs: dict[str, Any],
) -> Path:
    """Atomically publish the run's reference manifest."""
    from phenotypic.sdk_._atomic_io import atomic_write_json
    from phenotypic.sdk_._io_constants import reference_manifest_path

    path = reference_manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = {
        name: {"images": plan.images_by_dataset.get(name, {}), "digests": plan.digests.get(name, {})}
        for name in sorted(set(plan.images_by_dataset) | set(plan.digests))
    }
    atomic_write_json(
        path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "table": str(Path(table_path).resolve()),
            "table_sha256": table_sha256,
            "read_kwargs": read_kwargs,
            "datasets": datasets,
        },
    )
    return path


def read_reference_manifest(output_dir: Path) -> dict | None:
    """The run's reference manifest, or ``None`` when the run needs none."""
    from phenotypic.sdk_._io_constants import reference_manifest_path

    path = reference_manifest_path(output_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def remove_reference_manifest(output_dir: Path) -> None:
    """Delete a stale manifest (the pipeline no longer reads reference metadata)."""
    from phenotypic.sdk_._io_constants import reference_manifest_path

    reference_manifest_path(output_dir).unlink(missing_ok=True)


@functools.lru_cache(maxsize=4)
def _manifest_base_context(table: str, table_sha256: str, read_kwargs_json: str) -> "ReferenceContext":
    from phenotypic._core._reference_context import ReferenceContext, ReferenceTableError

    context = ReferenceContext(Path(table), read_kwargs=json.loads(read_kwargs_json))
    if context.table_sha256 != table_sha256:
        raise ReferenceTableError(
            f"Reference table {table} changed since this run planned its references; "
            f"run the same command again to re-plan"
        )
    return context


@contextmanager
def worker_reference_context(output_dir: Path, dataset_name: str | None) -> Iterator["ReferenceContext | None"]:
    """Activate the run's ReferenceContext for one image of *dataset_name*.

    Yields ``None`` (and activates nothing) when the run has no manifest.
    """
    manifest = read_reference_manifest(output_dir)
    if manifest is None:
        yield None
        return
    if dataset_name is None:
        raise ValueError("A run with reference metadata needs the image's dataset name")
    base = _manifest_base_context(
        manifest["table"],
        manifest["table_sha256"],
        json.dumps(manifest["read_kwargs"], sort_keys=True),
    )
    images = manifest["datasets"].get(dataset_name, {}).get("images", {})
    with base.narrow(dataset=dataset_name, images=images) as context:
        yield context


def reference_digest_for(output_dir: Path | None, dataset_name: str, image_stem: str) -> str | None:
    """The image's reference digest for its work-id; ``None`` when the run has none."""
    if output_dir is None:
        return None
    manifest = read_reference_manifest(output_dir)
    if manifest is None:
        return None
    dataset = manifest["datasets"].get(dataset_name, {})
    return dataset.get("digests", {}).get(image_stem, UNPLANNED_DIGEST)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/unit/cli/test_cli_reference.py -q`
Expected: pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_reference.py tests/unit/cli/test_cli_reference.py
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_reference.py tests/unit/cli/test_cli_reference.py
git commit -m "feat(cli): reference planner, run manifest, and worker ReferenceContext"
```

---

### Task 5: Run preflight `PF-REF-*`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_preflight.py` (`FindingCode` literal ~line 49, `HINTS` ~line 84, new check, `CHECKS` tuple ~line 1428)
- Modify: `docs/source/tutorials/pages/cli_batch_processing.md` (`## Run Preflight Checks` table)
- Test: `tests/unit/cli/test_preflight_reference_checks.py`

**Interfaces:**
- Consumes: Task 4 (`plan_references(..., hash_images=False)`, `resolve_reference_table_path`), `PreflightContext` (`config`, `pipeline`, `datasets`, `mode`), `PreflightFinding(code, severity, message, subjects=())`, test helpers `make_config`, `make_datasets`, `make_context` from `tests/unit/cli/_preflight_support.py`.
- Produces: `check_reference_metadata(context) -> list[PreflightFinding]`; codes `PF-REF-NO-TABLE`, `PF-REF-TABLE`, `PF-REF-COLUMN`, `PF-REF-UNMATCHED`, `PF-REF-AMBIGUOUS`, `PF-REF-SELF`, `PF-REF-UNRESOLVED`.

- [ ] **Step 1: Write the failing tests**

Helpers (`tests/unit/cli/_preflight_support.py`): `make_context(pipeline, mode="full", datasets=(), **config_overrides)` and `make_datasets(*images, name="plate1")` (its `input_dir` is the first image's parent).

```python
"""PF-REF-*: the run preflight for reference metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from phenotypic import ImagePipeline
from phenotypic._cli._cli_preflight import check_reference_metadata
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank

from ._preflight_support import make_context, make_datasets


def _images(root: Path, *stems: str) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in stems:
        path = root / f"{stem}.tif"
        tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint8))
        paths.append(path)
    return paths


def _context(tmp_path, rows: dict | None, *, mode="full", stems=("t01", "t02"), pipeline=None):
    root = tmp_path / "in" / "plate1"
    _images(root, "blank", *stems)
    metadata = None
    if rows is not None:
        metadata = tmp_path / "layout.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
    datasets = make_datasets(*[root / f"{s}.tif" for s in stems], name="plate1")
    # make_context builds the ExecutionConfig itself from keyword overrides
    # (and sets measure_only / process_only_layer from `mode`).
    return make_context(
        pipeline or ImagePipeline(ops={"sb": SubtractBlank(), "d": OtsuDetector()}),
        mode,
        datasets,
        metadata_csv=metadata,
        output_dir=tmp_path / "out",
    )


def _codes(findings):
    return {(f.code, f.severity) for f in findings}


def test_ordinary_pipeline_has_no_findings(tmp_path):
    ctx = _context(tmp_path, None, pipeline=ImagePipeline(ops={"d": OtsuDetector()}))
    assert check_reference_metadata(ctx) == []


def test_measure_mode_is_skipped(tmp_path):
    assert check_reference_metadata(_context(tmp_path, None, mode="measure")) == []


def test_no_table_is_an_error(tmp_path):
    assert _codes(check_reference_metadata(_context(tmp_path, None))) == {("PF-REF-NO-TABLE", "error")}


def test_missing_column_is_an_error(tmp_path):
    findings = check_reference_metadata(_context(tmp_path, {"Metadata_ImageName": ["t01"]}))
    assert _codes(findings) == {("PF-REF-COLUMN", "error")}
    assert "Metadata_BlankImage" in findings[0].message


def test_table_without_image_name_is_a_table_error(tmp_path):
    findings = check_reference_metadata(_context(tmp_path, {"Metadata_BlankImage": ["blank"]}))
    assert _codes(findings) == {("PF-REF-TABLE", "error")}


def test_some_unmatched_is_a_warning(tmp_path):
    rows = {"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["blank"]}
    findings = check_reference_metadata(_context(tmp_path, rows))
    assert _codes(findings) == {("PF-REF-UNMATCHED", "warning")}
    assert findings[0].subjects == ("plate1/t02",)


def test_every_image_unresolved_escalates_to_error(tmp_path):
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["nope", "nope"]}
    assert _codes(check_reference_metadata(_context(tmp_path, rows))) == {("PF-REF-UNRESOLVED", "error")}


def test_ambiguous_and_self_are_reported(tmp_path):
    rows = {
        "Metadata_ImageName": ["t01", "t01", "t02"],
        "Metadata_BlankImage": ["blank", "other", "t02"],
    }
    assert _codes(check_reference_metadata(_context(tmp_path, rows))) == {
        ("PF-REF-AMBIGUOUS", "warning"),
        ("PF-REF-SELF", "warning"),
    }


def test_process_mode_is_checked(tmp_path):
    assert _codes(check_reference_metadata(_context(tmp_path, None, mode="process"))) == {
        ("PF-REF-NO-TABLE", "error")
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/cli/test_preflight_reference_checks.py -q`
Expected: `ImportError: cannot import name 'check_reference_metadata'`

- [ ] **Step 3: Implement**

Add the seven codes to the `FindingCode` literal after `"PF-META-UNVERIFIED"`, and to `HINTS`:

```python
    "PF-REF-NO-TABLE": (
        "Pass --metadata with a table that names each image's reference "
        "(e.g. a Metadata_BlankImage column), or remove the reference operation."
    ),
    "PF-REF-TABLE": (
        "Fix the reference table: it must be a readable CSV with a "
        "Metadata_ImageName (or ImageName) column."
    ),
    "PF-REF-COLUMN": (
        "Add the column the operation names to the table, or change the "
        "operation's column parameter to one the table has."
    ),
    "PF-REF-UNMATCHED": (
        "Add a row for each listed image, keyed by its file name without the "
        "extension, or leave the image out with --image-manifest."
    ),
    "PF-REF-AMBIGUOUS": (
        "Give each listed image exactly one non-empty value; its rows are empty "
        "or disagree."
    ),
    "PF-REF-SELF": (
        "A blank frame cannot be its own reference; leave blank frames out of "
        "the input with --image-manifest."
    ),
    "PF-REF-UNRESOLVED": (
        "Each named reference must match exactly one file (by stem or full "
        "name) in the image's own input directory."
    ),
```

New check, after `check_metadata_join`:

```python
def check_reference_metadata(context: PreflightContext) -> list[PreflightFinding]:
    """``PF-REF-*``: can the run's table serve its reference-metadata operations?

    Reads the table and lists directories only -- reference images are
    resolved to file names, never opened or hashed (startup hashes them).
    Per-image findings are warnings, escalated to errors when they cover
    every input image, per this module's severity rule.
    """
    if context.mode not in ("full", "process"):
        return []
    needs = context.pipeline.reference_columns()
    if not needs:
        return []
    from phenotypic._core._reference_context import ReferenceContext, ReferenceTableError

    from ._cli_reference import plan_references, resolve_reference_table_path

    described = "; ".join(f"{path} reads {list(cols)}" for path, cols in needs.items())
    table_path = resolve_reference_table_path(context.config, context.config.output_dir)
    if table_path is None:
        return [PreflightFinding(
            "PF-REF-NO-TABLE", "error",
            f"The pipeline reads reference metadata ({described}) but no --metadata was given.",
            subjects=tuple(needs),
        )]
    try:
        table = ReferenceContext(table_path)
    except ReferenceTableError as exc:
        return [PreflightFinding("PF-REF-TABLE", "error", str(exc), subjects=(str(table_path),))]
    missing = tuple(
        f"{path}: {column}" for path, cols in needs.items() for column in cols
        if not table.has_column(column)
    )
    if missing:
        return [PreflightFinding(
            "PF-REF-COLUMN", "error",
            f"{table_path} lacks columns the pipeline reads: {', '.join(missing)}",
            subjects=missing,
        )]
    plan = plan_references(table, context.pipeline, context.datasets, hash_images=False)
    findings: list[PreflightFinding] = []
    for code, labels, what in (
        ("PF-REF-UNMATCHED", plan.unmatched, "have no row in the reference table"),
        ("PF-REF-AMBIGUOUS", plan.ambiguous, "have empty or disagreeing reference values"),
        ("PF-REF-SELF", plan.self_referenced, "name themselves as their own reference"),
        ("PF-REF-UNRESOLVED", plan.unresolved, "name a reference image that matches no single file"),
    ):
        if labels:
            severity: Severity = "error" if len(labels) >= plan.total_images else "warning"
            findings.append(PreflightFinding(
                code, severity, f"{len(labels)} of {plan.total_images} input images {what}.",
                subjects=labels,
            ))
    return findings
```

Insert `check_reference_metadata,` into `CHECKS` directly after `check_metadata_join,`.

Docs table — in `docs/source/tutorials/pages/cli_batch_processing.md`, after the `PF-META-*` rows, add (same four columns: Code | Reported when | Severity | Remedy):

```markdown
| `PF-REF-NO-TABLE` | The pipeline has an operation that reads reference metadata (e.g. `SubtractBlank`) and no `--metadata` was given | error | Pass `--metadata` with the reference columns |
| `PF-REF-TABLE` | The reference table cannot be read or has no `ImageName` column | error | Fix the table |
| `PF-REF-COLUMN` | The table lacks a column an operation reads | error | Add the column, or change the operation's column parameter |
| `PF-REF-UNMATCHED` | Some images have no row in the reference table | warning (error if all) | Add rows, or leave the images out with `--image-manifest` |
| `PF-REF-AMBIGUOUS` | Some images' rows are empty or disagree for a reference column | warning (error if all) | Give each image one value |
| `PF-REF-SELF` | Some images name themselves as their reference | warning (error if all) | Leave blank frames out with `--image-manifest` |
| `PF-REF-UNRESOLVED` | A named reference image matches no single file in the image's directory | warning (error if all) | Name an existing file, by stem or full name |
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/cli/test_preflight_reference_checks.py tests/unit/test_docs_preflight_codes.py tests/unit/cli/test_cli_preflight_core.py -q`
Expected: pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_preflight.py tests/unit/cli/test_preflight_reference_checks.py
git add src/phenotypic/_cli/_cli_preflight.py docs/source/tutorials/pages/cli_batch_processing.md tests/unit/cli/test_preflight_reference_checks.py
git commit -m "feat(cli): PF-REF-* run preflight for reference metadata"
```

---

### Task 6: Startup publication, worker entry points, per-image work-id digest

**Files:**
- Create (function): `publish_reference_inputs` in `src/phenotypic/_cli/_cli_reference.py`
- Modify: `src/phenotypic/phenotypicCLI.py` (`_prepare_incremental_startup` line ~746; `_refuse_run_inputs_the_run_would_delete` call ~2325)
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py` (`compute_work_id` ~304, `work_id_for_image` ~368)
- Modify: `src/phenotypic/_cli/_cli_process_single.py` (`_worker_work_identity` ~123 and its callers; `process_single_image_core` apply at ~331; apply-only call at ~882)
- Modify: `src/phenotypic/_cli/_cli_process_only.py` (`process_single_apply_only_core` ~262, apply at ~346)
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py` (~600: pass `dataset_name`)
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py` (Stage-1 apply ~369, Stage-3 apply ~592)
- Test: `tests/unit/cli/test_cli_reference_wiring.py`

**Interfaces:**
- Consumes: Task 4 (`worker_reference_context`, `reference_digest_for`, `snapshot_reference_metadata`, `plan_references`, `write_reference_manifest`, `remove_reference_manifest`, `resolve_reference_table_path`, `input_read_kwargs`).
- Produces:
  - `publish_reference_inputs(config, datasets, output_dir) -> None`
  - `compute_work_id(..., reference_digest: str | None = None)` — payload key `"reference_digest"` only when not `None`
  - `process_single_apply_only_core(..., dataset_name: str | None = None)`
  - `_worker_work_identity(..., output_dir: Path | None = None)`

- [ ] **Step 1: Write the failing tests**

```python
"""Startup publishes the reference plan; work-ids carry the per-image digest."""

from __future__ import annotations

import inspect

from phenotypic._cli import _cli_failure_tracker as ft
from phenotypic._cli import _cli_process_only, _cli_process_single, _cli_staged_workers


def _ids(**overrides):
    base = dict(
        dataset="plate1", relative_image_path="plate1/t01.tif", input_sha256="a",
        pipeline_fingerprint="b", processing_config_digest="c", mode="full",
    )
    base.update(overrides)
    return ft.compute_work_id(**base)


def test_work_id_unchanged_without_a_reference_digest():
    assert _ids() == _ids(reference_digest=None)


def test_work_id_changes_with_the_reference_digest():
    assert _ids(reference_digest="x") != _ids(reference_digest="y")
    assert _ids(reference_digest="x") != _ids()


def test_every_worker_core_enters_the_reference_context():
    for module, name in (
        (_cli_process_single, "process_single_image_core"),
        (_cli_process_only, "process_single_apply_only_core"),
        (_cli_staged_workers, "stage1_preprocess_core"),
        (_cli_staged_workers, "stage3_merge_measure_core"),
    ):
        assert "worker_reference_context(" in inspect.getsource(getattr(module, name)), name
```

(The source-inspection test is a cheap tripwire against a fifth apply site being added without the context; Task 7's end-to-end runs are the behavioural proof.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/cli/test_cli_reference_wiring.py -q`
Expected: FAIL — `compute_work_id() got an unexpected keyword argument 'reference_digest'`.

- [ ] **Step 3: Work-id digest** — in `_cli_failure_tracker.py`, add `reference_digest: str | None = None` as the last keyword of `compute_work_id`, and before `return`:

```python
    if reference_digest is not None:
        # Present only for pipelines that read reference metadata, so every
        # existing work-id is unchanged. Per image, not per table: editing one
        # plate's blank re-runs that plate's frames and nothing else.
        payload["reference_digest"] = reference_digest
```

In `work_id_for_image`, pass:

```python
            reference_digest=reference_digest_for(
                config.output_dir, dataset, Path(image_path).stem
            ),
```

with `from ._cli_reference import reference_digest_for` imported inside the function. In `_cli_process_single._worker_work_identity` add `output_dir: Path | None = None` and pass `reference_digest=reference_digest_for(output_dir, dataset_name, Path(image).stem)` to `compute_work_id`; update every caller of `_worker_work_identity` (grep `_worker_work_identity(`) to pass `output_dir=output_dir`.

- [ ] **Step 4: Startup publication** — append to `_cli_reference.py`:

```python
def publish_reference_inputs(
    config: "ExecutionConfig", datasets: Sequence["Dataset"], output_dir: Path
) -> None:
    """Snapshot the table (process mode) and publish the run's reference manifest.

    Must run before the invocation computes any work-id, because work-ids read
    the manifest's per-image digests. Removes a stale manifest when the
    pipeline no longer reads reference metadata.
    """
    from phenotypic import ImagePipeline
    from phenotypic._core._reference_context import ReferenceContext, ReferenceTableError

    pipeline = ImagePipeline.from_json(config.pipeline_json)
    if config.measure_only or not pipeline.reference_columns():
        remove_reference_manifest(output_dir)
        return
    if config.process_only_layer is not None:
        config.metadata_csv = snapshot_reference_metadata(output_dir, config.metadata_csv)
    table_path = resolve_reference_table_path(config, output_dir)
    if table_path is None:
        raise ReferenceTableError(
            "The pipeline reads reference metadata but the run has no table; pass --metadata"
        )
    context = ReferenceContext(table_path)
    plan = plan_references(context, pipeline, datasets, hash_images=True)
    write_reference_manifest(
        output_dir,
        plan=plan,
        table_path=table_path,
        table_sha256=context.table_sha256 or "",
        read_kwargs=input_read_kwargs(config),
    )
```

In `phenotypicCLI._prepare_incremental_startup`, insert directly **after** the full-mode snapshot block and **before** `migrated_failures = 0`:

```python
    from phenotypic._cli._cli_reference import publish_reference_inputs

    publish_reference_inputs(config, datasets, output_dir)
```

At the `_refuse_run_inputs_the_run_would_delete` call (~line 2334), change the `--metadata` entry to `metadata_csv if cli_mode in ("full", "process") else None`.

Confirm by reading the main flow that no `work_id_for_image` call precedes `_prepare_incremental_startup` in the same invocation (grep `work_id_for_image(` in `phenotypicCLI.py` and check each is below the startup call or inside a worker). Record the result in the commit message body.

- [ ] **Step 5: Worker entry points** — wrap each apply with the context (import `from ._cli_reference import worker_reference_context` at the top of each module; `_cli_reference` imports nothing heavy at module level):

`_cli_process_single.py` (~331):

```python
        with continuing_provenance_application(image), provenance_success_sink(
            _write_checkpoint
        ), worker_reference_context(output_dir, dataset_name):
            measurements = pipeline.apply_and_measure(
                image, inplace=True, apply_post=False
            )
```

`_cli_process_only.py`: add `dataset_name: str | None = None` to `process_single_apply_only_core`'s signature (after `run_initiation`), and:

```python
        with continuing_provenance_application(image), worker_reference_context(
            output_dir, dataset_name
        ):
            pipeline.apply(image, inplace=True)
```

Pass `dataset_name=dataset.name` at `_cli_execution_strategies.py:600` and `dataset_name=dataset_name` at `_cli_process_single.py:882`.

`_cli_staged_workers.py` Stage 1 (~369) and Stage 3 (~592): wrap the existing `plan.pre_pipeline.apply(image, inplace=True)` and `replay_pipeline.apply(image, inplace=True)` in `with worker_reference_context(output_dir, dataset_name):`. Stage 3 matters: it re-runs the detector's whole top-level ancestor, which can contain a `SubtractBlank` sibling.

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/unit/cli/test_cli_reference_wiring.py tests/unit/cli/test_cli_reference.py -q`
Expected: pass.

- [ ] **Step 7: Affected surface (work-id + worker cores)**

Derive the surface from importers, not the directory:
`grep -rln "compute_work_id\|work_id_for_image\|_worker_work_identity\|process_single_apply_only_core\|stage1_preprocess_core\|stage3_merge_measure_core\|_prepare_incremental_startup" tests/ | sort -u`
Run that file list once with `uv run pytest <files> -q -p no:randomly` (as a Slurm job via the `run-phenotypic-test` skill if it exceeds ~5 minutes). Expected: pass; any failure is run in isolation before attribution.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_reference.py src/phenotypic/phenotypicCLI.py src/phenotypic/_cli/_cli_failure_tracker.py src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_process_only.py src/phenotypic/_cli/_cli_execution_strategies.py src/phenotypic/_cli/_cli_staged_workers.py tests/unit/cli/test_cli_reference_wiring.py
git add src/phenotypic/_cli src/phenotypic/phenotypicCLI.py tests/unit/cli/test_cli_reference_wiring.py
git commit -m "feat(cli): publish reference plan at startup; workers enter ReferenceContext"
```

---

### Task 7: CLI end-to-end runs

**Files:**
- Test: `tests/unit/cli/test_cli_reference_e2e.py`

**Interfaces:**
- Consumes: everything in Phase 2; `phenotypic_cli` (click) invoked with `CliRunner` as in `tests/unit/cli/test_cli_preflight_ordering.py`.

- [ ] **Step 1: Write the tests**

```python
"""SubtractBlank through the real CLI: process + full mode, continuation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner

from phenotypic import Image, ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli


def _rgb(level: int, colony: tuple[slice, slice] | None = None) -> np.ndarray:
    image = np.full((32, 32, 3), level, dtype=np.uint8)
    image[0:4, 0:4] = 90                      # a static scratch on the lid
    if colony is not None:
        image[colony] = 220
    return image


@pytest.fixture
def run_inputs(tmp_path: Path):
    root = tmp_path / "images" / "plate1"
    root.mkdir(parents=True)
    tifffile.imwrite(root / "blank.tiff", _rgb(20))
    tifffile.imwrite(root / "blank2.tiff", _rgb(30))
    tifffile.imwrite(root / "t01.tiff", _rgb(20, (slice(10, 16), slice(10, 16))))
    tifffile.imwrite(root / "t02.tiff", _rgb(20, (slice(18, 26), slice(18, 26))))
    manifest = tmp_path / "frames.txt"
    manifest.write_text("plate1/t01.tiff\nplate1/t02.tiff\n", encoding="utf-8")
    table = tmp_path / "blank_map.csv"
    pd.DataFrame({"ImageName": ["t01", "t02"], "BlankImage": ["blank", "blank"]}).to_csv(table, index=False)
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()}, meas={"size": MeasureSize()}).to_json(),
        encoding="utf-8",
    )
    return tmp_path, root, manifest, table, pipeline


def _cli(*args: str):
    return CliRunner().invoke(phenotypic_cli, [*args, "--njobs", "1"], catch_exceptions=False)


def _process(base, manifest, table, pipeline, *extra):
    args = [
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--output", str(base / "out"), "--image-manifest", str(manifest),
        "--mode", "process", "--layer", "detect_mat",
    ]
    if table is not None:
        args += ["--metadata", str(table)]
    return _cli(*args, *extra)


def _output(base: Path, stem: str) -> Path:
    (found,) = [p for p in (base / "out").rglob(f"{stem}.*") if ".phenotypic" not in p.parts]
    return found


def test_process_mode_subtracts_the_blank(run_inputs):
    base, root, manifest, table, pipeline = run_inputs
    result = _process(base, manifest, table, pipeline)
    assert result.exit_code == 0, result.output
    t01 = Image.imread(root / "t01.tiff")
    blank = Image.imread(root / "blank.tiff")
    expected = np.clip(t01.gray[:] - blank.gray[:], 0.0, 1.0)
    np.testing.assert_allclose(tifffile.imread(_output(base, "t01")), expected, atol=1e-6)
    assert (base / "out" / ".phenotypic" / "reference_metadata.csv").read_bytes() == table.read_bytes()


def test_process_mode_without_metadata_is_refused_before_any_write(run_inputs):
    base, _, manifest, _, pipeline = run_inputs
    result = _process(base, manifest, None, pipeline)
    assert result.exit_code != 0
    assert "PF-REF-NO-TABLE" in result.output
    assert not list((base / "out").rglob("t01.*"))


def test_continuation_reruns_only_the_image_whose_blank_changed(run_inputs):
    base, _, manifest, table, pipeline = run_inputs
    assert _process(base, manifest, table, pipeline).exit_code == 0
    t01, t02 = _output(base, "t01"), _output(base, "t02")
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (t01, t02)}

    # Same command, no --metadata: falls back to the snapshot, reuses both.
    assert _process(base, manifest, None, pipeline).exit_code == 0
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (t01, t02)} == before

    edited = base / "blank_map_v2.csv"
    pd.DataFrame({"ImageName": ["t01", "t02"], "BlankImage": ["blank2", "blank"]}).to_csv(edited, index=False)
    assert _process(base, manifest, edited, pipeline).exit_code == 0
    assert _output(base, "t01").read_bytes() != before[t01.name][1]
    assert _output(base, "t02").stat().st_mtime_ns == before[t02.name][0]


def test_full_mode_runs_and_joins_the_blank_column(run_inputs):
    base, _, manifest, table, pipeline = run_inputs
    result = _cli(
        "--pipeline", str(pipeline), "--input", str(base / "images"),
        "--output", str(base / "full"), "--image-manifest", str(manifest),
        "--metadata", str(table),
    )
    assert result.exit_code == 0, result.output
    measurements = pd.read_csv(base / "full" / "deliverables" / "measurements.csv")
    assert set(measurements["Metadata_BlankImage"].dropna()) == {"blank"}
```

Notes for the implementer:
- `--njobs 1` keeps the run in-process (loky would also work — workers read the manifest from disk — but in-process failures are easier to read).
- If the CLI's `--image-manifest` paths must be absolute, write absolute paths.
- If the mtime check in the continuation test is flaky on the filesystem's timestamp granularity, assert on the run's event log instead (`resolve_event_log_path(out)`): `t02` must have no second `"started"` event. Do not delete the assertion.

- [ ] **Step 2: Run**

Run: `uv run pytest tests/unit/cli/test_cli_reference_e2e.py -q`
Expected: pass. A failure here is a Task 6 defect — fix it there, then rerun.

- [ ] **Step 3: Commit**

```bash
uv run ruff check --fix tests/unit/cli/test_cli_reference_e2e.py
git add tests/unit/cli/test_cli_reference_e2e.py
git commit -m "test(cli): SubtractBlank end-to-end in process and full mode, with continuation"
```

**Phase 2 gate:** the affected surface once — `tests/unit/cli/` as a Slurm array per the `run-phenotypic-test` skill (it is large). Expected: no new failures against the `main` baseline in memory (`ebb6d7fc`, 0 failed); run any failure in isolation before attributing it.

---

## Phase 3 — GUI

### Task 8: Builder — reference-metadata picker, preview context, column dropdowns

**Files:**
- Create: `src/phenotypic/_gui/builder/_reference_metadata.py`
- Modify: `src/phenotypic/_gui/builder/_ids.py` (three ids, export list ~line 1098)
- Modify: `src/phenotypic/_gui/builder/_layout.py` (picker under `ACTIVE_IMAGE_LABEL` ~4126; store beside `STORE_IMAGE_PATH` ~4481; `build_inspector`/`_build_dag_inspector` gain `columns_provider`)
- Modify: `src/phenotypic/_gui/builder/_param_form.py` (`param_form` gains `columns_provider`)
- Modify: `src/phenotypic/_gui/builder/_callbacks.py` (picker callback; inspector render ~4017; preview request ~3672 and its `State`s ~5933/5977; preview run ~6090)
- Modify: `src/phenotypic/_gui/builder/_preview_cache.py` (`compute_scope` ~312, apply ~386)
- Modify: `src/phenotypic/_gui/builder/_preview_callbacks.py` (~121)
- Test: `tests/unit/gui/builder/test_reference_metadata.py`, extend `tests/unit/gui/test_operation_registry.py`

**Interfaces:**
- Consumes: Task 1 (`ReferenceContext`, `ReferenceTableError`), Task 2 (`_ColumnRefMarker("reference_metadata")`).
- Produces:
  - ids `STORE_REFERENCE_METADATA_PATH = "store-reference-metadata-path"`, `INPUT_REFERENCE_METADATA = "input-reference-metadata"`, `REFERENCE_METADATA_STATUS = "reference-metadata-status"`
  - `describe_reference_table(path: str | None) -> tuple[str, str]` → `(store_value, status_message)`
  - `reference_columns_provider(path: str | None) -> Callable[[str], list[str]] | None`
  - `preview_reference_context(reference_metadata: str | None, image_path: str | None)` — context manager
  - `reference_identity(reference_metadata: str | None) -> str` (`""` when unset)
  - `compute_scope(..., reference_metadata: str | None = None)`
  - `build_inspector(state, registry, *, columns_provider=None)`, builder `param_form(..., columns_provider=None)`

- [ ] **Step 1: Write the failing tests**

```python
"""Builder reference-metadata helpers and the preview fingerprint."""

from __future__ import annotations

import pandas as pd

from phenotypic import ReferenceContext
from phenotypic._gui.builder import _reference_metadata as rm


def _table(tmp_path):
    path = tmp_path / "blank_map.csv"
    pd.DataFrame({"ImageName": ["t01"], "BlankImage": ["t00"]}).to_csv(path, index=False)
    return path


def test_describe_empty_missing_and_valid(tmp_path):
    assert rm.describe_reference_table("") == ("", "")
    value, message = rm.describe_reference_table(str(tmp_path / "nope.csv"))
    assert value == "" and "not found" in message.lower()
    value, message = rm.describe_reference_table(str(_table(tmp_path)))
    assert value.endswith("blank_map.csv")
    assert "1 rows" in message and "2 columns" in message


def test_columns_provider_serves_only_reference_metadata(tmp_path):
    provide = rm.reference_columns_provider(str(_table(tmp_path)))
    assert "Metadata_BlankImage" in provide("reference_metadata")
    assert provide("measurements") == []
    assert rm.reference_columns_provider("") is None


def test_preview_context_activates_with_the_image_directory_as_root(tmp_path):
    image = tmp_path / "plates" / "t01.tif"
    with rm.preview_reference_context(str(_table(tmp_path)), str(image)) as ctx:
        assert ReferenceContext.current() is ctx
        assert ctx.image_root == image.parent
    with rm.preview_reference_context(None, str(image)) as ctx:
        assert ctx is None and ReferenceContext.current() is None


def test_reference_identity_follows_file_content(tmp_path):
    path = _table(tmp_path)
    first = rm.reference_identity(str(path))
    assert rm.reference_identity(None) == ""
    path.write_text(path.read_text() + "t02,t00\n", encoding="utf-8")
    assert rm.reference_identity(str(path)) != first
```

Extend `tests/unit/gui/test_operation_registry.py::TestColumnRefDetection` with:

```python
    def test_subtract_blank_column_is_a_reference_metadata_dropdown(self, registry):
        p = registry.get("SubtractBlank").parameters["blank_column"]
        assert p.column_ref is not None
        assert p.column_ref.source == "reference_metadata"
        assert p.column_ref.multi is False
```

And in `tests/unit/gui/builder/test_preview_cache_manifest.py` add (reuse that file's `_linear_root_state` helper and the `cached_scope` setup it uses for `tmp_path`/monkeypatch of the cache root):

```python
def test_reference_table_changes_the_root_fingerprint(tmp_path, monkeypatch) -> None:
    table = tmp_path / "blank_map.csv"
    table.write_text("ImageName,BlankImage\nt01,t00\n", encoding="utf-8")
    state = _linear_root_state([])
    plain = pc.compute_scope("s", state, [], None, None, None)
    with_ref = pc.compute_scope("s", state, [], None, None, None, reference_metadata=str(table))
    assert plain["fingerprint"] != with_ref["fingerprint"]
```

(Match `_linear_root_state`'s real argument shape and the cache-root fixture in that file.)

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder/test_reference_metadata.py -q`
Expected: `ImportError` for `_reference_metadata`.

- [ ] **Step 3: Helpers** — create `src/phenotypic/_gui/builder/_reference_metadata.py`:

```python
"""Builder-side reference metadata: the preview's ReferenceContext and column list.

The builder never stores a table on an operation. The session's picked table
acts as the preview's ambient context, exactly as ``--metadata`` does for the
CLI, and supplies the dropdown choices for ``RefColumn`` parameters.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

REFERENCE_SOURCE = "reference_metadata"


def describe_reference_table(path: Optional[str]) -> tuple[str, str]:
    """Validate a picked table: ``(value_to_store, status_message)``."""
    if not path:
        return "", ""
    from phenotypic._core._reference_context import ReferenceContext, ReferenceTableError

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return "", f"Not found: {candidate}"
    try:
        context = ReferenceContext(candidate)
    except ReferenceTableError as exc:
        return "", str(exc)
    return (
        str(candidate.resolve()),
        f"{candidate.name} · {context.table.height} rows · {len(context.columns)} columns",
    )


def reference_columns_provider(path: Optional[str]) -> Optional[Callable[[str], list[str]]]:
    """A ``columns_provider`` for the shared param form, or ``None`` without a table."""
    if not path:
        return None
    from phenotypic._core._reference_context import ReferenceContext, ReferenceTableError

    try:
        columns = list(ReferenceContext(path).columns)
    except ReferenceTableError:
        return None

    def provide(source: str) -> list[str]:
        return list(columns) if source == REFERENCE_SOURCE else []

    return provide


def reference_identity(reference_metadata: Optional[str]) -> str:
    """Content identity of the picked table for preview fingerprints."""
    if not reference_metadata:
        return ""
    path = Path(reference_metadata)
    if not path.is_file():
        return f"missing:{path}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def preview_reference_context(
    reference_metadata: Optional[str], image_path: Optional[str]
) -> Iterator[object]:
    """Activate the picked table around a preview; reference images resolve
    beside the preview image."""
    if not reference_metadata:
        yield None
        return
    from phenotypic._core._reference_context import ReferenceContext

    root = Path(image_path).parent if image_path else None
    with ReferenceContext(reference_metadata, image_root=root) as context:
        yield context
```

- [ ] **Step 4: Preview cache** — in `_preview_cache.compute_scope`, add `reference_metadata: str | None = None` as the last parameter; pass it to the recursive parent call; after `fingerprint_inputs = [sig, input_identity]`:

```python
    if not scope_path and reference_metadata:
        # Root only: nested scopes inherit it through parent_fp. Appended only
        # when set, so every existing fingerprint (and cache) is unchanged.
        fingerprint_inputs.append(reference_identity(reference_metadata))
```

and wrap line ~386:

```python
        with preview_reference_context(reference_metadata, image_path):
            pipeline.apply_with_intermediates(image, output_dir=sdir, full_layers=True)
```

(import both from `._reference_metadata` inside the function, like the module's other imports). In `_preview_callbacks.py:121`, pass `reference_metadata=` from the same place `image_path` comes from: add `State(ids.STORE_REFERENCE_METADATA_PATH, "data")` to that callback beside the image-path input it already reads, and thread the value.

- [ ] **Step 5: Top-level preview** — in `_callbacks.py`, at the request builder (~3655–3677) add `"reference_metadata": reference_metadata,` to the `request` dict and a `reference_metadata` parameter; add `State(ids.STORE_REFERENCE_METADATA_PATH, "data")` next to each `State(STORE_IMAGE_PATH, "data")` that feeds it (~5933, ~5977) and pass it through. At ~6090:

```python
            with preview_reference_context(
                request_data.get("reference_metadata"),
                image_path if isinstance(image_path, str) else None,
            ):
                result = pipeline.apply_with_intermediates(image)
```

- [ ] **Step 6: Picker** — ids in `_ids.py` (with docstrings like their neighbours, and in the export list):

```python
#: Session-level reference metadata table (path). Read by Run preview and by
#: the inspector's RefColumn dropdowns; never saved into the pipeline.
STORE_REFERENCE_METADATA_PATH = "store-reference-metadata-path"

#: Text input for the reference metadata table path.
INPUT_REFERENCE_METADATA = "input-reference-metadata"

#: One-line status under the reference metadata input (rows/columns or error).
REFERENCE_METADATA_STATUS = "reference-metadata-status"
```

In `_layout.py`, directly below the `ACTIVE_IMAGE_LABEL` div (~4126):

```python
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Reference metadata"),
                    dbc.Input(
                        id=ids.INPUT_REFERENCE_METADATA,
                        type="text",
                        placeholder="/path/to/blank_map.csv (optional)",
                        debounce=True,
                    ),
                ],
                size="sm",
                className="mt-2",
            ),
            html.Div(id=ids.REFERENCE_METADATA_STATUS, className="small text-muted"),
```

and beside the `STORE_IMAGE_PATH` store (~4481): `dcc.Store(id=ids.STORE_REFERENCE_METADATA_PATH, data=""),`.

In `_callbacks.py`, register:

```python
    @app.callback(
        Output(ids.STORE_REFERENCE_METADATA_PATH, "data"),
        Output(ids.REFERENCE_METADATA_STATUS, "children"),
        Input(ids.INPUT_REFERENCE_METADATA, "value"),
        prevent_initial_call=True,
    )
    def set_reference_metadata(path: object) -> tuple[str, str]:
        """Validate and store the session's reference metadata table."""
        from ._reference_metadata import describe_reference_table

        return describe_reference_table(path if isinstance(path, str) else None)
```

- [ ] **Step 7: Dropdowns** — builder `_param_form.param_form` gains `columns_provider: Callable[[str], list[str]] | None = None` and passes it to `_shared_param_form(..., columns_provider=columns_provider)`. Update the stale comment in `_gui/_param_forms.py` (~line 585: "builder ops carry no column-ref params, so this branch is dead code on the builder path") to say the builder supplies a provider for `reference_metadata`. `build_inspector` and `_build_dag_inspector` gain `*, columns_provider=None` and pass it to both `param_form(` calls (~3789, ~3931). The inspector callback (~4017) gains `State(ids.STORE_REFERENCE_METADATA_PATH, "data")` as its last `State` and calls `build_inspector(state, registry, columns_provider=reference_columns_provider(reference_path))`. (The dropdown refreshes the next time the inspector renders; the linear side loader at `_linear_layout.py:1087` keeps free text.)

- [ ] **Step 8: Run**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder/test_reference_metadata.py tests/unit/gui/builder/test_preview_cache_manifest.py tests/unit/gui/test_operation_registry.py tests/unit/gui/test_param_forms.py tests/unit/gui/test_apps_build_after_simplification.py -q`
Expected: pass (the last file proves every Dash app still builds with the new ids and callbacks).

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_gui/builder/_reference_metadata.py src/phenotypic/_gui/builder/_ids.py src/phenotypic/_gui/builder/_layout.py src/phenotypic/_gui/builder/_param_form.py src/phenotypic/_gui/builder/_callbacks.py src/phenotypic/_gui/builder/_preview_cache.py src/phenotypic/_gui/builder/_preview_callbacks.py src/phenotypic/_gui/_param_forms.py tests/unit/gui/builder/test_reference_metadata.py tests/unit/gui/builder/test_preview_cache_manifest.py tests/unit/gui/test_operation_registry.py
git add src/phenotypic/_gui tests/unit/gui
git commit -m "feat(gui): builder reference-metadata picker, preview context, column dropdowns"
```

---

### Task 9: Run console — require a table for reference pipelines

**Files:**
- Modify: `src/phenotypic/_gui/run_console/_ids.py` (`RC_REFERENCE_METADATA_REQUIRED`, export list ~316)
- Modify: `src/phenotypic/_gui/run_console/_form.py` (alert after `RC_STAGED_GPU_REFUSAL`, ~811)
- Modify: `src/phenotypic/_gui/run_console/_callbacks.py` (helper beside `_staged_gpu_capability` ~250; callback beside `show_staged_gpu_controls` ~1769; `update_run_disabled` ~2733)
- Test: `tests/unit/gui/run_console/test_reference_requirement.py`

**Interfaces:**
- Consumes: Task 2 (`ImagePipeline.reference_columns`); the form-state store `RC_STORE_FORM_STATE` whose payload has `"metadata_csv"` (`_callbacks.py:705`).
- Produces: `reference_metadata_requirement(pipeline_path: object, metadata_csv: object) -> str | None`; id `RC_REFERENCE_METADATA_REQUIRED = "rc-reference-metadata-required"`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic._gui.run_console._callbacks import reference_metadata_requirement
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank


def _write(tmp_path, ops):
    path = tmp_path / "pipeline.json"
    path.write_text(ImagePipeline(ops=ops).to_json(), encoding="utf-8")
    return str(path)


def test_reference_pipeline_without_metadata_is_blocked(tmp_path):
    message = reference_metadata_requirement(_write(tmp_path, {"sb": SubtractBlank()}), None)
    assert message is not None and "Metadata_BlankImage" in message


def test_reference_pipeline_with_metadata_is_allowed(tmp_path):
    assert reference_metadata_requirement(_write(tmp_path, {"sb": SubtractBlank()}), "/x/blank_map.csv") is None


def test_ordinary_or_unreadable_pipeline_is_not_blocked(tmp_path):
    assert reference_metadata_requirement(_write(tmp_path, {"d": OtsuDetector()}), None) is None
    assert reference_metadata_requirement(str(tmp_path / "missing.json"), None) is None
    assert reference_metadata_requirement(None, None) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/run_console/test_reference_requirement.py -q`
Expected: `ImportError: cannot import name 'reference_metadata_requirement'`.

- [ ] **Step 3: Implement** — helper beside `_staged_gpu_capability`:

```python
def reference_metadata_requirement(pipeline_path: object, metadata_csv: object) -> str | None:
    """Why Run must wait for a metadata table, or ``None`` when it need not.

    An unreadable pipeline is not reported here: the CLI's pipeline
    validation owns that message.
    """
    if not isinstance(pipeline_path, str) or not pipeline_path:
        return None
    if isinstance(metadata_csv, str) and metadata_csv:
        return None
    from phenotypic import ImagePipeline

    try:
        needs = ImagePipeline.from_json(Path(pipeline_path)).reference_columns()
    except (OSError, ValueError, TypeError):
        return None
    if not needs:
        return None
    reads = "; ".join(f"{path} reads {', '.join(cols)}" for path, cols in needs.items())
    return (
        f"This pipeline reads reference metadata ({reads}). Include a metadata "
        f"CSV with those columns before running."
    )
```

Id in `_ids.py`: `RC_REFERENCE_METADATA_REQUIRED = "rc-reference-metadata-required"` (+ export). In `_form.py`, after the `RC_STAGED_GPU_REFUSAL` alert:

```python
        dbc.Alert(
            id=ids.RC_REFERENCE_METADATA_REQUIRED,
            color="warning",
            is_open=False,
            className="mt-2",
        ),
```

Callback beside `show_staged_gpu_controls`:

```python
    @app.callback(
        Output(ids.RC_REFERENCE_METADATA_REQUIRED, "children"),
        Output(ids.RC_REFERENCE_METADATA_REQUIRED, "is_open"),
        Input(ids.RC_STORE_PIPELINE_PATH, "data"),
        Input(ids.RC_STORE_FORM_STATE, "data"),
    )
    def show_reference_metadata_requirement(pipeline_path: object, form_state: object) -> tuple[str, bool]:
        """Explain, and gate Run, when a reference pipeline has no metadata table."""
        metadata = form_state.get("metadata_csv") if isinstance(form_state, dict) else None
        message = reference_metadata_requirement(pipeline_path, metadata)
        return message or "", message is not None
```

In `update_run_disabled`, add `Input(ids.RC_REFERENCE_METADATA_REQUIRED, "is_open")` and a `reference_missing: Optional[bool]` parameter; return `True` when it is set (beside `pipeline_refused`).

- [ ] **Step 4: Run**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/run_console -q`
Expected: pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_gui/run_console/_ids.py src/phenotypic/_gui/run_console/_form.py src/phenotypic/_gui/run_console/_callbacks.py tests/unit/gui/run_console/test_reference_requirement.py
git add src/phenotypic/_gui/run_console tests/unit/gui/run_console/test_reference_requirement.py
git commit -m "feat(gui): run console requires a metadata table for reference pipelines"
```

### Task 10: GUI ledgers and tutorial capture

**Files:** `src/phenotypic/_gui/FEATURES.md`, `WORKFLOWS.md` (only if the skill says a workflow changed), the tutorial capture script.

- [ ] **Step 1:** Invoke the **`gui-tutorial-capture`** skill and follow it for two new affordances: the builder's *Reference metadata* input (+ status line) and the run console's *reference metadata required* alert. Add one `FEATURES.md` row each under the builder and run-console sections, marked as the skill prescribes.
- [ ] **Step 2:** Run the gates the skill names (at least `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/test_check_features_md.py tests/unit/gui/test_check_workflows_md.py -q`). Expected: pass.
- [ ] **Step 3:** Commit: `git commit -m "docs(gui): FEATURES rows for reference metadata picker and run gate"`.

**Phase 3 gate:** `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui -q` once, as a Slurm job if it exceeds a few minutes. Expected: no new failures.

---

## Phase 4 — Documentation and final regression

### Task 11: User and contributor docs

**Files:**
- Create: `docs/source/how_to/pages/reference_metadata.md`
- Modify: `docs/source/how_to/index.rst` (toctree, beside `pages/migrate_ome_zarr`)
- Modify: `src/phenotypic/abc_/CLAUDE.md` (a `RefMetadata` entry beside the `PlotImage` capability)
- Modify: `CLAUDE.md` (one Gotchas bullet)

- [ ] **Step 1: How-to page** — sections, each with a runnable snippet:
  1. *What it is for* — frame-0 media blank in time-lapse plates (ucr_033 as the motivating example, without project-private paths).
  2. *The table* — one row per image (or per colony; values must agree per image), `ImageName` + `BlankImage`; blanks named by stem in the same input directory.
  3. *Python* — `with ReferenceContext("blank_map.csv", image_root="images/plate1"): pipe.apply(img)`; prototyping with `ctx.lookup(...)` and `pipe.reference_columns()`.
  4. *CLI* — `--metadata blank_map.csv`, `--image-manifest` to leave blank frames out, `PF-REF-*` codes, continuation re-runs only images whose blank changed; process mode snapshot under `.phenotypic/`.
  5. *GUI* — builder *Reference metadata* input; run console gate.
  6. *Placement* — before any enhancer or directly after `SetDetectMode`; a later `SetDetectMode` discards it; `polarity` with the brighter/darker/both table from the spec.
- [ ] **Step 2: CLAUDE.md bullet** (Gotchas):

```markdown
- **Reference-metadata ops need a context:** an operation mixing in
  `RefMetadata` (e.g. `SubtractBlank`) holds no table path; it reads the
  active `phenotypic.ReferenceContext`. The CLI's `--metadata` is that
  context (startup writes `.phenotypic/reference_manifest.json`; workers read
  it), and a per-image reference digest enters the work-id only for such
  pipelines. Process mode snapshots the table to
  `.phenotypic/reference_metadata.csv`, preserved across `--restart`.
```

- [ ] **Step 3: Build check (rendering only)** — submit, per global instructions, as a Slurm job: `sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never -b html docs/source <scratch-out>` on `short`, then read the generated `reference_metadata.html` and confirm the code blocks and the polarity table rendered.
- [ ] **Step 4: Commit** — `git commit -m "docs: reference metadata how-to and CLAUDE.md notes"`.

### Task 12: Final regression

- [ ] **Step 1:** Run the full sharded `tests/unit` suite **once** with the committed batch script `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch` (via the `run-phenotypic-test` and `slurm-job` skills; never `-n auto`, never `-x`), in a worktree detached at this branch's HEAD SHA.
- [ ] **Step 2:** Compare with the `main` baseline (`ebb6d7fc`: 13,462 tests, 0 failed). Every new failure is re-run in isolation before attribution; report counts measured from the job output, not estimated.
- [ ] **Step 3:** Run doctests for the new modules: `uv run pytest --doctest-modules src/phenotypic/_core/_reference_context.py src/phenotypic/enhance/_subtract_blank.py -q`.
