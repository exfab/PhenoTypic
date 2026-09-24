# Per-image figures in the OME-Zarr store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every per-image (`PlotImage`) figure a pipeline produces is written into that image's `.ome.zarr` store. This happens in `--mode full`, staged Stage 3, `--mode measure` and `--mode process --process-format zarr`. `deliverables/plots/` is then filled by copying out of the promoted store.

**Architecture:** Three units, each depending only on the one before it:

1. **Build** (`plotting/_pipeline/_store_figures.py`) renders every `PlotImage` binding in memory into an immutable `StoredFigures` value. It writes nothing.
2. **Store write** (`sdk_/_image_figures.py`) writes that value into a store `.part` inside the existing root-last transaction, and returns the `attributes.phenotypic.figures` descriptor.
3. **Copy-out** (`plotting/_pipeline/_store_copyout.py`) runs after promotion. It reads the promoted store's descriptor, verifies each file's `sha256`, and republishes the files at today's `deliverables/plots/` paths. It also renders the deliverable HTML from each stored `plotly-json`.

`PlotCoordinator.emit_image` and its flat-path writer are removed.

**Tech Stack:** Python 3.12, pydantic v2, Zarr v3 / OME-Zarr 0.5 (hand-written group documents, as `tables/` does), Plotly 6 + Kaleido 1, matplotlib, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md`, revised 2026-09-22 after plan review; read its *Revision* section at the end. §-numbers below refer to it. The reviews are in `docs/superpowers/reports/2026-09-22-figures-in-ome-zarr/` (`plan-review.md`, `simplicity-review.md`). Finding ids (B1, M4, "minor 7") refer to `plan-review.md`.

## Global Constraints

- Use `uv run` for every command, never bare `python` or `pip`. Run `uv run ruff check --fix <explicit paths>` only, never bare.
- No module-level import of plotly, matplotlib, kaleido or zarr in any file this plan touches. Guards: `tests/unit/ci/test_startup_imports.py`, `tests/unit/ci/test_deferred_imports.py`.
- New store content lives under `attributes.phenotypic` only. Do not change `attributes.ome`, `OME/zarr.json` or `METADATA.ome.xml`. Do not bump `store_schema_version`.
- The format set is closed: `"plotly-json"` (`.plotly.json`, `application/vnd.plotly.v1+json`) and `"png"` (`.png`, `image/png`). The default `store` is `("plotly-json",)` for `backend="plotly"` and `("png",)` for `backend="mpl"`.
- Figures are best-effort: no figure error may fail an image. The only exception that propagates is `PlotPublicationBlocked` (a refused guard or fence), as in every handler today.
- Determinism: the same inputs, pipeline, version and environment give the same bytes. Only stochastic variation is removed (random ids, timestamps, `0x…` addresses).
- `PROCESS_LAYER_SEMANTICS_REVISION` goes 2 → 3. Full mode gets no revision.
- No new public API. The `_image_figures` names are **not** re-exported from `phenotypic.sdk_`; callers import the private module. The only new public name is `StoreFormat` in `phenotypic.abc_.plotting`, next to `figure`.
- Use Google-style docstrings and explicit names, and match the surrounding comment density.
- Tests per step: run only the touched test file(s). Per task: run the task's files. The full suite runs once, in Task 9, as a Slurm job through the **`run-phenotypic-test`** skill. Never use `-n auto`, and never use `-x` on a baseline.
- **Every new test must be shown to fail when the bug it guards is reintroduced.** A "prove it can fail" step does exactly that, then restores the code.

## Execution (orchestration, 2026-09-22)

**Dependency DAG** (→ = must finish before; shared files in brackets):

```
T1 (abc_ store formats) ──────────────┐
T2 (sdk_ _image_figures, ngff_) ──┬──> T3 (build; _writer, _backends) ──> T5 (copy-out; _writer, _failures, _coordinator) ──┐
                                  └──> T4 (store transaction; io_handler, _measurement_tables, output_manager, process_only) ──┤
                                                                                                                              v
                                                         T6 (wire all modes; retire emit_image; port tests) ──> T7 (properties) ──> T8 (docs) ──> T9 (regression)
```

Shared files: `_writer.py` (T3, T5); `_coordinator.py` (T5, T6); `_cli_process_only.py` (T4, T6).
No two clusters below touch disjoint files **and** have no ordering edge, so
execution is **sequential**; no parallel worktrees.

| Cluster | Tasks | Shape | Model / effort |
|---|---|---|---|
| C1 | T1 + T2 + T3 | Keystone (contract, value types, build) | Opus, high |
| C2 | T4 | Seam (root-last transaction, hard-link safety) | Opus, high |
| C3 | T5 | Keystone (copy-out) | Opus, high |
| C4 | T6 | Seam + test-port sweep (4 call sites, emit_image retired) | Opus, high |
| C5 | T7 | Leaf (properties; cache parity may expose a provider bug) | Opus, high |
| C5b | T7a + T7b | Keystone + Seam (apply-state figures; calibration provider; staged split; run folders) | Opus, high |
| C6 | T8 | Sweep (docs; Sphinx build on Slurm) | Sonnet, medium |
| — | T9 | Orchestrator (sharded Slurm regression) | — |

**Gates.**
- Plan review: done (`reports/…/plan-review.md`, `simplicity-review.md`).
- Per cluster (light): orchestrator reviews the diff and runs the cluster's test files; stop and ask on any design fork.
- Phase A = C1–C3 (backend units): `implementation-test-reviewer` (Opus) over the combined diff, then a **simplify pass on the backend layer before Phase B builds on it** (user preference), then re-run Phase A tests.
- Phase B = C4–C5 (wiring + properties): `implementation-test-reviewer` (Opus).
- End: simplify pass over Phase B + seams; then T9.

## File structure

| File | Status | Responsibility |
|---|---|---|
| `src/phenotypic/abc_/plotting/_store_formats.py` | create | Closed format table, defaults, `store=` validation. Stdlib only. |
| `src/phenotypic/abc_/plotting/_pht_plot.py` | modify | `figure(store=)`, `FigureSpec.store` |
| `src/phenotypic/abc_/plotting/__init__.py` | modify | export `StoreFormat` |
| `src/phenotypic/sdk_/ngff_.py` | modify | `FIGURES_GROUP`, `FIGURES_SCHEMA_VERSION`, `PhenotypicAttr.FIGURES` |
| `src/phenotypic/sdk_/_image_figures.py` | create | `StoredFigures` value types; write/apply/read the descriptor |
| `src/phenotypic/plotting/_pipeline/_store_formats.py` | create | Two deterministic serializers |
| `src/phenotypic/plotting/_pipeline/_store_figures.py` | create | `build_image_figures`, format resolution, error normalisation |
| `src/phenotypic/plotting/_pipeline/_backends.py` | modify | `declared_figure_spec`; the preflight judges image plots by a declared `png` |
| `src/phenotypic/plotting/_pipeline/_writer.py` | modify | extract `unique_page_stems` and `_commit_manifest` |
| `src/phenotypic/plotting/_pipeline/_failures.py` | modify | `record_plot_failure(error: BaseException \| str, *, page=None, fmt=None)` |
| `src/phenotypic/plotting/_pipeline/_store_copyout.py` | create | `publish_store_figures` |
| `src/phenotypic/plotting/_pipeline/_coordinator.py` | modify | add `publish_store_figures`; delete `emit_image`, `_publish_image_value` |
| `src/phenotypic/_core/_image_parts/_image_io_handler.py` | modify | `figures=` through `save2zarr` → `_save_store` → `_write_store_part` |
| `src/phenotypic/sdk_/_measurement_tables.py` | modify | `replace_image_tables(*, figures)` (required); `_rewrite_store_tables(clear_figures=)` |
| `src/phenotypic/_cli/_cli_output_manager.py` | modify | `save_image_store(figures=)`, `replace_image_store_measurements(*, figures)` (required) |
| `src/phenotypic/_cli/_cli_process_only.py` | modify | build + `write_process_only_layer(figures=)` |
| `src/phenotypic/_cli/_cli_process_single.py`, `_cli_staged_workers.py` | modify | wiring for full mode, measure mode and Stage 3 |
| `src/phenotypic/_cli/_cli_failure_tracker.py` | modify | revision 3 |
| `tests/unit/plotting/_store_fixtures.py` | create | `figure_store`, `emit_image_via_store` test helpers |

---

### Task 1: The store-format contract and `@figure(store=)`

**Files:**
- Create: `src/phenotypic/abc_/plotting/_store_formats.py`
- Modify: `src/phenotypic/abc_/plotting/_pht_plot.py` (`FigureSpec` ≈104; `figure()` ≈154–268)
- Modify: `src/phenotypic/abc_/plotting/__init__.py`
- Test: `tests/unit/abc_/plotting/test_store_formats.py`

**Interfaces:**
- Produces:
  - `StoreFormat = Literal["plotly-json", "png"]`
  - `StoreFormatInfo(extension: str, media_type: str, backends: frozenset[str])`
  - `STORE_FORMATS: Mapping[str, StoreFormatInfo]`
  - `default_store_formats(backend: str) -> tuple[str, ...]`
  - `resolve_store_formats(store, *, backend, owner) -> tuple[str, ...]`, which raises `TypeError`
  - `FigureSpec.store: tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

```python
"""@figure(store=...) -- the closed format set, defaults and validation (spec §2)."""
from __future__ import annotations

import pytest

from phenotypic.abc_.plotting import PhtPlot, figure
from phenotypic.abc_.plotting._store_formats import (
    STORE_FORMATS,
    default_store_formats,
)


def test_the_format_table_is_the_closed_set_with_its_media_types():
    assert {n: (i.extension, i.media_type) for n, i in STORE_FORMATS.items()} == {
        "plotly-json": (".plotly.json", "application/vnd.plotly.v1+json"),
        "png": (".png", "image/png"),
    }


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("plotly", ("plotly-json",)), ("mpl", ("png",))],
)
def test_an_omitted_store_takes_the_backend_default(backend, expected):
    class Plot(PhtPlot):
        @figure(title="t", backend=backend, primary=True)
        def draw(self, image):
            raise AssertionError("never rendered")

    assert Plot._class_primary_spec().store == expected
    assert default_store_formats(backend) == expected


def test_a_declared_store_is_kept_in_declared_order():
    class Plot(PhtPlot):
        @figure(title="t", backend="plotly", primary=True, store=("png", "plotly-json"))
        def draw(self, image):
            raise AssertionError

    assert Plot._class_primary_spec().store == ("png", "plotly-json")


@pytest.mark.parametrize(
    ("backend", "store", "match"),
    [
        ("mpl", ("plotly-json",), "cannot produce"),
        ("plotly", ("svg",), "unknown"),
        ("plotly", ("png", "png"), "duplicate"),
        ("plotly", (), "at least one"),
        ("plotly", "png", "tuple"),
    ],
)
def test_an_invalid_store_is_refused_at_class_definition(backend, store, match):
    with pytest.raises(TypeError, match=match):

        class Plot(PhtPlot):  # noqa: F841 - the definition itself must raise
            @figure(title="t", backend=backend, primary=True, store=store)
            def draw(self, image):
                raise AssertionError
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/abc_/plotting/test_store_formats.py -p no:cacheprovider -q`
Expected: collection error `ModuleNotFoundError: phenotypic.abc_.plotting._store_formats`.

- [ ] **Step 3: Write `_store_formats.py`**

```python
"""The closed set of formats a per-image figure can be stored in (spec §2).

Standard library only, like the rest of :mod:`phenotypic.abc_.plotting`:
validation runs at class-definition time, and importing a plotting library to
answer "is this name allowed?" would break the lazy-import contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

StoreFormat = Literal["plotly-json", "png"]


@dataclass(frozen=True)
class StoreFormatInfo:
    """How one store format is named on disk and labelled for a consumer.

    Attributes:
        extension: File suffix, including the leading dot.
        media_type: The media type a consumer dispatches on.
        backends: The figure backends that can produce this format.
    """

    extension: str
    media_type: str
    backends: frozenset[str]


#: The folder layout is storage only: a consumer reads ``media_type`` from the
#: descriptor, never the extension.
STORE_FORMATS: Mapping[str, StoreFormatInfo] = MappingProxyType({
    "plotly-json": StoreFormatInfo(
        ".plotly.json", "application/vnd.plotly.v1+json", frozenset({"plotly"})
    ),
    "png": StoreFormatInfo(".png", "image/png", frozenset({"plotly", "mpl"})),
})

_DEFAULTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "plotly": ("plotly-json",),
    "mpl": ("png",),
})


def default_store_formats(backend: str) -> tuple[str, ...]:
    """Return the formats stored when a figure declares none.

    Args:
        backend: ``"plotly"`` or ``"mpl"``.

    Returns:
        The backend's default format tuple.

    Raises:
        ValueError: If *backend* is not a known figure backend.
    """
    try:
        return _DEFAULTS[backend]
    except KeyError:
        raise ValueError(f"unknown figure backend {backend!r}") from None


def resolve_store_formats(
    store: tuple[str, ...] | None, *, backend: str, owner: str
) -> tuple[str, ...]:
    """Validate a ``store=`` declaration, or supply the backend default.

    Args:
        store: The declared formats, or ``None`` for the default.
        backend: The figure's declared backend.
        owner: Method name, for the error message.

    Returns:
        The formats to store, in declared order.

    Raises:
        TypeError: On a bare string, an empty tuple, an unknown or duplicate
            name, or a format the backend cannot produce.
    """
    if store is None:
        return default_store_formats(backend)
    if isinstance(store, str):
        raise TypeError(
            f"@figure({owner!r}): store must be a tuple of format names, "
            f"got the string {store!r}; write store=({store!r},)"
        )
    formats = tuple(store)
    if not formats:
        raise TypeError(
            f"@figure({owner!r}): store=() is refused -- an image figure must "
            "store at least one format, because deliverables/plots is copied "
            "out of the store and an unstored figure would appear nowhere"
        )
    unknown = [name for name in formats if name not in STORE_FORMATS]
    if unknown:
        raise TypeError(
            f"@figure({owner!r}): unknown store format(s) {unknown}; "
            f"expected any of {list(STORE_FORMATS)}"
        )
    duplicates = sorted({name for name in formats if formats.count(name) > 1})
    if duplicates:
        raise TypeError(f"@figure({owner!r}): duplicate store format(s) {duplicates}")
    unsupported = [n for n in formats if backend not in STORE_FORMATS[n].backends]
    if unsupported:
        raise TypeError(
            f"@figure({owner!r}): backend={backend!r} cannot produce "
            f"store format(s) {unsupported}"
        )
    return formats


__all__ = [
    "STORE_FORMATS",
    "StoreFormat",
    "StoreFormatInfo",
    "default_store_formats",
    "resolve_store_formats",
]
```

- [ ] **Step 4: Thread `store` through `figure()` and `FigureSpec`**

In `_pht_plot.py`:
- Import `from ._store_formats import StoreFormat, resolve_store_formats` beside the `._output` import.
- On `FigureSpec`, add `store: tuple[str, ...]` after `backend`. Document it under `Attributes:` as *"Formats a `PlotImage` publication stores for this figure (spec §2)."*
- On `figure()`, add `store: tuple[StoreFormat, ...] | None = None,` after `backend`. Document it under `Args:` as *"Formats to store in the image's OME-Zarr store. `None` stores the backend default (`("plotly-json",)` / `("png",)`). Validated when the class is defined."* Extend `Raises:` with a `TypeError` entry: *"or if `store` is invalid"*.
- In `decorator`, right after the controls check, add `resolved_store = resolve_store_formats(store, backend=backend, owner=fn.__name__)` and pass `store=resolved_store` to `FigureSpec(...)`. `wrapper` does not change.

In `abc_/plotting/__init__.py`: add `from ._store_formats import StoreFormat`, and add `"StoreFormat"` to `__all__`.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/abc_/plotting/ -p no:cacheprovider -q`
Expected: all pass, including the existing `test_pht_plot.py`, `test_figure_backend.py` and `test_imports.py`. `grep -rn "FigureSpec(" src tests` should show only `_pht_plot.py`. `_image_plots.py` builds its specs with `replace(...)`, which carries the new field over.

- [ ] **Step 6: Prove it can fail**

Change `if not formats:` to `if False:` and run the file. Expect the `at least one` case to FAIL. Restore the line.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix src/phenotypic/abc_/plotting/ tests/unit/abc_/plotting/test_store_formats.py
git add src/phenotypic/abc_/plotting/ tests/unit/abc_/plotting/test_store_formats.py
git commit -m "feat(plotting): @figure(store=) over a closed set of store formats"
```

---

### Task 2: Store-side value types and the `figures/` writer

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`. Add the constants after `METADATA_TABLE_SCHEMA_VERSION` (≈93), and add `FIGURES` to `PhenotypicAttr` after `TABLES` (≈464).
- Create: `src/phenotypic/sdk_/_image_figures.py`
- Test: `tests/unit/sdk_/test_image_figures.py`

**Interfaces:**
- Produces, all in `phenotypic.sdk_._image_figures` and not re-exported:
  - `StoredFigureFile(format, media_type, filename, data: bytes)`
  - `StoredFigurePage(key, label, backend, metadata: Mapping, files: tuple[StoredFigureFile, ...])`
  - `StoredFigureBinding(binding_id, plot_class, directory, pages: tuple[StoredFigurePage, ...])`
  - `StoredFigureFailure(binding, page: str | None, format: str | None, error)`
  - `StoredFigures(bindings: tuple[...], failed: tuple[...])`
  - `write_image_figures(store_part: Path, figures: StoredFigures) -> dict`
  - `apply_image_figures_attributes(phenotypic: dict, fragment: dict | None) -> None`
  - `read_image_figures_descriptor(store_path: Path) -> dict | None`
- Also produces `ngff_.FIGURES_GROUP = "figures"`, `ngff_.FIGURES_SCHEMA_VERSION = 1` and `PhenotypicAttr.FIGURES = "figures"`.

This module is storage-neutral. Filenames, directory names, formats and media types are all decided upstream, in Task 3.

- [ ] **Step 1: Write the failing tests**

```python
"""figures/ group + attributes.phenotypic.figures descriptor (spec §1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    apply_image_figures_attributes,
    read_image_figures_descriptor,
    write_image_figures,
)

_JSON = b'{"data": []}'
_PNG = b"\x89PNG fake"


def _stored() -> StoredFigures:
    page = StoredFigurePage(
        key="default", label=None, backend="plotly", metadata={"plate": 1},
        files=(
            StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                             "default.plotly.json", _JSON),
            StoredFigureFile("png", "image/png", "default.png", _PNG),
        ),
    )
    return StoredFigures(
        bindings=(StoredFigureBinding("sym", "MeasureSymZones", "sym", (page,)),),
        failed=(StoredFigureFailure("orient", None, None, "RuntimeError: boom"),),
    )


def test_writer_lays_out_groups_files_and_a_hash_bound_descriptor(tmp_path: Path):
    fragment = write_image_figures(tmp_path, _stored())
    group = json.loads((tmp_path / "figures" / "zarr.json").read_text())
    assert group == {"zarr_format": 3, "node_type": "group", "attributes": {}}
    assert json.loads((tmp_path / "figures" / "sym" / "zarr.json").read_text()) == group
    assert (tmp_path / "figures/sym/default.plotly.json").read_bytes() == _JSON

    descriptor = fragment[ngff_.PhenotypicAttr.FIGURES]
    assert descriptor["schema_version"] == 1
    assert descriptor["bindings"]["sym"]["class"] == "MeasureSymZones"
    page = descriptor["bindings"]["sym"]["pages"][0]
    assert page["metadata"] == {"plate": 1}
    assert [f["format"] for f in page["files"]] == ["plotly-json", "png"]
    for entry in page["files"]:
        data = (tmp_path / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert descriptor["failed"] == [
        {"binding": "orient", "page": None, "format": None, "error": "RuntimeError: boom"}
    ]


def test_all_failed_writes_the_key_with_empty_bindings(tmp_path: Path):
    stored = StoredFigures(bindings=(), failed=_stored().failed)
    descriptor = write_image_figures(tmp_path, stored)[ngff_.PhenotypicAttr.FIGURES]
    assert descriptor["bindings"] == {}
    assert (tmp_path / "figures" / "zarr.json").is_file()


def test_apply_sets_and_removes_the_key():
    phenotypic: dict = {"figures": {"stale": True}}
    apply_image_figures_attributes(phenotypic, None)
    assert "figures" not in phenotypic
    apply_image_figures_attributes(phenotypic, {"figures": {"schema_version": 1}})
    assert phenotypic["figures"] == {"schema_version": 1}


def test_reader_returns_none_for_a_pre_feature_store(tmp_path: Path):
    (tmp_path / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group",
         "attributes": {"phenotypic": {"store_schema_version": 3}}}
    ))
    assert read_image_figures_descriptor(tmp_path) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/sdk_/test_image_figures.py -p no:cacheprovider -q`
Expected: `ModuleNotFoundError: phenotypic.sdk_._image_figures`.

- [ ] **Step 3: Add the `ngff_` constants**

```python
#: Per-image figures (spec 2026-09-22 §1). A Zarr v3 group holding non-Zarr
#: files, exactly as `tables/` holds `table.parquet`; described by
#: `attributes.phenotypic.figures`, never by `ome.series`.
FIGURES_GROUP: Final[str] = "figures"
FIGURES_SCHEMA_VERSION: Final[int] = 1
```

`PhenotypicAttr`: `FIGURES: Final[str] = "figures"` after `TABLES`.

- [ ] **Step 4: Write `_image_figures.py`**

```python
"""Per-image figures inside an OME-Zarr store (spec 2026-09-22 §1).

Storage-neutral: the caller decides directory names, filenames, formats and
media types; this module writes bytes where it is told, hashes them, and
describes them. It never imports a plotting library.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._atomic_io import atomic_write_json

#: Same document every `tables/*` group uses.
_GROUP_DOCUMENT: dict[str, object] = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {},
}


@dataclass(frozen=True)
class StoredFigureFile:
    """One rendering of one page. ``filename`` is final; no separators."""

    format: str
    media_type: str
    filename: str
    data: bytes


@dataclass(frozen=True)
class StoredFigurePage:
    """One page and the renderings that succeeded for it.

    ``metadata`` is JSON-native by the time it gets here: the builder
    refuses a page whose metadata does not serialize (spec §1).
    """

    key: str
    label: str | None
    backend: str
    metadata: Mapping[str, Any]
    files: tuple[StoredFigureFile, ...]


@dataclass(frozen=True)
class StoredFigureBinding:
    """One binding's pages. ``directory`` is the sanitized group name."""

    binding_id: str
    plot_class: str
    directory: str
    pages: tuple[StoredFigurePage, ...]


@dataclass(frozen=True)
class StoredFigureFailure:
    """A failure at the finest level available (spec §1)."""

    binding: str
    page: str | None
    format: str | None
    error: str


@dataclass(frozen=True)
class StoredFigures:
    """Everything one image's store will say about its figures, in order."""

    bindings: tuple[StoredFigureBinding, ...]
    failed: tuple[StoredFigureFailure, ...]


def write_image_figures(
    store_part: Path, figures: StoredFigures
) -> dict[str, object]:
    """Write ``figures/`` into an unpromoted part and return its descriptor.

    Args:
        store_part: An unpromoted ``*.ome.zarr.part`` directory in which
            ``figures/`` does not yet exist. Callers rewriting a promoted store
            remove the part's copied ``figures/`` first: those copies are hard
            links into the live store, and writing through one would change
            the published bytes (``replace_image_tables``).
        figures: The built figures.

    Returns:
        ``{"figures": descriptor}``, to apply with
        :func:`apply_image_figures_attributes`.
    """
    from . import ngff_

    group = Path(store_part) / ngff_.FIGURES_GROUP
    group.mkdir(parents=True, exist_ok=True)
    atomic_write_json(group / ngff_.STORE_ROOT_JSON, _GROUP_DOCUMENT)
    bindings: dict[str, object] = {}
    for binding in figures.bindings:
        directory = group / binding.directory
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / ngff_.STORE_ROOT_JSON, _GROUP_DOCUMENT)
        pages = []
        for page in binding.pages:
            entries = []
            for stored in page.files:
                (directory / stored.filename).write_bytes(stored.data)
                entries.append({
                    "format": stored.format,
                    "media_type": stored.media_type,
                    "path": f"{ngff_.FIGURES_GROUP}/{binding.directory}/{stored.filename}",
                    "sha256": hashlib.sha256(stored.data).hexdigest(),
                })
            pages.append({
                "key": page.key,
                "label": page.label,
                "backend": page.backend,
                "metadata": dict(page.metadata),
                "files": entries,
            })
        bindings[binding.binding_id] = {"class": binding.plot_class, "pages": pages}
    descriptor = {
        "schema_version": ngff_.FIGURES_SCHEMA_VERSION,
        "bindings": bindings,
        "failed": [
            {"binding": f.binding, "page": f.page, "format": f.format, "error": f.error}
            for f in figures.failed
        ],
    }
    return {ngff_.PhenotypicAttr.FIGURES: descriptor}


def apply_image_figures_attributes(
    phenotypic: dict[str, object], fragment: dict[str, object] | None
) -> None:
    """Make the root's ``figures`` key equal *fragment*, removal included.

    ``None`` removes the key: a pipeline with no ``PlotImage`` binding has no
    figures, and a measure-mode rebuild must drop a stale descriptor -- the
    same total-function rule as ``apply_image_tables_attributes``.
    """
    from . import ngff_

    if fragment is None:
        phenotypic.pop(ngff_.PhenotypicAttr.FIGURES, None)
    else:
        phenotypic[ngff_.PhenotypicAttr.FIGURES] = fragment[ngff_.PhenotypicAttr.FIGURES]


def read_image_figures_descriptor(store_path: Path) -> dict[str, Any] | None:
    """Return a store's figures descriptor, or ``None`` when it has none."""
    from . import ngff_

    descriptor = ngff_.read_phenotypic_attributes(Path(store_path)).get(
        ngff_.PhenotypicAttr.FIGURES
    )
    return descriptor if isinstance(descriptor, dict) else None


__all__ = [
    "StoredFigureBinding",
    "StoredFigureFailure",
    "StoredFigureFile",
    "StoredFigurePage",
    "StoredFigures",
    "apply_image_figures_attributes",
    "read_image_figures_descriptor",
    "write_image_figures",
]
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_image_figures.py tests/unit/ci/test_startup_imports.py -p no:cacheprovider -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/_image_figures.py src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_image_figures.py
git add src/phenotypic/sdk_/_image_figures.py src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_image_figures.py
git commit -m "feat(sdk): figures/ group writer and attributes.phenotypic.figures descriptor"
```

---

### Task 3: Build per-image figures in memory (serializers, build, preflight)

**Files:**
- Create: `src/phenotypic/plotting/_pipeline/_store_formats.py`
- Create: `src/phenotypic/plotting/_pipeline/_store_figures.py`
- Modify: `src/phenotypic/plotting/_pipeline/_backends.py`. Split `declared_figure_spec` out of `_declared_backends` (≈221), and change the preflight (≈166).
- Modify: `src/phenotypic/plotting/_pipeline/_writer.py`. Extract `unique_page_stems` from `_publish_plot_output_locked` (≈334–349).
- Test: `tests/unit/plotting/test_store_figures_build.py`, `tests/unit/plotting/test_backends.py` (append)

**Interfaces:**
- Consumes: `STORE_FORMATS`, `default_store_formats` (Task 1); `StoredFigure*` (Task 2).
- Produces:
  - `serialize_store_format(fmt, figure, *, binding_id, page_key) -> bytes`
  - `build_image_figures(pipeline, image) -> StoredFigures | None`. Returns `None` iff the pipeline has no `PlotImage` binding.
  - `normalize_figure_error(exc: BaseException) -> str`
  - `declared_figure_spec(plot) -> FigureSpec | None`
  - `unique_page_stems(names: Sequence[tuple[str, str]]) -> list[str]`, where each input pair is `(page_key, preferred_name)`

- [ ] **Step 1: Extract `unique_page_stems` in `_writer.py` (behaviour-preserving)**

Add above `publish_plot_output`:

```python
def unique_page_stems(names: Sequence[tuple[str, str]]) -> list[str]:
    """Return one filesystem stem per page, unique under case folding.

    Args:
        names: ``(page_key, preferred_name)`` per page, in page order. The
            manifest writer prefers the label; the store uses the key.

    Returns:
        Stems in the same order. Sanitization is many-to-one, so a collision
        gets a digest suffix derived from the page key -- stable across reruns.
    """
    used: dict[str, str] = {}
    stems: list[str] = []
    for key, preferred in names:
        try:
            stem = safe_path_component(preferred)
        except Exception:
            stem = "page"
        base_stem = stem
        folded = stem.casefold()
        attempt = 0
        while folded in used and used[folded] != key:
            digest_input = key if attempt == 0 else f"{key}:{attempt}"
            digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8]
            stem = f"{base_stem}-{digest}"
            folded = stem.casefold()
            attempt += 1
        used[folded] = key
        stems.append(stem)
    return stems
```

In `_publish_plot_output_locked`:
- Delete `used` and the per-page stem block.
- Before the loop, compute `stems = unique_page_stems([(p.key, p.label or p.key) for p in output.pages])`.
- Iterate with `for page, stem in zip(output.pages, stems):`.
- Add `Sequence` to the `collections.abc` import.

Run: `uv run pytest tests/unit/plotting/test_output_adapter.py -p no:cacheprovider -q`
Expected: PASS, with no behaviour change.

- [ ] **Step 2: Split `declared_figure_spec` out of `_declared_backends`**

In `_backends.py`, move the body of `_declared_backends` into:

```python
def declared_figure_spec(plot: Any) -> Any:
    """Return the ``FigureSpec`` *plot*'s ``inspect()`` renders, if declared.

    The three-step rule documented on :func:`_declared_backends`, returning
    the spec rather than its backend so a caller can also read ``spec.store``.
    """
    from phenotypic.abc_.plotting import PhtPlot

    if isinstance(plot, PhtPlot):
        owner: Any = type(plot)
        primary_spec = plot._primary_spec
    else:
        owner = plot.cls
        primary_spec = owner._class_primary_spec
    effective_inspect = owner.inspect
    declared = getattr(effective_inspect, "__figure_spec__", None)
    if declared is not None:
        return declared
    if effective_inspect is not PhtPlot.inspect:
        return None
    try:
        return primary_spec()
    except RuntimeError:
        return None
```

`_declared_backends` keeps its docstring, and its body becomes `spec = declared_figure_spec(plot); return spec.backend if spec is not None else None`.

- [ ] **Step 3: Write the failing build tests**

```python
"""build_image_figures: in memory, finest-grained failures (spec §1, §3 step 1)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pydantic import BaseModel

from phenotypic import ImagePipeline
from phenotypic.abc_.plotting import PlotImage, PlotOutput, PlotPage, figure
from phenotypic.plotting._pipeline._store_figures import (
    build_image_figures,
    normalize_figure_error,
)


class Bars(BaseModel, PlotImage):
    @figure(title="bars", backend="plotly", primary=True)
    def draw(self, image):
        import plotly.graph_objects as go

        return go.Figure(go.Bar(x=["a"], y=[1]))


class BarsWithPng(BaseModel, PlotImage):
    @figure(title="bars", backend="plotly", primary=True, store=("plotly-json", "png"))
    def draw(self, image):
        import plotly.graph_objects as go

        return go.Figure(go.Bar(x=["a"], y=[1]))


class MplLine(BaseModel, PlotImage):
    @figure(title="line", backend="mpl", primary=True)
    def draw(self, image):
        from matplotlib.figure import Figure

        fig = Figure()
        fig.subplots().plot([0, 1])
        return fig


class HandBuiltPages(BaseModel, PlotImage):
    """Overrides inspect(): no spec, so each page takes its backend default."""

    def inspect(self, subject=None, *, for_save=False, **overrides):
        import plotly.graph_objects as go
        from matplotlib.figure import Figure

        return PlotOutput(pages=(
            PlotPage(key="A b", figure=go.Figure(), label="First"),
            PlotPage(key="a-b", figure=Figure()),
            PlotPage(key="odd", figure=object()),
            PlotPage(key="np", figure=go.Figure(), metadata={"n": np.int64(3)}),
        ))


class Explodes(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise RuntimeError(f"bad object at {hex(id(self))}")


class ReturnsNone(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        return None


def _build(*plots):
    return build_image_figures(ImagePipeline(plots=list(plots)), object())


def test_no_image_binding_builds_nothing():
    assert build_image_figures(ImagePipeline(), object()) is None


def test_default_plotly_stores_plotly_json_only():
    stored = _build(Bars())
    [binding] = stored.bindings
    assert (binding.binding_id, binding.directory) == ("Bars", "Bars")
    [page] = binding.pages
    assert [(f.format, f.filename) for f in page.files] == [
        ("plotly-json", "default.plotly.json")
    ]
    assert json.loads(page.files[0].data)["data"][0]["type"] == "bar"
    assert stored.failed == ()


def test_mpl_default_stores_png():
    [page] = _build(MplLine()).bindings[0].pages
    assert [f.format for f in page.files] == ["png"]
    assert page.backend == "mpl"
    assert page.files[0].data.startswith(b"\x89PNG")


def test_a_declared_png_without_chrome_fails_that_format_only(monkeypatch):
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    stored = _build(BarsWithPng())
    [page] = stored.bindings[0].pages
    assert [f.format for f in page.files] == ["plotly-json"]
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("BarsWithPng", "default", "png")
    assert failure.error.startswith("PlotBackendUnavailable: ")


def test_a_declared_png_with_chrome_stores_what_kaleido_returns(monkeypatch):
    import plotly.io as pio

    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)
    monkeypatch.setattr(pio, "to_image", lambda fig, format: b"\x89PNG kaleido")
    [page] = _build(BarsWithPng()).bindings[0].pages
    assert [(f.format, f.data) for f in page.files][1] == ("png", b"\x89PNG kaleido")


def test_hand_built_pages_use_backend_defaults_and_collision_safe_names():
    stored = _build(HandBuiltPages())
    [binding] = stored.bindings
    names = {page.key: [f.filename for f in page.files] for page in binding.pages}
    assert names["A b"] == ["A-b.plotly.json"]
    [mpl_name] = names["a-b"]
    assert mpl_name.endswith(".png") and mpl_name != "A-b.png"
    assert "odd" not in names and "np" not in names
    by_page = {f.page: f for f in stored.failed}
    assert by_page["odd"].format is None
    assert by_page["odd"].error.startswith("TypeError: unsupported figure type")
    assert by_page["np"].format is None
    assert by_page["np"].error.startswith("TypeError: ")


def test_inspect_raising_omits_the_binding_and_normalises_the_address():
    stored = _build(Explodes(), Bars())
    assert [b.binding_id for b in stored.bindings] == ["Bars"]
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("Explodes", None, None)
    assert failure.error == "RuntimeError: bad object at 0x…"


def test_inspect_returning_none_is_a_failure_not_an_absence():
    stored = _build(ReturnsNone())
    assert stored.bindings == ()
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("ReturnsNone", None, None)


def test_a_binding_whose_every_page_failed_is_absent(monkeypatch):
    from phenotypic.plotting._pipeline import _backends

    class PngOnly(BaseModel, PlotImage):
        @figure(title="p", backend="plotly", primary=True, store=("png",))
        def draw(self, image):
            import plotly.graph_objects as go

            return go.Figure()

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    stored = _build(PngOnly())
    assert stored.bindings == ()
    assert [(f.page, f.format) for f in stored.failed] == [("default", "png")]


def test_an_unexpected_error_outside_inspect_stays_inside_the_binding(monkeypatch):
    from phenotypic.plotting._pipeline import _store_figures

    def _boom(plot):
        raise LookupError("resolver broke")

    monkeypatch.setattr(_store_figures, "declared_figure_spec", _boom)
    stored = _build(Bars())
    assert stored.bindings == ()
    assert stored.failed[0].error == "LookupError: resolver broke"


def test_normalize_figure_error_replaces_every_address():
    assert normalize_figure_error(ValueError("0xdead and 0xBEEF1")) == "ValueError: 0x… and 0x…"


def test_figures_are_closed_after_serialization(monkeypatch):
    from phenotypic.plotting._pipeline import _store_figures

    closed = []
    monkeypatch.setattr(_store_figures.FigureAdapter, "close", staticmethod(closed.append))
    _build(MplLine())
    assert len(closed) == 1


@pytest.mark.parametrize("backend", ["plotly", "mpl"])
def test_the_default_serializer_is_stable_across_processes(backend):
    """Spec §4: fresh interpreters (fresh hash seeds, fresh addresses) agree."""
    make = {
        "plotly": "import plotly.graph_objects as go\nfig = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2]))\n",
        "mpl": "from matplotlib.figure import Figure\nfig = Figure()\nfig.subplots().plot([1, 2, 3], [3, 1, 2])\n",
    }[backend]
    fmt = "plotly-json" if backend == "plotly" else "png"
    code = make + textwrap.dedent(f"""
        import hashlib
        from phenotypic.plotting._pipeline._store_formats import serialize_store_format
        data = serialize_store_format({fmt!r}, fig, binding_id="b", page_key="default")
        print(hashlib.sha256(data).hexdigest())
    """)

    def digest(seed: str) -> str:
        import os

        env = {**os.environ, "PYTHONHASHSEED": seed}
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            check=True, env=env,
        ).stdout.strip()

    assert digest("1") == digest("2")
```

- [ ] **Step 4: Run to verify they fail**

Run: `uv run pytest tests/unit/plotting/test_store_figures_build.py -p no:cacheprovider -q`
Expected: `ModuleNotFoundError: phenotypic.plotting._pipeline._store_figures`.

- [ ] **Step 5: Write `_store_formats.py` (the serializers)**

```python
"""One deterministic serializer per store format (spec §2 "Serializers").

Plotting libraries are imported inside each function (lazy-import contract).
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

from phenotypic.abc_.plotting import figure_backend_of


def _serialize_plotly_json(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    # `to_json` drops trace uids by default, the one per-object random field.
    return figure.to_json().encode("utf-8")


def _serialize_png(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    if figure_backend_of(figure) == "plotly":
        # Module attribute access, not a bound name, so a test that patches
        # `_backends.chrome_available` is honoured (see `_writer.py` imports).
        from . import _backends

        if not _backends.chrome_available():
            raise _backends.PlotBackendUnavailable(
                "Plotly PNG export needs Chrome (kaleido); install it with "
                "plotly_get_chrome"
            )
        import plotly.io as pio

        return pio.to_image(figure, format="png")
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()


_SERIALIZERS: dict[str, Callable[..., bytes]] = {
    "plotly-json": _serialize_plotly_json,
    "png": _serialize_png,
}


def serialize_store_format(
    fmt: str, figure: Any, *, binding_id: str, page_key: str
) -> bytes:
    """Serialize *figure* to one store format.

    Args:
        fmt: A name from ``STORE_FORMATS``.
        figure: A figure whose backend supports *fmt* (checked by the caller).
        binding_id: The plot binding id. Unused by the current formats; kept
            so a format that must salt generated ids has what it needs.
        page_key: The page key; same reason.

    Returns:
        The encoded bytes.

    Raises:
        KeyError: If *fmt* is not a store format.
        PlotBackendUnavailable: A Plotly PNG without Chrome.
    """
    return _SERIALIZERS[fmt](figure, binding_id=binding_id, page_key=page_key)


__all__ = ["serialize_store_format"]
```

- [ ] **Step 6: Write `_store_figures.py`**

```python
"""Build one image's figures in memory for the store (spec §3 step 1).

Writes nothing, so it is safe to call anywhere before a store transaction.
Everything per binding sits inside that binding's failure boundary; only
``PlotPublicationBlocked`` propagates, as in every handler.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from phenotypic.abc_.plotting import PlotImage, figure_backend_of
from phenotypic.abc_.plotting._store_formats import STORE_FORMATS, default_store_formats
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
)

from ._adapter import FigureAdapter
from ._backends import declared_figure_spec
from ._failures import _format_error
from ._output import normalize_plot_output
from ._store_formats import serialize_store_format
from ._writer import PlotPublicationBlocked, safe_path_component, unique_page_stems

logger = logging.getLogger(__name__)

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def normalize_figure_error(error: BaseException) -> str:
    """Spell *error* for the store: ``"Type: message"``, addresses masked.

    A CPython object address differs every run, so a deterministic failure
    would otherwise produce different store bytes each time (spec §1).
    """
    return _ADDRESS.sub("0x…", _format_error(error))


def build_image_figures(pipeline: Any, image: Any) -> StoredFigures | None:
    """Render every ``PlotImage`` binding of *pipeline* for *image*.

    Args:
        pipeline: An ``ImagePipeline`` with normalized plot bindings.
        image: The image each binding's ``inspect`` receives.

    Returns:
        ``None`` when the pipeline has no ``PlotImage`` binding -- the store
        then carries no ``figures`` key at all. Otherwise the built value,
        which holds no bindings if every one failed.

    Raises:
        PlotPublicationBlocked: Never swallowed.
    """
    image_bindings = [b for b in pipeline.get_plots() if isinstance(b.plot, PlotImage)]
    if not image_bindings:
        return None
    built: list[StoredFigureBinding] = []
    failed: list[StoredFigureFailure] = []
    for binding in image_bindings:
        try:
            value = binding.plot.inspect(image, for_save=True)
            if value is None:
                raise TypeError("inspect() returned None; expected a figure or PlotOutput")
            pages = _build_pages(binding, value, failed)
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - one figure never kills an image
            logger.warning("Plot %s failed while building its figure", binding.id, exc_info=exc)
            failed.append(StoredFigureFailure(binding.id, None, None, normalize_figure_error(exc)))
            continue
        if pages:
            built.append(StoredFigureBinding(
                binding_id=binding.id,
                plot_class=type(binding.plot).__name__,
                directory=safe_path_component(binding.id),
                pages=tuple(pages),
            ))
    return StoredFigures(bindings=tuple(built), failed=tuple(failed))


def _build_pages(
    binding: Any, value: Any, failed: list[StoredFigureFailure]
) -> list[StoredFigurePage]:
    """Serialize every page of one binding, recording page/format failures."""
    output = normalize_plot_output(value)
    try:
        spec = declared_figure_spec(binding.plot)
        stems = unique_page_stems([(page.key, page.key) for page in output.pages])
    except BaseException:
        for page in output.pages:
            FigureAdapter.close(page.figure)
        raise
    pages: list[StoredFigurePage] = []
    for page, stem in zip(output.pages, stems):
        try:
            built = _build_page(binding, page, stem, spec, failed)
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - per-page best effort
            failed.append(StoredFigureFailure(
                binding.id, page.key, None, normalize_figure_error(exc)
            ))
            built = None
        finally:
            FigureAdapter.close(page.figure)
        if built is not None:
            pages.append(built)
    return pages


def _build_page(
    binding: Any,
    page: Any,
    stem: str,
    spec: Any,
    failed: list[StoredFigureFailure],
) -> StoredFigurePage | None:
    """One page: backend check, metadata check, then each declared format."""
    backend = figure_backend_of(page.figure)
    if backend is None:
        raise TypeError(
            "unsupported figure type "
            f"{type(page.figure).__module__}.{type(page.figure).__qualname__}"
        )
    metadata = dict(page.metadata)
    # The measure-mode root is written without `default=`, so a numpy scalar
    # here would fail the whole store rewrite; refuse the page instead (spec §1).
    json.dumps(metadata)
    formats = spec.store if spec is not None else default_store_formats(backend)
    files: list[StoredFigureFile] = []
    for fmt in formats:
        info = STORE_FORMATS[fmt]
        try:
            if backend not in info.backends:
                raise TypeError(f"a {backend} figure cannot be stored as {fmt}")
            data = serialize_store_format(
                fmt, page.figure, binding_id=binding.id, page_key=page.key
            )
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - per-format best effort
            failed.append(StoredFigureFailure(
                binding.id, page.key, fmt, normalize_figure_error(exc)
            ))
            continue
        files.append(StoredFigureFile(fmt, info.media_type, f"{stem}{info.extension}", data))
    if not files:
        return None
    return StoredFigurePage(
        key=page.key, label=page.label, backend=backend,
        metadata=metadata, files=tuple(files),
    )


__all__ = ["build_image_figures", "normalize_figure_error"]
```

The reviewer checks these rules against spec §1 and §2:
- A declared spec's `store` applies to every page of that binding.
- A page whose backend contradicts a declared format fails that one format only.
- An unknown backend fails the whole page with `format: None`.
- Metadata that is not JSON-native also fails the whole page with `format: None`.
- A page with no file is omitted.
- A binding with no page is omitted.

- [ ] **Step 7: The preflight judges image plots by their declared `png`**

Append to `tests/unit/plotting/test_backends.py`:

```python
def test_image_plots_need_chrome_only_when_they_declare_png(monkeypatch):
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotImage, figure
    from phenotypic.plotting._pipeline import _backends

    class Img(BaseModel, PlotImage):
        @figure(title="t", backend="plotly", primary=True)
        def draw(self, image):
            raise AssertionError

    class ImgPng(BaseModel, PlotImage):
        @figure(title="t", backend="plotly", primary=True, store=("png",))
        def draw(self, image):
            raise AssertionError

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    assert _backends.preflight_plot_backends(ImagePipeline(plots=[Img()])) == []
    [line] = _backends.preflight_plot_backends(ImagePipeline(plots=[ImgPng()]))
    assert "ImgPng" in line
```

In `preflight_plot_backends`, at the top of the loop:

```python
        if isinstance(binding.plot, PlotImage):
            spec = declared_figure_spec(binding.plot)
            # An image plot renders PNG only if it declared it (spec §2);
            # an undeclared one stores its backend default, which for Plotly
            # needs no Chrome.
            if spec is not None and spec.backend == "plotly" and "png" in spec.store:
                plotly_ids.append(binding.id)
            elif spec is not None and spec.backend == "mpl":
                mpl_ids.append(binding.id)
            elif spec is not None:
                _require_importable("plotly", "plotly", [binding.id])
            continue
```

Import `PlotImage` inside the function, beside the existing function-scope imports.

Run: `uv run pytest tests/unit/plotting/test_backends.py -p no:cacheprovider -q`
Expected: the new test passes. An existing preflight test that used a `PlotImage` fixture and expected the old "HTML only" line needs one of two fixes:
- If the test is about aggregate wording, change its fixture to `PlotMeas`.
- If it is about image plots, change its assertion to the new rule.

- [ ] **Step 8: Run to verify they pass**

Run: `uv run pytest tests/unit/plotting/test_store_figures_build.py tests/unit/plotting/test_backends.py tests/unit/plotting/test_output_adapter.py -p no:cacheprovider -q`
Expected: PASS.

- [ ] **Step 9: Prove it can fail**

- Remove the `_ADDRESS.sub` call. Expect the address test to FAIL, then restore the call.
- Replace `json.dumps(metadata)` with `pass`. Expect `test_hand_built_pages…` to FAIL on `"np" not in names`, then restore it.

- [ ] **Step 10: Commit**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_figures_build.py tests/unit/plotting/test_backends.py
git add src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_figures_build.py tests/unit/plotting/test_backends.py
git commit -m "feat(plotting): build per-image figures in memory for the store"
```

---

### Task 4: Write figures inside the store transaction

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`:
  - `save2zarr` (≈1068)
  - `_save_store` (≈1128)
  - `_write_store_part` (≈1212; tables block at 1377–1389, root at 1391–1418)
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`: `_rewrite_store_tables` (≈632), `replace_image_tables` (≈698)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`: `save_image_store` (≈1842), `replace_image_store_measurements` (≈1941)
- Modify: `src/phenotypic/_cli/_cli_process_only.py`: `write_process_only_layer` (≈152)
- Modify: `tests/unit/cli/conftest.py:835` and `tests/unit/cli/test_embedded_table_inversion.py:293`. They call `replace_image_store_measurements`; add `figures=None` to each call.
- Test: `tests/unit/sdk_/test_image_figures_store.py`

**Interfaces:**
- Consumes: `StoredFigures`, `write_image_figures`, `apply_image_figures_attributes` (Task 2).
- Produces:
  - `figures: StoredFigures | None = None` on `Image.save2zarr`, `_save_store`, `_write_store_part`, `OutputManager.save_image_store` and `write_process_only_layer`.
  - A **required** keyword-only `figures: StoredFigures | None` on `replace_image_tables` and `OutputManager.replace_image_store_measurements`. These functions always rebuild the figures group; `None` removes it.
  - The private helper `_rewrite_store_tables(..., clear_figures: bool = False)`.

- [ ] **Step 1: Write the failing tests**

```python
"""figures/ rides the root-last transaction (spec §1, §3 step 2, measure mode)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from phenotypic import Image
from phenotypic._cli._embedded_measurement_tables import prepare_image_tables
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_image_figures_descriptor,
)
from phenotypic.sdk_._measurement_tables import replace_image_tables


def _figures(tag: bytes = b"one", binding: str = "sym") -> StoredFigures:
    page = StoredFigurePage("default", None, "plotly", {}, (
        StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                         "default.plotly.json", tag),
    ))
    return StoredFigures((StoredFigureBinding(binding, "X", binding, (page,)),), ())


def _tables():
    return prepare_image_tables(pd.DataFrame({"Object_Label": [1]}), None)


@pytest.fixture(scope="module")
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def test_save2zarr_writes_figures_and_an_independent_reader_opens_them(tmp_path, plate):
    import zarr

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    descriptor = read_image_figures_descriptor(store)
    assert descriptor["bindings"]["sym"]["pages"][0]["files"][0]["path"] == (
        "figures/sym/default.plotly.json"
    )
    root = zarr.open_group(str(store), mode="r")
    assert isinstance(root["figures"], zarr.Group)
    assert isinstance(root["figures/sym"], zarr.Group)
    ome = json.loads((store / "OME" / "zarr.json").read_text())
    assert "figures" not in ome["attributes"]["ome"]["series"]
    assert "figures" not in (store / "OME" / "METADATA.ome.xml").read_text()


def test_no_figures_means_no_key_and_no_group(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert read_image_figures_descriptor(store) is None
    assert not (store / "figures").exists()


def test_process_writer_carries_figures_inside_the_consolidated_store(tmp_path, plate):
    from phenotypic._cli._cli_process_only import write_process_only_layer

    out = tmp_path / "p.ome.zarr"
    write_process_only_layer(plate, "rgb", out, fmt="zarr", figures=_figures())
    assert (out / "figures/sym/default.plotly.json").read_bytes() == b"one"
    root = json.loads((out / "zarr.json").read_text())
    assert "figures/sym" in root["consolidated_metadata"]["metadata"]


def test_measure_rebuild_replaces_the_group_and_keeps_pixels_linked(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old", "gone"))
    pixel = next(
        p for p in (store / "rgb" / "0").rglob("*") if p.is_file() and p.name != "zarr.json"
    )
    pixel_inode = pixel.stat().st_ino
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
        figures=_figures(b"new", "kept"),
    )
    assert not (store / "figures/gone").exists()
    assert (store / "figures/kept/default.plotly.json").read_bytes() == b"new"
    assert list(read_image_figures_descriptor(store)["bindings"]) == ["kept"]
    assert pixel.stat().st_ino == pixel_inode


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="holds a descriptor across a directory rename, which Windows refuses",
)
def test_a_same_name_rebuild_never_writes_through_into_the_live_store(tmp_path, plate):
    """Spec §5: the part's copies are hard links; the new bytes must be new files."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old-bytes", "sym"))
    held = os.open(store / "figures/sym/default.plotly.json", os.O_RDONLY)
    try:
        replace_image_tables(
            store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
            figures=_figures(b"new-bytes", "sym"),
        )
        os.lseek(held, 0, os.SEEK_SET)
        assert os.read(held, 32) == b"old-bytes"
    finally:
        os.close(held)
    assert (store / "figures/sym/default.plotly.json").read_bytes() == b"new-bytes"


def test_measure_rebuild_with_no_bindings_removes_key_and_group(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"), figures=None
    )
    assert read_image_figures_descriptor(store) is None
    assert not (store / "figures").exists()


def test_measure_rebuild_writes_figures_before_the_root_is_promoted(
    tmp_path, plate, monkeypatch
):
    """Spec §5: judged at the promote, not from the final tree."""
    import hashlib

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    real_promote = ngff_.promote_store
    seen = {}

    def _spy(part, final, **kwargs):
        root = json.loads((Path(part) / "zarr.json").read_text())
        entry = root["attributes"]["phenotypic"]["figures"]["bindings"]["sym"]["pages"][0]["files"][0]
        data = (Path(part) / entry["path"]).read_bytes()
        seen["match"] = hashlib.sha256(data).hexdigest() == entry["sha256"]
        return real_promote(part, final, **kwargs)

    monkeypatch.setattr(ngff_, "promote_store", _spy)
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"), figures=_figures()
    )
    assert seen == {"match": True}


def test_migrates_table_replace_leaves_figures_untouched(tmp_path, plate):
    from phenotypic.sdk_._measurement_tables import replace_embedded_measurement_table

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    before = read_image_figures_descriptor(store)
    replace_embedded_measurement_table(
        store, _tables().measurements_payload(), objmap_target=ngff_.objmap_path("rgb")
    )
    assert read_image_figures_descriptor(store) == before
    assert (store / "figures/sym/default.plotly.json").read_bytes() == b"one"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/sdk_/test_image_figures_store.py -p no:cacheprovider -q`
Expected: `TypeError: ... unexpected keyword argument 'figures'`.

- [ ] **Step 3: Thread `figures` through the image writer**

`save2zarr`, `_save_store` and `_write_store_part` each gain the parameter `figures: "StoredFigures | None" = None`, documented in `Args:` as *"Per-image figures to write inside this store's transaction (spec 2026-09-22 §3). `None` writes no `figures` key."* `save2zarr` and `_save_store` pass it down to the next function.

In `_write_store_part`, after the tables block and **before** `# 4. root zarr.json LAST`, insert:

```python
        # Figures land in THIS part too, before the root, for the same reason
        # as the tables above: the root certifies their sha256 (spec §1).
        figures_fragment = None
        if figures is not None:
            from phenotypic.sdk_._image_figures import write_image_figures

            figures_fragment = write_image_figures(part, figures)
```

After the `apply_image_tables_attributes(...)` block, insert:

```python
        if figures_fragment is not None:
            from phenotypic.sdk_._image_figures import apply_image_figures_attributes

            apply_image_figures_attributes(phenotypic_attributes, figures_fragment)
```

Put the `StoredFigures` import under the file's `TYPE_CHECKING` block (add the block if there is none).

- [ ] **Step 4: Thread it through the process writer and the output manager**

- **`write_process_only_layer(..., figures: "StoredFigures | None" = None)`:** pass `figures` to `image._save_store(...)` in the `zarr` branch. In the `tiff` branch, add one comment: figures are never built for flat exports (spec non-goal).
- **`OutputManager.save_image_store(..., figures=None)`:** `if figures is not None: save_kwargs["figures"] = figures`.
- **`OutputManager.replace_image_store_measurements(..., *, figures, commit_guard=None, durable=None)`:** `figures` is required and keyword-only. Forward it as `replace_image_tables(..., figures=figures)`. Document it as *"The current pipeline's per-image figures; `None` removes the store's figures (spec §3 measure mode)."*
- Update the two test callers to pass `figures=None`.

- [ ] **Step 5: Rebuild figures inside the table transaction**

Give `_rewrite_store_tables` a new parameter, `clear_figures: bool = False` (after `commit_guard`), and add this after the `tables/` rmtree:

```python
        if clear_figures:
            # The copied figure files are HARD LINKS into the live store, like
            # everything copytree cloned above. Removing them here means the
            # new generation is written as new files; writing through a link
            # would change the published store before its new root exists.
            shutil.rmtree(part / ngff_.FIGURES_GROUP, ignore_errors=True)
```

Change the signature to `replace_image_tables(store_path, tables, *, figures, objmap_target=None, durable=None, commit_guard=None)`, with `figures` required, and document it under `Args:`. In `_populate`, after the tables `apply_…` call:

```python
            from ._image_figures import apply_image_figures_attributes, write_image_figures

            apply_image_figures_attributes(
                phenotypic,
                write_image_figures(part, figures) if figures is not None else None,
            )
```

Pass `clear_figures=True` to `_rewrite_store_tables`. Leave `replace_embedded_measurement_table` unchanged: it never clears `figures/`, so migrate hard-links the group across (spec §3).

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_image_figures_store.py tests/unit/cli/test_process_only_zarr.py tests/unit/cli/test_embedded_measurement_replacement.py tests/unit/cli/test_embedded_table_inversion.py -p no:cacheprovider -q`
Expected: PASS.

- [ ] **Step 7: Prove the hard-link guard can fail**

Pass `clear_figures=False` from `replace_image_tables`. Expect `test_a_same_name_rebuild_never_writes_through_into_the_live_store` to FAIL (the held descriptor reads `new-bytes`). Restore it.

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_io_handler.py src/phenotypic/sdk_/_measurement_tables.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_process_only.py tests/unit/sdk_/test_image_figures_store.py tests/unit/cli/conftest.py tests/unit/cli/test_embedded_table_inversion.py
git add src/phenotypic/_core/_image_parts/_image_io_handler.py src/phenotypic/sdk_/_measurement_tables.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_process_only.py tests/unit/sdk_/test_image_figures_store.py tests/unit/cli/conftest.py tests/unit/cli/test_embedded_table_inversion.py
git commit -m "feat(store): write per-image figures inside the root-last transaction"
```

---

### Task 5: Copy-out from the promoted store to `deliverables/plots/`

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_failures.py` (`record_plot_failure`)
- Modify: `src/phenotypic/plotting/_pipeline/_writer.py`. Extract `_commit_manifest` from `_publish_plot_output_locked` (≈455–473).
- Create: `src/phenotypic/plotting/_pipeline/_store_copyout.py`
- Modify: `src/phenotypic/plotting/_pipeline/_coordinator.py`. Add `publish_store_figures`.
- Create: `tests/unit/plotting/_store_fixtures.py`
- Test: `tests/unit/plotting/test_store_copyout.py`

**Interfaces:**
- Consumes:
  - `read_image_figures_descriptor` (Task 2)
  - `unique_page_stems` (Task 3)
  - `STORE_FORMATS` (Task 1)
  - `_atomic_write`, `_guarded_commit`, `_require_plot_publication`, `safe_path_component` (writer)
  - `exclusive_path_lock`
  - `ensure_plotlyjs_bundle`, `plotlyjs_src_for`
  - `_image_output_stem` (coordinator)
- Produces:
  - `publish_store_figures(store_path, plots_base, *, dataset, image_stem, plot_classes=None, publication_guard=None, commit_guard=None) -> None`
  - `PlotCoordinator.publish_store_figures(store_path, *, dataset, image_stem) -> None`
  - `record_plot_failure(..., error: BaseException | str, ..., page: str | None = None, fmt: str | None = None)`
  - `_commit_manifest(directory, manifest, *, publication_guard, commit_guard) -> None`
  - test helpers `figure_store(root, stored) -> Path` and `emit_image_via_store(coordinator, image=None, *, dataset="ds", image_stem="plate-1") -> StoredFigures | None`

**The output reproduces today's layout.** Reviewers check it against spec §3:
- **One `default` page (flat):** the files go to `<plots_base>/<safe(binding)>/<safe(dataset)>/<stem>-<hash>.<ext>`, with no manifest.
- **Anything else (directory):** the files go to `<…>/<stem>-<hash>/<page-stem>.<ext>`, plus a schema-2 `manifest.json` committed by `_commit_manifest` while holding `.publication.lock`. `<page-stem>` = `unique_page_stems([(key, label or key) …])`, which is the manifest writer's rule.
- **HTML:** for each stored `plotly-json`, a `<page-stem>.html` is generated, referencing the hoisted `plotly.min.js`.
- **Manifest fields:**
  - `files` maps format → filename, and includes `"html"`.
  - `renderers` stays a *capability*: `html: available` when any page is Plotly, `png: available` when any page is matplotlib.
  - `partial` lists a published page's store failures.
  - `failed` lists pages that had no file copied.
- **Failure lines:** the stored `error` is written verbatim. `page` and `format` are added as fields when known. `plot_class` comes from `plot_classes`, or `"<unresolved>"` when the binding is not in it.

- [ ] **Step 1: Extend `record_plot_failure`**

In `_failures.py`:
- Change the annotation to `error: BaseException | str`, documented as *"A `str` is recorded verbatim; the store already spelled it with `normalize_figure_error`, and re-wrapping would prefix a class twice."*
- Add the keyword arguments `page: str | None = None, fmt: str | None = None`, documented as *"The page key and store format a per-image failure is about, recorded as `page`/`format` fields."*
- In the body, use `"error": error if isinstance(error, str) else _format_error(error)`. Then add `if page is not None: entry["page"] = page` and `if fmt is not None: entry["format"] = fmt`.

- [ ] **Step 2: Extract `_commit_manifest` in `_writer.py` (behaviour-preserving)**

```python
def _commit_manifest(
    directory: Path,
    manifest: dict[str, Any],
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    """Replace ``directory/manifest.json`` with *manifest*, guarded, last."""
    manifest_path = directory / "manifest.json"
    temporary_manifest = directory / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        with _guarded_commit(publication_guard, commit_guard):
            os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
```

In `_publish_plot_output_locked`, replace the inline block at the end with a call to it. Run: `uv run pytest tests/unit/plotting/test_output_adapter.py -p no:cacheprovider -q` and expect PASS.

- [ ] **Step 3: Write the shared test fixtures**

`tests/unit/plotting/_store_fixtures.py`:

```python
"""Minimal stores for copy-out tests, and the build -> store -> copy-out path.

`figure_store` writes only what copy-out reads (figures/ + a root carrying the
descriptor), so tests need no pixels. `emit_image_via_store` puts its store
OUTSIDE the test's tmp_path, in a fresh directory per call, so assertions over
tmp_path see only deliverables and a second emit for the same stem never
collides (plan-review B1).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from phenotypic.sdk_._image_figures import (
    StoredFigures,
    apply_image_figures_attributes,
    write_image_figures,
)


def figure_store(root: Path, stored: StoredFigures) -> Path:
    """Write a promoted-looking store under *root* and return its path."""
    store = Path(root) / "p.ome.zarr"
    store.mkdir(parents=True)
    phenotypic: dict = {"store_schema_version": 3}
    apply_image_figures_attributes(phenotypic, write_image_figures(store, stored))
    (store / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"phenotypic": phenotypic}}
    ))
    return store


def emit_image_via_store(coordinator, image=None, *, dataset="ds", image_stem="plate-1"):
    """build -> minimal store -> copy-out: the path every CLI mode now takes."""
    from phenotypic.plotting._pipeline._store_figures import build_image_figures

    stored = build_image_figures(coordinator._pipeline, object() if image is None else image)
    if stored is None:
        return None
    output_root = coordinator._plots_base.parent.parent
    scratch = Path(tempfile.mkdtemp(prefix=f"{output_root.name}-store-", dir=output_root.parent))
    store = figure_store(scratch, stored)
    coordinator.publish_store_figures(store, dataset=dataset, image_stem=image_stem)
    return stored
```

- [ ] **Step 4: Write the failing copy-out tests**

```python
"""Copy-out: promoted store -> today's deliverables layout (spec §3 step 3)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.plotting._pipeline._coordinator import _image_output_stem
from phenotypic.plotting._pipeline._store_copyout import publish_store_figures
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
)
from tests.unit.plotting._store_fixtures import figure_store

_STEM = _image_output_stem("ds 1", "plate_01")


def _plotly_json() -> bytes:
    import plotly.graph_objects as go

    return go.Figure(go.Bar(x=["a"], y=[1])).to_json().encode()


def _page(key="default", label=None, formats=("plotly-json",), backend="plotly"):
    table = {
        "plotly-json": ("application/vnd.plotly.v1+json", ".plotly.json", _plotly_json()),
        "png": ("image/png", ".png", b"\x89PNG"),
    }
    stem = key.replace(" ", "-")
    return StoredFigurePage(key, label, backend, {"k": 1}, tuple(
        StoredFigureFile(fmt, table[fmt][0], f"{stem}{table[fmt][1]}", table[fmt][2])
        for fmt in formats
    ))


def _one(*pages, binding="sym", failed=()):
    return StoredFigures((StoredFigureBinding(binding, "MeasureSymZones", binding, pages),), failed)


def _publish(tmp_path, store, **kw):
    plots = tmp_path / "deliverables" / "plots"
    publish_store_figures(store, plots, dataset="ds 1", image_stem="plate_01", **kw)
    return plots


def _lines(plots):
    return [json.loads(line) for line in (plots / ".failures.jsonl").read_text().splitlines()]


def test_a_single_default_page_lands_flat_with_generated_html(tmp_path):
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    base = plots / "sym" / "ds-1"
    assert sorted(p.name for p in base.iterdir()) == [f"{_STEM}.html", f"{_STEM}.plotly.json"]
    assert 'src="../../plotly.min.js"' in (base / f"{_STEM}.html").read_text()
    assert (plots / "plotly.min.js").is_file()
    assert not (base / "manifest.json").exists()


def test_multi_page_writes_a_directory_and_manifest_v2(tmp_path):
    pages = (_page("first", "First"), _page("second", formats=("plotly-json", "png")))
    failed = (StoredFigureFailure("sym", "second", "png", "OSError: partial"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages, failed=failed)))
    directory = plots / "sym" / "ds-1" / _STEM
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert [p["key"] for p in manifest["pages"]] == ["first", "second"]
    assert manifest["pages"][0]["files"] == {"plotly-json": "First.plotly.json", "html": "First.html"}
    assert manifest["pages"][1]["files"]["png"] == "second.png"
    assert manifest["pages"][1]["partial"] == ["OSError: partial"]
    assert manifest["pages"][0]["metadata"] == {"k": 1}
    assert manifest["renderers"] == {"html": "available"}
    assert 'src="../../../plotly.min.js"' in (directory / "First.html").read_text()


def test_a_tampered_file_is_recorded_and_not_copied(tmp_path):
    store = figure_store(tmp_path / "s", _one(_page()))
    (store / "figures/sym/default.plotly.json").write_bytes(b"tampered")
    plots = _publish(tmp_path, store, plot_classes={"sym": "MeasureSymZones"})
    assert not list((plots / "sym").rglob("*.plotly.json"))
    [record] = _lines(plots)
    assert record["error"].startswith("ValueError: ") and "sha256" in record["error"]
    assert (record["page"], record["format"], record["plot_class"]) == (
        "default", "plotly-json", "MeasureSymZones"
    )


def test_descriptor_failures_are_recorded_verbatim_with_their_class(tmp_path):
    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom at 0x…"),)
    plots = _publish(
        tmp_path, figure_store(tmp_path / "s", _one(_page(), failed=failed)),
        plot_classes={"orient": "MeasureOrientationZones"},
    )
    [record] = _lines(plots)
    assert record["error"] == "RuntimeError: boom at 0x…"
    assert record["plot_class"] == "MeasureOrientationZones"
    assert "page" not in record and "format" not in record
    assert (record["binding_id"], record["dataset"], record["image_stem"]) == (
        "orient", "ds 1", "plate_01"
    )


def test_a_republished_page_loses_its_leftover_renderings(tmp_path):
    base = tmp_path / "deliverables" / "plots" / "sym" / "ds-1"
    base.mkdir(parents=True)
    (base / f"{_STEM}.png").write_bytes(b"a png from a run that stored png")
    _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    assert not (base / f"{_STEM}.png").exists()


def test_a_page_that_publishes_nothing_keeps_its_previous_files(tmp_path):
    base = tmp_path / "deliverables" / "plots" / "sym" / "ds-1"
    base.mkdir(parents=True)
    for suffix in (".html", ".png"):
        (base / f"{_STEM}{suffix}").write_bytes(b"previous")
    store = figure_store(tmp_path / "s", _one(_page()))
    (store / "figures/sym/default.plotly.json").write_bytes(b"tampered")
    _publish(tmp_path, store)
    assert sorted(p.name for p in base.iterdir()) == [f"{_STEM}.html", f"{_STEM}.png"]


def test_a_store_without_figures_publishes_nothing(tmp_path):
    store = tmp_path / "s" / "p.ome.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"phenotypic": {}}}
    ))
    plots = _publish(tmp_path, store)
    assert not plots.exists()


def test_a_refused_guard_propagates_before_anything_is_written(tmp_path):
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom"),)
    store = figure_store(tmp_path / "s", _one(_page(), failed=failed))
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, store, publication_guard=lambda: False)
    assert not (tmp_path / "deliverables").exists()
```

- [ ] **Step 5: Run to verify they fail**

Run: `uv run pytest tests/unit/plotting/test_store_copyout.py -p no:cacheprovider -q`
Expected: `ModuleNotFoundError: phenotypic.plotting._pipeline._store_copyout`.

- [ ] **Step 6: Write `_store_copyout.py`**

```python
"""Copy a promoted store's figures out to deliverables/plots (spec §3 step 3).

The store is the single source: this never renders a PNG. The one thing it
produces rather than copies is an HTML page for each stored ``plotly-json``,
so a Plotly figure stays browsable in deliverables. Best-effort: every failure
is recorded to ``.failures.jsonl``; only a refused guard propagates.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from phenotypic.abc_.plotting._store_formats import STORE_FORMATS
from phenotypic.sdk_ import CommitGuard
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_._image_figures import read_image_figures_descriptor

from ._failures import record_plot_failure
from ._writer import (
    PlotPublicationBlocked,
    _atomic_write,
    _commit_manifest,
    _guarded_commit,
    _require_plot_publication,
    safe_path_component,
    unique_page_stems,
)

logger = logging.getLogger(__name__)

#: `<unresolved>` is the coordinator's spelling for "class not knowable here".
_UNRESOLVED = "<unresolved>"

#: Every suffix a page can have in deliverables, for leftover removal only.
_DELIVERABLE_SUFFIXES = (".plotly.json", ".html", ".png")


def publish_store_figures(
    store_path: Path,
    plots_base: Path,
    *,
    dataset: str,
    image_stem: str,
    plot_classes: Mapping[str, str] | None = None,
    publication_guard: Callable[[], bool] | None = None,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Republish one promoted store's figures at today's deliverables paths.

    Args:
        store_path: A promoted ``*.ome.zarr`` store.
        plots_base: Resolved ``deliverables/plots`` directory.
        dataset: Dataset name (unsanitized; hashed into the output stem).
        image_stem: Image stem (unsanitized).
        plot_classes: ``binding_id -> class name`` from the pipeline. The
            descriptor records no class for a binding that failed outright.
        publication_guard: Optional GUI compare-and-set predicate.
        commit_guard: Optional commit guard.

    Raises:
        PlotPublicationBlocked: If a guard refuses. Never swallowed.
    """
    classes = dict(plot_classes or {})

    def _record(binding_id: str, error: BaseException | str, *, page=None, fmt=None) -> None:
        record_plot_failure(
            plots_base, binding_id=binding_id,
            plot_class=classes.get(binding_id, _UNRESOLVED), lifecycle="image",
            error=error, dataset=dataset, image_stem=image_stem, page=page, fmt=fmt,
        )

    try:
        descriptor = read_image_figures_descriptor(store_path)
        if descriptor is None:
            return
        # Before the first record, too: a refused guard means "do not touch
        # this tree", and the failure log lives in it (_coordinator F1).
        _require_plot_publication(publication_guard)
        from ._coordinator import _image_output_stem

        output_stem = _image_output_stem(dataset, image_stem)
        failures = list(descriptor.get("failed", []))
        bindings = dict(descriptor.get("bindings", {}))
    except PlotPublicationBlocked:
        raise
    except Exception as exc:  # noqa: BLE001 - copy-out is best-effort
        logger.warning("Copy-out could not read %s", store_path, exc_info=exc)
        _record("<store>", exc)
        return
    for failure in failures:
        try:
            _record(failure["binding"], failure["error"],
                    page=failure.get("page"), fmt=failure.get("format"))
        except Exception as exc:  # noqa: BLE001 - a malformed entry is one record
            _record("<store>", exc)
    for binding_id, binding in bindings.items():
        classes.setdefault(binding_id, binding.get("class", _UNRESOLVED))
        page_failures = [f for f in failures if f.get("binding") == binding_id]
        try:
            _publish_binding(
                Path(store_path), plots_base, binding_id, binding, page_failures,
                dataset=dataset, output_stem=output_stem, record=_record,
                publication_guard=publication_guard, commit_guard=commit_guard,
            )
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - copy-out is best-effort
            logger.warning("Copy-out of plot %s failed", binding_id, exc_info=exc)
            _record(binding_id, exc)


def _publish_binding(
    store: Path,
    plots_base: Path,
    binding_id: str,
    binding: dict[str, Any],
    page_failures: list[dict[str, Any]],
    *,
    dataset: str,
    output_stem: str,
    record: Callable[..., None],
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    pages = binding.get("pages", [])
    base = plots_base / safe_path_component(binding_id) / safe_path_component(dataset)
    flat = len(pages) == 1 and pages[0]["key"] == "default"
    directory = base if flat else base / output_stem
    stems = (
        [output_stem]
        if flat
        else unique_page_stems([(p["key"], p["label"] or p["key"]) for p in pages])
    )
    _require_plot_publication(publication_guard)
    directory.mkdir(parents=True, exist_ok=True)
    if flat:
        _publish_pages(store, plots_base, directory, pages, stems, page_failures,
                       record=record, binding_id=binding_id,
                       publication_guard=publication_guard, commit_guard=commit_guard)
        return
    # Same lock `publish_plot_output` takes for a manifest directory, so two
    # writers of one image's directory cannot interleave pages and manifest.
    with exclusive_path_lock(directory / ".publication.lock"):
        _require_plot_publication(publication_guard)
        published, failed = _publish_pages(
            store, plots_base, directory, pages, stems, page_failures,
            record=record, binding_id=binding_id,
            publication_guard=publication_guard, commit_guard=commit_guard,
        )
        renderers: dict[str, str] = {}
        if any(p["backend"] == "plotly" for p in published):
            renderers["html"] = "available"
        if any(p["backend"] == "matplotlib" for p in published):
            renderers["png"] = "available"
        _commit_manifest(
            directory,
            {
                "schema_version": 2, "plot_id": binding_id,
                "class": binding.get("class", binding_id),
                "renderers": renderers, "pages": published, "failed": failed,
            },
            publication_guard=publication_guard, commit_guard=commit_guard,
        )


def _publish_pages(
    store: Path,
    plots_base: Path,
    directory: Path,
    pages: list[dict[str, Any]],
    stems: list[str],
    page_failures: list[dict[str, Any]],
    *,
    record: Callable[..., None],
    binding_id: str,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Copy every page's stored files; return manifest pages and failures."""
    published: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for page, stem in zip(pages, stems):
        files: dict[str, str] = {}
        for entry in page["files"]:
            try:
                data = (store / entry["path"]).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(
                        f"stored {entry['path']} does not match its sha256 "
                        f"(descriptor {entry['sha256'][:12]}…, file {digest[:12]}…)"
                    )
                name = f"{stem}{STORE_FORMATS[entry['format']].extension}"
                _atomic_write(
                    directory / name, lambda dest, data=data: dest.write_bytes(data),
                    publication_guard=publication_guard, commit_guard=commit_guard,
                )
                files[entry["format"]] = name
                if entry["format"] == "plotly-json":
                    files["html"] = _write_html_from_json(
                        data, directory, stem, plots_base,
                        publication_guard=publication_guard, commit_guard=commit_guard,
                    )
            except PlotPublicationBlocked:
                raise
            except Exception as exc:  # noqa: BLE001 - per-file best effort
                record(binding_id, exc, page=page["key"], fmt=entry.get("format"))
        if not files:
            failed.append({"key": page["key"], "label": page["label"],
                           "error": "no stored file could be copied out"})
            continue
        _remove_leftovers(directory, stem, set(files.values()),
                          publication_guard=publication_guard, commit_guard=commit_guard)
        entry_out: dict[str, Any] = {
            "key": page["key"], "label": page["label"], "files": files,
            "backend": "matplotlib" if page["backend"] == "mpl" else "plotly",
            "metadata": page.get("metadata", {}),
        }
        partial = [f["error"] for f in page_failures if f.get("page") == page["key"]]
        if partial:
            entry_out["partial"] = partial
        published.append(entry_out)
    return published, failed


def _write_html_from_json(
    data: bytes,
    directory: Path,
    stem: str,
    plots_base: Path,
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> str:
    """Render the deliverables HTML for one stored Plotly JSON; return its name."""
    import plotly.io as pio

    from ._backends import ensure_plotlyjs_bundle, plotlyjs_src_for

    figure = pio.from_json(data.decode("utf-8"))
    src = plotlyjs_src_for(directory, ensure_plotlyjs_bundle(plots_base))
    name = f"{stem}.html"
    _atomic_write(
        directory / name,
        lambda dest: figure.write_html(dest, include_plotlyjs=src),
        publication_guard=publication_guard, commit_guard=commit_guard,
    )
    return name


def _remove_leftovers(
    directory: Path,
    stem: str,
    written: set[str],
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    """Remove *stem*'s renderings this pass did not write (today's stale rule).

    Only for a page this pass published: a page that published nothing keeps
    its previous files whole rather than half of them.
    """
    for suffix in _DELIVERABLE_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if path.name in written or not path.exists():
            continue
        with _guarded_commit(publication_guard, commit_guard):
            path.unlink(missing_ok=True)


__all__ = ["publish_store_figures"]
```

- [ ] **Step 7: Add the coordinator method**

```python
    def publish_store_figures(
        self, store_path: Path, *, dataset: str, image_stem: str
    ) -> None:
        """Copy one promoted store's figures to deliverables (spec §3 step 3)."""
        from ._store_copyout import publish_store_figures

        publish_store_figures(
            store_path, self._plots_base, dataset=dataset, image_stem=image_stem,
            plot_classes={
                binding.id: type(binding.plot).__name__
                for binding in self._pipeline.get_plots()
            },
            publication_guard=self._publication_guard, commit_guard=self._commit_guard,
        )
```

- [ ] **Step 8: Run to verify they pass**

Run: `uv run pytest tests/unit/plotting/test_store_copyout.py tests/unit/plotting/test_failure_record.py tests/unit/plotting/test_output_adapter.py -p no:cacheprovider -q`
Expected: PASS.

- [ ] **Step 9: Prove it can fail**

- **sha256 check:** replace `if digest != entry["sha256"]:` with `if False:`. Expect the tamper test to FAIL, then restore the line.
- **Guard ordering:** move the early `_require_plot_publication(publication_guard)` below the failure-record loop. Expect the refused-guard test to FAIL, because `deliverables/` now exists. Restore it.

- [ ] **Step 10: Commit**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_copyout.py tests/unit/plotting/_store_fixtures.py
git add src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_copyout.py tests/unit/plotting/_store_fixtures.py
git commit -m "feat(plotting): copy per-image figures out of the promoted store"
```

---

### Task 6: Wire every mode, retire `emit_image`, bump the process revision

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_single.py`: full mode (≈340–366) and measure mode (≈437–456)
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py`: Stage 3 (≈578–600)
- Modify: `src/phenotypic/_cli/_cli_process_only.py`: `process_single_apply_only_core` (≈320–352)
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:195-205`
- Modify: `src/phenotypic/plotting/_pipeline/_coordinator.py`: delete `emit_image` and `_publish_image_value`, then delete the imports ruff reports as unused
- Modify tests: `tests/unit/plotting/test_coordinator.py`, `tests/integration/plotting/test_publication_end_to_end.py`, `tests/unit/cli/test_embedded_measurement_replacement.py:155`, and `tests/unit/cli/test_work_id_semantics_revision.py` if it pins the value 2
- Test: `tests/integration/cli/test_figures_in_store.py`

**Interfaces:**
- Consumes:
  - `build_image_figures` (Task 3)
  - the `figures=` keywords (Task 4)
  - `PlotCoordinator.publish_store_figures` (Task 5)
  - `emit_image_via_store` (Task 5 fixtures)

- [ ] **Step 1: Write the failing CLI-level tests**

```python
"""Every mode writes figures into the store; deliverables are a copy (spec §3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_._image_figures import read_image_figures_descriptor


def _write_inputs(root: Path, *, with_plot: bool):
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize, MeasureSymZones

    root.mkdir(parents=True, exist_ok=True)
    image = root / "in" / "plate.tiff"
    image.parent.mkdir(exist_ok=True)
    imsave(str(image), load_synth_yeast_plate().rgb[:], check_contrast=False)
    sym = MeasureSymZones()
    pipeline = ImagePipeline(
        ops={"detect": OtsuDetector()},
        meas={"size": MeasureSize(), "sym": sym},
        plots=[sym] if with_plot else [],
    )
    path = root / "pipeline.json"
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return image, path


def _full(tmp_path, pipeline, image):
    from phenotypic._cli._cli_process_single import process_single_image_core

    out = tmp_path / "out"
    process_single_image_core(
        pipeline, image, out, "ds", "Image", {},
        OutputManager.from_config(out, ".tiff", save_overlays=False),
    )
    return out, zarr_store_path(out, "ds", "plate")


def test_full_mode_stores_figures_and_copies_them_out(tmp_path):
    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, pipeline, image)
    descriptor = read_image_figures_descriptor(store)
    assert descriptor["failed"] == []
    [page] = descriptor["bindings"]["sym"]["pages"]
    assert [f["format"] for f in page["files"]] == ["plotly-json"]
    deliverable = list((out / "deliverables/plots/sym/ds").glob("plate-*.plotly.json"))
    assert len(deliverable) == 1
    assert deliverable[0].read_bytes() == (store / page["files"][0]["path"]).read_bytes()
    assert len(list((out / "deliverables/plots/sym/ds").glob("plate-*.html"))) == 1


def test_measure_mode_rebuilds_figures_from_the_current_pipeline(tmp_path):
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    image, with_plot = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, with_plot, image)
    _, without_plot = _write_inputs(tmp_path / "second", with_plot=False)
    manager = OutputManager.from_config(out, ".tiff", save_overlays=False)
    process_single_store_measure_core(without_plot, store, out, "ds", "Image", manager)
    assert read_image_figures_descriptor(store) is None
    process_single_store_measure_core(with_plot, store, out, "ds", "Image", manager)
    assert list(read_image_figures_descriptor(store)["bindings"]) == ["sym"]


@pytest.mark.parametrize("fmt", ["zarr", "tiff"])
def test_process_mode_carries_figures_only_in_a_store(tmp_path, fmt):
    from phenotypic._cli._cli_process_only import process_single_apply_only_core

    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline, image_path=image, input_root=image.parent,
        output_dir=out, image_type="Image", layer="rgb", read_kwargs={},
        process_format=fmt,
    )
    if fmt == "zarr":
        store = out / "plate.ome.zarr"
        descriptor = read_image_figures_descriptor(store)
        assert list(descriptor["bindings"]) == ["sym"] and descriptor["failed"] == []
        assert not (store / "tables").exists(), "process mode writes no table"
    else:
        assert not list(out.rglob("*.plotly.json"))
    assert not (out / "deliverables").exists()


def test_process_revision_is_3():
    from phenotypic._cli import _cli_failure_tracker as tracker

    assert tracker.PROCESS_LAYER_SEMANTICS_REVISION == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/integration/cli/test_figures_in_store.py -p no:cacheprovider -q`
Expected: FAIL, because there is no descriptor and the revision is still 2.

- [ ] **Step 3: Wire full mode**

In `_cli_process_single.py`, replace the `PlotCoordinator(...).emit_image(...)` block (≈343–355) with:

```python
        from phenotypic.plotting._pipeline import PlotCoordinator
        from phenotypic.plotting._pipeline._store_figures import build_image_figures

        _check_active(active_check)
        figures = build_image_figures(pipeline, image)
```

Add `figures=figures` to the `save_image_store(...)` call that follows. Keep its `if saved_store is None: raise RuntimeError(...)` unchanged, and directly after it add:

```python
        # After promotion, before the completion record: a crash between the
        # two re-runs the image, so deliverables never lag a certified store.
        _check_active(active_check)
        PlotCoordinator(pipeline, output_dir, commit_guard=commit_guard).publish_store_figures(
            saved_store, dataset=dataset_name, image_stem=image_stem
        )
```

- [ ] **Step 4: Wire staged Stage 3**

Make the same change in `_cli_staged_workers.py` (≈578–600), using `plan.post_pipeline` in place of `pipeline`. The copy-out goes after the existing `if saved_store is None or not valid_staged_store(saved_store): raise ...`.

- [ ] **Step 5: Wire measure mode**

Replace the `output_manager.replace_image_store_measurements(...)` call and the `PlotCoordinator(...).emit_image(...)` block (≈437–456) with:

```python
    from phenotypic.plotting._pipeline import PlotCoordinator
    from phenotypic.plotting._pipeline._store_figures import build_image_figures

    # Built BEFORE the table replace so the figures ride the same root-last
    # transaction: a store's figures and its table always come from the same
    # pipeline (spec §3 "Measure mode semantics").
    figures = build_image_figures(pipeline, image)
    output_manager.replace_image_store_measurements(
        store_path,
        measurements,
        dataset_name,
        commit_guard=commit_guard,
        figures=figures,
    )
    PlotCoordinator(pipeline, output_dir, commit_guard=commit_guard).publish_store_figures(
        store_path, dataset=dataset_name, image_stem=stem
    )
```

Keep the CAN-3 comment above the replace call. Leave everything from "Marker refresh is the final successful per-image publication" onward unchanged.

- [ ] **Step 6: Wire process mode**

In `process_single_apply_only_core`:
- Before the `try`, add `figures = None`.
- Inside the `try`, after `pipeline.apply(...)` and **before** `set_provenance_status(image, "complete")`, add the block below.
- Pass `figures=figures` to `write_process_only_layer(...)`.

Process mode has no copy-out.

```python
        # Figures only when there is a store to hold them (spec §3 by mode),
        # and before the status is closed, as in full mode. Measurer-backed
        # bindings recompute inside inspect() here, because apply() never
        # filled their cache -- accepted (spec §3 process mode).
        if process_format == "zarr":
            from phenotypic.plotting._pipeline._store_figures import build_image_figures

            figures = build_image_figures(pipeline, image)
```

- [ ] **Step 7: Bump the revision**

In `_cli_failure_tracker.py`, append to the changelog comment:

```python
#: 2 -> 3: process-mode stores now carry the pipeline's per-image figures
#:         (spec 2026-09-22-figures-in-ome-zarr §3). Also invalidates in-flight
#:         ``tiff`` continuations -- deliberate; invalidating too much is safe.
PROCESS_LAYER_SEMANTICS_REVISION = 3
```

Run: `uv run pytest tests/unit/cli/test_work_id_semantics_revision.py tests/integration/cli/test_process_objmap_semantics.py -p no:cacheprovider -q`. If a test pins the literal `2`, change it to `3`.

- [ ] **Step 8: Retire `emit_image`**

- In `_coordinator.py`, delete `PlotCoordinator.emit_image` and `_publish_image_value`. Keep `_image_output_stem`: the copy-out uses it.
- In the class docstring, add one sentence: *image plots publish through `build_image_figures` → store → `publish_store_figures`*.
- Run `uv run ruff check src/phenotypic/plotting/_pipeline/_coordinator.py` and delete exactly the imports it reports as unused.
- In `tests/unit/cli/test_embedded_measurement_replacement.py:155`, patch `PlotCoordinator.publish_store_figures` instead of `emit_image`. The test's claim still holds: a failure after the table write leaves the old marker stale, because copy-out runs before the marker refresh.

- [ ] **Step 9: Port `tests/unit/plotting/test_coordinator.py`, test by test (plan-review B1)**

First, add `from tests.unit.plotting._store_fixtures import emit_image_via_store`.

"Swap" in the table below means: replace `coordinator.emit_image(x, dataset=D, image_stem=S)` with `emit_image_via_store(coordinator, x, dataset=D, image_stem=S)`. Every store lands outside `tmp_path`, in a fresh directory, so "tree is empty" assertions over `tmp_path` still mean "deliverables are empty".

`grep -n "emit_image" tests/unit/plotting/test_coordinator.py` lists every call site. Each one is covered below:

| Test (current line) | Action |
|---|---|
| `test_image_plot_uses_deliverables_plot_layout` (92) | Swap. An mpl `_ImagePlot` stores `png`, so the assertions hold. |
| `test_image_plot_strict_mode_propagates_publication_failure` (105) | **Delete**, because `strict` is gone. Its guarantee (the failure is visible) moves to Task 3's `test_inspect_raising_omits_the_binding…` and Task 5's `test_descriptor_failures_are_recorded_verbatim…`. |
| `test_image_plot_disambiguates_sanitized_and_casefold_collisions` (119) | Swap. |
| `test_image_plot_output_name_is_stable_for_reruns` (137) | Swap. |
| `test_multi_page_image_plot_disambiguates_invocation_directories` (150) | Swap. |
| `test_a_multi_page_plotly_image_plot_writes_exactly_one_bundle` (424) | Swap. The `broken` page is now a build failure recorded at copy-out, so change the final assertion to `[(e["lifecycle"], e["page"]) for e in entries] == [("image", "broken")] * 3`. The bundle, 6-HTML-page and hoisted-record assertions hold. |
| `_emit_image` helper (539) and the `emit_image` param of `test_every_emit_point_records_one_failure…` (560) | Make the helper call `emit_image_via_store(coordinator)`. `plot_class` is `_RaisingImagePlot` via `plot_classes`, and the error regex holds because the text is stored verbatim with no address in it. |
| `test_a_single_figure_plotly_image_plot_publishes_html` (707) | Swap. The default `plotly-json` also lands, so add `assert len(list(directory.glob("*.plotly.json"))) == 2`. The other assertions hold. |
| `test_a_flat_image_render_failure_records_the_real_exception_class` (754) | Rewrite the assertions. There is now **one** record, with no second "produced no file" record. It must have `error.startswith("TypeError: unsupported figure type builtins.object")`, `page == "default"`, no `format` key, `plot_class == "_UnsupportedFigureImagePlot"`, and the same lifecycle, dataset and stem. |
| `_emit_image_flat` (793), used by the `emit_image-flat` / `emit_image-multi-page` params of `test_a_refused_publication_guard_propagates_and_writes_nothing` | Make the helper call `emit_image_via_store(coordinator)`. The test holds: the guard refuses before anything under `tmp_path` is written. |
| `test_a_fenced_commit_propagates_with_its_cause_and_records_nothing` (877–880, three `emit_image` params) | Holds via `_emit_image_flat`. The flat mpl case makes 1 png commit. The multi-page mpl case makes 2 png commits plus 1 manifest commit, and `allow=2` fences the manifest. |
| `test_the_flat_path_commits_through_the_commit_guard` (909) | Swap. There is one png commit, so `entered == 1` holds. |
| `test_the_flat_path_rechecks_the_publication_guard_before_commit` (925) | Swap, and change `answers = iter([True])` to `iter([True, True])`. The copy-out checks the guard at entry and again before `mkdir`, so the third check, inside the commit, is the one that must refuse. |
| `test_a_strict_flat_failure_keeps_the_renderer_as_its_cause` (954) | **Delete** (`strict`). |
| `test_a_partial_flat_render_publishes_what_it_can_and_records_once` (966) | Rewrite. Use a local plot declaring `@figure(backend="plotly", primary=True, store=("plotly-json", "png"))`, patch `_backends.chrome_available` to `True` and `plotly.io.to_image` to raise `OSError("raster exploded")`. Assert one `.html` and one `.plotly.json`, no `.png`, and the record list `== ["OSError: raster exploded"]` with `format == "png"`. |
| `test_a_rerun_without_chrome_removes_the_previous_png` (1022) | **Delete.** Its guarantee (a republished page loses its leftover PNG) is Task 5's `test_a_republished_page_loses_its_leftover_renderings`, which seeds the leftover by hand, since no default Plotly run writes a PNG any more. |
| `test_a_failed_flat_rerun_keeps_both_previous_renderings` (1045) | **Delete.** Its guarantee is Task 5's `test_a_page_that_publishes_nothing_keeps_its_previous_files`. |
| `test_a_rerun_as_matplotlib_removes_the_previous_html` (1073) | Swap both calls, and add `assert list(directory.glob("*.plotly.json")) == []`. |
| `test_the_flat_path_closes_its_matplotlib_figure` (1096) | Swap. In the docstring, say the close now happens in build (the `finally` in `_build_pages`), before the guard is consulted. |
| `test_a_multi_page_image_rerun_without_chrome_removes_the_previous_pngs` (1220) | Rewrite. Emit once via `emit_image_via_store`, seed `First.png` and `Second.png` in the invocation directory, and emit again. Assert no `.png`, 2 `.html`, 2 `.plotly.json`, and `_assert_manifest_matches_disk(directory)`. |

Keep `_emit_twice_chrome_then_none`, because the aggregate test (≈1197) still uses it.

- [ ] **Step 10: Port the integration file**

In `tests/integration/plotting/test_publication_end_to_end.py`, replace each `PlotCoordinator(pipeline, tmp_path).emit_image(image, dataset="ds 1", image_stem="plate_01"[, strict=True])` with:

```python
    stored = build_image_figures(pipeline, image)
    assert stored.failed == ()  # replaces strict=True: a failed build must not pass quietly
    store = image.save2zarr(tmp_path.parent / f"{tmp_path.name}-store.ome.zarr", figures=stored)
    PlotCoordinator(pipeline, tmp_path).publish_store_figures(
        store, dataset="ds 1", image_stem="plate_01"
    )
```

Test-specific changes:
- **`test_a_failing_plot_is_recorded_once…`:** omit the `stored.failed == ()` line, since failure is the test's subject. Its record assertions hold; `plot_class` now comes from the pipeline map, which gives the same value.
- **The Plotly test:** replace the assertion that PNG presence depends on `chrome_available()` with: `.plotly.json` and `.html` are present, and there is no `.png`.
- **The mpl test:** its assertions hold.

- [ ] **Step 11: Run the touched surface**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/integration/cli/test_figures_in_store.py tests/unit/plotting/ tests/integration/plotting/ tests/unit/cli/test_embedded_measurement_replacement.py tests/unit/cli/test_embedded_measurement_publication.py tests/unit/cli/test_process_only_zarr.py tests/unit/cli/test_work_id_semantics_revision.py -p no:cacheprovider -q`
Expected: PASS. Run any failure in isolation before attributing it (project rule).

- [ ] **Step 12: Prove it can fail**

Remove `figures=figures` from the full-mode `save_image_store` call. Expect `test_full_mode_stores_figures_and_copies_them_out` to FAIL, then restore it.

- [ ] **Step 13: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/_cli/_cli_process_only.py src/phenotypic/_cli/_cli_failure_tracker.py src/phenotypic/plotting/_pipeline/_coordinator.py tests/integration/cli/test_figures_in_store.py tests/unit/plotting/test_coordinator.py tests/integration/plotting/test_publication_end_to_end.py tests/unit/cli/test_embedded_measurement_replacement.py tests/unit/cli/test_work_id_semantics_revision.py
git add src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/_cli/_cli_process_only.py src/phenotypic/_cli/_cli_failure_tracker.py src/phenotypic/plotting/_pipeline/_coordinator.py tests/integration/cli/test_figures_in_store.py tests/unit/plotting/test_coordinator.py tests/integration/plotting/test_publication_end_to_end.py tests/unit/cli/test_embedded_measurement_replacement.py tests/unit/cli/test_work_id_semantics_revision.py
git commit -m "feat(cli): per-image figures live in the store in every mode; emit_image retired"
```

---

### Task 7: Cross-mode properties: reproducibility, cache parity, migrate, lazy imports

**Files:**
- Modify: `tests/unit/cli/test_process_only_zarr.py`
- Create: `tests/unit/measure/test_zone_figure_cache_parity.py`
- Modify: `tests/unit/cli/test_cli_provenance_migration.py`

- [ ] **Step 1: Process stores with a figure binding are byte-identical across processes**

Append to `tests/unit/cli/test_process_only_zarr.py`:

```python
def test_two_processes_with_a_figure_binding_write_byte_identical_stores(
    tmp_path: Path, source_image: Path
) -> None:
    """Spec §4: byte identity now covers figures/, and holds across fresh
    interpreters (fresh hash seeds, fresh object addresses)."""
    import os
    import subprocess
    import sys
    import textwrap

    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSymZones

    sym = MeasureSymZones()
    pipeline = tmp_path / "plotted.json.pht-pipe"
    ImagePipeline(ops=[OtsuDetector()], meas={"sym": sym}, plots=[sym]).to_json(pipeline)

    def run(out: Path, seed: str) -> Path:
        code = textwrap.dedent(f"""
            from pathlib import Path
            from phenotypic._cli._cli_process_only import process_single_apply_only_core
            process_single_apply_only_core(
                pipeline_path=Path({str(pipeline)!r}), image_path=Path({str(source_image)!r}),
                input_root=Path({str(source_image.parent)!r}), output_dir=Path({str(out)!r}),
                image_type="Image", layer="rgb", read_kwargs={{}}, process_format="zarr",
            )
        """)
        subprocess.run([sys.executable, "-c", code], check=True,
                       env={**os.environ, "PYTHONHASHSEED": seed})
        return out / f"{source_image.stem}{ngff_.STORE_SUFFIX}"

    first, second = run(tmp_path / "a", "1"), run(tmp_path / "b", "2")
    left, right = _tree_bytes(first), _tree_bytes(second)
    assert any(name.startswith("figures/sym/") for name in left)
    assert sorted(left) == sorted(right)
    assert [name for name in left if left[name] != right[name]] == []
```

Prove it can fail: append `str(time.time()).encode()` to what `_serialize_plotly_json` returns, and expect the test to FAIL. Revert the change.

- [ ] **Step 2: Cache parity**

```python
"""A zone figure does not depend on whether measure() just ran (spec §4)."""
from __future__ import annotations

import json

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureOrientationZones, MeasureSymZones
from phenotypic.plotting._pipeline._store_formats import serialize_store_format


def _detected() -> Image:
    image = Image(load_synth_yeast_plate())
    OtsuDetector().apply(image, inplace=True)
    return image


def _first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            found = _first_difference(left.get(key), right.get(key), f"{path}.{key}")
            if found:
                return found
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}[len]"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
    elif left != right:
        return path
    return None


@pytest.mark.parametrize("measurer_cls", [MeasureSymZones, MeasureOrientationZones])
def test_cache_hit_and_recompute_render_identical_bytes(measurer_cls, tmp_path):
    image = _detected()
    measurer = measurer_cls()
    measurer.measure(image)
    hit = measurer.inspect(image, for_save=True)

    reloaded = Image.load_zarr(image.save2zarr(tmp_path / "p.ome.zarr"))
    recomputed = measurer_cls().inspect(reloaded, for_save=True)

    def encode(fig):
        return serialize_store_format("plotly-json", fig, binding_id="b", page_key="default")

    left, right = encode(hit), encode(recomputed)
    assert left == right, _first_difference(json.loads(left), json.loads(right))
```

Run: `uv run pytest tests/unit/measure/test_zone_figure_cache_parity.py -p no:cacheprovider -q`
**If it FAILS, that is a real provider bug.** Spec §4 treats this as "a bug in the provider, not a tolerance". Stop and report the path from the assertion message to the orchestrator. Do not loosen the assertion.

- [ ] **Step 3: A figure-carrying store survives migrate unchanged**

Append to `tests/unit/cli/test_cli_provenance_migration.py`:

```python
def test_migrate_leaves_a_stores_figures_and_their_descriptor_untouched(
    tmp_path: Path,
) -> None:
    """Spec §1 'optional', non-goal 'migrate never fabricates a figure'."""
    from phenotypic._cli._cli_migrate import run_migrate

    store = tmp_path / "direct.ome.zarr"
    root = _write_store_root(store, {
        "schema_version": 1, "status": "complete", "pipeline": None,
        "retry_base_length": 0, "operations": [],
    }, root_version="")
    figure = store / "figures" / "sym" / "default.plotly.json"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"{}")
    document = json.loads(root.read_text())
    descriptor = {"schema_version": 1, "bindings": {}, "failed": []}
    document["attributes"]["phenotypic"]["figures"] = descriptor
    root.write_text(json.dumps(document))

    run_migrate(store, njobs=1)

    after = json.loads(root.read_text())["attributes"]["phenotypic"]
    assert after["figures"] == descriptor
    assert figure.read_bytes() == b"{}"
```

- [ ] **Step 4: Run the lazy-import guards and this task's files**

Run: `uv run pytest tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py tests/unit/cli/test_process_only_zarr.py tests/unit/measure/test_zone_figure_cache_parity.py tests/unit/cli/test_cli_provenance_migration.py -p no:cacheprovider -q`
Expected: PASS. If a guard fails, a module-level plotly/matplotlib/zarr import slipped in; move it into the function that uses it.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix tests/unit/cli/test_process_only_zarr.py tests/unit/measure/test_zone_figure_cache_parity.py tests/unit/cli/test_cli_provenance_migration.py
git add tests/unit/cli/test_process_only_zarr.py tests/unit/measure/test_zone_figure_cache_parity.py tests/unit/cli/test_cli_provenance_migration.py
git commit -m "test: figure reproducibility, cache parity, and migrate neutrality"
```

---

### Task 7a: Apply-state figures (§3a) and the calibration overlay as a `PlotImage`

*Added 2026-09-22 after PR #238 merged. Spec: §3a. The user chose to wire it in for real and to keep the stored overlay whenever it cannot be redrawn.*

**Files:**
- Modify: `src/phenotypic/abc_/plotting/_output.py` (or a new `_errors.py`) + `abc_/plotting/__init__.py` — export `FigureInputUnavailable(RuntimeError)`.
- Modify: `src/phenotypic/plotting/_pipeline/_store_figures.py` — `build_image_figures(pipeline, image, *, carry_from: Path | None = None)`; a binding raising `FigureInputUnavailable` is carried from `carry_from`'s descriptor (sha-verified) or else recorded as a binding-level failure.
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` — `CalibrateColorRpcc(ImageCorrector, PlotImage)`; weakref to the image `_operate` saw; undecorated `inspect(self, subject=None, *, for_save=False, **overrides)` → `show_tiles()`, or `FigureInputUnavailable` when there is no record or the record is for another image.
- Modify: `src/phenotypic/_cli/_cli_process_single.py` (measure mode: `carry_from=store_path`).
- Modify: `src/phenotypic/_cli/_cli_pipeline_split.py` — stop refusing a plot bound to a **pre-GPU operation** (still refuse a plot bound to the GPU detector itself); give `pre_pipeline` the bindings whose `ref` is a pre-op key, and keep them out of `post_pipeline` (whose ops no longer contain them).
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py` — Stage 1: after `plan.pre_pipeline.apply`, `build_image_figures(plan.pre_pipeline, image)` and pass `figures=` to Stage 1's `save_image_store`. Stage 3: `build_image_figures(plan.post_pipeline, image, carry_from=<the Stage-1 store it loaded>)`, and **also** carry every Stage-1 binding (they are not in `post_pipeline`): simplest is a helper `carry_bindings(carry_from, binding_ids)` merged into the Stage-3 `StoredFigures`.
- Tests: `tests/unit/plotting/test_store_figures_build.py` (carry semantics), `tests/unit/correction/test_calibration_plot_image.py` (new: provider contract), `tests/integration/cli/test_calibration_figure_in_store.py` (new: end-to-end on the synthetic checker frame from `tests/unit/correction/_checker_frames.py`), the staged split tests (grep `staged plotting supports only post-GPU`), and the Stage 1/3 worker tests.

**Required behaviour (each needs a test that can fail + a mutation):**
1. `FigureInputUnavailable` is public (`phenotypic.abc_.plotting`) and lazy-import safe.
2. `CalibrateColorRpcc` after `apply(frame, inplace=True)`: `inspect(frame)` returns the same figure `show_tiles()` does (compare PNG bytes via the store serializer); `inspect(other_image)` and `inspect()` on a fresh instance raise `FigureInputUnavailable`; the record/weakref never pins the image (`weakref`-released image → unavailable). Pydantic serialization of the op is unchanged (`to_json`/`from_json` round-trip, no new fields in `model_json_schema()`).
3. Build: a binding raising `FigureInputUnavailable` with `carry_from=None` → one binding-level failure `FigureInputUnavailable: …`; with `carry_from` holding the binding → carried byte-identical (descriptor entry + files + its page-level failed entries); tampered carried file → binding-level failure, nothing carried.
4. Full mode (`process_single_image_core`), pipeline `ops={"cal": CalibrateColorRpcc(rois=band_rois(), ...)}` + a detector + a measurer, `plots=[cal]`, on the synthetic frame written to disk: store has `figures/cal/default.png` (mpl → png); deliverables copy exists and equals it.
5. Process mode (zarr): the store carries the overlay.
6. Measure mode on the full-mode store: the overlay is **carried** — same descriptor entry, same bytes (sha) — while the table is replaced; a measure run on a store that never had the overlay records `FigureInputUnavailable` and writes no overlay.
7. Staged split: a plot bound to a pre-GPU op is accepted; one bound to the GPU detector is still refused. Stage 1 writes the overlay into its store; Stage 3's final store carries it byte-identical. (Use the existing staged-worker test fixtures; grep `stage1_preprocess_core(` / `stage3_merge_measure_core(` in `tests/`.)
8. The calibration record's own tests (`tests/unit/correction/test_calibration_overlay.py`, `test_calibrate_color_rpcc_review.py`) still pass unchanged.

**Manual check (not a unit test):** one full-mode run on `load_yeast_plate_full()` (the bundled *Rhodotorula* full plate with its checker) as a Slurm job via the **`slurm-job`** skill, if valid checker ROIs for that plate can be found in the calibration specs/tests (`docs/superpowers/specs/2026-09-21-in-frame-checker-color-correction/`, `tests/unit/correction/`); open the stored `figures/cal/default.png` and the deliverables copy. If no ROIs are documented, report that instead of guessing coordinates.

- [ ] Commit in two parts: (a) `FigureInputUnavailable` + build carry + `CalibrateColorRpcc` provider + unit tests; (b) CLI wiring (measure carry, staged split + Stage 1 build + Stage 3 carry) + integration tests.

---

### Task 7b: Run folders (§1a) — and Task 7a revised to match

*Added 2026-09-22, user decision. Spec: §1a (run folders `{date}-{pipeline hash}`, never wiped, descriptor keyed by run, consumer picks the current run) and the revised §3a (no copy between run folders; keep only within the same run folder; else `unavailable`). Executed by the same agent as Task 7a, sequentially, because the two share files.*

**Order.**
1. Finish and commit Task 7a's provider part: `FigureInputUnavailable` (fix its lazy-import test), `CalibrateColorRpcc` as `PlotImage`, and its unit tests. The carry/CLI parts of 7a are superseded below.
2. Run folders, store side:
   - `sdk_/_image_figures.py`: `StoredFigures` gains `run_id`, `date` and `pipeline_sha256`, plus `unavailable: tuple[str, ...]`. `write_image_figures` writes `figures/<run_id>/<binding>/…` and returns a fragment for **one run**. Merging that fragment into the root **preserves every other run's entry**.
   - `replace_image_tables`: `figures=None` (the default) means no new run and nothing touched. `KEEP_FIGURES` is retired. A `StoredFigures` clears and rewrites **only its own run folder** in the part; the other runs stay as hard links.
   - `_write_store_part` / `save2zarr`: when the final path already holds a store, carry its other run folders (files and descriptor entries) into the new part, verified by sha256. This covers full `--overwrite`, a re-derived process store, and Stage 3 over Stage 1.
3. Run id at the CLI:
   - `{date}` is the UTC date the CLI invocation started, taken from existing run state if there is one (grep for the run start timestamp or manifest under `.phenotypic/`). Otherwise, capture it once at invocation and pass it through the worker config to every image and stage. The same date must reach SLURM workers and staged Stage 1/3.
   - `{pipeline hash}` is the first 12 hex of the pipeline source sha256 recorded in provenance.
4. Build: `build_image_figures(pipeline, image, *, run_id, date, pipeline_sha256, keep_from: Path | None = None)`. A §3a binding that raises `FigureInputUnavailable` keeps its entry from `keep_from`'s **same `run_id`** folder, if one exists and its sha256 verifies. Otherwise it goes into `unavailable`.
   - Measure mode passes `keep_from=store_path`.
   - Stage 3 passes `keep_from=<the Stage-1 store>`.
   - Stage 1 builds only the bindings whose producer is in `pre_pipeline`. The staged split changes from 7a still apply.
5. Copy-out publishes **this run's** folder only (it takes `run_id`). A store with no folder for this run publishes nothing.

**Tests.** Each must be able to fail, and each gets a mutation.
- **Two runs, one store.** A full run, then a measure run with a different pipeline: both run folders are present, and the first is byte-identical.
- **Same run id.** A same-day, same-pipeline rerun replaces only that folder.
- **Full `--overwrite`** (a direct `save_image_store` over an existing store) keeps the other runs' folders.
- **Process-mode byte identity** is kept within one fixed date.
- **Run id** is `YYYY-MM-DD-<12 hex>` from a pinned date and the pipeline sha.
- **One run id for the whole run:** the staged Stage 1 → Stage 3 path uses a single run id across stages, even when the stages see different wall-clock days (monkeypatch the clock).
- **Copy-out** publishes only this run.
- **§3a:** measure mode on a store whose earlier run holds the calibration overlay puts `cal` in the new run's `unavailable`, and leaves the earlier folder untouched. Stage 3 keeps Stage 1's overlay in the same folder.
- **Existing tests:** update the Task 1–7 tests and fixtures that assert the flat `figures/<binding>/…` paths or the flat descriptor (`figure_store`, `emit_image_via_store`, `test_image_figures*.py`, `test_store_copyout.py`, `test_figures_in_store.py`, `test_process_only_zarr.py`, the parity and migrate tests). Keep what each one guarantees.

**Commits.** Separate commits, each green: (a) 7a provider; (b) store-side run folders plus updated tests; (c) CLI run id, build keep and copy-out; (d) calibration wiring and staged §3a, plus the integration tests.

---

### Task 8: Documentation

> **Executed against the revised spec (2026-09-23).** The steps below predate
> Revisions 12–14: §1a run folders (`figures/<run>/…`, `attributes.phenotypic.figures.runs`,
> never wiped except by `--overwrite`, the run's initial call recorded once in
> `state.config`/`job_metadata.json`) and §3a `FigureInputUnavailable` /
> `unavailable`. Where they disagree, the docs follow `design.md`, not these
> steps: the Step 4 snippet was adapted to the run-keyed descriptor. The edits
> also cover Phase B review MINOR-5, -8 and -9, `correction/CLAUDE.md`, and
> `docs/source/contrib_guide/tracked_state.md`.

**Files:**
- Modify: `.claude/skills/working-with-ome-zarr/SKILL.md`
- Modify: `src/phenotypic/_cli/CLAUDE.md`
- Modify: root `CLAUDE.md` (the `--mode process` bullet)
- Modify: `src/phenotypic/abc_/CLAUDE.md`
- Modify: `docs/source/extending/pages/custom_plotter.md`
- Modify: `docs/source/how_to/pages/zarr_storage.md`

- [ ] **Step 1: The store-contract skill**
  - Add a *Per-image figures* row to the store-contract table:
    - location: `figures/<binding>/…`
    - described by: `attributes.phenotypic.figures`
    - optional, and not listed in `ome.series`
  - Add one paragraph covering:
    - The media type is the contract.
    - The root binds each file's `sha256`.
    - The `bindings` key order is not part of the contract.
    - There is no `store_schema_version` bump.
    - Measure mode rebuilds the group inside the table transaction, and removes the part's hard-linked copies first.

- [ ] **Step 2: `_cli/CLAUDE.md` and the root `CLAUDE.md`**
  - In `_cli/CLAUDE.md`, add a short *"Per-image figures"* section covering:
    - the write path per mode, one sentence per row of the spec §3 table
    - copy-out runs after promotion and before the completion record
    - process revision 3; full mode is unchanged
    - `emit_image` is retired
  - In the root `CLAUDE.md`, add to the `--mode process` bullet: *"A store also carries the pipeline's per-image figures under `figures/` (revision 3); flat `tiff` exports carry none."*

- [ ] **Step 3: `abc_/CLAUDE.md` and the extending guide**
  - In `abc_/CLAUDE.md`, add the `@figure(store=...)` convention:
    - The closed set is `{plotly-json, png}`.
    - Each backend has its own default.
    - An invalid declaration raises `TypeError` when the class is defined.
    - An `inspect()` override falls back to the backend default, page by page.
  - In `custom_plotter.md`, add a section *"Storing figures with the image"* covering:
    - the `store=` parameter
    - the formats and their media types, as a table
    - the defaults
    - a default Plotly figure has no PNG in `deliverables/`, even with Chrome
    - an image-backed Plotly figure's JSON is MB-sized, because it embeds a PNG data URI (plan-review minor 13)
    - `deliverables/plots/` is a copy of the store
    - one runnable docstring-style example using `load_synth_yeast_plate()`

- [ ] **Step 4: The zarr storage how-to**

Add a section *"Reading an image's figures"* with this stdlib-only snippet:

```python
import hashlib, json
from pathlib import Path

store = Path("plate.ome.zarr")
figures = json.loads((store / "zarr.json").read_text())["attributes"]["phenotypic"].get("figures")
for binding_id, binding in (figures or {}).get("bindings", {}).items():
    for page in binding["pages"]:
        for entry in page["files"]:
            data = (store / entry["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            print(binding_id, page["key"], entry["media_type"], len(data))
```

- [ ] **Step 5: Verify the rendered pages**

Using the **`slurm-job`** skill, submit one job that runs `uv run sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never -b html docs/source <out>`. Never build locally, and never use `-j auto`. When it finishes, read the generated HTML for the two changed pages; an exit status of 0 is not the check.

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/working-with-ome-zarr/SKILL.md src/phenotypic/_cli/CLAUDE.md CLAUDE.md src/phenotypic/abc_/CLAUDE.md docs/source/extending/pages/custom_plotter.md docs/source/how_to/pages/zarr_storage.md
git commit -m "docs: per-image figures in the OME-Zarr store"
```

---

### Task 9: Final regression

- [ ] **Step 1: Types and lint on the changed surface.** Expect no new errors from either command:
  - `uv run mypy src/phenotypic/abc_/plotting src/phenotypic/plotting/_pipeline src/phenotypic/sdk_/_image_figures.py src/phenotypic/_cli/_cli_process_only.py`
  - `uv run ruff check $(git diff --name-only origin/main...HEAD -- '*.py')`
- [ ] **Step 2: The full sharded suite, once, as a Slurm job.**
  - Use the **`run-phenotypic-test`** skill with `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.
  - Run it in a worktree detached at this branch's HEAD SHA, with an `afterany` cleanup finalizer.
  - Compare the result against the recorded baseline: 11,106 tests, 81 known failures, all outside `sdk_`/`_cli`/`gui`.
  - Run each new failure in isolation before attributing it to this change.
- [ ] **Step 3: Report.** Take totals only from the job output you just read. List by name each failure attributed to this change, with its isolated-run result.

---

## Self-review

**Spec coverage.**

| Spec | Task |
|---|---|
| §1 layout, descriptor, media types, metadata | 1, 2, 3 |
| §1 presence, ordering, collision suffix | 2, 3, 4 |
| §1 failure granularity, `error` text | 3 |
| §1 namespace, no version bump | 4 (independent-reader test) |
| §1 hashes, copy-out verify | 5 |
| §2 signature, validation (including a bare `str`), defaults | 1 |
| §2 which declaration applies, `inspect()` override fallback | 3 |
| §2 serializers, lazy imports | 3, 7 |
| §3 build (whole binding inside the boundary, `None` counts as a failure) | 3 |
| §3 store write | 4 |
| §3 copy-out (lock, manifest commit, `renderers` capability, `partial`, failure fields and class) | 5 |
| §3 by-mode table, process builds before the status closes | 6 |
| §3 measure semantics, hard-link safety, migrate path untouched | 4, 6, 7 |
| §3 continuation (revision 3, full mode unchanged) | 6 |
| §4 byte-identical process store across processes | 7 |
| §4 cross-process serializers | 3 |
| §4 cache parity | 7 |
| §5 every row | 1–7 |
| §6 docs | 8 |
| Preflight follows the declared `png` (review) | 3 |

**Review findings disposition.**

| Finding | Resolution |
|---|---|
| B1 | Task 6 Step 9 (enumerated test by test) + `_store_fixtures` |
| M1 | `plot_classes` |
| M2 | Same-name hard-link test |
| M3 | `skipif win32` |
| M4 | Metadata check in `_build_page` |
| M5 | Removed: `html` is out of the store |
| M6 | `promote_store` spy |
| M7 | `stored.failed == ()` |
| M8 | `error` verbatim + `page`/`format` fields |
| Minors 1–8 | Spec revision items 6, 8, 10 and 9, plus the copy-out code |
| Minors 9, 11 | Moot: no Chrome lane |
| Minor 10 | Import path fixed |
| Minor 12 | Task 7 Step 1 |
| Minor 13 | Spec §2 + Task 8 |
| Minor 14 | Bare `str` rejected; collision suffix is now in spec §1. Windows `MAX_PATH` has the same gap as tables and is left as it is. |

**Simplicity cuts.**

| Cut | Resolution |
|---|---|
| 1–3 | Accepted by the user |
| 4 | `figures` is a required keyword; no `rebuild_figures` |
| 5 | Preflight folded into Task 3 |
| 6 | `_commit_manifest` reused |
| 7 | Delegate dropped; the four call sites use `build_image_figures(pipeline, image)` |
| 8 | No re-export |
| 9 | Duplicate tests dropped |

**Type consistency.** These names are spelled identically in Tasks 2–7:
- `StoredFigures`
- `StoredFigureBinding(binding_id, plot_class, directory, pages)`
- `StoredFigurePage(key, label, backend, metadata, files)`
- `StoredFigureFile(format, media_type, filename, data)`
- `StoredFigureFailure(binding, page, format, error)`
- the `figures=` keyword on every writer
- `record_plot_failure(..., page=, fmt=)`
