# Per-image figures in the OME-Zarr store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every per-image (`PlotImage`) figure a pipeline produces is written into the image's `.ome.zarr` store in `--mode full`, staged Stage 3, `--mode measure` and `--mode process --process-format zarr`. `deliverables/plots/` is then populated by copying out of the promoted store.

**Architecture:** Three separate units.
1. **Build** (`plotting/_pipeline/_store_figures.py`) renders each binding in memory into an immutable `StoredFigures` value. It writes nothing.
2. **Store write** (`sdk_/_image_figures.py`) writes that value into a `.part` directory inside the existing root-last transaction and returns the `attributes.phenotypic.figures` descriptor.
3. **Copy-out** (`plotting/_pipeline/_store_copyout.py`) runs after promotion. It reads the promoted store's descriptor, verifies each file's `sha256`, and republishes the files at today's `deliverables/plots/` paths.

`PlotCoordinator.emit_image` and its flat-path writer are removed. The four CLI call sites go through build → store → copy-out.

**Tech Stack:** Python 3.12, pydantic v2, Zarr v3 / OME-Zarr 0.5 (hand-written group documents, as `tables/` does it), Plotly 6 + Kaleido 1 (Chrome), matplotlib, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md`. Read it before any task; section numbers below (§1–§6) refer to it.

## Global Constraints

- `uv run` for every command; never bare `python`/`pip`. `uv run ruff check --fix <explicit paths>` only — never bare.
- **Lazy imports:** no module-level import of plotly, matplotlib, kaleido or zarr in any file this plan touches. Guards: `tests/unit/ci/test_startup_imports.py`, `tests/unit/ci/test_deferred_imports.py`.
- Everything new in the store lives under `attributes.phenotypic`. Do not change `attributes.ome`, `OME/zarr.json` or `METADATA.ome.xml`, and do not bump `store_schema_version`.
- Figures are **best-effort**. One figure failing never fails an image. The only exception to propagate is `PlotPublicationBlocked` (a refused guard), as in every handler today.
- **Determinism:** the same inputs, pipeline, version and environment give the same bytes. The only variation removed is stochastic (random ids, timestamps, `0x…` addresses).
- Store format names, closed set: `"plotly-json"`, `"html"`, `"png"`, `"svg"`. Extensions: `.plotly.json`, `.html`, `.png`, `.svg`. Media types: `application/vnd.plotly.v1+json`, `text/html`, `image/png`, `image/svg+xml`.
- Default `store`: `("plotly-json",)` for `backend="plotly"`, `("png",)` for `backend="mpl"`.
- `PROCESS_LAYER_SEMANTICS_REVISION` goes 2 → 3. Full mode gets no revision.
- Google-style docstrings; explicit names (no `run()`/`process()`); match the surrounding comment density.
- Tests: per step, run only the touched test file(s). Per task, run the task's files. The full suite runs once at the end, as a Slurm job, through the **`run-phenotypic-test`** skill (Task 12). Never `-n auto`, never `-x` on a baseline run.
- **Each new test must be shown to fail when the bug it guards is reintroduced.** A step marked "prove it can fail" does exactly that and then restores the code.

## Additions to the spec made while planning

Tasks 5 and 7 need three clarifications the spec does not state. They are written into the spec in Task 11:

1. **Descriptor pages carry `"metadata"`** (the `PlotPage.metadata` mapping, JSON-native). The copy-out rebuilds manifest v2, whose page entries have a `metadata` field. Without it in the descriptor, the copy-out cannot reproduce today's manifest.
2. **Build is a module-level function** `build_image_figures(pipeline, image)`. `PlotCoordinator.build_image_figures(image)` delegates to it. Process mode has no `deliverables/` and so no `plots_base` to construct a coordinator with.
3. **Chrome absence is spelled `PlotBackendUnavailable: …`**, using the existing `_backends.PlotBackendUnavailable`. The spec's `ChromeNotFoundError` was illustrative.

## File structure

| File | Status | Responsibility |
|---|---|---|
| `src/phenotypic/abc_/plotting/_store_formats.py` | create | Closed format table, defaults, `@figure(store=)` validation. Stdlib only. |
| `src/phenotypic/abc_/plotting/_pht_plot.py` | modify | `figure(store=)`, `FigureSpec.store` |
| `src/phenotypic/abc_/plotting/__init__.py` | modify | export `StoreFormat` |
| `src/phenotypic/plotting/_pipeline/_store_formats.py` | create | One deterministic serializer per format |
| `src/phenotypic/sdk_/_image_figures.py` | create | `StoredFigures` value types; write/apply/read the descriptor |
| `src/phenotypic/sdk_/ngff_.py` | modify | `FIGURES_GROUP`, `FIGURES_SCHEMA_VERSION`, `PhenotypicAttr.FIGURES` |
| `src/phenotypic/sdk_/__init__.py` | modify | re-export the new sdk_ names |
| `src/phenotypic/plotting/_pipeline/_store_figures.py` | create | `build_image_figures`, format resolution, error normalisation |
| `src/phenotypic/plotting/_pipeline/_backends.py` | modify | `declared_figure_spec` (split out of `_declared_backends`); preflight counts only image plots that store Chrome formats |
| `src/phenotypic/plotting/_pipeline/_writer.py` | modify | extract `unique_page_stems` from `_publish_plot_output_locked` |
| `src/phenotypic/plotting/_pipeline/_store_copyout.py` | create | `publish_store_figures` |
| `src/phenotypic/plotting/_pipeline/_failures.py` | modify | `record_plot_failure(error: BaseException \| str)` |
| `src/phenotypic/plotting/_pipeline/_coordinator.py` | modify | add `build_image_figures`, `publish_store_figures`; delete `emit_image`, `_publish_image_value` |
| `src/phenotypic/_core/_image_parts/_image_io_handler.py` | modify | `figures=` through `save2zarr` → `_save_store` → `_write_store_part` |
| `src/phenotypic/sdk_/_measurement_tables.py` | modify | `replace_image_tables(figures=, rebuild_figures=)` |
| `src/phenotypic/_cli/_cli_output_manager.py` | modify | `save_image_store(figures=)`, `replace_image_store_measurements(figures=, rebuild_figures=)` |
| `src/phenotypic/_cli/_cli_process_only.py` | modify | build + `write_process_only_layer(figures=)` |
| `src/phenotypic/_cli/_cli_process_single.py` | modify | full + measure wiring |
| `src/phenotypic/_cli/_cli_staged_workers.py` | modify | Stage 3 wiring |
| `src/phenotypic/_cli/_cli_failure_tracker.py` | modify | revision 3 |
| `tests/unit/cli/_kaleido_utils.py`, `.github/pytest-shards.json`, `.github/workflows/run-pytest.yml`, `tests/unit/ci/test_pytest_shard_manifest.py` | modify | the Chrome lane where Chrome-dependent tests must run |

---

### Task 1: Plotly-SVG determinism probe (decides `svg` for Plotly)

**Files:**
- Create: `docs/superpowers/plans/2026-09-22-figures-in-ome-zarr/probe_plotly_svg.py`
- Modify: this plan (record the outcome under "Probe outcome" below)

This task writes no product code. Its outcome selects one of two small code variants, which Task 2 and Task 3 both spell out. The probe drives Plotly directly, not `phenotypic`, so it is not a `logic_validation_scripts/` script. It lives beside this plan, as the project rule requires for executable artifacts of a change.

- [ ] **Step 1: Make Chrome available locally**

Run: `uv run plotly_get_chrome -y`
Then: `uv run python -c "from choreographer.browsers.chromium import Chromium; print(Chromium.find_browser(skip_local=False))"`
Expected: a path. If either command fails (no network, missing system libraries), skip to Step 4 and record outcome **B** with the reason. Being unable to run the probe is not evidence that the bytes are stable.

- [ ] **Step 2: Write the probe**

```python
"""Probe: is Plotly SVG byte-stable across processes once its ids are pinned?

Run twice in fresh interpreters; exits 0 and prints the digest. The caller
compares the two digests. Depends only on plotly/kaleido, never on phenotypic.
"""
from __future__ import annotations

import hashlib
import re
import sys

import plotly.graph_objects as go
import plotly.io as pio

_ID = re.compile(rb'\bid="([^"]+)"')


def pin_svg_ids(svg: bytes, salt: str) -> bytes:
    """Rewrite every id and every #reference to it to a stable sequence."""
    ids = list(dict.fromkeys(_ID.findall(svg)))
    for index, old in enumerate(ids):
        new = f"{salt}-{index}".encode()
        svg = re.sub(rb'id="' + re.escape(old) + rb'"', b'id="' + new + b'"', svg)
        svg = re.sub(rb"#" + re.escape(old) + rb"\b", b"#" + new, svg)
    return svg


def render_probe_figure() -> go.Figure:
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], mode="lines+markers"))
    fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]], xaxis="x2", yaxis="y2"))
    fig.update_layout(xaxis2={"domain": [0.6, 1]}, xaxis={"domain": [0, 0.4]})
    return fig


if __name__ == "__main__":
    raw = pio.to_image(render_probe_figure(), format="svg")
    pinned = pin_svg_ids(raw, "pht")
    residual = set(re.findall(rb"url\(#([^)]+)\)|href=\"#([^\"]+)\"", pinned))
    print(hashlib.sha256(raw).hexdigest(), hashlib.sha256(pinned).hexdigest())
    print("residual-refs:", sorted({a or b for a, b in residual}))
    sys.exit(0)
```

- [ ] **Step 3: Run it twice in fresh processes**

Run: `for i in 1 2; do uv run python docs/superpowers/plans/2026-09-22-figures-in-ome-zarr/probe_plotly_svg.py; done`
Expected: two lines, each `<raw digest> <pinned digest>`, plus the residual refs.
- **Outcome A** (Plotly SVG ships): the two *pinned* digests are equal, **and** every residual ref names an id of the form `pht-<n>` (the rewrite reached every reference).
- **Outcome B** (`svg` is mpl-only): anything else, including Step 1 failing.

- [ ] **Step 4: Record the outcome in this plan and commit**

Edit the line below to `A` or `B`, with the two digest lines or the Step 1 error pasted underneath.

**Probe outcome:** _(filled in by Task 1)_

```bash
git add docs/superpowers/plans/2026-09-22-figures-in-ome-zarr/
git commit -m "chore(plan): Plotly SVG determinism probe and its outcome"
```

---

### Task 2: The store-format contract and `@figure(store=)`

**Files:**
- Create: `src/phenotypic/abc_/plotting/_store_formats.py`
- Modify: `src/phenotypic/abc_/plotting/_pht_plot.py` (the `FigureSpec` dataclass and `figure()` near lines 104–268)
- Modify: `src/phenotypic/abc_/plotting/__init__.py`
- Test: `tests/unit/abc_/plotting/test_store_formats.py`

**Interfaces:**
- Produces: `StoreFormat` (the `Literal`), `StoreFormatInfo(extension: str, media_type: str, backends: frozenset[str])`, `STORE_FORMATS: Mapping[str, StoreFormatInfo]`, `default_store_formats(backend: str) -> tuple[str, ...]`, `resolve_store_formats(store, *, backend, owner) -> tuple[str, ...]` (raises `TypeError`), and `FigureSpec.store: tuple[str, ...]`.

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


def _spec(cls):
    return cls._class_primary_spec()


def test_the_format_table_is_the_closed_set_with_its_media_types():
    assert {name: (info.extension, info.media_type) for name, info in STORE_FORMATS.items()} == {
        "plotly-json": (".plotly.json", "application/vnd.plotly.v1+json"),
        "html": (".html", "text/html"),
        "png": (".png", "image/png"),
        "svg": (".svg", "image/svg+xml"),
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

    assert _spec(Plot).store == expected
    assert default_store_formats(backend) == expected


def test_a_declared_store_is_kept_in_declared_order():
    class Plot(PhtPlot):
        @figure(title="t", backend="plotly", primary=True, store=("png", "plotly-json"))
        def draw(self, image):
            raise AssertionError

    assert _spec(Plot).store == ("png", "plotly-json")


@pytest.mark.parametrize(
    ("backend", "store", "match"),
    [
        ("mpl", ("plotly-json",), "plotly-json"),
        ("mpl", ("html",), "html"),
        ("plotly", ("jpeg",), "unknown"),
        ("plotly", ("png", "png"), "duplicate"),
        ("plotly", (), "at least one"),
    ],
)
def test_an_invalid_store_is_refused_at_class_definition(backend, store, match):
    with pytest.raises(TypeError, match=match):

        class Plot(PhtPlot):  # noqa: F841 - the definition itself must raise
            @figure(title="t", backend=backend, primary=True, store=store)
            def draw(self, image):
                raise AssertionError
```

With **outcome B**, also add a case to the parametrized refusal: `("plotly", ("svg",), "svg")`. With **outcome A**, add a positive test instead: `store=("svg",)` on `backend="plotly"` is accepted.

- [ ] **Step 2: Run the tests to verify they fail**

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

StoreFormat = Literal["plotly-json", "html", "png", "svg"]


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


#: Keyed in the order the spec lists them. The folder layout is storage only:
#: a consumer reads ``media_type`` from the descriptor, never the extension.
STORE_FORMATS: Mapping[str, StoreFormatInfo] = MappingProxyType({
    "plotly-json": StoreFormatInfo(
        ".plotly.json", "application/vnd.plotly.v1+json", frozenset({"plotly"})
    ),
    "html": StoreFormatInfo(".html", "text/html", frozenset({"plotly"})),
    "png": StoreFormatInfo(".png", "image/png", frozenset({"plotly", "mpl"})),
    # Outcome B of the Task 1 probe: matplotlib only. Outcome A: {"plotly", "mpl"}.
    "svg": StoreFormatInfo(".svg", "image/svg+xml", frozenset({"mpl"})),
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
        TypeError: On an empty tuple, an unknown or duplicate name, or a
            format the backend cannot produce.
    """
    if store is None:
        return default_store_formats(backend)
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
    unsupported = [
        name for name in formats if backend not in STORE_FORMATS[name].backends
    ]
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
- Import: `from ._store_formats import StoreFormat, resolve_store_formats` beside the `._output` import.
- Add the attribute `store: tuple[str, ...]` to `FigureSpec`, after `backend`. Document it in the `Attributes:` block as *"Formats a `PlotImage` publication stores for this figure (spec §2)."*
- Add the parameter `store: tuple[StoreFormat, ...] | None = None,` to `figure()`, after `backend`. Document it in `Args:` as *"Formats to store in the image's OME-Zarr store. `None` stores the backend default (`("plotly-json",)` / `("png",)`). Validated when the class is defined."*
- Inside `decorator`, immediately after the controls check, add `resolved_store = resolve_store_formats(store, backend=backend, owner=fn.__name__)`, and pass `store=resolved_store` to the `FigureSpec(...)` constructor.
- Keep `store` out of `wrapper`. It is metadata only.

In `abc_/plotting/__init__.py`, add `from ._store_formats import StoreFormat` and put `"StoreFormat"` in `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/abc_/plotting/ -p no:cacheprovider -q`
Expected: all pass, including the existing `test_pht_plot.py`, `test_figure_backend.py` and `test_imports.py`. `FigureSpec` is built only through `figure()`, so no other constructor call needs the new field. Confirm with `grep -rn "FigureSpec(" src tests`: expect only `_pht_plot.py`, plus `replace(...)` in `_image_plots.py`, which copies the field.

- [ ] **Step 6: Prove the refusal test can fail**

Temporarily change `if not formats:` to `if False:`. Run the file. Expect the `at least one` case to FAIL. Restore the line.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix src/phenotypic/abc_/plotting/ tests/unit/abc_/plotting/test_store_formats.py
git add src/phenotypic/abc_/plotting/ tests/unit/abc_/plotting/test_store_formats.py
git commit -m "feat(plotting): @figure(store=) over a closed set of store formats"
```

---

### Task 3: Deterministic serializers, and the Chrome lane that must run their Chrome cases

**Files:**
- Create: `src/phenotypic/plotting/_pipeline/_store_formats.py`
- Modify: `tests/unit/cli/_kaleido_utils.py`
- Modify: `.github/pytest-shards.json` (the `plots-post-viz` entry), `.github/workflows/run-pytest.yml` (the Linux shard job, after "Install Playwright browser")
- Modify: `tests/unit/ci/test_pytest_shard_manifest.py`
- Test: `tests/unit/plotting/test_store_serializers.py`

**Interfaces:**
- Consumes: `STORE_FORMATS` (Task 2).
- Produces: `serialize_store_format(format: str, figure, *, binding_id: str, page_key: str) -> bytes`, plus a pinned-id helper `pin_svg_ids(svg: bytes, salt: str) -> bytes` (outcome A only).

- [ ] **Step 1: Make the Chrome marker strict on a lane that declares Chrome**

In `tests/unit/cli/_kaleido_utils.py`, replace the `requires_kaleido_chrome = pytest.mark.skipif(...)` definition with:

```python
import os

#: Set on a CI shard that installs Chrome. There a missing browser is a
#: FAILURE, not a skip: a Chrome-only check that skips on every lane is a
#: silent green (spec §5). Everywhere else the marker still skips.
_CHROME_REQUIRED = os.environ.get("PHENOTYPIC_REQUIRE_CHROME") == "1"

requires_kaleido_chrome = pytest.mark.skipif(
    not _CHROME_REQUIRED and not _kaleido_chrome_available(),
    reason=(
        "kaleido >= 1 requires Chrome for Plotly PNG export; "
        "install Chrome or run `plotly_get_chrome`"
    ),
)
```

(Put `import os` with the module's other imports.)

- [ ] **Step 2: Declare the lane**

In `.github/pytest-shards.json`, add `"chrome": true` to the `plots-post-viz` entry. Add `"chrome": false` to every other entry, so the key is total.

In `.github/workflows/run-pytest.yml`, in the Linux shard job, directly after the "Install Playwright browser" step, add:

```yaml
      - name: Install Chrome for Kaleido
        if: matrix.shard.chrome
        run: uv run plotly_get_chrome -y
```

In the shard job's "Run … tests" step, add `env:` with `PHENOTYPIC_REQUIRE_CHROME: ${{ matrix.shard.chrome && '1' || '' }}`.

In `tests/unit/ci/test_pytest_shard_manifest.py`, add:

```python
def test_the_chrome_lane_installs_chrome_and_makes_its_marker_strict() -> None:
    """Spec §5: the Chrome-dependent serializer tests must run somewhere."""
    shards = json.loads(SHARDS.read_text(encoding="utf-8"))
    assert all("chrome" in shard for shard in shards)
    chrome = [shard["name"] for shard in shards if shard["chrome"]]
    assert chrome == ["plots-post-viz"]
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(
        r"if: matrix\.shard\.chrome\s*\n\s*run: uv run plotly_get_chrome -y", workflow
    )
    assert "PHENOTYPIC_REQUIRE_CHROME: ${{ matrix.shard.chrome && '1' || '' }}" in workflow
```

(Use the file's existing `SHARDS`/`WORKFLOW` path constants. If the shards constant has a different name, `grep -n "pytest-shards.json" tests/unit/ci/test_pytest_shard_manifest.py` gives it.)

- [ ] **Step 3: Write the failing serializer tests**

```python
"""Store serializers are byte-deterministic across processes (spec §2, §4)."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.unit.cli._kaleido_utils import requires_kaleido_chrome

#: Built fresh in each subprocess, so any per-process randomness (hash seeds,
#: uuid4 ids, object addresses) would show up as differing bytes.
_FIGURES = {
    "plotly": textwrap.dedent("""
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2]))
    """),
    "mpl": textwrap.dedent("""
        from matplotlib.figure import Figure
        fig = Figure()
        fig.subplots().plot([1, 2, 3], [3, 1, 2])
    """),
}


def _digest_in_fresh_process(backend: str, fmt: str) -> str:
    code = _FIGURES[backend] + textwrap.dedent(f"""
        import hashlib
        from phenotypic.plotting._pipeline._store_formats import serialize_store_format
        data = serialize_store_format({fmt!r}, fig, binding_id="b", page_key="default")
        print(hashlib.sha256(data).hexdigest())
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("backend", "fmt"),
    [("plotly", "plotly-json"), ("plotly", "html"), ("mpl", "png"), ("mpl", "svg")],
)
def test_a_chrome_free_serializer_is_stable_across_processes(backend, fmt):
    assert _digest_in_fresh_process(backend, fmt) == _digest_in_fresh_process(backend, fmt)


@requires_kaleido_chrome
@pytest.mark.parametrize("fmt", ["png"])  # outcome A: ["png", "svg"]
def test_a_chrome_serializer_is_stable_across_processes(fmt):
    assert _digest_in_fresh_process("plotly", fmt) == _digest_in_fresh_process("plotly", fmt)


def test_html_names_its_div_after_the_binding_and_page_not_a_uuid():
    import plotly.graph_objects as go

    from phenotypic.plotting._pipeline._store_formats import serialize_store_format

    html = serialize_store_format(
        "html", go.Figure(), binding_id="sym", page_key="default"
    ).decode()
    other = serialize_store_format(
        "html", go.Figure(), binding_id="sym", page_key="second"
    ).decode()
    assert 'id="pht-' in html
    assert html != other
    assert "cdn.plot.ly" in html


def test_plotly_png_without_chrome_raises_backend_unavailable(monkeypatch):
    import plotly.graph_objects as go

    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._store_formats import serialize_store_format

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    with pytest.raises(_backends.PlotBackendUnavailable, match="Chrome"):
        serialize_store_format("png", go.Figure(), binding_id="b", page_key="default")
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/unit/plotting/test_store_serializers.py tests/unit/ci/test_pytest_shard_manifest.py -p no:cacheprovider -q`
Expected: the serializer tests fail with `ModuleNotFoundError`, and the shard test passes (Steps 1–2 are already in place). If the shard test fails, fix the manifest or workflow edit before moving on.

- [ ] **Step 5: Write the serializers**

```python
"""One deterministic serializer per store format (spec §2 "Serializers").

Each takes ``(figure, *, binding_id, page_key)`` and returns bytes. Anything a
format would otherwise randomise -- Plotly's HTML div id, matplotlib's SVG ids
and ``Date`` -- is derived from the binding id and page key instead, so the
same figure always yields the same bytes. What the environment legitimately
changes (a different Chrome's PNG) is left alone: that is signal, not noise.

Plotting libraries are imported inside each function (lazy-import contract).
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any, Callable

from phenotypic.abc_.plotting import figure_backend_of


def _stable_token(binding_id: str, page_key: str) -> str:
    """Return a short id that depends only on the page's identity."""
    identity = f"{binding_id}\0{page_key}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:16]


def _serialize_plotly_json(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    return figure.to_json().encode("utf-8")


def _serialize_html(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    import plotly.io as pio

    # CDN, not the run's hoisted bundle: a store cannot reference a file
    # outside itself, and embedding costs 4.8 MB per figure (spec §2).
    return pio.to_html(
        figure,
        include_plotlyjs="cdn",
        full_html=True,
        div_id=f"pht-{_stable_token(binding_id, page_key)}",
    ).encode("utf-8")


def _serialize_png(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    if figure_backend_of(figure) == "plotly":
        # Imported from the module, not bound by name, so a test that patches
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


def _serialize_svg(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    from matplotlib import rc_context

    buffer = BytesIO()
    with rc_context({"svg.hashsalt": _stable_token(binding_id, page_key)}):
        figure.savefig(buffer, format="svg", metadata={"Date": None})
    return buffer.getvalue()


_SERIALIZERS: dict[str, Callable[..., bytes]] = {
    "plotly-json": _serialize_plotly_json,
    "html": _serialize_html,
    "png": _serialize_png,
    "svg": _serialize_svg,
}


def serialize_store_format(
    fmt: str, figure: Any, *, binding_id: str, page_key: str
) -> bytes:
    """Serialize *figure* to one store format.

    Args:
        fmt: A name from ``STORE_FORMATS``.
        figure: A Plotly or matplotlib figure whose backend supports *fmt*
            (the caller has already checked; see ``_store_figures``).
        binding_id: The plot binding id; salts every generated id.
        page_key: The page key; salts every generated id.

    Returns:
        The encoded bytes.

    Raises:
        KeyError: If *fmt* is not a store format.
        PlotBackendUnavailable: A Plotly PNG without Chrome.
    """
    return _SERIALIZERS[fmt](figure, binding_id=binding_id, page_key=page_key)


__all__ = ["serialize_store_format"]
```

**Outcome A only:** replace `_serialize_svg` with a dispatcher. For Plotly: `pio.to_image(figure, format="svg")` is Chrome-gated exactly as in `_serialize_png`, then `pin_svg_ids(raw, f"pht-{_stable_token(...)}")`. Copy `pin_svg_ids` verbatim from the Task 1 probe and export it. For mpl, keep the body above. Also add `"svg"` to the Chrome test's parametrize list.

- [ ] **Step 6: Run them to verify they pass**

Run: `uv run pytest tests/unit/plotting/test_store_serializers.py -p no:cacheprovider -q`
Expected: PASS. The Chrome case SKIPs locally unless Task 1 installed Chrome, in which case it passes.

- [ ] **Step 7: Prove the determinism test can fail**

Temporarily change `div_id=...` to `div_id=None`. Run the `html` case and expect FAIL (Plotly then generates a uuid). Restore the line.

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/_store_formats.py tests/unit/plotting/test_store_serializers.py tests/unit/cli/_kaleido_utils.py tests/unit/ci/test_pytest_shard_manifest.py
git add src/phenotypic/plotting/_pipeline/_store_formats.py tests/unit/plotting/test_store_serializers.py tests/unit/cli/_kaleido_utils.py tests/unit/ci/test_pytest_shard_manifest.py .github/
git commit -m "feat(plotting): deterministic store serializers and a Chrome lane that must run"
```

---

### Task 4: The store-side value types and the `figures/` writer

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`: constants after `METADATA_TABLE_SCHEMA_VERSION` (≈line 93); `FIGURES` in `PhenotypicAttr` (≈line 484)
- Create: `src/phenotypic/sdk_/_image_figures.py`
- Modify: `src/phenotypic/sdk_/__init__.py` (re-export; follow how `write_image_tables` is exported there)
- Test: `tests/unit/sdk_/test_image_figures.py`

**Interfaces:**
- Produces:
  - `StoredFigureFile(format: str, media_type: str, filename: str, data: bytes)`
  - `StoredFigurePage(key: str, label: str | None, backend: str, metadata: Mapping[str, Any], files: tuple[StoredFigureFile, ...])`
  - `StoredFigureBinding(binding_id: str, plot_class: str, directory: str, pages: tuple[StoredFigurePage, ...])`
  - `StoredFigureFailure(binding: str, page: str | None, format: str | None, error: str)`
  - `StoredFigures(bindings: tuple[StoredFigureBinding, ...], failed: tuple[StoredFigureFailure, ...])`
  - `write_image_figures(store_part: Path, figures: StoredFigures) -> dict[str, object]`
  - `apply_image_figures_attributes(phenotypic: dict, fragment: dict | None) -> None`
  - `read_image_figures_descriptor(store_path: Path) -> dict | None`
  - `ngff_.FIGURES_GROUP = "figures"`, `ngff_.FIGURES_SCHEMA_VERSION = 1`, `PhenotypicAttr.FIGURES = "figures"`

The value types are storage-neutral. Filenames and directory names are decided upstream (Task 5), so this module knows nothing about formats or plotting.

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
    page = descriptor["bindings"]["sym"]["pages"][0]
    assert descriptor["bindings"]["sym"]["class"] == "MeasureSymZones"
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

- [ ] **Step 3: Add the constants to `ngff_.py`**

```python
#: Per-image figures (spec 2026-09-22 §1). A Zarr v3 group holding non-Zarr
#: files, exactly as `tables/` holds `table.parquet`; described by
#: `attributes.phenotypic.figures`, never by `ome.series`.
FIGURES_GROUP: Final[str] = "figures"
FIGURES_SCHEMA_VERSION: Final[int] = 1
```

Add `FIGURES: Final[str] = "figures"` to `PhenotypicAttr`, after `TABLES`.

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
    """One rendering of one page. ``filename`` is final; no path separators."""

    format: str
    media_type: str
    filename: str
    data: bytes


@dataclass(frozen=True)
class StoredFigurePage:
    """One page and the renderings that succeeded for it."""

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
    """A failure at the finest level available (spec §1 "Failure granularity")."""

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
        store_part: An unpromoted ``*.ome.zarr.part`` directory. Nothing is
            written into a promoted store: the root that certifies these files
            is written after them, in the same transaction.
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
                # A fresh file in a fresh part. Never write through an existing
                # path here: in a measure-mode rewrite the part's files are hard
                # links into the LIVE store (see `replace_image_tables`).
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
    figures, and a measure-mode rebuild must drop a stale descriptor, not
    keep it (the same total-function rule as ``apply_image_tables_attributes``).
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

Re-export these eight names plus `FIGURES_GROUP` from `phenotypic.sdk_`, the same way the tables names are exported. Check with `grep -n "write_image_tables" src/phenotypic/sdk_/__init__.py`, which shows whether they are eager imports or `__getattr__` lazy entries; follow that pattern. `tests/unit/ci/test_startup_imports.py` must stay green.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_image_figures.py tests/unit/ci/test_startup_imports.py -p no:cacheprovider -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/_image_figures.py src/phenotypic/sdk_/ngff_.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_image_figures.py
git add src/phenotypic/sdk_/ tests/unit/sdk_/test_image_figures.py
git commit -m "feat(sdk): figures/ group writer and attributes.phenotypic.figures descriptor"
```

---

### Task 5: Build per-image figures in memory

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_backends.py`: split `_declared_backends` (≈line 221) into `declared_figure_spec`
- Modify: `src/phenotypic/plotting/_pipeline/_writer.py`: extract the stem loop from `_publish_plot_output_locked` (≈lines 334–349)
- Create: `src/phenotypic/plotting/_pipeline/_store_figures.py`
- Modify: `src/phenotypic/plotting/_pipeline/_coordinator.py`: add `build_image_figures`
- Test: `tests/unit/plotting/test_store_figures_build.py`

**Interfaces:**
- Consumes: `STORE_FORMATS`, `default_store_formats` (Task 2); `serialize_store_format` (Task 3); the `StoredFigure*` types (Task 4).
- Produces:
  - `build_image_figures(pipeline, image) -> StoredFigures | None` (`None` ⇔ no `PlotImage` binding)
  - `normalize_figure_error(exc: BaseException) -> str`
  - `declared_figure_spec(plot) -> FigureSpec | None`
  - `unique_page_stems(names: Sequence[tuple[str, str]]) -> list[str]` (input `(page_key, preferred_name)`; the manifest writer passes `label or key`, the store passes `key`)
  - `PlotCoordinator.build_image_figures(image) -> StoredFigures | None`

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

In `_publish_plot_output_locked`, delete the `used` dict and the stem loop body. Compute `stems = unique_page_stems([(p.key, p.label or p.key) for p in output.pages])` before the loop, and iterate `for page, stem in zip(output.pages, stems):`. Add `Sequence` to the `collections.abc` import.

Run: `uv run pytest tests/unit/plotting/ -p no:cacheprovider -q -k "disambiguat or collision or manifest"`
Expected: PASS, with no behaviour change.

- [ ] **Step 2: Split `declared_figure_spec` out of `_declared_backends`**

In `_backends.py`, rename the body of `_declared_backends` into:

```python
def declared_figure_spec(plot: Any) -> Any:
    """Return the ``FigureSpec`` *plot*'s ``inspect()`` renders, if declared.

    The three-step rule documented on :func:`_declared_backends`, returning the
    spec rather than its backend so a caller can also read ``spec.store``.
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

and make `_declared_backends` a two-liner that keeps its docstring: `spec = declared_figure_spec(plot); return spec.backend if spec is not None else None`.

- [ ] **Step 3: Write the failing build tests**

```python
"""build_image_figures: in-memory, finest-grained failures (spec §1, §3 step 1)."""
from __future__ import annotations

import json

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
        ))


class Explodes(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise RuntimeError(f"bad object at {hex(id(self))}")


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


def test_a_declared_png_without_chrome_fails_that_format_only(monkeypatch):
    from phenotypic.plotting._pipeline import _backends

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    stored = _build(BarsWithPng())
    [page] = stored.bindings[0].pages
    assert [f.format for f in page.files] == ["plotly-json"]
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("BarsWithPng", "default", "png")
    assert failure.error.startswith("PlotBackendUnavailable: ")


def test_hand_built_pages_use_backend_defaults_and_collision_safe_names():
    stored = _build(HandBuiltPages())
    [binding] = stored.bindings
    names = {page.key: [f.filename for f in page.files] for page in binding.pages}
    assert names["A b"] == ["A-b.plotly.json"]
    [mpl_name] = names["a-b"]
    assert mpl_name.endswith(".png") and mpl_name != "A-b.png"
    assert "odd" not in names
    [failure] = stored.failed
    assert (failure.page, failure.format) == ("odd", None)
    assert failure.error.startswith("TypeError: unsupported figure type")


def test_inspect_raising_omits_the_binding_and_normalises_the_address():
    stored = _build(Explodes(), Bars())
    assert [b.binding_id for b in stored.bindings] == ["Bars"]
    [failure] = stored.failed
    assert (failure.binding, failure.page, failure.format) == ("Explodes", None, None)
    assert failure.error == "RuntimeError: bad object at 0x…"


def test_normalize_figure_error_replaces_every_address():
    assert normalize_figure_error(ValueError("0xdead and 0xBEEF1")) == "ValueError: 0x… and 0x…"


def test_figures_are_closed_after_serialization(monkeypatch):
    from phenotypic.plotting._pipeline import _store_figures

    closed = []
    monkeypatch.setattr(_store_figures.FigureAdapter, "close", staticmethod(closed.append))
    _build(MplLine())
    assert len(closed) == 1
```

- [ ] **Step 4: Run to verify they fail**

Run: `uv run pytest tests/unit/plotting/test_store_figures_build.py -p no:cacheprovider -q`
Expected: `ModuleNotFoundError: phenotypic.plotting._pipeline._store_figures`.

- [ ] **Step 5: Write `_store_figures.py`**

```python
"""Build one image's figures in memory for the store (spec §3 step 1).

Writes nothing, so it is safe to call anywhere before a store transaction.
Every failure is captured at the finest level available and returned inside
the value; only ``PlotPublicationBlocked`` propagates, as in every handler.
"""
from __future__ import annotations

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
        which may hold no bindings if every one failed.

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
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - one figure never kills an image
            logger.warning("Plot %s failed during image inspect", binding.id, exc_info=exc)
            failed.append(StoredFigureFailure(binding.id, None, None, normalize_figure_error(exc)))
            continue
        pages = _build_pages(binding, value, failed)
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
    """Serialize every page of one binding, recording per-page/format failures."""
    output = normalize_plot_output(value)
    spec = declared_figure_spec(binding.plot)
    stems = unique_page_stems([(page.key, page.key) for page in output.pages])
    pages: list[StoredFigurePage] = []
    for page, stem in zip(output.pages, stems):
        try:
            backend = figure_backend_of(page.figure)
            if backend is None:
                failed.append(StoredFigureFailure(
                    binding.id, page.key, None,
                    "TypeError: unsupported figure type "
                    f"{type(page.figure).__module__}.{type(page.figure).__qualname__}",
                ))
                continue
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
                files.append(StoredFigureFile(
                    fmt, info.media_type, f"{stem}{info.extension}", data
                ))
            if files:
                pages.append(StoredFigurePage(
                    key=page.key, label=page.label, backend=backend,
                    metadata=dict(page.metadata), files=tuple(files),
                ))
        finally:
            FigureAdapter.close(page.figure)
    return pages


__all__ = ["build_image_figures", "normalize_figure_error"]
```

Rules this implements, which the reviewer checks against spec §1/§2:
- The spec's `store` applies to every page the declared method produced.
- A page whose backend contradicts a declared format fails **that format**, not the page.
- A page whose backend is unknown fails with `format: null`.
- A page with zero successful files is omitted.
- A binding with zero pages is omitted.

- [ ] **Step 6: Add the coordinator delegate**

In `_coordinator.py`, add this method to `PlotCoordinator` (place it where `emit_image` is; `emit_image` itself is removed in Task 9):

```python
    def build_image_figures(self, image: Any) -> "StoredFigures | None":
        """Build this pipeline's per-image figures in memory (spec §3 step 1)."""
        from ._store_figures import build_image_figures

        return build_image_figures(self._pipeline, image)
```

Add `from phenotypic.sdk_._image_figures import StoredFigures` under `if TYPE_CHECKING:`, and add `TYPE_CHECKING` to the `typing` import.

- [ ] **Step 7: Run to verify they pass**

Run: `uv run pytest tests/unit/plotting/test_store_figures_build.py tests/unit/plotting/test_backends.py -p no:cacheprovider -q`
Expected: PASS.

- [ ] **Step 8: Prove two tests can fail**

(a) Delete the `_ADDRESS.sub` call; expect the address test to FAIL. Restore it.
(b) Change `if pages:` to `if True:` in `build_image_figures`. Nothing fails yet, because the all-failed binding case is not covered in this file. Add this test, confirm it fails under the mutation, then restore the line and confirm it passes:

```python
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
```

- [ ] **Step 9: Commit**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_figures_build.py
git add src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_figures_build.py
git commit -m "feat(plotting): build per-image figures in memory for the store"
```

---

### Task 6: Write figures inside the store transaction, in every mode

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`: `save2zarr` (≈1068), `_save_store` (≈1128), `_write_store_part` (≈1212; the tables block at 1377–1389, the root at 1391–1418)
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`: `_rewrite_store_tables` (≈632), `replace_image_tables` (≈698)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`: `save_image_store` (≈1842), `replace_image_store_measurements` (≈1941)
- Modify: `src/phenotypic/_cli/_cli_process_only.py`: `write_process_only_layer` (≈152)
- Test: `tests/unit/sdk_/test_image_figures_store.py`

**Interfaces:**
- Consumes: `StoredFigures`, `write_image_figures`, `apply_image_figures_attributes` (Task 4).
- Produces: a keyword `figures: StoredFigures | None = None` on `Image.save2zarr`, `Image._save_store`, `Image._write_store_part`, `OutputManager.save_image_store` and `write_process_only_layer`. `replace_image_tables` and `OutputManager.replace_image_store_measurements` gain `figures: StoredFigures | None = None, rebuild_figures: bool = False`.

- [ ] **Step 1: Write the failing tests**

```python
"""figures/ rides the root-last transaction (spec §1, §3 step 2, measure mode)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_image_figures_descriptor,
)


def _figures(tag: bytes = b"one", binding: str = "sym") -> StoredFigures:
    page = StoredFigurePage("default", None, "plotly", {}, (
        StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                         "default.plotly.json", tag),
    ))
    return StoredFigures((StoredFigureBinding(binding, "X", binding, (page,)),), ())


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


def test_measure_rebuild_replaces_figures_without_touching_live_bytes(tmp_path, plate):
    from phenotypic.sdk_ import replace_image_tables
    from phenotypic.sdk_._measurement_tables import prepare_image_tables
    import pandas as pd

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old", "gone"))
    old_file = store / "figures/gone/default.plotly.json"
    pixel = next(
        p for p in (store / "rgb" / "0").rglob("*") if p.is_file() and p.name != "zarr.json"
    )
    pixel_inode = pixel.stat().st_ino
    held = os.open(old_file, os.O_RDONLY)
    try:
        replace_image_tables(
            store,
            prepare_image_tables(pd.DataFrame({"Object_Label": [1]}), None),
            objmap_target=ngff_.objmap_path("rgb"),
            figures=_figures(b"new", "kept"),
            rebuild_figures=True,
        )
        # the replaced file's inode still holds the OLD bytes: nothing wrote through it
        assert os.pread(held, 16, 0) == b"old"
    finally:
        os.close(held)
    assert not (store / "figures/gone").exists()
    assert (store / "figures/kept/default.plotly.json").read_bytes() == b"new"
    assert list(read_image_figures_descriptor(store)["bindings"]) == ["kept"]
    assert (store / pixel.relative_to(store)).stat().st_ino == pixel_inode


def test_measure_rebuild_with_no_bindings_removes_key_and_group(tmp_path, plate):
    from phenotypic.sdk_ import replace_image_tables
    from phenotypic.sdk_._measurement_tables import prepare_image_tables
    import pandas as pd

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    replace_image_tables(
        store,
        prepare_image_tables(pd.DataFrame({"Object_Label": [1]}), None),
        objmap_target=ngff_.objmap_path("rgb"),
        figures=None,
        rebuild_figures=True,
    )
    assert read_image_figures_descriptor(store) is None
    assert not (store / "figures").exists()


def test_a_table_only_replace_leaves_figures_untouched(tmp_path, plate):
    from phenotypic.sdk_ import replace_image_tables
    from phenotypic.sdk_._measurement_tables import prepare_image_tables
    import pandas as pd

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    before = read_image_figures_descriptor(store)
    replace_image_tables(
        store,
        prepare_image_tables(pd.DataFrame({"Object_Label": [1]}), None),
        objmap_target=ngff_.objmap_path("rgb"),
    )
    assert read_image_figures_descriptor(store) == before
    assert (store / "figures/sym/default.plotly.json").read_bytes() == b"one"
```

Before running, check two names in this test:
- **`prepare_image_tables`**: `grep -rn "def prepare_image_tables" src/phenotypic` shows its module. Import it from there, and match the call to its signature (the output manager calls `prepare_image_tables(baseline, metadata_snapshot_or_None)`). If the synthetic frame is rejected for lacking a required column, use the frame the existing `tests/unit/sdk_` table tests build. `grep -rln "prepare_image_tables(" tests/unit/sdk_` finds them.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/sdk_/test_image_figures_store.py -p no:cacheprovider -q`
Expected: `TypeError: ... got an unexpected keyword argument 'figures'`.

- [ ] **Step 3: Thread `figures` through the image writer**

- `save2zarr`: add `figures: "StoredFigures | None" = None`. Document it in `Args:` as *"Per-image figures to write inside this store's transaction (spec 2026-09-22 §3). `None` writes no `figures` key."* Pass `figures=figures` to `_save_store`.
- `_save_store`: add the same parameter and pass it to `_write_store_part`.
- `_write_store_part`: add the same parameter. After the tables block and **before** `# 4. root zarr.json LAST`, insert:

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

Add `from phenotypic.sdk_._image_figures import StoredFigures` under the file's `TYPE_CHECKING` block. If there is none, use a string annotation and add one.

- [ ] **Step 4: Thread it through the process writer and the output manager**

- `write_process_only_layer(..., commit_guard=None, figures: "StoredFigures | None" = None)`: pass `figures=figures` to `image._save_store(...)` in the `zarr` branch. The `tiff` branch ignores it; add a one-line comment saying figures are never built for flat exports (spec non-goal).
- `OutputManager.save_image_store(..., measurements=None, figures=None)`: `if figures is not None: save_kwargs["figures"] = figures`.
- `OutputManager.replace_image_store_measurements(..., commit_guard=None, figures=None, rebuild_figures=False)`: forward both to `replace_image_tables`.

- [ ] **Step 5: Rebuild figures inside the table transaction**

In `_measurement_tables.py`:

`_rewrite_store_tables(..., commit_guard, clear_figures: bool = False)`: after `shutil.rmtree(part / ngff_.TABLES_GROUP, ignore_errors=True)`, add:

```python
        if clear_figures:
            # Same reasoning as `tables/` above, plus one more: the copied
            # figure files are HARD LINKS into the live store, so the new
            # generation must be written as new files, never through these.
            shutil.rmtree(part / ngff_.FIGURES_GROUP, ignore_errors=True)
```

`replace_image_tables(..., commit_guard=None, figures: "StoredFigures | None" = None, rebuild_figures: bool = False)`: add both parameters to `Args:`. *`rebuild_figures`* is documented as: *"Replace the whole `figures/` group with *figures* (`None` removes it). `False` leaves the store's figures exactly as they are."* Inside `_populate`, after the tables `apply_…` call:

```python
            if rebuild_figures:
                from ._image_figures import (
                    apply_image_figures_attributes,
                    write_image_figures,
                )

                apply_image_figures_attributes(
                    phenotypic,
                    write_image_figures(part, figures) if figures is not None else None,
                )
```

Pass `clear_figures=rebuild_figures` to `_rewrite_store_tables`. Leave `replace_embedded_measurement_table` as it is: it never clears `figures/`, so migrate hard-links it across (spec §3).

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_image_figures_store.py tests/unit/cli/test_process_only_zarr.py tests/unit/cli/test_embedded_measurement_replacement.py -p no:cacheprovider -q`
Expected: all PASS.

- [ ] **Step 7: Prove the hard-link guard can fail**

Temporarily pass `clear_figures=False` in `replace_image_tables`. Expect `test_measure_rebuild_replaces_figures_without_touching_live_bytes` to FAIL: the stale `gone/` survives, and on a same-name rebuild the write goes through the link. Restore it.

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_io_handler.py src/phenotypic/sdk_/_measurement_tables.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_process_only.py tests/unit/sdk_/test_image_figures_store.py
git add src/phenotypic/ tests/unit/sdk_/test_image_figures_store.py
git commit -m "feat(store): write per-image figures inside the root-last transaction"
```

---

### Task 7: Copy-out from the promoted store to `deliverables/plots/`

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_failures.py`: `record_plot_failure(error: BaseException | str)`
- Create: `src/phenotypic/plotting/_pipeline/_store_copyout.py`
- Modify: `src/phenotypic/plotting/_pipeline/_coordinator.py`: add `publish_store_figures`
- Test: `tests/unit/plotting/test_store_copyout.py`

**Interfaces:**
- Consumes: `read_image_figures_descriptor` (Task 4); `unique_page_stems`, `_atomic_write`, `_guarded_commit`, `safe_path_component`, `PlotPublicationBlocked` (writer); `ensure_plotlyjs_bundle`, `plotlyjs_src_for` (backends); `_image_output_stem` (coordinator).
- Produces: `publish_store_figures(store_path, plots_base, *, dataset, image_stem, publication_guard=None, commit_guard=None) -> None` and `PlotCoordinator.publish_store_figures(store_path, *, dataset, image_stem) -> None`.

**Output layout, reproduced exactly as today:**
- A binding with one page keyed `default`: files at `<plots_base>/<safe(binding)>/<safe(dataset)>/<stem>-<hash>.<ext>`, where `<stem>-<hash>` = `_image_output_stem(dataset, image_stem)`. No manifest.
- Otherwise: `<…>/<stem>-<hash>/<page-stem>.<ext>` plus `manifest.json` (schema 2). `<page-stem>` = `unique_page_stems([(key, label or key) …])`, the same rule as `_publish_plot_output_locked`.
- For each stored `plotly-json`, an `.html` is rendered beside it, referencing the hoisted `plotly.min.js` by relative src.
- Manifest page `files` map format → filename. It includes `"html"` for a generated HTML. `backend` is spelled `"plotly"`/`"matplotlib"`, as the writer spells it. `renderers` is `{"html": "available"}` when any page is Plotly, plus `{"png": "available"}` when any page carries a PNG. `failed` lists pages with no copied file.

- [ ] **Step 1: Let `record_plot_failure` take a pre-spelled message**

In `_failures.py`, change the annotation to `error: BaseException | str` and document it: *"A `str` is recorded verbatim. The store already spelled it with `normalize_figure_error`, and re-wrapping it would prefix a class name twice."* Change the entry line to `"error": error if isinstance(error, str) else _format_error(error),`.

- [ ] **Step 2: Write the failing tests**

```python
"""Copy-out: promoted store -> today's deliverables layout (spec §3 step 3)."""
from __future__ import annotations

import hashlib
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
    apply_image_figures_attributes,
    write_image_figures,
)


def figure_store(tmp_path: Path, figures: StoredFigures) -> Path:
    """A minimal promoted store: figures/ plus a root carrying the descriptor.

    Copy-out reads only these, so no pixels are needed. Shared with the ported
    coordinator tests (Task 9) -- keep it importable.
    """
    store = tmp_path / "p.ome.zarr"
    store.mkdir()
    phenotypic: dict = {"store_schema_version": 3}
    apply_image_figures_attributes(phenotypic, write_image_figures(store, figures))
    (store / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"phenotypic": phenotypic}}
    ))
    return store


def _plotly_json() -> bytes:
    import plotly.graph_objects as go

    return go.Figure(go.Bar(x=["a"], y=[1])).to_json().encode()


def _page(key="default", label=None, formats=("plotly-json",)):
    table = {
        "plotly-json": ("application/vnd.plotly.v1+json", ".plotly.json", _plotly_json()),
        "png": ("image/png", ".png", b"\x89PNG"),
    }
    stem = key.replace(" ", "-")
    return StoredFigurePage(key, label, "plotly", {"k": 1}, tuple(
        StoredFigureFile(fmt, table[fmt][0], f"{stem}{table[fmt][1]}", table[fmt][2])
        for fmt in formats
    ))


def _one(*pages, binding="sym", failed=()):
    return StoredFigures((StoredFigureBinding(binding, "MeasureSymZones", binding, pages),), failed)


def _publish(tmp_path, store, **kw):
    plots = tmp_path / "deliverables" / "plots"
    publish_store_figures(store, plots, dataset="ds 1", image_stem="plate_01", **kw)
    return plots


def test_a_single_default_page_lands_flat_with_generated_html(tmp_path):
    plots = _publish(tmp_path, figure_store(tmp_path, _one(_page())))
    base = plots / "sym" / "ds-1"
    stem = _image_output_stem("ds 1", "plate_01")
    assert sorted(p.name for p in base.iterdir()) == [f"{stem}.html", f"{stem}.plotly.json"]
    html = (base / f"{stem}.html").read_text()
    assert 'src="../../plotly.min.js"' in html
    assert (plots / "plotly.min.js").is_file()
    assert not (base / "manifest.json").exists()


def test_multi_page_writes_a_directory_and_manifest_v2(tmp_path):
    store = figure_store(tmp_path, _one(_page("first", "First"), _page("second", formats=("plotly-json", "png"))))
    plots = _publish(tmp_path, store)
    directory = plots / "sym" / "ds-1" / _image_output_stem("ds 1", "plate_01")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert [p["key"] for p in manifest["pages"]] == ["first", "second"]
    assert manifest["pages"][0]["files"] == {"plotly-json": "First.plotly.json", "html": "First.html"}
    assert manifest["pages"][1]["files"]["png"] == "second.png"
    assert manifest["pages"][0]["metadata"] == {"k": 1}
    assert manifest["renderers"] == {"html": "available", "png": "available"}
    assert 'src="../../../plotly.min.js"' in (directory / "First.html").read_text()


def test_a_tampered_file_is_recorded_and_not_copied(tmp_path):
    store = figure_store(tmp_path, _one(_page()))
    (store / "figures/sym/default.plotly.json").write_bytes(b"tampered")
    plots = _publish(tmp_path, store)
    assert not list((plots / "sym").rglob("*.plotly.json"))
    [line] = (plots / ".failures.jsonl").read_text().splitlines()
    record = json.loads(line)
    assert record["lifecycle"] == "image" and "sha256" in record["error"]


def test_descriptor_failures_become_failure_lines_verbatim(tmp_path):
    failed = (StoredFigureFailure("orient", None, None, "RuntimeError: boom at 0x…"),)
    plots = _publish(tmp_path, figure_store(tmp_path, _one(_page(), failed=failed)))
    [line] = (plots / ".failures.jsonl").read_text().splitlines()
    record = json.loads(line)
    assert record["error"] == "RuntimeError: boom at 0x…"
    assert (record["binding_id"], record["dataset"], record["image_stem"]) == ("orient", "ds 1", "plate_01")


def test_a_rerun_removes_a_leftover_rendering_of_a_republished_page(tmp_path):
    stem = _image_output_stem("ds 1", "plate_01")
    base = tmp_path / "deliverables" / "plots" / "sym" / "ds-1"
    base.mkdir(parents=True)
    (base / f"{stem}.png").write_bytes(b"old png from a run that stored png")
    _publish(tmp_path, figure_store(tmp_path, _one(_page())))
    assert not (base / f"{stem}.png").exists()


def test_a_store_without_figures_publishes_nothing(tmp_path):
    store = tmp_path / "p.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"phenotypic": {}}}
    ))
    plots = _publish(tmp_path, store)
    assert not plots.exists() or not any(plots.rglob("*"))


def test_a_refused_guard_propagates(tmp_path):
    from phenotypic.plotting._pipeline import PlotPublicationBlocked

    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, figure_store(tmp_path, _one(_page())), publication_guard=lambda: False)
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/unit/plotting/test_store_copyout.py -p no:cacheprovider -q`
Expected: `ModuleNotFoundError: phenotypic.plotting._pipeline._store_copyout`.

- [ ] **Step 4: Write `_store_copyout.py`**

```python
"""Copy a promoted store's figures out to deliverables/plots (spec §3 step 3).

The store is the single source: this never renders from a figure object. The
one thing it produces rather than copies is an HTML page for each stored
``plotly-json``, so a Plotly figure stays browsable in deliverables.
Best-effort: every failure is recorded to ``.failures.jsonl``; only a refused
guard (:class:`PlotPublicationBlocked`) propagates.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable

from phenotypic.abc_.plotting._store_formats import STORE_FORMATS
from phenotypic.sdk_ import CommitGuard
from phenotypic.sdk_._image_figures import read_image_figures_descriptor

from ._failures import record_plot_failure
from ._writer import (
    PlotPublicationBlocked,
    _atomic_write,
    _guarded_commit,
    _require_plot_publication,
    safe_path_component,
    unique_page_stems,
)

logger = logging.getLogger(__name__)

#: Every suffix a page can have in deliverables -- stored ones plus the
#: generated HTML. Used only to remove a republished page's leftovers.
_DELIVERABLE_SUFFIXES = (".plotly.json", ".html", ".png", ".svg")


def publish_store_figures(
    store_path: Path,
    plots_base: Path,
    *,
    dataset: str,
    image_stem: str,
    publication_guard: Callable[[], bool] | None = None,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Republish one promoted store's figures at today's deliverables paths.

    Args:
        store_path: A promoted ``*.ome.zarr`` store.
        plots_base: Resolved ``deliverables/plots`` directory.
        dataset: Dataset name (unsanitized; hashed into the output stem).
        image_stem: Image stem (unsanitized).
        publication_guard: Optional GUI compare-and-set predicate.
        commit_guard: Optional commit guard.

    Raises:
        PlotPublicationBlocked: If a guard refuses. Never swallowed.
    """
    from ._coordinator import _image_output_stem

    try:
        descriptor = read_image_figures_descriptor(store_path)
    except Exception as exc:  # noqa: BLE001 - copy-out is best-effort
        record_plot_failure(
            plots_base, binding_id="<store>", plot_class="<store>",
            lifecycle="image", error=exc, dataset=dataset, image_stem=image_stem,
        )
        return
    if descriptor is None:
        return
    output_stem = _image_output_stem(dataset, image_stem)
    for failure in descriptor.get("failed", []):
        record_plot_failure(
            plots_base,
            binding_id=failure["binding"],
            plot_class="<stored>",
            lifecycle="image",
            error=_located(failure["page"], failure["format"], failure["error"]),
            dataset=dataset,
            image_stem=image_stem,
        )
    for binding_id, binding in descriptor.get("bindings", {}).items():
        try:
            _publish_binding(
                Path(store_path), plots_base, binding_id, binding,
                dataset=dataset, image_stem=image_stem, output_stem=output_stem,
                publication_guard=publication_guard, commit_guard=commit_guard,
            )
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - copy-out is best-effort
            logger.warning("Copy-out of plot %s failed", binding_id, exc_info=exc)
            record_plot_failure(
                plots_base, binding_id=binding_id, plot_class=binding.get("class", "<stored>"),
                lifecycle="image", error=exc, dataset=dataset, image_stem=image_stem,
            )


def _located(page: str | None, fmt: str | None, error: str) -> str:
    """Prefix a stored error with where it happened, when that is known."""
    where = [f"page={page}"] if page is not None else []
    where += [f"format={fmt}"] if fmt is not None else []
    return f"[{' '.join(where)}] {error}" if where else error


def _publish_binding(
    store: Path,
    plots_base: Path,
    binding_id: str,
    binding: dict[str, Any],
    *,
    dataset: str,
    image_stem: str,
    output_stem: str,
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
    manifest_pages: list[dict[str, Any]] = []
    manifest_failed: list[dict[str, Any]] = []
    for page, stem in zip(pages, stems):
        files: dict[str, str] = {}
        for entry in page["files"]:
            suffix = STORE_FORMATS[entry["format"]].extension
            source = store / entry["path"]
            try:
                data = source.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(
                        f"stored {entry['path']} does not match its sha256 "
                        f"(descriptor {entry['sha256'][:12]}…, file {digest[:12]}…)"
                    )
                name = f"{stem}{suffix}"
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
                record_plot_failure(
                    plots_base, binding_id=binding_id,
                    plot_class=binding.get("class", "<stored>"), lifecycle="image",
                    error=_located(page["key"], entry["format"], f"{type(exc).__name__}: {exc}"),
                    dataset=dataset, image_stem=image_stem,
                )
        if files:
            _remove_leftovers(
                directory, stem, set(files.values()),
                publication_guard=publication_guard, commit_guard=commit_guard,
            )
            manifest_pages.append({
                "key": page["key"], "label": page["label"], "files": files,
                "backend": "matplotlib" if page["backend"] == "mpl" else "plotly",
                "metadata": page.get("metadata", {}),
            })
        else:
            manifest_failed.append({
                "key": page["key"], "label": page["label"],
                "error": "no stored file could be copied out",
            })
    if not flat:
        _write_manifest(
            directory, binding_id, binding.get("class", binding_id),
            manifest_pages, manifest_failed,
            publication_guard=publication_guard, commit_guard=commit_guard,
        )


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
    """Remove *stem*'s renderings this pass did not write (today's stale rule)."""
    for suffix in _DELIVERABLE_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if path.name in written or not path.exists():
            continue
        with _guarded_commit(publication_guard, commit_guard):
            path.unlink(missing_ok=True)


def _write_manifest(
    directory: Path,
    plot_id: str,
    plot_class: str,
    pages: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    renderers: dict[str, str] = {}
    if any(page["backend"] == "plotly" for page in pages):
        renderers["html"] = "available"
    if any("png" in page["files"] for page in pages):
        renderers["png"] = "available"
    manifest = {
        "schema_version": 2, "plot_id": plot_id, "class": plot_class,
        "renderers": renderers, "pages": pages, "failed": failed,
    }
    temporary = directory / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        with _guarded_commit(publication_guard, commit_guard):
            os.replace(temporary, directory / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["publish_store_figures"]
```

The suffix comes from the descriptor's `format` through `STORE_FORMATS`, never from parsing a filename: a stem may contain dots.

- [ ] **Step 5: Add the coordinator method**

```python
    def publish_store_figures(
        self, store_path: Path, *, dataset: str, image_stem: str
    ) -> None:
        """Copy one promoted store's figures to deliverables (spec §3 step 3)."""
        from ._store_copyout import publish_store_figures

        publish_store_figures(
            store_path, self._plots_base, dataset=dataset, image_stem=image_stem,
            publication_guard=self._publication_guard, commit_guard=self._commit_guard,
        )
```

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/unit/plotting/test_store_copyout.py tests/unit/plotting/test_failure_record.py -p no:cacheprovider -q`
Expected: PASS.

- [ ] **Step 7: Prove the sha256 check can fail**

Replace `if digest != entry["sha256"]:` with `if False:`. Expect the tamper test to FAIL. Restore it.

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_copyout.py
git add src/phenotypic/plotting/_pipeline/ tests/unit/plotting/test_store_copyout.py
git commit -m "feat(plotting): copy per-image figures out of the promoted store"
```

---

### Task 8: The preflight warns only about Chrome formats an image figure declares

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_backends.py`: `preflight_plot_backends` (≈166)
- Test: `tests/unit/plotting/test_backends.py` (append)

After this change, a `PlotImage` binding produces a PNG only when its store declares `"png"` (or Plotly `"svg"` under outcome A). The old warning, *"N Plotly plots will publish HTML only, without PNG"*, is still true for aggregate plots, but it is now misleading for image plots, which never produce a PNG by default.

- [ ] **Step 1: Write the failing test**

```python
def test_a_default_plotly_image_plot_needs_no_chrome_warning(monkeypatch):
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
    assert "ImgPng" in line and "store" in line
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/plotting/test_backends.py -p no:cacheprovider -q -k chrome_warning`
Expected: FAIL. The first assertion gets a non-empty warning.

- [ ] **Step 3: Split image bindings out of the classification**

At the top of the loop in `preflight_plot_backends`, route `PlotImage` bindings through the store formats:

```python
    from phenotypic.abc_.plotting import PlotImage
    from phenotypic.abc_.plotting._store_formats import default_store_formats

    chrome_formats = {"png", "svg"}
    image_store_ids: list[str] = []
    ...
    for binding in pipeline.get_plots():
        if isinstance(binding.plot, PlotImage):
            spec = declared_figure_spec(binding.plot)
            backend = spec.backend if spec is not None else None
            if backend == "mpl":
                mpl_ids.append(binding.id)
            elif backend == "plotly":
                _require_importable("plotly", "plotly", [binding.id])
                if chrome_formats & set(spec.store):
                    image_store_ids.append(binding.id)
            continue
        ...existing classification...
```

Change the early return to `if not (plotly_ids or undeclared_ids or image_store_ids) or chrome_available(): return []`. Before the install hint, append:

```python
    if image_store_ids:
        parts.append(
            f"{len(image_store_ids)} image plots declare a store format that "
            f"needs Chrome and will record it as failed: {', '.join(image_store_ids)}."
        )
```

Update the docstring's first paragraph with one sentence: *"Image plots are judged by their declared store formats (spec 2026-09-22 §2): only a declared `png`/Plotly `svg` needs Chrome."*

- [ ] **Step 4: Run the whole backends file**

Run: `uv run pytest tests/unit/plotting/test_backends.py tests/unit/cli/ -p no:cacheprovider -q -k "preflight or backend"`
Expected: PASS. An existing preflight test that used a `PlotImage` fixture and expected the old "HTML only" line must be updated to the new semantics. Change the fixture to `PlotMeas` if the test is about aggregates. If it is about image plots, assert the new line.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/_backends.py tests/unit/plotting/test_backends.py
git add src/phenotypic/plotting/_pipeline/_backends.py tests/unit/plotting/test_backends.py
git commit -m "fix(plotting): preflight judges image plots by their declared store formats"
```

---

### Task 9: Wire every mode; retire `emit_image`; bump the process revision

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_single.py`: full mode (≈344–366), measure mode (≈437–456)
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py`: Stage 3 (≈578–600)
- Modify: `src/phenotypic/_cli/_cli_process_only.py`: `process_single_apply_only_core` (≈330–352)
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:205`
- Modify: `src/phenotypic/plotting/_pipeline/_coordinator.py`: delete `emit_image`, `_publish_image_value`; drop the now-unused imports (`_render_page`, `_remove_stale_sibling`, `_format_error`, `FigureAdapter`, `normalize_plot_output`, `_require_plot_publication` — delete only those ruff reports as unused)
- Modify: `tests/unit/plotting/test_coordinator.py`, `tests/integration/plotting/test_publication_end_to_end.py`, `tests/unit/cli/test_embedded_measurement_replacement.py:155`
- Modify: `tests/unit/cli/test_work_id_semantics_revision.py` (if it pins the value 2)
- Test: `tests/integration/cli/test_figures_in_store.py`

**Interfaces:**
- Consumes: `PlotCoordinator.build_image_figures`, `PlotCoordinator.publish_store_figures`, `build_image_figures`, and every `figures=` keyword from Task 6.

- [ ] **Step 1: Write the failing CLI-level tests**

```python
"""Every mode writes figures into the store; deliverables are a copy (spec §3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_._image_figures import read_image_figures_descriptor


def _write_inputs(tmp_path: Path, *, with_plot: bool, measure_only_plot: bool = False):
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize, MeasureSymZones

    image = tmp_path / "in" / "plate.tiff"
    image.parent.mkdir()
    imsave(str(image), load_synth_yeast_plate().rgb[:], check_contrast=False)
    sym = MeasureSymZones()
    meas = {"size": MeasureSize(), "sym": sym} if not measure_only_plot else {"sym": sym}
    pipeline = ImagePipeline(
        ops={"detect": OtsuDetector()}, meas=meas, plots=[sym] if with_plot else []
    )
    path = tmp_path / "pipeline.json"
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
    [page] = descriptor["bindings"]["sym"]["pages"]
    assert [f["format"] for f in page["files"]] == ["plotly-json"]
    deliverable = list((out / "deliverables" / "plots" / "sym" / "ds").glob("plate-*.plotly.json"))
    assert len(deliverable) == 1
    assert deliverable[0].read_bytes() == (store / page["files"][0]["path"]).read_bytes()


def test_full_mode_without_image_bindings_writes_no_figures(tmp_path):
    image, pipeline = _write_inputs(tmp_path, with_plot=False)
    _out, store = _full(tmp_path, pipeline, image)
    assert read_image_figures_descriptor(store) is None
    assert not (store / "figures").exists()


def test_measure_mode_rebuilds_figures_from_the_current_pipeline(tmp_path):
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    image, with_plot = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, with_plot, image)
    (tmp_path / "second").mkdir()
    _, without_plot = _write_inputs(tmp_path / "second", with_plot=False)
    process_single_store_measure_core(
        without_plot, store, out, "ds", "Image",
        OutputManager.from_config(out, ".tiff", save_overlays=False),
    )
    assert read_image_figures_descriptor(store) is None
    process_single_store_measure_core(
        with_plot, store, out, "ds", "Image",
        OutputManager.from_config(out, ".tiff", save_overlays=False),
    )
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
        assert list(read_image_figures_descriptor(store)["bindings"]) == ["sym"]
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
Expected: FAIL. There is no `figures` descriptor yet, and the revision is 2.

- [ ] **Step 3: Wire full mode**

In `_cli_process_single.py`, replace the `PlotCoordinator(...).emit_image(...)` block (≈343–355) and the following `save_image_store` call with:

```python
        from phenotypic.plotting._pipeline import PlotCoordinator

        _check_active(active_check)
        coordinator = PlotCoordinator(pipeline, output_dir, commit_guard=commit_guard)
        figures = coordinator.build_image_figures(image)
        _check_active(active_check)
        set_provenance_status(image, "complete")
        saved_store = output_manager.save_image_store(
            image,
            dataset_name,
            image_stem,
            work_id=work_id,
            commit_guard=commit_guard,
            measurements=measurements,
            figures=figures,
        )
        if saved_store is None:
            raise RuntimeError(
                f"Final image store publication failed for {dataset_name}/{image_stem}"
            )
        # After promotion, before the completion record: a crash between the
        # two re-runs the image, so deliverables never lag a certified store.
        _check_active(active_check)
        coordinator.publish_store_figures(
            saved_store, dataset=dataset_name, image_stem=image_stem
        )
```

Keep the existing `if saved_store is None: raise` text exactly as it is. The block above repeats it only to show where the copy-out goes.

- [ ] **Step 4: Wire staged Stage 3**

Apply the same transformation in `_cli_staged_workers.py` (≈578–600) with `plan.post_pipeline`. The copy-out goes after the existing `if saved_store is None or not valid_staged_store(saved_store): raise ...` check.

- [ ] **Step 5: Wire measure mode**

In `process_single_store_measure_core`, replace the `replace_image_store_measurements(...)` call and the `PlotCoordinator(...).emit_image(...)` block (≈437–456) with:

```python
    from phenotypic.plotting._pipeline import PlotCoordinator

    coordinator = PlotCoordinator(pipeline, output_dir, commit_guard=commit_guard)
    # Built BEFORE the table replace so the figures ride the same root-last
    # transaction: a store's figures and its table always come from the same
    # pipeline (spec §3 "Measure mode semantics").
    figures = coordinator.build_image_figures(image)
    output_manager.replace_image_store_measurements(
        store_path,
        measurements,
        dataset_name,
        commit_guard=commit_guard,
        figures=figures,
        rebuild_figures=True,
    )
    coordinator.publish_store_figures(store_path, dataset=dataset_name, image_stem=stem)
```

Keep the long comment above the replace call (CAN-3). Keep the "Marker refresh is the final successful per-image publication" comment and code after it unchanged.

- [ ] **Step 6: Wire process mode**

In `process_single_apply_only_core`, right after `set_provenance_status(image, "complete")` and still inside the same `try`, add:

```python
        # Figures only when there is a store to hold them (spec §3 by mode).
        # Measurer-backed bindings recompute inside inspect() here, because
        # apply() never filled their cache -- accepted (spec §3 process mode).
        figures = None
        if process_format == "zarr":
            from phenotypic.plotting._pipeline._store_figures import build_image_figures

            figures = build_image_figures(pipeline, image)
```

Add `figures = None` before the `try`, so the name exists on the tiff path. Pass `figures=figures` to `write_process_only_layer(...)`. Process mode does no copy-out.

- [ ] **Step 7: Bump the revision**

In `_cli_failure_tracker.py`, append to the changelog comment:

```python
#: 2 -> 3: process-mode stores now carry the pipeline's per-image figures
#:         (spec 2026-09-22-figures-in-ome-zarr §3). Also invalidates in-flight
#:         ``tiff`` continuations -- deliberate; invalidating too much is safe.
PROCESS_LAYER_SEMANTICS_REVISION = 3
```

Run: `uv run pytest tests/unit/cli/test_work_id_semantics_revision.py tests/integration/cli/test_process_objmap_semantics.py -p no:cacheprovider -q`
If a test pins the literal `2`, change it to `3`. Tests that compute `shipped - 1` need no change.

- [ ] **Step 8: Retire `emit_image` and port its tests**

Delete `PlotCoordinator.emit_image` and `_publish_image_value`, and update the class docstring to say that image plots publish through `build_image_figures` → store → `publish_store_figures`. Keep `_image_output_stem`, which the copy-out uses. Run `uv run ruff check src/phenotypic/plotting/_pipeline/_coordinator.py` and delete only the imports it reports as unused.

Port every test that calls `emit_image`. Give `tests/unit/plotting/test_coordinator.py` one helper:

```python
from tests.unit.plotting.test_store_copyout import figure_store


def _emit_image_via_store(coordinator, tmp_path, image, *, dataset="ds", image_stem="plate-1"):
    """build -> minimal store -> copy-out: the path every CLI mode now takes."""
    stored = coordinator.build_image_figures(image)
    if stored is None:
        return None
    store = figure_store(tmp_path / f"store-{image_stem}", stored)
    coordinator.publish_store_figures(store, dataset=dataset, image_stem=image_stem)
    return stored
```

(`figure_store` needs `(tmp_path / ...)` to exist: have the helper `mkdir(parents=True, exist_ok=True)` first. If importing across test modules trips on package layout, move `figure_store` to `tests/unit/plotting/_store_fixtures.py` and import it from there in both files.)

Then, for each `emit_image` call site listed by `grep -n "emit_image" tests/unit/plotting/test_coordinator.py`:
- Layout, stable-name and collision tests (≈92, 119, 137, 150, 424, 707): replace the call with `_emit_image_via_store(...)`. An mpl `_ImagePlot` still yields `<stem>-<hash>.png`. A Plotly one yields `.plotly.json` + `.html` and **no PNG even with Chrome**. Update any PNG-with-Chrome assertion to that.
- Strict-mode test (≈105): `strict` no longer exists. Delete the test. Its guarantee ("a failure is visible") is now `test_inspect_raising_omits_the_binding_and_normalises_the_address` (Task 5) plus `test_descriptor_failures_become_failure_lines_verbatim` (Task 7). Name both in the commit message.
- Guard and fence tests (≈539–573, 793–811, 877–880, 915, 941, 954): route through `_emit_image_via_store`. The guard now fires inside `publish_store_figures`, so the expected `PlotPublicationBlocked` / no-write assertions hold unchanged.
- The failure-class test (≈754–800): the durable line's `error` now starts with the real exception class. The descriptor spelling (`normalize_figure_error`) keeps it, so assert `record["error"].startswith("<ExpectedClass>: ")`.
- Stale-sibling tests (≈1034–1093): copy-out's `_remove_leftovers` preserves "only a republished page loses its leftovers", so keep the assertions and swap the call.

In `tests/integration/plotting/test_publication_end_to_end.py`, replace `PlotCoordinator(pipeline, tmp_path).emit_image(image, dataset="ds 1", image_stem="plate_01", strict=True)` with the build → `image.save2zarr(tmp_path / "s.ome.zarr", figures=stored)` → `publish_store_figures` sequence. Assert `.plotly.json` + `.html` for Plotly, whatever Chrome says (the old assertion `png is chrome_available()` no longer holds for image plots). Keep the aggregate tests unchanged.

In `tests/unit/cli/test_embedded_measurement_replacement.py:155`, patch `PlotCoordinator.publish_store_figures` instead of `emit_image`. The test's claim ("a failure after the table write leaves the old marker stale") still holds, because copy-out runs before the marker refresh.

- [ ] **Step 9: Run the touched surface**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/integration/cli/test_figures_in_store.py tests/unit/plotting/ tests/integration/plotting/ tests/unit/cli/test_embedded_measurement_replacement.py tests/unit/cli/test_embedded_measurement_publication.py tests/unit/cli/test_process_only_zarr.py -p no:cacheprovider -q`
Expected: PASS. Run any failure in isolation before attributing it (project rule).

- [ ] **Step 10: Prove the wiring test can fail**

Remove `figures=figures` from the full-mode `save_image_store` call. Expect `test_full_mode_stores_figures_and_copies_them_out` to FAIL. Restore it.

- [ ] **Step 11: Commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_process_single.py src/phenotypic/_cli/_cli_staged_workers.py src/phenotypic/_cli/_cli_process_only.py src/phenotypic/_cli/_cli_failure_tracker.py src/phenotypic/plotting/_pipeline/_coordinator.py tests/integration/cli/test_figures_in_store.py tests/unit/plotting/test_coordinator.py tests/integration/plotting/test_publication_end_to_end.py tests/unit/cli/test_embedded_measurement_replacement.py
git add -A src/phenotypic tests
git commit -m "feat(cli): per-image figures live in the store in every mode; emit_image retired"
```

---

### Task 10: The cross-mode properties — reproducibility, cache parity, migrate, lazy imports

**Files:**
- Modify: `tests/unit/cli/test_process_only_zarr.py`: add a byte-identical test with a figure binding
- Create: `tests/unit/measure/test_zone_figure_cache_parity.py`
- Modify: `tests/unit/cli/test_cli_provenance_migration.py`: a figure-carrying direct store survives migrate
- Test run: `tests/unit/ci/test_startup_imports.py`, `tests/unit/ci/test_deferred_imports.py`

- [ ] **Step 1: The byte-identical process store, now with figures**

Append to `tests/unit/cli/test_process_only_zarr.py`:

```python
@pytest.fixture
def plotted_pipeline_file(tmp_path: Path) -> Path:
    """A detector plus a measurer-backed figure binding (spec §3, §4)."""
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSymZones

    sym = MeasureSymZones()
    path = tmp_path / "plotted.json.pht-pipe"
    ImagePipeline(ops=[OtsuDetector()], meas={"sym": sym}, plots=[sym]).to_json(path)
    return path


def test_two_runs_with_a_figure_binding_produce_byte_identical_stores(
    tmp_path: Path, source_image: Path, plotted_pipeline_file: Path
) -> None:
    """Spec §4: the byte-identity contract now includes figures/."""
    first = _run_to_store(plotted_pipeline_file, source_image, tmp_path / "a")
    second = _run_to_store(plotted_pipeline_file, source_image, tmp_path / "b")
    left, right = _tree_bytes(first), _tree_bytes(second)
    assert any(name.startswith("figures/sym/") for name in left)
    assert sorted(left) == sorted(right)
    assert [name for name in left if left[name] != right[name]] == []
```

Prove it can fail: in `_serialize_html`, temporarily make the div id `uuid4().hex`; the test does not notice, because `html` is not stored by default. That shows this test guards the *default* path only; `plotly-json` stability across processes is pinned in Task 3. Revert. Then temporarily append `str(time.time())` to the bytes in `_serialize_plotly_json`, expect FAIL, and revert.

- [ ] **Step 2: Cache parity**

```python
"""A zone figure does not depend on whether measure() just ran (spec §4)."""
from __future__ import annotations

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

    assert encode(hit) == encode(recomputed)
```

Run: `uv run pytest tests/unit/measure/test_zone_figure_cache_parity.py -p no:cacheprovider -q`
If `OtsuDetector().apply(image, inplace=True)` is not the right signature, match `tests/unit/detect` usage (`grep -rn "OtsuDetector()" tests/unit/detect | head -3`).

**If it FAILS, that is a real provider bug** (spec §4: "a bug in the provider, not a tolerance"). Stop and report it to the orchestrator with the first differing JSON path. Do not loosen the assertion. Compare with `json.loads` to find the path.

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

Also add the absence case, using the existing test above it as the template: a store with no `figures` key still migrates and gains none. Assert `"figures" not in after`.

- [ ] **Step 4: Run the lazy-import guards**

Run: `uv run pytest tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py -p no:cacheprovider -q`
Expected: PASS. A failure means a module-level plotly/matplotlib/zarr import slipped into a Task 2–7 file. Move it into the function that uses it.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix tests/unit/cli/test_process_only_zarr.py tests/unit/measure/test_zone_figure_cache_parity.py tests/unit/cli/test_cli_provenance_migration.py
git add tests/
git commit -m "test: figure reproducibility, cache parity, and migrate neutrality"
```

---

### Task 11: Documentation and the spec clarifications

**Files:**
- Modify: `.claude/skills/working-with-ome-zarr/SKILL.md`
- Modify: `src/phenotypic/_cli/CLAUDE.md`
- Modify: root `CLAUDE.md` (the `--mode process` bullet)
- Modify: `src/phenotypic/abc_/CLAUDE.md`
- Modify: `docs/source/extending/pages/custom_plotter.md`
- Modify: `docs/source/how_to/pages/zarr_storage.md`
- Modify: `docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md`

- [ ] **Step 1: Spec clarifications**

In the spec:
- In the §1 descriptor example, add `"metadata": {}` to the page object, and add a bullet under **Rules**: *"`metadata` is the page's `PlotPage.metadata`, JSON-native; the copy-out reproduces manifest v2 from it."*
- In §3 step 1, add a sentence: *"Implemented as the module function `build_image_figures(pipeline, image)`; `PlotCoordinator.build_image_figures(image)` delegates. Process mode calls the function, having no `plots_base`."*
- In §1, replace the example error `ChromeNotFoundError: <message>` with `PlotBackendUnavailable: Plotly PNG export needs Chrome (kaleido); install it with plotly_get_chrome`.
- In §2, replace the Plotly-SVG gate paragraph with the recorded outcome (A or B) and one line of evidence from Task 1.

- [ ] **Step 2: The store contract skill**

In `.claude/skills/working-with-ome-zarr/SKILL.md`, add a *Per-image figures* row to the store-contract table (location `figures/<binding>/…`; described by `attributes.phenotypic.figures`; optional; not in `ome.series`). Add one paragraph on the descriptor: media type is the contract, `sha256` bound by the root, no `store_schema_version` bump, and measure mode rebuilds the group in the table transaction.

- [ ] **Step 3: `_cli/CLAUDE.md` and root `CLAUDE.md`**

In `_cli/CLAUDE.md`, add a short section, *"Per-image figures"*: the write path per mode (the §3 table in one sentence per row), the copy-out running after promotion and before the completion record, process revision 3, full mode unchanged, and `emit_image` retired. In root `CLAUDE.md`, in the `--mode process` bullet, add: *"A store also carries the pipeline's per-image figures under `figures/` (revision 3); flat `tiff` exports carry none."*

- [ ] **Step 4: `abc_/CLAUDE.md` and the extending guide**

In `abc_/CLAUDE.md`, add the `@figure(store=...)` convention: the closed set, the per-backend defaults, class-definition `TypeError`s, and the fact that an `inspect()` override falls back to the backend default per page.

In `docs/source/extending/pages/custom_plotter.md`, add a section *"Storing figures with the image"*. Cover:
- `store=`;
- the four formats and their media types, as a table;
- determinism (why `html` uses the CDN);
- the defaults;
- that a default Plotly figure has no PNG in `deliverables/`, even with Chrome;
- that `deliverables/plots/` is a copy of the store.

Include one runnable docstring-style example using `load_synth_yeast_plate()`.

- [ ] **Step 5: The zarr storage how-to**

In `docs/source/how_to/pages/zarr_storage.md`, add *"Reading an image's figures"*: read `zarr.json` → `attributes.phenotypic.figures` → pick the file whose `media_type` you can render → verify `sha256`. Include a stdlib-only snippet:

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

- [ ] **Step 6: Verify the docs build renders the changed pages**

Use the **`slurm-job`** skill to submit one job running `uv run sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never -b html docs/source <out>` (per the global rule: never locally, never `-j auto`). Then read the generated HTML for the two changed pages. Exit 0 is not the check.

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/working-with-ome-zarr/SKILL.md src/phenotypic/_cli/CLAUDE.md CLAUDE.md src/phenotypic/abc_/CLAUDE.md docs/source/ docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md
git commit -m "docs: per-image figures in the OME-Zarr store"
```

---

### Task 12: Final regression

- [ ] **Step 1: Types and lint on the changed surface**

Run: `uv run mypy src/phenotypic/abc_/plotting src/phenotypic/plotting/_pipeline src/phenotypic/sdk_/_image_figures.py src/phenotypic/_cli/_cli_process_only.py`
Run: `uv run ruff check $(git diff --name-only origin/main...HEAD -- '*.py')`
Expected: no new errors.

- [ ] **Step 2: The full sharded suite, once, as a Slurm job**

Use the **`run-phenotypic-test`** skill with the committed batch script `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`, run in a worktree detached at this branch's HEAD SHA (global rule: a parallel gate measures one tree). Compare against the recorded baseline (memory: 11,106 tests / 81 known failures, all outside `sdk_`/`_cli`/`gui`). Run each new failure in isolation before attributing it.

- [ ] **Step 3: Report**

Report test totals only from the job output you just read. List any failure attributed to this change, with its isolated-run result.

---

## Self-review

**Spec coverage.**

| Spec | Task |
|---|---|
| §1 layout, descriptor, media types | 2 (table), 4 (writer) |
| §1 presence, ordering | 4, 5, 6 |
| §1 failure granularity, `error` text | 5 |
| §1 namespace, no version bump | 6 (independent-reader test) |
| §1 hashes, copy-out verify, no re-hash in continuation | 7; continuation untouched |
| §2 signature, validation, defaults | 2 |
| §2 which declaration applies, `inspect()` override fallback | 5 (`declared_figure_spec`) |
| §2 serializers, CDN HTML, Plotly SVG gate | 1, 3 |
| §2 lazy imports | 10 |
| §3 build, store write, copy-out | 5, 6, 7 |
| §3 by-mode table | 9 |
| §3 measure semantics, migrate path untouched | 6, 9, 10 |
| §3 continuation (revision 3, full unchanged) | 9 |
| §4 byte-identical process store | 10 |
| §4 cross-process serializer stability | 3 |
| §4 cache parity | 10 |
| §5 tests | 2–10 (each row appears in its task) |
| §5 Chrome lane | 3 |
| §6 docs | 11 |
| Blast radius: preflight semantics | 8 |

**Type consistency.** The following names are identical across Tasks 4–9: `StoredFigures` / `StoredFigureBinding(binding_id, plot_class, directory, pages)` / `StoredFigurePage(key, label, backend, metadata, files)` / `StoredFigureFile(format, media_type, filename, data)` / `StoredFigureFailure(binding, page, format, error)`, the keyword `figures=` on every writer, and `rebuild_figures=` on the two replace functions.
