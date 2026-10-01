# Plot subfolders and per-ROI calibration overlays: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every stored plot gets its own folder (`figures/<run>/<binding>/<plot>/<file>`), and `CalibrateColorRpcc` stores one tile overlay per ROI.

**Architecture:**
- An optional `PlotPage.plot` names a page's folder.
- One shared helper turns a page list into `(plot folder, file stem)` pairs, by the existing `unique_page_stems` rule. The store builder, the store copy-out and the direct publisher all use it.
- The figures descriptor goes to `schema_version` 2. Every page and failure records its `plot`. Version 1 runs are read as flat pages, and are never moved.

**Tech Stack:**
- Python 3.12, `uv`, pytest.
- Zarr v3 group documents, written as plain JSON by `sdk_/_image_figures.py`.
- matplotlib and Plotly figures, through the existing serializers.

**Spec:** `docs/superpowers/specs/2026-09-30-plot-subfolders/design.md` (approved 2026-09-30). Executors read both.

## Global Constraints

- **Run everything through `uv`:** `uv run pytest …`, `uv run ruff check --fix <explicit paths>`. Never run bare `ruff check --fix`.
- **Run pytest with `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg`** and `-o addopts="" -m "not slow" -p no:cacheprovider`. Never `-n auto`, never `-x` for a run whose numbers are quoted.
- **Tests per stage:**
  - per task: the task's own test files;
  - per phase (after Tasks 3, 5 and 7): the 17-file figure set below, as a Slurm job;
  - once at the end: the full sharded suite, via `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.
- **Baseline at `ba725001`:** the 17-file figure set gives **313 passed, 0 failed** (Slurm job 29302628).
- **The 17-file figure set:**
  - `tests/integration/cli/`: `test_calibration_figure_in_store.py`, `test_figures_in_store.py`, `test_figures_run_date_cli.py`
  - `tests/integration/plotting/`: `test_publication_end_to_end.py`
  - `tests/unit/abc_/plotting/`: `test_imports.py`
  - `tests/unit/cli/`: `test_cli_figures_run_date.py`, `test_process_only_zarr.py`, `test_staged_figures_keep.py`
  - `tests/unit/correction/`: `test_calibration_overlay.py`, `test_calibration_plot_image.py`
  - `tests/unit/plotting/`: `test_coordinator.py`, `test_output_adapter.py`, `test_public_imports.py`, `test_store_copyout.py`, `test_store_figures_build.py`
  - `tests/unit/sdk_/`: `test_image_figures.py`, `test_image_figures_store.py`
- **Outside the 17-file set, also run at each phase gate:**
  - `tests/unit/plotting/test_plot_meas_time_series.py`
  - `tests/gui/results_viewer/test_mutation_guard.py`, which needs the Qt env: `uv sync --group dev --group test-qt --extra gui --extra napari`
  Neither ran in the baseline. Run both once at `ba725001` first, so their results can be compared by name.
- **The sweep rule (plan review I1).** One level deeper, a check against the old flat location passes while testing nothing. So in every test file a task touches:
  - every negative check (`not … .exists()`, `glob(...) == []`) is made against the new location, or becomes `rglob`;
  - every `glob`-based positive check, and every `first == second` comparison of globbed lists, also asserts the result is **non-empty**.
  Each task's sweep step lists the known hits and greps for more:
  ```bash
  grep -n 'glob(\|exists()\|iterdir\|first == second' <the task's touched test files>
  ```
- **Figures descriptor `schema_version`:** written `2`; readable `{1, 2}`.
- **Deliverables manifest `schema_version`:** `3`.
- **By mode (spec §3):** full, measure and staged store figures **and** copy them to deliverables. `--mode process --process-format zarr` stores them in the image's store **only**, with no `deliverables/`. A TIFF process export carries none. No task changes this; Task 7 pins it.
- **Per-ROI file names:** `roi_<index>` (spec D3), where `<index>` is `RoiOverlay.roi_index`.
- **`PlotPage.plot` refuses `/`** (spec D6). `zarr.json` and `manifest.json` are reserved plot-folder names (case-folded), so they get the digest suffix.
- **The plan review** is `docs/superpowers/reports/2026-09-30-plot-subfolders/plan-review.md` (0 blocker, 8 important, 14 minor). This revision folds in I1–I8 and minor items 1, 2, 5, 6, 7, 9, 10, 11 and 14. Minor items 3, 4, 8, 12 and 13 are accepted as noted there.
- **A page with no `plot` is stored at `<key>/<key>.<ext>`** (spec D2).
- **Commits:** one per task, ending with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SJAhjtbTyyECE3caGanGEs
  ```

## Decisions this plan makes (implementing the spec; confirm at review)

- **P1: deliverables file names.**
  - The store copy-out copies each stored file under **the store's own relative
    path**, `<plot folder>/<file>`. So deliverables name files by **key**, as the
    store does: `tiles/roi_0.png`, not `Tile overlay, ROI 0.png`.
  - This is what "mirror the store" (D4) means file for file. Today the copy-out
    names files by label.
  - The direct publisher (aggregate plots: `PlotMeas` / `PlotAnalysis` /
    `PlotQc`) has no store to mirror, so it keeps today's label-preferred file
    names and only gains the plot folder.
- **P2:** the spec's `page_plot(page)` helper is implemented as the property
  `PlotPage.plot_name`.
- **P3:** a page kept flat from a version 1 run is written with `"plot": null`
  in the version 2 descriptor. A null `plot` means flat everywhere.

## Review Focus

1. **Names that clean to the same folder:** for example plots `"Tiles"` and
   `"tiles"`, or a plot named with spaces. They must get distinct folders,
   through the stable digest suffix, never overwrite each other. Test: Task 3,
   `test_plot_folders_that_collide_get_distinct_names`.
2. **Re-measuring a store whose same-run folder an older PhenoTypic wrote**
   (version 1, flat): the kept calibration pages must stay flat and verify, not
   be refused. Test: Task 3, `test_a_flat_v1_binding_is_kept_flat`.
3. **An older (version 1-only) PhenoTypic meeting a version 2 store:** it adds
   no run and touches no figure. Test: Task 2,
   `test_a_v1_only_writer_adds_no_run_to_a_v2_store`.
4. **A pipeline with one ROI:** it stores `tiles/roi_0` and `delta_e/delta_e`,
   with no empty or stray folder. Test: Task 6, `test_one_roi_stores_one_overlay`.
5. **A page that failed outright inside a plot folder:** its failure keeps its
   `plot` through the descriptor and into the deliverables manifest's `failed`.
   Test: Task 4, `test_a_failed_page_keeps_its_plot_in_the_manifest`.

---

## File map

| File | Responsibility | Tasks |
|---|---|---|
| `src/phenotypic/abc_/plotting/_output.py` | `PlotPage.plot`, `plot_name`, uniqueness per `(plot, key)` | 1 |
| `src/phenotypic/sdk_/ngff_.py` | `FIGURES_SCHEMA_VERSION = 2`, `READABLE_FIGURES_SCHEMA_VERSIONS` | 2 |
| `src/phenotypic/sdk_/_image_figures.py` | path helpers, descriptor version 2 writer, version 1/2 readers, carry | 2 |
| `src/phenotypic/plotting/_pipeline/_writer.py` | `plot_page_paths` helper (Task 3); direct publisher with plot folders and manifest version 3 (Task 5) | 3, 5 |
| `src/phenotypic/plotting/_pipeline/_store_figures.py` | build pages into plot folders; keep version 1 and version 2 bindings | 3 |
| `src/phenotypic/plotting/_pipeline/_store_copyout.py` | mirror the store into deliverables, manifest version 3, no flat case | 4 |
| `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` | per-ROI `tiles` pages | 6 |
| tests, as named per task | | 1-7 |
| docs | figures spec note, `CLAUDE.md` + `AGENTS.md`, `custom_plotter.md` | 8 |

---

### Task 1: `PlotPage.plot` and per-`(plot, key)` uniqueness

**Files:**
- Modify: `src/phenotypic/abc_/plotting/_output.py` (`PlotPage`, `PlotOutput`)
- Test: `tests/unit/abc_/plotting/test_plot_page_plot.py` (new)

**Interfaces:**
- Produces: `PlotPage(key, figure, label=None, metadata={}, plot=None)` and the property `PlotPage.plot_name -> str` (`plot` if set, else `key`). `PlotOutput` refuses a duplicate `(plot_name, key)` with `ValueError("plot output contains duplicate page keys: [...]")`, keeping today's message prefix.

- [ ] **Step 1: Write the failing tests**

```python
"""PlotPage.plot names a page's folder (spec 2026-09-30 §1)."""
from __future__ import annotations

import pytest

from phenotypic.abc_.plotting import PlotOutput, PlotPage


def test_plot_defaults_to_the_key():
    page = PlotPage(key="delta_e", figure=object())
    assert page.plot is None
    assert page.plot_name == "delta_e"


def test_plot_names_the_folder_when_given():
    page = PlotPage(key="roi_0", plot="tiles", figure=object())
    assert (page.plot_name, page.key) == ("tiles", "roi_0")


@pytest.mark.parametrize("plot", ["", 3, "a/b", "/tiles", "tiles/"])
def test_an_empty_non_string_or_slashed_plot_is_refused(plot):
    with pytest.raises(ValueError, match="plot"):
        PlotPage(key="k", plot=plot, figure=object())


def test_the_same_key_in_two_plots_is_allowed():
    output = PlotOutput(pages=(
        PlotPage(key="roi_0", plot="tiles", figure=object()),
        PlotPage(key="roi_0", plot="masks", figure=object()),
    ))
    assert [(p.plot_name, p.key) for p in output.pages] == [("tiles", "roi_0"), ("masks", "roi_0")]


def test_the_same_key_in_one_plot_is_refused():
    with pytest.raises(ValueError, match="duplicate page keys"):
        PlotOutput(pages=(
            PlotPage(key="roi_0", plot="tiles", figure=object()),
            PlotPage(key="roi_0", plot="tiles", figure=object()),
        ))


def test_two_bare_pages_with_one_key_are_still_refused():
    with pytest.raises(ValueError, match="duplicate page keys"):
        PlotOutput(pages=(PlotPage(key="a", figure=object()), PlotPage(key="a", figure=object())))


def test_a_bare_page_and_a_plotted_page_with_one_key_coexist():
    # (plot "a", key "a") and (plot "tiles", key "a") are different pages.
    PlotOutput(pages=(PlotPage(key="a", figure=object()),
                      PlotPage(key="a", plot="tiles", figure=object())))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest -o addopts="" -p no:cacheprovider tests/unit/abc_/plotting/test_plot_page_plot.py -q`
Expected: FAIL with `TypeError: PlotPage.__init__() got an unexpected keyword argument 'plot'`.

- [ ] **Step 3: Implement**

In `_output.py`, replace `PlotPage` and `PlotOutput.__post_init__`:

```python
@dataclass(frozen=True)
class PlotPage:
    """One independently saveable figure page.

    Args:
        key: Stable logical page key; names the page's file.
        figure: Plotly or Matplotlib figure.
        label: Optional human-readable page label.
        metadata: Immutable-by-convention selector metadata.
        plot: The plot this page belongs to; names the folder its file is
            stored in. ``None`` means the page is a plot of its own, stored
            at ``<key>/<key>`` (spec 2026-09-30 §1).
    """

    key: str
    figure: FigureLike
    label: str | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )
    plot: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("plot page key must be a non-empty string")
        if self.plot is not None and (
            not isinstance(self.plot, str) or not self.plot or "/" in self.plot
        ):
            raise ValueError(
                "plot page plot must be None or a non-empty string without '/'"
            )

    @property
    def plot_name(self) -> str:
        """The plot (folder) this page is stored under: ``plot``, else ``key``."""
        return self.plot if self.plot is not None else self.key


@dataclass(frozen=True)
class PlotOutput:
    """Ordered pages returned by a plotting invocation."""

    pages: tuple[PlotPage, ...]

    def __post_init__(self) -> None:
        ids = [(page.plot_name, page.key) for page in self.pages]
        duplicates = sorted({f"{plot}/{key}" for plot, key in ids if ids.count((plot, key)) > 1})
        if duplicates:
            raise ValueError(f"plot output contains duplicate page keys: {duplicates}")
```

- [ ] **Step 4: Run the new tests and the existing plotting-API tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest -o addopts="" -p no:cacheprovider tests/unit/abc_/plotting -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/abc_/plotting/_output.py tests/unit/abc_/plotting/test_plot_page_plot.py
git add src/phenotypic/abc_/plotting/_output.py tests/unit/abc_/plotting/test_plot_page_plot.py
git commit -m "feat(plotting): PlotPage.plot names a page's folder; pages unique per (plot, key)"
```

---

### Task 2: Descriptor version 2, path helpers, writer, readers, carry

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py:102`
- Modify: `src/phenotypic/sdk_/_image_figures.py` (`StoredFigurePage`, `StoredFigureFailure`, `figure_file_path`, `split_figure_file_path`, `write_image_figures`, `_carry_file`, `known_figures_schema`, `read_figure_run`)
- Test: `tests/unit/sdk_/test_image_figures.py`

**Interfaces:**
- Consumes: nothing from Task 1. This module never imports the plotting API.
- Produces:
  - `ngff_.FIGURES_SCHEMA_VERSION == 2`, and `ngff_.READABLE_FIGURES_SCHEMA_VERSIONS == frozenset({1, 2})`.
  - `StoredFigurePage(key, label, backend, metadata, files, plot: str | None = None, directory: str | None = None)`. `plot` is the logical plot name and `directory` the cleaned folder name; both `None` means a flat (version 1) page.
  - `StoredFigureFailure(binding, page, format, error, plot: str | None = None)`.
  - `figure_file_path(run_id, directory, filename, plot_directory: str | None = None) -> str`.
  - `split_figure_file_path(path) -> tuple[str, str, str | None, str]`, which is `(run_id, binding_dir, plot_dir_or_None, filename)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/sdk_/test_image_figures.py`; update the 3 existing assertions noted in Step 4)

```python
def _stored_v2(run: FigureRun = _RUN) -> StoredFigures:
    tiles = tuple(
        StoredFigurePage(
            key=f"roi_{i}", label=None, backend="mpl", metadata={},
            files=(StoredFigureFile("png", "image/png", f"roi_{i}.png", _PNG + bytes([i])),),
            plot="tiles", directory="tiles",
        )
        for i in range(2)
    )
    delta = StoredFigurePage(
        key="delta_e", label=None, backend="mpl", metadata={},
        files=(StoredFigureFile("png", "image/png", "delta_e.png", _PNG + b"d"),),
        plot="delta_e", directory="delta_e",
    )
    return StoredFigures(
        run=run,
        bindings=(StoredFigureBinding("cal", "CalibrateColorRpcc", "cal", (*tiles, delta)),),
        failed=(StoredFigureFailure("cal", "roi_2", "png", "OSError: x", plot="tiles"),),
    )


def test_writer_lays_out_plot_folders_with_group_documents(tmp_path: Path):
    fragment = write_image_figures(tmp_path, _stored_v2())
    group = {"zarr_format": 3, "node_type": "group", "attributes": {}}
    for level in ("cal", "cal/tiles", "cal/delta_e"):
        document = tmp_path / "figures" / _RUN.run_id / level / "zarr.json"
        assert json.loads(document.read_text(encoding="utf-8")) == group
    assert (tmp_path / f"figures/{_RUN.run_id}/cal/tiles/roi_1.png").read_bytes() == _PNG + b"\x01"
    descriptor = fragment[ngff_.PhenotypicAttr.FIGURES]
    assert descriptor["schema_version"] == 2
    run = descriptor["runs"][_RUN.run_id]
    pages = run["bindings"]["cal"]["pages"]
    assert [(p["plot"], p["key"]) for p in pages] == [
        ("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e"),
    ]
    assert pages[0]["files"][0]["path"] == f"figures/{_RUN.run_id}/cal/tiles/roi_0.png"
    assert run["failed"] == [
        {"binding": "cal", "page": "roi_2", "plot": "tiles", "format": "png", "error": "OSError: x"}
    ]


def test_a_flat_page_is_written_flat_with_a_null_plot(tmp_path: Path):
    fragment = write_image_figures(tmp_path, _stored())
    page = fragment["figures"]["runs"][_RUN.run_id]["bindings"]["sym"]["pages"][0]
    assert page["plot"] is None
    assert page["files"][1]["path"] == f"figures/{_RUN.run_id}/sym/default.png"


@pytest.mark.parametrize("path, expected", [
    ("figures/r/b/f.png", ("r", "b", None, "f.png")),
    ("figures/r/b/p/f.png", ("r", "b", "p", "f.png")),
])
def test_split_accepts_both_layouts(path, expected):
    assert split_figure_file_path(path) == expected


@pytest.mark.parametrize("path", [
    "figures/r/f.png", "figures/r/b/p/q/f.png", "figures/r/../p/f.png",
    "figures/r/b/./f.png", "/figures/r/b/f.png", "tables/r/b/f.png",
])
def test_split_refuses_every_other_shape(path):
    with pytest.raises(ValueError):
        split_figure_file_path(path)


def test_figure_file_path_round_trips_both_layouts():
    from phenotypic.sdk_._image_figures import figure_file_path

    for plot in (None, "tiles"):
        path = figure_file_path("r", "b", "f.png", plot)
        assert split_figure_file_path(path) == ("r", "b", plot, "f.png")


def test_a_v1_store_gains_a_v2_run_and_keeps_its_v1_run_byte_for_byte(tmp_path: Path):
    store = _store_with(tmp_path, _stored(_OTHER))
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    root["attributes"]["phenotypic"]["figures"]["schema_version"] = 1   # as 0.19 wrote it
    old_entry = dict(root["attributes"]["phenotypic"]["figures"]["runs"][_OTHER.run_id])
    old_bytes = (store / f"figures/{_OTHER.run_id}/sym/default.png").read_bytes()

    part = tmp_path / "p.ome.zarr.part"
    part.mkdir()
    carried = carry_figure_runs(store, part, exclude=_RUN.run_id)
    phenotypic = {"figures": root["attributes"]["phenotypic"]["figures"]}
    apply_image_figures_attributes(phenotypic, carried)
    apply_image_figures_attributes(phenotypic, write_image_figures(part, _stored_v2()))

    assert phenotypic["figures"]["schema_version"] == 2
    assert phenotypic["figures"]["runs"][_OTHER.run_id] == old_entry
    assert (part / f"figures/{_OTHER.run_id}/sym/default.png").read_bytes() == old_bytes
    assert (part / f"figures/{_RUN.run_id}/cal/tiles/roi_0.png").is_file()


def test_read_figure_run_reads_v1_and_v2(tmp_path: Path):
    store = _store_with(tmp_path, _stored())
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    for version in (1, 2):
        root["attributes"]["phenotypic"]["figures"]["schema_version"] = version
        (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
        assert read_figure_run(store, _RUN.run_id)["date"] == "2026-09-22"


def test_a_v1_only_writer_adds_no_run_to_a_v2_store(monkeypatch):
    """Review Focus 3: an older PhenoTypic leaves a v2 descriptor alone."""
    v2 = {"schema_version": 2, "runs": {"a": {"n": 1}}}
    phenotypic = {"figures": json.loads(json.dumps(v2))}
    monkeypatch.setattr(ngff_, "FIGURES_SCHEMA_VERSION", 1)
    monkeypatch.setattr(ngff_, "READABLE_FIGURES_SCHEMA_VERSIONS", frozenset({1}))
    apply_image_figures_attributes(
        phenotypic, {"figures": {"schema_version": 1, "runs": {"b": {"n": 2}}}}
    )
    assert phenotypic == {"figures": v2}


def test_a_boolean_schema_version_is_not_a_known_one():
    from phenotypic.sdk_._image_figures import known_figures_schema

    assert known_figures_schema({"schema_version": True}) is False


def test_carry_links_a_v2_run_into_its_plot_folders(tmp_path: Path):
    store = _store_with(tmp_path, _stored_v2(_OTHER))
    part = tmp_path / "p.ome.zarr.part"
    part.mkdir()
    carry_figure_runs(store, part, exclude=_RUN.run_id)
    for name in ("tiles/roi_0.png", "tiles/roi_1.png", "delta_e/delta_e.png"):
        carried = part / f"figures/{_OTHER.run_id}/cal/{name}"
        assert carried.read_bytes() == (store / f"figures/{_OTHER.run_id}/cal/{name}").read_bytes()
    assert (part / f"figures/{_OTHER.run_id}/cal/tiles/zarr.json").is_file()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest -o addopts="" -p no:cacheprovider tests/unit/sdk_/test_image_figures.py -q`
Expected: FAIL. The new tests fail with `TypeError: ... unexpected keyword argument 'plot'` and a `split_figure_file_path` unpacking error.

- [ ] **Step 3: Implement**

`src/phenotypic/sdk_/ngff_.py`, replacing line 102:

```python
#: The figures descriptor version this writer writes (spec 2026-09-30 §2):
#: version 2 adds a plot folder per page and ``plot`` on pages and failures.
FIGURES_SCHEMA_VERSION: Final[int] = 2
#: Versions this reader understands. Version 1 (flat pages) is a subset of 2.
READABLE_FIGURES_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({1, 2})
```

`src/phenotypic/sdk_/_image_figures.py`, with the dataclasses gaining their trailing fields:

```python
@dataclass(frozen=True)
class StoredFigurePage:
    """One page and the renderings that succeeded for it.

    ``metadata`` is strict, key-sorted JSON by the time it gets here: the
    builder round-trips it and refuses a page it cannot (spec §1).
    ``plot`` is the logical plot name and ``directory`` its sanitized folder
    (spec 2026-09-30 §2); both ``None`` is a flat page, as a version 1 run
    stored it.
    """

    key: str
    label: str | None
    backend: str
    metadata: Mapping[str, Any]
    files: tuple[StoredFigureFile, ...]
    plot: str | None = None
    directory: str | None = None


@dataclass(frozen=True)
class StoredFigureFailure:
    """A failure at the finest level available (spec §1)."""

    binding: str
    page: str | None
    format: str | None
    error: str
    plot: str | None = None
```

Path helpers:

```python
def figure_file_path(
    run_id: str, directory: str, filename: str, plot_directory: str | None = None
) -> str:
    """Store-relative path of one figure file (spec §1a; 2026-09-30 §2)."""
    from . import ngff_

    middle = f"{directory}/{plot_directory}" if plot_directory is not None else directory
    return f"{ngff_.FIGURES_GROUP}/{run_id}/{middle}/{filename}"


def split_figure_file_path(path: str) -> tuple[str, str, str | None, str]:
    """Invert :func:`figure_file_path`: ``(run_id, directory, plot_directory, filename)``.

    ``plot_directory`` is ``None`` for a flat (version 1) page.

    Raises:
        ValueError: If *path* is not laid out as this writer lays it out --
            which also refuses ``..``, ``.`` and absolute paths.
    """
    from . import ngff_

    raw = str(path).split("/")
    parts = PurePosixPath(path).parts
    if (
        len(parts) not in (4, 5)
        or len(raw) != len(parts)           # PurePosixPath drops "." and "" components
        or parts[0] != ngff_.FIGURES_GROUP  # also refuses an absolute path, whose parts[0] is "/"
        or any(part in {".", ".."} for part in parts)
    ):
        raise ValueError(
            f"figure path {path!r} is not "
            f"{ngff_.FIGURES_GROUP}/<run>/<binding>/[<plot>/]<file>"
        )
    if len(parts) == 4:
        return parts[1], parts[2], None, parts[3]
    return parts[1], parts[2], parts[3], parts[4]
```


In `write_image_figures`, replace the per-page file loop and the page dict:

```python
        for page in binding.pages:
            page_directory = (
                directory if page.directory is None else directory / page.directory
            )
            if page.directory is not None:
                _ensure_group(page_directory)
            entries = []
            for stored in page.files:
                target = ngff_.long_path(page_directory / stored.filename)
                try:
                    os.unlink(target)
                except FileNotFoundError:
                    pass
                with open(target, "xb") as handle:
                    handle.write(stored.data)
                entries.append({
                    "format": stored.format,
                    "media_type": stored.media_type,
                    "path": figure_file_path(
                        run_id, binding.directory, stored.filename, page.directory
                    ),
                    "sha256": hashlib.sha256(stored.data).hexdigest(),
                })
            pages.append({
                "key": page.key,
                "plot": page.plot,
                "label": page.label,
                "backend": page.backend,
                "metadata": dict(page.metadata),
                "files": entries,
            })
```

Keep the two comment lines above `os.unlink`. The failure list:

```python
        "failed": [
            {"binding": f.binding, "page": f.page, "plot": f.plot,
             "format": f.format, "error": f.error}
            for f in figures.failed
        ],
```

`_carry_file`:

```python
        file_run, directory, plot_directory, _filename = split_figure_file_path(stored["path"])
        …
        _ensure_group(store_part / ngff_.FIGURES_GROUP / run_id / directory)
        if plot_directory is not None:
            _ensure_group(store_part / ngff_.FIGURES_GROUP / run_id / directory / plot_directory)
        _link_or_copy(source, store_part / stored["path"])
```

The version gate (`known_figures_schema` and `read_figure_run`):

```python
def _readable_version(version: object) -> bool:
    from . import ngff_

    return type(version) is int and version in ngff_.READABLE_FIGURES_SCHEMA_VERSIONS


def known_figures_schema(descriptor: object) -> bool:
    """…(docstring unchanged; "knows" now means READABLE_FIGURES_SCHEMA_VERSIONS)…"""
    return not isinstance(descriptor, Mapping) or _readable_version(
        descriptor.get("schema_version")
    )
```

In `read_figure_run`, replace the version check:

```python
    version = descriptor.get("schema_version")
    if not _readable_version(version):
        raise ValueError(
            f"figures schema_version {version!r} is not supported "
            f"(this reader knows {sorted(ngff_.READABLE_FIGURES_SCHEMA_VERSIONS)})"
        )
```

`carry_figure_runs` and `apply_image_figures_attributes` need no change. `_fragment` stamps `FIGURES_SCHEMA_VERSION` (now `2`) on carried and new runs alike, which is what upgrades a version 1 descriptor.

- [ ] **Step 4: Update the existing tests that pinned version 1 or the 3-tuple**

In `tests/unit/sdk_/test_image_figures.py`:
- `test_writer_lays_out_one_run_folder_and_a_hash_bound_descriptor`: `assert descriptor["schema_version"] == 2`; the split assertion becomes `== (_RUN.run_id, "sym", None, Path(entry["path"]).name)`; the failure dict gains `"plot": None`.
- `test_apply_never_merges_with_an_unknown_schema`: `newer = {"schema_version": 3, "layout": {"x": 1}}`.
- `test_read_figure_run_picks_one_run_and_refuses_an_unknown_schema`: set `schema_version` to `3` and match `"schema_version 3"`.

Known hits outside that file (plan review I3). Use `3` for "a newer, unknown version" from now on:
- `tests/unit/sdk_/test_image_figures_store.py:260`: `_relabel_as_newer` sets `["schema_version"] = 2` as the unknown layout. Change it to `3`. It is used by the tests at `:282` and `:295`.
- `tests/unit/plotting/test_store_copyout.py:341` (`d.update(schema_version=2)`) and `:346` (`"schema_version 2"`): change both to `3`.

Then grep **every** form, not just literals, and update each hit the same way (`2` → `3` where it means "unknown"; `1` → `2` where it means "what this writer writes"). Leave deliverables-manifest versions for Tasks 4 and 5:

```bash
grep -rn 'schema_version' tests/unit/sdk_ tests/unit/plotting tests/unit/cli tests/integration
```

Make the version 1 fixture real (minor 6). In `test_a_v1_store_gains_a_v2_run_and_keeps_its_v1_run_byte_for_byte`, after setting `schema_version = 1`, strip the keys 0.19 never wrote, so the carried entry has the true version 1 shape:

```python
    v1_run = root["attributes"]["phenotypic"]["figures"]["runs"][_OTHER.run_id]
    for binding in v1_run["bindings"].values():
        for page in binding["pages"]:
            page.pop("plot")
    for failure in v1_run["failed"]:
        failure.pop("plot")
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    old_entry = json.loads(json.dumps(v1_run))
```

Then build `phenotypic` from that same `root` (move the `old_entry` line after the strip, replacing the earlier one).

Warning wording (minor 7): the three "this writer knows %r" warnings (`_image_figures.py:368, 493`, and `_measurement_tables.py:790`) print `sorted(ngff_.READABLE_FIGURES_SCHEMA_VERSIONS)` instead of `FIGURES_SCHEMA_VERSION`.

- [ ] **Step 5: Run the tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest -o addopts="" -p no:cacheprovider tests/unit/sdk_/test_image_figures.py tests/unit/sdk_/test_image_figures_store.py -q`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/ngff_.py src/phenotypic/sdk_/_image_figures.py tests/unit/sdk_/test_image_figures.py
git add src/phenotypic/sdk_/ngff_.py src/phenotypic/sdk_/_image_figures.py tests/unit/sdk_/
git commit -m "feat(store): figures descriptor v2 with plot folders; read v1 and v2"
```

---

### Task 3: Build pages into plot folders; keep version 1 and version 2 bindings

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_writer.py` (add `plot_page_paths` beside `unique_page_stems`, and export it in `__all__`)
- Modify: `src/phenotypic/plotting/_pipeline/_store_figures.py` (`_build_pages`, `_build_page`, `_SameRunKeeper.keep`, `_kept_binding`, imports)
- Test: `tests/unit/plotting/test_store_figures_build.py`

**Interfaces:**
- Consumes: Task 1 `PlotPage.plot_name`; Task 2 `StoredFigurePage(…, plot=, directory=)`, `StoredFigureFailure(…, plot=)`, and `split_figure_file_path -> (run, dir, plot_dir | None, file)`.
- Produces: `plot_page_paths(pages: Sequence[tuple[str, str, str]]) -> list[tuple[str, str]]`. It maps `(plot_name, key, preferred_name)` per page to `(plot_directory, file_stem)` per page. Tasks 4 and 5 use it.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/plotting/test_store_figures_build.py`, which already defines `_build`, `_keep`, `_files`, `ApplyState` and imports `TEST_RUN`)

Add these to the file's existing module-level imports. `json` and `re` are already imported there; do not import them twice (minor 14):

```python
from phenotypic.plotting._pipeline._writer import plot_page_paths
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_figure_run,
    write_image_figures,
)
from tests.unit.plotting._store_fixtures import figure_store


class PerRoiPages(BaseModel, PlotImage):
    """Two `tiles` pages and a bare `delta_e`, like the calibration overlay."""

    mode: str = "draw"

    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        if self.mode == "gone":
            raise FigureInputUnavailable("the as-shot pixels are gone")

        def fig(y):
            f = Figure()
            f.subplots().plot([0, y])
            return f

        return PlotOutput(pages=(
            PlotPage(key="roi_0", plot="tiles", figure=fig(1)),
            PlotPage(key="roi_1", plot="tiles", figure=fig(2)),
            PlotPage(key="delta_e", figure=fig(3)),
        ))


def test_plot_page_paths_groups_pages_by_plot():
    assert plot_page_paths([
        ("tiles", "roi_0", "roi_0"), ("tiles", "roi_1", "roi_1"), ("delta_e", "delta_e", "delta_e"),
    ]) == [("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e")]


def test_plot_folders_that_collide_get_distinct_names():
    """Review Focus 1: case-folded collisions get the stable digest suffix."""
    paths = plot_page_paths([("Tiles", "a", "a"), ("tiles", "a", "a")])
    folders = [folder for folder, _stem in paths]
    assert folders[0] == "Tiles"
    assert re.fullmatch(r"tiles-[0-9a-f]{8}", folders[1])
    assert [stem for _folder, stem in paths] == ["a", "a"]       # one per folder, no clash
    assert paths == plot_page_paths([("Tiles", "a", "a"), ("tiles", "a", "a")])  # stable


def test_reserved_names_never_become_plot_folders():
    """Minor 1: a plot named like the group document or the manifest."""
    folders = [folder for folder, _stem in plot_page_paths([
        ("zarr.json", "a", "a"), ("Manifest.JSON", "b", "b"),
    ])]
    assert all(re.fullmatch(r"(zarr|Manifest)\.(json|JSON)-[0-9a-f]{8}", f) for f in folders)


def test_two_pages_of_one_plot_whose_stems_collide_get_distinct_files():
    """I2: the per-plot file pass, not only the folder pass, is collision-safe."""
    paths = plot_page_paths([("tiles", "A b", "A b"), ("tiles", "a-b", "a-b")])
    assert paths[0] == ("tiles", "A-b")
    assert paths[1][0] == "tiles" and re.fullmatch(r"a-b-[0-9a-f]{8}", paths[1][1])


def test_a_multi_plot_output_is_built_into_plot_folders():
    [binding] = _build(PerRoiPages()).bindings
    assert [(p.plot, p.directory, p.key, p.files[0].filename) for p in binding.pages] == [
        ("tiles", "tiles", "roi_0", "roi_0.png"),
        ("tiles", "tiles", "roi_1", "roi_1.png"),
        ("delta_e", "delta_e", "delta_e", "delta_e.png"),
    ]


def test_a_foldered_v2_binding_is_kept_byte_identical(tmp_path):
    first = _build(PerRoiPages())
    store = figure_store(tmp_path / "first", first)
    kept = _keep(store, PerRoiPages(mode="gone"))
    assert kept == first
    assert _files(figure_store(tmp_path / "again", kept)) == _files(store)


def _flat_v1_store(tmp_path, *, plot_claimed):
    """A run folder an older PhenoTypic wrote: flat page, no `plot`, version 1."""
    flat = StoredFigures(TEST_RUN, (StoredFigureBinding(
        "PerRoiPages", "PerRoiPages", "PerRoiPages",
        (StoredFigurePage("tiles", None, "mpl", {},
                          (StoredFigureFile("png", "image/png", "tiles.png", b"\x89PNG v1"),)),),
    ),), ())
    store = figure_store(tmp_path / "v1", flat)
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    figures = root["attributes"]["phenotypic"]["figures"]
    figures["schema_version"] = 1
    [page] = figures["runs"][TEST_RUN.run_id]["bindings"]["PerRoiPages"]["pages"]
    if plot_claimed is None:
        page.pop("plot")                      # exactly as 0.19 wrote it
    else:
        page["plot"] = plot_claimed           # a v2 claim over a flat path
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    return store


def test_a_kept_flat_page_is_written_flat_with_a_null_plot(tmp_path):
    """I6 / P3: the null plot survives the write and the file stays flat."""
    kept = _keep(_flat_v1_store(tmp_path, plot_claimed=None), PerRoiPages(mode="gone"))
    again = figure_store(tmp_path / "again", kept)
    [page] = read_figure_run(again, TEST_RUN.run_id)["bindings"]["PerRoiPages"]["pages"]
    assert page["plot"] is None
    assert page["files"][0]["path"] == f"figures/{TEST_RUN.run_id}/PerRoiPages/tiles.png"


def test_a_binding_spread_over_two_binding_folders_is_refused(tmp_path):
    """I6: the single-binding-folder check still holds with plot folders."""
    store = figure_store(tmp_path / "s", _build(PerRoiPages()))
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    pages = root["attributes"]["phenotypic"]["figures"]["runs"][TEST_RUN.run_id]["bindings"]["PerRoiPages"]["pages"]
    moved = pages[2]["files"][0]
    source = store / moved["path"]
    moved["path"] = moved["path"].replace("/PerRoiPages/", "/Elsewhere/")
    (store / moved["path"]).parent.mkdir(parents=True)
    source.rename(store / moved["path"])
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    kept = _keep(store, PerRoiPages(mode="gone"))
    assert kept.bindings == ()
    assert "one directory" in kept.failed[0].error


def test_a_page_failure_records_its_plot_through_the_descriptor(tmp_path):
    """I6 / Review Focus 5: the builder stamps `plot` on page failures."""
    stored = _build(HandBuiltPages())
    by_page = {f.page: f for f in stored.failed}
    assert by_page["odd"].plot == "odd" and by_page["np"].plot == "np"
    (tmp_path / "s").mkdir()
    fragment = write_image_figures(tmp_path / "s", stored)
    failed = fragment["figures"]["runs"][TEST_RUN.run_id]["failed"]
    assert {(f["page"], f["plot"]) for f in failed} >= {("odd", "odd"), ("np", "np")}


def test_a_flat_v1_binding_is_kept_flat(tmp_path):
    """Review Focus 2: an older PhenoTypic wrote this run's folder flat."""
    kept = _keep(_flat_v1_store(tmp_path, plot_claimed=None), PerRoiPages(mode="gone"))
    [page] = kept.bindings[0].pages
    assert (page.plot, page.directory, page.files[0].filename) == (None, None, "tiles.png")


def test_a_page_whose_plot_disagrees_with_its_path_is_refused(tmp_path):
    kept = _keep(_flat_v1_store(tmp_path, plot_claimed="tiles"), PerRoiPages(mode="gone"))
    assert kept.bindings == ()
    [failure] = kept.failed
    assert (failure.binding, failure.page) == ("PerRoiPages", None)
    assert "not laid out" in failure.error
```

Update the one existing test this changes:
- `test_hand_built_pages_use_backend_defaults_and_collision_safe_names`: its bare pages `"A b"` and `"a-b"` are now each their own plot (D2). So the collision digest moves from the **file** to the **folder**.
- Assert by `(directory, filename)`:
  ```python
      where = {page.key: (page.directory, [f.filename for f in page.files]) for page in binding.pages}
      assert where["A b"] == ("A-b", ["A-b.plotly.json"])
      folder, [mpl_name] = where["a-b"]
      assert re.fullmatch(r"a-b-[0-9a-f]{8}", folder) and mpl_name == "a-b.png"
  ```
- Its failure assertions are unchanged.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/plotting/test_store_figures_build.py -q`
Expected: FAIL with `ImportError: cannot import name 'plot_page_paths'`.

- [ ] **Step 3: Implement `plot_page_paths`** (in `_writer.py`, directly after `unique_page_stems`)

```python
def plot_page_paths(pages: Sequence[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """Return ``(plot_directory, file_stem)`` per page (spec 2026-09-30 §2).

    Plot folders are unique within the binding and file stems unique within
    their folder, both by :func:`unique_page_stems`, so a collision gets the
    same stable digest suffix it always has.

    Args:
        pages: ``(plot_name, page_key, preferred_name)`` per page, in page
            order. The store passes the key as the preferred name; the direct
            publisher passes the label.
    """
    plots = list(dict.fromkeys(plot for plot, _key, _preferred in pages))
    # Reserved first (spec D6), so a plot named like the group document or the
    # manifest gets the digest suffix instead of colliding with that file.
    reserved = [(name, name) for name in _RESERVED_PLOT_NAMES]
    named = unique_page_stems(reserved + [(plot, plot) for plot in plots])
    plot_directories = dict(zip(plots, named[len(reserved):]))
    stems = [""] * len(pages)
    for plot in plots:
        members = [i for i, (owner, _key, _preferred) in enumerate(pages) if owner == plot]
        named = unique_page_stems([(pages[i][1], pages[i][2]) for i in members])
        for i, stem in zip(members, named):
            stems[i] = stem
    return [(plot_directories[plot], stems[i]) for i, (plot, _key, _preferred) in enumerate(pages)]
```

Above it, add the constant:

```python
#: Plot-folder names that would collide with a file beside them: a binding's
#: Zarr group document, and the deliverables manifest (spec D6).
_RESERVED_PLOT_NAMES: tuple[str, ...] = ("zarr.json", "manifest.json")
```

`unique_page_stems` already compares case-folded, so `Zarr.JSON` is caught too. Add `"plot_page_paths"` to `_writer.py`'s `__all__`.

- [ ] **Step 4: Build into folders** (`_store_figures.py`)

Import `plot_page_paths` alongside `unique_page_stems`, and drop `unique_page_stems` if nothing else uses it. In `_build_pages`:

```python
        spec = declared_figure_spec(binding.plot)
        paths = plot_page_paths([(p.plot_name, p.key, p.key) for p in output.pages])
    …
    for page, (plot_directory, stem) in zip(output.pages, paths):
        try:
            built = _build_page(binding, page, plot_directory, stem, spec, failed)
        …
        except Exception as exc:  # noqa: BLE001 - per-page best effort
            failed.append(StoredFigureFailure(
                binding.id, page.key, None, normalize_figure_error(exc), plot=page.plot_name
            ))
```

In `_build_page(binding, page, plot_directory, stem, spec, failed)`, give the per-format failure `plot=page.plot_name`, and make the return:

```python
    return StoredFigurePage(
        key=page.key, label=page.label, backend=backend,
        metadata=metadata, files=tuple(files),
        plot=page.plot_name, directory=plot_directory,
    )
```

- [ ] **Step 5: Keep both layouts** (`_SameRunKeeper.keep` and `_kept_binding`)

In `keep`, carry `plot` on kept failures:

```python
        failures = [
            StoredFigureFailure(f["binding"], f["page"], f["format"], f["error"], f.get("plot"))
            for f in entry.get("failed", [])
            if f.get("binding") == binding.id
        ]
```

`_kept_binding`'s loop becomes:

```python
        figures_root = (self._store / FIGURES_GROUP).resolve()
        directories: set[str] = set()
        pages: list[StoredFigurePage] = []
        for page in stored["pages"]:
            files: list[StoredFigureFile] = []
            page_directories: set[str | None] = set()
            for entry in page["files"]:
                run_id, directory, plot_directory, filename = split_figure_file_path(entry["path"])
                if run_id != self._run.run_id:
                    raise ValueError(
                        f"{entry['path']!r} is not in run folder {self._run.run_id!r}"
                    )
                directories.add(directory)
                page_directories.add(plot_directory)
                data = _read_stored_file(self._store, figures_root, entry)
                files.append(StoredFigureFile(
                    entry["format"], entry["media_type"], filename, data
                ))
            plot = page.get("plot")
            if len(page_directories) > 1 or (plot is None) != (None in page_directories):
                raise ValueError(
                    f"stored page {page['key']!r} of {binding.id!r} is not laid out "
                    f"as this writer lays it out (plot {plot!r}, folders "
                    f"{sorted(map(str, page_directories))})"
                )
            pages.append(StoredFigurePage(
                key=page["key"], label=page["label"], backend=page["backend"],
                metadata=page["metadata"], files=tuple(files),
                plot=plot, directory=next(iter(page_directories), None),
            ))
        if len(directories) != 1:
            …unchanged…
```

A page with no files cannot occur here: the writer omits a page that stored nothing (spec §1), so `page_directories` is never empty. The `(plot is None) != (None in page_directories)` check is what refuses a version 2 page stored flat and a version 1 page stored in a folder.

- [ ] **Step 6: Run the tests**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/plotting/test_store_figures_build.py tests/unit/cli/test_staged_figures_keep.py tests/unit/sdk_ -q`
Expected: all pass, except assertions that pin the old flat filenames for bare pages. Each such assertion changes from `f"{key}.png"` at `<binding>/` to `<key>/<key>.png` (D2). Update each one, then rerun until everything passes.

**Sweep (I1).** Known vacuous checks in this task's files:
- `test_store_figures_build.py:505`, `test_a_drawable_binding_is_redrawn_not_kept`: the tamper loop `for png in (...ApplyState).glob("*.png")` no longer finds anything, because the PNG is now in `ApplyState/tiles/`. Use `rglob("*.png")` and assert the list is non-empty before the loop, or the test cannot tell redraw from keep.
- `:468` (`[png] = ….glob("*.png")`) fails loudly; switch it to `rglob`.

Then run the sweep grep from Global Constraints over `test_store_figures_build.py` and `test_staged_figures_keep.py`, and fix every hit by the rule.

Keep the integration tests bisectable (minor 11): in `tests/integration/cli/test_figures_in_store.py:136`, change the pinned `sym/default.plotly.json` to `sym/default/default.plotly.json` in this task.

- [ ] **Step 7: Lint, commit, phase gate A**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/_writer.py src/phenotypic/plotting/_pipeline/_store_figures.py tests/unit/plotting/test_store_figures_build.py
git add src/phenotypic/plotting/_pipeline/_writer.py src/phenotypic/plotting/_pipeline/_store_figures.py tests/unit/
git commit -m "feat(store): build figures into plot folders; keep v1 flat and v2 foldered bindings"
```

**Phase gate A:** submit the 17-file figure set as a Slurm job (the baseline job's script, `logs/pht_figures_baseline.sh` in AutoConvertRaw-GC, pointed at this worktree). Failures are expected only in copy-out (Task 4) and calibration (Task 6) assertions. List every failing test **by name**. Any failure outside `test_store_copyout.py`, `test_coordinator.py`, `test_calibration_*` and `test_publication_end_to_end.py` must be explained before going on. Also run the two out-of-set files (Global Constraints) and compare them by name with their `ba725001` results.

---

### Task 4: Deliverables copy-out mirrors the store

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_store_copyout.py` (`_publish_binding`, `_publish_pages`, the `_publish_binding` docstring)
- Test: `tests/unit/plotting/test_store_copyout.py`

**Interfaces:**
- Consumes: Task 2 `split_figure_file_path`, and the descriptor's page `"plot"` and failure `"plot"`.
- Produces: `plots/<binding>/<dataset>/<output_stem>/manifest.json` (version 3). Each page's files sit at the page's store-relative folder and name, so manifest `files` values are relative POSIX paths, for example `"tiles/roi_0.png"`.

- [ ] **Step 1: Rewrite the layout tests first**

In `test_store_copyout.py`:

```python
def test_a_single_default_page_lands_in_its_plot_folder_with_generated_html(tmp_path):
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page())))
    directory = plots / "sym" / "ds-1" / _STEM
    assert sorted(p.name for p in (directory / "default").iterdir()) == [
        "default.html", "default.plotly.json",
    ]
    html = (directory / "default" / "default.html").read_text(encoding="utf-8")
    assert 'src="../../../../plotly.min.js"' in html
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["pages"][0]["plot"] == "default"
    assert manifest["pages"][0]["files"] == {
        "plotly-json": "default/default.plotly.json", "html": "default/default.html",
    }


def test_a_failed_page_keeps_its_plot_in_the_manifest(tmp_path):
    """Review Focus 5."""
    failed = (StoredFigureFailure("sym", "roi_1", None, "TypeError: nope", plot="tiles"),)
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page(), failed=failed)))
    manifest = _manifest(plots)
    assert manifest["failed"] == [
        {"key": "roi_1", "plot": "tiles", "label": None, "error": "TypeError: nope"}
    ]
```

Replace the file's `_page` fixture, so a page is built the way Task 3's builder builds it (plot folder = cleaned plot name), or flat, as a version 1 run stored it:

```python
def _page(key="default", label=None, formats=("plotly-json",), backend="plotly",
          plot=None, flat=False):
    table = {
        "plotly-json": ("application/vnd.plotly.v1+json", ".plotly.json", _plotly_json()),
        "png": ("image/png", ".png", b"\x89PNG"),
    }
    stem = key.replace(" ", "-")
    plot = None if flat else (plot or key)
    return StoredFigurePage(
        key, label, backend, {"k": 1},
        tuple(StoredFigureFile(fmt, table[fmt][0], f"{stem}{table[fmt][1]}", table[fmt][2])
              for fmt in formats),
        plot=plot,
        directory=None if flat else plot.replace(" ", "-"),
    )
```

`test_a_flat_v1_page_is_copied_into_the_image_folder` uses `_page(flat=True)`.

Also:
- **Rename and rewrite** `test_multi_page_writes_a_directory_and_manifest_v2` as `…_manifest_v3`. Its files become key-named inside plot folders: `{"plotly-json": "first/first.plotly.json", "html": "first/first.html"}`. The label stays in the manifest's `"label"` (decision P1).
- **Delete** `test_a_failed_second_page_does_not_flip_the_layout_to_flat`. It guards the removed flat case.
- **Rewrite** `test_a_failed_lone_default_page_writes_nothing` as `test_a_failed_lone_default_page_writes_a_manifest_of_its_failure`. The binding now always publishes a manifest directory, whose `failed` lists `{"key": "default", "plot": "default", …}`.
- **Add** `test_a_flat_v1_page_is_copied_into_the_image_folder`: a store page with `plot=None, directory=None` lands at `<stem>/<file>`, and the manifest gives `"plot": null`.
- **Rewrite** `test_a_republished_page_loses_its_leftover_renderings` (I6). Seed the leftover where it now lives, `<stem>/default/default.png`, publish a plotly-json-only page, and assert `default.png` is gone **from that plot folder** while `default.plotly.json` and `default.html` are present.
- **Add** the I4 and I5 tests:

```python
def test_a_page_with_no_copyable_file_is_failed_with_its_plot(tmp_path):
    """I4: the "no stored file could be copied out" entry carries `plot`."""
    store = figure_store(tmp_path / "s", _one(_page("roi_0", plot="tiles")))
    (store / run_path("sym/tiles/roi_0.plotly.json")).write_bytes(b"tampered")
    manifest = _manifest(_publish(tmp_path, store))
    assert manifest["pages"] == []
    [failed] = manifest["failed"]
    assert (failed["key"], failed["plot"]) == ("roi_0", "tiles")


def test_a_refused_guard_creates_no_plot_folder(tmp_path):
    """I5: the guard is asked before a plot folder is created."""
    from phenotypic.plotting._pipeline import PlotPublicationBlocked   # as the file's sibling tests do

    calls = iter([True, True, True, False])   # read-guard, binding dir, lock, then the plot-folder mkdir
    store = figure_store(tmp_path / "s", _one(_page("roi_0", plot="tiles")))
    with pytest.raises(PlotPublicationBlocked):
        _publish(tmp_path, store, publication_guard=lambda: next(calls, False))
    image_dir = tmp_path / "deliverables" / "plots" / "sym" / "ds-1" / _STEM
    assert image_dir.is_dir()                  # the refusal came after the image folder ...
    assert not (image_dir / "tiles").exists()  # ... and before the plot folder
```

(A refused guard propagates `PlotPublicationBlocked`, as `test_a_refused_guard_propagates_before_anything_is_written` relies on. The `True` count must equal the guard calls made before the plot-folder check, which come from `publish_store_figures` and `_publish_binding`. Count them and adjust the sequence so the single `False` lands on the plot-folder check. The two closing asserts prove where it landed.)

- **Fix the vacuous refused-guard test** (I1): `test_a_refused_guard_mid_page_leaves_no_half_page` asserts `not (base / f"{_STEM}.plotly.json").exists()`, a path no layout writes any more. Assert `list((base / _STEM).rglob("*.plotly.json")) == []` instead, after first asserting that the refusal happened mid-page.
- **Pinned paths and manifests** (I3): `:73` is covered by the rename above. Update the version 2 manifest literal at `:279` to version 3 with `"plot"` on its entries. The three `run_path("sym/default.plotly.json")` uses at `:125`, `:164` and `:352` become `run_path("sym/default/default.plotly.json")`. The first two "tamper" a path that no longer exists, so without this they would silently test nothing.

- [ ] **Step 2: Run them to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/plotting/test_store_copyout.py -q`
Expected: FAIL on the rewritten layout tests.

- [ ] **Step 3: Implement**

`_publish_binding`: delete the `flat` computation and its branch, so the function always publishes the manifest directory. Keep `directory = base / output_stem` and the lock. Replace the stem computation and the call:

```python
    directory = base / output_stem
    _require_plot_publication(publication_guard)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_path_lock(directory / ".publication.lock"):
        _require_plot_publication(publication_guard)
        published, failed = _publish_pages(
            store, plots_base, directory, pages, page_failures,
            record=record,
            publication_guard=publication_guard, commit_guard=commit_guard,
        )
        failed += [
            {"key": key, "plot": plot, "label": None, "error": error}
            for (key, plot), error in failed_only.items()
        ]
        …renderers unchanged…
        _commit_manifest(directory, {
            "schema_version": 3, "plot_id": binding_id, "class": plot_class,
            "renderers": renderers, "pages": published, "failed": failed,
        }, publication_guard=publication_guard, commit_guard=commit_guard)
```

`failed_only` is keyed by `(page, plot)`:

```python
    published_ids = {(p["key"], p.get("plot")) for p in pages}
    failed_only: dict[tuple[str, str | None], str] = {}
    for failure in page_failures:
        page_id = (failure["page"], failure.get("plot"))
        if page_id not in published_ids:
            failed_only.setdefault(page_id, str(failure.get("error")))
```

Update the docstring: "Publish one binding as a manifest directory mirroring the store (spec 2026-09-30 §3)."

`_publish_pages(store, plots_base, directory, pages, page_failures, *, record, …)`. The `stems` parameter is removed, and names come from the stored paths:

```python
    for page in pages:
        files: dict[str, str] = {}
        errors: list[BaseException] = []
        for entry in page["files"]:
            try:
                data = _read_stored_file(store, figures_root, entry)
                _run, _binding, plot_directory, name = split_figure_file_path(entry["path"])
                stem = name[: -len(STORE_FORMATS[entry["format"]].extension)]
                page_directory = directory if plot_directory is None else directory / plot_directory
                if not page_directory.exists():
                    # The writer's contract: the guard is asked immediately
                    # before any directory is created (I5).
                    _require_plot_publication(publication_guard)
                    page_directory.mkdir()
                relative = name if plot_directory is None else f"{plot_directory}/{name}"
                _atomic_write(page_directory / name, lambda dest, d=data: dest.write_bytes(d),
                              publication_guard=publication_guard, commit_guard=commit_guard)
                files[entry["format"]] = relative
            …
            if entry["format"] != "plotly-json":
                continue
            try:
                html = _write_html_from_json(data, page_directory, stem, plots_base, …)
                files["html"] = html if plot_directory is None else f"{plot_directory}/{html}"
            …
```

Changes that follow from the new names:
- `_discard_page(directory, files)` already joins `directory / written`, so the relative values still resolve.
- `_remove_leftovers` now runs per page, in `page_directory`, with the page's stem and the base names of its files:
  ```python
  _remove_leftovers(page_directory, stem, {Path(v).name for v in files.values()}, …)
  ```
  `stem` and `page_directory` are set from the page's stored files (all of one page's formats share a stem and a folder), and `_remove_leftovers` runs only when `files` is non-empty, so both are always bound there.
- The manifest page entry gains `"plot": page.get("plot")`.
- The `if not files:` branch (today `_store_copyout.py:277-279`) also records the plot (I4):
  ```python
  failed.append({"key": page["key"], "plot": page.get("plot"), "label": page["label"],
                 "error": "no stored file could be copied out"})
  ```
- The per-page failure lookup matches on `(key, plot)`:
  ```python
  partial = [f["error"] for f in page_failures
             if (f.get("page"), f.get("plot")) == (page["key"], page.get("plot"))]
  ```
- `unique_page_stems` is no longer imported here. Remove it from the import list if nothing else uses it.

- [ ] **Step 4: Run the tests**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/plotting/test_store_copyout.py tests/unit/plotting/test_coordinator.py -q`
Expected: all pass. Update any remaining `test_coordinator.py` assertion that pinned the flat or label-named layout, to the mirrored layout. `:97`, `:115` and `:899` fail loudly.

**Sweep (I1).** Known vacuous checks:
- `test_coordinator.py:126-130` (`…_stable_for_reruns`): `list(dir.glob("*.png"))` is `[]` both times, so `first == second` proves nothing. Use `sorted(dir.rglob("*.png"))` and assert it is non-empty.
- `test_coordinator.py:717` and `:963` (`glob("*.png") == []`) become `rglob`.
- `test_coordinator.py:718` (`not (directory / "manifest.json").exists()`): read the test's intent; a binding directory now always has a manifest unless nothing was published, so assert whatever the test means at the new depth.

Then run the sweep grep over `test_store_copyout.py` and `test_coordinator.py`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/_store_copyout.py tests/unit/plotting/test_store_copyout.py tests/unit/plotting/test_coordinator.py
git add src/phenotypic/plotting/_pipeline/_store_copyout.py tests/unit/plotting/
git commit -m "feat(deliverables): copy-out mirrors the store's plot folders; manifest v3; no flat case"
```

---

### Task 5: The direct publisher gains plot folders (aggregate plots)

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_writer.py` (`_publish_plot_output_locked`)
- Test: `tests/unit/plotting/test_output_adapter.py`

**Interfaces:**
- Consumes: Task 1 `PlotPage.plot_name`; Task 3 `plot_page_paths`.
- Produces: an aggregate plot directory `plots/<id>/` holding `<plot>/<file>`, and `manifest.json` version 3 with `"plot"` on pages and failures, whose `files` values are relative paths. File stems keep today's label preference (decision P1).

- [ ] **Step 1: Write the failing test** (in `test_output_adapter.py`)

```python
def test_pages_land_in_plot_folders_with_a_v3_manifest(tmp_path):
    def figure():
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        return fig

    output = PlotOutput(pages=(
        PlotPage(key="a", plot="tiles", label="Tile A", figure=figure()),
        PlotPage(key="b", plot="tiles", figure=figure()),
        PlotPage(key="chart", figure=figure()),
    ))
    manifest = publish_plot_output(output, tmp_path / "agg", plot_id="agg", plots_base=tmp_path)
    assert manifest["schema_version"] == 3
    assert [(p["plot"], p["key"], p["files"]["png"]) for p in manifest["pages"]] == [
        ("tiles", "a", "tiles/Tile-A.png"),
        ("tiles", "b", "tiles/b.png"),
        ("chart", "chart", "chart/chart.png"),
    ]
    for page in manifest["pages"]:
        assert (tmp_path / "agg" / page["files"]["png"]).is_file()
```

(`safe_path_component("Tile A") == "Tile-A"`, checked at `ba725001`. `plt` is already imported by this file, and `PlotOutput`/`PlotPage` by its `phenotypic.abc_.plotting` import.)

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/plotting/test_output_adapter.py -q`
Expected: FAIL (`KeyError: 'plot'`, or `schema_version` 2).

- [ ] **Step 3: Implement** (in `_publish_plot_output_locked`)

```python
    paths = plot_page_paths(
        [(page.plot_name, page.key, page.label or page.key) for page in output.pages]
    )
    for page, (plot_directory, stem) in zip(output.pages, paths):
        page_directory = directory / plot_directory
        created = not page_directory.exists()
        if created:
            _require_plot_publication(publication_guard)   # before directory creation, per the contract
            page_directory.mkdir()
        try:
            files, errors, backend = _render_page(
                page.figure, page_directory, stem,
                plots_base=base, plot_id=plot_id,
                publication_guard=publication_guard, commit_guard=commit_guard,
            )
        …unchanged FigureAdapter.close…
        if files:
            _remove_stale_sibling(page_directory, stem, backend, files,
                                  publication_guard=publication_guard, commit_guard=commit_guard)
        …record_plot_failure unchanged…
        if not files:
            if created and not any(page_directory.iterdir()):
                page_directory.rmdir()   # minor 9: no empty plot folder for a page that rendered nothing
            failed.append({"key": page.key, "plot": page.plot_name, "label": page.label,
                           "error": …unchanged…})
            continue
        entry = {
            "key": page.key, "plot": page.plot_name, "label": page.label,
            "files": {fmt: f"{plot_directory}/{name}" for fmt, name in files.items()},
            …backend, metadata unchanged…
        }
```

Then set `"schema_version": 3` in the manifest dict. `_render_page` computes the HTML's `plotly.min.js` src from the directory it is given (`plotlyjs_src_for(page_dir, bundle)`), so passing `page_directory` keeps the relative path right one level deeper.

- [ ] **Step 4: Update the existing direct-publish assertions and run**

Every `test_output_adapter.py` assertion that reads `entry["files"]["png"]` as a bare name now gets `"<plot>/<name>"`. A bare page has plot = key, so `"default"` becomes `"default/default.png"`.

**Keep the collision tests meaningful (I2).** `test_matplotlib_pages_publish_with_collision_safe_names` (`:70-87`) and `test_hash_suffix_is_rechecked_for_page_filename_collision` (`:89-110`) use bare pages. Each bare page is now its own folder, so they would pass with `unique_page_stems` deleted. Give every page in each of them `plot="same"`, so the file names still have to be made unique within one folder, and assert the files land in `same/`.

**Stale sibling inside a plot folder (I6).** Add:

```python
def test_a_rerun_removes_the_old_rendering_inside_its_plot_folder(tmp_path, monkeypatch):
    stale = tmp_path / "agg" / "tiles" / "a.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    import plotly.graph_objects as go
    from phenotypic.plotting._pipeline import _backends
    monkeypatch.setattr(_backends, "chrome_available", lambda: False)   # plotly → html only
    publish_plot_output(PlotOutput(pages=(PlotPage(key="a", plot="tiles", figure=go.Figure()),)),
                        tmp_path / "agg", plot_id="agg", plots_base=tmp_path)
    assert not stale.exists()
    assert (tmp_path / "agg" / "tiles" / "a.html").is_file()
```

(Patch `chrome_available` where `_render_page` looks it up. Read `test_a_plotly_page_publishes_html_without_chrome` at `:205` and copy its monkeypatch target exactly.)

**Out-of-set tests (I3):**
- `tests/unit/plotting/test_plot_meas_time_series.py:334-338` pins `files["png"] == ["BY4741.png", …]` and `destination / "BY4741.png"`. The folder comes from the key, so these become `"strain-str-BY4741/BY4741.png"` (per `safe_path_component`; confirm it from the published manifest rather than hand-deriving it). Leave `:127-147`, which reads `tmp_path/"manifest.json"`, as it is.
- `tests/gui/results_viewer/test_mutation_guard.py:556-600` counts guard calls (`checks == 3`, perturbing at the third). The new plot-folder guard call shifts which operation the third call guards. Update the count so the perturbation still lands on the **PNG commit** guard, and assert that it does: the test must still prove a commit-time refusal.

**Sweep (I1):** `test_output_adapter.py:221` (`not (sym/"Only.png").exists()`) and `:242` (`not (plots_base/"sym"/"plotly.min.js").exists()`) aim at the old depth. Make the first an `rglob`. The second's intent ("no bundle per directory") now also covers `sym/<plot>/`, so assert `list((plots_base / "sym").rglob("plotly.min.js")) == []`. Then run the sweep grep over `test_output_adapter.py` and `test_plot_meas_time_series.py`.

Then run:

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/plotting -q`
Expected: all pass. Then, in the Qt env (Global Constraints): `QT_QPA_PLATFORM=offscreen uv run pytest -o addopts="" -p no:cacheprovider tests/gui/results_viewer/test_mutation_guard.py -q`. Expected: all pass.

- [ ] **Step 5: Lint, commit, phase gate B**

```bash
uv run ruff check --fix src/phenotypic/plotting/_pipeline/_writer.py tests/unit/plotting/test_output_adapter.py
git add src/phenotypic/plotting/_pipeline/_writer.py tests/unit/plotting/
git commit -m "feat(deliverables): aggregate plots publish into plot folders; manifest v3"
```

**Phase gate B:** rerun the 17-file figure set as a Slurm job. Remaining failures are expected only in `test_calibration_*` and `test_figures_in_store.py` / `test_publication_end_to_end.py`, wherever they pin calibration's two pages or a flat path. List them by name.

---

### Task 6: `CalibrateColorRpcc` stores one overlay per ROI

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` (the `plot_pages` / `inspect` method near line 372, and its docstring "Two pages")
- Test: `tests/unit/correction/test_calibration_plot_image.py`

**Interfaces:**
- Consumes: Task 1 `PlotPage(plot=…)`; `render_calibration_overlay(record)` (`_calibration_overlay.py:521`); `CalibrationOverlayRecord.rois: list[RoiOverlay]`; `RoiOverlay.roi_index: int`.
- Produces: pages `[(plot "tiles", key f"roi_{roi.roi_index}") for roi in record.rois] + [(plot "delta_e", key "delta_e")]`.

- [ ] **Step 1: Write the failing tests** (replace `test_inspect_draws_the_overlay_and_the_delta_e_chart` and `test_a_skipped_frame_still_yields_both_pages`; update `test_an_image_pipeline_listing_it_under_plots_stores_both_figures`)

```python
from phenotypic.correction._color_correction._calibration_overlay import (
    render_calibration_overlay,
)


def _ids(output) -> list[tuple[str, str]]:
    return [(page.plot_name, page.key) for page in output.pages]


def test_inspect_draws_one_overlay_per_roi_and_the_delta_e_chart():
    operation, frame = _applied()
    record = operation.calibration_record
    assert len(record.rois) == 2
    output = operation.inspect(frame, for_save=True)
    assert _ids(output) == [("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e")]
    for roi, page in zip(record.rois, output.pages):
        alone = render_calibration_overlay(record.model_copy(update={"rois": [roi]}))
        assert _png(page.figure) == _png(alone)        # exactly that ROI, nothing else
    assert _png(output.pages[2].figure) == _png(operation.show_delta_bar_plot())
    assert _png(output.pages[0].figure) != _png(output.pages[1].figure)


def test_a_skipped_frame_still_draws_every_roi():
    operation = frozen_op(on_qc_fail="skip")
    frame = Image(arr=render_frame(gain=1.6))  # saturated card
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operation.apply(frame, inplace=True)
    assert operation.calibration_record.verdict == "skipped"
    output = operation.inspect(frame)
    assert _ids(output) == [("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e")]
    (ax,) = output.pages[2].figure.axes
    assert ax.containers == []
    assert all(_png(page.figure).startswith(b"\x89PNG") for page in output.pages)


def test_one_roi_stores_one_overlay():
    """Review Focus 4."""
    # One band is 12 patches; a degree-3 fit needs 13 (`require_rank`), and that
    # fails before QC, so `on_qc_fail` cannot help. Degree 2 fits 12 (probed
    # 2026-09-30: verdict "corrected", 12/24 fitted).
    operation = frozen_op(rois=band_rois()[:1], lattice_prior=[band_prior()], degree=2)
    frame = Image(arr=render_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operation.apply(frame, inplace=True)
    assert operation.calibration_record.verdict == "corrected"
    assert _ids(operation.inspect(frame)) == [("tiles", "roi_0"), ("delta_e", "delta_e")]
```

(`band_rois` and `band_prior` come from `._checker_frames`; add them to the import.)

Keep the existing "no subject: the held image" check (I8). In `test_inspect_draws_one_overlay_per_roi_and_the_delta_e_chart`, add at the end:

```python
    held = operation.inspect()
    assert [_png(p.figure) for p in held.pages] == [_png(p.figure) for p in output.pages]
```

In `test_an_image_pipeline_listing_it_under_plots_stores_both_figures`, rename it to `…_stores_every_figure_in_its_plot_folder`, and assert:

```python
    assert [(p.directory, p.key, p.files[0].filename) for p in binding.pages] == [
        ("tiles", "roi_0", "roi_0.png"), ("tiles", "roi_1", "roi_1.png"),
        ("delta_e", "delta_e", "delta_e.png"),
    ]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/correction/test_calibration_plot_image.py -q`
Expected: FAIL. The keys are `["tiles", "delta_e"]`.

- [ ] **Step 3: Implement** (replace the `return PlotOutput(...)` at `_calibrate_color_rpcc.py:372`; import `render_calibration_overlay` where `show_tiles` already does)

```python
        # All pages or none: a failure in any page fails the whole binding,
        # which publishes nothing. One overlay per ROI (spec 2026-09-30 §4),
        # each drawn from the record with only that ROI, so the frame-level
        # title (verdict, patches fitted) stays on every one.
        record = self._calibration_record
        overlays = tuple(
            PlotPage(
                key=f"roi_{roi.roi_index}",
                plot="tiles",
                figure=render_calibration_overlay(record.model_copy(update={"rois": [roi]})),
                label=f"Tile overlay, ROI {roi.roi_index}",
            )
            for roi in record.rois
        )
        return PlotOutput(pages=(
            *overlays,
            PlotPage(key="delta_e", figure=self.show_delta_bar_plot(),
                     label="Delta E00 before and after"),
        ))
```

Update the docstring's "Returns" section: "One `tiles` page per ROI (`roi_<index>`), each an overlay of that ROI alone, then `delta_e`, what :meth:`show_delta_bar_plot` returns. All are always present, so every image stores the same pages." `show_tiles()` is unchanged.

- [ ] **Step 4: Run the tests**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -p no:cacheprovider tests/unit/correction -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py tests/unit/correction/test_calibration_plot_image.py
git add src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py tests/unit/correction/
git commit -m "feat(correction): CalibrateColorRpcc stores one tile overlay per ROI"
```

---

### Task 7: End to end and determinism

**Files:**
- Modify: `tests/integration/cli/test_calibration_figure_in_store.py` (`PAGES`, `_overlay`, `_deliverable`, and the two-page assertions at lines 128-136, 154 and 237)
- Modify: `tests/integration/cli/test_figures_in_store.py`, `tests/integration/plotting/test_publication_end_to_end.py`: only the assertions that pin a flat path
- Test: the same files, plus a new determinism test in `test_calibration_figure_in_store.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Update the expectations**

```python
#: Each page's (plot, key), and the file it is copied out to (spec 2026-09-30, D4 mirror).
PAGES = {
    ("tiles", "roi_0"): "tiles/roi_0.png",
    ("tiles", "roi_1"): "tiles/roi_1.png",
    ("delta_e", "delta_e"): "delta_e/delta_e.png",
}


def _overlay(store: Path, run_id: str) -> tuple[dict, dict[tuple[str, str], bytes]]:
    entry = _runs(store)[run_id]["bindings"]["cal"]
    pngs = {}
    for page in entry["pages"]:
        [stored] = page["files"]
        pngs[(page["plot"], page["key"])] = (store / stored["path"]).read_bytes()
    return entry, pngs


def _deliverable(out: Path) -> dict[tuple[str, str], bytes]:
    [directory] = (out / "deliverables" / "plots" / "cal" / "ds").glob("plate-*")
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["schema_version"] == 3
    return {page: (directory / name).read_bytes() for page, name in PAGES.items()}
```

In `test_full_mode_stores_the_overlay_png_and_copies_it_out`:

```python
    assert [(p["plot"], p["key"], p["backend"]) for p in entry["pages"]] == [
        ("tiles", "roi_0", "mpl"), ("tiles", "roi_1", "mpl"), ("delta_e", "delta_e", "mpl"),
    ]
    for page in entry["pages"]:
        assert [(f["format"], f["path"]) for f in page["files"]] == [
            ("png", f"figures/{run_id}/cal/{page['plot']}/{page['key']}.png")
        ]
    assert _deliverable(out) == data
```

Other assertions to update:
- `test_process_mode_zarr_stores_the_overlay`: `assert list(data) == list(PAGES)`.
- Line 237: `assert list(stage1[1]) == list(PAGES)`.
- The measure-mode keep test: its assertions compare page dicts and bytes from `_overlay`, so they follow the new keys. Read it and confirm it still asserts the kept bytes are byte-identical.

(`_write_inputs` builds a pipeline with this test file's two-band ROIs. Read the file header to confirm two ROIs, and adjust `PAGES` if the fixture uses a different count.)

- [ ] **Step 2: Pin process mode: figures in the store, nothing in deliverables** (spec §3 "by mode")

Extend the existing `test_process_mode_zarr_stores_the_overlay`. Do not add a third test: `test_figures_in_store.py::test_process_mode_carries_figures_only_in_a_store` already pins "no deliverables" for zarr and tiff (minor 10). Append:

```python
    for plot, key in PAGES:
        assert (store / f"figures/{run_id}/cal/{plot}/{key}.png").is_file()
    assert not (out / "deliverables").exists()
```

It must pass on top of Tasks 1-6. It pins existing behaviour: process mode never had a deliverables tree.

**Sweep (I1)** in `tests/integration/plotting/test_publication_end_to_end.py`:
- `:135` (`glob("*.png") == []`) becomes `rglob`.
- `:139` (`not (directory / "manifest.json").exists()`): restate it at the new depth by its intent.
- The `src="../../plotly.min.js"` near `:108` becomes `../../../../plotly.min.js` for an image figure, which is the copy-out depth. The lines above these fail loudly; these two would not.

- [ ] **Step 3: Add the determinism test** (process mode, same UTC day, two output folders)

```python
def test_two_same_day_process_runs_write_byte_identical_stores(tmp_path):
    from phenotypic._cli._cli_process_only import process_single_apply_only_core

    image, pipeline = _write_inputs(tmp_path)
    stores = []
    for name in ("a", "b"):
        out = tmp_path / name
        process_single_apply_only_core(
            pipeline_path=pipeline, image_path=image, input_root=image.parent,
            output_dir=out, image_type="Image", layer="rgb", read_kwargs={},
            process_format="zarr", run_initiation=RunInitiation(DAY, f"{DAY}T12:00:00.000Z", 7),
        )
        stores.append(out / "plate.ome.zarr")

    def tree(store: Path) -> dict[str, bytes]:
        return {p.relative_to(store).as_posix(): p.read_bytes()
                for p in sorted(store.rglob("*")) if p.is_file()}

    first, second = (tree(s) for s in stores)
    assert first == second
    assert any(path.startswith("figures/") and "/cal/tiles/roi_1.png" in path for path in first)
```

- [ ] **Step 4: Run the integration files**

Run: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -o addopts="" -m "not slow" -p no:cacheprovider tests/integration/cli/test_calibration_figure_in_store.py tests/integration/cli/test_figures_in_store.py tests/integration/plotting/test_publication_end_to_end.py -q`
Expected: all pass. Update any flat-path assertion in the latter two files to `<binding>/<key>/<key>.<ext>`.

- [ ] **Step 5: Commit, phase gate C**

```bash
uv run ruff check --fix tests/integration/cli/test_calibration_figure_in_store.py tests/integration/cli/test_figures_in_store.py tests/integration/plotting/test_publication_end_to_end.py
git add tests/integration/
git commit -m "test(figures): end-to-end per-ROI overlays in plot folders; same-day byte identity"
```

**Phase gate C:** the 17-file figure set as a Slurm job must give **0 failures**. Quote the count and the tree SHA.

---

### Task 8: Documentation, then the full regression

**Files:**
- Modify: `docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md` (notes at `### Layout` and `### §1a`)
- Modify: `CLAUDE.md` and `AGENTS.md`, line 128 in both (they are kept identical)
- Modify: `docs/source/extending/pages/custom_plotter.md` (the `PlotPage` example near line 178, and the deliverables layout table, lines 227-229)

- [ ] **Step 1: Superseded notes in the figures spec.** Directly under `### Layout`, and under the `### §1a` heading:

```markdown
> **Superseded in part, 2026-09-30** (`docs/superpowers/specs/2026-09-30-plot-subfolders/design.md`):
> runs written by the release that ships `feat/plot-subfolders` add one level,
> `<binding>/<plot>/<file>`, and the descriptor is `schema_version` 2. Version 1
> runs keep the layout below and remain readable.
```

- [ ] **Step 2: `CLAUDE.md` / `AGENTS.md` line 128.** Replace "`figures/<run>/` (revision 3)" with "`figures/<run>/<binding>/<plot>/<file>` (revision 3; descriptor `schema_version` 2 since `feat/plot-subfolders`)". Apply the identical edit to both files.

- [ ] **Step 3: `custom_plotter.md`.** After the `PlotPage(key="area", …)` example, add:

````markdown
A page can name the plot it belongs to with `plot=`. Pages sharing a plot are
stored in one folder, one file per page, while a page without `plot=` is a
plot of its own:

```python
PlotOutput(pages=(
    PlotPage(key="roi_0", plot="tiles", figure=overlay_0),
    PlotPage(key="roi_1", plot="tiles", figure=overlay_1),
    PlotPage(key="delta_e", figure=chart),            # stored at delta_e/delta_e
))
```
````

Fix the statements the new layout makes false (I7):
- the sentence just above line 235, "A page's filename comes from its `label`, or its `key`". It now applies to aggregate plots only. Image figures copied out from the store are named by **key** (decision P1); say both.
- `:235`: `PlotColonyArea` publishes `plots/PlotColonyArea/default/default.html`.
- `:332-333`: it stores `PlotColonySizes/default/default.plotly.json`.
- the manifest remarks around `:395` and `:421`: manifest `schema_version` 3, and `files` values are paths relative to the manifest, including the plot folder.

Then replace the layout table rows at lines 227-229 with:

```markdown
| `PlotImage` | `<dataset>/<stem>-<hash>/`, holding one folder per plot (`<plot>/<key>.<ext>`, plus `.html` for a stored `plotly-json`) and a `manifest.json` (`schema_version` 3) |
| `PlotMeas`, `PlotAnalysis`, `PlotQc` | `plots/<id>/`, holding one folder per plot (`<plot>/<label or key>.<ext>`) and a `manifest.json` (`schema_version` 3) |
```

- [ ] **Step 3b: The release note goes in the PR description (spec D7; no changelog file).** Draft it now in the PR body, from spec §6's list.

- [ ] **Step 4: Commit the docs**

```bash
git add docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md CLAUDE.md AGENTS.md docs/source/extending/pages/custom_plotter.md
git commit -m "docs(figures): plot subfolders, descriptor v2, manifest v3"
```

- [ ] **Step 5: Full sharded regression, once**

Submit `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch` against this worktree; read its header for how it takes the tree, and use the `slurm-job` procedure. Compare the failing test names with a baseline run of the same script at `ba725001`, by **name**, not count. Any test that fails here but not at `ba725001` must be fixed or explained before the PR.

- [ ] **Step 6: Mypy on the touched modules**

Run: `uv run mypy src/phenotypic/abc_/plotting/_output.py src/phenotypic/sdk_/_image_figures.py src/phenotypic/plotting/_pipeline/_writer.py src/phenotypic/plotting/_pipeline/_store_figures.py src/phenotypic/plotting/_pipeline/_store_copyout.py src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py`
Expected: no new errors relative to `ba725001` (compare the error list).

---

## Spec coverage (self-review)

| Spec | Task |
|---|---|
| §1 `PlotPage.plot`, default = key, uniqueness per `(plot, key)`, field not key convention | 1 |
| §2 layout, group documents, names by `unique_page_stems`, descriptor version 2, `plot` on pages and failures, version 1 read as flat, version 1 → 2 upgrade, older-writer safety, path helpers, determinism | 2, 3, 7 |
| §3 store readers (`_kept_binding`, carry; clear and `latest_run_date` unchanged) | 2, 3 |
| §3 by mode: process mode stores figures, writes no deliverables | 7 |
| §3 deliverables mirror, manifest version 3, flat case removed, both writers, version 1 page copied flat, `_remove_stale_sibling` scoped per folder and not sweeping | 4, 5 |
| §4 per-ROI overlays, `roi_<index>`, `delta_e` unchanged, refused frame draws all, `show_tiles` unchanged | 6 |
| §5 testing matrix | 1-7 |
| §6 documentation; release note in the PR description (D7) | 8 |
| D6 `plot` refuses `/`; reserved folder names | 1, 3 |
| Plan review I1-I8 | sweeps in 3, 4, 5, 7; I2 in 3, 5; I3 in 2, 4, 5; I4, I5 in 4; I6 in 3, 4, 5; I7 in 8; I8 in 6 |
