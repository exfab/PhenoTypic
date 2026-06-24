# Overlays → `deliverables/` Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the CLI's per-image overlay PNGs (and their manifest) out of `results/<dataset>/overlays/` into `deliverables/overlays/<dataset>/` so overlays are packaged with the user-facing deliverables, and funnel every overlay path through the single canonical `dataset_overlays_dir()` helper (CLI and GUI alike).

**Architecture:** One canonical path builder in `phenotypic.sdk_._io_constants` (`dataset_overlays_dir`, plus a new `overlays_dir` root + relocated `overlay_manifest_path`) is the single source of truth. Every writer (CLI `OutputManager`, `phenotypicCLI`, recompile-SLURM, dashboard manifest plugin) and every reader (GUI `OutputRoot`, dashboard JS) is rewired to that helper. The GUI's `OutputRoot` already exposes `overlay_path()`/`has_overlay()` as its public API; this refactor removes the 3 re-spelled path joins still living inside `_output_root.py` and makes them delegate to the shared helper.

**Tech Stack:** Python 3.11+, pydantic v2, polars, Dash/Flask (GUI), pytest, `uv` runner, ruff + mypy.

## Global Constraints

- **`uv` is the sole runner.** Never bare `python`/`pip`. `uv run pytest …`, `uv run ruff check --fix`, `uv run mypy src/phenotypic`.
- **Resolve every output path via `phenotypic.sdk_` helpers** — never hand-join `results`/`deliverables`/`overlays` names. (Root + GUI CLAUDE.md rule.)
- **Hard cutover** — new code reads/writes only `deliverables/overlays/`. No back-compat fallback to the legacy `results/<ds>/overlays/`. Old output dirs must be re-run/recompiled to regenerate overlays (matches the prior `deliverables/` move precedent).
- **Target layout:** `deliverables/overlays/<dataset>/<stem>.png` (per-dataset subdir; no flat layout).
- **Manifest:** `deliverables/overlays/overlay_manifest.json` (moved out of `progress/`).
- **Constant names are unchanged** — `DIR_OVERLAYS = "overlays"` and `OVERLAY_MANIFEST_JSON = "overlay_manifest.json"` keep their values; only the path-builder *bodies* and doc comments change. This minimizes churn.
- **GUI gate:** any PR touching `src/phenotypic/gui/` must modify `src/phenotypic/gui/FEATURES.md` (the `features-md-gate` job). `✅ shipping` rows need a resolvable `path::test` Test ref (pre-commit validates).
- **Out of scope — do NOT touch the tune-cache overlays.** `gui/tune/_overlays.py` and `tests/unit/gui/tune/test_overlays.py` / `test_curate_overlays.py` / `test_overlay_cache.py` / `test_difference.py` operate on `<output>/.pht-tune-cache/overlays/` — a *different* overlays concept (tune-side difference visualizations). They are unrelated to the CLI output overlays and stay untouched.

---

## Design / Spec Summary (review this first)

### Current state (verified)

| Concern | Current location | Defined / used in |
|---|---|---|
| Overlay PNGs | `<out>/results/<ds>/overlays/<stem>.png` | written by `OutputManager.save_overlay` |
| Overlay manifest | `<out>/progress/overlay_manifest.json` | written by `ImageViewerPlugin.prepare_data` + chunk writer; read by dashboard JS |
| Canonical dir helper | `dataset_overlays_dir(out, ds)` → `results/<ds>/overlays/` | `sdk_/_io_constants.py:1142` |
| Manifest path helper | `overlay_manifest_path(out)` → `progress/overlay_manifest.json` | `sdk_/_io_constants.py:1212` |

### Target state

| Concern | New location |
|---|---|
| Overlay PNGs | `<out>/deliverables/overlays/<ds>/<stem>.png` |
| Overlay manifest | `<out>/deliverables/overlays/overlay_manifest.json` |
| `dataset_overlays_dir(out, ds)` | `deliverables/overlays/<ds>/` |
| new `overlays_dir(out)` | `deliverables/overlays/` |
| `overlay_manifest_path(out)` | `deliverables/overlays/overlay_manifest.json` |

### Why each consumer is affected

- **Writers** all currently compute the overlay dir three different ways (the canonical helper, a hand-join `out/DIR_RESULTS/ds/DIR_OVERLAYS`, and `OutputManager.results_dir/.../layer`). All three must move to `deliverables/`.
- **GUI** already routes external reads through `OutputRoot.overlay_path()`/`has_overlay()`; only the 3 in-file joins inside `_output_root.py` re-spell the path. They get redirected to the shared `dataset_overlays_dir()` (which is the "deduped helper" the request asks for). Dataset *discovery* still enumerates from `results/` (hdf/measurements stay there); only the overlay existence checks move.
- **Dashboard** (`analysis.html`, in `deliverables/`) builds overlay `<img src>` as `results/<ds>/overlays/<img>` *without* `ROOT_PREFIX` — already latently wrong post-`deliverables` move (degrades to "Image not found"). After this refactor the path becomes `overlays/<ds>/<img>` (a `deliverables/` sibling), which is *correct*. The manifest fetch moves from `ROOT_PREFIX + 'progress/overlay_manifest.json'` to the bare `'overlays/overlay_manifest.json'`.

### Safety notes (verified, no action needed but don't regress)

- `finalize_post_master_outputs` does **not** rmtree `deliverables/` — overlays written per-image survive finalize. (Only `_cleanup_scratch` rmtrees a `$SCRATCH/.phenotypic_stage_*` dir, which holds copied parquets only.)
- `write_json_atomic` already does `target_path.parent.mkdir(parents=True, exist_ok=True)`, so writing the manifest into `deliverables/overlays/` works even when no overlays exist.
- Overlays are written **directly to the final output dir** by each worker (`save_overlay` → `get_output_path`); there is no per-node staging/merge of overlays, so the SLURM path is unaffected beyond the dir change.

### Full impact inventory (every file that changes)

**Source — path helpers**
- `src/phenotypic/sdk_/_io_constants.py` — `dataset_overlays_dir` body + doc; new `overlays_dir`; `overlay_manifest_path` body + doc; `DIR_OVERLAYS`/`OVERLAY_MANIFEST_JSON` doc comments.
- `src/phenotypic/sdk_/__init__.py` — export `overlays_dir`.

**Source — CLI writers**
- `src/phenotypic/_cli/_cli_output_manager.py` — `get_output_path` overlays branch; `create_structure` overlays mkdir; imports (`+dataset_overlays_dir`, `−DIR_OVERLAYS`).
- `src/phenotypic/phenotypicCLI.py:1595` — hand-join → `dataset_overlays_dir`; imports.
- `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py:184` — hand-join → `dataset_overlays_dir`; imports.

**Source — dashboard**
- `src/phenotypic/_cli/_dashboard/_analysis/_image_viewer.py` — `prepare_data` scan dir + manifest target; `js()` overlay `<img>` path; imports.
- `src/phenotypic/_cli/_dashboard/_generator.py` — `fetchAnalysisData` manifest fetch split; `cmd-png` wget target.

**Source — GUI**
- `src/phenotypic/gui/results_viewer/_output_root.py` — dedup the 3 joins through `dataset_overlays_dir`; docstrings/error strings; imports.
- Docstring-only: `gui/_shared/tiles.py`, `gui/results_viewer/colony_view/_cropper.py`, `gui/results_viewer/__main__.py`.
- `src/phenotypic/gui/FEATURES.md` — gate edit (overlay thumbnail-route row note).

**Docs / generated**
- `src/phenotypic/_cli/_cli_readme_generator.py` — ASCII tree + Saved-Layers table.
- `docs/source/api_reference/cli_reference.rst`, `docs/source/how_to/pages/gui_hub.md`, `docs/source/how_to/pages/generate_reports.md`, `docs/source/tutorials/gui/06_view_results.md`, `…/20_results_timeline.md`, `…/16_tune_copilot.md`, `…/18_browse.md` — user-facing path references.
- Root `CLAUDE.md` + `src/phenotypic/gui/CLAUDE.md` — layout gotchas.
- (Skip `docs/_build/**`, `docs/build/**` — build artifacts. Skip `docs/source/api_reference/api/*.rst` — gitignored autosummary stubs, regenerated on build.)

**Tests** (path-rename + a few semantic edits) — enumerated in Task 6/7.

---

## Task 1: Canonical path helpers + constants (`sdk_`)

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (lines ~400-401, ~328-329, ~1142-1144, ~1212-1214)
- Modify: `src/phenotypic/sdk_/__init__.py` (import block ~129, `__all__` ~341)
- Test: `tests/unit/sdk_/test_io_constants.py` (lines 429, 490, 566; add one test)

**Interfaces:**
- Produces: `overlays_dir(output_dir: Path) -> Path` → `<out>/deliverables/overlays/`; `dataset_overlays_dir(output_dir: Path, dataset: str) -> Path` → `<out>/deliverables/overlays/<dataset>/`; `overlay_manifest_path(output_dir: Path) -> Path` → `<out>/deliverables/overlays/overlay_manifest.json`. All later tasks consume these.

- [ ] **Step 1: Update the failing tests first**

In `tests/unit/sdk_/test_io_constants.py`, change the two `dataset_overlays_dir` assertions (lines 429 and 566):

```python
assert dataset_overlays_dir(output, "ds1") == output / "deliverables" / "overlays" / "ds1"
```

Change the `overlay_manifest_path` assertion (line 490):

```python
assert overlay_manifest_path(output) == output / "deliverables" / "overlays" / "overlay_manifest.json"
```

Add a new test for `overlays_dir` next to the `dataset_overlays_dir` test (and import `overlays_dir` in the existing import block alongside `dataset_overlays_dir` at line ~53):

```python
def test_overlays_dir_roots_under_deliverables(self):
    output = Path("/tmp/out")
    assert overlays_dir(output) == output / "deliverables" / "overlays"
```

(Leave `assert DIR_OVERLAYS == "overlays"` at line 282 unchanged — the dir name is unchanged.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/sdk_/test_io_constants.py -k "overlay" -v`
Expected: FAIL — `dataset_overlays_dir` still returns `results/...`; `overlays_dir` is undefined (ImportError).

- [ ] **Step 3: Update `DIR_OVERLAYS` / `OVERLAY_MANIFEST_JSON` doc comments**

In `src/phenotypic/sdk_/_io_constants.py`, line ~400, change the comment:

```python
#: Overlay PNG subdirectory: ``<output>/deliverables/overlays/<ds>/``.
DIR_OVERLAYS: Final[str] = "overlays"
```

Line ~328:

```python
#: Manifest of overlay PNGs (one entry per per-image overlay). Lives alongside
#: the overlays under ``<output>/deliverables/overlays/``.
OVERLAY_MANIFEST_JSON: Final[str] = "overlay_manifest.json"
```

- [ ] **Step 4: Add `overlays_dir` and rewrite `dataset_overlays_dir`**

Replace the current `dataset_overlays_dir` (lines ~1142-1144) with both helpers:

```python
def overlays_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/overlays/`` — the overlay package root."""
    return deliverables_dir(output_dir) / DIR_OVERLAYS


def dataset_overlays_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/deliverables/overlays/<dataset>/``."""
    return overlays_dir(output_dir) / dataset
```

- [ ] **Step 5: Relocate `overlay_manifest_path`**

Replace the current `overlay_manifest_path` (lines ~1212-1214):

```python
def overlay_manifest_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/overlays/overlay_manifest.json``."""
    return overlays_dir(output_dir) / OVERLAY_MANIFEST_JSON
```

- [ ] **Step 6: Export `overlays_dir`**

In `src/phenotypic/sdk_/__init__.py`, add `overlays_dir,` to the import block next to `dataset_overlays_dir` (line ~129) and add `"overlays_dir",` to `__all__` next to `"dataset_overlays_dir"` (line ~341).

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_io_constants.py -k "overlay" -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_io_constants.py
git commit -m "refactor(io): root overlays + manifest under deliverables/"
```

---

## Task 2: CLI writer — `OutputManager`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` (imports ~30-50; `create_structure` ~1056-1057; `get_output_path` ~1083-1096)
- Test: `tests/unit/cli/test_cli_v2.py` (lines ~392, ~424, ~1261, ~1430-1438); `tests/integration/cli/test_cli_hdf_output.py` (lines ~129, ~155-162, ~196, ~240)

**Interfaces:**
- Consumes: `dataset_overlays_dir` (Task 1).
- Produces: `OutputManager.get_output_path(ds, "overlays", stem)` → `<base>/deliverables/overlays/<ds>/<stem>.png`; `create_structure` provisions that dir.

- [ ] **Step 1: Update the failing tests first**

`tests/unit/cli/test_cli_v2.py` — the `get_output_path` overlays assertion (lines ~1430-1438):

```python
        path = manager.get_output_path("my_dataset", "overlays", "image2")
        assert (
            path
            == temp_output_dir
            / "output"
            / "deliverables"
            / "overlays"
            / "my_dataset"
            / "image2.png"
        )
```

The `create_structure` overlay-dir existence checks (lines ~392 and ~424) — change `results/dataset1/overlays` to `deliverables/overlays/dataset1`:

```python
        assert (temp_output_dir / "deliverables" / "overlays" / "dataset1").exists()
```

The single-image overlay path (line ~1261):

```python
        overlay_file = output_dir / "deliverables" / "overlays" / "input" / "single.png"
```

(Leave the `not (… / "overlays").exists()` top-level guards at lines ~399, ~1121 unchanged — they assert no stray *root-level* `overlays/`, which stays true.)

`tests/integration/cli/test_cli_hdf_output.py` — the overlay path (lines ~129, ~196, ~240) and the dataset-dir whitelist (lines ~155-162). The overlay no longer lives under `dataset_dir`:

```python
        overlay_file = output_dir / "deliverables" / "overlays" / "plates" / "plate_001.png"
```

```python
        actual_children = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
        assert actual_children == {"hdf", "measurements"}, (
            f"Unexpected dataset-level folders after forward run. "
            f"Got {sorted(actual_children)}; expected {{'hdf', 'measurements'}} "
            f"(overlays now live under deliverables/overlays/<ds>/)."
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/cli/test_cli_v2.py -k "overlay or get_output_path or create_structure" tests/integration/cli/test_cli_hdf_output.py -v`
Expected: FAIL — paths still resolve under `results/`.

- [ ] **Step 3: Swap the import**

In `src/phenotypic/_cli/_cli_output_manager.py`, in the `from phenotypic.sdk_ import (…)` block: remove `DIR_OVERLAYS,` and add `dataset_overlays_dir,` (keep alphabetical-ish grouping with the other helpers like `measurements_by_feature_dir`).

- [ ] **Step 4: Special-case the `overlays` layer in `get_output_path`**

In `get_output_path` (around line 1083), add an early return for overlays before the generic `results_dir/.../layer` join. The generic join (line ~1096) stays for measurements/hdf/save_layers:

```python
        # Overlays are a user-facing deliverable, not a per-image result:
        # route them to <base>/deliverables/overlays/<ds>/<stem>.png.
        if layer == "overlays":
            return dataset_overlays_dir(self.base_dir, dataset_name) / f"{image_stem}.png"

        # Determine extension
        if layer == "measurements":
            ext = ".parquet"
        elif layer == "overlays":
            ext = ".png"
```

(The now-dead `elif layer == "overlays"` ext branch can be left as-is harmlessly, but prefer to delete it since the early return makes it unreachable. Also delete the stale `"overlays"` mention in the `get_output_path` `Args:` docstring is optional — keep `layer` doc accurate by noting overlays route to deliverables.)

- [ ] **Step 5: Provision the new dir in `create_structure`**

Replace line ~1056-1057:

```python
            if self.save_overlays:
                dataset_overlays_dir(self.base_dir, dataset.name).mkdir(
                    parents=True, exist_ok=True
                )
```

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/unit/cli/test_cli_v2.py -k "overlay or get_output_path or create_structure" tests/integration/cli/test_cli_hdf_output.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/_cli/_cli_output_manager.py tests/unit/cli/test_cli_v2.py tests/integration/cli/test_cli_hdf_output.py
git commit -m "refactor(cli): OutputManager writes overlays under deliverables/"
```

---

## Task 3: CLI hand-joined write sites (`phenotypicCLI` + recompile-SLURM)

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py` (import ~184; mkdir ~1594-1597)
- Modify: `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py` (import ~18; `_overlay_tasks_for_dataset` ~184)
- Test: `tests/unit/cli/test_cli_recompile.py`, `tests/unit/cli/test_cli_recompile_slurm.py` (overlay-regeneration assertions — update any `results/<ds>/overlays` paths to `deliverables/overlays/<ds>`)

**Interfaces:**
- Consumes: `dataset_overlays_dir` (Task 1).

- [ ] **Step 1: Update failing tests first**

In `tests/unit/cli/test_cli_recompile.py` and `tests/unit/cli/test_cli_recompile_slurm.py`, change every fixture/assertion that builds `… / "results" / <ds> / "overlays"` to `… / "deliverables" / "overlays" / <ds>`. (Run `grep -n "overlays" tests/unit/cli/test_cli_recompile*.py` to enumerate.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/cli/test_cli_recompile.py tests/unit/cli/test_cli_recompile_slurm.py -k "overlay" -v`
Expected: FAIL.

- [ ] **Step 3: Fix `phenotypicCLI.py`**

In the `from phenotypic.sdk_ import (…)` block (line ~184), remove `DIR_OVERLAYS,` and add `dataset_overlays_dir,`. (Confirm `DIR_OVERLAYS` has no other use in the file: `grep -n DIR_OVERLAYS src/phenotypic/phenotypicCLI.py` — expect only the import + line 1595.)

Replace the mkdir loop body (line ~1594-1597):

```python
    for dataset_name in {ds for ds, _ in work}:
        dataset_overlays_dir(output_dir, dataset_name).mkdir(
            parents=True, exist_ok=True
        )
```

- [ ] **Step 4: Fix `_cli_recompile_slurm_scripts.py`**

In its `from phenotypic.sdk_ import (…)` block (line ~18), remove `DIR_OVERLAYS,` and add `dataset_overlays_dir,`. (Keep `DIR_RESULTS` — still used by the `hdf_dir` join on line ~181.)

Replace line ~184:

```python
    overlay_dir = dataset_overlays_dir(output_dir, dataset_name)
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/cli/test_cli_recompile.py tests/unit/cli/test_cli_recompile_slurm.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/phenotypicCLI.py src/phenotypic/_cli/_cli_recompile_slurm_scripts.py tests/unit/cli/test_cli_recompile.py tests/unit/cli/test_cli_recompile_slurm.py
git commit -m "refactor(cli): recompile + overlay-regen paths use dataset_overlays_dir"
```

---

## Task 4: Dashboard manifest writer + JS

**Files:**
- Modify: `src/phenotypic/_cli/_dashboard/_analysis/_image_viewer.py` (imports ~8; `prepare_data` ~24-45; `js()` overlay path ~196-197)
- Modify: `src/phenotypic/_cli/_dashboard/_generator.py` (`fetchAnalysisData` ~1719-1728; `cmd-png` wget builder ~1197)
- Modify: `src/phenotypic/_cli/_cli_chunk_writer.py` (import ~35; `_ensure_overlay_manifest` guard ~415)
- Test: `tests/unit/cli/test_cli_analysis_plugins.py` (`test_overlay_manifest` ~390-417, and the second overlay fixture ~464-489)

**Interfaces:**
- Consumes: `overlays_dir`, `overlay_manifest_path` (Task 1).
- **Chunk-writer fix (do NOT skip):** `_ensure_overlay_manifest` (`_cli_chunk_writer.py:415`) guards on `progress_dir / OVERLAY_MANIFEST_JSON` — the *old* location. After the move, `prepare_data` writes to `deliverables/overlays/`, so this guard would always miss and needlessly re-run the plugin on every checkpoint early-return. The guard must read the new location.

- [ ] **Step 1: Update failing tests first**

In `tests/unit/cli/test_cli_analysis_plugins.py::test_overlay_manifest` (~399-417), move the fixture and the manifest assertion:

```python
        from phenotypic.sdk_ import dataset_overlays_dir, overlay_manifest_path

        overlay_dir = dataset_overlays_dir(tmp_dir, "plate1")
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "img001.png").touch()
        (overlay_dir / "img002.png").touch()

        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir(parents=True)

        ctx = AnalysisPrepareContext(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            merged_df=None,
        )
        plugin = ImageViewerPlugin()
        plugin.prepare_data(ctx)

        manifest = json.loads(overlay_manifest_path(tmp_dir).read_text())
        assert "datasets" in manifest
        assert "plate1" in manifest["datasets"]
        assert len(manifest["datasets"]["plate1"]) == 2
```

Apply the same fixture move to the second overlay-creating test (~464, and its `progress_dir / "overlay_manifest.json"` assertion at ~489 → `overlay_manifest_path(tmp_dir)`).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/cli/test_cli_analysis_plugins.py -k "overlay" -v`
Expected: FAIL — plugin still scans `results/` and writes to `progress/`.

- [ ] **Step 3: Rewrite `ImageViewerPlugin.prepare_data` + imports**

In `_image_viewer.py`, change the import (line ~8):

```python
from phenotypic.sdk_ import overlays_dir, overlay_manifest_path
```

Replace `prepare_data` (lines ~24-45):

```python
    def prepare_data(self, ctx: AnalysisPrepareContext) -> None:
        """Write ``overlay_manifest.json`` by scanning the overlay package."""
        from .._analysis_helpers import write_json_atomic

        package_dir = overlays_dir(ctx.output_dir)
        datasets: Dict[str, List[str]] = {}

        if package_dir.is_dir():
            for dataset_dir in sorted(package_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                png_files = sorted(f.name for f in dataset_dir.glob("*.png"))
                if png_files:
                    datasets[dataset_dir.name] = png_files

        write_json_atomic(
            {"datasets": datasets},
            overlay_manifest_path(ctx.output_dir),
        )
```

- [ ] **Step 4: Fix the overlay `<img>` path in `js()`**

In `_image_viewer.py` `loadOverlayImage` (lines ~196-197), the dashboard HTML lives in `deliverables/`, so the overlay package is a bare sibling — drop the `results/` prefix and the per-ds `/overlays/` infix:

```python
            "  var overlayPath = 'overlays/' + encodeURIComponent(ds)"
            " + '/' + encodeURIComponent(img);\n"
```

- [ ] **Step 5: Split the manifest fetch in `_generator.py`**

In `fetchAnalysisData` (lines ~1719-1728), `analysis_stats.json` stays under `progress/` but the overlay manifest is now a `deliverables/` sibling. Replace the shared loop:

```javascript
    async function fetchAnalysisData() {
      const sources = [
        { key: 'stats', url: ROOT_PREFIX + 'progress/analysis_stats.json' },
        { key: 'overlay', url: 'overlays/overlay_manifest.json' },
      ];
      for (const src of sources) {
        try {
          const resp = await fetch(src.url + '?' + Date.now());
          if (resp.ok) analysisData[src.key] = await resp.json();
        } catch(e) { /* file not ready yet */ }
      }
    }
```

- [ ] **Step 6: Point the "overlay images only" download at the new location**

In `_generator.py` (~1197), the `cmd-png` wget targets `base + 'results/'`. Overlays now live under deliverables; target them directly so the command grabs overlays (not the diagnostic `inspect/` PNGs that remain in `results/`):

```javascript
        'wget -r -np -nH -e robots=off --cut-dirs=' + cutDirs + ' -A "*.png"' + auth + ' ' + base + 'deliverables/overlays/';
```

(Verify the `cutDirs` computation still yields a sensible local tree; if `cutDirs` is derived to strip `results/`, adjust the comment accordingly. This is a convenience command, not load-bearing for correctness.)

- [ ] **Step 7: Fix the chunk-writer manifest guard**

In `_cli_chunk_writer.py`, swap the import (line ~35): remove `OVERLAY_MANIFEST_JSON,` and add `overlay_manifest_path,` (keep `CHUNK_MANIFEST_JSON` — still used at line ~115). Replace the guard (line ~415):

```python
    manifest_path = overlay_manifest_path(output_dir)
    if manifest_path.exists():
        return
```

- [ ] **Step 8: Run to verify tests pass**

Run: `uv run pytest tests/unit/cli/test_cli_analysis_plugins.py -k "overlay" -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/_cli/_dashboard/_analysis/_image_viewer.py src/phenotypic/_cli/_dashboard/_generator.py src/phenotypic/_cli/_cli_chunk_writer.py tests/unit/cli/test_cli_analysis_plugins.py
git commit -m "refactor(dashboard): overlay manifest + img paths under deliverables/"
```

---

## Task 5: GUI dedup — `OutputRoot`

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py` (imports ~20-25; `discover` ~92-197; `overlay_path` ~214-227; `_scan_overlay_index` ~398-419; docstrings/error strings)
- Test: `tests/gui/results_viewer/test_output_root.py`

**Interfaces:**
- Consumes: `dataset_overlays_dir` (Task 1).
- Produces (unchanged signatures, so external GUI callers are untouched): `OutputRoot.overlay_path(dataset, stem) -> Path`, `OutputRoot.has_overlay(dataset, stem) -> bool`. **This is the dedup deliverable** — every external GUI consumer (`_shared/tiles.py`, `_tile_routes.py`, `timeline_view/_thumb_routes.py`, `colony_view/_grid.py`, `_viewer_card.py`, `_qc_tab/review/_callbacks.py`) already calls these two methods; this task makes the methods (and discovery) the *only* place a path is built, and routes them through the shared `dataset_overlays_dir`.

- [ ] **Step 1: Update failing tests first**

In `tests/gui/results_viewer/test_output_root.py`, change any fixture that builds `root/"results"/<ds>/"overlays"` so it (a) still creates a `results/<ds>/` dataset dir for discovery, and (b) creates the overlay under `deliverables/overlays/<ds>/`. Pattern:

```python
        # discovery still enumerates datasets from results/
        (root / "results" / "d1" / "measurements").mkdir(parents=True)
        overlays = root / "deliverables" / "overlays" / "d1"
        overlays.mkdir(parents=True)
        (overlays / "img-1.png").touch()
```

Add/adjust an assertion that `overlay_path` points at deliverables:

```python
    assert output_root.overlay_path("d1", "img-1") == \
        root / "deliverables" / "overlays" / "d1" / "img-1.png"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/gui/results_viewer/test_output_root.py -v`
Expected: FAIL (discovery finds no overlays / `overlay_path` returns `results/...`).

- [ ] **Step 3: Swap the import**

In `_output_root.py`, change the `from phenotypic.gui._config import (…)` block to drop `DIR_OVERLAYS` (keep `DIR_MEASUREMENTS`, `RESULTS_DIRNAME`, `VIEWER_CACHE_DIRNAME`), and add to the `from phenotypic.sdk_ import (…)` block:

```python
from phenotypic.sdk_ import (
    dataset_overlays_dir,
    master_measurements_parquet_path,
    measurements_parquet_path,
    resolve_pipeline_config_path,
)
```

- [ ] **Step 4: Redirect `overlay_path` to the shared helper**

Replace the body (line ~227):

```python
        return dataset_overlays_dir(self.root, dataset) / f"{stem}.png"
```

Update its docstring `Returns:` to `<root>/deliverables/overlays/<dataset>/<stem>.png`.

- [ ] **Step 5: Redirect the discovery existence check**

Replace line ~174-176:

```python
        datasets_with_overlays = [
            ds for ds in datasets if dataset_overlays_dir(root, ds).is_dir()
        ]
```

- [ ] **Step 6: Redirect `_scan_overlay_index`**

Change its signature to take `root` instead of `results_dir` (it must build the deliverables path), and update its single call site in `discover` (line ~197) to pass `root`:

```python
def _scan_overlay_index(
    root: Path, datasets_with_overlays: list[str]
) -> frozenset[tuple[str, str]]:
    """Snapshot every ``(dataset, stem)`` whose overlay PNG exists on disk."""
    pairs: set[tuple[str, str]] = set()
    for dataset in datasets_with_overlays:
        for entry in dataset_overlays_dir(root, dataset).iterdir():
            if entry.suffix.lower() == ".png" and entry.is_file():
                pairs.add((dataset, entry.stem))
    return frozenset(pairs)
```

Call site (line ~197): `overlay_index = _scan_overlay_index(root, datasets_with_overlays)`.

- [ ] **Step 7: Fix the prose**

Update the `discover` docstring layout line (~95, ~156) and the warning/error strings (~185-188) to read `<root>/deliverables/overlays/<dataset>/<image_stem>.png`. Keep the `results/` requirement language for *dataset discovery* (hdf/measurements stay there). The `--save-overlays` hint stays valid.

- [ ] **Step 8: Run to verify they pass**

Run: `uv run pytest tests/gui/results_viewer/test_output_root.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_output_root.py tests/gui/results_viewer/test_output_root.py
git commit -m "refactor(gui): OutputRoot overlay access funnels through dataset_overlays_dir"
```

---

## Task 6: Migrate GUI / integration test fixtures

The bulk of the remaining work is mechanical, but with one **critical caveat**: `OutputRoot.discover` enumerates datasets from `results/` and requires at least one `results/<ds>/` dir. Many fixtures created `results/<ds>/overlays/` as a *side effect* of making the dataset dir. After moving overlays out, those fixtures must still create a `results/<ds>/` dir (e.g. a `measurements/` subdir) **in addition to** the deliverables overlay.

**Transformation rule (per fixture):**
1. Keep/ensure a `results/<ds>/` directory exists (add `(root/"results"/<ds>/"measurements").mkdir(parents=True, exist_ok=True)` if the fixture previously relied solely on the overlays dir to create it).
2. Move the overlay PNG creation to `deliverables/overlays/<ds>/<stem>.png`.

**Files (run `grep -n "results.*overlays\|overlays.*\.png" <file>` in each to find lines):**

- [ ] `tests/gui/_shared/test_tiles.py` (~340, ~470)
- [ ] `tests/gui/results_viewer/colony_view/test_grid.py`
- [ ] `tests/gui/results_viewer/colony_view/test_crop_routes.py`
- [ ] `tests/gui/results_viewer/timeline_view/test_grid.py`
- [ ] `tests/gui/results_viewer/timeline_view/test_thumb_routes.py`
- [ ] `tests/gui/results_viewer/timeline_view/test_layout.py`
- [ ] `tests/gui/builder/test_image_renderer.py`
- [ ] `tests/unit/gui/results_viewer/test_filter_panel.py` (~186, ~201)
- [ ] `tests/unit/gui/results_viewer/test_navigation_layout.py` (~165, ~173)
- [ ] `tests/unit/gui/results_viewer/test_qc_review_data.py` (~69, ~72)
- [ ] `tests/integration/gui/test_filter_offcanvas_layout.py` (~24 docstring, ~39, ~42)
- [ ] `tests/integration/gui/test_analysis_handoff.py` (~43-44)
- [ ] `tests/integration/gui/test_qc_review_recompute.py` (~90-93, ~208-210)
- [ ] `tests/integration/gui/test_qc_tab_registry.py` (~45-47)
- [ ] `tests/integration/gui/test_viewer_handoff.py` (~29, ~44)
- [ ] `tests/integration/gui/test_timeline_thumb_url.py` (~41-44)
- [ ] `tests/integration/gui/test_triage_callbacks.py` (~67-71)
- [ ] `tests/e2e/gui/test_filter_offcanvas.py`, `test_heatmap_tab.py`, `test_qc_review_splitter.py`, `test_qc_tab.py`, `test_radial_triage.py`, `test_results_timeline.py`
- [ ] `tests/integration/cli/test_staged_gpu_local.py`, `tests/integration/cli/test_staged_slurm_live.py`
- [ ] `tests/unit/cli/test_cli_output_manager_inspect.py`, `tests/unit/cli/test_cli_save_inspect_integration.py`, `tests/unit/cli/test_slurm_process_only_scripts.py`, `tests/unit/cli/test_staged_routing.py` (verify whether each builds overlay paths or only sets `save_overlays`; edit only the ones that build paths)

> **Before editing each file**, check for a shared fixture helper: `grep -rn "def .*output_root\|def .*cli_out\|results.*overlays" tests/gui/conftest.py tests/integration/gui/conftest.py tests/conftest.py`. If a shared builder exists, fix it once there instead of per-file. Prefer the smallest number of edits.

> **Confirm out-of-scope tune tests are untouched:** `tests/unit/gui/tune/test_overlays.py`, `test_curate_overlays.py`, `test_overlay_cache.py`, `test_difference.py`, and `tests/unit/tune/test_build_pipeline_nested.py`, `test_sigma_color_none_categorical.py` — these are tune-cache overlays. Run them to confirm green and do NOT edit.

- [ ] **Step A: Apply the transformation file-by-file (or via the shared helper).**
- [ ] **Step B: Run the affected suites incrementally**

```bash
uv run pytest tests/gui/ tests/integration/gui/ tests/integration/cli/test_cli_hdf_output.py -q
```
Expected: PASS. (For Qt/napari suites, run with `QT_QPA_PLATFORM=offscreen` and the `qt-test` group as per project CLAUDE.md.)

- [ ] **Step C: Confirm tune suites stay green**

```bash
uv run pytest tests/unit/gui/tune -q
```
Expected: PASS, untouched.

- [ ] **Step D: Commit**

```bash
git add tests/
git commit -m "test: migrate overlay fixtures to deliverables/overlays/<ds>/"
```

---

## Task 7: Docs, generated README, FEATURES.md, CLAUDE.md

**Files:**
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py` (~110, ~126)
- Modify: `src/phenotypic/gui/FEATURES.md` (gate requirement)
- Modify: user docs (rst/md listed below)
- Modify: root `CLAUDE.md`, `src/phenotypic/gui/CLAUDE.md`

- [ ] **Step 1: Generated README (`_cli_readme_generator.py`)**

In `_generate_layout_section`, move the `overlays/` line out of the per-dataset block into the `deliverables/` block of the ASCII tree (around lines ~95-114). New deliverables block adds:

```
|   +-- overlays/                     # Detection overlay PNGs, one per input image (per-dataset subfolders)
```

and the per-dataset block (lines ~108-110) loses its `overlays/` row, leaving `hdf/` and `measurements/`.

In `_generate_layers_section` (the "Saved Layers" table), move the `overlays/` row from the per-dataset table into the deliverables description (or retitle the section). Minimum: change the `overlays/` row text to note it lives under `deliverables/overlays/<dataset>/`.

- [ ] **Step 2: Update the README generator test if present**

Run `grep -rln "overlays" tests/ | xargs grep -l "readme\|README\|layout_section\|Saved Layers" 2>/dev/null`. If a README-generator test asserts the tree/table, update its expected strings.

- [ ] **Step 3: FEATURES.md gate edit**

Touch `src/phenotypic/gui/FEATURES.md` with an accurate note. Update the **Overlay thumbnail route** row (line ~361) description to reflect the new source path, keeping its existing valid Test ref:

```
| Overlay thumbnail route | `GET /<VIEWER_THUMB_URL_SEGMENT>/<dataset>/<stem>` | The Phase 1 thumbnail factory serves a bucketed downscaled PNG of `output_root.overlay_path(dataset, stem)` (now sourced from `deliverables/overlays/<dataset>/`), cached under `.viewer_cache/timeline_thumbs`. | ✅ shipping | unit | tests/gui/results_viewer/timeline_view/test_thumb_routes.py::test_thumb_happy_path_returns_bucketed_png |
```

- [ ] **Step 4: User-facing docs**

For each, replace `results/<dataset>/overlays/` (or similar) references with `deliverables/overlays/<dataset>/`:
- `docs/source/api_reference/cli_reference.rst`
- `docs/source/how_to/pages/gui_hub.md`
- `docs/source/how_to/pages/generate_reports.md`
- `docs/source/tutorials/gui/06_view_results.md`
- `docs/source/tutorials/gui/20_results_timeline.md`
- `docs/source/tutorials/gui/16_tune_copilot.md`
- `docs/source/tutorials/gui/18_browse.md`

(Skip `docs/_build/**`, `docs/build/**`, and `docs/source/api_reference/api/*.rst` — build artifacts / gitignored autosummary stubs.)

- [ ] **Step 5: CLAUDE.md gotchas**

- Root `CLAUDE.md` — the "Output location — `deliverables/`" gotcha lists what lives in `deliverables/` vs `results/`. Add `deliverables/overlays/<ds>/<stem>.png` to the deliverables list and remove "overlays" from the `results/<ds>/` description.
- `src/phenotypic/gui/CLAUDE.md` — the "Output layout — `deliverables/`" note says `RESULTS_DIRNAME (results/, per-image hdf/measurements/overlays)`. Change to `per-image hdf/measurements` and add overlays to the deliverables enumeration.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_readme_generator.py src/phenotypic/gui/FEATURES.md docs/ CLAUDE.md src/phenotypic/gui/CLAUDE.md
git commit -m "docs: overlays now packaged under deliverables/overlays/"
```

---

## Task 8: Whole-repo verification

- [ ] **Step 1: No stray hand-joins or stale path references remain**

```bash
grep -rn "results.*overlays\|DIR_RESULTS.*DIR_OVERLAYS\|/ DIR_OVERLAYS\|results/<.*>/overlays" src/ | grep -v "pht-tune-cache"
```
Expected: no production hand-joins of `results/.../overlays`. (Hits should only be the canonical helper, tune-cache, or updated docstrings referencing `deliverables`.)

- [ ] **Step 2: Lint + types**

```bash
uv run ruff check --fix
uv run mypy src/phenotypic
```
Expected: clean (in particular, no unused `DIR_OVERLAYS` imports left behind).

- [ ] **Step 3: Targeted regression suite**

```bash
uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/cli tests/integration/cli/test_cli_hdf_output.py tests/gui tests/integration/gui -q
```
Expected: PASS. (Use the full dev env per project CLAUDE.md: `uv sync --group dev --group qt-test --extra gui --extra napari` and `QT_QPA_PLATFORM=offscreen` for Qt suites.)

- [ ] **Step 4: Live smoke (recommended, not gated)**

Run a tiny forward CLI run on `load_synth_yeast_plate()`-style inputs and confirm:
- `deliverables/overlays/<ds>/<stem>.png` exists; no `results/<ds>/overlays/`.
- `deliverables/overlays/overlay_manifest.json` exists with the dataset → PNG mapping.
- Launch `uv run phenotypic-gui --root <out>` → results viewer image picker shows overlays; open `deliverables/analysis.html` → Image Viewer renders the overlay (no "Image not found").

- [ ] **Step 5: Final commit / PR**

```bash
git add -A && git commit -m "chore: finalize overlays→deliverables refactor"
```

---

## Self-Review (completed against the impact inventory)

- **Spec coverage:** every row of the impact inventory maps to a task — helpers (T1), OutputManager (T2), hand-joins (T3), dashboard writer+JS + chunk-writer guard (T4), GUI dedup (T5), test fixtures (T6), docs/README/FEATURES/CLAUDE (T7), verification (T8).
- **Type consistency:** `dataset_overlays_dir(output_dir, dataset)` and `overlays_dir(output_dir)` signatures are used identically in T2–T5; `_scan_overlay_index` signature change (`results_dir` → `root`) is paired with its only call site in the same task.
- **No placeholders:** every code step shows the exact replacement; mechanical test edits give the explicit transformation rule + the discovery-dir caveat + the non-mechanical exact edits (whitelist assertion, `get_output_path` assertion, manifest test).
- **Open risk flagged for the reviewer:** the dashboard `measurementPath` helper (`_image_viewer.py`) builds `results/<ds>/measurements/<file>` *without* `ROOT_PREFIX` — already latently broken post-`deliverables` move and **outside this refactor's scope** (measurements stay in `results/`). Fixing it would mean prefixing with `ROOT_PREFIX`; recommend a separate follow-up so this PR stays scoped to overlays.
```