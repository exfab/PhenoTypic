# Design — Consolidate logo/brand assets under `src/phenotypic/_assets`

- **Date:** 2026-06-04
- **Branch:** `worktree-dedup-asset-location`
- **Status:** Approved (design); pending implementation plan

## 1. Context & motivation

Logo and brand image assets are currently scattered across **four** locations
with byte-identical duplicates and dead files:

| Location | Role |
|----------|------|
| `src/phenotypic/_cli/_assets/` | `LogoArtOnly.png` — **orphan** (only thing in `package-data`, referenced by nothing) |
| `src/phenotypic/_cli/_dashboard/_assets/` | `LogoArtOnly.png` (used by CLI dashboard), `light_logo.png` (**orphan**), `plotly.min.js`, `hyparquet.min.js` |
| `src/phenotypic/gui/_shared/_static/` | `dashboard_logo.svg` — used by all GUI sub-apps |
| `docs/source/_static/assets/` | 7.9 MB brand library (logos in many sizes) **+** decorative landing-page card icons |

Byte-identical duplicate groups (SHA-256 prefix / size):

| Hash | Bytes | Copies | Canonical role |
|------|-------|--------|----------------|
| `ed32b501` | 74831 | `_cli/_assets/LogoArtOnly.png` (orphan) + `_cli/_dashboard/_assets/LogoArtOnly.png` (used) | **App CLI-dashboard logo** |
| `dc181a12` | 511060 | `docs/.../400x150/dashboard_logo.svg` + `gui/_shared/_static/dashboard_logo.svg` (used) | **App GUI logo** |
| `dd0adbf6` | 74748 | `docs/.../500x500_png/LogoArtOnly.png` + `_cli/_dashboard/_assets/light_logo.png` (orphan) | Brand variant (`500x500_png/LogoArtOnly.png`) |

**Latent packaging bug (fixed by this work):** `package-data` lists only
`_cli/_assets/*.png` and `gui/_shared/_static/*.svg`. The directory the CLI
dashboard actually reads — `_cli/_dashboard/_assets/` — is **not** packaged, so
in a **built wheel** the dashboard logo (`_load_logo_data_uri`) and the
interactive-plot sidecars (`plotly.min.js`, `hyparquet.min.js` via
`_write_js_sidecar`) silently fail (their `except` branches swallow the missing
file). Works only from a source/editable checkout today.

## 2. Goals / non-goals

**Goals**
1. One canonical home for all logo/brand images and the dashboard's bundled JS:
   `src/phenotypic/_assets/`.
2. Collapse the three duplicate groups to a single canonical copy each; delete
   the two orphan files.
3. CLI dashboard, GUI, and Sphinx all resolve their logos from
   `src/phenotypic/_assets` through a single accessor.
4. Fix `package-data` so every bundled asset ships in the wheel.

**Non-goals**
- Decorative landing-page card icons (`getting_started_rocket`, `user_guide_book`,
  `api_ref_sign`, `dev_guide`, `examples`, `tutorial`, `contact_us`, `downloads`)
  stay in `docs/source/_static/assets/` — they are not logos and are referenced
  by `docs/source/index.rst`.
- GUI tutorial screenshots (`docs/source/_static/gui_images/`), OpenSeadragon
  vendored sprites (`gui/results_viewer/_assets/openseadragon/images/`), the DAG
  diagram (`docs/diagrams/builder_dag_canvas.svg`), and all sample data
  (`src/phenotypic/data/`) are untouched.
- No re-theming, no logo redesign, no URL changes visible to end users.

## 3. Chosen decisions (from brainstorming)

- **Scope:** full brand/logo library moves into the package (all `*logo*`,
  `*exfab*`, `*sponsor*`, `LogoArtOnly`, `ExFabLogo` variants).
- **JS sidecars:** move `plotly.min.js` + `hyparquet.min.js` into `_assets` too —
  one bundled-assets home, one `package-data` entry.
- **Dedup depth:** single source + delete dups. Sphinx points at the package
  copy; the duplicate logo files are removed from `docs/_static/assets`.
- **Resolution:** central accessor module (Approach A).

## 4. Target layout

```
src/phenotypic/_assets/
  __init__.py                    # accessor: ASSET_DIR, asset_bytes(), logos_dir()
  logos/
    LogoArtOnly.png              # ed32b501 — CLI dashboard logo (canonical)
    dashboard_logo.svg           # dc181a12 — GUI logo (canonical)
    ExFabLogo.svg                # 72499d7c
    200x150/                     # brand variant library (see move map)
    400x150/                     #   light_logo_exfab.svg + gradient_logo_exfab.svg (Sphinx)
    500x500/
    500x500_png/
  vendor/
    plotly.min.js                # de3be007
    hyparquet.min.js             # 8a2d282a
```

## 5. Move / collapse map (exact)

`git mv` everything so history follows. `git rm` the orphans.

**Canonical app logos → `_assets/logos/` root**
- `_cli/_dashboard/_assets/LogoArtOnly.png` → `_assets/logos/LogoArtOnly.png`
- `gui/_shared/_static/dashboard_logo.svg` → `_assets/logos/dashboard_logo.svg`
- `docs/source/_static/assets/ExFabLogo.svg` → `_assets/logos/ExFabLogo.svg`

**Brand variants → `_assets/logos/<size>/` (from `docs/source/_static/assets/<size>/`)**
- `200x150/`: `dark_logo (2).svg`, `dark_logo.png`, `dark_logo.svg`,
  `dark_logo_sponsor.png`, `dark_logo_sponsor.svg`, `light_logo.png`,
  `light_logo.svg`, `light_logo_sponsor.png`, `light_logo_sponsor.svg`
- `400x150/`: `light_logo_exfab.svg`, `gradient_logo_exfab.svg`
  (the `400x150/dashboard_logo.svg` is `dc181a12` — **dropped**, collapsed into
  `_assets/logos/dashboard_logo.svg`)
- `500x500/`: `LogoArtOnly.svg`, `dark_logo.svg`, `dark_logo_sponsor.svg`,
  `light_logo.svg`, `light_logo_sponsor.svg`, `light_logo_sponsor_centered.svg`
- `500x500_png/`: `LogoArtOnly.png` (`dd0adbf6`), `dark_logo.png`,
  `light_logo.png`, `light_logo_sponsor.png`, `light_logo_sponsor_centered.png`

**JS sidecars → `_assets/vendor/`**
- `_cli/_dashboard/_assets/plotly.min.js` → `_assets/vendor/plotly.min.js`
- `_cli/_dashboard/_assets/hyparquet.min.js` → `_assets/vendor/hyparquet.min.js`

**Orphans → `git rm` (byte-identical to a canonical copy, referenced by nothing)**
- `_cli/_assets/LogoArtOnly.png` (`ed32b501`)
- `_cli/_dashboard/_assets/light_logo.png` (`dd0adbf6`)

**Emptied directories to remove:** `_cli/_assets/`, `_cli/_dashboard/_assets/`,
`gui/_shared/_static/` (and `gui/_shared/_static`'s parent only if it has no
other contents — `_shared/` keeps its `.py` files).

**Stay in `docs/source/_static/assets/` (decorative icons, untouched):**
`api_ref_sign`, `contact_us`, `dev_guide`, `downloads`, `examples`,
`getting_started_rocket`, `tutorial`, `user_guide_book` (png/svg across sizes).

## 6. New accessor module — `src/phenotypic/_assets/__init__.py`

Single source of truth for the asset root, built on `importlib.resources`.

```python
"""Bundled static assets (logos + dashboard JS) for PhenoTypic.

Canonical home for every image/JS file the runtime or docs build ships.
Resolve assets through this module rather than re-spelling the package path.
"""
from __future__ import annotations
from importlib.resources import files
from pathlib import Path

#: Filesystem path to this package's asset root. This package's `__init__.py`
#: lives in the asset dir, so `__file__`-relative resolution matches the
#: existing on-disk-install pattern in `gui/_shared/_blueprint.py`.
ASSET_DIR: Path = Path(__file__).resolve().parent

def logos_dir() -> Path:
    """Directory holding logo/brand images (Flask static root for the GUI)."""
    return ASSET_DIR / "logos"

def asset_bytes(relpath: str) -> bytes:
    """Read an asset's bytes by POSIX-style relative path, e.g. ``"logos/LogoArtOnly.png"``."""
    return files(__name__).joinpath(*relpath.split("/")).read_bytes()
```

- `asset_bytes` uses the `Traversable` API (zip-safe) for the base64 embed.
- `ASSET_DIR` / `logos_dir()` expose a concrete `Path` for Flask
  `send_from_directory` and Sphinx — consistent with the current code, which
  already assumes an on-disk package (`Path(__file__).parent`).

## 7. Consumer edits

1. **`src/phenotypic/_cli/_dashboard/_generator.py`**
   - `_load_logo_data_uri()` (line ~212): replace
     `files("phenotypic._cli._dashboard").joinpath("_assets", "LogoArtOnly.png").read_bytes()`
     with `phenotypic._assets.asset_bytes("logos/LogoArtOnly.png")`.
   - `_write_js_sidecar()` (line ~151): replace
     `files("phenotypic._cli._dashboard").joinpath("_assets", filename)` with
     `files("phenotypic._assets").joinpath("vendor", filename)` (or an accessor
     helper). The two callers pass `"plotly.min.js"` / `"hyparquet.min.js"`.
   - Keep the base64 data-URI embedding and the `progress/` sidecar URLs
     unchanged — only the source path changes.

2. **`src/phenotypic/gui/_shared/_blueprint.py`**
   - `_STATIC_DIR = _HERE / "_static"` → `_STATIC_DIR = logos_dir()` (import from
     `phenotypic._assets`).
   - `SHARED_LOGO_PATH = "_shared/dashboard_logo.svg"` stays — the served URL is
     unchanged, so `tests/integration/gui/test_smoke_shell.py` keeps passing.
   - Update module docstring (references `_static/` next to the module) and the
     `gui/_shared/__init__.py` docstring (references `_shared/_static/dashboard_logo.svg`).

3. **`docs/source/conf.py`**
   - Source logos from the package. Sketch:
     ```python
     from pathlib import Path
     import phenotypic
     _PKG_LOGOS = Path(phenotypic.__file__).parent / "_assets" / "logos"
     # Logo assets live in phenotypic/_assets/logos (single source of truth).
     html_static_path = ["_static", str(_PKG_LOGOS)]   # tree copied into _static/
     LIGHT_LOGO_PATH = "400x150/light_logo_exfab.svg"  # resolved under _static/
     DARK_LOGO_PATH  = "400x150/gradient_logo_exfab.svg"
     html_logo = LIGHT_LOGO_PATH
     ```
   - Exact mechanism (whether the theme's `image_light`/`image_dark` want a
     `_static`-relative path vs. `html_logo`'s conf-relative path) is verified by
     a `sphinx-build` smoke during implementation; the principle is fixed: logos
     come from `phenotypic/_assets/logos`, with a comment saying so.
   - `conf.py` already imports `phenotypic` (guarded `try/except ImportError`,
     for `version`). The logo path now also needs `phenotypic.__file__`, which is
     fine for any real docs build (autodoc requires the package installed). Keep
     the resolution inside the same import guard so a bare `conf.py` import
     without the package still degrades gracefully rather than crashing.

## 8. Packaging — `pyproject.toml` `[tool.setuptools.package-data]`

Replace:
```toml
"phenotypic" = [
    "data/*.jpg",
    "data/*.png",
    "_cli/_assets/*.png",
    "gui/_shared/_static/*.svg",
]
```
with:
```toml
"phenotypic" = [
    "data/*.jpg",
    "data/*.png",
    "_assets/logos/*",
    "_assets/logos/*/*",
    "_assets/vendor/*",
]
```
Explicit per-level globs avoid relying on setuptools `**` support. Verified by
building a wheel and listing `_assets/**` membership.

## 9. Tests

- **New unit test** (`tests/unit/...` near the dashboard generator tests):
  `phenotypic._assets.asset_bytes("logos/LogoArtOnly.png")` is non-empty and PNG
  magic; `_load_logo_data_uri()` returns a string starting with
  `data:image/png;base64,`.
- **New unit test:** `logos_dir()` exists and contains `dashboard_logo.svg`.
- **Optional dedup invariant:** no two files under `_assets/logos/` share a
  SHA-256 (guards future re-introduction of duplicates).
- **Update** `tests/integration/gui/test_smoke_shell.py` comment (lines ~226-230)
  that says the file is served "from `gui/_shared/_static/`" → now
  `phenotypic/_assets/logos/`. Assertion (URL `_shared/dashboard_logo.svg`)
  is unchanged.

## 10. Verification checklist

1. `uv run ruff check --fix`
2. `uv run mypy src/phenotypic`
3. `uv run pytest tests/integration/gui tests/unit -k "dashboard or logo or shell or asset"`
4. `uv build` (or `python -m build`) → inspect wheel: `_assets/logos/...`,
   `_assets/vendor/plotly.min.js`, `hyparquet.min.js` all present.
5. `uv run sphinx-build -b html docs/source /tmp/docs_smoke` → confirm the logo
   renders / is copied to `_static/`, no missing-image warnings.
6. `git status` after `git mv` — confirm old dirs gone, no stray tracked files
   (worktree disk-safety habit).

## 11. Risks & edge cases

- **`importlib.resources` + Flask:** `send_from_directory` needs a real dir;
  `ASSET_DIR` assumes an unzipped install (true for wheels/editable here, same
  assumption the current code makes). Not zip-safe — acceptable, documented.
- **setuptools globbing depth:** brand library is at most two levels under
  `_assets/logos/` (`<size>/<file>`); `_assets/logos/*` + `_assets/logos/*/*`
  cover root + one size level. Verified against wheel contents.
- **Filename with a space/parens:** `200x150/dark_logo (2).svg` — quote in any
  shell `git mv`; safe in `package-data` globs.
- **Sphinx static-path collision:** copying the whole logos tree into the docs
  `_static/` adds the brand library to the built site (acceptable; they are docs
  assets). No name clash with existing `_static/assets/` icons because logos land
  under their `<size>/` subpaths.
- **`docs/source/_static/assets/` not emptied:** the `<size>/` folders keep their
  decorative icons, so the directories remain; only logo files are removed.
- **Blueprint now serves the whole `logos/` tree:** pointing `_STATIC_DIR` at
  `logos_dir()` means `GET /<mount>/_shared/<any logo path>` resolves, not just
  `dashboard_logo.svg`. All of it is public brand imagery, so low-risk; noted for
  awareness. `send_from_directory` already blocks path traversal outside the
  root.
