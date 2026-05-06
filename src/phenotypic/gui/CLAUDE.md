# PhenoTypic GUI Module Guide

The GUI is a Dash-based hub composed of four mounted sub-apps:

| Mount        | Module                              | Purpose                                  |
| ------------ | ----------------------------------- | ---------------------------------------- |
| `/`          | `gui/shell/`                        | Top-bar chrome, sidebar, home page       |
| `/builder/`  | `gui/builder/`                      | Pipeline builder (dash-cytoscape graph)  |
| `/results/`  | `gui/results_viewer/`               | Output viewer (OpenSeadragon + tables)   |
| `/run/`      | `gui/run_console/`                  | Run console (form + log tail + recents)  |

Composition lives in [shell/_app.py](shell/_app.py) (`compose_hub`); the
hub's WSGI seam is a `werkzeug.middleware.dispatcher.DispatcherMiddleware`
that strips each mount prefix before forwarding to the sub-app's Flask
server.

---

## Where to put things

| Need                                     | File                                                  |
| ---------------------------------------- | ----------------------------------------------------- |
| New CLI default / port / log format      | [_config.py](_config.py) (`DEFAULT_*`, `LOG_FORMAT`)  |
| New brand color / type / radius / shadow | [_design.py](_design.py) (`COLOR_*`, `TEXT_*`, …)     |
| New mount prefix                         | [_config.py](_config.py) (`MOUNT_*`) + register in [shell/_app.py](shell/_app.py) |
| New `app.server.config` key              | [_config.py](_config.py) (`CFG_*`; pick `CFG_OPERATION_REGISTRY` for the builder, `CFG_RUN_REGISTRY` for the run console) |
| New sandbox/cache subdirectory name      | [_config.py](_config.py) (`SANDBOX_*`, `*_DIRNAME`)   |
| New tool-internal Dash component ID      | tool's own `_ids.py` (e.g. `builder/_ids.py`)         |
| New cross-tool path store ID             | [shell/_ids.py](shell/_ids.py)                        |

**Default rule:** if a string literal would appear in two or more files,
move it to `_config.py` (Python identifiers) or `_design.py` (CSS-shaped
values). Don't re-spell.

---

## Shared constants — always import, never re-spell

### Launcher defaults

```python
from phenotypic.gui._config import (
    DEFAULT_HOST,        # "127.0.0.1"
    DEFAULT_PORT,        # 8050
    LOG_FORMAT,          # logging.basicConfig format
    SSH_TUNNEL_HINT,     # "ssh -L 8050:localhost:8050 user@cluster"
    add_launcher_args,   # add --host/--port/--debug to a parser
    configure_launcher_logging,  # logging.basicConfig with LOG_FORMAT
    print_launcher_banner,       # consistent banner (title + url + hint)
)
```

Every `__main__.py` and `_launcher.py` calls `add_launcher_args(parser)`
and `configure_launcher_logging(debug=args.debug)`. Don't re-implement
either or you re-introduce drift like the `"%(name)s:"` vs
`"%(name)s "` inconsistency we cleaned up.

### Mount prefixes

```python
from phenotypic.gui._config import (
    MOUNT_HOME,             # "/"
    MOUNT_BUILDER,          # "/builder/"
    MOUNT_VIEWER,           # "/results/"
    MOUNT_RUN,              # "/run/"
    SANDBOX_API_PREFIX,     # "/sandbox/api"
    RUNS_BLUEPRINT_PREFIX,  # "/runs"
)
```

Every dispatcher mount, tab href, redirect, and Flask blueprint
`url_prefix` must spell these via the constant. The shell layout's
`_TAB_HREFS` dict and the composer's dispatcher mounts BOTH read these
so they can never disagree.

### Flask `app.server.config` keys

```python
from phenotypic.gui._config import (
    CFG_URL_PREFIX,            # current Dash app's mount-point prefix
    CFG_OPERATION_REGISTRY,    # builder's OperationRegistry
    CFG_RUN_REGISTRY,          # run console's RunRegistry (also stashed on shell)
    CFG_RUNNER,                # process-wide LocalRunner
    CFG_IMAGE_ROOT,            # builder image root
    CFG_SANDBOX_ROOT,          # frozen sandbox root (string)
    CFG_OUTPUT_ROOT,           # results viewer OutputRoot
    CFG_FILTERED_STATE,        # results viewer FilteredMeasurements
)
```

Reads use `app.server.config[CFG_*]` or `current_app.config.get(CFG_*)`;
writes use the same constant. The two `*_REGISTRY` keys are deliberately
distinct — even though the builder and run-console Dash apps live on
separate Flask servers, naming the keys after the registry's TYPE
prevents a future cross-tool callback from grabbing the wrong handle.

### Sandbox / cache paths

```python
from phenotypic.gui._config import (
    SANDBOX_GUI_DIRNAME,           # ".phenotypic-gui"
    SANDBOX_PRESETS_SUBDIR,        # "presets"
    SANDBOX_BUILDER_TILES_SUBDIR,  # "builder_tiles"
    RUN_LOG_DIRNAME,               # ".gui_log" (inside a run's output dir)
    VIEWER_CACHE_DIRNAME,          # ".viewer_cache" (inside an output root)
)

presets = sandbox.root / SANDBOX_GUI_DIRNAME / SANDBOX_PRESETS_SUBDIR
log_dir = output_dir / RUN_LOG_DIRNAME
```

### Branding strings

```python
from phenotypic.gui._config import (
    TITLE_HUB,       # "PhenoTypic GUI"
    TITLE_BUILDER,   # "PhenoTypic Pipeline Builder"
    TITLE_VIEWER,    # "PhenoTypic Results Viewer"
    TITLE_RUN,       # "PhenoTypic Run Console"
)

dash.Dash(..., title=TITLE_BUILDER)
```

### Threading

```python
from phenotypic.gui._config import THREAD_NAME_PREFIX  # "phenotypic-gui"

ThreadPoolExecutor(thread_name_prefix=f"{THREAD_NAME_PREFIX}-slurm")
threading.Thread(name=f"{THREAD_NAME_PREFIX}-idle-release", ...)
```

---

## Design tokens — Python and CSS share the same source

[_design.py](_design.py) is the single source of truth for **all** design
tokens. `inject_design_tokens(app)` splices a `<style>` block carrying
`--font-*`, `--color-*`, `--oi-*`, `--text-*`, `--sp-*`, `--radius-*`,
`--shadow-*`, `--ease-*`, and `--transition` into every Dash app's
`index_string`. The hub composer's `wrap_in_chrome` calls it on every
mount, and each sub-app `_app.py` calls it as well — both paths are
idempotent via a marker comment.

### Using tokens in CSS

```css
.my-card {
    background: var(--color-surface);
    color: var(--color-body);
    border: 1px solid var(--color-border);
    padding: var(--sp-3);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}
```

**DO NOT** declare a `:root { --color-* }` block inside a tool's CSS
file — `_design.py`'s injection covers all four mounts. Tool-specific
overrides are fine, but only for values truly unique to that tool. For
example, [builder/assets/builder.css](builder/assets/builder.css)
keeps `--color-interactive: var(--oi-purple)` (point-picker mixin) and
`--text-2xl: 1.875rem` (canvas titles only); both reference shared
tokens or extend the type scale and never redefine a shared value.

### Using tokens in Python inline styles

```python
from phenotypic.gui._design import (
    COLOR_BG, COLOR_BLUE, COLOR_GOLD, COLOR_MUTED, COLOR_NAVY,
    COLOR_SURFACE, OI_VERMILION,
)

html.Div(style={"color": COLOR_NAVY, "background": COLOR_SURFACE})
```

**Never** write `"color": "#003660"` inline. Pyright will not catch it,
but a future palette change will. The same goes for `f"1px solid #1b75bc"`
— write `f"1px solid {COLOR_BLUE}"`.

### Palette rules (DESIGN.md)

`COLOR_*` (navy/blue/gold/etc.) are UI-only — never use them as data
series colors. `OI_*` (Okabe-Ito) are data-only — never use them for UI
chrome. Series order is fixed: navy, orange, sky, green, blue, purple
(vermilion reserved for error / alert). Yellow may not be used as text
on white backgrounds. See [../../../DESIGN.md](../../../DESIGN.md) for
badge contrast variants and prohibited combinations.

---

## Adding a new GUI feature

1. **Update [FEATURES.md](FEATURES.md)** — every user-visible affordance
   gets a row. The `gui-e2e` workflow rejects any PR that touches
   `gui/` without modifying `FEATURES.md`.
2. **If it's an end-to-end flow worth a tutorial**, also update
   [WORKFLOWS.md](WORKFLOWS.md), add `_capture_<id>` in
   `scripts/capture_gui_tutorial_screenshots.py`, and add a walkthrough
   page under `docs/source/how_to/pages/gui_walkthrough/`. The
   `gui-docs` CI gate enforces the round-trip.
3. **Use [_config.py](_config.py)** for any new shared constant.
4. **Use [_design.py](_design.py)** for any color, type size, radius,
   shadow, or motion value.
5. **Re-run `uv run python scripts/capture_gui_tutorial_screenshots.py`**
   after any visible chrome change and commit refreshed PNGs.

---

## Common gotchas

- **`requests_pathname_prefix=url_prefix, routes_pathname_prefix=MOUNT_HOME`** — sub-apps
  mounted under `DispatcherMiddleware` see their mount prefix stripped
  before Dash routes. Standalone launches collapse to identical prefixes
  because `url_prefix` defaults to `MOUNT_HOME` ("/").
- **Don't import dash from `_config.py` or `_design.py`** — they stay
  cheap to import everywhere, including from blueprint and test code.
- **Registry keys are split by type** — the builder's
  `OperationRegistry` lives under `CFG_OPERATION_REGISTRY`
  (`"pheno_operation_registry"`) and the run console's `RunRegistry`
  lives under `CFG_RUN_REGISTRY` (`"pheno_run_registry"`). They were
  previously overloaded behind a single `pheno_registry` key; the
  rename makes the type explicit at every call site.
- **`inject_design_tokens` is idempotent** — both the sub-app `_app.py`
  factories AND `wrap_in_chrome` call it; only the first call inserts
  the `<style>` block (marker comment de-dupes).
