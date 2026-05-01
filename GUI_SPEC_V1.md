# PhenoTypic Unified GUI — v1 Specification

> **Revision history**
> - **2026-04-30 (initial):** drafted from brainstorm.
> - **2026-04-30 (post-review):** plan-reviewer findings applied. Key changes:
>   factories gain `url_prefix=` kwarg (Dash makes `requests_pathname_prefix`
>   read-only post-construction); click refactor dropped in favour of
>   `phenotypic-gui` console script; `_ViewerProxy` WSGI proxy made explicit;
>   `ToolSession` gains `touch()` + `idle_seconds()`; `RunRegistry` lock made
>   explicit; `LocalRunHandle` uses `collections.deque(maxlen=)` not `queue.Queue`;
>   SLURM job-id from `progress/job_metadata.json` not stdout; `_runs_blueprint`
>   uses `<path:file>` catch-all; ID-collision check narrowed to intra-app +
>   chrome-vs-existing; FEATURES.md enforcement clarified as two-layer (pre-commit
>   + CI); Playwright CI gated on path filters; Release-button UX copy made honest
>   about RSS retention.
>
> Living reference for the unified Dash GUI hub. Every section captures a design decision
> reached during the v1 brainstorm and includes a checklist of concrete deliverables so
> implementation progress can be tracked at a glance.
>
> **Status legend:** 🔭 planned · 🚧 in progress · ✅ shipping · ❌ cancelled / out of scope
>
> Companion documents:
> - [`src/phenotypic/gui/FEATURES.md`](src/phenotypic/gui/FEATURES.md) — per-affordance test ledger (created in Phase 0)
> - [`/Users/alex/.claude/plans/section-9-looks-good-wise-moore.md`](/Users/alex/.claude/plans/section-9-looks-good-wise-moore.md) — phase-by-phase implementation plan

---

## Section 1 — Goals, non-goals, and high-level architecture

### Goal

A unified Dash GUI launched via `python -m phenotypic.gui` (or the `phenotypic-gui`
console script) that hosts the existing pipeline builder and results viewer, plus a
new "Run pipeline" console, all under one URL. A persistent left sidebar serves as a
sandboxed file browser (the project root, set at launch). Tools share a sandbox but
track their own paths.

### Non-goals (v1)

- ❌ Cloud / multi-user deployment. SSH-tunnel-to-loopback remains the only supported
  remote pattern. A TODO note marks the future `--mode=cloud` auth gate.
- ❌ Multi-tenant / per-session state. Process-wide state (e.g., the results viewer's
  shared `FilteredMeasurements`) is preserved — collaborators sharing a tunnel still
  share state.
- ❌ Job manager (cancel / retry / persistent history). Recent Runs is read-only;
  in-flight runs are observed, not managed.
- ❌ Porting the napari sweep viewer to the browser. Stays a separate launcher.
- ❌ Surfacing the older Panel-based `PipelineBuilder`. Already removed from
  `src/phenotypic/gui/__init__.py`.

### Architecture: Flask-mount

One Werkzeug `DispatcherMiddleware` hosts the shell Dash app + each existing
`create_app()` factory under a path prefix. No rewrite of existing app internals.

```
python -m phenotypic.gui --root /scratch/alex
        │
        ▼
   shell.create_app(root)         ◄── new module: phenotypic.gui.shell
        │
        ├── shell_app.server (Flask)         (top of WSGI tree; receives all traffic)
        │     ├── @blueprint /sandbox/api/*  ── JSON API for sidebar tree + classifier
        │     ├── @blueprint /runs/<rel>/<path:file>
        │     │                              ── static, sandbox-scoped serve of output dirs
        │     └── delegates everything else to:
        │           DispatcherMiddleware
        │             ├── /          → shell Dash (sidebar + top bar + home)
        │             ├── /builder/  → builder.create_app(image_root, url_prefix="/builder/")
        │             ├── /results/  → _ViewerProxy(viewer_session)  ◄── lazy-init via ToolSession
        │             └── /run/      → run_console.create_app(sandbox, url_prefix="/run/")
        │
        ▼
  app.run(host=..., port=..., debug=...)
```

**Two routing layers, by intent.** Flask blueprints (`/sandbox/api/*`, `/runs/...`)
register directly on `shell_app.server` so they answer regardless of which Dash app the
user is currently viewing. Dash sub-apps mount under `DispatcherMiddleware`, with each
constructed with its `requests_pathname_prefix` set at construction time (Dash makes
this read-only post-construction — see Section 6).

Each Dash sub-app keeps its own callbacks, `assets_folder`, and layout. The shell Dash
app owns sidebar, top bar, and home page. The viewer is lazily instantiated through a
WSGI proxy (`_ViewerProxy`) so memory-heavy `OutputRoot.discover` is deferred until first
access and can be released later.

**Why Flask-mount over a single multi-page Dash app:** minimal refactor of the existing
builder/viewer — their `create_app()` factories gain only an optional `url_prefix=`
parameter and (for the viewer) optional `output_root=None`. They keep working as
standalone `python -m phenotypic.gui.builder` invocations because both new parameters
have safe defaults.

### Section 1 deliverables

- [ ] 🔭 Architecture documented in `docs/source/user_guide/gui.rst`
- [ ] 🔭 Cloud-deploy non-goal called out in user docs (with SSH tunnel pattern)
- [ ] 🔭 Loopback default (`127.0.0.1`) preserved across all entry points

---

## Section 2 — Module layout

New code lives under `src/phenotypic/gui/shell/` and `src/phenotypic/gui/run_console/`.
Existing builder + results_viewer dirs are untouched.

```
src/phenotypic/gui/
├── __init__.py                      ← add lazy attrs for shell.create_app + launch_gui + SandboxRoot
├── __main__.py                      ← NEW; argparse → shell.launch_gui (mirrors builder/__main__.py)
├── builder/                         ← unchanged
├── results_viewer/                  ← optional-output_root refactor (Section 6)
├── sweep/                           ← unchanged
├── shell/                           ← NEW
│   ├── __init__.py
│   ├── _app.py                      ← shell Dash factory + DispatcherMiddleware composer
│   ├── _launcher.py                 ← launch_gui() banner + app.run wrapper
│   ├── _sandbox.py                  ← SandboxRoot dataclass + safe-resolve helpers
│   ├── _classifier.py               ← classify(path) → Capabilities (img/cfg/out/dashboard/perms)
│   ├── _session.py                  ← ToolSession primitive (lazy build / release / lock)
│   ├── _runs_registry.py            ← process-wide RunRecord registry
│   ├── _runs_blueprint.py           ← /runs/<output_dir>/<file> static blueprint
│   ├── _routes.py                   ← /sandbox/api/{root,children,classify} JSON blueprint
│   ├── _layout.py                   ← top bar + sidebar + chrome-wrap helper
│   ├── _sidebar.py                  ← tree + badges + hidden/symlink toggles
│   ├── _home.py                     ← landing pane
│   ├── _release_button.py           ← reusable per-tool Release button + RSS readout
│   ├── _ids.py                      ← namespaced SHELL_*
│   ├── _callbacks.py
│   └── _assets/shell.css
└── run_console/                     ← NEW (sibling, like builder/, results_viewer/)
    ├── __init__.py
    ├── __main__.py                  ← standalone parity with builder/results_viewer
    ├── _app.py                      ← create_app(sandbox: SandboxRoot)
    ├── _layout.py                   ← form + iframe panel + log tail + recent runs
    ├── _form.py                     ← pipeline/input/output pickers + mode + advanced + slurm
    ├── _runner.py                   ← LocalRunner (Popen + stdout queue + SIGTERM-on-stop)
    ├── _slurm.py                    ← shell-out wrapper over phenotypic._cli SLURM submission
    ├── _recent_runs.py              ← classifier-driven scanner over sandbox
    ├── _state.py                    ← per-session UI scratch
    ├── _ids.py
    ├── _callbacks.py
    └── _assets/run_console.css
```

Tests follow the same pattern as the existing CLI suite:

```
tests/unit/gui/shell/
tests/unit/gui/run_console/
tests/integration/gui/                ← Flask test-client smoke (no browser)
tests/e2e/gui/                        ← Playwright; gated on PLAYWRIGHT=1 locally, REQUIRED in CI
```

### Section 2 deliverables

- [ ] 🔭 `src/phenotypic/gui/shell/` skeleton scaffolded
- [ ] 🔭 `src/phenotypic/gui/run_console/` skeleton scaffolded
- [ ] 🔭 `src/phenotypic/gui/__main__.py` argparse wrapper
- [ ] 🔭 Test directories `tests/unit/gui/{shell,run_console}/` + `tests/integration/gui/` + `tests/e2e/gui/`
- [ ] 🔭 `pytest-playwright` + `playwright` added to dev dependency group
- [ ] 🔭 `gui-e2e` CI job stub

---

## Section 3 — Sandbox & file browser

### `SandboxRoot` (the security primitive)

A frozen dataclass constructed once at launch; every callback that reads a path goes
through it. Centralizing this is the most important thing for not-screwing-up a future
cloud-deploy mode.

```python
@dataclass(frozen=True)
class SandboxRoot:
    root: Path                       # absolute, resolved at launch

    def resolve(self, p: str | Path) -> Path:
        """Resolve user-supplied path; raise ValueError if it escapes root."""
        candidate = (self.root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise ValueError(f"Path {candidate} is outside sandbox {self.root}") from e
        return candidate

    def contains(self, p: Path) -> bool: ...
    def list_children(self, p: Path, *, show_hidden: bool, follow_external_symlinks: bool = False) -> list[Path]: ...
```

**Rules:**
- Built once in `shell._launcher.launch_gui(...)`. Stashed on `app.server.config["sandbox_root"]`.
- Symlinks pointing **outside** the root: hidden by default. A toggle in the sidebar
  reveals them as inert (greyed; refused on click). Existing `_directory_browser.py`
  logic reused.
- Hidden files / dotfiles: hidden by default; sidebar toggle reveals them.
- The Run console's pipeline/input/output pickers all funnel through
  `SandboxRoot.resolve()`. Same for the `/runs/...` static blueprint and
  `/sandbox/api/...` JSON routes.
- TODO comment at the top of `_sandbox.py`: *"Cloud-deploy hardening: when
  `--mode=cloud` ships, add a per-request auth check + per-user effective root here.
  Don't bolt auth onto callbacks individually."*

### Capability classifier

Pure function, file-stat only, no parsing of file contents.

```python
@dataclass(frozen=True)
class Capabilities:
    is_image_dir: bool        # has ≥1 file with extension in IMAGE_EXTS
    has_pipeline_json: bool   # has ≥1 *.json with top-level "operations" (cheap 4KB peek)
    is_cli_output: bool       # has BOTH master_measurements.parquet AND results/ dir
    has_dashboard: bool       # has dashboard.html (Recent Runs uses this to enable iframe)
    image_count: int | None   # only filled for is_image_dir; capped at 1000
    bad_perms: bool           # PermissionError on listdir

def classify(path: Path) -> Capabilities: ...
```

- Stat-only; no parquet open, no image decode. Cheap → can run on every visible
  sidebar node + every Recent Runs row.
- LRU-cached per `(path, mtime)`. Cache invalidated by a manual "Refresh" button (no
  filesystem watcher in v1 — too OS-specific).
- Pipeline-json detection peeks at the first 4 KB and looks for `"operations"`. Cheap
  + good enough for a UI badge.

### Sidebar component

- Tree built lazily (one level per expand, like the existing `_directory_browser.py`).
- Each node renders: `📁 name` + capability badges:
  - `img` (blue, matches builder's stage colour) — directory has images
  - `cfg` (gold) — file is a pipeline JSON
  - `out` (green) — directory looks like a CLI output (parquet + results/)
  - `?` (grey) — `bad_perms`
- Toggles at top: `[☐ Hidden files] [☐ External symlinks]` (defaults off).
- "Refresh" button next to toggles flushes the classifier cache.
- Sidebar selection is a `dcc.Store`. It does NOT auto-route to a tool. The
  currently-active tool's form has a small `[↩ from sidebar]` button that pulls the
  selection into its own per-tool path state. Sidebar is a navigator, not a global
  selection.

### JSON API for the tree

| Route | Returns |
|---|---|
| `GET /sandbox/api/root` | `{root, mtime, badges}` for the root |
| `GET /sandbox/api/children?path=<rel>&hidden=0&symlinks=0` | `[{name, type, rel_path, badges}]` |
| `GET /sandbox/api/classify?path=<rel>` | `Capabilities` JSON for one path |

**Registration.** These are Flask blueprints registered directly on `shell_app.server`
(NOT placed under `DispatcherMiddleware` mounts). Reachable from any page because the
Werkzeug listener routes any unmatched URL to the top-level Flask app. Reasons:
(a) sidebar JS fetches them async without round-tripping through Dash's callback
system, (b) easy to test independently with the Flask test client, (c) cloud-deploy
auth can wrap them with a single `@before_request` later — touching one location
rather than every Dash callback.

### Static blueprint for output dirs

```
/runs/<rel>/<path:file>          ← Flask <path:> catch-all so nested dashboard polls work
```

The dashboard's internal JS polls `progress/manifest.json` and `progress/failures.jsonl`
using **relative URLs**. When `dashboard.html` is iframed at
`/runs/plate_2026-04/output/dashboard.html`, the browser resolves polls to
`/runs/plate_2026-04/output/progress/manifest.json` — only reachable if the route
parameter is a `<path:file>` catch-all (multi-segment). A single-segment `<file>`
parameter would silently 404 the polling requests and the dashboard would render but
appear frozen.

Like the sandbox API, this blueprint registers on `shell_app.server`, not under
`DispatcherMiddleware`. Every request runs `SandboxRoot.resolve(...)` first and
rejects path traversal with a 404.

### Section 3 deliverables

- [ ] 🔭 `SandboxRoot` dataclass with `resolve` / `contains` / `list_children`
- [ ] 🔭 Out-of-root + symlink-escape paths raise `ValueError`
- [ ] 🔭 TODO comment at top of `_sandbox.py` for cloud-deploy auth gate
- [ ] 🔭 `Capabilities` dataclass + `classify()` function
- [ ] 🔭 LRU cache on classifier keyed by `(path, mtime)`
- [ ] 🔭 4KB pipeline-json peek (`"operations"` key)
- [ ] 🔭 `_runs_blueprint.py` serves `/runs/<dir>/<file>` with sandbox containment
- [ ] 🔭 `_routes.py` exposes `/sandbox/api/{root,children,classify}`
- [ ] 🔭 Sidebar tree (lazy expand, capability badges, hidden/symlinks toggles, Refresh)
- [ ] 🔭 Sidebar selection store + `[↩ from sidebar]` hand-off pattern

---

## Section 4 — Shell chrome (top bar, navigation, per-tool path state, lifecycle)

### Top bar

```
[ « ] [ PhenoTypic GUI ] [ root: /scratch/alex ]   …   [ Home ] [ Builder ] [ Viewer ] [ Run ]   …   [ ⓘ ]
```

- **Title + root display** (left). Root path is read-only; clicking copies to clipboard.
  No "Change root" button — sandbox is frozen at launch.
- **Tab nav** (centre). Anchors: `/`, `/builder/`, `/results/`, `/run/`. Real
  navigations between mounted sub-apps. Active tab set server-side from `flask.request.path`.
- **Help button** (right). Modal: SSH-tunnel reminder, "Clear classifier cache", link
  to docs, version info.
- **RSS readout** (top-right corner). Updates every 5s; reads `psutil.Process().memory_info().rss`.

Chrome is rendered by *each* sub-app's layout, not a parent frame.
`shell/_layout.py` exports `build_top_bar(active_tab)` and `build_sidebar(...)`. The
shell composer post-processes each mounted Dash app's `app.layout` once, wrapping it
in chrome — without touching their `_layout.py` source files. Standalone launches
(`python -m phenotypic.gui.builder`) bypass wrapping.

**`wrap_in_chrome` is BOTH a layout mutator AND a callback registrar.** The chrome
contains interactive elements (RSS readout `dcc.Interval`, sidebar refresh button,
release-button click handlers) whose callbacks must register on the *specific* Dash
app being wrapped — each mounted Dash instance has its own callback dispatch, so the
chrome callbacks must be registered on each one separately. This is not a workaround;
it's how Dash multi-app composition works.

```python
# shell/_layout.py
def wrap_in_chrome(app: dash.Dash, *, active_tab: str, sandbox: SandboxRoot) -> None:
    """Mutate app.layout in place AND register chrome callbacks on app."""
    body = app.layout
    app.layout = html.Div([
        build_top_bar(active_tab=active_tab, sandbox=sandbox),
        html.Div([
            build_sidebar(sandbox=sandbox),
            html.Main(body, id=ids.SHELL_MAIN_PANE),
        ], className="shell-body"),
        dcc.Store(id=ids.SHELL_SIDEBAR_SELECTION_STORE),
        dcc.Store(id=ids.SHELL_CLASSIFIER_CACHE_STORE),
        dcc.Interval(id=ids.SHELL_RSS_INTERVAL, interval=5_000),
    ])
    _register_chrome_callbacks(app, sandbox)   # RSS readout, sidebar refresh, release click

def _register_chrome_callbacks(app: dash.Dash, sandbox: SandboxRoot) -> None:
    @app.callback(Output(ids.SHELL_RSS_LABEL, "children"),
                  Input(ids.SHELL_RSS_INTERVAL, "n_intervals"))
    def _update_rss(_): ...
    # … sidebar refresh, release-button handlers …
```

All chrome IDs live under a `SHELL_*` namespace in `shell/_ids.py` to avoid colliding
with the existing apps' IDs.

### Per-tool path state

Each tool's pickers are completely independent. State lives in tool-local `dcc.Store`s:

- Builder: `builder-image-root`
- Viewer: `viewer-output-root`
- Run: `run-pipeline-path`, `run-input-dir`, `run-output-dir`

Tools never *read* the sidebar selection as their source of truth; instead, each
tool's picker has a small `[↩ from sidebar]` button that copies the current sidebar
selection into the tool's own store. Switching tabs preserves each tool's state.

### Lifecycle (memory-conscious; agreed mid-brainstorm)

**Constraint of Flask-mount:** all sub-apps share one Python process. We can drop
Python references (so the GC reclaims `master_df`, etc.), but we **can't guarantee
RSS returns to the OS** — Python's allocators (especially pandas/numpy pools) tend
to hold on. "Release" in v1 means *logical release* (next access re-loads from disk),
not *OS-level memory return*.

**Lazy-init per tool.** Heavy state isn't built at hub boot:

```python
class ToolSession:
    def __init__(self, name: str, *, build, teardown=lambda _: None):
        self.name = name
        self._build = build
        self._teardown = teardown
        self._state = None
        self._last_access: float = 0.0          # monotonic timestamp; idle timer reads this
        self._lock = threading.Lock()

    def get(self):                              # lazy build on first access
        with self._lock:
            if self._state is None:
                self._state = self._build()
            self._last_access = time.monotonic()
            return self._state

    def touch(self) -> None:                    # bump activity without forcing build
        self._last_access = time.monotonic()    # called by /runs/ blueprint, /sandbox/api/, etc.

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_access if self._state is not None else 0.0

    def release(self) -> None:                  # drop refs; next get() rebuilds
        with self._lock:
            if self._state is not None:
                self._teardown(self._state)
                self._state = None
                gc.collect()
```

The viewer's callbacks fetch state via
`current_app.config["viewer_session"].get()`. First click on `/results/` triggers the
load; the rest of the GUI stays light at boot.

**Why `touch()` matters.** Iframe viewing (the dashboard polling its progress files)
hits the `/runs/` blueprint, not Dash callbacks. Without an explicit `touch()` call
from the blueprint, the idle timer would release tools while the user is actively
viewing them. The sandbox API blueprint and the runs blueprint both call
`session.touch()` for any session that should remain "live" while their routes are hit.

**Idle timer implementation.** A single background daemon thread runs every 60s,
walks all registered sessions, and calls `release()` on any session whose
`idle_seconds() > idle_release_minutes * 60`. It does NOT use a one-shot
`threading.Timer` (which races with `touch()` resets); it polls monotonic time. The
release acquires the same lock as `get()` so an in-flight `get()` is never
mid-rebuild during a release.

**Manual Release button per tool.** Top-right of each tool's main pane. Button label:
**"Release loaded data"**. Tooltip / inline help text:
*"Drops Python references to this tool's loaded state (e.g., parquet, curation,
intermediate caches). The next visit rebuilds from disk. NOTE: process RSS may stay
elevated even after release because Python's allocators retain freed memory pages.
For hard memory bounds, restart the GUI."* The current-RSS readout next to the button
shows the *process-wide* RSS (not the per-tool delta) — the user sees the truth.

**Idle auto-release.** After **N minutes of no callback activity** for a tool, its
session releases automatically. Default N = 10, configurable via `--idle-release-minutes`.
The Run console is **exempt** — see below.

**Tab-lifecycle hooks deferred to v1.5.** Browser-`pagehide` + `sendBeacon` +
reference-counting is the cleanest "release the moment you navigate away" approach,
but adds enough complexity (multi-tab refcounting, race conditions) that v1 ships
with manual button + idle timer only.

**Run console is split into "UI" and "registry".** Subprocess monitoring lives on
the shell, not the run console session. The registry persists across UI release/rehydrate.

```
shell/_runs_registry.py    ── process-wide singleton on app.server.config
                              { run_id: { mode, output_dir, pid|slurm_id, status, started_at, log_path } }
                              survives Run console UI release

run_console/_app.py        ── UI layer; reads from registry
                              session.release() drops UI scratch; registry untouched
```

So: navigating away from `/run/` (or hitting Release) drops form-state + the in-memory
log buffer, **but** subprocesses stay alive (shell-owned children of the GUI process),
the registry keeps tracking them, Recent Runs still shows them. Coming back rehydrates
the UI from the registry.

For SLURM submissions, the registry holds the `slurm_array_id`; status updates come
from polling the existing CLI's progress files. SLURM jobs survive even GUI restarts
(registry is rebuilt from sandbox scan on boot).

### Killing the GUI

- **Local runs** are subprocess children. `Ctrl-C` on the GUI sends `SIGTERM` to
  children → they die. Confirmation dialog if GUI exits while local runs are in flight.
- **SLURM runs** are detached at submission. GUI exit → SLURM keeps going → next
  launch reads them from sandbox.
- Optional v1.5: `--detach-local-runs` for `nohup`/double-fork. Adds zombie-reaping
  complexity. Deferred.

### Empty / error states

| Situation | Behaviour |
|---|---|
| `--root` doesn't exist | `launch_gui` exits with clear error before booting Flask |
| `--root` is empty | Sidebar shows "Empty directory. Drop images or pipeline.json files here." |
| External symlink with reveal-toggle on | Disabled-looking node; click → toast: "External symlinks aren't followed." |
| `bad_perms` folder | Toast: "Can't read this directory (permission denied)." |
| Hand-off button when sidebar selection is empty | Button disabled |

### Section 4 deliverables

- [ ] 🔭 Top bar with root display + tab nav + RSS readout + help modal
- [ ] 🔭 `wrap_in_chrome(app)` mutates `app.layout` AND registers chrome callbacks (RSS interval, sidebar refresh, release click) on `app`
- [ ] 🔭 Per-tool path state stores (`builder-image-root`, `viewer-output-root`, `run-*`)
- [ ] 🔭 `[↩ from sidebar]` hand-off buttons in each tool's pickers
- [ ] 🔭 `ToolSession` primitive with `get` / `touch` / `release` / `idle_seconds` / threading lock
- [ ] 🔭 Manual Release button per tool with **honest UX copy** ("Release loaded data" + tooltip about RSS retention)
- [ ] 🔭 Idle auto-release: single daemon thread polling monotonic time every 60s; configurable via `--idle-release-minutes` (default 10)
- [ ] 🔭 `/runs/` and `/sandbox/api/` blueprints call `session.touch()` to reset idle timers
- [ ] 🔭 `RunRegistry` singleton with `threading.Lock` on `app.server.config`
- [ ] 🔭 Registry rehydrates from sandbox scan on boot
- [ ] 🔭 GUI-shutdown hook SIGTERMs in-flight local runs (with confirmation dialog)
- [ ] 🔭 Empty/error state copy

---

## Section 5 — Run console internals

### Five flows

```
A. Local run        : form → spawn subprocess → stream stdout → iframe dashboard.html when ready
B. SLURM submit     : form → call existing _cli SLURM submission → register job-id → iframe dashboard.html
C. Dry-run          : form → spawn subprocess with --dry-run → stream stdout (no iframe)
D. Recent run open  : sidebar/list click → re-point iframe at <output_dir>/dashboard.html
E. Tab unmount     : navigate away → UI session releases; registry + subprocesses survive
```

### `_runner.py` — local subprocess wrapper

```python
class LocalRunHandle:
    run_id: str                 # e.g. "local_2026-04-30_12-04-01"
    pid: int
    output_dir: Path
    started_at: datetime
    log_path: Path                    # <output_dir>/.gui_log/stdout.log (tee'd from PIPE)
    proc: subprocess.Popen
    stdout_buffer: collections.deque  # maxlen=5000; true ring buffer (drops oldest on overflow)
    buffer_lock: threading.Lock       # serializes writer thread vs UI reader

class LocalRunner:
    def start(self, *, sandbox, pipeline, input_dir, output_dir, options) -> LocalRunHandle: ...
    def stop(self, run_id: str) -> None:                  # SIGTERM, then SIGKILL after 10s
    def tail(self, run_id: str, since: int) -> list[str]: # ring-buffer slice for the UI poll
```

- `Popen` with `stdout=PIPE, stderr=STDOUT, bufsize=1, text=True`. A daemon thread per
  run reads lines, appends to `log_path` on disk, and pushes to `stdout_buffer` under
  `buffer_lock`.
- **`collections.deque(maxlen=5000)` is used deliberately, not `queue.Queue`.** A
  bounded `queue.Queue` with `put()` blocks the reader thread when full → backs up
  the subprocess's stdout pipe → can deadlock the subprocess. A `deque(maxlen=N)`
  drops the oldest line silently on overflow, which is the desired ring-buffer
  behaviour for a tail panel. The on-disk `log_path` is the canonical record; the
  in-memory buffer is just for the UI panel.
- `output_dir` is resolved through `SandboxRoot.resolve(...)` and verified writable
  before spawn.
- `stop()` sends `SIGTERM`, waits 10s, then `SIGKILL`. Registry status flips to `cancelled`.
- `atexit` hook SIGTERMs all live handles on GUI shutdown.

### Dashboard iframe wiring

```
/runs/<output_dir>/<file...>  ← Flask blueprint, sandbox-scoped
```

After `LocalRunner.start()`:

1. UI sets `dcc.Store` with `iframe_target: pending`.
2. Polling callback (every 500ms, max 10s) checks `os.path.exists(output_dir/'dashboard.html')`.
3. As soon as it exists, store flips to `iframe_target: /runs/<rel>/dashboard.html`.
4. Clientside callback writes the iframe `src` from the store (avoids re-render churn).
5. If dashboard doesn't appear in 10s, iframe stays empty and the live log panel does
   the work — usually means the run died before reaching `generate_dashboard()`.

### `postMessage` channel — the one upgrade door

Inside `dashboard.html`, ~10 lines of JS guarded by `if (window.parent !== window)`:

```js
window.parent.postMessage({type:"phenotypic:select-failure", dataset, image_file}, "*");
```

The shell installs a window-level listener on the parent page that:
- Validates origin (`location.origin`, or `null` when iframe is same-document).
- Validates message shape.
- Dispatches via `dcc.Store(id="shell-iframe-event")` → Dash callback.

For v1 we **write** the listener (the door is open) but **don't wire** any
Builder/Viewer routing yet — we just log the event. That keeps the hook tested
without committing to a specific cross-app workflow.

### `_slurm.py` — SLURM submission

Thin shell-out wrapper over the existing `phenotypic._cli._cli_slurm_submission`
pathway. **Critical:** invoke via subprocess (`python -m phenotypic … --slurm k=v …`),
do NOT import CLI internals — ensures GUI-submitted SLURM runs are indistinguishable
from hand-typed CLI submissions.

The Run form's "SLURM mode" panel collects:

| Typed fields (most common) | Free-form `k=v` table |
|---|---|
| partition, account, mem-gb, time-limit, array-size, cpus-per-task | anything else (e.g. `gres=gpu:1`, custom modules) |

**Job-id resolution.** The CLI already writes
`<output_dir>/progress/job_metadata.json` containing structured `chunk_job_ids` keyed
by chunk index. After the submission subprocess exits, `_slurm.py` reads this file
and extracts `chunk_job_ids["0"]` (or the full dict for multi-chunk array jobs). This
is the authoritative job-id source — **do NOT parse the CLI's Rich-formatted stdout**
(format may change; locale/terminal-width can break regexes; structured file is
canonical).

Submission returns a `SlurmRunHandle` with `slurm_array_id` (or `slurm_job_ids` for
multi-chunk runs). Status polls the manifest in `<output_dir>/progress/` (same source
the dashboard polls). If `job_metadata.json` is absent after the submitter exits with
non-zero, the registry status is set to `failed` and the submitter's stderr is
surfaced to the user as the failure reason.

### `_recent_runs.py` — Recent runs panel

- On sandbox boot, walk `<root>` to depth `--scan-depth` (default 1; configurable for
  nested layouts).
- Run classifier on each directory; keep ones with `is_cli_output`.
- Sort by mtime desc; cap at 50 entries.
- Each row: name, mtime, status badge from `manifest.json` (`done` / `failed N` /
  `in progress` / `unknown`).
- Click → set `iframe_target` to `/runs/<rel>/dashboard.html`.
- "Refresh" button rescans + flushes classifier cache for output dirs.

### `_runs_registry.py` — process-wide registry

```python
@dataclass
class RunRecord:
    run_id: str
    mode: Literal["local", "slurm"]
    output_dir: Path
    started_at: datetime
    pid: int | None
    slurm_array_id: str | None
    status: Literal["running", "done", "failed", "cancelled", "unknown"]
    log_path: Path | None     # local mode only

class RunRegistry:
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = threading.Lock()    # serializes register / update_status / list

    def register(self, record: RunRecord) -> None:
        with self._lock:
            self._records[record.run_id] = record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def list(self) -> list[RunRecord]:
        with self._lock:
            return list(self._records.values())

    def update_status(self, run_id: str, status: str) -> None:
        with self._lock:
            if run_id in self._records:
                self._records[run_id] = replace(self._records[run_id], status=status)
```

The lock is mandatory — multiple Dash callback threads (and the runner's daemon
thread, which updates status on subprocess exit) concurrently mutate the registry.


### Save-as-preset

Stores form state (sans paths) to `<root>/.phenotypic-gui/presets/<name>.json`.
Loading fills the form except for paths. No schema versioning yet (v1 ships single schema).

### Concurrency limits

v1 caps **simultaneous local runs at 1** (configurable via `--max-local-runs`,
default 1). SLURM submissions are unlimited. The "Run" button is disabled while a
local run is active; SLURM submit always available.

### Section 5 deliverables

- [ ] 🔭 `LocalRunner` (Popen + `collections.deque(maxlen=5000)` ring buffer + SIGTERM-then-SIGKILL + atexit)
- [ ] 🔭 `LocalRunHandle.buffer_lock` serializes writer thread vs UI reader
- [ ] 🔭 Stdout tee'd to `<output_dir>/.gui_log/stdout.log` (canonical record on disk)
- [ ] 🔭 Iframe-target polling (500ms × 10s) for `dashboard.html`
- [ ] 🔭 Clientside callback writes iframe `src`
- [ ] 🔭 `postMessage` listener registered on parent (logs only in v1)
- [ ] 🔭 ~10 lines of guarded JS added to `_cli/_dashboard/_generator.py`
- [ ] 🔭 `_slurm.py` subprocess wrapper (no CLI internals imported)
- [ ] 🔭 SLURM job-id parsed from `<output_dir>/progress/job_metadata.json` (NOT stdout)
- [ ] 🔭 SLURM form: typed common fields + free-form `k=v` rows
- [ ] 🔭 Recent Runs scanner with classifier-driven status badges
- [ ] 🔭 Recent Runs row click re-points iframe
- [ ] 🔭 `RunRegistry` with explicit `threading.Lock` on register / get / list / update_status
- [ ] 🔭 Registry rehydration from sandbox scan on boot
- [ ] 🔭 Save-as-preset to `<root>/.phenotypic-gui/presets/`
- [ ] 🔭 `--max-local-runs` cap (default 1)
- [ ] 🔭 Validate/Dry-run button (no iframe; log only)
- [ ] 🔭 Cancel button (SIGTERM + SIGKILL)

---

## Section 6 — Builder + Viewer integration

The headline: **minimal but real changes** to the existing factories. Each
`create_app()` factory gains an optional `url_prefix=` parameter (Dash makes
`requests_pathname_prefix` read-only after construction, so it MUST be set at
constructor time). Standalone launchers continue to work because `url_prefix="/"` is
the default. The viewer additionally accepts `output_root: OutputRoot | None` so the
shell can lazy-instantiate it through a `ToolSession`.

### Factory signature changes

```python
# builder/_app.py — minimal change
def create_app(image_root: Path | None = None, *,
               registry: OperationRegistry | None = None,
               url_prefix: str = "/") -> dash.Dash:
    app = dash.Dash(__name__, requests_pathname_prefix=url_prefix,
                              routes_pathname_prefix=url_prefix,
                              suppress_callback_exceptions=True, ...)
    ...

# results_viewer/_app.py — slightly larger change
def create_app(output_root: OutputRoot | None = None, *,
               url_prefix: str = "/") -> dash.Dash:
    app = dash.Dash(__name__, requests_pathname_prefix=url_prefix,
                              routes_pathname_prefix=url_prefix,
                              suppress_callback_exceptions=True, ...)
    if output_root is None:
        # Empty-state pathway — chrome-only layout w/ "pick a directory" hand-off.
        app.layout = build_empty_state_layout()
        return app
    # Normal path (unchanged):
    app.server.config["output_root"] = output_root
    _tile_routes.register(app, output_root)
    filtered_state = FilteredMeasurements.load(output_root.root, output_root.master_df)
    app.server.config["filtered_state"] = filtered_state
    colony_crop_routes.register(app, output_root)
    app.layout = build_app_layout(output_root, filtered_state)
    register_callbacks(app, output_root)
    return app
```

The viewer's empty-state pathway is **four conditional branches** (skip
`_tile_routes.register`, skip `FilteredMeasurements.load`, skip
`colony_crop_routes.register`, swap layout). Small but real.

### What the shell does at boot

```python
# shell/_app.py
def create_app(sandbox: SandboxRoot) -> dash.Dash:
    # Construct each Dash app with its prefix set at constructor time.
    builder_app = builder.create_app(image_root=sandbox.root, url_prefix="/builder/")
    run_app     = run_console.create_app(sandbox=sandbox,    url_prefix="/run/")

    # The viewer is lazy: its factory is called only on first /results/ request.
    def _build_viewer():
        from phenotypic.gui.results_viewer import create_app as _viewer_create
        viewer_app = _viewer_create(output_root=None, url_prefix="/results/")
        wrap_in_chrome(viewer_app, active_tab="results", sandbox=sandbox)
        return viewer_app

    def _teardown_viewer(viewer_app):
        # Clear refs to OutputRoot/FilteredMeasurements held on the viewer's Flask config.
        viewer_app.server.config.pop("filtered_state", None)
        viewer_app.server.config.pop("output_root", None)

    viewer_session = ToolSession("viewer", build=_build_viewer, teardown=_teardown_viewer)

    shell_app = _build_shell_dash(sandbox)
    wrap_in_chrome(builder_app, active_tab="builder", sandbox=sandbox)
    wrap_in_chrome(run_app,     active_tab="run",     sandbox=sandbox)

    # Stash sessions + sandbox on the shell's Flask config for callbacks/blueprints.
    shell_app.server.config["sandbox_root"]   = sandbox
    shell_app.server.config["viewer_session"] = viewer_session
    shell_app.server.config["runs_registry"]  = RunRegistry()
    shell_app.server.config["session_registry"] = [viewer_session]   # for idle-release thread

    # Flask blueprints register on shell_app.server (NOT under DispatcherMiddleware):
    _routes.register_sandbox_api(shell_app.server, sandbox)
    _runs_blueprint.register(shell_app.server, sandbox, viewer_session)

    # Sub-apps mount under DispatcherMiddleware:
    shell_app.server.wsgi_app = DispatcherMiddleware(shell_app.server.wsgi_app, {
        "/builder": builder_app.server,
        "/results": _ViewerProxy(viewer_session),     # lazy WSGI proxy
        "/run":     run_app.server,
    })

    _start_idle_release_thread(viewer_session, idle_release_minutes=10)
    _assert_no_id_collisions(builder_app, run_app)    # see narrowed scope below
    return shell_app
```

### `_ViewerProxy` — the load-bearing piece of the lifecycle

The DispatcherMiddleware mounts dict accepts any WSGI callable. To make
`session.release()` + rebuild work transparently, we use a tiny proxy that delegates
to whatever the session currently holds:

```python
# shell/_app.py
class _ViewerProxy:
    """WSGI proxy that resolves the viewer Dash app on every request.

    DispatcherMiddleware caches the mounts dict's values once. If we put the viewer's
    Flask server directly there, it would go stale after `session.release()` rebuilds
    a new Dash instance with a new server. The proxy resolves `session.get().server`
    per-request, transparently picking up the rebuilt instance.
    """
    def __init__(self, session: ToolSession) -> None:
        self._session = session

    def __call__(self, environ, start_response):
        viewer_app = self._session.get()      # lazy build on first hit; rebuild after release
        return viewer_app.server.wsgi_app(environ, start_response)
```

This is the entire mechanism. Without it, the lifecycle design doesn't work — the
DispatcherMiddleware would still be pointed at the released-and-stale viewer.

### `_wrap_layout_in_chrome` — what it does and doesn't touch

It mutates `app.layout` exactly once at runtime, replacing it with a chrome-wrapped
tree. It does **not** touch the existing app's `_callbacks.py`, `_ids.py`, or
`_layout.py` source files. Wrapping happens at runtime, after `create_app()`
returns. Standalone launchers skip wrapping by calling `create_app()` directly
without the shell.

### `requests_pathname_prefix`

Each Dash app needs to know where it's mounted so it builds correct asset URLs.
Shell sets these on existing apps post-instantiation. Each app's `assets_folder`
(already set to its own `_assets/`) is unaffected — `dash.Dash` namespaces asset
URLs under the prefix automatically.

### ID collision check (intra-app + chrome-vs-existing)

Each mounted Dash app has its own callback dispatch namespace. Two apps using the
same id (e.g., both having `"main-pane"`) is **legitimate** — they live in different
Dash instances and never compete. So the check is NOT cross-app; it has two narrower
duties:

1. **Intra-app:** for each app, no id appears twice in its own layout tree.
2. **Chrome-vs-existing:** before `wrap_in_chrome` injects shell IDs (the
   `SHELL_*` namespace), assert none of those IDs are already present in the app's
   pre-wrap layout. Catches the case where the builder accidentally added a
   `top-bar-rss` id that would conflict with the chrome's own callback target.

```python
def _assert_no_id_collisions(*apps: dash.Dash) -> None:
    for app in apps:
        intra = collections.Counter()
        for c in _walk_components(app.layout):
            if hasattr(c, "id") and c.id:
                intra[c.id] += 1
        dupes = [k for k, v in intra.items() if v > 1]
        if dupes:
            raise RuntimeError(f"intra-app id collisions in {app}: {dupes}")

def _assert_chrome_ids_clear(app: dash.Dash) -> None:
    """Called BEFORE wrap_in_chrome on each app."""
    pre_wrap_ids = {c.id for c in _walk_components(app.layout) if hasattr(c, "id") and c.id}
    chrome_ids = set(SHELL_ID_REGISTRY)   # exported from shell/_ids.py
    overlap = pre_wrap_ids & chrome_ids
    if overlap:
        raise RuntimeError(f"app {app.title!r} uses chrome-reserved ids: {overlap}")
```

Both fail loudly at boot rather than at runtime with mysterious callback misfires.

### Builder-specific notes

- `image_root` already accepted by `builder.create_app(image_root=...)`. Shell passes
  `sandbox.root`. Builder keeps its own `_directory_browser.py` for in-page picks.
- `OperationRegistry` is small and cheap; no `ToolSession` needed.
- New affordance: image-source picker gains a `[↩ from sidebar]` button.

### Viewer-specific notes

- `OutputRoot.discover(...)` is heavy (loads `master_measurements.parquet`). This is
  what motivated the lifecycle work; viewer is the one tool wrapped in `ToolSession`.
- **Refactor:** `results_viewer.create_app(output_root: OutputRoot | None)`. When
  mounted via shell, no output is set initially → empty-state pane with
  `[↩ from sidebar]` hand-off. When user picks one, `ToolSession` calls
  `OutputRoot.discover(...)` and rebuilds heavy state.
- Standalone launch (`python -m phenotypic.gui.results_viewer --output-root /some/path`)
  still works — it provides explicit path; shell case uses empty-state.
- Viewer's `[Release]` button drops loaded `master_df` + `FilteredMeasurements`. Next
  access re-runs `OutputRoot.discover`.

### Run console doesn't need a `ToolSession`

UI scratch is small; heavy/long-lived state is the `RunRegistry` on the shell. Run
console gets a `[Release]` button too, but it just clears log buffer + form scratch.
Registry — and therefore in-flight runs — untouched.

### What we don't change

- No changes to either app's `_callbacks.py`. No changes to `_ids.py`. No changes to
  layouts beyond the small additions noted (Builder's hand-off button, Viewer's
  empty-state path).
- No changes to existing tests. New shell tests validate composition.
- The Builder and Viewer's existing modal file pickers (`_modal_browser.py`) stay for
  in-page picks. Shell sidebar is additive.

### Section 6 deliverables

- [ ] 🔭 `builder.create_app(image_root, *, url_prefix="/")` accepts `url_prefix`
- [ ] 🔭 `results_viewer.create_app(output_root=None, *, url_prefix="/")` accepts both
- [ ] 🔭 Each sub-app constructs its `dash.Dash` with `requests_pathname_prefix=url_prefix` at the constructor
- [ ] 🔭 `DispatcherMiddleware` composition in `shell/_app.py` mounts `/builder/`, `/results/` (proxy), `/run/`
- [ ] 🔭 `_ViewerProxy` WSGI proxy class implemented + lazy-build via `ToolSession`
- [ ] 🔭 `wrap_in_chrome(app, ...)` mutates `app.layout` AND registers chrome callbacks on `app`
- [ ] 🔭 `_assert_no_id_collisions` (intra-app only) runs at boot
- [ ] 🔭 `_assert_chrome_ids_clear(app)` runs before each `wrap_in_chrome` call
- [ ] 🔭 `results_viewer.create_app(output_root=None)` empty-state pathway (4 conditional branches)
- [ ] 🔭 Viewer teardown clears `app.server.config["filtered_state"]` and `["output_root"]` before drop
- [ ] 🔭 Builder image-source picker `[↩ from sidebar]` button
- [ ] 🔭 Viewer empty-state main pane + hand-off button
- [ ] 🔭 Existing standalone launchers (`python -m phenotypic.gui.{builder,results_viewer}`) unchanged with `url_prefix="/"` default

---

## Section 7 — Entry points & CLI integration

### Three callable surfaces, one underlying function

```python
# shell/_launcher.py
def launch_gui(
    root: Path | str = Path.cwd(),
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    idle_release_minutes: int = 10,
    max_local_runs: int = 1,
    scan_depth: int = 1,
) -> None:
    """Validate root, build sandbox, compose Flask, banner, app.run."""
```

**Surface 1 — `python -m phenotypic.gui` (canonical module entry):**
- New `src/phenotypic/gui/__main__.py` with argparse → `launch_gui(...)`
- Mirrors the structure of `builder/__main__.py` and `results_viewer/__main__.py`

**Surface 2 — `phenotypic-gui` (console script, hyphenated):**

Plan-reviewer verified that converting `phenotypic_cli` from `@click.command` to
`@click.group(invoke_without_command=True)` **cannot preserve the existing
`uv run phenotypic pipeline.json input/ --dry-run` invocation pattern** without a
non-trivial subclass of `click.Group` overriding `parse_args`. Click groups treat
tokens after positional args as subcommand names; `allow_interspersed_args=True`
fixes that but causes the `gui` token itself to be absorbed into the
`pipeline_json` positional slot, breaking subcommand dispatch.

**Decision:** the click refactor is dropped. The hyphenated `phenotypic-gui`
console script is the primary (and only) standalone script entry. Add to
`pyproject.toml`:

```toml
[project.scripts]
phenotypic     = "phenotypic.phenotypicCLI:phenotypic_cli"
phenotypic-gui = "phenotypic.gui.shell._launcher:main"      # new
```

Rationale:
- Zero regression risk to the existing CLI. `phenotypic` keeps its current
  `@click.command` semantics; every existing invocation pattern still works.
- One-line change to `pyproject.toml`.
- Users get two short commands that do exactly one thing each, which is
  arguably clearer than overloading `phenotypic` with subcommands.
- Documentation must explicitly note that `phenotypic gui` (without hyphen, as a
  subcommand of `phenotypic`) is **not supported** and explain why (the click
  parser conflict above).

**Surface 3 — Python API (programmatic):**
```python
from phenotypic.gui import launch_gui
launch_gui(root="/scratch/alex", port=8051)        # blocks until Ctrl-C
```
Exported from `phenotypic.gui.__init__.py` via the existing lazy-`__getattr__` pattern.

### Banner

```
PhenoTypic GUI
  root          : /scratch/alex
  url           : http://127.0.0.1:8050/
  pages         : /  /builder/  /results/  /run/
  ssh tunnel    : ssh -L 8050:localhost:8050 user@cluster
  cache nuke    : rm -rf /scratch/alex/.phenotypic-gui
```

### Standalone launchers preserved

- `python -m phenotypic.gui.builder` — unchanged. Boots only the builder, no chrome wrapping.
- `python -m phenotypic.gui.results_viewer --output-root <path>` — unchanged. Boots
  only the viewer with eager output-root loading.
- `python -m phenotypic.gui.run_console` — **NEW**, parity with the others. Useful for
  dev iteration on the Run page without rebuilding the entire shell.

### `[project.scripts]` after this change

```toml
[project.scripts]
phenotypic     = "phenotypic.phenotypicCLI:phenotypic_cli"   # unchanged
phenotypic-gui = "phenotypic.gui.shell._launcher:main"       # NEW
```

`uv run phenotypic-gui --root .` and `uv run python -m phenotypic.gui --root .` both
launch the hub. `uv run phenotypic pipeline.json ./images …` continues to work
exactly as it does today.

### Documentation deltas

- `README.md`: short "Launch the GUI" section.
- `docs/source/user_guide/gui.rst` (NEW): walk through hub home, sidebar, picking dirs,
  running pipelines locally vs SLURM, the iframe dashboard, the Release button.
- SSH-tunnel pattern; cloud-deploy non-goal; "closing GUI kills local runs" caveat.

### Section 7 deliverables

- [ ] 🔭 `shell/_launcher.py::launch_gui()` with banner
- [ ] 🔭 `shell/_launcher.py::main()` argparse wrapper (entry for `phenotypic-gui` script)
- [ ] 🔭 `src/phenotypic/gui/__main__.py` argparse wrapper (delegates to `_launcher.main()`)
- [ ] 🔭 `pyproject.toml`: add `phenotypic-gui = "phenotypic.gui.shell._launcher:main"`
- [ ] 🔭 NO changes to `phenotypicCLI.py` click structure (refactor dropped)
- [ ] 🔭 Existing CLI invocations regression-tested as a paranoia check (sample of 25-option flows)
- [ ] 🔭 `phenotypic.gui.launch_gui` exported from package via lazy `__getattr__`
- [ ] 🔭 `run_console/__main__.py` standalone launcher (parity with builder/viewer)
- [ ] 🔭 `README.md` "Launch the GUI" section
- [ ] 🔭 `docs/source/user_guide/gui.rst` walkthrough — explicitly notes `phenotypic gui` (no hyphen) is not supported
- [ ] 🔭 `CLAUDE.md` Quick Start updated with new invocations

---

## Section 8 — Testing strategy

### Layer 1 — Unit tests

| Module | What we test | Key cases |
|---|---|---|
| `shell/_sandbox.py` | `SandboxRoot.resolve` / `contains` | absolute outside-root → `ValueError`; relative `..` → `ValueError`; symlink escape → `ValueError`; legitimate child → resolved path |
| `shell/_classifier.py` | `classify(path)` | empty / images-only / cli-output / mixed / permission denied; LRU cache invalidates on mtime change |
| `shell/_session.py` | `ToolSession` lifecycle | lazy build; release drops state; double-release safe; build called once until release; thread safety |
| `shell/_runs_registry.py` | `RunRegistry` | register/get/list; status transitions; rehydration from sandbox scan; concurrent updates |
| `shell/_runs_blueprint.py` | `/runs/<dir>/<file>` | serves files inside sandbox; rejects path traversal; 404 outside; 403 on permission errors |
| `shell/_routes.py` | `/sandbox/api/*` | `children` shape with badges; respects toggles; sandbox containment |
| `run_console/_runner.py` | `LocalRunner` | starts subprocess; captures stdout to disk + queue; SIGTERM on `stop()`; SIGKILL after timeout; `atexit` cleanup |
| `run_console/_slurm.py` | submission glue | builds correct `--slurm k=v` args; delegates to `_cli_slurm_submission` (mocked); registers `SlurmRunHandle` |
| `run_console/_recent_runs.py` | scanner | sorts by mtime desc; cap respected; status from manifest |

**Fixtures:** `tests/conftest.py`-level synthetic sandbox tree:
```python
@pytest.fixture
def fake_sandbox(tmp_path):
    # tmp_path/plate_A/raw/{img1.tif,img2.tif}
    # tmp_path/plate_A/output/{master_measurements.parquet, results/, dashboard.html, progress/manifest.json}
    # tmp_path/pipeline.json
    # tmp_path/.hidden/secret.tif
    # symlink → ../outside
```

### Layer 2 — Integration (Flask test client; no browser)

| Test | What it does |
|---|---|
| `test_smoke_shell.py` | Boot composed app on ephemeral port; HTTP 200 on `/`, `/builder/`, `/results/`, `/run/`, `/sandbox/api/root`; `/runs/<missing>` → 404; `/runs/<existing>/dashboard.html` → 200 |
| `test_no_id_collisions.py` | Build all sub-apps; assertion passes; intentional duplicate raises clearly |
| `test_lifecycle.py` | Hit `/results/`, assert session state present; click release; assert state gone; hit again; assert rebuilt |
| `test_recent_runs_rehydrate.py` | Pre-populate sandbox with 3 fake output dirs; boot shell; assert all 3 in registry with correct statuses |
| `test_run_local_smoke.py` | Synthetic plate + tiny pipeline; click "Run" via Flask test client; assert subprocess spawned; wait for `dashboard.html`; assert iframe target updates; SIGTERM cleanup |
| `test_postmessage_listener.py` | Render chrome; inject synthetic `postMessage`; assert `shell-iframe-event` store updates |

### Layer 3 — Playwright (gated locally on `PLAYWRIGHT=1`; gated in CI on path filters)

| Test | What it does |
|---|---|
| `test_hub_navigation.py` | Boot shell; navigate `/` → `/builder/` → `/results/` → `/run/`; assert each renders; assert tab highlight follows |
| `test_sidebar_classifier_badges.py` | Boot with `fake_sandbox`; expand tree; assert `img`/`out`/`cfg` badges on the right nodes |
| `test_run_local_e2e.py` | Fill form, click Run, wait for iframe to populate, assert log lines appear, assert dashboard renders |
| `test_release_button.py` | Hit viewer; click release; assert object-graph drop (skip RSS-shrink assertion if unstable) |
| `test_select_failure_postmessage.py` | Force-render iframe with fake `dashboard.html` containing postMessage hook; click failure; assert shell logs event |

**CI gating.** Playwright requires browser binary installation
(`playwright install chromium`), which adds 2–4 minutes of CI time and ~200 MB of
disk. To avoid charging that cost on every PR, the `gui-e2e` GitHub Actions job runs
only when paths matter:

```yaml
# .github/workflows/gui-e2e.yml (sketch)
on:
  pull_request:
    paths:
      - "src/phenotypic/gui/**"
      - "src/phenotypic/_cli/_dashboard/**"
      - "tests/e2e/gui/**"
      - "pyproject.toml"
  push:
    branches: [main]
```

Unrelated PRs (touching only e.g. `src/phenotypic/detect/`) skip the browser-install
step entirely. The job is **required to pass** when triggered, never optional.

### Mocking strategy

- Subprocess execution: fake `Popen` for unit tests; integration uses tiny synthetic-plate
  pipeline that runs in <2s.
- SLURM submission: fully mocked at `_cli_slurm_submission` boundary; never invoke
  `sbatch` in tests. Manual cluster smoke covers that path.
- Filesystem: real (not mocked); `tmp_path` is cheaper than mocking `os.scandir`.

### What we deliberately don't test

- Existing builder + viewer's internal callbacks. They have their own suites.
- Dashboard's internal rendering. Tested via existing `_cli_dashboard_generator` tests.
- True OS-level RSS reclamation. We assert Python-reference drop, not RSS shrink.

### Coverage target

Match existing CLI suite: ~85% line coverage on new modules. Lifecycle, sandbox safety,
and registry rehydration should be near 100%.

### Section 8 deliverables

- [ ] 🔭 Unit tests for each module listed above (≥85% coverage on new code)
- [ ] 🔭 Integration tests (Flask test client) for boot + composition + lifecycle
- [ ] 🔭 Playwright fixtures (server-on-ephemeral-port + fake_sandbox + browser)
- [ ] 🔭 Playwright suite for navigation + badges + run flow + release + postMessage
- [ ] 🔭 `gui-e2e` CI job runs full Playwright suite
- [ ] 🔭 Coverage gate enforced in CI
- [ ] 🔭 Existing CLI suite passes with zero regressions after click refactor

---

## Section 9 — Feature ledger, future work, migration

### `src/phenotypic/gui/FEATURES.md`

A structured living document that every GUI PR updates. Downstream agents and humans
consult it as the canonical "what should the GUI do, and where is each thing tested?".
One section per page + cross-cutting; each section is a markdown table with stable
columns (Feature · Element · Expected behaviour · Status · Test layer · Test ref) so
a downstream parser can `re.split(r"\|", line)`.

```markdown
# PhenoTypic GUI Feature Inventory

> Canonical list of every user-visible feature and the tests that protect it.
> Every PR that adds, removes, or changes a UI affordance must update this file.
> Downstream automation parses the tables — keep columns stable.

## Conventions
- Status: ✅ shipping · 🚧 in progress · 🔭 planned
- Test layer: unit · integration · e2e · manual
- Test ref: file:line, dotted test path, or `n/a (manual)`

## Shell — top bar
| Feature | Element | Expected behaviour | Status | Test layer | Test ref |
|---|---|---|---|---|---|
| Root display | `top-bar-root-label` | Shows --root path; click copies to clipboard | … | … | … |
| Tab nav (Home) | `top-bar-tab-home` | Anchor → `/`; active highlight when path == `/` | … | … | … |
| Tab nav (Builder/Viewer/Run) | `top-bar-tab-{builder,results,run}` | Real navigation between mounted apps | … | … | … |
| RSS readout | `top-bar-rss` | Updates every 5s; shows psutil total RSS | … | … | … |
| Help modal | `top-bar-help-button` | Opens modal with SSH/cache/version | … | … | … |

## Shell — sidebar (file browser)
| Feature | Element | Expected behaviour | Status | Test layer | Test ref |
|---|---|---|---|---|---|
| Tree expand | `sidebar-node-{rel}` | Click expands one level; lazy-loads | … | … | … |
| Capability badges | rendered inline | Badge present iff classifier returns true | … | … | … |
| Hidden files toggle | `sidebar-toggle-hidden` | Re-renders without dotfiles when off | … | … | … |
| External symlinks toggle | `sidebar-toggle-symlinks` | Reveals greyed; clicks rejected | … | … | … |
| Refresh | `sidebar-refresh` | Flushes classifier cache | … | … | … |

## Run console
| Feature | Element | Expected behaviour | Status | Test layer | Test ref |
|---|---|---|---|---|---|
| Pipeline picker | `run-pipeline-picker` | Modal browser scoped to sandbox | … | … | … |
| Input dir picker | `run-input-picker` | Modal browser scoped to sandbox | … | … | … |
| Output dir picker | `run-output-picker` | Defaults to `output_<timestamp>` | … | … | … |
| `[↩ from sidebar]` hand-off | `run-input-from-sidebar` | Copies sidebar selection into input | … | … | … |
| Mode toggle | `run-mode-{local,slurm}` | Swaps which advanced section is shown | … | … | … |
| Dry-run checkbox | `run-flag-dry-run` | Adds `--dry-run` to subprocess args | … | … | … |
| Resume checkbox | `run-flag-resume` | Adds `--resume` | … | … | … |
| Advanced collapse | `run-advanced-collapse` | Sample/nrows/ncols/workers/log-level | … | … | … |
| SLURM collapse | `run-slurm-collapse` | Typed fields + free-form k=v rows | … | … | … |
| Run button (local) | `run-button-run` | Spawns subprocess; waits for dashboard.html; updates iframe | … | … | … |
| Run button (SLURM) | `run-button-run` | Submits via _cli_slurm_submission; registers job | … | … | … |
| Validate (dry-run) | `run-button-validate` | Spawns with --dry-run; no iframe; log only | … | … | … |
| Save preset | `run-button-save-preset` | Writes to `<root>/.phenotypic-gui/presets/` | … | … | … |
| Live log tail | `run-log-panel` | Streams stdout (local); hidden in SLURM | … | … | … |
| Dashboard iframe | `run-dashboard-iframe` | `src` set after dashboard.html exists; postMessage active | … | … | … |
| Recent runs panel | `run-recent-list` | Read-only; click → re-points iframe | … | … | … |
| Cancel running | `run-button-cancel` | SIGTERM → SIGKILL after 10s; status flips | … | … | … |
| Release button | `run-button-release` | Drops UI scratch; subprocesses survive | … | … | … |

## Builder
(reference existing inventory; only add the new `[↩ from sidebar]` button row)

## Viewer
(reference existing inventory; add empty-state behaviour + `[Release]` button rows)

> **Important for the Viewer Release row:** the `Expected behaviour` column must read
> *"Drops in-memory references (master_df, FilteredMeasurements); next access re-loads from disk."* —
> NOT "RSS readout drops." Process RSS may not return to the OS due to Python's
> allocator behaviour. Asserting RSS shrinkage in tests is unreliable and would falsely
> promise something the system doesn't deliver.

## Cross-cutting
| Feature | Test layer | Test ref |
|---|---|---|
| Sandbox containment | unit | tests/unit/gui/shell/test_sandbox.py |
| Capability classifier accuracy | unit | tests/unit/gui/shell/test_classifier.py |
| Run registry rehydration on boot | integration | tests/integration/gui/test_recent_runs_rehydrate.py |
| `postMessage select-failure` plumbing | integration | tests/integration/gui/test_postmessage_listener.py |
| Idle auto-release | unit | tests/unit/gui/shell/test_session.py::test_idle_release |
| ID collision check at boot | integration | tests/integration/gui/test_no_id_collisions.py |
```

### Implementation enforcement

`FEATURES.md` lands in **Phase 0** with all rows in `🔭 planned`. Each subsequent
phase flips rows to `🚧` while in flight and `✅` when shipped + tested.

**Two-layer enforcement** (a single pre-commit hook can't see PR-level diffs):

1. **Pre-commit hook (local; checks the working tree only):**
   - Validates `FEATURES.md` syntax — every row has the required columns.
   - For every row with `Status == "✅ shipping"`, asserts the `Test ref` resolves to
     a real file path on disk. Rows in `🔭` or `🚧` are skipped (the test files don't
     necessarily exist yet during in-flight work — Phase 0's own PR would otherwise
     fail trivially).

2. **CI workflow step (PR-level; uses git diff):**
   - Runs `git diff origin/main -- src/phenotypic/gui/` and `git diff origin/main -- FEATURES.md`.
   - If the first diff is non-empty and the second is empty, fails with:
     *"PR touches src/phenotypic/gui/ but does not update FEATURES.md."*
   - Same `Test ref` resolution check as the pre-commit hook (gated on `✅` rows).
   - Final-merge gate: when the PR is the implementation merge, additionally assert
     no `🚧` rows remain.

### Future work / TODOs

| Item | Where it lives | Notes |
|---|---|---|
| Cloud-deploy auth gate (`--mode=cloud`) | TODO at top of `shell/_sandbox.py` + `FEATURES.md` 🔭 | Out of scope v1 |
| Tab-lifecycle auto-release (`pagehide` + sendBeacon) | Mentioned as v1.5 | Manual button + idle timer cover v1 |
| Cancel/retry from Recent Runs (D-tier) | Mentioned in Q5 | Read-only in v1 |
| Subprocess isolation for true RSS reclaim | Mentioned in Section 4 | Only if memory pressure becomes a real complaint |
| Multi-local-runs (`--max-local-runs > 1`) | Flag exists; default 1 | Higher values usable, not formally tested |
| Persistent state across launches (`<root>/.phenotypic-gui/state.json`) | Mentioned; not in v1 | Cheap to add later |
| napari sweep viewer integration | Out of scope | Stays on its own launcher |
| Detached local runs (`--detach-local-runs`) | Out of scope | Keeps current "GUI death = SIGTERM children" |
| Native Dash dashboard (replacing iframe) | Out of scope v1 | Plugin API rewrite; defer |
| Builder / Viewer routing from postMessage | Hook present, routing absent in v1 | Wire when concrete cross-app workflow is identified |

### Migration plan

1. **No breaking changes** to existing standalone launchers. They continue to ship and work.
2. **One small additive change** to `builder.create_app(image_root, *, url_prefix="/")`:
   adds an optional kwarg-only `url_prefix` parameter for setting
   `requests_pathname_prefix` at construction (Dash makes this read-only after
   construction — see Section 6). Default `"/"` matches existing standalone behaviour.
3. **One small additive change** to `results_viewer.create_app(output_root: OutputRoot | None = None, *, url_prefix="/")`:
   adds the optional `url_prefix` plus makes `output_root` optional. Existing
   callers (standalone `__main__.py`) pass a value, so nothing breaks. New shell
   mounting passes `None` initially and uses the empty-state pathway.
4. **`generate_dashboard`** gains a small JS block guarded by
   `if (window.parent !== window)`. Standalone dashboard.html opens are unaffected.
5. **`pyproject.toml`** gains one new entry: `phenotypic-gui` console script. The
   existing `phenotypic` script is unchanged. **No click structural refactor.**
6. **Existing tests** keep passing without edits; new tests are additive.

### Section 9 deliverables

- [ ] 🔭 `src/phenotypic/gui/FEATURES.md` created in Phase 0 with all v1 rows
- [ ] 🔭 Pre-commit hook validating FEATURES.md syntax + resolving `Test ref` for `✅` rows only
- [ ] 🔭 CI workflow step rejecting PRs that touch `src/phenotypic/gui/` without `FEATURES.md` edit
- [ ] 🔭 CI final-merge gate rejecting PRs with any `🚧` row remaining
- [ ] 🔭 Cloud-deploy TODO comment at top of `shell/_sandbox.py`
- [ ] 🔭 Migration path verified: existing `phenotypic` CLI zero-regression (no click refactor performed)
- [ ] 🔭 `phenotypic-gui` console script added to `[project.scripts]`

---

## Quick implementation tracker

A consolidated checklist mirroring the per-section deliverables above. Tick each
as it lands so progress is visible at a glance.

### Phase 0 — Scaffolding & FEATURES.md
- [ ] 🔭 `shell/` skeleton scaffolded
- [ ] 🔭 `run_console/` skeleton scaffolded
- [ ] 🔭 `FEATURES.md` written with all v1 rows in `🔭`
- [ ] 🔭 `pytest-playwright` + `playwright` in dev deps
- [ ] 🔭 `gui-e2e` CI job stub

### Phase 1 — Sandbox + Classifier + ToolSession
- [ ] 🔭 `SandboxRoot` (resolve / contains / list_children)
- [ ] 🔭 `Capabilities` + `classify` + LRU cache
- [ ] 🔭 `ToolSession` primitive

### Phase 2 — Flask blueprints
- [ ] 🔭 `/runs/<dir>/<file>` blueprint
- [ ] 🔭 `/sandbox/api/{root,children,classify}` blueprint

### Phase 3 — Shell Dash app
- [ ] 🔭 Top bar (root, tabs, RSS, help)
- [ ] 🔭 Sidebar (lazy tree, badges, toggles, refresh)
- [ ] 🔭 Home pane
- [ ] 🔭 Release button + RSS readout
- [ ] 🔭 `shell._launcher.launch_gui()` + banner
- [ ] 🔭 `__main__.py` argparse

### Phase 4 — RunRegistry + LocalRunner + dashboard iframe
- [ ] 🔭 `RunRegistry` + sandbox-rehydrate
- [ ] 🔭 `LocalRunner` (Popen + queue + SIGTERM/SIGKILL + atexit)
- [ ] 🔭 Recent Runs scanner
- [ ] 🔭 postMessage JS in `_cli/_dashboard/_generator.py`

### Phase 5 — Mount sub-apps
- [ ] 🔭 `results_viewer.create_app(output_root=None)` empty-state path
- [ ] 🔭 `DispatcherMiddleware` composition
- [ ] 🔭 `_wrap_layout_in_chrome` (no source edits)
- [ ] 🔭 `_assert_no_id_collisions` at boot
- [ ] 🔭 `run_console/__main__.py` standalone parity

### Phase 6 — Run console UI
- [ ] 🔭 Form: pickers + mode + advanced + slurm
- [ ] 🔭 Run / Validate / Cancel / Save-preset callbacks
- [ ] 🔭 Iframe target polling + clientside src write
- [ ] 🔭 Live log tail panel (local mode)
- [ ] 🔭 SLURM submit via subprocess (no CLI imports)
- [ ] 🔭 `--max-local-runs` cap

### Phase 7 — `phenotypic-gui` console script (NO click refactor)
- [ ] 🔭 `phenotypic-gui = "phenotypic.gui.shell._launcher:main"` added to `[project.scripts]`
- [ ] 🔭 `shell/_launcher.py::main()` argparse entry implemented
- [ ] 🔭 `phenotypicCLI.py` left untouched
- [ ] 🔭 Paranoia regression-test: existing CLI suite still passes (it should — we changed nothing)

### Phase 8 — Documentation
- [ ] 🔭 `docs/source/user_guide/gui.rst`
- [ ] 🔭 `README.md` "Launch the GUI" section
- [ ] 🔭 `CLAUDE.md` Quick Start updated

### Phase 9 — Final E2E sweep
- [ ] 🔭 Full Playwright suite green on real synthetic-plate sandbox
- [ ] 🔭 Manual smoke (steps 1–7 from plan) completed
- [ ] 🔭 All `🚧` rows in `FEATURES.md` flipped to `✅`

---

## Success criteria (copied from implementation plan)

- All existing tests pass with zero regressions.
- New unit + integration + E2E suites all green; coverage ≥ 85% on new modules.
- `FEATURES.md` complete: zero `🚧` rows; every `✅` row has a working test reference.
- Manual smoke passes end-to-end on a real synthetic-plate sandbox.
- `uv run phenotypic pipeline.json ./images` (existing CLI) and
  `uv run phenotypic gui` (new subcommand) both work.
- TODO comment for cloud-deploy auth gate is present at the top of `shell/_sandbox.py`.
- Documentation updated (`gui.rst`, `README.md`, `CLAUDE.md`).
