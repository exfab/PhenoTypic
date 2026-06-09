# Design: Shared GUI source image root

- **Date:** 2026-06-08
- **Status:** Draft for user review, no implementation plan yet
- **Author:** Alexander Nguyen with Codex
- **Scope:** Unified GUI hub only, including Shell, Builder, Run, Tune, Viewer, and Analysis

## 1. Summary

Add a shell-owned `source_image_root` context that is visible in the top bar and
available to every GUI page. The context represents the mutable directory that
contains the source plate images. It is distinct from the immutable sandbox root
and distinct from the CLI output root used by Viewer and Analysis.

The recommended transport is a small browser-local `dcc.Store` mounted by the
shared shell chrome. Each page already receives that chrome through
`wrap_in_chrome()`, so the same store id and top-bar label can exist in Shell,
Builder, Run, Tune, Viewer, and Analysis. Every server-side consumer must
re-resolve the payload through `SandboxRoot` before touching the filesystem.

## 2. Evidence basis and assumptions

### Established from local code

- The hub is composed from separate Dash apps mounted through
  `DispatcherMiddleware` in `src/phenotypic/gui/shell/_app.py`. The shell wraps
  Builder, Run, Tune, Viewer, and Analysis with the same chrome by calling
  `wrap_in_chrome()` in `src/phenotypic/gui/shell/_layout.py`.
- The chrome already mounts a persistent browser-local store for sidebar
  collapse state in `wrap_in_chrome()`. This proves the same browser storage
  pattern works across mounted pages in this app.
- Run already stores its selected input directory in `RC_STORE_INPUT_DIR`.
- Tune already stores its selected image source in `TUNE_IMAGE_SOURCE_STORE` and
  validates candidates through `resolve_image_source()`.
- Viewer and Analysis bind to a validated CLI output root through
  `OutputRoot.discover()` and the `/sandbox/api/viewer/output-root` handoff.
  They do not currently use a source image root as their primary context.

### Established from Dash documentation

- `dcc.Store` stores JSON data in the browser. Its `local` mode uses
  `window.localStorage`, persists after the browser quits, and is appropriate
  for small JSON payloads. Source: Dash `dcc.Store` docs,
  https://dash.plotly.com/dash-core-components/store.
- Dash recommends explicit stores, disk, or shared server-side stores for shared
  callback state, and warns against mutating globals because Dash is stateless
  and may run callbacks in multiple processes. Source: Dash sharing-data docs,
  https://dash.plotly.com/sharing-data-between-callbacks.

### Assumptions

- The GUI remains a single-user, SSH-tunneled workstation or cluster tool for
  this feature version.
- The selected source folder should persist across browser navigation and page
  reloads, but does not need to be written to disk across GUI server restarts.
- The source folder is usually inside the launch sandbox. Out-of-sandbox source
  image roots remain out of scope for this design because the existing GUI
  picker and route model is sandbox-bounded.

## 3. Terminology

| Term | Meaning |
|---|---|
| `sandbox.root` | Immutable launch-time containment boundary. All browser-selected paths must resolve inside it. |
| `source_image_root` | Mutable directory containing source plate images. This is the shared context added by this design. |
| `output_root` | CLI run output directory, discovered by `OutputRoot.discover()`. Viewer and Analysis use this as their primary loaded context. |
| page-local store | Existing per-page Dash stores such as `RC_STORE_INPUT_DIR` and `TUNE_IMAGE_SOURCE_STORE`. |

## 4. Goals

- Show the current source image root in the top bar on every mounted page.
- Make the current source image root available to callbacks on every page.
- Keep path containment explicit and server-side. Browser storage is transport,
  not authority.
- Mirror the shared source into existing page-local stores where appropriate,
  without deleting those page-local stores.
- Preserve Viewer and Analysis output-root handoff behavior.
- Keep the first implementation small and reversible.

## 5. Non-goals

- Do not replace `sandbox.root`.
- Do not replace Viewer or Analysis `output_root`.
- Do not make the GUI multi-user or worker-safe beyond the current Dash app
  expectations.
- Do not store large image lists, DataFrames, thumbnails, or loaded arrays in
  browser storage.
- Do not introduce a new database, Redis, Flask session backend, or file-backed
  settings store for v1.
- Do not add an unrestricted filesystem picker. All browser-driven path
  selection stays sandbox-bounded.

## 6. Locked decisions

| Question | Decision |
|---|---|
| Storage transport | Browser-local `dcc.Store(storage_type="local")` mounted by shell chrome. |
| Authority | Server-side resolver validates every payload against `SandboxRoot`. |
| Public concept name | `source_image_root`, not `root`, `working_dir`, or `input_dir`. |
| Top-bar behavior | Show a compact source status pill on every page, with a clear action. |
| Setting source | Existing page pickers and sidebar handoff write the shared source. No new full picker in the top bar for v1. |
| Run integration | Mirror shared source with `RC_STORE_INPUT_DIR`. |
| Tune integration | Mirror shared source with `TUNE_IMAGE_SOURCE_STORE`. |
| Builder integration | Use shared source to seed image/pipeline browse stores after page load. Keep construction-time `CFG_IMAGE_ROOT` as the sandbox root. |
| Viewer and Analysis integration | Make source context available and visible, but keep `OutputRoot` as the loaded data context. |
| Payload size | Small JSON only. No file inventories or image metadata beyond optional count and label. |

## 7. Architecture

### 7.1 New shell ids

Add the following ids to `src/phenotypic/gui/shell/_ids.py`:

```python
SHELL_SOURCE_IMAGE_ROOT_STORE = "shell-source-image-root-store"
SHELL_SOURCE_IMAGE_ROOT_LABEL = "shell-source-image-root-label"
SHELL_SOURCE_IMAGE_ROOT_CLEAR = "shell-source-image-root-clear"
```

Optional later ids, not required for v1:

```python
SHELL_SOURCE_IMAGE_ROOT_STATUS = "shell-source-image-root-status"
SHELL_SOURCE_IMAGE_ROOT_USE_SELECTION = "shell-source-image-root-use-selection"
```

### 7.2 Payload schema

The store data is a JSON object or `None`.

```json
{
  "abs_path": "/absolute/path/to/images",
  "rel_path": "plates/batch-01",
  "label": "batch-01",
  "image_count": 384,
  "source": "run-console",
  "validated": true,
  "version": 1
}
```

Field rules:

| Field | Required | Meaning |
|---|---:|---|
| `abs_path` | yes | Absolute resolved directory path. Used only after server-side revalidation. |
| `rel_path` | yes | Path relative to `sandbox.root`, used for compact labels and round trips. |
| `label` | yes | Human-readable short label for the top bar. |
| `image_count` | no | Optional best-effort count from classifier or picker. |
| `source` | yes | One of `sidebar`, `run-console`, `tune`, `builder`, `manual`, or `unknown`. |
| `validated` | yes | Must be `true` for values written by server callbacks. Consumers still revalidate. |
| `version` | yes | Payload schema version, currently `1`. |

Do not store image filename lists. The store should stay small and stable.

### 7.3 Shared source context helper

Add a small shell-owned helper module:

`src/phenotypic/gui/shell/_source_context.py`

Responsibilities:

- Normalize and validate payload shape.
- Resolve payloads through `SandboxRoot`.
- Build payloads from `Path` plus source metadata.
- Format compact top-bar labels.
- Optionally count image-like files using the existing classifier where
  available.

Proposed functions:

```python
def source_payload_from_path(
    sandbox: SandboxRoot,
    path: Path | str,
    *,
    source: str,
) -> dict[str, object] | None:
    """Return a versioned source-image-root payload, or None if invalid."""


def resolve_source_image_root(
    sandbox: SandboxRoot,
    payload: object,
) -> Path | None:
    """Return a sandbox-contained source directory from a store payload."""


def source_label(payload: object) -> str:
    """Return a compact top-bar label."""
```

Validation requirements:

- Reject missing or non-dict payloads.
- Reject unsupported `version`.
- Reject non-string paths.
- Resolve with `sandbox.resolve()`.
- Reject non-directories.
- Reject symlink escapes through the existing `SandboxRoot` behavior.
- Treat `image_count` as advisory only.

### 7.4 Top bar UI

Update `build_top_bar()` in `src/phenotypic/gui/shell/_layout.py` to show a
compact source pill near the existing sandbox root label.

Suggested text:

- Empty: `source: unset`
- Set: `source: batch-01`

The full resolved path goes in the `title` attribute. The top bar should not
show a long absolute path inline because the tab strip is already dense.

Mount the new store in `wrap_in_chrome()`:

```python
dcc.Store(
    id=SHELL_SOURCE_IMAGE_ROOT_STORE,
    storage_type="local",
    data=None,
)
```

Register a chrome callback on every app instance:

- Input: `SHELL_SOURCE_IMAGE_ROOT_STORE.data`
- Outputs: `SHELL_SOURCE_IMAGE_ROOT_LABEL.children`,
  `SHELL_SOURCE_IMAGE_ROOT_LABEL.title`
- Behavior: format the label from store payload. Do not touch disk in this
  display-only callback unless the callback is explicitly validating.

Register a clear callback:

- Input: `SHELL_SOURCE_IMAGE_ROOT_CLEAR.n_clicks`
- Output: `SHELL_SOURCE_IMAGE_ROOT_STORE.data`
- Behavior: return `None` on a real click.

### 7.5 Setter model

Do not add a new top-bar directory browser in v1. The app already has
sandbox-bounded pickers in Run and Tune, plus sidebar selection. The source
context should be set from those flows:

- Run input picker confirm.
- Run sidebar handoff `Set as input dir`.
- Tune image-source picker confirm.
- Optional Builder load-image picker confirm when the selected file's parent is
  useful as a source folder.
- Optional sidebar directory selection when the selected directory has
  `is_image_dir` capability.

This keeps the design simple and avoids a duplicated global picker. A top-bar
picker can be added later if users need a page-independent source selector.

## 8. Page integration

### 8.1 Shell and Home

Shell owns the source store and status label. Home does not need page-specific
logic beyond showing the shared chrome.

Acceptance:

- The source status appears on `/`, `/builder/`, `/run/`, `/tune/`,
  `/results/`, and `/analysis/`.
- The same browser-local value appears after navigating between mounts.

### 8.2 Run Console

Run has the strongest fit because its input directory is semantically the same
as `source_image_root`.

Existing local state:

- `RC_STORE_INPUT_DIR` stores the selected CLI input directory.
- `sync_form_state()` reads `RC_STORE_INPUT_DIR` and places it into
  `RunConsoleState.input_dir`.
- `to_argv()` converts `input_dir` into the CLI `--input` argument.

New behavior:

1. When `RC_STORE_INPUT_DIR` changes to a valid directory, write a normalized
   payload to `SHELL_SOURCE_IMAGE_ROOT_STORE`.
2. When `SHELL_SOURCE_IMAGE_ROOT_STORE` changes and `RC_STORE_INPUT_DIR` is
   empty, mirror the resolved source path into `RC_STORE_INPUT_DIR`.
3. Do not overwrite an existing run input directory automatically once the user
   has chosen one in the current Run page session, unless the triggering action
   is explicitly a source update from Run itself.

The third rule prevents a stale localStorage value from unexpectedly changing a
run form that the user is already editing.

### 8.3 Tune

Tune's Curate view already has the same concept under a page-local name:
`TUNE_IMAGE_SOURCE_STORE`.

Existing local state:

- `TUNE_IMAGE_SOURCE_STORE` stores the selected image source.
- `_list_plate_names()` validates through `resolve_image_source()` before
  listing files.
- Overlay rendering uses the selected image source and plate filename.

New behavior:

1. On Tune Curate first load, if `TUNE_IMAGE_SOURCE_STORE` is empty and the
   shared source store resolves, initialize `TUNE_IMAGE_SOURCE_STORE` from the
   shared source.
2. When Tune confirms an image source, write the normalized payload to
   `SHELL_SOURCE_IMAGE_ROOT_STORE`.
3. When the shared source changes, update the Tune label and prompt only if the
   local Tune source has not been explicitly set in this loaded Curate session.

The existing run-derived default from `root.images_dir` remains higher priority
than a stale browser-local source. A bound tune run is more specific than a
global source preference.

Priority order for Tune source:

1. User-confirmed Tune image source in current page session.
2. Bound tune run `images_dir`.
3. Shared `source_image_root`.
4. Unset prompt.

### 8.4 Builder

Builder needs a caveat. Its construction-time `image_root` is currently set in
`create_app()` and stored in Flask config as `CFG_IMAGE_ROOT`. Many Builder
callbacks use `_image_root()` to seed browse-dir stores and bound save/load
trees.

The shared source store should not replace `CFG_IMAGE_ROOT` in v1. Keep
`CFG_IMAGE_ROOT = sandbox.root` as the safe browse boundary. Use
`source_image_root` only as an initial browse location inside that boundary.

New behavior:

- Save modal: keep bounded by `CFG_IMAGE_ROOT`. Optionally seed browse dir from
  `source_image_root` when it resolves, else fall back to `CFG_IMAGE_ROOT`.
- Load pipeline JSON modal: optionally seed browse dir from `source_image_root`
  when it resolves, else fall back to `CFG_IMAGE_ROOT`.
- Load image modal: seed browse dir from `source_image_root` when it resolves,
  else fall back to `CFG_IMAGE_ROOT`.
- Point picker tile/cache routes: keep using construction-time `CFG_IMAGE_ROOT`
  as the cache root and security boundary.

This design gives Builder the shared source context without refactoring its app
factory or route registration.

### 8.5 Viewer

Viewer's primary context remains `OutputRoot`. This is not negotiable for v1
because the viewer loads measurements, overlays, DZI cache paths, filters, and
metadata from CLI output layout.

New behavior:

- Source status is visible through chrome.
- Viewer callbacks may read `source_image_root` in future features that need
  original source files, such as opening an original plate image next to an
  overlay.
- Viewer empty-state handoff remains based on sidebar-selected CLI output
  directories and `/sandbox/api/viewer/output-root`.

Do not use `source_image_root` as a fallback `output_root`.

### 8.6 Analysis

Analysis mirrors Viewer: its primary context remains `OutputRoot`, recipe state,
and measurement schema.

New behavior:

- Source status is visible through chrome.
- Analysis may consume source context for future source-aware plots or
  provenance display.
- Empty-state output handoff remains unchanged and continues to bind both
  Viewer and Analysis through the existing output-root route.

Do not use `source_image_root` as a fallback analysis root.

## 9. Data flow

### 9.1 Setting source from Run

```text
Run input picker confirm
    -> RC_STORE_INPUT_DIR
    -> source_payload_from_path(sandbox, input_dir, source="run-console")
    -> SHELL_SOURCE_IMAGE_ROOT_STORE
    -> top bar label updates on every page
```

### 9.2 Setting source from Tune

```text
Tune Image Source confirm
    -> resolve_image_source(sandbox, browsed)
    -> TUNE_IMAGE_SOURCE_STORE
    -> source_payload_from_path(sandbox, resolved, source="tune")
    -> SHELL_SOURCE_IMAGE_ROOT_STORE
```

### 9.3 Reading source on another page

```text
SHELL_SOURCE_IMAGE_ROOT_STORE
    -> resolve_source_image_root(sandbox, payload)
    -> page-specific callback uses Path or declines gracefully
```

### 9.4 Navigating between WSGI mounts

```text
/run/ writes localStorage key
    -> browser navigates to /tune/
    -> /tune/ mounts its own Dash app and chrome
    -> dcc.Store with the same id reads the existing localStorage value
    -> Tune initialization callback can mirror it if appropriate
```

This is the same broad pattern already used for sidebar collapse persistence.

## 10. Error handling

- Empty source: display `source: unset`; consuming callbacks return their
  existing empty-state UI.
- Malformed payload: display `source: invalid`; server consumers treat as unset.
- Out-of-sandbox payload: reject and treat as unset. Log at warning level.
- Non-directory payload: reject and treat as unset.
- Directory with no obvious image files: allow as source only if it is a real
  directory, but show a warning where the page already has warning UI. Run
  already supports unusual image types through `--image-type`.
- Deleted directory after selection: top bar may still show the stored label,
  but server consumers must reject on use. A later enhancement can add
  validation status in the top bar.

## 11. Files to modify during implementation

| File | Change |
|---|---|
| `src/phenotypic/gui/shell/_ids.py` | Add shared source ids. |
| `src/phenotypic/gui/shell/_source_context.py` | New helper module for payload creation, validation, resolving, and labels. |
| `src/phenotypic/gui/shell/_layout.py` | Add top-bar source pill, clear button, and local `dcc.Store`. |
| `src/phenotypic/gui/shell/_callbacks.py` | Register source label, clear, and optional sidebar-selection source callbacks. |
| `src/phenotypic/gui/run_console/_callbacks.py` | Mirror `RC_STORE_INPUT_DIR` to shared source and initialize input dir from shared source when safe. |
| `src/phenotypic/gui/tune/_callbacks.py` | Mirror `TUNE_IMAGE_SOURCE_STORE` to shared source and initialize from shared source when safe. |
| `src/phenotypic/gui/builder/_callbacks.py` | Seed browse-dir stores from shared source when valid, while keeping `CFG_IMAGE_ROOT` as boundary. |
| `src/phenotypic/gui/FEATURES.md` | Add user-visible affordance rows with test refs. |
| Tests | Add unit, integration, and e2e coverage listed below. |

No change is required to `WORKFLOWS.md` unless implementation adds a new
end-to-end tutorial flow. This design only extends existing shell, Run, Tune,
and Builder flows.

## 12. Testing

### 12.1 Unit tests

Add `tests/unit/gui/shell/test_source_context.py`:

- `source_payload_from_path` accepts an in-sandbox directory.
- It rejects out-of-sandbox paths.
- It rejects files and missing paths.
- `resolve_source_image_root` rejects malformed payloads.
- `resolve_source_image_root` rejects unsupported schema versions.
- `source_label` handles unset, valid, and malformed payloads.

Add or extend shell layout tests:

- `wrap_in_chrome()` mounts `SHELL_SOURCE_IMAGE_ROOT_STORE`.
- `build_top_bar()` renders the source label and clear action.
- Source store uses `storage_type="local"`.

### 12.2 Integration tests

Run Console:

- Confirming an input directory writes `RC_STORE_INPUT_DIR` and the shared source
  store.
- Loading a preset with `input_dir` writes or refreshes shared source.
- A preexisting shared source initializes an empty run input field.
- A preexisting shared source does not overwrite a non-empty input field.

Tune:

- Confirming an image source writes `TUNE_IMAGE_SOURCE_STORE` and the shared
  source store.
- A preexisting shared source initializes Tune Curate only when no run-derived
  source exists.
- Bound tune run `images_dir` wins over stale shared source.

Builder:

- Opening Load Image seeds the browse dir from shared source when valid.
- Opening Load Image falls back to `CFG_IMAGE_ROOT` when shared source is unset
  or invalid.
- Save and Load JSON remain bounded by `CFG_IMAGE_ROOT`.

Viewer and Analysis:

- Source status appears in chrome on loaded and empty states.
- Existing output-root handoff tests still pass unchanged.

### 12.3 E2E tests

Add or extend an e2e test that:

1. Opens `/run/`, chooses an input directory, and sees the top bar source label.
2. Navigates to `/tune/` and verifies the same source label is visible.
3. Opens Curate with no run-derived source and verifies the shared source can
   populate the image source.
4. Navigates to `/builder/`, opens the image picker, and verifies the picker
   starts at the shared source.
5. Navigates to `/results/` or `/analysis/` and verifies the source label is
   visible while output-root handoff remains separate.

### 12.4 Regression tests

Keep existing tests green for:

- `tests/e2e/gui/test_lazy_expand_handoff.py`
- `tests/e2e/gui/test_run_console.py`
- `tests/integration/gui/test_tune_image_source.py`
- `tests/integration/gui/test_analysis_handoff.py`
- `tests/integration/gui/test_smoke_shell.py`

## 13. Acceptance criteria

- The top bar source status is visible on every mounted GUI page.
- The source value persists across mount navigation and browser reloads.
- Run can set and consume the shared source as its input directory.
- Tune can set and consume the shared source as its image source.
- Builder can use the shared source as an initial browsing location without
  widening its sandbox boundary.
- Viewer and Analysis show the shared source but still bind output roots through
  `OutputRoot.discover()`.
- Browser-local payloads are never trusted without server-side `SandboxRoot`
  validation.
- No large data objects are stored in browser storage.
- `FEATURES.md` includes the new user-visible affordances with real test refs.

## 14. Implementation order

1. Add ids and `_source_context.py` with unit tests.
2. Add store, label, clear action, and chrome callbacks.
3. Wire Run source mirroring.
4. Wire Tune source mirroring.
5. Wire Builder browse-dir seeding.
6. Add Viewer and Analysis smoke assertions for source status visibility.
7. Update `FEATURES.md`.
8. Run unit and targeted integration tests.
9. Run e2e smoke across Run, Tune, Builder, Viewer, and Analysis.

Do not invoke a full implementation plan until the user approves this spec.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stale localStorage points to a deleted directory. | Revalidate on every server-side use and degrade to unset. |
| Shared source unexpectedly overwrites page-local choices. | Only initialize empty page-local fields, or update on explicit source-setting actions. |
| Builder accidentally narrows its security boundary to source root. | Keep `CFG_IMAGE_ROOT` as sandbox root and use source only as initial browse location. |
| Viewer/Analysis semantics get confused with source images. | Keep `output_root` as primary context and treat source as supplemental. |
| Browser store grows too large. | Store only path metadata, not image lists. |
| Callback loops between shared and local stores. | Use one-way mirror callbacks with guards and `no_update` when values already match. |

## 16. Open follow-ups after v1

- Add a top-bar source picker if users want to set source without visiting Run
  or Tune.
- Persist source context to `.phenotypic-gui/` if source should survive server
  restarts.
- Add a validated source status indicator that detects deleted or unreadable
  directories immediately.
- Add source-aware Viewer features, such as opening original plate images next
  to overlays.

These are follow-ups, not requirements for the first implementation.
