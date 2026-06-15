# Design: Global GUI settings and metadata CSV browsing

- **Date:** 2026-06-15
- **Status:** Draft for user review, no implementation plan yet
- **Author:** Alexander Nguyen with Codex
- **Scope:** Unified GUI hub shell, Browse tab, and Run Console metadata handoff

## 1. Summary

Move the GUI's global path context out of the top-bar status strip and into a
new shell-owned settings menu. The menu becomes the single place to inspect or
change:

- the immutable sandbox root,
- the shared source/input image folder,
- a new global metadata CSV file.

The metadata CSV is global state. Browse uses it to show metadata for the
currently selected source image. Run Console uses the same selected CSV when it
builds CLI arguments, passing it through to the existing `--metadata` pipeline
join path.

Browse metadata matching uses the public schema token `METADATA.IMAGE_NAME`
from `phenotypic.schema`. Treat that enum member as the string column name.
For the selected source image, Browse compares `Path(filename).stem` to the
`METADATA.IMAGE_NAME` column.

## 2. Evidence basis and assumptions

### Established from local code

- `METADATA.IMAGE_NAME` is the public schema member for the framework image
  name column. It renders as `Metadata_ImageName` through the custom
  `MeasurementInfo` enum subclass. See `src/phenotypic/schema/_metadata.py`.
- The shell already owns shared source image root state through
  `SHELL_SOURCE_IMAGE_ROOT_STORE` and validates it with
  `resolve_source_image_root()`. See
  `src/phenotypic/gui/shell/_source_context.py`.
- The current top bar renders sandbox root and source root inline. See
  `src/phenotypic/gui/shell/_layout.py`.
- Browse already builds the current image from the shared source root,
  selected dataset, and filename, then stores an encoded sandbox-relative
  path. See `src/phenotypic/gui/browse/_callbacks.py`.
- Browse currently has one metadata panel for image display metadata:
  dimensions, size, capture time, and camera. See
  `src/phenotypic/gui/browse/_layout.py`.
- Run Console state currently tracks pipeline path, input directory, output
  directory, execution mode, flags, advanced args, and SLURM args, but not a
  metadata CSV path. See `src/phenotypic/gui/run_console/_state.py`.
- The CLI config already has a `metadata_csv` field, and finalization joins a
  metadata CSV onto measurements through `join_metadata()`. See
  `src/phenotypic/_cli/_cli_types.py` and
  `src/phenotypic/_cli/_cli_output_manager.py`.

### Assumptions

- The GUI remains a single-user, SSH-tunneled workstation or cluster tool.
- Browser-local stores are acceptable transport for small global GUI settings,
  provided every filesystem use revalidates through `SandboxRoot`.
- Metadata CSV files are expected to live inside the sandbox so the existing
  picker security model applies.
- For Browse display, `METADATA.IMAGE_NAME` should match the selected source
  filename stem, not the full filename with suffix.
- One metadata row per image stem is the expected CSV shape. Duplicate stems
  are data-quality errors for Browse display.

## 3. Goals

- Replace the top-bar root/source status clutter with a compact settings entry.
- Make sandbox root, source image folder, and metadata CSV visible in one
  global menu.
- Keep source image folder behavior compatible with the existing shared source
  root store and Browse tab.
- Add a sandbox-bounded metadata CSV picker.
- Let Browse show CSV metadata for the selected image stem.
- Let Run Console pass the global metadata CSV into CLI validation and run
  commands.
- Use `METADATA.IMAGE_NAME` directly for the metadata image-name column token.
- Keep the first implementation reversible and aligned with existing shell
  state patterns.

## 4. Non-goals

- Do not add a full Settings route for this feature version.
- Do not allow out-of-sandbox metadata CSV selection.
- Do not infer metadata CSV files automatically from source folders.
- Do not support arbitrary metadata join-key configuration in Browse.
- Do not edit metadata CSV files in the GUI.
- Do not persist settings to a project file or database.
- Do not change the CLI metadata join semantics.
- Do not require Browse metadata display to duplicate the full CLI join logic.

## 5. Chosen approach

Use a shell-owned global settings menu.

The top bar should keep the application title, navigation, RSS readout, and
help button. The current inline `root: ... source: ... x` controls should be
replaced by a compact settings button. The button opens a popover or modal with
three setting rows:

| Setting | Behavior |
|---|---|
| Sandbox root | Read-only absolute path. Full path visible, copyable if practical. |
| Source image folder | Existing shared source folder value, with Pick and Clear actions. |
| Metadata CSV | New global CSV value, with Pick and Clear actions. |

The implementation should prefer a popover if the content remains compact and
usable at common laptop widths. Use a modal instead if the folder and CSV
pickers need tree navigation inside the settings surface. The logical contract
is the settings menu, not the exact Bootstrap primitive.

## 6. Global state model

### 6.1 Existing source image folder

Keep `SHELL_SOURCE_IMAGE_ROOT_STORE` and its existing payload schema as the
source image folder state. Rename user-facing copy from "source" to "input
folder" where that improves clarity, but keep code names stable unless a later
implementation plan intentionally migrates ids.

Existing consumers continue to resolve the payload with
`resolve_source_image_root(sandbox, payload)` before touching the filesystem.

### 6.2 New metadata CSV store

Add a new shell-owned store:

```python
SHELL_METADATA_CSV_STORE = "shell-metadata-csv-store"
```

The store payload is a small JSON object or `None`:

```json
{
  "abs_path": "/absolute/path/to/layout.csv",
  "rel_path": "metadata/layout.csv",
  "label": "layout.csv",
  "validated": true,
  "version": 1,
  "has_image_name": true,
  "row_count": 384,
  "unique_image_names": true
}
```

Field rules:

| Field | Required | Meaning |
|---|---:|---|
| `abs_path` | yes | Absolute resolved CSV path, used only after server revalidation. |
| `rel_path` | yes | Sandbox-relative path for labels and round trips. |
| `label` | yes | Compact display label, usually filename. |
| `validated` | yes | `true` only for values written by server callbacks. |
| `version` | yes | Payload schema version, initially `1`. |
| `has_image_name` | yes | Whether the CSV has a `METADATA.IMAGE_NAME` column. |
| `row_count` | yes | Number of CSV rows read during validation. |
| `unique_image_names` | yes | Whether non-null `METADATA.IMAGE_NAME` values are unique. |

Validation is advisory for display. Consumers still re-resolve the file before
use. A selected CSV with `has_image_name == false` can still be passed to the
CLI because CLI metadata joining is based on columns common to metadata and
measurements. Browse display, however, cannot show per-image metadata without
`METADATA.IMAGE_NAME`.

### 6.3 Helper module

Add a shell helper module next to the source context helper:

`src/phenotypic/gui/shell/_metadata_context.py`

Responsibilities:

- Build a metadata CSV payload from a candidate path.
- Resolve a payload back to a sandbox-contained CSV path.
- Format setting labels and titles.
- Read only enough CSV data to validate headers, row count, and duplicate
  `METADATA.IMAGE_NAME` values.
- Provide a Browse-facing lookup helper or a shared lower-level CSV reader.

Proposed functions:

```python
def metadata_payload_from_path(
    sandbox: SandboxRoot,
    path: Path | str,
) -> MetadataCsvPayload | None:
    """Return a validated metadata CSV payload, or None if invalid."""


def resolve_metadata_csv(
    sandbox: SandboxRoot,
    payload: object,
) -> Path | None:
    """Return a sandbox-contained CSV path from a store payload."""


def metadata_csv_label(payload: object) -> str:
    """Return the compact label shown in the settings menu."""
```

Use `from phenotypic.schema import METADATA` and the token
`METADATA.IMAGE_NAME` directly. Do not re-spell `"Metadata_ImageName"` in new
implementation code except in tests asserting user-visible strings.

## 7. Settings UI

### 7.1 Top bar

Replace the inline root/source cluster with a settings action.

Recommended visual shape:

- A compact icon button in the left or right top-bar group.
- Tooltip: `GUI settings`.
- Accessible label: `GUI settings`.
- The current root/source values are not repeated inline.

The settings surface should show:

- sandbox root full path,
- input folder label plus full path on hover or secondary text,
- metadata CSV label plus validation state,
- Pick and Clear actions for mutable settings.

### 7.2 Source/input folder picker

Reuse the existing source picker flow and callbacks where possible. The user
should not experience two competing source pickers. The settings menu simply
becomes the place where that picker is exposed.

### 7.3 Metadata CSV picker

Add a sandbox-bounded file picker filtered to `.csv` files. It should behave
like other GUI file pickers:

- browsing starts at the current metadata CSV parent, current source folder, or
  sandbox root, in that priority order,
- clicking folders navigates,
- clicking a CSV selects it,
- confirming validates and writes `SHELL_METADATA_CSV_STORE`,
- cancel leaves the current store unchanged,
- clear sets the store to `None`.

## 8. Browse metadata display

### 8.1 Matching rule

For the current Browse image:

1. Decode the current image token to the sandbox-relative path.
2. Resolve it through `SandboxRoot`.
3. Compute `image_stem = original_path.stem`.
4. Resolve the global metadata CSV.
5. Read the CSV and find rows where `csv[METADATA.IMAGE_NAME] == image_stem`.

Use `METADATA.IMAGE_NAME` directly as the column token.

### 8.2 Display model

Keep the existing image display metadata panel. Add a separate CSV metadata
section under it. This keeps EXIF/file metadata distinct from experiment
metadata.

Recommended states:

| State | Browse display |
|---|---|
| No CSV selected | `No metadata CSV selected`. |
| CSV missing/unreadable | `Metadata CSV is unavailable`. |
| CSV lacks `METADATA.IMAGE_NAME` | `Metadata CSV has no image-name column`. |
| No matching stem | `No metadata row for <stem>`. |
| Duplicate matching stems | `Multiple metadata rows for <stem>`. |
| One matching row | Render metadata fields from that row. |

For one matching row, display all CSV columns except `METADATA.IMAGE_NAME`.
Empty values render as blank or `-`. Very wide rows should use a compact table
or wrapped key-value grid, not a long chip row that pushes the viewport.

### 8.3 Performance

The first implementation may read the selected CSV on each image change if the
file is expected to be small. To avoid unnecessary rereads, a small in-process
cache keyed by `(csv_path, mtime_ns, size)` is acceptable. Do not put full CSV
rows in browser storage.

If the CSV grows large enough for reads to become visible in the UI, a later
optimization can build a stem-indexed cache in `shell/_metadata_context.py`.

## 9. Run Console integration

Add `metadata_csv` to `RunConsoleState` and to its JSON round trip. When the
global metadata CSV store resolves to a CSV file, Run Console should include it
in validation and run argv:

```text
--metadata /absolute/path/to/layout.csv
```

Run Console should treat metadata as optional. Validation and run should still
work when no CSV is selected.

The settings menu owns the user-facing metadata picker. Run Console can show a
read-only summary or badge so users can see that the run will use metadata, but
it should not introduce a second independent metadata picker.

SLURM submission should receive the same argv so the existing CLI job metadata
sidecar records the metadata CSV path through the normal CLI path.

## 10. Error handling

- Invalid or stale browser payloads resolve to `None`.
- CSV picker rejects non-files, directories, non-CSV suffixes, unreadable
  paths, and paths outside the sandbox.
- Browse metadata lookup never blocks image viewing. It only changes the
  metadata section state.
- Duplicate `METADATA.IMAGE_NAME` values are surfaced as a warning state in
  Browse instead of choosing one row.
- Missing `METADATA.IMAGE_NAME` is a Browse display warning, not a global CSV
  selection failure.
- CLI run behavior remains delegated to the CLI after the GUI passes
  `--metadata`.

## 11. Testing plan

### Unit tests

- Metadata payload validation accepts an in-sandbox `.csv` file and rejects
  out-of-sandbox paths, directories, non-CSV files, and malformed payloads.
- Metadata payload validation reports `has_image_name`, `row_count`, and
  `unique_image_names`.
- Browse metadata lookup uses `METADATA.IMAGE_NAME` and matches by
  `Path(filename).stem`.
- Browse metadata lookup returns explicit states for missing CSV, missing
  image-name column, no match, duplicate match, and one match.
- `RunConsoleState` JSON round trip preserves `metadata_csv`.
- `to_argv()` appends `--metadata <path>` only when metadata is selected.

### Integration tests

- The shell settings menu renders sandbox root, source/input folder, and
  metadata CSV rows.
- Selecting a metadata CSV writes the global shell metadata store.
- Browse updates its CSV metadata section when the selected image changes.
- Run Console receives the global metadata CSV and includes it in the command
  used by validate/run callbacks.

### E2E smoke

- Open Browse, select an input folder, select a metadata CSV, choose an image,
  and verify the image's CSV metadata appears.
- Open Run Console after selecting metadata and verify the rendered command or
  launched argv includes `--metadata`.

## 12. Documentation and ledgers

Implementation will touch `src/phenotypic/gui/`, so the GUI feature ledger must
be updated.

Expected `FEATURES.md` rows:

- global settings menu,
- metadata CSV global store,
- metadata CSV picker,
- Browse CSV metadata panel,
- Run Console metadata handoff.

This is an extension of existing Browse and Run workflows. A new
`WORKFLOWS.md` tutorial row is not required unless implementation turns the
settings flow into a new multi-step tutorial with dedicated screenshots.

Existing Browse tutorial docs should be updated to mention optional CSV
metadata display if screenshots or copy become stale.

## 13. Open implementation choices

These are intentionally left for the implementation plan:

- Use a compact popover or modal for the settings surface.
- Share one generic sandbox file picker between source and metadata, or add a
  small metadata-specific picker.
- Cache CSV contents in the shell helper or Browse helper.
- Whether Run Console shows metadata as a read-only row in its form or only in
  a command summary badge.

## 14. Decision log

| Question | Decision |
|---|---|
| Settings placement | Top-bar settings menu, not a full Settings tab. |
| Metadata scope | Global GUI setting. |
| Browse metadata key | `METADATA.IMAGE_NAME` from `phenotypic.schema`. |
| Browse match value | Selected source image filename stem. |
| Duplicate metadata rows | Warning state, no arbitrary row selection. |
| Run integration | Pass selected metadata CSV to CLI `--metadata`. |
