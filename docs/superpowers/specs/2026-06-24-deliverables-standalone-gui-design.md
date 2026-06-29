# Standalone `deliverables/` Bundle Support for the GUI

**Date:** 2026-06-24
**Status:** Design — pending implementation plan
**Branch (suggested):** `feature/deliverables-standalone-gui`

## Problem

The results viewer cannot open a `deliverables/` folder on its own. Today it
requires three things that live at the **output-dir root**, outside
`deliverables/`:

1. **`results/<dataset>/` must exist.** `OutputRoot.discover` raises
   `FileNotFoundError` if `results/` is missing or has no dataset subdirectories,
   and it *enumerates the dataset list by listing those directories*
   (`results_viewer/_output_root.py:151-169`). This is both a structural gate and
   the source of the dataset names.
2. **`qc/` holds all QC + curation durable state.** The QC review tab, the Error
   tab's verified mode, and the curation-label store all read `<root>/qc/...`
   (`qc_summary.parquet`, `qc_members.parquet`, `qc_config.json`,
   `review_state.json`, `curation_labels.parquet`, `custom_categories.json`).
   None of these live under `deliverables/`.
3. The pixel viewers source colony/timeline crops **only** from the baked,
   downscaled overlay PNGs under `deliverables/overlays/`. There is no path that
   uses the full-resolution per-image HDF, even when `results/` is present (see
   the `crop_hdf_rgb` TODO at `gui/_shared/tiles.py:117`).

We want `deliverables/` to be a **self-contained, portable bundle**: zip it, send
it to a collaborator, and they get the *full* review + curation + QC + error-triage
workflow with no `results/` and no root `qc/`. When `results/` **is** available, the
GUI should opportunistically use the higher-resolution per-image HDF for sharper
crops and a per-layer toggle — a strict superset, never a requirement.

## Goals

- A folder containing only `deliverables/` (master + mirror + overlays + qc)
  discovers, boots, and supports **full parity**: measurements table, dashboards,
  per-colony curation that persists, QC review, and error triage.
- The user can point the GUI at **either** a `deliverables/` folder directly **or**
  a parent output root that contains it (auto-detected).
- When `results/` is present, the pixel viewers use **full-resolution HDF crops**
  and expose a **per-layer toggle** (rgb / detect_mat / objmap). When absent, they
  fall back to the overlay PNG transparently, per image.
- Existing full-run outputs keep working unchanged; existing on-disk runs with a
  root `qc/` are migrated once, in place.

## Non-Goals

- **No GUI re-measure / recompile.** Re-deriving measurements from per-image data
  remains a CLI/run-console concern. `results/` unlocks display fidelity only.
- No new tutorial/WORKFLOWS page is required (it may be added later; see CI Gates).
- No change to the CLI's output layout *other than* relocating `qc/` under
  `deliverables/` (`results/`, `.phenotypic/`, `progress/` are unchanged).

## Key Decisions (resolved during brainstorming)

| Decision | Choice |
|---|---|
| Standalone scope | **Full parity** — view + curate + QC review + error triage |
| `results/` bonus | **Full-res HDF crops + per-layer toggle** (no GUI re-measure) |
| QC state | **Relocate `qc/` → `deliverables/qc/`** (canonical), legacy root as fallback |
| Bundle root | **Auto-detect** — accept `deliverables/` itself or a parent containing it |
| Path-resolution refactor | **Approach A** — a `BundleLayout` value object in `sdk_` |
| Legacy qc migration | **MOVE** (hard cutover), idempotent, on discover + finalize |

---

## Architecture

### Section 1 — `BundleLayout` (the path-resolution seam)

A single frozen dataclass in `sdk_/_io_constants.py` becomes the authority on
on-disk topology and capability. It decouples "the folder that holds
`master_measurements.parquet`" (the **deliverables base**) from "the optional
output root that holds `results/` + `.phenotypic/`".

```python
@dataclass(frozen=True)
class BundleLayout:
    deliverables_base: Path      # folder directly holding master_measurements.parquet
    output_root: Path | None     # parent holding results/ + .phenotypic/; None when standalone

    @classmethod
    def detect(cls, path: Path) -> "BundleLayout":
        path = Path(path).resolve()
        # case 1: pointed straight AT a deliverables dir
        if (path / MASTER_MEASUREMENTS_PARQUET).is_file():
            output_root = (
                path.parent
                if path.name == DIR_DELIVERABLES and (path.parent / DIR_RESULTS).is_dir()
                else None
            )
            return cls(deliverables_base=path, output_root=output_root)
        # case 2: pointed at a parent containing deliverables/
        if (path / DIR_DELIVERABLES / MASTER_MEASUREMENTS_PARQUET).is_file():
            return cls(deliverables_base=path / DIR_DELIVERABLES, output_root=path)
        raise FileNotFoundError(
            f"{path} is neither a deliverables bundle nor a run output directory "
            "containing deliverables/master_measurements.parquet."
        )

    @property
    def has_results(self) -> bool:
        return self.output_root is not None and (self.output_root / DIR_RESULTS).is_dir()

    def hdf_path(self, dataset: str, stem: str) -> Path | None:
        """Full-res per-image HDF, or None when unavailable for this image."""
        if not self.has_results:
            return None
        candidate = dataset_hdf_dir(self.output_root, dataset) / f"{stem}.h5"
        return candidate if candidate.is_file() else None
```

**Promotion guard.** In case 1, the parent is adopted as `output_root` *only* when
the pointed-at folder is literally named `deliverables` **and** a sibling
`results/` exists. This prevents a standalone bundle (named anything) from
accidentally adopting an unrelated sibling `results/` directory.

**Path accessors split by where the artifact lives:**

- **Always from `deliverables_base`:** `master_parquet`, `master_csv`,
  `mirror_parquet`, `analysis_parquet`, `pipeline_json`, `overlays_dir(ds)`,
  `errors_dir`, `verified_parquet`, and the **canonical** `qc_dir`.
- **Only from `output_root` (capability-gated, `None` when absent):**
  `results_dir`, `hdf_path(ds, stem)`.

These accessors are added as methods/properties on `BundleLayout` (thin wrappers
over the existing filename constants) so call sites never hand-join.

**CLI side stays trivial.** The CLI always has a real output root with
`deliverables/` beneath it, so it does not need `BundleLayout`. We only repoint the
canonical helper:

```python
def qc_dir(output_dir: Path) -> Path:
    return deliverables_dir(output_dir) / DIR_QC      # was: output_dir / DIR_QC
```

`DIR_QC` stays `"qc"`; only its parent moves. Every dependent helper
(`qc_summary_parquet_path`, `qc_members_parquet_path`, `qc_config_json_path`,
`qc_review_state_path`, `curation_labels_parquet_path`,
`custom_categories_json_path`) resolves through `qc_dir`, so they all follow with
no further edits. Every existing `qc_*` write site (`run_qc`,
`CurationLabels._save_locked`, the review-state writer, `reemit_error_deliverables`)
calls those helpers and therefore writes into `deliverables/qc/` automatically.

A new `resolve_qc_dir(output_dir)` adds the legacy-root read fallback:
`deliverables/qc` if it exists, else legacy `output_dir/qc` if it exists, else the
canonical `deliverables/qc`.

### Section 2 — Discovery refactor (`results_viewer/_output_root.py`)

`OutputRoot.discover(root)` becomes: `layout = BundleLayout.detect(root)`, then
build the frozen `OutputRoot` from `layout`. `OutputRoot` holds the `BundleLayout`
and delegates path resolution to it; the existing `OutputRoot.root`/`results_dir`
properties are re-expressed in terms of `layout`.

- **Delete the hard `results/` gate** (`_output_root.py:151-169`). Validity is
  decided solely by the discovery sentinel (`master_measurements.parquet`, already
  checked at `:113-120`). The dataset list no longer comes from `results/`.
- **Dataset enumeration becomes data-driven.** Authoritative source is
  `sorted(master_df["Metadata_Dataset"].unique())`, unioned with
  `deliverables/overlays/*` subdirectories (and `results/*` when present) to catch
  a dataset that has overlays/results but zero surviving rows. Modern masters
  always carry `Metadata_Dataset`.
- **`_ensure_required_columns` becomes capability-aware.** If `Metadata_Dataset` is
  present → no-op. If absent **and** `has_results` → backfill from per-image
  parquets (today's path, `_output_root.py:342-356`). If absent **and** standalone
  → raise a precise error instructing the user to recompile (the column cannot be
  fabricated from pixels that are not in the bundle).
- **Viewer cache** (`.viewer_cache/`): write under `output_root` when present, else
  `deliverables_base/.viewer_cache` (hidden, regenerable); if that location is not
  writable (read-only mount), fall back to a per-session temp dir keyed by the
  bundle path.

Net effect: a folder with just `deliverables/` discovers and boots; a full run with
`results/` is a strict superset that lights up the Section 4 bonus paths.

### Section 3 — QC relocation + backward compatibility

The relocation itself is the one-line `qc_dir` change in Section 1. The
`errors/` directory and `verified.parquet` are already under `deliverables/`, so
the entire curation/QC surface ends up inside the bundle.

**The split-state hazard.** Read-fallback alone is unsafe: if the GUI reads legacy
`root/qc/curation_labels.parquet` but writes new labels to `deliverables/qc/`, the
durable state forks across two files. Migration must therefore be **active**, not
lazy-read-only:

- `migrate_legacy_qc(output_root)` — idempotent. If `output_root/qc/` exists and
  `deliverables/qc/` does not, **move** it (hard cutover; no duplication; matches
  the `deliverables/` cutover and the `migrate_legacy_machine_state` precedent).
  After the move, `deliverables/qc/` is canonical. No-op when the destination
  already exists, when there is no legacy dir, or when there is no `output_root`
  (standalone).
- Invoked from **(1)** `OutputRoot.discover` when `output_root` is present, and
  **(2)** `finalize_post_master_outputs` so headless recompiles relocate too.
- `BundleLayout.qc_dir` still encodes the read fallback for the read-only-mount
  edge case where the move cannot run.
- Standalone bundles never hit this path — there is no `output_root`, so qc is
  already inside `deliverables/`.

**Concurrency note.** MOVE assumes the GUI and a headless CLI recompile are not
operating on the same directory simultaneously. If that ever changes, switch the
migration to COPY at the cost of a stale duplicate. Documented here so the
assumption is explicit.

### Section 4 — Pixel-fidelity tiering (`gui/_shared/tiles.py` + tile/thumb routes)

**Implement the `crop_hdf_rgb` TODO** (`tiles.py:117`) as a sibling to
`crop_overlay`:

```python
crop_hdf_rgb(h5_path, layer, cx, cy, size, mtime_ns, bbox=None)
```

- Reads **only the requested layer** from the HDF via the `sdk_.hdf_` layer-read
  utility — **not** a full `Image` construction. (Project memory-discipline rule:
  these arrays are large and copying them is expensive; load just the layer, crop,
  release.)
- Crops the full-resolution window centered on `(cx, cy)`, reusing `crop_overlay`'s
  padding and `_dim_outside_bbox` dimming. `objmap` is colorized through the
  renderer's existing `_label_map_to_rgb`.
- Decoded layers are cached in an LRU keyed by `(path, mtime_ns, layer)` — the same
  pattern as `_load_overlay_rgb`, with smaller capacity since full-res layers are
  heavier.

**A dispatcher picks the source per-image:**

```python
def crop_colony(layout, ds, stem, layer, cx, cy, size, bbox=None):
    h5 = layout.hdf_path(ds, stem)
    if h5 is not None:
        return crop_hdf_rgb(h5, layer, cx, cy, size, h5_mtime_ns(h5), bbox)
    return crop_overlay(layout.overlay_path(ds, stem), cx, cy, size, ..., bbox)
```

Centroid/bbox coordinates still come from the in-memory master frame, so both
crop paths consume identical inputs.

**Layer toggle UI** (rgb / detect_mat / objmap): a small segmented control rendered
**when `layout.has_results`** (the run carries HDFs); hidden in standalone (the
overlay PNG is a single baked composite with no separable layers). The toggle is a
viewer-level control, but the per-image dispatcher still governs each tile — an
individual image whose `.h5` is missing falls back to its overlay regardless of the
toggle position. Wired into `colony_view`,
`timeline_view`, the OSD DZI tile route (`results_viewer/_tile_routes.py`), and the
timeline thumb route (`timeline_view/_thumb_routes.py`).

**Deep-zoom (DZI).** To make the layer toggle work in the OSD viewer too, the tiler
renders the selected HDF layer's image to a cached PNG, then tiles as today; the
cache path gains a `<layer>` dimension
(`.viewer_cache/dzi/<ds>/<stem>/<layer>/`). This is the heaviest piece (tiling a
full-resolution raw image); standalone falls back to tiling the overlay PNG exactly
as now.

### Section 5 — Mode signaling / degradation UX

With full parity + relocated qc, **no tab is ever disabled** in standalone mode —
every feature works. The only difference is pixel fidelity, so signaling stays
light:

- **One mode badge** in the results-viewer header, driven by `layout.has_results`:
  `Full run` vs `Standalone bundle`. The shell's `_classifier.py` already labels
  directories in the sidebar — extend it to distinguish the two so the badge is
  consistent hub-wide.
- **Per-viewer fidelity hint**: a small `Full-res (HDF)` / `Overlay` indicator near
  the colony/timeline view. Because capability is checked **per image**
  (`hdf_path` returns `None` for a missing `.h5`), a mid-run full directory with
  some HDFs absent simply shows `Overlay` on those individual tiles — no
  special-casing; the dispatcher already handles it.
- The **layer toggle** self-signals: its absence is the affordance.

---

## Testing

**Unit**
- `BundleLayout.detect`: case 1 (deliverables dir, with and without a sibling
  `results/`), case 2 (parent containing `deliverables/`), error path, the
  `name == "deliverables"` promotion guard, `has_results`/`hdf_path` (None vs Path).
- qc-helper relocation: `qc_dir` → `deliverables/qc`; `resolve_qc_dir` fallback
  precedence; dependent helpers follow.
- `migrate_legacy_qc`: moves when legacy present + dest absent; idempotent no-op
  when dest exists / no legacy / standalone.
- `crop_hdf_rgb`: full-res dimensions vs overlay, layer selection
  (rgb/detect_mat/objmap), per-image overlay fallback when HDF absent, cache keyed
  by layer.
- `crop_colony` dispatcher: picks HDF vs overlay correctly.

**Discovery**
- New **standalone-bundle** boot test: deliverables-only folder (qc inside, no
  `results/`) discovers and boots.
- Full-run still boots and lights up the bonus paths.
- Dataset enumeration from `master_df` + overlay union; missing `Metadata_Dataset`
  → backfill-with-results vs clear-error-when-standalone.

**GUI e2e (Playwright)** — per the project rule to verify Dash callbacks in a live
browser, not just unit tests:
- Open a standalone bundle; exercise QC review + curation persistence (asserting
  writes land in `deliverables/qc/`) and the Error tab's verified mode.
- Assert the layer toggle is present in a full run and absent in standalone.
- Curation/QC callbacks are the wrong-arity-closure class previously flagged, so
  live verification is mandatory.

**Fixtures**
- The shared viewer-output fixture moves qc under `deliverables/qc/`.
- Add a **standalone-bundle** variant (deliverables only).
- Add a **legacy** variant (root `qc/`) for the migration test.

---

## CI Gates & Docs

- **`FEATURES.md`** (hard gate): rows for the layer toggle, the mode badge, and the
  fidelity indicator. Each `✅ shipping` row needs a resolvable `Test ref`.
- **`WORKFLOWS.md`**: a "Open a shared deliverables bundle" tutorial flow is
  **optional** and **deferred** — adding a row forces the `_capture_<id>` +
  tutorial-page + screenshot round-trip. Revisit after the core ships.
- **Docs to flip** (they currently assert qc stays at root): `gui/CLAUDE.md`, the
  `_io_constants` qc-helper docstrings, the QC `CLAUDE.md`, and the root
  `CLAUDE.md` gotcha that lists `qc/` as non-deliverable.

---

## Implementation Phasing

Each phase is a code-review gate; a simplify pass + regression run follows the
final phase.

1. **`BundleLayout` + qc relocation + `migrate_legacy_qc`** (sdk_ + CLI;
   write-location change only, no GUI behavior change yet). Unit tests.
2. **Discovery refactor** — drop the `results/` gate, data-driven dataset
   enumeration, capability-aware `_ensure_required_columns`. Standalone fixture +
   discovery tests.
3. **Pixel tiering** — `crop_hdf_rgb` + `crop_colony` dispatcher + layer toggle +
   DZI layer dimension. `FEATURES.md` rows.
4. **Mode signaling** — header badge + `_classifier.py` extension.
5. **Docs/CLAUDE.md sweep + GUI e2e**.

## Risks & Mitigations

- **Test-fixture blast radius.** Many viewer tests construct the old root-`qc/`
  layout. Mitigation: update the shared fixture first (Phase 1), run the full
  viewer suite before proceeding.
- **DZI full-res tiling cost.** Tiling a raw full-res layer is heavier than tiling
  the overlay PNG. Mitigation: layer-keyed cache; standalone keeps the overlay
  path; treat DZI layer support as the last sub-task of Phase 3 and gate it behind
  `has_results`.
- **Migration on a shared dir.** MOVE assumes no concurrent GUI + CLI recompile on
  the same directory (documented in Section 3); switch to COPY if that invariant
  changes.
- **Old runs viewed before any finalize.** `migrate_legacy_qc` runs on
  `OutputRoot.discover`, so simply opening an old run in the GUI relocates its qc
  in place.
