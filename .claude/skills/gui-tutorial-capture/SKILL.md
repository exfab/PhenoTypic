---
name: gui-tutorial-capture
description: How to keep PhenoTypic's GUI feature ledgers and tutorial screenshots in sync when changing GUI chrome. Use when adding or modifying any user-visible GUI affordance under src/phenotypic/gui/, or when a CI gui-checks job (features-md-gate, workflows-md-gate, smoke-capture) fails.
---

# GUI feature ledgers & tutorial capture

Two CI-gated ledgers track the GUI surface; touching `src/phenotypic/gui/`
requires keeping them current.

## `FEATURES.md` — every affordance

`src/phenotypic/gui/FEATURES.md` lists every individual user-visible affordance
(button, badge, store, callback, route). The `gui-checks` workflow's
**`features-md-gate`** job rejects any PR that touches `src/phenotypic/gui/`
without modifying `FEATURES.md`. Pre-commit also validates the `Test ref` column
on `✅ shipping` rows.

## `WORKFLOWS.md` — every end-to-end flow

`src/phenotypic/gui/WORKFLOWS.md` lists every end-to-end user flow worth a
tutorial page. Adding a row **requires** a matching `_capture_<id>` function in
`scripts/capture_gui_tutorial_screenshots.py` **and** a walkthrough page under
`docs/source/tutorials/gui/`. The **`workflows-md-gate`** job runs
`scripts/check_workflows_md.py` (also a pre-commit hook) to enforce the
round-trip.

## Regenerating screenshots

Run after any visible chrome change and commit the refreshed PNGs alongside the
source change:

```bash
uv run python scripts/capture_gui_tutorial_screenshots.py
```

The capture regenerates the **full** screenshot set, so unrelated tutorials'
PNGs shift by a few bytes (font-rendering noise) on every run. **Commit them all
— do not cherry-pick or `git checkout --` the collateral.** Full regeneration +
commit-everything keeps the workflow simple and the committed render internally
consistent; the accepted cost is occasional binary churn in history.

The `gui-checks` workflow's **`smoke-capture`** job regenerates the PNGs on
Ubuntu and uploads them as a build artifact for spot-checking, but cross-platform
font rendering means the committed PNGs should come from a developer
workstation, not CI.
