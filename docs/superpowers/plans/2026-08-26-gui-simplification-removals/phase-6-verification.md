# Phase 6 — Verification & docs

**Spec:** §7, §6 (last paragraph). **Depends on:** phases 1-5.

**Deliverable:** the positive checks spec §7 requires, plus `gui/CLAUDE.md` brought in line
with the five remaining mounts.

> **Why positive checks.** Removal is verified by absence, which is the weakest kind of
> test — an absent module cannot fail a test that no longer exists. These carry the weight
> instead: they assert what must still *work*.
>
> Spec §7 names three. **Two survive review**: check 1 (the apps build) reduced to the half
> that is not already covered, and check 3 (curation), re-pointed at the tests that actually
> prove it. Check 2 (no dangling references) is cut — see task 6.2 for why.

---

### Task 6.1: Both apps import and build a layout

**Files:**
- Test: `tests/unit/gui/test_apps_build_after_simplification.py` (create)

**Interfaces:**
- Consumes: nothing; `test_layout_tab_shape.py` (phase 1, edited in phase 5) owns tab shape.
- Produces: the check that catches a missed import of a deleted module — specifically in
  the shell, which no other test builds. Spec §7 check 1.

- [ ] **Step 1: Write the test**

```python
"""The hub app still constructs after the simplification.

Spec section 7, check 1 -- reduced to its load-bearing half. A deleted module
that some module still imports fails here at ``create_app`` time, which is the
only place it *can* fail now that the tests for the deleted surfaces are gone
too.

The results viewer's own tab shape is NOT re-asserted here: phase 1 created
``test_layout_tab_shape.py`` for exactly that and phase 5 edited it to the
final two-tab list. Restating it would ship a second ``dbc.Tabs`` walker beside
that file's, two walkers over one structure. What is genuinely uncovered is the
SHELL's ``create_app`` -- no phase 1-5 test builds it, and it is what catches a
missed import after the Tune unmount.
"""


def test_hub_app_constructs(tmp_output_root):
    from phenotypic.gui.shell._app import create_app

    assert create_app(root=tmp_output_root) is not None
```

`tmp_output_root` must be a real, discoverable output root. **Reuse the existing fixture**
rather than writing one — the results-viewer e2e suite already builds sandboxes with
`write_master` / `write_measurements_mirror` from `tests._output_layout`:

```bash
uv run grep -rn "write_master\|OutputRoot.discover\|tmp_output_root" tests/ | head -20
```
Match `create_app`'s real signature in both apps before writing the calls above.

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/test_apps_build_after_simplification.py -v
```
Expected: PASS. A `ModuleNotFoundError` here names the exact missed import — fix it in the
phase that owned that module, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/test_apps_build_after_simplification.py
git commit -m "test(gui): assert both apps build after the simplification"
```

---

### Task 6.2: ~~No dangling references~~ — **cut**

> **Dropped in round-1 review.** An earlier draft added
> `tests/unit/gui/test_no_dangling_removed_references.py`, an `rglob` + substring scan
> asserting no `gui/` module mentions a removed name. Four reasons it does not earn its
> keep:
>
> 1. **The real failure mode fails loudly at `create_app`**, which task 6.1 already covers.
>    The residual case it claims to catch — a lazy in-function import or an `importlib`
>    string — is hypothetical: spec §1.1 enumerates every call site of all six deleted
>    modules and every one is a top-level import.
> 2. **It is a substring scan**, so `_capture_time` matches `_capture_timeout`,
>    `read_capture_time`, and any prose mention. The draft conceded this and instructed that
>    docstring hits "should be reworded" — a test dictating prose.
> 3. Its `test_nav_model_carries_no_tune_leaf` **duplicates phase 4's** version and does it
>    worse: `"tune" not in repr(NAV_MODEL).lower()` trips on any word containing "tune" and
>    passes if the leaf is spelled differently.
> 4. Ruff's `F401` plus task 6.1's import test already cover what matters.
>
> Nothing replaces it. If a deleted module is still referenced, `create_app` raises
> `ModuleNotFoundError` and task 6.1 fails — which is the loud, specific failure this scan
> would have restated less precisely.

---

### Task 6.3: Colony curation still works — unmodified

**Files:**
- Verify only: `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py`

**Interfaces:**
- Consumes: everything. This is spec §7 check 3 and the executable statement of §5.

> **`test_colony_callbacks_helpers.py` is the constraint, not the proof.** All 15 of its
> tests drive pure `_triage_callbacks` helpers against **hand-built `ctx.triggered` dicts**
> in `tmp_path`. Nothing asserts that a radial exists on a tile, and nothing asserts that
> anything reaches disk — so it would pass **unmodified** while the radial ceased to be
> rendered at all. Keep the unmodified rule (it is the executable statement that §5 was
> respected), but the chain is proved by three other files, all collected by
> `pyproject.toml:218`'s `testpaths` and **none of them run by any command in this plan
> before this fix**.

- [ ] **Step 1: Prove the file is byte-unchanged across the whole plan**

```bash
git diff --stat <baseline-sha> -- \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py
```
where `<baseline-sha>` is the commit this plan started from. Expected: **empty output**.

Per spec §7: "If a test in `test_colony_callbacks_helpers.py` needs editing, §5 has been
violated." A non-empty diff is a **stop-and-escalate**, not something to reconcile.

- [ ] **Step 2: Run the three tests that actually prove the chain**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/gui/results_viewer/colony_view/test_grid.py::test_build_grid_tiles_carry_radial_trigger_not_old_remove_button \
  tests/integration/gui/test_triage_callbacks.py::test_colony_wedge_mark_writes_category_parquet_and_drops_mirror \
  tests/unit/cli/test_cli_error_outputs.py \
  -v
```

| Test | What it is the proof of |
|---|---|
| `test_grid.py:454` | the **only** assertion that a cell carries `colony-radial-trigger` — exactly what a deck.gl rewrite endangers |
| `test_triage_callbacks.py:227` | `deliverables/errors/debris.parquet` gains the object and the mirror drops it — the executable statement of spec §5 |
| `test_cli_error_outputs.py` | the `reemit_error_deliverables` end of the round-trip |

Expected: PASS. Note the first lives under `tests/gui/`, not `tests/unit/gui/` — a
`tests/unit/gui` invocation never touches it.

- [ ] **Step 3: Run the rest of the colony suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  tests/unit/gui/results_viewer/colony_view \
  tests/gui/results_viewer/colony_view -n 4 -q
```
Expected: PASS.

For the record, the chain verified during review and unchanged by this plan: radial →
`apply_wedge_mark` / `bulk_mark` → `mutate_and_payload` → `_save_locked` →
`_publish_if_current` (`_curation_labels.py:773`) → `_write_curated_mirror` +
`_write_category_parquets` + `_write_labels_parquet`. `colony_view/_callbacks.py:46-51`
imports the same helpers QC does, so **no hop needs QC or Error mounted** — which is spec
§5's claim, confirmed.

- [ ] **Step 4: No commit** — this task is verification only.

---

### Task 6.4: Bring `gui/CLAUDE.md` in line

**Files:**
- Modify: `src/phenotypic/gui/CLAUDE.md`

- [ ] **Step 1: Cut the sub-app table to five mounts**

```bash
uv run grep -n "tune\|Timeline\|Heatmap\|Error\|QC\|mount" src/phenotypic/gui/CLAUDE.md
```
Remove Tune from the mount table so it lists five, and mark the Error-analysis-tab section
unmounted with a pointer to the spec (spec §6, last paragraph).

- [ ] **Step 2: Check the root `CLAUDE.md` GUI section too**

```bash
uv run grep -n "tune\|Timeline" CLAUDE.md
```
The root file's "GUI hub" section lists sub-apps; if it names Tune as reachable, correct it.
`AGENTS.md` is a symlink to `CLAUDE.md`, so it follows automatically — do not edit it
separately.

- [ ] **Step 3: Final full gate run**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui -n 4 -q
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --smoke
```
Expected: all green. Then run the full unit suite as a Slurm job per the
**`run-phenotypic-test`** skill and report it as "green except the known baseline failure".

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/CLAUDE.md CLAUDE.md
git commit -m "docs(gui): record the five remaining mounts and the unmounted tabs"
```
