# 01 — Placement & Information Architecture

## Decision: evolve `/tune/`, don't add a top-level page

The hub already mounts a read-only `/tune/` Dash sub-app
(`src/phenotypic/gui/tune/`) with the sub-views Monitor / Curate / Space /
Launch. Two of those — **Space** (runs `infer_search_space`) and **Launch**
(renders the `python -m phenotypic.tune run …` command) — are embryonic versions
of the authoring surface we need. Building a separate "Setup Tuning" mount would
either duplicate or delete those shipped, tested, tutorialized views.

We therefore **grow `/tune/` in place** rather than add a mount. This is the
lowest-debt option (reuses the mount, chrome, sub-view switching, design tokens,
and the builder's param-form machinery), keeps the whole tune journey in one
tab, and preserves the option to promote authoring to its own mount later if
non-expert discoverability ever demands it. See D1 in the decision log for the
full tradeoff against a dedicated page and a distributed-by-reuse-seam option.

## The hamburger IA: Setup / Run / Monitor

Inside `/tune/`, a hamburger control in the page bar opens a left drawer with
three destinations following the verb chain **author → deploy → inspect**:

| Destination | Verb | Contents |
|-------------|------|----------|
| **Setup** | what to tune & how to judge | Pipeline · Search space · Scorer (doc 02) |
| **Run** | how to run it | Strategy & budget · Advanced eval · Compute target · Deploy (doc 03) |
| **Monitor** | inspect | The existing read-only co-pilot: Monitor + Curate sub-tabs, run switcher, export-best (doc 03) |

This replaces the current flat Monitor/Curate/Space/Launch sub-tab row:

- **Space** is folded into **Setup → Search space** (it stops being a peer
  destination; its `infer_search_space` logic survives 1:1 as the section's
  prefill).
- **Launch** (today a read-only command *mirror* with an explicit "no-re-optimize
  lock" — `tune/_launch.py` never spawns a process and keeps `optuna` out of
  `sys.modules`) is **superseded by Run**. Run is a *new executing surface*, not
  a repurposing of the locked view: it reuses the pure `render_launch_command`
  helper for its command preview but adds the gated executor. The lazy-`optuna`
  constraint is preserved (only the deploy path imports it). See doc 03 and D13.
- **Monitor + Curate** move under the **Monitor** destination unchanged.

This is a larger change than a rename: the existing `_ids.py`
`SUBTAB_ORDER = ("monitor","curate","space","launch")` and the Launch/Space
callbacks are production code. The plan must treat Setup and Run as **new view
bodies** wired into a new hamburger nav, with Space's inference and Launch's
command-render helpers reused — not as edits to the existing sub-tab strip.

Rationale for the hamburger over a flat tab strip: the three destinations are
*modes* with different empty-state semantics (Setup/Run author with no bound
run; Monitor requires a run), and the drawer scales cleanly without crowding the
page bar. See D2.

## Why the verb split lands where it does

- **Setup = pipeline + search space + scorer.** The scorer defines the
  objective — *what "best" means* — so it is part of the experiment definition
  and lives with the search space, not on Run. (Earlier iterations placed it on
  Run; D3 moved it back.)
- **Run = strategy + budget + advanced + compute.** These are genuine
  per-launch knobs; the CLI already treats `--strategy`/`--n-trials` as
  overrides of the spec. The relaunch loop (tweak strategy, redeploy) wants them
  here, reinforced by the live-runs counter and auto-advance (doc 03). See D4.
- **Monitor = inspect + export.** Unchanged analysis surface plus the
  export-best action that turns a finished study into a runnable pipeline.

## Navigation & state

- The hamburger drawer is a CSS-transform slide-in with a scrim; selecting a
  destination swaps the active view (CSS view-swap, same pattern as today's
  sub-tab toggling — no route change).
- A **live-runs counter** appears in three coordinated places: the page bar
  (clickable → Monitor), the drawer's Monitor item, and the hub's Tune tab chip.
  Deploying increments it; a run finishing, or a Local run being cancelled,
  decrements it (doc 03).
- **Crumb**: the page bar shows the active destination name + a one-line
  subtitle.

## Empty-state gating (entry)

Setup is gated on a chosen pipeline (D10):

- On load with no pipeline, the **Search space** and **Scorer** sections render
  **folded, name-bar dimmed, and unselectable** (a `🔒 pick a pipeline first`
  summary; clicking the header is a no-op). The footer is muted ("Choose a
  pipeline to begin") and **Continue to Run is disabled**; the drawer's Run
  destination is also inert until a pipeline exists.
- Choosing a pipeline unlocks both sections (Search space auto-opens), fills the
  op-chain, and enables the footer.
- Three entry paths feed the pipeline gate: the **file picker**, a **Builder →
  "Tune this pipeline"** hand-off, and **Open spec** (re-edit a saved
  `.json.pht-tune`; doc 04).
- **Builder hand-off mechanism.** The builder and tune apps are separate Flask
  servers behind `DispatcherMiddleware`, so the hand-off cannot pass an in-memory
  `ImagePipeline`. It carries the **pipeline config path** (a
  `.json.pht-pipe` the builder writes to the sandbox), via a **shell-level
  `dcc.Store`** — the same cross-app pattern the merge introduced for the shared
  source-image-root (`SHELL_SOURCE_IMAGE_ROOT_STORE`). The builder gets a "Tune
  this pipeline" button that writes the path to that store and navigates to
  `/tune/`; Setup reads the store on load and resolves the pipeline. (A URL query
  param is the fallback if a store proves awkward across the mount boundary.)

## Files to touch (GUI)

Within `src/phenotypic/gui/tune/`:

- `_layout.py` / view modules — replace the flat sub-tab strip with the
  hamburger drawer + the three destination views; add the Setup and Run view
  bodies; keep Monitor/Curate.
- `_ids.py` — new component IDs for the drawer, destinations, Setup/Run controls
  (follow the tool-local `_ids.py` convention).
- `_callbacks.py` — destination switching, the empty-state lock/unlock, footer
  state, live-runs counter.
- `_app.py` — no new mount; the app gains the authoring + deploy views.

Shared infra (reuse, don't re-spell): `gui/_param_forms.py` (knob/scorer
widgets), `gui/_operation_registry.py` (pydantic field walk), `gui/_design.py`
(tokens), `gui/_config.py` (constants, sandbox paths). The run/deploy engine is
extracted to a shared module in doc 03.

## Ledger & CI obligations

Any PR touching `src/phenotypic/gui/` must update the two CI-gated ledgers:

- **`gui/FEATURES.md`** — one row per new affordance (hamburger, each
  destination, the domain editor, validation badges, deploy button, run
  switcher, Local cancel, export-best, save/load, etc.) with a `Test ref`. The
  `features-md-gate` job rejects PRs that touch `gui/` without editing this.
- **`gui/WORKFLOWS.md`** — the end-to-end "author → deploy → monitor → export"
  flow gets a row, which **requires** a matching `_capture_*` function in
  `scripts/capture_gui_tutorial_screenshots.py` and a walkthrough page under
  `docs/source/tutorials/gui/`. The `workflows-md-gate` job enforces the
  round-trip. Plan for regenerating the full screenshot set and committing all
  PNGs (font-render churn is expected and accepted).
