# 03 — Run (Deploy) & Monitor

## Run = "how to run it"

The Run destination holds the per-launch knobs and the deploy action. Sections:

- **Strategy & budget** — strategy (`grid` / `random` / `optuna`), sampler
  (`tpe` / `cmaes` / `gp` / `nsga2`), ASHA pruning, trial budget, max failures,
  seed. The sampler row only shows for Optuna. A budget field shows a rough
  **runtime estimate** (trials × per-eval cost ÷ workers).
- **Advanced** (collapsed) — evaluator/robust-eval internals: held-out fraction,
  group key, stability weight λ, ASHA rung settings. Sane defaults; most users
  never open it.
- **Compute target & output** — Local vs SLURM, **image source**, output dir,
  workers, storage URL, and (SLURM) partition/mem/time. A read-only **resolved
  command** preview shows the exact `python -m phenotypic.tune run …` invocation
  (via `render_launch_command`).

  **Image source (`-i <images>`)** comes from the hub's **shared
  source-image-root** added in the recent merge
  (`SHELL_SOURCE_IMAGE_ROOT_STORE`, resolved via
  `shell/_source_context.resolve_source_image_root`, sandbox-bounded). The Run
  form displays the resolved root and offers a per-run override picker
  (`_directory_picker` pattern). This closes the gap that a brand-new authoring
  run has no prior `run.json` to read `images_dir` from — the images come from
  the shared root, not a previous run's marker.

The objective (scorer) and search space are **not** here — they live in Setup
(doc 01/02). Run is purely method + compute + launch.

### Pre-flight: strategy ↔ search-space coupling

Strategy choice constrains the search space. The canonical conflict: **grid
cannot enumerate a continuous float** (Optuna can). When strategy is `grid` and
an active float knob has no step, Run shows a red pre-flight banner naming the
offending knob and pointing back to Setup ("give it a step or pin it"). This
surfaces at deploy time what would otherwise be a buried runtime error. It feeds
the blocked-deploy contract (doc 04) and disables Deploy.

## Relationship to the existing Launch view (no-re-optimize lock)

The current `/tune/` **Launch** view (`tune/_launch.py`) is a read-only command
*mirror* for an already-bound, finished run: it renders the `python -m
phenotypic.tune run …` string and **deliberately never spawns a process** (the
"no-re-optimize lock"), and it deliberately keeps `optuna` out of `sys.modules`.

The new **Run** destination is therefore a **new executing surface alongside the
retired command-mirror**, not a repurposing of the locked view. We:

- reuse the pure `tune/_command.render_launch_command(...)` helper for the
  read-only command **preview** on Run, and
- keep the heavy `optuna` import **lazy** (only the deploy path imports it), so
  the read-only Monitor/Curate path stays light.

This makes the "lift" explicit: deploy is intentional and gated (validation +
auto-advance), distinct from the analysis co-pilot's no-execute stance.

## The shared runner

Deploy reuses the run-console's process engine. `LocalRunner` (Popen + bounded
log ring-buffer + SIGTERM/SIGKILL + `atexit`) **already lives at a Dash-free
module** `gui/run_console/_runner.py`, and the SLURM helper at
`gui/run_console/_slurm.py` (also Dash-free). The tune Run view **imports them
directly** and points the runner at `python -m phenotypic.tune run` instead of
`python -m phenotypic`.

Moving these to a top-level `gui/_runner.py` is **optional cosmetic
consolidation, not a prerequisite** — they are already extracted from
`_callbacks.py`. The plan should default to importing in place and only move if a
naming-clarity reason emerges. (Corrects the earlier "required extraction"
framing.)

Run-state must use a persistent registry (the run-console's `RunRegistry`
pattern), not just `LocalRunner.is_running()` — the latter only tracks local
Popen handles, so SLURM runs and the live-runs counter need the registry to
survive view releases.

## Deploy

`Deploy run ▶` is gated by the aggregated validation footer (doc 04) — it is
hard-guarded, not just visually disabled. On a valid spec:

1. The full `TuningSpec` is serialized (with the `phenotypic_version` stamp) and
   the run's canonical copy is written to
   `deliverables/tuning_spec.json.pht-tune` (doc 04).
2. The runner launches `python -m phenotypic.tune run <spec> -i <images> -o
   <output> …` either locally (Popen) or as a SLURM fleet.
3. The view **auto-advances to Monitor** and the **live-runs counter
   increments**. The counter is backed by the persistent `RunRegistry` (source
   of truth), surfaced through a per-session `dcc.Store(storage_type="session")`
   so two browser tabs deploying at once don't race a shared server-side store.

Auto-advance is deliberate: the relaunch loop is "configure → fire → watch →
come back and fire a variant," and the live counter + run switcher make Monitor
the place runs accumulate.

### Local vs SLURM divergence

The two targets diverge *after* launch and Monitor must reflect it:

- **Local** — the runner tails subprocess stdout; Monitor shows a streaming
  **log** with per-trial lines (params, score, pruned, ★ best).
- **SLURM** — there is **no local stdout**; workers run on the cluster and write
  trials to the shared study store. Monitor shows a **fleet card** instead:
  array-task states (done/running/queued, when available), partition/mem/time, and
  a note that Monitor **polls the study DB** (e.g. every 3 s) for progress, with
  per-task output in `slurm-%A_%a.out` on the cluster. The charts come from the
  store, not a tail.
  - **Reachability (D13):** polling only works when the GUI host can reach the
    study store — a SQLite DB on a shared filesystem, or a reachable Postgres
    `--storage-url`. When the store is **unreachable** (the common detached
    remote-Postgres cluster pattern), Monitor **degrades gracefully**: it shows
    "running on cluster - submitted N tasks" with a hint to inspect externally,
    rather than erroring. The deploy step records mode, output directory, and
    storage URL; it does **not** require or parse a SLURM job id in v1.

The deploy target chosen on Run determines which live view Monitor opens; the
run switcher lets the user flip between any live/finished run regardless of mode.

## Monitor = inspect + export

Monitor is the existing read-only co-pilot, extended:

### Run switcher

A row of pills lists every live/finished study (name · target · progress).
Selecting one swaps the live view (Local log vs SLURM fleet) and the charts.
This is what the live-runs counter points at.

### Cancellation

Each running **Local** pill has a **✕** that opens a confirm dialog:

- Local → "Send `SIGTERM`? The N trials already in the journal are kept; the
  study can be resumed."

Confirming greys the pill, tags it *cancelled*, removes the ✕, and **decrements
the live-runs counter**. (The ✕ stops event propagation so it doesn't also
select the run.)

SLURM cancellation is **not in v1**. SLURM pills do not show the ✕, and Monitor
does not shell out to `scancel`.

### Monitor / Curate sub-tabs (unchanged)

- **Monitor** — objective-over-trials, parameter importance, the
  generalization-gap badge, plus the live view (log or fleet).
- **Curate** — pin two trials, pan/zoom-linked A/B overlays with a difference
  toggle.

### Export best pipeline (closing the loop)

The biggest functional addition: a **Best so far** result zone above the charts
showing the winning trial, its score, and its params, with actions:

- **⤓ Export best pipeline** — applies the winning knob values to the base
  pipeline (`build_pipeline(base, params)`) and writes a runnable
  `best_pipeline.json.pht-pipe` (doc 04). This is the whole payoff of tuning:
  the run produces `best_params.json`, but the *usable artifact* is a pipeline,
  and the GUI is where that conversion belongs.
- **Open in Builder** / **Send to Run Console** — hand the exported pipeline to
  the other mounts for inspection or a full processing run.

For a still-running study the best is marked "may still improve"; the action is
available throughout, not only at completion.

**Multi-objective / Pareto (D13).** When the scorer is
`CompositeScorer(multi_objective=True)` there is no single best — the study
yields a **Pareto front**. In that case the result zone switches from a single
"Best so far" card to a **Pareto view**: the front's trials (each a
non-dominated trade-off across the named objectives) are listed/plotted, and the
user **picks a trial to export**, writing
`deliverables/pareto/best_<objective>.json.pht-pipe` (the per-objective naming
already exists in `_io_constants.py`). Single-objective studies keep the simple
single-winner card. The scorer's objective count is known from the spec
(`is_multi_objective(scorer)`), so the Monitor view selects the right result UI
without guessing.

## Files to touch (GUI)

- `gui/_runner.py` (new) — extracted shared runner; `gui/run_console/` updated
  to import it.
- `gui/tune/` Run view + callbacks — strategy/budget/compute form, pre-flight,
  deploy wiring, auto-advance.
- `gui/tune/` Monitor — run switcher, Local/SLURM live-view swap, local cancel dialog,
  best-result export zone. The export action calls `build_pipeline(...)` and the
  `best_pipeline_path` helper (doc 04).
