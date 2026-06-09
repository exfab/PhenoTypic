# Screening & Parameter Importance (the `ScreeningPhase`)

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md).
Deep dive on **master §4** (the `ScreeningPhase`) and **decision D8**: how the engine
answers *"which knobs actually matter for your plates?"* from the optimizer's own
trials, and optionally **freezes** the irrelevant ones for a focused second round.

- **Status:** Design settled (pre-implementation). The fANOVA path lands in **Phase 2**
  (with Optuna); the zero-extra-dependency fallback + the report ship in **Phase 1**.
- **Maps to:** master §4 (`ScreeningPhase`), D8, §2 (fANOVA literature), §7
  (multi-objective), §8 (`param_importance.json`), §14 (open thresholds). Consumes the
  trial scores produced by [`robust-evaluation.md`](robust-evaluation.md), prunes the
  generous space from [`search-space-inference.md`](search-space-inference.md), and
  forward-references [`optuna-integration.md`](optuna-integration.md) for the
  Optuna `get_param_importances` / study details.

---

## 1. Purpose and where it fits

Screening serves two jobs from **one** computation:

1. **Analysis (always on, independently valuable).** Rank parameters by how much they
   influence the objective — a plain-English *"which knobs matter"* answer that helps a
   human or agent understand their pipeline, and triages the **`needs_review`** guesses
   from search-space inference (a low-importance inferred range can be confidently
   narrowed/frozen; a high-importance one warrants double-checking).
2. **Action (opt-in).** **Freeze** the low-importance parameters at good values and spend
   the remaining budget on a **focused second round** over the parameters that matter.

It **reuses the optimizer's own trials** — no separate screening budget — and runs by
default once a run has more than **~6 free parameters** (tunable).

---

## 2. What master §4 / D8 lock (documented, not re-litigated)

**fANOVA over the optimizer's own trials** (Optuna `get_param_importances`), capturing
main effects **and interactions**; **fANOVA > Morris OAT** (our space is
categorical/conditional and interactions can dominate — settled in the self-review);
low-importance params may be **frozen for a focused second round**; **PED-ANOVA** for
top-performance subspaces; a **zero-extra-dependency fallback** for the no-Optuna path.

> **Böck carry-forward (load-bearing).** fANOVA decomposes the variance of the
> *objective*, so its inputs must be **grid-independent**. Robust-evaluation's
> **Scorer-owned, threshold-anchored normalization** (never min–max over trials)
> guarantees exactly this (master §2; reference-free doc §B.3). Screening inherits clean
> inputs; it must not introduce any trial-set-relative rescaling of its own.

---

## 3. The screening lifecycle

**Analysis-first, with opt-in two-round freezing.** Importance is *always* computed and
reported; freezing is a deliberate gate, never an automatic gamble, and the search space
is **never mutated underneath a live surrogate** (which would destabilise TPE/GP).

```
explore round  : W warm-up trials, all params free, PRUNING OFF (§4)  ── importance sample
       ↓ fANOVA / fallback importance (§4, §5)
   freeze gate  : freeze low-importance params at good values (§6)     ── opt-in
       ↓ warm-start
focused round  : remaining budget on the reduced space, pruning ON     ── exploitation
       ↓
     winner     : best held-out objective across BOTH rounds
```

- **Budget split, not budget add.** The total trial budget splits into the explore round
  (`W` = the warm-up floor, §4) and the focused round (remainder). Screening spends no
  extra trials.
- **Round transition = fresh study + warm-start.** The focused round is a **fresh study
  on the reduced (frozen) space**, **warm-started** by enqueuing the **top-k explore
  configs** (kept-param values retained, frozen params pinned). This reuses what explore
  learned about the *kept* params without continuing a study whose space changed
  mid-flight.
- **Winner = best across both rounds.** The explore round produced valid full-space
  configs and the focused round produces reduced-space ones; the winner is the best
  held-out objective across **all** trials from both rounds — so freezing can never make
  the final result *worse* than the explore best.
- **Manual / agent freezing.** A human (CLI) or agent (MCP steering) can read the report
  and freeze/keep params by hand instead of using the auto-gate (D7 review ethos).

---

## 4. The importance method

**Primary — fANOVA (Optuna).** `get_param_importances` fits a random-forest surrogate on
the trials and decomposes the objective's variance into per-parameter **main effects**
and **interactions**; **PED-ANOVA** (master §2) gives importance within top-performance
subspaces. This is the field-standard hyperparameter-importance method.

**Fallback — RF + permutation importance (no Optuna).** Since **scikit-learn is a core
dependency**, the no-Optuna path fits a `RandomForestRegressor` surrogate and takes
**`sklearn.inspection.permutation_importance`** — the standard model-agnostic importance
(preferred over impurity-based `feature_importances_`, which is biased toward
high-cardinality features). It captures **main effects** well and handles non-monotonic /
categorical knobs, but does **not** cleanly decompose interactions; the report flags
*"interactions not estimated — install the `tune` extra for fANOVA."* Refines the
master's "correlation/variance" wording (which would have the severe blind spot that a
param with an **interior optimum** shows ~0 correlation despite mattering).

**Importance is computed from the explore round only — and the explore round runs
unpruned.** Multi-fidelity pruning (robust-evaluation §7) early-stops bad candidates, so
importance computed over *surviving* trials would be **survivorship-biased**: parameters
that determine survival look *less* important than they are (every survivor already has a
"good" value). To avoid this, the **explore round disables pruning** (full fidelity for
every warm-up trial) so the importance sample is unbiased; pruning still applies in the
*focused* round (exploitation, not measurement). The focused round's pruned,
reduced-space trials do **not** feed importance.

**Warm-up trust.** fANOVA's surrogate needs enough trials. Don't compute a *freeze-grade*
importance until `W ≥ max(absolute_floor, c·n_params)` **and** the top-k importance
**ranking is stable** across the last couple of recomputations. Below that, the report
shows importance marked *"warming up — not freeze-grade."* (`floor`, `c`, and the
stability window are conservative defaults, tunable — master §14.)

**Multi-objective (master §7).** For a Pareto study, importance is computed **per
objective** (Optuna's `target`), including the stability/dispersion objective when it is
exposed separately (robust-evaluation §9). The report shows per-objective bars; freezing
uses the across-objective rule (§6).

---

## 5. Importance over the conditional / nested space

The search space is conditional: when `<Op>.__enabled__ = False`, that op's child knobs
are **inactive** in those trials, so the trial table has holes that plain fANOVA cannot
fit. Screening uses **hierarchical, per-activation** importance:

- A **global** fANOVA over the **always-active** params **and the presence
  `__enabled__` knobs**, on all trials.
- A **separate** fANOVA per optional-op group, over only the trials where it was
  **active**, reported as **conditional importance** — *"`sigma` matters this much, given
  `GaussianBlur` is enabled."* This is a PED-ANOVA-style subspace computation and is
  bounded to **one run per optional op** by the search-space **depth-cap of 1** (presence
  is top-level only).

Guards:
- **Min-trials-per-group.** A rarely-active group (e.g. the op was enabled in only a
  handful of trials) yields noisy child importance; below a threshold the report shows
  *"insufficient data"* rather than a number — the **failure mode is flagged, not
  silent** (the decisive advantage over imputing a sentinel, which would silently leak
  the parent's presence effect into the child's importance).
- **No cross-tier ranking.** A top-level importance share and a within-subset conditional
  share are not on the same scale; the report keeps them in separate tiers and never
  ranks them in one list.

---

## 6. Freezing

When the auto-gate is enabled (or a human/agent accepts a recommendation):

- **Which to freeze — cumulative-tail over *total* importance.** Freeze the least-important
  params whose importance **collectively** accounts for `< ε` (keep the params covering
  ~90–95% of explained variance). The cutoff is over **total** importance (main effect +
  interactions), **not** main-effect-only — a param with a small main effect but a large
  *interaction* contribution must **not** be frozen. (`ε` is tunable, master §14.)
- **Freeze at — the top-k trials' central tendency.** Fix each frozen param at the
  **median** (numeric) / **mode** (categorical) of its value across the best-performing
  trials — "the value good configs tend to use" — which is robust to the noise a
  low-importance param carries in any single trial. (Refines master D8's "defaults.")
- **Freezing = a `Fixed` domain.** Mechanically, freezing converts the param to a
  `Fixed(value)` in the `SearchSpace` (search-space-inference) for the focused round.
- **Multi-objective rule — freeze only if low across *all* objectives.** A param
  important for *any* objective (including stability/dispersion) stays free. Conservative
  and correct.
- **Conservative fallback freezing.** In the RF-permutation fallback (main effects only),
  interactions are unknown, so freezing is **more conservative** — it freezes fewer
  params and flags *"interactions unverified; install the `tune` extra for
  interaction-aware freezing."*

**Safety valve + wrong-freeze recovery.** Never freeze on an unstable importance estimate
(§4). The focused round is **validated on held-out** (robust-evaluation §8); if it
**underperforms the explore round** on held-out, the freeze was likely wrong, and the
engine **falls back to the best explore-round config** as the winner, **flags** the freeze
as likely-bad, and **recommends re-running without it** — it does **not** silently keep a
worse focused result, **nor** auto-unfreeze mid-study (consistent with §3), **nor** spend
extra budget on an automatic continuation.

---

## 7. The importance report and surfaces

**`param_importance.json`** (master §8) carries: per-parameter importance shares; the
**two-tier** conditional structure (top-level + per-group); the top-N **interaction
pairs** (fANOVA path only); the **freeze recommendations** (which params, at what
central-tendency values); and **honesty flags** — the method used (fANOVA vs RF-permutation
fallback), insufficient-data groups, warm-up/stability status, and per-objective breakdown.

Plus a plain-English **"which knobs matter"** summary (D8).

**Surfaces:**
- **CLI** — prints the importance table + the freeze report; `--screen/--no-screen`
  toggles the auto-gate (master §6).
- **MCP `tune_param_importance`** — returns the structured report so an agent can decide
  freezing / steering (master §6).
- **Dash** — importance bars **consume** `param_importance.json`; the rendering itself
  lives in [`dash-copilot-design.md`](dash-copilot-design.md) (planned).

---

## 8. Error handling & graceful degradation

| Situation | Behaviour |
|-----------|-----------|
| Too few trials (`< W` / unstable ranking) | importance shown but marked *"warming up — not freeze-grade"*; no auto-freeze |
| Optuna absent | RF + permutation-importance fallback; interactions flagged as not estimated |
| Rarely-active conditional group | child importance shown as *"insufficient data"*, never a spurious number |
| Multi-objective | per-objective importance; freeze only across-all-objectives |
| Bad freeze detected on held-out | fall back to best explore config + advise re-run (no extra spend, §6) |
| `≤ ~6` free params | screening off by default (little to prune); analysis still available on request |

---

## 9. Testing

- **fANOVA over a synthetic conditional space** — always-active params scored over all
  trials; each conditional group scored over its active subset; rarely-active group →
  *"insufficient data"*; no cross-tier ranking.
- **Survivorship bias** — importance computed on the **explore round only**; the explore
  round is unpruned; focused-round trials do not feed importance.
- **Freezing math** — cumulative-tail over **total** importance; a low-main-effect /
  high-interaction param is **not** frozen; freeze-at = top-k central tendency; freeze
  produces a `Fixed` domain.
- **Multi-objective** — per-objective importance; a param important for one objective is
  not frozen.
- **Fallback** — RF permutation importance ranks main effects sensibly on a synthetic
  function; interactions flagged as not estimated; conservative freezing.
- **Lifecycle** — warm-start enqueues the top-k explore configs onto the reduced space;
  winner is the best across both rounds; bad-freeze recovery falls back to the explore
  best.
- **Warm-up guard** — no freeze-grade importance before the floor + ranking stability.

Fixed seeds throughout (project reproducibility requirement).

---

## 10. Resolved choices / open questions

**Resolved (recorded so they aren't re-litigated):**

1. **Lifecycle** — analysis-first; opt-in discrete two-round freeze gate; space never
   mutated mid-surrogate.
2. **Round transition** — fresh study on the reduced space + warm-start (enqueue top-k
   explore configs); budget splits explore(`W`)+focused; **winner = best across both
   rounds**.
3. **Importance source** — explore round only; **explore round runs unpruned** to avoid
   survivorship bias.
4. **Method** — fANOVA (Optuna) primary + PED-ANOVA subspaces; **RF + permutation
   importance** fallback (sklearn core, main effects, interactions flagged).
5. **Conditional handling** — hierarchical + per-activation; min-trials-per-group guard;
   no cross-tier ranking.
6. **Freezing** — cumulative-tail over **total** importance; freeze-at top-k central
   tendency; multi-objective = freeze across all objectives; conservative fallback
   freezing; freeze = `Fixed` domain.
7. **Wrong-freeze recovery** — fall back to best explore config + advise; no extra spend,
   no mid-study unfreeze.

**Still open (planning / empirical):**

- Warm-up floor + `c·n_params` and the ranking-stability window before a freeze-grade
  estimate (master §14).
- The cumulative-tail `ε` for freezing.
- The screening trigger (`> ~6` free params).
- Importance **confidence**: the cheap ranking-stability guard ships in v1; bootstrap
  confidence intervals on the shares are a future option.
- Whether to decompose importance w.r.t. the **level vs dispersion** sub-terms separately
  (v1 decomposes the final objective only).
