# Companion: render the `phenotypic.tune` data-flow graph

This file is a **self-contained prompt for Claude Desktop**. Paste the whole
file into a Claude Desktop conversation and ask it to *render the data-flow
graph as an artifact*. It carries two equivalent specs — **Mermaid** (renders
inline as an artifact) and **Graphviz DOT** (for a higher-fidelity layout) —
plus a legend so the diagram is self-explanatory.

The narrative companion is [`tune-with-optuna.md`](./tune-with-optuna.md).

---

## Prompt to paste into Claude Desktop

> You are given a Mermaid spec and a Graphviz DOT spec describing the
> hyperparameter-tuning data flow of the `phenotypic.tune` module. Render the
> **Mermaid** version as a diagram artifact. Keep every node and edge label
> verbatim. Do not invent steps. After rendering, give a 3-sentence summary of
> the loop using only the nodes shown. If I then ask for "the DOT version",
> render the Graphviz spec instead.

---

## Legend

- **Rounded boxes** = process steps (a function does work).
- **Rectangles** = data artifacts (`SearchSpace`, `params`, `EvaluationResult`,
  `Trial`, study store).
- **Diamonds** = decisions (prune?, exhausted?, build ok?).
- **Solid arrows** = forward data flow within one trial.
- **Dashed arrows** = feedback / cross-trial learning (the *tell* edge) and
  loop-back.
- **The cycle** Strategy → params → Evaluator → result → store → (tell) →
  Strategy is the ask-and-tell loop. Everything below `store.best()` is the
  one-time post-study finalize.

---

## Mermaid spec

```mermaid
flowchart TD
    %% ---- one-time setup ----
    PIPE[/"base ImagePipeline"/]
    INFER("infer_search_space<br/>Tier-1 TuneSpec / Tier-2 heuristics<br/>numeric: [d/4, d·4], log if span &gt;10x/100x")
    SPACE[/"SearchSpace (Knobs)"/]
    PIPE --> INFER --> SPACE

    %% ---- ask-and-tell loop ----
    SPACE --> STRAT
    STRAT("Strategy.suggest()<br/>TPE: max EI ∝ l(x)/g(x)<br/>CMA-ES / GP / NSGA-II / random / grid")
    PARAMS[/"params + PruningChannel"/]
    STRAT --> PARAMS

    PARAMS --> BUILD{"build_pipeline<br/>builds ok?"}
    BUILD -- no --> FAIL[/"EvaluationResult<br/>failed=True, cost=1.0"/]
    BUILD -- yes --> RUNG

    RUNG("rung ladder<br/>sizes = max(6, ⌈n/3⌉), x3, … , n")
    SCORE("Scorer.score_image → cost terms in [0,1]<br/>supervised: Dice/IoU on matched objects<br/>ref-free: shape / contrast / sizeCV<br/>qc: exp(-ln2·m/thr) · composite: augmented Tchebycheff / weighted mean")
    AGG("robust-aggregate per term<br/>median + λ·IQR, clamp01  (λ=0.5)")
    FINAL("scorer.finalize → running scalar")
    PRUNE{"between rungs:<br/>should_prune()?"}

    RUNG --> SCORE --> AGG --> FINAL --> PRUNE
    PRUNE -- "yes (not final rung)" --> PARTIAL[/"EvaluationResult<br/>pruned=True (partial)"/]
    PRUNE -- "no / final rung" --> RESULT[/"EvaluationResult<br/>score, terms, objectives?, gap, suspicious"/]

    RESULT --> STORE
    PARTIAL --> STORE
    FAIL --> STORE
    STORE[("StudyStore<br/>append Trial; budget number+=1, failures+=failed")]

    STORE -. "tell: register_result(params, result)" .-> STRAT
    STORE --> EXH{"exhausted? / n_trials / max_failures"}
    EXH -. "no" .-> STRAT
    EXH -- yes --> BEST

    %% ---- one-time finalize ----
    BEST("store.best()")
    IMP("param importance<br/>fANOVA (interactions) | RF-permutation (main effects)")
    PARETO("multi-objective:<br/>Pareto front → knee point")
    GEN("held-out re-eval →<br/>generalization gap (report-only)")
    BEST --> IMP
    BEST --> PARETO
    BEST --> GEN
```

---

## Graphviz DOT spec

```dot
digraph tune_dataflow {
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10];
    edge [fontname="Helvetica", fontsize=9];

    // shapes: box=data, box+rounded=process, diamond=decision, cylinder=store
    pipe   [shape=box, style=filled, fillcolor="#eef", label="base ImagePipeline"];
    infer  [shape=box, style=rounded, label="infer_search_space\nTier-1 TuneSpec / Tier-2 heuristics\nnumeric [d/4, d*4]; log if span >10x/100x"];
    space  [shape=box, style=filled, fillcolor="#eef", label="SearchSpace (Knobs)"];

    strat  [shape=box, style=rounded, label="Strategy.suggest()\nTPE: max EI ∝ l(x)/g(x)\nCMA-ES / GP / NSGA-II / random / grid"];
    params [shape=box, style=filled, fillcolor="#eef", label="params + PruningChannel"];

    build  [shape=diamond, label="build_pipeline\nbuilds ok?"];
    fail   [shape=box, style=filled, fillcolor="#fdd", label="EvaluationResult\nfailed=True, cost=1.0"];

    rung   [shape=box, style=rounded, label="rung ladder\nsizes = max(6, ceil(n/3)), x3, ..., n"];
    score  [shape=box, style=rounded, label="Scorer.score_image -> cost terms[0,1]\nsupervised Dice/IoU | ref-free shape/contrast/sizeCV\nqc exp(-ln2*m/thr) | composite augmented Tchebycheff / weighted mean"];
    agg    [shape=box, style=rounded, label="robust-aggregate per term\nmedian + lambda*IQR, clamp01  (lambda=0.5)"];
    final  [shape=box, style=rounded, label="scorer.finalize -> running scalar"];
    prune  [shape=diamond, label="between rungs:\nshould_prune()?"];
    partial[shape=box, style=filled, fillcolor="#ffd", label="EvaluationResult\npruned=True (partial)"];
    result [shape=box, style=filled, fillcolor="#dfd", label="EvaluationResult\nscore, terms, objectives?, gap, suspicious"];

    store  [shape=cylinder, style=filled, fillcolor="#efe", label="StudyStore\nappend Trial; budget number+=1, failures+=failed"];
    exh    [shape=diamond, label="exhausted?\nn_trials / max_failures"];

    best   [shape=box, style=rounded, label="store.best()"];
    imp    [shape=box, style=rounded, label="param importance\nfANOVA (interactions) | RF-permutation (main effects)"];
    pareto [shape=box, style=rounded, label="multi-objective:\nPareto front -> knee point"];
    gen    [shape=box, style=rounded, label="held-out re-eval ->\ngeneralization gap (report-only)"];

    // setup
    pipe -> infer -> space -> strat;

    // ask
    strat -> params -> build;
    build -> fail   [label="no"];
    build -> rung   [label="yes"];

    // evaluate
    rung -> score -> agg -> final -> prune;
    prune -> partial [label="yes (not final rung)"];
    prune -> result  [label="no / final rung"];

    // collect + tell (feedback dashed)
    result  -> store;
    partial -> store;
    fail    -> store;
    store -> strat [label="tell: register_result(params, result)", style=dashed, constraint=false];
    store -> exh;
    exh -> strat [label="no", style=dashed, constraint=false];
    exh -> best  [label="yes"];

    // finalize
    best -> imp;
    best -> pareto;
    best -> gen;
}
```

---

## Node reference (maps each box to source)

| Node | Source | Math / role |
|---|---|---|
| `infer_search_space` | `_search_space/_infer.py:685` | fields → Knobs; `[d/4, d·4]`, log auto-trip |
| `Strategy.suggest()` | `_strategies/_optuna.py:268` | TPE EI ∝ `l(x)/g(x)`; sampler at `:248` |
| `build_pipeline` | `_evaluation/_builder.py` | clone base + overlay params |
| rung ladder | `_evaluation/_evaluator.py:222` | `max(6, ⌈n/3⌉)`, ×3, …, n |
| `Scorer.score_image` | `_scoring/*` | four strategies, **cost** terms ∈ [0,1] (oriented by `to_cost`) |
| `Scorer.to_cost` | `_scoring/_orient.py` | natural value → cost ∈ [0,1] (Sense + anchor) |
| robust-aggregate | `_evaluator.py:55` + `_aggregate_math.py:28` | `median + λ·IQR` (clamped), λ=0.5 |
| `should_prune()` | `_optuna.py:218` (ASHA) | top `1/reduction_factor` survive |
| `StudyStore` | `_study_store.py` / `_study/_optuna_store.py` | journal or Optuna RDB |
| param importance | `_screening.py:85` | fANOVA vs RF-permutation |
| Pareto knee | `_study/_pareto.py:54,115` | dominance + max chord distance |
| generalization gap | `_generalization.py:58` | loss-space `heldout_cost − cal_cost`; `rel>0.15 ∧ abs>0.05` flag |
