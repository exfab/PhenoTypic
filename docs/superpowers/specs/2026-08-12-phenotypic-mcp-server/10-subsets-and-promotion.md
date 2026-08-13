# PhenoTypic MCP Server — §10 Development Subsets and the Promotion Gate

Status: **draft, pending review**
Date: 2026-08-12

## 10.1 The subset is the unit of development

**Everything from triage through campaign execution runs on a subset. The full
dataset is touched exactly once, after an explicit human promotion.**

This single structural choice resolves the autonomy question §8.6 raised. The
reason "let campaigns carry deploy arms" felt dangerous was that a full-dataset
SLURM run could launch on a pipeline nobody had looked at. Scope the development
loop to a subset and that risk disappears: an unattended campaign spends subset
compute, which is bounded and cheap by construction, and the expensive
irreversible step keeps its own gate.

```
Phase 0  TRIAGE     → assay + SUBSET                  (you + agent)
Phase 1  PLAN       → explore + converge ON THE SUBSET (you + agent)
   ▼  ── campaign approval ──
Phase 2  EXECUTE    → tune arms ON THE SUBSET          (agent alone, may amend,
Phase 3  REPORT     → leaderboard + winner              may carry deploy arms)
   ▼  ── ★ PROMOTION GATE ★ ── human, mandatory, separate ──
Phase 4  DEPLOY     → the full dataset                 (attended decision)
```

The two gates answer different questions, which is why one cannot substitute for
the other:

| Gate | Question it asks | When |
|---|---|---|
| Campaign approval (§8.2) | *Is this a sensible experiment to run?* | Before subset compute |
| **Promotion (§10.5)** | *Is this winner good enough to spend the full dataset on?* | Before full-dataset compute |

## 10.2 The subset artifact

`<workspace>/subsets/<name>.subset.json`

```json
{
  "schema_version": 1,
  "name": "plates-dev-24",
  "parent": {"path": "data/plates", "digest": "sha256:1a4c…", "n_images": 480},
  "selection": {
    "method": "user_named",
    "note": "Alex picked these to span the low- and high-contrast batches"
  },
  "images": ["plateA_01.tif", "plateA_07.tif", "…"],
  "n_images": 24,
  "digest": "sha256:77b2…",
  "coverage": {
    "measured_on": 4,
    "contrast_eta": {"min": 0.22, "max": 0.71},
    "note": "spans low→moderate; no high-contrast plate included"
  }
}
```

Three things it must record, because each one changes how much the results mean:

- **`parent` with a digest** — so a promotion can verify the full dataset has not
  changed since development, and so `campaign_status.comparable` (§8.3) has its
  dataset identity.
- **`selection.method`** — how these images were chosen. `user_named` and
  `stratified` support very different confidence in the result.
- **`coverage`** — what range of assay traits the subset actually spans, measured
  during triage. A subset that contains only easy plates will tune to a pipeline
  that fails on the hard ones, and nothing downstream can detect that from the
  cost alone.

## 10.3 Where subsets come from

**v1 — the agent does not invent subsets.**

| Method | v1 | How |
|---|---|---|
| `user_named` | **yes** | You give a list, a glob, or a directory. The honest default. |
| `first_n` / `random_n` | **yes** | Mechanical sampling with a recorded seed. Cheap, and honestly labelled as unstratified. |
| `stratified` | **seam reserved** | Sample to span the assay-trait range — the future iteration |

The seam matters more than the feature. `selection.method` is an open string
with a recorded `note` and, for sampled methods, a `seed`; adding `stratified`
later means a new method value and a generator, **not** a schema change or a new
tool signature. This is the same extensibility discipline as the trait map
(§9.3.0): the artifact's shape is fixed, its vocabulary is not.

`subset_put` (`W0`) writes one; `subset_get` reads it back with coverage.
A future `subset_generate` slots in beside them without disturbing either.

**Why the agent does not auto-generate subsets in v1:** a stratified sample needs
trait measurements across the *whole* dataset to stratify on, which means probing
far more than four images — turning a `W0`/`W1` planning step into a substantial
compute job. That is a real feature with a real cost, and it deserves its own
design rather than being smuggled into triage.

## 10.4 What runs unattended (resolves OQ-8.1 and OQ-8.2)

**Both permissions are granted, scoped to the subset.**

| Capability | Allowed? | Bound |
|---|---|---|
| Amend a campaign mid-flight (replace a failed arm) | **yes** | Must stay inside the approved budget, compute profile, and scorer. Logged with the reason. |
| Carry a deploy arm | **yes** | **Subset only.** A deploy arm targeting the full dataset is refused. |
| Deploy to the full dataset | **no** | Requires promotion (§10.5) |

So an overnight campaign can lose an arm at trial 31, substitute a replacement
inside the envelope you approved, finish, and even run the winner end-to-end
across the subset — producing real measurements, a real dashboard, and real QC
output for you to look at in the morning. What it cannot do is touch the other
456 plates.

An amendment that would exceed the approved budget, change the scorer, or switch
compute profile is **not** an amendment; it needs a fresh `campaign_approve`.
The envelope is what you actually agreed to, and it is checkable.

## 10.5 The promotion gate

`promotion_request` (`W0`) → `promotion_approve` (`W0`) → `deploy_start {scope: "full"}`

`deploy_start` gains a `scope` argument:

| `scope` | Requires | Meaning |
|---|---|---|
| `"subset"` (default) | `plan_token` | Runs against the subset. Reachable from a campaign arm. |
| `"full"` | `plan_token` **and** `promotion_token` | Runs against `subset.parent` |

`promotion_request` assembles the decision you are actually making, in one
response:

```json
{"ok":true,"data":{
  "pipeline":"pipelines/edge-v3-tuned.json.pht-pipe",
  "provenance":{"from_study":"studies/phase-edge","trial":47,
                "prefab_baseline":{"pipeline":"FilamentousFungiPipeline","best_cost":0.31}},
  "subset":{"name":"plates-dev-24","n_images":24,
            "score":0.081,"gap":{"value":0.06,"verdict":"ok"}},
  "full":{"path":"data/plates","n_images":480,"digest_matches_parent":true},
  "estimate":{"node_hours":18.4,"basis":"subset run: 3.4 s/image measured"},
  "warnings":[
    {"code":"subset_coverage_gap",
     "message":"Subset spans contrast_eta 0.22–0.71; no high-contrast plate included. 
                113 of 480 parent images were not represented by any measured trait range."}]}}
```

Two properties this must have:

- **The estimate is measured, not guessed.** The subset run already produced real
  per-image timing, so the full-dataset node-hour figure has a basis. This is the
  strongest argument for subset-first development independent of safety: it makes
  the cost of the expensive step *knowable* before you commit.
- **Coverage gaps are surfaced as warnings, not buried.** A winner tuned on 24
  easy plates may fail on the hard ones, and cost alone cannot reveal that. The
  promotion review is the only place a human can catch it.

`promotion_approve` records the decision, mints the token, and appends a lineage
row. The token is bound to `(pipeline digest, parent digest, scope)` — if the
full dataset gained images since the request, the token is stale and the review
happens again with `code: "promotion_stale"`.

**The server cannot verify a human approved.** As with campaign approval (§8.2),
`promotion_approve` is a call the agent makes after you say so in chat. It is
provenance so the artifact and the transcript agree, not authentication.

## 10.6 Risks worth stating

- **An unrepresentative subset is the dominant failure mode**, and it is silent:
  every score looks healthy while the pipeline is tuned to a easy slice.
  `coverage` and the promotion warning are the mitigations; neither is a
  guarantee, and `user_named` selection puts the responsibility with you.
- **A subset small enough to be cheap may be too small for the held-out split.**
  `HeldOutConfig.min_heldout_plates` defaults to 6, so a subset under ~12 images
  cannot support a meaningful generalization gap. `subset_put` warns below that
  threshold rather than letting the gap silently degrade to noise.
- **Subset compute is bounded but not free.** An unattended campaign with deploy
  arms still consumes an allocation. The campaign budget and profile caps (§5.2)
  are what bound it, and they bind on subset runs exactly as on any other.

## 10.7 Open questions

- **OQ-10.1 — should promotion re-probe?** The promotion estimate extrapolates
  subset timing to the full dataset. If the full set contains larger images or a
  different modality, that extrapolation is wrong. A cheap probe of 2 images
  drawn from `parent \ subset` would catch it, at the cost of one more `W1` step
  in the promotion flow.
