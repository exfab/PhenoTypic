# PhenoTypic MCP Server — §10 Development Subsets and the Promotion Gate

Status: **draft, reviewed once, revised**
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
    "method": "MetadataGroupSubsetSelector",
    "params": {"n": 24, "seed": 0,
               "grouping_metadata": "data/plate_batches.csv",
               "group_key": "Metadata_Batch", "allocation": "equal"},
    "rationale": "8 batches x 3 plates each; equal allocation so the two rare
                  low-contrast batches are not swamped by the six common ones"
  },
  "images": ["plateA/plateA_01.tif", "plateA/plateA_07.tif", "plateB/plateB_03.tif", "…"],
  "n_images": 24,
  "digest": "sha256:77b2…",
  "coverage": {
    "measured_on": 4,
    "contrast_michelson": {"min": 0.031, "max": 0.094},
    "note": "spans low→moderate on the per-cell Michelson measure (§9.3.2);
             no high-contrast batch included"
  }
}
```

Three things it must record, because each one changes how much the results mean:

- **`parent` with a digest** — so a promotion can verify the full dataset has not
  changed since development, and so `campaign_status.comparable` (§8.3) has its
  dataset identity. `images` entries are **parent-relative paths**, not bare
  filenames: `scan_directory_structure` treats one level of subdirectories as
  separate datasets, so a bare name cannot disambiguate two datasets that both
  contain `plate_001.tif` (§10.3.1).
- **`selection`** — the selector class, its params, and its seed. A
  `RandomSubsetSelector` and a `MetadataGroupSubsetSelector` support very
  different confidence in the result, and a recorded seed makes either
  reproducible.
- **`coverage`** — what range of assay traits the subset actually spans, measured
  during triage. A subset that contains only easy plates will tune to a pipeline
  that fails on the hard ones, and nothing downstream can detect that from the
  cost alone.

## 10.3 Where subsets come from — the selector hierarchy

Subset selection is a **pluggable strategy**, following the same pattern as
every other extensible thing in this codebase: a pydantic ABC, concrete
subclasses, `{class, params}` serialization, and resolution by bare class name.

### `SubsetSelector` — the base class

New public subpackage `phenotypic/subset/`, added to
`_find_class_in_phenotypic`'s submodule list so selectors serialize and resolve
exactly like operations and scorers do.

```python
class SubsetSelector(BaseModel, ABC):
    """Choose a development subset from a parent image set.

    Args:
        n: Target subset size.
        seed: RNG seed; recorded on the artifact so a selection is reproducible.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n: int = Field(..., ge=1)
    seed: int = 0

    @abstractmethod
    def _select(self, candidates: list[ImageRef]) -> list[str]: ...

    def availability(self) -> tuple[bool, str]: ...   # (usable?, why not)
    def cost_class(self) -> Literal["W0", "W1", "W2"]: ...

    def select(self, candidates: list[ImageRef]) -> SubsetSelection:
        """Template: check availability, delegate, then dedup, order, and
        record the rationale so the artifact explains itself."""
```

`SubsetSelection` is frozen and carries `images`, `method`, `params`, `seed`,
and a human-readable `rationale` — which becomes `selection` on the subset
artifact (§10.2).

Two methods worth calling out:

- **`availability()`** mirrors `Scorer.availability()` (§4.1), so
  `subset_generate` can report which selectors are usable *before* the agent
  commits — the same affordance that stops the most common tuning failure.
- **`cost_class()`** is what keeps an expensive selector from being smuggled
  into triage. Selection cost is not uniform, and the difference is structural:

  | Selector | Cost | Because |
  |---|---|---|
  | `RandomSubsetSelector` | `W0` | Needs only the file list |
  | `MetadataGroupSubsetSelector` | `W0` | Needs only the metadata CSV, which already exists |
  | `EmbeddingSubsetSelector` | `W2` | Must encode **every parent image** |

  An earlier draft deferred all stratification to a future iteration on the
  grounds that stratifying requires measuring the whole dataset. That was too
  blunt: it is true of *trait* and *embedding* stratification, and false of
  **metadata** stratification, where the grouping already exists on disk. So
  metadata sampling ships now.

### The three selectors

**`RandomSubsetSelector`** — uniform without replacement, seeded.

```json
{"class": "RandomSubsetSelector", "params": {"n": 24, "seed": 0}}
```

Honest and unstratified. The right default when no metadata exists, and the
right *baseline* even when it does.

**`MetadataGroupSubsetSelector`** — sample across metadata groups.

```json
{"class": "MetadataGroupSubsetSelector",
 "params": {"n": 24, "seed": 0,
            "grouping_metadata": "data/plate_batches.csv",
            "group_key": "Metadata_Batch",
            "allocation": "proportional"}}
```

| Param | Meaning |
|---|---|
| `grouping_metadata` | CSV supplying the grouping column. **Named distinctly on purpose** — three different CSVs appear in this spec: `deploy_plan.metadata_csv` (joined onto the output mirror), this one (subset stratification), and `QCScorer.check.metadata` (the expected counts the whole objective is scored against). Passing the wrong one at the scorer produces a meaningless objective rather than an error. |
| `group_key` | The column in `grouping_metadata` naming each plate's group |

**Only the new class was renamed.** `grouping_metadata` is this spec's choice
because `MetadataGroupSubsetSelector` does not exist yet. `ExpectedVsDetectedCount.metadata`
**ships today with no alias** (`analysis/qc/_expected_vs_detected.py:208`), so it
keeps its name and is disambiguated in prose. A draft of this spec renamed it to
`expected_counts_csv` in two worked examples — a field that does not exist, which
would raise `missing` on `metadata` and `extra_forbidden` on the invention: the
exact failure §4.2's pre-submit checks exist to prevent, written into the
example. **A spec may name a class it is introducing; it may not rename one that
already ships.**
| `allocation` | `proportional` (mirror group sizes) or `equal` (same count per group, so a rare condition is not lost) |
| `min_per_group` | Floor per group; groups smaller than it are taken whole |

**Only a class this spec introduces may be renamed.** `grouping_metadata` is
this spec's choice because `MetadataGroupSubsetSelector` does not exist yet.
`ExpectedVsDetectedCount.metadata` **ships today with no alias**
(`analysis/qc/_expected_vs_detected.py:208`), so it keeps its name and is
disambiguated in prose only. A draft of this spec renamed it to
`expected_counts_csv` in two worked examples — a field that does not exist, which
would raise `missing` on `metadata` and `extra_forbidden` on the invention: the
exact failure §4.2's pre-submit checks exist to prevent, written into the
example.

**The selector performs its own CSV→filename join. It does *not* reuse
`_resolve_groups`.**

An earlier draft claimed it did — "the same vocabulary the tune split already
uses" — and that reusing it would guarantee the held-out split reached Tier 2
(whole-group hold-out) rather than the weaker within-group tier. **Both halves
are false, verified by reproduction.** `_resolve_groups`
(`tune/_evaluation/_split.py:114-133`) is a pure in-memory
`image.metadata.get(group_key)` lookup with no CSV and no join, and a freshly
read image carries only:

```
MetadataImage_BitDepth, MetadataImage_FileSuffix, MetadataImage_ImageName,
MetadataImage_ImageType, MetadataImage_UUID
```

`img.metadata.get("Metadata_Batch")` returns `None`. External CSV columns reach
data only through `join_metadata` (`_cli/_cli_output_manager.py:83-175`), which
operates on the **measurement DataFrame** inside `finalize_post_master_outputs`
— i.e. after a full pipeline run has measured every image, the exact opposite of
Phase-0 triage.

So the claimed payoff does not follow: `_resolve_groups` returns `{}` for any
externally-sourced key, and `derive_split` falls through to the within-group or
data-poor tier **silently** — no error, just a weaker generalization estimate
than the reader was promised.

What the selector actually does: read `grouping_metadata`, join rows to images
**by filename / parent-relative path**, and stratify on that. `group_key` names a
column in that CSV. It is a name shared with the tune split's vocabulary and
nothing more.

**If the held-out split should also benefit from the grouping**, something must
populate `image.metadata[group_key]` from the CSV at tune load time. That is an
engine change to `phenotypic.tune`, it is **not** in §7's P1–P6, and it is not
assumed anywhere in this design. Until it exists, stratifying a subset does not
change how the split is derived.

**`EmbeddingSubsetSelector`** — placeholder, and it **fails loudly**.

```json
{"class": "EmbeddingSubsetSelector",
 "params": {"n": 24, "seed": 0, "model": "<unset>", "strategy": "kmeans_medoids"}}
```

Intended shape: embed every parent image with a vision model, cluster, and take
medoids — giving visual coverage without any metadata or hand-labelling.

Until implemented, `availability()` returns
`(False, "EmbeddingSubsetSelector is not implemented; no embedding backend is configured")`
and `_select` **raises `NotImplementedError`**. It does **not** silently fall
back to random. A placeholder that quietly degrades to a different strategy is
the worst possible failure here: the artifact would record
`method: "embedding"`, the agent and the human would both believe the subset had
visual coverage, and nothing would contradict them. Per the project's
test-integrity rule, a check that cannot run must fail rather than skip — the
same logic applies to a selector that cannot select.

Its `cost_class()` returns `W2` even while unimplemented, so the routing story
is already correct when it lands: embedding a 480-image parent is a scheduled
job, not a planning step.

### `subset_generate` (`W0` or as `cost_class()` reports)

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Workspace subset name |
| `parent` | `str` | — | Parent image directory |
| `selector` | `object` | — | `{class, params}` |
| `dry_run` | `bool` | `false` | Return the selection without writing |

Returns the chosen images, the per-group allocation when applicable, and the
recorded rationale.

### `subset_put` (`W0`) — a human-named subset

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Workspace subset name |
| `parent` | `str` | — | Parent image directory |
| `images` | `array[str]` | — | Parent-relative paths (§10.2), or globs resolved against `parent` |
| `note` | `str?` | `null` | Why these — recorded as the selection rationale |
| `coverage` | `object?` | `null` | Measured trait ranges from triage (§9.3) |
| `overwrite` / `dry_run` | `bool` | `false` | As §3 |

Records `selection.method: "user_named"` with the note. A human-picked subset is
first-class — `user_named` is a selection method, not a lesser one — and this is
the call `phenotypic-assay-triage` step 3 makes when you name the images
yourself.

`subset_get {name}` returns the artifact plus whether staging used symlinks or
copies (§10.3.1).

Because selectors resolve by bare class name like every other extensible class,
adding a fourth is a new subclass plus one `__init__.py` export. No tool
signature changes, no schema bump.

## 10.3.1 How the boundary is *enforced*

"The full dataset is touched exactly once" is a claim about mechanism, and a
claim like that is worth nothing unless something refuses.

The refusal cannot be `deploy_start` alone. `tune_start` and `pipeline_probe`
both take a raw `images` path, and `W2` tune work is explicitly allowed to run
unattended and to route to a full `sbatch` fleet — so an ordinary `tune_start`
pointed at the parent directory would spend full-dataset compute without ever
approaching the promotion gate.

**So subset-scoped tools take a `subset_id`, not a path:**

| Tool | Before | Now |
|---|---|---|
| `pipeline_probe` | `images: str` | `subset_id: str` — raw path allowed **only** while the workspace has no subset at all (bounded `W1`: ≤4 images) |
| `tune_start` | `images: str` | `subset_id: str` |
| `campaign_put` | `dataset.images` path | `subset_id`, recorded on the campaign |
| `deploy_start` | `images` + `scope` | `subset_id` + `scope`; `scope:"full"` resolves to `subset.parent` |

A raw parent path in a subset-scoped phase is refused with
`code: "subset_required"`. This also gives `campaign_status.comparable` (§8.3)
its dataset identity for free: every arm in a campaign shares one `subset_id`,
so arms are comparable by construction rather than by after-the-fact digest
comparison.

### Neither engine accepts a file list — the subset must be materialized

An earlier draft said the server "resolves `subset_id` → the subset's recorded
images list and **passes those to the engine**". **That is not implementable.**
Both call surfaces take a single *path*, not a list:

| Engine | Input | Consumed by |
|---|---|---|
| `python -m phenotypic.tune run` | `-i/--input`, help text literally "image directory" (`tune/__main__.py:49`) | `_load_images(input_dir)` → `Path(input_dir).iterdir()`, **non-recursive directory scan** (`_run.py:235-279`) |
| `python -m phenotypic` | `-i/--input`, `click.Path(dir_okay=True, file_okay=True)`, **no `multiple=True`** (`phenotypicCLI.py:721-730`) | `scan_directory_structure(input_path)` — walks root images, or one level of subdirectories as separate datasets (`_cli_directory_scanner.py:28-117`) |

There is no manifest flag, no repeated `-i`, no file-list parameter on either.
`--sample N` only randomly *thins* datasets already discovered by the scan; it
cannot select named images.

So the server **materializes staging directories** and passes those. **Two
layouts, because the two engines want opposite things:**

```
<workspace>/.phenotypic-mcp/subset-staging/<subset-digest>/
├── flat/                     # for tune — _load_images is a NON-RECURSIVE iterdir
│   ├── plateA_01.tif -> …/data/plates/plateA/plateA_01.tif
│   ├── plateA_07.tif -> …
│   └── plateB_03.tif -> …
└── nested/                   # for deploy — Metadata_Dataset comes from subdir names
    ├── plateA/plateA_01.tif -> …
    ├── plateA/plateA_07.tif -> …
    └── plateB/plateB_03.tif -> …
```

**A single layout cannot serve both, and picking either one alone breaks the
other.** `_load_images` (`tune/_tune_cli/_run.py:259-262`) is
`Path(input_dir).iterdir()` filtered to files — at the root of a *nested*
staging directory it sees only subdirectories, matches zero images, and the run
dies on `SystemExit("no images found under …")` (`tune/__main__.py:202-204`).
Conversely `scan_directory_structure` derives `Metadata_Dataset` from subdirectory
names, so handing *deploy* a flat directory silently relabels every row's dataset
to the staging folder name — the exact corruption nesting exists to prevent.

The split is cheap and safe because the two engines genuinely differ in what they
need: **tune has no dataset concept at all** (`_load_images` returns a flat
filename-sorted list of `GridImage`s; grouping for scoring comes from the
scorer's own CSV, not from directories), while deploy's whole output schema keys
off dataset identity. Both layouts are symlink trees under one digest, so the
marginal cost is inodes, not bytes.

An earlier draft specified only the nested layout and listed `tune_start` among
the tools materializing through it — which would have failed outright on the
spec's own headline example, a `data/plates` parent with `plateA/`/`plateB/`
subdirectories.

**Fidelity is a check, not just a property.** The staging builder verifies that
the layout it produced round-trips: `nested/` must reproduce exactly the dataset
names `scan_directory_structure` would derive from the parent for those images.
Nothing in the engines can catch a mismatch — `scan_directory_structure` only
rejects *internally* inconsistent directories (root images **and** subdirectories
together, `_cli_directory_scanner.py:97-103`); it has no way to know what the
parent looked like. So the check lives in the builder or nowhere.

Four properties it must have:

1. **It mirrors the parent's dataset substructure.** `scan_directory_structure`
   treats one level of subdirectories as separate datasets and rejects mixed
   structures, so a flat staging dir would silently collapse a multi-dataset
   parent into one dataset and change every `Metadata_Dataset` value. The
   subset artifact's `images` entries are therefore **parent-relative paths**
   (`plateA/plateA_01.tif`), not bare filenames — an earlier §10.2 example
   showed bare names, which cannot disambiguate two datasets containing
   `plate_001.tif`.
2. **Symlinks by default, copies on fallback.** Symlinks are cheap and a
   subset may be staged repeatedly across an unattended campaign. But Windows
   symlink creation needs elevated privileges or Developer Mode, and this
   project supports Windows — so the server probes once, falls back to copying,
   and **reports which it used** in `subset_get`. A silent copy of a large
   subset is a surprise worth surfacing.
3. **Keyed by the subset digest**, so re-staging an unchanged subset is a no-op
   and two concurrent arms share one staging directory rather than racing.
4. **It lives under `.phenotypic-mcp/`**, not under `runs/` or the parent — it
   is server scratch, and `--restart`/`--overwrite` semantics must never reach
   the parent images through it.

This staging layer is **new work that §1.6's reuse inventory missed** and §7's
prerequisites did not list. It is tracked as P6.

**The raw-path fallback is bounded to cheap tools.** `pipeline_probe` may take a
path while no subset exists, because it is capped at 4 images and holds the
compute slot — it cannot reach fleet scale. `tune_start`, `campaign_put`, and
`deploy_start` have **no** fallback: they refuse with `subset_required`. An
agent must therefore create a subset before anything unattended or
fleet-scale, which is what makes §10.1's invariant structural rather than
opt-in.

The single exception is `scope: "full"`, which is the *point* of the promotion
gate and is guarded by `promotion_token`.

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

```
promotion_request → [human says yes] → promotion_approve
                                            ↓
                          deploy_plan  {scope:"full"}   ← plan_token for the PARENT
                                            ↓
                          deploy_start {scope:"full"}   ← plan_token + promotion_token
```

**Both** `deploy_plan` and `deploy_start` take `scope`. This matters: a campaign
arm can mint a `plan_token` only for `scope:"subset"` (§10.4), so a full-dataset
run has no other way to obtain one — the plan must be drawn explicitly against
the parent, which is also what produces the sbatch preview and array sizing for
480 images rather than 24.

| `scope` | Requires | Runs against |
|---|---|---|
| `"subset"` (default) | `plan_token` | the subset's image list; reachable from a campaign arm |
| `"full"` | `plan_token` (scope=full) **and** `promotion_token` | `subset.parent` |

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
    {"code":"subset_coverage_unverified",
     "message":"Subset spans contrast_michelson 0.031–0.094 across 4 measured images. The
                parent's 480 images were NOT characterized, so whether the subset
                represents them is unknown, not confirmed. Selection was
                'user_named'."}]}}
```

Two properties this must have:

- **The estimate is measured, not guessed.** The subset run already produced real
  per-image timing, so the full-dataset node-hour figure has a basis. This is the
  strongest argument for subset-first development independent of safety: it makes
  the cost of the expensive step *knowable* before you commit.
- **Coverage is reported honestly, including its limits.** A winner tuned on 24
  easy plates may fail on the hard ones, and cost alone cannot reveal that. But
  v1 measures traits only on the subset — §10.3 rules full-parent
  characterization out of scope precisely because it is a substantial compute
  job. So the warning says the subset's range is **unverified against the
  parent**, not that a specific number of parent images fall outside it.
  Claiming the latter would require exactly the dataset-wide probing v1 defers,
  and asserting it anyway would be the more dangerous error: a false assurance
  of representativeness is worse than an admitted unknown.

**Full scope bypasses staging deliberately**, running against `subset.parent`
directly — the `flat/`+`nested/` split (§10.3.1) exists only for subset-scoped
work. The parent's *structure* is not re-scanned at promotion, and does not need
to be: a structural regression (say a stray image dropped beside the `plateA/`
subdirectories, which would trip `scan_directory_structure`'s mixed-structure
rejection at submit time) also changes the parent's file-set digest, so
`digest_matches_parent: false` catches it first and forces a fresh
`promotion_request`. Stated because §10.3.1 makes fidelity an explicit check for
staging, and a reader is entitled to ask why the full path has no equivalent.

`promotion_approve {promotion_id, human_response, note?}` records the decision,
mints the token, and appends a lineage row:

```json
{"ok":true,"data":{"promotion_id":"prom_2c81","status":"approved",
  "promotion_token":"pm_5d17…","expires":"2026-08-14T09:12:00Z"}}
```

`promotion_request`'s response carries `pending_human_ack: true` and an
`ack_prompt` summarizing the ask (winner, subset score, gap, node-hours,
coverage warnings), and `human_response` here is required — same reasoning as
§8.3: it cannot authenticate, but it makes skipping the human an explicit
fabrication rather than a silent default. The token is bound to `(pipeline digest, parent digest, scope)` — if the
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
  The real cliff is sharp and sits at 6: `derive_split`
  (`tune/_evaluation/_split.py:191-199`) returns `kind="none"` with an **empty**
  held-out set when `n_plates < min_heldout_plates` (default 6,
  `_evaluation/_held_out.py:48`) — every plate becomes calibration and the
  generalization gap is not merely noisy but absent. Above 6 there is no second
  discontinuity: Tier 3 sizing is `n_held = max(1, round(0.2 · n))`, growing
  smoothly (1 plate at n=6–9, 2 at 10–12, 3 at 13–17).

  So `subset_put` **errors** below 6 and **warns** below roughly 15, where a
  single held-out plate makes the gap a one-sample estimate.

  **Unless the parent itself is that small.** A 4-plate pilot workspace cannot
  produce a 6-image subset, and since `subset_id` is a hard requirement for
  `tune_start` / `campaign_put` / `deploy_start`, a hard floor would lock such a
  workspace out of tuning and deployment entirely with no stated next action.
  So when `n_images >= parent.n_images` — the subset *is* the parent — the error
  downgrades to `subset_too_small_for_heldout` and the run proceeds with
  `kind: "none"`: every plate calibrates, there is no held-out gap, and
  `tune_status` reports `gap: null` with that reason rather than a number that
  does not exist. Small pilots are a real workflow, not an edge case; what they
  cannot have is a generalization estimate, and saying so is better than
  refusing. An earlier draft
  cited "~12" as following from `min_heldout_plates = 6`; nothing in the split
  logic produces 12, and the real hazard is the hard zero at 6.
- **Subset compute is bounded but not free.** An unattended campaign with deploy
  arms still consumes an allocation. The campaign budget and profile caps (§5.2)
  are what bound it, and they bind on subset runs exactly as on any other.

## 10.6.1 Does promotion re-probe? Only when the headers disagree

The promotion estimate extrapolates subset per-image timing to the parent. That
extrapolation is wrong when the parent holds images the subset does not
represent *dimensionally* — larger frames, a different bit depth, a second
modality mixed in.

Always probing would add a `W1` step, and a `LocalComputeSlot` acquisition, to
every promotion. Never probing would let a silently-wrong estimate through. Both
are avoidable, because **the thing that breaks the extrapolation is readable
without decoding a single pixel.**

So promotion runs a two-tier check:

| Tier | Cost | What it does |
|---|---|---|
| **Always** — header sweep | `W0`, no decode, no slot | Read dimensions, bit depth, and channel count from every parent image header. Compare the distribution against the subset's. |
| **Only on mismatch** — re-probe | `W1`, 2 images | Probe 2 images drawn from `parent \ subset`, chosen from the *mismatching* stratum, and re-derive the estimate from that timing |

Header reads are cheap enough to run over a 480-image parent (TIFF/PNG headers,
not pixel data), and they catch the dominant failure directly: cost scales with
pixel count, so a parent whose images match the subset's dimensions and depth
extrapolates soundly, and one whose images do not is exactly the case worth
spending two probes on.

The promotion response reports which tier ran:

```json
"estimate":{"node_hours":18.4,"basis":"subset run: 3.4 s/image measured",
            "extrapolation_check":"headers match (1024x1536, 16-bit, 3ch across
                                   all 480); no re-probe needed"}
```

and on mismatch:

```json
"estimate":{"node_hours":41.7,
            "basis":"re-probed 2 images from the 4096x4096 stratum at 9.1 s/image",
            "extrapolation_check":"MISMATCH — 113 parent images are 4096x4096
                                   while every subset image is 1024x1536"}
```

Note that the header sweep also gives a *bounded, honest* version of the
coverage gap §10.5 warns about. It cannot tell you whether the parent spans a
biological trait range the subset misses — that would need the full-dataset
characterization §10.3 rules out of v1 — but it **can** state exactly how many
parent images differ dimensionally, which is a real fact rather than an
extrapolated one.

## 10.7 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-10.1 promotion re-probe~~ → **header sweep always, re-probe only on
  mismatch** (§10.6.1). What breaks the extrapolation is readable from headers
  without decoding, so the common case costs nothing and the failing case gets a
  measured estimate.
