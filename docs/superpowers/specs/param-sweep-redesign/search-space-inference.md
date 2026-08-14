# Search-Space Inference (`infer_search_space`)

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md).
Deep dive on **master §5** and **decision D7**: how a pydantic `ImagePipeline`
is turned into a *generous, reviewable* proposal of tunable domains that a human
or agent edits before any tuning budget is spent.

- **Status:** Design settled (pre-implementation). Lands in **Phase 3** (master §12).
- **Maps to:** master §5 (`infer_search_space`), D7 (auto-derived space, human/agent
  reviews before tuning).

---

## 1. Purpose and where it fits

`infer_search_space(pipeline_or_json) -> InferredSearchSpace` is the **bootstrap**
that converts a configured pipeline into a candidate search space. It exists
because operations are pydantic v2 models with typed, constrained, self-describing
fields (`model_json_schema()` carries each field's docstring-derived description),
so the tunable domains can be **mined from the operation contract** instead of
hand-written.

The output is deliberately **generous** — it over-includes rather than under-includes
— because the downstream **screening / importance pass** ([screening-importance.md](screening-importance.md), planned)
prunes low-influence knobs after a few trials. The cost of an extra *categorical or
bounded* knob is a screening freeze; the cost of a *missing* knob is an optimum the
engine can never reach. So inference errs toward inclusion, and every guess is
flagged for review (D7).

Inference is **not** the optimizer-facing object. It produces a reviewable
*proposal*; calling `.to_search_space()` collapses it to the clean `SearchSpace`
the strategies consume (§7).

---

## 2. Two tiers and their precedence

Every field resolves through two tiers, **Tier 1 first**:

1. **Tier 1 — per-field tuning metadata (`TuneSpec`).** A precise, opt-in marker the
   operation author attaches to a field (§3). When present, it is authoritative.
2. **Tier 2 — type / constraint heuristics.** When no `TuneSpec`, the field's
   annotation, pydantic constraints, and default drive an automatic guess (§4).

`TuneSpec` always wins over the heuristics. A `TuneSpec(tunable=False)` is the
explicit "never tune this" escape hatch and short-circuits Tier 2 entirely.

---

## 3. The `TuneSpec` marker

### Mechanism

`TuneSpec` is an `Annotated` extra that mirrors the existing field-marker pattern
(`ColumnRef`, `NdArrayField`, `OperationField` in `tools_/typing_.py`): it is a
**complete no-op at runtime** and is read *only* by `infer_search_space`, via
`op.model_fields[name].metadata` (where pydantic v2 stores `Annotated` extras).

```python
from typing import Annotated, ClassVar
from phenotypic.tune import TuneSpec

class BlurGauss(ImageEnhancer):
    sigma:    Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 2.0
    truncate: Annotated[float, TuneSpec(tunable=False)]      = 4.0   # never tuned
```

At runtime `sigma` is still a plain `float`; `BlurGauss(sigma=999.0)` constructs
exactly as before. The marker carries `__eq__`/`__hash__` (like `_ColumnRefMarker`)
so a duplicate in an `Annotated` chain de-dupes.

**Marker placement — walk the annotation tree.** pydantic surfaces only the
*outermost* `Annotated` extras in `model_fields[name].metadata`; a marker nested under
`Optional` or a container (the natural `Optional[Annotated[float, TuneSpec(...)]]`)
would be **silently missed**. So inference resolves the `TuneSpec` by **walking the
annotation tree** — exactly as the GUI `OperationRegistry` already does for
`_OperationFieldMarker` (`gui/_operation_registry.py`'s `_has_operation_field_marker`)
— rather than reading the top-level metadata list alone. Authors may then place
`TuneSpec` either outermost (`Annotated[Optional[float], TuneSpec(...)]`) or nested;
both resolve.

### Hint-only, not a validator

`TuneSpec` declares the **search** domain, never the **valid** domain. Validity stays
the job of pydantic `Field(ge=, le=)`. The two are genuinely different: `truncate` is
*valid* on `(0, ∞)` but you would only *search* `[2, 5]`; `sigma` accepts any positive
float but you search a slice. Conflating them would make it impossible to express
"valid to ∞, search only this window" or to hand-pass an out-of-search-but-valid value.

Where a field carries **both** a `Field(ge=, le=)` constraint and a `TuneSpec`, the
search range must lie within the validity bounds. Inference asserts the invariant

```
TuneSpec[low, high] ⊆ [ge, le]
```

and **errors at inference time** if an author writes a `TuneSpec` that escapes the
field's own constraints (you cannot search where the value is invalid). The check reads
all four `annotated_types` bounds pydantic emits into the field metadata —
`Ge`/`Gt`/`Le`/`Lt` — and respects strictness (`low ≥ ge`, `low > gt`, etc.).

**Limitation — validator-enforced bounds are invisible.** This project's prevailing
convention enforces numeric bounds in a `field_validator`, *not* `Field(ge=, le=)`
(e.g. `InoculumDetector` deliberately keeps a bound as a validator, not `Field(gt=0)`).
Those bounds live in imperative code, never in `model_fields[name].metadata`, so the
`⊆` check **cannot see them** — a `TuneSpec` exceeding a *validator*-enforced bound
passes inference and only fails later at apply/trial time (the backstop). An author
adding a `TuneSpec` to a validator-bounded field should also express the bound as
`Field(ge=, le=)` so the invariant can guard it.

### Field set

```python
TuneSpec(low=None, high=None, *, step=None, log=False, categories=None, tunable=True)
```

| Field        | Purpose                                          | Example                                      |
|--------------|--------------------------------------------------|----------------------------------------------|
| `low`, `high`| numeric search bounds (positional)               | `TuneSpec(0.5, 5.0)`                          |
| `log`        | log-scale sampling                               | `TuneSpec(1e-3, 1.0, log=True)`               |
| `step`       | discretize (int step / quantized float)          | `TuneSpec(2, 20, step=2)`                     |
| `categories` | override / subset the auto-derived categorical   | `TuneSpec(categories=["reflect", "nearest"])` |
| `tunable`    | `False` excludes the field outright              | `TuneSpec(tunable=False)`                     |

A distribution/prior weight is deliberately **omitted** for now (YAGNI — Optuna can
grow one behind the same marker later).

---

## 4. Tier-2 type / constraint heuristics

When a field has no `TuneSpec`, its annotation, constraints, and default produce an
automatic domain. The full mapping:

| Field shape                                   | Inferred domain                              | `source`              | `needs_review` |
|-----------------------------------------------|----------------------------------------------|-----------------------|:--------------:|
| `bool`                                        | `Categorical([True, False])`                 | `bool`                | no             |
| `Enum` / `Literal[...]`                        | `Categorical(members)` (`categories` subsets)| `enum` / `literal`    | no             |
| bounded `int`/`float` (`Field(ge=, le=)`)      | `IntRange`/`FloatRange`; `log` auto-on when `lo>0 ∧ hi/lo ≳ 100` | `bounded` | no |
| unbounded numeric, default `d > 0`            | `[d/unbounded_factor, d·unbounded_factor]` (linear) | `unbounded_heuristic` | **yes**  |
| unbounded numeric, default `d ≤ 0`            | — *excluded* —                               | (excluded)            | —              |
| free-form `str`, paths, names                 | — *excluded* — (open set)                    | (excluded)            | —              |
| `NdArrayField`, `OperationField`, containers  | — *excluded* — (not scalar-tunable)          | (excluded)            | —              |
| `T \| None` (union with `None`)               | infer over `T` if `T` is tunable             | (T's source)          | **yes**        |
| multi-type union (`A \| B`, neither `None`)    | — *excluded* —                               | (excluded)            | —              |

> **Reality check — `bounded` is rare today.** Operations in the current codebase
> almost never declare `Field(ge=, le=)`; numeric bounds, where they exist, are enforced
> in `field_validator`s (invisible here — see §3). So in practice **most numeric knobs
> fall through to the unbounded heuristic** and arrive `needs_review=True`; the `bounded`
> branch is correct but seldom taken until operations adopt `Field`-declared constraints.
> This inverts the naive expectation that "only unbounded knobs need review" — by
> default, nearly all do — and correspondingly raises the `proposal.needs_review`
> autonomy gate (§7) for most pipelines. That conservatism is deliberate, and it is the
> argument for migrating high-value tunable fields to `Field(ge=, le=)` or giving them a
> `TuneSpec`.

### The unbounded window

For an unbounded numeric with a positive default, the window is multiplicative:

```python
_DEFAULT_UNBOUNDED_FACTOR: Final[float] = 4.0   # span 16× (= factor²)

def infer_search_space(pipeline_or_json, *,
                       unbounded_factor: float = _DEFAULT_UNBOUNDED_FACTOR,
                       recurse_nested: bool = True) -> InferredSearchSpace:
    ...
```

`factor = 4` (span 16×) is the defensible middle: factor 2 (4× span) is too timid and
risks excluding the optimum; factor 10 (100× span) trips log-scale and makes the
optimizer waste samples. **Crucially, the "be generous, screening prunes" philosophy
argues for including *extra knobs*, not for *wider ranges*** — screening *freezes*
low-importance parameters, it does not *shrink* a continuous range, so a too-wide
range is pure wasted sampling. Hence a moderate width, not a maximal one. The factor
is exposed as a keyword so the MCP / a power user can retune it without editing source.

The window anchors on the **instance's current value**, not the class default
(they differ only when the input pipeline overrode the default). `--auto-space` exists
to bootstrap from a concrete pipeline, so a `pipeline.json` with `sigma=3.0` yields
`[0.75, 12.0]` — centered on the user/agent's operating point. A bad anchor is caught
by `needs_review`, optimizer exploration, and screening. Falls back to the class
default if the instance value is unset. For integer fields the window bounds round
**outward** (floor the low, ceil the high), so the inferred range never excludes a
value the multiplicative window implied — e.g. `min_size=50` → `[12, 200]`.

### Degenerate defaults (`d ≤ 0`): surface, don't fabricate

A multiplicative window assumes a positive scale to anchor on. It collapses at `d = 0`
(`cval` → `[0, 0]`) and inverts at `d < 0`. Because `needs_review` means "*plausible*
guess, sanity-check it," and a window around 0 or a negative default is not even
plausible, inference **does not invent one**. The field is excluded *and listed
prominently* in the proposal's "couldn't infer — declare a `TuneSpec`" section with the
reason `"unbounded numeric with non-positive default; add a TuneSpec to tune"`. It is
visible, never silently dropped, and the fix is one annotation.

### No docstring-range parsing

`BlurGauss.sigma`'s docstring literally says *"Typical range: 0.5–5.0"* — tempting,
but free-text parsing is fragile, and a *silently wrong* range is worse than a *visibly
flagged* guess. Inference never parses it. Instead, the proposal already surfaces the
full docstring `description` (§7), so a reviewer sees "Typical range: 0.5–5.0" right
next to the inferred `[0.5, 8.0]` and corrects it — and the durable fix is the author
adding a `TuneSpec(0.5, 5.0)`. The surfaced description *subsumes* a parsed hint.

---

## 5. `Presence` auto-wrapping (optional ops)

An operation declares itself droppable with a ClassVar, matching the existing
behavior-flag convention (`_HIGHER_IS_BAD`, `_exposes_agg_func` on the QC checks):

```python
class BlurGauss(ImageEnhancer):
    _tune_optional: ClassVar[bool] = True   # an optional denoiser — safe to drop
```

Default `False` on `BaseOperation`. Inference auto-wraps a pipeline position in a
presence choice **only** when `type(op)._tune_optional is True` — it is **never guessed**
from "this looks optional." A detector never opts in (you cannot drop the detector);
optional denoisers / refiners do. A reviewer can still add or remove presence by hand
in the proposal; the flag only drives the *automatic* wrapping.

### Flat conditional representation

A present-or-absent op is encoded **flat**, not as a subtree:

- the position gets a synthetic knob `<Op>.__enabled__: Categorical([True, False])`
  (`source="presence_optin"`, `needs_review=False`), and
- the op's own param-knobs are tagged `conditional_on={"<Op>.__enabled__": True}`.

A strategy samples `__enabled__` first; when `False`, it skips the conditional children.
The `__enabled__` **dunder** makes a collision with a real operation field essentially
impossible and visually flags the knob as synthetic.

This reproduces the **legacy `Presence` Cartesian semantics exactly** — absent collapses
to a *single* combination, present multiplies by the children — which is precisely what
the master §9 regression lock requires (`Presence` + nested params must equal the **frozen
golden `generate_sweep_manifest` fixture** — `sweep` is deleted in the hard cutover, master
§9, so the lock is against the golden, not live code). It also maps 1:1 onto Optuna's
define-by-run
(`suggest_categorical` then conditional `suggest_*`) and is trivial for the homegrown
`GridStrategy`/`RandomStrategy` to flatten.

---

## 6. Nested-operation recursion (one level)

Some fields hold *other* operations via `OperationField`: a **single** nested op
(`FilamentousFungiDetector.inoculum_detector`) or a **list**
(`CompositeDetector.ops`). Inference recurses **exactly one level** into them by
default (`recurse_nested=True`), applying the same Tier-1 → Tier-2 rules and nested
`_tune_optional` wrapping to the nested instance(s).

For:

```python
[ BlurGauss(sigma=2.0),                                    # position 0
  CompositeDetector(ops=[                                     # position 1
      OtsuDetector(ignore_zeros=False),
      WatershedDetector(min_size=50),
  ]) ]
```

inference emits:

| Knob key                       | Domain                | Provenance (source · class) |
|--------------------------------|-----------------------|------------------------------|
| `0.sigma`                      | `FloatRange(0.5, 8.0)`| unbounded_heuristic · BlurGauss |
| `1.ops[0].ignore_zeros`        | `Categorical([T,F])`  | bool · OtsuDetector          |
| `1.ops[1].min_size`            | `IntRange(12, 200)`   | unbounded_heuristic · WatershedDetector |

A nested knob's `description` comes from the **nested class's** `model_json_schema()`;
its provenance records the nested class name for display *and* apply-time safety.

*(This example reflects the ops as they exist today — un-annotated — so `0.sigma`
resolves via the unbounded heuristic to `[0.5, 8.0]`. Had `BlurGauss` carried the
illustrative `TuneSpec(0.5, 5.0, log=True)` from §3, `0.sigma` would instead resolve
`source="tune_spec"`, domain `FloatRange(0.5, 5.0, log=True)` — Tier 1 over Tier 2.)*

### Canonical knob keys are root-relative paths

Recursion forces a naming upgrade: the canonical key is a **root-relative path with a
parent identifier** — `1.ops[0].ignore_zeros` — which also fixes a latent
top-level bug where two `BlurGauss`s in one pipeline would collide on a bare
`BlurGauss.sigma`. The exact form of the *parent-identifier* segment (raw position
index vs. a class-occurrence tag like `BlurGauss#0`) is owned by the `SearchSpace`
naming design (master); this doc uses the position-index form in examples and only
requires that recursion have *some* stable parent identifier. The position-index form
is stable against same-class duplication but shifts under top-level reordering — a
tradeoff the `SearchSpace` doc finalizes.

### Contained risks

1. **List-position identity is fragile.** `ops[0]` is keyed by *position*;
   reordering the list would silently retarget the knob and misalign a saved
   `best_pipeline.json` / `trials.parquet`. **Mitigation:** provenance records the
   nested class, and the params→pipeline builder **validates the class on apply** —
   asserting `ops[0]` is still an `OtsuDetector`, erroring loudly if not. (List
   members are themselves optional — the real annotation is `list[OperationField | None]`
   — so recursion **skips `None` slots** and apply-time validation expects a slot may be
   `None`.)
2. **Heterogeneous members.** An `ops` list can mix classes, so `[0].*` and `[1].*`
   carry disjoint knob sets keyed off the *current* contents — consistent with the
   instance-value philosophy.
3. **Depth cap = 1, presence is top-level only.** A nested op's *own* `OperationField`
   children are excluded (no chaining). And in v1 **`_tune_optional` presence-wrapping
   applies only to top-level pipeline positions** — a nested op is recursed for its
   parameters but never gets its own `__enabled__`. Together these cap every
   `conditional_on` chain at depth 1 (a single `<TopOp>.__enabled__`), so grid
   enumeration and any future Optuna conditional `suggest_*` never face a two-level
   conditional. Lifting this (nested presence → a two-level `conditional_on` chain) is
   deferred.
4. **Regression lock is safe.** Legacy `Sweep`/`generate_sweep_manifest` never swept
   nested-op params, so recursion is strictly **additive**: a space with no nested knobs
   enumerates byte-identically to legacy (§9 holds). Nested knobs are outside the lock's
   surface.
5. **Apply-back builder.** Setting `ops[0].sigma` means constructing the leaf op,
   injecting it into the parent's kwargs, and reconstructing the parent (ops are
   keyword-only / immutable). The Evaluator's params→pipeline builder walks nested paths
   — mechanical, but more than flat-field assignment.

`recurse_nested=False` yields the flat-only proposal (no nested knobs).

---

## 7. The `InferredSearchSpace` proposal

`infer_search_space` returns a reviewable proposal, **not** the optimizer-facing object.

> **Type layer:** these are **frozen pydantic value-models** (and the domains a pydantic
> discriminated union), per [`engine-architecture.md`](engine-architecture.md) §5, which
> **owns the canonical type definitions** — so `SearchSpace`/`InferredSearchSpace` round-trip
> through `TuningSpec`. The `@dataclass` sketch below illustrates the *shape* only.

```python
@dataclass(frozen=True)
class Knob:
    key: str                      # canonical root-relative path, e.g. "1.ops[0].ignore_zeros"
    domain: Domain                # Categorical | IntRange | FloatRange
    source: KnobSource            # tune_spec | bool | enum | literal | bounded
                                  #   | unbounded_heuristic | presence_optin
    needs_review: bool
    description: str              # from the owning class's model_json_schema()
    conditional_on: dict[str, object] | None = None   # e.g. {"BlurGauss.__enabled__": True}

@dataclass(frozen=True)
class Excluded:
    key: str
    reason: str                   # human-readable; e.g. "unbounded numeric with non-positive default; add a TuneSpec"

@dataclass(frozen=True)
class InferredSearchSpace:
    knobs: list[Knob]
    excluded: list[Excluded]

    def to_search_space(self) -> SearchSpace: ...        # collapse to the clean optimizer object
    @property
    def n_knobs(self) -> int: ...
    @property
    def n_needs_review(self) -> int: ...
    @property
    def n_excluded(self) -> int: ...
    @property
    def needs_review(self) -> bool: ...                  # any(k.needs_review for k in knobs)
```

- `description` is sourced from `model_json_schema()["properties"][field]["description"]`,
  which the project guarantees is auto-derived from the Google-style `Args:` docstring —
  so the "knob's docstring description as context" (§5 of the master) comes for free.
- `.to_search_space()` is the only thing the optimizer ever calls; the `source` /
  `needs_review` / `description` / `excluded` data never leaks into the strategies.
- The summary properties double as the **autonomous-vs-review gate**: an MCP agent can
  check `proposal.needs_review` (or `n_needs_review`) to decide whether to pause for human
  confirmation before tuning, rather than silently optimizing over shaky guesses.

---

## 8. Inference core and surfaces

### One core, two adapters

`infer_search_space` accepts **either** a live `ImagePipeline` **or** a `pipeline.json`
path/dict, and normalizes both to a list of live operation **instances** — because
deserializing the JSON through the existing
`SerializablePipeline._deserialize_pipeline_config` / `_find_class_in_phenotypic`
registry already yields a live `ImagePipeline` (an unresolvable class name raises the
same clear error `OperationField` deserialization does). That pipeline holds its ops in
the **insertion-ordered `pipeline._ops` dict**, so the adapter reads `_ops.values()` in
order as the position basis for the `0.`, `1.`, … key prefixes (the live-pipeline entry
point reads the same). So there is exactly **one inference core**,
`_infer_from_operations(ops)`, and the live-pipeline and json-path entry points are thin
adapters over it. No new registry, no parallel code path.

### Drivers

- **CLI `--auto-space`** (from a `pipeline.json`): prints a review table — reliable knobs
  (`✓`), `needs_review` guesses (`⚠`), and the "couldn't infer" / excluded list with
  reasons — then proceeds with the (possibly edited) space.
- **MCP `tune_infer_space(pipeline_json)`**: returns the structured proposal — domains,
  `source`, `needs_review`, and each knob's docstring `description` — as agent context, so
  the agent can widen/narrow/freeze knobs and decide whether `needs_review` warrants a
  human check.

---

## 9. Review and the screening handoff

The proposal is **generous by design** (§1); the human/agent edits it (D7) — narrowing
shaky ranges, dropping irrelevant knobs, adding presence — and then tuning begins.
After a few trials, the **screening / importance pass** ([screening-importance.md](screening-importance.md), planned)
ranks knobs by influence (and interaction) and freezes the low-importance ones for a
focused second round. So inference need not be precise — it needs to be *complete and
honest*: include plausibly-relevant knobs, flag every guess, and never silently drop a
field.

**Böck-trap carry-forward.** Inference emits *grid-independent* domains — there is no
normalization-over-the-tested-set anywhere in this stage — so it cannot introduce the
min–max normalization instability that would corrupt fANOVA inputs downstream (master
§2; reference-free doc §B.3 (Böck), recommendation 2). The dependency only flows the other way:
screening *consumes* trial scores, which must be grid-independent for separate reasons.

---

## 10. Testing

- **Tier-2 branch coverage** — `infer_search_space` over synthetic pydantic ops
  exercising every row of the §4 table: bool, `Enum`/`Literal`, bounded with/without
  log auto-trip, unbounded `d>0`, unbounded `d≤0` exclusion, `str`/path exclusion,
  `NdArrayField`/`OperationField`/container exclusion, `T | None`, multi-type union.
- **Tier-1 precedence + invariant** — a `TuneSpec` overrides the heuristic; a
  `TuneSpec` escaping a field's `Field(ge=, le=)`/`Gt`/`Lt` raises at inference time;
  `tunable=False` excludes; a `TuneSpec` nested under `Optional`/a container is still
  resolved (annotation-tree walk, not top-level metadata only).
- **Presence** — `_tune_optional=True` emits a `<Op>.__enabled__` knob with
  `conditional_on` children; `False` emits none.
- **Regression lock** — `GridStrategy` enumeration over a conditional space
  (`__enabled__` + tagged children) **equals** the current `generate_sweep_manifest`
  Cartesian product, against a saved fixture (master §9). Nested-recursion knobs are
  asserted *absent* when `recurse_nested=False`, preserving byte-identity.
- **Nested recursion** — one-level recursion emits path-keyed knobs for single and
  list `OperationField`; depth is capped (nested-of-nested excluded); `None` list members
  are skipped; nested ops are never presence-wrapped in v1; apply-time class-validation
  raises when a list member's class changed.
- **Adapters** — `infer_search_space(live_pipeline)` and
  `infer_search_space(pipeline_json_path)` produce the **same** proposal; an unresolvable
  class name raises the registry error.
- **Anchoring** — the unbounded window centers on the instance's current value, falling
  back to the class default when unset; `unbounded_factor` widens/narrows it.

Fixed seeds throughout (project reproducibility requirement).

---

## 11. Resolved design choices

These were open questions during design; recording the resolutions so they are not
re-litigated.

1. **Unbounded window factor** — kept at `4` (span 16×; generosity belongs to knob
   *inclusion*, not range *width*), exposed as the `unbounded_factor` keyword with a
   named `Final` default so it is retunable without a source edit.
2. **Nested-op recursion** — **on by default** (`recurse_nested=True`), **one level**,
   **lists recursed** with `[i]` indexing + class-validation on apply; depth capped at 1;
   additive vs. the regression lock.
3. **Docstring-range hints** — **never parsed**; the surfaced `description` already
   carries the hint, and the durable fix is a `TuneSpec`.
4. **Presence sentinel name** — `<Op>.__enabled__` (dunder: collision-proof, visibly
   synthetic).

### Deferred to the `SearchSpace` design (master)

- The exact **parent-identifier segment** of a canonical knob key (raw position index vs.
  a class-occurrence tag), given the top-level reordering tradeoff (§6).
- The `Domain` / `SearchSpace` / `to_search_space()` types themselves — a **hard upstream
  dependency**: Phase 3 inference cannot land until the master `SearchSpace` design is
  settled (master §14: `Sweep` range types in-place vs. a richer `SearchSpace`).

### Still genuinely open (planning)

- Whether a *future* version recurses **beyond one level** into nested ops, and the
  knob-naming / conditional-depth scheme that would require.
- Whether nested ops may eventually be **presence-wrapped** (`_tune_optional` on a nested
  op → a two-level `conditional_on` chain), which the v1 top-level-only rule (§6) defers.
