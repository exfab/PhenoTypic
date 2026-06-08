# Structured Knob Targets (typed parameter references)

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md)
and [`search-space-inference.md`](search-space-inference.md). A pre-Phase-5 refinement to the
**search-space authoring surface**: replace the stringly-typed knob keys (`"0.sigma"`) with
typed **target classes** + early cross-validation + a discovery catalog, so that programmatic
and **agent/MCP** construction is hard to get wrong — while keeping the canonical string as the
internal/serialized rendering so the engine and the golden lock are untouched.

- **Status:** Design settled (pre-implementation; all OQs resolved 2026-06-05). Lands **before
  Phase 5** (the GUI `6c` space-edit forms and the future MCP workstream build on it).
- **Maps to:** master §5 (knob-key grammar / position-index), search-space-inference §3/§6/§8
  (nested grammar, `InferredSearchSpace`, the inference that *generates* keys),
  dash-copilot-design §5 (`6c` consumes the proposal as forms), and the **MCP workstream** (the
  primary motivation — an LLM authoring keys from scratch is error-prone).
- **Follows:** the project's pydantic-v2 conventions (annotated fields, no `__init__`,
  `field_validator`/`model_validator`, `Literal` discriminators) and the closed-set typing rules
  in the root + `tools_/CLAUDE.md`.

---

## 1. Problem

A `Knob` references a pipeline parameter by a **string key** in one of three grammars
(master §5, search-space-inference §6):

| Case | String key |
|------|-----------|
| Flat field on a top-level op | `"0.sigma"` |
| Presence toggle (op on/off) | `"0.GaussianBlur.__enabled__"` (or bare `"0.__enabled__"`) |
| Depth-1 nested-op leaf | `"0.refiners[1].min_size"` |

Stringly-typed keys are error-prone for **programmatic / agent (MCP)** construction: a typo
(`"sigam"`), a wrong index, or a malformed grammar (`__enabled__` dunder, `[i]` brackets) is
easy to produce and only fails **late** — deep in `build_pipeline`, at evaluation time. The
goal is to make the *authoring* surface typed and **validated where the caller submits**, while
leaving the proven internal machinery alone.

**Non-goal / honest scope.** A typed class does **not** make the leaf *field name* stop being a
string (pydantic field names are strings). The anti-mistake levers are **structure** (an `int`
op index; explicit presence/nesting instead of dunder/bracket grammar), **early validation**
(cross-check against the actual pipeline at spec construction, with did-you-mean), and
**discovery** (a catalog the agent *selects* from rather than authoring). This design delivers
all three.

---

## 2. The `.key` bridge (what stays untouched)

Every target renders a **`.key` property** equal to today's canonical string. The internal
machinery keeps consuming `.key`:

- `build_pipeline` (`_evaluation/_builder.py`) keeps parsing `.key` strings — **no change**.
- The strategies, the engine, `trials.parquet`, and the **grid byte-compat golden lock**
  (`test_grid_golden_manifest.py` / `test_grid_byte_compat_lock.py`) are **unaffected** (they
  operate on the canonical string, which is unchanged).

So targets are a typed authoring/serialization layer that *compiles down to* the existing
strings. This is what keeps the change low-risk and additive.

---

## 3. The target union

A discriminated union (`kind` discriminator) — one frozen pydantic model per grammar case — in
a new `tune/_search_space/_targets.py`. The bare names (`Param`/`Presence`/`Nested`) read
unambiguously because they are accessed through the `targets` subpackage (§8):
`targets.Param(op=0, field="sigma")`.

```python
class Param(BaseModel):             # kind="param"     →  "0.sigma"
    kind: Literal["param"] = "param"
    op: int
    field: str
    op_class: Optional[str] = None        # optional cross-check (see §5)

class Presence(BaseModel):          # kind="presence"  →  "0.GaussianBlur.__enabled__"
    kind: Literal["presence"] = "presence"
    op: int
    op_class: Optional[str] = None        # also renders the classed key form

class Nested(BaseModel):            # kind="nested"    →  "0.refiners[1].min_size"
    kind: Literal["nested"] = "nested"
    op: int
    field: str
    index: int
    leaf: str
    op_class: Optional[str] = None

KnobTarget = Annotated[Param | Presence | Nested, Field(discriminator="kind")]
```

- **`.key` property** on each renders the canonical string. `Presence.key` renders the
  classed form (`"0.GaussianBlur.__enabled__"`) when `op_class` is set, else the bare
  (`"0.__enabled__"`) form — both already parse in `build_pipeline`.
- **`parse_key(s: str) -> KnobTarget`** — the inverse, reusing the existing parse logic in
  `_builder.py::_parse_key` (a single shared parser; the builder's `FlatKey`/`PresenceKey`/
  `NestedKey` dataclasses are an internal apply-time representation and stay private — the public
  target union is the authoring representation, and `parse_key` bridges a string into it). Used
  for the `key=` convenience (§4) and the legacy-string load path (§7).
- All three are `frozen=True`, keyword-only, with Google `Args:` docstrings (so
  `model_json_schema()` documents the MCP tool surface).

**Depth.** `Nested` is depth-1 (single `(field, index, leaf)`), matching the current cap.
Raising the cap is out of scope here (a separate, moderate change to the parser/applier/inference
— see the depth assessment in the project notes); the union would generalize `Nested` to a
`path` of segments then.

---

## 4. `Knob` — dual constructor (decision B)

`Knob.target: KnobTarget` is the canonical field. A `model_validator(mode="before")` coerces a
legacy `key="0.sigma"` into `parse_key("0.sigma")`, so **both** spellings construct:

```python
from phenotypic.tune import Knob, FloatRange
from phenotypic.tune import targets

Knob(target=targets.Param(op=0, field="sigma"), domain=FloatRange(low=0.5, high=8.0))
Knob(key="0.sigma",                              domain=FloatRange(low=0.5, high=8.0))   # coerced
```

- `Knob.key` remains as a **read property** → `self.target.key` (so existing `.key` readers and
  `SearchSpace.keys()` keep working).
- `conditional_on` references **targets** instead of strings:
  `conditional_on=((targets.Presence(op=0, op_class="GaussianBlur"), True),)`. A legacy string
  parent key is coerced the same way.
- `SearchSpace` is otherwise unchanged: `.keys()` → `[k.target.key for k in knobs]`; add a
  `.targets()` accessor.

Keeping `key=` is "for now" (decision B) — a one-line migration aid and escape hatch; the
canonical/authoring surface is `target=`.

---

## 5. Validation — the cross-check (decision A=2, posture C)

The structural win is **early validation against the actual pipeline**. A
`model_validator(mode="after")` on `TuningSpec` (which holds *both* `pipeline` and
`search_space`) walks every knob target and validates it:

- **op in range** — `0 <= op < len(pipeline ops)`; else a clear out-of-range error.
- **`op_class` cross-check (decision A=2)** — when a target carries `op_class`, assert the op
  actually at `op` *is* that class (`op=0, op_class="GaussianBlur"` fails if index 0 is an
  `OtsuDetector`). This catches the "wrong op / index drift" mistake class directly.
- **field / leaf exists** — `Param.field` ∈ `type(op).model_fields`; `Nested.leaf` ∈ the
  nested op's `model_fields`. On a miss, a **`difflib` did-you-mean** suggestion plus the list of
  available fields.
- **nested resolution** — `Nested.field` is an op-valued list field, `index` in range, the slot
  is not `None`.
- **rich errors regardless of `op_class`** — even a bare-index target's error names *what is
  actually at that index* ("op 0 is a `GaussianBlur`; for the `OtsuDetector` use op 1"), so an
  agent gets an actionable correction.

**`op_class` posture = C (optional on the class, always populated by discovery/inference).**
`op_class: Optional[str] = None` on every target, so **hand-authoring may omit it** (zero
verbosity). But **every programmatic producer always sets it** — `pipeline_targets` (§6),
`infer_search_space`, the GUI `6c` form, and the MCP tool all emit `op_class`-bearing targets, so
every realistic path is wrong-op-cross-checked for free. The only uncovered case is a
deliberately-bare hand-written target, which still gets the field-exists check + the rich
"op 0 is actually a GaussianBlur" error.

This complements (does not replace) the existing **apply-time `⊆` backstop** in `build_pipeline`
(`_rebuild_op_or_raise_with_keys`), which still catches *validator*-enforced value bounds that
`model_fields` metadata can't see. The new cross-validator catches *targeting* errors early;
the apply-time backstop catches *value* errors at reconstruction. (A `SearchSpace` built without
a pipeline can't be cross-checked at `SearchSpace` construction — validation lives at
`TuningSpec`, which is exactly where an MCP submits a complete spec.)

---

## 6. Discovery catalog (decision D — in scope)

So the agent/GUI **selects** valid targets rather than authoring them. Lives in the `targets`
subpackage alongside the refs (§8):

```python
class TunableParam(BaseModel):
    target: KnobTarget                       # the structured ref to use (op_class always set)
    op_class: str
    value_type: Literal["float", "int", "bool", "categorical"]   # not 'kind' — avoids
                                             # colliding with the target union's discriminator
    default: Any                             # current value on the pipeline
    suggested_domain: Optional[Domain]       # from TuneSpec / heuristic inference
    description: str                         # the field docstring / TuneSpec text
    needs_review: bool

def pipeline_targets(pipeline) -> list[TunableParam]: ...
```

- Built on the **existing `_infer.py` mining** (which already walks ops, reads `TuneSpec`
  markers, classifies field types, and infers domains) — `pipeline_targets` re-surfaces that as
  per-parameter descriptors and **auto-fills `op_class`** (so every catalog target is
  cross-checked per posture C).
- Relationship to existing inference (unchanged in role): `infer_search_space()` keeps returning
  the **opinionated default `SearchSpace`** (picks fields, sets bounds, flags `needs_review`);
  `pipeline_targets()` returns the **full structured catalog** the GUI `6c` form and the MCP
  "what can I tune?" tool present. `InferredSearchSpace.knobs` now carry structured `target`s too
  (they are `Knob`s). The two share the mining; neither duplicates it.

---

## 7. Serialization (decision E — structured)

A `Knob` serializes its target **structurally**:

```json
{"target": {"kind": "param", "op": 0, "field": "sigma", "op_class": "GaussianBlur"},
 "domain": {"kind": "float", "low": 0.5, "high": 8.0}}
```

The discriminated union round-trips natively through pydantic. **On load**, `parse_key` also
accepts the **legacy string form** (`{"key": "0.sigma", ...}` or a bare `"0.sigma"`), so a frozen
pre-change `tuning_spec.json` / fixture still validates. Structured is the canonical emitted form
(self-describing for agent + human inspection); the string form is accepted but not emitted.

---

## 8. Public surface + namespacing (decision = subpackage)

The **entire param-reference + discovery surface lives in `phenotypic.tune.targets`** — the bare
class names get their context from the subpackage, and the top-level `phenotypic.tune.__all__`
(39 symbols) gains **zero** new entries:

```python
from phenotypic.tune import targets               # targets.Param, targets.Presence, …
from phenotypic.tune.targets import Param, Presence, Nested, KnobTarget, \
                                    TunableParam, pipeline_targets
```

- In `targets`: `Param`, `Presence`, `Nested`, `KnobTarget` (union alias), `parse_key`,
  `TunableParam`, `pipeline_targets`.
- Stays top-level: `infer_search_space` (it returns a `SearchSpace`, the higher-level
  construct), and the consumers `Knob` / `SearchSpace` / `TuningSpec`.

---

## 9. What changes / what doesn't

| Area | Change |
|------|--------|
| `tune/_search_space/_targets.py` (new) | `Param` / `Presence` / `Nested`, `KnobTarget`, `parse_key`, `.key` renders, `TunableParam`, `pipeline_targets` |
| `tune/targets.py` or `tune/targets/__init__.py` (new) | The public `targets` subpackage re-exporting the above |
| `tune/_search_space/_space.py` (`Knob`) | `target` field; `key=` coercion; `.key` property; `conditional_on` → targets |
| `tune/_spec.py` (`TuningSpec`) | `model_validator(after)` cross-check (op range, class, field/leaf, did-you-mean) |
| `tune/_search_space/_infer.py` | Emit structured targets (op_class-bearing); back `pipeline_targets` |
| `_builder.py`, engine, strategies, `trials.parquet`, golden lock | **Unchanged** (`.key` bridge) |
| Tests / fixtures | Update `SearchSpace(knobs=...)` call sites to structured (or rely on `key=` coercion) |

---

## 10. Testing

- **`.key` round-trip** — for each kind, `parse_key(t.key) == t` (modulo an optional `op_class`),
  and `t.key` equals the legacy canonical string.
- **`Knob` dual constructor** — `key=` and `target=` produce equal knobs; `.key` reads through.
- **Cross-validator** — a `TuningSpec` with: an out-of-range op; an `op_class` mismatch; a missing
  field (asserts the did-you-mean suggestion text); a `Nested` into a non-list / `None` slot /
  bad index — each raises with an actionable message. A valid spec passes.
- **Discovery** — `pipeline_targets(synth_pipe)` returns descriptors with filled `op_class`,
  `value_type`, defaults, `suggested_domain`, `needs_review`; every target carries `op_class`
  (posture C); `infer_search_space` still returns its default space (unchanged behavior).
- **Serialization** — a spec round-trips structured; a legacy string-form fixture still loads.
- **Locks stay green** — `build_pipeline` unchanged; the grid byte-compat golden lock and the full
  `tests/unit/tune` suite stay green (`-n 8`).

---

## 11. Resolved choices (all settled 2026-06-05)

1. **Op anchoring (A)** — position `int` **+ an optional `op_class` cross-check** (validated when
   present, auto-filled by discovery/inference). Full op-identity (uuid / `(class, occurrence)`) is
   deferred future headroom; the engine still uses position internally.
2. **Keep the string key (B)** — `Knob` accepts both `target=` (canonical) and `key=` (coerced),
   "for now". The internal `parse_key` stays regardless.
3. **Decomposition (C)** — the **3-class discriminated union** (clean MCP schema), not a single
   class with a `kind` field + optional attrs.
4. **Discovery scope (D)** — `TunableParam` + `pipeline_targets()` are **in this change**.
5. **Serialization (E)** — **structured** target in `tuning_spec.json`, legacy string accepted on
   load.
6. **Naming** — bare `Param` / `Presence` / `Nested` (no suffix), union `KnobTarget`, kwarg
   `target=`; read through the `targets` subpackage (`targets.Param(...)`).
7. **Namespacing** — the whole param-reference + discovery surface lives in
   `phenotypic.tune.targets`; top-level `__all__` is unchanged. `infer_search_space` stays
   top-level.
8. **`op_class` posture (C)** — optional on the class; **always populated** by discovery,
   inference, the GUI `6c`, and the MCP — so every programmatic path is wrong-op-cross-checked,
   with no hand-authoring burden.

**Deferred (not part of this change):** nesting depth >1 (`Nested` → a `path` of segments + the
recursive applier), and full op-identity anchoring.
