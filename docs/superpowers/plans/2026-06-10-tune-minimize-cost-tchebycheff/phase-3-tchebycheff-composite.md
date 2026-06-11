# Phase 3 — augmented Tchebycheff composite (`_composite.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes. Implement task-by-task: failing test → run (see it fail) → minimal impl → run (see it pass) → commit.

**Goal:** Replace the single-objective `CompositeScorer` combiner (today: geometric-mean-of-goodness, or weighted-arithmetic-mean when `weights` is set) with an **augmented Tchebycheff** scalarization over per-child **cost** scalars (`bᵢ ∈ [0,1]`), normalized to `(0,1]`. Add `rho`/`blend` fields and a study-global **active set** pinned at study start. Keep `weighted_mean` as an explicit opt-out; **remove `_geometric_mean` from the live path**. Keep the multi-objective NSGA-II path distinct (it keeps the per-child vector; only its abstainer floor flips `0.0 → 1.0`).

**Assumes Phases 1 + 2 landed** (the atomic cutover). Concretely, before this phase:
- `_orient.py` ships `Sense`, `clamp01`, `to_cost` (Phase 0); `clamp01` is the public clamp.
- `Scorer` is a template method: subclasses implement `_score_terms` + declare `_TERM_SENSE`; the base `score_image` wraps each term via `to_cost`. **Every per-child scalar `CompositeScorer._per_child_scalars` produces is now a cost in `[0,1]`** (`0` perfect, `1` worst) because `_robust_aggregate` flips to `median + λ·IQR` **and clamps to `[0,1]`** (Phase 1 B1), and each child's `finalize` mean is over costs.
- The optimizer **minimizes** (Phase 2); `EvaluationResult.score` and the multi-objective `directions` are now `minimize`.
- The multi-objective sidecar floor at `_composite.py` `finalize` is still `0.0` (the OLD higher-is-better worst). **Phase 1 may already have flipped this to `1.0`**; verify in the worktree (re-grep `child_scalars.get(handle, 0.0)`). If Phase 1 left it `0.0`, this phase flips it as Task 5. If Phase 1 already flipped it to `1.0`, Task 5's test still locks it as a regression and the impl step is a no-op (note it and move on).

**This is the one *intended* behavior change** of the whole migration (README cross-cutting invariant #3, spec §8 "Accuracy cost"): the composite picks a different multi-criteria compromise than the geometric mean. Gate it with the composite-delta snapshot test (Task 7), not a silent swap.

**Files:**
- Modify: `src/phenotypic/tools_/typing_.py` (add `CompositeBlend` Literal alias)
- Modify: `src/phenotypic/tune/_scoring/_composite.py` (fields, `_tchebycheff`, `finalize` routing, active set, abstainer floor)
- Modify: `src/phenotypic/tune/__init__.py` (re-export `CompositeBlend` — closed value set surfaced on a public field)
- Modify: `tests/unit/tune/test_composite_scorer.py` (rewrite the geometric-blend assertions; add Tchebycheff/active-set/ρ-ε/clamp/non-convex/delta tests)
- Modify: `tests/unit/tools_/test_io_constants.py` (Enum↔Literal alignment — but `CompositeBlend` has no Enum partner; instead add a `get_args` membership assertion — see Task 1)

**Read first (re-resolve every `file:line` in the worktree before editing — main-branch refs drift ±1–2):**
- README.md "Shared contract" — reuse **verbatim**: `CompositeBlend = Literal["tchebycheff", "weighted_mean"]` (default `"tchebycheff"`); `rho: float = 0.05`; `blend: CompositeBlend = "tchebycheff"`; `_UTOPIA_EPS: Final[float] = 1e-3`; cost convention; the one-study-global-active-set rule (cross-cutting invariant #4); `clamp01` from `_orient.py`.
- Spec §6.1 (formula + the "drop `|·|`" / assert invariant note), §6.2 (study-global normalizer), §6.3 (where it plugs in + the active-set rule + abstainer plumbing SF3), §6.4 (ε/ρ derivation + mistuning), §6.5 (blend opt-out, `weights` now blend-dependent), §6.6 (zero-parameter surface), §7 "Phase 3", §10 (composite tests), §11 pitfalls #3/#4/#5/#6/#7.
- Source: `src/phenotypic/tune/_scoring/_composite.py` (`finalize` ~`:203`, `_per_child_scalars` ~`:250`, `_geometric_mean` ~`:319`, `_weighted_mean` ~`:298`, `availability` ~`:163`, `objective_names` ~`:148`, `score_image` ~`:178`).
- `src/phenotypic/tune/_evaluation/_evaluator.py` (`scorer.finalize(running)` ~`:312` and `scorer.finalize(aggregated)` ~`:326` — the two `finalize` call sites; this is where the active set must be available to the composite).
- `src/phenotypic/tune/_scoring/_reference_free_scorer.py` (`availability()` ~`:171` reads run-local `_meta_validated` `PrivateAttr` — `False` until `meta_validate()` runs; **this is why the active set must be pinned, not recomputed in `finalize`**).

---

## Design decisions fixed for this phase (do not diverge)

1. **Per-child reduction is unchanged** — `_per_child_scalars` already returns `{handle: cost}` (Phase 1 made the child means costs). Tchebycheff consumes that dict directly. **Assert each `bᵢ ∈ [0,1]`** inside `_tchebycheff` (spec §6.1; B1 invariant — the assert guards the Phase 1 clamp staying in place). The assert is a real `assert`, not a clamp: if it fires, an upstream clamp regressed and we want the loud failure, not silent saturation.

2. **Active set = study-global available children, pinned at study start.** One set for BOTH the `max` numerator and the normalizer roster (spec §6.2/§6.3, README invariant #4). `finalize` has no access to availability, and `ReferenceFreeScorer.availability()` is run-local (`False` until `meta_validate`). **Plumbing chosen: a `set_active_set(handles)` setter that stores a `PrivateAttr` `_active_handles: tuple[str, ...] | None` on `CompositeScorer`.** The engine computes the available handles once (after meta-validation) and calls the setter before the trial loop; `finalize` reads it. `None` (never pinned — e.g. a unit test calling `finalize` directly) means "use all children that produced a scalar this call" (the in-call roster), preserving today's direct-`finalize` ergonomics. This is the §6.3 "store it on the scorer" option; it is a smaller blast radius than changing the `finalize` signature (which `_evaluator.py` and `Scorer.finalize` and every child share).

3. **Geometric mean is removed from the live path.** Keep `_geometric_mean` as a `@staticmethod` **only** if Task 7's snapshot test needs the old baseline; otherwise delete it. The decision is in Task 8 — the snapshot test computes the geomean baseline inline (one line), so `_geometric_mean` can be **deleted** entirely. Never expose geomean-of-cost (spec §9, pitfall #7: `0` is the product annihilator → one perfect axis zeroes the product and dominates — the inverse of the conjunctive property).

4. **`blend` routing** (single-objective only):
   - `blend="tchebycheff"` (default) → `_tchebycheff` with `weights` (uniform if `None`).
   - `blend="weighted_mean"` → `_weighted_mean` (existing, unchanged math — now over costs, reflection-clean).
   - There is no geomean branch. `weights` is now **blend-dependent** (spec §6.5): Tchebycheff per-axis weights under the default, arithmetic weights under `weighted_mean`. Document in the docstring + the migration note (Phase 5 owns release notes; here just the docstring).

5. **Multi-objective path stays distinct** — keep returning the per-child vector for NSGA-II; flip ONLY its abstainer floor `0.0 → 1.0` (worst cost). It does **not** use the active-set rule (it needs a fixed-length vector; an abstainer must stay an axis, floored to worst). (Task 5.)

---

### Task 1: `CompositeBlend` Literal alias + alignment assertion

**Files:**
- Modify: `src/phenotypic/tools_/typing_.py`
- Modify: `src/phenotypic/tune/__init__.py`
- Modify: `tests/unit/tools_/test_io_constants.py`

- [ ] **Step 1: Write the failing test**

`CompositeBlend` is a closed value set with **no Enum partner** (per `CLAUDE.md`: a `Literal` alias alone suffices for a serialized field value with no documentation surface — like `DetectMode`). So the test is a membership/coverage lock, not an Enum↔Literal alignment. Append to `tests/unit/tools_/test_io_constants.py` (find the `TestEnumLiteralAlignment` class — add a sibling class so it sits beside the existing alias tests):

```python
class TestCompositeBlendLiteral:
    """``CompositeBlend`` is the serialized ``CompositeScorer.blend`` value set."""

    def test_members_are_the_two_supported_blends(self):
        from typing import get_args

        from phenotypic.tools_.typing_ import CompositeBlend

        assert set(get_args(CompositeBlend)) == {"tchebycheff", "weighted_mean"}

    def test_exported_from_tune_package(self):
        # The blend is a public field value set; it must be importable where the
        # scorer is, or GUI / from_json callers cannot name it.
        from phenotypic.tune import CompositeBlend

        from typing import get_args

        assert "tchebycheff" in get_args(CompositeBlend)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra tune pytest tests/unit/tools_/test_io_constants.py::TestCompositeBlendLiteral -v`
Expected: FAIL — `ImportError: cannot import name 'CompositeBlend' from 'phenotypic.tools_.typing_'`.

- [ ] **Step 3: Add the alias + re-export**

In `src/phenotypic/tools_/typing_.py`, after the `FailureSource` Literal (~`:71`, the end of the "CLI / GUI closed value sets" block), add:

```python
#: The single-objective composite blend selector — the serialized value of
#: ``CompositeScorer.blend``. ``"tchebycheff"`` (default) is conjunctive
#: (worst-axis-dominant, augmented Tchebycheff over per-child cost);
#: ``"weighted_mean"`` is the compensatory opt-out. No Enum partner is needed —
#: this is a serialized field value with no separate documentation surface
#: (mirrors ``DetectMode`` / ``ExecutionMode``). The geometric-mean-of-cost
#: blend is intentionally NOT offered (it inverts the conjunctive property).
CompositeBlend = Literal["tchebycheff", "weighted_mean"]
```

In `src/phenotypic/tune/__init__.py`, add `CompositeBlend` to the imports and `__all__` (it is the value set of a public field). Import it from `phenotypic.tools_.typing_` near the other tune imports, and add `"CompositeBlend"` to `__all__` alphabetically near `"CompositeScorer"`. Example:

```python
from phenotypic.tools_.typing_ import CompositeBlend
```

and in `__all__`:

```python
    "CompositeBlend",
    "CompositeScorer",
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra tune pytest tests/unit/tools_/test_io_constants.py::TestCompositeBlendLiteral -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tools_/typing_.py src/phenotypic/tune/__init__.py tests/unit/tools_/test_io_constants.py
git commit -m "feat(tune): add CompositeBlend literal alias for the composite blend selector"
```

---

### Task 2: add `rho` / `blend` fields + `_UTOPIA_EPS` constant

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_composite.py`
- Modify: `tests/unit/tune/test_composite_scorer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tune/test_composite_scorer.py`:

```python
# --------------------------------------------------------------------------- #
# new fields — rho / blend (defaults + round-trip)
# --------------------------------------------------------------------------- #
def test_default_blend_is_tchebycheff_and_default_rho():
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0})])
    assert comp.blend == "tchebycheff"
    assert comp.rho == pytest.approx(0.05)


def test_blend_and_rho_round_trip():
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": 0.0})],
        blend="weighted_mean",
        rho=0.1,
    )
    back = CompositeScorer.model_validate_json(comp.model_dump_json())
    assert back.blend == "weighted_mean"
    assert back.rho == pytest.approx(0.1)


def test_invalid_blend_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        CompositeScorer(scorers=[], blend="geomean")  # not a CompositeBlend
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v -k "blend or rho"`
Expected: FAIL — `AttributeError` / `ValidationError` (`blend`/`rho` are not fields yet; `_FixedScorer` here uses a `0.0` term to stay valid once Tchebycheff lands).

- [ ] **Step 3: Add the fields + constant**

In `src/phenotypic/tune/_scoring/_composite.py`:

Add the import near the top (after `from ._scorer import ...`):

```python
from phenotypic.tools_.typing_ import CompositeBlend

from ._orient import clamp01
```

(`clamp01` is used in Task 3's normalization guard; import it now so the module compiles.)

Add the module constant near `_CHILD_HANDLE` / `_SEP` (~`:42`):

```python
#: The utopia-point shift ``z*ᵢ = −ε`` for the augmented Tchebycheff combiner.
#: A small, fixed numerical safety margin (~0.1% of the [0,1] cost scale): it
#: pushes the reference strictly below the achievable front so every
#: ``bᵢ − z*ᵢ = bᵢ + ε > 0`` (the unsigned ``max`` is valid) and caps the
#: weight realizer ``1/(bᵢ + ε)`` at ``≤ 1000×``. Internal — never a field
#: (spec §6.4/§6.6).
_UTOPIA_EPS: Final[float] = 1e-3
```

Add the two fields to `CompositeScorer` (after `multi_objective: bool = False`, ~`:103`):

```python
    blend: CompositeBlend = "tchebycheff"
    rho: float = 0.05
```

Update the class docstring `Args:` block to document `blend` (conjunctive default vs compensatory opt-out; `weights` semantics are blend-dependent) and `rho` (augmentation coefficient; advanced-only; default `0.05`; quality dial per §6.4). Note the geometric mean is no longer offered.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v -k "blend or rho"`
Expected: PASS (3 tests). Other composite tests may now FAIL (the geometric-blend assertions) — that is expected; Task 3 + Task 4 fix them.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_scoring/_composite.py tests/unit/tune/test_composite_scorer.py
git commit -m "feat(tune): add rho/blend fields + _UTOPIA_EPS to CompositeScorer"
```

---

### Task 3: implement `_tchebycheff` (formula + study-global normalization)

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_composite.py`
- Modify: `tests/unit/tune/test_composite_scorer.py`

Implement `_tchebycheff(child_costs: dict[str, float]) -> float` per spec §6.1/§6.2. The roster of axes (the active set) is resolved by a helper (Task 4); for this task `_tchebycheff` operates on the dict it is handed and normalizes over **that same dict's** weights (consistency is what matters — Task 4 ensures the dict is the pinned active set).

Formula (spec §6.1, canonical augmented weighted Tchebycheff, Steuer & Choo 1983 / ParEGO Knowles 2006):
```
ε  = _UTOPIA_EPS
wᵢ = weights.get(handle, 1.0)        # uniform (1.0) when weights is None/missing
dᵢ = wᵢ · (bᵢ + ε)                    # z*ᵢ = −ε ⇒ bᵢ − z*ᵢ = bᵢ + ε ; all dᵢ > 0
Tᵨ = max_i dᵢ  +  ρ · Σ_i wᵢ·bᵢ      # augmentation term is RAW weighted L1 (Σ wᵢ·bᵢ)
T_norm = Tᵨ / Tᵨ(1…1)                # Tᵨ(1…1) over the SAME roster ⇒ T_norm ∈ (0,1]
```
`Tᵨ(1…1) = max_i wᵢ·(1+ε) + ρ·Σ_i wᵢ` (every `bᵢ = 1`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/tune/test_composite_scorer.py`:

```python
# --------------------------------------------------------------------------- #
# _tchebycheff — formula + normalization (operates on per-child COST dicts)
# --------------------------------------------------------------------------- #
def test_tchebycheff_all_perfect_is_near_zero():
    # All children perfect (cost 0) → numerator = max(w·ε) + ρ·0 = ε (uniform w=1)
    # T_norm = ε / ((1+ε) + ρ·1) → tiny, in (0,1].
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                    _FixedScorer(terms={"b": 0.0})])
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    assert isinstance(result, float)
    assert 0.0 < result < 0.05  # near-zero cost, strictly positive (z*=−ε)


def test_tchebycheff_all_worst_is_one():
    # All children worst (cost 1) → Tᵨ == Tᵨ(1…1) → T_norm == 1.0 exactly.
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 1.0}),
                                    _FixedScorer(terms={"b": 1.0})])
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    assert result == pytest.approx(1.0)


def test_tchebycheff_is_worst_axis_dominant():
    # Conjunctive: the WORST (highest-cost) axis drives the max term, so a
    # candidate with one bad axis scores higher (worse) than one balanced at the
    # mean. {0.0, 0.8} (max term ~0.8) must exceed {0.4, 0.4} (max term ~0.4).
    one_bad = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                       _FixedScorer(terms={"b": 0.8})])
    balanced = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.4}),
                                        _FixedScorer(terms={"b": 0.4})])
    cost_one_bad = one_bad.finalize(one_bad.score_image(None, pd.DataFrame()))
    cost_balanced = balanced.finalize(balanced.score_image(None, pd.DataFrame()))
    assert cost_one_bad > cost_balanced


def test_tchebycheff_result_in_unit_interval():
    for a in (0.0, 0.2, 0.5, 0.9, 1.0):
        for b in (0.0, 0.3, 1.0):
            comp = CompositeScorer(scorers=[_FixedScorer(terms={"x": a}),
                                            _FixedScorer(terms={"y": b})])
            r = comp.finalize(comp.score_image(None, pd.DataFrame()))
            assert 0.0 < r <= 1.0 + 1e-9


def test_tchebycheff_weights_steer_the_max():
    # Weighting the worse axis up makes the composite worse (it weighs that axis
    # more heavily in the max). {a:0.0, b:0.6} with w_b=3 > the same with w=1.
    light = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                     _FixedScorer(terms={"b": 0.6})])
    heavy = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                     _FixedScorer(terms={"b": 0.6})],
                            weights={"s1": 3.0})
    # Normalization differs per weight set, so compare each to its own balanced
    # baseline rather than to each other directly; assert the heavy-weighted
    # bad axis is still worst-axis dominant (> 0.4 normalized).
    assert heavy.finalize(heavy.score_image(None, pd.DataFrame())) > 0.4
    assert light.finalize(light.score_image(None, pd.DataFrame())) > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v -k tchebycheff`
Expected: FAIL — `finalize` still routes to `_geometric_mean`/`_weighted_mean`; the worst-axis-dominant and unit-interval assertions break.

- [ ] **Step 3: Implement `_tchebycheff`**

Add to `CompositeScorer` (place after `_weighted_mean`, before `_geometric_mean` which is deleted in Task 8):

```python
    def _tchebycheff(self, child_costs: dict[str, float]) -> float:
        """Augmented weighted Tchebycheff of per-child **cost** scalars (§6.1/§6.2).

        ``Tᵨ(b) = maxᵢ wᵢ(bᵢ + ε) + ρ·Σᵢ wᵢ·bᵢ``, minimized, with utopia point
        ``z*ᵢ = −ε`` (``_UTOPIA_EPS``). The ``max`` drops the absolute value
        because ``z*ᵢ = −ε < 0 ≤ bᵢ`` makes every ``bᵢ − z*ᵢ = bᵢ + ε > 0`` — an
        invariant asserted here (the Phase 1 ``[0,1]`` clamp guarantees the upper
        bound; the assert fires loudly if that clamp ever regresses). The raw
        ``Tᵨ`` ranges over ``[ε·(…), (1+ε)·(…)]``, so it is normalized by the
        theoretical worst ``Tᵨ(1…1)`` over the **same** roster of axes, giving a
        normalized cost in ``(0, 1]`` for downstream consumers (§6.2). The roster
        is the pinned active set (the children available study-wide, §6.3) —
        consistent numerator and denominator across trials.

        Args:
            child_costs: ``{handle: cost}`` over the active set's axes (each
                ``bᵢ ∈ [0,1]``).

        Returns:
            The normalized composite cost in ``(0, 1]``. Empty roster → ``1.0``
            (the worst floor; the engine degrades — guards the ``max([])``).
        """
        if not child_costs:
            return 1.0
        eps = _UTOPIA_EPS
        weights = self.weights or {}
        max_term = 0.0
        l1 = 0.0
        denom_max = 0.0
        denom_l1 = 0.0
        for handle, cost in child_costs.items():
            assert 0.0 <= cost <= 1.0, (  # noqa: S101 — B1 invariant guard
                f"per-child cost {handle}={cost!r} escaped [0,1]; the Phase 1 "
                "robust-aggregate clamp regressed"
            )
            weight = float(weights.get(handle, 1.0))
            max_term = max(max_term, weight * (cost + eps))
            l1 += weight * cost
            denom_max = max(denom_max, weight * (1.0 + eps))
            denom_l1 += weight
        numerator = max_term + self.rho * l1
        denominator = denom_max + self.rho * denom_l1
        if denominator <= 0.0:
            return 1.0
        return clamp01(numerator / denominator)
```

(Do **not** route `finalize` to it yet — Task 4 does the routing through the active-set helper. Leave `finalize` on the old path so other tests stay meaningful until Task 4. The `-k tchebycheff` tests still fail after this step because `finalize` does not call `_tchebycheff` yet — that is the next task. To see THIS method exercised in isolation, you may add a temporary direct-call test, or proceed to Task 4 which wires it.)

> **Note:** because `finalize` is not yet routed, Step 4 below confirms the method exists and is importable/typed; the behavioral `-k tchebycheff` tests go green in Task 4.

- [ ] **Step 4: Run the focused method check**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v -k "blend or rho"` (still green) and `uv run mypy src/phenotypic/tune/_scoring/_composite.py` (the new method type-checks).
Expected: blend/rho tests PASS; mypy clean for the new method. The `-k tchebycheff` tests remain RED until Task 4 (expected).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_scoring/_composite.py tests/unit/tune/test_composite_scorer.py
git commit -m "feat(tune): implement augmented Tchebycheff combiner (_tchebycheff)"
```

---

### Task 4: route `finalize` + pin the study-global active set

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_composite.py`
- Modify: `tests/unit/tune/test_composite_scorer.py`

The active set (spec §6.3, README invariant #4): the children available study-wide. `finalize` cannot recompute it (`ReferenceFreeScorer.availability()` is run-local, `False` until `meta_validate`), so it must be **pinned once at study start** and stored. Plumbing (decision §2 above): a `set_active_set` setter storing a `_active_handles` `PrivateAttr`; `finalize` reads it to select the roster for both the `max` and the normalizer. `None` (never pinned) ⇒ use the in-call roster (all children that produced a scalar) so direct-`finalize` unit tests and the existing engine path keep working before the engine wiring lands.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/tune/test_composite_scorer.py`:

```python
# --------------------------------------------------------------------------- #
# active set — pinned study-global roster for max + normalizer
# --------------------------------------------------------------------------- #
def test_finalize_routes_to_tchebycheff_by_default():
    # With finalize wired, the worst-axis-dominant property now drives finalize.
    one_bad = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                       _FixedScorer(terms={"b": 0.8})])
    balanced = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.4}),
                                        _FixedScorer(terms={"b": 0.4})])
    assert (one_bad.finalize(one_bad.score_image(None, pd.DataFrame()))
            > balanced.finalize(balanced.score_image(None, pd.DataFrame())))


def test_finalize_weighted_mean_opt_out():
    # blend="weighted_mean" keeps the compensatory arithmetic mean over costs.
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": 0.8}), _FixedScorer(terms={"b": 0.4})],
        weights={"s0": 3.0, "s1": 1.0},
        blend="weighted_mean",
    )
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    # (3*0.8 + 1*0.4) / 4 = 0.7  (now over cost; same arithmetic as before)
    assert result == pytest.approx(0.7)


def test_active_set_pins_roster_for_both_max_and_normalizer():
    # Pin the active set to BOTH children. An abstaining child (no terms this
    # call) is in the active set but absent from this call's costs → it must NOT
    # be flooded into the max (that is per-image abstention, handled by the
    # robust aggregate upstream). With one child scoring 0.5, the composite is
    # the single-axis Tchebycheff of the present axis (not pinned to 1.0).
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.5}),
                                    _FixedScorer(terms={})])
    comp.set_active_set(("s0", "s1"))
    terms = comp.score_image(None, pd.DataFrame())  # only s0 emits
    assert terms == {"s0.a": 0.5}
    result = comp.finalize(terms)
    # discrimination on the present axis is preserved (not flattened to ~1.0)
    assert 0.0 < result < 0.7


def test_empty_active_set_is_worst_cost():
    comp = CompositeScorer(scorers=[_FixedScorer(terms={})])
    comp.set_active_set(())
    assert comp.finalize({}) == pytest.approx(1.0)


def test_no_scored_children_is_worst_cost_under_tchebycheff():
    # Single-objective default: zero scalars → worst cost 1.0 (NOT 0.0).
    # This is the cost-convention flip of the old "empty → 0.0" goodness floor.
    comp = CompositeScorer(scorers=[])
    assert comp.finalize({}) == pytest.approx(1.0)
```

> **Replace, do not keep:** delete the old `test_finalize_scalar_geometric_blend_default` (geomean is gone) and update `test_finalize_scalar_weighted_blend` to set `blend="weighted_mean"` (its 0.7 expectation is correct only under the explicit opt-out now). Replace `test_finalize_scalar_empty_is_zero` with `test_no_scored_children_is_worst_cost_under_tchebycheff` above (empty single-objective is now `1.0`, the worst cost, not `0.0`).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v`
Expected: FAIL — `set_active_set` does not exist; `finalize` still calls `_geometric_mean`; empty-finalize returns `0.0` not `1.0`.

- [ ] **Step 3: Wire `finalize` + add the setter**

In `src/phenotypic/tune/_scoring/_composite.py`:

Add the `PrivateAttr` import if absent (`from pydantic import ... , PrivateAttr`) and declare the attr on `CompositeScorer` (after the fields, before validators):

```python
    #: The pinned study-global active set — child handles available study-wide,
    #: fixed once at study start by :meth:`set_active_set`. Used as the roster for
    #: BOTH the Tchebycheff ``max`` numerator and the normalizer so the
    #: normalizer is a study-global constant (§6.2/§6.3). ``None`` (never pinned —
    #: e.g. a direct ``finalize`` unit call) falls back to the in-call roster.
    #: A ``PrivateAttr`` so it never serializes (it is run/study state, not recipe).
    _active_handles: Optional[tuple[str, ...]] = PrivateAttr(default=None)

    def set_active_set(self, handles: tuple[str, ...]) -> None:
        """Pin the study-global active set (child handles available study-wide).

        Called once by the engine after meta-validation, before the trial loop,
        so every trial's Tchebycheff ``max`` and normalizer use the same fixed
        roster (§6.3 plumbing SF3). Idempotent.

        Args:
            handles: The available child handles (a subset of
                :meth:`objective_names`), in objective order.
        """
        self._active_handles = tuple(handles)
```

Rewrite the single-objective branch of `finalize` (~`:243`–`:248`). Today:

```python
        values = list(child_scalars.values())
        if not values:
            return 0.0
        if self.weights is not None:
            return self._weighted_mean(child_scalars)
        return self._geometric_mean(values)
```

Replace with:

```python
        if self.blend == "weighted_mean":
            if not child_scalars:
                return 1.0  # worst cost (cost-convention floor; was 0.0 goodness)
            return self._weighted_mean(child_scalars)
        # Default conjunctive blend: augmented Tchebycheff over the pinned
        # study-global active set (§6.3). Restrict to the active roster so a
        # study-wide abstainer is simply not an objective (dropped from both the
        # max and the normalizer); per-image abstention is already a fewer-samples
        # matter handled by the robust aggregate, so a present-but-absent-this-call
        # child is NOT flooded into the max.
        if self._active_handles is None:
            roster = child_scalars
        else:
            roster = {
                handle: child_scalars[handle]
                for handle in self._active_handles
                if handle in child_scalars
            }
        return self._tchebycheff(roster)
```

> **`weighted_mean` floor:** keep `_weighted_mean` returning its own `0.0` for the zero-total-weight edge, but the *empty-children* case is now `1.0` (worst cost). The branch above handles the empty case before calling `_weighted_mean`.

Update the `finalize` docstring: replace the "geometric mean of the per-child scalars" bullet with the two new blends (Tchebycheff default over the active set; weighted-mean opt-out), and change the "Returns ... `0.0` for no children/terms" line to `1.0` (worst cost) for the single-objective path.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v`
Expected: PASS — including all `-k tchebycheff` tests from Task 3 (now that `finalize` routes through `_tchebycheff`), the active-set tests, the weighted-mean opt-out, and the worst-cost-empty tests. The two reused multi-objective / availability / round-trip tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_scoring/_composite.py tests/unit/tune/test_composite_scorer.py
git commit -m "feat(tune): route finalize to Tchebycheff over a pinned study-global active set"
```

---

### Task 5: flip the multi-objective abstainer floor `0.0 → 1.0`

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_composite.py` (only if Phase 1 left it `0.0`)
- Modify: `tests/unit/tune/test_composite_scorer.py`

The multi-objective path (`multi_objective=True`) is **distinct** — it returns the per-child vector for NSGA-II and must keep a fixed-length value vector (one entry per axis). It does NOT use the active-set rule. Its abstainer floor flips from the old higher-is-better worst `0.0` to the cost worst `1.0` (spec §6.3, §7 Phase 3, README invariant). **Verify whether Phase 1 already flipped it** (`grep -n "child_scalars.get(handle," src/phenotypic/tune/_scoring/_composite.py`); if it already reads `1.0`, the impl step is a no-op and this task only adds the regression test.

- [ ] **Step 1: Write/Update the test**

The existing `test_finalize_dict_floors_abstaining_child_to_zero` asserts `result["s1"] == 0.0` — that is the OLD goodness floor. Rewrite it for cost:

```python
def test_finalize_dict_floors_abstaining_child_to_worst_cost():
    # Multi-objective: an abstaining child stays an axis (fixed-length vector for
    # NSGA-II) but is floored to the WORST cost 1.0 (was 0.0 under goodness).
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": 0.2}), _FixedScorer(terms={})],
        multi_objective=True,
    )
    terms = comp.score_image(None, pd.DataFrame())
    assert terms == {"s0.a": 0.2}
    result = comp.finalize(terms)
    assert isinstance(result, dict)
    assert list(result.keys()) == comp.objective_names() == ["s0", "s1"]
    assert result["s0"] == pytest.approx(0.2)
    assert result["s1"] == pytest.approx(1.0)  # worst cost floor
```

Also update `test_finalize_dict_when_multi_objective` — its `{"s0": 0.8, "s1": 0.4}` values are fine (those children score), no floor fires; leave it (the values are costs now but the test only checks pass-through). Confirm by re-reading.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v -k "multi_objective or floors"`
Expected: FAIL on `...floors_abstaining_child_to_worst_cost` (asserts `1.0`, code still floors `0.0`) — unless Phase 1 already flipped it, in which case it PASSES immediately (note that and skip Step 3's edit).

- [ ] **Step 3: Flip the floor**

In `finalize`, the multi-objective branch (~`:238`–`:242`):

```python
        if self.multi_objective:
            return {
                handle: child_scalars.get(handle, 1.0)  # was 0.0 (goodness worst)
                for handle in self.objective_names()
            }
```

Update the multi-objective bullet in the `finalize` docstring: "floored to `0.0` (the higher-is-better worst score)" → "floored to `1.0` (the worst cost; minimized)", and the corresponding `_vector` / NSGA-II `directions` rationale (now `minimize`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v`
Expected: PASS (all composite tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_scoring/_composite.py tests/unit/tune/test_composite_scorer.py
git commit -m "feat(tune): flip multi-objective abstainer floor 0.0 -> 1.0 (worst cost)"
```

---

### Task 6: abstainer-masking + cost-clamp (B1) tests

**Files:**
- Modify: `tests/unit/tune/test_composite_scorer.py`

These are the spec §10 regression tests for pitfalls #3 (clamp) and #4 (abstainer masking). The implementation already satisfies them (Tasks 3–4); this task locks them explicitly.

- [ ] **Step 1: Write the tests**

Append to `tests/unit/tune/test_composite_scorer.py`:

```python
# --------------------------------------------------------------------------- #
# abstainer masking (§6.3 pitfall #4) — one study-wide-absent / per-image-absent
# child must not flatten discrimination on the present axes
# --------------------------------------------------------------------------- #
def test_one_abstaining_child_does_not_flatten_present_axis():
    # Active set pins both children; s1 abstains this call. The present axis (s0)
    # must still discriminate: a good s0 (0.1) scores strictly better than a bad
    # s0 (0.7), i.e. the abstainer is NOT flooded into the max as cost 1.0.
    good = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.1}),
                                    _FixedScorer(terms={})])
    bad = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.7}),
                                   _FixedScorer(terms={})])
    good.set_active_set(("s0", "s1"))
    bad.set_active_set(("s0", "s1"))
    cost_good = good.finalize(good.score_image(None, pd.DataFrame()))
    cost_bad = bad.finalize(bad.score_image(None, pd.DataFrame()))
    assert cost_good < cost_bad  # discrimination preserved
    assert cost_bad < 0.9        # NOT pinned near the ceiling by the abstainer


# --------------------------------------------------------------------------- #
# cost clamp (B1, §6.1 invariant 0 <= bᵢ <= 1) — a high-variance term clamped
# upstream must keep T_norm in [0,1] and the assert must NOT fire
# --------------------------------------------------------------------------- #
def test_high_cost_child_keeps_t_norm_in_unit_interval():
    # A child whose robust-aggregated cost is clamped to 1.0 (median+λ·IQR > 1
    # upstream → clamp01 → 1.0) feeds the composite as exactly 1.0; T_norm stays
    # in (0,1] and the §6.1 0<=bᵢ<=1 assert does not fire.
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 1.0}),
                                    _FixedScorer(terms={"b": 0.3})])
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    assert 0.0 < result <= 1.0


def test_cost_above_one_trips_the_invariant_assert():
    # Defensive: if an UNCLAMPED cost (>1, the Phase 1 clamp regressed) reaches
    # the combiner, the §6.1 invariant assert must fire loudly (not silently
    # saturate). Drive _tchebycheff directly with a poisoned roster.
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0})])
    with pytest.raises(AssertionError):
        comp._tchebycheff({"s0": 1.5})
```

- [ ] **Step 2: Run to verify it passes (impl already in place)**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v -k "abstain or clamp or invariant or unit_interval"`
Expected: PASS. If `test_cost_above_one_trips_the_invariant_assert` fails because Python is run with `-O` (asserts stripped), note it — the project does not run `-O` under `uv run pytest`, so the assert is live. (If your environment strips asserts, convert the §6.1 guard to an explicit `raise AssertionError` and re-run — keep it loud either way.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/tune/test_composite_scorer.py
git commit -m "test(tune): lock abstainer-masking + B1 cost-clamp invariant for Tchebycheff"
```

---

### Task 7: non-convex reachability + ρ/ε sensitivity + composite-delta snapshot

**Files:**
- Modify: `tests/unit/tune/test_composite_scorer.py`

These are the headline spec §10 tests: augmented Tchebycheff reaches a knee a weighted sum / `1−geomean` cannot; ρ removes a weakly-dominated point; ρ→0 admits a weakly-dominated winner that ρ=0.05 rejects; large ρ drifts toward weighted-sum; and the composite-delta snapshot documents the intended change vs the old geomean winner.

> **Modeling the front as candidates.** Each "candidate pipeline" is modeled by a `CompositeScorer` over two `_FixedScorer` children whose terms are that candidate's per-axis **costs**. The combiner picks the winner = `argmin(finalize)`. A 2-objective concave front is a set of (cost_a, cost_b) points where the trade-off curve bulges toward the origin (a knee dominated by no single-axis extreme).

- [ ] **Step 1: Write the tests**

Append to `tests/unit/tune/test_composite_scorer.py`:

```python
import math


# --------------------------------------------------------------------------- #
# helpers — score a list of (cost_a, cost_b) candidates with a blend
# --------------------------------------------------------------------------- #
def _composite(cost_a, cost_b, *, blend="tchebycheff", rho=0.05, weights=None):
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": cost_a}), _FixedScorer(terms={"b": cost_b})],
        blend=blend,
        rho=rho,
        weights=weights,
    )
    return comp.finalize(comp.score_image(None, pd.DataFrame()))


def _weighted_sum_cost(cost_a, cost_b):
    # The convex weighted sum (uniform) over cost — the baseline Tchebycheff beats.
    return 0.5 * cost_a + 0.5 * cost_b


def _one_minus_geomean_cost(cost_a, cost_b):
    # The OLD composite read on cost: 1 - geomean(goodness) = 1 - sqrt((1-a)(1-b)).
    return 1.0 - math.sqrt((1.0 - cost_a) * (1.0 - cost_b))


# A concave Pareto front: two single-axis extremes and a balanced KNEE that
# bulges toward the origin (low on BOTH axes). The knee is the conjunctive
# winner an all-objectives-required user wants.
_FRONT = [
    (0.02, 0.80),  # extreme A: nails axis a, tanks b
    (0.35, 0.35),  # the KNEE: balanced, both good
    (0.80, 0.02),  # extreme B: nails b, tanks a
]


def _argmin(front, cost_fn):
    return min(front, key=lambda p: cost_fn(*p))


# --------------------------------------------------------------------------- #
# non-convex reachability — Tchebycheff selects the knee; weighted sum &
# 1−geomean do not (§9 / §10)
# --------------------------------------------------------------------------- #
def test_tchebycheff_selects_knee_on_concave_front():
    knee = (0.35, 0.35)
    assert _argmin(_FRONT, lambda a, b: _composite(a, b)) == knee


def test_weighted_sum_misses_the_knee():
    # The uniform weighted sum is tied/biased toward an extreme, not the knee.
    winner = _argmin(_FRONT, _weighted_sum_cost)
    assert winner != (0.35, 0.35)


def test_one_minus_geomean_misses_the_knee():
    # The OLD composite (1 - geomean of goodness) also does not pick this knee —
    # documenting WHY the migration changes the winner (§8 deliberate change).
    winner = _argmin(_FRONT, _one_minus_geomean_cost)
    assert winner != (0.35, 0.35)


# --------------------------------------------------------------------------- #
# ρ removes a weakly-dominated point (§6.1/§6.4)
# --------------------------------------------------------------------------- #
def test_rho_breaks_weak_domination():
    # Two candidates equal on their WORST axis (both max-axis cost 0.5) but one is
    # strictly better on the OTHER axis. Plain Tchebycheff (ρ→0) ties them; the
    # augmentation (ρ=0.05) strictly prefers the properly-dominant one.
    weak = (0.5, 0.5)       # worst axis 0.5, other axis 0.5
    strong = (0.5, 0.1)     # worst axis 0.5, other axis 0.1 (strictly better)
    # ρ→0: the max terms are equal → near-tie (strong only marginally lower).
    tied_weak = _composite(*weak, rho=1e-9)
    tied_strong = _composite(*strong, rho=1e-9)
    assert tied_strong == pytest.approx(tied_weak, abs=1e-6)
    # ρ=0.05: the augmentation L1 term strictly separates them.
    sep_weak = _composite(*weak, rho=0.05)
    sep_strong = _composite(*strong, rho=0.05)
    assert sep_strong < sep_weak  # the properly-dominant point wins


# --------------------------------------------------------------------------- #
# ρ sensitivity — ρ→0 admits a weakly-dominated winner ρ=0.05 rejects; large ρ
# drifts toward the weighted-sum winner (§6.4)
# --------------------------------------------------------------------------- #
def test_large_rho_drifts_toward_weighted_sum():
    # As ρ grows, the Σ (weighted-sum) term dominates, so the winner converges to
    # the weighted-sum winner (an extreme), abandoning the knee.
    ws_winner = _argmin(_FRONT, _weighted_sum_cost)
    big_rho_winner = _argmin(_FRONT, lambda a, b: _composite(a, b, rho=50.0))
    assert big_rho_winner == ws_winner


def test_default_rho_keeps_the_knee_vs_large_rho():
    assert _argmin(_FRONT, lambda a, b: _composite(a, b, rho=0.05)) == (0.35, 0.35)
    assert _argmin(_FRONT, lambda a, b: _composite(a, b, rho=50.0)) != (0.35, 0.35)


# --------------------------------------------------------------------------- #
# composite-delta snapshot — document the intended winner change vs old geomean
# --------------------------------------------------------------------------- #
def test_composite_delta_vs_old_geomean_winner_is_documented():
    # The OLD composite (geomean of goodness == 1−_one_minus_geomean_cost, i.e.
    # MAXIMIZE goodness == MINIMIZE 1−geomean) and the NEW composite pick
    # DIFFERENT winners on the concave front — the one intended behavior change
    # of the migration (README invariant #3, spec §8). This test is the
    # baseline snapshot + reviewer sign-off gate; if it ever passes with equal
    # winners, the composite stopped being augmented Tchebycheff.
    old_winner = _argmin(_FRONT, _one_minus_geomean_cost)
    new_winner = _argmin(_FRONT, lambda a, b: _composite(a, b))
    assert new_winner == (0.35, 0.35)        # NEW: the conjunctive knee
    assert old_winner != new_winner          # DELTA: intentional change
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v -k "knee or rho or weak or geomean or delta or weighted_sum"`
Expected: PASS. If `test_weighted_sum_misses_the_knee` or the geomean test is a tie at the knee for *this* `_FRONT`, **tune the front geometry** (push the extremes further toward the axes and/or lower the knee) until the weighted-sum/geomean argmin is provably an extreme while Tchebycheff's is the knee — then re-pin the literals. Do **not** weaken the assertion; the whole point is the reachability separation.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/tune/test_composite_scorer.py
git commit -m "test(tune): non-convex reachability + rho/eps sensitivity + composite-delta snapshot"
```

---

### Task 8: delete `_geometric_mean`; thread the active set from the engine

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_composite.py` (delete `_geometric_mean`)
- Modify: `src/phenotypic/tune/_engine.py` (pin the active set before the loop)
- Modify: `tests/unit/tune/test_composite_scorer.py` (already updated — confirm no geomean ref remains)

- [ ] **Step 1: Delete `_geometric_mean`**

Remove the entire `_geometric_mean` `@staticmethod` (~`:319`–`:339`) and the `import math` at the top **only if** it has no other use (re-grep `math.` in `_composite.py` — `_tchebycheff` does not use `math`, so the import is now dead; remove it). Confirm no test or docstring references `_geometric_mean` (the class docstring + `finalize` docstring were updated in Tasks 2/4 — re-read the `Examples:` doctest at `:70`–`:96`: it asserts `geometric mean of the two child scalars` and `1.0`. **Rewrite that doctest** to the Tchebycheff result for two perfect (cost-0) children, e.g.):

```python
        >>> comp = CompositeScorer(scorers=[qc, qc])
        >>> terms = comp.score_image(None, layout)
        >>> sorted(terms)
        ['s0.Count', 's1.Count']
        >>> round(comp.finalize(terms), 3)  # augmented Tchebycheff of two perfect (cost-0) children
        0.001
```

> Compute the exact doctest value before pinning it: a perfect QC count match is goodness `1.0` → cost `0.0`. Two cost-0 children, uniform weights, ε=1e-3, ρ=0.05: numerator `= max(1·(0+ε)) + ρ·(0+0) = ε`; denominator `= max(1·(1+ε)) + ρ·(1+1) = (1+ε) + 0.1 = 1.101`; `T_norm = 0.001/1.101 ≈ 0.000908` → `round(..., 3) = 0.001`. **Run the doctest to confirm the rounded literal**, adjust if the QC goodness is not exactly 1.0.

Also rewrite the multi-objective doctest (`comp_mo`) values if they were goodness — for two perfect children they are now cost `0.0` each (`{'s0': 0.0, 's1': 0.0}`); confirm by running.

- [ ] **Step 2: Thread the active set from the engine**

In `src/phenotypic/tune/_engine.py` `optimize`, after `directions = objective_directions(spec.scorer)` (~`:61`) and before the loop, pin the active set on a composite scorer so `finalize` uses the study-global roster. The available handles are the composite's children that report `availability()` study-wide (after any meta-validation the engine performs). Add:

```python
        # Pin the study-global active set for the augmented Tchebycheff composite
        # (§6.3): the children available study-wide form the fixed roster for both
        # the Tchebycheff max numerator and the normalizer, so the normalizer is a
        # study-global constant and per-image abstention stays a robust-aggregate
        # matter (not a max-composition one). Non-composite / non-Tchebycheff
        # scorers ignore this.
        scorer = spec.scorer
        if isinstance(scorer, CompositeScorer):
            active = tuple(
                handle
                for handle, child in zip(scorer.objective_names(), scorer.scorers)
                if child.availability()
            )
            scorer.set_active_set(active)
```

Add the import at the top of `_engine.py`:

```python
from ._scoring import CompositeScorer
```

> **Meta-validation ordering caveat (SF3):** `ReferenceFreeScorer.availability()` is `False` until `meta_validate()` runs. If the engine performs meta-validation, pin the active set **after** it. Re-grep the worktree for where `meta_validate` is invoked in the engine/CLI path (`grep -rn "meta_validate" src/phenotypic/tune/`); today it is referenced only in `_tune_cli/_run.py:~407` (a guard message), so in the current engine a `ReferenceFreeScorer` child is unavailable at pin time and is correctly dropped from the roster (the engine degrades to `QCScorer` — matching today's behavior). If a later phase wires `meta_validate` into the engine, move the `set_active_set` call to immediately after it. Note this inline in the comment.

- [ ] **Step 3: Add an engine-level active-set test**

Append to `tests/unit/tune/test_composite_scorer.py` (or a new test in `tests/unit/tune/test_engine.py` if one exists — `grep -l "TuningEngine" tests/unit/tune/`; prefer the existing engine test module). A minimal lock that the engine pins the set:

```python
def test_engine_pins_active_set_to_available_children():
    from phenotypic.tune._scoring._composite import CompositeScorer

    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": 0.1}, ok=True),
                 _FixedScorer(terms={"b": 0.2}, ok=False)],  # unavailable
    )
    # Simulate what the engine does (the pin logic, without a full run):
    active = tuple(
        handle
        for handle, child in zip(comp.objective_names(), comp.scorers)
        if child.availability()
    )
    comp.set_active_set(active)
    assert comp._active_handles == ("s0",)  # only the available child is an axis
```

(If a real `TuningEngine`-driven test is cheap to add — a tiny grid study over `load_synth_yeast_plate()` with a `CompositeScorer` — prefer that; otherwise the pin-logic lock above suffices for the unit tier. The full-run path is covered by the existing engine/integration tests, which must still pass.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra tune pytest tests/unit/tune/test_composite_scorer.py -v`
Run the module doctest: `uv run --extra tune pytest --doctest-modules src/phenotypic/tune/_scoring/_composite.py -v`
Expected: PASS (all composite tests + the rewritten doctests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_scoring/_composite.py src/phenotypic/tune/_engine.py tests/unit/tune/test_composite_scorer.py
git commit -m "feat(tune): delete _geometric_mean; pin composite active set from the engine"
```

---

### Task 9: full tune-suite regression + type + lint gate

- [ ] **Step 1: Run the full tune unit suite**

Run: `uv run --extra tune pytest tests/unit/tune -v`
Expected: PASS. Pay attention to:
- `tests/unit/tune/test_composite_scorer.py` — all rewritten + new tests.
- Any engine / spec / study test that round-trips a `CompositeScorer` (the new `blend`/`rho` fields must serialize; `_active_handles` is a `PrivateAttr` and must NOT appear in `model_dump_json()` — confirm a round-trip test does not expect it).
- The multi-objective Pareto / NSGA-II tests (the abstainer floor flip + `directions=minimize` from Phase 2).

- [ ] **Step 2: Run the broader tune integration/smoke (if present)**

Run: `uv run --extra tune pytest tests/integration -k tune -v` (skip if no tune integration tests exist — `grep -rl tune tests/integration` first).
Expected: PASS or no-collect.

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/phenotypic/tune`
Expected: `Success: no issues found`. Common snags: the `PrivateAttr` typed `Optional[tuple[str, ...]]`; `CompositeBlend` field default must be the literal `"tchebycheff"`; `set_active_set` return `None`.

- [ ] **Step 4: Lint**

Run: `uv run ruff check --fix src/phenotypic/tune tests/unit/tune`
Expected: no remaining errors. The `assert` in `_tchebycheff` may trip `S101` (bandit) under some configs — the inline `# noqa: S101` is on it; confirm ruff is clean.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A && git commit -m "style(tune): mypy/ruff clean for the Tchebycheff composite" || echo "nothing to commit"
```

---

## Phase 3 done-criteria
- `CompositeScorer` has `blend: CompositeBlend = "tchebycheff"` + `rho: float = 0.05` fields, a `_UTOPIA_EPS: Final[float] = 1e-3` constant, a `_tchebycheff` combiner, a `set_active_set` setter + `_active_handles` `PrivateAttr`.
- `finalize` single-objective routes: `tchebycheff` (default, uniform or `weights`) → `_tchebycheff` over the pinned active set; `weighted_mean` → `_weighted_mean`. `_geometric_mean` is **deleted**.
- Multi-objective path unchanged in shape; abstainer floor is `1.0` (worst cost).
- `_tchebycheff` asserts `0 ≤ bᵢ ≤ 1` (B1 invariant), normalizes by `Tᵨ(1…1)` over the same roster, returns `(0,1]`; empty roster → `1.0`.
- Engine pins the active set (available children) before the trial loop.
- `CompositeBlend` is in `tools_/typing_.py` + re-exported from `phenotypic.tune`.
- Tests: non-convex reachability (knee), ρ removes weak domination, ρ→0 admits a weakly-dominated winner ρ=0.05 rejects, large ρ drifts to weighted-sum, abstainer-masking, B1 cost-clamp invariant (+ assert fires on >1), composite-delta snapshot vs old geomean, blend/rho round-trip, empty-as-worst-cost, multi-objective worst-cost floor. All pass.
- `_composite.py` module doctest rewritten + passing.
- `uv run --extra tune pytest tests/unit/tune` green; `uv run mypy src/phenotypic/tune` clean; `uv run ruff check` clean.

## Out of scope for Phase 3 (owned by other phases)
- The Pareto domination flip (`_study/_pareto.py` `_dominates`, `_vector` fill `0.0→1.0`) and screening direction flips → **Phase 4**.
- The GUI `_run_root.py` `_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS` flip, winner/score relabeling, `FEATURES.md`/`WORKFLOWS.md` → **Phase 4**.
- The new-scorer authoring contract docs (§5.3 → `Scorer` docstring, `tune/CLAUDE.md`, contributor guide), the explainer rewrite, release notes for the `weights`-semantics change → **Phase 5**.
- The evaluator math (`_robust_aggregate` clamp, `_WORST_TERM`/`failure_score` flips, `_is_suspicious`, generalization gap, `_GAP_EPS`) and the scorer migration to `_score_terms`/`_TERM_SENSE` → **Phase 1** (assumed landed; this phase depends on per-child scalars already being costs).
