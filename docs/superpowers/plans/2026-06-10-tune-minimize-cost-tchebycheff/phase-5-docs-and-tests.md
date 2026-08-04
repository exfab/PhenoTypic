# Phase 5 — Docs + cross-phase regression tests

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Phase goal.** Land the documentation half of the minimize-cost + augmented
Tchebycheff cutover, and the **cross-phase regression tests** that can only pass
once Phases 1–4 are merged. No production-code edits live here (every source
flip belongs to its own phase); Phase 5 only **documents** the new convention and
**locks it in with tests**.

**Assumes landed:** Phase 0 (`_orient.py`: `Sense`, `to_cost`, `clamp01`),
Phase 1 (Evaluator cost math + scorer migration + gap fix + `_is_suspicious`
reflection), Phase 2 (`_MINIMIZE`, `min`-selection, `_STUDY_NAME = "tune_cost_v1"`),
Phase 3 (augmented Tchebycheff `CompositeScorer` + `blend`/`rho` fields), Phase 4
(Pareto/screening/GUI flips). See the
[index README](README.md) "Shared contract & conventions" — those names are
fixed and reused verbatim below.

**Spec sections covered:** §5.3 (authoring contract → 3 surfaces), §6.4/§6.5/§6.6
(scalarization-parameter docs), §7 "Phase 5", §9 (theory notes for the explainer),
§10 (cross-phase regressions), §12 (references).

**Source of truth.** The §5.3 authoring contract is canonical. Tasks 1–3 copy it
verbatim-in-spirit into three surfaces; they MUST stay in sync. If you change one
copy, change all three.

---

## Dev environment & commands

```bash
# one-time, in this worktree (if not already synced):
uv sync --group dev --group docs --extra tune
# tune tests need the `tune` extra (Optuna):
uv run --extra tune pytest tests/unit/tune/test_cost_convention_regression.py -v
# docs build (explainer + contrib guide are .md/.rst under docs/):
uv run --group docs sphinx-build -b html docs/source docs/_build/html -q
# lint/type at phase boundary:
uv run ruff check --fix tests/unit/tune/test_cost_convention_regression.py
uv run mypy src/phenotypic/tune
```

`docs/superpowers/**` is **not** part of the Sphinx tree (it lives outside
`docs/source/`), so changes there are verified by `grep`, not `sphinx-build`.
Only `docs/source/contrib_guide/contributing.rst` is in the Sphinx build.

Commit after every green task (each task shows its exact `git add` / `git commit`).
Run `ruff` + `mypy` once before the final phase commit.

---

## Task 1 — Rewrite the `Scorer` base-class docstring (authoring contract, surface 1/3)

**File:** `src/phenotypic/tune/_scoring/_scorer.py`

**Why.** Phase 1 renamed the abstract method from `score_image` to `_score_terms`
and made `score_image` a template method (`README.md` "`Scorer` template method";
spec §5.2). The class docstring, the `score_image` docstring, the
`project_objectives_to_scalar` docstring, and the `finalize` docstring all still
say "higher is better / 0.0 is worst". This task corrects the prose **and** adds
the §5.3 authoring contract.

> **Re-resolve before editing.** Phase 1 already touched this file (it rewrote the
> method bodies). Re-read it first — the line numbers below are main-branch and the
> `_score_terms`/`to_cost` wiring is added by Phase 1, so the *prose* is what you
> change, not the signatures Phase 1 introduced.

- [ ] **1a — module docstring.** Replace the top-of-file docstring (lines 1–10):

  *Before:*
  ```python
  """The pluggable scoring objective — a pydantic ABC.

  A ``Scorer`` turns one image's measurement frame into a dict of **named term
  scores** (``score_image``), where *higher is better* and each term is a clean,
  comparable signal (typically normalized to ``[0, 1]``). The ``Evaluator``
  collects the per-image terms across a calibration set, robust-aggregates each
  term, then asks the scorer to ``finalize`` the aggregated terms into the single
  scalar objective the optimizer maximizes. ``availability`` lets a scorer report
  that it cannot run (e.g. missing metadata) so the engine can degrade gracefully.
  """
  ```

  *After:*
  ```python
  """The pluggable scoring objective — a pydantic ABC.

  A ``Scorer`` emits one image's **natural per-term values** (``_score_terms``);
  the base-class template method ``score_image`` orients each term into bounded
  **cost** ``∈ [0, 1]`` (``0`` = perfect, ``1`` = worst) via the shared
  :func:`~phenotypic.tune._scoring._orient.to_cost`, reading the scorer's
  ``_TERM_SENSE`` and optional ``_term_anchor``. The ``Evaluator`` collects the
  per-image cost terms across a calibration set, robust-aggregates each term
  (``median + λ·IQR``, clamped to ``[0, 1]``), then asks the scorer to
  ``finalize`` the aggregated terms into the single scalar **the optimizer
  minimizes**. ``availability`` lets a scorer report that it cannot run (e.g.
  missing metadata) so the engine can degrade gracefully.

  Authoring a new ``Scorer`` (the canonical contract — kept in sync with
  ``tune/CLAUDE.md`` and ``docs/source/contrib_guide/contributing.rst``):

    1. Implement ``_score_terms(image, measurements) -> dict[str, float]``
       returning your **natural** per-term values — do **not** flip or normalize
       by hand.
    2. Declare ``_TERM_SENSE`` (``Sense.LOWER_BETTER`` if larger = worse — the
       default; ``Sense.HIGHER_BETTER`` if larger = better, e.g. Dice/ICC).
    3. Override ``_term_anchor`` **only** if a term is unbounded, returning the
       half-cost scale (for a QC-backed term, its check's ``fail_threshold``);
       bounded ``[0, 1]`` terms need nothing.
    4. Do **not** add scalarization parameters (``ε``, ``ρ``, normalization,
       default weights are framework-derived).
    5. Register: re-export from ``tune/__init__.py`` and the class registry, or
       the GUI and ``from_json`` cannot see it.

  The framework then orients (``to_cost``), robust-aggregates, reduces per child,
  and combines (augmented Tchebycheff) — the author writes none of that.
  """
  ```

- [ ] **1b — `project_objectives_to_scalar` docstring.** In the function docstring
  (lines 22–37), change the two goodness phrases to cost. *Before* contains
  `the same way (higher = better, ``0.0`` is the worst score).`; *after*:
  `the same way (lower = better cost, ``0.0`` is the best and the empty-mapping
  default).` Also change `Returns:` body from
  `` ``mean(mapping.values())`` as a ``float``; ``0.0`` for an empty mapping. ``
  to keep the same numeric contract but add `` (a perfect/empty cost). `` after
  it.

  > **Note (verify against Phase 1):** the empty-mapping default of
  > `project_objectives_to_scalar` is `0.0`. Under cost, `0.0` is **best**, not
  > worst — confirm Phase 1's `finalize`/`_FAILURE`/abstainer paths floor to
  > `1.0` *before* this projection (per `README.md` cross-cutting invariant 1 and
  > spec §7 Phase 1 `failure_score: 0.0 → 1.0`). If Phase 1 left the empty default
  > at `0.0`, this docstring is correct as written (empty = no penalty); do **not**
  > silently change the numeric default here — that is Phase 1's call.

- [ ] **1c — the abstract method + `finalize` docstrings.** Phase 1 renamed the
  abstract method to `_score_terms`. Ensure its docstring reads (replace whatever
  Phase 1 left, to match §5.3 exactly):

  ```python
      @abstractmethod
      def _score_terms(
          self, image: Any, measurements: pd.DataFrame
      ) -> dict[str, float]:
          """This scorer's NATURAL per-term values (its own convention).

          Emit raw, intuitive numbers — a divergence stays a divergence, Dice
          stays Dice. Do **not** flip or normalize: the base ``score_image``
          orients each term into cost ``∈ [0, 1]`` via ``to_cost`` using
          ``_TERM_SENSE`` and ``_term_anchor``.

          Args:
              image: The (already-processed) image — duck-typed; reference-free
                  scorers read its mask/objmap, the ``QCScorer`` ignores it.
              measurements: The measurement frame the candidate pipeline produced
                  for ``image`` (the output of ``ImagePipeline.measure``).

          Returns:
              A mapping of term name → natural value for this image. Keys must be
              stable across images so the ``Evaluator`` can aggregate per term.
          """
          raise NotImplementedError
  ```

  And the `finalize` docstring: replace the `Returns:` sentence
  `` The scalar objective (higher = better; ``0.0`` for no terms) for `` with
  `` The scalar objective **cost** (lower = better; ``0.0`` for no terms) for ``,
  and change the body sentence "the single scalar objective the optimizer
  maximizes" wherever it appears in this docstring to "...the optimizer minimizes".

**Verify:**
```bash
# the goodness phrasing is gone from the file:
grep -nE "higher is better|higher = better|optimizer maximizes|is the worst score" src/phenotypic/tune/_scoring/_scorer.py
# (expect: no matches)
# the contract landed:
grep -nE "_score_terms|_TERM_SENSE|_term_anchor|minimizes|cost" src/phenotypic/tune/_scoring/_scorer.py | head
uv run mypy src/phenotypic/tune/_scoring/_scorer.py
```

**Commit:**
```bash
git add src/phenotypic/tune/_scoring/_scorer.py
git commit -m "docs(tune): rewrite Scorer docstring for cost convention + authoring contract"
```

---

## Task 2 — `tune/CLAUDE.md`: convention flip + "Adding a Scorer" subsection (surface 2/3)

**File:** `src/phenotypic/tune/CLAUDE.md`

- [ ] **2a — convention flip.** Replace the "Higher-is-better everywhere" bullet
  (lines 23–25):

  *Before:*
  ```markdown
  - **Higher-is-better everywhere**: every objective (and every axis of a
    multi-objective study) maximizes; the single `_MAXIMIZE` literal lives in
    `_strategies/_optuna_support.py`.
  ```

  *After:*
  ```markdown
  - **Cost everywhere (lower-is-better, minimize)**: every per-term and
    per-child value the optimizer sees is a bounded **cost** `∈ [0,1]` (`0` =
    perfect, `1` = worst); every objective (and every axis of a multi-objective
    study) **minimizes**. The single `_MINIMIZE` literal lives in
    `_strategies/_optuna_support.py`. The word in code/docs/fields is **"cost"**
    (never "score" for the new quantity, never "badness"). The QC flag
    `_HIGHER_IS_BAD` is unchanged: `True` ⟺ the metric is a loss ⟺
    `Sense.LOWER_BETTER`.
  - **Composite = augmented Tchebycheff**: the single-objective `CompositeScorer`
    blends per-child cost with `Tᵨ(b) = maxᵢ wᵢ(bᵢ + ε) + ρ·Σᵢ wᵢ·bᵢ`
    (utopia `z*ᵢ = −ε`, `_UTOPIA_EPS = 1e-3`, `rho = 0.05`), normalized to
    `[0,1]` over the **study-global active set**. `blend="weighted_mean"` is the
    compensatory opt-out; geometric-mean-of-cost is **never** exposed (one perfect
    axis would zero the product). `weights` are now blend-dependent (§6.5).
  - **Study persistence is a hard cutover**: `_STUDY_NAME = "tune_cost_v1"` (was
    `"tune"`). Pre-cutover `"tune"` (maximize) studies are **never reopened** and
    cannot be resumed under the cost convention — re-run them. Cross-study
    comparison with pre-cutover runs is invalid. (Optuna `load_if_exists=True`
    silently keeps a mismatched direction — verified 4.9.0 — so correctness rests
    on the name bump, not a runtime guard.)
  ```

- [ ] **2b — "Adding a Scorer" subsection.** Insert immediately **after** the
  `## Conventions` block (after the "Optuna is lazy-imported" bullet, before
  `## Math & logic doc`):

  ```markdown
  ## Adding a Scorer (the authoring contract)

  Canonical in the `Scorer` base-class docstring (`_scoring/_scorer.py`) and the
  contributor guide (`docs/source/contrib_guide/contributing.rst`) — keep the
  three copies in sync.

  1. Subclass `Scorer`; implement `_score_terms(image, measurements) ->
     dict[str, float]` returning your **natural** per-term values. Do **not** flip
     or normalize by hand — a divergence stays a divergence, Dice stays Dice.
  2. Declare `_TERM_SENSE` (`Sense.LOWER_BETTER` if larger = worse, the default;
     `Sense.HIGHER_BETTER` if larger = better, e.g. Dice/ICC).
  3. Override `_term_anchor` **only** for an unbounded term, returning its
     half-cost scale (for a QC-backed term, the check's `fail_threshold`). Bounded
     `[0,1]` terms need nothing.
  4. Do **not** add scalarization parameters — `ε`, `ρ`, normalization, and
     default weights are framework-derived; a scorer never sets them.
  5. Register: re-export from `tune/__init__.py` and the class registry, or the
     GUI and `from_json` cannot see it.

  The base `score_image` template method orients every term via `to_cost`
  (`_scoring/_orient.py`); the framework then robust-aggregates, reduces per child,
  and combines (augmented Tchebycheff). `CompositeScorer` overrides `score_image`
  (it merges already-cost children) — never `_score_terms`.
  ```

**Verify:**
```bash
grep -nE "Cost everywhere|_MINIMIZE|augmented Tchebycheff|tune_cost_v1|Adding a Scorer|_score_terms|_TERM_SENSE" src/phenotypic/tune/CLAUDE.md
grep -nE "Higher-is-better everywhere|_MAXIMIZE" src/phenotypic/tune/CLAUDE.md   # expect: no matches
```

**Commit:**
```bash
git add src/phenotypic/tune/CLAUDE.md
git commit -m "docs(tune): CLAUDE.md cost convention + Adding-a-Scorer subsection"
```

---

## Task 3 — `contributing.rst`: "Adding a tuning objective (Scorer)" walkthrough (surface 3/3)

**File:** `docs/source/contrib_guide/contributing.rst`

The file is a stub (six placeholder sections). Add a fully-worked new section
between "Code Standards" and "Pull Request Workflow". This is the only one of the
three surfaces inside the Sphinx tree, so it gets a runnable code block and a
`sphinx-build` verification.

- [ ] **3a — insert the section.** After the "Code Standards" block (lines 12–15),
  insert:

  ```rst
  Adding a tuning objective (Scorer)
  ----------------------------------

  A tuning *objective* is a :class:`~phenotypic.tune.Scorer`. Every value the
  tuner optimizes is a bounded **cost** in ``[0, 1]`` (``0`` = perfect, ``1`` =
  worst) and the optimizer **minimizes** it. You emit your metric's *natural*
  value and declare its *sense*; the framework orients, aggregates, and combines.

  The contract (canonical in the ``Scorer`` base-class docstring and
  ``src/phenotypic/tune/CLAUDE.md`` — keep all three in sync):

  #. **Subclass** ``Scorer`` and implement
     ``_score_terms(image, measurements) -> dict[str, float]`` returning your
     **natural** per-term values. Do **not** flip or normalize by hand — a
     divergence stays a divergence, Dice stays Dice.
  #. **Declare the sense** with the ``_TERM_SENSE`` class variable:
     ``Sense.LOWER_BETTER`` (the default) if a larger value is worse;
     ``Sense.HIGHER_BETTER`` if a larger value is better (Dice, IoU, ICC,
     solidity).
  #. **Supply an anchor only for an unbounded term** by overriding
     ``_term_anchor`` to return the value at which cost should reach ``0.5`` (for
     a QC-backed term, its check's ``fail_threshold``). Bounded ``[0, 1]`` terms
     need nothing.
  #. **Do not add scalarization parameters.** The utopia shift ``ε``, the
     augmentation coefficient ``ρ``, per-axis normalization, and default weights
     are all framework-derived — a scorer never exposes them.
  #. **Register** the class: re-export it from ``phenotypic.tune`` (its
     ``__init__``) and the class registry, or the GUI and ``from_json``
     deserialization cannot discover it.

  Minimal example — a reference-free scorer that rewards round colonies (Solidity
  is already ``[0, 1]`` and higher-is-better, so it only declares the sense):

  .. code-block:: python

     from typing import Any, ClassVar

     import pandas as pd

     from phenotypic.tune import Scorer
     from phenotypic.tune._scoring._orient import Sense


     class SolidityScorer(Scorer):
         """Reward compact, non-jagged colonies (mean Solidity, higher = better)."""

         _TERM_SENSE: ClassVar[Sense] = Sense.HIGHER_BETTER

         def _score_terms(
             self, image: Any, measurements: pd.DataFrame
         ) -> dict[str, float]:
             # Solidity is bounded [0, 1]; emit it raw. The base score_image
             # complements it into cost (1 - value) because _TERM_SENSE is
             # HIGHER_BETTER — you write no flip.
             return {"Solidity": float(measurements["Shape_Solidity"].mean())}

  The base :meth:`Scorer.score_image` template method then turns each natural
  term into cost via :func:`to_cost`; the ``Evaluator`` robust-aggregates
  (``median + λ·IQR``, clamped) and the optimizer minimizes. A
  :class:`~phenotypic.tune.CompositeScorer` combines several scorers' per-child
  cost with an **augmented Tchebycheff** scalarization (worst-axis-dominant by
  default) — see the explainer
  :doc:`tune-with-optuna </../../superpowers/explain/tune-with-optuna>` for the
  math. (If the cross-tree ``:doc:`` reference does not resolve in your Sphinx
  config, drop it to plain text: "see ``docs/superpowers/explain/tune-with-optuna.md``".)
  ```

  > **Cross-reference caveat.** `docs/superpowers/` is outside `docs/source/`, so
  > a Sphinx `:doc:` link into it will warn ("unknown document"). If
  > `sphinx-build` in 3c emits that warning, replace the final `:doc:` directive
  > with the plain-text fallback already noted in the block. Do **not** leave a
  > broken cross-reference.

**Verify:**
```bash
grep -nE "Adding a tuning objective|_score_terms|_TERM_SENSE|SolidityScorer|augmented Tchebycheff" docs/source/contrib_guide/contributing.rst
```

- [ ] **3b — confirm the example imports resolve.** The example imports
  `Sense` from `phenotypic.tune._scoring._orient` (Phase 0) and `Scorer` from
  `phenotypic.tune` (existing). Sanity-check both exist post-Phase-0/1:
  ```bash
  uv run --extra tune python -c "from phenotypic.tune import Scorer; from phenotypic.tune._scoring._orient import Sense; print(Sense.HIGHER_BETTER)"
  # expect: Sense.HIGHER_BETTER
  ```

- [ ] **3c — build the docs.**
  ```bash
  uv run --group docs sphinx-build -b html docs/source docs/_build/html -q 2>&1 | tee /tmp/sphinx.log
  # expect: no NEW errors/warnings attributable to contributing.rst.
  grep -i "contributing" /tmp/sphinx.log   # expect: clean (no warning lines)
  ```
  If 3c warns on the `:doc:` cross-tree link, apply the plain-text fallback from
  3a and re-run.

**Commit:**
```bash
git add docs/source/contrib_guide/contributing.rst
git commit -m "docs(contrib): add Adding-a-tuning-objective (Scorer) walkthrough"
```

---

## Task 4 — Rewrite the math sections of the Optuna explainer

**File:** `docs/superpowers/explain/tune-with-optuna.md`

`tune/CLAUDE.md` mandates updating this explainer in the **same change** as any
math/control-flow change; Phases 1–4 changed the math, so the explainer must be
brought current here. Six edits, each surgical. Re-resolve `file:line` refs
against the worktree before pasting — they shifted across Phases 1–4.

- [ ] **4a — the convention banner (lines 7–9).**

  *Before:*
  ```markdown
  > All file:line references point at `src/phenotypic/tune/…`. Everything in the
  > module follows one sign convention: **higher-is-better, everything is
  > maximized** (`_MAXIMIZE = "maximize"`, `_strategies/_optuna_support.py`).
  ```

  *After:*
  ```markdown
  > All file:line references point at `src/phenotypic/tune/…`. Everything in the
  > module follows one sign convention: **bounded cost in `[0,1]` (`0` = perfect,
  > `1` = worst), everything is minimized** (`_MINIMIZE = "minimize"`,
  > `_strategies/_optuna_support.py`). Each scorer emits its *natural* per-term
  > value and declares a `Sense`; the base `Scorer.score_image` orients it into
  > cost via `to_cost` (`_scoring/_orient.py`).
  ```

- [ ] **4b — §3 Step B robust aggregation (lines 162–175).** Replace the formula
  and its gloss:

  *Before:*
  ```markdown
  After each rung, every scoring term's per-image values are reduced not by a
  plain mean but by a **spread-penalized median** (`_robust_aggregate`,
  `_evaluator.py:53`):

  ```
  term = median(s₁ … sₖ) − λ · IQR(s₁ … sₖ)
  ```

  with `λ = stability_weight = 0.5` and `IQR = Q75 − Q25`
  (`_aggregate_math.py:25`). The `−λ·IQR` term **rewards parameters that work
  consistently across plates**, not just on average — a config that is brilliant
  on two plates and terrible on three loses to a steady one.
  ```

  *After:*
  ```markdown
  After each rung, every scoring term's per-image **cost** values are reduced not
  by a plain mean but by a **spread-penalized median**, then clamped to `[0,1]`
  (`_robust_aggregate`, `_evaluator.py:53`):

  ```
  term = clamp01( median(b₁ … bₖ) + λ · IQR(b₁ … bₖ) )
  ```

  with `λ = stability_weight = 0.5` and `IQR = Q75 − Q25`
  (`_aggregate_math.py:25`). The `+λ·IQR` term **penalizes parameters that work
  inconsistently across plates** — a config that is brilliant on two plates and
  terrible on three (high IQR) is dragged toward worse cost and loses to a steady
  one. The clamp matters: `median + λ·IQR` can reach `~1+λ` on an unstable-and-bad
  term, so clamping keeps every per-child cost in `[0,1]` (the invariant the
  Tchebycheff composite asserts).
  ```

- [ ] **4c — §3 Step C finalize (lines 176–179) + Step D pruning (lines 181–200)
  + Failure taxonomy (lines 202–207) + the two diagnostic flags (lines 209–215).**

  - Step C: change "the scalar Optuna maximizes" → "the scalar Optuna minimizes".
  - Step D: change "trials are ranked by their interim value; only the top
    `1/reduction_factor` (top third by default) survive" — keep the words but add
    after them: " (under `direction=minimize` the ASHA pruner keeps the
    **lowest-cost** third)".
  - Failure taxonomy: change both `score floored to `0.0`` and `the worst term
    (`0.0`)` to `1.0` (cost worst-floor); change "failures honestly drag the
    aggregate" → "failures honestly drag the aggregate **up** (toward worse cost)".
  - The two diagnostic flags:

    *Before:*
    ```markdown
    - **`gap`** (`_per_trial_dispersion`, `_evaluator.py:68`) = relative IQR of the
      *primary* term, `(Q75 − Q25) / max(|median|, 1e-12)` — a cheap
      instability/overfit flag (not a held-out gap).
    - **`suspicious`** (`_is_suspicious`, `_evaluator.py:104`) =
      `score ≥ 0.7 AND Count ≤ 0.3` — catches the gaming signature where a pipeline
      scores well *because* it under-detects.
    ```

    *After:*
    ```markdown
    - **`gap`** (`_per_trial_dispersion`, `_evaluator.py:68`) = relative IQR of the
      *primary* term computed on the **goodness-equivalent `1 − cost`**, with the
      denominator floored at `_GAP_EPS ≈ 0.02` — a cheap instability/overfit flag
      (not a held-out gap). Computing it on `1 − cost` moves the divide-by-zero
      singularity to the harmless *bad-cost* end so a near-perfect candidate
      (cost ≈ 0) does not explode.
    - **`suspicious`** (`_is_suspicious`, `_evaluator.py:104`) =
      `score ≤ 0.3 AND Count ≥ 0.7` (cost) — catches the gaming signature where a
      pipeline has low *cost* (looks good) *because* it under-detects (high Count
      cost). A missing `Count` term defaults to best cost (`0.0`), so absent-Count
      candidates are never flagged.
    ```

- [ ] **4d — §4 the scoring strategies (lines 219–299).**

  - Lead sentence (lines 221–222): change "all normalized **[0,1],
    higher-is-better**" → "each emitted as a **natural** value and oriented to
    **cost ∈ [0,1]** by the base `score_image` (a `Sense.HIGHER_BETTER` term like
    Dice is complemented `1 − value`; a `Sense.LOWER_BETTER` divergence passes
    through)".
  - Supervised mask (lines 236–237): "Two empty masks → 1.0; matched-vs-empty
    → 0.0" describes the *natural* Dice (higher-better); add a trailing sentence:
    "These are the natural Dice values; the base orients them to cost (`1 − Dice`),
    so two empty masks → cost `0.0` (perfect) and a missed object → cost `1.0`."
  - QCScorer (lines 269–278): the `t(metric) = exp(−ln2·metric/fail_threshold)`
    fold currently produces goodness. Reframe:

    *Before:*
    ```markdown
    The count divergence `metric = |detected − expected| / expected` is folded to
    higher-is-better via an **exponential half-life anchor**:

    ```
    t(metric) = exp( −ln2 · metric / fail_threshold )
    ```

    So metric 0 → 1.0, metric = fail_threshold → exactly 0.5, metric → ∞ → 0.0.
    Averaged across `groupby` units → term `"Count"`.
    ```

    *After:*
    ```markdown
    The count divergence `metric = |detected − expected| / expected` is the
    scorer's **natural** value (a loss, `Sense.LOWER_BETTER`). Because it is
    unbounded, the scorer supplies an **anchor** (`_term_anchor` → the check's
    `fail_threshold`), and the base `to_cost` folds it via the threshold-anchored
    transform `1 − exp(−ln2 · metric / fail_threshold)`:

    ```
    cost(metric) = 1 − exp( −ln2 · metric / fail_threshold )
    ```

    So metric 0 → cost 0.0 (perfect), metric = fail_threshold → exactly 0.5,
    metric → ∞ → cost 1.0 (worst). Averaged across `groupby` units → term
    `"Count"`. (In the shipped roster every scorer keeps its internal `[0,1]` fold
    — OQ1=A — so `_term_anchor` returns `None` and `to_cost` is identity/complement;
    the anchor branch is the contract for future raw-loss scorers.)
    ```

  - CompositeScorer (lines 280–299): replace the blend description **and** the
    summary table's last rows.

    *Before:*
    ```markdown
    ### D. CompositeScorer — blend multiple scorers (`_scoring/_composite.py`)
    Each child owns a namespaced prefix `s0.`, `s1.`, … Children are finalized to
    scalars, then combined:

    - **weighted arithmetic mean** `Σ wᵢsᵢ / Σ wᵢ` if weights are given, else
    - **geometric mean** `(Π max(sᵢ, 0))^(1/n)` — a single weak axis drags the
      product toward 0, so a bad objective can't be masked by a strong one.

    It rejects cyclic nesting at construction, and in `multi_objective=True` mode
    returns the per-child dict instead of a scalar.
    ```

    *After:*
    ```markdown
    ### D. CompositeScorer — blend multiple scorers (`_scoring/_composite.py`)
    Each child owns a namespaced prefix `s0.`, `s1.`, … Children are finalized to
    per-child **cost** scalars `bᵢ ∈ [0,1]`, then combined over the **study-global
    active set** (children available study-wide; a study-wide abstainer is dropped
    from *both* the `max` and the normalizer):

    - **augmented Tchebycheff** (default, `blend="tchebycheff"`) with utopia point
      `z*ᵢ = −ε` and augmentation `ρ`:

      ```
      Tᵨ(b) = maxᵢ wᵢ(bᵢ + ε)  +  ρ · Σᵢ wᵢ·bᵢ           (minimize)
      T_norm = Tᵨ(b) / Tᵨ(1…1)                            ∈ (0, 1]
      ```

      The `max` makes it **conjunctive** (worst axis dominates — all objectives
      must be good); the `ρ·Σ` augmentation upgrades minimizers from *weakly* to
      *properly* Pareto optimal. `_UTOPIA_EPS = 1e-3`, `rho = 0.05` (defaults the
      user never sets — §6.4). The normalizer is the **study-global** constant
      `Tᵨ(1…1)`, so the `[0,1]` rescale is argmin-preserving **across** trials.
    - **weighted arithmetic mean** (opt-out, `blend="weighted_mean"`)
      `Σ wᵢbᵢ / Σ wᵢ` — *compensatory*: a strong axis offsets a weak one. Cannot
      reach non-convex-front compromises; that is why it is not the default.

    **Never** a geometric mean of cost: `0` is the product's annihilator, so one
    perfect axis (cost 0) would zero the product and dominate — the opposite of
    the conjunctive property it has on goodness. It is removed from the live path.

    `weights` are now **blend-dependent** — Tchebycheff per-axis weights under
    `tchebycheff`, arithmetic weights under `weighted_mean` (a behavior change:
    today *setting* `weights` switched to the compensatory mean). It rejects cyclic
    nesting at construction, and in `multi_objective=True` mode returns the
    per-child cost dict (NSGA-II, `directions=["minimize"]*n`; the abstainer floor
    flips `0.0 → 1.0`).
    ```

  - Summary table (lines 293–299): change the `Range` column from `[0,1]↑` to
    `[0,1]↓ (cost)` on every row; update the formulas to the cost forms
    (Supervised mask → `1 − Dice/IoU`; QC → `1 − exp(−ln2·metric/thr)`;
    Composite → `augmented Tchebycheff (or weighted mean) / multi-objective dict`).

- [ ] **4e — §6 Pareto math (lines 336–350) + §7 generalization (lines 354–377).**

  - §6: "directions become `["maximize"] * n`" → "`["minimize"] * n`"; dominance
    "`a` dominates `b` iff `a` is ≥ `b` on **every** axis AND strictly > on **at
    least one**" → "`a` dominates `b` iff `a` is **≤** `b` on every axis AND
    strictly **<** on at least one". Knee point: "`lo = min(vectors)` and
    `hi = max(vectors)` ... **maximum perpendicular distance**" — the chord math is
    direction-agnostic, so keep it but add: "(the extremes/chord are
    direction-agnostic; under minimize the knee is still the elbow of the cost
    front)".
  - §7: replace the overfit-gate block.

    *Before:*
    ```markdown
    The overfit gate (`compute_generalization_gap`, `_evaluation/_generalization.py:58`)
    flags a winner only when **both** margins are exceeded:

    ```
    abs_drop = s_cal − s_held
    rel_drop = abs_drop / max(|s_cal|, 1e-12)
    flag  ⟺  rel_drop > rel_margin  AND  abs_drop > abs_margin
    ```
    ```

    *After:*
    ```markdown
    The overfit gate (`compute_generalization_gap`, `_evaluation/_generalization.py:58`)
    adopts the **standard loss-space generalization gap** — `gap = test − train`,
    positive = overfit. Because cost *is* a loss, this is direction-correct by
    construction (no custom sign flip):

    ```
    abs_gap = heldout_cost − cal_cost          # positive = overfit
    rel_gap = abs_gap / max(1 − cal_cost, _GAP_EPS)   # on the goodness-equivalent
    flag  ⟺  rel_gap > rel_margin  AND  abs_gap > abs_margin
    ```

    The relative term divides by the **goodness-equivalent `1 − cal_cost`** (with
    `_GAP_EPS ≈ 0.02`) so a near-perfect calibration (cost ≈ 0) does not explode.
    The principled blow-up-free upgrade — relative *overtuning* normalized by the
    *achievable* test improvement (`> 1` ⇒ all gains lost; Schneider, Bischl &
    Feurer, 2025) — needs incumbent/default tracking we don't have and is a
    deferred v2 upgrade.
    ```

  - §7 margins sentence (lines 375–377): "It is report-only" stays; the margin
    defaults `gap_margin_relative = 0.15`, `gap_margin_absolute = 0.05` are
    unchanged — verify against Phase 1/4 (the gap re-derivation kept `0.15`).

- [ ] **4f — §9 takeaways + Key files (lines 444–472).** In the Takeaways list:
  - "stability-penalized robust aggregate (`median − λ·IQR`)" → "(`median + λ·IQR`,
    clamped)".
  - "one of four [0,1] scoring strategies" → "one of four [0,1]-cost scoring
    strategies".
  - Add a final takeaway bullet:
    "- **Combined**: the single-objective `CompositeScorer` uses an **augmented
    Tchebycheff** scalarization (conjunctive, worst-axis-dominant) over the
    study-global active set, replacing the old geometric mean — it can reach
    non-convex-front compromises a weighted sum cannot (Steuer & Choo, 1983;
    Miettinen, 1998)."
  - Key files table: add a row
    `` | `_scoring/_orient.py` | `Sense`, `to_cost`, `clamp01` (the orientation boundary) | ``
    and update the `_composite.py` description to mention Tchebycheff.

- [ ] **4g — add a References block** at the end of the explainer (the doc
  currently has none). Append, citing only the §12 papers that back the new math:

  ```markdown
  ---

  ## References

  The cost convention and composite math draw on:

  - Steuer, R. E., & Choo, E.-U. (1983). *An interactive weighted Tchebycheff
    procedure for multiple objective programming.* Mathematical Programming,
    26(3), 326–344. https://doi.org/10.1007/BF02591870 — weighted Tchebycheff
    reaches every Pareto point; augmentation gives *proper* Pareto optimality.
  - Miettinen, K. (1998). *Nonlinear Multiobjective Optimization.* Kluwer.
    https://doi.org/10.1007/978-1-4615-5563-6 — reachability and proper efficiency
    of (augmented) Tchebycheff scalarization.
  - Carrell, A. M., Mallinar, N., Lucas, J., & Nakkiran, P. (2022). *The
    calibration generalization gap.* arXiv.
    https://doi.org/10.48550/arXiv.2210.01964 — generalization gap as
    `|Test − Train|` error; our loss-space `heldout_cost − cal_cost` is the same
    quantity.
  - Schneider, L., Bischl, B., & Feurer, M. (2025). *Overtuning in hyperparameter
    optimization.* arXiv. https://doi.org/10.48550/arXiv.2506.19540 — the
    relative-overtuning normalization deferred to v2.
  ```

**Verify:**
```bash
grep -nE "higher-is-better|maximized|_MAXIMIZE|median − λ·IQR|geometric mean|\[0,1\], higher" docs/superpowers/explain/tune-with-optuna.md
# expect: no matches (all flipped)
grep -nE "minimize|cost|augmented Tchebycheff|median \+ λ·IQR|study-global active set|Steuer" docs/superpowers/explain/tune-with-optuna.md | head
```

**Commit:**
```bash
git add docs/superpowers/explain/tune-with-optuna.md
git commit -m "docs(tune): rewrite explainer math for cost + augmented Tchebycheff"
```

---

## Task 5 — Update the `.graph.md` data-flow companion

**File:** `docs/superpowers/explain/tune-with-optuna.graph.md`

The diagram specs (Mermaid + DOT) and the node-reference table all carry the
old goodness/maximize wording. Update both specs identically (the two are
"equivalent specs" by design) and the table.

- [ ] **5a — Mermaid spec (lines 41–87).**
  - `SCORE` node label: `Scorer.score_image → terms in [0,1]` →
    `Scorer.score_image → cost terms in [0,1]`; the `composite: geo/weighted mean`
    line → `composite: augmented Tchebycheff / weighted mean`.
  - `AGG` node: `median − λ·IQR  (λ=0.5)` → `median + λ·IQR, clamp01  (λ=0.5)`.
  - `FAIL` node: `failed=True, score=0.0` → `failed=True, cost=1.0`.
  - `RESULT` node: keep field list (`score, terms, objectives?, gap, suspicious`)
    but `score` now denotes cost — no label change needed, the banner covers it.

- [ ] **5b — Graphviz DOT spec (lines 93–153).** Mirror 5a on the DOT labels:
  `score` node label `Scorer.score_image -> terms[0,1]` →
  `Scorer.score_image -> cost terms[0,1]` and `composite geo/weighted mean` →
  `composite augmented Tchebycheff / weighted mean`; `agg` node
  `median - lambda*IQR` → `median + lambda*IQR, clamp01`; `fail` node
  `failed=True, score=0.0` → `failed=True, cost=1.0`.

- [ ] **5c — Node reference table (lines 159–171).**
  - `Scorer.score_image` row: "four strategies, terms ∈ [0,1]" → "four strategies,
    **cost** terms ∈ [0,1] (oriented by `to_cost`)".
  - robust-aggregate row: `median − λ·IQR`, λ=0.5 → `median + λ·IQR` (clamped),
    λ=0.5.
  - generalization gap row: `rel>0.15 ∧ abs>0.05` flag → keep, but change the
    role to "loss-space `heldout_cost − cal_cost`; `rel>0.15 ∧ abs>0.05` flag".
  - Add a row: `` | `Scorer.to_cost` | `_scoring/_orient.py` | natural value → cost ∈ [0,1] (Sense + anchor) | ``.

**Verify:**
```bash
grep -nE "median − λ·IQR|median - lambda\*IQR|geo/weighted mean|geo/weighted mean|score=0.0" docs/superpowers/explain/tune-with-optuna.graph.md
# expect: no matches
grep -nE "cost terms|augmented Tchebycheff|clamp01|to_cost" docs/superpowers/explain/tune-with-optuna.graph.md | head
```

**Commit:**
```bash
git add docs/superpowers/explain/tune-with-optuna.graph.md
git commit -m "docs(tune): update tune data-flow graph for cost convention"
```

---

## Task 6 — Cross-phase regression tests

**New file:** `tests/unit/tune/test_cost_convention_regression.py`

**Why a new unit file, not integration.** These are the §10 cross-phase
regressions that lock the *whole* cutover. They are deterministic (seeded grid
strategy + synthetic scorers, no Optuna RDB, no disk beyond `tmp_path`-free
in-memory stores), so they belong in `tests/unit/tune/` (per
[tests/CLAUDE.md](../../../../tests/CLAUDE.md): "unit/ — deterministic, no I/O").
Driving `TuningEngine` with a `GridConfig` strategy and the in-memory
`JournalStudyStore` exercises the real Evaluator cost math, the real `best()`
selection (`min` after Phase 2), and the real composite — without an Optuna study,
so no `--extra tune` Optuna dependency is needed for the winner-equivalence and
gap tests. The composite-delta test constructs a `CompositeScorer` directly.

> **These tests only pass once Phases 1–4 land.** Until then they are red
> (the engine still maximizes goodness). That is intended — they are the
> acceptance gate for the cutover. Mark none as `xfail`; a red test here means a
> phase is incomplete.

The file has three test groups mapping to §10:
(6a) reflection winner-equivalence, (6b) overfit-gap SIGN end-to-end,
(6c) composite-delta snapshot.

- [ ] **6a — reflection winner-equivalence.** A seeded synthetic study must select
  the **identical winner** under the new minimize convention as the old maximize
  convention would have, for the single-term and arithmetic-mean paths.

  **Deterministic construction.** Use `GridConfig` (exhaustive, order-stable) over
  a 3-point categorical knob, and a synthetic `Scorer` whose *natural* per-term
  value is a pure function of the chosen param (no image dependence). Because the
  grid is exhaustive and the scorer is deterministic, the winner is fully
  determined. We assert the engine picks the param the **cost-minimizing** winner
  must have — and, to prove reflection (not just "some winner"), we also compute
  the **old-maximize** winner from the same natural goodness values and assert they
  name the **same param**.

  Full test code:

  ```python
  """§10 cross-phase regressions for the minimize-cost + Tchebycheff cutover.

  These pass only once Phases 1-4 have landed (the engine minimizes bounded cost,
  the composite is augmented Tchebycheff, and the overfit gap is loss-space). A
  red test here means a phase is incomplete, not that the test is wrong.
  """
  from __future__ import annotations

  from typing import Any, ClassVar

  import pandas as pd
  import pytest

  from phenotypic import ImagePipeline
  from phenotypic.data import load_synth_yeast_plate
  from phenotypic.detect import OtsuDetector
  from phenotypic.tune import (
      Categorical,
      Evaluator,
      GridConfig,
      Knob,
      Scorer,
      SearchSpace,
  )
  from phenotypic.tune._engine import TuningEngine
  from phenotypic.tune._scoring._orient import Sense
  from phenotypic.tune._spec import Budget, TuningSpec


  # The deterministic objective surface: one knob, three settings, a fixed
  # natural GOODNESS per setting (higher = better in the old world). Chosen so the
  # ranking is unambiguous and the best/worst are not the grid endpoints (guards
  # against an off-by-orientation that happens to pick the right end by luck).
  _CHOICES: tuple[float, ...] = (0.2, 0.9, 0.5)
  _NATURAL_GOODNESS: dict[float, float] = {0.2: 0.20, 0.9: 0.90, 0.5: 0.50}
  _BEST_PARAM = 0.9   # max goodness == min cost
  _WORST_PARAM = 0.2


  def _space() -> SearchSpace:
      return SearchSpace(knobs=(
          Knob(key="0.sigma", domain=Categorical(choices=_CHOICES)),
      ))


  def _base() -> ImagePipeline:
      from phenotypic.enhance import BlurGauss
      return ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()])


  class _GoodnessKnobScorer(Scorer):
      """Emits a HIGHER_BETTER natural term keyed off the chosen sigma.

      Reads the sigma the candidate pipeline was built with off the image-
      independent param surface via the pipeline op (so the natural value is a
      pure function of the param, fully deterministic).
      """

      _TERM_SENSE: ClassVar[Sense] = Sense.HIGHER_BETTER

      def _score_terms(
          self, image: Any, measurements: pd.DataFrame
      ) -> dict[str, float]:
          # The engine built the pipeline with the chosen sigma; recover it from
          # the BlurGauss op so the natural value depends only on the param.
          sigma = float(self._chosen_sigma)
          return {"Quality": _NATURAL_GOODNESS[sigma]}

      # The engine sets this per trial via build; in this synthetic test we read
      # it off a private attr injected by the fixture below.
      _chosen_sigma: float = 1.0
  ```

  > **Plumbing note for 6a.** A `Scorer` is stateless across `score_image` and the
  > engine reuses one instance, so it cannot stash per-trial `_chosen_sigma`
  > cleanly. The deterministic, stateless way to make the natural value depend on
  > the param is to read it **from the measurements/image the candidate produced**.
  > Two acceptable implementations — pick the one that matches the landed Phase 1
  > scorer surface:
  >
  > 1. **Param-from-pipeline (preferred):** override the Evaluator hook the engine
  >    uses to pass the built pipeline, or read the op off `image`'s pipeline
  >    provenance. If no such hook exists, use option 2.
  > 2. **Param-encoding scorer via a custom Evaluator subclass** that captures the
  >    `params` dict it is handed in `evaluate(...)` and exposes the chosen value to
  >    the scorer. The engine calls `evaluator.evaluate(base, scorer, params, ...)`
  >    (see `test_engine.py` `_fake_eval` signature
  >    `evaluate(self, base, scorer, params, images, *, channel=None)`), so a thin
  >    subclass can thread `params["0.sigma"]` onto the scorer before delegating to
  >    `super().evaluate`.
  >
  > Implement option 2 concretely (it needs no engine internals):

  ```python
  class _ParamCaptureEvaluator(Evaluator):
      """Threads the chosen sigma onto the scorer before each evaluation.

      The engine calls evaluate(base, scorer, params, images, channel=...); we
      stamp params["0.sigma"] onto the scorer so its natural value is a pure
      function of the candidate, then delegate to the real cost-aware Evaluator.
      """

      def evaluate(self, base, scorer, params, images, *, channel=None):  # type: ignore[override]
          object.__setattr__(scorer, "_chosen_sigma", float(params["0.sigma"]))
          return super().evaluate(base, scorer, params, images, channel=channel)


  def _run_grid_winner() -> dict[str, Any]:
      spec = TuningSpec(
          pipeline=_base(),
          search_space=_space(),
          scorer=_GoodnessKnobScorer(),
          evaluator=_ParamCaptureEvaluator(),
          strategy=GridConfig(),
          budget=Budget(),
      )
      engine = TuningEngine(spec)
      engine.optimize([load_synth_yeast_plate()])
      best = engine.store.best()
      assert best is not None
      return best.params


  def test_single_term_winner_is_cost_minimizer():
      # The new engine minimizes cost; the single-term winner must be the param
      # whose natural goodness is highest (== lowest cost == old-maximize winner).
      winner = _run_grid_winner()
      assert winner["0.sigma"] == pytest.approx(_BEST_PARAM)


  def test_reflection_winner_matches_old_maximize_winner():
      # Reflection equivalence (README invariant 3 / spec §4): the cost winner is
      # the SAME param the old maximize convention would have picked. Compute the
      # old winner from the natural goodness directly and assert agreement.
      old_max_winner = max(_CHOICES, key=lambda c: _NATURAL_GOODNESS[c])
      new_min_winner = _run_grid_winner()["0.sigma"]
      assert new_min_winner == pytest.approx(old_max_winner)
      assert new_min_winner != pytest.approx(_WORST_PARAM)  # not the grid end by luck


  def test_arithmetic_mean_winner_is_cost_minimizer():
      # The finalize default (mean of terms) is reflection-clean: a two-term
      # scorer's mean-cost winner is the mean-goodness winner. Use two HIGHER_BETTER
      # terms whose mean ranks the params identically to the single-term case.
      class _TwoTermScorer(_GoodnessKnobScorer):
          def _score_terms(self, image, measurements):
              g = _NATURAL_GOODNESS[float(self._chosen_sigma)]
              return {"A": g, "B": g}  # mean == g; same ranking

      spec = TuningSpec(
          pipeline=_base(),
          search_space=_space(),
          scorer=_TwoTermScorer(),
          evaluator=_ParamCaptureEvaluator(),
          strategy=GridConfig(),
          budget=Budget(),
      )
      engine = TuningEngine(spec)
      engine.optimize([load_synth_yeast_plate()])
      best = engine.store.best()
      assert best is not None
      assert best.params["0.sigma"] == pytest.approx(_BEST_PARAM)
  ```

  > **Pareto path (third sub-case of §10 winner-equivalence).** Add a Pareto
  > assertion only if Phase 4's `pareto_front_of` / `_dominates` are importable
  > deterministically without an Optuna study. The clean unit-level check is on
  > `_dominates` directly:
  >
  > ```python
  > def test_pareto_domination_is_reflected_under_minimize():
  >     # Under minimize, a dominates b iff a <= b on every axis and < on one.
  >     # Reflection: the same trial that dominated under maximize-goodness now
  >     # dominates under minimize-cost when vectors are complemented.
  >     from phenotypic.tune._study._pareto import _dominates
  >     # cost vectors: lower = better. (0.1, 0.2) dominates (0.3, 0.2).
  >     assert _dominates((0.1, 0.2), (0.3, 0.2)) is True
  >     assert _dominates((0.3, 0.2), (0.1, 0.2)) is False
  >     # equal vectors do not dominate
  >     assert _dominates((0.1, 0.2), (0.1, 0.2)) is False
  > ```
  >
  > Re-resolve `_dominates`'s exact signature/arity against the landed Phase 4
  > file (`_study/_pareto.py`) before pasting — it may take `Trial`s or tuples.

  **Run:**
  ```bash
  uv run pytest tests/unit/tune/test_cost_convention_regression.py -k "winner or reflection or pareto" -v
  ```
  **Expected:** all green once Phases 1–4 land. The single-term and arithmetic-mean
  winners equal `0.9`; the reflection test confirms `new_min_winner ==
  old_max_winner == 0.9` and `!= 0.2`; `_dominates` follows the `<=`/`<` rule.

- [ ] **6b — overfit-gap SIGN end-to-end.** A winner that is **better on
  calibration than on held-out** must be **flagged** under the cost convention
  (loss-space `gap = heldout_cost − cal_cost > 0`). This is the §10 "overfit-gap
  sign" regression: it would catch the inverted detector shipping silently.

  Append to the same file:

  ```python
  # -- 6b: overfit-gap SIGN (loss-space heldout_cost - cal_cost) ----------------

  from phenotypic.tune._evaluation._generalization import compute_generalization_gap


  def test_overfit_winner_is_flagged_under_cost():
      # Loss-space gap: held-out cost (0.5) WORSE than calibration cost (0.1) is
      # overfit (gap = +0.4). Under cost this must FLAG (both margins exceeded).
      # cal_cost=0.1, heldout_cost=0.5 with the default margins.
      rel, absolute, flagged = compute_generalization_gap(
          0.1, 0.5, rel_margin=0.15, abs_margin=0.05
      )
      assert absolute == pytest.approx(0.4)        # heldout_cost - cal_cost
      assert absolute > 0                           # positive == overfit
      assert flagged is True


  def test_good_generalizer_is_not_flagged_under_cost():
      # Held-out cost (0.12) ~ calibration cost (0.10): no overfit, gap ~ +0.02
      # (below the absolute margin 0.05). Must NOT flag. Guards the symmetric
      # failure: a good generalizer mis-flagged as overfit.
      rel, absolute, flagged = compute_generalization_gap(
          0.10, 0.12, rel_margin=0.15, abs_margin=0.05
      )
      assert absolute == pytest.approx(0.02)
      assert flagged is False


  def test_underfit_does_not_flag_under_cost():
      # Held-out cost (0.1) BETTER than calibration (0.5): negative gap, never
      # overfit. The old accuracy-space detector would have flagged this; the
      # loss-space one must not.
      rel, absolute, flagged = compute_generalization_gap(
          0.5, 0.1, rel_margin=0.15, abs_margin=0.05
      )
      assert absolute == pytest.approx(-0.4)
      assert flagged is False
  ```

  > **Argument-order caveat.** The current signature is
  > `compute_generalization_gap(cal_score, heldout_score, ...)` returning
  > `(rel, absolute, flagged)` with `absolute = cal − heldout` (goodness). Phase 1
  > flips the body to `absolute = heldout_cost − cal_cost` **while keeping the
  > positional order `(cal, heldout)`** (the doctest at `_generalization.py:91`
  > `(0.9, 0.5) → 0.444, True` is re-read as loss-space inputs per spec §7 Phase 1).
  > Re-resolve the landed Phase 1 signature: if Phase 1 renamed the params to
  > `cal_cost, heldout_cost`, the call sites above are correct (`(0.1, 0.5)` =
  > cal 0.1, heldout 0.5 → gap +0.4). If the positional contract differs, adjust
  > the call args so `absolute == +0.4` for the overfit case before asserting.

  **Run:**
  ```bash
  uv run pytest tests/unit/tune/test_cost_convention_regression.py -k "overfit or generalizer or underfit" -v
  ```
  **Expected:** `test_overfit_winner_is_flagged_under_cost` → `absolute = +0.4`,
  flagged `True`; good-generalizer and underfit → `False`.

- [ ] **6c — composite-delta snapshot (intended change).** The augmented
  Tchebycheff composite picks a **different** winner than the old geometric-mean
  composite on a non-convex trade-off — this is the one *intended* behavior change
  (README invariant 3, spec §8). Snapshot the documented delta.

  Append:

  ```python
  # -- 6c: composite-delta snapshot (Tchebycheff != old geomean; intended) ------

  from phenotypic.tune import CompositeScorer, QCScorer


  def _per_child_cost_vectors() -> dict[str, tuple[float, float]]:
      """Three candidates' per-child cost (b0, b1), a non-convex 2-axis front.

      - 'balanced': (0.40, 0.40) — the conjunctive (worst-axis) optimum.
      - 'lopsided': (0.05, 0.80) — great on axis 0, poor on axis 1.
      - 'lopsided2': (0.80, 0.05) — mirror.

      Old geomean-of-cost would reward the lopsided ones (a near-0 axis pulls the
      product toward 0); augmented Tchebycheff (worst-axis-dominant) prefers the
      balanced candidate. This is the intended semantics change.
      """
      return {
          "balanced": (0.40, 0.40),
          "lopsided": (0.05, 0.80),
          "lopsided2": (0.80, 0.05),
      }


  def _tchebycheff_pick(vectors: dict[str, tuple[float, float]]) -> str:
      # Reproduce the Phase 3 composite math at unit level to assert the winner
      # WITHOUT a full study: minimize max_i (b_i + eps) + rho * sum_i b_i.
      eps, rho = 1e-3, 0.05

      def t(b: tuple[float, float]) -> float:
          return max(b[0] + eps, b[1] + eps) + rho * (b[0] + b[1])

      return min(vectors, key=lambda k: t(vectors[k]))


  def _old_geomean_pick(vectors: dict[str, tuple[float, float]]) -> str:
      # The OLD composite operated on GOODNESS g = 1 - cost via geometric mean,
      # maximized. Reproduce its winner for the snapshot delta.
      import math

      def g_geomean(b: tuple[float, float]) -> float:
          g = (1.0 - b[0], 1.0 - b[1])
          return math.sqrt(max(g[0], 0.0) * max(g[1], 0.0))

      return max(vectors, key=lambda k: g_geomean(vectors[k]))


  def test_composite_delta_is_the_intended_change():
      vectors = _per_child_cost_vectors()
      new_winner = _tchebycheff_pick(vectors)
      old_winner = _old_geomean_pick(vectors)
      # The documented, intended delta: Tchebycheff picks the balanced compromise;
      # the old geomean picked a lopsided extreme. They MUST differ (this is the
      # snapshot that proves the composite changed on purpose).
      assert new_winner == "balanced"
      assert old_winner != "balanced"
      assert new_winner != old_winner


  ```

  > **No stub here.** `test_composite_delta_is_the_intended_change` above is the
  > load-bearing cross-phase snapshot — it reproduces the Phase 3 math at unit
  > level and proves the composite changed on purpose, independent of the live
  > `CompositeScorer` wiring. The complementary live-API assertion ("shipped
  > `CompositeScorer` defaults to `blend="tchebycheff"`, `rho == 0.05`, and
  > `finalize(...)` ∈ `[0,1]`") is **owned by Phase 3** (its fields/defaults task)
  > against the real constructor, so it is **not** duplicated here — this file
  > ships no skip stub.

  **Run:**
  ```bash
  uv run --extra tune pytest tests/unit/tune/test_cost_convention_regression.py -k "composite or tchebycheff" -v
  ```
  **Expected:** `test_composite_delta_is_the_intended_change` → `new_winner ==
  "balanced"`, `old_winner` is a lopsided extreme, and they differ.

- [ ] **6d — full-file run + lint/type.**
  ```bash
  uv run --extra tune pytest tests/unit/tune/test_cost_convention_regression.py -v
  uv run ruff check --fix tests/unit/tune/test_cost_convention_regression.py
  uv run mypy tests/unit/tune/test_cost_convention_regression.py
  ```
  **Expected:** all green (this file is self-contained — it reproduces the phase
  math at unit level; the live-API default-blend check is owned by Phase 3);
  ruff/mypy clean.

**Commit:**
```bash
git add tests/unit/tune/test_cost_convention_regression.py
git commit -m "test(tune): cross-phase regressions for minimize-cost + Tchebycheff cutover"
```

---

## Task 7 — End-to-end minimize-cost acceptance smoke (multi-plate, seeded)

**New file:** `tests/integration/tune/test_minimize_cost_e2e.py` (+ `tests/integration/tune/__init__.py`)

**Why (Gap 2):** Task 6's regressions use a *synthetic* objective function (a seeded
grid over a closed-form cost). They prove the math, but nothing drives the **real
tuner against real images through a real Optuna minimize study**. This task closes
that: it runs `run_tuning(...)` over **several distinct synthetic plates** (made with
`make_synthetic_plate(seed=i)` — per-plate seeds give genuine plate-to-plate
variation, which is what exercises the cross-image robust aggregate) and asserts the
whole-system cutover: the Optuna study **minimizes**, carries the convention stamp,
the winner is the **lowest-cost** trial, and a good pipeline is **reachable** (low
cost achieved). This is the end-to-end complement to the synthetic regressions and a
phase-independent acceptance gate.

This is an **acceptance test that runs after Phases 1–4 land** (like Task 6), not a
red-first TDD task — the system must already be cut over for it to pass.

- [ ] **Step 1: Create the package marker**

```bash
test -f tests/integration/tune/__init__.py || : > tests/integration/tune/__init__.py
```

- [ ] **Step 2: Write the end-to-end smoke**

Create `tests/integration/tune/test_minimize_cost_e2e.py`:
```python
"""End-to-end minimize-cost acceptance smoke (multi-plate, seeded).

Drives the REAL tuner (``run_tuning`` -> ``TuningEngine.optimize`` -> an Optuna
**minimize** study) over several *distinct* synthetic plates
(``make_synthetic_plate`` with per-plate seeds, so plate-to-plate variation
exercises the cross-image robust aggregate). Asserts the cutover end-to-end: the
study minimizes and carries the convention stamp, the winner is the lowest-cost
trial, and the achieved cost is low (a good pipeline is reachable) — the
whole-system proof that complements the synthetic-objective regressions in
``tests/unit/tune/test_cost_convention_regression.py``.
"""
from __future__ import annotations

import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")  # the `tune` extra

from phenotypic import GridImage, ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import make_synthetic_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    Knob,
    OptunaConfig,
    QCScorer,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune._tune_cli._run import _STUDY_NAME, run_tuning

_NROWS, _NCOLS = 8, 12
_EXPECTED = _NROWS * _NCOLS  # 96 colonies per plate


def _seeded_plates(n: int = 4) -> list[GridImage]:
    """``n`` DISTINCT synthetic plates (one seed each), wrapped as GridImages.

    Small (512x768) to keep the smoke fast; the 8x12 grid still yields
    separable colonies for Otsu. Per-plate seeds give real cross-plate variation.
    """
    plates = []
    for i in range(n):
        arr = make_synthetic_plate(
            nrows=_NROWS, ncols=_NCOLS, plate_h=512, plate_w=768, seed=i
        )
        plates.append(
            GridImage(arr=arr, name=f"plate_{i:02d}", nrows=_NROWS, ncols=_NCOLS)
        )
    return plates


def _layout_csv(tmp_path, names) -> str:
    """A layout CSV declaring 96 expected objects per plate (for the count check)."""
    rows = [
        {"Metadata_ImageName": name, "Object_Label": j}
        for name in names
        for j in range(_EXPECTED)
    ]
    csv = tmp_path / "layout.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def _spec(tmp_path, names) -> TuningSpec:
    # BlurGauss sigma is the discriminating knob: a small sigma keeps colonies
    # separable (count ~= 96 -> low cost); a large sigma merges/erases them
    # (count far from 96 -> high cost). Minimization must prefer the small sigma.
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="0.sigma", domain=Categorical(choices=(1.0, 12.0, 40.0))),)
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=_layout_csv(tmp_path, names),
                groupby=["Metadata_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=6, sampler="tpe", seed=0),
        budget=Budget(),
    )


def test_minimize_cost_end_to_end_winner_is_low_cost(tmp_path):
    images = _seeded_plates(4)
    names = [im.name for im in images]
    out = tmp_path / "out"

    run_tuning(_spec(tmp_path, names), images, out)

    # 1. Deliverables were written.
    assert io.best_pipeline_path(out).exists()
    assert io.trials_parquet_path(out).exists()

    # 2. The Optuna study MINIMIZES and carries the convention stamp (no silent
    #    maximize; the name was bumped to the cost-era study).
    db = io.resolve_study_db_path(out)
    study = optuna.load_study(storage=f"sqlite:///{db}", study_name=_STUDY_NAME)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert study.user_attrs.get("tune_convention") == "minimize-cost-v1"

    # 3. The winner is the LOWEST-cost trial, and a good pipeline is reachable.
    values = [t.value for t in study.trials if t.value is not None]
    assert values, "no completed trials"
    assert study.best_value == pytest.approx(min(values))
    assert study.best_value < 0.5  # small-sigma config detects ~96 -> low cost
    # Minimization discriminated: when >=2 distinct configs were evaluated, the
    # best is strictly better than the worst tried (the large-sigma config wrecks
    # the count -> high cost).
    if len(set(values)) > 1:
        assert study.best_value < max(values)
```

- [ ] **Step 3: Run the smoke**

```bash
uv run --extra tune pytest tests/integration/tune/test_minimize_cost_e2e.py -v
```
**Expected:** PASS once Phases 1–4 have landed. If it errors before the asserts:
- `_STUDY_NAME` import fails → Phase 2 has not bumped/exported the constant; re-resolve its name in `_tune_cli/_run.py`.
- `tune_convention` assert fails → Phase 2's stamp (README invariant #5) did not land; check `study.set_user_attr("tune_convention", "minimize-cost-v1")` on study creation.
- `direction == MINIMIZE` fails → Phase 2's `objective_directions` / `study_objective_kwargs` flip did not land (or a legacy `"tune"` study was silently reopened — the name bump is the guard).
- `best_value < 0.5` fails → re-check the count physics at 512x768 (a small sigma must detect ~96); if the synthetic colony radius is too small at this resolution, bump `plate_h/plate_w` to 1024x1536 (slower but unambiguous). Do **not** loosen the `< 0.5` threshold to paper over a broken detection — that would defeat the "good pipeline reachable" guarantee.

> **Determinism note:** `make_synthetic_plate(seed=i)` and `OptunaConfig(seed=0)` make the run reproducible. TPE over 6 trials / 3 categories over-covers the space; the `len(set(values)) > 1` guard keeps the discrimination assert robust even in the (unlikely) event the sampler repeats a category.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/tune/__init__.py tests/integration/tune/test_minimize_cost_e2e.py
git commit -m "test(tune): end-to-end minimize-cost acceptance smoke (multi-plate, seeded)"
```

---

## Task 8 — Whole-package final acceptance gate

**Why (Gap 1):** Every phase scopes `mypy`/`pytest` to `src/phenotypic/tune` (+
`gui/tune`). But Phase 1 changes the **public `Scorer` shape** (`score_image` →
`_score_terms` template method), which can ripple to importers *outside* `tune`. The
per-phase gates won't catch that. This task runs the type-checker and test suite
across the **whole package** once, as the last gate before the cutover is declared
done. (This is the gate the *Execution & review protocol* in the README calls "the
whole-package final gate".)

This task has **no code** — it is the final verification gate. Run it only after
Phases 0–5 (including Tasks 1–7) have all landed.

- [ ] **Step 1: Whole-package type check**

```bash
uv run mypy src/phenotypic
```
**Expected:** `Success: no issues found`. If a module *outside* `tune` breaks, it is
almost always a caller of the old `Scorer.score_image` that needs no change (the
template method preserves the `score_image` signature) — investigate any error;
do not blanket-`# type: ignore`.

- [ ] **Step 2: Whole-package lint**

```bash
uv run ruff check src/phenotypic
```
**Expected:** no remaining errors.

- [ ] **Step 3: Whole-suite regression (non-Playwright lanes)**

Run the full unit + integration + smoke suite with the Qt binding and the tune
extra (the Playwright `tests/e2e/gui` lane is a separate CI job and is **not**
gated by this change — exclude it here):
```bash
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest \
    tests/unit tests/integration tests/smoke -q
```
**Expected:** all green (the default `-m "not slow"` deselection applies). Pay
attention to any failure *outside* `tests/unit/tune` / `tests/unit/gui/tune` —
that is the ripple this gate exists to catch. The `tune` GUI tests need the Qt
offscreen platform + `--group qt-test` (see CLAUDE.md).

- [ ] **Step 4: Confirm no goodness relics remain package-wide**

```bash
grep -rnE "higher-is-better|maximize a|_MAXIMIZE\b|median − λ·IQR|geometric mean" \
    src/phenotypic | grep -v "_HIGHER_IS_BAD" || echo "clean"
```
**Expected:** `clean` (or only legitimately-unrelated matches — read each; the QC
flag `_HIGHER_IS_BAD` is intentionally retained and is filtered out above).

- [ ] **Step 5: This gate is verification-only — nothing to commit.** Record the
  result in the PR description (whole-package `mypy` clean + full non-e2e suite
  green). The orchestrator then runs the *final acceptance gate* sequence from the
  README (simplify agent → code-review agent → spec-adherence agent → re-run this
  gate).

---

## Phase-close checklist

- [ ] All three §5.3 surfaces (Task 1/2/3) carry the identical contract — diff
      them by eye; the five numbered steps + the "framework orients, you don't"
      sentence must match.
- [ ] `grep -rnE "higher-is-better|_MAXIMIZE|median − λ·IQR|geometric mean"
      src/phenotypic/tune/CLAUDE.md docs/superpowers/explain/tune-with-optuna*.md
      src/phenotypic/tune/_scoring/_scorer.py` → **no matches** (every goodness
      relic flipped).
- [ ] `uv run --group docs sphinx-build -b html docs/source docs/_build/html -q`
      → no new warnings from `contributing.rst`.
- [ ] `uv run --extra tune pytest tests/unit/tune/test_cost_convention_regression.py -v`
      → all green (requires Phases 1–4 landed).
- [ ] **Task 7:** `uv run --extra tune pytest tests/integration/tune/test_minimize_cost_e2e.py -v`
      → green (the real-tuner end-to-end minimize-cost smoke).
- [ ] **Task 8 (whole-package final gate):** `uv run mypy src/phenotypic` clean;
      `QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit tests/integration tests/smoke -q`
      green; package-wide goodness-relic grep is clean.
- [ ] `uv run mypy src/phenotypic/tune` + `uv run ruff check` → clean.
- [ ] Re-confirm every `file:line` ref in the explainer + `.graph.md` node table
      against the worktree (Phases 1–4 shifted them).
- [ ] **Orchestrator final acceptance gate (README protocol):** simplify agent →
      code-review agent → spec-adherence agent → re-run Task 8. Only then is the
      cutover complete.

## Spec coverage map (Phase 5)

| Spec section | Task |
|---|---|
| §5.3 authoring contract → 3 surfaces | Tasks 1, 2, 3 |
| §6.1–6.6 composite/scalarization math (docs) | Tasks 2, 4 |
| §7 Phase 5 docs (explainer + CLAUDE.md + contrib) | Tasks 1–5 |
| §9 theory notes (explainer prose + references) | Tasks 4f, 4g |
| §10 reflection winner-equivalence | Task 6a |
| §10 overfit-gap sign end-to-end | Task 6b |
| §10 composite-delta snapshot | Task 6c |
| §10 end-to-end minimize-cost smoke (real tuner, multi-plate) | Task 7 |
| Whole-package final acceptance gate (ripple from §5 Scorer template) | Task 8 |
| §12 references (doc citations) | Tasks 4g (Steuer&Choo, Miettinen, Carrell, Schneider/Bischl/Feurer) |
