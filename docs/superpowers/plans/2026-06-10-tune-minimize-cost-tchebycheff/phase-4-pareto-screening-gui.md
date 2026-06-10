# Phase 4 — Pareto + screening + GUI relabel (cost / minimize)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes. Implement task-by-task: failing test → run (red) → minimal impl → run (green) → commit.

**Goal:** Finish the cost cutover in the three remaining direction-bearing surfaces — the Pareto-front math (`_study/_pareto.py`), the screening-freeze winner/selection logic (`_screening_freeze.py`), and the read-only `/tune/` GUI Monitor (`gui/tune/_study_read.py`, `_run_root.py`, `_winner.py`, `_callbacks.py`) — so every place that ranks or picks a trial treats **lower cost = better** and the GUI labels what it shows as **cost (lower is better)**.

**Why this phase is mechanical (and own-PR):** Phases 1–2 already flipped the optimizer direction, the per-term/per-child cost math, `Trial.score`/`objectives` semantics, and `JournalStudyStore.best()` (`max → min`). What is left is every *downstream consumer* that re-implements a max/sort/sentinel against `score` or `objectives` and so still assumes higher-is-better. None of these compute cost; they only *order* by it. Reflection equivalence holds at the winner level (cross-cutting invariant #3): a lower-cost vector must dominate, the screening winner must be the `min`, and the GUI running-best must descend.

**Hard dependency on Phases 1–3 (do not start until they land):**
- **Phase 2** flips `JournalStudyStore.best()` to `min(valid, key=lambda t: t.score)` and ships `_STUDY_NAME = "tune_cost_v1"` plus the **friendly legacy-study detector** (the helper that recognizes a pre-cutover `"tune"` / maximize study and returns a human message instead of Optuna's raw error). Phase 4 *reuses* `best()` and that detector — it does not redefine them. Step 4-D below imports the Phase-2 detector by name; if Phase 2 named it differently, re-point the import (the behavior — friendly message on a legacy study — is the contract, not the symbol).
- **Phase 2** also fixes `_run_root.py`'s `_DEFAULT_STUDY_NAME` (`"tune" → "tune_cost_v1"`). Phase 4 only touches `_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS` in that file. If Phase 2 has not yet flipped `_DEFAULT_STUDY_NAME`, do **not** flip it here — flag it back to the Phase-2 worker; this phase owns only the directions placeholder.
- **Phase 1** raised `_GAP_EPS` and recomputes the generalization gap / relative dispersion on the goodness-equivalent (`1 − cost`). That is why `Trial.gap` stays a non-negative relative dispersion in `[0, ~1]` and `GAP_FLAG_THRESHOLD = 0.15` is **unchanged** here (see Task 4-C, Step "GAP re-review").

**Read first:**
- The "Shared contract & conventions" + "Cross-cutting invariants" sections of [`README.md`](README.md) (cost convention; invariant #3 winner-equivalence; invariant #5 no-silent-maximize).
- Spec §7 "Phase 4" (`docs/superpowers/specs/2026-06-09-tune-minimize-badness-augmented-tchebycheff-design.md:680`), §11 pitfalls (Pareto domination, screening reflection, silent-load detector reuse), and the §11 "Decisions made" OQ6 entry (`:916`).

**Files touched in this phase:**
- `src/phenotypic/tune/_study/_pareto.py` (`_dominates`, `_vector`; verify `knee_point_of`)
- `src/phenotypic/tune/_screening_freeze.py` (six direction sites)
- `src/phenotypic/gui/tune/_study_read.py` (`running_best`, `shortlist`, y-axis label, docstrings)
- `src/phenotypic/gui/tune/_run_root.py` (`_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS`)
- `src/phenotypic/gui/tune/_winner.py` (doctest)
- `src/phenotypic/gui/tune/_callbacks.py` (Monitor read path: reuse the legacy detector)
- `src/phenotypic/gui/FEATURES.md` (gate: a cost-relabel row)
- `src/phenotypic/gui/WORKFLOWS.md` (gate: re-review; no row change expected — see Task 4-F)
- Tests: `tests/unit/tune/test_pareto.py`, `tests/unit/tune/test_screening_freeze.py`, `tests/unit/gui/tune/test_study_read.py`, `tests/unit/gui/tune/test_monitor_figures.py`, `tests/unit/gui/tune/test_run_root.py`

**`_screening.py` is intentionally untouched.** Its importance sorts (`sorted(..., key=lambda kv: kv[1], reverse=True)` at `_screening.py:115` and `:210`) order **importances** — non-negative variance-attribution / permutation magnitudes, **sign-independent of the objective's direction**. fANOVA decomposes variance and RF-permutation measures prediction-error increase; neither cares whether the target is maximized or minimized. Leave both `reverse=True` sorts as-is. A regression test in Task 4-B Step 6 locks this so a future reader does not "helpfully" flip them.

**Commands (run from this worktree root):**
```bash
# one-time (if not already synced for tune):
uv sync --group dev --extra tune
# pure tune tests (Pareto, screening):
uv run --extra tune pytest tests/unit/tune/test_pareto.py -v
# GUI tune tests need a Qt binding + offscreen platform (the tune GUI
# test dir imports the plotly/dash surface; pytest-qt aborts collection
# without a binding). Verified invocation:
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/test_study_read.py -v
# type + lint at the phase boundary:
uv run mypy src/phenotypic/tune src/phenotypic/gui/tune
uv run ruff check --fix src/phenotypic/tune src/phenotypic/gui/tune
```
Commit after every green task.

---

## Task 4-A: Pareto domination flips to minimize (`_study/_pareto.py`)

`_dominates` (`_study/_pareto.py:53`) currently encodes higher-is-better (`>=` / `>`). Under cost, a vector dominates when it is **no worse (≤) on every** axis and **strictly better (<) on at least one**. `_vector` (`:47`) fills a missing objective with `0.0` (the best goodness); under cost the worst fill is `1.0`. `knee_point_of` (`:113`) is direction-agnostic (it projects onto the chord between the extremes; the elbow is the same point regardless of axis polarity) — verify, do not change.

### Step 1: Write the failing test

Append to `tests/unit/tune/test_pareto.py` (after `test_pareto_front_excludes_dominated`, mirroring its hand-built style but in cost coordinates):

```python
def test_pareto_front_excludes_dominated_under_cost():
    """Cost coordinates (lower is better): the dominated point is the HIGH-cost one.

    Points (cost_seg, cost_qc): A=(0.1,0.8), B=(0.5,0.5), C=(0.8,0.1) are mutually
    non-dominated (each wins an axis by being lower). D=(0.6,0.6) is dominated by
    B (0.5,0.5 ≤ on both, strictly on both). The front is {A, B, C}.
    """
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"seg": 0.1, "qc": 0.8}, score=0.45))  # A
    store.append(_trial(1, objectives={"seg": 0.5, "qc": 0.5}, score=0.50))  # B
    store.append(_trial(2, objectives={"seg": 0.8, "qc": 0.1}, score=0.45))  # C
    store.append(_trial(3, objectives={"seg": 0.6, "qc": 0.6}, score=0.60))  # D dominated
    front_numbers = {t.number for t in store.pareto_front()}
    assert front_numbers == {0, 1, 2}


def test_lower_cost_vector_dominates_higher_cost_vector():
    """A strictly-lower-cost trial dominates a strictly-higher-cost one (cost B1)."""
    from phenotypic.tune._study._pareto import _dominates

    assert _dominates([0.2, 0.3], [0.5, 0.6]) is True   # lower on both → dominates
    assert _dominates([0.5, 0.6], [0.2, 0.3]) is False  # higher on both → dominated
    assert _dominates([0.2, 0.6], [0.2, 0.3]) is False  # ties one, worse on other
    assert _dominates([0.2, 0.3], [0.2, 0.3]) is False  # equal vectors never dominate


def test_vector_missing_axis_fills_worst_cost():
    """A trial missing an axis is filled with 1.0 (worst cost), not 0.0 (best)."""
    from phenotypic.tune._study._pareto import _vector

    partial = _trial(0, objectives={"seg": 0.2}, score=0.2)
    assert _vector(partial, ["seg", "qc"]) == [0.2, 1.0]
```

> **Note on `_dominates` import:** it is private but importing it directly in the test is fine — the existing `test_pareto.py` already reaches into the store API; here we lock the pure-function contract that the §11-pitfall-1 (inverted domination) regression must not re-break.

### Step 2: Run to verify it fails

```bash
uv run --extra tune pytest tests/unit/tune/test_pareto.py::test_lower_cost_vector_dominates_higher_cost_vector tests/unit/tune/test_pareto.py::test_vector_missing_axis_fills_worst_cost tests/unit/tune/test_pareto.py::test_pareto_front_excludes_dominated_under_cost -v
```
Expected: FAIL — `_dominates([0.2,0.3],[0.5,0.6])` returns `False` (old `>=`/`>` logic), and `_vector` fills `0.0` not `1.0`.

### Step 3: Minimal implementation

In `src/phenotypic/tune/_study/_pareto.py`:

`_vector` (`:50`) — flip the missing-axis fill:
```python
    return [float(objectives.get(key, 1.0)) for key in keys]
```

`_dominates` (`:68–70`) — flip both comparisons (lower-is-better):
```python
    no_worse = all(left <= right for left, right in zip(lhs, rhs))
    strictly_better = any(left < right for left, right in zip(lhs, rhs))
    return no_worse and strictly_better
```

Also update the prose that asserts direction so the docstrings do not lie:
- Module docstring (`:7`): `"higher-is-better — robust-eval §5"` → `"cost, lower-is-better — robust-eval §5 / cost convention"`.
- `_vector` docstring (`:48`): `"(``0.0`` for any missing)"` → `"(``1.0`` — worst cost — for any missing)"`.
- `_dominates` docstring (`:54`): `"(higher-is-better objectives)"` → `"(cost objectives, lower-is-better)"`; and the body line `"at least as good on every"` stays correct in words, but change the parenthetical `"and strictly better on at least one"` is still correct — only the comparison direction changed.

`knee_point_of` (`:113`) — **no code change.** Add one sentence to its docstring confirming direction-agnosticism, e.g. after "the max-curvature elbow": `"The chord/projection geometry is direction-agnostic — the elbow is the same front point whether axes are goodness or cost — so this is unchanged under the cost cutover."`

### Step 4: Run to verify it passes

```bash
uv run --extra tune pytest tests/unit/tune/test_pareto.py -v
```
Expected: PASS — the new cost tests **and** the existing goodness-coordinate tests still pass. The existing `test_pareto_front_excludes_dominated` (goodness coords A=(0.9,0.2)…) will now compute a *different* front because the comparison flipped; **this is expected and is the bug if it does not** — but check: that test's points were chosen so D=(0.4,0.4) is dominated by B=(0.5,0.5) under `>=`. Under `<=`, B no longer dominates D; instead the all-low point would. **Re-read the existing test before this task and rewrite its expectation if it asserts the old higher-is-better front** — it must now express cost coordinates. If the existing goodness test still passes unchanged, that means its points happen to be symmetric; verify by hand, and if it fails, convert it to cost coordinates (this is the intended migration of the test fixture, not a regression).

> **Migration note:** the existing `test_pareto_front_excludes_dominated`, `test_pareto_front_ignores_failed_trials`, `test_pareto_front_ignores_objectiveless_trials`, `test_knee_point_is_max_distance_to_chord`, and `test_pareto_front_duplicate_objectives_keeps_one_representative` use objective values as *goodness*. After the flip, re-interpret each as cost (the numbers are arbitrary axis values; only the relative ordering of dominated-ness matters). For `test_knee_point_is_max_distance_to_chord` the knee is the **same trial** (knee is direction-agnostic), so it should pass unchanged — confirm. For the front tests, the *dominated* trial flips identity; update each expected `front_numbers` set so the test expresses "the high-cost interior point is dominated". Keep the test count the same.

### Step 5: Commit

```bash
git add src/phenotypic/tune/_study/_pareto.py tests/unit/tune/test_pareto.py
git commit -m "feat(tune): flip Pareto domination + missing-axis fill to minimize-cost"
```

---

## Task 4-B: Screening-freeze winner/selection flips to minimize (`_screening_freeze.py`)

Six direction-bearing sites in `_screening_freeze.py` rank/select trials by `Trial.score`. Under cost, every one inverts. Enumerated (line numbers main ±2):

| # | Site | Symbol / line | Change |
|---|------|---------------|--------|
| 1 | `freeze_value` top-k ranking | `sorted(..., key=lambda t: t.score, reverse=True)` (`:175–179`) | `reverse=True → reverse=False` |
| 2 | `_warm_started_store` top-k seed | `sorted(..., key=lambda t: t.score, reverse=True)` (`:364–368`) | `reverse=True → reverse=False` |
| 3 | `_genuinely_focused_best` | `max(fresh, key=lambda t: t.score)` (`:433`) | `max → min` |
| 4 | `_resolve_winner` explore sentinel | `explore_best.score if … else float("-inf")` (`:446`) | `float("-inf") → float("inf")` |
| 5 | `_resolve_winner` focused sentinel | `focused_best.score if … else float("-inf")` (`:450`) | `float("-inf") → float("inf")` |
| 6 | `_resolve_winner` recovery test | `if focused_score < explore_score:` (`:456`) | `< → >` |
| 7 | `_resolve_winner` union winner | `max((t … ), key=lambda t: t.score, default=explore_best)` (`:475–479`) | `max → min` |
| 8 | `_apply_focused_penalty` | `trial.score - self._focused_penalty` (`:410`) | `- → +` |

> The spec calls this "six sites"; the table lists eight concrete edits because two are the paired `-inf` sentinels and two are paired `sorted(reverse=…)`. Same set, finer granularity. The penalty seam (#8) must add, not subtract: under cost a *penalty* makes a focused trial **worse = higher cost**, so the test that exercises the wrong-freeze recovery (`_focused_score_penalty`) raises focused cost above explore cost, tripping the `focused_score > explore_score` recovery branch.

### Step 1: Write the failing tests

Append to `tests/unit/tune/test_screening_freeze.py`:

```python
def test_freeze_value_numeric_is_top_k_median_by_lowest_cost():
    """Best (lowest-cost) trials carry x in {2,4,6}; median = 4 over the top 3.

    The cost mirror of ``test_freeze_value_numeric_is_top_k_median``: the
    high-cost outlier (score=0.99) must be excluded from the top-k, so the
    central tendency is the median of the three low-cost configs.
    """
    trials = [
        Trial(number=0, params={"x": 2.0}, score=0.10, terms={}, n_images=1),
        Trial(number=1, params={"x": 4.0}, score=0.20, terms={}, n_images=1),
        Trial(number=2, params={"x": 6.0}, score=0.30, terms={}, n_images=1),
        Trial(number=3, params={"x": 99.0}, score=0.99, terms={}, n_images=1),
    ]
    assert freeze_value("x", trials, top_k=3) == 4.0


def test_genuinely_focused_best_is_lowest_cost():
    """The focused round's best is the MIN-cost fresh (non-warm-start) trial."""
    spec = _constant_grid_spec()  # reuse the file's existing spec helper
    controller = ScreeningController(spec, config=ScreeningConfig())
    controller._warm_count = 1
    controller.focused_store = JournalStudyStore([
        Trial(number=0, params={"x": 1}, score=0.05, terms={}, n_images=1),  # warm seed
        Trial(number=1, params={"x": 2}, score=0.40, terms={}, n_images=1),  # fresh
        Trial(number=2, params={"x": 3}, score=0.10, terms={}, n_images=1),  # fresh, best
    ])
    best = controller._genuinely_focused_best()
    assert best is not None and best.number == 2 and best.score == 0.10
```

> **Spec helper:** the file already drives `ScreeningController` over a tiny grid in its orchestration tests. Reuse whatever spec builder those tests use (e.g. a `_constant_grid_spec()` / inline `TuningSpec`). If there is no shared helper, build a minimal `TuningSpec` inline exactly as the existing `test_screening_freeze.py` orchestration tests do — match their pattern; do not invent a new fixture shape.

Append a recovery-direction test (cost: a *penalized* — higher-cost — focused round trips recovery):

```python
def test_wrong_freeze_recovery_when_focused_cost_exceeds_explore(tmp_path):
    """A focused round penalized to HIGHER cost than explore flags the freeze.

    The cost mirror of the existing wrong-freeze recovery test: the test seam
    ``_focused_score_penalty`` now ADDS cost (makes focused worse), so the
    genuinely-focused best is worse (higher) than the explore best, tripping the
    ``focused_score > explore_score`` recovery branch → ``freeze_flagged`` and the
    explore best is returned.
    """
    # Reuse the existing orchestration harness (images + spec + forced freeze
    # report) from the file's recovery test; set _focused_score_penalty to a
    # positive value large enough to push focused cost above explore cost.
    ...  # mirror the existing recovery test body, asserting:
    #   result.freeze_flagged is True
    #   result.winner is the explore best (lowest-cost explore trial)
```

> Locate the existing recovery test in `test_screening_freeze.py` (it sets `_focused_score_penalty=` to force the G3 path) and write the cost mirror **next to it**, reusing its `images`/`spec`/`importance_report_fn` setup verbatim — only the penalty sign-meaning and the assertion direction change.

### Step 2: Run to verify it fails

```bash
uv run --extra tune pytest tests/unit/tune/test_screening_freeze.py -k "lowest_cost or focused_cost" -v
```
Expected: FAIL — `freeze_value` still ranks `reverse=True` (picks the 99.0 outlier into top-k → median 6.0), `_genuinely_focused_best` returns the `max` (number 1), and the recovery branch uses `<` with subtraction.

### Step 3: Minimal implementation

Apply all eight edits in `src/phenotypic/tune/_screening_freeze.py`:

```python
# Site 1 — freeze_value (:175)
    ranked = sorted(
        (t for t in trials if not t.failed and key in t.params),
        key=lambda t: t.score,
        reverse=False,            # cost: lowest score = best
    )[:top_k]
```
```python
# Site 2 — _warm_started_store (:364)
    top = sorted(
        (t for t in self.explore_store.trials if not t.failed),
        key=lambda t: t.score,
        reverse=False,            # cost: lowest score = best
    )[: self._config.top_k]
```
```python
# Site 3 — _genuinely_focused_best (:433)
        return min(fresh, key=lambda t: t.score)
```
```python
# Sites 4 + 5 — _resolve_winner sentinels (:446, :450)
        explore_best.score if explore_best is not None else float("inf")
        ...
        focused_best.score if focused_best is not None else float("inf")
```
```python
# Site 6 — _resolve_winner recovery test (:456)
        if focused_score > explore_score:   # cost: focused WORSE than explore
```
```python
# Site 7 — _resolve_winner union winner (:475)
        winner = min(
            (t for t in union if not t.failed),
            key=lambda t: t.score,
            default=explore_best,
        )
```
```python
# Site 8 — _apply_focused_penalty (:410)
                        update={"score": trial.score + self._focused_penalty}
```

Update the direction-bearing docstrings so they stop saying "highest" / "best held-out" in goodness terms:
- `freeze_value` docstring (`:162`): `"top_k highest-scoring non-failed trials"` → `"top_k lowest-cost non-failed trials"`.
- `_genuinely_focused_best` / `_resolve_winner` references to "best" stay fine in words (best = lowest cost now); add a one-liner to `_apply_focused_penalty`'s docstring: `"Under the cost convention a penalty ADDS cost (makes a focused trial worse)."`.
- `_resolve_winner` recovery comment (`:453–455`): `"underperformed explore on held-out"` is still correct in words; update the inline math comment to read `"focused cost exceeded explore cost"`.
- The class `__init__` docstring for `_focused_score_penalty` (`:240–242`): `"subtract this from every fresh focused-round score"` → `"add this to every fresh focused-round score (raise its cost)"`.

### Step 4: Run to verify it passes

```bash
uv run --extra tune pytest tests/unit/tune/test_screening_freeze.py -v
```
Expected: PASS — new cost tests + all existing screening tests. **Re-read the existing `freeze_value` / recovery tests:** the existing `test_freeze_value_numeric_is_top_k_median` (scores 0.9/0.8/0.7/0.1, x in {2,4,6,99}) picked the high-score trials. Under cost-flip it now picks the *low-score* trials {2.0 (0.9? no — 0.9 is now worst)}… → it will select x={99.0, 6.0, 4.0} → median 6.0, breaking the assert. **Migrate the existing test's scores** so the three "good" configs carry the *lowest* scores (e.g. swap 0.9↔0.1) — the fixture encoded goodness; re-encode it as cost so "good = low" and the median stays 4.0. Same for any other existing screening test that orders by score. Keep the test names; only the score numbers / expected winner identity migrate.

### Step 5: Lock `_screening.py` is unchanged

Add a regression test asserting the importance sorts are direction-independent. Append to `tests/unit/tune/test_screening_freeze.py` (or `test_screening_multiobjective.py` — pick the file that already imports `compute_param_importance_report`):

```python
def test_importance_sort_is_sign_independent_of_cost():
    """Importance ranking is over non-negative variance attribution, not score
    direction: it must NOT flip with the cost cutover. Two stores whose targets
    are exact negations produce the same importance *ranking* (the RF-permutation
    magnitude depends on |variance explained|, not on sign)."""
    from phenotypic.tune._screening import compute_param_importance_report
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    def _store(flip):
        s = JournalStudyStore()
        for n in range(8):
            x = float(n)
            base = 0.05 * n              # x drives the target strongly
            score = (1.0 - base) if flip else base
            s.append(Trial(number=n, params={"x": x, "noise": 0.0},
                           score=score, terms={}, n_images=1))
        return s

    rank_lo = list(compute_param_importance_report(_store(False)).importances)
    rank_hi = list(compute_param_importance_report(_store(True)).importances)
    assert rank_lo[0] == rank_hi[0] == "x"   # x dominates either way; sort unflipped
```

Run:
```bash
uv run --extra tune pytest tests/unit/tune/test_screening_freeze.py::test_importance_sort_is_sign_independent_of_cost -v
```
Expected: PASS with **no change** to `_screening.py`. (If this fails, do not flip `_screening.py` — the failure means the test's target is too weak; strengthen the signal.)

### Step 6: Commit

```bash
git add src/phenotypic/tune/_screening_freeze.py tests/unit/tune/test_screening_freeze.py tests/unit/tune/test_screening_multiobjective.py
git commit -m "feat(tune): flip screening-freeze winner/selection to minimize-cost; lock importance sort direction-free"
```

---

## Task 4-C: GUI study-read flips to cost (`gui/tune/_study_read.py`)

`running_best` re-implements the cumulative best with `max` (`:104`) and labels the curve "best score" (docstring `:85–108`); `shortlist` ranks top-k with `reverse=True` (`:167`, `:173`); `build_objective_figure` labels the y-axis `"score"` (`:250`). All assume higher-is-better. `gap_badge` and `_is_gap_flagged` read `Trial.gap` which (per Phase 1) is a non-negative dispersion — **unchanged**. `best()` is consumed but flipped in Phase 2 — not touched here.

### Step 1: Write the failing tests

Edit `tests/unit/gui/tune/test_study_read.py`. The existing `_store()` helper encodes goodness (running max). Add cost-oriented tests (do not delete the existing ones yet — Step 4 migrates them):

```python
def test_running_best_is_monotone_non_increasing_under_cost():
    """The running best is the cumulative MIN of the trial costs, in order."""
    trials = JournalStudyStore([
        _trial(0, 0.70),
        _trial(1, 0.50),
        _trial(2, 0.60),
        _trial(3, 0.30),
        _trial(4, 0.40),
    ]).trials
    curve = running_best(trials)
    assert curve == [0.70, 0.50, 0.50, 0.30, 0.30]
    assert all(b <= a for a, b in zip(curve, curve[1:]))  # non-increasing


def test_shortlist_top_k_is_lowest_cost():
    """Top-k by cost = the LOWEST-cost trials (best), not the highest."""
    store = JournalStudyStore([
        _trial(0, 0.30), _trial(1, 0.50), _trial(2, 0.40),
        _trial(3, 0.70), _trial(4, 0.60),
    ])
    picks = shortlist(store, k=2)
    numbers = [t.number for t in picks]
    assert 0 in numbers and 2 in numbers      # lowest-cost two are 0 (0.30), 2 (0.40)
    # Score-ascending order (lowest cost first).
    scores = [t.score for t in picks]
    assert scores == sorted(scores)
```

For the y-axis label, edit `tests/unit/gui/tune/test_monitor_figures.py`:

```python
def test_build_objective_figure_best_trace_is_monotone_non_increasing():
    from phenotypic.gui.tune._study_read import build_objective_figure

    trials = [_trial(0, 0.7), _trial(1, 0.5), _trial(2, 0.6), _trial(3, 0.3)]
    fig = build_objective_figure(trials)
    best_traces = [
        tr for tr in fig.data if getattr(tr, "mode", None) and "lines" in tr.mode
    ]
    assert best_traces, "expected a running-best line trace"
    ys = list(best_traces[0].y)
    assert ys == [0.7, 0.5, 0.5, 0.3]
    assert ys == sorted(ys, reverse=True)  # cost: non-increasing
    # y-axis is relabeled to cost (lower is better).
    assert "cost" in fig.layout.yaxis.title.text.lower()
```

### Step 2: Run to verify it fails

```bash
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/test_study_read.py tests/unit/gui/tune/test_monitor_figures.py -k "cost or non_increasing" -v
```
Expected: FAIL — `running_best` uses `max` (curve ascends), `shortlist` picks high scores, y-axis says `"score"`.

### Step 3: Minimal implementation

In `src/phenotypic/gui/tune/_study_read.py`:

`running_best` body (`:101–108`) — flip `max → min`:
```python
    curve: list[float] = []
    best_so_far: float | None = None
    for trial in trials:
        best_so_far = (
            trial.score if best_so_far is None else min(best_so_far, trial.score)
        )
        curve.append(best_so_far)
    return curve
```
`running_best` docstring (`:86–100`): `"monotone non-decreasing cumulative-best score curve"` → `"monotone non-increasing cumulative-best **cost** curve"`; `"best score seen … (higher is better — robust-eval §5), so the curve never decreases"` → `"lowest cost seen … (lower is better — cost convention), so the curve never increases"`; `"they scored the failure floor, which simply never advances the running best"` stays correct (failure floor is the worst = highest cost, never lowers the min).

`shortlist` — both sorts (`:167`, `:173`):
```python
    top_k = sorted(valid, key=lambda t: t.score)[:k]          # ascending = lowest cost
    ...
    return sorted(picked.values(), key=lambda t: t.score)     # ascending
```
`shortlist` docstring (`:140–165`): `"top-*k* by scalar ``score`` (the obvious winners)"` → `"top-*k* by lowest cost"`; `"returned score-descending"` → `"returned cost-ascending (lowest first)"`; `"a high scalar score can still miss a front-defining trade-off"` → `"a low scalar cost can still miss a front-defining trade-off"`.

`build_objective_figure` — y-axis (`:250`) and series name (`:232`):
```python
            name="trial cost",
            ...
            yaxis={"title": "cost (lower is better)"},
```
`build_objective_figure` docstring (`:207–222`): `"running-best line + raw-score scatter"` → `"running-best (lowest-cost) line + raw per-trial cost scatter"`.

`gap_badge` / `_is_gap_flagged` (`:111–137`): **no logic change.** Add a one-line clarifying comment in `_is_gap_flagged` — `"gap is a non-negative relative dispersion on the goodness-equivalent (Phase 1); higher = more unstable, independent of the cost flip"`.

`_ReadableStore.best()` Protocol docstring (`:76–78`): `"the non-failed trial with the highest score"` → `"the non-failed trial with the lowest cost"` (mirror the Phase-2 `JournalStudyStore.best()` docstring; this is the Protocol slice, prose-only).

### Step 4: Migrate the existing goodness tests + run

Re-encode the existing `test_study_read.py` fixtures (`_store()` and the standalone tests) as cost: the "winner" must be the **lowest**-score trial. Concretely:
- `test_running_best_is_monotone_non_decreasing` → rename to `..._non_increasing`, flip the expected curve, assert `b <= a`.
- `test_gap_badge_flags_high_dispersion_winner` → **unchanged** (gap semantics did not flip; a gap of 0.25 > 0.15 still flags).
- `test_shortlist_includes_top_scorers_and_gap_flagged_deduped` → swap "top-2 by score are 3,4" to "lowest-cost two are …"; flip the score-order assertion to ascending.

Run:
```bash
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/test_study_read.py tests/unit/gui/tune/test_monitor_figures.py -v
```
Expected: PASS (all, including migrated existing tests).

### Step 5: GAP re-review (state the decision, no code change)

`GAP_FLAG_THRESHOLD = 0.15` (`:52`) is **unchanged** in Phase 4. Rationale, recorded here and in a one-line code comment update: Phase 1 already redefined `Trial.gap` as the relative across-plate dispersion computed on the **goodness-equivalent** (`1 − cost`) with `_GAP_EPS` raised to ~0.02. Because the gap is computed in goodness space, the "swings by more than ~15% plate-to-plate" interpretation in the docstring (`:48–52`) is preserved verbatim under the cost cutover — the threshold's meaning did not move, so the number does not change. Update only the docstring's cross-reference: `"(``_study_store.Trial.gap``)"` line, append `"computed on the goodness-equivalent ``1 − cost`` (Phase 1)"`. Do **not** touch the `0.15` value.

### Step 6: Commit

```bash
git add src/phenotypic/gui/tune/_study_read.py tests/unit/gui/tune/test_study_read.py tests/unit/gui/tune/test_monitor_figures.py
git commit -m "feat(tune-gui): study-read running-best/shortlist/y-axis relabel to cost (lower=better)"
```

---

## Task 4-D: Monitor read path reuses the Phase-2 legacy-study detector (`gui/tune/_callbacks.py`)

`read_study_for_monitor` (`_callbacks.py:470`) opens the live `OptunaStudyStore` on a worker thread; on **any** open/connect error it degrades to the journal with the generic `_NOTE_LIVE_UNREACHABLE` note (`:527–533`). After the study-name bump (Phase 2, `_STUDY_NAME = "tune_cost_v1"`), a pre-cutover `"tune"` study can no longer be opened by name — but a user pointing the Monitor at an **old run directory** whose marker still records `study_name="tune"` would hit the generic "couldn't reach the live study" note, which is misleading. Reuse the Phase-2 friendly detector so the note explains the real cause: the run predates the cost cutover and must be re-run.

> **Scope guard (invariant #5):** Phase 4 must never *open* a legacy maximize study as minimize (the silent-load hazard). The name bump already prevents that at the optimizer. Here we only improve the **read-only** degrade message; we do not attempt to read a legacy study's trials.

### Step 1: Write the failing test

The friendly-message logic is a pure branch over `(root, raised_exception)`. Extract it into a module-level helper so it is unit-testable without a live study (per the GUI-review lesson: callback wiring bugs only fire on the live `/_dash-update-component` route; pull the branch into a helper). Add to `tests/unit/gui/tune/test_monitor.py`:

```python
def test_legacy_study_degrade_note_is_friendly(tmp_path):
    """A pre-cutover run (study_name='tune') degrades with a re-run message,
    not the generic 'couldn't reach the live study' note."""
    from phenotypic.gui.tune._callbacks import _monitor_degrade_note
    from phenotypic.gui.tune._run_root import TuneRunRoot

    legacy = TuneRunRoot(
        path=tmp_path, trials_path=None, storage_url="sqlite:///x.db",
        study_name="tune", directions=None, images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )
    note = _monitor_degrade_note(legacy, RuntimeError("study not found"))
    assert "re-run" in note.lower() or "pre-cutover" in note.lower()


def test_current_study_degrade_note_is_generic(tmp_path):
    """A current-convention run keeps the generic unreachable note."""
    from phenotypic.gui.tune._callbacks import (
        _NOTE_LIVE_UNREACHABLE, _monitor_degrade_note,
    )
    from phenotypic.gui.tune._run_root import TuneRunRoot

    current = TuneRunRoot(
        path=tmp_path, trials_path=None, storage_url="sqlite:///x.db",
        study_name="tune_cost_v1", directions=None, images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )
    assert _monitor_degrade_note(current, RuntimeError("timeout")) == _NOTE_LIVE_UNREACHABLE
```

### Step 2: Run to verify it fails

```bash
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/test_monitor.py -k degrade_note -v
```
Expected: FAIL — `ImportError: cannot import name '_monitor_degrade_note'`.

### Step 3: Minimal implementation

Add the helper near the other Monitor notes in `_callbacks.py` (after `read_study_for_monitor`). Reuse the Phase-2 detector — import it function-local to keep the module import optuna-free:

```python
#: The note for a pre-cutover ("tune", maximize) run reached after the cost
#: cutover: its study can't be opened by the new name, and even if it could the
#: stored cost/score meaning is inverted, so the only safe action is a re-run.
_NOTE_LEGACY_STUDY: str = (
    "this run predates the cost cutover (study 'tune') -- its results use the "
    "old higher-is-better convention and can't be monitored live. Re-run it "
    "with the current phenotypic.tune to get cost-convention results."
)


def _monitor_degrade_note(root: "TuneRunRoot", error: BaseException) -> str:
    """Pick the Monitor degrade note for a failed live-study open.

    A pre-cutover run (its marker/spec still names the legacy ``"tune"`` study)
    gets a friendly re-run message; everything else keeps the generic
    "couldn't reach the live study" note. Reuses the Phase-2 legacy-study
    detector (the same predicate the CLI startup guard uses) so the GUI and CLI
    can't disagree about what "legacy" means.
    """
    # Lazy import: the detector lives behind the tune extra; importing it
    # function-local keeps _callbacks importable without optuna.
    from phenotypic.tune._tune_cli._run import is_legacy_study_name  # Phase 2 symbol

    if is_legacy_study_name(root.study_name):
        return _NOTE_LEGACY_STUDY
    return _NOTE_LIVE_UNREACHABLE
```

> **Phase-2 symbol name:** `is_legacy_study_name` is the *expected* name of the Phase-2 detector predicate (a `str -> bool` that returns `True` for `"tune"` and any other pre-`tune_cost_v1` name). If Phase 2 shipped it under a different name or a different module, re-point this single import — the contract is "True for a pre-cutover study name". If Phase 2 only shipped a richer detector (e.g. one that inspects an Optuna study's stored `direction`), prefer the name-based predicate here: the read-only monitor must not connect to a legacy study just to classify it, so a **name-only** check is the right tool on this path. If no Phase-2 helper exists yet, add a one-line local fallback `return root.study_name != "tune_cost_v1"` and leave a `# TODO(phase-2): replace with the shared detector` — but flag the gap to the Phase-2 worker.

Wire the helper into both degrade branches of `read_study_for_monitor` (`:519–533`):
```python
    except FutureTimeout:
        logger.warning(... )
        return _load_journal(root), _monitor_degrade_note(root, FutureTimeout())
    except Exception as exc:  # noqa: BLE001 - any open/connect error degrades
        logger.warning(... )
        return _load_journal(root), _monitor_degrade_note(root, exc)
```
(The timeout branch passes a synthesized `FutureTimeout()` so the helper's signature is uniform; the note for a current run is unchanged, and a legacy run that times out still gets the legacy note — its name is the dispositive signal.)

### Step 4: Run to verify it passes

```bash
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/test_monitor.py -v
```
Expected: PASS.

### Step 5: Commit

```bash
git add src/phenotypic/gui/tune/_callbacks.py tests/unit/gui/tune/test_monitor.py
git commit -m "feat(tune-gui): friendly legacy-study note on the Monitor degrade path"
```

---

## Task 4-E: `_run_root.py` directions placeholder + `_winner.py` doctest

`_run_root.py` synthesizes a 2-axis **maximize** vector for a multi-objective run from the `is_multi_objective` boolean (`:46`). Under cost every axis minimizes. `_winner.py`'s doctest pins `score=0.9` (a *good* goodness, now a *bad* cost) — change to a low cost so the example reads as a strong candidate.

### Step 1: Write/adjust the failing tests

`tests/unit/gui/tune/test_run_root.py` already asserts `len(root.directions) >= 2` (`:124–125`) — strengthen it to pin the value. Edit that test:

```python
    # is_multi_objective=True → two-axis minimize directions (cost convention).
    assert root.directions is not None
    assert root.directions == ["minimize", "minimize"]
```

Also fix the two `test_monitor_figures.py` multi-objective roots that hardcode `directions=["maximize", "maximize"]` (`:85`) → `["minimize", "minimize"]` (they only test `len > 1`, so either value passes `monitor_pareto_visible`; flip them for consistency and to avoid a stale "maximize" literal).

### Step 2: Run to verify it fails

```bash
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/test_run_root.py -k direction -v
```
Expected: FAIL — placeholder is `["maximize", "maximize"]`.

### Step 3: Minimal implementation

`src/phenotypic/gui/tune/_run_root.py:46`:
```python
_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS: list[str] = ["minimize", "minimize"]
```
Update its docstring (`:42–46`): `"a 2-axis maximize vector"` → `"a 2-axis minimize vector"`; `"Every tuning objective is higher-is-better (robust-eval §5), so the synthesized axes are both ``"maximize"``"` → `"Every tuning objective is a cost (lower-is-better — cost convention), so the synthesized axes are both ``"minimize"``"`.

> **`_DEFAULT_STUDY_NAME` (`:38`) is Phase-2's.** If it still reads `"tune"` here, leave it and flag the Phase-2 worker — do not flip it in this phase (avoids a merge collision with Phase 2's persistence commit).

`src/phenotypic/gui/tune/_winner.py` doctest (`:78`): change `score=0.9` → `score=0.05` (a low cost = strong candidate) so the example narrative matches the cost convention. The doctest only checks the *override landed* (`.sigma == 3.0`), so the score value is cosmetic — but it must not read as "0.9 is good". Also update the `study_name="tune"` in the doctest's `TuneRunRoot(...)` (`:73`) to `"tune_cost_v1"` for consistency **only if** Phase 2 has not already done so; if Phase 2 owns the doctest study-name fix, skip it here.

### Step 4: Run to verify it passes

```bash
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/test_run_root.py tests/unit/gui/tune/test_monitor_figures.py -v
uv run --extra tune pytest --doctest-modules src/phenotypic/gui/tune/_winner.py -v
```
Expected: PASS.

### Step 5: Commit

```bash
git add src/phenotypic/gui/tune/_run_root.py src/phenotypic/gui/tune/_winner.py tests/unit/gui/tune/test_run_root.py tests/unit/gui/tune/test_monitor_figures.py
git commit -m "feat(tune-gui): MO directions placeholder + winner doctest to minimize-cost"
```

---

## Task 4-F: FEATURES.md gate + WORKFLOWS.md re-review

Any change under `src/phenotypic/gui/` trips `gui-checks`' `features-md-gate` (rejects a PR touching `gui/` without modifying `FEATURES.md`). Tasks 4-C/4-D/4-E touch `gui/tune/`, so `FEATURES.md` **must** change. The visible affordance that changed is the **Monitor objective figure's semantics**: it now plots **cost (lower is better)** with a non-increasing running-best, and the y-axis is relabeled. Add a row and update the existing objective-figure row's expected behaviour.

### Step 1: Update the existing Monitor objective-figure row

In `src/phenotypic/gui/FEATURES.md`, find the `Monitor — objective figure` row (around `:325`). Update its "Expected behaviour" cell:

> `The objective figure plots the raw per-trial **cost** plus a monotone **non-increasing** running-best (lowest-cost) trace; the y-axis is labelled "cost (lower is better)"; an empty journal renders a safe empty figure.`

Update its `Test ref` to the renamed test:
> `tests/unit/gui/tune/test_monitor_figures.py::test_build_objective_figure_best_trace_is_monotone_non_increasing`

### Step 2: Add a cost-convention relabel row

Add one new `✅ shipping` row in the Tune co-pilot Monitor section (after the objective-figure row), documenting the cost relabel as a user-visible convention:

```
| Monitor — cost convention     | `build_objective_figure` y-axis + `running_best` / `shortlist` | Tuning now MINIMIZES a bounded [0,1] cost (0=perfect, 1=worst); the Monitor labels the objective y-axis "cost (lower is better)", the running-best descends, and the shortlist ranks lowest-cost-first. | ✅ shipping | unit | tests/unit/gui/tune/test_study_read.py::test_running_best_is_monotone_non_increasing_under_cost |
```

Match the existing column widths loosely — the validator parses the markdown table cells, not the padding; keep the pipe count consistent with the section's header row.

### Step 3: WORKFLOWS.md re-review (expected: no row change)

Open `src/phenotypic/gui/WORKFLOWS.md` and read the `tune_copilot` row (`:52`). Its description says "read trial progress (objective + importance figures, gap badge, trials table)" — generic enough that the cost relabel does **not** change the *flow*; the user still "watches the live read". **No WORKFLOWS.md row change is required** (the affordance label changed, not the end-to-end flow), so no `_capture_tune_copilot` rewrite and no screenshot recapture is mandated by the round-trip gate.

> **If** a reviewer wants the screenshot to show the new "cost (lower is better)" axis label, that is an optional polish: run `uv run python scripts/capture_gui_tutorial_screenshots.py` and commit the **full** regenerated PNG set (per the project rule: commit everything, do not cherry-pick the collateral churn). This is not gate-required for Phase 4 because no WORKFLOWS.md row text changed. Default: **skip** the recapture unless the reviewer asks.

### Step 4: Verify the FEATURES gate passes locally

```bash
# The features-md-gate validates Test ref existence on ✅ rows + that gui/ changes
# touched FEATURES.md. Reproduce the Test-ref check the pre-commit hook runs:
uv run python scripts/check_features_md.py 2>/dev/null || \
  echo "(if the validator script name differs, find it under scripts/ and run it)"
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest \
  tests/unit/gui/tune/test_study_read.py::test_running_best_is_monotone_non_increasing_under_cost \
  tests/unit/gui/tune/test_monitor_figures.py::test_build_objective_figure_best_trace_is_monotone_non_increasing -v
```
Expected: the two `Test ref`s resolve (the tests exist and pass) and the validator is green.

### Step 5: Commit

```bash
git add src/phenotypic/gui/FEATURES.md
git commit -m "docs(tune-gui): FEATURES.md cost-convention relabel row (Monitor objective figure)"
```

---

## Task 4-G: Type + lint + full-suite gate (phase boundary)

### Step 1: Type-check

```bash
uv run mypy src/phenotypic/tune src/phenotypic/gui/tune
```
Expected: `Success: no issues found`. (The flips are value/comparison swaps and prose; no signature changes, so mypy should be clean. If `_monitor_degrade_note`'s `error` param is flagged unused, keep it — it documents intent and lets a future detector inspect the exception; add `# noqa`-free by referencing it in the docstring only, or rename to `_error` if mypy/ruff complains about an unused arg.)

### Step 2: Lint

```bash
uv run ruff check --fix src/phenotypic/tune src/phenotypic/gui/tune tests/unit/tune tests/unit/gui/tune
```
Expected: no remaining errors. If ruff flags the unused `error` arg in `_monitor_degrade_note` (ARG001), rename it `_error` (the leading underscore is ruff's unused-arg escape).

### Step 3: Run the full Phase-4 test surface

```bash
uv run --extra tune pytest tests/unit/tune/test_pareto.py tests/unit/tune/test_screening_freeze.py tests/unit/tune/test_screening_multiobjective.py tests/unit/tune/test_run_tuning_pareto.py -v
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/unit/gui/tune/ -v
QT_QPA_PLATFORM=offscreen uv run --extra tune --group qt-test pytest tests/integration/gui/tune/ -v
```
Expected: all green. Pay attention to `tests/unit/tune/test_run_tuning_pareto.py` and `test_screening_multiobjective.py` — they exercise the full multi-objective + screening pipeline end-to-end and will catch any flipped-site interaction. If a pre-existing test there still encodes goodness (asserts a high-score winner), migrate its expected winner to lowest-cost (same migration pattern as Tasks 4-A/4-B).

### Step 4: Commit any lint fixes

```bash
git add -A && git commit -m "style(tune): lint Phase-4 cost-flip changes" || echo "nothing to commit"
```

---

## Phase 4 done-criteria

- `_dominates`/`_vector` minimize cost; `knee_point_of` unchanged (verified direction-agnostic); a lower-cost vector dominates (Task 4-A test) — closes §11 pitfall #1 (inverted domination).
- All eight `_screening_freeze.py` direction sites flipped; screening winner = `min`-cost across both rounds; wrong-freeze recovery trips when focused **cost exceeds** explore cost; `_screening.py` importance sorts unchanged + locked by a sign-independence test.
- GUI Monitor: `running_best` non-increasing, `shortlist` lowest-cost-first, y-axis "cost (lower is better)"; `_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS = ["minimize","minimize"]`; `_winner.py` doctest reads as a low-cost candidate.
- `GAP_FLAG_THRESHOLD = 0.15` unchanged (gap is goodness-space per Phase 1) — rationale documented in the threshold's docstring.
- A pre-cutover run shows the friendly legacy-study note on the read-only Monitor (reuses the Phase-2 detector; name-only check, never opens a legacy study) — honors invariant #5 (no silent maximize).
- `FEATURES.md` gate satisfied (objective-figure row updated + a cost-convention row added, both with resolving `Test ref`s); `WORKFLOWS.md` reviewed, **no row change** (flow unchanged), screenshot recapture optional/skipped.
- `mypy` + `ruff` clean on `src/phenotypic/tune` and `src/phenotypic/gui/tune`; the full Phase-4 unit + integration GUI tune suites pass.
- **Reflection winner-equivalence (invariant #3):** confirm against the Phase-5 cross-phase regression that the new minimize winner equals the old maximize winner for single-term / arithmetic-mean / Pareto paths (the composite is the one intended behavior change — Phase 3).
