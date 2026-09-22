# C1 review — `@figure` backend declaration, implementation and tests

- **Date:** 2026-09-21
- **Worktree:** `.claude/worktrees/figure-backend-routing`, branch `fix/figure-backend-routing`
- **Cluster:** `2b11e6c8` (shared predicate), `935c66f0`, `72f2ab3e` (annotation sweeps),
  `afdb77d4` (required `backend`, branching wrapper, return check), `c449cbd2`
  (`report()` guard)
- **Subject:** spec `design.md` §1; plan Tasks 1–3
- **Method:** static reading, an independent AST re-count off git objects, and **seven
  one-line mutations applied to the implementation and reverted**, run by the
  orchestrator against `tests/unit/abc_/plotting/` (31 tests) and, where relevant,
  `tests/unit/plotting/test_output_adapter.py` (9 more).

---

## Verdict

**Safe to build C2–C6 on.** No defect in this cluster changes runtime behaviour, and
the two substantive tests the prior review flagged as suspect both survive mutation:
the Plotly theming assertion and the matplotlib `rc_context` assertion each fail when
the behaviour they claim to guard is removed. The `report()` guard is sited correctly,
and its placement is independently pinned by a test.

Two findings are real and both are in the **tests**, not the code. The implementation
is correct today; these two tests would not notice if it stopped being. Neither blocks
C2–C6 — four added assertions close both.

One thing I would fix **before** C2–C6 rather than after: the decorator's entire type
surface still declares `go.Figure` (§ *Implementation defects*, D1). C2 and C5 will
read `FigureSpec.backend` and thread figures further through the publication layer, and
propagating a knowingly-wrong annotation outward is cheaper to stop now than to unpick
across five more commits.

Mutation results, in full:

| | Mutation (one line, reverted) | Result | Predicted |
|---|---|---|---|
| Baseline | — | 40 passed | — |
| M1 | `backend` gets a default `= "plotly"` | 1 failed — `test_backend_is_required` (DID NOT RAISE) | ✔ |
| M2 | plotly branch returns `built`, skipping `apply_theme` | 1 failed — `test_a_plotly_figure_is_themed_after_construction` | ✔ |
| M3 | mpl figure built **outside** `phenotypic_mpl_context()` | 1 failed — `test_the_mpl_theme_is_live_while_the_figure_is_built` | ✔ |
| M4 | `report()` mpl guard removed (`if False:`) | 3 failed — all three refusal tests | ✔ |
| M5 | guard moved behind the controls branch | 1 failed — **only** the controls refusal test | ✔ |
| M6 | `figure_backend_of` loses its `__name__ == "Figure"` check | **40 passed** | ✔ |
| M7 | `_require_backend` names the wrong remedy backend | **31 passed** | ✔ |

Each mutation printed its own diff before its run, so M6 and M7 are genuine coverage
gaps rather than seds that matched nothing. The tree was verified clean against a
pre-run `sha256sum` snapshot afterwards.

**M3 is the most reassuring result in the set.** It is the only guard against the
rc_context-vs-post-pass inversion — the failure the spec's Background says produces an
unthemed matplotlib figure that looks like it worked — and it fails with a legible
diagnosis rather than a bare `False`:

```
assert cycler(... '#17becf']) == cycler(... '#D55E00'])
```

matplotlib's default cycle against Okabe-Ito. Both of that test's clauses are
load-bearing: `observed_cycle` catches building outside the context, and `before ==
after` catches an unscoped `rcParams.update` that never restores.

**M5 is the direct proof that B6 was necessary, and the check that keeps it fixed.**
Moving the guard behind the controls branch — which is what siting it in
`_compose_control_free_figure` amounts to — fails *only*
`test_report_refuses_a_matplotlib_provider_that_declares_controls`. That one test is
the entire defence of the guard's placement. Without it, the composer siting would have
passed review looking correct. B6 came from measurement, not from reading.

---

## Tests that cannot fail

### T1 — `figure_backend_of`'s class-name check has zero coverage

**Mutation that should have broken it:** delete the guard in
`src/phenotypic/abc_/plotting/_output.py:30-31`.

```python
    if type(figure).__name__ != "Figure":
        return None
```

**Result: 40 passed** — the whole of `tests/unit/abc_/plotting/` *and*
`tests/unit/plotting/test_output_adapter.py`.

The predicate is a conjunction of two independent checks: the module prefix and the
class name. `test_returns_none_for_an_unsupported_object`
(`tests/unit/abc_/plotting/test_output.py:22`) passes `object()`, `None` and
`"not a figure"` — every one of which has module `builtins` and is therefore screened
out by the *module* half. The name half is never reached by any assertion in the tree.
`test_output_adapter.py`'s `PlotPage("bad", object())` is the same shape and screens
the same way.

What the guard actually protects: with it removed, `figure_backend_of(go.Scatter())`
returns `"plotly"` (module `plotly.graph_objs._scatter`) and
`figure_backend_of(ax)` returns `"mpl"` for any matplotlib artist. Every plotly and
matplotlib object in existence is classified as a figure. That reaches
`FigureAdapter.backend_name` and `_require_backend` identically, since both now
delegate here — so the one module both layers import would hand both of them a wrong
answer, and nothing in the suite would say so.

**Close it with two lines** in `test_returns_none_for_an_unsupported_object` — the case
that is missing is a plotly-or-matplotlib object that is not a `Figure`:

```python
    assert figure_backend_of(go.Scatter()) is None
    assert figure_backend_of(MplFigure().add_subplot()) is None
```

### T2 — the remedy half of the mismatch message is unpinned

**Mutation that should have broken it:** `_pht_plot.py:149`

```python
    other = "mpl" if declared == "plotly" else "plotly"   →   other = declared
```

**Result: 31 passed.**

`_require_backend` builds an error in two halves: a *diagnosis* (what you declared,
what came back) and a *remedy* (`Declare backend={other!r}, or return a {expected}`).
The two mismatch tests assert only substrings that are present regardless of whether
the remedy is right:

- `test_plotly_declared_method_returning_matplotlib_raises` asserts `"wrong"`,
  `"'plotly'"` and `"matplotlib.figure.Figure"`. With `other = declared` the message
  reads *"...declared backend 'plotly' but the method returned
  matplotlib.figure.Figure. Declare backend='plotly', or return a
  plotly.graph_objects.Figure."* — all three substrings still present.
- `test_mpl_declared_method_returning_plotly_raises` asserts `"'mpl'"` and `"plotly"`.
  `"plotly"` matches the *actual-type* clause, so it is satisfied by the diagnosis
  alone and says nothing about the remedy.

The same is true of `expected` (`:150-153`): in the plotly-declared test the string
`"matplotlib.figure.Figure"` appears in the actual-type clause whatever `expected`
holds, so swapping the two branches is equally invisible.

This matters more than a message nit. Spec §1 specifies the message *shape*, and the
remedy is its actionable half — the whole reason the spec chose "mismatch is an error,
not a fallback" is that the error has to tell the author what to do. Under this
mutation a user who returned a matplotlib figure is told to declare `backend='plotly'`,
which is what they already declared: sent in a circle by the error designed to stop
that.

**Close it with one assertion per test:**

```python
    assert "Declare backend='mpl'" in message      # plotly-declared test
    assert "Declare backend='plotly'" in message   # mpl-declared test
```

### T3 — `test_agrees_with_the_publisher_vocabulary` is missing its third arm *(unmeasured)*

Spec §5 specifies this test as *"`figure_backend_of` agrees with
`FigureAdapter.backend_name` across Plotly, matplotlib, **and an unsupported
object**"*. The implemented test (`test_output.py:27`) covers two of the three; the
unsupported arm is absent, so nothing asserts that `backend_name(object())` raises
`TypeError` naming the type.

I am flagging this differently from T1 and T2: **I did not measure it.** The raise
branch is exercised indirectly by
`test_unsupported_page_fails_without_suppressing_sibling`
(`tests/unit/plotting/test_output_adapter.py:110`), but that test asserts only that the
*good* page published — it never asserts the error type or message, so whether it would
survive `raise TypeError(...)` → `return "plotly"` is a question I would need a mutation
to answer, and did not run one. Treat this as an identified gap of unknown depth rather
than a confirmed hole. The fix is one line either way:

```python
    with pytest.raises(TypeError, match="unsupported figure type"):
        FigureAdapter.backend_name(object())
```

---

## Implementation defects

### D1 — the decorator's type surface still declares Plotly-only, and that is why a wrong return type is invisible to mypy

`backend="mpl"` is now a first-class, required declaration, but every annotation and
docstring on the path it takes still says `go.Figure`:

| Location | Declares |
|---|---|
| `_pht_plot.py:127` | `FigureSpec.method: Callable[..., "go.Figure"]` |
| `:170` | `figure(...) -> Callable[[Callable[..., "go.Figure"]], Callable[..., "go.Figure"]]` |
| `:203-204` | `decorator(fn: Callable[..., "go.Figure"]) -> Callable[..., "go.Figure"]` |
| `:230` | `def wrapper(...) -> "go.Figure"` — **returns a matplotlib figure in the mpl branch** |
| `:307` | `BoundFigures.render(...) -> "go.Figure"` |
| `:411` | `_render_spec(...) -> "go.Figure"` |
| `:171` | *"Mark a method as a figure builder and lazily apply the house theme"* — the mpl branch applies no post-pass |
| `:190` | *"Returns: A decorator for a Plotly figure-building method."* |
| `:315` | *"Returns: The themed Plotly figure."* |

This is not merely stale prose. Because `fn` is declared `Callable[..., "go.Figure"]`,
mypy infers `built: go.Figure` in **both** branches, so `return built` type-checks in
the mpl branch and a method annotated `-> go.Figure` that returns an `MplFigure` raises
no static complaint. The annotation is the reason the exact mistake `_require_backend`
exists to catch at runtime cannot be caught statically — the type system has been told
the wrong thing, so it agrees with the bug.

Minimum honest fix is a backend-neutral alias; `FigureLike` already exists in
`_output.py` for precisely this and is already imported by this package. Note this
lands **before** C5, which threads figures further through the publication layer.

### D2 — the deliberately removed working path is not legible to a future reader

The guard refuses `report()` for *any* `mpl` spec, including a lone one that needs no
composition and, after `afdb77d4`, demonstrably worked
(`_compose_control_free_figure` short-circuits a single spec at `:521-522`). Plan Task 3
Step 2 records this as a user decision — predictability over one incidentally-working
case — but **none of that reaches the code**:

- the inline comment (`:487-489`) explains only the *placement* (B6), not the
  single-spec decision;
- the `Raises:` entry (`:471-472`) says *"Composing matplotlib figures is not
  supported"*;
- the `Returns:` block (`:466-467`) is unchanged and still describes the old behaviour.

So a reader who calls `report()` on a one-figure matplotlib provider is told
`cannot compose matplotlib figures (one)` when there is manifestly nothing to compose.
That reads as a bug in the guard, and the natural repair — "obviously it should
short-circuit a single spec, same as the composer does" — silently reverts a decision
that was taken deliberately. One sentence in the `Raises:` entry prevents it, e.g.
*"including a single figure, which needs no composition: the refusal is uniform by
choice, and `inspect()` is the supported call."*

### D3 — the site count in `afdb77d4`'s commit message is wrong *(corrected in `18acc5b4`)*

Recorded here because the reconciliation is the durable part. `afdb77d4`'s message
claims **44** `@figure` applications with exactly 1 lacking `backend=`. Counted at every
revision in the cluster, off git objects only:

| Revision | `src/` | `test_figure_backend` | `test_pht_plot` | `test_notebook_adapter` | **total** | missing `backend=` |
|---|---|---|---|---|---|---|
| `2b11e6c8~1` | 27 | — | 9 | 2 | **38** | 38 |
| `935c66f0` | 27 | — | 9 | 2 | **38** | 11 |
| `72f2ab3e` | 27 | — | 9 | 2 | **38** | 0 |
| **`afdb77d4`** | 27 | 7 | 9 | 2 | **45** | **1** |
| `c449cbd2` | 27 | 11 | 9 | 2 | **49** | 1 |
| `HEAD` | 27 | 11 | 9 | 2 | **49** | 1 |

The tree at `afdb77d4` held **45**, not 44. Task 3 then added exactly four sites
(7 → 11), giving **45 + 4 = 49** at HEAD. The substantive claim — one bare site, the
deliberate one — was correct then and is correct now; the total was off by one.

**Why it was off by one.** The counting script carried `except SyntaxError: continue`
and ran under the node's **system `python3`, which is 3.6.8**. Same script, same clean
tree, two interpreters:

| Interpreter | sites found | files silently skipped |
|---|---|---|
| system `python3` **3.6.8** | 48 | **44** |
| `.venv/bin/python` 3.12.10 | **49** | 0 |

Among the 44 files 3.6.8 cannot parse is `measure/_measure_symzones.py`, which holds
one of the 27 `src/` sites — so the skip costs exactly one site, at HEAD and at
`afdb77d4` alike. 49 → 48 now; 45 → **44** then. The script printed a confident total
having read part of the tree, and its success output was indistinguishable from its
partial output.

**An arithmetic story that fits and is still false.** My first explanation in this
review was that 44 was the count of sites *carrying* `backend=` — 45 − 1, the bare site
— mislabelled as the total. That arithmetic is correct and the conclusion is wrong: the
missing one is the *skipped* site, not the *bare* site. The two subtractions land on
the same number for unrelated reasons, which is exactly what made the wrong explanation
plausible enough to write down. Worth recording as its own caution: a number that
reconciles is not thereby explained, and a tidy derivation is weak evidence next to
reproducing the mechanism.

**The lesson is not "AST beats grep".** Both instruments failed, for different reasons
and with the same symptom. The grep was wrong because a line-oriented tool was asked a
syntax question — it matched the error-message string containing `@figure(` at
`_pht_plot.py:174`. The AST walk was wrong because it could decline to read an input
and still report a total. Neither printed anything that distinguished a complete answer
from a partial one.

So: **any sweep that can skip an input must count and print the skips.** That is the
same defect class as the two decorative tests this review was convened to hunt — a
success signal identical to the no-op signal — relocated from the subject into the
measuring instrument.

For anyone re-deriving these numbers: use
`/rhome/anguy344/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/bin/python3.12`
(stdlib-only parse, imports nothing from the project), and print the skip count.

---

## Spec divergence

Only one, and it is in the tests:

- **§5, `figure_backend_of` agreement test** — specified across three inputs (Plotly,
  matplotlib, unsupported); implemented across two. See T3.

Two near-misses that are **not** divergences, checked and cleared:

- **§5, "the returned [mpl] figure is not post-processed"** is not asserted directly.
  It is covered only incidentally — `apply_theme` on an `MplFigure` would raise
  `AttributeError` on `.layout`, so a post-pass could not pass the test silently.
  Adequate, but the coverage is accidental rather than designed.
- **§1 message shape** is implemented exactly as the spec's example, `other` and
  `expected` included — the defect in T2 is that nothing *tests* it, not that it is
  wrong.

Everything else in §1 matches: the required keyword-only `backend` with no default; the
asymmetric theming (Plotly themes the result, matplotlib themes the construction, no
mpl post-pass); `_require_backend` raising rather than coercing or sniffing; the single
shared predicate in `_output.py` with `FigureAdapter` delegating; and the `report()`
guard sited after `iter_figures()` and its empty check, **before** the controls branch.

**Nothing here is more complex than the spec requires.** `_require_backend` is twenty
lines of which twelve are the message; the wrapper is a two-branch `if`; no new
abstraction, no configuration, no indirection. The backend vocabulary stays deliberately
split (`"mpl"` at the decorator, `"matplotlib"` on the wire) with `backend_name`
performing the one mapping, exactly as the plan's global constraint requires.

---

## Confirmed sound

- **All 27 `src/` and 11 test annotations are `backend="plotly"` and every one is
  correct.** Checked per site, not assumed: no `@figure` method in the six annotated
  `src/` files returns a matplotlib figure. `_diagnostics_plotter.py` does build
  matplotlib figures, but only in `diagnostics()` / `_diagnostics_matplotlib()`, which
  are **undecorated** — which independently confirms the spec's premise that matplotlib
  output reaches publication today only by bypassing the decorator.
- **The one bare `@figure` in the repo** (`test_figure_backend.py:19`) is deliberate,
  carries a three-line comment saying so, and sits inside
  `pytest.raises(TypeError, match="backend")`. M1 proves it is load-bearing: giving
  `backend` a default makes that test the only failure in the suite.
- **The delegation changed no behaviour.** The old `FigureAdapter._is_plotly` was
  `module.startswith("plotly.") and __name__ == "Figure"` — the identical conjunction
  `figure_backend_of` now performs. `test_output_adapter.py` is green at baseline, which
  is the proof the wire format is untouched.
- **`FigureSpec` name collision is harmless.** The plotting-ABC class has exactly one
  construction site (`_pht_plot.py:250`, keyword form) plus one `dataclasses.replace`
  at `plotting/_image_plots.py:24`, which carries the new required field forward
  automatically. The ~36 other constructions in the tree are the unrelated GUI
  scatter-tab class. Worth noting for future required fields: a `replace()` call is
  invisible to any search for construction sites.
- **The stdlib-only contract holds.** Both theme imports are inside the wrapper body;
  `figure_backend_of` identifies by module string precisely so `_output.py` needs no
  plotly or matplotlib import. `test_imports.py` green at baseline.
- **`PlotDiagnostics` still routes through the wrapper.** Its methods are undecorated
  and delegate to `DiagnosticsPlotter`'s decorated ones, so the backend check and
  theming happen on the inner call. No gap.

---

## Pre-existing observations

Reported separately, as asked. None is introduced by this cluster and none blocks it.

1. **`_diagnostics_plotter.py:8`** carries a module-level
   `from matplotlib.gridspec import GridSpec` — a heavy import at module scope in a
   package the laziness gotcha governs. Pre-existing; not in this diff.
2. **`_primary_spec()` (`_pht_plot.py:396-398`)** silently returns `primaries[0]` when
   several specs declare `primary=True`. It raises a clear error for *zero* primaries
   among multiple specs, so the silence on the over-specified case is asymmetric.
3. **`_compose_control_free_figure` would call `make_subplots(rows=0, ...)`** on an
   empty spec list. Unreachable via `report()` (which raises first) but reachable by a
   subclass calling it directly.
4. **Two `report()` overrides bypass the new guard** —
   `correction/_color_correction/_color_correction_report.py:266` and
   `grid/_grid_fit_report.py:114`. Both are all-Plotly today, and spec §1 names
   overriding `report()` as the sanctioned escape hatch, so this is by design rather
   than a defect. Flagged because a future matplotlib provider that overrides
   `report()` gets no guard and no warning that it is opting out of one.
5. **Line length is not enforced.** `pyproject.toml` sets `line-length = 79` but
   declares no `[tool.ruff.lint] select`, so ruff runs its default `E4`/`E7`/`E9`/`F`
   set and **E501 is not in it**. The four lines this cluster pushed past 79 are not
   lint failures, and `935c66f0`'s line-length rationale for not reflowing the
   multi-line decorators is cosmetic rather than a constraint. Noted only so nobody
   "fixes" it on the belief that CI cares.
