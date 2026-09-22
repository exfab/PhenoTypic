# Plan review — Explicit figure backends and non-silent plot failures

- **Date:** 2026-09-20
- **Reviewed:** `docs/superpowers/plans/2026-09-20-figure-backend-routing/plan.md` (2,239 lines, 14 tasks, 9 clusters)
- **Against:** `docs/superpowers/specs/2026-09-20-figure-backend-routing/design.md`, `DEFERRED.md`
- **Worktree:** `.claude/worktrees/figure-backend-routing` @ `181c113c` (branch `fix/figure-backend-routing`)
- **Reviewer:** analysis only; no plan, spec or source file was edited.

---

## Verdict

**Not safe to execute as written.** The plan is unusually careful — the counts are
right, nearly every `file:line` resolves, the dependency DAG matches the real file
overlaps, and the declined-parallelism reasoning holds. But it has one defect that
defeats its own headline goal and several that make its gates lie.

The decisive one is **B1**: the single-page image path in `_coordinator.py:360-380`
does not call `publish_plot_output` at all. It writes a PNG directly. Every task in
the plan that produces HTML lives inside `_writer.py`, which that path never
reaches — so after all fourteen tasks, a `PlotImage` binding returning one Plotly
figure (the common case, and the case the spec's own example uses) still publishes
**PNG only**, still fails on a machine without Chrome, and still leaves an empty
directory. The spec's central claim — "a machine without Chrome is not a machine
that cannot publish plots" — would remain false for image plots.

The second structural one is **B6**: Task 3 puts the `report()` matplotlib guard
inside `_compose_control_free_figure`, and a provider that declares any control
never reaches that method. Measured with a spy, not inferred. So a spec
requirement ships unimplemented on one of its two paths, and neither of the plan's
two `report()` tests declares a control, so nothing would catch it.

Four further blockers make cluster gates unreliable rather than wrong-output:
a pre-existing test the rewrite breaks with no task assigned (**B2**), an F4
regression test that passes on the unfixed code (**B3**), a sweep verification grep
that reports failure on a correct sweep for 19 of 27 sites (**B4**), and a
post-implementation task pointed at a file that was missing from this worktree
(**B5** — resolved mid-review when the files were copied in, but they remain
uncommitted).

Twelve non-blocking items follow. Answers to the nine questions you asked are
folded into the sections below and summarised in the index at the end.

**What is measured vs. read.** B1, B2, B4, B5 and the citation checks are read
from source and confirmed by grep; B6, the double-close question, the monkeypatch
reach, the theming assertions and the probe-cost numbers are measured by two
probes whose output is quoted inline. One item (B3) is settled by a user decision
rather than by evidence, and is written against that decision.

---

## Blocking

### B1 — Single-page image plots never reach the HTML writer at all

`PlotCoordinator._publish_image_value` has two branches:

```
src/phenotypic/plotting/_pipeline/_coordinator.py:360
        if len(output.pages) != 1 or output.pages[0].key != "default":
            publish_plot_output(...)          # :361-368  multi-page -> the writer
            return
        self._require_publication()           # :370
        destination = base / f"{output_stem}.png"          # :371
        ...
        FigureAdapter.save_png(output.pages[0].figure, temporary)   # :375
```

and `normalize_plot_output` assigns exactly the key that selects the second branch:

```
src/phenotypic/plotting/_pipeline/_output.py:16
    return PlotOutput(pages=(PlotPage(key="default", figure=value),))
```

So a `PlotImage` whose `inspect()` returns a bare `go.Figure` — which is what
every one of the 27 annotated sites does, and what Task 10's own `_ObjectCount`
does — takes the `:370-380` path: no manifest, no `renderers`, no `failed`, no
HTML, and `save_png` raising `ChromeNotFoundError` on a Chrome-less machine.
Confirmed by the existing test at `tests/unit/plotting/test_coordinator.py:96`,
which globs `*.png` directly under `plots/image/dataset/` with no per-page
subdirectory.

Task 9 Step 6 touches this function, but only to *"add `plots_base=self._plots_base`
to the `publish_plot_output(...)` call"* — which patches the multi-page branch only.
No task in the plan converts `:370-380`.

Consequences:

- The spec's §2 outcome is not delivered for image plots, which are the lifecycle
  that produces the empty directories F2 describes.
- Task 10 (`tests/integration/plotting/test_publication_end_to_end.py`) asserts
  `pages = list((plots / "_ObjectCount" / "ds-1").glob("plate_01-*.html"))` and
  `len(pages) == 1`. That test **fails**, and it fails only at cluster C6, after
  C1–C5 have been built and gated green. It is also run with `strict=True`, so on
  a Chrome-less machine the `save_png` failure re-raises before the assertion.
- Spec §4 explicitly documents this layout as
  `plots/<id>/<dataset>/<stem>-<hash>.{html,png}`, so this is a spec requirement
  with no task, not a design choice.

**This path was not overlooked — it was identified and then dropped.** The audit the
spec cites as its Origin singles it out by line number. Its E4
(`claim-verification.md:164-188`) corrects the artifact's "publishes *nothing*"
claim with:

> True for the single-page image path, where the write is `FigureAdapter.save_png`
> → `os.replace` inline (`_coordinator.py:370-380`) and the whole thing is
> swallowed by `emit_image`'s handler. **Not** true elsewhere.

and its E5 (`:193-213`) traces the `<stem>-<hash>` filename through
`_image_output_stem` and pins it to
`tests/unit/plotting/test_coordinator.py:98-100`. So the one path where the
spec's motivating symptom is *literally* true — nothing on disk, run green — is the
one path the implementation plan never reaches. That inversion is the strongest
argument for fixing it before dispatch rather than letting C6 find it.

**Correction.** Three shapes are available. I recommend the third, and the first
two are worth stating because the obvious one is a trap.

**(a) Delete the short-circuit and route everything through `publish_plot_output`.
Do not do this.** It looks like the clean fix and it carries two costs that are not
visible from the call site:

- **It serialises a whole dataset on one lock.** `publish_plot_output` acquires
  `exclusive_path_lock(directory / ".publication.lock")` (`_writer.py:84`). On the
  flat path, `directory` is `plots/<id>/<dataset>/` — *shared by every image in the
  dataset*. Today that path takes no directory lock at all (only the cheap
  `_require_publication` predicate at `:370`). Routing through the writer would put
  all 1,536 images of a plate run behind a single interprocess lock, across
  parallel local workers and SLURM array tasks alike. That is a throughput
  regression this change has no reason to take.
- **It renames every file and collides them.** `publish_plot_output` derives the
  filename from `page.label or page.key` (`_writer.py:109-111`), and a normalized
  bare figure has `key="default"` and `label=None`. Every image in a dataset would
  therefore want `default.png`, and the existing collision-suffix loop at
  `:117-124` only disambiguates *within one call* — across calls it would just
  overwrite. The current `<stem>-<hash>` naming from `_image_output_stem`
  (`_coordinator.py:393-402`) exists precisely to make per-image names unique and
  rerun-stable, and `tests/unit/plotting/test_coordinator.py:133-144`
  (`test_image_plot_output_name_is_stable_for_reruns`) pins it. Recovering it would
  mean a `flat_stem` parameter *plus* a manifest-suppression mode inside a function
  whose docstring says the manifest is authoritative — two new modes to buy back
  what already worked.

**(b) Teach the short-circuit both renderings inline at `:370-380`.** No lock
change, no filename change, and it is the smallest diff. The cost is that the
backend decision, the `chrome_available` gate, the bundle hoist and the
temp-then-replace dance all exist twice, and the second copy has no manifest — so
`renderers` and `failed` stay unreachable for image plots.

**(c) Extract the per-page rendering and call it from both. Recommended.** Task 5
is already rewriting exactly this logic; have it produce a module-level helper in
`_writer.py` rather than an inline loop body:

```python
def _render_page(
    page, directory, stem, *,
    png_ok: bool, plots_base: Path,
    publication_guard, commit_guard,
) -> tuple[dict[str, str], str | None]:
    """Write every rendering this page supports. Returns (files, error)."""
```

`_publish_plot_output_locked` calls it per page with `stem` from the collision
loop; `_publish_image_value` calls it once with `stem=output_stem` and keeps its
own `_require_publication`, its own flat directory and its own absence of a lock.
One definition of "what files does this page produce", no lock regression, no
filename change, and no manifest semantics to invent. `renderers`/`failed` remain
a property of the manifest-writing path only, which matches spec §4 — that section
documents flat `<stem>-<hash>.{html,png}` files and never a flat manifest.

Under (c), the `.failures.jsonl` write (S3) becomes the *only* durable record for
the image lifecycle, which makes S3 a requirement rather than a nicety.

Whichever shape is chosen, the fix belongs in **C5** — it edits `_coordinator.py`,
which C5 already owns — and must land before C6, which is otherwise the gate that
discovers it.

### B2 — `test_concurrent_plot_publications_do_not_mix_generations` breaks, with no task to fix it

`tests/unit/plotting/test_output_adapter.py:124-172` drives `publish_plot_output`
with a local `_FakeFigure` class and monkeypatches three `FigureAdapter`
staticmethods:

```
tests/unit/plotting/test_output_adapter.py:133-137
    monkeypatch.setattr(FigureAdapter, "save_png", save_png)
    monkeypatch.setattr(FigureAdapter, "backend_name", lambda _figure: "fake")
    monkeypatch.setattr(FigureAdapter, "close", lambda _figure: None)
```

Task 5's rewrite stops calling `FigureAdapter.backend_name` in the page loop and
calls `figure_backend_of(page.figure)` instead — a module-level function the test
does not patch. `figure_backend_of` requires `type(figure).__name__ == "Figure"`;
`_FakeFigure` fails that, so every page is classified `None`, diverted to `failed`,
and `continue`d. Measured against the predicate as Task 1 defines it:
`[Q11] backend_of(_FakeFigure) -> None`. `manifest["pages"]` becomes `[]`, and the
test dies at `generations.pop()` with `KeyError`, before it ever reaches the
`page["file"]` read at `:169`.

The plan accounts for exactly two pre-existing failures in this file — Task 5 Step 5
says *"Pre-existing tests reading `manifest["pages"][…]["file"]` (`:80`, `:106`) FAIL —
Task 6 updates them"* — and Task 6 Step 1 lists only those two line numbers. `:169`
is not mentioned, and the failure at `:169` is not a `"file"` → `"files"` rename.

This matters beyond the one test: it is the only concurrency guard on the writer,
and C3's gate is "run the cluster's own tests". An agent seeing a red
`test_output_adapter.py` that the plan *told it to expect* will plausibly wave it
through.

**Correction.** Name this test in Task 6 Step 1 and give it a concrete fix. Two
options, and the choice is a real design decision the plan should make explicitly:

1. Give `_FakeFigure` the name `Figure` and a `__module__` of `"plotly.fake"` so it
   is classified as Plotly — but then it must also survive `save_html`, so the test
   must patch `save_html` too.
2. Patch the writer's view of the predicate:
   `monkeypatch.setattr(_writer, "figure_backend_of", lambda _f: "mpl")`, which
   keeps the test PNG-only and unchanged in spirit. This requires Task 5 to import
   `figure_backend_of` **at module level** in `_writer.py`, not inside the function
   — see S4.

Option 2 is smaller and preserves what the test is actually about.

### B3 — The F4 regression test passes on the unfixed code

Task 9 Step 1's `test_emit_qc_prelude_failure_does_not_escape_or_misname` patches
`coordinator_module.MeasurementInput.__init__` to raise, and Step 2 predicts
*"FAIL — … `UnboundLocalError` escapes `emit_qc`"*. It does not. `MeasurementInput()`
is called at `_coordinator.py:240`, which is **after** the `binding` assignment at
`:239`:

```
src/phenotypic/plotting/_pipeline/_coordinator.py:239
                binding = configured.model_copy(update={"plot": plot})
                input_ref = binding.input or MeasurementInput()
```

So on today's code `binding` is already bound when the handler at `:259-263` reads
`binding.id`. The test is green before the fix and green after it — it proves
nothing. F4's actual window is the prelude at `:227-238`, which the test never
enters.

The test's post-fix assertion was also the wrong shape for the fix the plan
described, because the spec contradicted itself: §3 said the prelude failure is a
programming error that should propagate, §5's test row said *"remaining QC bindings
still emit"*. **That contradiction has since been resolved by the user in favour of
§5** — see the decision below. What follows is the correction against the decided
shape.

#### The decided shape

`binding = None` before the `try`, prelude **inside** the `try`, handler guarded,
loop continues:

```python
for configured in self._pipeline.get_plots():
    binding = None
    try:
        ...                                   # prelude, INSIDE the try
        binding = configured.model_copy(update={"plot": plot})
        ...
    except Exception as exc:
        name = binding.id if binding is not None else configured.id
        self._record_failure_by_name(name, exc, lifecycle="qc")
```

Note this needs a second recording entry point. Task 9 Step 3's `_record_failure`
takes a `binding` and reads both `binding.id` and `type(binding.plot).__name__`;
in the `binding is None` case neither is available, and `configured.plot` may be
the wrong object (it is the pre-`model_copy` plot, which for a QC ref is not the
one that would have been emitted). Add `_record_failure_by_name(name, error, *,
lifecycle, …)` and have `_record_failure` delegate to it, rather than passing a
half-built binding. `plot_class` for the unbound case should be
`type(configured.plot).__name__` with the caveat that it may name the recipe entry
rather than the check — acceptable, and better than omitting it.

#### The test must still be able to fail

The injection point is the only way into the prelude, and it survives the change of
shape:

```python
class _RaisingModules(dict):
    def get(self, key, default=None):
        raise RuntimeError("prelude exploded")

coordinator.emit_qc(
    pd.DataFrame(), registry,
    successful_modules=_RaisingModules({"anything": object()}),
)
```

It must be non-empty — `modules = successful_modules or {}` at `:223` discards a
falsy mapping and the injection would never be reached.

**The post-fix assertions, concretely.** Under the §5 shape there is no
`pytest.raises`; the test has two obligations and the second is the one that
matters:

```python
# 1. The loop continued: a LATER binding still emitted.
#    Configure two QC plots and assert the second one's output exists.
assert (plots_dir(tmp_path) / "_QcSecond" / "manifest.json").is_file()

# 2. The failure was recorded against THIS iteration's identity, not a
#    previous binding's. This is the half of F4 that is silent rather than
#    loud, and it is the half most easily written as a tautology.
entries = [json.loads(l) for l in (plots_dir(tmp_path) / ".failures.jsonl")
           .read_text().splitlines()]
assert [e["binding_id"] for e in entries] == ["_QcFirst"]
```

**Why obligation 2 is easy to get wrong.** A single-binding test satisfies it
trivially — with one plot in the pipeline there is no "previous binding" to be
misnamed as, so `binding_id == "_QcFirst"` passes whether the handler reads
`configured.id` or a stale `binding.id`. The test is only meaningful with **at
least two QC bindings where the *second* iteration is the one that raises**, so
that a stale `binding` would name the first. Concretely: make `_RaisingModules.get`
raise only on its second call. Without that, the misnaming guard is a tautology and
F4's quieter half stays unguarded.

On today's code the same test fails loudly — `UnboundLocalError` raised from inside
the handler on the first iteration — so it is a genuine red-to-green.

#### Recorded consequence of choosing §5

The user's choice is made and I am not relitigating it, but it has a cost that
should be on the record rather than rediscovered later.

**What §5 costs:** the prelude at `:227-238` is dict and attribute access —
`configured.ref`, `configured.id`, `modules.get`, `module.check`, `isinstance`, and
the `assert ref.key is not None` at `:231`. A failure in any of those is a
programming error or a violated invariant, not a plot failure. Keeping them inside
the `try` means such a failure is caught, written to `.failures.jsonl` as though a
figure had misbehaved, and the run stays green. Specifically, the `assert` at `:231`
encodes the invariant *"a `qc`-slot ref always carries a key"*; if that is ever
violated, §5's shape records it as a plot failure and continues, where §3's shape
would have surfaced it.

**What §5 buys**, and why it is a defensible trade: `emit_qc` is the one aggregate
path documented as best-effort, it iterates *every* configured plot rather than a
pre-filtered list, and a malformed binding anywhere in that list currently kills
every plot after it. §5 keeps one bad binding from costing the other nine their
output — which is the same argument §3 makes for per-figure softness, applied one
level up. The disagreement was never about whether failures should be visible; both
shapes make this one visible, and `.failures.jsonl` is what makes §5's version
non-silent.

**Follow-up worth filing.** Yes — a `DEFERRED.md` item is warranted, narrowly
scoped: *"distinguish invariant violations from plot failures in `emit_qc`"*, e.g.
by letting `_record_failure_by_name` re-raise for a chosen set (`AssertionError`,
`AttributeError` on the binding itself) while swallowing the rest. That is a real
design question, it is not what this change is about, and filing it is how §5's
cost stops being invisible. Reconsider it if `.failures.jsonl` ever records a
prelude-shaped failure in real use — which is exactly the evidence test
`DEFERRED.md` already applies to the `FigureAdapter` sniffing question.

### B4 — Task 11's completion grep reports failure on a correct sweep

Task 11 Step 4:

> Run: `grep -rn "@figure(" --include=*.py src/ | grep -v "_pht_plot.py" | grep -vc "backend="`
> Expected: `0`.

**19 of the 27 sites are multi-line decorations** whose `@figure(` line carries no
arguments at all:

| File | multi-line | single-line |
|---|---|---|
| `_core/_image_parts/plot_accessor/_diagnostics_plotter.py` | 12 | 0 |
| `measure/_measure_orientation_zones.py` | 3 | 0 |
| `correction/_color_correction/_color_correction_report.py` | 2 | 2 |
| `measure/_measure_symzones.py` | 1 | 0 |
| `_core/_image_parts/plot_accessor/_detect_modes_plotter.py` | 1 | 0 |
| `grid/_grid_fit_report.py` | 0 | 6 |
| **Total** | **19** | **8** |

e.g. `src/phenotypic/measure/_measure_symzones.py:515-519`:

```python
    @figure(
            title="Symmetric-radius overlay",
            primary=True,
            controls={"base_layer": BASE_LAYER},
    )
```

After a perfect sweep this grep prints **19**, not 0. Step 2's worked example
(`@figure(title="Noise profile", backend="plotly", section="noise")`) is the
single-line form, which describes 8 of 27 sites. A Sonnet agent told "Expected: 0"
will either report the sweep incomplete, or collapse 19 decorators onto one line to
satisfy the grep — churn in six files, and the `measure/` ones would exceed the
79-character `line-length`.

**Correction.** Replace Step 4's grep with a form that reads the whole decorator,
e.g.

```bash
grep -rn -A6 "@figure(" --include=*.py src/ | grep -v "_pht_plot.py" | grep -c 'backend='
```

Expected: `27`. Add to Step 2: *"most sites are multi-line; add `backend="plotly",`
as its own line after the `title=` line. Do not reflow the decorator."* Note that
Step 3 (import all six modules) is the *real* guard — a missed site raises
`TypeError: figure() missing 1 required keyword-only argument: 'backend'` naming
the file and line — so Step 4 is a convenience check that currently produces a
false alarm.

Task 12's equivalent grep is **correct**: all 11 `tests/` sites are single-line
(9 in `test_pht_plot.py`, 2 in `test_notebook_adapter.py`), so
`grep -rn "@figure(" tests/ | grep -vc "backend="` returns 11 today and 0 after,
with `test_figure_backend.py`'s own sites already carrying `backend=` on the
`@figure(` line.

### B5 — Task 13's audit was missing from the worktree *(resolved during review; still uncommitted)*

**Status update.** Both directories were copied into the worktree at 20:07, part
way through this review, and I have since read the audit. The blocking half is
resolved — C9 can now open the file. What remains is that both are **still
untracked**:

```
?? docs/superpowers/artifacts/2026-09-20-pipeline-figure-storage/
?? docs/superpowers/reports/2026-09-20-pipeline-figure-storage/
```

so they are not on the branch and would not survive a fresh worktree or clone.
Commit them. The original finding follows, for the record.

Reading the audit also **strengthened B1** — see the E4 quotation there — and
confirmed the binding-id rule I had independently derived: its E2
(`claim-verification.md:95-133`) reaches the same conclusion from the same lines
(`_bindings.py:165-171`), including the `model`-slot case that made the old doc
text look right.

---

Task 13 Step 1: *"Read the audit at
`docs/superpowers/reports/2026-09-20-pipeline-figure-storage/claim-verification.md`
for the evidence behind the binding-id rule."* The spec's **Origin** block cites the
same file plus
`docs/superpowers/artifacts/2026-09-20-pipeline-figure-storage/`.

Neither existed here when the review began:

```
$ ls -d docs/superpowers/*/2026-09-20-pipeline-figure-storage
ls: cannot access '...': No such file or directory
```

Both were untracked directories in the main checkout only (they appear as `??`
entries in this session's opening `git status`). A worktree checked out from
`origin/main` could not see them, so C9 would have blocked on a file it could not
open — and every claim the plan attributes to "the audit" was unsourced from here.
I re-derived the two load-bearing ones independently before the files arrived, and
both held.

**Correction.** Commit the artifact and report directories to the branch. Until
they are committed, the same failure recurs in the next worktree. (Excising the
citation from Task 13 Step 1 is the weaker alternative — now that I have read the
audit, it is genuinely the best statement of the three-case binding-id rule and
Task 13 is better off with it than without.)

### B6 — Task 3's guard is unreachable on the notebook-controls path

Spec §1: *"if any visible spec declares `backend="mpl"`, `PhtPlot.report()` raises
`TypeError`"*. Task 3 implements that by inserting the guard into
`_compose_control_free_figure`. But `report()` branches away from that method
whenever any spec declares a control:

```
src/phenotypic/abc_/plotting/_pht_plot.py:427-433
        if any(spec.controls for spec in specs):
            from phenotypic.sdk_.viz.notebook._adapter import (
                build_notebook_dashboard,
            )
            return build_notebook_dashboard(self, subject)
        return self._compose_control_free_figure(subject)
```

**Measured, not inferred.** A probe installed a spy on
`_compose_control_free_figure` and called `report()` on three providers:

```
[Q9d which providers reach _compose_control_free_figure]
    -> composer reached by: ['_OneMpl', '_OneMpl:AttributeError',
                             '_TwoMpl', '_TwoMpl:AttributeError']
```

The controls-declaring provider is **absent from that list** — it never reaches the
composer, so Task 3's guard never fires for it. `[Q9c]` shows what happens instead:
`AttributeError: 'Figure' object has no attribute 'layout'` — the raw F1 failure,
naming neither the decorator nor the backend, which is the exact outcome §1 exists
to replace.

So a spec requirement ships with no implementation on one of its two paths, and
nothing in Tasks 3–14 would catch it: the plan's two `report()` tests (`_AllMpl`,
`_Mixed`) declare no controls, so both take the composer path.

**Correction.** Move the guard from `_compose_control_free_figure` into `report()`,
placed after the `specs` fetch at `:422` and **before** the `any(spec.controls …)`
branch at `:427`:

```python
        specs = self.iter_figures()
        if not specs:
            raise RuntimeError(...)
        mpl_specs = [spec.name for spec in specs if spec.backend == "mpl"]
        if mpl_specs:
            raise TypeError(
                f"{type(self).__name__}.report(): cannot compose matplotlib "
                f"figures ({', '.join(sorted(mpl_specs))}). ..."
            )
        if any(spec.controls for spec in specs):
            ...
```

Add a third test to Task 3 Step 1 — an `mpl` provider **with** a `Control` — or the
gap simply reopens.

**On whether it should be in both places: no.** `_compose_control_free_figure` has
exactly one caller in the entire tree (`_pht_plot.py:433`), so a second copy is
dead defence. The one case neither placement covers is a subclass that overrides
`report()` entirely — and that is fine, because spec §1 names a `report()` override
as the *sanctioned escape hatch* for matplotlib providers. Guarding the base
implementation is precisely the right scope.

---

## Should fix

### S1 — `plots_base` is added to two signatures but never threaded

Task 5 Step 4 says to add `plots_base: Path | None = None` to *both*
`publish_plot_output` and `_publish_plot_output_locked`. It does not say to forward
it in the delegating call at `_writer.py:86-93`. An implementer following the text
literally gets a `plots_base` that is always `None` inside the locked function.
Task 5's own `test_the_html_references_the_hoisted_bundle` catches it, so this
costs a debugging cycle rather than shipping — but it is one sentence to prevent.

### S2 — A page that publishes one of two renderings loses the other's failure silently

In Task 5's loop, `page_error` is set by the HTML handler and then **overwritten**
by the PNG handler, and is consulted only in the `if not files:` branch:

```python
        if not files:
            failed.append({... "error": page_error or "no renderer produced a file"})
```

So for a Plotly page where HTML succeeds and PNG fails (Chrome present, per-figure
raster error), or where PNG succeeds and HTML fails, the failure is logged at
`WARNING` and then **discarded** — absent from `"failed"`, absent from
`.failures.jsonl`, and invisible in `"renderers"` (which reports the process-wide
capability, not this page's outcome). That is precisely the "best-effort means
silent" pattern §3 exists to remove, reintroduced one level down.

**Correction.** Accumulate rather than overwrite (`page_errors: list[str]`) and
record partial failures even when `files` is non-empty — either as a per-page
`"partial"` key or as a `.failures.jsonl` line.

### S3 — The writer never writes `.failures.jsonl`, which the spec requires

Spec §3: *"Written from all five `except` blocks in `_coordinator.py` … **plus the
per-page failures in `_writer.py:140`**, through one helper."* Task 5 and Task 6
produce only the manifest `"failed"` array; no task calls `record_plot_failure`
from `_writer.py`. Task 8 creates the helper and Task 9 wires the coordinator only.

This is reachable now that Task 5 gives the writer a `plots_base`. It is a spec
requirement with no task.

### S4 — Two monkeypatch targets, one safe, one that would silently test nothing

You asked specifically. The answer differs per function:

- **`chrome_available` — the patch works.** Task 5's code does
  `from ._backends import (chrome_available, …)` **inside**
  `_publish_plot_output_locked`. A function-scope `from X import y` re-executes on
  every call and re-reads the module attribute, so
  `monkeypatch.setattr(_backends, "chrome_available", lambda: False)` is observed.
  Same for Task 7's `preflight_plot_backends`, which is defined *in* `_backends.py`
  and resolves `chrome_available` as a module global. Both tests are genuine.
- **`figure_backend_of` — also function-scope in Task 5, and that is the problem
  for B2.** Because it is imported inside the function, a test cannot patch it on
  `_writer` (there is no module attribute to patch). Moving it to a module-level
  `from phenotypic.abc_.plotting import figure_backend_of` in `_writer.py` is safe
  — `_writer.py` already imports `PlotOutput` from that package at `:18` — and it
  is what makes B2's option 2 available.

No monkeypatch in the plan silently passes while testing nothing. The risk you
were worried about does not materialise.

### S5 — `test_a_plotly_figure_is_themed_after_construction` is a tautology

Task 2 Step 1:

```python
    assert "phenotypic" in str(fig.layout.template.layout.font.family or "") or \
        fig.layout.template is not None
```

`apply_theme` sets `fig.layout.template = f"plotly+{PHENOTYPIC_TEMPLATE_NAME}"`
(`sdk_/viz/figures/_theme.py:227`), whose rendered font family is the DESIGN.md
stack, not the string `"phenotypic"` — so the first clause is False. The second
clause is true for *any* `go.Figure`, themed or not.

**Measured, not inferred.** A probe run in this worktree:

```
[Q7 layout.template on an UNthemed figure] -> untheme template is None? False;
    themed template is None? False; untheme font.family=None
```

So the assertion passes on a wrapper that skips theming entirely. This is the
"success message identical to the no-op message" shape, and it sits on the one
test in Task 2 that is supposed to prove the Plotly branch does its job.

**Correction, with the constant checked.** An unthemed figure's
`template.layout.font.family` is `None` and a themed one's is a real stack, so that
attribute is the discriminator. But **it is `FONT_FAMILY_MONO`, not `FONT_FAMILY`** —
which is why it was worth measuring rather than naming from memory:

```
[Q7b] FONT_FAMILY='\'IBM Plex Sans\', -apple-system, ...';
      raw.template.layout.font.family=None;
      themed.template.layout.font.family='\'JetBrains Mono\', ui-monospace, ...'
```

That is deliberate, not a bug: `_theme.py:171` sets the template's **base**
`layout.font` to `FONT_FAMILY_MONO` per DESIGN.md "02" (all numeric data in mono),
and applies `FONT_FAMILY` to titles, axis titles and the legend at `:174`, `:180`,
`:187`, `:189`. An assertion written against `FONT_FAMILY` would fail on a
correctly themed figure.

Replace the whole `or` expression with:

```python
from phenotypic.sdk_.viz.figures import FONT_FAMILY_MONO

fig = _Plotly().inspect(object())
assert fig.layout.template.layout.font.family == FONT_FAMILY_MONO
```

Or, if coupling the test to a design token is unwanted, assert the template
identity instead — `fig.layout.template` resolved from
`"plotly+phenotypic"` — which is what `apply_theme` actually sets. Either
discriminates; `is not None` does not.

### S6 — The two new path helpers are dead on arrival

Tasks 4 and 8 add `plotlyjs_bundle_path(output_dir)` and
`plot_failures_jsonl_path(output_dir)` to `sdk_/_io_constants.py` and export both
from `sdk_/__init__.py` — and then **nothing calls either**. Both consumers take a
`plots_base` and hand-join the constant: `plots_base / PLOTLYJS_BUNDLE`,
`plots_base / PLOT_FAILURES_JSONL`.

That is not an oversight in the plan so much as an unavoidable consequence: the
coordinator's `_plots_base` is not always `plots_dir(output_dir)` — the GUI passes
`layout.plots_dir` from a portable bundle (`_gui/_plot_refresh.py:132`,
`_io_constants.py:2765`), which has no `deliverables/` segment. So an
`output_dir`-keyed helper genuinely cannot serve these call sites.

The spec justifies the helpers with *"The project rule against hand-joined output
names is not negotiable for a new artifact"* — but shipping an uncalled helper
satisfies the letter and not the rule. **Pick one:** either key both helpers on
`plots_base` (`plotlyjs_bundle_path(plots_base)`) and actually call them, or drop
them and keep only the `Final[str]` constants. Answering your question 9: this is
the clearest piece of unnecessary surface in the change.

### S7 — Task 3's Step 2 expectation is wrong for `_AllMpl` *(the larger half of this item was promoted to B6)*

The guard-placement half of this finding is now **B6** — probe `[Q9d]` confirmed the
bypass. What remains here is smaller and does not block.

**Step 2's stated expectation is wrong for `_AllMpl`.** `_compose_control_free_figure`
short-circuits a single spec at `:451-452` (`return self._render_spec(specs[0], subject)`)
without touching `make_subplots`. In the post-Task-2 world — which is the world
Task 3's test runs in — a `backend="mpl"` method returns its figure unthemed, so
`_AllMpl().report()` returns the matplotlib figure with no error, and the test
fails with `DID NOT RAISE` rather than the predicted `AttributeError`. For
`_Mixed` (two specs) the prediction is closer but still misnamed: the composer
reaches `for trace in rendered.data`, so the attribute is `data`, not `layout`.

**A correction to my own earlier phrasing.** I previously wrote that
`_AllMpl().report()` "works today". It does not — probe `[Q9]` shows it raising
`AttributeError: 'Figure' object has no attribute 'layout'`, which is F1 itself
firing in the decorator at `_pht_plot.py:192` before the composer is ever reached.
The path only becomes working *after* Task 2 removes the unconditional
`apply_theme`. The consequence is unchanged — Task 3's guard closes a path that
Task 2 opens two tasks earlier — but it is worth stating accurately, because
"Task 3 removes a working path" and "Task 3 closes a path Task 2 just created" are
different claims and only the second is true. The plan's justification
("it would fail obscurely inside Plotly") holds for the ≥2-figure case only; the
single-figure narrowing is a deliberate choice and should be stated as one.

### S8 — `strict=True` citations are 61 lines off

Spec F3 and the plan's File-Structure table both cite
`src/phenotypic/_cli/_cli_staged_workers.py:526`, and Task 9 Step 7 says *"at ~`:531`"*.
The actual call is:

```
src/phenotypic/_cli/_cli_staged_workers.py:583      ).emit_image(
src/phenotypic/_cli/_cli_staged_workers.py:587          strict=True,
```

`:526` is a `# NOTE (ledger FLOW-21):` comment. The companion citations
`_cli_process_single.py:348,450` are also off by two — the calls are at `:350` and
`:452`. The instruction is unambiguous enough to execute (there is exactly one
`strict=True` in the file), but a cluster gate that spot-checks citations will
flag it.

### S9 — Task 7 Step 5 runs a test file no task creates, and its command can fail on exit code 5

Task 7's **Files** block lists *"Test: … `tests/unit/cli/test_cli_validation.py`
(append; create if absent)"* — but none of Task 7's six steps writes that file; Step 1
appends only to `tests/unit/plotting/test_backends.py`. Step 5 then runs

```
uv run pytest tests/unit/plotting/test_backends.py -v && uv run pytest tests/unit/cli/ -k valid -v
```

`tests/unit/cli/test_cli_validation.py` does not exist in the tree, and `-k valid`
over `tests/unit/cli/` may select nothing, which pytest reports as exit code 5
("no tests ran"). With `&&` chaining, the agent sees a non-zero exit and a green
first suite. Either drop the second half or write the CLI test the Files block
promises.

### S10 — Spec §2's "workers probe lazily" is not what the plan implements

Spec §2 describes two strategies: eager in the submitting process, and in workers
*"the first PNG write attempt **is** the probe"*. The plan instead calls the memoised
`chrome_available()` eagerly at the top of every `_publish_plot_output_locked`.

For the Chrome-absent case this is behaviourally identical (one probe per process,
then free — and see S12: that probe is far cheaper than the spec implies). For the
Chrome-**present** case it moves an unmeasured
first-launch cost onto the first publish rather than onto the first figure that was
going to be rendered anyway — which Task 4 Step 6 is designed to measure and gate
on, so the risk is handled. The simplification is fine; the plan should just say it
is deviating, and the spec's §5 row *"Worker-side lazy probe: after the first Chrome
failure, later pages skip PNG without re-attempting"* should be mapped onto
`test_the_probe_is_memoised` rather than left as an unmapped test row.

Relatedly, the spec's once-per-process **announcement** (§2, the `WARNING` naming
the binding ids) is emitted only from `validate_pipeline` in the submitting
process. Workers log nothing above `DEBUG` (`_backends.chrome_available` uses
`logger.debug`). For a local `--njobs` run that never passes through
`validate_pipeline`, the user gets no announcement at all and only the manifest
`renderers` key records it. Acceptable given §3's durability argument, but worth a
sentence in the plan.

### S11 — A blocked publication can now leave an orphan HTML page

In the original loop the `PlotPublicationBlocked` handler unlinks the temp and
re-raises, so a blocked page leaves nothing on disk. In Task 5's loop, if HTML is
committed and the *PNG* write is then blocked by the `publication_guard`, the HTML
file has already been `os.replace`d into place, the exception propagates out of
`_publish_plot_output_locked`, and no manifest is written — leaving an HTML page in
a directory no manifest lists, produced by a generation the guard rejected.

This is a narrow GUI-only path (`publication_guard` is `None` for every CLI caller),
and `tests/gui/results_viewer/test_mutation_guard.py:553` exercises only the
matplotlib/PNG case so it will still see `checks == 3` and pass. But the writer's
whole contract is *"the manifest is replaced last and therefore lists only durable
page files"* (`_writer.py:65-66`), and this breaks the converse. Worth a
`files`-scoped cleanup in the `PlotPublicationBlocked` handler.

### S12 — The spec's "0.59 s to fail" is two different quantities stated as one

Spec §2 records the Chrome probe as *"Measured: **0.59 s to fail** when Chrome is
absent"*, and F2's evidence row repeats it. Re-measured in this worktree:

| Condition | Cost to fail |
|---|---|
| cold — fresh interpreter, `plotly.io` not yet imported | **0.59 s** (the spec's figure) |
| warm — same process, plotly already imported, first attempt | **0.03 s** |
| warm — second attempt in the same process | **0.02 s** |

These are not noise around one number: the failure cost is **import-dominated**,
and once `plotly.io` is loaded the probe is effectively free. That matters because
§2 uses the 0.59 s figure to argue the probe is cheap enough to site on a hot path
— an argument the warm number makes *stronger*, not weaker, since any process that
reaches publication has already imported plotly to build the figure.

**Correction.** State both numbers and say which is which. The plan should also
stop repeating "one 0.6 s probe per process" (Task 4 Step 6's expected value, and
my own phrasing in S10) as though a worker pays it — a worker that has rendered a
figure pays ~0.02 s.

This changes no decision in the change. It is flagged because a spec that states
one measurement as *the* measurement is what a later reader quotes.

**Not in scope of this finding:** Task 4 Step 6, which measures the **success**
path and halts the run if it exceeds ~5 s. That is the genuinely unknown half —
Chrome is not installed on this cluster — and the step is correctly designed as a
gate rather than an assumption.

---

## Considered and fine

**Scope of the `_writer.py` rewrite (your question 1).** Every name the Task 5 code
uses is in scope. `used` (`:106`), `pages` (`:107`), `logger` (`:23`), `uuid`
(`:10`), `os` (`:8`), `hashlib` (`:5`), `Path` (`:11`), `Callable` (`:12`), `Any`
(`:13`), `publication_commit` and `CommitGuard` (`:16`), `PlotPublicationBlocked`
(`:28`), `safe_path_component` (`:32`), `_require_plot_publication` (`:180`,
module scope so forward reference is fine). `_atomic_write` at module scope with
`publication_guard`/`commit_guard` passed explicitly is correct.

**`_atomic_write` semantics (your question 1, second half).** It preserves both
guarantees. The `finally: temporary.unlink(missing_ok=True)` subsumes the original
per-branch `temporary.unlink(missing_ok=True)`; after a successful `os.replace` the
temp no longer exists and the unlink is a no-op. The `PlotPublicationBlocked`
re-raise is preserved by the explicit `except PlotPublicationBlocked:` in each
branch, which closes the figure and re-raises exactly as `:136-139` did. (The one
gap is S11, which is about ordering across two writes, not about `_atomic_write`.)
Writing HTML to a `.tmp`-suffixed path works — `write_html` does not infer format
from the extension; I confirmed it by inspecting a page the probe wrote to
`.Only.png.deadbeef.tmp`, which is valid HTML.

**The loop `lambda`s are not a late-binding bug (your question 2).** Confirmed by
reading, not assumed: `_atomic_write` calls `write(temporary)` as its first
statement, synchronously, before the closure can outlive the iteration. Nothing
stores the lambda. Ruff will not flag it either — `[tool.ruff]` in `pyproject.toml`
sets only `line-length` and `extend-exclude`, so the default `E4/E7/E9/F` rule set
applies and `B023` (flake8-bugbear) is not enabled.

**`FigureAdapter.close` placement (your question 3) — no path leaks a figure.**
Traced all five exits from the page body: `backend is None` → `failed` + `continue`
(and `close` on a non-matplotlib object is a no-op anyway, so this matches the old
behaviour exactly, where `backend_name`'s `TypeError` led to a `close` that did
nothing); `PlotPublicationBlocked` in either branch → explicit `close` + `raise`;
normal completion, empty `files`, and non-empty `files` all pass through the single
`FigureAdapter.close(page.figure)` before the `if not files` check. The double-close
concern is real in shape — `save_png` closes matplotlib figures in its own `finally`
(`_adapter.py:43-48`) and the loop then calls `close` again — and **measured
harmless**: `[Q1c] savefig+close+close OK, size=2397`, with the pyplot-registered
triple-close and the bare-`Figure` double-close also clean. `plt.close` on a figure
with no live manager is a no-op, so the second call costs nothing and breaks
nothing.

**The matplotlib theming test is genuine, and it is the good one.** Task 2's
`test_the_mpl_theme_is_live_while_the_figure_is_built` is the test the spec's
Background says catches treating the two themes symmetrically, and unlike its
Plotly sibling (S5) it discriminates properly. Measured:
`[Q8] inside == phenotypic_rc()['axes.prop_cycle'] -> True; before == after ->
True; before == inside -> False`. All three clauses are load-bearing: the theme is
live inside the body, the caller's global `rcParams` are restored, and the themed
value genuinely differs from the default — so a wrapper that dropped the
`rc_context` would fail it.

**Task 1's shared predicate is sound, and its vocabulary mapping is exact.**
Measured against the implementation Task 1 specifies:

```
[Q6]  [('plotly.graph_objs._figure.Figure', 'plotly'),
       ('matplotlib.figure.Figure', 'mpl'), ('matplotlib.figure.Figure', 'mpl'),
       ('builtins.object', None), ('builtins.NoneType', None),
       ('builtins.str', None), ('builtins.int', None)]
[Q6b] [('plotly', 'plotly'), ('mpl', 'matplotlib'),
       "object() -> TypeError: unsupported figure type builtins.object"]
```

Both a `plt.figure()` and a bare `matplotlib.figure.Figure()` classify as `mpl`;
every non-figure returns `None` without raising, as the docstring promises. `[Q6b]`
confirms the `mpl` → `matplotlib` mapping the manifest depends on and that the
unsupported case still raises `TypeError` with the same message text — so the
delegation genuinely does not change the wire format, which is what
`test_output_adapter.py:86` pins.

**The bundle hoist behaves exactly as the spec's table claims.** `[Q3]
get_plotlyjs() chars=4847452` matches the spec's "4,847,452 chars, verified" to the
character. `[Q2] srcs=['../plotly.min.js']` confirms the string form of
`include_plotlyjs` is emitted verbatim as the script `src` and writes nothing
beside the page, `[Q2b]` confirms it at a deeper relative path, and `[Q2c]` confirms
`write_html` to a `.tmp`-suffixed destination produces valid HTML — which
`_atomic_write` depends on and no test in the plan covers directly.

**`figure_backend_of` on a closed matplotlib figure (your question 4) is valid.**
`plt.close()` tears down the figure's canvas/manager; it does not rebind, replace,
or mutate the object's `type()`, and `output.pages` holds a live reference
throughout. `type(figure).__module__` / `__name__` are class attributes, unreachable
from anything `close` touches.

**`_pht_plot.py` laziness (your question 6).** `_require_backend`'s call-time
`from ._output import figure_backend_of` is *unnecessary* — `_output.py` is
stdlib-only (`Any, Mapping, TypeAlias` from `typing`, plus `json`/`math`/
`dataclasses`/`datetime`), so a module-level import would be equally safe and
marginally cheaper. It is not wrong, just redundant. `figure_backend_of` itself
stays stdlib-only: it reads `type(figure).__module__` and `__name__` and imports
nothing.

One correction to the plan's *attribution*, though: the Global Constraints claim
these modules are *"Guarded by `tests/unit/ci/test_deferred_imports.py`"*. That test
is table-driven over `DEFERRED_SITES`, and neither `abc_/plotting/_pht_plot.py` nor
`abc_/plotting/_output.py` appears in the table. The real guard is
`tests/unit/abc_/plotting/test_imports.py::test_plotting_subpackage_adds_no_ui_or_runtime_plotting_imports`,
which spawns a subprocess and asserts that importing `phenotypic.abc_.plotting`
pulls in no `plotly.*`, `matplotlib.*`, `dash.*`, `ipywidgets.*` **or
`phenotypic.plotting.*`**. Task 1 Step 6 happens to run that file, so the plan is
operationally covered; only the attribution is wrong. (That last clause is also why
`figure_backend_of` must live in `abc_/plotting/_output.py` and not be imported
from `_pipeline` — the spec's layering argument is enforced by a real test.)

**Task 2's red window does not break anything else (your question 7).** Nothing in
the repo imports `tests/unit/abc_/plotting/test_pht_plot.py`, so its collection
error is module-scoped: `test_imports.py` and the new `test_figure_backend.py` in
the same directory still collect and run. The two affected files
(`test_pht_plot.py`, `tests/unit/viz/test_notebook_adapter.py`) are not on any
Task 3–10 command line, so those clusters' own runs stay readable. The residual
risk is a CI push mid-sequence, which would be red from Task 2 through Task 12 —
the plan commits locally, but it does not say "do not push until C8", and it should.

**Counts and structural claims verified independently.** 27 `@figure(` under `src/`
excluding `_pht_plot.py` (whose 28th hit is the error-message f-string at `:174`),
distributed exactly as the plan's table says — 12 / 6 / 4 / 3 / 1 / 1. 11 under
`tests/` — 9 and 2. All four existing `tests/integration/` subdirectories
(`cli`, `gui`, `packaging`, `tune`) carry `__init__.py`, and `tests/integration` is
in `testpaths` (`pyproject.toml:219`), so Task 10 Step 0 is correct. I also
re-derived the two claims the plan attributes to the unreachable audit: every one
of the 27 decorated methods returns a Plotly figure (`-> go.Figure` on 24 of them;
the three in `measure/` return a locally-built `fig` from Plotly traces), and the
binding-id rule at `_bindings.py:165-171` is exactly the three cases §4 states
(`ref.key` when present, else `type(raw).__name__`).

**Citations spot-checked, and nearly all resolve exactly.** `_pht_plot.py:102-128`
(`FigureSpec`), `:131-208` (`figure`), `:174`, `:192`, `:439`;
`_theme.py:227` (`fig.layout.template = …`); `_output.py:16`;
`_writer.py:149`, `:160-177`; `_coordinator.py:226/239/261`, `:360`;
`_adapter.py:129-153`; `_io_constants.py:806` (`DIR_PLOTS`), `:1105-1107`
(`plots_dir`); `_cli_validation.py:21-54`; `_bindings.py:165-171`;
`_gui/_plot_refresh.py:132`; `_cli_interactive.py:132`;
`_cli_readme_generator.py:106`; `test_output_adapter.py:80/86/106`. Three misses:
S8's `_cli_staged_workers.py:526`, `test_pht_plot.py:97` (the control-key
`@figure(` is at `:98`), and the spec's `_image_pipeline_core.py:704` — the
`registry[("model", None)]` line is `:705`, and the real path is
`src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`, not
`src/phenotypic/_core/_image_pipeline_core.py`.

**Task 12's one-off instruction is right.** Annotating the `@figure(` inside
`pytest.raises(ValueError)` at `test_pht_plot.py:98` is correct and necessary:
Task 2 puts the `backend not in ("plotly", "mpl")` check as the first statement of
`figure()`'s body, before `declared_controls`, so a valid `backend="plotly"` falls
through to the control-key `ValueError` raised inside `decorator(fn)` — the test
still reaches its subject. Without the annotation the site would raise `TypeError`
and the test would fail for the wrong reason.

**No production code reads the plot manifest.** Verified: the only `manifest`
reference in the GUI (`_plot_refresh.py:80`, `manifest_entry=result.manifest_entry`)
is the *analysis* manifest, a different artifact. Nothing globs `plots/**/*.png`
in `src/`, nothing enumerates `plots_dir` children, and the two documentation
strings the spec names are exactly as cited. The `schema_version: 2` bump is as
cheap as the spec claims.

**The nine-cluster split and the declined parallelism hold up.** The file-level DAG
matches reality: C1 and C2 genuinely share no files; C3's two tasks both rewrite
`_publish_plot_output_locked` and cannot be split without an un-green intermediate
commit; C4 and C5 are correctly isolated as seams (C5 in particular — five handler
bodies, a `try` boundary, and a cross-file `strict=` removal); C2's pairing of
Tasks 4 and 8 is forced by their shared edits to `_io_constants.py` and
`sdk_/__init__.py`. Declining C1∥C2 for a single cluster of wall-clock against a
second worktree is the right call, and declining C7∥C8 is obviously right. One
adjustment follows from B1: whichever task fixes `_publish_image_value` belongs in
C5, not a tenth cluster — it touches `_coordinator.py`, which C5 already owns, and
it must land before C6 can pass.

**Smaller things that check out.** `ensure_plotlyjs_bundle`'s double-checked lock
(cheap `is_file()`/size check, then `exclusive_path_lock`, then re-check) is the
right shape and matches Plotly's own `"directory"` non-rewriting behaviour;
`exclusive_path_lock(lock_path)` takes exactly one positional path
(`sdk_/_file_locking.py:22`). `record_plot_failure`'s never-raise contract is
genuinely exercised by `test_recording_never_raises` — `Path.mkdir(exist_ok=True)`
on an existing *file* raises `FileExistsError`, which the bare `except Exception`
catches. `plotlyjs_src_for`'s three parametrised depths compute correctly against
`os.path.relpath`, and they match the three real layouts
(`plots/<id>/`, `plots/<id>/<ds>/`, `plots/<id>/<ds>/<stem>-<hash>/`). `FigureSpec`
is a frozen dataclass with no defaulted fields, so inserting `backend` after
`section` is safe — every construction site is keyword-only (`:194-205`).
`apply_theme` mutates and returns the same figure (`_theme.py:225-228`), so
`return apply_theme(built)` is correct. `sdk_/__init__.py` uses a plain eager
import list, so the new export instructions work as written.

---

## Unverified

1. ~~`report()` on a matplotlib provider.~~ **Resolved, and it promoted the item to
   B6.** `[Q9d]` shows the composer reached by `_OneMpl` and `_TwoMpl` and **not**
   by the controls-declaring provider. Probe 1's `NameError` was a defect in my
   probe (I imported `figure` into a sibling function's scope), not a property of
   `report()`. Side-effect worth noting: `[Q9]`/`[Q9e]` raise
   `AttributeError: 'Figure' object has no attribute 'layout'`, which is F1 itself
   firing in today's decorator — that corrected a claim I had made in S7, noted
   there as a self-correction.
2. ~~Double-`plt.close()` on a matplotlib figure.~~ **Resolved — safe.** Measured:
   `[Q1a] pyplot-created triple close OK`, `[Q1b] bare Figure double close OK`,
   `[Q1c] savefig+close+close OK, size=2397`. The third is the exact
   `save_png`-then-`FigureAdapter.close` sequence Task 5 creates, and the PNG is
   intact. `[Q1d]` also confirms `type()` survives: `matplotlib.figure.Figure`
   after close, which is what Task 6's second `figure_backend_of` call depends on
   (your question 4). Your question 3 is fully answered: no double-close hazard,
   no unclosed path.
3. ~~`go.Figure().layout.template` on an unthemed figure.~~ **Resolved, and it
   confirms the test is vacuous** — `untheme template is None? False`. Folded into
   S5 with a concrete replacement assertion.
4. ~~`ImagePipeline(ops=…, plots=…)` and the bare-plot binding id.~~ **Resolved
   statically, both fine.** `ImagePipelineCore` is a pydantic model declaring
   `ops: Dict[str, …]` (`_pipeline_parts/_image_pipeline_core.py:202`),
   `meas` (`:206`) and `plots: List[Any]` (`:222`), so the two-kwarg construction
   the plan's Task 7 and Task 9 tests use is valid. And `_identity_ref`
   (`_bindings.py:357-364`) returns `None` unless the plot object is
   *identity-equal* to an entry in the pipeline's op/meas/model/qc registry — a
   freshly-constructed `_Exploding()` passed only via `plots=` is not, so
   `_bindings.py:166-170` falls through to `type(raw).__name__` and the plan's
   `entries[0]["binding_id"] == "_Exploding"` assertion holds. `PlotMeas`/`PlotQc`
   do inherit `PhtPlot` (`_lifecycle.py:16,24`), so normalization accepts the bare
   instances. Measured too: `[Q5] -> [('_Exploding', '_Exploding', None)]`.
5. **`chrome_available()`'s cost on the *success* path.** The failure path is now
   measured (S12: 0.59 s cold, 0.02–0.03 s warm, verdict `False` on this cluster).
   The success path cannot be measured here — Chrome is not installed — and Task 4
   Step 6 correctly gates on it. I did not pre-empt that step.
6. **GUI tree-snapshot tests around plot publication.** `plots/` gains
   `plotly.min.js` (4.8 MB), `.plotlyjs.lock`, and possibly `.failures.jsonl`.
   `tests/gui/results_viewer/test_mutation_guard.py:553` uses a matplotlib figure
   and triggers no failure, so it should see no new files and should still count
   `checks == 3` — but I did not run `tests/gui/` to confirm no other snapshot test
   is perturbed. Worth adding `tests/gui/results_viewer/test_mutation_guard.py` to
   Task 9 Step 8's command line.
7. **The full-suite baseline.** Task 14 Step 5 quotes "11,106 tests with 81
   pre-existing failures". I did not re-measure it and state no total of my own.

---

## Index: the nine questions you asked

| # | Question | Answer | Where |
|---|---|---|---|
| 1 | Does the Task 5 page-loop replacement fit the surrounding function? Are all names in scope? Does `_atomic_write` preserve the `PlotPublicationBlocked` re-raise and the `close()` guarantees? | **Yes** on scope (all 11 names check out) and **yes** on both semantics — with one ordering gap across two writes | *Considered and fine*; S11 |
| 2 | Is the `lambda` capturing `page`/`src` a late-binding bug? | **No.** `_atomic_write` invokes it synchronously as its first statement; ruff's `B023` is not even enabled here | *Considered and fine* |
| 3 | Does the single `FigureAdapter.close` double-close a matplotlib figure? Does any path leave one unclosed? | **No path leaves one unclosed** (all five exits traced), and the double-close is **measured harmless** — `savefig+close+close OK`, PNG intact | *Considered and fine* |
| 4 | Is `figure_backend_of` valid on a figure closed earlier in the loop? | **Yes.** `plt.close` does not alter `type()`, and `output.pages` holds a live reference | *Considered and fine* |
| 5 | Does `emit_qc`'s `continue` still work outside the `try`? Can anything newly escape? Is this better than `binding = None`? | **Superseded by a user decision.** The spec contradicted itself (§3 vs §5) and the user chose §5 — `binding = None`, prelude inside the `try`, loop continues. B3 is rewritten against that shape, with its cost recorded rather than argued. The finding that survives regardless: the plan's test does not reproduce F4 at all | **B3** |
| 6 | Is `_require_backend`'s call-time `._output` import necessary? Does `figure_backend_of` stay stdlib-only? | **Not necessary** (module-level would be equally safe); `figure_backend_of` **is** stdlib-only. The plan attributes the guard to the wrong test | *Considered and fine* |
| 7 | Does the deliberate red window break any other file or gate? | **No** — nothing imports `test_pht_plot.py`, collection errors are module-scoped, and no Task 3–10 command line touches the two red files. Add "do not push until C8" | *Considered and fine* |
| 8 | Do the `monkeypatch.setattr(_backends, "chrome_available", …)` tests actually reach the consumer? | **Yes, measured** — `[Q4] before=True after_patch=False`. A function-scope `from X import y` re-reads the module attribute on every call. No test silently passes while testing nothing. But the same function-scope style applied to `figure_backend_of` is what makes B2 hard to fix cleanly | S4; **B2** |
| 9 | Is anything here more than the spec needs? | **Yes, one thing:** `plotlyjs_bundle_path` and `plot_failures_jsonl_path` are added, exported, and never called. Everything else earns its place | S6 |

---

## Separately: pre-existing PhenoTypic bugs (not plan defects)

1. **`emit_qc`'s handler can raise from inside itself** — `_coordinator.py:226`
   (try) / `:239` (assign) / `:261` (read). This is the spec's F4 and is in scope,
   but it is a live bug on `main` today, not something the plan introduces. Its
   quieter half is worse than the crash: on the second and later iterations
   `binding` still holds the *previous* binding, so the warning names the wrong
   plot.
2. **`_publish_image_value`'s flat path writes no manifest at all** —
   `_coordinator.py:370-380`. Independently of this change, a single-page image
   plot produces a PNG with no `manifest.json`, so there is no record of its
   backend, label or metadata, and a failed write leaves the directory
   indistinguishable from "nothing was configured". B1 is the HTML consequence of
   this asymmetry; the missing manifest is a pre-existing gap on its own.
3. **`FigureAdapter.save_png`'s matplotlib branch closes the caller's figure**
   (`_adapter.py:43-48`), while its Plotly branch does not. Callers therefore
   cannot rely on the figure surviving a save, which is why the writer has to call
   `close` defensively afterwards and why Task 5's ordering question arises at all.
   Not in scope here; worth an issue.
