# Deferred — strict backends everywhere

Deferred from `design.md` (2026-09-20). **Not** a rejected idea; a sequenced one.

## What was deferred

Removing backend sniffing from `FigureAdapter` entirely, so that every figure in
the system arrives with a declared backend and nothing is ever inferred:

```python
class FigureAdapter:
    @staticmethod
    def save_png(figure, path, *, backend: Literal["plotly", "mpl"]): ...
    @staticmethod
    def to_dash_component(figure, *, backend: Literal["plotly", "mpl"]): ...
    # _is_plotly / _is_matplotlib deleted
```

This would also propagate the declaration down the publication path —
`PlotPage` gaining a `backend` field, the writer validating declared-against-actual,
and the manifest's `"backend"` recording a declaration rather than a sniff.

## Why it was deferred, not rejected

v1 closes the actual defect. The bug is that **`@figure` assumes Plotly**;
`FigureAdapter`'s dispatch is correct, tested, and not implicated. Declaring the
backend at the decorator removes the inference from the path that was broken.

`FigureAdapter` is a different job. It also serves figures that never passed
through `@figure`:

- a provider that overrides `inspect()` and builds its own `PlotOutput` — today the
  **only** way a matplotlib figure reaches publication at all;
- a direct `publish_plot_output` call, which `tests/unit/plotting/test_output_adapter.py`
  makes on every one of its cases;
- `to_dash_component`, called from GUI code with a figure whose provenance the call
  site does not track.

Taking sniffing away without first giving every one of those a way to declare would
either break them or force a sniffed fallback — reintroducing the automatic routing
the change exists to remove, one layer lower.

## What it needs first

1. **A declaration channel for undecorated figures.** `PlotPage(backend=...)` is the
   obvious one, but an overridden `inspect()` has no `FigureSpec` to inherit from, so
   the default has to be decided rather than defaulted. Leaving it `None` and sniffing
   is the failure mode above.
2. **A migration for direct adapter callers.** `publish_plot_output` and
   `to_dash_component` are public-ish within the package; their signatures change.
3. **Evidence that it buys something.** v1 should run first. If `.failures.jsonl`
   never records a backend mismatch below the decorator, the remaining sniffing is
   not costing anything and this stays deferred.

## Reconsider when

- A provider legitimately needs to return matplotlib from an overridden `inspect()`,
  making the undecorated path load-bearing rather than theoretical; or
- a backend mismatch is recorded below the decorator in real use; or
- `PlotPage` is being changed for another reason and the field is nearly free.

---

# Deferred — distinguish invariant violations from plot failures in `emit_qc`

Recorded 2026-09-20, from the pre-dispatch plan review.

## The choice that was made

`emit_qc`'s handler read `binding.id` while `binding` was assigned ten lines into
the `try`, so a prelude failure raised `UnboundLocalError` **from inside the
handler** on the first iteration and named the *previous* plot on any later one.

Two shapes fix it. The spec argued for both in different sections — §3 for
narrowing the `try` so the prelude sits outside it, §5's test row for keeping the
loop going. **The user chose the second:** `binding = None` before the `try`, the
prelude inside it, the handler guarded with a `configured.id` fallback.

## What that costs, stated plainly

The prelude is pure dict and attribute access. A failure there is a programming
error — a bad `assert`, a missing attribute, a malformed QC entry — not a plot
failure. Keeping it inside the `try` means such an error is **swallowed and
recorded as a plot failure**, which is the behaviour §3 objected to.

The narrow-the-`try` shape makes the unbound case *structurally impossible* rather
than merely handled, and lets a programming error surface as one. The chosen shape
handles it correctly but keeps the swallow.

## The argument on the other side

`emit_qc` iterates **every** configured plot rather than a pre-filtered list, so
under the narrow-`try` shape one malformed binding kills every plot after it. That
is the same argument §3 makes for per-figure softness, one level up — which is what
makes this a defensible trade rather than a concession.

## Reconsider when

- `.failures.jsonl` ever records a prelude-shaped failure in real use — an
  `AssertionError` from `:231`, an `AttributeError` on `module.check` — filed as
  though a figure misbehaved while the run stayed green; or
- the `configured.id` fallback appears in a record, which means the binding
  identity in that row was a guess; or
- QC bindings gain enough prelude complexity that "pure dict and attribute access"
  stops being true; or
- the `configured.id` fallback is ever observed in a record, which means the
  prelude failed and the binding identity was a guess.

The change is small and local: move `try:` to just after the `model_copy`, delete
the `binding = None` line and the fallback, and invert the test's assertion from
"the loop continued" to `pytest.raises(RuntimeError)`.

# Deferred — run identity in `.failures.jsonl` records

Raised by the C5 gate (`reports/2026-09-20-figure-backend-routing/c5-review.md`, F4).

## What was deferred

`record_plot_failure` writes `ts, binding_id, plot_class, lifecycle, error` (plus
`dataset`/`image_stem` for image plots) and nothing that says *which run* produced
the record. The file is append-only and never reset, so records from reruns,
`--mode measure` re-emits and GUI refreshes accumulate side by side. Adding
`SLURM_JOB_ID` (when set) and, where available, the lifecycle epoch would let a
reader separate them.

## Why it was deferred, not rejected

The worst producer of misattributed records — a fenced Stage-3 worker writing into
a run it no longer owns — is removed by the C5 gate's F1 fix: a guard or fence
rejection now propagates as `PlotPublicationBlocked` instead of being recorded. What
remains is ordinary accumulation across legitimate runs, which `ts` already orders.
Adding fields is backward-compatible for a JSONL reader, so nothing is lost by
waiting.

## Reconsider when

- a consumer of `.failures.jsonl` appears (the GUI, a QC report, a dashboard) and
  needs to show "failures from this run" rather than "all failures ever"; or
- a rerun's records are mistaken for a current failure in practice.

# Deferred — `_`-prefixed `PlotAnalysis` class names (pre-existing)

A user `PlotAnalysis` whose class name starts with `_` fails analysis-id validation
at `registry.get(type(binding.plot).__name__)` in `PlotCoordinator.emit_analyses`,
before `inspect` runs. Pre-existing and out of this change's scope. Since C5 it is
at least *recorded* (lifecycle `analysis`), but the record blames id validation, not
the class name the user chose — and `_Name` is exactly what someone writes in a
notebook. Fix belongs with the analysis registry's id rules, not the plot writer.
