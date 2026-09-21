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
