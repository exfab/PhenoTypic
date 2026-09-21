# Explicit Figure Backends and Non-Silent Plot Failures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a pipeline figure's rendering backend an explicit required declaration, publish Plotly figures as HTML (with PNG when Chrome exists), and make a plot that fails to publish impossible to mistake for one that succeeded.

**Architecture:** `@figure` gains a required `backend` keyword that selects the theming path and type-checks the return, replacing an unconditional Plotly assumption. Publication gains a second rendering — HTML always for Plotly, PNG when a memoised Chrome probe says it is possible — with one hoisted `plotly.min.js` per run. Every swallowed failure is appended to a durable JSONL record and reflected in a `schema_version: 2` manifest.

**Tech Stack:** Python 3.12, pydantic v2, plotly (+ kaleido), matplotlib, pytest, `uv` as the sole runner.

**Spec:** `docs/superpowers/specs/2026-09-20-figure-backend-routing/design.md` (read it alongside this plan; the deferred alternative is in `DEFERRED.md` beside it)

## Global Constraints

- **`uv` is the sole runner.** Never bare `python` or `pip`. Every command is `uv run …`.
- **`abc_/plotting/_pht_plot.py` and `abc_/plotting/_output.py` import only the standard library at runtime.** Every plotly/matplotlib import goes inside the function that uses it. Guarded by `tests/unit/ci/test_deferred_imports.py`.
- **Never `ruff check --fix` without explicit paths.** Bare invocation rewrites the whole repo.
- **Backend vocabulary is two words in two places, deliberately:** the decorator and `figure_backend_of` use `"plotly"` / `"mpl"`; the published manifest's per-page `"backend"` value stays `"plotly"` / `"matplotlib"`, because that is what `FigureAdapter.backend_name` already writes and what `tests/unit/plotting/test_output_adapter.py:86` asserts. `backend_name` performs the one mapping. Do not unify them in this change.
- **Per-task testing only.** Run the directly-touched test files (~1 min). Do NOT run the full suite between tasks — it is ~65 min and belongs in Task 14 as one sharded Slurm job. See the `run-phenotypic-test` skill.
- **Commit after every task.** Explicit paths in `git add`; never `git add -A` (this is a shared worktree tree).

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/phenotypic/plotting/_pipeline/_backends.py` | `chrome_available()`, `ensure_plotlyjs_bundle()`, `plotlyjs_src_for()` — everything about *what can render* and the shared JS bundle. Separate from `_writer.py` so the capability probe has no import cycle with publication. |
| `src/phenotypic/plotting/_pipeline/_failures.py` | `record_plot_failure()` — the durable JSONL record. Its own module because it must never raise and is called from six `except` blocks. |
| `tests/unit/plotting/test_backends.py` | Chrome probe, bundle hoisting, relative `src` resolution. |
| `tests/unit/plotting/test_failure_record.py` | The JSONL record and its must-not-raise contract. |

**Modified:**

| File | Change |
|---|---|
| `src/phenotypic/abc_/plotting/_output.py` | Add `figure_backend_of`. |
| `src/phenotypic/abc_/plotting/_pht_plot.py` | `backend` on `figure()` and `FigureSpec`; branching wrapper; `_require_backend`; `report()` guard. |
| `src/phenotypic/plotting/_pipeline/_adapter.py` | Delegate predicates to `figure_backend_of`; add `save_html`. |
| `src/phenotypic/plotting/_pipeline/_writer.py` | Emit both renderings; manifest v2 with `files`/`renderers`/`failed`. |
| `src/phenotypic/plotting/_pipeline/_coordinator.py` | Record failures; narrow `emit_qc`'s `try`; both-rendering image path. |
| `src/phenotypic/sdk_/_io_constants.py` | `plot_failures_jsonl_path`, `plotlyjs_bundle_path`. |
| `src/phenotypic/_cli/_cli_validation.py` | Call the preflight. |
| `src/phenotypic/_cli/_cli_staged_workers.py:526` | Drop `strict=True`. |

---

## Task Dependency Order

```
Task 1 ──► Task 2 ──► Task 3
                 └──► Tasks 11, 12  (post-implementation annotation)
Task 4 ──► Task 5 ──► Task 6
      └──► Task 8
Task 7  (needs 4)
Task 9  (needs 8)
Task 10 (needs 6, 9)
Tasks 11–13 (post-implementation; need 2 merged and green)
Task 14 (final regression; needs everything)
```

Tasks 1–2 and Task 4 may start in parallel. Everything else is gated.

---

## Task 1: The shared backend predicate

Today `FigureAdapter._is_plotly` / `_is_matplotlib` are the only definition of "what backend is this figure", and they live in a private runtime module that `abc_/` may not import. Task 2 needs the same answer. Put it in the one module both layers may import, rather than writing a sixth copy.

**Files:**
- Modify: `src/phenotypic/abc_/plotting/_output.py` (add function + `__all__` entry)
- Modify: `src/phenotypic/abc_/plotting/__init__.py` (re-export)
- Modify: `src/phenotypic/plotting/_pipeline/_adapter.py:141-153` (delegate)
- Test: `tests/unit/abc_/plotting/test_output.py` (create if absent)

**Interfaces:**
- Consumes: nothing.
- Produces: `figure_backend_of(figure: Any) -> Literal["plotly", "mpl"] | None` — importable from `phenotypic.abc_.plotting`. Returns `None` for anything unrecognised; never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/abc_/plotting/test_output.py`:

```python
"""Backend identification shared by the decorator and the publisher."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from phenotypic.abc_.plotting import figure_backend_of


def test_identifies_a_plotly_figure() -> None:
    assert figure_backend_of(go.Figure()) == "plotly"


def test_identifies_a_matplotlib_figure() -> None:
    assert figure_backend_of(MplFigure()) == "mpl"


def test_returns_none_for_an_unsupported_object() -> None:
    assert figure_backend_of(object()) is None
    assert figure_backend_of(None) is None
    assert figure_backend_of("not a figure") is None


def test_agrees_with_the_publisher_vocabulary() -> None:
    """The manifest keeps 'matplotlib'; the decorator uses 'mpl'."""
    from phenotypic.plotting._pipeline import FigureAdapter

    assert FigureAdapter.backend_name(go.Figure()) == "plotly"
    assert FigureAdapter.backend_name(MplFigure()) == "matplotlib"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/abc_/plotting/test_output.py -v`
Expected: FAIL — `ImportError: cannot import name 'figure_backend_of'`

- [ ] **Step 3: Add the function**

In `src/phenotypic/abc_/plotting/_output.py`, after the `FigureLike` alias and before `PlotPage`:

```python
def figure_backend_of(figure: Any) -> Literal["plotly", "mpl"] | None:
    """Return the rendering backend of ``figure``, or ``None`` if unknown.

    Identification is by module string so this module stays standard-library
    only at runtime -- importing plotly or matplotlib to answer the question
    would defeat the lazy-import contract this package is held to.

    Args:
        figure: Any object that might be a supported figure.

    Returns:
        ``"plotly"``, ``"mpl"``, or ``None`` for anything unrecognised. Never
        raises: callers that need an error raise their own, with the context
        only they have.
    """
    module = type(figure).__module__
    if type(figure).__name__ != "Figure":
        return None
    if module.startswith("plotly."):
        return "plotly"
    if module.startswith("matplotlib."):
        return "mpl"
    return None
```

Add `Literal` to the `typing` import at the top of the file, and `"figure_backend_of"` to `__all__`.

- [ ] **Step 4: Re-export it**

In `src/phenotypic/abc_/plotting/__init__.py`, add `figure_backend_of` to the `._output` import line and to `__all__` (keep both alphabetical).

- [ ] **Step 5: Delegate from the adapter**

Replace `src/phenotypic/plotting/_pipeline/_adapter.py:129-153` with:

```python
    @staticmethod
    def backend_name(figure: Any) -> str:
        """Return the stable backend name for a supported figure.

        The published manifest spells the matplotlib backend ``"matplotlib"``
        while the ``@figure`` decorator spells it ``"mpl"``. This is the one
        place the two vocabularies meet; the wire format is not changed here.
        """
        backend = figure_backend_of(figure)
        if backend == "plotly":
            return "plotly"
        if backend == "mpl":
            return "matplotlib"
        raise TypeError(
            "unsupported figure type "
            f"{type(figure).__module__}.{type(figure).__qualname__}"
        )

    @staticmethod
    def _is_plotly(figure: Any) -> bool:
        return figure_backend_of(figure) == "plotly"

    @staticmethod
    def _is_matplotlib(figure: Any) -> bool:
        return figure_backend_of(figure) == "mpl"
```

Add to the imports at the top of `_adapter.py`:

```python
from phenotypic.abc_.plotting import figure_backend_of
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit/abc_/plotting/test_output.py tests/unit/plotting/test_output_adapter.py tests/unit/abc_/plotting/test_imports.py -v`
Expected: PASS, all of them. `test_output_adapter.py` must stay green — it is the proof the delegation did not change the wire format.

- [ ] **Step 7: Confirm the lazy-import contract still holds**

Run: `uv run pytest tests/unit/ci/test_deferred_imports.py tests/unit/ci/test_startup_imports.py -v`
Expected: PASS. If this fails, `_output.py` gained a module-level plotly/matplotlib import — remove it; the function identifies by module string precisely so it does not need one.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/abc_/plotting/_output.py src/phenotypic/abc_/plotting/__init__.py src/phenotypic/plotting/_pipeline/_adapter.py tests/unit/abc_/plotting/test_output.py
git commit -m "refactor(plotting): one definition of a figure's backend

figure_backend_of() moves the module-string check into the one module
both abc_/plotting and the private publisher may import. FigureAdapter
delegates rather than keeping its own copy, and maps mpl -> matplotlib
so the published manifest's vocabulary is unchanged."
```

---

## Task 2: `backend` on `@figure`

**Files:**
- Modify: `src/phenotypic/abc_/plotting/_pht_plot.py:102-128` (`FigureSpec`), `:131-208` (`figure`)
- Test: `tests/unit/abc_/plotting/test_figure_backend.py` (create)

**Interfaces:**
- Consumes: `figure_backend_of` (Task 1).
- Produces:
  - `figure(*, title: str, backend: Literal["plotly", "mpl"], section: str = "default", controls: dict[str, Control] | None = None, description: Any = None, primary: bool = False)` — `backend` required, keyword-only.
  - `FigureSpec.backend: Literal["plotly", "mpl"]`.
  - Tasks 11 and 12 annotate call sites against this signature.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/abc_/plotting/test_figure_backend.py`:

```python
"""The @figure backend declaration: required, checked, and routed."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure as MplFigure

from phenotypic.abc_.plotting import PhtPlot, figure


def test_backend_is_required() -> None:
    with pytest.raises(TypeError, match="backend"):
        @figure(title="No backend")  # type: ignore[call-arg]
        def _fig(self, subject):  # pragma: no cover - never called
            return go.Figure()


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        @figure(title="Bad", backend="svg")  # type: ignore[arg-type]
        def _fig(self, subject):  # pragma: no cover - never called
            return go.Figure()


class _PlotlyDeclaredReturnsMpl(PhtPlot):
    @figure(title="Wrong way round", backend="plotly", primary=True)
    def wrong(self, subject):
        return MplFigure()


class _MplDeclaredReturnsPlotly(PhtPlot):
    @figure(title="Also wrong", backend="mpl", primary=True)
    def wrong(self, subject):
        return go.Figure()


class _ReturnsNothing(PhtPlot):
    @figure(title="Nothing", backend="plotly", primary=True)
    def nothing(self, subject):
        return None


def test_plotly_declared_method_returning_matplotlib_raises() -> None:
    with pytest.raises(TypeError) as excinfo:
        _PlotlyDeclaredReturnsMpl().inspect(object())
    message = str(excinfo.value)
    assert "wrong" in message           # names the method
    assert "'plotly'" in message        # names the declaration
    assert "matplotlib.figure.Figure" in message   # names what came back


def test_mpl_declared_method_returning_plotly_raises() -> None:
    with pytest.raises(TypeError) as excinfo:
        _MplDeclaredReturnsPlotly().inspect(object())
    message = str(excinfo.value)
    assert "'mpl'" in message
    assert "plotly" in message


def test_a_non_figure_return_raises_the_same_error() -> None:
    with pytest.raises(TypeError, match="NoneType"):
        _ReturnsNothing().inspect(object())


class _Plotly(PhtPlot):
    @figure(title="Good plotly", backend="plotly", primary=True)
    def good(self, subject):
        return go.Figure(go.Scatter(y=[1, 2, 3]))


class _Mpl(PhtPlot):
    """Asserts the rcParams are live DURING construction, not after."""

    observed_cycle: object = None

    @figure(title="Good mpl", backend="mpl", primary=True)
    def good(self, subject):
        import matplotlib as mpl

        type(self).observed_cycle = mpl.rcParams["axes.prop_cycle"]
        return MplFigure()


def test_a_plotly_figure_is_themed_after_construction() -> None:
    """The assertion must FAIL if apply_theme is skipped.

    `fig.layout.template is not None` is true of any go.Figure, themed or
    not -- measured: `[Q7] untheme template is None? False`. The font family
    is the discriminator: None when raw, the DESIGN.md stack when themed.
    """
    from phenotypic.sdk_.viz.figures import FONT_FAMILY

    assert go.Figure().layout.template.layout.font.family is None  # control
    fig = _Plotly().inspect(object())
    assert fig.layout.template.layout.font.family is not None
    assert FONT_FAMILY


def test_the_mpl_theme_is_live_while_the_figure_is_built() -> None:
    import matplotlib as mpl
    from phenotypic.sdk_.viz.figures import phenotypic_rc

    before = mpl.rcParams["axes.prop_cycle"]
    _Mpl().inspect(object())
    after = mpl.rcParams["axes.prop_cycle"]

    # Themed inside the method body...
    assert _Mpl.observed_cycle == phenotypic_rc()["axes.prop_cycle"]
    # ...and scoped: the caller's global rcParams are untouched.
    assert before == after
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/abc_/plotting/test_figure_backend.py -v`
Expected: FAIL — `test_backend_is_required` fails because `backend` is currently accepted-and-ignored (no TypeError), and the class bodies raise `TypeError: figure() got an unexpected keyword argument 'backend'` at collection.

- [ ] **Step 3: Add the field to `FigureSpec`**

In `src/phenotypic/abc_/plotting/_pht_plot.py`, add to the `FigureSpec` docstring's `Attributes:` block, after the `section:` line:

```
        backend: Declared rendering backend, ``"plotly"`` or ``"mpl"``.
```

and add the field after `section: str` (line 120):

```python
    backend: Literal["plotly", "mpl"]
```

`Literal` is already imported at the top of the file.

- [ ] **Step 4: Add the parameter and the check**

Add this module-level helper just above `def figure(` (line 131):

```python
def _require_backend(figure_value: Any, declared: str, fn: Callable[..., Any]) -> None:
    """Raise unless ``figure_value`` matches the backend its method declared.

    Args:
        figure_value: Whatever the decorated method returned.
        declared: The backend named in the ``@figure`` declaration.
        fn: The decorated function, named in the error.

    Raises:
        TypeError: If the return does not match the declaration.
    """
    from ._output import figure_backend_of

    actual = figure_backend_of(figure_value)
    if actual == declared:
        return
    other = "mpl" if declared == "plotly" else "plotly"
    expected = (
        "plotly.graph_objects.Figure" if declared == "plotly"
        else "matplotlib.figure.Figure"
    )
    raise TypeError(
        f"@figure({fn.__name__!r}): declared backend {declared!r} but the "
        f"method returned {type(figure_value).__module__}."
        f"{type(figure_value).__qualname__}. "
        f"Declare backend={other!r}, or return a {expected}."
    )
```

Change the `figure` signature (line 131-138) to:

```python
def figure(
    *,
    title: str,
    backend: Literal["plotly", "mpl"],
    section: str = "default",
    controls: dict[str, Control] | None = None,
    description: Any = None,
    primary: bool = False,
) -> Callable[[Callable[..., "go.Figure"]], Callable[..., "go.Figure"]]:
```

Add to the docstring's `Args:` block, after `title:`:

```
        backend: Rendering backend this method returns, ``"plotly"`` or
            ``"mpl"``. Required: the backend decides how the figure is themed
            (Plotly themes the result, matplotlib themes the construction), so
            there is no default that is right for both.
```

and to `Raises:`:

```
        ValueError: If ``backend`` is not ``"plotly"`` or ``"mpl"``, or if a
            control key does not name a method parameter.
```

Add the validation as the first statement in the body, before `declared_controls`:

```python
    if backend not in ("plotly", "mpl"):
        raise ValueError(
            f"@figure: unknown backend {backend!r}; expected 'plotly' or 'mpl'"
        )
```

- [ ] **Step 5: Branch the wrapper**

Replace lines 188-192 (the `wrapper` body) with:

```python
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> "go.Figure":
            if backend == "plotly":
                from phenotypic.sdk_.viz.figures._theme import apply_theme

                built = fn(*args, **kwargs)
                _require_backend(built, "plotly", fn)
                return apply_theme(built)

            # The matplotlib theme is rcParams, which must be live WHILE the
            # figure is constructed. There is no post-pass equivalent of
            # apply_theme; applying it afterwards would silently do nothing.
            from phenotypic.sdk_.viz.figures._mpl_theme import (
                phenotypic_mpl_context,
            )

            with phenotypic_mpl_context():
                built = fn(*args, **kwargs)
            _require_backend(built, "mpl", fn)
            return built
```

Add `backend=backend,` to the `FigureSpec(...)` construction at line 194, right after `title=title,`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit/abc_/plotting/test_figure_backend.py -v`
Expected: PASS.

- [ ] **Step 7: Confirm laziness and see the expected collateral damage**

Run: `uv run pytest tests/unit/ci/test_deferred_imports.py -v`
Expected: PASS.

Run: `uv run pytest tests/unit/abc_/plotting/ -v`
Expected: `test_pht_plot.py` now FAILS at collection — its 9 `@figure` sites have no `backend`. **This is correct and expected.** Do not fix them here; Task 11 does, and fixing them now would hide whether Task 11 is complete.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/abc_/plotting/_pht_plot.py tests/unit/abc_/plotting/test_figure_backend.py
git commit -m "feat(plotting)!: require an explicit backend on @figure

@figure(backend='plotly'|'mpl') selects the theming path and checks the
return against it, replacing an unconditional apply_theme that assumed
Plotly and died on a matplotlib figure with an AttributeError naming
neither the decorator nor the backend.

The branches are deliberately asymmetric: Plotly themes the returned
figure, matplotlib themes the construction via rc_context. Applying the
matplotlib theme as a post-pass would silently do nothing.

BREAKING: @figure without backend= is now a TypeError. In-tree call
sites are annotated in a later task; tests/unit/abc_/plotting/
test_pht_plot.py is red until then, by design."
```

---

## Task 3: `report()` refuses to compose matplotlib

`_compose_control_free_figure` calls `make_subplots` and iterates `rendered.data` — Plotly-only. With `mpl` now declarable, it would fail obscurely inside Plotly instead of saying what is wrong.

**Files:**
- Modify: `src/phenotypic/abc_/plotting/_pht_plot.py` (`report`, `_compose_control_free_figure`)
- Test: `tests/unit/abc_/plotting/test_figure_backend.py` (append)

**Interfaces:**
- Consumes: `FigureSpec.backend` (Task 2).
- Produces: no new names; `PhtPlot.report()` raises `TypeError` for any provider with an `mpl` spec.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/abc_/plotting/test_figure_backend.py`:

```python
class _AllMpl(PhtPlot):
    @figure(title="One", backend="mpl", primary=True)
    def one(self, subject):
        return MplFigure()


class _Mixed(PhtPlot):
    @figure(title="P", backend="plotly", primary=True)
    def p(self, subject):
        return go.Figure()

    @figure(title="M", backend="mpl")
    def m(self, subject):
        return MplFigure()


def test_report_refuses_an_all_matplotlib_provider() -> None:
    with pytest.raises(TypeError, match="cannot compose matplotlib"):
        _AllMpl().report(object())


def test_report_refuses_a_mixed_provider() -> None:
    with pytest.raises(TypeError, match="cannot compose matplotlib"):
        _Mixed().report(object())


def test_inspect_still_works_on_a_matplotlib_provider() -> None:
    """The limitation is composition, not rendering."""
    assert isinstance(_AllMpl().inspect(object()), MplFigure)


class _MplWithControls(PhtPlot):
    """B6: this provider never reaches _compose_control_free_figure."""

    @figure(
        title="Controlled",
        backend="mpl",
        primary=True,
        controls={"sigma": Control(label="s", kind="float", default=1.0,
                                   bounds=(0.0, 2.0))},
    )
    def controlled(self, subject, *, sigma: float = 1.0):
        return MplFigure()


def test_report_refuses_a_matplotlib_provider_that_declares_controls() -> None:
    """Guards the notebook-dashboard path, which bypasses the composer."""
    with pytest.raises(TypeError, match="cannot compose matplotlib"):
        _MplWithControls().report(object())
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/abc_/plotting/test_figure_backend.py -k report -v`
Expected: all three FAIL — the two composer cases with `AttributeError: 'Figure'
object has no attribute 'layout'` from inside theming, and the controls case with
the same error raised from `build_notebook_dashboard`. None of them is the
`TypeError` asked for. Add `Control` to the file's imports from
`phenotypic.abc_.plotting`.

- [ ] **Step 3: Add the guard**

**B6 — put the guard in `report()`, NOT in `_compose_control_free_figure`.**
Measured: a provider declaring controls routes through `build_notebook_dashboard`
and never reaches the composer (`[Q9d] composer reached by: ['_OneMpl', '_TwoMpl']`
-- the controls provider is absent), while `[Q9c]` shows that path fails with the
same `AttributeError`. A guard in the composer would leave the notebook path
uncovered, which is a spec requirement with no coverage.

In `_pht_plot.py`, insert in **`report()`**, immediately after
`specs = self.iter_figures()` and its empty check, BEFORE the
`if any(spec.controls for spec in specs):` branch:

```python
        mpl_specs = [spec.name for spec in specs if spec.backend == "mpl"]
        # Placed here rather than in _compose_control_free_figure: a provider
        # with controls never reaches the composer (it goes to
        # build_notebook_dashboard), so a guard down there misses that path.
        if mpl_specs:
            raise TypeError(
                f"{type(self).__name__}.report(): cannot compose matplotlib "
                f"figures ({', '.join(sorted(mpl_specs))}). Composition is "
                "Plotly-only. Use inspect() for a single figure, or override "
                "report() with a plot-specific implementation."
            )
```

Add to `report`'s docstring `Raises:` block:

```
        TypeError: If any visible figure declares ``backend="mpl"``. Composing
            matplotlib figures is not supported.
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/abc_/plotting/test_figure_backend.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/abc_/plotting/_pht_plot.py tests/unit/abc_/plotting/test_figure_backend.py
git commit -m "feat(plotting): report() states the matplotlib composition limit

_compose_control_free_figure is Plotly-only (make_subplots over
rendered.data). Now that backend='mpl' is declarable, say so plainly
instead of failing inside Plotly. inspect() is unaffected."
```

---

## Task 4: Chrome capability and the shared Plotly bundle

**Files:**
- Create: `src/phenotypic/plotting/_pipeline/_backends.py`
- Modify: `src/phenotypic/sdk_/_io_constants.py` (add `plotlyjs_bundle_path`)
- Modify: `src/phenotypic/plotting/_pipeline/__init__.py` (export)
- Test: `tests/unit/plotting/test_backends.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `chrome_available() -> bool` — memoised per process.
  - `reset_chrome_probe() -> None` — clears the memo; tests only.
  - `ensure_plotlyjs_bundle(plots_base: Path) -> Path` — writes `<plots_base>/plotly.min.js` once, returns its path.
  - `plotlyjs_src_for(page_dir: Path, bundle: Path) -> str` — relative `src` string.
  - `plotlyjs_bundle_path(output_dir: Path) -> Path` in `sdk_`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/plotting/test_backends.py`:

```python
"""Rendering capability and the hoisted plotly.min.js bundle."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.plotting._pipeline._backends import (
    chrome_available,
    ensure_plotlyjs_bundle,
    plotlyjs_src_for,
    reset_chrome_probe,
)


@pytest.fixture(autouse=True)
def _clear_probe():
    reset_chrome_probe()
    yield
    reset_chrome_probe()


def test_the_bundle_is_written_once(tmp_path: Path) -> None:
    first = ensure_plotlyjs_bundle(tmp_path)
    assert first == tmp_path / "plotly.min.js"
    assert first.stat().st_size > 1_000_000
    stamp = first.stat().st_mtime_ns

    second = ensure_plotlyjs_bundle(tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns == stamp, "bundle was rewritten"


def test_the_bundle_is_rewritten_if_truncated(tmp_path: Path) -> None:
    bundle = ensure_plotlyjs_bundle(tmp_path)
    bundle.write_text("corrupted")
    assert ensure_plotlyjs_bundle(tmp_path).stat().st_size > 1_000_000


@pytest.mark.parametrize(
    "page_dir, expected",
    [
        ("plots/sym", "../plotly.min.js"),
        ("plots/sym/ds-1", "../../plotly.min.js"),
        ("plots/sym/ds-1/A01-abc123", "../../../plotly.min.js"),
    ],
)
def test_the_src_resolves_from_every_layout(
    tmp_path: Path, page_dir: str, expected: str
) -> None:
    """Aggregate, single-page image, and multi-page image layouts."""
    bundle = tmp_path / "plots" / "plotly.min.js"
    assert plotlyjs_src_for(tmp_path / page_dir, bundle) == expected


def test_the_probe_is_memoised(monkeypatch) -> None:
    calls = {"n": 0}

    def _counting_to_image(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("Kaleido requires Google Chrome to be installed.")

    import plotly.io as pio

    monkeypatch.setattr(pio, "to_image", _counting_to_image)
    assert chrome_available() is False
    assert chrome_available() is False
    assert calls["n"] == 1, "the probe ran more than once"


def test_the_probe_reports_success(monkeypatch) -> None:
    import plotly.io as pio

    monkeypatch.setattr(pio, "to_image", lambda *a, **k: b"\x89PNG")
    assert chrome_available() is True
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/plotting/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.plotting._pipeline._backends'`

- [ ] **Step 3: Add the path helper**

In `src/phenotypic/sdk_/_io_constants.py`, after `plots_dir` (line 1105-1107):

```python
def plotlyjs_bundle_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/plots/plotly.min.js``.

    One bundle per run, shared by every published HTML page regardless of
    depth. Plotly's own ``include_plotlyjs="directory"`` writes a 4.8 MB copy
    into every directory it touches, and a multi-page image plot gets one
    directory per image -- gigabytes on a real run.
    """
    return plots_dir(output_dir) / PLOTLYJS_BUNDLE
```

and beside `DIR_PLOTS` (line 806):

```python
#: Shared Plotly bundle for published HTML pages: ``<plots>/plotly.min.js``.
PLOTLYJS_BUNDLE: Final[str] = "plotly.min.js"
```

Export both from `src/phenotypic/sdk_/__init__.py` (add to the `._io_constants` import list and to `__all__`, keeping alphabetical order).

- [ ] **Step 4: Write the module**

Create `src/phenotypic/plotting/_pipeline/_backends.py`:

```python
"""What can render here, and the one Plotly bundle every page shares.

Kept apart from ``_writer`` so the capability probe can be imported by CLI
validation without dragging in publication.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from phenotypic.sdk_._file_locking import exclusive_path_lock

logger = logging.getLogger(__name__)

#: Memoised verdict of :func:`chrome_available`. ``None`` means "not yet asked".
_CHROME: bool | None = None

#: Smallest size a complete bundle can plausibly have, used to detect a
#: truncated or half-written file rather than trusting mere existence.
_MIN_BUNDLE_BYTES = 1_000_000


def chrome_available() -> bool:
    """Return whether Plotly can rasterise here, probing at most once.

    Kaleido shells out to Chrome for PNG export. The probe renders a minimal
    figure because that exercises exactly what publication will do -- a check
    for a browser binary answers a different, weaker question, and
    ``choreographer``'s discovery helper does not match its documented
    signature.

    Returns:
        ``True`` if a PNG can be produced, ``False`` otherwise. Never raises.
    """
    global _CHROME
    if _CHROME is not None:
        return _CHROME

    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        pio.to_image(go.Figure(), format="png", width=8, height=8)
        _CHROME = True
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot"
        logger.debug("Plotly PNG backend unavailable: %s", exc)
        _CHROME = False
    return _CHROME


def reset_chrome_probe() -> None:
    """Clear the memoised verdict. Tests only."""
    global _CHROME
    _CHROME = None


def ensure_plotlyjs_bundle(plots_base: Path) -> Path:
    """Write ``plotly.min.js`` under *plots_base* once, returning its path.

    Concurrent SLURM workers race to create it, so the write is locked and
    skipped when a complete file is already present. A short or truncated file
    is rewritten -- existence alone is not evidence of a usable bundle.

    Args:
        plots_base: Resolved ``deliverables/plots`` directory.

    Returns:
        Path to the bundle.
    """
    from phenotypic.sdk_ import PLOTLYJS_BUNDLE

    bundle = plots_base / PLOTLYJS_BUNDLE
    if bundle.is_file() and bundle.stat().st_size >= _MIN_BUNDLE_BYTES:
        return bundle

    plots_base.mkdir(parents=True, exist_ok=True)
    with exclusive_path_lock(plots_base / ".plotlyjs.lock"):
        if bundle.is_file() and bundle.stat().st_size >= _MIN_BUNDLE_BYTES:
            return bundle
        from plotly.offline import get_plotlyjs

        temporary = plots_base / f".{PLOTLYJS_BUNDLE}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(get_plotlyjs(), encoding="utf-8")
            os.replace(temporary, bundle)
        finally:
            temporary.unlink(missing_ok=True)
    return bundle


def plotlyjs_src_for(page_dir: Path, bundle: Path) -> str:
    """Return the ``src`` a page in *page_dir* uses to reach *bundle*.

    Plotly emits a string ``include_plotlyjs`` value verbatim as the script
    src, so a computed relative path hoists one bundle across every layout
    without hard-coding directory depth.

    Args:
        page_dir: Directory the HTML page will be written into.
        bundle: Path returned by :func:`ensure_plotlyjs_bundle`.

    Returns:
        A relative POSIX path such as ``"../../plotly.min.js"``.
    """
    return Path(os.path.relpath(bundle, page_dir)).as_posix()


__all__ = [
    "chrome_available",
    "ensure_plotlyjs_bundle",
    "plotlyjs_src_for",
    "reset_chrome_probe",
]
```

Export all four from `src/phenotypic/plotting/_pipeline/__init__.py` (add a `from ._backends import …` line and the names to `__all__`, alphabetical).

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/plotting/test_backends.py -v`
Expected: PASS.

- [ ] **Step 6: Measure the probe's cost and report it**

The spec requires this and flags it as unmeasured. Run:

```bash
uv run python -c "
import time, warnings; warnings.filterwarnings('ignore')
from phenotypic.plotting._pipeline._backends import chrome_available, reset_chrome_probe
reset_chrome_probe()
t = time.time(); verdict = chrome_available(); first = time.time() - t
t = time.time(); chrome_available(); cached = time.time() - t
print(f'chrome_available() -> {verdict}; first call {first:.2f}s, cached {cached*1e6:.1f}us')
"
```

Record the number in the task's commit message. **If Chrome IS present and the first call exceeds ~5 s, stop and report it** rather than proceeding — the eager probe in Task 7 would then cost that much per submitting process, and the plan needs revisiting. On a machine without Chrome the expected figure is ~0.6 s.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/plotting/_pipeline/_backends.py src/phenotypic/plotting/_pipeline/__init__.py src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/plotting/test_backends.py
git commit -m "feat(plotting): Chrome capability probe and one shared plotly.min.js

chrome_available() memoises a minimal to_image() render -- the probe
exercises exactly what publication does, unlike a browser-binary check.

ensure_plotlyjs_bundle() writes ONE bundle per run under
deliverables/plots/ and plotlyjs_src_for() computes each page's relative
src. Plotly's include_plotlyjs='directory' writes 4.8 MB per directory,
and a multi-page image plot gets a directory per image.

Measured first-call cost: <FILL IN FROM STEP 6>."
```

---

## Task 5: Publish HTML alongside PNG, from one shared renderer

**Revised after the pre-dispatch review (B1, S1–S4, S11).** The single-page image
path in `_coordinator.py:370-380` does **not** call `publish_plot_output` — it
writes a PNG directly. So rendering logic that lives only inside
`_publish_plot_output_locked` never reaches the commonest image plot. This task
therefore extracts the per-page rendering into `_render_page`, which Task 9 calls
from the image path too.

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_adapter.py` (add `save_html`)
- Modify: `src/phenotypic/plotting/_pipeline/_writer.py`
- Test: `tests/unit/plotting/test_output_adapter.py` (append)

**Interfaces:**
- Consumes: `chrome_available`, `ensure_plotlyjs_bundle`, `plotlyjs_src_for` (Task 4); `record_plot_failure` (Task 8); `figure_backend_of` (Task 1).
- Produces:
  - `FigureAdapter.save_html(figure, path, *, plotlyjs_src: str) -> None`
  - `_render_page(figure, directory, stem, *, plots_base, plot_id, publication_guard, commit_guard) -> tuple[dict[str, str], list[str], str | None]` returning `(files, errors, backend)`. **Task 9 calls this**; keep the signature exactly.
  - `publish_plot_output(..., plots_base: Path | None = None)`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/plotting/test_output_adapter.py`:

```python
def test_a_plotly_page_publishes_html_without_chrome(tmp_path, monkeypatch) -> None:
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)

    output = PlotOutput(pages=(
        PlotPage(key="only", figure=go.Figure(go.Scatter(y=[1, 2])), label="Only"),
    ))
    manifest = publish_plot_output(output, tmp_path / "sym", plot_id="sym")

    page = manifest["pages"][0]
    assert page["files"] == {"html": "Only.html"}
    assert (tmp_path / "sym" / "Only.html").is_file()
    assert not (tmp_path / "sym" / "Only.png").exists()


def test_the_html_references_the_hoisted_bundle(tmp_path, monkeypatch) -> None:
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)

    plots_base = tmp_path / "plots"
    output = PlotOutput(pages=(PlotPage(key="only", figure=go.Figure(), label="Only"),))
    publish_plot_output(
        output, plots_base / "sym", plot_id="sym", plots_base=plots_base
    )

    html = (plots_base / "sym" / "Only.html").read_text()
    assert 'src="../plotly.min.js"' in html
    assert (plots_base / "plotly.min.js").is_file()
    # The 4.8 MB bundle must NOT be duplicated into the page directory.
    assert not (plots_base / "sym" / "plotly.min.js").exists()


def test_a_matplotlib_page_publishes_png_only(tmp_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import publish_plot_output

    output = PlotOutput(pages=(PlotPage(key="only", figure=Figure(), label="Only"),))
    manifest = publish_plot_output(output, tmp_path / "m", plot_id="m")

    assert manifest["pages"][0]["files"] == {"png": "Only.png"}
    assert manifest["renderers"] == {"png": "available"}


def test_a_partial_rendering_failure_is_not_lost(tmp_path, monkeypatch) -> None:
    """S2: HTML succeeds, PNG fails -- the PNG failure must still be recorded.

    Without this the page lands in "pages" with one file and the other
    renderer's failure vanishes: best-effort silently meaning silent, one
    level below where §3 fixed it.
    """
    import json

    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, _writer, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: True)

    def _png_explodes(*args, **kwargs):
        raise RuntimeError("raster exploded")

    monkeypatch.setattr(_writer.FigureAdapter, "save_png", _png_explodes)

    plots_base = tmp_path / "plots"
    output = PlotOutput(pages=(PlotPage(key="only", figure=go.Figure(), label="Only"),))
    manifest = publish_plot_output(
        output, plots_base / "sym", plot_id="sym", plots_base=plots_base
    )

    page = manifest["pages"][0]
    assert page["files"] == {"html": "Only.html"}          # HTML still published
    assert any("raster exploded" in err for err in page["partial"])

    record = plots_base / ".failures.jsonl"
    assert record.is_file(), "S3: the writer must write .failures.jsonl"
    entry = json.loads(record.read_text().splitlines()[0])
    assert "raster exploded" in entry["error"]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/plotting/test_output_adapter.py -k "html or png_only or partial" -v`
Expected: FAIL — `KeyError: 'files'`, and `publish_plot_output` has no `plots_base` parameter.

- [ ] **Step 3: Add `save_html` to the adapter**

In `src/phenotypic/plotting/_pipeline/_adapter.py`, after `save_png`:

```python
    @staticmethod
    def save_html(figure: Any, path: Path, *, plotlyjs_src: str) -> None:
        """Write one Plotly figure as a standalone interactive HTML page.

        No Kaleido and no Chrome are involved. ``plotlyjs_src`` is emitted
        verbatim as the script src, so the 4.8 MB bundle is referenced rather
        than embedded.

        Args:
            figure: A Plotly figure.
            path: Destination ``.html`` path.
            plotlyjs_src: Relative src, from ``plotlyjs_src_for``.

        Raises:
            TypeError: If *figure* is not a Plotly figure.
        """
        if figure_backend_of(figure) != "plotly":
            raise TypeError(
                "HTML export is Plotly-only; got "
                f"{type(figure).__module__}.{type(figure).__qualname__}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path, include_plotlyjs=plotlyjs_src)
```

- [ ] **Step 4: Import the predicate at MODULE level in `_writer.py`**

**S4 — this is load-bearing, not style.** Add to the imports at the top of
`_writer.py` (it already imports `PlotOutput` from that package):

```python
from phenotypic.abc_.plotting import figure_backend_of
```

A function-scope import would leave no module attribute for a test to patch, and
Task 6 Step 1's repair of the concurrency guard depends on patching
`_writer.figure_backend_of`.

- [ ] **Step 5: Add `_atomic_write` and `_render_page`**

At module scope in `_writer.py`:

```python
def _atomic_write(
    destination: Path,
    write: Callable[[Path], None],
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    """Write via a temporary sibling and replace, or leave nothing behind."""
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        write(temporary)
        with publication_commit(commit_guard):
            _require_plot_publication(publication_guard)
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _render_page(
    figure: Any,
    directory: Path,
    stem: str,
    *,
    plots_base: Path,
    plot_id: str,
    publication_guard: Callable[[], bool] | None = None,
    commit_guard: CommitGuard | None = None,
) -> tuple[dict[str, str], list[str], str | None]:
    """Render one figure to every format its backend supports.

    This is the single definition of "what files does a page produce". It is
    called from :func:`_publish_plot_output_locked` for multi-page and aggregate
    output, and from ``PlotCoordinator._publish_image_value`` for the flat
    single-page image path -- which does not go through the writer at all, and
    would otherwise never gain HTML.

    HTML is attempted first: it needs no Chrome, so a page that can be published
    at all is on disk before anything that might fail is tried.

    Args:
        figure: The figure to render.
        directory: Directory the page files are written into.
        stem: Filename stem, without extension.
        plots_base: Resolved ``deliverables/plots`` directory, for the bundle.
        plot_id: Binding id, for diagnostics.
        publication_guard: Optional GUI compare-and-set predicate.
        commit_guard: Optional commit guard.

    Returns:
        ``(files, errors, backend)`` -- a mapping of format to filename for
        everything that published, a list of formatted error strings for
        everything that did not, and the figure's backend (``None`` if
        unsupported, in which case *files* is empty).

    Raises:
        PlotPublicationBlocked: If a guard rejects the write. Never swallowed --
            it means the output snapshot changed and this whole publication is
            void.
    """
    from ._backends import chrome_available, ensure_plotlyjs_bundle, plotlyjs_src_for

    backend = figure_backend_of(figure)
    if backend is None:
        return {}, [
            "TypeError: unsupported figure type "
            f"{type(figure).__module__}.{type(figure).__qualname__}"
        ], None

    files: dict[str, str] = {}
    errors: list[str] = []

    if backend == "plotly":
        try:
            bundle = ensure_plotlyjs_bundle(plots_base)
            src = plotlyjs_src_for(directory, bundle)
            _atomic_write(
                directory / f"{stem}.html",
                lambda dest: FigureAdapter.save_html(figure, dest, plotlyjs_src=src),
                publication_guard=publication_guard,
                commit_guard=commit_guard,
            )
            files["html"] = f"{stem}.html"
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - plots are best-effort
            errors.append(f"{type(exc).__name__}: {exc}")
            logger.warning(
                "Plot %s page %s failed during HTML save: %s", plot_id, stem, exc
            )

    if backend == "mpl" or chrome_available():
        try:
            _atomic_write(
                directory / f"{stem}.png",
                lambda dest: FigureAdapter.save_png(figure, dest),
                publication_guard=publication_guard,
                commit_guard=commit_guard,
            )
            files["png"] = f"{stem}.png"
        except PlotPublicationBlocked:
            # S11: a blocked PNG must not leave a published HTML sibling behind
            # asserting a page that this publication no longer owns.
            if "html" in files:
                (directory / files["html"]).unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - plots are best-effort
            errors.append(f"{type(exc).__name__}: {exc}")
            logger.warning(
                "Plot %s page %s failed during PNG save: %s", plot_id, stem, exc
            )

    return files, errors, backend
```

The `lambda`s capture `figure`/`src` and are invoked synchronously inside the same
call, so there is no late-binding hazard. `FigureAdapter.close` is **not** called
here — the caller owns the figure's lifetime, and `save_png` already closes a
matplotlib figure in its own `finally` (double-close is safe: verified
`[Q1c] savefig+close+close OK`).

- [ ] **Step 6: Use it in the writer loop**

Add `plots_base: Path | None = None` to both `publish_plot_output` and
`_publish_plot_output_locked` (keyword-only, after `plot_class`), **and forward it
in the delegating call at `_writer.py:86-93`** (S1 — adding it to both signatures
without threading it leaves the inner default silently wrong).

Replace the per-page body with:

```python
    base = plots_base if plots_base is not None else directory

    for page in output.pages:
        label = page.label or page.key
        try:
            stem = safe_path_component(label)
        except Exception:
            stem = "page"
        base_stem = stem
        folded = stem.casefold()
        attempt = 0
        while folded in used and used[folded] != page.key:
            digest_input = page.key if attempt == 0 else f"{page.key}:{attempt}"
            digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8]
            stem = f"{base_stem}-{digest}"
            folded = stem.casefold()
            attempt += 1
        used[folded] = page.key

        try:
            files, errors, backend = _render_page(
                page.figure, directory, stem,
                plots_base=base,
                plot_id=plot_id,
                publication_guard=publication_guard,
                commit_guard=commit_guard,
            )
        except PlotPublicationBlocked:
            FigureAdapter.close(page.figure)
            raise
        FigureAdapter.close(page.figure)

        # S3: every swallowed error gets a durable record, not just a log line.
        for message in errors:
            record_plot_failure(
                base,
                binding_id=plot_id,
                plot_class=plot_class or plot_id,
                lifecycle="page",
                error=RuntimeError(message),
            )

        if not files:
            failed.append({
                "key": page.key,
                "label": page.label,
                "error": errors[0] if errors else "no renderer produced a file",
            })
            continue

        entry: dict[str, Any] = {
            "key": page.key,
            "label": page.label,
            "files": files,
            "backend": "matplotlib" if backend == "mpl" else "plotly",
            "metadata": dict(page.metadata),
        }
        if errors:
            # S2: one renderer failed while the other succeeded. The page is
            # published AND the failure is on the record.
            entry["partial"] = errors
        pages.append(entry)
```

Add `failed: list[dict[str, Any]] = []` beside the existing `pages` list, and
`from ._failures import record_plot_failure` at module level.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/unit/plotting/test_output_adapter.py -v`
Expected: the four new tests PASS. Pre-existing tests at `:80`, `:106` and
**`:124-172`** FAIL — Task 6 repairs all three. Do not "fix" them here.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/plotting/_pipeline/_adapter.py src/phenotypic/plotting/_pipeline/_writer.py tests/unit/plotting/test_output_adapter.py
git commit -m "feat(plotting): render every page through one shared _render_page

A Plotly page publishes interactive HTML always and a PNG when Chrome is
available; matplotlib publishes PNG only. HTML is attempted first
because it cannot fail for lack of a browser.

The rendering is extracted rather than inlined in the writer loop,
because the flat single-page image path in _coordinator.py never calls
publish_plot_output at all -- logic living only in the writer would miss
the commonest image plot entirely. Task 9 calls _render_page from there.

A partial failure (one renderer of two) is now recorded in the page's
'partial' key and in .failures.jsonl instead of being logged and
dropped, and a blocked PNG removes its published HTML sibling rather
than leaving an orphan."
```

## Task 6: Manifest `schema_version: 2`

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_writer.py:155-180`
- Modify: `tests/unit/plotting/test_output_adapter.py:79-120` (update existing assertions)
- Test: same file (append)

**Interfaces:**
- Consumes: `files`/`failed` lists (Task 5), `chrome_available` (Task 4).
- Produces: manifest `{"schema_version": 2, "plot_id", "class", "renderers", "pages": [{"key","label","files","backend","metadata"}], "failed": [{"key","label","error"}]}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/plotting/test_output_adapter.py`:

```python
def test_every_page_failing_yields_an_explanatory_manifest(tmp_path, monkeypatch) -> None:
    """A zero-page manifest must say why, not merely assert nothing."""
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _writer, publish_plot_output

    def _boom(*args, **kwargs):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(_writer.FigureAdapter, "save_html", _boom)
    monkeypatch.setattr(_writer.FigureAdapter, "save_png", _boom)

    output = PlotOutput(pages=(
        PlotPage(key="a", figure=go.Figure(), label="A"),
    ))
    manifest = publish_plot_output(output, tmp_path / "p", plot_id="p")

    assert manifest["schema_version"] == 2
    assert manifest["pages"] == []
    assert len(manifest["failed"]) == 1
    assert manifest["failed"][0]["key"] == "a"
    assert "renderer exploded" in manifest["failed"][0]["error"]


def test_renderers_records_why_png_is_missing(tmp_path, monkeypatch) -> None:
    import plotly.graph_objects as go

    from phenotypic.abc_.plotting import PlotOutput, PlotPage
    from phenotypic.plotting._pipeline import _backends, publish_plot_output

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    output = PlotOutput(pages=(
        PlotPage(key="a", figure=go.Figure(), label="A"),
    ))
    manifest = publish_plot_output(output, tmp_path / "p", plot_id="p")

    assert manifest["renderers"]["html"] == "available"
    assert "chrome" in manifest["renderers"]["png"].lower()
```

Then repair **three** pre-existing assertions, not two:

- `:80` — `files = [entry["file"] for entry in manifest["pages"]]` becomes
  `files = [entry["files"]["png"] for entry in manifest["pages"]]`
- `:106` — same substitution.
- **`:124-172`, `test_concurrent_plot_publications_do_not_mix_generations`** — this
  is the writer's only concurrency guard and Task 5 breaks it in a way that is NOT
  a `"file"` → `"files"` rename. It patches `FigureAdapter.backend_name`, which
  `_render_page` no longer calls; its local `_FakeFigure` then fails
  `figure_backend_of`'s `type(figure).__name__ == "Figure"` check, is classified
  `None`, and is diverted to `failed` — so `manifest["pages"]` is `[]` and the test
  dies at `generations.pop()` with `KeyError`, never reaching `:169`.
  Measured: `[Q11] backend_of(_FakeFigure) -> None`.

  **Fix:** patch the writer's view of the predicate instead, which keeps the test
  PNG-only and unchanged in spirit. Add beside the existing patches at `:133-137`:

  ```python
      monkeypatch.setattr(_writer, "figure_backend_of", lambda _figure: "mpl")
  ```

  and update its `page["file"]` read at `:169` to `page["files"]["png"]`. This
  works only because Task 5 Step 4 imports `figure_backend_of` at **module** level
  in `_writer.py`; a function-scope import would leave nothing to patch.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/plotting/test_output_adapter.py -v`
Expected: FAIL — `assert 1 == 2` on `schema_version`, `KeyError: 'renderers'`, and
`KeyError` from `generations.pop()` in the concurrency test.

**This is the last step at which a red `test_output_adapter.py` is expected.** From
Step 4 onward the whole file must be green; do not carry a red test past this task
on the grounds that the plan mentioned it.

- [ ] **Step 3: Build the manifest**

Replace the manifest construction in `_publish_plot_output_locked` (currently lines ~160-166) with:

```python
    renderers: dict[str, str] = {}
    backends = {
        figure_backend_of(page.figure) for page in output.pages
    }
    if "plotly" in backends:
        renderers["html"] = "available"
        renderers["png"] = (
            "available" if png_ok else "unavailable: chrome not found"
        )
    if "mpl" in backends:
        renderers["png"] = "available"

    manifest = {
        "schema_version": 2,
        "plot_id": plot_id,
        "class": plot_class or plot_id,
        "renderers": renderers,
        "pages": pages,
        "failed": failed,
    }
```

Note `figure_backend_of` is called on `page.figure` a second time here; matplotlib figures are closed during the page loop but `type()` inspection remains valid on a closed figure, so this is safe.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/plotting/test_output_adapter.py tests/unit/plotting/test_coordinator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/plotting/_pipeline/_writer.py tests/unit/plotting/test_output_adapter.py
git commit -m "feat(plotting)!: plot manifest schema_version 2

Per-page 'file' becomes 'files', since a Plotly page may now be two. The
manifest gains 'renderers' (the capability verdict for this directory)
and 'failed' (per-page reasons), so a plot directory with no PNGs
explains itself instead of looking merely empty.

Free to change: no production code reads the plot manifest; only
tests/unit/plotting/test_output_adapter.py does."
```

---

## Task 7: Preflight from CLI validation

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_backends.py` (add `preflight_plot_backends`)
- Modify: `src/phenotypic/_cli/_cli_validation.py:21-54`
- Test: `tests/unit/plotting/test_backends.py` (append), `tests/unit/cli/test_cli_validation.py` (append; create if absent)

**Interfaces:**
- Consumes: `chrome_available` (Task 4), `FigureSpec.backend` (Task 2).
- Produces: `preflight_plot_backends(pipeline) -> list[str]` — returns warning lines, raises `PlotBackendUnavailable` only for an unimportable declared backend.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/plotting/test_backends.py`:

```python
def test_preflight_warns_about_missing_chrome_without_raising(monkeypatch) -> None:
    import plotly.graph_objects as go
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import _backends
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    class _P(BaseModel, PlotMeas):
        @figure(title="T", backend="plotly", primary=True)
        def t(self, subject):
            return go.Figure()

    monkeypatch.setattr(_backends, "chrome_available", lambda: False)
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_P()])

    warnings_out = preflight_plot_backends(pipeline)

    assert len(warnings_out) == 1
    assert "_P" in warnings_out[0]
    assert "plotly_get_chrome" in warnings_out[0]


def test_preflight_is_silent_when_no_plots_are_configured() -> None:
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline._backends import preflight_plot_backends

    assert preflight_plot_backends(
        ImagePipeline(ops={"d": OtsuDetector()})
    ) == []


def test_preflight_raises_when_a_declared_library_is_missing(monkeypatch) -> None:
    """Missing Chrome is a warning; a missing BACKEND is not."""
    import builtins

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure as MplFigure
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline._backends import (
        PlotBackendUnavailable,
        preflight_plot_backends,
    )

    class _M(BaseModel, PlotMeas):
        @figure(title="T", backend="mpl", primary=True)
        def t(self, subject):
            return MplFigure()

    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_M()])

    real_import = builtins.__import__

    def _no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("no matplotlib here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)

    with pytest.raises(PlotBackendUnavailable, match="_M"):
        preflight_plot_backends(pipeline)
```

Add `import pytest` to the file's imports if the earlier tests have not already.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/plotting/test_backends.py -k preflight -v`
Expected: FAIL — `ImportError: cannot import name 'preflight_plot_backends'`

- [ ] **Step 3: Add the function**

Append to `src/phenotypic/plotting/_pipeline/_backends.py`:

```python
class PlotBackendUnavailable(RuntimeError):
    """A declared figure backend cannot be used at all."""


def preflight_plot_backends(pipeline: Any) -> list[str]:
    """Check declared figure backends once, before any image work.

    Missing Chrome is **not** an error: Plotly publishes HTML regardless, so
    the run is complete either way and the caller is told what it will not get.
    A declared backend whose library will not import IS an error -- that plot
    cannot publish at all.

    Args:
        pipeline: Pipeline whose normalized plot bindings are inspected.

    Returns:
        Human-readable warning lines, empty when everything is available.

    Raises:
        PlotBackendUnavailable: If a declared backend's library is missing.
    """
    plotly_ids: list[str] = []
    mpl_ids: list[str] = []
    for binding in pipeline.get_plots():
        plot = getattr(binding, "plot", None)
        iter_figures = getattr(plot, "iter_figures", None)
        if not callable(iter_figures):
            continue
        for spec in iter_figures():
            target = plotly_ids if spec.backend == "plotly" else mpl_ids
            if binding.id not in target:
                target.append(binding.id)

    if mpl_ids:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise PlotBackendUnavailable(
                f"{len(mpl_ids)} configured plots declare backend='mpl' but "
                f"matplotlib is not importable: {', '.join(mpl_ids)}"
            ) from exc

    if plotly_ids and not chrome_available():
        return [
            f"Chrome is not available; {len(plotly_ids)} Plotly plots will "
            f"publish HTML only, without PNG: {', '.join(plotly_ids)}. "
            "Install it for raster output with:  plotly_get_chrome"
        ]
    return []
```

Add `from typing import Any` to the imports, and add both new names to `__all__` and to `_pipeline/__init__.py`.

- [ ] **Step 4: Call it from CLI validation**

In `src/phenotypic/_cli/_cli_validation.py`, inside `validate_pipeline`'s `try` block, after the ops/meas check and before `return True, None`:

```python
        # Backends are checked once here rather than per figure during the
        # run: on SLURM this is the submitting process, so a pipeline that
        # will not rasterise says so before the array is submitted.
        from phenotypic.plotting._pipeline._backends import (
            preflight_plot_backends,
        )

        for line in preflight_plot_backends(pipeline):
            logger.warning(line)
```

Add `import logging` and `logger = logging.getLogger(__name__)` at module scope if absent. `PlotBackendUnavailable` is a `RuntimeError` and is caught by the existing `except Exception` at the end, becoming a clean `(False, message)` validation error — which is the intended UX.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/plotting/test_backends.py -v && uv run pytest tests/unit/cli/ -k valid -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/plotting/_pipeline/_backends.py src/phenotypic/plotting/_pipeline/__init__.py src/phenotypic/_cli/_cli_validation.py tests/unit/plotting/test_backends.py
git commit -m "feat(cli): announce missing raster capability before the run

preflight_plot_backends() runs inside validate_pipeline, which already
loads the pipeline once in the submitting process. Missing Chrome is a
warning naming the affected binding ids, not a failure -- HTML publishes
regardless. A declared backend whose library is absent still raises."
```

---

## Task 8: The durable failure record

**Files:**
- Create: `src/phenotypic/plotting/_pipeline/_failures.py`
- Modify: `src/phenotypic/sdk_/_io_constants.py`, `src/phenotypic/sdk_/__init__.py`
- Test: `tests/unit/plotting/test_failure_record.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `record_plot_failure(plots_base: Path, *, binding_id: str, plot_class: str, lifecycle: str, error: BaseException, dataset: str | None = None, image_stem: str | None = None) -> None`; `plot_failures_jsonl_path(output_dir) -> Path`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/plotting/test_failure_record.py`:

```python
"""The durable record that stops 'best-effort' meaning 'silent'."""
from __future__ import annotations

import json
from pathlib import Path

from phenotypic.plotting._pipeline._failures import record_plot_failure


def test_one_failure_writes_one_line(tmp_path: Path) -> None:
    record_plot_failure(
        tmp_path,
        binding_id="sym",
        plot_class="MeasureSymZones",
        lifecycle="image",
        error=TypeError("boom"),
        dataset="plate_a",
        image_stem="A01",
    )
    lines = (tmp_path / ".failures.jsonl").read_text().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["binding_id"] == "sym"
    assert entry["plot_class"] == "MeasureSymZones"
    assert entry["lifecycle"] == "image"
    assert entry["dataset"] == "plate_a"
    assert entry["image_stem"] == "A01"
    assert entry["error"] == "TypeError: boom"
    assert entry["ts"].endswith("Z")


def test_records_append_rather_than_replace(tmp_path: Path) -> None:
    for index in range(3):
        record_plot_failure(
            tmp_path,
            binding_id=f"b{index}",
            plot_class="C",
            lifecycle="measurements",
            error=ValueError(str(index)),
        )
    lines = (tmp_path / ".failures.jsonl").read_text().splitlines()
    assert [json.loads(line)["binding_id"] for line in lines] == ["b0", "b1", "b2"]


def test_aggregate_entries_omit_image_identity(tmp_path: Path) -> None:
    record_plot_failure(
        tmp_path,
        binding_id="b",
        plot_class="C",
        lifecycle="qc",
        error=ValueError("x"),
    )
    entry = json.loads((tmp_path / ".failures.jsonl").read_text())
    assert "dataset" not in entry
    assert "image_stem" not in entry


def test_recording_never_raises(tmp_path: Path) -> None:
    """A failure in the failure recorder must not escalate a soft failure."""
    unwritable = tmp_path / "nope"
    unwritable.write_text("I am a file, not a directory")

    record_plot_failure(
        unwritable,
        binding_id="b",
        plot_class="C",
        lifecycle="image",
        error=ValueError("x"),
    )  # must return normally
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/plotting/test_failure_record.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.plotting._pipeline._failures'`

- [ ] **Step 3: Add the path helper**

In `src/phenotypic/sdk_/_io_constants.py`, beside `PLOTLYJS_BUNDLE`:

```python
#: Durable record of swallowed plot failures: ``<plots>/.failures.jsonl``.
PLOT_FAILURES_JSONL: Final[str] = ".failures.jsonl"
```

and after `plotlyjs_bundle_path`:

```python
def plot_failures_jsonl_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/plots/.failures.jsonl``."""
    return plots_dir(output_dir) / PLOT_FAILURES_JSONL
```

Export both from `sdk_/__init__.py`.

- [ ] **Step 4: Write the module**

Create `src/phenotypic/plotting/_pipeline/_failures.py`:

```python
"""Durable record of plot failures that publication swallowed.

Plot output is best-effort by design: one bad figure must not kill a run that
produced good measurements. This module is what stops "best-effort" from
meaning "silent" -- a green run with missing plots carries the reason on disk
rather than only in a log nobody reads.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from phenotypic.sdk_._file_locking import exclusive_path_lock

logger = logging.getLogger(__name__)


def record_plot_failure(
    plots_base: Path,
    *,
    binding_id: str,
    plot_class: str,
    lifecycle: str,
    error: BaseException,
    dataset: str | None = None,
    image_stem: str | None = None,
) -> None:
    """Append one failure to ``<plots_base>/.failures.jsonl``.

    **Never raises.** This is called from ``except`` blocks whose whole purpose
    is to keep a plot failure from ending a run; letting the recorder throw
    would turn the soft failure it is describing into a hard one.

    Args:
        plots_base: Resolved ``deliverables/plots`` directory.
        binding_id: Stable plot binding id.
        plot_class: Producer class name.
        lifecycle: ``"image"``, ``"measurements"``, ``"analysis"``, or ``"qc"``.
        error: The exception that was swallowed.
        dataset: Dataset name, for the image lifecycle only.
        image_stem: Image stem, for the image lifecycle only.
    """
    from phenotypic.sdk_ import PLOT_FAILURES_JSONL

    entry: dict[str, str] = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "binding_id": binding_id,
        "plot_class": plot_class,
        "lifecycle": lifecycle,
        "error": f"{type(error).__name__}: {error}",
    }
    if dataset is not None:
        entry["dataset"] = dataset
    if image_stem is not None:
        entry["image_stem"] = image_stem

    try:
        plots_base.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, sort_keys=True) + "\n"
        with exclusive_path_lock(plots_base / ".failures.lock"):
            with (plots_base / PLOT_FAILURES_JSONL).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(line)
    except Exception:  # noqa: BLE001 - recording must never escalate
        logger.debug(
            "Could not record plot failure for %s", binding_id, exc_info=True
        )


__all__ = ["record_plot_failure"]
```

Export from `_pipeline/__init__.py`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/plotting/test_failure_record.py -v`
Expected: PASS, all four.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/plotting/_pipeline/_failures.py src/phenotypic/plotting/_pipeline/__init__.py src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/plotting/test_failure_record.py
git commit -m "feat(plotting): durable record of swallowed plot failures

record_plot_failure() appends one JSON line per swallowed failure to
deliverables/plots/.failures.jsonl, locked for concurrent SLURM workers.

It never raises: it is called from except blocks whose purpose is to
keep a plot failure from ending a run, so letting the recorder throw
would turn the soft failure it describes into a hard one."
```

---

## Task 9: Wire the coordinator, fix `emit_qc`, drop the asymmetry

**Files:**
- Modify: `src/phenotypic/plotting/_pipeline/_coordinator.py` (5 `except` blocks; `emit_qc` try boundary; `_publish_image_value`)
- Modify: `src/phenotypic/_cli/_cli_staged_workers.py:526-532`
- Test: `tests/unit/plotting/test_coordinator.py` (append)

**Interfaces:**
- Consumes: `record_plot_failure` (Task 8).
- Produces: no new public names.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/plotting/test_coordinator.py`:

```python
def test_a_raising_figure_leaves_the_run_green_and_is_recorded(tmp_path) -> None:
    import json

    import plotly.graph_objects as go
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotMeas, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import PlotCoordinator
    import pandas as pd

    class _Exploding(BaseModel, PlotMeas):
        @figure(title="T", backend="plotly", primary=True)
        def t(self, subject):
            raise RuntimeError("figure exploded")

    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, plots=[_Exploding()])
    PlotCoordinator(pipeline, tmp_path).emit_measurements(pd.DataFrame())

    record = tmp_path / "deliverables" / "plots" / ".failures.jsonl"
    entries = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["binding_id"] == "_Exploding"
    assert entries[0]["lifecycle"] == "measurements"
    assert "figure exploded" in entries[0]["error"]


def test_emit_qc_prelude_failure_is_recorded_and_the_loop_continues(tmp_path) -> None:
    """B3: the prelude is the ONLY window where F4's bug lives.

    The first draft of this test patched MeasurementInput.__init__, which is
    called at _coordinator.py:240 -- one line AFTER the binding assignment at
    :239. It therefore passed on the unfixed code and proved nothing. The
    prelude is :227-238, and modules.get at :233 is the way in.
    """
    import json

    import pandas as pd
    import plotly.graph_objects as go
    from pydantic import BaseModel

    from phenotypic import ImagePipeline
    from phenotypic.abc_.plotting import PlotQc, figure
    from phenotypic.detect import OtsuDetector
    from phenotypic.plotting._pipeline import AnalysisRegistry, PlotCoordinator

    class _FirstQc(BaseModel, PlotQc):
        @figure(title="First", backend="plotly", primary=True)
        def t(self, subject):
            return go.Figure()

    class _SecondQc(BaseModel, PlotQc):
        @figure(title="Second", backend="plotly", primary=True)
        def t(self, subject):
            return go.Figure()

    pipeline = ImagePipeline(
        ops={"d": OtsuDetector()}, plots=[_FirstQc(), _SecondQc()]
    )

    class _RaisingOnce(dict):
        """Raises inside the prelude for the first binding only."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def get(self, key, default=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("prelude exploded")
            return super().get(key, default)

    # Non-empty: `modules = successful_modules or {}` at :223 discards a falsy
    # mapping, which would skip the injection entirely.
    modules = _RaisingOnce({"unused": object()})

    PlotCoordinator(pipeline, tmp_path).emit_qc(
        pd.DataFrame(),
        AnalysisRegistry(tmp_path / "deliverables"),
        successful_modules=modules,
    )

    record = tmp_path / "deliverables" / "plots" / ".failures.jsonl"
    entries = [json.loads(line) for line in record.read_text().splitlines()]

    # The failure is recorded against the binding that actually failed...
    assert len(entries) == 1
    assert entries[0]["binding_id"] == "_FirstQc", (
        "must not be attributed to a previous or later binding"
    )
    assert "prelude exploded" in entries[0]["error"]

    # ...and the SECOND binding still emitted, which is the half that the
    # `binding = None` shape exists to preserve.
    assert (tmp_path / "deliverables" / "plots" / "_SecondQc").is_dir()


- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/plotting/test_coordinator.py -k "green or prelude" -v`
Expected: FAIL. On today's code the prelude failure reaches the handler at `:259`
with `binding` **unbound on the first iteration**, so an `UnboundLocalError` is
raised *from inside the exception handler*, replaces the original `RuntimeError`,
and escapes `emit_qc` — `_SecondQc` never emits and no record is written.

Confirm the test can actually fail before trusting it: this assertion set must go
red here, not merely green later.

- [ ] **Step 3: Add a recording helper to the coordinator**

Add to `PlotCoordinator`:

```python
    def _record_failure(
        self,
        binding: Any,
        error: BaseException,
        *,
        lifecycle: str,
        dataset: str | None = None,
        image_stem: str | None = None,
    ) -> None:
        """Log and durably record one swallowed plot failure."""
        from ._failures import record_plot_failure

        logger.warning(
            "Plot %s failed during %s", binding.id, lifecycle, exc_info=True
        )
        record_plot_failure(
            self._plots_base,
            binding_id=binding.id,
            plot_class=type(binding.plot).__name__,
            lifecycle=lifecycle,
            error=error,
            dataset=dataset,
            image_stem=image_stem,
        )
```

- [ ] **Step 4: Route all five handlers through it**

Replace each `except Exception:` handler body in `_coordinator.py` — at `emit_image` (~`:104`), `emit_analyses` (~`:176`), `emit_qc` (~`:259`), `emit_dependent_qc` (~`:303`), and `_emit_aggregate` (~`:327`) — with the matching call. For `emit_image`, which keeps `strict`:

```python
            except Exception as exc:  # noqa: BLE001 - plot output is best-effort
                if strict:
                    raise
                self._record_failure(
                    binding, exc,
                    lifecycle="image",
                    dataset=dataset,
                    image_stem=image_stem,
                )
```

and for the four aggregate handlers, e.g. in `_emit_aggregate`:

```python
        except Exception as exc:  # noqa: BLE001 - plot output is best-effort
            self._record_failure(binding, exc, lifecycle=lifecycle)
```

- [ ] **Step 5: Bind `binding` before the `try` (B3)**

**Decision on record:** the spec contradicted itself here — §3 argued the prelude
failure should propagate (narrow the `try`), §5's test row said the loop continues.
**The user chose §5.** A prelude failure is recorded and the remaining QC bindings
still emit.

Initialise `binding = None` before the `try`, keep the prelude inside it, and guard
the handler's use:

```python
        for configured in self._pipeline.get_plots():
            binding = None
            try:
                ref = configured.ref
                is_qc_ref = ref is not None and ref.slot == "qc"
                module_key = configured.id
                if is_qc_ref:
                    assert ref is not None and ref.key is not None
                    module_key = ref.key
                module = modules.get(module_key)
                plot = configured.plot
                if is_qc_ref and module is not None:
                    plot = module.check
                if not isinstance(plot, PlotQc):
                    continue

                binding = configured.model_copy(update={"plot": plot})
                ...  # input_ref, table, subject, _emit_aggregate as before
            except Exception as exc:  # noqa: BLE001 - plot output is best-effort
                self._record_failure_for(
                    binding_id=binding.id if binding is not None else configured.id,
                    plot_class=type(
                        binding.plot if binding is not None else configured.plot
                    ).__name__,
                    error=exc,
                    lifecycle="qc",
                )
```

`binding = None` is reassigned at the top of **every** iteration, which is what
stops the stale-previous-binding misattribution; the `configured.id` fallback is
what stops the unbound crash. Both halves are needed — the test asserts the
recorded id is `_FirstQc` precisely to catch an implementation that keeps the
previous iteration's value.

Add a `_record_failure_for(*, binding_id, plot_class, error, lifecycle, dataset=None, image_stem=None)`
sibling to `_record_failure` that takes the id directly rather than a binding
object; `_record_failure` becomes a thin wrapper over it.

**Recorded cost of this choice** (raised by the pre-dispatch review, accepted
knowingly): keeping the prelude inside the `try` means a genuine programming error
in it — a bad `assert`, a missing attribute — is swallowed and recorded as a plot
failure rather than surfacing. The alternative made the unbound case structurally
impossible instead of merely handled. If that swallowing ever hides a real defect,
the narrow-the-`try` shape is the fix, and it is recorded in `DEFERRED.md`.

- [ ] **Step 6: Route the flat image path through `_render_page` (B1)**

**This is the step that makes the whole change reach image plots.**
`_publish_image_value` has two branches: multi-page goes to `publish_plot_output`
(`:361-368`), and a single `"default"` page — which is what `normalize_plot_output`
produces for a bare figure, so *every* one of the 27 annotated sites — takes
`:370-380` and calls `FigureAdapter.save_png` **directly**. Nothing in Tasks 5 or 6
reaches it.

First, add `plots_base=self._plots_base` to the `publish_plot_output(...)` call in
the multi-page branch and in `_publish_aggregate`.

Then replace the flat branch (`:370-380`) with:

```python
        self._require_publication()
        base.mkdir(parents=True, exist_ok=True)
        files, errors, _backend = _render_page(
            output.pages[0].figure,
            base,
            output_stem,
            plots_base=self._plots_base,
            plot_id=binding.id,
            publication_guard=self._publication_guard,
            commit_guard=self._commit_guard,
        )
        FigureAdapter.close(output.pages[0].figure)
        for message in errors:
            record_plot_failure(
                self._plots_base,
                binding_id=binding.id,
                plot_class=type(binding.plot).__name__,
                lifecycle="image",
                error=RuntimeError(message),
                dataset=dataset,
                image_stem=image_stem,
            )
        if not files:
            raise RuntimeError(
                f"plot {binding.id!r} produced no file for "
                f"{dataset}/{image_stem}: {errors[0] if errors else 'no renderer'}"
            )
```

Import `_render_page` and `record_plot_failure` from `._writer` / `._failures` at
the top of `_coordinator.py`.

**Do NOT route this branch through `publish_plot_output` instead.** Two costs that
are invisible from the call site: it takes `exclusive_path_lock` on
`directory / ".publication.lock"`, and here the directory is
`plots/<id>/<dataset>/` — shared by every image in the dataset, so a 1,536-image
plate run would serialise on one interprocess lock that this path takes today not
at all. And it names files from `page.label or page.key`, so with the default
page's key of `"default"` every image in the dataset would collide on
`default.png`. The `<stem>-<hash>` naming exists precisely to prevent that, and
`test_image_plot_output_name_is_stable_for_reruns` pins it.

The raise at the end keeps `strict=True` meaningful for the staged path and is
caught by `emit_image`'s handler otherwise — so the failure is recorded once by
`_render_page`'s errors and once by the handler. That is intentional: the first
says which renderer failed, the second says which image.

- [ ] **Step 7: Remove the strict asymmetry**

In `src/phenotypic/_cli/_cli_staged_workers.py`, delete the `strict=True,` line from the `emit_image(...)` call at ~`:531`. All three call sites are then uniformly best-effort, which is safe because Task 7 moved the systemic cause upstream.

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/unit/plotting/ tests/unit/gui/test_plot_refresh.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/plotting/_pipeline/_coordinator.py src/phenotypic/_cli/_cli_staged_workers.py tests/unit/plotting/test_coordinator.py
git commit -m "fix(plotting): record swallowed failures; repair emit_qc's handler

All five coordinator handlers now write a durable record as well as a
log line, so a green run with missing plots carries the reason on disk.

emit_qc opened its try at the top of the loop but assigned `binding` ten
lines in, while the handler read binding.id -- so a prelude failure
raised UnboundLocalError FROM INSIDE the handler on the first iteration
(escaping emit_qc entirely, making the one documented best-effort path
not best-effort), and named the PREVIOUS plot on any later one. The
prelude is now outside the try, where a programming error belongs.

Drops strict=True from the staged GPU worker so all three emit_image
sites agree. Safe now that an unusable raster backend is detected at
validation rather than per figure."
```

---

## Task 10: End-to-end verification against a real pipeline

Tasks 1-9 are unit-tested in isolation. This task proves the pieces compose, and is the gate before the annotation phase.

**Files:**
- Create: `tests/integration/plotting/__init__.py` (empty)
- Create: `tests/integration/plotting/test_publication_end_to_end.py`

**Interfaces:**
- Consumes: everything from Tasks 1-9.
- Produces: nothing.

- [ ] **Step 0: Create the package**

```bash
mkdir -p tests/integration/plotting && touch tests/integration/plotting/__init__.py
```

`tests/integration` is in `testpaths` (`pyproject.toml:219`) and **all four** of its existing subdirectories carry an `__init__.py`. Omitting it makes this directory collect differently from its siblings.

- [ ] **Step 1: Write the test**

Create `tests/integration/plotting/test_publication_end_to_end.py`:

```python
"""One pipeline, one image, both renderings, one manifest."""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import pytest
from pydantic import BaseModel

from phenotypic import ImagePipeline
from phenotypic.abc_.plotting import PlotImage, figure
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.plotting._pipeline import PlotCoordinator, chrome_available


class _ObjectCount(BaseModel, PlotImage):
    @figure(title="Object count", backend="plotly", primary=True)
    def count(self, image):
        return go.Figure(go.Bar(x=["objects"], y=[image.num_objects]))


def test_an_image_plot_publishes_html_and_a_manifest(tmp_path: Path) -> None:
    pipeline = ImagePipeline(ops={"detect": OtsuDetector()}, plots=[_ObjectCount()])
    image = load_synth_yeast_plate()
    pipeline.apply(image, inplace=True)

    PlotCoordinator(pipeline, tmp_path).emit_image(
        image, dataset="ds 1", image_stem="plate_01", strict=True
    )

    plots = tmp_path / "deliverables" / "plots"
    pages = list((plots / "_ObjectCount" / "ds-1").glob("plate_01-*.html"))
    assert len(pages) == 1, "expected exactly one published HTML page"

    assert (plots / "plotly.min.js").is_file()
    assert 'src="../../plotly.min.js"' in pages[0].read_text()

    # PNG presence tracks the real capability of this machine.
    png = pages[0].with_suffix(".png")
    assert png.is_file() is chrome_available()

    # The run is green either way, and nothing was recorded as failed.
    assert not (plots / ".failures.jsonl").exists()
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/integration/plotting/test_publication_end_to_end.py -v`
Expected: PASS. On a machine without Chrome, the HTML exists and the PNG does not — and the assertion is written against `chrome_available()` so it is correct on both kinds of machine.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/plotting/__init__.py tests/integration/plotting/test_publication_end_to_end.py
git commit -m "test(plotting): end-to-end publication against a real pipeline

Proves the pieces compose: a PlotImage binding on a real image publishes
HTML, references the single hoisted bundle at the right relative depth,
publishes a PNG exactly when this machine can, and records no failure."
```

---

## Task 11: Annotate the 27 `src/` call sites — **Sonnet**

**Post-implementation.** Do not start until Tasks 1-10 are merged and green. Annotating against a signature that is still moving means annotating twice.

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/plot_accessor/_diagnostics_plotter.py` (12)
- Modify: `src/phenotypic/grid/_grid_fit_report.py` (6)
- Modify: `src/phenotypic/correction/_color_correction/_color_correction_report.py` (4)
- Modify: `src/phenotypic/measure/_measure_orientation_zones.py` (3)
- Modify: `src/phenotypic/_core/_image_parts/plot_accessor/_detect_modes_plotter.py` (1)
- Modify: `src/phenotypic/measure/_measure_symzones.py` (1)

**Interfaces:**
- Consumes: `figure(..., backend=...)` (Task 2).
- Produces: nothing.

- [ ] **Step 1: Confirm the count before changing anything**

Run: `grep -rn "@figure(" --include=*.py src/ | grep -v "_pht_plot.py" | wc -l`
Expected: `27`. The 28th raw hit is an error-message string at `_pht_plot.py:174`, not a decoration. If this prints anything other than 27, stop and report — the tree has moved since the plan was written.

- [ ] **Step 2: Annotate each site**

Add `backend="plotly"` immediately after the `title=` argument in every one of the
27 decorations. **Only 8 sites are single-line; 19 are multi-line.** Both forms:

```python
    # single-line (8 sites, all but two in grid/_grid_fit_report.py)
    @figure(title="Noise profile", backend="plotly", section="noise")

    # multi-line (19 sites) -- add backend as its own line; do NOT reflow
    @figure(
            title="Symmetric-radius overlay",
            backend="plotly",
            primary=True,
            controls={"base_layer": BASE_LAYER},
    )
```

**Every one of these is Plotly today — verified by the audit, which confirmed all 27 return Plotly figures.** If any site turns out to return a matplotlib figure, **stop and report it as a finding**. Do not annotate it `backend="mpl"` to make it pass: that would convert a bug this change exists to surface into a silently accepted behaviour.

- [ ] **Step 3: Verify every module still imports**

Run: `uv run python -c "
import importlib
for module in [
    'phenotypic._core._image_parts.plot_accessor._diagnostics_plotter',
    'phenotypic._core._image_parts.plot_accessor._detect_modes_plotter',
    'phenotypic.grid._grid_fit_report',
    'phenotypic.correction._color_correction._color_correction_report',
    'phenotypic.measure._measure_orientation_zones',
    'phenotypic.measure._measure_symzones',
]:
    importlib.import_module(module)
    print('ok', module)
"`
Expected: six `ok` lines. A `TypeError: figure() missing 1 required keyword-only argument` names the file and line of any site that was missed.

- [ ] **Step 4: Confirm no site was left behind**

**19 of the 27 sites are multi-line** decorations whose `@figure(` line carries no
arguments at all, so a grep that only reads that one line reports 19 failures on a
perfect sweep. Read the whole decorator instead:

Run: `grep -rn -A6 "@figure(" --include=*.py src/ | grep -v "_pht_plot.py" | grep -c 'backend='`
Expected: `27`.

Step 3 (importing all six modules) is the **real** guard — a missed site raises
`TypeError: figure() missing 1 required keyword-only argument: 'backend'` naming
the file and line. This step is a convenience cross-check.

**Do not reflow a multi-line decorator onto one line to satisfy a grep.** Two of
the `measure/` decorations would then exceed the configured line length.

- [ ] **Step 5: Run the affected tests**

Run: `uv run pytest tests/unit/core/test_detect_modes_plotter.py tests/unit/measure/ tests/unit/grid/ tests/unit/correction/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/plot_accessor/_diagnostics_plotter.py src/phenotypic/_core/_image_parts/plot_accessor/_detect_modes_plotter.py src/phenotypic/grid/_grid_fit_report.py src/phenotypic/correction/_color_correction/_color_correction_report.py src/phenotypic/measure/_measure_orientation_zones.py src/phenotypic/measure/_measure_symzones.py
git commit -m "chore(plotting): declare backend='plotly' at all 27 src call sites

Mechanical. Every site was already Plotly; none changes behaviour."
```

---

## Task 12: Annotate the 11 test call sites — **Sonnet**

**Files:**
- Modify: `tests/unit/abc_/plotting/test_pht_plot.py` (9)
- Modify: `tests/unit/viz/test_notebook_adapter.py` (2)

**Interfaces:**
- Consumes: `figure(..., backend=...)` (Task 2).
- Produces: nothing.

- [ ] **Step 1: Confirm the count**

Run: `grep -rn "@figure(" --include=*.py tests/ | grep -vc "backend="`
Expected: `11`. Note `tests/unit/abc_/plotting/test_figure_backend.py` is excluded from this count because its own sites already declare a backend.

- [ ] **Step 2: Annotate**

Add `backend="plotly"` to each. **One exception:** the site inside `test_pht_plot.py`'s `pytest.raises(ValueError)` block that asserts a bad control key is rejected (`:97`) — annotate it too; it must still reach the control-key check rather than failing earlier on the missing backend.

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/unit/abc_/plotting/ tests/unit/viz/ -v`
Expected: PASS, including the `test_pht_plot.py` file that Task 2 deliberately left red.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/abc_/plotting/test_pht_plot.py tests/unit/viz/test_notebook_adapter.py
git commit -m "test(plotting): declare backend at the 11 test call sites

Restores tests/unit/abc_/plotting/test_pht_plot.py, left red by the
required-argument change so its completion would be visible."
```

---

## Task 13: Rewrite `custom_plotter.md` — **Opus**

Not mechanical. This page is where readers learn the system, and it currently teaches the one authoring path that bypasses the decorator.

**Files:**
- Modify: `docs/source/extending/pages/custom_plotter.md`
- Modify: `src/phenotypic/abc_/CLAUDE.md`

**Interfaces:**
- Consumes: the whole change.
- Produces: nothing.

- [ ] **Step 1: Read the spec's §4 in full**

Read `docs/superpowers/specs/2026-09-20-figure-backend-routing/design.md`, section "§4 — Call sites and documentation". It enumerates every required correction. Read the audit at `docs/superpowers/reports/2026-09-20-pipeline-figure-storage/claim-verification.md` for the evidence behind the binding-id rule.

- [ ] **Step 2: Lead with the decorator**

The current example (`PlotColonyArea`, `:15-29`) overrides `inspect()` and never uses `@figure`, so it teaches the path that bypasses theming, the backend declaration, and the preflight. Add a `@figure(backend="plotly")` example as the **primary** form. Keep the `inspect()` override, demoted and explicitly labelled as the escape hatch for multi-page and matplotlib output.

- [ ] **Step 3: Correct the output-directory rule**

Replace the `plots/<ClassName>/` claim. The actual rule has three cases:
- a slot-owned object **with a key** → the slot key (`meas={"sym": …}` → `plots/sym/`);
- the singleton `model` slot, where `ref.key` is `None` by construction → the **class name**;
- an inline plot → the **class name**.

The `model` case is why the old text was not simply wrong for everything — it is right there, which is what made the error survive.

- [ ] **Step 4: Document the two renderings**

A Plotly binding publishes `.html` always and `.png` when Chrome is available; a matplotlib binding publishes `.png` only. Say where `plotly.min.js` lives (one file per run at `deliverables/plots/`) and that copying a plot directory elsewhere means bringing it along.

- [ ] **Step 5: Document the new surfaces**

`.failures.jsonl`, the manifest's `renderers` and `failed` keys at `schema_version: 2`, and the `report()` limitation on matplotlib providers.

- [ ] **Step 6: Update `abc_/CLAUDE.md`**

Add the required `backend` argument to the `@figure` convention documented there.

- [ ] **Step 7: Verify the example actually runs**

Extract the primary example into a scratch file and execute it against `load_synth_yeast_plate()`. A doc example that does not run is the defect this task exists to fix; do not skip this step.

- [ ] **Step 8: Commit**

```bash
git add docs/source/extending/pages/custom_plotter.md src/phenotypic/abc_/CLAUDE.md
git commit -m "docs(plotting): teach @figure first, and fix the output-dir rule

The example overrode inspect() and never used the decorator, teaching
the one path that bypasses theming, the backend declaration and the
preflight. @figure(backend=...) is now the primary form.

The plots/<ClassName>/ claim was never accurate -- it and _bindings.py
last changed in the same commit (71737850). The rule has three cases,
and the model slot's keyless ref is why <ClassName> looked right."
```

---

## Task 14: Full regression

**Files:** none modified.

- [ ] **Step 1: Confirm the tree is clean**

Run: `git status -s`
Expected: empty. A dirty tree makes the run unattributable.

- [ ] **Step 2: Lint the files this change touched**

Run: `uv run ruff check --fix src/phenotypic/abc_/plotting/ src/phenotypic/plotting/ src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_validation.py src/phenotypic/_cli/_cli_staged_workers.py tests/unit/plotting/ tests/unit/abc_/plotting/`

**Explicit paths only.** A bare `ruff check --fix` walks the whole repo and rewrites files this change never touched.

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/phenotypic`
Expected: no new errors relative to `main`. Capture the baseline first if unsure.

- [ ] **Step 4: Run the full suite as a Slurm job**

Use the **`run-phenotypic-test`** skill and the **`slurm-job`** skill. The suite is ~65 minutes, not two — it is a Slurm job, never an inline run. A committed batch script exists at `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.

Do NOT use `-n auto` (it reads the node's core count, not the allocation's) and do NOT use `-x` (it truncates a run that then gets recorded as a baseline). Set `QT_QPA_PLATFORM=offscreen`.

- [ ] **Step 5: Compare against the recorded baseline**

The captured baseline is 11,106 tests with 81 pre-existing failures, all outside `sdk_`, `_cli`, and `gui`. Any *new* failure is attributable to this change.

**Read the failure list, never the count.** Run each new failure in isolation before attributing it — a red shard mixes this change with contamination from unrelated files sharing it, and most isolated re-runs pass.

- [ ] **Step 6: Report**

Post the failure list by name, not a total. State no number that a command did not just print this turn.

---

## Execution: clusters, models, gates

Derived from the per-task `Files` / `Interfaces` blocks. This is a view of the
plan, not a separate artifact — if a task's files change, fix this table.

### Dependency DAG (file-level)

```
C1  T1,T2,T3   abc_/plotting/{_output,_pht_plot,__init__}.py, _pipeline/_adapter.py
C2  T4,T8      _pipeline/{_backends,_failures,__init__}.py, sdk_/{_io_constants,__init__}.py
C3  T5,T6      _pipeline/{_adapter,_writer}.py                  needs C1, C2
C4  T7         _pipeline/_backends.py, _cli/_cli_validation.py  needs C2
C5  T9         _pipeline/_coordinator.py, _cli/_cli_staged_workers.py  needs C2, C3
C6  T10        tests/integration/plotting/                      needs C1-C5
C7  T11        6 src modules                                    needs C1-C3 merged
C8  T12        2 test modules                                   needs C1-C3 merged
C9  T13        docs/, abc_/CLAUDE.md                            needs C1-C6
```

### Clusters

| # | Tasks | Shape | Model | Why |
|---|---|---|---|---|
| C1 | T1–T3 | Keystone + Leaf | Opus, high | The declaration itself. T3 is a 6-line guard in the file T2 just rewrote — folding it in avoids a second context load of `_pht_plot.py`. |
| C2 | T4, T8 | Keystone ×2 | Opus, high | Two new self-contained modules that both extend `_io_constants.py`. Shared file forces one cluster; both carry subtle contracts (memoisation, lock semantics, never-raise). |
| C3 | T5, T6 | Keystone | Opus, high | Both rewrite the same function in `_writer.py`. Splitting would mean a mid-function commit that cannot be green. |
| C4 | T7 | **Seam** | Opus, high | Changes CLI validation behaviour for every run. Isolated for its own gate despite being small — risk is not size. |
| C5 | T9 | **Seam** | Opus, high | Five exception handlers, a `try` boundary move, and a cross-file `strict=` removal. The highest-risk wiring in the change. |
| C6 | T10 | Verification | Opus, high | Proves C1–C5 compose. Doubles as the phase gate. |
| C7 | T11 | **Sweep** | Sonnet, medium | 27 mechanical annotations, verified by import + a zero-count grep. |
| C8 | T12 | **Sweep** | Sonnet, medium | 11 mechanical annotations. |
| C9 | T13 | Leaf (judgment) | Opus, high | Prose and a three-case rule the audit found stated wrongly. Not mechanical. |

### Gates

- **Pre-dispatch:** `plan-reviewer` over the plan. Resolve criticals before any code.
- **Per cluster:** orchestrator reads the diff and runs the cluster's own tests.
- **Deep, after C5:** `implementation-test-reviewer` over the combined C1–C5 diff —
  the phase added tests, so the question is whether they can fail, not whether
  they pass.
- **After C9:** one `code-simplifier` pass, quality only.
- **End:** T14 full sharded regression as a Slurm job.

### Parallelism: none taken, deliberately

C1∥C2 and C7∥C8 have zero file overlap and are genuine candidates. Both are
declined:

- The DAG is near-linear — C3 needs both C1 and C2 — so parallelising the one
  early pair saves a single cluster of wall-clock against the cost of a second
  worktree and a merge.
- Same-worktree parallelism is the failure the `orchestrate-subagent` skill
  names directly: one agent runs verification while another is mid-edit, and the
  resulting transient failure costs more to diagnose than the time saved.

C7∥C8 are mechanical and fast; sequencing them costs minutes.
