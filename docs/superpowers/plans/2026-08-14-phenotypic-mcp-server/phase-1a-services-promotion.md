# Phase 1a — Promote the Dash-free tier to `phenotypic/_services/`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Every task below ends with a reviewer step.** See the plan README's
> *Review protocol* — a task with an unaddressed correctness finding does not
> hand off to the next one.

**Implements:** §7 P2, §1.4. **Spec:**
[`../../specs/2026-08-12-phenotypic-mcp-server/01-architecture.md`](../../specs/2026-08-12-phenotypic-mcp-server/01-architecture.md)

**Goal:** Move the six capabilities the MCP server and the GUI both need into a
tier that imports no Dash, so the server can depend on a tested API instead of
another surface's private modules.

**Why promotion rather than importing in place:** `phenotypic.mcp` importing
`phenotypic.gui._operation_registry` would make one user-facing surface's private
module the de-facto API of another, with no test protecting the boundary.
Promotion costs one refactor and buys a layer that can be versioned on its own
terms (§1.4).

**The eager `__init__.py` files are the root cause, and they ARE in scope.**
`gui/shell/__init__.py:17-20` and `gui/tune/__init__.py:18` eagerly import their
Dash app factories, so importing *any* module from those packages drags `dash`,
`dash_bootstrap_components`, `flask`, and `werkzeug` into `sys.modules` even when
the module's own content is clean. Measured, one subprocess per module:

```
phenotypic.gui.shell._sandbox      ['dash','dash_bootstrap_components','flask','werkzeug']
phenotypic.gui.shell._classifier   [same]
phenotypic.gui.tune._space         [same]
phenotypic.gui.run_console._state  [same]
phenotypic.gui._config             CLEAN
phenotypic.gui._operation_registry CLEAN
```

**An earlier draft of this document called that "deferred cleanup, not in scope",
and it was wrong** — wrong in a way that surfaces as a red purity gate mid-cluster,
on a task whose own instructions forbid weakening the gate to get past it. Nine
promoted modules import back into those packages. Concretely: Task 5 moves
`RunRegistry`, and `_runs_registry.py:59` does
`from phenotypic.gui.shell._classifier import classify`; that one import executes
`gui/shell/__init__.py` and fails the Task 1 gate. **Task 2 is necessary but not
sufficient** — the dependency Task 5 declares on Task 2 is real, but the mechanism
that actually bites is the package `__init__`, not `IMAGE_EXTS`.

**Task 2.5 fixes it**, using the same `__getattr__` pattern `gui/__init__.py:31`
and `gui/run_console/__init__.py:25` already use — roughly 20 lines, ordered
before Task 5. That is far cheaper than the alternative, which is expanding Task 7
by five modules: `_setup_authoring.py:20-28` alone reaches `gui._config`,
`gui.shell._metadata_context`, `gui.shell._sandbox` (including two privates),
`gui.shell._source_context`, and `gui.tune._space`.

The MCP server still never imports `phenotypic.gui` — that half of the original
claim stands. What changed is that `_services` cannot avoid it either, so the leak
has to be **fixed** rather than routed around.

**Task order changed (B2):** Task 8 now runs **before** Task 7.
`gui/tune/_command.py:13-17` imports from `gui.tune._run_argv`, which Task 8
promotes; in the original order Task 7's output would import
`phenotypic.gui.tune._run_argv`, whose package `__init__.py:19` eagerly imports
`._app` → dash, failing the gate again.

See [review-findings.md](review-findings.md) for the full register.

---

## File structure this phase creates

| File | Responsibility |
|---|---|
| `src/phenotypic/_services/__init__.py` | Package marker. **Lazy** — no eager submodule imports, or the purity gate becomes meaningless the first time one module grows a heavy dependency |
| `src/phenotypic/_services/registry.py` | Operation discovery + param introspection (from `gui/_operation_registry.py`) |
| `src/phenotypic/_services/sandbox.py` | `SandboxRoot` filesystem sandbox (from `gui/shell/_sandbox.py`) |
| `src/phenotypic/_services/runs.py` | `RunRegistry` + `LocalRunner` (from `gui/shell/_runs_registry.py`, `gui/run_console/_runner.py`) |
| `src/phenotypic/_services/tune_spec.py` | Spec authoring, validation, export, **and the pure half of `_space.py`** |
| `src/phenotypic/_services/argv.py` | `RunConsoleState` + `to_argv`, and the tune argv builder |
| `src/phenotypic/gui/tune/_space_view.py` | The Dash half of `_space.py`, importing the pure half back |
| `tests/unit/services/test_import_purity.py` | The gate that makes all of the above permanent |
| `tests/unit/services/test_shim_equivalence.py` | Catches the `_REGISTRY` double-singleton failure |

Each `gui/*` module left behind becomes a **re-export shim**, so GUI behaviour is
unchanged and its 43 `SandboxRoot` / 15 `RunRegistry` call sites keep working
untouched.

---

### Task 1: The import-purity gate

Written **first**, before anything moves, so it fails for the right reason and is
proven able to fail. This is the test that makes the `_space.py` split (Task 6)
permanent rather than aspirational.

**Files:**
- Create: `src/phenotypic/_services/__init__.py`
- Create: `tests/unit/services/__init__.py` (empty — every `tests/unit/*` subdir
  is a package here; `tests/unit/cli`, `core`, `enhance`, `gui` all carry one)
- Create: `tests/unit/services/test_import_purity.py`

**Interfaces:**
- Produces: the `phenotypic._services` package namespace every later task moves into.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_import_purity.py
"""The boundary that makes `_services` a layer rather than a folder."""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

FORBIDDEN = ("dash", "dash_bootstrap_components", "flask", "werkzeug")

# One subprocess per module: a single process would let module A's clean import
# be vouched for by module B having already been imported, and vice versa.
_PROBE = """
import importlib, sys
importlib.import_module({module!r})
leaked = sorted(m for m in {forbidden!r} if m in sys.modules)
print(",".join(leaked))
"""

def _service_modules() -> list[str]:
    import phenotypic._services as services

    return [
        f"phenotypic._services.{m.name}"
        for m in pkgutil.iter_modules(services.__path__)
    ]

def test_services_package_exists_and_is_lazy():
    import phenotypic._services as services

    assert services.__path__, "phenotypic._services must be a package"

@pytest.mark.parametrize("module", _service_modules())
def test_service_module_imports_no_dash(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    leaked = [name for name in proc.stdout.strip().split(",") if name]
    assert not leaked, f"{module} dragged {leaked} into sys.modules"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/services/test_import_purity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic._services'`
at collection time.

- [ ] **Step 3: Create the package**

```python
# src/phenotypic/_services/__init__.py
"""Dash-free service tier shared by the GUI and the MCP server.

Modules here import only the standard library and other ``phenotypic``
internals. Nothing in this package may import ``dash``,
``dash_bootstrap_components``, ``flask``, or ``werkzeug`` — the boundary is
enforced by ``tests/unit/services/test_import_purity.py``.

This module is deliberately empty of submodule imports: eagerly importing them
here would make one heavy dependency contaminate every consumer, which is the
failure this tier exists to prevent.
"""

from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/services/test_import_purity.py -v`
Expected: PASS — `test_services_package_exists_and_is_lazy` passes; the
parametrized test collects zero cases because the package is empty. **That is
correct at this point and is the last time it is acceptable.**

- [ ] **Step 5: Prove the gate can fail**

Temporarily create `src/phenotypic/_services/_scratch.py` containing
`import dash`, re-run, and confirm `test_service_module_imports_no_dash[...]`
FAILS with `dragged ['dash'] into sys.modules`. Delete the file.
**Do not skip this step** — a purity gate that cannot fail is the exact class of
worthless test §6.5 names.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_services/__init__.py tests/unit/services/test_import_purity.py
git commit -m "test(services): add the import-purity gate before the tier exists"
```

- [ ] **Step 7: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 2: Relocate `IMAGE_EXTS` below the GUI

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (add the constant)
- Modify: `src/phenotypic/gui/_config.py:429` (re-export instead of define)
- Modify: `src/phenotypic/gui/shell/_classifier.py:34` (repoint the import)
- Test: `tests/unit/services/test_image_exts_relocation.py`

**Why:** `rehydrate_from_sandbox` — the boot-recovery method §2.4 depends on —
calls `classify()`, and `_classifier.py:34` reaches `IMAGE_EXTS` through
`gui/builder/_directory_browser.py`, which imports `dash` at `:20-21`. Promoting
`runs.py` (Task 5) without this drags Dash in behind it.

**Drift note (DR1):** the spec describes `IMAGE_EXTS` as *defined* in
`_directory_browser.py`. It is not, any more: it is defined at
`gui/_config.py:429` and re-exported from `_directory_browser.py:23` for
back-compat. `gui/_config.py` is already Dash-free. It still cannot be the home,
because `_services` importing from `phenotypic.gui` would invert the layering the
architecture diagram asserts — so it moves one level further down, to `sdk_`.

**Interfaces:**
- Produces: `phenotypic.sdk_._io_constants.IMAGE_EXTS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_image_exts_relocation.py
def test_image_exts_lives_in_sdk():
    from phenotypic.sdk_._io_constants import IMAGE_EXTS

    assert isinstance(IMAGE_EXTS, frozenset)
    assert ".tif" in IMAGE_EXTS

def test_every_alias_is_the_same_object():
    """Three import paths, one object — a copy would drift silently."""
    from phenotypic.gui._config import IMAGE_EXTS as via_config
    from phenotypic.gui.builder._directory_browser import IMAGE_EXTS as via_browser
    from phenotypic.sdk_._io_constants import IMAGE_EXTS as canonical

    assert via_config is canonical
    assert via_browser is canonical

def test_classifier_does_not_reach_through_the_dash_module():
    """The whole point: classify() must not pull in _directory_browser."""
    import inspect

    from phenotypic.gui.shell import _classifier

    source = inspect.getsource(_classifier)
    assert "_directory_browser" not in source
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/services/test_image_exts_relocation.py -v`
Expected: FAIL — `ImportError: cannot import name 'IMAGE_EXTS'` from
`phenotypic.sdk_._io_constants`.

- [ ] **Step 3: Move the definition**

Cut the `IMAGE_EXTS: frozenset[str] = frozenset(...)` literal from
`gui/_config.py:429` into `sdk_/_io_constants.py` beside the other filename
constants, keeping its docstring. Then in `gui/_config.py`:

```python
from phenotypic.sdk_._io_constants import IMAGE_EXTS  # re-exported for back-compat
```

and in `gui/shell/_classifier.py`, replace line 34:

```python
from phenotypic.sdk_._io_constants import IMAGE_EXTS
```

Leave `_directory_browser.py:23` alone — it already re-exports from `_config`,
which now re-exports from `sdk_`, so the object identity chain holds.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services/test_image_exts_relocation.py tests/unit/gui -q`
Expected: PASS, and no GUI regression.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/gui/_config.py \
        src/phenotypic/gui/shell/_classifier.py \
        tests/unit/services/test_image_exts_relocation.py
git commit -m "refactor(sdk): move IMAGE_EXTS below the GUI so classify() is Dash-free"
```

- [ ] **Step 6: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 2.5: Make the eager GUI package `__init__`s lazy

**Added after review (B1).** Without this, Task 5 fails the Task 1 gate and Task 7
grows by five modules. It is the root-cause fix the original draft deferred.

**Files:**
- Modify: `src/phenotypic/gui/shell/__init__.py`
- Modify: `src/phenotypic/gui/tune/__init__.py`
- Test: `tests/unit/services/test_lazy_gui_packages.py`

**Interfaces:**
- Produces: no new symbols. `phenotypic.gui.shell` and `phenotypic.gui.tune` keep
  **exactly** their current public names; only the import timing changes.

**The pattern is already in this repo — copy it, do not invent one.**
`gui/run_console/__init__.py` is the cleanest template: a `TYPE_CHECKING` import
for the type checker, `__all__` unchanged, and a module-level `__getattr__` (PEP
562) that imports on first attribute access. `gui/__init__.py:31` uses the same
idiom at larger scale.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_lazy_gui_packages.py
"""The eager package __init__s are why a content-clean module still drags Dash.

Task 5 promotes RunRegistry, whose _runs_registry.py:59 imports `classify` from
gui.shell._classifier. That single import executes gui/shell/__init__.py. If the
package is eager, _services/runs.py fails the Task 1 purity gate through no fault
of its own content.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

FORBIDDEN = ("dash", "dash_bootstrap_components", "flask", "werkzeug")

_PROBE = """
import importlib, sys
importlib.import_module({module!r})
print(",".join(sorted(m for m in {forbidden!r} if m in sys.modules)))
"""


@pytest.mark.parametrize(
    "module",
    [
        "phenotypic.gui.shell._sandbox",
        "phenotypic.gui.shell._classifier",
        "phenotypic.gui.shell._runs_registry",
        "phenotypic.gui.tune._space",
        "phenotypic.gui.tune._run_argv",
    ],
)
def test_submodule_import_does_not_execute_the_dash_app_factory(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, forbidden=FORBIDDEN)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    leaked = [n for n in proc.stdout.strip().split(",") if n]
    assert not leaked, f"{module} dragged {leaked} in via its package __init__"


@pytest.mark.parametrize(
    ("package", "symbol"),
    [
        ("phenotypic.gui.shell", "create_app"),
        ("phenotypic.gui.shell", "launch_gui"),
        ("phenotypic.gui.shell", "SandboxRoot"),
        ("phenotypic.gui.shell", "ToolSession"),
        ("phenotypic.gui.tune", "create_app"),
        ("phenotypic.gui.tune", "TuneRunRoot"),
        ("phenotypic.gui.tune", "TuneRunRootError"),
    ],
)
def test_public_api_is_unchanged(package: str, symbol: str) -> None:
    """Laziness must be invisible: every name still resolves on access."""
    import importlib

    assert getattr(importlib.import_module(package), symbol) is not None


def test_unknown_attribute_still_raises_attribute_error() -> None:
    import phenotypic.gui.shell as shell

    with pytest.raises(AttributeError):
        shell.definitely_not_a_real_symbol
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --no-sync pytest tests/unit/services/test_lazy_gui_packages.py -v`
Expected: the five `test_submodule_import_does_not_execute_the_dash_app_factory`
cases FAIL with `dragged ['dash', 'dash_bootstrap_components', 'flask',
'werkzeug'] in via its package __init__`. The API tests should already pass —
they are the regression guard, not the target.

- [ ] **Step 3: Convert both packages**

`gui/shell/__init__.py` — replace the four eager imports at `:17-20`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type-checker only; never executed at runtime
    from phenotypic.gui.shell._app import create_app  # noqa: F401
    from phenotypic.gui.shell._launcher import launch_gui, main  # noqa: F401
    from phenotypic.gui.shell._sandbox import SandboxRoot  # noqa: F401
    from phenotypic.gui.shell._session import ToolSession  # noqa: F401

__all__ = ["SandboxRoot", "ToolSession", "create_app", "launch_gui", "main"]

_LAZY = {
    "create_app": ("phenotypic.gui.shell._app", "create_app"),
    "launch_gui": ("phenotypic.gui.shell._launcher", "launch_gui"),
    "main": ("phenotypic.gui.shell._launcher", "main"),
    "SandboxRoot": ("phenotypic.gui.shell._sandbox", "SandboxRoot"),
    "ToolSession": ("phenotypic.gui.shell._session", "ToolSession"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(name) from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)
```

Keep the existing `__all__` contents exactly as they are — read them from the file
rather than trusting this sketch.

`gui/tune/__init__.py` — same shape. **Only `create_app` needs to be lazy**; it is
the one reaching `._app`. Verify whether `._run_root` is import-clean and, if it
is, leave `TuneRunRoot` / `TuneRunRootError` eager:

```bash
uv run --no-sync python -c "
import importlib, sys
importlib.import_module('phenotypic.gui.tune._run_root')
print([m for m in ('dash','flask','werkzeug') if m in sys.modules] or 'CLEAN')"
```

If it reports CLEAN, keep those two eager and make only `create_app` lazy — a
smaller change is a smaller regression surface. **Preserve the module docstring**;
it documents the package's optuna-free import contract, which is a separate
guarantee this task must not disturb.

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/unit/services/test_lazy_gui_packages.py -v`
Expected: all cases PASS — the five submodules now import clean, and every public
name still resolves.

- [ ] **Step 5: Prove the GUI did not notice**

Run: `uv run --no-sync pytest tests/unit/gui tests/integration/gui -q`
Expected: PASS, unchanged. A lazy `__init__` that breaks a real Dash call site is
worse than the leak it fixed.

Then confirm the app factories still work end to end:

```bash
uv run --no-sync python -c "
from phenotypic.gui.shell import create_app
from phenotypic.gui.tune import create_app as tune_app
print('both factories resolve:', callable(create_app), callable(tune_app))"
```

- [ ] **Step 6: Prove the new test can fail**

Restore one eager import in `gui/shell/__init__.py`, confirm the parametrized
purity cases FAIL again, then revert. This is the test the rest of the phase leans
on — an unverified version of it is what let the original scoping error through.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/gui/shell/__init__.py src/phenotypic/gui/tune/__init__.py \
        tests/unit/services/test_lazy_gui_packages.py
git commit -m "refactor(gui): make the shell and tune package __init__s lazy

Importing any submodule of gui.shell or gui.tune executed the package __init__
and pulled in dash/flask/werkzeug, so a content-clean module could not be
promoted without dragging the GUI stack behind it."
```

- [ ] **Step 8: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). This one matters more than most: it changes import
timing for two packages with many call sites, and "the GUI still works" is a claim
the whole phase now rests on. Require it to check that no public name was dropped
from either `__all__`, that the `TYPE_CHECKING` imports do not execute at runtime,
and that Step 6's mutation was actually performed.

---

### Task 3: Promote the operation registry

**Files:**
- Create: `src/phenotypic/_services/registry.py` (moved from `gui/_operation_registry.py`)
- Modify: `src/phenotypic/gui/_operation_registry.py` → re-export shim
- Test: `tests/unit/services/test_shim_equivalence.py`

**Interfaces:**
- Produces: `phenotypic._services.registry.get_registry() -> OperationRegistry`,
  `OperationRegistry`, `OperationInfo`, `ParamInfo`

**The failure this task must not cause.** `_REGISTRY` is a module-level global
(`:811-823`). If the shim re-*creates* a registry instead of re-*exporting* the
function, two singletons exist and `discover()` runs twice — and nothing else
would notice. §1.4 chose the module global over the GUI's per-app
`app.server.config[CFG_OPERATION_REGISTRY]` caching precisely because a stdio
server has no analogue of the latter.

`discover()` is lazy today and **must stay lazy** — it is only called on first
`get_registry()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_shim_equivalence.py
"""Each gui.* shim must re-export the same object, not a parallel one."""

def test_get_registry_is_one_function():
    from phenotypic._services.registry import get_registry as canonical
    from phenotypic.gui._operation_registry import get_registry as shim

    assert shim is canonical

def test_get_registry_is_one_singleton():
    from phenotypic._services.registry import get_registry as canonical
    from phenotypic.gui._operation_registry import get_registry as shim

    assert shim() is canonical()

def test_discovery_stays_lazy():
    """Importing the module must not walk eight packages."""
    import importlib

    import phenotypic._services.registry as registry

    importlib.reload(registry)
    assert registry._REGISTRY is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/services/test_shim_equivalence.py -v`
Expected: FAIL — `No module named 'phenotypic._services.registry'`.

- [ ] **Step 3: Move the module and write the shim**

```bash
git mv src/phenotypic/gui/_operation_registry.py src/phenotypic/_services/registry.py
```

Fix the moved module's own relative imports, then create the shim:

```python
# src/phenotypic/gui/_operation_registry.py
"""Back-compat shim. The implementation lives in :mod:`phenotypic._services.registry`.

Re-exports the *same* objects — in particular the ``_REGISTRY`` singleton lives
in the promoted module's namespace, so both import paths share one instance.
"""

from __future__ import annotations

from phenotypic._services.registry import (  # noqa: F401
    OperationInfo,
    OperationRegistry,
    ParamInfo,
    get_registry,
)

__all__ = ["OperationInfo", "OperationRegistry", "ParamInfo", "get_registry"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services tests/unit/gui -q`
Expected: PASS, including the purity gate now collecting a real module.

- [ ] **Step 5: Prove the singleton test can fail**

Temporarily give the shim its own `_REGISTRY = None` and a local `get_registry`;
confirm `test_get_registry_is_one_singleton` FAILS. Revert.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/_services/registry.py src/phenotypic/gui/_operation_registry.py \
           tests/unit/services/test_shim_equivalence.py
git commit -m "refactor(services): promote the operation registry, shim the GUI path"
```

- [ ] **Step 7: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 4: Promote `SandboxRoot`

**Files:**
- Create: `src/phenotypic/_services/sandbox.py` (from `gui/shell/_sandbox.py`)
- Modify: `src/phenotypic/gui/shell/_sandbox.py` → re-export shim
- Test: extend `tests/unit/services/test_shim_equivalence.py`

**Interfaces:**
- Produces: `phenotypic._services.sandbox.SandboxRoot`

`SandboxRoot` **is the entire security boundary** of the MCP server (§6.4: there
is no authentication), so it gets an adversarial test of its own in Phase 2A.
Here it only moves; 43 GUI call sites keep importing through the shim.

- [ ] **Step 1: Add the failing assertion**

```python
def test_sandbox_root_is_one_class():
    from phenotypic._services.sandbox import SandboxRoot as canonical
    from phenotypic.gui.shell._sandbox import SandboxRoot as shim

    assert shim is canonical
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_shim_equivalence.py::test_sandbox_root_is_one_class -v`
Expected: FAIL — `No module named 'phenotypic._services.sandbox'`.

- [ ] **Step 3: Move and shim**

```bash
git mv src/phenotypic/gui/shell/_sandbox.py src/phenotypic/_services/sandbox.py
```

```python
# src/phenotypic/gui/shell/_sandbox.py
"""Back-compat shim; implementation in :mod:`phenotypic._services.sandbox`."""

from __future__ import annotations

from phenotypic._services.sandbox import SandboxRoot  # noqa: F401

__all__ = ["SandboxRoot"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services tests/unit/gui tests/integration/gui -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(services): promote SandboxRoot"
```

- [ ] **Step 6: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 5: Promote `RunRegistry` and `LocalRunner`

**Files:**
- Create: `src/phenotypic/_services/runs.py` (from `gui/shell/_runs_registry.py` + `gui/run_console/_runner.py`)
- Modify: both originals → re-export shims
- Test: extend `tests/unit/services/test_shim_equivalence.py`

**Interfaces:**
- Produces: `RunRegistry` (`.allocate`, `.compare_and_set`,
  `.rehydrate_from_sandbox`, `.observe_local_exit`, `.cancel_generation`),
  `RunRecord`, `LocalRunner` (`.start`, `.stop`, `.snapshot_log`)

**Depends on Task 2.** `rehydrate_from_sandbox` → `classify()` →
`_classifier.py` → `IMAGE_EXTS`; without Task 2 this import chain reaches Dash
and the purity gate fails on `runs.py`. That failure is the gate working — do
not weaken it, do Task 2 first.

Reusing `RunRegistry` is what buys the server interprocess locking on allocation,
nonterminal-generation rejection, generation-fenced CAS, and boot recovery
(§2.4). None of it is reimplemented.

- [ ] **Step 1: Add the failing assertions**

```python
def test_run_registry_is_one_class():
    from phenotypic._services.runs import RunRegistry as canonical
    from phenotypic.gui.shell._runs_registry import RunRegistry as shim

    assert shim is canonical

def test_local_runner_is_one_class():
    from phenotypic._services.runs import LocalRunner as canonical
    from phenotypic.gui.run_console._runner import LocalRunner as shim

    assert shim is canonical
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/services/test_shim_equivalence.py -v`
Expected: FAIL — `No module named 'phenotypic._services.runs'`.

- [ ] **Step 3: Move both into one module, shim both originals**

`RunRegistry` and `LocalRunner` change together (allocate → start → CAS is one
flow), so they live in one file per the plan's file-structure rule. Move the
contents of both modules into `_services/runs.py`, then reduce each original to a
re-export shim in the shape of Task 4's.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services tests/unit/gui tests/integration/gui -q`
Expected: PASS — in particular
`tests/integration/gui/test_recent_runs_rehydrate.py`, which exercises the
`classify()` chain Task 2 untangled.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(services): promote RunRegistry and LocalRunner"
```

- [ ] **Step 6: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 6: Split `gui/tune/_space.py` into a pure half and a view half

**Files:**
- Create: `src/phenotypic/_services/tune_spec.py` (pure half — extended by Task 7)
- Create: `src/phenotypic/gui/tune/_space_view.py` (Dash half)
- Modify: `src/phenotypic/gui/tune/_space.py` → shim re-exporting both halves
- Modify: `src/phenotypic/gui/tune/_layout.py:642`, `_callbacks.py:1388,2227` if
  they import view symbols directly
- Test: `tests/unit/services/test_space_split.py`

**This is the one genuinely new refactor in P2, not a move.** `_space.py` carries
`import dash_bootstrap_components as dbc` and `from dash import html` at
`:33-34`, and the split is **forced**: `_setup_authoring.py:28` does
`from phenotypic.gui.tune._space import apply_space_edits, space_to_spec`, so
Task 7 cannot promote `_setup_authoring` without either splitting this file or
dragging Dash into `_services`.

| Half | Symbols (verified line numbers) | Destination |
|---|---|---|
| Pure | `_build_search_space` (`:134`), `apply_space_edits` (`:161`), `space_to_spec` (`:209`) | `_services/tune_spec.py` |
| View | `_knob_form` (`:396`), `setup_knob_forms` (`:468`), `build_space_view` (`:503`) | `gui/tune/_space_view.py` |

`_load_space_source` (imported at `_callbacks.py:2227`) is pure — it reads a spec
file — so it goes with the pure half.

**Interfaces:**
- Produces: `phenotypic._services.tune_spec.space_to_spec`,
  `.apply_space_edits`, `._build_search_space`, `._load_space_source`
- Consumes: nothing from earlier tasks

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_space_split.py
import inspect
import subprocess
import sys

def test_pure_half_is_importable_without_dash():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import phenotypic._services.tune_spec as t; import sys;"
         " print('dash' in sys.modules)"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"

def test_pure_symbols_moved():
    from phenotypic._services.tune_spec import apply_space_edits, space_to_spec

    assert callable(space_to_spec)
    assert callable(apply_space_edits)

def test_view_half_imports_the_pure_half_not_the_reverse():
    from phenotypic._services import tune_spec
    from phenotypic.gui.tune import _space_view

    assert "phenotypic.gui" not in inspect.getsource(tune_spec)
    assert "_services.tune_spec" in inspect.getsource(_space_view)

def test_legacy_import_path_still_works():
    """_setup_authoring.py:28 and three call sites import from _space."""
    from phenotypic.gui.tune._space import (  # noqa: F401
        apply_space_edits,
        build_space_view,
        setup_knob_forms,
        space_to_spec,
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_space_split.py -v`
Expected: FAIL — `No module named 'phenotypic._services.tune_spec'`.

- [ ] **Step 3: Perform the split**

Move the three pure functions (plus `_load_space_source` and any private helper
used **only** by them) into `_services/tune_spec.py` with no Dash imports. Move
the three view functions into `gui/tune/_space_view.py`, which imports what it
needs back:

```python
# src/phenotypic/gui/tune/_space_view.py
from phenotypic._services.tune_spec import _build_search_space, space_to_spec
```

Reduce `gui/tune/_space.py` to a shim re-exporting both halves, so
`_setup_authoring.py:28`, `_layout.py:642`, and `_callbacks.py:1388,2227` need no
edit:

```python
# src/phenotypic/gui/tune/_space.py
"""Back-compat shim. Pure half: :mod:`phenotypic._services.tune_spec`.
Dash half: :mod:`phenotypic.gui.tune._space_view`."""

from __future__ import annotations

from phenotypic._services.tune_spec import (  # noqa: F401
    _build_search_space,
    _load_space_source,
    apply_space_edits,
    space_to_spec,
)
from phenotypic.gui.tune._space_view import (  # noqa: F401
    _knob_form,
    build_space_view,
    setup_knob_forms,
)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services tests/unit/gui/tune -q`
Expected: PASS, including `tests/unit/gui/tune/test_setup_authoring.py`.

- [ ] **Step 5: Prove the split holds**

Add `import dash` to `_services/tune_spec.py`, confirm both
`test_pure_half_is_importable_without_dash` and the Task 1 purity gate FAIL,
then remove it.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(tune): split _space.py into a pure half and a Dash view"
```

- [ ] **Step 7: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 7: Promote spec authoring, validation, and export

**Files:**
- Modify: `src/phenotypic/_services/tune_spec.py` (extend with the moved modules)
- Modify: `src/phenotypic/gui/tune/{_setup_authoring,_command,_validation,_export}.py` → shims
- Test: extend `tests/unit/services/test_shim_equivalence.py`

**Interfaces:**
- Produces: `phenotypic._services.tune_spec.export_best_from_run`,
  `.prepare_best_from_run`, `.publish_prepared_export`, plus the authoring and
  validation entry points Phase 2B's `tune_put_spec` calls

- [ ] **Step 1: Confirm the module list before moving**

Run: `uv run python -c "import phenotypic.gui.tune._command"` and
`grep -rn "^import \|^from " src/phenotypic/gui/tune/_command.py`.
§1.4 lists `_command.py` among the four; if it turns out to be Dash-bearing or
absent, record it in the plan's drift register and split it the way Task 6 split
`_space.py`. **Do not silently drop a module from the move.**

- [ ] **Step 2: Add the failing assertion**

```python
def test_export_is_one_function():
    from phenotypic._services.tune_spec import export_best_from_run as canonical
    from phenotypic.gui.tune._export import export_best_from_run as shim

    assert shim is canonical
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_shim_equivalence.py -v`
Expected: FAIL — `ImportError: cannot import name 'export_best_from_run'`.

- [ ] **Step 4: Move the four modules' contents into `_services/tune_spec.py`, shim each original**

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/services tests/unit/gui/tune tests/integration/gui -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(services): promote tune spec authoring, validation, export"
```

- [ ] **Step 7: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 8: Promote argv construction

**Files:**
- Create: `src/phenotypic/_services/argv.py`
- Modify: `src/phenotypic/gui/run_console/_state.py`, `src/phenotypic/gui/tune/_run_argv.py` → shims
- Test: `tests/unit/services/test_argv_promotion.py`

**`to_argv` cannot travel alone.** Its signature is
`to_argv(state: RunConsoleState)` and `RunConsoleState` is defined in the same
file at `:70`. The dataclass is clean — plain, already JSON-serializable, no Dash
coupling — so it moves with the function. Leaving it behind would make
`_services/argv.py` import back up into `gui/`, inverting the layering.

**Interfaces:**
- Produces: `phenotypic._services.argv.RunConsoleState`,
  `.to_argv(state) -> list[str]`, `.tune_run_argv(...) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_argv_promotion.py
def test_state_and_builder_move_together():
    from phenotypic._services.argv import RunConsoleState, to_argv

    assert to_argv.__annotations__["state"] in (RunConsoleState, "RunConsoleState")

def test_shims_are_the_same_objects():
    from phenotypic._services.argv import RunConsoleState as canonical
    from phenotypic.gui.run_console._state import RunConsoleState as shim

    assert shim is canonical

def test_argv_module_does_not_import_gui():
    import inspect

    from phenotypic._services import argv

    assert "phenotypic.gui" not in inspect.getsource(argv)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_argv_promotion.py -v`
Expected: FAIL — `No module named 'phenotypic._services.argv'`.

- [ ] **Step 3: Move `RunConsoleState` + `to_argv` + the tune argv builder; shim both originals**

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services tests/unit/gui/run_console tests/integration/gui/test_run_console_callbacks.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(services): promote to_argv with RunConsoleState"
```

- [ ] **Step 6: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

### Task 9: Extract a pure sbatch-spec builder

**Files:**
- Modify: `src/phenotypic/_cli/_cli_slurm_array_scripts.py:116-368`
- Test: `tests/unit/cli/test_build_array_script_spec_is_pure.py`

**Why this is not a call-through.** `deploy_plan` (§5.3) must render an sbatch
preview **without touching the run's output directory**, but
`generate_array_job_script` has real side effects under the *real* output
directory: `script_dir.mkdir(...)` (`:184-185`), `log_dir.mkdir(...)` (`:198`),
and `write_slurm_array_script` → `path.write_text(...)` + `path.chmod(0o755)`.
Calling it for a preview would populate
`<output_dir>/.phenotypic/slurm_scripts/` and `logs/` **before you approve
anything**, and would then trip `deploy_start`'s own `output_not_empty` check on
the directory the preview swore it only looked at.

`SlurmArrayScriptSpec.render()` is already pure. What is entangled is the ~150
lines of argument, `cmd_parts`, and dispatch-block construction that build the
spec alongside the write.

**Interfaces:**
- Produces: `build_array_script_spec(...) -> SlurmArrayScriptSpec` — **no I/O**.
  Phase 2C's `deploy_plan` calls it directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/test_build_array_script_spec_is_pure.py
"""deploy_plan previews an sbatch script; a preview that writes is not a preview."""

import hashlib
from pathlib import Path

def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()

def test_build_array_script_spec_writes_nothing(tmp_path, array_script_kwargs):
    from phenotypic._cli._cli_slurm_array_scripts import build_array_script_spec

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    before = _tree_digest(output_dir)

    spec = build_array_script_spec(output_dir=output_dir, **array_script_kwargs)

    assert _tree_digest(output_dir) == before, "the builder touched the output dir"
    assert spec.render(), "the spec must still render a script"

def test_generator_and_builder_agree(tmp_path, array_script_kwargs):
    """The real generator must consume the extracted builder, not duplicate it."""
    from phenotypic._cli._cli_slurm_array_scripts import (
        build_array_script_spec,
        generate_array_job_script,
    )

    out_a = tmp_path / "a"
    out_a.mkdir()
    written = Path(generate_array_job_script(output_dir=out_a, **array_script_kwargs))

    out_b = tmp_path / "b"
    out_b.mkdir()
    previewed = build_array_script_spec(output_dir=out_b, **array_script_kwargs).render()

    assert written.read_text() == previewed
```

Add an `array_script_kwargs` fixture to `tests/unit/cli/conftest.py` supplying a
minimal valid call — read `generate_array_job_script`'s signature and mirror its
required arguments exactly.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_build_array_script_spec_is_pure.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_array_script_spec'`.

- [ ] **Step 3: Extract the builder**

Split `generate_array_job_script` in two: everything that computes the
`SlurmArrayScriptSpec` moves into `build_array_script_spec(...)` with no `mkdir`,
`write_text`, or `chmod`; the original keeps the directory creation and the write
and now reads:

```python
def generate_array_job_script(*, output_dir, **kwargs):
    spec = build_array_script_spec(output_dir=output_dir, **kwargs)
    script_dir = ...  # unchanged mkdir / log_dir / write_slurm_array_script
    return write_slurm_array_script(script_dir / name, spec.render())
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/cli -q && uv run pytest tests/unit/gui/run_console/test_slurm_live_harness.py -q`
Expected: PASS — byte-identical scripts, no behaviour change for the real path.

- [ ] **Step 5: Prove the purity test can fail**

Add a `(output_dir / "scratch").mkdir()` inside `build_array_script_spec`,
confirm `test_build_array_script_spec_writes_nothing` FAILS, then remove it.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(cli): extract a pure build_array_script_spec for deploy previews"
```

- [ ] **Step 7: Spawn a reviewer**

Dispatch `execute-plan-orchestration:implementation-test-reviewer`, scoped to this
task's diff (`git show HEAD`). Brief it with the task's goal, the spec section it
implements, and the specific claim its tests are meant to pin. Require it to check
at minimum:

- **No false greens.** Each new test must fail when the behaviour it guards is
  mutated. This task's "prove it can fail" step is a claim; the reviewer verifies it.
- **No scope leak.** Nothing outside this task's stated **Files** changed.
- **Interfaces hold.** The names and types in the **Interfaces** block match what
  was actually produced — later tasks are written against them, and a rename here
  silently breaks a task nobody is looking at yet.

Read the findings. Fix them in a follow-up commit, or record why not. **Do not
start the next task with an unaddressed correctness finding.**

---

## Phase 1a exit gate

All must hold before Phase 1b starts:

- [ ] `uv run pytest tests/unit/services -v` — green, and the purity gate
      collects one case per promoted module (five modules, not zero).
- [ ] `uv run pytest tests/unit/gui tests/integration/gui -q` — green,
      **unchanged**. The GUI must not notice this phase happened.
- [ ] `uv run pytest tests/gui -q` — green. **`tests/gui` IS in `testpaths`**
      (`pyproject.toml:200`, added by `aa40014ab`), so CI runs it; a regression
      here fails the build rather than hiding.
- [ ] The CI ledger gates stay green: `FEATURES.md`, `WORKFLOWS.md`, smoke-capture.
- [ ] `uv run mypy src/phenotypic` — no new errors.
- [ ] `uv run ruff check src/phenotypic/_services src/phenotypic/gui src/phenotypic/_cli tests/unit/services`
- [ ] Every "prove it can fail" step above was actually run, with the failure observed.
