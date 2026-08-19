# Phase 1a — simplify pass

Quality only. No behaviour change, no test weakened, no re-export removed.

Scope was the Phase 1a diff (`git diff upstream/main...HEAD`, minus `docs/`):
`src/phenotypic/_services/*`, the eleven `gui/` shims, the two lazy package
`__init__`s, `tests/unit/services/*`, and the
`_cli_slurm_array_scripts.py` builder/generator split with its test.

---

## Verification (after the changes)

| Gate | Result | Baseline measured on `572e27cdd` before editing |
|---|---|---|
| `tests/unit/services` | **61 passed** | 61 passed |
| `tests/unit/cli` | **552 passed** (see note) | same |
| `tests/unit/gui` + `tests/integration/gui` | **1746 passed, 3 skipped** | 1746 passed, 3 skipped |
| `ruff check <changed paths>` | **All checks passed!** | — |
| mypy, cold cache, `src/phenotypic` | **417 errors in 124 files** | 417 errors in 124 files |

- The services count is **unchanged at exactly 61** — no test node was added,
  removed, or merged. That was a design constraint on this pass, not a
  coincidence; see "considered and left" below.
- mypy was run cold into a fresh cache directory both times and the two outputs
  were compared **line by line** (`diff <(sort base) <(sort after)`), not by
  count. The diff is **empty** — byte-identical error text, same files, same
  lines.
- CLI note: `tests/unit/services` + `tests/unit/cli` run together as
  `-n auto` gives **612 passed, 1 failed** both **before and after** the change.
  The failure is
  `test_cli_terminal_failures.py::test_concurrent_process_appends_do_not_lose_records`
  (`SpawnProcess.exitcode is None`) — a multiprocessing test starved on this
  4-core allocation. It is present at the unmodified baseline, touches nothing
  in this diff, and **passes when run alone** (`1 passed in 18.57s`). Not caused
  by, and not fixed by, this pass.

### Mutation checks on the refactored gates

Three test modules were rewired onto a shared boundary helper, so the gates were
re-proven to still bite rather than assumed to:

| Mutation | Expected | Observed |
|---|---|---|
| `from phenotypic.gui import builder as _leak` appended to `_services/argv.py` | the aliased-reach form is caught | `test_argv_module_does_not_import_gui` FAILED, reporting both `phenotypic.gui` and `phenotypic.gui.builder` |
| Both `GUI_IMPORT_ALLOWLIST` entries widened to `{"phenotypic.gui"}` | the equality pins refuse a widened entry | 3 FAILED: both `test_allowlist_entry_matches_what_the_module_actually_reaches` cases **and** `test_pure_half_reaches_exactly_its_allowlisted_gui_modules` |
| `import dash` appended to `_services/sandbox.py` | every probe-based gate fires | 7 FAILED across all three modules (`test_import_purity`, `test_lazy_gui_packages`, `test_space_split`) |

A fourth mutation (`raise ImportError` inside `_services/sandbox.py`, to exercise
the helper's returncode assert) turned out to be unusable: `tests/unit/test_fixtures.py`
is loaded as a pytest plugin and imports the whole package, so pytest dies before
collection. The returncode assert is carried over verbatim from the original
code, unmodified.

---

## What changed

### 1. `# noqa: F401` — 7 of 11 were dead, and the 4 live ones now say why

Measured, not eyeballed: every shim was copied to a scratch dir with the marker
stripped and run through `ruff check --isolated --select F401`. Names listed in
`__all__` already count as used, so the marker fired on **nothing** in
`_operation_registry.py`, `shell/_runs_registry.py`, `run_console/_runner.py`,
`run_console/_state.py`, `tune/_command.py`, `tune/_validation.py`,
`tune/_run_argv.py` — removed there.

It is genuinely load-bearing in exactly the four shims that forward a **private**
name (not in `__all__`): `shell/_sandbox.py`, `tune/_export.py`,
`tune/_setup_authoring.py`, `tune/_space.py`. Those now read
`# noqa: F401 - re-exported`, matching the convention `shell/_source_context.py`
and `tune/_domain_editor.py` already used. The marker now means one thing —
"a private name travels through here" — instead of being decoration.

The `if TYPE_CHECKING:` block in `tune/_space.py` keeps the bare `# noqa: F401`,
matching `gui/shell/__init__.py` and `gui/tune/__init__.py`: those names are
type-checker-only, not re-exports.

### 2. Shim name ordering

Two shims were written against a different convention from the other nine, which
sort `SCREAMING_CASE` → `ClassName` → `_private` → `lowercase`:

- `tune/_setup_authoring.py` had `_normalize_setup_metadata_groupby` wedged
  between `SETUP_DRAFT_VERSION` and `SetupAuthoringResult`; moved to the private
  block, matching `_export.py` and `_space.py`.
- `tune/_export.py`'s `__all__` had `PreparedPipelineExport` trailing the
  functions; moved to the front.
- `shell/_runs_registry.py`'s `__all__` had `RunStatus` before `RunRecord`
  (its import list has them the other way round); sorted, plus a stray trailing
  blank line dropped.

**Every one of these was verified by AST**, not by reading the diff: for each of
the eleven touched shims, the set of imported names and the set of `__all__`
entries were extracted from `HEAD` and from the working tree and compared.
All eleven report `imports SAME | __all__ SAME`. That check caught a real
mistake mid-pass — a botched two-step edit had dropped
`_normalize_setup_metadata_groupby` entirely, which is precisely the re-export
removal the brief warns has broken this project four times. It was restored
before anything ran.

### 3. `tests/unit/services/_boundary.py` — one copy of the two boundary gates

The AST import-walk existed **three** times (not two): `gui_modules_reached` in
`test_import_purity.py`, an inline walk in `test_argv_promotion.py`, and
`_parsed_imports` + `_gui_modules_reached` in `test_space_split.py`. The
subprocess leak probe (`FORBIDDEN` + `_PROBE`) existed twice, character for
character, in `test_import_purity.py` and `test_lazy_gui_packages.py`.

New private helper module (not a `conftest.py` — these are plain callables the
tests import by name, and the parametrize lists are built at collection time):

- `FORBIDDEN` + `forbidden_imports_after_importing(module)` — the one-subprocess-
  per-module probe. Both the "two libraries deliberately NOT listed" comment and
  the "one subprocess per module" comment moved with it.
- `parsed_import_names(module)` — accepts a module object or a dotted name.
- `gui_modules_reached(module)`
- `shallowest_modules(names)` — the allowlist-grain reduction `test_space_split`
  needs and the other two do not.

**Nothing was weakened in the merge.** The three copies differed in exactly one
respect: `test_import_purity`'s version skipped relative imports (`node.level == 0`),
the other two did not. The canonical helper keeps the `level == 0` filter and
documents why — a `from .x import y` inside these packages can never *name*
`phenotypic.gui`, and folding its `.x` / `x.y` spellings into the result adds
names no caller can interpret. This is behaviour-neutral for every existing
assertion: no `_services` module uses a relative import at all (checked), and
`_space_view`'s import of the pure half is absolute.

Each test keeps its own assertion and its own failure message — the helper
returns data, it does not assert on the caller's behalf, so
`"dragged {leaked} into sys.modules"` and `"dragged {leaked} in via its package
__init__"` both survive as distinct diagnostics.

### 4. `tests/unit/cli/test_build_array_script_spec_is_pure.py`

- Six copies of `output_dir = tmp_path / "run"; output_dir.mkdir()` → one
  `output_dir` fixture. The "start from a known empty state" reason (which
  `_tree_digest` depends on) is now stated once, in the fixture.
- `test_building_a_spec_reads_the_inputs_it_hashes` hand-rolled a
  `pytest.MonkeyPatch()` with `try/finally: monkey.undo()` while its sibling
  `test_generator_consumes_the_builder` used the `monkeypatch` fixture. Switched
  to the fixture — same teardown guarantee, six fewer lines, one convention. The
  `assert mod.file_sha256 is real_sha` pre-check (which proves the two patch
  targets are the same object before patching either) is kept.
- The module docstring cross-referenced
  `:func:`test_building_a_spec_reads_every_input_image``, which does not exist —
  the function is `test_building_a_spec_reads_the_inputs_it_hashes`. Fixed.

---

## Considered and deliberately left

**`test_shim_equivalence.py`'s eight near-identical `getattr(shim, name) is
getattr(canonical, name)` loops.** This is the largest single block of repetition
in the phase and a parametrized table would compress it by roughly half. Left
alone for two reasons. First, folding the seven "reexports every name" tests into
one table produces **eight** parametrized cases (the runs test covers two shims),
so the suite count moves 61 → 62 — and the brief's instruction on a moved number
is to revert, not to rationalise. Second, each of those tests carries a distinct,
hard-won docstring: which call site imports which name, that `ColumnRefSpec` was
the name the plan's shim sketch omitted, that `write_setup_draft` /
`build_authored_setup_spec` reach the module only through the multi-line
parenthesised imports that produced the X2 and X3 incidents. In a table those
become comments on data rows, which is a worse place for a warning than a
docstring on the test that enforces it. Compression here would trade real
provenance for line count.

**The subsumed single-name identity tests** (`test_get_registry_is_one_function`
is a strict subset of `test_registry_shim_reexports_every_public_name`, and the
same holds for the sandbox / runs / export / command pairs). Removing them is
not weakening in the strict sense, but it is also not worth anything: they are
three lines each and they name the specific invariant a reader looks for first.
Left.

**Three different lazy-import shapes** — `_LAZY: dict[str, tuple[str, str]]` in
`gui/shell/__init__.py`, `_VIEW_NAMES: frozenset` in `gui/tune/_space.py`, and a
bare `if name == "create_app"` in `gui/tune/__init__.py`. This looks like drift
between agents but each is the minimal correct form for a different shape:
five names across four modules (needs a mapping), eight names from one module
(needs only membership), one name from one module (needs neither). Unifying on
the dict would make two of the three strictly more verbose to buy a symmetry
nobody reads. Left.

**Near-identical "must not import the GUI" docstring paragraphs in
`_services/*.py`.** On inspection they are not near-identical: `registry.py`
states the constraint in one sentence, `argv.py` names the two test modules that
enforce it, `tune_spec.py` adds the no-rendering-surface and no-`optuna` rules,
and `sandbox.py` / `runs.py` say nothing about it at all. The canonical statement
already lives once, in `_services/__init__.py`. Trimming the module-level ones
would delete module-specific facts to remove a resemblance rather than a
duplication. Left.

**`_services/tune_spec.py`'s loader and fingerprint helpers.** `_try_load_spec` /
`_try_load_pipeline` / `load_pipeline_or_spec` all read a JSON file and validate
it; `path_content_fingerprint` / `authored_content_fingerprint` both SHA-256 a
file. They look mergeable and are not: the first three differ in error semantics
(two degrade to `None` on any failure, one propagates) and in dispatch (one
branches on the suffix), and the two fingerprints differ in what they hash (path
+ content + a missing/unreadable sentinel, versus content only). These are moved
bodies, unchanged from their pre-promotion originals; merging them would be a
behaviour change wearing a cleanup's clothes. Left.

**`_cli_slurm_array_scripts.py`'s builder/generator split.** Reviewed, nothing to
do. `_array_script_names` is already the shared helper that stops the two halves
drifting, and both call sites destructure only the half they need
(`job_name, _` / `_, script_name`), which reads correctly.

**The `strict=True` xfail.** There is no live `xfail` anywhere in this phase's
tests — the only occurrence is a comment in `test_lazy_gui_packages.py`
recording that the Task 6 split removed one "rather than rotting into a permanent
expected failure". Nothing to preserve; nothing touched. The allowlist equality
pin is intact and, per the mutation table above, still fails on a widened entry.
