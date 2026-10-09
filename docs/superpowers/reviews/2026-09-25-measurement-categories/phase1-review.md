# Phase 1 review: `CATEGORIES`, `Entry.categories`, the 18 tags, category badges

Reviewer: Phase 1 deep-review gate (implementation and test strength).
Scope: `git diff 57f8521e 7de8e443`, which covers commits `57edc22f` (Task 2), `eaed911c` (Task 3)
and `7de8e443` (Task 7 plus the C1 fix). Checked against plan Tasks 2, 3, 7 and 8, the Global
Constraints, the Review Focus, spec §4 and §6.2, and `schema/CLAUDE.md`.

**Verdict: pass. No CRITICAL or HIGH findings.** 0 CRITICAL / 0 HIGH / 3 MEDIUM / 7 LOW.

The implementation is correct for every input I probed. The 18 tags are exact, and they change
no `desc`, `bio_desc` or `image` (proven by AST, §A). mypy is clean. The C1 fix is complete:
`_render_info_table` has exactly two callers, and both now build 6-tuples.

The problems are in the tests. Three mutants survive:

- a Categories cell emitted under the wrong column header;
- `CATEGORIES.__new__` no longer calling its validator;
- `members()` no longer de-duplicating classes.

Separately, the chosen badge colour, `dark`, is nearly invisible in the docs' dark mode.

---

## A. Execution evidence (run by main, verbatim excerpts)

Probes ran in the live worktree at `7de8e443`. Mutations ran in a scratch worktree detached at
`7de8e443` with its own venv. Each mutated file was restored and re-hashed, and the final
`git status` was clean.

**Probe A: enum and normalization behaviour**

```
lookup True True
pickle True True
deepcopy True
hash True True
fmt 'StartingMetrics' 'StartingMetrics' 'StartingMetrics' <CATEGORIES.STARTING_METRICS: 'StartingMetrics'>
names ['STARTING_METRICS']
disp StartingMetrics -> 'Starting Metrics'
disp CIELabColor -> 'CIE Lab Color'
disp RGBValues -> 'RGB Values'
disp Tier1Traits -> 'Tier1 Traits'
disp Size2D -> 'Size2 D'
disp GrowthRate24h -> 'Growth Rate24h'
disp QCFlags -> 'QC Flags'
disp A -> 'A'
disp ABTest -> 'AB Test'
regex accepts trailing newline: True
CategoryEntry ACCEPTED trailing newline
dict frozenset({<CATEGORIES.STARTING_METRICS: 'StartingMetrics'>})
gen frozenset({<CATEGORIES.STARTING_METRICS: 'StartingMetrics'>})
TypeError for NoneType
TypeError for bytes
TypeError for list
entry eq False hash-equal False
_new_member_ TypeError: CATEGORIES members must be declared as CategoryEntry(...); got 'x'
n 18 ['ColorLab_L*Medoid', 'ColorLab_a*Medoid', 'ColorLab_b*Medoid', 'Intensity_IntegratedIntensity', 'Size_Area', ...]
```

In that last line, `list` means `["StartingMetrics"]`, a list holding one raw string.

**Probe B: Task 3 touched only `categories=`** (AST comparison of every `Entry(...)` call at the
two SHAs, ignoring the `categories` keyword)

```
src/phenotypic/schema/_size.py identical-except-categories: True 14 14
src/phenotypic/schema/_color_lab.py identical-except-categories: True 11 11
src/phenotypic/schema/_intensity.py identical-except-categories: True 12 12
```

**Probe C: types.** `uv run mypy` on the six changed source files: `Success: no issues found in 6
source files`.

**Mutations**

| # | Mutation | Command scope | Result |
|---|---|---|---|
| M1 | C1 revert: `_quality_check.py` builds 5-tuples again | `test_rst_rendering.py` + `sdk_/test_quality_check_info.py`; also `test_analysis_package_imports` alone | **Killed**, but by an `IndexError` raised during collection or import (no pytest summary line), before any named test ran |
| M2 | Categories cell emitted after the Biology cell; header order unchanged | `tests/unit/schema` + `test_measurements_ref_extension.py` | **Survived** (144 passed, 3 skipped) |
| M3 | `CATEGORIES.__new__` no longer calls `_validate_entry` | `test_categories.py` | **Survived** (24 passed) |
| M4a | `members()` drops `or info in seen` | `test_categories.py` + `test_rst_rendering.py` | **Survived** (36 passed) |
| M4b | `has_cat = True` | same | **Killed** by `test_uncategorized_table_has_no_categories_column` |

**Probe D (the focused surface, and the colour test with skips shown):** see §E.

---

## B. Findings

### MEDIUM

#### M-1. No test ties the Categories cell to the Categories column (mutant M2 survived)

`_measurement_info.py:214-237`. The header order (Type, Categories, Biology, Image) and the
cell order are written in two separate blocks, and nothing checks that they agree.

`SIZE` carries every optional column at once: `AREA` has a `bio_desc` and an `image`
(`_size.py:28-40`). So when M2 emitted the Categories cell after the Biology cell, the rendered
table showed biology prose under "Categories" and the badge pills under "Biology". Nevertheless,
all 144 tests stayed green. `test_categorized_table_has_a_categories_badge_column`
(`test_rst_rendering.py:88-96`) only checks that the header and the badge string appear
somewhere in the table. docutils does not catch this either, because the cell count per row is
unchanged.

**Fix:** add a structural test that parses `SIZE.rst_table()`:

1. Split the text on `"\n   * - "` and collect each row's `"     - "` cells.
2. Assert that every row has as many cells as the header.
3. Assert that the `Area` row's cell at `header.index("Categories")` is the badge string.
4. Assert that the same row's cell at `header.index("Biology")` is its `bio_desc`.

This kills M2 and any future reordering of the header.

#### M-2. `_CATEGORY_BADGE_COLOR = "dark"` is almost invisible in dark mode

`_measurement_info.py:69`. The docs use `pydata_sphinx_theme` with a `theme-switcher`
(`docs/source/conf.py:189`, `navbar_end`), so dark mode is reachable, and "auto" follows the OS.

- **pydata's colour map:** in `pydata_sphinx_theme/assets/styles/extensions/_sphinx_design.scss:62`,
  pydata maps sphinx-design's `dark` to one fixed value, `"dark": $foundation-dark-gray` (`#222832`,
  `variables/_color.scss:125`). That value has no dark-mode variant, unlike `light`/`muted`
  (lines 49-60) and the pst semantic colours `primary`/`secondary`/`info`/`success`/`warning`
  (`_color.scss:127-160`).
- **The outline badge:** `bdg-ref-dark-line` renders as `sd-outline-dark sd-text-dark`
  (`sphinx_design/badges_buttons.py:56-57`). That is `#222832` text and border on pydata's
  near-black dark background.

The test only asks whether the colour is in `SEMANTIC_COLORS` (`test_classification.py:215-219`),
and it is. The spec leaves the colour open (`:bdg-ref-<color>-line:`, design §6.2 item 2), so
changing it needs no spec amendment.

**Fix:** use a mode-aware colour. Choose one that none of the solid Type pills already uses, so
the outline badges still read as a separate axis:

- `secondary` and `muted` are already taken by the Quality and Identity pills (`_BADGE_SPECS`,
  `:60-61`).
- `danger` is unused but reads as a warning.
- So `info`-line, or `primary`-line, are the natural choices.

Also check the result by eye in the Task 9 docs build (both themes), because no unit test can see
contrast. If `dark` stays, a scoped CSS override in `docs/source/_static/custom.css` for
`html[data-theme="dark"] .sd-outline-dark` works too.

#### M-3. `test_analysis_package_imports` doesn't do what its `reload` implies, and none of the three C1 guards is what killed M1

`test_rst_rendering.py:123-128`.

- **What `reload` actually does:** `importlib.reload(phenotypic.analysis)` re-executes only
  `analysis/__init__.py`. Its `from .qc import (...)` (`analysis/__init__.py:22`) re-binds names
  from submodules that are already in `sys.modules`. So `QualityCheck.__init_subclass__`
  (`analysis/abc_/_quality_check.py:493-511`) does **not** run again for `ICC`, `RelativeMAD`,
  `MaxModifiedZScore` and the rest (`analysis/qc/_icc.py:44` etc.). The `reload` line is a no-op
  for the thing it names. All the test's force comes from the `import` statement, and only when
  that is the first import in the process.
- **What killed M1:** the mutant died during collection or import, with an `IndexError` on
  stderr, before any named test ran. Something imported at collection time (the `tests/unit`
  conftest chain, which main presumed; I did not trace the exact module) already pulls in
  `phenotypic.analysis`. So in practice a C1 regression is caught by the whole run crashing, not
  by either new test.
- **What would catch it anyway:** the pre-existing
  `tests/unit/sdk_/test_quality_check_info.py` calls `QUALITY_CHECK.append_rst_to_doc` seven
  times (lines 45-84), and the tier-5 startup guard imports `phenotypic.analysis` in a subprocess
  (`tests/unit/ci/test_startup_imports.py:256`).

To answer the brief's direct question: yes. `test_quality_check_docs_render_with_category_column`
alone would have caught C1 if collection had survived, because it calls the overridden
`QUALITY_CHECK.append_rst_to_doc` directly, and `row[5]` then raises `IndexError`
(`_measurement_info.py:205`).

This is MEDIUM rather than LOW because the test's claim is misleading. A future reader will take
it as proof that concrete checks re-render.

**Fix:** replace the reload with something that states what it proves. Either:

- a subprocess, `subprocess.run([sys.executable, "-c", "import phenotypic.analysis"],
  check=True)`, which is independent of import order; or
- a docstring assertion on a concrete check, such as
  `assert "Metric family: **QC_" in ICC.__doc__`.

Or delete the test and rely on the render test plus the tier-5 guard.

### LOW

#### L-1. `CategoryEntry` accepts a label with a trailing newline

`_categories.py:29,51`. `_LABEL_RE = r"^[A-Z][A-Za-z0-9]*$"` is used with `.match()`, and in
Python `$` also matches just before a final `\n`. Probe A: `CategoryEntry ACCEPTED trailing
newline`.

The label feeds the output file stem (Phase 2), the Sphinx anchor, and the badge text. A label of
`"Foo\n"` would produce a file named `Foo\n.csv` and break the RST role. This needs an authoring
typo, hence LOW.

**Fix:** `_LABEL_RE.fullmatch(...)` (or `\Z`), and add `"StartingMetrics\n"` to the
parametrized rejection list at `test_categories.py:57`.

#### L-2. `test_members_must_be_category_entries` never exercises `__new__` (mutant M3 survived)

`test_categories.py:51-55` calls `CATEGORIES._validate_entry` directly. Its comment says an enum
with members can't be subclassed, which is true. But Probe A shows the original `__new__` is
reachable as `CATEGORIES._new_member_`, and it raises the right `TypeError`.

**Fix:** `with pytest.raises(TypeError, match="CategoryEntry"):
CATEGORIES._new_member_(CATEGORIES, "x")`. That kills M3.

#### L-3. `members()` de-duplication and ordering are untested (mutant M4a survived)

`_categories.py:116-129`. No compatibility alias is in `schema.__all__` today (`_LEGACY_METADATA_NAMES`
resolves through `__getattr__` and is excluded from `__all__`, `schema/__init__.py:152-179`). So
the `info in seen` guard never fires, and the `len(members) == 18` pin can't see it go missing.

The ordering claim in the docstring ("`__all__` order, then member order") is also unpinned. Task
8 will render the Categories page from this order, so a change would silently reorder the docs.

**Fix:** in `test_members_finds_tagged_public_members`, append `"FUTURE_TAGGED"` to `__all__`
twice, or append an alias name bound to the same class, and assert that `FUTURE_TAGGED.VALUE`
appears once. Pin the order with
`[m.value for m in CATEGORIES.STARTING_METRICS.members()][:4] == ["ColorLab_L*Medoid",
"ColorLab_a*Medoid", "ColorLab_b*Medoid", "Intensity_IntegratedIntensity"]`, which is what
Probe A shows.

#### L-4. Two `CATEGORIES` tests can't fail with one member and the current label regex

- **`test_every_display_name_is_readable`** (`test_categories.py:39-43`) is tautological given
  `_LABEL_RE`. The substitution only inserts spaces, and every label is `[A-Z][A-Za-z0-9]*`. So
  `"".join(words) == label` and "each word starts upper or with a digit" hold for any label the
  regex admits.
- **The regex's weak spots:** Probe A shows where it produces something questionable, and the
  test accepts all of it: `Size2D -> 'Size2 D'`, `GrowthRate24h -> 'Growth Rate24h'`,
  `Tier1Traits -> 'Tier1 Traits'`. None of these is a current label, so this is only a
  forward-looking concern.
- **`test_in_order_follows_declaration_order`** (`:72-73`) passes a single member, so it can't
  tell a sort from no sort.

**Fix:** pin a table of `label -> display_name` expectations against `_CAMEL_BOUNDARY_RE`
directly, including `CIELabColor` and one digit case, with whatever split the team decides is
right. For `in_order`, call it with a local two-member enum:
`CATEGORIES.in_order.__func__(Local, [Local.B, Local.A])`.

#### L-5. The stdlib-only AST guard only walks the top level of the module

`test_categories.py:137-143` iterates `tree.body`. An import under a top-level `try:` or `if`
other than `TYPE_CHECKING` would escape it. It holds today. Walking `ast.walk(tree)` and
excluding the `if TYPE_CHECKING:` block and function bodies would close the gap.

#### L-6. `categories=` quietly accepts a dict (its keys) and rejects `None`

Probe A: `{c: "why"}` becomes `frozenset({c})`, and `None` raises `TypeError`. Both are
defensible. But the other optional `Entry` fields all accept `None`, and a mapping probably means
the author intended something the field can't hold.

**Fix (optional):** reject `Mapping` explicitly in `_normalize_categories`
(`_measurement_info.py:91`), or document both behaviours in the `Entry` `Args:` block.

#### L-7. Iterating a `frozenset` of `str`-enum members depends on the hash seed

`CATEGORIES` members hash as their string value (Probe A: `hash True`), and `str` hashing is
randomized per process. So iterating `member.categories` directly gives a different order in
each process once a column carries two or more categories.

Every current consumer goes through `CATEGORIES.in_order` (`category_badges`,
`_measurement_info.py:612`), and so do all the Phase 2/3 consumers the plan specifies (plan lines
1136, 1685, 1722). So nothing is wrong today. The risk is a later consumer that writes
`for c in member.categories`.

**Fix:** state in the `schema/CLAUDE.md` Categories section (Task 9) that `.categories` has no
order and that consumers must use `CATEGORIES.in_order`.

---

## C. Checks that came back clean

- **Enum mechanics** (Probe A):
  - lookup by value (`CATEGORIES("StartingMetrics")`) and by name work;
  - pickle round-trips return the identical member, for both `CATEGORIES` and a tagged
    `MeasurementInfo` member;
  - `deepcopy` returns the identical member;
  - `str`, `format` and f-strings all give the bare label;
  - `repr` is the standard enum repr;
  - exactly one member exists. The staticmethod and properties are descriptors, so they are not
    members, and the bare annotations `label: str` / `desc: str` create none either.
- **`str` mixin semantics:** `hash(c) == hash("StartingMetrics")`, so
  `"StartingMetrics" in member.categories` is `True`. The `_normalize_categories` docstring says
  this ("reading is looser than writing"), and `members()` relies on it only for real members.
- **Normalization:**
  - a bare member becomes a single-element set, not its characters;
  - a generator and a list are consumed once into a `frozenset`;
  - raw `str`, `bytes`, a list containing a string, and non-iterables are rejected by type,
    never by equality.

  **`frozenset` is the right stored type.** `Entry` is `@dataclass(frozen=True, slots=True)` with
  a generated `__hash__` over all fields, so the stored value must be hashable (a list would make
  `Entry` unhashable). Members share the object, so it must be immutable. Order comes from
  `in_order`, not from the set.
- **`Entry` equality and hash:** they now depend on `categories` (Probe A: `entry eq False`).
  Nothing in `src/` uses an `Entry` as a dict key or compares entries:
  - `grep` shows no `asdict`/`astuple`/`replace`/`fields(Entry` in `schema`, `sdk_`, `util` or
    `tune`;
  - enum aliasing keys on the member's `_value_` (the prefixed header), not on the `Entry`.

  So the change is inert.
- **`cast` in `MeasurementInfo.__new__`** (`_measurement_info.py:538`): it hides nothing.
  `__post_init__` always replaces the field with a `frozenset`, and the only way around that is
  constructing an `Entry` with `object.__new__`, which no code does. mypy is clean (Probe C).
- **The 18 tags:**
  - `grep -c "= Entry("` in `_size.py` is 14, and `categories=CATEGORIES.STARTING_METRICS`
    appears 14 + 3 + 1 times;
  - `members()` returns exactly the pinned 18 headers (Probe A, `n 18`);
  - Probe B proves every other argument of every `Entry` in the three files is AST-identical
    across the two SHAs, so no `desc`, `bio_desc` or `image` changed. The long ColorLab lines
    were only reflowed.
- **Consumers of the new member attribute:** every generic discovery walk filters on
  `issubclass(x, MeasurementInfo)`, so `CATEGORIES` (not a subclass) is skipped. The walks
  checked:
  - `schema/_rembi.py:40-50`
  - `util/_measurement_outputs.py:233-239`
  - `analysis/qc/_expected_vs_detected.py:79-89`
  - `sdk_/_metadata_helpers.py:28-43`
  - `_gui/shell/_metadata_context.py:92-104`
  - the docs extension's `_public_measurement_info_classes`
  - the coverage gate `test_classification_coverage.py:14-19`

  The per-member serialization surfaces are unaffected:
  - no code serializes a member via `vars()` or `__dict__` (the only `vars()` use is the legacy
    guard, `_measurement_info.py:296`);
  - JSON and `model_json_schema` see a member as its `str` value;
  - pipeline serialization stores values;
  - the GUI inspector renders `__doc__` as raw `<pre>` text (`_gui/builder/_layout.py:2613-2629`);
  - `parse_param_descriptions` reads the `Args:` block, which the appended table follows
    unchanged in structure.
- **Import-light rule:** `_categories.py` imports only `__future__`, `re`, `collections.abc`,
  `dataclasses`, `enum` and `typing` at module level. The `MeasurementInfo` import is under
  `TYPE_CHECKING`, and `phenotypic.schema` is imported only inside `members()`. The import graph
  is acyclic: `_measurement_info` imports `_categories`, and `_categories` imports
  `_measurement_info` only lazily.
- **Badge RST:**
  - The role `bdg-ref-dark-line` exists: sphinx-design registers `bdg-ref-<color>-line` for every
    `SEMANTIC_COLORS` entry (`sphinx_design/badges_buttons.py:38-41`), and `dark` is one of them
    (`shared.py:28`).
  - The syntax is the correct `` :role:`text <target>` `` form.
  - The badge cannot contain `|`, `` ` ``, `<` or `>`: `display_name` and `anchor` derive from a
    label restricted to `[A-Za-z0-9]` (L-1 aside).
  - The pills are space-joined and use `in_order`.
- **Anchor consistency with Task 8:**
  - `CATEGORIES.anchor` is `measurement-category-<label.lower()>`;
  - Task 8's `_category_section` emits `.. _{category.anchor}:` (plan line 2061);
  - Task 8's gate test extracts `<(measurement-category-[a-z0-9]+)>` from every `rst_table()`
    (plan line 2008), which matches, because labels are `[A-Za-z0-9]` and lowercased.
  - The references use `reftype="any"` (`badges_buttons.py:120`), which lowercases `ref`
    targets, so a label with capitals would still resolve.

  One expected consequence: between this commit and Task 8, the badges on the Measurements page
  and in 21 measurer docstrings are dead links. They only produce warnings, since the docs build
  has no `-W`. So don't publish docs from a Phase 1 or Phase 2 SHA.
- **Completeness of the 6-tuple change:**
  - `grep -rn _render_info_table src/` lists exactly the definition (`:184`), the `rst_table`
    caller (`:680`) and `_quality_check.py:70`;
  - no test builds rows directly;
  - `docs/source/_extensions/measurements_ref.py:66` calls only `rst_table(header=...,
    use_headers=True)`;
  - all the other docstring appenders (`measure/*`, `analysis/*`, `grid/_auto_grid_finder.py:1378`,
    `analysis/abc_/_quality_check.py:506-511`) go through `rst_table` or the QC override.

  Existing table assertions are unaffected: the caption (`test_metric_family.py:124`), the
  list-table count (`test_measurements_ref_extension.py:137`), and the note position
  (`test_change_note.py:70`, `test_measurements_ref_extension.py:277`).

---

## D. What's at risk outside the focused runs

- **`tests/unit/analysis`, `tests/unit/util`, `tests/unit/cli`:** C1's blast radius. Any
  row-shape regression now crashes collection (M1). So the risk is a loud red run, not a silent
  one. `util` and `_cli` producer discovery import `phenotypic.analysis`.
- **`tests/unit/gui/results_viewer` (`colony_view/_grid.py:105-130`):** this derives measurement
  prefixes by walking `MeasurementInfo.__subclasses__()` at module scope. The test-local
  `TAGGED`/`FUTURE_TAGGED` classes (`test_categories.py:98, 115`) linger in `__subclasses__()`.
  If the colony view is first imported in the same xdist worker after `test_categories.py`, it
  gains `Tagged_`/`FutureTagged_` prefixes. That is harmless, because no column carries them, and
  earlier schema tests already do the same. Worth knowing if a prefix-snapshot test ever appears.
- **`tests/unit/schema/test_measurement_info_format.py`:** the "universal attribute surface" test
  doesn't check `categories`. Adding `assert isinstance(member.categories, frozenset)` would make
  the attribute part of that contract.
- **The Python 3.11 lane:** probes ran on the 3.12 venv. The enum paths used (custom `__new__`
  setting `_value_`, an overridden `__str__`, `__format__` taking the str-overridden branch)
  behave the same on 3.11, but CI's 3.11 job is the first real check.
- **The sphinx-design colour test:** `test_badge_spec_colors_are_valid_sphinx_design_semantic_colors`
  is `importorskip`-gated, so it runs only where sphinx-design is installed. Probe D's second
  command shows whether it ran or was skipped.

---

## E. Probe D (focused surface)

Verbatim, from main's background task output:

```
======================= 400 passed in 280.20s (0:04:40) ========================
=== D2

======================= 1 passed, 20 deselected in 0.49s =======================

[exited with code 0]
```

- **D1** (`tests/unit/schema`, `sdk_/test_quality_check_info.py`,
  `test_measurements_ref_extension.py` and both startup-import guards): 400 passed, 0 failed,
  and no skips reported under `-rs`.
- **D2** (the sphinx-design colour test, run with `--group docs`): it **ran and passed** rather
  than being skipped. So `dark` really is in `SEMANTIC_COLORS`, which M-2 does not dispute; M-2
  is about contrast, not validity.
