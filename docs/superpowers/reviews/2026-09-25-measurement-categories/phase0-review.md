# Phase 0 review — `category()` → `metric_family()` (commit `2539b06d`)

Reviewer: Phase 0 deep-review gate (implementation + test-strength).
Scope: `git show 2539b06d` (parent `718ef536`), against plan Task 1 and spec §3.

**Verdict: pass. No CRITICAL or HIGH findings.** 0 CRITICAL / 0 HIGH / 2 MEDIUM / 5 LOW.
The rename is complete for every executable form I could find. Both guard hooks are
individually load-bearing and individually tested (proven by a probe, §A). The symzones deletion
is behaviour-neutral. The gaps are in what the tests *don't* pin: the `CATEGORY` half of the
guard, and non-`.py` surfaces.

---

## A. Execution evidence (probe run by main, verbatim)

In-process mutation: `_refuse_legacy_category` patched to fire only from one hook at a time.
No file was edited.

```
baseline                     test_legacy_category_override_is_refused_at_class_creation: PASS
baseline                     test_legacy_category_on_memberless_subclass_is_refused: PASS
no __new__ guard             test_legacy_category_override_is_refused_at_class_creation: FAIL NotImplementedError:
no __new__ guard             test_legacy_category_on_memberless_subclass_is_refused: PASS
no __init_subclass__ guard   test_legacy_category_override_is_refused_at_class_creation: PASS
no __init_subclass__ guard   test_legacy_category_on_memberless_subclass_is_refused: FAIL Failed: DID NOT RAISE <class 'TypeError'>
CATEGORY-dropped mutant: CATEGORY override ACCEPTED (no test covers this)
guard cost per call: 2.41 us; x403 members = 0.970 ms
```

What this shows:

- **Removing the `__new__` guard is caught** by the member-ful test. Without that guard,
  member construction calls the base `metric_family()` and raises `NotImplementedError`,
  which `pytest.raises(TypeError)` rejects.
- **Removing the `__init_subclass__` guard is caught** by the member-less test.
- So the two hooks are not redundant: each has exactly one test that kills its removal.
- **Cost is negligible.** It is ~2.4 µs per member. There are 403 `= Entry(` members in `src/`,
  so the total is ≈1 ms per process at first schema import. No action needed.

---

## B. Findings

### MEDIUM-1 — The `CATEGORY` half of the guard is untested (surviving mutant)

`src/phenotypic/schema/_measurement_info.py:236`:
`_LEGACY_FAMILY_NAMES: Final = ("category", "CATEGORY")`.

Dropping `"CATEGORY"` from the tuple leaves every test green (probe line
`CATEGORY-dropped mutant: … ACCEPTED`). Both guard tests define only `category`.

The commit message claims more than this. It says a `CATEGORY` override is refused, and that
"a member named CATEGORY is refused too". The `_curation.py:21-23` comment relies on that
second claim. Neither is pinned.

**Fix.** Add two cases to `tests/unit/schema/test_metric_family.py`:

```python
def test_legacy_CATEGORY_property_override_is_refused() -> None:
    with pytest.raises(TypeError, match="metric_family"):
        class LEGACY_PROP(MeasurementInfo):
            @classmethod
            def metric_family(cls) -> str:
                return "LegacyProp"
            @property
            def CATEGORY(self) -> str:
                return "LegacyProp"
            VALUE = Entry("Value", "A value.")


def test_member_named_CATEGORY_is_refused() -> None:
    with pytest.raises(TypeError, match="CATEGORY"):
        class LEGACY_MEMBER(MeasurementInfo):
            @classmethod
            def metric_family(cls) -> str:
                return "LegacyMember"
            CATEGORY = Entry("Category", "A value.")
```

Optionally, parametrize the existing two tests over both names.

### MEDIUM-2 — The legacy-API scan covers only `src/**/*.py` and literal dotted forms

`tests/unit/schema/test_metric_family.py:13,47-54`.

**Would it catch a leftover in a `.md` inside `src/`? No.** The scan uses
`_SRC.rglob("*.py")`. `src/phenotypic/schema/CLAUDE.md`, `src/phenotypic/_gui/FEATURES.md`
and every other non-`.py` file are unscanned.

**It also does not scan any of these:**

- `docs/sour*`: `.md`, `.rst`, `.ipynb`, and `_extensions/*.py`.
- `.claude/skills/**`.
- The root `CLAUDE.md`.
- `tests/`.

Plan review M1 exists because `.claude/skills/adding-an-operation/SKILL.md:76` taught a
`def category` example. That was fixed by hand this time, and nothing stops it coming back.

**The regex only matches literal dotted or `def` forms.** It misses the non-literal forms the
commit itself had to find by runtime failure: `ns["category"]` in
`tests/unit/schema/test_classification.py:15`, plus `getattr(x, "category")`,
`hasattr(..., "CATEGORY")` and `setattr`.

The runtime guard covers *definitions* in any file that gets imported. What remains exposed:

- *call sites* in files the unit suite never imports: docs notebooks and the Sphinx extension;
- *prose and code examples* in markdown.

Current state: I found no remaining legacy calls anywhere outside `docs/superpowers/`.
`git grep -nE "category\(\)|\.CATEGORY|CATEGORY\b"` over docs, `.claude`, `*.md`, `*.ipynb`,
`*.rst`, `*.json` and `*.yaml` has one hit: the intentional history note at
`src/phenotypic/schema/CLAUDE.md:20`. So this is a regression-guard gap, not a live defect.

**Fix.** Widen the scan:

- roots: `src/phenotypic`, `docs/sour*` (excluding `docs/superpowers`), `.claude/skills` and
  `CLAUDE.md`;
- suffixes: `{.py, .md, .rst, .ipynb}`;
- allowlist: `schema/CLAUDE.md`'s history line (match on the phrase "hard-renamed from").

Also add `["']category["']\]` and `(get|has|set)attr\([^)]*["'](category|CATEGORY)["']` to the
pattern, restricted to the schema/sdk_ scope (see LOW-4) so GUI triage dict keys don't
false-positive.

### LOW-1 — The README guard iterates `schema.__all__`, not the "discovered measurers" the spec names

Spec §7 (Rename row): "The README renders a non-empty table for every **discovered
measurer**". The test (`tests/unit/cli/test_readme_measurement_tables.py:80-92`) parametrizes
over public `MeasurementInfo` classes in `schema.__all__`.

The real path is `_generate_measurements_section` → `measurer.get_measurement_infoclasses()`
(`src/phenotypic/abc_/_measure_features.py:333`). A measurer whose schema is not exported from
`schema.__all__` is not covered. I did not verify whether any such measurer exists.

**Does it genuinely detect the swallowed exception? Yes.** Every statement in
`_generate_measurement_table` (`_cli_readme_generator.py:214-241`) is inside the
`try/except Exception: return ""`. `"".startswith("\n### <family>\n")` is `False`, so any
exception in the body (a missed rename, a bad attribute) fails the parametrized case.

**Fix (optional).** Add a second parametrization over
`{c for m in MeasureFeatures-subclasses for c in m().get_measurement_infoclasses()}`, or
assert that set ⊆ the public set. To check whether the gap is real today:

```
uv run python -c "import phenotypic.measure as M, phenotypic.schema as S; from phenotypic.abc_ import MeasureFeatures; pub={getattr(S,n) for n in S.__all__}; print({c.__name__ for n in dir(M) if isinstance(getattr(M,n),type) and issubclass(getattr(M,n),MeasureFeatures) for c in getattr(M,n).__dict__.get('_measurement_infoclasses',()) or ()} - {c.__name__ for c in pub if isinstance(c,type)})"
```

### LOW-2 — Stale "category" wording where "category" will soon mean curated categories

- **`src/phenotypic/_gui/FEATURES.md:424`** still reads "derived from every
  `MeasurementInfo` category … named a `TextureGray_` category … omitted many categories".
  It was not updated.
- **`src/phenotypic/_gui/results_viewer/colony_view/_grid.py:93`**: the constant is still
  named `_AXIS_ELIGIBLE_CATEGORIES`, while its comment (`:89`) now says "Metric families". Its
  only references are in `_grid.py:93,104,132,254`, plus a historical plan.
- **`src/phenotypic/sdk_/_metadata_helpers.py:343-344`**: the deprecated public
  `metadata_category_for_label` docstring says "Return the shared category". After Phase 1,
  "category" is ambiguous here.

**Fix.**

- Update the FEATURES.md prose. The gate validates only the test reference, not the
  description text.
- Rename the constant to `_AXIS_ELIGIBLE_FAMILIES`.
- Change the helper's docstring to "shared metric family". Keep the deprecated function's name.

### LOW-3 — Guard error message misattributes or misdescribes in two edge cases

`_measurement_info.py:239-248`:
`f"{cls.__name__} defines {name!r} … the category() classmethod is now metric_family() and
the CATEGORY property is now METRIC_FAMILY"`.

- It walks the MRO, so a hit on a non-`MeasurementInfo` mixin base (`klass is not cls`) is
  reported as `cls` defining it.
- A **member** named `CATEGORY`/`category` gets advice about a classmethod/property rename.
  That member was harmless before for lowercase `category`. For uppercase `CATEGORY` it already
  shadowed the property.

**Fix.** Report `klass.__name__`, and branch on
`isinstance(vars(klass)[name], (Entry, _proto_member_types))`. Simplest: check
`isinstance(v, (classmethod, property, staticmethod)) or callable(v)` for the rename message,
and use a "reserved member name" message otherwise. Cosmetic; no behaviour change.

### LOW-4 — The scan regex is repo-wide over `src/` and will false-positive on unrelated future code

`_LEGACY_API` matches any `def category(` or `.category()` anywhere under `src/phenotypic`.
Error-tab, curation and operation-registry code legitimately use "category". For example,
`src/phenotypic/_gui/builder/_linear_layout.py:54` does `getattr(info, "category", None)` on
operation-registry info.

A future `def category(self)` on a registry dataclass or a curation store would fail this
schema test for an unrelated reason. Nothing collides today.

**Fix.** Pick one:

- scope the scan to `schema/`, `sdk_/constants_.py`, and files that import `MeasurementInfo`;
- or AST-scan for `ClassDef`s whose bases resolve to `MeasurementInfo` subclasses, plus
  `Attribute(attr in {"category","CATEGORY"})` on names bound to schema classes.

The first is enough.

### LOW-5 — The guard runs for every member rather than once per class

It is measured at ~1 ms total (§A), so this is not a real cost. It is recorded only because
the brief asked. It could short-circuit after the first member with a class-level sentinel, but
that adds state for no measurable gain. **No action recommended.**

---

## C. Verified correct (no finding)

1. **The rename is complete for executable forms.**
   `git grep -nE "\.category\(|def category|\bCATEGORY\b|_known_categories|['\"]category['\"]|['\"]CATEGORY['\"]"`
   outside `docs/superpowers` returns only these:
   - the guard constant and message (`_measurement_info.py:236,247`);
   - the `_curation.py:22` comment;
   - the `schema/CLAUDE.md:20` history note;
   - unrelated concepts: `unicodedata.category`, GUI triage/radial dict keys, error-tab
     publication columns, operation-registry `getattr(info, "category")`
     (`builder/_linear_layout.py:54`, `builder/_preview_callbacks.py:53`), and
     `tests/migration/_scenarios.py:category_for`.

   There are no `super().category`, `setattr`, `category=classmethod(...)` or pickled forms.
   `_known_categories` → `_known_families` has no stale reference. 33 test references to
   `metric_family` exist.
2. **Unrelated "category" concepts are untouched.** The diff's removed lines with "categor"
   are all the header-prefix sense. Error/curation/registry/migration code is unchanged.
   `ErrorCategory`'s class name and `CURATION.ERROR_CATEGORY` (value `Curation_Category`) are
   unchanged.
3. **Serialized forms are unaffected.** Enum values (`Size_Area`, …) are unchanged, so these
   are all unaffected:
   - pickles (by-value lookup through `Enum.__new__`, not the member `__new__`);
   - pipeline JSON;
   - stored `measurement_columns`;
   - master/mirror parquet headers.

   No golden or snapshot file in `tests/` or `docs/sour*` contains the old
   `Category: **` caption. The only hit is a historical HTML mockup under
   `docs/superpowers/artifacts/`. Provenance hashes cover pipeline bytes, not docstrings.
4. **The guard has no false positives from bases.** Every schema class's MRO is
   `… → MeasurementInfo → str → Enum → object`. None of `str`, `Enum` or `object` has
   `category`/`CATEGORY` in its `__dict__`. `MeasurementInfo` is the only multi-base schema
   class. `ConstantLabels` (`sdk_/constants_.py:28`) is a plain subclass. No pydantic or mixin
   base is involved. Annotations (`category: str` without a value) do not populate `vars()`.
5. **Guard ordering holds on both supported Pythons.**
   - 3.12 (`enum.py:594-601`): `type.__new__` runs `__set_name__` (member creation) before
     `__init_subclass__`, and re-raises the original exception with the note stripped. The probe
     confirms this.
   - 3.11: `EnumType.__new__` unwraps the `RuntimeError` from `__set_name__` to its
     `__cause__`, so the `TypeError` surfaces there too. This is the same mechanism the existing
     "raw tuple → TypeError" contract relies on. The 3.11 CI matrix
     (`package-integrity.ci.yml:45`) runs only packaging tests anyway.
6. **The symzones filter deletion is behaviour-neutral** (`_measure_symzones.py:253-256`).
   `CATEGORY` was a builtin `property` (`_measurement_info.py:9` imports only `Enum`), so
   `SYMMETRIC_ZONES.CATEGORY` returned the property object. `str.__ne__(property)` →
   `NotImplemented` both ways → identity fallback → always `True`. The filter never excluded a
   member.
7. **Failed test classes lingering in `__subclasses__()` are harmless.** The two refused
   classes from `test_metric_family.py` stay reachable until GC. They are member-less (`list()`
   is empty). The walkers either filter by `__module__.startswith("phenotypic")`
   (`test_measurement_info_format.py:19`, `test_measurement_assets.py`) or catch
   `NotImplementedError` (`colony_view/_grid.py:125-130`, which runs once at import).
8. **RST caption consumers.** Nothing parses `Category:`. The `QUALITY_CHECK` override
   (`_quality_check.py:62`) reuses `_render_info_table`, so it inherits the new caption
   consistently. The docs extension (`docs/sour*/_extensions/measurements_ref.py`) has no
   "categor" references.

---

## D. At-risk surface the 344-test focused suite did not run

These import a touched module (mechanically, by `git grep -l`) and were not in the Step 8
list. They belong in the plan's Phase 0 importer-derived gate:

- **GUI, module-scope derivation in `_grid.py`:**
  - `tests/gui/results_viewer/colony_view/test_grid.py`
  - `tests/gui/results_viewer/colony_view/test_grid_axis_columns.py`
  - `tests/gui/results_viewer/colony_view/test_grid_measurement.py`
  - `tests/gui/results_viewer/test_mutation_guard.py`
  - `tests/unit/gui/results_viewer/test_colony_view_cap.py`
  - `tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py`
- **Growth models (`metric_token` → `_known_families`, `qualified_header` → `METRIC_FAMILY`):**
  - `tests/unit/analysis/test_model_fitter_headers.py`
  - `tests/unit/analysis/test_log_growth_model.py`
  - `tests/unit/analysis/test_linear_softplus.py`
  - `tests/unit/analysis/test_double_softplus.py`
- **Symzones / orientation zones (dead filter removed; `_orientation_zones.py` renamed):**
  - `tests/unit/measure/test_measure_symmetric_zones.py`
  - `tests/unit/measure/test_orientation_zone_migration_golden.py`
  - `tests/unit/measure/test_orientation_zone_segmentation.py`
  - `tests/unit/measure/test_zone_segmentation_regression.py`
  - `tests/unit/measure/test_zone_figure_cache_parity.py`
  - `tests/unit/measure/test_symmetric_zones_figure.py`

  Step 8's `-k` filter may miss the orientation ones.
- **`sdk_/constants_.py` (`GAMMA_ENCODINGS`, `PIPE_STATUS`):**
  - `tests/unit/core/test_xyz_conversion.py`
  - `tests/unit/core/test_image_dtype_conversion.py`
  - `tests/unit/correction/test_color_checker_geometric_median.py`
  - `tests/unit/grid/test_grid_image.py`
- **CLI:**
  - `tests/unit/cli/test_schema_gate.py`
  - `tests/unit/cli/test_run_identity.py`
- **Tune:** `tests/unit/tune/*` covering `score/_reference_free_scorer.py` (a docstring-only
  change there, so low risk).
- **Docs:** a Sphinx build. The caption text changes on 21 measurer/API docstrings and the
  Measurements/Metadata pages, and `sphinx.ext.doctest` is enabled (`conf.py:95`), which would
  run the `SHAPE`/`metric_family` doctest in `_measurement_info.py`. Not needed for Phase 0
  correctness. Run it with `-D nbsphinx_execute=never` at the docs task.

Expected outcome for all of the above: pass. None depends on the removed names at runtime
except through the renamed calls, which are all updated.
