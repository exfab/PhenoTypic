# Measurement categories and the `measurements_by_category/` split

**Date:** 2026-09-25
**Branch:** `feat/measurement-tags` (off `origin/main` @ `a8b6e17c`)
**Status:** design approved in brainstorm; awaiting written-spec review

## 1. Objective

Give measurement columns a curated, repo-defined **category** axis — a
many-to-many grouping such as *Starting Metrics* — and have every CLI run that
finalizes measurements publish one spreadsheet per category under
`deliverables/measurements_by_category/`, beside the existing
`measurements_by_feature/`. Give categories their own generated docs page
(each category's `desc` output verbatim from the enum), and badge each
categorized column on the Measurements page.

To free the word, the existing per-enum header prefix, today called
`category()`, is renamed **`metric_family()`**.

### Non-goals

- User-defined or runtime categories. Categories are always repo-defined; the
  closed vocabulary is `schema/_categories.py`.
- Any change to kind/tier classification. Categories make no trust claim.
- Renaming or restructuring `measurements_by_feature/` (keyed by *measurer*,
  despite its name).
- Clearing stale split files (see §8, Follow-ups).
- GUI surfaces. No GUI code reads either split today (`_gui/_config.py:32`
  mentions `measurements_by_feature/` only in a docstring).

## 2. Vocabulary

| Term | Meaning | Cardinality |
|---|---|---|
| **metric family** | The string prefixed onto every member of one `MeasurementInfo` enum: `SIZE.metric_family() == "Size"` → `Size_Area`. Pure spelling. | one per enum |
| **category** | A curated grouping of measurement members across families, e.g. `CATEGORIES.STARTING_METRICS`. | many-to-many |
| kind / tier | Existing trust classification (`_tiers.py`). Unchanged. | one per member |

## 3. Phase 0 — rename `category()` → `metric_family()`

A hard rename with no alias, landed as one mechanical commit before any
category code.

- `MeasurementInfo.category()` → `metric_family()`; instance property
  `.CATEGORY` → `.METRIC_FAMILY`. Member construction
  (`_measurement_info.py:442`) builds the value from `cls.metric_family()`.
- Every override is renamed: the 40 in `src/phenotypic/schema/` and the ones in
  `src/phenotypic/sdk_/constants_.py` (`GAMMA_ENCODINGS`, the status enum, …).
  All `.category()` calls (12) and `.CATEGORY` uses (4) in `src/`, plus the 17
  test files that reference them.
- **Hard break for out-of-tree subclasses.** A `MeasurementInfo` subclass that
  defines `category` fails at class creation with a `TypeError` naming
  `metric_family()`. The check must run before member construction, since
  `__new__` is what would otherwise call the missing method; the plan picks the
  hook (`__init_subclass__` vs. a check in `__new__`) after verifying Enum
  class-creation ordering on the project's Python.
- Wording: "category-prefixed" → "family-prefixed" in `MeasurementInfo`
  docstrings, `schema/CLAUDE.md`, root `CLAUDE.md` (Gotchas: "Measurement columns
  are category-prefixed"), and `docs/source/explanation/metadata_namespace.md`
  (the one docs page naming `category()`).
- **README hazard.** `_cli_readme_generator.py:_generate_measurement_table`
  calls `info_cls.category()` inside a broad `except Exception` that returns
  `""`. A missed rename there silently drops every measurement table from every
  run's README. It is renamed with the rest, and a guard test (§7) asserts that
  every measurer's table renders.
- No CHANGELOG exists in the repo; the breaking change is recorded in the commit
  body and the PR description.

## 4. Schema: `CATEGORIES` and `Entry.categories`

### 4.1 `schema/_categories.py`

Stdlib-only, following the schema import-light rule (`schema/CLAUDE.md`,
Conventions). It mirrors the `MeasurementInfo` shape (a label and a desc per
member) but deliberately **does not subclass `MeasurementInfo`**: at least six
sites discover measurement enums by `issubclass(x, MeasurementInfo)`
(`docs/source/_extensions/measurements_ref.py`, `schema/_rembi.py:47`,
`util/_measurement_outputs.py:225`, `abc_/_measure_features.py:349,366`,
`analysis/qc/_expected_vs_detected.py:85`,
`plotting/_plot_meas_time_series.py:404,418`), and a subclass would surface as a
measurement table and fail the classification coverage gate.

```python
@dataclass(frozen=True)
class CategoryEntry:
    label: str   # CamelCase token: file stem and anchor slug, e.g. "StartingMetrics"
    desc: str    # technical: what the category groups and why

class CATEGORIES(str, Enum):
    # value == label (no family prefix); members expose .label, .desc
    STARTING_METRICS = CategoryEntry(
        "StartingMetrics",
        "Core per-colony magnitudes to examine first: the size measurements, "
        "integrated grayscale intensity, and the CIELAB medoid colour.",
    )
```

- `__new__` accepts only a `CategoryEntry` (`TypeError` otherwise), matching
  `MeasurementInfo`'s `Entry`-only rule.
- `.display_name` → the label split on CamelCase boundaries
  (`"StartingMetrics"` → `"Starting Metrics"`). Used for docs headings, badges
  and the README. There is no separate title field.
- `.members()` → a tuple of every public `MeasurementInfo` member carrying this
  category, in `phenotypic.schema.__all__` export order and then member order.
  It is built lazily by a function-local `import phenotypic.schema` (the module
  itself stays stdlib-only) and is deliberately **not cached**. It walks ~40
  classes, only docs and tests call it, and a cache would hide classes
  registered later. (Amended 2026-09-25 after plan review M3; approved by the
  user.)
- `desc` is technical text; agents may author it. There is no `bio_desc`.
- `CATEGORIES` and `CategoryEntry` are exported from `phenotypic.schema`.

### 4.2 `Entry.categories`

- A new keyword-only field `categories`, default empty. Authors write the
  member directly, never `.value`:

  ```python
  AREA = Entry("Area", "...", categories=CATEGORIES.STARTING_METRICS)
  # several: categories=(CATEGORIES.STARTING_METRICS, CATEGORIES.OTHER)
  ```

- It accepts a **bare `CATEGORIES` member** or any iterable of members, and is
  normalized to a `frozenset` in `__post_init__` (via `object.__setattr__`,
  since `Entry` is frozen).
- **The bare member is checked first.** `CATEGORIES` is a `str` enum, so a bare
  member is itself an iterable string. Iterating it would yield its characters
  (`{"S", "t", "a", …}`). `__post_init__` tests `isinstance(value, CATEGORIES)`
  before any iteration, and a test pins that
  `Entry(..., categories=CATEGORIES.STARTING_METRICS).categories ==
  frozenset({CATEGORIES.STARTING_METRICS})`.
- `__post_init__` raises `TypeError` for anything that is not a `CATEGORIES`
  member, **including a raw string equal to a member's value**
  (`"StartingMetrics"`). The check is by `isinstance`, never by equality, so
  every category is spelled through `_categories.py`.
- `MeasurementInfo.__new__` stores it on the member as `.categories`.
- **Metadata owners may not carry categories.** A `MetadataInfo` member with a
  non-empty `categories` is refused by a schema test. Metadata columns already
  appear in every split as context.

### 4.3 Initial membership: `STARTING_METRICS` (18 members)

| Family | Members |
|---|---|
| `SIZE` | all 14 (enumerated from `list(SIZE)` on `a8b6e17c`): Area, IntegratedIntensity, Perimeter, ConvexArea, BboxArea, MajorAxisLength, MinorAxisLength, MinFeretDiameter, MaxFeretDiameter, InscribedRadius, MedianRadius, MeanRadius, RobustMeanRadius, MaxRadius |
| `ColorLab` | `L*Medoid`, `a*Medoid`, `b*Medoid` |
| `INTENSITY` | `IntegratedIntensity` |

Both integrated-intensity columns are tagged on purpose: a pipeline may run
either `MeasureSize` or `MeasureIntensity`. When both run, the category file
carries both columns.

The pin test lists all 18 headers explicitly. A `SIZE` member added later is
**not** auto-categorized; it needs its own `categories=` and a pin-test update.

## 5. Split and CLI output

### 5.1 One grouping engine (`util/_measurement_outputs.py`)

- **Context columns** are defined once, for both splits: the columns *not*
  owned by any measurement producer's `MeasurementInfo` (the set the feature
  split already treats as context: metadata, object label, grid, joined
  external metadata). Measurement columns outside a split's group are dropped
  from that split, never carried as context. So a `StartingMetrics` file holds
  the context columns plus its 18 columns, not every measurement in the run.
- Extract a private `_split_by_groups(df, context, groups)` that owns the
  pandas/polars same-type selection and column ordering (master order, context
  first, as today). `split_measurements(df)` (by measurer, unchanged output) is
  re-expressed on it.
- New public `split_measurements_by_category(df) -> dict[str, frame]`, keyed by
  category label. A column belongs to a category when its member carries that
  category. The member is resolved with `member_for_header()` across public
  `MeasurementInfo` classes, so dynamic texture and metric-qualified headers
  work. A column in several categories appears in each. A category with no
  present columns has no key. It is exported from `phenotypic.util`.

### 5.2 CLI writer (`_cli/_cli_output_manager.py`)

- Extract the atomic CSV + Parquet loop in `split_master_by_feature` into a
  private `_write_split(split_dir, frames) -> dict[str, Path]` (atomic writes,
  per-file warning on failure, CSV-saved-if-Parquet-fails, unchanged).
- New `split_master_by_category(post_df, output_dir)` writes to
  `measurements_by_category_dir(output_dir)` via `_write_split`.

### 5.3 Where it runs: the recompile/finalization path

There is one finalization path, and the category split lives on it:

```
full / measure : aggregate_measurements → _aggregate_measurements_unlocked
                 → finalize_run                    (_cli_output_manager.py:1497)
recompile      : run_recompile_task → finalize_run  (_cli_recompile_worker.py:973-975)
finalize_run   → finalize_post_master_outputs      (_cli_finalize_run.py:552, sole caller)
               → split_master_by_feature           (_cli_output_manager.py:1294)
               → split_master_by_category          (new, immediately after)
```

- **Requirement:** `split_master_by_category` is called **only** from
  `finalize_post_master_outputs`, beside the feature split, inside the same
  `_guarded_terminal_best_effort(commit_guard, …)` wrapper. A failure logs a
  warning and never affects the master, the mirror, or the aggregate proof.
- It therefore runs for `full`, `measure` and `recompile`, locally, in the
  SLURM finalizer, under `--wait` (where the in-process and finalizer
  aggregations are serialized by `.aggregate_publication.lock` and write the
  same bytes), and for `--mode migrate` (which aggregates through the same
  path). `--mode process` does not measure and writes neither split. The chunk
  writer writes neither (chunks are intermediate).
- **Input frame:** `post_df`, the post-applied, metadata-joined frame published
  as `measurements.{csv,parquet}`, the same one the feature split reads.
- **Existing runs** gain the folder with `--mode recompile`. There is no
  continuation impact: splits are derivations of the published mirror, not work
  units, so no work-id digest or `*_REVISION` bump is needed.

### 5.4 Layout and paths

```
deliverables/
├── measurements_by_feature/     MeasureSize.{csv,parquet}, …   (unchanged)
└── measurements_by_category/    StartingMetrics.{csv,parquet}
```

- `sdk_/_io_constants.py`: `DIR_MEASUREMENTS_BY_CATEGORY: Final[str] =
  "measurements_by_category"` and `measurements_by_category_dir(output_dir)`,
  both exported from `phenotypic.sdk_` (and listed in `__all__`), per the rule
  never to hand-join deliverable names.

### 5.5 README (`_cli/_cli_readme_generator.py`)

- The output-structure tree gains `measurements_by_feature/` (missing today) and
  `measurements_by_category/`.
- Per-family measurement tables gain a Categories column (display names), shown
  only when some member of that table has a category.
- A new "Measurement Categories" section lists each category that has columns
  the run's **configured** measurers declare, with its `desc` and those columns.
  `READMEGenerator` sees the pipeline, not the measurement frame, so this matches
  how the existing measurement tables are generated. A measurer that fails on
  every image still gets an entry. (Amended 2026-09-25 from "present in this
  run" after plan review M4; approved by the user.)

## 6. Docs

1. **A generated Categories page** (`docs/source/_extensions/measurements_ref.py`
   writes `measurements_ref/categories/index.rst` on every build, beside the
   Measurements and Metadata pages). Nothing on it is hand-maintained: adding a
   `CATEGORIES` member or tagging an `Entry` changes the page on the next build.
   - A fixed intro, written in the extension: categories are curated,
     many-to-many groupings with no trust claim, plus a link to
     `measurement-categories` on the explanation page.
   - For each `CATEGORIES` member, in enum order:
     - an anchor `measurement-category-<label.lower()>` and a heading
       (`display_name`)
     - its **`desc`, output verbatim** from the enum
     - the file it produces: `deliverables/measurements_by_category/<label>.{csv,parquet}`
     - a list-table *Column | Family | Type*: the Family cell is a `:ref:` to
       the existing `measurement-info-<slug>` anchor, and the Type cell is the
       member's existing tier badge (`use_badge`)
   - **It is a subpage of the "Measurements" header tab**, a sibling of
     Metadata:
     - Navbar: a third `<li>` in the Measurements dropdown
       (`docs/source/_templates/navbar-nav.html`), after Measurements and
       Metadata, using the same markup and
       `_pn.startswith('measurements_ref/categories/')` active state. The tab
       itself highlights on this page with no change, since its test is
       `_pn.startswith('measurements_ref/')`.
     - Sidebar/toctree: `../categories/index` joins `../metadata/index` in the
       Measurements page's hidden toctree (`_build_reference_page`), so the
       page is a child of Measurements.
     - It is also linked from the explanation page's `measurement-categories`
       section. It is **not** added to the Explanation toctree, so it has one
       parent.
   - It is generated under `measurements_ref/` rather than `explanation/`
     because the extension `rmtree`s and regenerates its own folder. Writing
     generated files into the hand-authored `explanation/` folder would put
     them under that cleanup rule or leave them stale.
   - The Measurements page itself gets no per-category column list (a table
     gathering one category's columns from across families); that lives only on
     the Categories page. The Measurements page's family tables carry the
     category badges instead (item 2).
2. **Family tables** (`MeasurementInfo.rst_table()`): a "Categories" column with
   one badge per category, `:bdg-ref-<color>-line:` (the outline variant, so it
   reads as distinct from the solid Type pills), linking to the category anchor.
   The `-line` ref-badge roles are registered by sphinx-design for every
   semantic color (`setup_badges_and_buttons`, verified in the docs env).
   It follows the existing rule of appearing only when some member has a
   category. `rst_table()` also feeds 21 measurer/API docstrings; Sphinx labels
   are global, so the links resolve from those pages too. The badge color and
   anchor pattern are constants next to `_BADGE_SPECS`.
3. **Explanation page** (`docs/source/explanation/measurement_classification_system.md`):
   a new `(measurement-categories)=` section, a short hand-written paragraph.
   Categories are curated, many-to-many groupings; unlike kind/tier they make
   **no trust claim**. It links to the generated Categories page for the list,
   so no category name or `desc` is repeated by hand. It also defines *metric
   family* in one sentence as the column prefix.
   `schema/CLAUDE.md` is updated alongside, since it must stay consistent with
   this page.
4. **CLI docs** (`docs/source/tutorials/pages/cli_modes.md`): the output tree
   gains `measurements_by_category/`.
5. **Module guides**: `schema/CLAUDE.md` gains a "Categories" section (the
   vocabulary, the no-subclass rule, how to add a category), and root
   `CLAUDE.md`'s `deliverables/` bullet names the new folder.

## 7. Testing

Focused tests per phase; the full sharded regression runs once, at the end.

| Area | Guards |
|---|---|
| Rename | No `def category(` / `.category()` / `.CATEGORY` left in `src/`. A subclass defining `category` raises `TypeError` at class creation. The README renders a non-empty table for every discovered measurer (the swallowed-exception guard). |
| Schema | `Entry(categories=CATEGORIES.X)` (bare member) yields `frozenset({CATEGORIES.X})`, not its characters. A tuple of members normalizes to a `frozenset`. A raw string equal to a member's value is rejected, as is any non-member. `CATEGORIES` rejects a non-`CategoryEntry` value. No `MetadataInfo` member carries categories. Every category has ≥1 member. `STARTING_METRICS.members()` is pinned to its exact 18 headers. `display_name` is readable for every label. The startup import guards (`tests/unit/ci/test_startup_imports.py`, `test_deferred_imports.py`) pass. |
| Split | Pandas and polars in, same type out. Context columns are identical to the feature split's. A column in two categories appears in both. A category with no present columns has no key. `split_measurements` output is unchanged by the refactor. A dynamic-header member resolves through `member_for_header()`. |
| CLI | `finalize_post_master_outputs` writes `measurements_by_category/StartingMetrics.{csv,parquet}` with equal contents. A raising category split still leaves the master and mirror published. `--mode recompile` of an existing finalized tree writes the folder. `split_master_by_category` has exactly one call site (`finalize_post_master_outputs`). The `sdk_` path helper resolves under `deliverables/`. |
| Docs | The generated Categories page has one section per `CATEGORIES` member, with its anchor, and its body contains that member's `desc` verbatim. Adding a test-local category changes the generated page (it is generated, not hand-written). The navbar has a Categories entry. Every category anchor that `rst_table()` emits exists on the generated Categories page (this closes the dead-link gap `schema/CLAUDE.md` warns about, since badge refs use `reftype="any"` and only warn). The category badge color is in sphinx-design's `SEMANTIC_COLORS`. |

The docs build runs as a Slurm job (`sphinx-build -j "$SLURM_CPUS_PER_TASK"
-D nbsphinx_execute=never`), followed by reading the generated Measurements HTML.
Exit 0 is not the check.

No logic-validation script: nothing here rests on a numeric invariant.

## 8. Follow-ups (out of scope)

- **Stale split files.** Neither split clears its folder, so a recompile that
  drops every column of a category leaves the old file behind. This is inherited
  from `measurements_by_feature/`; fixing it for both is a behaviour change to
  the existing split.
- **Integrated-intensity descriptions disagree.** `SIZE.INTEGRATED_INTENSITY`'s
  `desc` says "sum × area"; `INTENSITY.INTEGRATED_INTENSITY`'s says "sum". Both
  measurers compute the sum. Correct the `SIZE` desc separately.
- **`measurements_by_feature/` naming.** It is keyed by measurer class, which
  reads oddly beside *metric family*. Leave as is.
