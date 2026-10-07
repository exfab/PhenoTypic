# Phase 3 review: Tasks 8 and 9 (docs)

Commits: `47efe825` (generated Categories subpage + navbar), `71e4e256` (prose + module guides).
Reviewed against spec `design.md` §6 and plan Tasks 8–9. Evidence was read from the source, the
generated RST (`docs/source/measurements_ref/categories/index.rst`), the built HTML under
`docs/_build/categories/`, and the build log `slurm_logs/cat-docs_29105308.log`.

**Severity count: 0 CRITICAL, 0 HIGH, 1 MEDIUM, 6 LOW.**

## What was verified and holds

- **§6.1 page contents.** Each section gets the anchor `.. _measurement-category-startingmetrics:`
  (`measurements_ref.py` `_category_section`, from `CATEGORIES.anchor`, `_categories.py:93-96`).
  The heading is `display_name`. `desc` is emitted raw and appears verbatim in the HTML. The line
  `Written to ``deliverables/measurements_by_category/StartingMetrics.csv`` (and ``.parquet``)` is
  present. The table is Column | Metric family | Type with 18 rows. Built HTML: 3 `ColorLab` + 1
  `Intensity` + 14 `Size` family links resolve to `../measurements/index.html#measurement-info-*`,
  plus 15 success and 3 primary Type badges. No raw `:bdg-ref`/`:ref:` text is left in the HTML.
- **`*` in member values.** `ColorLab_L*Medoid` and the others sit inside double backticks, so
  they are inline literals and render as `<span class="pre">ColorLab_L*Medoid</span>`. The log has
  no warning for the categories page. The three `Inline emphasis` warnings at
  `measurements/index.rst:164/168/172` come from the pre-existing `*GeoMedian` in the ColorLab
  `desc`, which `main:src/phenotypic/schema/_color_lab.py` already contains.
- **Toctree and navbar.** The Measurements page's hidden toctree is `../metadata/index` then
  `../categories/index`. On the Categories page the sidebar shows Metadata plus Categories (current)
  and the breadcrumb is Measurements > Categories. The navbar dropdown item has
  `current active` only on this page, and the Measurements tab is active via
  `_pn.startswith('measurements_ref/')` (`navbar-nav.html:73`). The only other reference to the page
  is the explanation page's `{doc}` link, so it is not in the Explanation toctree.
- **Cross-references.**
  - `:ref:`measurement-categories`` resolves to `explanation/measurement_classification_system.html#measurement-categories`.
  - The MyST `{doc}`Categories </measurements_ref/categories/index>`` renders as
    `../measurements_ref/categories/index.html`.
  - The 18 `sd-outline-info` badges on the Measurements page, and the badges on the
    MeasureSize/MeasureIntensity/MeasureColor API pages, link to
    `…categories/index.html#measurement-category-startingmetrics`.
  - The log has no `undefined label`.
- **Import safety at `config-inited`.** `_build_pages` already imports `phenotypic.schema`.
  `_build_categories_page` imports `CATEGORIES` lazily, and `.members()` imports the schema lazily
  (`_categories.py:112-114`).
- **Regeneration.** `rmtree(output_dir)` (`measurements_ref.py:170-171`) covers the new
  `categories/` child. The generated tree is gitignored (`.gitignore:76`).
- **Legacy dotted forms.** The regex from `tests/unit/schema/test_metric_family.py:29-33` finds no
  `.category()`, `.CATEGORY` or `def category(` in either diff or in any touched file. The
  schema/CLAUDE.md mention of `` `category()` `` has no leading dot.
- **Prose claims checked against code.**
  - "Hash as their `str` value": `type(m).__hash__ is str.__hash__` and
    `hash(m) == hash("StartingMetrics")` both hold, so iteration order is randomized per process.
  - `in_order` ordering: `_categories.py:98-102`, and `category_badges` uses it at
    `_measurement_info.py:621-624`.
  - `_categories.py` has stdlib-only module imports.
  - "At least six" `issubclass(…, MeasurementInfo)` discovery sites: 7 in `src/` and 1 in the
    extension.
  - Metadata rule: enforced by `test_categories.py:158-163`.
  - Every category has a member: `test_categories.py:258-260`.
  - The dark-mode claim holds: pydata 0.16.1 defines `--pst-color-dark:#222832` in both theme
    blocks, while `--pst-color-info` differs per theme (`#276be9` / `#79a3f2`).
  - Single call site: `split_master_by_category` is called only at `_cli_output_manager.py:1309`,
    inside `finalize_post_master_outputs`, and that function's only caller is
    `_cli_finalize_run.py:552`.
  - Modes that reach that caller: full, measure and recompile through
    `aggregate_measurements`/`finalize_run`, and migrate through
    `_cli_migrate.py:1051` (`_publish_migration_aggregate`).
  - Modes that don't: the chunk writer (`_cli_chunk_writer.py`) has no split call.
  - The README section exists (`_cli_readme_generator.py:105,218-301`).

## MEDIUM

### M-1: The explanation page repeats a category name and paraphrases its `desc` by hand, against spec §6.3

`docs/source/explanation/measurement_classification_system.md:65-67`:

> *Starting Metrics*, for example, gathers the size magnitudes, integrated intensity, and the
> CIELAB medoid colour, which come from three different families.

Spec §6.3 says the section "links to the generated Categories page for the list, **so no category
name or `desc` is repeated by hand**." Plan Task 9 Step 1 wrote this sentence in anyway, so the
code follows the plan and the plan drifted from the spec.

The consequence is the one the spec rule exists to prevent. If `STARTING_METRICS` is renamed, its
`desc` edited, or its membership changed (the pin test is updated, as schema/CLAUDE.md instructs),
this sentence goes stale silently. No test reads the explanation page.

Fix, either of:

- Make the example generic ("a category such as a *starting set* can gather size, intensity and
  colour columns from three families…"), with no member name.
- Keep the sentence and amend spec §6.3 to allow one illustrative example. Record the decision
  either way.

## LOW

### L-1: A zero-member category makes docutils drop that section's table with an ERROR

`_category_section` always emits `.. list-table::` with `:header-rows: 1`. When
`category.members()` is empty, only the header row exists, and docutils 0.21.2 raises
`Insufficient data supplied (1 row(s)); no data remaining for table body`
(`docutils/parsers/rst/directives/tables.py:66-72`). The table is replaced by a system-message
block, and the build still exits 0 (no `-W`).

Mitigation that exists: `test_every_category_has_a_member` (`tests/unit/schema/test_categories.py:258`)
fails first, so this cannot ship green.

Optional fix: in `_category_section`, emit a one-line "No columns carry this category yet." paragraph
instead of the table when `members` is empty. That lets a docs build during a
category-in-progress edit render cleanly.

### L-2: A category `desc` is emitted as raw RST with no guard

`measurements_ref.py` `_category_section` writes `category.desc` unescaped, as the spec requires
("verbatim"). The one current `desc` is safe. A future `desc` with a word-initial `*` (the exact
bug `*GeoMedian` already causes on the Measurements page), a backtick, or `|` would emit a docutils
warning or mis-render. The build exits 0, so nothing would catch it.

`CategoryEntry.__post_init__` (`_categories.py:49-55`) checks only non-emptiness.

Fix: add a check in `CategoryEntry.__post_init__`, or a test in `test_categories.py`, that rejects
RST inline-markup start characters (`` ` ``, `*` or `|` at a word start). The "verbatim" contract
and the `category.desc in page` test then keep holding.

### L-3: The "one spreadsheet per category" wording overclaims

- The extension intro (`measurements_ref.py` `_CATEGORIES_INTRO`) says "Every run that measures
  objects writes one spreadsheet per category".
- The explanation page (`measurement_classification_system.md:70`) says "Each run that measures
  objects writes one spreadsheet per category".

In fact `split_measurements_by_category` omits a category with no present column
(`util/_measurement_outputs.py:70-71,161`), and `split_master_by_category` then writes nothing. A run
configured with only `MeasureShape`/`MeasureTexture` produces no `measurements_by_category/` at all.
`_cli/CLAUDE.md:866-867` states it correctly ("one per `CATEGORIES` member with a present column").

Fix: "…one spreadsheet per category that has at least one column in the run".

### L-4: "Every measurement column is named `<Family>_<Label>`" is too strong

`measurement_classification_system.md:61`. The `metric_qualified` and `texture` header schemes
emit `{family}_{metric}_{label}` and `{family}_{label}-deg###-scale##`
(`src/phenotypic/schema/CLAUDE.md` "Dynamic output headers").

Fix, matching spec §6.3's "defines *metric family* … as the column prefix": "Every measurement
column begins with its metric family: `Size_Area` belongs to the **Size** family."

### L-5: `--mode migrate` in the root CLAUDE.md finalize claim is broader than the code

`CLAUDE.md:566-569` says the splits are written by the finalization path "shared by full, measure,
recompile and `--mode migrate`". Only a **full-run** migrate reaches `aggregate_measurements`
(`_cli_migrate.py:1037-1056`). Direct-store and process-tree migrations are provenance-only and never
finalize, as the root CLAUDE.md CLI section itself says. The same phrasing is in the
`split_master_by_category` docstring (`_cli_output_manager.py:1408-1409`) and the call-site comment
(`:1305-1306`).

Fix: "…and a full-run `--mode migrate`".

### L-6: Stale neighbours in the edited guides

- `src/phenotypic/_cli/CLAUDE.md:956-957` ("Per-feature splits and named analysis artifacts derive
  from the mirror") and `:1016` (chunk-writer carve-out: "post, per-feature splits, analysis … are
  deferred to final aggregation") still name only the feature split, though the category split
  obeys both rules. Add "and per-category" to each so the carve-out stays exhaustive.
- `src/phenotypic/schema/CLAUDE.md:6-8`: the updated `Entry(...)` signature adds
  `categories=frozenset()` but still omits `rembi_module=None`, which sits before it
  (`_measurement_info.py:139`). This omission predates the change, but the line was just edited.
- The root `CLAUDE.md` Gotchas bullet "Authoring `MeasurementInfo` members" says agents should
  "only author/edit the **`label`** … and **`desc`**" of an `Entry`. The new schema/CLAUDE.md
  section tells agents to "tag the members", which means writing `categories=` on an `Entry`. A
  reader of the root rule could read tagging as forbidden. Add a clause to the root bullet, e.g.
  "`categories=` tags are also agent-authored; see schema/CLAUDE.md", or scope the rule to
  `bio_desc`/`image` explicitly.

## Tests (47efe825)

They are adequate for the spec §7 Docs row: one section per category with its anchor and verbatim
`desc`, the generated-from-tags test, badge anchors resolving, and the navbar entry and active-state
string.

One weak spot: `test_categories_page_lists_every_member_once_with_its_type_badge` asserts
`member.use_badge in row_block`. That is vacuous when `use_badge == ""`. None of the 18 current
members has an empty badge, so this is informational only.
