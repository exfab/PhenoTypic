# Chrome font: Comfortaa to Nunito Sans

- **Date:** 2026-09-29
- **Branch:** `claude/magical-keller-kw5lu5`
- **Status:** implemented on `claude/magical-keller-kw5lu5` (2026-09-29); N1 to N3 were decided
  in the artifact at `docs/superpowers/artifacts/2026-09-29-figure-typography/`
- **Follow-on:** the figure-typography spec (same artifact) and the move of `DESIGN.md`
  to the Google DESIGN.md token format (option C of the 2026-09-29 brainstorm) both
  build on the font roles this spec settles, so this change lands first.

## Objective

Replace Comfortaa with Nunito Sans as the GUI chrome family, in both the display role
(content headings, large stat values) and the body role (prose, button and tab labels,
component titles). The mono role (JetBrains Mono) is unchanged.

## Non-goals

This spec does not touch figure or chart typography. The Plotly template
(`sdk_/viz/figures/_theme.py`) keeps IBM Plex Sans and the matplotlib theme
(`sdk_/viz/figures/_mpl_theme.py`) is left as it is; both are the subject of the
figure-typography spec. It also does not restructure `DESIGN.md` into front-matter
tokens, which is the later option C work, and it does not vendor font files into the
package, since the GUI keeps loading its faces from Google Fonts with the existing
system fallback stacks.

## Background

Comfortaa became the chrome family in an earlier revision of `DESIGN.md` (v1.2). Two
properties of that face shaped the rest of the typography section. Comfortaa ships no
italic, which the Google Fonts API confirms by answering a request for
`Comfortaa:ital@1` with HTTP 400 "The requested font families are not available".
Because of that, italic binomial species names were routed to a fourth role,
`--font-species`, set in IBM Plex Serif italic, so that *Rhodotorula toruloides* would
render as a designed italic rather than a browser-synthesized oblique. The role is
defined in `_gui/_design.py:215` and applied through the `.is-species` class
(`_gui/_design.py:716`), but no layout in `src/phenotypic/_gui/` currently applies
`.is-species`, so the role has no rendered call site today.

Nunito Sans removes the reason for that detour. The Google Fonts API serves it with a
true italic, a weight axis from 200 to 1000, an optical-size axis from 6 to 12 and a
width axis from 75 to 125 (queried 2026-09-29 through
`fonts.googleapis.com/css2?family=Nunito+Sans:ital,opsz,wght@...`). It is distributed
under the SIL Open Font License, like Comfortaa.

The two faces do not set text at the same size. Measured from the `hmtx` advance widths
of the Regular cuts served by Google Fonts, Nunito Sans sets four representative GUI
strings at 0.846 to 0.873 of Comfortaa's width, and its x-height is 0.486 em against
Comfortaa's 0.547 em. At an unchanged pixel ladder the chrome will therefore read
smaller and looser, and text that was close to truncating will gain room. The
measurement is reproducible with `fontTools` against the two TTFs; the sample strings
were "Colony area by strain", "Run console", "Detect colonies on 384-pin plates" and
"Measurements exported".

## Design

### The swap itself

`_gui/_design.py` was built for this change. Its role comment (`:161`) says that
swapping a role means changing its `_*_PRIMARY` constant and updating
`_GOOGLE_FONTS_URL`, after which every call site inherits the new family through the
`--font-*` custom properties and the `FONT_FAMILY_*` constants. The swap is therefore
two edits in that file:

1. `_DISPLAY_PRIMARY` and `_BODY_PRIMARY` (`:168`, `:169`) become `"Nunito Sans"`.
2. `_GOOGLE_FONTS_URL` (`:181`) replaces `family=Comfortaa:wght@400;500;600;700` with a
   Nunito Sans request covering weights 400, 500, 600 and 700 plus the 400 and 600
   italics, so a binomial in body text and one in a weight-600 heading (N2) both use a
   designed cut rather than a synthesized bold.

The sans fallback stack stays as it is. `_FALLBACK_SANS` already covers macOS, Windows,
Linux and Android system sans faces, and those are a closer offline stand-in for Nunito
Sans than they were for the rounded Comfortaa.

### Decisions

**N1. Italic species names use Nunito Sans Italic.** `_SPECIES_PRIMARY` becomes
`"Nunito Sans"` and the `.is-species` rule keeps its `font-style: italic`. A binomial
set in the italic of the surrounding text is the ordinary typographic convention, and
the serif exception existed only because Comfortaa had no italic. The `--font-species`
role token survives, so a later reversal is a one-line change. `_FALLBACK_SPECIES`
moves from the serif stack to `_FALLBACK_SANS` for the same reason. IBM Plex Serif stays
in the Google Fonts request, because the chart subsystem still uses it for donut center
values (`DESIGN.md` 06); only its role as the species face ends.

**N2. Display styles move from weight 400 to 600.** Nunito Sans at 400 is a plainer,
lower-contrast heading than Comfortaa at 400, and the heavier weight holds the
hierarchy between headings and body text. The change touches the five display-family
text styles (Display, Title, Header, H2, H3) in the `.text-*` classes
(`_gui/_design.py:696` to `:700`) and in the `DESIGN.md` 02.4 table. UI Title already
sits at 600 and Button at 500, so after this change headings and UI titles share a
weight and are told apart by size alone. The implementation should confirm on the
rendered hub that this still reads as two levels; if it does not, UI Title can drop to
500 without touching headings.

**N3. Body text rises from 15 px to 16 px.** At 16 px the Nunito Sans x-height is
0.486 x 16 = 7.8 px, against Comfortaa's 0.547 x 15 = 8.2 px, so most of the lost
x-height returns while text still sets narrower than before. The ladder is expressed in
rem against the browser's 16 px root, so the change is `TEXT_BASE` from `0.9375rem` to
`1rem` (`_gui/_design.py:402`).

Two consequences follow, and the author accepted the recommendation for both on
2026-09-29. `FONT_SIZE_BODY` is also the size of the mono Data Value styles
(`.text-data`, `.text-data--muted`), so raising it would enlarge numeric table cells
set in JetBrains Mono, which this change does not otherwise touch. The data styles
therefore get their own alias, `FONT_SIZE_DATA` / `--font-size-data`, pinned at the old
`0.9375rem`, so the font swap does not reflow tables. Two existing tests require every
`FONT_SIZE_*` alias to equal exactly one `TEXT_*` primitive, one to one
(`test_semantic_font_size_aliases_resolve_to_primitives` and
`test_semantic_font_size_aliases_cover_full_scale` in
`tests/unit/gui/test_config_and_design.py`). Once `TEXT_BASE` is 1rem no primitive
holds 15 px, so the change adds a primitive `TEXT_DATA = "0.9375rem"` between
`TEXT_SM` and `TEXT_BASE`, and the two tests and `test_type_scale_is_monotonic` gain
the new pair. Body Large (`TEXT_MD`, 17 px)
would otherwise sit only 1 px above body, so it rises to `1.125rem` (18 px) to keep a
visible step. The fourteen other `--font-size-body` call sites in the sub-app
stylesheets (tune, run console, builder, analysis) are prose and UI labels, so they
move to 16 px as intended.

### Role rule: Nunito Sans for text, mono for data

The author set the rule on 2026-09-29: Nunito Sans carries all general text and
formatting, and every data value and table value is set in the mono family. The four
Dash `DataTable`s already comply, each setting `style_cell` to `FONT_FAMILY_MONO`
(`tune/_callbacks.py:1198`, `results_viewer/_viewer_card.py:422`,
`results_viewer/_error_tab/_layout.py:154`, `builder/_image_renderer.py:499`). Three
plain HTML tables do not, because their cells inherit the body family:

| Table | Cells not in mono | Stylesheet |
|---|---|---|
| Browse CSV metadata (`browse/_callbacks.py:268`) | every cell | `browse/_assets/browse.css:300` |
| Run console Recent Runs (`run_console/_layout.py:202`) | Mode and Dashboard (Output is already mono) | `run_console/_assets/run_console.css:289`, `:294` |
| Analysis post-op preview (`analysis/_post_preview.py:96`) | Before and After values (Column is already mono) | `analysis/_assets/analysis.css:68` |

Their value cells move to `var(--font-mono)` at `var(--font-size-data)`, and their
header cells take the Label style the Typography section already assigns to table
headers. `DESIGN.md` states the rule in the Typography section.

### Documents and tests that change with it

The inventory below comes from `grep -rn Comfortaa` over the tree, excluding dated
historical documents under `docs/superpowers/`, which record past states and are left
alone.

| File | What changes |
|---|---|
| `src/phenotypic/_gui/_design.py` | `_DISPLAY_PRIMARY`, `_BODY_PRIMARY`, `_SPECIES_PRIMARY`, `_FALLBACK_SPECIES`, `_GOOGLE_FONTS_URL` and the role comment block (`:6`, `:153` to `:193`); `TEXT_BASE` and `TEXT_MD` (`:402`, `:403`); the display text-style weights (`:696` to `:700`); the new `FONT_SIZE_DATA` alias and the `.text-data` classes that use it (`:712`, `:713`) |
| Browse, run console and analysis stylesheets | Table value cells to mono (see Role rule) |
| `src/phenotypic/_gui/shell/_assets/shell.css` | Header comment naming the role fonts (`:14`) |
| `src/phenotypic/_gui/FEATURES.md` | "Semantic typography tokens" row (`:591`); required by the `features-md-gate` job for any change under `_gui/` |
| `tests/unit/gui/test_config_and_design.py` | `test_font_family_constants_carry_role_fonts` (`:433` to `:443`) pins Comfortaa for display and body and a serif species stack; all three assertions change |
| `tests/unit/viz/test_theme.py` | `test_chart_body_font_intentionally_differs_from_gui_chrome` (`:108` to `:116`) pins Comfortaa; its intent (chart and chrome fonts differ) still holds |
| `DESIGN.md` | 33 mentions: the header, Overview, Key Characteristics, 02.1, 02.4, 02.7, and component recipes in 05, 09, 13 and 14; the 02.2 size table (body 16 px) and 02.4 weights |
| `docs/source/tutorials/gui/` screenshots | Regenerated with `scripts/capture_gui_tutorial_screenshots.py` per the `gui-tutorial-capture` skill, committing the full set |

No test compares rendered pixels, so the metric change cannot fail a test by itself;
the tutorial screenshots are the only visual record, and they are regenerated rather
than compared.

### Pre-existing defects to fix in the same pass

Two defects in the sections this change edits are cheap to fix while the file is open.
The CSS block in `DESIGN.md` 02.3 (lines 535 to 590) has been broken into one token per
line, so `--tracking-tight: -0.02em;` currently reads as eight separate lines and is
not copyable as CSS; `_gui/_design.py:573` holds the correct values to restore it from.
The Absolute Constraints block still contains an editing instruction addressed to its
author ("Add these to the existing 'Absolute Constraints' block", line 131), which
should be deleted.

## Verification

The focused checks are the two test files named above plus
`tests/unit/ci/test_startup_imports.py`, which guards that `_gui/_design.py` stays
cheap to import. After those pass, the tutorial capture script regenerates the
screenshots, and a manual look at the hub (`uv run phenotypic-gui --root <dir>`) at
the builder, results viewer and run console confirms that no label truncates or wraps
differently in a way that breaks a layout.

## References

- Google Fonts CSS API responses for `Nunito+Sans` and `Comfortaa`, queried 2026-09-29.
- SIL Open Font License 1.1, <https://openfontlicense.org>.
