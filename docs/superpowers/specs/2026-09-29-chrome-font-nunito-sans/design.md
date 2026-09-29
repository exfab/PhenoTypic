# Chrome font: Comfortaa to Nunito Sans

- **Date:** 2026-09-29
- **Branch:** `claude/magical-keller-kw5lu5`
- **Status:** draft; three decisions (N1 to N3) are open and are being chosen in the
  artifact at `docs/superpowers/artifacts/2026-09-29-figure-typography/`
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
   Nunito Sans request covering the same weights, plus italic cuts if N1 adopts them.

The fallback stacks stay as they are. `_FALLBACK_SANS` already covers macOS, Windows,
Linux and Android system sans faces, and those are a closer offline stand-in for Nunito
Sans than they were for the rounded Comfortaa.

### Open decisions

**N1. Italic species names.** The recommendation is to set species names in Nunito
Sans Italic by pointing `_SPECIES_PRIMARY` at `"Nunito Sans"` and keeping the
`.is-species` rule's `font-style: italic`. A binomial set in the italic of the
surrounding text is the ordinary typographic convention, and the serif exception
existed only because Comfortaa had no italic. The alternative keeps IBM Plex Serif
italic, which preserves the current look at the cost of a second text family in the
chrome. Either way the role token survives, so a later reversal is a one-line change.

**N2. Display weight.** `DESIGN.md` 02.1 sets display styles at weight 400 to keep
Comfortaa headings light. Nunito Sans at 400 is a plainer, lower-contrast heading, so
the choice is between keeping 400 and raising display styles to 600.

**N3. Size compensation.** The ladder is rem-based on a 15 px body (`DESIGN.md` 02.2).
Given the smaller x-height, the choice is between keeping the ladder unchanged and
raising the body root to 16 px. The artifact shows both at true scale.

### Documents and tests that change with it

The inventory below comes from `grep -rn Comfortaa` over the tree, excluding dated
historical documents under `docs/superpowers/`, which record past states and are left
alone.

| File | What changes |
|---|---|
| `src/phenotypic/_gui/_design.py` | `_DISPLAY_PRIMARY`, `_BODY_PRIMARY`, `_GOOGLE_FONTS_URL`, the role comment block (`:6`, `:153` to `:193`); `_SPECIES_PRIMARY` if N1 is adopted |
| `src/phenotypic/_gui/shell/_assets/shell.css` | Header comment naming the role fonts (`:14`) |
| `src/phenotypic/_gui/FEATURES.md` | "Semantic typography tokens" row (`:591`); required by the `features-md-gate` job for any change under `_gui/` |
| `tests/unit/gui/test_config_and_design.py` | `test_font_family_constants_carry_role_fonts` (`:433` to `:443`) pins Comfortaa and IBM Plex Serif |
| `tests/unit/viz/test_theme.py` | `test_chart_body_font_intentionally_differs_from_gui_chrome` (`:108` to `:116`) pins Comfortaa; its intent (chart and chrome fonts differ) still holds |
| `DESIGN.md` | 33 mentions: the header, Overview, Key Characteristics, 02.1, 02.4, 02.7, and component recipes in 05, 09, 13 and 14 |
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
