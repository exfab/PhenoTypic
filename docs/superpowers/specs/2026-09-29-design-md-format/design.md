# DESIGN.md in the Google DESIGN.md format

- **Date:** 2026-09-29
- **Branch:** `claude/magical-keller-kw5lu5`
- **Status:** design settled; C1 to C3 confirmed by the author on 2026-09-29; awaiting
  implementation plan
- **Depends on:** `2026-09-29-chrome-font-nunito-sans` (its tokens are the ones the
  front matter records) and `2026-09-29-figure-style-defaults` (its section is placed
  here)

## Objective

Restructure the repository's `DESIGN.md` into the DESIGN.md format published by Google
Labs, so that its design tokens sit in machine-readable YAML front matter, its
sections carry the headings that format's tooling recognizes, and
`npx @google/design.md lint DESIGN.md` passes with no errors or warnings. This is
option C of the 2026-09-29 brainstorm.

## Non-goals

`DESIGN.md` stays guidance. This spec adds no test that compares it with
`_gui/_design.py` or the theme modules, which the author declined on 2026-09-29, and
no CI job that fails a pull request on a lint finding. It does not generate code from
the tokens either. The format's `export css-vars` command could one day produce the
CSS block that `_gui/_design.py` writes by hand, but that is a separate change.

## Background

### The format

The format is published at <https://github.com/google-labs-code/design.md> under
Apache-2.0 and is labelled alpha. The specification quoted here is the one the CLI
prints with `npx @google/design.md spec` at version 0.4.0, fetched on 2026-09-29.
Front matter holds typed token groups: `colors`, `typography`, `rounded`, `spacing`
and `components`, plus `version`, `name`, `description` and `omitted`. A dimension may
use only `px`, `em` or `rem`. Tokens may reference each other as `{colors.primary}`.
The body uses `##` headings, and eight of them are recognized, in this order: Overview,
Colors, Typography, Layout, Elevation & Depth, Shapes, Components, Do's and Don'ts.
Other headings are preserved without error, while a duplicated heading is an error.
The spec calls the tokens "the normative values" and the prose the context for
applying them.

### What the linter does with this repository today

Run against the current `DESIGN.md`, the linter reports a single warning, "No YAML
content found", because the file has no front matter. A probe file written to test the
format's edges on 2026-09-29 produced four further findings that shape this design:

1. A custom top-level key such as `figures:` draws a warning under the rule
   `token-like-ignored`. The key is not silently accepted, contrary to what an earlier
   reading of the documentation suggested.
2. A numbered heading such as `## 01 -- Color Palette` is not recognized as Colors; it
   counts as an unknown section.
3. The rule `section-order` warns when recognized sections appear out of order.
4. The rule `contrast-ratio` checks each component's `textColor` against its
   `backgroundColor` and warns below WCAG AA's 4.5:1. The probe's Okabe-Ito yellow on
   the `#FBFEF8` canvas measured 1.30:1.

## Design

### Front matter

The front matter records the values the GUI uses, taken from `_gui/_design.py` after
the Nunito Sans change lands.

| Group | Contents |
|---|---|
| `colors` | UI palette (navy, blue, gold), surface and neutral tokens, semantic colors, the Okabe-Ito data colors as `oi-*`, and the darkened badge variants |
| `typography` | One token per text style in `DESIGN.md` 02.4 (Display through Data Micro, 15 styles), with family, size in px or rem, weight, line height and letter spacing |
| `rounded` | The radius ladder under the existing CSS names: `sm` 3px, `base` 6px, `md` 10px, `lg` 16px |
| `spacing` | The spacing scale as `_design.py` defines it, `SPACING_1` (4 px) through `SPACING_16` (64 px) |
| `components` | Buttons, badges, alerts and inputs, with each variant's text and background colors, so the contrast rule checks every badge the Badges section specifies |

The components group earns its place through the contrast check. The Badges section
already prescribes darkened Okabe-Ito variants for text on white, and recording each
badge as a component makes the linter confirm that every pairing meets AA.

### Section map

Numbering is removed from every heading, because the numbered forms are not
recognized. Recognized sections take the canonical names; domain sections the format
does not know keep descriptive names and sit between Components and Do's and Don'ts,
except Logo and Branding, which belongs next to the Overview.

| Current | New heading |
|---|---|
| Overview; Absolute Constraints | `## Overview`, with the constraints kept as a subsection (C3) |
| 00 Logo and Branding | `## Logo and Branding` |
| 01 Color Palette | `## Colors` |
| 02 Typography | `## Typography` |
| 03 Spacing & Layout; 13 Dashboard Shell & Layout | `## Layout` |
| 04 Shapes & Elevation | split into `## Elevation & Depth` and `## Shapes` |
| 05 Components; 14 Feedback & Loading States | `## Components` |
| 06 Data Visualization; 11 Extended Chart Types; 12 Chart Support Elements | `## Data Visualization` |
| (new) | `## Figures`, from `2026-09-29-figure-style-defaults` |
| 09 Image Display & Viewers | `## Image Display` |
| 10 Well-Plate Grid | `## Well-Plate Grid` |
| 15 Export & Provenance Strip | `## Export & Provenance` |
| 07 Code Integration | `## Code Integration` |
| 08 Usage Rules & Anti-Patterns | `## Do's and Don'ts` |

Code Integration loses its matplotlib `rcParams` block, which the Figures section and
`_mpl_theme.py` replace, and keeps the napari label colors and the CSS custom
properties.

### Scoping the absolute rules

Three Absolute Constraints and one Don't are written over all charts but describe
screen charts only (see `2026-09-29-figure-style-defaults`): mono type for numbers,
the navy-first series order, and the navy-to-sky sequential ramp. Each gains the scope
"in GUI chrome and on-screen charts", and the Figures section states from its side
that none of them binds static figures.

### Cross-references

Section numbers disappear, so every reference to one must change to the heading name.
Inside `DESIGN.md`, 42 lines refer to another section by number (`grep -cE "section
[0-9]{2}|§ ?[0-9]{2}"`). Outside it, 41 lines in 20 files under `src/` and `tests/` cite
a numbered section, 14 of them in `_gui/_design.py`; the plan lists them by the same
grep. Heading names are the more stable anchor, since a heading survives a reordering
that a number does not.

### Pre-existing defects

The restructure rewrites the Typography section, so it restores the CSS block in 02.3
that a formatter broke into one token per line, and it deletes the leftover editing
note in the Absolute Constraints ("Add these to the existing ... block"). The Nunito
Sans spec names both as well; whichever change lands first fixes them.

## Decisions (confirmed 2026-09-29)

**C1. Where figure values live.** Figure sizes are in points and widths in
millimetres, which the format's dimension type cannot express, and a custom
`figures:` key draws a standing `token-like-ignored` warning. Figure values
stay in the Figures section's own tables and in `_mpl_theme.py`, out of the front
matter, so the lint stays clean. The rejected alternative accepted one permanent
warning in exchange for having the figure values machine-readable next to the others.

**C2. How the linter runs.** The linter is a documented manual command,
`npx @google/design.md@0.4.0 lint DESIGN.md`, named in `CLAUDE.md` beside the other
checks, with the version pinned because the format is alpha. A pre-commit hook would
catch findings earlier, but it would block commits on a document the author wants to
remain guidance, and it adds a Node dependency to every contributor's commit path.

**C3. Where the Absolute Constraints go.** The format puts Do's and Don'ts last, while
the constraints matter most to an agent reading the file for the first time. They stay
as a short "Absolute Constraints" subsection inside the Overview, which the format
permits, and the longer Do and Don't lists move to the final section.

## Verification

After the restructure, `npx @google/design.md@0.4.0 lint DESIGN.md` reports no errors
and no warnings. Its `info` findings, such as the token summary, are acceptable. The
same grep that counted the numbered references finds none left. The two tests that pin
`DESIGN.md` wording in docstrings, `tests/unit/viz/test_theme.py` and
`tests/unit/gui/test_config_and_design.py`, still pass, and so does
`tests/unit/abc_/plotting/test_figure_backend.py`, which cites the Typography section.

## References

- Google Labs, DESIGN.md format and CLI, <https://github.com/google-labs-code/design.md>;
  specification text from `npx @google/design.md spec`, version 0.4.0, 2026-09-29.
- Linter output for the current `DESIGN.md` and for a probe file, both run on
  2026-09-29 with the same version.
