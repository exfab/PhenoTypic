# Frontend Design Style Guide

> Scientific Analysis Dashboard Design System v1.2 -- Light theme -- Data-intensive
> research & bioanalysis applications (PhenoTypic).
>
> **Single source of truth** for all dashboard UI and data-visualization work. Audience:
> human designers, frontend developers, and agentic coding assistants. Fonts: Comfortaa
> (display + body), JetBrains Mono (mono), IBM Plex Serif (italic species names only).
> Canvas: #FBFEF8.

---

## Overview

PhenoTypic is a scientific analysis dashboard design system: light theme, built for
data-intensive research and bioanalysis. The surface reads as a calm, evidence-first
instrument, not a marketing site or a consumer app. The default page is a near-white
canvas (`{colors.canvas}` / `--color-bg` -- `#FBFEF8`) with pure-white card surfaces
(`--color-white` -- `#ffffff`) lifting off it, and deep navy (`{colors.navy}` --
`#003660`)
as the primary ink for headings, primary actions, and the wordmark. Navy is deliberate
and
slightly desaturated so it reads as branded rather than as default black.

Around navy sits a restrained UI palette: light blue (`{colors.blue}` -- `#1b75bc`) for
interactive and secondary roles, and gold (`{colors.gold}` -- `#febc11`) as a single
deliberate emphasis accent used sparingly (a target line, a brand highlight), never as a
data series and never as a primary button fill. The brand's primary CTA stays navy.

The system's defining rule is a hard separation of three closed color worlds:

- **UI chrome** -- navy / blue / gold plus the surface and neutral tokens. Buttons, nav,
  headings, borders, links.
- **Data visualization** -- the colorblind-safe Okabe-Ito palette, used only inside
  charts,
  plate maps, and image overlays.
- **Branding** -- the logo's colony hues, used in the mark and nowhere else.

These never cross. A data color never paints UI chrome; a logo color never enters a
chart;
the gold accent never becomes a data series. This three-way closure is the single most
important characteristic of the system.

Type carries the second voice. Comfortaa -- a rounded geometric sans -- carries both
content headings / large stat values (display) and body copy / UI chrome titles (body) at
weights 400-700, giving the chrome one warm, approachable voice. Latin species names
(e.g. *Rhodotorula toruloides*) are the one exception: because Comfortaa ships no true
italic, they are set in IBM Plex Serif's italic cut so the binomial reads as a real
italic, not a synthesized slant. JetBrains
Mono
carries every number, axis label, badge, caption, and code token. Mono-for-all-data is a
signature: it preserves optical column alignment and gives the surface a data-forward
read.

Shape is restrained: a `--radius-sm` 3px / `--radius` 6px / `--radius-md` 10px /
`--radius-lg` 16px ladder, engineered rather than playful. The brand never uses pill
CTAs.

Elevation is quiet and navy-tinted: shadows always use `rgba(0, 54, 96, ...)`, never
gray
or black. The one permitted dark surface in an otherwise light system is the image
stage,
where the pixels are the data.

Data integrity is a first-class design constraint, not a styling afterthought: a fixed
six-series order, no red-green encodings, vermilion reserved for error and alert, and a
six-series ceiling before an "other" bucket. The system is colorblind-safe by
construction.

**Key Characteristics:**

- **Three closed color systems** -- UI chrome (navy / blue / gold), data (Okabe-Ito,
  CB-safe), and branding (logo only). Crossing them is a hard error, not a judgment
  call.
- **Navy is the ink and the conversion color.** Every primary action, heading, and
  wordmark
  is `{colors.navy}` `#003660`. Blue is secondary / interactive; gold is a rare emphasis
  accent, never a button fill and never a data series.
- **Mono for all data.** Every number, axis label, badge, caption, and code token
  renders
  in JetBrains Mono, for optical column alignment and a data-forward voice.
- **Comfortaa across chrome.** A single rounded geometric sans (Comfortaa) carries both
  content headings / stat values (weight 400) and body / component titles (500 / 600).
  Italic species names are the lone serif exception (IBM Plex Serif italic), since
  Comfortaa has no true italic face.
- **Colorblind-safe by construction.** Fixed Okabe-Ito six-series order, no red-green
  colormaps, vermilion reserved for error / alert, six categorical series maximum before
  an
  "other" category.
- **Navy-tinted, layered elevation.** Shadows are always navy-tinted, never gray or
  black,
  and quiet by default. Heavy elevation is reserved for modal-level surfaces only.
- **Restrained geometry.** A 3 / 6 / 10 / 16px radius ladder. Engineered, never pill.
- **Light theme, one dark exception.** A near-white `#FBFEF8` canvas with pure-white
  cards throughout; the only intentionally dark surface is the image stage, where the
  pixels themselves are the content.

---

## Absolute Constraints

These rules are **never** overridden by component context, user request, or convenience.
Agents must treat violations as hard errors.

- **NEVER** use data colors (Okabe-Ito) for buttons, navigation, headings, text links,
  input borders, or any UI chrome.
- **NEVER** use `#F0E442` (yellow) as text color, stroke, or thin line on white or light
  backgrounds.
- **NEVER** render numeric data, axis labels, badge text, captions, or code outside
  `font-family: 'JetBrains Mono'`.
- **NEVER** apply `--shadow-lg` to inline cards or panel components.
- **NEVER** combine `--oi-blue` (`#0072B2`) and `--color-blue` (`#1b75bc`) in the same
  chart.
- **NEVER** reorder Okabe-Ito series. Series order is fixed:
  navy, orange, sky, green, blue, purple (vermilion reserved for error/alert).
- **NEVER** use more than 6 categorical series in a single chart without introducing an
  "other" category.
- **NEVER** use raw Okabe-Ito hex values as text on white without applying the darkened
  contrast variants listed in the Badges section.
- **NEVER** use red-green colormaps.
- **NEVER** use em dashes. Use double hyphens (`--`) or restructure the sentence.

Add these to the existing "Absolute Constraints" block. They extend, never contradict,
the current rules.

- **NEVER** render a single fluorescence / intensity channel in a hue. Single channels
  display in grayscale. Color is reserved for multi-channel composites.
- **NEVER** use a red-green channel pairing in a composite overlay. Use green / magenta
  (preferred, harmonizes with `--oi-green`) or cyan / red. This is the image-layer
  expression of the existing "no red-green colormaps" rule.
- **NEVER** place a scale bar without a mono-font length label, and never render the bar
  in a color that fails contrast against the image region beneath it (default to white
  with a 1px `rgba(0,0,0,0.4)` outline, or pure black on bright fields).
- **NEVER** draw chart annotation lines (thresholds, means) in a categorical series
  color. Use `--oi-grey` (`#BBBBBB`) or `--color-muted` so annotations never read as a
  data series.
- **NEVER** apply a continuous sequential colorbar built from a categorical series. Use
  the single-variable navy-to-blue ramp already defined in section 06.
- **NEVER** let an image overlay (mask, ROI, box) obscure the underlying pixels at full
  opacity by default. Masks default to outline-only or <= 45% fill.

---

---

## 00 -- Logo and Branding

### Asset Inventory

Three brand assets exist. They are not interchangeable; each has a defined role.

| Asset                    | Contents                                                      | viewBox     | Aspect | Role                                                   |
|--------------------------|---------------------------------------------------------------|-------------|--------|--------------------------------------------------------|
| `light_logo_exfab.svg`   | PhenoTypic wordmark + colony mark + ExFAB/NSF lockup          | 300 x 112.5 | ~8:3   | Wide banner for light surfaces (sidebar header, splash); NOT the navy topbar -- see section 13 |
| `dashboard_logo.svg`     | ExFAB wordmark + "AN NSF BIOFOUNDRY" + NSF seal + colony mark | 300 x 187.5 | ~8:5   | Splash / login / about / exported-figure footer        |
| `LogoArtOnly.png` (icon) | Circular colony "petri dish" mark only                        | 500 x 500   | 1:1    | Favicon, collapsed sidebar, app icon, compact contexts |

- All three are **light-background** assets. There is no dark-background variant yet;
  see
  "Backgrounds" below before placing any of them on a dark surface.
- The colony mark is the shared visual anchor across all three and reads as the product
  symbol on its own.

### Placement by Context

| Context                         | Asset                          | Notes                                                           |
|---------------------------------|--------------------------------|-----------------------------------------------------------------|
| Topbar (navy bar)               | none (wordmark title)          | The topbar is navy (section 13); no light lockup goes on a dark surface. The white view-title carries identity. Add a logo only after a dark-background variant exists. |
| Sidebar header (expanded)       | `light_logo_exfab.svg` or mark | Mark alone if sidebar is <= 240px                               |
| Sidebar (collapsed rail)        | icon mark                      | Centered, 28-32px                                               |
| Login / splash / hero           | `dashboard_logo.svg`           | The fuller lockup with the NSF seal suits first-screen branding |
| About / footer / attribution    | `dashboard_logo.svg`           | Where the NSF Biofoundry credit belongs                         |
| Exported figure footer          | `dashboard_logo.svg`           | Small, alongside the provenance strip (section 15)              |
| Browser favicon / PWA icon      | icon mark                      | 16, 32, 180 (apple-touch), 512 (PWA)                            |

> The topbar is a navy (dark) bar (section 13), so the light-background
> `light_logo_exfab.svg` banner is NOT placed there -- dropping a light lockup on a
> dark surface is forbidden below ("Backgrounds"). The topbar carries the product
> name through the white view-title wordmark instead. If a logo is wanted on the
> topbar, commission a dark-background variant first; do not invert or recolor the
> existing assets. The wide banner / icon mark guidance below applies to light
> surfaces (sidebar header, splash, footers).

### Topbar Implementation

Ties to section 13 (App Shell). The navy topbar carries the white view-title wordmark
(no logo asset). The CSS below applies only if a dark-background logo variant is later
added; it is left for reference.

```css
.topbar-logo {
    height: 32px; /* banner scales to fit the 56px bar with padding */
    width: auto;
    display: block;
}

.topbar-logo--icon { /* collapsed / narrow state */
    height: 28px;
    width: 28px;
}
```

- Prefer the SVG assets in the live UI for crispness at any DPI. Reserve PNG for the
  favicon and any context that cannot consume SVG.

### Clear Space & Minimum Size

Recommended defaults; adjust if your shell is unusually dense.

| Asset                  | Clear space (all sides)    | Min size                                                                  |
|------------------------|----------------------------|---------------------------------------------------------------------------|
| `light_logo_exfab.svg` | height of the "e" in eXFAB | ~180px wide (below this the "AN NSF BIOFOUNDRY" line stops being legible) |
| `dashboard_logo.svg`   | height of the "e" in eXFAB | ~160px wide                                                               |
| icon mark              | 25% of mark diameter       | 24px (favicon floor; colony detail is lost smaller)                       |

The fine tagline text in both lockups is the limiting factor for minimum size. When you
need the brand smaller than the legible floor, switch to the icon mark instead of
shrinking a lockup.

### Backgrounds

- Place lockups on `--color-white` or `--color-bg` only. Both assets carry a subtle
  light container and are tuned for light surfaces.
- **Do not** drop a light-background lockup onto a dark panel (for example, the dark
  image stage in section 09). If a logo is needed on a dark surface, commission a proper
  dark-background variant; do not invert or recolor these files, which would break the
  NSF seal and the colony mark colors.

### Brand Mark Colors vs Data Colors -- Constraint

The colony mark uses soft decorative colony hues (pinks, purples, greens, yellows,
blues) plus a thin circuit-trace motif. These are **brand decoration**, not the data
palette.

- **NEVER** sample a color out of the logo for a chart series, badge, or UI element.
  Charts use the Okabe-Ito series order; UI uses the primary palette. The logo sits
  outside both systems.
- This preserves the same strict UI / data color separation the rest of this document
  enforces. The logo is a third, closed category: branding only.

### Do / Don't

**Do**

- Use the wide banner on light surfaces (sidebar header, splash), the full lockup for
  splash and footers, the icon mark for favicon and collapsed states. The navy topbar
  carries the wordmark title, not a logo asset (section 13).
- Keep lockups on light surfaces with the clear space above.
- Use SVG in the UI; PNG only for favicons.

**Don't**

- Recolor, invert, or restyle any asset to fit a dark surface.
- Stretch or change the aspect ratio (8:3 banner, 8:5 lockup, 1:1 mark).
- Shrink a lockup below its legible floor; switch to the icon mark instead.
- Borrow logo colony colors for data or UI.

---

## 01 -- Color Palette

### Primary Colors -- UI Only

Brand identity, UI structure, and interactive elements. These colors are **not** used
for data series.

| Token           | Hex       | Usage                                     |
|-----------------|-----------|-------------------------------------------|
| `--color-navy`  | `#003660` | Headings, nav, primary CTA, table headers |
| `--color-blue`  | `#1b75bc` | Links, interactive states, accent borders |
| `--color-gold`  | `#febc11` | Highlights, brand accent, emphasized CTA  |
| `--color-white` | `#ffffff` | Card surfaces, panel backgrounds          |
| `--color-bg`    | `#FBFEF8` | App background, inset/recessed areas      |

### Data Colors -- Okabe-Ito -- Visualization Only

Colorblind-safe palette designed by Masataka Okabe and Kei Ito. Safe across
deuteranopia, protanopia, and tritanopia. Used **exclusively** for data visualization:
plot series, progress bars, status badges, and semantic states.

| Token            | Name           | Hex       | CB Safe   | Primary Use                |
|------------------|----------------|-----------|-----------|----------------------------|
| `--oi-orange`    | Orange         | `#E69F00` | All types | Series 2, warning          |
| `--oi-sky`       | Sky Blue       | `#56B4E9` | All types | Series 3, info             |
| `--oi-green`     | Bluish Green   | `#009E73` | All types | Series 4, success          |
| `--oi-vermilion` | Vermilion      | `#D55E00` | All types | Error/alert series only    |
| `--oi-blue`      | Blue           | `#0072B2` | All types | Series 5                   |
| `--oi-purple`    | Reddish Purple | `#CC79A7` | All types | Series 6                   |
| `--oi-yellow`    | Yellow         | `#F0E442` | All types | Series 7, large fills only |
| `--oi-grey`      | Grey           | `#BBBBBB` | All types | Reference/control/null     |

> **Source:** Okabe M, Ito K (2008). "Color Universal Design."
> *jfly.iam.u-tokyo.ac.jp/color*

> **Yellow (`#F0E442`):** Contrast ratio on white is ~1.9:1. Reserve for filled chart
> elements (bars, area fills, large markers) where size compensates. Never use as text
> or thin lines on white.

> **`--oi-blue` (`#0072B2`):** Visually close to `--color-blue` (`#1b75bc`) at small
> sizes. Use for series 5 only after 4 prior series are present, or substitute grey for
> reference/control lines.

> **Grey (`#BBBBBB`):** Not formally part of Okabe-Ito but universally CB-safe. Strongly
> recommended for reference lines, baseline means, and negative control series.

### Surface & Neutral Tokens

| Token             | Hex       | Usage                                      |
|-------------------|-----------|--------------------------------------------|
| `--color-border`  | `#dde3ed` | Card borders, input outlines, dividers     |
| `--color-rule`    | `#e8ecf2` | Chart gridlines, table row rules, dividers |
| `--color-muted`   | `#8892a4` | Secondary text, labels, captions, axes     |
| `--color-body`    | `#2e3a4e` | Primary body text                          |
| `--color-heading` | `#003660` | All heading text (alias of navy)           |

### Semantic Color Assignments

| State   | Token                  | Hex       | Use Case                                   |
|---------|------------------------|-----------|--------------------------------------------|
| Success | `--color-oi-green`     | `#009E73` | Completed jobs, BUSCO pass, positive delta |
| Info    | `--color-oi-sky`       | `#56B4E9` | Pipeline notices, metadata callouts        |
| Warning | `--color-oi-orange`    | `#E69F00` | Threshold failures, outlier flags          |
| Error   | `--color-oi-vermilion` | `#D55E00` | OOM errors, failed segments, abort states  |

### Color Selection Decision Logic

Use these rules to determine which palette an element draws from:

- **Buttons, nav, headings, links, tabs, inputs, modals:**
  Primary palette only -- `#003660`, `#1b75bc`, `#febc11`, `#ffffff`, `#FBFEF8`
- **Chart series, progress fills, heatmap cells, donut segments, bar fills:**
  Fixed series order -- `#003660`, `#E69F00`, `#56B4E9`, `#009E73`, `#0072B2`, `#CC79A7`
  Use `#D55E00` only for error/alert/failed series.
  Use `#BBBBBB` for reference lines, baseline means, negative controls.
- **State indicators (success/info/warning/error):**
  `#009E73`, `#56B4E9`, `#E69F00`, `#D55E00`
- **Single-variable heatmaps:**
  Ramp from `rgba(0,54,96,0.08)` through `#56B4E9` to `#003660`.
  Use `#D55E00` at 0.7 opacity for failed or null wells.
- **Badges on white surfaces:**
  Use the darkened text variants from the Badges section.
- **`#febc11` as text on white:**
  Substitute `#a87a00` (darkened for WCAG AA compliance).

---

## 02 -- Typography

Five layers, narrowest to broadest:

1. **Font families** -- three role fonts (CSS tokens + `_design.py` constants).
2. **Size primitives + semantic aliases** -- the raw ladder and the named size tokens.
3. **Line-height & tracking tokens.**
4. **Text styles** -- named composite recipes (Title, Header, Body, Caption ...). What
   every component references.
5. **Element defaults + typography rules.**

Call-site discipline (carried over from the original spec, unchanged):

- New code references a **text style** (CSS class `.text-*`, or the matching
  `TEXT_STYLE_*` Python constant).
- When only a size is needed, use a **semantic alias** (`--font-size-*` /
  `FONT_SIZE_*`), never a raw `--text-*` primitive.
- Raw `--text-*` primitives are kept for back-compat and must not appear in new call
  sites.
- Python inline styles import `FONT_FAMILY_*` and `FONT_SIZE_*` from
  `gui/_design.py`; never hardcode a literal.

---

### 02.1 -- Font Families

Four role tokens. Nothing else may declare a `font-family`.

```css
--font-display: 'Comfortaa', -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
--font-body:    'Comfortaa', -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
--font-mono:    'JetBrains Mono', ui-monospace, "SFMono-Regular", Menlo, "Liberation Mono", monospace;
--font-species: 'IBM Plex Serif', Georgia, "Times New Roman", Times, serif;
```

```html

<link
        href="https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
        rel="stylesheet"
/>
```

`gui/_design.py` constants (the Python call-site source of truth):

```python
FONT_FAMILY_DISPLAY = "'Comfortaa', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
FONT_FAMILY_BODY = "'Comfortaa', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
FONT_FAMILY_MONO = "'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, 'Liberation Mono', monospace"
FONT_FAMILY_SPECIES = "'IBM Plex Serif', Georgia, 'Times New Roman', Times, serif"
```

Role intent:

| Token            | Role    | Carries                                                                      |
|------------------|---------|------------------------------------------------------------------------------|
| `--font-display` | Display | Content headings, large stat values                                          |
| `--font-body`    | Body    | Prose, button / tab labels, component titles                                 |
| `--font-mono`    | Mono    | All numeric data, axis labels, badge / overline / label text, captions, code |
| `--font-species` | Species | **Italic** binomial species names only (the one non-Comfortaa text surface)  |

> **The chrome runs on one family.** Comfortaa carries both the display and body roles,
> so headings, stat values, prose, and UI titles all share a single rounded geometric
> sans. The role tokens still exist independently, so a future split back into two
> families is mechanical -- change `_DISPLAY_PRIMARY` / `_BODY_PRIMARY` in
> `gui/_design.py` and every call site inherits it with no edits.

> **Italics need a serif.** Comfortaa ships no true italic face, so `font-style: italic`
> on a Comfortaa run would render a browser-synthesized oblique. Italic species names
> (`.is-species`) therefore switch to `--font-species` (IBM Plex Serif italic) for a real
> italic cut. Do not apply `--font-species` to anything but italic binomials.

> **Charts keep IBM Plex.** The `@import` also loads **IBM Plex Sans** and **IBM Plex
> Serif** (regular + italic) even though no chrome `--font-*` token references IBM Plex
> Sans. The chart subsystem (`viz/figures/_theme.py`, §06) is deliberately *not* migrated
> to Comfortaa -- plot titles / legend names stay IBM Plex Sans and donut center values
> stay IBM Plex Serif -- and those plots render inside GUI pages, so the families must
> stay loaded. Chrome chooses Comfortaa; charts keep IBM Plex; the two intentionally
> differ.

> **Weights.** Comfortaa ships 400 / 500 / 600 / 700; JetBrains Mono ships 400 / 500 / 600;
> IBM Plex Sans 400 / 500 / 600 / 700 and IBM Plex Serif 400 / 500 / 600 (+ italics) are
> loaded for the chart subsystem. Display styles default to 400 to keep headings light;
> you may raise Header / Title to 500 / 600 / 700 for heavier hierarchy. Only those
> families and weights are loaded; do not reference a weight outside the imported set.

---

### 02.2 -- Size Primitives & Semantic Aliases

The scale is rem-based, rooted on 15px body text. **New call sites use the semantic
alias** (right column). The raw `--text-*` primitive is back-compat only.

| Role            | Primitive     | Size             | Semantic alias         | Python const         |
|-----------------|---------------|------------------|------------------------|----------------------|
| Display         | `--text-3xl`  | 2.5rem / 40px    | `--font-size-display`  | `FONT_SIZE_DISPLAY`  |
| Title           | `--text-2xl`  | 1.875rem / 30px  | `--font-size-title`    | `FONT_SIZE_TITLE`    |
| Header (h2)     | `--text-xl`   | 1.5rem / 24px    | `--font-size-header-1` | `FONT_SIZE_HEADER_1` |
| H2 (h3)         | `--text-lg`   | 1.25rem / 20px   | `--font-size-header-2` | `FONT_SIZE_HEADER_2` |
| H3 / Body Large | `--text-md`   | 1.0625rem / 17px | `--font-size-body-lg`  | `FONT_SIZE_BODY_LG`  |
| Body            | `--text-base` | 0.9375rem / 15px | `--font-size-body`     | `FONT_SIZE_BODY`     |
| Body Small      | `--text-sm`   | 0.8125rem / 13px | `--font-size-body-sm`  | `FONT_SIZE_BODY_SM`  |
| Label / Caption | `--text-xs`   | 0.6875rem / 11px | `--font-size-caption`  | `FONT_SIZE_CAPTION`  |
| Data Micro      | `--text-2xs`  | 0.625rem / 10px  | `--font-size-micro`    | `FONT_SIZE_MICRO`    |
| (reserve)       | `--text-4xl`  | 3.25rem / 52px   | --                     | --                   |

```css
--font-size-display:
var

(
--text-3xl

)
;
--font-size-title:
var

(
--text-2xl

)
;
--font-size-header-1:
var

(
--text-xl

)
;
--font-size-header-2:
var

(
--text-lg

)
;
--font-size-body-lg:
var

(
--text-md

)
;
--font-size-body:
var

(
--text-base

)
;
--font-size-body-sm:
var

(
--text-sm

)
; /* renamed from --font-size-label; see 02.7 */
--font-size-caption:
var

(
--text-xs

)
;
--font-size-micro:
var

(
--text-2xs

)
; /* new: chart axis / sparkline floor */
```

> SVG-rendered chart internals (axis ticks, scale-bar labels) may go to 8px directly,
> since they are drawn, not DOM text. `--font-size-micro` (10px) is the DOM-text floor.

---

### 02.3 -- Line-height & Tracking Tokens

```css
--leading-display:

1.1
;
--leading-tight:

1.2
;
--leading-snug:

1.3
;
--leading-normal:

1.45
;
--leading-relaxed:

1.6
;

--tracking-tight:

-
0.02
em

;
--tracking-snug:

-
0.01
em

;
--tracking-normal:

0
;
--tracking-button:

0.01
em

;
--tracking-wide:

0.08
em

;
--tracking-wider:

0.12
em

;
```

| Line-height         | Used by                      | Tracking            | Used by               |
|---------------------|------------------------------|---------------------|-----------------------|
| `--leading-display` | Display                      | `--tracking-tight`  | Display               |
| `--leading-tight`   | Header, Label, Data Micro    | `--tracking-snug`   | Title, Header         |
| `--leading-snug`    | Title, H2, H3, UI Title      | `--tracking-normal` | H2, H3, Body, Caption |
| `--leading-normal`  | Caption, Data Value          | `--tracking-button` | Button                |
| `--leading-relaxed` | Body Large, Body, Body Small | `--tracking-wide`   | Label                 |
|                     |                              | `--tracking-wider`  | Overline              |

---

### 02.4 -- Text Styles  (the named categories)

Every piece of text uses exactly one of these. Components reference the **Style** name.
"Default color" is overridable in context; family, size, weight, line-height, and
tracking are fixed.

| Style                  | Family  | Size (alias)           | Weight | Line-height         | Tracking                               | Transform | Default color     | Used for                                                |
|------------------------|---------|------------------------|--------|---------------------|----------------------------------------|-----------|-------------------|---------------------------------------------------------|
| **Display**            | display | `--font-size-display`  | 400    | `--leading-display` | `--tracking-tight`                     | none      | `--color-heading` | Stat-card values, hero numbers                          |
| **Title**              | display | `--font-size-title`    | 400    | `--leading-snug`    | `--tracking-snug`                      | none      | `--color-heading` | Page / view title (h1)                                  |
| **Header**             | display | `--font-size-header-1` | 400    | `--leading-tight`   | `--tracking-snug`                      | none      | `--color-heading` | Major section heading (h2), modal title                 |
| **H2**                 | display | `--font-size-header-2` | 400    | `--leading-snug`    | `--tracking-normal`                    | none      | `--color-heading` | Subsection heading (h3)                                 |
| **H3**                 | display | `--font-size-body-lg`  | 400    | `--leading-snug`    | `--tracking-normal`                    | none      | `--color-heading` | Minor heading (h4)                                      |
| **Body Large**         | body    | `--font-size-body-lg`  | 400    | `--leading-relaxed` | `--tracking-normal`                    | none      | `--color-body`    | Lead paragraph, intro copy                              |
| **Body**               | body    | `--font-size-body`     | 400    | `--leading-relaxed` | `--tracking-normal`                    | none      | `--color-body`    | Default paragraph copy                                  |
| **Body Small**         | body    | `--font-size-body-sm`  | 400    | `--leading-relaxed` | `--tracking-normal`                    | none      | `--color-body`    | Dense / secondary prose                                 |
| **UI Title**           | body    | `--font-size-body-sm`  | 600    | `--leading-snug`    | `--tracking-normal`                    | none      | `--color-heading` | Chart / card / alert / panel titles                     |
| **Button**             | body    | `--font-size-body-sm`  | 500    | 1                   | `--tracking-button`                    | none      | per variant       | Button labels, nav tab labels                           |
| **Label / Overline**   | mono    | `--font-size-caption`  | 500    | `--leading-tight`   | `--tracking-wide` / `--tracking-wider` | uppercase | `--color-muted`   | Form labels, table headers, overlines, badge text       |
| **Caption**            | mono    | `--font-size-caption`  | 400    | `--leading-normal`  | `--tracking-normal`                    | none      | `--color-muted`   | Hints, figure captions, chart subtitle, scale-bar label |
| **Data Value**         | mono    | `--font-size-body`     | 500    | `--leading-normal`  | `--tracking-normal`                    | none      | `--color-heading` | Numeric table cells, stat deltas, tooltip values        |
| **Data Value (muted)** | mono    | `--font-size-body`     | 400    | `--leading-normal`  | `--tracking-normal`                    | none      | `--color-muted`   | Table secondary cells                                   |
| **Data Micro**         | mono    | `--font-size-micro`    | 400    | `--leading-tight`   | 0.02em                                 | none      | `--color-muted`   | Chart axis ticks, sparkline labels, dense data          |

Notes:

- **Label vs Overline** share the recipe; only tracking differs. `--tracking-wide`
  (0.08em) for inline labels and table headers, `--tracking-wider` (0.12em) for
  standalone
  section overlines.
- **Data Value** size follows its container; `--font-size-body` is the default. The
  fixed
  parts are mono family, weight 500, heading color. The **muted** variant (weight 400,
  `--color-muted`) is for secondary table cells.
- **Species names** add the `.is-species` class (italic), which also switches the run to
  `--font-species` (IBM Plex Serif italic) -- Comfortaa has no true italic, so the
  family swap is what makes the binomial a real italic rather than a synthesized oblique.
  Apply it to Title / Header / H2 / H3 species runs as needed; only IBM Plex Serif italic
  400 / 500 are loaded.

---

### 02.5 -- Text Styles in CSS

A component applies one class; it never re-declares these properties. Classes reference
the semantic aliases, not raw primitives.

```css
.text-display {
    font-family: var(--font-display);
    font-size: var(--font-size-display);
    font-weight: 400;
    line-height: var(--leading-display);
    letter-spacing: var(--tracking-tight);
    color: var(--color-heading);
}

.text-title {
    font-family: var(--font-display);
    font-size: var(--font-size-title);
    font-weight: 400;
    line-height: var(--leading-snug);
    letter-spacing: var(--tracking-snug);
    color: var(--color-heading);
}

.text-header {
    font-family: var(--font-display);
    font-size: var(--font-size-header-1);
    font-weight: 400;
    line-height: var(--leading-tight);
    letter-spacing: var(--tracking-snug);
    color: var(--color-heading);
}

.text-h2 {
    font-family: var(--font-display);
    font-size: var(--font-size-header-2);
    font-weight: 400;
    line-height: var(--leading-snug);
    color: var(--color-heading);
}

.text-h3 {
    font-family: var(--font-display);
    font-size: var(--font-size-body-lg);
    font-weight: 400;
    line-height: var(--leading-snug);
    color: var(--color-heading);
}

.text-body-lg {
    font-family: var(--font-body);
    font-size: var(--font-size-body-lg);
    font-weight: 400;
    line-height: var(--leading-relaxed);
    color: var(--color-body);
}

.text-body {
    font-family: var(--font-body);
    font-size: var(--font-size-body);
    font-weight: 400;
    line-height: var(--leading-relaxed);
    color: var(--color-body);
}

.text-body-sm {
    font-family: var(--font-body);
    font-size: var(--font-size-body-sm);
    font-weight: 400;
    line-height: var(--leading-relaxed);
    color: var(--color-body);
}

.text-ui-title {
    font-family: var(--font-body);
    font-size: var(--font-size-body-sm);
    font-weight: 600;
    line-height: var(--leading-snug);
    color: var(--color-heading);
}

.text-button {
    font-family: var(--font-body);
    font-size: var(--font-size-body-sm);
    font-weight: 500;
    line-height: 1;
    letter-spacing: var(--tracking-button);
}

.text-label {
    font-family: var(--font-mono);
    font-size: var(--font-size-caption);
    font-weight: 500;
    line-height: var(--leading-tight);
    letter-spacing: var(--tracking-wide);
    text-transform: uppercase;
    color: var(--color-muted);
}

.text-overline {
    font-family: var(--font-mono);
    font-size: var(--font-size-caption);
    font-weight: 500;
    line-height: var(--leading-tight);
    letter-spacing: var(--tracking-wider);
    text-transform: uppercase;
    color: var(--color-muted);
}

.text-caption {
    font-family: var(--font-mono);
    font-size: var(--font-size-caption);
    font-weight: 400;
    line-height: var(--leading-normal);
    color: var(--color-muted);
}

.text-data {
    font-family: var(--font-mono);
    font-size: var(--font-size-body);
    font-weight: 500;
    line-height: var(--leading-normal);
    color: var(--color-heading);
}

.text-data--muted {
    font-family: var(--font-mono);
    font-size: var(--font-size-body);
    font-weight: 400;
    line-height: var(--leading-normal);
    color: var(--color-muted);
}

.text-data-micro {
    font-family: var(--font-mono);
    font-size: var(--font-size-micro);
    font-weight: 400;
    line-height: var(--leading-tight);
    letter-spacing: 0.02em;
    color: var(--color-muted);
}

.is-species {
    font-style: italic;
}

/* modifier for display-family headings */
```

> **Python parity:** mirror these as a `TEXT_STYLE` mapping in `gui/_design.py` (one
> entry
> per style returning the family / size / weight / leading / tracking / color), so
> Python
> call sites apply a style by name exactly as CSS does by class. Add `--text-2xs`, the
> line-height tokens, and the tracking tokens to the section 07 `:root` block; the other
> primitives and aliases already exist there.

---

### 02.6 -- Element Defaults & Typography Rules

#### HTML element defaults

| Element       | Style                        |
|---------------|------------------------------|
| `h1`          | Title                        |
| `h2`          | Header                       |
| `h3`          | H2                           |
| `h4`          | H3                           |
| `p`           | Body                         |
| `p.lead`      | Body Large                   |
| `small`       | Caption                      |
| `label`, `th` | Label                        |
| `button`, tab | Button                       |
| `code`, `kbd` | mono inline (see rule below) |
| numeric `td`  | Data Value                   |

#### Typography rules (carried over from the original, mapped to styles)

- **Headings** use the display family at weight 400. Italic cut for Latin species names
  (e.g. *Rhodotorula toruloides*) via `.is-species`.
- **Labels and overlines:** Label / Overline style (mono, uppercase, muted).
- **Stat card values:** Display style.
- **Data values** always render in the mono family to preserve optical column alignment
  (Data Value style).
- **Table numeric cells:** Data Value style (mono, weight 500, `--color-heading`).
- **Table secondary cells:** Data Value (muted) style (mono, `--color-muted`).
- **Inline code:** mono family, `background: #edf2f7`, `color: #003660`,
  `padding: 1px 5px`,
  `border-radius: 3px`.
- **Maximum line length:** 65ch for Body, 52ch for Body Large.

---

### 02.7 -- Reconciliations & Flags

Confirm before treating this as final.

1. **Label alias renamed.** The original named the 13px rung `--font-size-label`, but
   every
   component renders labels at 11px mono. This spec renames the 13px rung to
   `--font-size-body-sm` (Body Small / UI Title / Button) and routes the Label /
   Overline
   style to `--font-size-caption` (11px), matching real usage. Update any
   `--font-size-label`
   references in `gui/_design.py` and CSS to `--font-size-body-sm`, or keep a deprecated
   alias `--font-size-label: var(--font-size-body-sm)` during migration.

2. **Heading vs. component-title weight.** Content headings (Title, Header, H2, H3) and
   component titles (**UI Title**) now share one family -- Comfortaa -- distinguished by
   weight (headings 400, UI Title 600) rather than the old serif-to-sans shift. Both map
   to Comfortaa via `--font-display` / `--font-body`, which carry the same stack.

3. **Body family.** Body is Comfortaa (same family as display). The chrome runs on a
   single rounded geometric sans; only italic species names (`--font-species`, IBM Plex
   Serif) and mono data (`--font-mono`, JetBrains Mono) step outside it. To split display
   and body back into two families, change `_DISPLAY_PRIMARY` / `_BODY_PRIMARY` in
   `gui/_design.py` and update the import.

---

## 03 -- Spacing & Layout

8-point base grid. All spacing tokens are multiples of `0.25rem` (4px).

| Token     | Value   | px   |
|-----------|---------|------|
| `--sp-1`  | 0.25rem | 4px  |
| `--sp-2`  | 0.5rem  | 8px  |
| `--sp-3`  | 0.75rem | 12px |
| `--sp-4`  | 1rem    | 16px |
| `--sp-5`  | 1.25rem | 20px |
| `--sp-6`  | 1.5rem  | 24px |
| `--sp-8`  | 2rem    | 32px |
| `--sp-10` | 2.5rem  | 40px |
| `--sp-12` | 3rem    | 48px |
| `--sp-16` | 4rem    | 64px |

- **Section rhythm:** Major sections separated by `--sp-16` top margin and a
  `1px solid --color-rule` divider.
- **Card internal padding:** `--sp-6` (24px).

### Grid & Container

#### Container

Two container modes. Both live inside the App Shell content area (to the right of the
240px sidebar, section 13).

| Mode           | Width                               | Gutters                                         | Use                                                                    |
|----------------|-------------------------------------|-------------------------------------------------|------------------------------------------------------------------------|
| **Contained**  | `max-width: 1600px`, centered       | `--sp-8` (32px) desktop, `--sp-4` (16px) mobile | Default. Keeps chart sizes and line lengths sane on ultrawide displays |
| **Full-bleed** | Fills the content area edge to edge | Outer `--sp-8` edge padding only                | Wide data tables, plate maps, image montages that benefit from width   |

```css
.container {
    width: 100%;
    max-width: 1600px;
    margin-inline: auto;
    padding-inline: var(--sp-8);
}

.container--bleed {
    max-width: none;
    padding-inline: var(--sp-8);
}
```

- Contained bands sit on the `--color-bg` (`#FBFEF8`) canvas; cards within are
  `--color-white` with a `--color-border` hairline and the resting shadow (section 04).
- Full-bleed is an opt-in for data-dense surfaces only, not a default. Body prose still
  respects the Body / Body Large max line lengths from section 02.

#### Panel Grid (12-column)

The canonical layout primitive for a view. 12 columns, `--sp-6` (24px) gap. This is the
same grid the App Shell references; do not define a second grid.

```css
.panel-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: var(--sp-6);
}

.col-3 {
    grid-column: span 3;
}

.col-4 {
    grid-column: span 4;
}

.col-6 {
    grid-column: span 6;
}

.col-8 {
    grid-column: span 8;
}

.col-12 {
    grid-column: span 12;
}
```

Desktop spans:

| Element                                    | Span        | Per row                    |
|--------------------------------------------|-------------|----------------------------|
| Stat / KPI card                            | `col-3`     | 4-up                       |
| Secondary chart                            | `col-4`     | 3-up                       |
| Standard chart card                        | `col-6`     | 2-up                       |
| Primary / featured chart                   | `col-8`     | with a `col-4` companion   |
| Well-plate, wide data table, image montage | `col-12`    | full                       |
| Image panel                                | `col-6` min | never below a ~360px stage |

Mixed sizing is encouraged, mirroring the reference's feature-cards-span-two idea: a
primary chart (`col-8`) pairs with a supporting `col-4` panel (legend, stat list, or
controls) on the same row, and a hero KPI can span `col-6` among `col-3` siblings.

#### Grid Patterns (canonical compositions)

```
+--------------------------------------------------------------+
|  [ KPI col-3 ] [ KPI col-3 ] [ KPI col-3 ] [ KPI col-3 ]     |  stat row
+--------------------------------------------------------------+
|  [ primary chart            col-8        ] [ context col-4 ] |  focus + context
+--------------------------------------------------------------+
|  [ chart col-6            ] [ chart col-6                  ] |  chart pair
+--------------------------------------------------------------+
|  [ well-plate / wide table / image montage     col-12     ] |  full-width data
+--------------------------------------------------------------+
```

- **Stat row** -- four `col-3` cards across the top of a view.
- **Focus + context** -- one `col-8` primary chart plus one `col-4` supporting panel.
- **Chart pair** -- two `col-6` charts side by side.
- **Full-width data** -- plate map, wide table, or image montage at `col-12`.

Collapse behavior for each pattern at each breakpoint is defined in the Responsive
Strategy section (next).

#### Section Rhythm (carried from section 03)

- Major view sections are separated by `--sp-16` (64px) top margin and a `1px solid
  --color-rule` divider.
- Panel internal padding is `--sp-6` (24px). Dense data panels (tables, plate maps) may
  drop to `--sp-4` (16px) to maximize content area.
- Gap between panels in the grid is `--sp-6` (24px); never collapse panels flush.

### Responsive Strategy

#### Breakpoints

| Name         | Width         | Key changes                                                            |
|--------------|---------------|------------------------------------------------------------------------|
| Mobile       | < 480px       | Single column; sidebar becomes a drawer; topbar collapses to hamburger |
| Mobile-Large | 480 to 767px  | As Mobile; thumbnail grids may show 2 across                           |
| Tablet       | 768 to 991px  | 2-up panel grids; sidebar collapses to icon rail                       |
| Desktop      | 992 to 1599px | Full multi-up grids; sidebar expanded                                  |
| Wide         | >= 1600px     | Container stops growing at `1600px` and centers (Grid & Container)     |

```css
/* breakpoint reference values */
--bp-mobile-lg:

480
px

;
--bp-tablet:

768
px

;
--bp-desktop:

992
px

;
--bp-wide:

1600
px

;
```

CSS variables cannot be used inside media query conditions; the tokens above are the
documented values to write into `@media (min-width: 992px)` and equivalents.

#### Touch Targets

| Control                                                | Minimum target                                                                                | Standard       |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------|
| Primary / secondary buttons, nav links, tabs           | 44 x 44px                                                                                     | WCAG 2.5.5 AAA |
| Dense icon controls (image toolbar, table row actions) | 24 x 24px absolute floor, with >= `--sp-2` (8px) spacing between adjacent targets             | WCAG 2.5.8 AA  |
| Plate wells / thumbnails on touch                      | tap to select; if the rendered cell is below 24px, require zoom (section 10) before selection | --             |

- Default buttons reach 44px via vertical padding (`--sp-3` 12px) plus the Button line
  box; do not shrink primary actions below this on touch surfaces.
- The image toolbar (section 09) uses icon buttons; on touch, increase their hit area to
  44px even when the icon glyph stays small.

#### Collapsing Strategy

Each grid pattern from Grid & Container collapses as follows.

| Pattern (desktop)                   | Tablet (768 to 991)         | Mobile (< 768)   |
|-------------------------------------|-----------------------------|------------------|
| Stat row (4 x `col-3`)              | 2-up                        | 1-up (stacked)   |
| Chart pair (2 x `col-6`)            | 1-up                        | 1-up             |
| Focus + context (`col-8` + `col-4`) | stack: primary then context | stacked          |
| Full-width data (`col-12`)          | stays full width            | stays full width |
| Image panel (`col-6`)               | full width                  | full width       |
| Thumbnail grid                      | auto-fill (fewer columns)   | 2 across, then 1 |

App shell and navigation:

- **Sidebar:** expanded (240px) at Desktop, icon rail at Tablet, off-canvas drawer at
  Mobile (section 13).
- **Topbar:** navy bar with the full link row + white view-title wordmark at Desktop;
  links collapse to a hamburger below `--bp-tablet` (sections 00 / 13). No logo asset
  sits on the navy bar.
- **Wide data table:** never reflow columns; scroll horizontally inside its container at
  Tablet and Mobile, with the first column frozen.
- **Well-plate:** never shrink wells past legibility; below the format minimum width
  (section 10) switch to horizontal scroll plus zoom rather than scaling wells down.

#### Image Behavior

Imagery in this system is data, not decoration. The marketing convention of full-bleed
photographic heroes does not apply.

- **Image panels** preserve native pixel aspect ratio and letterbox against the stage
  background (section 09). The stage scales down to a ~360px minimum, below which the
  panel
  takes full width (`col-12`).
- **Thumbnail / montage grids** are already fluid via
  `repeat(auto-fill, minmax(96px, 1fr))`
  (section 09); columns reduce naturally with width.
- **Charts** reflow to container width and reduce axis-tick density (fewer ticks) rather
  than letting labels overlap; the Data Micro style is the floor (section 02 / 11).
- **Plate maps** scroll and zoom below their minimum width rather than shrinking wells
  (section 10).
- **Scale bars, colorbars, and legends** persist at every breakpoint and in exports;
  they
  are never dropped to save space (section 09 / 12 / 15).
- No decorative or portrait photography anywhere in the surface.

---

## 04 -- Shapes & Elevation

### Border Radius

| Token         | Value | Usage                                        |
|---------------|-------|----------------------------------------------|
| `--radius-sm` | 3px   | Inline code, badges, small chips             |
| `--radius`    | 6px   | Buttons, inputs, small cards                 |
| `--radius-md` | 10px  | Stat cards, chart cards, main cards          |
| `--radius-lg` | 16px  | Hero panels, modal surfaces, feature banners |

### Elevation & Depth

All shadows are navy-tinted, `rgba(0, 54, 96, ...)`, never gray or black, so depth reads
as part of the brand rather than a default browser drop-shadow. Elevation is quiet by
default; heavy shadow is reserved for genuinely transient surfaces. The ladder assigns
each shadow token a level and a use.

| Level | Name     | Treatment                                                | Token         | Use                                                       |
|-------|----------|----------------------------------------------------------|---------------|-----------------------------------------------------------|
| **0** | Flat     | No border, no shadow                                     | --            | Canvas bands, full-bleed sections, inline content         |
| **1** | Hairline | `1px solid --color-border` on `--color-white`, no shadow | --            | Inputs, table containers, recessed / inset surfaces       |
| **2** | Resting  | Hairline + `--shadow-sm`                                 | `--shadow-sm` | Default cards, stat cards, chart cards at rest            |
| **3** | Raised   | Hairline + `--shadow`                                    | `--shadow`    | Hovered cards, the featured / primary panel, swipe handle |
| **4** | Floating | `--shadow-md`                                            | `--shadow-md` | Dropdowns, popovers, tooltips, sticky bars, modals        |
| **5** | Overlay  | `--shadow-lg`                                            | `--shadow-lg` | Full-page overlays, lightbox, fullscreen image stage      |

- **Hover transition:** interactive cards animate from Level 2 to Level 3 on hover via
  `transition: border-color, box-shadow` (the same motion used by thumbnails and chart
  cards in sections 09 and 11). Resting state returns to Level 2.
- **Do not stack:** a surface already at a level does not gain a second shadow when
  nested
  inside another elevated surface. Pick one level per surface.

#### Elevation Tokens (recipes)

```css
--shadow-sm:

0
1
px

3
px
rgba

(
0
,
54
,
96
,
0.07
)
,
0
1
px

2
px
rgba

(
0
,
54
,
96
,
0.04
)
; /* Level 2 -- resting  */
--shadow:

0
4
px

12
px
rgba

(
0
,
54
,
96
,
0.08
)
,
0
1
px

3
px
rgba

(
0
,
54
,
96
,
0.05
)
; /* Level 3 -- raised   */
--shadow-md:

0
8
px

24
px
rgba

(
0
,
54
,
96
,
0.10
)
,
0
2
px

6
px
rgba

(
0
,
54
,
96
,
0.06
)
; /* Level 4 -- floating */
--shadow-lg:

0
16
px

40
px
rgba

(
0
,
54
,
96
,
0.12
)
,
0
4
px

12
px
rgba

(
0
,
54
,
96
,
0.07
)
; /* Level 5 -- overlay  */
```

Each token is a two-stop shadow: a soft, wide far-shadow plus a tight near-shadow, which
gives a natural falloff. This layered pairing is the system's only atmospheric effect.

#### Decorative Depth

- **Navy tint, never neutral.** Depth carries the brand hue. A gray or black shadow is a
  defect in this system, not a stylistic alternative.
- **The hairline is the primary separator.** On the near-white `#FBFEF8` canvas, the
  canvas-to-card lightness gap is about one percent, so the `1px --color-border` at
  Level 1
  does most of the work of separating a card from the page; the shadow is a secondary
  cue.
  This is why nearly every card keeps its hairline even at higher levels.
- **One intentional dark surface.** The image stage (section 09) is deliberately dark
  because the pixels are the data. That is a content canvas, not an elevation level, and
  no
  UI chrome ever uses a dark fill to imply depth.
- **Restraint.** Most surfaces live at Level 1 or 2. Reserve Level 4 and 5 for surfaces
  that genuinely float above the page or sit over a scrim. Do not push a resting card to
  Level 4 to make it stand out; use the `col` span or a navy accent bar instead.

#### Reconciliation Flags

1. **Radius splits out.** This section now covers depth only. The Border Radius table
   from
   the old section 04 should become its own short "Shapes" section (content unchanged:
   the
   3 / 6 / 10 / 16px ladder), matching the reference's separation of Shapes from
   Elevation.
2. **Modal level.** Your original token table placed modals at `--shadow-md` and labeled
   `--shadow-lg` as "hero, full-page overlays." I kept modals at Level 4 (`--shadow-md`)
   per that original intent, moved "hero" off the heaviest level since hero bands are
   normally flat (Level 0), and reframed Level 5 as full-page overlays and the lightbox.
   If
   you prefer the reference's "modal is the heaviest surface," bump modals to Level 5.

---

## 05 -- Components

### Stat Cards

Each stat card has a 3px top accent bar whose color communicates category at a glance.
Use primary colors for top-level KPIs; Okabe-Ito for categorically meaningful metrics.

**Accent bar color assignments:**

| Color                      | Hex       | Metric Type                                |
|----------------------------|-----------|--------------------------------------------|
| Navy `--color-navy`        | `#003660` | Primary KPIs (total count, overall status) |
| Blue `--color-blue`        | `#1b75bc` | Measurement values (diameter, area)        |
| Gold `--color-gold`        | `#febc11` | Coverage / utilization percentages         |
| Orange `--oi-orange`       | `#E69F00` | Warning states, flagged conditions         |
| Bluish Green `--oi-green`  | `#009E73` | Quality / completeness / BUSCO scores      |
| Sky Blue `--oi-sky`        | `#56B4E9` | Throughput / job counts                    |
| Vermilion `--oi-vermilion` | `#D55E00` | Error / failure counts                     |

**Delta text colors:** positive `#009E73`, negative `#D55E00`, neutral `#8892a4`.

**Anatomy:**

```
+-- 3px accent bar (color by category) -------------------------+
|  LABEL -- JetBrains Mono -- 11px -- uppercase -- --color-muted       |
|                                                               |
|  Value -- Comfortaa -- 2.5rem                               |
|  Delta -- JetBrains Mono -- 11px -- #009E73 (up) / #D55E00 (down)   |
+---------------------------------------------------------------+
```

**CSS:**

```css
.stat-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--sp-5) var(--sp-6);
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent-color); /* set per instance */
}
```

---

### Progress Bars

Use Okabe-Ito series order for multi-fill progress. Track background is always
`--color-rule`.

**Variants:**

| Variant   | Height  | Use Case                                         |
|-----------|---------|--------------------------------------------------|
| Standard  | 6px     | Multi-step pipelines, comparison series          |
| Thick     | 10px    | Primary KPI progress                             |
| Segmented | 6--10px | Proportional composition (e.g. strain breakdown) |

**Color series order** for multi-progress lists:

| Position  | Name           | Hex       |
|-----------|----------------|-----------|
| 1         | Navy           | `#003660` |
| 2         | Orange         | `#E69F00` |
| 3         | Sky Blue       | `#56B4E9` |
| 4         | Bluish Green   | `#009E73` |
| 5         | Blue           | `#0072B2` |
| 6         | Reddish Purple | `#CC79A7` |
| 7 (error) | Vermilion      | `#D55E00` |

Reserve `#D55E00` for error-state fills only.

**Label layout:** progress name (Comfortaa 500) left, value (JetBrains Mono) right,
`justify-content: space-between`. Stack items with `gap: 16px`.

**CSS:**

```css
.progress-track {
    height: 6px;
    background: var(--color-rule);
    border-radius: 9999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 9999px;
    background-image: linear-gradient(
            90deg,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.25) 100%
    );
}
```

---

### Buttons

Primary buttons use the brand palette. Okabe-Ito colors are **not** used for buttons --
reserved exclusively for data. Exception: danger variant uses `--oi-vermilion`.

| Variant | Background     | Text            | Border           | Use Case             |
|---------|----------------|-----------------|------------------|----------------------|
| Primary | `--color-navy` | White           | `--color-navy`   | Submit, Run, Confirm |
| Blue    | `--color-blue` | White           | `--color-blue`   | Primary interactive  |
| Gold    | `--color-gold` | `--color-navy`  | `--color-gold`   | Emphasized CTA       |
| Outline | Transparent    | `--color-navy`  | `--color-border` | Secondary actions    |
| Ghost   | Transparent    | `--color-muted` | none             | Tertiary / utility   |
| Danger  | Transparent    | `#D55E00`       | `#D55E00`        | Destructive actions  |

**Sizes:**

| Size    | Font Size | Padding                           |
|---------|-----------|-----------------------------------|
| sm      | 11px      | `0.3rem 0.75rem`                  |
| default | 13px      | `0.5rem 1.125rem`                 |
| lg      | 15px      | `0.7rem 1.5rem`                   |
| icon    | --        | `width/height 2.25rem, padding 0` |

**Shared styles:** Comfortaa, `font-weight: 500`, `letter-spacing: 0.01em`,
`border-radius: var(--radius)`, `border-width: 1.5px`,
`transition: all 180ms cubic-bezier(0.22, 1, 0.36, 1)`.

**Hover states:**

- Filled variants: darken background ~8%, add
  `box-shadow: 0 4px 12px rgba(0,54,96,0.28)`
- Outline: `border-color: #1b75bc`, `color: #1b75bc`,
  `background: rgba(27,117,188,0.04)`
- Danger: `background: #D55E00`, `color: #fff`

---

### Badges

Monospaced, uppercase micro-labels for status, classification, and categorical tagging.

**Structure:**
`[colored dot 5px] [text -- JetBrains Mono -- 10.5px -- uppercase -- letter-spacing: 0.08em]`

**Styling:** `display: inline-flex`, `align-items: center`, `gap: 5px`,
`font-size: 0.65rem`, `font-weight: 500`, `padding: 0.2rem 0.55rem`,
`border-radius: 9999px`, `border: 1px solid`.

**Color variants (background / text / border):**

| Variant   | Background               | Text (darkened for AA) | Border                   |
|-----------|--------------------------|------------------------|--------------------------|
| Navy      | `rgba(0,54,96,0.08)`     | `#003660`              | `rgba(0,54,96,0.15)`     |
| Blue      | `rgba(27,117,188,0.08)`  | `#1b75bc`              | `rgba(27,117,188,0.20)`  |
| Orange    | `rgba(230,159,0,0.10)`   | `#9A6B00`              | `rgba(230,159,0,0.25)`   |
| Sky Blue  | `rgba(86,180,233,0.10)`  | `#0B6E9E`              | `rgba(86,180,233,0.25)`  |
| Green     | `rgba(0,158,115,0.08)`   | `#006B4F`              | `rgba(0,158,115,0.20)`   |
| Vermilion | `rgba(213,94,0,0.08)`    | `#D55E00`              | `rgba(213,94,0,0.20)`    |
| Purple    | `rgba(204,121,167,0.10)` | `#8B3D6E`              | `rgba(204,121,167,0.25)` |

> All badge text colors are darkened from raw Okabe-Ito values to meet WCAG AA (4.5:1)
> on white surfaces. **Do not** use raw Okabe-Ito hex as badge text.

Include a 5px colored dot for active/live states (Running, Processing). Omit dot for
static labels.

---

### Alerts & Callouts

Left border in semantic Okabe-Ito color, background tint at 7--10% opacity against white
card surfaces.

**Structure:** `display: flex`, `gap: var(--sp-4)`, `padding: var(--sp-4) var(--sp-5)`,
`border-radius: var(--radius)`, `border-left: 4px solid`.

**Icon:** `1rem`, `flex-shrink: 0`, `margin-top: 1px`.
**Title:** Comfortaa, `font-weight: 600`, `font-size: 13px`.
**Body:** `opacity: 0.85`, `line-height: 1.5`.

| Type    | Border    | Background              | Text      |
|---------|-----------|-------------------------|-----------|
| Info    | `#56B4E9` | `rgba(86,180,233,0.08)` | `#0B5E87` |
| Success | `#009E73` | `rgba(0,158,115,0.07)`  | `#005C43` |
| Warning | `#E69F00` | `rgba(230,159,0,0.10)`  | `#7A5500` |
| Error   | `#D55E00` | `rgba(213,94,0,0.08)`   | `#8A3C00` |

---

### Data Tables

| Element          | Style                                                                      |
|------------------|----------------------------------------------------------------------------|
| Header border    | `2px solid --color-navy` (bottom)                                          |
| Header text      | JetBrains Mono, 11px, uppercase, `letter-spacing: 0.08em`, `--color-muted` |
| Cell padding     | `12px 16px`                                                                |
| Row divider      | `1px solid --color-rule`                                                   |
| Row hover        | `background: rgba(27,117,188,0.03)`                                        |
| Numeric values   | JetBrains Mono, `--color-heading`, `font-weight: 500`                      |
| Secondary values | JetBrains Mono, `--color-muted`                                            |

**Column ordering convention:** identifier, taxonomy/category, measurements (JetBrains
Mono),
quality metrics, status badge.

---

### Form Inputs

**Base styles:** `background: #ffffff`, `border: 1.5px solid --color-border`,
`border-radius: var(--radius)`, `padding: 0.5rem 0.875rem`, Comfortaa 13px,
`color: --color-body`, `transition: border-color 180ms, box-shadow 180ms`.

| State   | Border                       | Focus Ring                        |
|---------|------------------------------|-----------------------------------|
| Default | `1.5px solid --color-border` | --                                |
| Focus   | `1.5px solid --color-blue`   | `0 0 0 3px rgba(27,117,188,0.12)` |
| Valid   | `1.5px solid #009E73`        | `0 0 0 3px rgba(0,158,115,0.12)`  |
| Error   | `1.5px solid #D55E00`        | `0 0 0 3px rgba(213,94,0,0.12)`   |

**Labels:** JetBrains Mono, 11px, uppercase, `letter-spacing: 0.08em`, `--color-muted`,
placed above input.
**Hint text:** 11px, `--color-muted`.
**Valid text:** 11px, `#006B4F`.
**Error text:** 11px, `#D55E00`.

---

### Navigation Tabs

```css
/* Track */
border-bottom:

2
px solid
var

(
--color-rule

)
; /* full width */

/* Tab */
font-family:
var

(
--font-body

)
;
font-size:

13
px

;
font-weight:

500
;
padding:

12
px

20
px

;
border-bottom:

2
px solid transparent

;
margin-bottom:

-
2
px

;

/* States */
/* active  */
color:
var

(
--color-navy

)
;
border-color:
var

(
--color-navy

)
;
/* inactive */
color:
var

(
--color-muted

)
;
/* hover (inactive) */
color:
var

(
--color-body

)
;
border-color:
var

(
--color-border

)
;
```

---

## 06 -- Data Visualization

### Categorical Series Order

Apply Okabe-Ito colors in this fixed order for all multi-series charts. Navy anchors
series 1, keeping the primary series harmonious with the UI chrome.

| Series          | Name           | Hex       | Notes                                         |
|-----------------|----------------|-----------|-----------------------------------------------|
| 1               | Navy           | `#003660` | Primary series, anchors to UI brand           |
| 2               | Orange         | `#E69F00` | Warm anchor, distinct from all others         |
| 3               | Sky Blue       | `#56B4E9` | High luminance, good for thin lines           |
| 4               | Bluish Green   | `#009E73` | Perceptually equidistant from orange & sky    |
| 5               | Blue           | `#0072B2` | Use carefully near `--color-blue` UI elements |
| 6               | Reddish Purple | `#CC79A7` | High distinctiveness                          |
| 7 (alert/error) | Vermilion      | `#D55E00` | Error / failed / alert series **only**        |
| ref             | Grey           | `#BBBBBB` | Reference lines, negative controls, null      |

### Chart Styling Rules

| Element           | Style                                                  |
|-------------------|--------------------------------------------------------|
| Gridlines         | `1px`, `#e8ecf2`                                       |
| Axes              | `1.5px`, `#dde3ed`                                     |
| Axis labels       | JetBrains Mono, 7--8px, `--color-muted`                |
| Chart title       | IBM Plex Sans, 13px, weight 600, `--color-heading`           |
| Chart subtitle    | JetBrains Mono, 11px, `--color-muted`                  |
| Spines            | Top and right hidden; bottom and left only             |
| Data point dots   | 3.5px radius, filled with series color                 |
| Area fills        | Series color at 7% opacity beneath line                |
| Line stroke width | 2px, `stroke-linejoin: round`, `stroke-linecap: round` |
| Error bars        | 1px, series color, 60% opacity                         |

### Donut / Pie Charts

```
ring stroke-width:  20px on r=40 SVG circle (circumference ~251px)
center value:       IBM Plex Serif, font-size 11, font-weight 600, fill #003660
center unit:        JetBrains Mono, font-size 6.5, fill #8892a4
legend dot:         10px circle
legend name:        IBM Plex Sans, 11px, color #2e3a4e
legend pct:         JetBrains Mono, 11px, color #8892a4
```

### Heatmap Colorscale (Single-Variable)

For single-variable intensity maps (e.g. 96-well plate colony density):

| Intensity     | Value                | Notes                 |
|---------------|----------------------|-----------------------|
| Low           | `rgba(0,54,96,0.08)` | Near-transparent navy |
| Mid           | `#56B4E9`            | Okabe-Ito sky blue    |
| High          | `#003660`            | Full navy             |
| Failed / null | `#D55E00` at 70%     | Okabe-Ito vermilion   |

Avoids red-green colormaps which fail under deuteranopia. The navy-to-blue ramp is
maximally distinct from the vermilion fail state under all CB types.

---

## 07 -- Code Integration

### matplotlib / seaborn rcParams

Apply this block when generating matplotlib figures:

```python
import matplotlib as mpl

OKABE_ITO = [
    "#003660",  # navy     -- series 1, UI-harmonized
    "#E69F00",  # orange   -- series 2
    "#56B4E9",  # sky blue -- series 3
    "#009E73",  # green    -- series 4
    "#0072B2",  # blue     -- series 5
    "#CC79A7",  # purple   -- series 6
    "#D55E00",  # vermilion -- error/alert series
]

mpl.rcParams.update({
    "axes.prop_cycle"  : mpl.cycler(color=OKABE_ITO),
    "axes.facecolor"   : "#ffffff",
    "figure.facecolor" : "#FBFEF8",
    "axes.edgecolor"   : "#dde3ed",
    "axes.grid"        : True,
    "grid.color"       : "#e8ecf2",
    "grid.linewidth"   : 0.8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["IBM Plex Sans", "Helvetica Neue", "Arial"],
    "axes.labelcolor"  : "#2e3a4e",
    "xtick.color"      : "#8892a4",
    "ytick.color"      : "#8892a4",
    "axes.titlecolor"  : "#003660",
    "axes.titleweight" : "600",
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
})
```

### napari Label Layer Colors

For categorical label overlays (e.g. colony segmentation masks):

```python
# RGBA tuples normalized 0-1, for napari label layer color dict
OKABE_ITO_NAPARI = {
    1: (0 / 255, 54 / 255, 96 / 255, 1.0),  # navy
    2: (230 / 255, 159 / 255, 0 / 255, 1.0),  # orange
    3: (86 / 255, 180 / 255, 233 / 255, 1.0),  # sky blue
    4: (0 / 255, 158 / 255, 115 / 255, 1.0),  # bluish green
    5: (0 / 255, 114 / 255, 178 / 255, 1.0),  # blue
    6: (204 / 255, 121 / 255, 167 / 255, 1.0),  # reddish purple
    7: (213 / 255, 94 / 255, 0 / 255, 1.0),  # vermilion (error)
}
```

### CSS Custom Properties

Include this `:root` block in all generated CSS files:

```css
:root {
    /* Primary palette -- UI only */
    --color-navy: #003660;
    --color-blue: #1b75bc;
    --color-gold: #febc11;
    --color-white: #ffffff;
    --color-bg: #FBFEF8;
    --color-surface: #ffffff;
    --color-border: #dde3ed;
    --color-rule: #e8ecf2;
    --color-muted: #8892a4;
    --color-body: #2e3a4e;
    --color-heading: #003660;

    /* Data palette -- Okabe-Ito -- visualization only */
    --oi-orange: #E69F00;
    --oi-sky: #56B4E9;
    --oi-green: #009E73;
    --oi-vermilion: #D55E00;
    --oi-blue: #0072B2;
    --oi-purple: #CC79A7;
    --oi-yellow: #F0E442; /* large fills only */
    --oi-grey: #BBBBBB; /* reference / control */

    /* Semantic aliases */
    --color-success: var(--oi-green);
    --color-info: var(--oi-sky);
    --color-warning: var(--oi-orange);
    --color-danger: var(--oi-vermilion);

    /* Typography */
    --font-display: 'Comfortaa', system-ui, sans-serif;
    --font-body: 'Comfortaa', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Courier New', monospace;
    --font-species: 'IBM Plex Serif', Georgia, serif; /* italic binomials only */

    /* Type scale */
    --text-xs: 0.6875rem; /*  11px */
    --text-sm: 0.8125rem; /*  13px */
    --text-base: 0.9375rem; /*  15px */
    --text-md: 1.0625rem; /*  17px */
    --text-lg: 1.25rem; /*  20px */
    --text-xl: 1.5rem; /*  24px */
    --text-2xl: 1.875rem; /*  30px */
    --text-3xl: 2.5rem; /*  40px */
    --text-4xl: 3.25rem; /*  52px */
    --text-2xs: 0.625rem; /*  10px */

    /* Semantic size aliases -- use these, not raw --text-* */
    --font-size-display: var(--text-3xl);
    --font-size-title: var(--text-2xl);
    --font-size-header-1: var(--text-xl);
    --font-size-header-2: var(--text-lg);
    --font-size-body-lg: var(--text-md);
    --font-size-body: var(--text-base);
    --font-size-body-sm: var(--text-sm);
    --font-size-caption: var(--text-xs);
    --font-size-micro: var(--text-2xs);

    /* Line-height */
    --leading-display: 1.1;
    --leading-tight: 1.2;
    --leading-snug: 1.3;
    --leading-normal: 1.45;
    --leading-relaxed: 1.6;

    /* Letter-spacing */
    --tracking-tight: -0.02em;
    --tracking-snug: -0.01em;
    --tracking-normal: 0;
    --tracking-button: 0.01em;
    --tracking-wide: 0.08em;
    --tracking-wider: 0.12em;

    /* Spacing (8pt grid) */
    --sp-1: 0.25rem;
    --sp-2: 0.5rem;
    --sp-3: 0.75rem;
    --sp-4: 1rem;
    --sp-5: 1.25rem;
    --sp-6: 1.5rem;
    --sp-8: 2rem;
    --sp-10: 2.5rem;
    --sp-12: 3rem;
    --sp-16: 4rem;

    /* Border radius */
    --radius-sm: 3px;
    --radius: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;

    /* Shadows (navy-tinted) */
    --shadow-sm: 0 1px 3px rgba(0, 54, 96, 0.07), 0 1px 2px rgba(0, 54, 96, 0.04);
    --shadow: 0 4px 12px rgba(0, 54, 96, 0.08), 0 1px 3px rgba(0, 54, 96, 0.05);
    --shadow-md: 0 8px 24px rgba(0, 54, 96, 0.10), 0 2px 6px rgba(0, 54, 96, 0.06);
    --shadow-lg: 0 16px 40px rgba(0, 54, 96, 0.12), 0 4px 12px rgba(0, 54, 96, 0.07);

    /* Motion */
    --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
    --transition: 180ms var(--ease-out);
}
```

---

## 08 -- Usage Rules & Anti-Patterns

### Do

- **Keep the three color worlds separate.** UI chrome (navy / blue / gold plus
  neutrals),
  data (Okabe-Ito), and branding (the logo) never borrow from one another.
- **Reserve navy `#003660` for primary actions, headings, and the wordmark.** Use blue
  `#1b75bc` for interactive and secondary roles, and gold `#febc11` only as a rare
  emphasis
  accent (darken to `#a87a00` for text on white).
- **Use Okabe-Ito for chart series, progress fills, and semantic states** (success /
  info /
  warning / error), in the fixed order navy, orange, sky, green, blue, purple.
- **Render every number, axis label, badge, caption, and code token in the mono family**
  (JetBrains Mono), to keep optical column alignment and a data-forward voice.
- **Use Comfortaa for content headings, large stat values, body, and component titles**
  -- one rounded geometric sans across all chrome. Italic species names are the lone
  exception (IBM Plex Serif italic, via `.is-species` / `--font-species`).
- **Pair color with a second signal on every status.** A dot, an icon, or a text word,
  so
  meaning survives for colorblind readers.
- **Keep shadows navy-tinted and quiet, and let the hairline border do the primary
  separating** on the near-white `#FBFEF8` canvas.
- **Hold buttons and inputs at `--radius` 6px and cards at `--radius-md` 10px.** Keep
  the
  geometry restrained and engineered.
- **Put a scale bar on every calibrated image, and a persistent colorbar or legend on
  every
  data figure.** They stay in exports too.
- **Show single fluorescence channels in grayscale and composites in green / magenta**,
  not
  green / red.
- **Reference text styles and tokens by name.** The document is the single source of
  truth;
  never hardcode a font, size, or hex at a call site.

### Don't

- **Don't use data colors (Okabe-Ito) for buttons, nav, headings, links, input borders,
  or
  any UI chrome.**
- **Don't reorder the Okabe-Ito series,** and don't exceed six categorical series
  without
  introducing an "other" category.
- **Don't use red-green colormaps,** and don't rely on a green-success / red-error pair
  without a dot or label to carry the meaning.
- **Don't use `#F0E442` (yellow) as text, stroke, or a thin line on white.** Reserve it
  for
  large filled chart elements only.
- **Don't combine `--oi-blue` (`#0072B2`) and `--color-blue` (`#1b75bc`) in the same
  chart.**
  They read as the same color at small sizes.
- **Don't render data, labels, captions, or code in anything but the mono family.**
- **Don't use gray or black shadows,** and don't apply `--shadow-lg` to inline cards or
  panels (reserve it for overlays).
- **Don't use raw Okabe-Ito hex as text on white** without the darkened contrast variant
  from the Badges section.
- **Don't use chromatic data colors as button fills, and don't use gold as a data
  series.**
- **Don't render pill-shaped CTAs,** and don't stretch the logo or sample its colony
  colors
  for UI or data.
- **Don't use em dashes.** Use double hyphens or restructure the sentence.

---

## 09 -- Image Display & Viewers

The dashboard surfaces acquired image data: plate scans, colony fields, brightfield and
fluorescence microscopy, and segmentation results. Images are **data**, so the same
discipline that governs charts governs image color: grayscale per channel, CB-safe
composites, and overlays that never destroy the underlying signal.

### Image Panel -- Anatomy

```
+-- chart-card surface (radius-md, shadow-sm, 1px --color-border) -----+
|  TITLE -- Comfortaa 13 600 --color-heading          [toolbar: icons right] |
|  subtitle -- JetBrains Mono 11 --color-muted                                |
|  +-- image stage (background #0e1620 for fluor, #FBFEF8 for bright)+ |
|  |                                                                  | |
|  |    [ image pixels ]              [overlay layer: mask/ROI/box]   | |
|  |                                                                  | |
|  |                                          [scale bar, lower-right]| |
|  +------------------------------------------------------------------+ |
|  [channel chips]   [metadata strip -- JetBrains Mono 11 --color-muted]      |
+----------------------------------------------------------------------+
```

- **Stage background:** dark (`#0e1620`) for fluorescence so low-signal pixels read;
  light (`--color-bg`) for brightfield, colony plates, and gels. The stage is the one
  place a near-black surface is permitted in this light-theme system, because it is the
  data canvas, not UI chrome.
- **Stage radius:** `--radius` (6px), `overflow: hidden`. The card around it uses
  `--radius-md`.
- **Aspect:** preserve native pixel aspect ratio. Never stretch. Letterbox with the
  stage background.

### Image Toolbar

Icon buttons, `icon` size from the Buttons section, `ghost` variant. Group order:

| Group      | Controls                                  |
|------------|-------------------------------------------|
| Navigation | zoom in, zoom out, fit-to-view, reset 1:1 |
| Display    | channel toggle, LUT/colormap, brightness  |
| Overlay    | mask on/off, ROI on/off, labels on/off    |
| Capture    | fullscreen, download (see section 15)     |

- Zoom level reads in a mono chip at lower-left: `JetBrains Mono 11 --color-muted`, e.g.
  `220%`.
- Active toggle state uses the same active-tab treatment: `--color-navy` icon, 2px
  bottom or ring accent. Inactive icons are `--color-muted`.

### Scale Bar  (microscopy / calibrated images)

A scale bar is mandatory for any calibrated image and is the single most important
trust signal in a microscopy panel.

```
position:    lower-right (default) or lower-left, --sp-4 inset
bar:         height 3px, white #ffffff, outline 1px rgba(0,0,0,0.4)
             (on bright fields: pure black, no outline)
label:       JetBrains Mono 11px, same color as bar, centered above bar
             e.g. "20 um"  (use "um", never the micro glyph, to stay ASCII-safe)
length:      snap to a round calibrated value (5, 10, 20, 50, 100 um ...)
```

State the calibration source in the metadata strip so the bar is auditable.

> **Status -- deferred (pending core calibration support).** No scale bar ships in
> the results viewer yet, and it is intentionally blocked rather than skipped: a bar
> requires a physical pixel size (um/px), which the core image model does not carry
> (`_core/_image_parts/_image_io_handler.py` has `# TODO: implement calibration
> schema`; only an experimental `RESOLUTION` tag exists). Because nothing is
> calibrated today, the "mandatory for calibrated images" rule above is vacuously
> satisfied. Implementing the bar is a cross-cutting feature, not a styling fix:
> (1) add the calibration schema to the core `Image` model, (2) plumb um/px through
> the pipeline into the results viewer, then (3) render the bar (e.g. the
> openseadragon-scalebar plugin, not currently bundled). Render the bar only when a
> calibration value is present; omit it (do not fake one) otherwise.

### Channels & LUTs

This is the image-data analog of the categorical series order. The rules below are
the established microscopy-accessibility conventions, not house preference.

- **Single channel:** grayscale only. Do not pseudocolor a lone channel.
- **Composite (2 channels):** green / magenta preferred; cyan / red acceptable. Do not
  use green / red.
- **Composite (3 channels):** green / magenta / cyan, or magenta / yellow / cyan. Always
  offer a one-click "split to grayscale panels" view beside the composite.
- **Brightfield:** displayed as captured; no LUT.

Channel chips below the stage carry a mono label and a swatch of the assigned LUT:

```
[ GFP   ]  swatch = green ramp     JetBrains Mono 10.5 uppercase
[ mCh   ]  swatch = magenta ramp
[ merge ]  swatch = split gradient
```

> **Sources (CB-safe LUT convention):** Single channels in grayscale, individual
> channels shown alongside any color composite, and green/magenta (or cyan/red) in place
> of red/green are standard fluorescence-figure accessibility guidance. See ASCB,
> "How to make scientific figures accessible to readers with color-blindness" (2025);
> the Node, "Color-blind people are your audience too" (2021); Bankhead, *Analyzing
> Fluorescence Microscopy Images with ImageJ*. Aligns with Okabe & Ito (2008), already
> cited in section 01. Verify current ASCB URL before publishing the doc.

### Overlay Layers -- Masks, ROIs, Detections

Overlays reuse the `OKABE_ITO_NAPARI` label mapping already defined in section 07, so a
mask color in the dashboard matches the same label in napari.

| Overlay type        | Default render                                          |
|---------------------|---------------------------------------------------------|
| Segmentation mask   | outline 1.5px series color, OR <= 45% fill, never 100%  |
| Labeled instances   | outline series color + mono ID at centroid, 10px        |
| ROI (user-drawn)    | `--color-blue` 1.5px dashed, 8% fill                    |
| Detection box       | series color 1.5px solid, mono confidence chip top-left |
| Reference / control | `--oi-grey` outline, to read as non-data                |

- Opacity is user-adjustable but defaults conservative so pixels stay legible.
- A legend for label classes uses the section 12 categorical legend.

> **Exception -- high-cardinality instance maps (decided, intentional).** The
> `OKABE_ITO_NAPARI` mapping above is for **low-cardinality categorical** overlays
> (a handful of label *classes*: e.g. mask / ROI / detection-type). It is **not**
> applied to dense **instance** segmentations where every colony is its own label --
> an arrayed plate routinely has hundreds of objects, and cycling a 7-color palette
> across them makes touching colonies share a color, which defeats the overlay's
> purpose (telling neighbors apart). For those instance maps the GUI keeps
> scikit-image `label2rgb` (with a matplotlib `tab20` fallback), which spreads many
> perceptually-distinct hues. This is a deliberate usability-over-brand-consistency
> call. It governs the builder "Run preview" detector/refiner overlay and the tune
> Curate overlay, which share one renderer (`gui/builder/_image_renderer.py:
> to_overlay_rgb_array`). Apply `OKABE_ITO_NAPARI` only when the label count is small
> enough that each class gets a stable, distinct brand color; revisit if a future
> overlay is genuinely class-based (few categories) rather than per-instance.

### Comparison Modes

| Mode          | Use case                         | Layout                          |
|---------------|----------------------------------|---------------------------------|
| Side-by-side  | two conditions, raw vs segmented | 2-up grid, shared zoom/pan      |
| Swipe slider  | before/after, channel A vs B     | single stage, draggable divider |
| Composite tog | channels overlaid vs split       | toolbar toggle, no relayout     |
| Linked sync   | many fields, one zoom state      | grid, `sync-zoom` shared state  |

Swipe divider handle: `--color-navy` 2px line, 28px circular grip, `--shadow-sm`.

### Thumbnail Grid / Montage / Gallery

For plate fields, well montages, and image sets.

```css
.thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
    gap: var(--sp-2);
}

.thumb {
    aspect-ratio: 1 / 1;
    border-radius: var(--radius-sm);
    border: 1.5px solid var(--color-border);
    overflow: hidden;
    cursor: pointer;
    transition: border-color var(--transition), box-shadow var(--transition);
}

.thumb:hover {
    border-color: var(--color-blue);
    box-shadow: var(--shadow-sm);
}

.thumb.selected {
    border-color: var(--color-navy);
    box-shadow: var(--shadow);
}

.thumb.flagged {
    border-color: var(--oi-vermilion);
}

/* QC fail */
```

- Caption under each thumb: mono 11px, well ID or field index.
- A status dot (5px, from Badges) marks pass / warn / fail per thumb.
- Clicking a thumb opens the full Image Panel; do not open a raw lightbox without the
  toolbar and scale bar.

### Image States

| State    | Treatment                                                            |
|----------|----------------------------------------------------------------------|
| Loading  | skeleton block at stage aspect, shimmer (section 14)                 |
| Failed   | centered `--oi-vermilion` icon + mono "Image failed to load" + retry |
| Empty    | centered `--color-muted` icon + mono "No image for this selection"   |
| Decoding | progress bar (thick variant) at stage bottom, navy fill              |

---

## 10 -- Well-Plate Grid

The canonical microbiology layout: 96-well (8x12) and 384-well (16x24). Well fill uses
the **single-variable** colorscale from section 06, not categorical series, because a
plate map encodes one continuous variable (density, OD, growth).

### Plate -- Anatomy

```
       1    2    3   ...  12          row/col headers: JetBrains Mono 10 --color-muted
   A  ( )  ( )  ( )  ... ( )
   B  ( )  ( )  ( )  ... ( )          well: circle, radius scales to fit
   ...                                fill: navy-to-blue ramp by value
   H  ( )  ( )  ( )  ... ( )          stroke: 1px --color-border

   [ colorbar ]  low ........ high    continuous legend (section 12)
```

### Well States & Value Mapping

| Well state        | Fill                                           | Stroke                      |
|-------------------|------------------------------------------------|-----------------------------|
| Value (low->high) | `rgba(0,54,96,0.08)` -> `#56B4E9` -> `#003660` | 1px `--color-border`        |
| Failed / null     | `#D55E00` at 70% opacity                       | 1px `#D55E00`               |
| Empty / no sample | `--color-bg`                                   | 1px dashed `--color-border` |
| Control           | `--oi-grey` fill                               | 1px `#8892a4`               |
| Selected          | current fill                                   | 2px `--color-navy`          |
| Hover             | current fill + `--shadow-sm`                   | 1.5px `--color-blue`        |

This is identical to the section 06 heatmap rule, so a plate map and a heatmap of the
same data read the same. Do not introduce a second ramp.

### Well Interaction

- **Hover tooltip:** well ID, raw value, derived metric (all mono). See section 12.
- **Drag-select:** rectangular marquee, `--color-blue` 1.5px dashed, 8% fill (matches
  ROI). Selected wells get the 2px navy stroke.
- **Row / column headers** are clickable to select an entire row or column.

### Plate Density / Spacing

| Format | Well shape | Gap    | Min stage width |
|--------|------------|--------|-----------------|
| 96     | circle     | --sp-2 | ~520px          |
| 384    | rounded sq | --sp-1 | ~720px          |
| 1536   | square     | 1px    | scroll / zoom   |

Below the minimum width, enable zoom/pan (reuse the Image Toolbar navigation group)
rather than shrinking wells past legibility.

---

## 11 -- Extended Chart Types

All chart types inherit the section 06 series order, gridline, axis, and spine rules.
Listed below are the type-specific additions. Axis labels and all numeric annotations
remain `JetBrains Mono` per the typography constraints.

### Bar Charts (grouped / stacked)

| Property      | Value                                                        |
|---------------|--------------------------------------------------------------|
| Bar fill      | series order; single-series defaults to `--color-navy`       |
| Bar corner    | `--radius-sm` on the free end only (top for vertical)        |
| Group gap     | 0.2 of band width; bar gap within group 0.08                 |
| Stacked order | series order bottom-up; "other" category last in `--oi-grey` |
| Value labels  | JetBrains Mono 11, `--color-heading`, only when < ~12 bars   |
| Baseline      | 1.5px `--color-border` at zero                               |

Reserve `--oi-vermilion` for a failed / alert bar only, never as series 7 by default.

### Box Plot / Violin   (distributions -- common in this field)

| Element         | Style                                                  |
|-----------------|--------------------------------------------------------|
| Box fill        | series color at 18% opacity                            |
| Box outline     | series color 1.5px                                     |
| Median line     | series color 2px                                       |
| Whiskers        | series color 1px, cap 1px                              |
| Outlier points  | series color, 2.5px radius, 60% opacity                |
| Violin shape    | series color outline 1.5px, fill 12%; overlay box thin |
| Jittered points | optional, 2px, 40% opacity, slight x-jitter            |

Show n per group as a mono caption beneath each category label. Distribution plots
without n are not publication-honest.

### Scatter + Regression

| Element         | Style                                                |
|-----------------|------------------------------------------------------|
| Points          | series color, 3.5px radius (matches section 06 dots) |
| Point opacity   | 70%; drop to 40% above ~500 points to show density   |
| Regression line | `--color-navy` 2px, or series color if grouped       |
| Confidence band | same color at 12% opacity fill                       |
| Identity / y=x  | `--oi-grey` 1px dashed                               |
| R-squared / fit | mono chip, upper-left, `--color-muted`               |

### Growth Curve / Time-Series   (OD600, kinetics)

| Element         | Style                                                    |
|-----------------|----------------------------------------------------------|
| Line            | series order, 2px, round join/cap                        |
| Error band      | mean +/- SD or SEM, series color 12% fill; state which   |
| Replicate lines | optional thin 1px, 30% opacity, same color as mean       |
| Markers         | optional 3.5px at measured timepoints only               |
| Log axis        | label as "log10" in mono; gridlines at decades           |
| Phase markers   | `--oi-grey` dashed verticals + mono label (lag/log/stat) |

### Histogram / KDE

- Bars: single fill `--color-navy` at 85%, no per-bar coloring.
- KDE overlay: `--color-navy` 2px line on 8% fill.
- Bin count or bandwidth stated in a mono caption; binning choices change the story, so
  surface them.

### Small Multiples / Facet Grid

- Shared axes across facets; label only the outer edge.
- Facet title: JetBrains Mono 11 uppercase `--color-muted`, top-left of each cell.
- 1px `--color-rule` between cells, never heavy borders.
- One series color across all facets when the facet *is* the grouping variable.

### Sparkline   (inline, in tables / stat cards)

- 1.5px line, `--color-blue`, no axes, no gridlines.
- Optional end-dot 3px in `--color-navy`; min/max dots in `--oi-grey`.
- Height 18-24px; lives in a table cell or beneath a stat-card value.

### Volcano / MA Plot   [optional -- omics, include only if relevant]

- Non-significant points `--oi-grey` 60%; up `--oi-green`; down `--oi-purple`
  (avoids the red-green default volcano).
- Threshold lines `--oi-grey` 1px dashed (fold-change and p-value cutoffs).
- This pairing keeps significance direction CB-distinguishable.

---

## 12 -- Chart Support Elements

### Continuous Colorbar  (for plate maps, heatmaps, intensity overlays)

```
orientation: horizontal (default) or vertical
gradient:    rgba(0,54,96,0.08) -> #56B4E9 -> #003660   (section 06 ramp)
track:       height 10px (h) / width 10px (v), radius 9999px
ticks:       3-5, JetBrains Mono 10, --color-muted
end labels:  low / high or numeric, JetBrains Mono 11
null swatch: separate 14px square, #D55E00 @70%, label "fail/null"
```

Never build a sequential colorbar from categorical series colors.

### Categorical Legend

- Marker: 10px circle (line/area charts) or 10px rounded square (bar/box).
- Name: `IBM Plex Sans 11 --color-body`. Optional value/pct:
  `JetBrains Mono 11 --color-muted`.
- Order matches series order exactly. Wrap, never scroll, for <= 6 entries; introduce
  "other" beyond that (existing constraint).
- Interactive legends: clicking dims a series to 20% opacity rather than removing it, so
  axis scale stays stable.

### Tooltip

```css
.tooltip {
    background: var(--color-navy);
    color: #ffffff;
    border-radius: var(--radius-sm);
    padding: var(--sp-2) var(--sp-3);
    box-shadow: var(--shadow-md);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.5;
    pointer-events: none;
}
```

- Label row in IBM Plex Sans; all numeric values in JetBrains Mono.
- Series swatch (8px) precedes each value in multi-series tooltips.
- Position above the cursor; flip below near the top edge.

### Threshold & Reference Annotations

- Reference line: `--oi-grey` 1px dashed, mono label at the right margin.
- Highlighted band (e.g. acceptable range): `--oi-grey` at 8% fill.
- Target / spec line: `--color-gold` 1.5px is permitted here as a deliberate brand
  emphasis, but never for a data series.

### Chart States

| State   | Treatment                                                       |
|---------|-----------------------------------------------------------------|
| Loading | axis frame drawn, plot area shimmer skeleton (section 14)       |
| Empty   | centered mono "No data for this selection", muted icon          |
| Error   | `--oi-vermilion` icon + mono message + retry, axis frame hidden |
| Partial | render available series + mono caption "n of m series loaded"   |

---

## 13 -- Dashboard Shell & Layout

### App Shell

```
+-- topbar (h 56px, --color-navy, 1px bottom rule) --------------------+
|  view title (display 20)                    [global filters] [user]   |
+----------------+------------------------------------------------------+
| sidebar 240px  |  content (--color-bg, padding --sp-8)                |
| --color-white  |   +-- panel grid -------------------------------+    |
| 1px right rule |   | stat cards row                              |    |
| nav items      |   | chart cards / image panels (grid)           |    |
|                |   +---------------------------------------------+    |
+----------------+------------------------------------------------------+
```

- **Topbar:** the topbar is a deliberate dark surface -- `--color-navy` background
  with white ink and a gold (`--color-gold`) underline on the active tab. This is an
  intentional brand choice (the one chrome exception to the otherwise-light theme,
  alongside the section 09 image stage); it is NOT the white topbar an earlier draft
  specified. White-on-navy carries the wordmark identity directly, so the topbar does
  not place a logo asset (a light-background lockup on the navy bar would violate the
  section 00 "no light lockup on a dark surface" rule; a dark-background logo variant
  would be required first).
- **Sidebar nav item:** Comfortaa 13, `--color-muted` default; active gets
  `--color-navy` text + 3px left accent bar in `--color-navy` + `rgba(0,54,96,0.05)`
  background. Reuses the active-tab logic. The sidebar itself is `--color-white`.

### Panel Grid

```css
.panel-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: var(--sp-6);
}

/* common spans */
.col-3 {
    grid-column: span 3;
}

/* stat card        */
.col-6 {
    grid-column: span 6;
}

/* half-width chart  */
.col-8 {
    grid-column: span 8;
}

/* primary chart     */
.col-12 {
    grid-column: span 12;
}

/* full-width / plate */
```

- Stat cards: 4-up (`col-3`) on wide, 2-up on tablet, stacked on mobile.
- Image panels prefer `col-6` or larger; never below ~360px stage width.

### Filter / Toolbar Bar

- Sits directly under the topbar or atop a panel: `--color-white`, 1px bottom rule,
  `padding: var(--sp-3) var(--sp-6)`.
- Controls: form inputs (existing spec) + ghost buttons. Active filters render as
  removable badges (existing spec) with a 5px dot.

### Breadcrumb

- JetBrains Mono 11, `--color-muted`; separators `/` in `--color-border`; current crumb
  `--color-heading`.

---

## 14 -- Feedback & Loading States

### Skeleton Loaders

```css
.skeleton {
    background: linear-gradient(
            90deg,
            var(--color-rule) 25%,
            #eef2f7 37%,
            var(--color-rule) 63%
    );
    background-size: 400% 100%;
    border-radius: var(--radius-sm);
    animation: shimmer 1.4s ease-in-out infinite;
}

@keyframes shimmer {
    0% {
        background-position: 100% 0;
    }
    100% {
        background-position: 0 0;
    }
}
```

- Match the skeleton block to the eventual content shape (stat value bar, chart area,
  image stage). Never spin a generic loader where the layout is known.

### Empty States

- Centered, vertically generous (`--sp-12` padding). Muted icon (24px), Comfortaa 13
  title, JetBrains Mono 11 sub-line, optional primary button to act.
- Voice: state what is missing and the next step, not just "no data".

### Toasts / Notifications

- Bottom-right stack, `--color-white`, `--radius-md`, `--shadow-md`, 4px left accent in
  the semantic color (success / info / warning / error from section 01).
- Auto-dismiss 4-6s; errors persist until dismissed.
- Title Comfortaa 13 600; body Comfortaa 13; any code/IDs JetBrains Mono.

### Modal / Dialog

The system references modals in shadows and radius but does not spec them; this fills
it.

```css
.modal {
    background: var(--color-white);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg); /* one of the few valid --shadow-lg uses */
    max-width: 560px;
    padding: var(--sp-8);
}

.modal-overlay {
    background: rgba(0, 54, 96, 0.45);
    backdrop-filter: blur(2px);
}
```

- Title: display family, `--font-size-header-1`, `--color-heading`.
- Footer actions right-aligned; primary (navy) rightmost, ghost cancel left of it.

---

## 15 -- Export & Provenance Strip

A small but high-value addition for a research dashboard, and a direct tie-in to the
reproducibility discipline in the seminar: every exported figure or image should carry
its provenance.

- **Download control:** ghost icon button (section 09 toolbar) offering PNG (raster) and
  SVG / PDF (vector) for plots; PNG / TIFF for images. State format in a mono dropdown.
- **Provenance strip** (optional footer on exported panels): JetBrains Mono 10,
  `--color-muted`,
  e.g. `dataset_id | pipeline v | UTC timestamp | colormap`. Keeps a figure auditable
  once it leaves the dashboard.
- **Scale-bar persistence:** scale bars and colorbars must be burned into image exports,
  not just shown in the live UI.
