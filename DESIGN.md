# Frontend Design Style Guide

> Scientific Analysis Dashboard Design System v1.1 -- Light theme -- Data-intensive
> research & bioanalysis applications
>
> **Audience:** Human designers, frontend developers, and agentic coding assistants.
> Follow this document as the single source of truth for all dashboard UI and data
> visualization work in the PhenoTypic project.

---

## Absolute Constraints

These rules are **never** overridden by component context, user request, or convenience.
Agents must treat violations as hard errors.

- **NEVER** use data colors (Okabe-Ito) for buttons, navigation, headings, text links,
  input borders, or any UI chrome.
- **NEVER** use `#F0E442` (yellow) as text color, stroke, or thin line on white or light
  backgrounds.
- **NEVER** render numeric data, axis labels, badge text, captions, or code outside
  `font-family: 'DM Mono'`.
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

---

## 00 -- Logo and Branding

Use the following logo to the left of the dashboard view title: @src/phenotypic/_assets/logos/500x500_png/LogoArtOnly.png

---

## 01 -- Color Palette

### Primary Colors -- UI Only

Brand identity, UI structure, and interactive elements. These colors are **not** used
for data series.

| Token           | Hex       | Usage                                        |
|-----------------|-----------|----------------------------------------------|
| `--color-navy`  | `#003660` | Headings, nav, primary CTA, table headers    |
| `--color-blue`  | `#1b75bc` | Links, interactive states, accent borders    |
| `--color-gold`  | `#febc11` | Highlights, brand accent, emphasized CTA     |
| `--color-white` | `#ffffff` | Card surfaces, panel backgrounds             |
| `--color-bg`    | `#f5f7fa` | App background, inset/recessed areas         |

### Data Colors -- Okabe-Ito -- Visualization Only

Colorblind-safe palette designed by Masataka Okabe and Kei Ito. Safe across
deuteranopia, protanopia, and tritanopia. Used **exclusively** for data visualization:
plot series, progress bars, status badges, and semantic states.

| Token            | Name           | Hex       | CB Safe     | Primary Use                 |
|------------------|----------------|-----------|-------------|-----------------------------|
| `--oi-orange`    | Orange         | `#E69F00` | All types   | Series 2, warning           |
| `--oi-sky`       | Sky Blue       | `#56B4E9` | All types   | Series 3, info              |
| `--oi-green`     | Bluish Green   | `#009E73` | All types   | Series 4, success           |
| `--oi-vermilion` | Vermilion      | `#D55E00` | All types   | Error/alert series only     |
| `--oi-blue`      | Blue           | `#0072B2` | All types   | Series 5                    |
| `--oi-purple`    | Reddish Purple | `#CC79A7` | All types   | Series 6                    |
| `--oi-yellow`    | Yellow         | `#F0E442` | All types   | Series 7, large fills only  |
| `--oi-grey`      | Grey           | `#BBBBBB` | All types   | Reference/control/null      |

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

| State   | Token                | Hex       | Use Case                                     |
|---------|----------------------|-----------|----------------------------------------------|
| Success | `--color-oi-green`   | `#009E73` | Completed jobs, BUSCO pass, positive delta   |
| Info    | `--color-oi-sky`     | `#56B4E9` | Pipeline notices, metadata callouts          |
| Warning | `--color-oi-orange`  | `#E69F00` | Threshold failures, outlier flags            |
| Error   | `--color-oi-vermilion` | `#D55E00` | OOM errors, failed segments, abort states  |

### Color Selection Decision Logic

Use these rules to determine which palette an element draws from:

- **Buttons, nav, headings, links, tabs, inputs, modals:**
  Primary palette only -- `#003660`, `#1b75bc`, `#febc11`, `#ffffff`, `#f5f7fa`
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

The active GUI font is **Roboto** (Google Fonts), used for all three font
roles: display, body, and mono. The same family covers every role today
because the codebase favours one consistent surface; the role-specific
CSS custom properties (`--font-display`, `--font-body`, `--font-mono`)
exist anyway so a future swap to a multi-family stack (e.g. the
DM Serif Display / DM Sans / DM Mono trio) is mechanical — change
[`gui/_design.py`](src/phenotypic/gui/_design.py) and every call site
picks up the new family without edits.

```css
--font-display: 'Roboto', Georgia, "Times New Roman", Times, serif;
--font-body:    'Roboto', -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
--font-mono:    'Roboto', ui-monospace, "SFMono-Regular", Menlo, "Liberation Mono", monospace;
```

Google Fonts import (already wired by `gui/_design.py`):

```html
<link
  href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"
  rel="stylesheet"
/>
```

### Font Assignment Rules

| Content Type                                                       | Role                | Weight |
|--------------------------------------------------------------------|---------------------|--------|
| Headings (h1--h3), large display numbers, species names (italic)   | `var(--font-display)` | 400  |
| Body copy, lead paragraphs, button labels, tab labels, UI prose    | `var(--font-body)`    | varies |
| Numeric data, axis labels, badge text, form labels, captions, code | `var(--font-mono)`    | varies |

### Type Scale

The scale is rem-based and rooted on 15px body text. New code reaches
for the **semantic** aliases (right column); the raw `--text-*`
primitives are kept for back-compat but should not appear in new
call sites.

| Role            | Primitive    | Size              | Semantic alias          |
|-----------------|--------------|-------------------|-------------------------|
| Display         | `--text-3xl` | 2.5rem / 40px     | `--font-size-display`   |
| Title           | `--text-2xl` | 1.875rem / 30px   | `--font-size-title`     |
| Header 1 (h4)   | `--text-xl`  | 1.5rem / 24px     | `--font-size-header-1`  |
| Header 2 (h5)   | `--text-lg`  | 1.25rem / 20px    | `--font-size-header-2`  |
| Body lead       | `--text-md`  | 1.0625rem / 17px  | `--font-size-body-lg`   |
| Body            | `--text-base`| 0.9375rem / 15px  | `--font-size-body`      |
| Label / Overline| `--text-sm`  | 0.8125rem / 13px  | `--font-size-label`     |
| Caption         | `--text-xs`  | 0.6875rem / 11px  | `--font-size-caption`   |

Python inline styles import the matching `FONT_SIZE_*` and
`FONT_FAMILY_*` constants from
[`gui/_design.py`](src/phenotypic/gui/_design.py) — never hardcode a
rem/px literal or a font-family string at a call site.

### Typography Rules

- **Headings** use the display family at weight 400. Italic cut for
  Latin species names (e.g. *Rhodotorula toruloides*).
- **Labels and overlines:** mono family, `text-transform: uppercase`,
  `letter-spacing: 0.12em`, `color: var(--color-muted)`.
- **Stat card values:** display family, `--font-size-display`, `color:
  var(--color-heading)`.
- **Data values** always render in the mono family to preserve optical
  column alignment.
- **Table numeric cells:** mono family, `font-weight: 500`, `color:
  var(--color-heading)`.
- **Table secondary cells:** mono family, `color: var(--color-muted)`.
- **Inline code:** `background: #edf2f7`, `color: #003660`, `padding:
  1px 5px`, `border-radius: 3px`.
- **Maximum line length:** 65ch for body prose, 52ch for lead text.

---

## 03 -- Spacing

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

---

## 04 -- Border Radius & Shadows

### Border Radius

| Token         | Value | Usage                                          |
|---------------|-------|------------------------------------------------|
| `--radius-sm` | 3px   | Inline code, badges, small chips               |
| `--radius`    | 6px   | Buttons, inputs, small cards                   |
| `--radius-md` | 10px  | Stat cards, chart cards, main cards            |
| `--radius-lg` | 16px  | Hero panels, modal surfaces, feature banners   |

### Shadows

All shadows use `rgba(0, 54, 96, ...)` (navy-tinted). **Do not** use gray or black
shadows.

| Token         | Value                                                            | Usage                      |
|---------------|------------------------------------------------------------------|----------------------------|
| `--shadow-sm` | `0 1px 3px rgba(0,54,96,0.07), 0 1px 2px rgba(0,54,96,0.04)`   | Default card resting state |
| `--shadow`    | `0 4px 12px rgba(0,54,96,0.08), 0 1px 3px rgba(0,54,96,0.05)`  | Elevated cards             |
| `--shadow-md` | `0 8px 24px rgba(0,54,96,0.10), 0 2px 6px rgba(0,54,96,0.06)`  | Modals, dropdowns          |
| `--shadow-lg` | `0 16px 40px rgba(0,54,96,0.12), 0 4px 12px rgba(0,54,96,0.07)`| Hero, full-page overlays   |

---

## 05 -- Components

### Stat Cards

Each stat card has a 3px top accent bar whose color communicates category at a glance.
Use primary colors for top-level KPIs; Okabe-Ito for categorically meaningful metrics.

**Accent bar color assignments:**

| Color                       | Hex       | Metric Type                                |
|-----------------------------|-----------|--------------------------------------------|
| Navy `--color-navy`         | `#003660` | Primary KPIs (total count, overall status) |
| Blue `--color-blue`         | `#1b75bc` | Measurement values (diameter, area)        |
| Gold `--color-gold`         | `#febc11` | Coverage / utilization percentages         |
| Orange `--oi-orange`        | `#E69F00` | Warning states, flagged conditions         |
| Bluish Green `--oi-green`   | `#009E73` | Quality / completeness / BUSCO scores      |
| Sky Blue `--oi-sky`         | `#56B4E9` | Throughput / job counts                    |
| Vermilion `--oi-vermilion`  | `#D55E00` | Error / failure counts                     |

**Delta text colors:** positive `#009E73`, negative `#D55E00`, neutral `#8892a4`.

**Anatomy:**

```
+-- 3px accent bar (color by category) -------------------------+
|  LABEL -- DM Mono -- 11px -- uppercase -- --color-muted       |
|                                                               |
|  Value -- DM Serif Display -- 2.5rem                          |
|  Delta -- DM Mono -- 11px -- #009E73 (up) / #D55E00 (down)   |
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
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--accent-color); /* set per instance */
}
```

---

### Progress Bars

Use Okabe-Ito series order for multi-fill progress. Track background is always
`--color-rule`.

**Variants:**

| Variant   | Height | Use Case                                         |
|-----------|--------|--------------------------------------------------|
| Standard  | 6px    | Multi-step pipelines, comparison series          |
| Thick     | 10px   | Primary KPI progress                             |
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

**Label layout:** progress name (DM Sans 500) left, value (DM Mono) right,
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
    rgba(255,255,255,0) 0%,
    rgba(255,255,255,0.25) 100%
  );
}
```

---

### Buttons

Primary buttons use the brand palette. Okabe-Ito colors are **not** used for buttons --
reserved exclusively for data. Exception: danger variant uses `--oi-vermilion`.

| Variant | Background        | Text            | Border           | Use Case               |
|---------|-------------------|-----------------|------------------|------------------------|
| Primary | `--color-navy`    | White           | `--color-navy`   | Submit, Run, Confirm   |
| Blue    | `--color-blue`    | White           | `--color-blue`   | Primary interactive    |
| Gold    | `--color-gold`    | `--color-navy`  | `--color-gold`   | Emphasized CTA         |
| Outline | Transparent       | `--color-navy`  | `--color-border` | Secondary actions      |
| Ghost   | Transparent       | `--color-muted` | none             | Tertiary / utility     |
| Danger  | Transparent       | `#D55E00`       | `#D55E00`        | Destructive actions    |

**Sizes:**

| Size    | Font Size | Padding              |
|---------|-----------|----------------------|
| sm      | 11px      | `0.3rem 0.75rem`     |
| default | 13px      | `0.5rem 1.125rem`    |
| lg      | 15px      | `0.7rem 1.5rem`      |
| icon    | --        | `width/height 2.25rem, padding 0` |

**Shared styles:** DM Sans, `font-weight: 500`, `letter-spacing: 0.01em`,
`border-radius: var(--radius)`, `border-width: 1.5px`,
`transition: all 180ms cubic-bezier(0.22, 1, 0.36, 1)`.

**Hover states:**

- Filled variants: darken background ~8%, add `box-shadow: 0 4px 12px rgba(0,54,96,0.28)`
- Outline: `border-color: #1b75bc`, `color: #1b75bc`,
  `background: rgba(27,117,188,0.04)`
- Danger: `background: #D55E00`, `color: #fff`

---

### Badges

Monospaced, uppercase micro-labels for status, classification, and categorical tagging.

**Structure:**
`[colored dot 5px] [text -- DM Mono -- 10.5px -- uppercase -- letter-spacing: 0.08em]`

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
**Title:** DM Sans, `font-weight: 600`, `font-size: 13px`.
**Body:** `opacity: 0.85`, `line-height: 1.5`.

| Type    | Border    | Background                | Text      |
|---------|-----------|---------------------------|-----------|
| Info    | `#56B4E9` | `rgba(86,180,233,0.08)`   | `#0B5E87` |
| Success | `#009E73` | `rgba(0,158,115,0.07)`    | `#005C43` |
| Warning | `#E69F00` | `rgba(230,159,0,0.10)`    | `#7A5500` |
| Error   | `#D55E00` | `rgba(213,94,0,0.08)`     | `#8A3C00` |

---

### Data Tables

| Element          | Style                                                                   |
|------------------|-------------------------------------------------------------------------|
| Header border    | `2px solid --color-navy` (bottom)                                       |
| Header text      | DM Mono, 11px, uppercase, `letter-spacing: 0.08em`, `--color-muted`    |
| Cell padding     | `12px 16px`                                                             |
| Row divider      | `1px solid --color-rule`                                                |
| Row hover        | `background: rgba(27,117,188,0.03)`                                     |
| Numeric values   | DM Mono, `--color-heading`, `font-weight: 500`                          |
| Secondary values | DM Mono, `--color-muted`                                                |

**Column ordering convention:** identifier, taxonomy/category, measurements (DM Mono),
quality metrics, status badge.

---

### Form Inputs

**Base styles:** `background: #ffffff`, `border: 1.5px solid --color-border`,
`border-radius: var(--radius)`, `padding: 0.5rem 0.875rem`, DM Sans 13px,
`color: --color-body`, `transition: border-color 180ms, box-shadow 180ms`.

| State   | Border                       | Focus Ring                          |
|---------|------------------------------|-------------------------------------|
| Default | `1.5px solid --color-border` | --                                  |
| Focus   | `1.5px solid --color-blue`   | `0 0 0 3px rgba(27,117,188,0.12)`  |
| Valid   | `1.5px solid #009E73`        | `0 0 0 3px rgba(0,158,115,0.12)`   |
| Error   | `1.5px solid #D55E00`        | `0 0 0 3px rgba(213,94,0,0.12)`    |

**Labels:** DM Mono, 11px, uppercase, `letter-spacing: 0.08em`, `--color-muted`,
placed above input.
**Hint text:** 11px, `--color-muted`.
**Valid text:** 11px, `#006B4F`.
**Error text:** 11px, `#D55E00`.

---

### Navigation Tabs

```css
/* Track */
border-bottom: 2px solid var(--color-rule); /* full width */

/* Tab */
font-family: var(--font-body);
font-size: 13px;
font-weight: 500;
padding: 12px 20px;
border-bottom: 2px solid transparent;
margin-bottom: -2px;

/* States */
/* active  */ color: var(--color-navy); border-color: var(--color-navy);
/* inactive */ color: var(--color-muted);
/* hover (inactive) */ color: var(--color-body); border-color: var(--color-border);
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

| Element           | Style                                                    |
|-------------------|----------------------------------------------------------|
| Gridlines         | `1px`, `#e8ecf2`                                         |
| Axes              | `1.5px`, `#dde3ed`                                       |
| Axis labels       | DM Mono, 7--8px, `--color-muted`                        |
| Chart title       | DM Sans, 13px, weight 600, `--color-heading`            |
| Chart subtitle    | DM Mono, 11px, `--color-muted`                          |
| Spines            | Top and right hidden; bottom and left only               |
| Data point dots   | 3.5px radius, filled with series color                   |
| Area fills        | Series color at 7% opacity beneath line                  |
| Line stroke width | 2px, `stroke-linejoin: round`, `stroke-linecap: round`  |
| Error bars        | 1px, series color, 60% opacity                           |

### Donut / Pie Charts

```
ring stroke-width:  20px on r=40 SVG circle (circumference ~251px)
center value:       DM Serif Display, font-size 11, font-weight 600, fill #003660
center unit:        DM Mono, font-size 6.5, fill #8892a4
legend dot:         10px circle
legend name:        DM Sans, 11px, color #2e3a4e
legend pct:         DM Mono, 11px, color #8892a4
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
    "axes.prop_cycle":    mpl.cycler(color=OKABE_ITO),
    "axes.facecolor":     "#ffffff",
    "figure.facecolor":   "#f5f7fa",
    "axes.edgecolor":     "#dde3ed",
    "axes.grid":          True,
    "grid.color":         "#e8ecf2",
    "grid.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DM Sans", "Helvetica Neue", "Arial"],
    "axes.labelcolor":    "#2e3a4e",
    "xtick.color":        "#8892a4",
    "ytick.color":        "#8892a4",
    "axes.titlecolor":    "#003660",
    "axes.titleweight":   "600",
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
})
```

### napari Label Layer Colors

For categorical label overlays (e.g. colony segmentation masks):

```python
# RGBA tuples normalized 0-1, for napari label layer color dict
OKABE_ITO_NAPARI = {
    1: (  0/255,  54/255,  96/255, 1.0),  # navy
    2: (230/255, 159/255,   0/255, 1.0),  # orange
    3: ( 86/255, 180/255, 233/255, 1.0),  # sky blue
    4: (  0/255, 158/255, 115/255, 1.0),  # bluish green
    5: (  0/255, 114/255, 178/255, 1.0),  # blue
    6: (204/255, 121/255, 167/255, 1.0),  # reddish purple
    7: (213/255,  94/255,   0/255, 1.0),  # vermilion (error)
}
```

### CSS Custom Properties

Include this `:root` block in all generated CSS files:

```css
:root {
  /* Primary palette -- UI only */
  --color-navy:    #003660;
  --color-blue:    #1b75bc;
  --color-gold:    #febc11;
  --color-white:   #ffffff;
  --color-bg:      #f5f7fa;
  --color-surface: #ffffff;
  --color-border:  #dde3ed;
  --color-rule:    #e8ecf2;
  --color-muted:   #8892a4;
  --color-body:    #2e3a4e;
  --color-heading: #003660;

  /* Data palette -- Okabe-Ito -- visualization only */
  --oi-orange:    #E69F00;
  --oi-sky:       #56B4E9;
  --oi-green:     #009E73;
  --oi-vermilion: #D55E00;
  --oi-blue:      #0072B2;
  --oi-purple:    #CC79A7;
  --oi-yellow:    #F0E442;  /* large fills only */
  --oi-grey:      #BBBBBB;  /* reference / control */

  /* Semantic aliases */
  --color-success: var(--oi-green);
  --color-info:    var(--oi-sky);
  --color-warning: var(--oi-orange);
  --color-danger:  var(--oi-vermilion);

  /* Typography */
  --font-display: 'DM Serif Display', Georgia, serif;
  --font-body:    'DM Sans', system-ui, sans-serif;
  --font-mono:    'DM Mono', 'Courier New', monospace;

  /* Type scale */
  --text-xs:   0.6875rem;   /*  11px */
  --text-sm:   0.8125rem;   /*  13px */
  --text-base: 0.9375rem;   /*  15px */
  --text-md:   1.0625rem;   /*  17px */
  --text-lg:   1.25rem;     /*  20px */
  --text-xl:   1.5rem;      /*  24px */
  --text-2xl:  1.875rem;    /*  30px */
  --text-3xl:  2.5rem;      /*  40px */
  --text-4xl:  3.25rem;     /*  52px */

  /* Spacing (8pt grid) */
  --sp-1:  0.25rem;
  --sp-2:  0.5rem;
  --sp-3:  0.75rem;
  --sp-4:  1rem;
  --sp-5:  1.25rem;
  --sp-6:  1.5rem;
  --sp-8:  2rem;
  --sp-10: 2.5rem;
  --sp-12: 3rem;
  --sp-16: 4rem;

  /* Border radius */
  --radius-sm: 3px;
  --radius:    6px;
  --radius-md: 10px;
  --radius-lg: 16px;

  /* Shadows (navy-tinted) */
  --shadow-sm: 0 1px 3px rgba(0,54,96,0.07), 0 1px 2px rgba(0,54,96,0.04);
  --shadow:    0 4px 12px rgba(0,54,96,0.08), 0 1px 3px rgba(0,54,96,0.05);
  --shadow-md: 0 8px 24px rgba(0,54,96,0.10), 0 2px 6px rgba(0,54,96,0.06);
  --shadow-lg: 0 16px 40px rgba(0,54,96,0.12), 0 4px 12px rgba(0,54,96,0.07);

  /* Motion */
  --ease-out:   cubic-bezier(0.22, 1, 0.36, 1);
  --transition: 180ms var(--ease-out);
}
```

---

## 08 -- Usage Rules & Anti-Patterns

### Do

- Keep Okabe-Ito colors strictly in the data layer. UI elements use primary brand colors only.
- Follow the series order. Orange second, sky blue third -- changing the order breaks CB
  distinctiveness guarantees at adjacent series.
- Use grey `#BBBBBB` for reference lines, baseline means, and negative control series.
- Use vermilion `#D55E00` only for error / alert / failed series.
- Darken Okabe-Ito colors when using as text on white (see badge contrast values).
- Reserve yellow `#F0E442` for large filled elements only (bars, area fills, large markers).

### Don't

- Don't use Okabe-Ito colors for buttons, navigation, or text. Data-only palette.
- Don't reorder series arbitrarily. The order was optimized for perceptual separation
  under CB simulation.
- Don't render `#F0E442` as thin lines or text on white -- contrast ratio is ~1.9:1.
- Don't use `--oi-blue` (`#0072B2`) alongside `--color-blue` (`#1b75bc`) in the same chart.
- Don't exceed 6 categorical series in a single chart without introducing an "other" category.
- Don't apply `--shadow-lg` to inline cards; reserve for modals and hero panels.
- Don't use red-green colormaps.
- Don't use em dashes in any generated text.
