# Scientific Analysis Dashboard — Style Guide

> Design System v1.1 · Light theme · Data-intensive research & bioanalysis applications

---

## 01 — Color Palette

### Primary Colors

Brand identity, UI structure, and interactive elements. These colors are **not** used
for data series.

| Token           | Hex       | Usage                                        |
|-----------------|-----------|----------------------------------------------|
| `--color-navy`  | `#003660` | Headings · Nav · Primary CTA · Table headers |
| `--color-blue`  | `#1b75bc` | Links · Interactive states · Accent borders  |
| `--color-gold`  | `#febc11` | Highlights · Brand accent                    |
| `--color-white` | `#ffffff` | Card surfaces · Panel backgrounds            |
| `--color-bg`    | `#f5f7fa` | App background · Inset/recessed areas        |

### Secondary Colors — Okabe-Ito

Colorblind-safe palette designed by Masataka Okabe and Kei Ito. Safe across
deuteranopia, protanopia, and tritanopia. Used exclusively for data visualization: plot
series, progress bars, status badges, and semantic states.

| Token                  | Name           | Hex       | CB Safety   | Primary use                 |
|------------------------|----------------|-----------|-------------|-----------------------------|
| `--color-oi-orange`    | Orange         | `#E69F00` | ✓ All types | Series 1 · Warning          |
| `--color-oi-sky`       | Sky Blue       | `#56B4E9` | ✓ All types | Series 2 · Info             |
| `--color-oi-green`     | Bluish Green   | `#009E73` | ✓ All types | Series 3 · Success          |
| `--color-oi-vermilion` | Vermilion      | `#D55E00` | ✓ All types | Series 4 · Error/Alert      |
| `--color-oi-blue`      | Blue           | `#0072B2` | ✓ All types | Series 5                    |
| `--color-oi-purple`    | Reddish Purple | `#CC79A7` | ✓ All types | Series 6                    |
| `--color-oi-yellow`    | Yellow         | `#F0E442` | ✓ All types | Series 7 · large fills only |

> **Source:** Okabe M, Ito K (2008). "Color Universal Design."
*jfly.iam.u-tokyo.ac.jp/color*

> **Note on yellow (`#F0E442`):** Passes colorblind simulation but contrast ratio on
> white is ~1.9:1. Reserve for filled chart elements (bars, area fills, large markers)
> where size compensates. Never use as text color or thin lines on white.

> **Note on `--color-oi-blue` (`#0072B2`):** Visually close to `--color-blue #1b75bc` at
> small sizes. Use for series 5 only after 4 prior series are present, or substitute grey
> for reference/control lines.

### Surface & Neutral Tokens

| Token             | Hex       | Usage                                        |
|-------------------|-----------|----------------------------------------------|
| `--color-border`  | `#dde3ed` | Card borders · Input outlines                |
| `--color-rule`    | `#e8ecf2` | Dividers · Chart gridlines · Table row rules |
| `--color-muted`   | `#8892a4` | Secondary text · Labels · Captions · Axes    |
| `--color-body`    | `#2e3a4e` | Primary body text                            |
| `--color-heading` | `#003660` | All heading text (alias of navy)             |

### Semantic Color Assignments

| State   | Color                            | Hex       | Use case                                     |
|---------|----------------------------------|-----------|----------------------------------------------|
| Success | Bluish Green `--color-oi-green`  | `#009E73` | Completed jobs · BUSCO pass · Positive delta |
| Info    | Sky Blue `--color-oi-sky`        | `#56B4E9` | Pipeline notices · Metadata callouts         |
| Warning | Orange `--color-oi-orange`       | `#E69F00` | Threshold failures · Outlier flags           |
| Error   | Vermilion `--color-oi-vermilion` | `#D55E00` | OOM errors · Failed segments · Abort states  |

---

## 02 — Typography

Three-font stack: **DM Serif Display** for editorial headings and large numerics, **DM
Sans** for body and UI copy, **DM Mono** for all labels, data values, and code.

```css
--font-display: 'DM Serif Display', Georgia, serif;
--font-body:    'DM Sans', system-ui, sans-serif;
--font-mono:    'DM Mono', 'Courier New', monospace;
```

Google Fonts import:

```html
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1
  &family=DM+Mono:wght@300;400;500
  &family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300
  &display=swap" rel="stylesheet">
```

### Type Scale

| Role             | Token         | Size             | Font             | Weight | Line Height |
|------------------|---------------|------------------|------------------|--------|-------------|
| H1               | `--text-4xl`  | 3.25rem / 52px   | DM Serif Display | 400    | 1.1         |
| H2               | `--text-3xl`  | 2.5rem / 40px    | DM Serif Display | 400    | 1.2         |
| H3               | `--text-2xl`  | 1.875rem / 30px  | DM Serif Display | 400    | 1.25        |
| H4               | `--text-lg`   | 1.25rem / 20px   | DM Sans          | 600    | 1.4         |
| H5               | `--text-md`   | 1.0625rem / 17px | DM Sans          | 600    | 1.4         |
| Lead / Intro     | `--text-lg`   | 1.25rem / 20px   | DM Sans          | 300    | 1.7         |
| Body             | `--text-base` | 0.9375rem / 15px | DM Sans          | 400    | 1.65        |
| Caption          | `--text-xs`   | 0.6875rem / 11px | DM Mono          | 400    | 1.5         |
| Label / Overline | `--text-xs`   | 0.6875rem / 11px | DM Mono          | 500    | —           |
| Inline code      | `--text-sm`   | 0.8125rem / 13px | DM Mono          | 400    | —           |

### Typography Rules

- **Headings** use DM Serif Display at weight 400. Italic cut is appropriate for Latin
  species names (e.g. *Rhodotorula toruloides*).
- **Labels and overlines** use DM Mono, uppercase, `letter-spacing: 0.12em`. Apply
  `--color-muted` for secondary labels.
- **Data values** always render in DM Mono to preserve optical column alignment.
- **Inline code** renders against `background: #edf2f7`, `color: --color-navy`,
  `padding: 1px 5px`, `border-radius: 3px`.
- **Maximum line length:** 65ch for body prose, 52ch for lead text.

---

## 03 — Spacing

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

**Section rhythm:** Major sections separated by `--sp-16` top margin and a 1px
`--color-rule` divider. Card internal padding is `--sp-6`.

---

## 04 — Border Radius & Shadows

### Border Radius

| Token         | Value | Usage                                          |
|---------------|-------|------------------------------------------------|
| `--radius-sm` | 3px   | Inline code · Badges · Small chips             |
| `--radius`    | 6px   | Buttons · Inputs · Small cards                 |
| `--radius-md` | 10px  | Stat cards · Chart cards · Main cards          |
| `--radius-lg` | 16px  | Hero panels · Modal surfaces · Feature banners |

### Shadows

All shadows use `rgba(0, 54, 96, …)` (navy-tinted) to stay harmonious with the palette.

| Token         | Value                                                           | Usage                      |
|---------------|-----------------------------------------------------------------|----------------------------|
| `--shadow-sm` | `0 1px 3px rgba(0,54,96,0.07), 0 1px 2px rgba(0,54,96,0.04)`    | Default card resting state |
| `--shadow`    | `0 4px 12px rgba(0,54,96,0.08), 0 1px 3px rgba(0,54,96,0.05)`   | Elevated cards             |
| `--shadow-md` | `0 8px 24px rgba(0,54,96,0.10), 0 2px 6px rgba(0,54,96,0.06)`   | Modals · Dropdowns         |
| `--shadow-lg` | `0 16px 40px rgba(0,54,96,0.12), 0 4px 12px rgba(0,54,96,0.07)` | Hero · Full-page overlays  |

---

## 05 — Components

### Stat Cards

Each stat card has a 3px top accent bar whose color communicates category at a glance.
Use primary colors for top-level KPIs; Okabe-Ito for categorically meaningful metrics.

**Accent bar color assignments:**

| Color                            | Hex       | Suggested metric type                      |
|----------------------------------|-----------|--------------------------------------------|
| Navy `--color-navy`              | `#003660` | Primary KPIs (total count, overall status) |
| Blue `--color-blue`              | `#1b75bc` | Measurement values (diameter, area)        |
| Gold `--color-gold`              | `#febc11` | Coverage / utilization percentages         |
| Orange `--color-oi-orange`       | `#E69F00` | Warning states · Flagged conditions        |
| Bluish Green `--color-oi-green`  | `#009E73` | Quality / completeness / BUSCO scores      |
| Sky Blue `--color-oi-sky`        | `#56B4E9` | Throughput / job counts                    |
| Vermilion `--color-oi-vermilion` | `#D55E00` | Error / failure counts                     |

**Anatomy:**

```
┌─ 3px accent bar (color by category) ──────────────────────┐
│  LABEL · DM Mono · 11px · uppercase · --color-muted       │
│                                                            │
│  Value · DM Serif Display · 2.5rem                        │
│  Delta  · DM Mono · 11px · #009E73 (up) / #D55E00 (down)  │
└────────────────────────────────────────────────────────────┘
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

| Variant   | Height | Use case                                         |
|-----------|--------|--------------------------------------------------|
| Standard  | 6px    | Multi-step pipelines · Comparison series         |
| Thick     | 10px   | Primary KPI progress                             |
| Segmented | 6–10px | Proportional composition (e.g. strain breakdown) |

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

Primary buttons remain anchored to the brand palette. Okabe-Ito colors are **not** used
for buttons — reserved exclusively for data.

| Variant | Background     | Text            | Border           | Use case               |
|---------|----------------|-----------------|------------------|------------------------|
| Primary | `--color-navy` | White           | `--color-navy`   | Submit · Run · Confirm |
| Blue    | `--color-blue` | White           | `--color-blue`   | Primary interactive    |
| Gold    | `--color-gold` | `--color-navy`  | `--color-gold`   | Emphasized CTA         |
| Outline | Transparent    | `--color-navy`  | `--color-border` | Secondary actions      |
| Ghost   | Transparent    | `--color-muted` | none             | Tertiary / utility     |
| Danger  | Transparent    | `#D55E00`       | `#D55E00`        | Destructive actions    |

---

### Badges

Monospaced, uppercase micro-labels for status, classification, and categorical tagging.

**Structure:**
`[colored dot 5px] [text · DM Mono · 10.5px · uppercase · letter-spacing: 0.08em]`

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
> on white surfaces.

---

### Alerts & Callouts

Left border in semantic Okabe-Ito color, background tint at 7–10% opacity against white
card surfaces.

| Type    | Border Hex | Background              | Text      |
|---------|------------|-------------------------|-----------|
| Info    | `#56B4E9`  | `rgba(86,180,233,0.08)` | `#0B5E87` |
| Success | `#009E73`  | `rgba(0,158,115,0.07)`  | `#005C43` |
| Warning | `#E69F00`  | `rgba(230,159,0,0.10)`  | `#7A5500` |
| Error   | `#D55E00`  | `rgba(213,94,0,0.08)`   | `#8A3C00` |

---

### Data Tables

| Element          | Style                                                                   |
|------------------|-------------------------------------------------------------------------|
| Header border    | 2px solid `--color-navy` (bottom)                                       |
| Header text      | DM Mono · 11px · uppercase · `letter-spacing: 0.08em` · `--color-muted` |
| Cell padding     | `--sp-3` vertical · `--sp-4` horizontal                                 |
| Row divider      | 1px solid `--color-rule`                                                |
| Row hover        | `background: rgba(27,117,188,0.03)`                                     |
| Numeric values   | DM Mono · `--color-heading` · `font-weight: 500`                        |
| Secondary values | DM Mono · `--color-muted`                                               |

---

### Form Inputs

| State   | Border                       | Focus ring                        |
|---------|------------------------------|-----------------------------------|
| Default | `1.5px solid --color-border` | —                                 |
| Focus   | `1.5px solid --color-blue`   | `0 0 0 3px rgba(27,117,188,0.12)` |
| Valid   | `1.5px solid #009E73`        | `0 0 0 3px rgba(0,158,115,0.12)`  |
| Error   | `1.5px solid #D55E00`        | `0 0 0 3px rgba(213,94,0,0.12)`   |

---

## 06 — Data Visualization

### Categorical Series Order

Apply Okabe-Ito colors in this order for all multi-series charts. Navy anchors series 1,
keeping the primary series harmonious with the UI chrome.

| Series          | Name           | Hex       | Notes                                             |
|-----------------|----------------|-----------|---------------------------------------------------|
| 1               | Navy           | `#003660` | Primary series · anchors to UI brand              |
| 2               | Orange         | `#E69F00` | Warm anchor, distinct from all others             |
| 3               | Sky Blue       | `#56B4E9` | High luminance, good for thin lines               |
| 4               | Bluish Green   | `#009E73` | Perceptually equidistant from orange & sky        |
| 5               | Blue           | `#0072B2` | Use carefully near `--color-blue` UI elements     |
| 6               | Reddish Purple | `#CC79A7` | High distinctiveness                              |
| 7 (alert/error) | Vermilion      | `#D55E00` | Error / failed / alert series only                |
| ref             | Grey           | `#BBBBBB` | Reference lines · negative controls · null series |

> Grey `#BBBBBB` is not formally part of Okabe-Ito but is universally CB-safe and
> strongly recommended for reference/control series.

### Chart Styling Rules

| Element           | Style                                                    |
|-------------------|----------------------------------------------------------|
| Gridlines         | 1px · `#e8ecf2`                                          |
| Axes              | 1.5px · `#dde3ed`                                        |
| Axis labels       | DM Mono · 7–8px · `--color-muted`                        |
| Chart title       | DM Sans · 13px · weight 600 · `--color-heading`          |
| Chart subtitle    | DM Mono · 11px · `--color-muted`                         |
| Data point dots   | 3.5px radius · filled with series color                  |
| Area fills        | Series color at 7% opacity beneath line                  |
| Line stroke width | 2px · `stroke-linejoin: round` · `stroke-linecap: round` |
| Error bars        | 1px · series color · 60% opacity                         |

### Heatmap Colorscale (Single-Variable)

For single-variable intensity maps (e.g. 96-well plate colony density):

| Intensity     | Hex                  | Notes                 |
|---------------|----------------------|-----------------------|
| Low           | `rgba(0,54,96,0.08)` | Near-transparent navy |
| Mid           | `#56B4E9`            | Okabe-Ito sky blue    |
| High          | `#003660`            | Full navy             |
| Failed / null | `#D55E00` at 70%     | Okabe-Ito vermilion   |

Avoids red-green colormaps which fail under deuteranopia. The navy-to-blue ramp is
maximally distinct from the vermilion fail state under all CB types.

### matplotlib / seaborn Integration

```python
import matplotlib as mpl

OKABE_ITO = [
    "#003660",  # navy     — series 1, UI-harmonized
    "#E69F00",  # orange   — series 2
    "#56B4E9",  # sky blue — series 3
    "#009E73",  # green    — series 4
    "#0072B2",  # blue     — series 5
    "#CC79A7",  # purple   — series 6
    "#D55E00",  # vermilion — error/alert series
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

### napari Layer Color Assignments

For categorical label overlays (e.g. colony segmentation masks):

```python
# RGBA tuples normalized 0–1, for napari label layer color dict
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

---

## 07 — Complete CSS Token Reference

```css
:root {
  /* Primary palette — UI only */
  --color-navy:    #003660;
  --color-blue:    #1b75bc;
  --color-gold:    #febc11;

  /* Secondary palette — Okabe-Ito · data visualization only */
  --color-oi-orange:    #E69F00;
  --color-oi-sky:       #56B4E9;
  --color-oi-green:     #009E73;
  --color-oi-vermilion: #D55E00;
  --color-oi-blue:      #0072B2;
  --color-oi-purple:    #CC79A7;
  --color-oi-yellow:    #F0E442;  /* large fills only */
  --color-oi-grey:      #BBBBBB;  /* reference / control */

  /* Semantic aliases */
  --color-success: var(--color-oi-green);
  --color-info:    var(--color-oi-sky);
  --color-warning: var(--color-oi-orange);
  --color-danger:  var(--color-oi-vermilion);

  /* Surfaces & neutrals */
  --color-white:   #ffffff;
  --color-bg:      #f5f7fa;
  --color-surface: #ffffff;
  --color-border:  #dde3ed;
  --color-rule:    #e8ecf2;
  --color-muted:   #8892a4;
  --color-body:    #2e3a4e;
  --color-heading: #003660;

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

## 08 — Usage Rules & Anti-Patterns

### Do

- Keep Okabe-Ito colors strictly in the data layer. UI elements (buttons, nav, headings)
  use primary brand colors only.
- Follow the series order. Orange second, sky blue third — changing the order breaks CB
  distinctiveness guarantees at adjacent series.
- Use grey `#BBBBBB` for reference lines, baseline means, and negative control series.
- Use vermilion `#D55E00` only for error / alert / failed series.
- Darken Okabe-Ito colors when using as text on white (see badge contrast values in
  section 05).
- Reserve yellow `#F0E442` for large filled elements only (bars, area fills, large
  markers).

### Don't

- Don't use Okabe-Ito colors for buttons, navigation, or text. They are a data-only
  palette.
- Don't reorder series arbitrarily. The Okabe-Ito series order was optimized for
  perceptual separation under CB simulation.
- Don't render `#F0E442` as thin lines or text on white — contrast ratio is ~1.9:1.
- Don't use `--color-oi-blue #0072B2` alongside `--color-blue #1b75bc` in the same
  chart.
- Don't exceed 6 categorical series in a single chart without introducing an "other"
  category.
- Don't apply `--shadow-lg` to inline cards; reserve it for modals and hero panels.
