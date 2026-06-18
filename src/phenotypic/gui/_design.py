"""Single-source-of-truth design tokens for the GUI.

Mirrors ``DESIGN.md`` v1.2. :func:`inject_design_tokens` splices four
``<style>`` blocks into every Dash app's ``index_string``:

* :data:`FONT_TOKENS_CSS` -- ``@import``s the role fonts (Comfortaa for
  display + body, JetBrains Mono for mono, IBM Plex Serif kept only for
  italic species names) and declares the ``--font-display`` / ``--font-body``
  / ``--font-mono`` / ``--font-species`` custom properties.
* :data:`DESIGN_TOKENS_CSS` -- the brand + Okabe-Ito palettes, type scale,
  semantic ``--font-size-*`` aliases, line-height / tracking, spacing,
  radius, shadow, and ease/transition tokens.
* :data:`BOOTSTRAP_OVERRIDE_CSS` -- remaps ``dbc.themes.BOOTSTRAP`` onto the
  brand palette so ``color="primary"`` etc. render navy, not Bootstrap blue.
* :data:`BASE_STYLES_CSS` -- conservative element defaults plus the named
  ``.text-*`` style classes (DESIGN.md "02.5 / 02.6").

To swap a role font: change ``_DISPLAY_PRIMARY`` / ``_BODY_PRIMARY`` /
``_MONO_PRIMARY`` / ``_SPECIES_PRIMARY`` and update ``_GOOGLE_FONTS_URL``.
Every mounted Dash app picks up the change on next reload -- no CSS file
edits required.

To swap a design color or sizing token: edit the matching ``COLOR_*``,
``TEXT_*``, ``FONT_SIZE_*``, ``RADIUS_*``, ``SHADOW_*``, or ``EASE_*``
constant; the CSS blocks are rebuilt at import time.

Python callers that need an inline-style hex, font family, or font size
(e.g. Plotly figure colors, ``html.Div(style={"color": COLOR_NAVY})``,
``style_cell={"fontFamily": FONT_FAMILY_MONO, "fontSize": FONT_SIZE_BODY_SM}``)
should ``import`` the constant instead of re-spelling literals -- see
``DESIGN.md`` for the full UI-vs-data palette rules and the typography
scale. ``COLOR_*`` are UI-only; ``OI_*`` are data-only.
"""
from __future__ import annotations

__all__ = [
    # ---- Brand / UI palette (UI ONLY -- never charts) ----
    "COLOR_NAVY",
    "COLOR_BLUE",
    "COLOR_GOLD",
    "COLOR_WHITE",
    "COLOR_BG",
    "COLOR_SURFACE",
    "COLOR_BORDER",
    "COLOR_RULE",
    "COLOR_MUTED",
    "COLOR_BODY",
    "COLOR_HEADING",
    "COLOR_IMAGE_STAGE_DARK",
    # ---- Semantic state colors (data / state only) ----
    "COLOR_SUCCESS",
    "COLOR_INFO",
    "COLOR_WARNING",
    "COLOR_DANGER",
    # ---- Okabe-Ito data palette (DATA ONLY -- never UI chrome) ----
    "OI_NAVY",
    "OI_ORANGE",
    "OI_SKY",
    "OI_GREEN",
    "OI_VERMILION",
    "OI_BLUE",
    "OI_PURPLE",
    "OI_YELLOW",
    "OI_GREY",
    "OKABE_ITO",
    "OKABE_ITO_NAPARI",
    # ---- Darkened OI text variants (badge / alert text on white) ----
    "OI_ORANGE_TEXT",
    "OI_SKY_TEXT",
    "OI_GREEN_TEXT",
    "OI_PURPLE_TEXT",
    "OI_VERMILION_TEXT",
    # ---- Visual tokens (non-palette) ----
    "TILE_DIM_RGB",
    # ---- Type scale / radius / shadow / motion ----
    "TEXT_2XS",
    "TEXT_XS",
    "TEXT_SM",
    "TEXT_BASE",
    "TEXT_MD",
    "TEXT_LG",
    "TEXT_XL",
    "TEXT_2XL",
    "TEXT_3XL",
    "TEXT_4XL",
    # ---- Semantic font-size aliases (preferred call form) ----
    "FONT_SIZE_DISPLAY",
    "FONT_SIZE_TITLE",
    "FONT_SIZE_HEADER_1",
    "FONT_SIZE_HEADER_2",
    "FONT_SIZE_BODY_LG",
    "FONT_SIZE_BODY",
    "FONT_SIZE_BODY_SM",
    "FONT_SIZE_CAPTION",
    "FONT_SIZE_MICRO",
    "FONT_SIZE_LABEL",  # DEPRECATED alias of FONT_SIZE_BODY_SM
    # ---- Python-side font-family constants ----
    "FONT_FAMILY_DISPLAY",
    "FONT_FAMILY_BODY",
    "FONT_FAMILY_MONO",
    "FONT_FAMILY_SPECIES",
    "SPACING_1",
    "SPACING_2",
    "SPACING_3",
    "SPACING_4",
    "SPACING_5",
    "SPACING_6",
    "SPACING_8",
    "SPACING_10",
    "SPACING_12",
    "SPACING_16",
    # ---- Line-height / tracking ----
    "LEADING_DISPLAY",
    "LEADING_TIGHT",
    "LEADING_SNUG",
    "LEADING_NORMAL",
    "LEADING_RELAXED",
    "TRACKING_TIGHT",
    "TRACKING_SNUG",
    "TRACKING_NORMAL",
    "TRACKING_BUTTON",
    "TRACKING_WIDE",
    "TRACKING_WIDER",
    "RADIUS_SM",
    "RADIUS",
    "RADIUS_MD",
    "RADIUS_LG",
    "SHADOW_SM",
    "SHADOW",
    "SHADOW_MD",
    "SHADOW_LG",
    "EASE_OUT",
    "TRANSITION",
    # ---- Error category colors ----
    "ERROR_CATEGORY_COLORS",
    "category_color",
    # ---- CSS bundles ----
    "FONT_TOKENS_CSS",
    "DESIGN_TOKENS_CSS",
    "BOOTSTRAP_OVERRIDE_CSS",
    "BASE_STYLES_CSS",
    # ---- Injector ----
    "inject_design_tokens",
]

# ---------------------------------------------------------------------------
# Role fonts (Google Fonts) -- DESIGN.md "02.1 Font Families"
# ---------------------------------------------------------------------------
#
# Four role families (NOT one family across all roles):
#
#   display -- Comfortaa     : content headings, large stat values.
#   body    -- Comfortaa     : prose, UI/component titles, button + tab labels.
#   mono    -- JetBrains Mono : ALL numeric data, axis labels, badge / label /
#              caption text, and code tokens.
#   species -- IBM Plex Serif : ITALIC binomial species names ONLY. Comfortaa
#              ships no true italic face, so italic *Genus species* is set in a
#              real serif italic instead of a browser-synthesized oblique.
#
# Comfortaa carries both the display and body voice (one rounded geometric
# sans across chrome); JetBrains Mono carries the data voice; IBM Plex Serif
# is retained solely for italic species names. To swap a role, change its
# ``_*_PRIMARY`` below and update ``_GOOGLE_FONTS_URL`` to load the new family
# -- every call site inherits via the ``--font-*`` custom properties /
# ``FONT_FAMILY_*`` constants.

_DISPLAY_PRIMARY = "Comfortaa"
_BODY_PRIMARY = "Comfortaa"
_MONO_PRIMARY = "JetBrains Mono"
_SPECIES_PRIMARY = "IBM Plex Serif"

# The @import loads more than the four chrome roles: the chart subsystem
# (``phenotypic.sdk_.viz.figures._theme``) is intentionally NOT migrated to
# Comfortaa -- it keeps IBM Plex Sans for plot titles / legend names and IBM
# Plex Serif for donut center values (DESIGN.md "06 -- Charts"). Those plots
# render inside GUI Dash pages, so the IBM Plex families must stay loaded here
# even though no ``--font-*`` chrome token references IBM Plex Sans. Comfortaa
# carries display + body; JetBrains Mono carries data; IBM Plex Serif (italic)
# also backs ``--font-species`` for binomial names.
_GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Comfortaa:wght@400;500;600;700"
    "&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400;1,500"
    "&family=IBM+Plex+Sans:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500;600"
    "&display=swap"
)

# Cross-platform fallbacks: kick in if the Google Font is blocked or slow to
# load. The stacks cover macOS / iOS, Windows, Linux, and Android in turn
# before bottoming out on the generic CSS family. Display + body share the
# sans stack (Comfortaa is a rounded sans); species falls back to a serif so
# italic binomials stay serif even offline.
_FALLBACK_SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", '
    '"Helvetica Neue", Arial, sans-serif'
)
_FALLBACK_DISPLAY = _FALLBACK_SANS
_FALLBACK_BODY = _FALLBACK_SANS
_FALLBACK_MONO = (
    'ui-monospace, "SFMono-Regular", Menlo, Consolas, '
    '"Liberation Mono", "Courier New", monospace'
)
_FALLBACK_SPECIES = 'Georgia, "Times New Roman", Times, serif'

# Python-side font-family strings -- mirror the CSS custom properties
# below. Use these from Python inline ``style={...}`` dicts and from
# call-sites that don't see CSS variables (Cytoscape stylesheets,
# Plotly layout, dash-table style_cell, etc.).

FONT_FAMILY_DISPLAY: str = f"'{_DISPLAY_PRIMARY}', {_FALLBACK_DISPLAY}"
FONT_FAMILY_BODY: str = f"'{_BODY_PRIMARY}', {_FALLBACK_BODY}"
FONT_FAMILY_MONO: str = f"'{_MONO_PRIMARY}', {_FALLBACK_MONO}"
FONT_FAMILY_SPECIES: str = f"'{_SPECIES_PRIMARY}', {_FALLBACK_SPECIES}"

FONT_TOKENS_CSS = f"""\
@import url("{_GOOGLE_FONTS_URL}");

:root {{
  --font-display: {FONT_FAMILY_DISPLAY};
  --font-body:    {FONT_FAMILY_BODY};
  --font-mono:    {FONT_FAMILY_MONO};
  --font-species: {FONT_FAMILY_SPECIES};
}}
"""

# ---------------------------------------------------------------------------
# Brand / UI palette (PRIMARY -- UI only, never charts)
# ---------------------------------------------------------------------------
#
# Values mirror DESIGN.md "01 -- Color Palette / Primary Colors". Used
# by ``--color-*`` CSS custom properties and importable by Python
# inline-style callers.

COLOR_NAVY: str = "#003660"
COLOR_BLUE: str = "#1b75bc"
COLOR_GOLD: str = "#febc11"
COLOR_WHITE: str = "#ffffff"
COLOR_BG: str = "#FBFEF8"  # near-white warm canvas (DESIGN.md "01 -- Color Palette")
COLOR_SURFACE: str = "#ffffff"
COLOR_BORDER: str = "#dde3ed"
COLOR_RULE: str = "#e8ecf2"
COLOR_MUTED: str = "#8892a4"
COLOR_BODY: str = "#2e3a4e"
COLOR_HEADING: str = COLOR_NAVY  # Same as navy; kept named for semantic call-sites.

#: The one permitted dark surface in this light-theme system: the image stage
#: where the pixels are the data (DESIGN.md "09 -- Image Display"). Use for the
#: fluorescence/OSD canvas background; never for UI chrome.
COLOR_IMAGE_STAGE_DARK: str = "#0e1620"

# ---------------------------------------------------------------------------
# Okabe-Ito data palette (DATA series only -- never UI chrome)
# ---------------------------------------------------------------------------
#
# Colorblind-safe series. See DESIGN.md "Absolute Constraints":
#
#   * NEVER use these colors for UI chrome (buttons, borders, headings).
#   * Series order is fixed: navy, orange, sky, green, blue, purple
#     (vermilion reserved for error/alert).
#   * Yellow may not be used as a thin line / text on white.

OI_ORANGE: str = "#E69F00"
OI_SKY: str = "#56B4E9"
OI_GREEN: str = "#009E73"
OI_VERMILION: str = "#D55E00"
OI_BLUE: str = "#0072B2"
OI_PURPLE: str = "#CC79A7"
OI_YELLOW: str = "#F0E442"
OI_GREY: str = "#BBBBBB"

#: Series 1 of the fixed Okabe-Ito order is navy -- the same hex as the UI
#: ``COLOR_NAVY`` but exported under a data-palette name so chart call sites
#: read as data, not chrome. See DESIGN.md "06 -- Categorical Series Order".
OI_NAVY: str = COLOR_NAVY

#: Fixed categorical series order (DESIGN.md "06"). Index 0..5 are the six
#: categorical series; index 6 (vermilion) is error/alert only; grey is for
#: reference / control / null lines.
OKABE_ITO: tuple[str, ...] = (
    OI_NAVY,
    OI_ORANGE,
    OI_SKY,
    OI_GREEN,
    OI_BLUE,
    OI_PURPLE,
    OI_VERMILION,
)

#: napari label-layer color map (1-indexed) mirroring ``OKABE_ITO`` so a mask
#: color in the dashboard matches the same label in napari (DESIGN.md "07").
#: RGBA tuples normalized 0-1.
OKABE_ITO_NAPARI: dict[int, tuple[float, float, float, float]] = {
    1: (0 / 255, 54 / 255, 96 / 255, 1.0),  # navy
    2: (230 / 255, 159 / 255, 0 / 255, 1.0),  # orange
    3: (86 / 255, 180 / 255, 233 / 255, 1.0),  # sky blue
    4: (0 / 255, 158 / 255, 115 / 255, 1.0),  # bluish green
    5: (0 / 255, 114 / 255, 178 / 255, 1.0),  # blue
    6: (204 / 255, 121 / 255, 167 / 255, 1.0),  # reddish purple
    7: (213 / 255, 94 / 255, 0 / 255, 1.0),  # vermilion (error)
}

# ---------------------------------------------------------------------------
# Error-category → color map (shared by radial wedges, tile badges, ANOVA plots)
# ---------------------------------------------------------------------------
#
# Core error-category tokens map to fixed OI slots; ``other`` maps to grey.
# Custom tokens (registered at runtime) cycle ``_CUSTOM_PALETTE``.
# Colors are drawn from the DATA palette (``OI_*``) only -- never ``COLOR_*``.

ERROR_CATEGORY_COLORS: dict[str, str] = {
    "oversegmented": OI_ORANGE,
    "undersegmented": OI_SKY,
    "merged": OI_PURPLE,
    "background_noise": OI_BLUE,
    "debris": OI_GREEN,
    "other": OI_GREY,
}

#: Palette custom categories cycle through (OI data colors minus the reserved
#: core/Other slots and the alert vermilion / unreadable yellow).
_CUSTOM_PALETTE: tuple[str, ...] = (OI_ORANGE, OI_SKY, OI_GREEN, OI_BLUE, OI_PURPLE)


def category_color(token: str, custom_index: int = 0) -> str:
    """Return the display color for a category token.

    Core tokens map to their fixed OI slot; custom tokens cycle
    ``_CUSTOM_PALETTE`` by their registration index.

    Args:
        token: The category token string (e.g. ``"debris"``, ``"halo"``).
        custom_index: Zero-based registration index for custom categories;
            ignored when ``token`` is a core category. Wraps around
            ``_CUSTOM_PALETTE`` so any non-negative integer is valid.

    Returns:
        A hex color string from the Okabe-Ito data palette.
    """
    if token in ERROR_CATEGORY_COLORS:
        return ERROR_CATEGORY_COLORS[token]
    return _CUSTOM_PALETTE[custom_index % len(_CUSTOM_PALETTE)]


# Semantic state colors -- map to Okabe-Ito (DATA / state only, never chrome).
# Use for chart/state fills and as the *source* hue of badges/alerts; for TEXT
# on white use the darkened AA variants below.
COLOR_SUCCESS: str = OI_GREEN
COLOR_INFO: str = OI_SKY
COLOR_WARNING: str = OI_ORANGE
COLOR_DANGER: str = OI_VERMILION

# Darkened Okabe-Ito TEXT variants for WCAG AA (4.5:1) on white surfaces
# (DESIGN.md "05 -- Badges"). NEVER use a raw OI_* hex as badge / alert / status
# text on white; use these instead.
OI_ORANGE_TEXT: str = "#9A6B00"
OI_SKY_TEXT: str = "#0B6E9E"
OI_GREEN_TEXT: str = "#006B4F"
OI_PURPLE_TEXT: str = "#8B3D6E"
OI_VERMILION_TEXT: str = "#D55E00"  # vermilion meets AA as-is; alerts use #8A3C00

# ---------------------------------------------------------------------------
# Visual tokens (non-palette)
# ---------------------------------------------------------------------------
#
# Values that are colours but not part of the brand/UI or data palettes.

#: The blend-toward colour for the tile-spotlight dim pass — the surroundings
#: of each colony crop fade toward this RGB. Black ``(0, 0, 0)`` matches the
#: ``pad_value`` of :func:`phenotypic.gui._shared.tiles.crop_overlay` so the
#: out-of-image padding and the dimmed in-image surroundings read as one
#: continuous black backdrop. Tuple form (not a hex string) so it drops
#: straight into the NumPy blend and the PIL ``Image.new`` fill.
TILE_DIM_RGB: tuple[int, int, int] = (0, 0, 0)

# ---------------------------------------------------------------------------
# Type scale (rem-based; tuned for ~15 px body)
# ---------------------------------------------------------------------------

TEXT_2XS: str = "0.625rem"  # ~10 px -- data micro floor (chart axis, sparkline)
TEXT_XS: str = "0.6875rem"  # ~11 px -- captions, footnotes
TEXT_SM: str = "0.8125rem"  # ~13 px -- secondary UI labels
TEXT_BASE: str = "0.9375rem"  # ~15 px -- body text
TEXT_MD: str = "1.0625rem"  # ~17 px -- emphasized body
TEXT_LG: str = "1.25rem"  # ~20 px -- subhead
TEXT_XL: str = "1.5rem"  # ~24 px -- builder canvas titles
TEXT_2XL: str = "1.875rem"  # ~30 px -- page / dashboard top titles
TEXT_3XL: str = "2.5rem"  # ~40 px -- large stat numerics, hero numbers
TEXT_4XL: str = "3.25rem"  # ~52 px -- reserve (hero display)

# ---------------------------------------------------------------------------
# Semantic typography roles -- single source of truth for `font-size`.
# ---------------------------------------------------------------------------
#
# New code (CSS or Python inline styles) should use these semantic names
# rather than the raw `TEXT_*` rem-scale primitives above. Each Python
# constant has a matching `--font-size-*` CSS custom property spliced in
# by `inject_design_tokens()`.
#
#   FONT_SIZE_DISPLAY   -- large stat numerics, hero numbers
#   FONT_SIZE_TITLE     -- page / dashboard top titles
#   FONT_SIZE_HEADER_1  -- section heads
#   FONT_SIZE_HEADER_2  -- sub-section heads
#   FONT_SIZE_BODY_LG   -- emphasized / lead body
#   FONT_SIZE_BODY      -- default body copy
#   FONT_SIZE_BODY_SM   -- UI titles, button labels, dense body (13 px)
#   FONT_SIZE_CAPTION   -- form labels, overlines, badge text (11 px)
#   FONT_SIZE_MICRO     -- chart axis ticks, sparkline floor (10 px)
#
# NOTE (DESIGN.md "02.7"): the 13 px rung was renamed from ``FONT_SIZE_LABEL``
# to ``FONT_SIZE_BODY_SM`` and the Label / Overline role moved down to
# ``FONT_SIZE_CAPTION`` (11 px). ``FONT_SIZE_LABEL`` is kept as a DEPRECATED
# alias of ``FONT_SIZE_BODY_SM``; new call sites use the new names.

FONT_SIZE_DISPLAY: str = TEXT_3XL  # 2.5rem
FONT_SIZE_TITLE: str = TEXT_2XL  # 1.875rem
FONT_SIZE_HEADER_1: str = TEXT_XL  # 1.5rem
FONT_SIZE_HEADER_2: str = TEXT_LG  # 1.25rem
FONT_SIZE_BODY_LG: str = TEXT_MD  # 1.0625rem
FONT_SIZE_BODY: str = TEXT_BASE  # 0.9375rem
FONT_SIZE_BODY_SM: str = TEXT_SM  # 0.8125rem
FONT_SIZE_CAPTION: str = TEXT_XS  # 0.6875rem
FONT_SIZE_MICRO: str = TEXT_2XS  # 0.625rem
FONT_SIZE_LABEL: str = FONT_SIZE_BODY_SM  # DEPRECATED -- use FONT_SIZE_BODY_SM

# ---------------------------------------------------------------------------
# 8 pt spacing grid (DESIGN.md "Spacing")
# ---------------------------------------------------------------------------

SPACING_1: str = "0.25rem"  # 4 px
SPACING_2: str = "0.5rem"  # 8 px
SPACING_3: str = "0.75rem"  # 12 px
SPACING_4: str = "1rem"  # 16 px
SPACING_5: str = "1.25rem"  # 20 px
SPACING_6: str = "1.5rem"  # 24 px
SPACING_8: str = "2rem"  # 32 px
SPACING_10: str = "2.5rem"  # 40 px
SPACING_12: str = "3rem"  # 48 px
SPACING_16: str = "4rem"  # 64 px -- major section rhythm

# ---------------------------------------------------------------------------
# Radius / shadow / motion
# ---------------------------------------------------------------------------

RADIUS_SM: str = "3px"
RADIUS: str = "6px"
RADIUS_MD: str = "10px"
RADIUS_LG: str = "16px"

SHADOW_SM: str = "0 1px 3px rgba(0,54,96,0.07), 0 1px 2px rgba(0,54,96,0.04)"
SHADOW: str = "0 4px 12px rgba(0,54,96,0.08), 0 1px 3px rgba(0,54,96,0.05)"
SHADOW_MD: str = "0 8px 24px rgba(0,54,96,0.10), 0 2px 6px rgba(0,54,96,0.06)"
SHADOW_LG: str = "0 16px 40px rgba(0,54,96,0.12), 0 4px 12px rgba(0,54,96,0.07)"

# Line-height & tracking tokens (DESIGN.md "02.3"). Python mirrors of the
# injected ``--leading-*`` / ``--tracking-*`` custom properties.
LEADING_DISPLAY: str = "1.1"
LEADING_TIGHT: str = "1.2"
LEADING_SNUG: str = "1.3"
LEADING_NORMAL: str = "1.45"
LEADING_RELAXED: str = "1.6"

TRACKING_TIGHT: str = "-0.02em"
TRACKING_SNUG: str = "-0.01em"
TRACKING_NORMAL: str = "0"
TRACKING_BUTTON: str = "0.01em"
TRACKING_WIDE: str = "0.08em"
TRACKING_WIDER: str = "0.12em"

EASE_OUT: str = "cubic-bezier(0.22, 1, 0.36, 1)"
TRANSITION: str = f"180ms {EASE_OUT}"

# ---------------------------------------------------------------------------
# Combined :root block injected into every Dash app
# ---------------------------------------------------------------------------
#
# Order matches the original ``shell.css`` ``:root`` block so dependents
# that override individual variables see the same cascade.

DESIGN_TOKENS_CSS = f"""\
:root {{
  /* ---- Brand / UI ---- */
  --color-navy:    {COLOR_NAVY};
  --color-blue:    {COLOR_BLUE};
  --color-gold:    {COLOR_GOLD};
  --color-white:   {COLOR_WHITE};
  --color-bg:      {COLOR_BG};
  --color-surface: {COLOR_SURFACE};
  --color-border:  {COLOR_BORDER};
  --color-rule:    {COLOR_RULE};
  --color-muted:   {COLOR_MUTED};
  --color-body:    {COLOR_BODY};
  --color-heading: {COLOR_HEADING};

  /* ---- Okabe-Ito data palette ---- */
  --oi-navy:      {OI_NAVY};
  --oi-orange:    {OI_ORANGE};
  --oi-sky:       {OI_SKY};
  --oi-green:     {OI_GREEN};
  --oi-vermilion: {OI_VERMILION};
  --oi-blue:      {OI_BLUE};
  --oi-purple:    {OI_PURPLE};
  --oi-yellow:    {OI_YELLOW};
  --oi-grey:      {OI_GREY};

  /* ---- Darkened OI text variants (badge / alert / status text on white) ---- */
  --oi-orange-text:    {OI_ORANGE_TEXT};
  --oi-sky-text:       {OI_SKY_TEXT};
  --oi-green-text:     {OI_GREEN_TEXT};
  --oi-purple-text:    {OI_PURPLE_TEXT};
  --oi-vermilion-text: {OI_VERMILION_TEXT};

  /* ---- Semantic state aliases (data / state only, never UI chrome) ---- */
  --color-success: {COLOR_SUCCESS};
  --color-info:    {COLOR_INFO};
  --color-warning: {COLOR_WARNING};
  --color-danger:  {COLOR_DANGER};

  /* ---- Image stage (the one permitted dark surface) ---- */
  --color-image-stage-dark: {COLOR_IMAGE_STAGE_DARK};

  /* ---- Type scale ---- */
  --text-2xs:  {TEXT_2XS};
  --text-xs:   {TEXT_XS};
  --text-sm:   {TEXT_SM};
  --text-base: {TEXT_BASE};
  --text-md:   {TEXT_MD};
  --text-lg:   {TEXT_LG};
  --text-xl:   {TEXT_XL};
  --text-2xl:  {TEXT_2XL};
  --text-3xl:  {TEXT_3XL};
  --text-4xl:  {TEXT_4XL};

  /* ---- Semantic font-size aliases (preferred over --text-*) ---- */
  --font-size-display:  var(--text-3xl);
  --font-size-title:    var(--text-2xl);
  --font-size-header-1: var(--text-xl);
  --font-size-header-2: var(--text-lg);
  --font-size-body-lg:  var(--text-md);
  --font-size-body:     var(--text-base);
  --font-size-body-sm:  var(--text-sm);
  --font-size-caption:  var(--text-xs);
  --font-size-micro:    var(--text-2xs);
  --font-size-label:    var(--text-sm);  /* DEPRECATED alias of --font-size-body-sm */

  /* ---- Line-height ---- */
  --leading-display: {LEADING_DISPLAY};
  --leading-tight:   {LEADING_TIGHT};
  --leading-snug:    {LEADING_SNUG};
  --leading-normal:  {LEADING_NORMAL};
  --leading-relaxed: {LEADING_RELAXED};

  /* ---- Letter-spacing ---- */
  --tracking-tight:  {TRACKING_TIGHT};
  --tracking-snug:   {TRACKING_SNUG};
  --tracking-normal: {TRACKING_NORMAL};
  --tracking-button: {TRACKING_BUTTON};
  --tracking-wide:   {TRACKING_WIDE};
  --tracking-wider:  {TRACKING_WIDER};

  /* ---- 8 pt spacing grid ---- */
  --sp-1:  {SPACING_1};
  --sp-2:  {SPACING_2};
  --sp-3:  {SPACING_3};
  --sp-4:  {SPACING_4};
  --sp-5:  {SPACING_5};
  --sp-6:  {SPACING_6};
  --sp-8:  {SPACING_8};
  --sp-10: {SPACING_10};
  --sp-12: {SPACING_12};
  --sp-16: {SPACING_16};

  /* ---- Radius / shadow / motion ---- */
  --radius-sm: {RADIUS_SM};
  --radius:    {RADIUS};
  --radius-md: {RADIUS_MD};
  --radius-lg: {RADIUS_LG};

  --shadow-sm: {SHADOW_SM};
  --shadow:    {SHADOW};
  --shadow-md: {SHADOW_MD};
  --shadow-lg: {SHADOW_LG};

  --ease-out:   {EASE_OUT};
  --transition: {TRANSITION};
}}
"""

# ---------------------------------------------------------------------------
# Bootstrap remap -- map dbc.themes.BOOTSTRAP onto the brand tokens
# ---------------------------------------------------------------------------
#
# Every sub-app loads stock ``dbc.themes.BOOTSTRAP``, so ``color="primary"``
# etc. would otherwise render Bootstrap blue/grey/red. This single injected
# layer points Bootstrap's CSS variables and button variants at the brand
# palette so no per-call-site ``color=`` change is needed. Okabe-Ito is never a
# button fill here; only the danger variant uses vermilion (DESIGN.md "05").

_NAVY_HOVER = "#00284a"  # navy darkened ~8% for filled-button hover

BOOTSTRAP_OVERRIDE_CSS = f"""\
:root {{
  --bs-body-font-family: var(--font-body);
  --bs-font-monospace:   var(--font-mono);
  --bs-body-bg:          var(--color-bg);
  --bs-body-color:       var(--color-body);
  --bs-border-radius:    var(--radius);
  --bs-border-color:     var(--color-border);

  --bs-primary:   var(--color-navy);    --bs-primary-rgb:   0,54,96;
  --bs-secondary: var(--color-muted);   --bs-secondary-rgb: 136,146,164;
  --bs-success:   var(--oi-green);      --bs-success-rgb:   0,158,115;
  --bs-info:      var(--oi-sky);        --bs-info-rgb:      86,180,233;
  --bs-warning:   var(--oi-orange);     --bs-warning-rgb:   230,159,0;
  --bs-danger:    var(--oi-vermilion);  --bs-danger-rgb:    213,94,0;

  --bs-link-color: var(--color-blue);   --bs-link-color-rgb: 27,117,188;
  --bs-link-hover-color: var(--color-navy);
}}

.btn {{
  font-family: var(--font-body);
  font-weight: 500;
  letter-spacing: var(--tracking-button);
  border-width: 1.5px;
  border-radius: var(--radius);
}}
.btn-primary {{
  --bs-btn-bg: var(--color-navy);  --bs-btn-border-color: var(--color-navy);
  --bs-btn-hover-bg: {_NAVY_HOVER};  --bs-btn-hover-border-color: {_NAVY_HOVER};
  --bs-btn-active-bg: {_NAVY_HOVER}; --bs-btn-active-border-color: {_NAVY_HOVER};
  --bs-btn-color: #fff; --bs-btn-hover-color: #fff; --bs-btn-active-color: #fff;
}}
.btn-secondary, .btn-outline-secondary, .btn-outline-primary {{
  --bs-btn-bg: transparent; --bs-btn-color: var(--color-navy);
  --bs-btn-border-color: var(--color-border);
  --bs-btn-hover-bg: rgba(27,117,188,0.04); --bs-btn-hover-color: var(--color-blue);
  --bs-btn-hover-border-color: var(--color-blue);
  --bs-btn-active-bg: rgba(27,117,188,0.08); --bs-btn-active-color: var(--color-blue);
}}
.btn-danger, .btn-outline-danger {{
  --bs-btn-bg: transparent; --bs-btn-color: var(--oi-vermilion);
  --bs-btn-border-color: var(--oi-vermilion);
  --bs-btn-hover-bg: var(--oi-vermilion); --bs-btn-hover-color: #fff;
  --bs-btn-hover-border-color: var(--oi-vermilion);
}}
.btn-link {{ --bs-btn-color: var(--color-blue); --bs-btn-hover-color: var(--color-navy); text-decoration: none; }}
.text-muted {{ color: var(--color-muted) !important; }}
.text-monospace {{ font-family: var(--font-mono) !important; }}
"""

# ---------------------------------------------------------------------------
# Base element defaults + named text-style classes (DESIGN.md "02.5 / 02.6")
# ---------------------------------------------------------------------------
#
# Element defaults are conservative -- they establish the serif/mono *identity*
# (family, weight, heading color) for raw tags without forcing sizes, so they
# don't fight component-set sizes. The ``.text-*`` classes are the full named
# recipes; apply one class to a node to get the exact spec style.

BASE_STYLES_CSS = """\
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-display);
  font-weight: 400;
  color: var(--color-heading);
}
code, kbd, samp, pre {
  font-family: var(--font-mono);
}
code:not(pre code), kbd {
  background: #edf2f7;
  color: var(--color-navy);
  padding: 1px 5px;
  border-radius: var(--radius-sm);
}

.text-display    { font-family: var(--font-display); font-size: var(--font-size-display);   font-weight: 400; line-height: var(--leading-display); letter-spacing: var(--tracking-tight); color: var(--color-heading); }
.text-title      { font-family: var(--font-display); font-size: var(--font-size-title);     font-weight: 400; line-height: var(--leading-snug);    letter-spacing: var(--tracking-snug);  color: var(--color-heading); }
.text-header     { font-family: var(--font-display); font-size: var(--font-size-header-1);  font-weight: 400; line-height: var(--leading-tight);   letter-spacing: var(--tracking-snug);  color: var(--color-heading); }
.text-h2         { font-family: var(--font-display); font-size: var(--font-size-header-2);  font-weight: 400; line-height: var(--leading-snug);    color: var(--color-heading); }
.text-h3         { font-family: var(--font-display); font-size: var(--font-size-body-lg);   font-weight: 400; line-height: var(--leading-snug);    color: var(--color-heading); }

.text-body-lg    { font-family: var(--font-body); font-size: var(--font-size-body-lg); font-weight: 400; line-height: var(--leading-relaxed); color: var(--color-body); }
.text-body       { font-family: var(--font-body); font-size: var(--font-size-body);    font-weight: 400; line-height: var(--leading-relaxed); color: var(--color-body); }
.text-body-sm    { font-family: var(--font-body); font-size: var(--font-size-body-sm); font-weight: 400; line-height: var(--leading-relaxed); color: var(--color-body); }

.text-ui-title   { font-family: var(--font-body); font-size: var(--font-size-body-sm); font-weight: 600; line-height: var(--leading-snug); color: var(--color-heading); }
.text-button     { font-family: var(--font-body); font-size: var(--font-size-body-sm); font-weight: 500; line-height: 1; letter-spacing: var(--tracking-button); }

.text-label      { font-family: var(--font-mono); font-size: var(--font-size-caption); font-weight: 500; line-height: var(--leading-tight); letter-spacing: var(--tracking-wide);  text-transform: uppercase; color: var(--color-muted); }
.text-overline   { font-family: var(--font-mono); font-size: var(--font-size-caption); font-weight: 500; line-height: var(--leading-tight); letter-spacing: var(--tracking-wider); text-transform: uppercase; color: var(--color-muted); }
.text-caption    { font-family: var(--font-mono); font-size: var(--font-size-caption); font-weight: 400; line-height: var(--leading-normal); color: var(--color-muted); }
.text-data       { font-family: var(--font-mono); font-size: var(--font-size-body);    font-weight: 500; line-height: var(--leading-normal); color: var(--color-heading); }
.text-data--muted{ font-family: var(--font-mono); font-size: var(--font-size-body);    font-weight: 400; line-height: var(--leading-normal); color: var(--color-muted); }
.text-data-micro { font-family: var(--font-mono); font-size: var(--font-size-micro);   font-weight: 400; line-height: var(--leading-tight); letter-spacing: 0.02em; color: var(--color-muted); }

.is-species      { font-family: var(--font-species); font-style: italic; }
"""

# ---------------------------------------------------------------------------
# Injection helper
# ---------------------------------------------------------------------------

_MARKER = "<!-- phenotypic-design-tokens -->"


def inject_design_tokens(app) -> None:  # type: ignore[no-untyped-def]
    """Splice the font + design-token CSS into ``app.index_string``.

    Idempotent via a marker comment so callers don't have to coordinate.
    Both the standalone sub-app factories and ``wrap_in_chrome`` invoke
    this; only the first call inserts the block.

    The token ``<style>`` is inserted immediately after Dash's ``{%css%}``
    placeholder so any sub-app stylesheet still loads first -- the
    ``--font-*``, ``--color-*``, ``--text-*``, ``--radius-*``,
    ``--shadow-*``, and ``--ease-*`` declarations land in the cascade
    before they're referenced via ``var(--…)``.

    Four blocks are spliced, in cascade order: the font ``@import`` + role
    families, the design-token ``:root`` block, the Bootstrap remap (so
    ``dbc.themes.BOOTSTRAP`` inherits the brand palette), and the base element
    defaults + named ``.text-*`` style classes. The remap and base styles land
    *after* Bootstrap and the sub-app stylesheet so they win the cascade.
    """
    if _MARKER in app.index_string:
        return
    style_block = (
        f"{_MARKER}\n"
        f"<style>\n{FONT_TOKENS_CSS}</style>\n"
        f"<style>\n{DESIGN_TOKENS_CSS}</style>\n"
        f"<style>\n{BOOTSTRAP_OVERRIDE_CSS}</style>\n"
        f"<style>\n{BASE_STYLES_CSS}</style>"
    )
    if "{%css%}" in app.index_string:
        app.index_string = app.index_string.replace(
                "{%css%}", "{%css%}\n" + style_block, 1
        )
    else:  # pragma: no cover -- defensive: custom templates without {%css%}
        app.index_string = app.index_string.replace(
                "</head>", style_block + "\n</head>", 1
        )
