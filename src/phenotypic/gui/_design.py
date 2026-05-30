"""Single-source-of-truth design tokens for the GUI.

Splices a small CSS block into every Dash app's ``index_string`` that:

* ``@import``s the active Google Font.
* Declares the ``--font-display``, ``--font-body``, and ``--font-mono``
  CSS custom properties used throughout ``shell.css``, ``builder.css``,
  ``results_viewer.css``, and ``run_console.css``.
* Declares the brand color, Okabe-Ito data, type-scale, semantic
  ``--font-size-*`` aliases, radius, shadow, and ease/transition tokens
  that previously duplicated across ``shell.css`` and ``builder.css``
  ``:root`` blocks.

To swap fonts: change ``_GOOGLE_FONT_FAMILY`` + ``_GOOGLE_FONT_URL`` to
one of the commented alternatives below. Every mounted Dash app picks
up the new font on next reload -- no CSS file edits required.

To swap a design color or sizing token: edit the matching ``COLOR_*``,
``TEXT_*``, ``FONT_SIZE_*``, ``RADIUS_*``, ``SHADOW_*``, or ``EASE_*``
constant. The :data:`DESIGN_TOKENS_CSS` block is rebuilt at module
import time and re-injected into every Dash app via
:func:`inject_design_tokens`.

Python callers that need an inline-style hex, font family, or font size
(e.g. Plotly figure colors, ``html.Div(style={"color": COLOR_NAVY})``,
``style_cell={"fontFamily": FONT_FAMILY_MONO, "fontSize": FONT_SIZE_LABEL}``)
should ``import`` the constant instead of re-spelling literals -- see
``DESIGN.md`` for the full UI-vs-data palette rules and the typography
scale.
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
    # ---- Okabe-Ito data palette (DATA ONLY -- never UI chrome) ----
    "OI_ORANGE",
    "OI_SKY",
    "OI_GREEN",
    "OI_VERMILION",
    "OI_BLUE",
    "OI_PURPLE",
    "OI_YELLOW",
    "OI_GREY",
    # ---- Visual tokens (non-palette) ----
    "TILE_DIM_RGB",
    # ---- Type scale / radius / shadow / motion ----
    "TEXT_XS",
    "TEXT_SM",
    "TEXT_BASE",
    "TEXT_MD",
    "TEXT_LG",
    "TEXT_XL",
    "TEXT_2XL",
    "TEXT_3XL",
    # ---- Semantic font-size aliases (preferred call form) ----
    "FONT_SIZE_DISPLAY",
    "FONT_SIZE_TITLE",
    "FONT_SIZE_HEADER_1",
    "FONT_SIZE_HEADER_2",
    "FONT_SIZE_BODY_LG",
    "FONT_SIZE_BODY",
    "FONT_SIZE_LABEL",
    "FONT_SIZE_CAPTION",
    # ---- Python-side font-family constants ----
    "FONT_FAMILY_DISPLAY",
    "FONT_FAMILY_BODY",
    "FONT_FAMILY_MONO",
    "SPACING_1",
    "SPACING_2",
    "SPACING_3",
    "SPACING_4",
    "SPACING_5",
    "SPACING_6",
    "SPACING_8",
    "RADIUS_SM",
    "RADIUS",
    "RADIUS_MD",
    "RADIUS_LG",
    "SHADOW_SM",
    "SHADOW",
    "SHADOW_MD",
    "EASE_OUT",
    "TRANSITION",
    # ---- CSS bundles ----
    "FONT_TOKENS_CSS",
    "DESIGN_TOKENS_CSS",
    # ---- Injector ----
    "inject_design_tokens",
]

# ---------------------------------------------------------------------------
# Active font (Google Fonts)
# ---------------------------------------------------------------------------

# _GOOGLE_FONT_FAMILY = "Rock 3D"
# _GOOGLE_FONT_URL = (
#     "https://fonts.googleapis.com/css2?family=Rock+3D&display=swap"
# )

# Alternatives -- uncomment one block (and comment the active one above)
# to switch the GUI font. All are also Google Fonts.
#
# _GOOGLE_FONT_FAMILY = "Mate SC"
# _GOOGLE_FONT_URL = (
#     "https://fonts.googleapis.com/css2?family=Mate+SC&display=swap"
# )
#
_GOOGLE_FONT_FAMILY = "Roboto"
_GOOGLE_FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"
)

# Cross-platform fallbacks: kick in if the Google Font is blocked,
# slow to load, or (in the Google Sans case) not actually served. The
# stacks cover macOS / iOS, Windows, Linux, and Android in turn before
# bottoming out on the generic CSS family.
_FALLBACK_DISPLAY = 'Georgia, "Times New Roman", Times, serif'
_FALLBACK_BODY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", '
    '"Helvetica Neue", Arial, sans-serif'
)
_FALLBACK_MONO = (
    'ui-monospace, "SFMono-Regular", Menlo, Consolas, '
    '"Liberation Mono", "Courier New", monospace'
)

FONT_TOKENS_CSS = f"""\
@import url("{_GOOGLE_FONT_URL}");

:root {{
  --font-display: '{_GOOGLE_FONT_FAMILY}', {_FALLBACK_DISPLAY};
  --font-body:    '{_GOOGLE_FONT_FAMILY}', {_FALLBACK_BODY};
  --font-mono:    '{_GOOGLE_FONT_FAMILY}', {_FALLBACK_MONO};
}}
"""

# Python-side font-family strings -- mirror the CSS custom properties
# above. Use these from Python inline ``style={...}`` dicts and from
# call-sites that don't see CSS variables (Cytoscape stylesheets,
# Plotly layout, dash-table style_cell, etc.).

FONT_FAMILY_DISPLAY: str = f"'{_GOOGLE_FONT_FAMILY}', {_FALLBACK_DISPLAY}"
FONT_FAMILY_BODY: str = f"'{_GOOGLE_FONT_FAMILY}', {_FALLBACK_BODY}"
FONT_FAMILY_MONO: str = f"'{_GOOGLE_FONT_FAMILY}', {_FALLBACK_MONO}"

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
COLOR_BG: str = "#f5f7fa"
COLOR_SURFACE: str = "#ffffff"
COLOR_BORDER: str = "#dde3ed"
COLOR_RULE: str = "#e8ecf2"
COLOR_MUTED: str = "#8892a4"
COLOR_BODY: str = "#2e3a4e"
COLOR_HEADING: str = COLOR_NAVY  # Same as navy; kept named for semantic call-sites.

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

TEXT_XS: str = "0.6875rem"  # ~11 px -- captions, footnotes
TEXT_SM: str = "0.8125rem"  # ~13 px -- secondary UI labels
TEXT_BASE: str = "0.9375rem"  # ~15 px -- body text
TEXT_MD: str = "1.0625rem"  # ~17 px -- emphasized body
TEXT_LG: str = "1.25rem"  # ~20 px -- subhead
TEXT_XL: str = "1.5rem"  # ~24 px -- builder canvas titles
TEXT_2XL: str = "1.875rem"  # ~30 px -- page / dashboard top titles
TEXT_3XL: str = "2.5rem"  # ~40 px -- large stat numerics, hero numbers

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
#   FONT_SIZE_LABEL     -- form labels, picker labels, secondary chrome
#   FONT_SIZE_CAPTION   -- footnotes, badge text, overlines

FONT_SIZE_DISPLAY: str = TEXT_3XL  # 2.5rem
FONT_SIZE_TITLE: str = TEXT_2XL  # 1.875rem
FONT_SIZE_HEADER_1: str = TEXT_XL  # 1.5rem
FONT_SIZE_HEADER_2: str = TEXT_LG  # 1.25rem
FONT_SIZE_BODY_LG: str = TEXT_MD  # 1.0625rem
FONT_SIZE_BODY: str = TEXT_BASE  # 0.9375rem
FONT_SIZE_LABEL: str = TEXT_SM  # 0.8125rem
FONT_SIZE_CAPTION: str = TEXT_XS  # 0.6875rem

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
  --oi-orange:    {OI_ORANGE};
  --oi-sky:       {OI_SKY};
  --oi-green:     {OI_GREEN};
  --oi-vermilion: {OI_VERMILION};
  --oi-blue:      {OI_BLUE};
  --oi-purple:    {OI_PURPLE};
  --oi-yellow:    {OI_YELLOW};
  --oi-grey:      {OI_GREY};

  /* ---- Type scale ---- */
  --text-xs:   {TEXT_XS};
  --text-sm:   {TEXT_SM};
  --text-base: {TEXT_BASE};
  --text-md:   {TEXT_MD};
  --text-lg:   {TEXT_LG};
  --text-xl:   {TEXT_XL};
  --text-2xl:  {TEXT_2XL};
  --text-3xl:  {TEXT_3XL};

  /* ---- Semantic font-size aliases (preferred over --text-*) ---- */
  --font-size-display:  var(--text-3xl);
  --font-size-title:    var(--text-2xl);
  --font-size-header-1: var(--text-xl);
  --font-size-header-2: var(--text-lg);
  --font-size-body-lg:  var(--text-md);
  --font-size-body:     var(--text-base);
  --font-size-label:    var(--text-sm);
  --font-size-caption:  var(--text-xs);

  /* ---- 8 pt spacing grid ---- */
  --sp-1: {SPACING_1};
  --sp-2: {SPACING_2};
  --sp-3: {SPACING_3};
  --sp-4: {SPACING_4};
  --sp-5: {SPACING_5};
  --sp-6: {SPACING_6};
  --sp-8: {SPACING_8};

  /* ---- Radius / shadow / motion ---- */
  --radius-sm: {RADIUS_SM};
  --radius:    {RADIUS};
  --radius-md: {RADIUS_MD};
  --radius-lg: {RADIUS_LG};

  --shadow-sm: {SHADOW_SM};
  --shadow:    {SHADOW};
  --shadow-md: {SHADOW_MD};

  --ease-out:   {EASE_OUT};
  --transition: {TRANSITION};
}}
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
    """
    if _MARKER in app.index_string:
        return
    style_block = (
        f"{_MARKER}\n"
        f"<style>\n{FONT_TOKENS_CSS}</style>\n"
        f"<style>\n{DESIGN_TOKENS_CSS}</style>"
    )
    if "{%css%}" in app.index_string:
        app.index_string = app.index_string.replace(
                "{%css%}", "{%css%}\n" + style_block, 1
        )
    else:  # pragma: no cover -- defensive: custom templates without {%css%}
        app.index_string = app.index_string.replace(
                "</head>", style_block + "\n</head>", 1
        )
