"""Single-source-of-truth design tokens for the GUI.

Splices a small CSS block into every Dash app's ``index_string`` that:

* ``@import``s the active Google Font.
* Declares the ``--font-display``, ``--font-body``, and ``--font-mono``
  CSS custom properties used throughout ``shell.css``, ``builder.css``,
  ``results_viewer.css``, and ``run_console.css``.

To swap fonts: change ``_GOOGLE_FONT_FAMILY`` + ``_GOOGLE_FONT_URL`` to
one of the commented alternatives below. Every mounted Dash app picks
up the new font on next reload -- no CSS file edits required.
"""
from __future__ import annotations

__all__ = ["FONT_TOKENS_CSS", "inject_design_tokens"]

# ---------------------------------------------------------------------------
# Active font (Google Fonts)
# ---------------------------------------------------------------------------

# _GOOGLE_FONT_FAMILY = "Rock 3D"
# _GOOGLE_FONT_URL = (
#     "https://fonts.googleapis.com/css2?family=Rock+3D&display=swap"
# )

# Alternatives -- uncomment one block (and comment the active one above)
# to switch the GUI font. Both are also Google Fonts.
#
_GOOGLE_FONT_FAMILY = "Mate SC"
_GOOGLE_FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Mate+SC&display=swap"
)
#
# _GOOGLE_FONT_FAMILY = "Roboto"
# _GOOGLE_FONT_URL = (
#     "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"
# )


FONT_TOKENS_CSS = f"""\
@import url("{_GOOGLE_FONT_URL}");

:root {{
  --font-display: '{_GOOGLE_FONT_FAMILY}', Georgia, serif;
  --font-body:    '{_GOOGLE_FONT_FAMILY}', system-ui, sans-serif;
  --font-mono:    '{_GOOGLE_FONT_FAMILY}', 'Courier New', monospace;
}}
"""

# ---------------------------------------------------------------------------
# Injection helper
# ---------------------------------------------------------------------------

_MARKER = "<!-- phenotypic-design-tokens -->"


def inject_design_tokens(app) -> None:  # type: ignore[no-untyped-def]
    """Splice :data:`FONT_TOKENS_CSS` into ``app.index_string``.

    Idempotent via a marker comment so callers don't have to coordinate.
    Both the standalone sub-app factories and ``wrap_in_chrome`` invoke
    this; only the first call inserts the block.

    The token ``<style>`` is inserted immediately after Dash's ``{%css%}``
    placeholder so any sub-app stylesheet still loads first -- the
    ``--font-*`` declarations land in the cascade before they're
    referenced via ``var(--font-body)`` etc.
    """
    if _MARKER in app.index_string:
        return
    style_block = f"{_MARKER}\n<style>\n{FONT_TOKENS_CSS}</style>"
    if "{%css%}" in app.index_string:
        app.index_string = app.index_string.replace(
                "{%css%}", "{%css%}\n" + style_block, 1
        )
    else:  # pragma: no cover -- defensive: custom templates without {%css%}
        app.index_string = app.index_string.replace(
                "</head>", style_block + "\n</head>", 1
        )
