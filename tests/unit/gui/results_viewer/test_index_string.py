"""Unit tests for ``_index_string_with_prefix`` JS-string escaping.

The factory injects ``window.__phenotypicAppPrefix = "<prefix>";`` into
``app.index_string`` so ``results_viewer.js`` can build hub-aware URLs.
The escape pass guards against unusual prefix values: backslashes,
double quotes, and the ``</`` digraph that would otherwise terminate
the inline ``<script>`` tag prematurely.
"""
from __future__ import annotations

import re

from phenotypic.gui.results_viewer._app import _index_string_with_prefix


def _extract_prefix_literal(index_string: str) -> str:
    """Pull the JS string literal out of the injected ``<script>``."""
    match = re.search(
        r'window\.__phenotypicAppPrefix\s*=\s*"((?:[^"\\]|\\.)*)";',
        index_string,
    )
    assert match is not None, "no prefix literal injected"
    return match.group(1)


def test_default_prefix_is_root() -> None:
    template = _index_string_with_prefix("/")
    assert _extract_prefix_literal(template) == "/"


def test_hub_prefix_is_results_slash() -> None:
    template = _index_string_with_prefix("/results/")
    assert _extract_prefix_literal(template) == "/results/"


def test_escapes_double_quote() -> None:
    template = _index_string_with_prefix('/weird"prefix/')
    literal = _extract_prefix_literal(template)
    # Confirm the ``\"`` survived; the regex's capture group includes
    # backslash escapes verbatim.
    assert '\\"' in literal


def test_escapes_backslash() -> None:
    template = _index_string_with_prefix(r"/back\slash/")
    literal = _extract_prefix_literal(template)
    assert "\\\\" in literal


def test_escapes_closing_script_tag() -> None:
    """``</script>`` in the prefix must NOT terminate the script tag.

    Without the ``</`` -> ``<\\/`` escape, an HTML parser would close the
    inline ``<script>`` tag at the first ``</`` and treat anything after
    as raw HTML. The escape pass replaces ``</`` with the JS-legal
    ``<\\/`` so the script body stays intact.
    """
    template = _index_string_with_prefix("/</script><img>/")
    # The literal ``</`` digraph must be absent from the entire injected
    # template body — otherwise the browser ends the <script> tag early.
    body = template[template.index("__phenotypicAppPrefix"):]
    assert "</script><img" not in body
    # The escaped form is present.
    assert "<\\/script><img" in body
