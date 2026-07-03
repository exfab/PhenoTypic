"""Unit tests for Dash index-string URL-prefix injection."""
from __future__ import annotations

from phenotypic.gui._url_prefix import dash_index_string_with_app_prefix


def test_dash_index_string_exposes_app_prefix() -> None:
    """The helper injects the browser-visible prefix for client assets."""
    template = dash_index_string_with_app_prefix("/results/")

    assert 'window.__phenotypicAppPrefix = "/results/";' in template


def test_dash_index_string_escapes_prefix_for_inline_script() -> None:
    """Inline script injection escapes values that can break JavaScript/HTML."""
    template = dash_index_string_with_app_prefix('/x\\"</script>/')

    assert 'window.__phenotypicAppPrefix = "/x\\\\\\"<\\/script>/";' in template
    assert '"/x\\"</script>/"' not in template


def test_dash_index_string_preserves_dash_placeholders() -> None:
    """The helper keeps every placeholder Dash needs to render an app."""
    template = dash_index_string_with_app_prefix("/")

    for placeholder in (
        "{%metas%}",
        "{%title%}",
        "{%favicon%}",
        "{%css%}",
        "{%app_entry%}",
        "{%config%}",
        "{%scripts%}",
        "{%renderer%}",
    ):
        assert placeholder in template

