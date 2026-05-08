"""Unit tests for the shared GUI constants + design tokens.

These modules (``phenotypic.gui._config`` and ``phenotypic.gui._design``)
are the single source of truth for launcher defaults, mount prefixes,
Flask config keys, sandbox subdirectory names, and CSS design tokens.
The tests below assert their public surface and helpers behave as
documented so a future drift (e.g. accidentally dropping a token from
the injection block) fails loudly.
"""
from __future__ import annotations

import argparse

import pytest

from phenotypic.gui import _config, _design


# ---------------------------------------------------------------------------
# _config public surface
# ---------------------------------------------------------------------------


class TestConfigConstants:
    """Public-surface guarantees for ``phenotypic.gui._config``."""

    def test_launcher_defaults(self) -> None:
        """Default host/port match the CLI launchers and SSH-tunnel hint."""
        assert _config.DEFAULT_HOST == "127.0.0.1"
        assert _config.DEFAULT_PORT == 8050
        assert "%(asctime)s" in _config.LOG_FORMAT
        # SSH hint must reference the default port so the two stay synced.
        assert str(_config.DEFAULT_PORT) in _config.SSH_TUNNEL_HINT

    def test_mount_prefixes_end_with_slash(self) -> None:
        """``MOUNT_*`` strings always end with ``/`` (matches Dash url-prefix contract)."""
        for name in ("MOUNT_HOME", "MOUNT_BUILDER", "MOUNT_VIEWER", "MOUNT_RUN"):
            value: str = getattr(_config, name)
            assert value.startswith("/"), f"{name}={value!r} must start with '/'"
            assert value.endswith("/"), f"{name}={value!r} must end with '/'"

    def test_blueprint_prefixes_no_trailing_slash(self) -> None:
        """``SANDBOX_API_PREFIX`` / ``RUNS_BLUEPRINT_PREFIX`` are Flask-style (no trailing slash)."""
        assert _config.SANDBOX_API_PREFIX == "/sandbox/api"
        assert _config.RUNS_BLUEPRINT_PREFIX == "/runs"

    def test_cfg_keys_share_pheno_namespace(self) -> None:
        """All ``CFG_*`` Flask-config keys use the ``pheno_`` namespace
        (or are documented standalone keys)."""
        # Most live in the pheno_ namespace.
        for name in ("CFG_URL_PREFIX", "CFG_OPERATION_REGISTRY",
                     "CFG_RUN_REGISTRY", "CFG_RUNNER",
                     "CFG_IMAGE_ROOT", "CFG_SANDBOX_ROOT"):
            value: str = getattr(_config, name)
            assert value.startswith("pheno_"), f"{name}={value!r} should start with pheno_"
        # The two viewer-internal keys keep their historical names.
        assert _config.CFG_OUTPUT_ROOT == "output_root"
        assert _config.CFG_FILTERED_STATE == "filtered_state"

    def test_registry_keys_are_distinct(self) -> None:
        """Each Registry type has its own Flask config key (no overload)."""
        assert _config.CFG_OPERATION_REGISTRY != _config.CFG_RUN_REGISTRY
        assert _config.CFG_OPERATION_REGISTRY == "pheno_operation_registry"
        assert _config.CFG_RUN_REGISTRY == "pheno_run_registry"

    def test_titles_are_distinct_and_human_readable(self) -> None:
        titles = {_config.TITLE_HUB, _config.TITLE_BUILDER,
                  _config.TITLE_VIEWER, _config.TITLE_RUN}
        assert len(titles) == 4
        for title in titles:
            assert title.startswith("PhenoTypic ")

    def test_sandbox_subdirectories(self) -> None:
        assert _config.SANDBOX_GUI_DIRNAME == ".phenotypic-gui"
        assert _config.RUN_LOG_DIRNAME == ".gui_log"
        assert _config.VIEWER_CACHE_DIRNAME == ".viewer_cache"

    def test_module_does_not_import_dash(self) -> None:
        """``_config`` must be cheap to import everywhere (no Dash/Flask deps)."""
        import sys

        # Force a fresh import via reimport pathway.
        sys.modules.pop("phenotypic.gui._config", None)
        import phenotypic.gui._config  # noqa: F401

        # After the reimport, the heavy GUI deps must NOT have been pulled in
        # transitively from _config alone. (Other tests in the session may
        # have imported them already, so we check that _config did not
        # *trigger* the import.)
        # We can only assert what _config itself imports at module top.
        import inspect
        src = inspect.getsource(phenotypic.gui._config)
        assert "import dash" not in src
        assert "import flask" not in src
        assert "import werkzeug" not in src


class TestAddLauncherArgs:
    """``add_launcher_args`` adds a consistent --host/--port/--debug block."""

    def test_adds_three_flags_with_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        _config.add_launcher_args(parser)
        ns = parser.parse_args([])
        assert ns.host == _config.DEFAULT_HOST
        assert ns.port == _config.DEFAULT_PORT
        assert ns.debug is False

    def test_user_overrides_take_effect(self) -> None:
        parser = argparse.ArgumentParser()
        _config.add_launcher_args(parser)
        ns = parser.parse_args(["--host", "0.0.0.0", "--port", "9000", "--debug"])
        assert ns.host == "0.0.0.0"
        assert ns.port == 9000
        assert ns.debug is True

    def test_include_debug_false_omits_flag(self) -> None:
        parser = argparse.ArgumentParser()
        _config.add_launcher_args(parser, include_debug=False)
        # --debug should now be unknown -> SystemExit on parse.
        with pytest.raises(SystemExit):
            parser.parse_args(["--debug"])
        ns = parser.parse_args([])
        assert ns.host == _config.DEFAULT_HOST
        assert ns.port == _config.DEFAULT_PORT


class TestPrintLauncherBanner:
    """``print_launcher_banner`` writes consistent stdout."""

    def test_banner_contains_title_url_and_tunnel_hint(self, capsys) -> None:
        from pathlib import Path

        root = Path("/tmp/sandbox")
        _config.print_launcher_banner(
            title=_config.TITLE_HUB,
            host=_config.DEFAULT_HOST,
            port=_config.DEFAULT_PORT,
            root=root,
        )
        out = capsys.readouterr().out
        assert _config.TITLE_HUB in out
        assert "http://127.0.0.1:8050/" in out
        assert "ssh -L 8050:localhost:8050" in out
        # ``Path.__str__`` is platform-native (``/tmp/sandbox`` on POSIX,
        # ``\tmp\sandbox`` on Windows) and the banner echoes it verbatim.
        assert str(root) in out

    def test_extra_lines_appear_indented(self, capsys) -> None:
        from pathlib import Path

        _config.print_launcher_banner(
            title=_config.TITLE_VIEWER,
            host="127.0.0.1",
            port=8050,
            root=Path("/tmp/output"),
            extra_lines=("Clear tile cache    : rm -rf /tmp/output/.viewer_cache",),
        )
        out = capsys.readouterr().out
        assert "Clear tile cache" in out
        assert "/tmp/output/.viewer_cache" in out


# ---------------------------------------------------------------------------
# _design public surface
# ---------------------------------------------------------------------------


class TestDesignTokens:
    """The design-token constants are the single source of truth for
    every ``--color-*`` / ``--text-*`` / etc. CSS variable."""

    def test_brand_colors_are_documented_hex(self) -> None:
        assert _design.COLOR_NAVY == "#003660"
        assert _design.COLOR_BLUE == "#1b75bc"
        assert _design.COLOR_GOLD == "#febc11"
        assert _design.COLOR_HEADING == _design.COLOR_NAVY

    def test_okabe_ito_palette_is_distinct(self) -> None:
        oi = {
            _design.OI_ORANGE, _design.OI_SKY, _design.OI_GREEN,
            _design.OI_VERMILION, _design.OI_BLUE, _design.OI_PURPLE,
            _design.OI_YELLOW, _design.OI_GREY,
        }
        # Eight distinct values per the Okabe-Ito spec.
        assert len(oi) == 8

    def test_type_scale_is_monotonic(self) -> None:
        """The shared type scale (xs..3xl) must be increasing in rem."""
        sizes = [
            _design.TEXT_XS, _design.TEXT_SM, _design.TEXT_BASE,
            _design.TEXT_MD, _design.TEXT_LG, _design.TEXT_XL,
            _design.TEXT_2XL, _design.TEXT_3XL,
        ]
        floats = [float(s.removesuffix("rem")) for s in sizes]
        assert floats == sorted(floats)
        # Strictly increasing (no duplicates).
        assert len(set(floats)) == len(floats)

    def test_semantic_font_size_aliases_resolve_to_primitives(self) -> None:
        """Each ``FONT_SIZE_*`` alias must point at one of the ``TEXT_*`` rem
        primitives. Catches a future rename of a primitive that forgets to
        update the alias."""
        primitives = {
            _design.TEXT_XS, _design.TEXT_SM, _design.TEXT_BASE,
            _design.TEXT_MD, _design.TEXT_LG, _design.TEXT_XL,
            _design.TEXT_2XL, _design.TEXT_3XL,
        }
        aliases = {
            "FONT_SIZE_DISPLAY": _design.FONT_SIZE_DISPLAY,
            "FONT_SIZE_TITLE": _design.FONT_SIZE_TITLE,
            "FONT_SIZE_HEADER_1": _design.FONT_SIZE_HEADER_1,
            "FONT_SIZE_HEADER_2": _design.FONT_SIZE_HEADER_2,
            "FONT_SIZE_BODY_LG": _design.FONT_SIZE_BODY_LG,
            "FONT_SIZE_BODY": _design.FONT_SIZE_BODY,
            "FONT_SIZE_LABEL": _design.FONT_SIZE_LABEL,
            "FONT_SIZE_CAPTION": _design.FONT_SIZE_CAPTION,
        }
        for name, value in aliases.items():
            assert value in primitives, (
                f"{name}={value!r} must equal one of TEXT_XS..TEXT_3XL"
            )

    def test_semantic_font_size_aliases_cover_full_scale(self) -> None:
        """The eight semantic aliases collectively cover all eight rem
        primitives — no gaps and no two aliases mapping to the same size."""
        alias_values = {
            _design.FONT_SIZE_DISPLAY, _design.FONT_SIZE_TITLE,
            _design.FONT_SIZE_HEADER_1, _design.FONT_SIZE_HEADER_2,
            _design.FONT_SIZE_BODY_LG, _design.FONT_SIZE_BODY,
            _design.FONT_SIZE_LABEL, _design.FONT_SIZE_CAPTION,
        }
        primitives = {
            _design.TEXT_XS, _design.TEXT_SM, _design.TEXT_BASE,
            _design.TEXT_MD, _design.TEXT_LG, _design.TEXT_XL,
            _design.TEXT_2XL, _design.TEXT_3XL,
        }
        assert alias_values == primitives

    def test_font_family_constants_carry_active_google_font(self) -> None:
        """The Python-side ``FONT_FAMILY_*`` strings must lead with the
        active Google Font and end with a generic CSS family fallback."""
        assert _design.FONT_FAMILY_DISPLAY.startswith("'Roboto'")
        assert _design.FONT_FAMILY_BODY.startswith("'Roboto'")
        assert _design.FONT_FAMILY_MONO.startswith("'Roboto'")
        assert _design.FONT_FAMILY_DISPLAY.rstrip().endswith("serif")
        assert _design.FONT_FAMILY_BODY.rstrip().endswith("sans-serif")
        assert _design.FONT_FAMILY_MONO.rstrip().endswith("monospace")

    def test_spacing_grid_matches_8pt_system(self) -> None:
        """``SPACING_*`` constants line up with the 8 pt grid."""
        # 0.25 / 0.5 / 0.75 / 1 / 1.25 / 1.5 / 2 rem
        expected = ["0.25rem", "0.5rem", "0.75rem", "1rem",
                    "1.25rem", "1.5rem", "2rem"]
        actual = [_design.SPACING_1, _design.SPACING_2, _design.SPACING_3,
                  _design.SPACING_4, _design.SPACING_5, _design.SPACING_6,
                  _design.SPACING_8]
        assert actual == expected


class TestInjectDesignTokens:
    """``inject_design_tokens`` splices a single self-contained token block."""

    def _make_app(self):
        """Build a minimal Dash app for injection tests."""
        import dash

        return dash.Dash(__name__, suppress_callback_exceptions=True)

    def test_all_token_groups_appear_in_index_string(self) -> None:
        app = self._make_app()
        _design.inject_design_tokens(app)
        idx = app.index_string

        # Marker comment present.
        assert "phenotypic-design-tokens" in idx

        # Every token group has at least one representative variable.
        assert "--font-display:" in idx
        assert "--color-navy:" in idx and _design.COLOR_NAVY in idx
        assert "--oi-purple:" in idx and _design.OI_PURPLE in idx
        assert "--text-base:" in idx and _design.TEXT_BASE in idx
        assert "--text-2xl:" in idx and _design.TEXT_2XL in idx
        assert "--text-3xl:" in idx and _design.TEXT_3XL in idx
        # Semantic font-size aliases — preferred call form going forward.
        assert "--font-size-display:" in idx
        assert "--font-size-title:" in idx
        assert "--font-size-header-1:" in idx
        assert "--font-size-header-2:" in idx
        assert "--font-size-body-lg:" in idx
        assert "--font-size-body:" in idx
        assert "--font-size-label:" in idx
        assert "--font-size-caption:" in idx
        assert "--sp-3:" in idx and _design.SPACING_3 in idx
        assert "--radius:" in idx and _design.RADIUS in idx
        assert "--shadow-sm:" in idx and _design.SHADOW_SM in idx
        assert "--ease-out:" in idx and _design.EASE_OUT in idx
        assert "--transition:" in idx

    def test_inject_is_idempotent(self) -> None:
        """Multiple calls do not duplicate the style block."""
        app = self._make_app()
        _design.inject_design_tokens(app)
        first = app.index_string
        _design.inject_design_tokens(app)
        _design.inject_design_tokens(app)
        # Marker appears exactly once.
        assert app.index_string.count("phenotypic-design-tokens") == 1
        # And the index string is unchanged after re-injection.
        assert app.index_string == first

    def test_inject_lands_after_dash_css_placeholder(self) -> None:
        """The token block follows ``{%css%}`` so it overrides Dash defaults."""
        app = self._make_app()
        _design.inject_design_tokens(app)
        idx = app.index_string
        # The placeholder still exists and the marker appears AFTER it.
        css_pos = idx.find("{%css%}")
        marker_pos = idx.find("phenotypic-design-tokens")
        assert css_pos != -1
        assert marker_pos != -1
        assert marker_pos > css_pos
