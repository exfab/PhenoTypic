"""Tests for the consolidated ``phenotypic._assets`` package.

Guards the single-source asset layout: the accessor resolves bundled
logos, the canonical app logos are present, and no duplicate logo files
have crept back under ``logos/``.
"""

from __future__ import annotations

import hashlib

from phenotypic._assets import ASSET_DIR, asset_bytes, logos_dir

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestAssetAccessor:
    def test_asset_dir_points_at_assets_package(self):
        assert ASSET_DIR.name == "_assets"
        assert ASSET_DIR.is_dir()

    def test_logos_dir_exists(self):
        assert logos_dir().is_dir()
        assert logos_dir() == ASSET_DIR / "logos"

    def test_cli_dashboard_logo_is_a_png(self):
        raw = asset_bytes("logos/LogoArtOnly.png")
        assert raw[:8] == _PNG_MAGIC

    def test_gui_logo_present(self):
        svg = logos_dir() / "dashboard_logo.svg"
        assert svg.exists()
        assert svg.read_bytes().lstrip()[:4] in (b"<svg", b"<?xm")

    def test_sphinx_exfab_logos_present(self):
        d = logos_dir() / "400x150"
        assert (d / "light_logo_exfab.svg").exists()
        assert (d / "gradient_logo_exfab.svg").exists()

    def test_retired_dashboard_vendor_js_is_absent(self):
        assert not (ASSET_DIR / "vendor" / "plotly.min.js").exists()
        assert not (ASSET_DIR / "vendor" / "hyparquet.min.js").exists()


class TestNoDuplicateLogos:
    def test_no_byte_identical_duplicates_under_logos(self):
        """Single-source invariant: every logo file is unique by content.

        Re-introducing a duplicate copy (the very thing this consolidation
        removed) would make two paths share a SHA-256 and fail here.
        """
        seen: dict[str, str] = {}
        for path in sorted(logos_dir().rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rel = path.relative_to(logos_dir()).as_posix()
            assert digest not in seen, (
                f"Duplicate logo content: {rel} == {seen[digest]}"
            )
            seen[digest] = rel
