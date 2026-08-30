"""The builder renders through the ONE vendored Viv artifact, not a copy.

Committing a second ~2.6 MB bundle beside the builder's own assets would put
two artifacts under one build recipe (``tools/viv-bundle/``), and with no npm
in CI nothing would notice them drifting apart. The blueprint in
``gui/_shared/_viv_assets.py`` is what makes one copy reachable from two Dash
apps; these tests pin that there is one, that it is served, and that the
builder never talks to ``window.__vivBundle`` directly.
"""

from __future__ import annotations

from pathlib import Path

import dash
import pytest

from phenotypic.gui import builder as builder_pkg
from phenotypic.gui._shared import (
    VIV_BUNDLE_PATH,
    VIV_FACADE_PATH,
    register_viv_assets,
    viv_script_urls,
)

BUILDER_DIR = Path(builder_pkg.__file__).parent


@pytest.fixture
def client():
    app = dash.Dash("viv-assets")
    app.layout = dash.html.Div()
    register_viv_assets(app.server)
    return app.server.test_client()


def test_builder_does_not_carry_its_own_bundle_copy() -> None:
    stray = list((BUILDER_DIR / "assets").rglob("viv-bundle*.js"))
    assert not stray, f"builder has its own bundle copy: {stray}"


def test_builder_does_not_carry_its_own_facade_copy() -> None:
    """The facade is the file whose behaviour every surface depends on.

    A second copy of it would be worse than a second bundle: the bundle is
    inert data, while a stale facade silently changes how a surface renders.
    """
    stray = list((BUILDER_DIR / "assets").rglob("viv_viewer*.js"))
    assert not stray, f"builder has its own facade copy: {stray}"


@pytest.mark.parametrize("path", [VIV_BUNDLE_PATH, VIV_FACADE_PATH])
def test_the_blueprint_serves_the_vendored_file(client, path: str) -> None:
    resp = client.get(f"/{path}")
    assert resp.status_code == 200
    assert resp.data


def test_the_blueprint_is_not_a_passthrough_to_the_asset_tree(client) -> None:
    """An allow-list of two files, not ``<path:filename>``.

    A passthrough would expose the results viewer's whole ``_assets/`` tree
    from an unrelated sub-app for no gain.
    """
    assert client.get("/_viv/results_viewer.js").status_code == 404


def test_registering_twice_is_a_no_op() -> None:
    """Both the sub-app factory and a future composer may call it."""
    app = dash.Dash("viv-assets-idempotent")
    app.layout = dash.html.Div()
    register_viv_assets(app.server)
    register_viv_assets(app.server)  # must not raise a name collision


def test_the_builder_app_links_both_scripts(tmp_path: Path) -> None:
    from phenotypic.gui.builder._app import create_app

    app = create_app(image_root=tmp_path, url_prefix="/builder/")
    assert app.config.external_scripts == viv_script_urls("/builder/")


def test_script_urls_carry_the_mount_prefix() -> None:
    assert viv_script_urls("/builder/") == [
        f"/builder/{VIV_BUNDLE_PATH}",
        f"/builder/{VIV_FACADE_PATH}",
    ]
    assert viv_script_urls("/") == [f"/{VIV_BUNDLE_PATH}", f"/{VIV_FACADE_PATH}"]


def test_nothing_in_the_builder_touches_the_bundle_directly() -> None:
    """Only ``viv_viewer.js`` may read ``window.__vivBundle``.

    That indirection is what lets the vendored artifact be replaced without
    editing a surface, so a direct read anywhere else is the leak that makes
    the facade optional.
    """
    offenders = [
        path
        for path in BUILDER_DIR.rglob("*")
        if path.suffix in {".py", ".js"}
        and "__vivBundle" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders, offenders
