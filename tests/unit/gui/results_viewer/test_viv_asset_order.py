"""On-disk facts about the vendored Viv assets, checkable without a browser.

Dash serves the Viv facade BEFORE the bundle it reads.

This is a property of Dash's asset walk, not of either file: `_assets/` is
walked with ``for current, _, files in sorted(os.walk(walk_dir))``, which
appends every root-level asset before any subdirectory asset. So
``viv_viewer.js`` (root) is emitted ahead of ``viv/viv-bundle.min.js``
(subdirectory), and ``window.__vivBundle`` does not exist while the facade
executes.

The facade handles this by deferring its bundle lookup to the first
``ready()`` call --- pinned behaviourally by
``tests/e2e/gui/test_viv_codec_reads_a_real_store.py::
test_the_facade_survives_loading_before_the_bundle``. This test pins the
*premise*: it renders a real Dash index over the real assets folder and
reads the script order out of the HTML, so a change to that order shows up
here rather than as a runtime failure in a browser nobody ran.
"""

from __future__ import annotations

import re
from pathlib import Path

import dash

import phenotypic.gui.results_viewer as results_viewer

REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS = Path(results_viewer.__file__).parent / "_assets"
BUNDLE_FILE = ASSETS / "viv" / "viv-bundle.min.js"
RECIPE_VERSION = REPO_ROOT / "tools" / "viv-bundle" / "VERSION"

FACADE = "/assets/viv_viewer.js"
BUNDLE = "/assets/viv/viv-bundle.min.js"


def _asset_script_order() -> list[str]:
    """Script ``src`` paths, in emission order, from a rendered Dash index."""
    app = dash.Dash("viv-asset-order-probe", assets_folder=str(ASSETS))
    # Dash 4 validates the layout in a before_request hook; a trivial layout
    # keeps that from 500-ing before the index is rendered.
    app.layout = dash.html.Div()
    # ``init_app`` normally does this; calling it directly keeps the probe
    # off the network and off a real server.
    app._walk_assets_directory()
    with app.server.test_request_context("/"):
        html = app.index()
    return [
        src.split("?", 1)[0]
        for src in re.findall(r'src="([^"]+)"', html)
        if "/assets/" in src
    ]


def test_the_facade_is_served_before_the_bundle() -> None:
    """Root-level assets sort ahead of subdirectory ones."""
    order = _asset_script_order()
    assert FACADE in order, order
    assert BUNDLE in order, order
    assert order.index(FACADE) < order.index(BUNDLE), (
        "Dash no longer serves viv_viewer.js before viv/viv-bundle.min.js "
        f"(order: {order}). The facade's deferred ready() exists ONLY "
        "because of this order -- if the order changed deliberately, update "
        "the comment in viv_viewer.js rather than deleting it, because the "
        "deferred form is still correct under either order."
    )


def test_the_bundle_is_served_on_every_results_viewer_page() -> None:
    """The 2.5 MiB artifact loads unconditionally, as OpenSeadragon does.

    Recorded rather than fixed: the deployment is localhost or an SSH tunnel
    (plan Global Constraints, accepted cost). If a later phase adds deferred
    loading this test is the one that says so out loud.
    """
    order = _asset_script_order()
    assert "/assets/openseadragon/openseadragon.min.js" in order
    assert BUNDLE in order


def test_the_committed_bundle_matches_its_recorded_version() -> None:
    """The vendored artifact and its recipe agree about what it is.

    There is no npm in CI by design, so nothing rebuilds the bundle to
    prove the two are in sync -- ``tools/viv-bundle/VERSION`` and the string
    ``build.mjs`` stamps into the artifact are the whole provenance of a
    committed binary. Rebuilding without bumping ``VERSION``, or bumping
    ``VERSION`` without rebuilding, is otherwise invisible until a browser
    behaves oddly.

    Reads only the banner, not the whole 2.5 MiB file.
    """
    recorded = RECIPE_VERSION.read_text(encoding="utf-8").strip()
    banner = BUNDLE_FILE.read_text(encoding="utf-8", errors="replace")[:512]
    assert recorded in banner, (
        f"tools/viv-bundle/VERSION says {recorded!r}, which does not appear "
        f"in the committed artifact's banner: {banner.splitlines()[0]!r}. "
        "Rebuild with `cd tools/viv-bundle && npm ci && node build.mjs`, or "
        "correct VERSION -- whichever is actually stale."
    )
