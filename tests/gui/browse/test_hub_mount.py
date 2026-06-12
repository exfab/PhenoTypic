from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._sandbox import SandboxRoot


def test_browse_tab_in_nav_model():
    from phenotypic.gui.shell._ids import SHELL_TAB_BROWSE, SHELL_TAB_HOME
    from phenotypic.gui.shell._layout import NAV_MODEL

    # Browse is a leaf immediately after Home.
    assert NAV_MODEL[0] == SHELL_TAB_HOME
    assert NAV_MODEL[1] == SHELL_TAB_BROWSE


def test_hub_serves_browse_mount(tmp_path):
    (tmp_path / "imgs").mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = app.server.test_client()
    resp = client.get("/browse/")
    assert resp.status_code == 200
    assert b"PhenoTypic" in resp.data
