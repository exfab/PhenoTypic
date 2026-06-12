from phenotypic.gui._config import (
    BROWSE_CACHE_TMP_SUBPATH,
    BROWSE_TILES_PREFIX,
    MOUNT_BROWSE,
    TITLE_BROWSE,
)
from phenotypic.gui.shell._ids import SHELL_TAB_BROWSE


def test_browse_mount_and_prefixes():
    assert MOUNT_BROWSE == "/browse/"
    assert BROWSE_TILES_PREFIX == "/tiles"
    assert BROWSE_CACHE_TMP_SUBPATH == ("phenotypic", "browse")
    assert TITLE_BROWSE == "PhenoTypic Source Browser"


def test_shell_tab_browse():
    assert SHELL_TAB_BROWSE == "shell-tab-browse"
