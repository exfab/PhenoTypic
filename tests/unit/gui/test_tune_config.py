"""``gui/tune`` config constants — mount prefix + browser title."""
from __future__ import annotations


def test_mount_tune_constant_and_title() -> None:
    """``MOUNT_TUNE`` is the tune sub-app prefix and the title names "Tune"."""
    from phenotypic.gui._config import MOUNT_TUNE, TITLE_TUNE

    assert MOUNT_TUNE == "/tune/"
    assert "Tune" in TITLE_TUNE
