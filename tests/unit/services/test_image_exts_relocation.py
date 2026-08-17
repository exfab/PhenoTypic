"""``IMAGE_EXTS`` must live below the GUI so ``classify()`` stays Dash-free."""

from __future__ import annotations


def test_image_exts_lives_in_sdk():
    from phenotypic.sdk_._io_constants import IMAGE_EXTS

    assert isinstance(IMAGE_EXTS, frozenset)
    assert ".tif" in IMAGE_EXTS


def test_every_alias_is_the_same_object():
    """Three import paths, one object — a copy would drift silently."""
    from phenotypic.gui._config import IMAGE_EXTS as via_config
    from phenotypic.gui.builder._directory_browser import IMAGE_EXTS as via_browser
    from phenotypic.sdk_._io_constants import IMAGE_EXTS as canonical

    assert via_config is canonical
    assert via_browser is canonical


def test_classifier_does_not_reach_through_the_dash_module():
    """The whole point: classify() must not pull in _directory_browser."""
    import inspect

    from phenotypic.gui.shell import _classifier

    source = inspect.getsource(_classifier)
    assert "_directory_browser" not in source
