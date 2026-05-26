"""Shared pytest utilities for tests that require Kaleido + Chrome.

kaleido >= 1.0 dropped its bundled browser and now requires an external
Chrome/Chromium install (managed by ``choreographer``).  Tests that call
``plotly_figure.write_image()`` therefore need a Chrome check at collection
time so they skip rather than fail on minimal containers.
"""

from __future__ import annotations

import pytest


def _kaleido_chrome_available() -> bool:
    """Return True iff choreographer can locate a Chrome-family executable.

    Uses ``Chromium.find_browser(skip_local=False)`` — the same code path
    kaleido 1.x uses internally — so the probe matches the actual runtime
    requirement.  Falls back to False on any import or runtime error.
    """
    try:
        from choreographer.browsers.chromium import Chromium

        return Chromium.find_browser(skip_local=False) is not None
    except Exception:
        return False


requires_kaleido_chrome = pytest.mark.skipif(
    not _kaleido_chrome_available(),
    reason=(
        "kaleido >= 1 requires Chrome for Plotly PNG export; "
        "install Chrome or run `plotly_get_chrome`"
    ),
)
