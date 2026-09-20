"""The backend must be the one this platform is supposed to use.

Every refusal test in the contract suite passes vacuously if the backend
degrades to "unavailable" -- a ctypes symbol that fails to bind would turn the
Windows lane green while shipping nothing. This test has no skip for that
reason.
"""

import os

import pytest

from phenotypic.sdk_ import _identity_io

pytestmark = pytest.mark.platform_io


def test_the_expected_backend_is_active_on_this_platform() -> None:
    """Fails loudly if a ctypes symbol failed to bind at import.

    This works only because the Windows backend binds its API at import time
    and sets ``SUPPORTED`` from the result. Bound lazily per call, this test
    would pass while every call raised.
    """
    expected = {"posix": "posix", "nt": "windows"}[os.name]
    assert _identity_io.active_backend_name() == expected
    assert _identity_io.identity_io_available() is True
