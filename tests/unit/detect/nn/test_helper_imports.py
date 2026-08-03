"""Import-boundary tests for private neural-network helper modules."""

from __future__ import annotations

import importlib.util
from types import ModuleType

from phenotypic.detect.nn import _helper


EXPECTED_HELPER_MODULES = [
    "_checkpoint_manager",
    "_dino_support",
    "_sam2_rle",
    "_tiling",
]


def test_helper_package_explicitly_imports_its_modules() -> None:
    """Static analyzers can discover every helper module as a real attribute."""
    assert _helper.__all__ == EXPECTED_HELPER_MODULES
    assert "__getattr__" not in vars(_helper)

    for name in EXPECTED_HELPER_MODULES:
        module = getattr(_helper, name)
        assert isinstance(module, ModuleType)
        assert module.__name__ == f"phenotypic.detect.nn._helper.{name}"


def test_helper_modules_are_not_available_at_nn_root() -> None:
    """The old private root module paths are intentionally unsupported."""
    for name in EXPECTED_HELPER_MODULES:
        assert importlib.util.find_spec(f"phenotypic.detect.nn.{name}") is None
