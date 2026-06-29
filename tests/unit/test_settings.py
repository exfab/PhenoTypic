from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from phenotypic.sdk_.exceptions_ import OperationIntegrityError
from phenotypic.sdk_.funcs_ import validate_measure_integrity


def test_root_settings_module_is_public_runtime_surface() -> None:
    """The public settings module lives at ``phenotypic.settings``."""
    settings = importlib.import_module("phenotypic.settings")

    assert hasattr(settings, "VALIDATE_OPS")
    assert hasattr(settings, "set_validate_ops")
    assert hasattr(settings, "validation")


def test_legacy_settings_underscore_module_is_removed() -> None:
    """``phenotypic.settings_`` is hard-removed, not kept as a shim."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic.settings_")


def test_validate_ops_is_read_live_after_funcs_import() -> None:
    """Changing ``settings.VALIDATE_OPS`` after import affects validators."""
    settings = importlib.import_module("phenotypic.settings")

    @validate_measure_integrity("sample.data")
    def mutate_sample(sample: SimpleNamespace) -> None:
        sample.data[0] = 2

    previous = settings.VALIDATE_OPS
    try:
        settings.set_validate_ops(False)
        sample = SimpleNamespace(data=np.array([1], dtype=np.uint8))
        mutate_sample(sample)

        settings.set_validate_ops(True)
        sample = SimpleNamespace(data=np.array([1], dtype=np.uint8))
        with pytest.raises(OperationIntegrityError):
            mutate_sample(sample)
    finally:
        settings.set_validate_ops(previous)


def test_validation_context_manager_restores_previous_value() -> None:
    """Temporary validation changes do not leak out of the context."""
    settings = importlib.import_module("phenotypic.settings")

    previous = settings.VALIDATE_OPS
    try:
        settings.set_validate_ops(False)
        with settings.validation(True):
            assert settings.VALIDATE_OPS is True
        assert settings.VALIDATE_OPS is False
    finally:
        settings.set_validate_ops(previous)
