from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from phenotypic.tools_.typing_ import OperationField, polymorphic_field


# --- A live, non-operation pydantic base to prove base-parameterization ---
class _Animal(BaseModel):
    name: str = "?"


class _Dog(_Animal):
    legs: int = 4


_AnimalField = polymorphic_field(base=_Animal)


class _AnimalHost(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pet: _AnimalField  # type: ignore[valid-type]


def test_guard_accepts_base_subclass_instance():
    # A live instance of the declared base (passed through, not dict-deserialized)
    host = _AnimalHost(pet=_Dog(name="rex"))
    assert isinstance(host.pet, _Dog)
    assert host.pet.name == "rex"


def test_guard_rejects_non_base_instance():
    class _Rock(BaseModel):
        pass

    with pytest.raises(ValidationError):
        _AnimalHost(pet=_Rock())


# --- OperationField back-compat: it must still round-trip a real operation ---
class _OpHost(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    op: OperationField  # type: ignore[valid-type]


def test_operationfield_roundtrips_real_operation():
    from phenotypic.detect import OtsuDetector

    host = _OpHost(op=OtsuDetector(ignore_zeros=True))
    dumped = host.model_dump(mode="json")
    assert dumped["op"]["class"] == "OtsuDetector"

    restored = _OpHost.model_validate(dumped)
    assert type(restored.op).__name__ == "OtsuDetector"
    assert restored.op.ignore_zeros is True


def test_operationfield_keeps_gui_marker():
    # The GUI OperationRegistry detects operation params via _OperationFieldMarker
    # in the Annotated chain. OperationField must keep it after the refactor.
    from phenotypic.tools_.typing_ import _OperationFieldMarker

    meta = OperationField.__metadata__
    assert any(isinstance(m, _OperationFieldMarker) for m in meta)
