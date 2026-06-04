from __future__ import annotations

from phenotypic._core._pipeline_parts._serializable_pipeline import (
    SerializablePipeline,
)


def test_registry_searches_phenotypic_tune(monkeypatch):
    import phenotypic.tune as tune_pkg

    class _ProbeClass:
        pass

    # Export a class under the phenotypic.tune namespace; the registry must find it.
    monkeypatch.setattr(tune_pkg, "_ProbeClass", _ProbeClass, raising=False)
    found = SerializablePipeline._find_class_in_phenotypic("_ProbeClass")
    assert found is _ProbeClass


def test_tune_package_imports():
    import phenotypic.tune  # noqa: F401
