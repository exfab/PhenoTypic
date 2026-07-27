"""Unit tests for Builder source-image-root browse seeding."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import source_payload_from_path


def test_builder_browse_seed_uses_valid_shared_source(tmp_path: Path) -> None:
    from phenotypic.gui.builder._callbacks import _browse_seed_from_source

    plates = tmp_path / "plates"
    plates.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, plates, source="manual")

    assert _browse_seed_from_source(tmp_path, payload) == str(plates.resolve())


def test_builder_browse_seed_falls_back_to_image_root_when_unset(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.builder._callbacks import _browse_seed_from_source

    assert _browse_seed_from_source(tmp_path, None) == str(tmp_path.resolve())


def test_builder_browse_seed_rejects_outside_image_root(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.builder._callbacks import _browse_seed_from_source

    image_root = tmp_path / "sandbox"
    image_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = {
        "abs_path": str(outside.resolve()),
        "rel_path": "../outside",
        "label": "outside",
        "image_count": None,
        "source": "manual",
        "validated": True,
        "version": 1,
    }

    assert _browse_seed_from_source(image_root, payload) == str(
        image_root.resolve()
    )


def test_builder_browse_seed_accepts_v1_compatibility_payload(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.builder._callbacks import _browse_seed_from_source

    plates = tmp_path / "plates"
    plates.mkdir()
    payload = {
        "abs_path": str(plates.resolve()),
        "rel_path": "plates",
        "label": "plates",
        "image_count": None,
        "source": "manual",
        "validated": False,
        "version": 1,
    }

    assert _browse_seed_from_source(tmp_path, payload) == str(plates.resolve())


def test_builder_browse_seed_rejects_v2_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.builder._callbacks import _browse_seed_from_source

    image_root = tmp_path / "active"
    image_root.mkdir()
    plates = image_root / "plates"
    plates.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_plates = other_root / "plates"
    other_plates.mkdir()
    payload = source_payload_from_path(
        SandboxRoot.from_path(other_root),
        other_plates,
        source="manual",
    )

    assert _browse_seed_from_source(image_root, payload) == str(
        image_root.resolve()
    )


def test_builder_browse_seed_falls_back_on_resolver_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phenotypic.gui.builder._callbacks import _browse_seed_from_source

    image_root = tmp_path / "sandbox"
    image_root.mkdir()
    bad_source = image_root / "bad-source"
    bad_source.mkdir()
    original_resolve = Path.resolve

    def _resolve_or_raise(path: Path, *args: object, **kwargs: object) -> Path:
        if path.name == "bad-source":
            raise RuntimeError("resolver loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _resolve_or_raise)
    payload = {
        "abs_path": str(bad_source),
        "rel_path": "bad-source",
        "label": "bad-source",
        "image_count": None,
        "source": "manual",
        "validated": True,
        "version": 1,
    }

    assert _browse_seed_from_source(image_root, payload) == str(
        original_resolve(image_root)
    )
