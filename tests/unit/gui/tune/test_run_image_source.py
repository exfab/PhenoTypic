from pathlib import Path

from phenotypic.gui.shell._source_context import source_payload_from_path
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.tune._run_image_source import resolve_run_images


def test_override_wins_when_inside_sandbox(tmp_path: Path):
    sandbox = SandboxRoot.from_path(tmp_path)
    shared = tmp_path / "shared"
    override = tmp_path / "override"
    shared.mkdir()
    override.mkdir()

    payload = source_payload_from_path(sandbox, shared, source="test")

    assert resolve_run_images(sandbox, payload, str(override)) == str(override)


def test_relative_override_resolves_inside_sandbox(tmp_path: Path):
    sandbox = SandboxRoot.from_path(tmp_path)
    images = tmp_path / "images"
    images.mkdir()

    assert resolve_run_images(sandbox, None, "images") == str(images)


def test_shared_source_used_when_override_empty(tmp_path: Path):
    sandbox = SandboxRoot.from_path(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    payload = source_payload_from_path(sandbox, shared, source="test")

    assert resolve_run_images(sandbox, payload, "") == str(shared)


def test_out_of_sandbox_override_is_rejected(tmp_path: Path):
    root = tmp_path / "sandbox"
    root.mkdir()
    sandbox = SandboxRoot.from_path(root)
    outside = tmp_path / "outside"
    outside.mkdir()

    assert resolve_run_images(sandbox, None, str(outside)) is None
