"""Unit tests for the shared GUI source-image-root context."""
from __future__ import annotations

from pathlib import Path

from phenotypic.gui.shell._sandbox import SandboxRoot


class _ExplodingSandbox:
    """Sandbox test double whose resolver always fails."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve(self, _candidate: object) -> Path:
        raise RuntimeError("resolver loop")


def test_source_payload_from_path_accepts_in_sandbox_directory(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import source_payload_from_path

    source_dir = tmp_path / "plates"
    source_dir.mkdir()
    (source_dir / "plate_001.tif").write_bytes(b"")
    sandbox = SandboxRoot.from_path(tmp_path)

    payload = source_payload_from_path(
        sandbox, source_dir, source="run-console"
    )

    assert payload is not None
    assert payload["abs_path"] == str(source_dir.resolve())
    assert payload["rel_path"] == "plates"
    assert payload["label"] == "plates"
    assert payload["source"] == "run-console"
    assert payload["validated"] is True
    assert payload["version"] == 1
    assert payload["image_count"] == 1


def test_source_payload_from_path_rejects_out_of_sandbox_path(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import source_payload_from_path

    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox = SandboxRoot.from_path(sandbox_root)

    assert (
        source_payload_from_path(sandbox, outside, source="manual") is None
    )


def test_source_payload_from_path_rejects_resolver_errors(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import source_payload_from_path

    sandbox = _ExplodingSandbox(tmp_path)

    assert source_payload_from_path(sandbox, "loop", source="manual") is None


def test_source_payload_from_path_rejects_files_and_missing_paths(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import source_payload_from_path

    sandbox = SandboxRoot.from_path(tmp_path)
    image_file = tmp_path / "plate.tif"
    image_file.write_bytes(b"")

    assert (
        source_payload_from_path(sandbox, image_file, source="builder") is None
    )
    assert (
        source_payload_from_path(
            sandbox, tmp_path / "missing", source="builder"
        )
        is None
    )


def test_resolve_source_image_root_rejects_malformed_payloads(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import resolve_source_image_root

    sandbox = SandboxRoot.from_path(tmp_path)

    assert resolve_source_image_root(sandbox, None) is None
    assert resolve_source_image_root(sandbox, "not-json") is None
    assert resolve_source_image_root(sandbox, {"abs_path": 123}) is None
    assert (
        resolve_source_image_root(
            sandbox,
            {
                "version": 999,
                "abs_path": str(tmp_path),
                "validated": True,
            },
        )
        is None
    )


def test_resolve_source_image_root_rejects_resolver_errors(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import resolve_source_image_root

    sandbox = _ExplodingSandbox(tmp_path)
    payload = {
        "abs_path": str(tmp_path / "loop"),
        "rel_path": "loop",
        "label": "loop",
        "image_count": None,
        "source": "manual",
        "validated": True,
        "version": 1,
    }

    assert resolve_source_image_root(sandbox, payload) is None


def test_resolve_source_image_root_returns_valid_directory(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import (
        resolve_source_image_root,
        source_payload_from_path,
    )

    source_dir = tmp_path / "plates"
    source_dir.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, source_dir, source="tune")

    assert payload is not None
    assert resolve_source_image_root(sandbox, payload) == source_dir.resolve()


def test_source_label_formats_unset_invalid_and_valid_payloads(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import (
        source_label,
        source_payload_from_path,
    )

    source_dir = tmp_path / "batch-a"
    source_dir.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, source_dir, source="unknown")

    assert source_label(None) == "source: unset"
    assert source_label({"abs_path": 1}) == "source: invalid"
    assert payload is not None
    assert source_label(payload) == "source: batch-a"
