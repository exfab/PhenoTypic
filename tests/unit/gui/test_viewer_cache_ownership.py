"""Cache ownership contracts for shell and standalone Results launchers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from phenotypic.gui.results_viewer import _output_root


@pytest.mark.parametrize(
    ("platform", "environment_name"),
    [
        ("linux", "XDG_CACHE_HOME"),
        ("win32", "LOCALAPPDATA"),
    ],
)
def test_user_viewer_cache_root_uses_platform_cache_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    environment_name: str,
) -> None:
    """Standalone cache ownership is deterministic and source-independent."""
    owner = tmp_path / f"{platform}-cache"
    monkeypatch.setattr(_output_root.sys, "platform", platform)
    monkeypatch.setenv(environment_name, str(owner))

    cache_root = _output_root.user_viewer_cache_root()

    assert cache_root == owner / "phenotypic" / "gui" / "viewer_cache"
    assert not cache_root.exists()


def test_macos_user_viewer_cache_root_uses_library_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS follows the standard per-user Library/Caches convention."""
    monkeypatch.setattr(_output_root.sys, "platform", "darwin")

    assert _output_root.user_viewer_cache_root() == (
        Path.home()
        / "Library"
        / "Caches"
        / "phenotypic"
        / "gui"
        / "viewer_cache"
    )


def test_results_launcher_passes_external_owner_and_reports_real_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standalone Results launcher neither infers nor reports source cache."""
    from phenotypic.gui.results_viewer import __main__ as launcher

    root = tmp_path / "output"
    cache_owner = tmp_path / "user-cache"
    source_cache = cache_owner / "source-key"
    discovered = SimpleNamespace(viewer_cache_dir=source_cache)
    discover_calls: list[tuple[Path, Path]] = []
    banner_calls: list[dict[str, object]] = []
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        launcher,
        "user_viewer_cache_root",
        lambda: cache_owner,
    )
    monkeypatch.setattr(
        launcher.OutputRoot,
        "discover",
        lambda path, *, cache_root: (
            discover_calls.append((path, cache_root)) or discovered
        ),
    )
    monkeypatch.setattr(
        launcher,
        "create_app",
        lambda output, *, url_prefix: SimpleNamespace(
            run=lambda **kwargs: run_calls.append(kwargs)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "print_launcher_banner",
        lambda **kwargs: banner_calls.append(kwargs),
    )

    launcher.launch_results_viewer(root)

    assert discover_calls == [(root.resolve(), cache_owner)]
    assert banner_calls[0]["extra_lines"] == (
        f"Clear tile cache    : rm -rf {source_cache}",
    )
    assert len(run_calls) == 1


def test_analysis_launcher_passes_per_user_cache_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone Analysis uses the same explicit external owner contract."""
    from phenotypic.gui.analysis import __main__ as launcher

    root = tmp_path / "output"
    cache_owner = tmp_path / "user-cache"
    discovered = object()
    discover_calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        launcher,
        "user_viewer_cache_root",
        lambda: cache_owner,
    )
    monkeypatch.setattr(
        launcher.OutputRoot,
        "discover",
        lambda path, *, cache_root: (
            discover_calls.append((path, cache_root)) or discovered
        ),
    )
    monkeypatch.setattr(
        launcher,
        "create_app",
        lambda **kwargs: SimpleNamespace(run=lambda **_kwargs: None),
    )
    monkeypatch.setattr(launcher, "configure_launcher_logging", lambda **_: None)
    monkeypatch.setattr(launcher, "print_launcher_banner", lambda **_: None)

    assert launcher.main(["--root", str(root)]) == 0
    assert discover_calls == [(root, cache_owner)]
