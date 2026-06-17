"""Gated SAM3 + ungated DINOv2 checkpoint managers (Spec 2a, Task 3).

These tests run WITHOUT ``huggingface_hub`` installed: the manager lazy-imports
it inside the module-level :func:`snapshot_download` indirection, which the tests
patch. The license-acceptance gate fires before any network call.
"""

import pytest


def test_sam3_manager_requires_acceptance(monkeypatch):
    from phenotypic.detect.nn._checkpoint_manager import Sam3CheckpointManager

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    mgr = Sam3CheckpointManager()
    with pytest.raises(RuntimeError, match="license"):
        mgr.download(interactive=False)  # acceptance gate fires before any network


def test_sam3_manager_accepts_then_downloads(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "sam3")
    calls: dict = {}
    monkeypatch.setattr(
        cm, "snapshot_download",
        lambda **kw: calls.update(kw) or "/fake/cache/sam3",
    )
    path = cm.Sam3CheckpointManager().download(interactive=False)
    assert calls["repo_id"] == "facebook/sam3"
    assert path == "/fake/cache/sam3"


def test_gated_download_401_is_actionable(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "sam3")

    def _raise(**kw):
        raise cm._GatedRepoError("403")

    monkeypatch.setattr(cm, "snapshot_download", _raise)
    with pytest.raises(RuntimeError, match="Request access"):
        cm.Sam3CheckpointManager().download(interactive=False)


def test_dinov3_manager_requires_acceptance(monkeypatch):
    from phenotypic.detect.nn._checkpoint_manager import Dinov3CheckpointManager

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    with pytest.raises(RuntimeError, match="license"):
        Dinov3CheckpointManager(size="base").download(interactive=False)


def test_dinov3_manager_accepts_then_downloads(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "dinov3")
    calls: dict = {}
    monkeypatch.setattr(
        cm, "snapshot_download",
        lambda **kw: calls.update(kw) or "/fake/cache/dinov3",
    )
    path = cm.Dinov3CheckpointManager(size="base").download(interactive=False)
    assert calls["repo_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert path == "/fake/cache/dinov3"


def test_dinov3_manager_maps_size_to_repo_id(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "dinov3")
    expected = {
        "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    for size, repo in expected.items():
        calls: dict = {}
        monkeypatch.setattr(
            cm, "snapshot_download",
            lambda **kw: calls.update(kw) or "/fake/cache/dinov3",
        )
        cm.Dinov3CheckpointManager(size=size).download(interactive=False)
        assert calls["repo_id"] == repo


def test_dinov3_manager_401_is_actionable(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "dinov3")

    def _raise(**kw):
        raise cm._GatedRepoError("403")

    monkeypatch.setattr(cm, "snapshot_download", _raise)
    with pytest.raises(RuntimeError, match="Request access"):
        cm.Dinov3CheckpointManager(size="base").download(interactive=False)


def test_dinov3_manager_rejects_unknown_size():
    from phenotypic.detect.nn._checkpoint_manager import Dinov3CheckpointManager

    with pytest.raises(ValueError, match="size"):
        Dinov3CheckpointManager(size="huge")


def test_dinov2_manager_is_ungated(monkeypatch):
    # DINOv2 needs no acceptance / token
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    monkeypatch.setattr(cm, "snapshot_download", lambda **kw: "/fake/dinov2")
    assert cm.Dinov2CheckpointManager(size="base").download() == "/fake/dinov2"


def test_dinov2_manager_maps_size_to_repo_id(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    calls: dict = {}
    monkeypatch.setattr(
        cm, "snapshot_download",
        lambda **kw: calls.update(kw) or "/fake/dinov2",
    )
    cm.Dinov2CheckpointManager(size="large").download()
    assert calls["repo_id"] == "facebook/dinov2-large"


def test_sam3_non_gated_error_propagates(monkeypatch):
    """A non-gated error (e.g. disk full) is re-raised unchanged, not reworded."""
    from phenotypic.detect.nn import _checkpoint_manager as cm

    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "sam3")

    def _raise(**kw):
        raise OSError("disk full")

    monkeypatch.setattr(cm, "snapshot_download", _raise)
    with pytest.raises(OSError, match="disk full"):
        cm.Sam3CheckpointManager().download(interactive=False)


# ---------------------------------------------------------------------------
# Download CLI routing (B1 — extends the existing --model-type click.Choice)
# ---------------------------------------------------------------------------


def test_cli_sam3_choice_and_accept_license(monkeypatch):
    from click.testing import CliRunner

    from phenotypic.detect.nn import _checkpoint_manager as cm
    from phenotypic.detect.nn._cli import nn_cli

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    monkeypatch.setattr(cm, "snapshot_download", lambda **kw: "/fake/cache/sam3")

    result = CliRunner().invoke(
        nn_cli, ["download", "--model-type", "sam3", "--accept-license"]
    )
    assert result.exit_code == 0, result.output
    assert "Cached" in result.output


def test_cli_sam3_without_accept_license_fails(monkeypatch):
    from click.testing import CliRunner

    from phenotypic.detect.nn import _checkpoint_manager as cm
    from phenotypic.detect.nn._cli import nn_cli

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    monkeypatch.setattr(cm, "snapshot_download", lambda **kw: "/fake/cache/sam3")

    # No --accept-license, non-tty input → gate raises → exit 1
    result = CliRunner().invoke(
        nn_cli, ["download", "--model-type", "sam3"], input="\n"
    )
    assert result.exit_code == 1
    assert "Failed" in result.output


def test_cli_dinov2_choice(monkeypatch):
    from click.testing import CliRunner

    from phenotypic.detect.nn import _checkpoint_manager as cm
    from phenotypic.detect.nn._cli import nn_cli

    calls: dict = {}
    monkeypatch.setattr(
        cm, "snapshot_download",
        lambda **kw: calls.update(kw) or "/fake/dinov2",
    )
    result = CliRunner().invoke(
        nn_cli, ["download", "--model-type", "dinov2", "--dino-size", "small"]
    )
    assert result.exit_code == 0, result.output
    assert calls["repo_id"] == "facebook/dinov2-small"


def test_cli_dinov3_choice_and_accept_license(monkeypatch):
    from click.testing import CliRunner

    from phenotypic.detect.nn import _checkpoint_manager as cm
    from phenotypic.detect.nn._cli import nn_cli

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    calls: dict = {}
    monkeypatch.setattr(
        cm, "snapshot_download",
        lambda **kw: calls.update(kw) or "/fake/cache/dinov3",
    )
    result = CliRunner().invoke(
        nn_cli,
        [
            "download",
            "--model-type",
            "dinov3",
            "--dino-size",
            "large",
            "--accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["repo_id"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert "Cached" in result.output


def test_cli_dinov3_without_accept_license_fails(monkeypatch):
    from click.testing import CliRunner

    from phenotypic.detect.nn import _checkpoint_manager as cm
    from phenotypic.detect.nn._cli import nn_cli

    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    monkeypatch.setattr(cm, "snapshot_download", lambda **kw: "/fake/cache/dinov3")
    result = CliRunner().invoke(
        nn_cli, ["download", "--model-type", "dinov3"], input="\n"
    )
    assert result.exit_code == 1
    assert "Failed" in result.output
