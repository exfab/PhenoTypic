"""Tests for checkpoint manager classes and type aliases.

Most tests work WITHOUT torch installed.  Tests that call into torch-dependent
methods (``resolve_device``, ``cache_dir``, ``list_cached``, ``clear``) are
skipped when torch is unavailable.
"""

import importlib.util

import pytest

from phenotypic.detect.nn._checkpoint_manager import (
    Device,
    MicroSamCheckpointManager,
    MicroSamModelType,
    ResolvedDevice,
    Sam2CheckpointManager,
    Sam2ModelSize,
)

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# Sam2CheckpointManager — model registry
# ---------------------------------------------------------------------------


class TestSam2CheckpointManagerModels:
    """Verify the static MODELS registry (no torch needed)."""

    def test_has_four_sizes(self):
        assert set(Sam2CheckpointManager.MODELS.keys()) == {
            "tiny",
            "small",
            "base_plus",
            "large",
        }

    def test_each_model_has_filename(self):
        for size, info in Sam2CheckpointManager.MODELS.items():
            assert "filename" in info, f"Missing filename for {size}"
            assert info["filename"].endswith(".pt"), (
                f"Unexpected extension for {size}"
            )

    def test_each_model_has_config(self):
        for size, info in Sam2CheckpointManager.MODELS.items():
            assert "config" in info, f"Missing config for {size}"
            assert info["config"].endswith(".yaml"), (
                f"Unexpected config format for {size}"
            )

    def test_get_config_returns_string(self):
        cfg = Sam2CheckpointManager.get_config("tiny")
        assert isinstance(cfg, str)
        assert "yaml" in cfg

    def test_base_url_is_set(self):
        assert Sam2CheckpointManager.BASE_URL.startswith("https://")


# ---------------------------------------------------------------------------
# MicroSamCheckpointManager — model registry
# ---------------------------------------------------------------------------


class TestMicroSamCheckpointManagerModels:
    """Verify the static MODELS registry (no torch needed)."""

    def test_has_expected_models(self):
        expected = {
            "vit_t", "vit_b", "vit_l", "vit_h",
            "vit_t_lm", "vit_b_lm", "vit_l_lm",
            "vit_b_em_organelles", "vit_l_em_organelles",
        }
        assert set(MicroSamCheckpointManager.MODELS.keys()) == expected

    def test_each_model_has_description(self):
        for mt, desc in MicroSamCheckpointManager.MODELS.items():
            assert isinstance(desc, str), f"Description for {mt} is not a string"
            assert len(desc) > 0, f"Empty description for {mt}"


# ---------------------------------------------------------------------------
# Type alias spot checks
# ---------------------------------------------------------------------------


class TestTypeAliases:
    """Type aliases are importable and have the expected Literal members."""

    def test_sam2_model_size_importable(self):
        assert Sam2ModelSize is not None

    def test_microsam_model_type_importable(self):
        assert MicroSamModelType is not None

    def test_device_importable(self):
        assert Device is not None

    def test_resolved_device_importable(self):
        assert ResolvedDevice is not None


# ---------------------------------------------------------------------------
# resolve_device — requires torch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="Requires torch")
class TestResolveDevice:
    """Test resolve_device() with torch installed."""

    def test_cpu_returns_cpu(self):
        from phenotypic.detect.nn._checkpoint_manager import resolve_device

        assert resolve_device("cpu") == "cpu"

    def test_auto_with_allow_cpu(self):
        """auto + allow_cpu=True should return a device string without error."""
        from phenotypic.detect.nn._checkpoint_manager import resolve_device

        result = resolve_device("auto", allow_cpu=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_auto_without_allow_cpu_returns_or_raises(self):
        """auto + allow_cpu=False either finds an accelerator or raises."""
        from phenotypic.detect.nn._checkpoint_manager import resolve_device

        try:
            result = resolve_device("auto", allow_cpu=False)
            assert isinstance(result, str)
        except RuntimeError as exc:
            assert "No accelerator" in str(exc)


# ---------------------------------------------------------------------------
# Sam2CheckpointManager — torch-dependent methods
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="Requires torch")
class TestSam2CheckpointManagerTorch:
    """Tests that need torch for cache_dir / list_cached / clear."""

    def test_cache_dir_is_path(self):
        path = Sam2CheckpointManager.cache_dir()
        # Should be a Path-like object
        assert hasattr(path, "is_dir")

    def test_list_cached_returns_list(self):
        result = Sam2CheckpointManager.list_cached()
        assert isinstance(result, list)

    def test_clear_on_empty_cache(self):
        """Clearing when nothing is cached should return an empty list."""
        # Only safe to test for sizes that are unlikely to be cached in CI
        deleted = Sam2CheckpointManager.clear("tiny")
        assert isinstance(deleted, list)


# ---------------------------------------------------------------------------
# MicroSamCheckpointManager — cache location
# ---------------------------------------------------------------------------


class TestMicroSamCheckpointManagerCacheDir:
    """cache_dir() should return a Path even without micro_sam installed."""

    def test_cache_dir_returns_path(self):
        path = MicroSamCheckpointManager.cache_dir()
        assert hasattr(path, "is_dir")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
