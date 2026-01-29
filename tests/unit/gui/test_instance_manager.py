"""Unit tests for InstanceManager."""

import tempfile
from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.gui import InstanceManager


class TestInstanceManager:
    """Test InstanceManager functionality."""

    def test_default_workspace(self):
        """Test default workspace location is ./pipelines/."""
        manager = InstanceManager()
        assert manager.workspace == Path.cwd() / "pipelines"
        assert manager.workspace.exists()
        manager.close()

    def test_custom_workspace(self, tmp_path):
        """Test custom workspace directory."""
        workspace = tmp_path / "custom"
        manager = InstanceManager(workspace=workspace)
        assert manager.workspace == workspace
        assert workspace.exists()
        manager.close()

    def test_save_pipeline(self, tmp_path):
        """Test saving pipeline to workspace."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        # Save pipeline
        filepath = manager.save_pipeline(pipeline, "test_pipeline")
        assert filepath.exists()
        assert filepath.name == "test_pipeline.json"

    def test_save_pipeline_adds_json_extension(self, tmp_path):
        """Test that .json extension is added automatically."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        filepath = manager.save_pipeline(pipeline, "test")
        assert filepath.name == "test.json"

    def test_save_pipeline_overwrite(self, tmp_path):
        """Test overwriting existing pipeline."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline1 = ImagePipeline([GaussianBlur(sigma=1.0)])
        pipeline2 = ImagePipeline([GaussianBlur(sigma=2.0)])

        # Save first pipeline
        manager.save_pipeline(pipeline1, "test")

        # Should raise error without overwrite
        with pytest.raises(FileExistsError):
            manager.save_pipeline(pipeline2, "test", overwrite=False)

        # Should succeed with overwrite=True
        manager.save_pipeline(pipeline2, "test", overwrite=True)

    def test_save_pipeline_invalid_name(self, tmp_path):
        """Test saving with invalid pipeline name."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        # Empty name
        with pytest.raises(ValueError):
            manager.save_pipeline(pipeline, "")

        # Path separators
        with pytest.raises(ValueError):
            manager.save_pipeline(pipeline, "../evil")

    def test_load_pipeline(self, tmp_path):
        """Test loading pipeline from workspace."""
        manager = InstanceManager(workspace=tmp_path)
        original = ImagePipeline([GaussianBlur(sigma=2.5)])

        # Save and load
        manager.save_pipeline(original, "test")
        loaded = manager.load_pipeline("test")

        # Check it's the same
        assert len(loaded._ops) == 1
        op = list(loaded._ops.values())[0]
        assert isinstance(op, GaussianBlur)
        assert op.sigma == 2.5

    def test_load_pipeline_adds_extension(self, tmp_path):
        """Test loading works with or without .json extension."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        manager.save_pipeline(pipeline, "test")

        # Both should work
        loaded1 = manager.load_pipeline("test")
        loaded2 = manager.load_pipeline("test.json")
        assert len(loaded1._ops) == len(loaded2._ops)

    def test_load_nonexistent_pipeline(self, tmp_path):
        """Test loading pipeline that doesn't exist."""
        manager = InstanceManager(workspace=tmp_path)

        with pytest.raises(FileNotFoundError):
            manager.load_pipeline("nonexistent")

    def test_list_pipelines(self, tmp_path):
        """Test listing saved pipelines."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        # Empty initially
        assert manager.list_pipelines() == []

        # Save some pipelines
        manager.save_pipeline(pipeline, "pipeline1")
        manager.save_pipeline(pipeline, "pipeline2")

        # List should be sorted
        pipelines = manager.list_pipelines()
        assert pipelines == ["pipeline1", "pipeline2"]

    def test_delete_pipeline(self, tmp_path):
        """Test deleting pipeline."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        manager.save_pipeline(pipeline, "test")
        assert "test" in manager.list_pipelines()

        manager.delete_pipeline("test")
        assert "test" not in manager.list_pipelines()

    def test_delete_nonexistent_pipeline(self, tmp_path):
        """Test deleting pipeline that doesn't exist."""
        manager = InstanceManager(workspace=tmp_path)

        with pytest.raises(FileNotFoundError):
            manager.delete_pipeline("nonexistent")

    def test_export_pipeline(self, tmp_path):
        """Test exporting pipeline to specific location."""
        manager = InstanceManager(workspace=tmp_path)
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        # Save to workspace
        manager.save_pipeline(pipeline, "test")

        # Export to different location
        export_path = tmp_path / "exports" / "exported.json"
        result = manager.export_pipeline("test", export_path)

        assert result.exists()
        assert result == export_path

    def test_context_manager(self, tmp_path):
        """Test context manager interface."""
        workspace = tmp_path / "temp"
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        with InstanceManager(workspace=workspace, auto_cleanup=False) as manager:
            manager.save_pipeline(pipeline, "test")
            assert workspace.exists()

        # Workspace should still exist (auto_cleanup=False)
        assert workspace.exists()

    def test_context_manager_auto_cleanup(self, tmp_path):
        """Test context manager with auto_cleanup."""
        workspace = tmp_path / "temp"
        pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])

        with InstanceManager(workspace=workspace, auto_cleanup=True) as manager:
            manager.save_pipeline(pipeline, "test")
            assert workspace.exists()

        # Workspace should be deleted (auto_cleanup=True)
        assert not workspace.exists()

    def test_repr(self, tmp_path):
        """Test string representation."""
        manager = InstanceManager(workspace=tmp_path, auto_cleanup=True)
        repr_str = repr(manager)
        assert "InstanceManager" in repr_str
        assert str(tmp_path) in repr_str
        assert "auto_cleanup=True" in repr_str
