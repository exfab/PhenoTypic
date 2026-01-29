"""Workspace session manager for PhenoTypic GUI.

This module provides InstanceManager for managing pipeline storage and retrieval.
No Panel/GUI dependencies - uses only stdlib and existing phenotypic dependencies.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from phenotypic import ImagePipeline


class InstanceManager:
    """Workspace session manager for PhenoTypic GUI.

    Manages saved pipelines in a workspace directory. Default location is
    project-relative `./pipelines/` folder.

    Args:
        workspace: Directory for pipeline storage. If None, uses `./pipelines/`
            relative to current working directory.
        auto_cleanup: If True, delete workspace on close. Default False for
            project-relative folders to preserve saved pipelines.

    Examples:
        >>> from phenotypic.gui import InstanceManager
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur
        >>>
        >>> # Use default ./pipelines/ directory
        >>> manager = InstanceManager()
        >>> pipeline = ImagePipeline([GaussianBlur(sigma=2.0)])
        >>> manager.save_pipeline(pipeline, "my_pipeline")
        PosixPath('pipelines/my_pipeline.json')
        >>>
        >>> # List saved pipelines
        >>> manager.list_pipelines()
        ['my_pipeline']
        >>>
        >>> # Load pipeline
        >>> loaded = manager.load_pipeline("my_pipeline")
        >>>
        >>> # Context manager support
        >>> with InstanceManager() as mgr:
        ...     mgr.save_pipeline(pipeline, "temp")
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        auto_cleanup: bool = False,
    ):
        """Initialize InstanceManager with workspace directory.

        Args:
            workspace: Directory for pipeline storage. None = ./pipelines/
            auto_cleanup: Delete workspace on close (default False)
        """
        if workspace is None:
            self.workspace = Path.cwd() / "pipelines"
        else:
            self.workspace = Path(workspace)

        self.auto_cleanup = auto_cleanup
        self._ensure_workspace()

    def _ensure_workspace(self) -> None:
        """Create workspace directory if it doesn't exist."""
        self.workspace.mkdir(parents=True, exist_ok=True)

    def save_pipeline(
        self,
        pipeline: ImagePipeline,
        name: str,
        overwrite: bool = False,
    ) -> Path:
        """Save pipeline to workspace as JSON.

        Args:
            pipeline: ImagePipeline to save
            name: Pipeline name (extension added automatically)
            overwrite: Allow overwriting existing pipeline

        Returns:
            Path to saved pipeline file

        Raises:
            FileExistsError: If pipeline exists and overwrite=False
            ValueError: If name is invalid (empty, contains path separators)
        """
        # Validate name
        if not name or not name.strip():
            raise ValueError("Pipeline name cannot be empty")
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"Invalid pipeline name: {name}")

        # Ensure .json extension
        if not name.endswith(".json"):
            name = f"{name}.json"

        filepath = self.workspace / name

        # Check for existing file
        if filepath.exists() and not overwrite:
            raise FileExistsError(
                f"Pipeline '{name}' already exists. Use overwrite=True to replace."
            )

        # Save pipeline as JSON
        pipeline_json = pipeline.to_json()
        filepath.write_text(pipeline_json, encoding="utf-8")

        return filepath

    def load_pipeline(self, name: str) -> ImagePipeline:
        """Load pipeline from workspace.

        Args:
            name: Pipeline name (with or without .json extension)

        Returns:
            Loaded ImagePipeline

        Raises:
            FileNotFoundError: If pipeline doesn't exist
        """
        # Add .json if missing
        if not name.endswith(".json"):
            name = f"{name}.json"

        filepath = self.workspace / name

        if not filepath.exists():
            raise FileNotFoundError(
                f"Pipeline '{name}' not found in workspace: {self.workspace}"
            )

        # Load pipeline from JSON
        pipeline_json = filepath.read_text(encoding="utf-8")
        return ImagePipeline.from_json(pipeline_json)

    def list_pipelines(self) -> list[str]:
        """List all saved pipelines in workspace.

        Returns:
            List of pipeline names (without .json extension)
        """
        if not self.workspace.exists():
            return []

        pipelines = []
        for filepath in self.workspace.glob("*.json"):
            # Remove .json extension
            name = filepath.stem
            pipelines.append(name)

        return sorted(pipelines)

    def delete_pipeline(self, name: str) -> None:
        """Delete pipeline from workspace.

        Args:
            name: Pipeline name (with or without .json extension)

        Raises:
            FileNotFoundError: If pipeline doesn't exist
        """
        # Add .json if missing
        if not name.endswith(".json"):
            name = f"{name}.json"

        filepath = self.workspace / name

        if not filepath.exists():
            raise FileNotFoundError(
                f"Pipeline '{name}' not found in workspace: {self.workspace}"
            )

        filepath.unlink()

    def export_pipeline(self, name: str, path: Path) -> Path:
        """Export pipeline to a specific location.

        Args:
            name: Pipeline name in workspace
            path: Destination path for export

        Returns:
            Path to exported file

        Raises:
            FileNotFoundError: If pipeline doesn't exist
        """
        # Load pipeline
        pipeline = self.load_pipeline(name)

        # Ensure destination directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save to destination
        if not str(path).endswith(".json"):
            path = path.with_suffix(".json")

        pipeline_json = pipeline.to_json()
        path.write_text(pipeline_json, encoding="utf-8")

        return path

    def close(self) -> None:
        """Close workspace, optionally deleting it if auto_cleanup is True."""
        if self.auto_cleanup and self.workspace.exists():
            shutil.rmtree(self.workspace)

    def __enter__(self) -> "InstanceManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"InstanceManager(workspace={self.workspace}, auto_cleanup={self.auto_cleanup})"
