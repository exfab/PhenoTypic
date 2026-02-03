"""Data structures for sweep execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import pandas as pd


@dataclass
class SweepResult:
    """Result from a single pipeline variant execution.

    Args:
        variant_id: Unique identifier for this variant (e.g., 'path0_combo3').
        pipeline_config: Dictionary of parameter values used for this variant.
        image_name: Name of the input image processed.
        success: Whether execution completed without errors.
        outputs: Dictionary mapping output names to file paths.
        metrics: Dictionary of computed metrics (object count, IoU, etc.).
        error: Error message if execution failed.
        execution_time: Time taken to execute in seconds.
    """

    variant_id: str
    pipeline_config: Dict[str, Any]
    image_name: str
    success: bool
    outputs: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variant_id": self.variant_id,
            "pipeline_config": self.pipeline_config,
            "image_name": self.image_name,
            "success": self.success,
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "metrics": self.metrics,
            "error": self.error,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepResult":
        """Create from dictionary."""
        return cls(
            variant_id=data["variant_id"],
            pipeline_config=data["pipeline_config"],
            image_name=data["image_name"],
            success=data["success"],
            outputs={k: Path(v) for k, v in data.get("outputs", {}).items()},
            metrics=data.get("metrics", {}),
            error=data.get("error"),
            execution_time=data.get("execution_time", 0.0),
        )


@dataclass
class SweepResults:
    """Aggregated results from a parameter sweep.

    Args:
        sweep_dir: Directory containing sweep outputs.
        results: List of individual SweepResult objects.
        manifest_path: Path to the JSON manifest file.
        created: Timestamp when sweep was executed.
        graph_config: Optional dictionary of the PipelineGraph configuration.
    """

    sweep_dir: Path
    results: List[SweepResult]
    manifest_path: Optional[Path] = None
    created: datetime = field(default_factory=datetime.now)
    graph_config: Optional[Dict[str, Any]] = None

    @property
    def successful(self) -> List[SweepResult]:
        """Get only successful results."""
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> List[SweepResult]:
        """Get only failed results."""
        return [r for r in self.results if not r.success]

    @property
    def success_rate(self) -> float:
        """Fraction of successful executions."""
        if not self.results:
            return 0.0
        return len(self.successful) / len(self.results)

    @property
    def total_time(self) -> float:
        """Total execution time across all variants."""
        return sum(r.execution_time for r in self.results)

    @property
    def variant_ids(self) -> List[str]:
        """List of all variant IDs."""
        return [r.variant_id for r in self.results]

    def get_result(self, variant_id: str) -> Optional[SweepResult]:
        """Get result by variant ID.

        Args:
            variant_id: The variant identifier to find.

        Returns:
            SweepResult if found, None otherwise.
        """
        for r in self.results:
            if r.variant_id == variant_id:
                return r
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis.

        Returns:
            DataFrame with columns for variant_id, image, success, time,
            all config parameters, and all metrics.

        Examples:
            >>> df = results.to_dataframe()
            >>> df.sort_values('object_count', ascending=False).head(5)
        """
        rows = []
        for r in self.results:
            row = {
                "variant_id": r.variant_id,
                "image": r.image_name,
                "success": r.success,
                "time": r.execution_time,
                "error": r.error,
            }
            # Flatten pipeline config
            for key, value in r.pipeline_config.items():
                # Handle nested configs (node_id -> param -> value)
                if isinstance(value, dict):
                    for param, val in value.items():
                        row[f"{key}.{param}"] = val
                else:
                    row[key] = value
            # Add metrics
            row.update(r.metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def filter_by_metric(
        self,
        metric: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> List[SweepResult]:
        """Filter results by metric range.

        Args:
            metric: Name of metric to filter on.
            min_value: Minimum value (inclusive).
            max_value: Maximum value (inclusive).

        Returns:
            List of matching SweepResult objects.
        """
        filtered = []
        for r in self.successful:
            if metric not in r.metrics:
                continue
            value = r.metrics[metric]
            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue
            filtered.append(r)
        return filtered

    def best_by_metric(
        self,
        metric: str,
        minimize: bool = False,
    ) -> Optional[SweepResult]:
        """Get the result with best metric value.

        Args:
            metric: Name of metric to optimize.
            minimize: If True, find minimum; if False (default), find maximum.

        Returns:
            SweepResult with best metric value, or None if no results.

        Examples:
            >>> best = results.best_by_metric('object_count')  # Maximize
            >>> best = results.best_by_metric('execution_time', minimize=True)
        """
        candidates = [r for r in self.successful if metric in r.metrics]
        if not candidates:
            return None

        if minimize:
            return min(candidates, key=lambda r: r.metrics[metric])
        else:
            return max(candidates, key=lambda r: r.metrics[metric])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": "1.0",
            "sweep_dir": str(self.sweep_dir),
            "created": self.created.isoformat(),
            "graph_config": self.graph_config,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepResults":
        """Create from dictionary."""
        return cls(
            sweep_dir=Path(data["sweep_dir"]),
            results=[SweepResult.from_dict(r) for r in data["results"]],
            created=datetime.fromisoformat(data["created"]),
            graph_config=data.get("graph_config"),
        )

    def save_manifest(self, path: Optional[Path] = None) -> Path:
        """Save results manifest to JSON file.

        Args:
            path: Output path. Defaults to sweep_dir/manifest.json.

        Returns:
            Path to saved manifest file.
        """
        path = path or self.sweep_dir / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        self.manifest_path = path
        return path

    @classmethod
    def load_manifest(cls, path: Path) -> "SweepResults":
        """Load results from manifest file.

        Args:
            path: Path to manifest.json file.

        Returns:
            Reconstructed SweepResults.
        """
        data = json.loads(path.read_text())
        results = cls.from_dict(data)
        results.manifest_path = path
        return results

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string.
        """
        lines = [
            f"Sweep Results: {self.sweep_dir.name}",
            f"  Created: {self.created.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Total variants: {len(self.results)}",
            f"  Successful: {len(self.successful)} ({self.success_rate:.1%})",
            f"  Failed: {len(self.failed)}",
            f"  Total time: {self.total_time:.1f}s",
        ]

        # Add metric summaries if available
        if self.successful:
            metrics = set()
            for r in self.successful:
                metrics.update(r.metrics.keys())

            if metrics:
                lines.append("  Metrics:")
                df = self.to_dataframe()
                for metric in sorted(metrics):
                    if metric in df.columns:
                        col = df[metric].dropna()
                        if len(col) > 0:
                            lines.append(
                                f"    {metric}: min={col.min():.2f}, "
                                f"max={col.max():.2f}, mean={col.mean():.2f}"
                            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SweepResults({len(self.results)} variants, "
            f"{len(self.successful)} successful, "
            f"dir={self.sweep_dir.name!r})"
        )
