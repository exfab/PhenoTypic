"""Interactive comparison widget for sweep results.

Provides side-by-side comparison of different pipeline variant outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

try:
    import param
    import panel as pn
    import numpy as np
    from skimage import io as skio

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    param = None
    pn = None

if TYPE_CHECKING:
    from ..explorer._sweep_results import SweepResults, SweepResult

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def load_image_safe(path: Path) -> Optional[np.ndarray]:
    """Safely load an image file.

    Args:
        path: Path to image file.

    Returns:
        Image array or None if loading fails.
    """
    try:
        if path.exists():
            return skio.imread(str(path))
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
    return None


def get_available_views(result: "SweepResult") -> List[str]:
    """Get list of available output views for a result.

    Args:
        result: SweepResult to check.

    Returns:
        List of view names that have saved outputs.
    """
    return list(result.outputs.keys())


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary as readable string.

    Args:
        metrics: Metrics dictionary.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"**{key}**: {value:.4f}")
        else:
            lines.append(f"**{key}**: {value}")
    return "\n".join(lines) if lines else "*No metrics*"


def format_config(config: Dict[str, Any]) -> str:
    """Format pipeline config as readable string.

    Args:
        config: Config dictionary.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    for key, value in sorted(config.items()):
        if isinstance(value, float):
            lines.append(f"**{key}**: {value:.4f}")
        else:
            lines.append(f"**{key}**: {value}")
    return "\n".join(lines) if lines else "*Default config*"


def compute_difference_map(
    img_a: np.ndarray,
    img_b: np.ndarray,
) -> np.ndarray:
    """Compute visual difference map between two images.

    Args:
        img_a: First image.
        img_b: Second image.

    Returns:
        Difference visualization (same shape as inputs).
    """
    # Handle shape mismatches
    if img_a.shape != img_b.shape:
        logger.warning("Image shapes don't match for difference")
        return np.zeros_like(img_a)

    # Convert to float for difference
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)

    # Compute absolute difference
    diff = np.abs(a - b)

    # Normalize to 0-255 for visualization
    if diff.max() > 0:
        diff = (diff / diff.max() * 255).astype(np.uint8)
    else:
        diff = diff.astype(np.uint8)

    return diff


# =============================================================================
# SweepComparisonWidget
# =============================================================================


if PANEL_AVAILABLE:

    class SweepComparisonWidget(param.Parameterized):
        """Interactive comparison widget for sweep results.

        Provides side-by-side viewing of different pipeline variant outputs
        with metrics comparison and difference visualization.

        Args:
            results: SweepResults to visualize.

        Examples:
            Basic usage:

            >>> from phenotypic.gui.viewer import SweepComparisonWidget
            >>> from phenotypic.gui.explorer import SweepResults
            >>> results = SweepResults.load_manifest('./results/manifest.json')
            >>> widget = SweepComparisonWidget(results=results)
            >>> widget.panel()

            In Jupyter:

            >>> widget.panel().servable()
        """

        # === Parameters ===
        results = param.ClassSelector(
            class_=object,  # Will be SweepResults at runtime
            doc="SweepResults to display",
        )

        variant_a = param.Selector(
            default=None,
            objects=[],
            doc="First variant to compare",
        )

        variant_b = param.Selector(
            default=None,
            objects=[],
            doc="Second variant to compare",
        )

        view_type = param.Selector(
            default="overlay",
            objects=["overlay", "objmask", "objmap", "enh_gray", "rgb", "gray"],
            doc="Output view to display",
        )

        image_name = param.Selector(
            default=None,
            objects=[],
            doc="Image to display (if multiple images processed)",
        )

        show_diff = param.Boolean(
            default=False,
            doc="Show difference between variants",
        )

        show_metrics = param.Boolean(
            default=True,
            doc="Show metrics comparison",
        )

        def __init__(self, results: "SweepResults", **params):
            """Initialize the comparison widget.

            Args:
                results: SweepResults to visualize.
                **params: Additional parameters.
            """
            super().__init__(results=results, **params)
            self._init_selectors()

        def _init_selectors(self) -> None:
            """Initialize selector options from results."""
            if self.results is None:
                return

            # Get unique variant IDs
            variant_ids = sorted(set(r.variant_id for r in self.results.results))
            self.param.variant_a.objects = variant_ids
            self.param.variant_b.objects = variant_ids

            if len(variant_ids) >= 1:
                self.variant_a = variant_ids[0]
            if len(variant_ids) >= 2:
                self.variant_b = variant_ids[1]
            elif len(variant_ids) >= 1:
                self.variant_b = variant_ids[0]

            # Get unique image names
            image_names = sorted(set(r.image_name for r in self.results.results))
            self.param.image_name.objects = image_names
            if image_names:
                self.image_name = image_names[0]

            # Get available views
            all_views = set()
            for result in self.results.successful:
                all_views.update(result.outputs.keys())
            if all_views:
                self.param.view_type.objects = sorted(all_views)
                if "overlay" in all_views:
                    self.view_type = "overlay"
                else:
                    self.view_type = sorted(all_views)[0]

        def _get_result(
            self,
            variant_id: str,
            image_name: str,
        ) -> Optional["SweepResult"]:
            """Get result for specific variant and image.

            Args:
                variant_id: Variant identifier.
                image_name: Image name.

            Returns:
                SweepResult or None if not found.
            """
            for result in self.results.results:
                if result.variant_id == variant_id and result.image_name == image_name:
                    return result
            return None

        def _load_view_image(
            self,
            result: "SweepResult",
            view_type: str,
        ) -> Optional[np.ndarray]:
            """Load image for specific view.

            Args:
                result: SweepResult to load from.
                view_type: Type of view to load.

            Returns:
                Image array or None.
            """
            if view_type not in result.outputs:
                return None
            return load_image_safe(result.outputs[view_type])

        @param.depends(
            "variant_a", "variant_b", "view_type", "image_name",
            "show_diff", "show_metrics"
        )
        def _comparison_view(self) -> pn.viewable.Viewable:
            """Build the comparison view.

            Returns:
                Panel layout with side-by-side images.
            """
            if self.results is None:
                return pn.pane.Markdown("*No results loaded*")

            result_a = self._get_result(self.variant_a, self.image_name)
            result_b = self._get_result(self.variant_b, self.image_name)

            if result_a is None and result_b is None:
                return pn.pane.Markdown("*No results found for selected variants*")

            # Build image panels
            panels = []

            # Variant A
            if result_a:
                img_a = self._load_view_image(result_a, self.view_type)
                if img_a is not None:
                    panels.append(pn.Column(
                        pn.pane.Markdown(f"### {self.variant_a}"),
                        pn.pane.PNG(img_a, sizing_mode="scale_width"),
                        self._metrics_panel(result_a) if self.show_metrics else None,
                        sizing_mode="stretch_width",
                    ))
                else:
                    panels.append(pn.pane.Markdown(
                        f"*View '{self.view_type}' not available for {self.variant_a}*"
                    ))
            else:
                panels.append(pn.pane.Markdown(f"*No result for {self.variant_a}*"))

            # Variant B
            if result_b:
                img_b = self._load_view_image(result_b, self.view_type)
                if img_b is not None:
                    panels.append(pn.Column(
                        pn.pane.Markdown(f"### {self.variant_b}"),
                        pn.pane.PNG(img_b, sizing_mode="scale_width"),
                        self._metrics_panel(result_b) if self.show_metrics else None,
                        sizing_mode="stretch_width",
                    ))
                else:
                    panels.append(pn.pane.Markdown(
                        f"*View '{self.view_type}' not available for {self.variant_b}*"
                    ))
            else:
                panels.append(pn.pane.Markdown(f"*No result for {self.variant_b}*"))

            # Difference view
            if self.show_diff and result_a and result_b:
                img_a = self._load_view_image(result_a, self.view_type)
                img_b = self._load_view_image(result_b, self.view_type)
                if img_a is not None and img_b is not None:
                    diff = compute_difference_map(img_a, img_b)
                    panels.append(pn.Column(
                        pn.pane.Markdown("### Difference"),
                        pn.pane.PNG(diff, sizing_mode="scale_width"),
                        sizing_mode="stretch_width",
                    ))

            return pn.Row(*panels, sizing_mode="stretch_width")

        def _metrics_panel(self, result: "SweepResult") -> pn.viewable.Viewable:
            """Build metrics panel for a result.

            Args:
                result: SweepResult to display.

            Returns:
                Panel with metrics.
            """
            return pn.Column(
                pn.pane.Markdown("**Config:**"),
                pn.pane.Markdown(format_config(result.pipeline_config)),
                pn.pane.Markdown("**Metrics:**"),
                pn.pane.Markdown(format_metrics(result.metrics)),
                pn.pane.Markdown(f"*Time: {result.execution_time:.2f}s*"),
                sizing_mode="stretch_width",
            )

        @param.depends("results")
        def _controls_view(self) -> pn.viewable.Viewable:
            """Build the controls panel.

            Returns:
                Panel with selector widgets.
            """
            return pn.Row(
                pn.widgets.Select.from_param(self.param.variant_a, name="Variant A"),
                pn.widgets.Select.from_param(self.param.variant_b, name="Variant B"),
                pn.widgets.Select.from_param(self.param.view_type, name="View"),
                pn.widgets.Select.from_param(self.param.image_name, name="Image"),
                pn.widgets.Checkbox.from_param(self.param.show_diff),
                pn.widgets.Checkbox.from_param(self.param.show_metrics),
                sizing_mode="stretch_width",
            )

        @param.depends("results")
        def _summary_view(self) -> pn.viewable.Viewable:
            """Build summary panel.

            Returns:
                Panel with results summary.
            """
            if self.results is None:
                return pn.pane.Markdown("")

            return pn.pane.Markdown(self.results.summary())

        def panel(self) -> pn.viewable.Viewable:
            """Get the Panel layout for display.

            Returns:
                Complete Panel layout.
            """
            if not pn.state._extensions_loaded:
                pn.extension()

            return pn.Column(
                pn.pane.Markdown("# Sweep Results Comparison"),
                self._controls_view,
                pn.layout.Divider(),
                self._comparison_view,
                pn.layout.Divider(),
                pn.pane.Markdown("### Summary"),
                self._summary_view,
                sizing_mode="stretch_both",
            )

        def next_variant_a(self) -> None:
            """Move to next variant A."""
            variants = self.param.variant_a.objects
            if not variants:
                return
            idx = variants.index(self.variant_a)
            self.variant_a = variants[(idx + 1) % len(variants)]

        def prev_variant_a(self) -> None:
            """Move to previous variant A."""
            variants = self.param.variant_a.objects
            if not variants:
                return
            idx = variants.index(self.variant_a)
            self.variant_a = variants[(idx - 1) % len(variants)]

        def next_variant_b(self) -> None:
            """Move to next variant B."""
            variants = self.param.variant_b.objects
            if not variants:
                return
            idx = variants.index(self.variant_b)
            self.variant_b = variants[(idx + 1) % len(variants)]

        def prev_variant_b(self) -> None:
            """Move to previous variant B."""
            variants = self.param.variant_b.objects
            if not variants:
                return
            idx = variants.index(self.variant_b)
            self.variant_b = variants[(idx - 1) % len(variants)]


else:
    # Stub class when Panel is not available
    class SweepComparisonWidget:
        """Placeholder when Panel is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SweepComparisonWidget requires Panel. "
                "Install with: pip install phenotypic[gui]"
            )
