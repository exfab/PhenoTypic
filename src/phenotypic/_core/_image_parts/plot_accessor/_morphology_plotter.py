from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import distance_transform_edt, label as ndi_label

from skimage.morphology import (
    disk,
    square,
    diamond,
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing,
)

from ._base_plotter import BasePlotter


class MorphologyPlotter(BasePlotter):
    """Provides morphological operation visualization methods for image processing pipelines.

    This class offers sophisticated visualization methods to help understand how
    morphological operations affect colony detection in arrayed microbial cultures
    on solid agar media. These plots are designed for pipeline development and
    parameter tuning rather than publication.
    """

    def morph_progression(
            self,
            operation: Literal["opening", "closing", "erosion", "dilation"] = "opening",
            kernel_sizes: Optional[list[int]] = None,
            shape: Literal["disk", "square", "diamond"] = "disk",
            use_binary: bool = False,
            figsize: Optional[Tuple[int, int]] = None,
            cmap: str = "tab10",
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize morphological operation effects across kernel sizes.

        Shows how morphological operations affect object boundaries as structuring
        element size increases. Each panel displays the original image with
        color-coded boundaries overlaid for a specific kernel size, enabling
        identification of critical transition points where colonies merge,
        separate, or disappear.

        Args:
            operation: Morphological operation to apply.
            kernel_sizes: List of kernel radii (pixels) to test.
            shape: Structuring element shape.
            use_binary: If True, operates on binary mask (objmask).
            figsize: Figure size as (width, height) in inches.
            cmap: Colormap for boundary overlays.

        Returns:
            Tuple containing (fig, axes) where axes is 2D array of subplots.

        Raises:
            ValueError: If no objects detected or parameters invalid.
        """
        # Validate parameters
        self._validate_figsize(figsize)
        self._validate_cmap(cmap)

        # Check data availability
        self._validate_objects_exist(use_binary)
        mask = self._get_mask_for_plotting(use_binary)

        # Auto-generate kernel sizes if not provided
        if kernel_sizes is None:
            # Generate 7-9 sizes from 1 to 15 pixels
            kernel_sizes = list(range(1, 16, 2))  # [1, 3, 5, 7, 9, 11, 13, 15]

        # Calculate grid layout
        n_kernels = len(kernel_sizes)
        n_cols = min(3, n_kernels)
        n_rows = int(np.ceil(n_kernels / n_cols))

        # Set figure size
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Get grayscale image for background
        gray = self._root_image.gray[:]

        # Get colormap - fix potential index bounds issue
        n_colors = len(kernel_sizes)
        colors = self._create_colormap(n_colors, cmap)

        # Process each kernel size
        for idx, k_size in enumerate(kernel_sizes):
            # Create structuring element
            match shape:
                case "disk":
                    selem = disk(k_size)
                case "square":
                    selem = square(2 * k_size + 1)
                case "diamond":
                    selem = diamond(k_size)
                case _:
                    raise ValueError(f"Unknown kernel shape: {shape}")

            # Apply morphological operation
            match operation:
                case "opening":
                    result = binary_opening(mask, footprint=selem)
                case "closing":
                    result = binary_closing(mask, footprint=selem)
                case "erosion":
                    result = binary_erosion(mask, footprint=selem)
                case "dilation":
                    result = binary_dilation(mask, footprint=selem)
                case _:
                    raise ValueError(f"Unknown operation: {operation}")

            # Extract boundary (result - erosion of result)
            if result.any():
                boundary = result & ~binary_erosion(result, footprint=disk(1))
            else:
                boundary = np.zeros_like(result)

            # Create overlay
            ax = axes[idx]
            ax.imshow(gray, cmap="gray")

            # Overlay boundary with color (idx guaranteed in bounds since we control loop)
            if boundary.any():
                overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
                color = colors[idx]
                overlay[boundary] = color
                ax.imshow(overlay, alpha=0.7)
                del overlay  # Explicitly free large overlay array

            ax.set_title(f"Kernel size: {k_size}")
            ax.axis("off")

            # Clean up large arrays to reduce memory usage
            del result, boundary, selem  # Clean up structuring element too

        # Hide unused subplots
        for idx in range(n_kernels, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
                f"Morphological {operation.capitalize()} Progression "
                f"({shape} structuring element)"
        )
        plt.tight_layout()

        return fig, axes

    def structural_response_curve(
            self,
            operation: Literal["opening", "closing", "erosion", "dilation"] = "opening",
            kernel_range: Tuple[int, int] | list[int] = (1, 20),
            metric: Literal["count", "total_area", "mean_size"] = "count",
            shape: Literal["disk", "square", "diamond"] = "disk",
            use_binary: bool = False,
            figsize: Tuple[int, int] = (10, 6),
            show_derivative: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot how object metrics respond to morphological kernel size changes.

        Quantifies the sensitivity of detection results to morphological operation
        parameters by measuring how object count, total area, or mean size changes
        across a range of kernel sizes.

        Args:
            operation: Morphological operation to test.
            kernel_range: Range or list of kernel sizes to test.
            metric: Metric to track across kernel sizes.
            shape: Structuring element shape.
            use_binary: If False, works with labeled objects.
            figsize: Figure size as (width, height) in inches.
            show_derivative: If True, adds derivative plot.

        Returns:
            Tuple containing (fig, ax) where ax is the primary axes.

        Raises:
            ValueError: If no objects detected or kernel_range is invalid.
        """
        # Validate parameters
        self._validate_figsize(figsize)

        # Check data availability
        self._validate_objects_exist(use_binary)
        mask_ref = self._get_mask_for_plotting(use_binary)

        # Parse kernel_range
        if isinstance(kernel_range, (list, tuple)) and len(kernel_range) == 2:
            kernel_sizes = list(range(kernel_range[0], kernel_range[1] + 1))
        else:
            kernel_sizes = list(kernel_range)

        if len(kernel_sizes) < 2:
            raise ValueError("Need at least 2 kernel sizes to plot a curve.")

        # Compute metric for each kernel size
        metric_values = []
        for k_size in kernel_sizes:
            # Create structuring element
            if shape == "disk":
                selem = disk(k_size)
            elif shape == "square":
                selem = square(2 * k_size + 1)
            elif shape == "diamond":
                selem = diamond(k_size)

            # Apply operation
            if operation == "opening":
                result = binary_opening(mask_ref, footprint=selem)
            elif operation == "closing":
                result = binary_closing(mask_ref, footprint=selem)
            elif operation == "erosion":
                result = binary_erosion(mask_ref, footprint=selem)
            elif operation == "dilation":
                result = binary_dilation(mask_ref, footprint=selem)

            # Label and compute metric
            labeled, num = ndi_label(result)

            if metric == "count":
                value = num
            elif metric == "total_area":
                value = np.sum(result)
            elif metric == "mean_size":
                value = np.sum(result) / num if num > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            metric_values.append(value)

            # Clean up large arrays to prevent memory accumulation
            del selem, result, labeled

        metric_values = np.array(metric_values)

        # Compute derivative (finite difference) - fix potential division by zero
        if len(kernel_sizes) > 2:
            derivative = np.gradient(metric_values, kernel_sizes)
        else:
            kernel_diffs = np.diff(kernel_sizes)
            if len(kernel_diffs) > 0 and kernel_diffs[0] != 0:
                derivative = np.diff(metric_values) / kernel_diffs
                derivative = np.concatenate([[derivative[0]], derivative])
            else:
                derivative = np.zeros_like(metric_values)

        # Create plot
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot primary metric
        color = "tab:blue"
        ax1.set_xlabel("Kernel Size (pixels)")
        ax1.set_ylabel(f"{metric.replace('_', ' ').title()}", color=color)
        ax1.plot(kernel_sizes, metric_values, "o-", color=color, linewidth=2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Mark stable regions (low derivative magnitude) - fix division by zero
        if show_derivative and len(kernel_sizes) > 2:
            ax2 = ax1.twinx()
            color = "tab:orange"
            ax2.set_ylabel("Rate of Change (derivative)", color=color)
            ax2.plot(kernel_sizes, derivative, "s--", color=color, alpha=0.7)
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.axhline(0, color="gray", linestyle=":", linewidth=1)

            # Shade stable zones (low absolute derivative) - avoid division by zero
            std_derivative = np.std(derivative)
            if std_derivative > 0:
                stable_threshold = std_derivative * 0.5
                stable_mask = np.abs(derivative) < stable_threshold
                if stable_mask.any():
                    for i in range(len(stable_mask) - 1):
                        if stable_mask[i] and stable_mask[i + 1]:
                            ax1.axvspan(
                                    kernel_sizes[i],
                                    kernel_sizes[i + 1],
                                    alpha=0.1,
                                    color="green",
                                    label="Stable region" if i == 0 else "",
                            )

        plt.title(
                f"{operation.capitalize()} Response: {metric.replace('_', ' ').title()} "
                f"vs. Kernel Size ({shape})"
        )
        plt.tight_layout()

        return fig, ax1

    def boundary_displacement(
            self,
            operation: Literal["opening", "closing", "erosion", "dilation"] = "opening",
            kernel_sizes: Optional[list[int]] = None,
            reference_size: Optional[int] = None,
            shape: Literal["disk", "square", "diamond"] = "disk",
            use_binary: bool = False,
            figsize: Tuple[int, int] = (12, 5),
            cmap: str = "plasma",
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize spatial sensitivity to morphological parameter changes.

        Creates a heatmap showing how much object boundaries shift spatially across
        different kernel sizes, revealing which image regions are most sensitive to
        morphological parameter choices.

        Args:
            operation: Morphological operation to test.
            kernel_sizes: List of kernel sizes to compare.
            reference_size: Kernel size to use as baseline.
            shape: Structuring element shape.
            use_binary: If False, uses objmap.
            figsize: Figure size as (width, height) in inches.
            cmap: Colormap for heatmap.

        Returns:
            Tuple containing (fig, axes) with heatmap and statistics panels.

        Raises:
            ValueError: If insufficient kernel sizes or reference_size not in kernel_sizes.
        """
        # Validate parameters
        self._validate_figsize(figsize)
        self._validate_cmap(cmap)

        # Check data availability
        self._validate_objects_exist(use_binary)
        mask_ref = self._get_mask_for_plotting(use_binary)

        # Auto-generate kernel sizes if not provided
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7, 9]

        if len(kernel_sizes) < 2:
            raise ValueError("Need at least 2 kernel sizes for displacement analysis.")

        # Select reference size
        if reference_size is None:
            reference_size = kernel_sizes[len(kernel_sizes) // 2]

        # Get grayscale for visualization
        gray = self._root_image.gray[:]

        # Helper function to compute morphological operation + signed distance
        def _apply_morph_and_distance(mask, k_size, operation, shape):
            """Apply morphological operation and compute signed distance transform."""
            # Create structuring element
            if shape == "disk":
                selem = disk(k_size)
            elif shape == "square":
                selem = square(2 * k_size + 1)
            elif shape == "diamond":
                selem = diamond(k_size)

            # Apply operation
            if operation == "opening":
                result = binary_opening(mask, footprint=selem)
            elif operation == "closing":
                result = binary_closing(mask, footprint=selem)
            elif operation == "erosion":
                result = binary_erosion(mask, footprint=selem)
            elif operation == "dilation":
                result = binary_dilation(mask, footprint=selem)

            # Compute signed distance transform (positive inside, negative outside)
            if result.any():
                dist_inside = distance_transform_edt(result)
                dist_outside = distance_transform_edt(~result)
                dist_signed = dist_inside - dist_outside
            else:
                dist_signed = -distance_transform_edt(~result)

            return dist_signed

        # Validate reference_size is in kernel_sizes
        if reference_size not in kernel_sizes:
            raise ValueError(
                f"reference_size {reference_size} not in kernel_sizes {kernel_sizes}"
            )

        # Compute reference distance map
        dist_ref = _apply_morph_and_distance(mask_ref, reference_size, operation, shape)

        # Compute displacement incrementally to avoid storing all distance maps
        displacement = np.zeros_like(dist_ref, dtype=np.float64)
        for k_size in kernel_sizes:
            if k_size != reference_size:
                # Compute distance for this kernel size
                dist_current = _apply_morph_and_distance(mask_ref, k_size, operation, shape)
                # Add to displacement accumulator
                displacement += np.abs(dist_current - dist_ref)
                # Explicitly free large distance array
                del dist_current

        # Normalize by number of comparisons - avoid division by zero
        n_comparisons = len(kernel_sizes) - 1
        displacement = self._safe_divide(displacement, n_comparisons)

        # Compute statistics
        mean_disp = np.mean(displacement)
        std_disp = np.std(displacement)
        p95_disp = np.percentile(displacement, 95)

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[3, 1])

        # Plot heatmap
        ax0 = axes[0]
        ax0.imshow(gray, cmap="gray", alpha=0.3)
        im = ax0.imshow(displacement, cmap=cmap, alpha=0.7)
        ax0.set_title(
                f"Boundary Displacement Heatmap\n"
                f"{operation.capitalize()} (ref: {reference_size}px)"
        )
        ax0.axis("off")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax0)
        cbar.set_label("Displacement (pixels)")

        # Add contours at quantiles
        quantiles = [25, 50, 75]
        contour_levels = [np.percentile(displacement, q) for q in quantiles]
        ax0.contour(displacement, levels=contour_levels, colors="white", alpha=0.5)

        # Statistics panel
        ax1 = axes[1]
        ax1.axis("off")

        stats_text = (
            f"Displacement Statistics\n"
            f"{'=' * 25}\n\n"
            f"Mean: {mean_disp:.2f} px\n"
            f"Std Dev: {std_disp:.2f} px\n"
            f"95th %ile: {p95_disp:.2f} px\n\n"
            f"Kernel sizes tested:\n{kernel_sizes}\n\n"
            f"Reference: {reference_size} px\n"
            f"Shape: {shape}\n"
            f"Operation: {operation}"
        )

        ax1.text(
                0.1,
                0.95,
                stats_text,
                transform=ax1.transAxes,
                verticalalignment="top",
                fontfamily="monospace",
                fontsize=9,
        )

        plt.tight_layout()

        return fig, np.array([ax0, ax1])


