from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from skimage.measure import regionprops_table

# Optional interactive widgets
try:
    import ipywidgets as widgets
    from IPython.display import display

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

from ._base_plotter import BasePlotter


class SizeDistributionPlotter(BasePlotter):
    """Provides size distribution visualization methods for colony analysis.

    This class offers methods to visualize and analyze the size distribution
    of detected colonies, including static and interactive plotting with
    size filtering previews.
    """

    def size_distribution(
            self,
            thresholds: Optional[list[int]] = None,
            figsize: Tuple[int, int] = (15, 10),
            log_scale: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize object size distribution with filtering preview panels (static version).

        Displays comprehensive size distribution statistics with preview panels showing
        the effects of different size thresholds on colony detection.

        Args:
            thresholds: List of size thresholds (pixels) to preview.
            figsize: Figure size as (width, height) in inches.
            log_scale: If True, uses log scale for x-axis in histogram.

        Returns:
            Tuple of (fig, axes) where axes is 2D array of subplots.

        Raises:
            ValueError: If no labeled objects detected or parameters invalid.
        """
        # Validate parameters
        self._validate_figsize(figsize)

        # Check for labeled objects
        objmap = self._root_image.objmap[:]
        if self._root_image.num_objects == 0:
            raise ValueError(
                    "No labeled objects detected. Apply an ObjectDetector first."
            )

        # Extract object sizes using regionprops
        props = regionprops_table(objmap, properties=["label", "area"])
        sizes = props["area"]

        if len(sizes) == 0:
            raise ValueError("No objects found in object map.")

        # Calculate statistics
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(sizes, percentiles)

        # Freedman-Diaconis rule for histogram bins
        iqr = np.percentile(sizes, 75) - np.percentile(sizes, 25)
        bin_width = 2 * iqr / (len(sizes) ** (1 / 3))
        n_bins = int(np.ceil((sizes.max() - sizes.min()) / bin_width))
        n_bins = max(10, min(n_bins, 100))  # Clamp between 10 and 100

        # Auto-select thresholds if not provided
        if thresholds is None:
            thresholds = [
                percentile_values[1],  # 5th percentile
                percentile_values[4],  # 50th percentile (median)
                percentile_values[7],  # 95th percentile
            ]

        # Create static visualization
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Panel A: Histogram with KDE
        ax_hist = fig.add_subplot(gs[0, 0])
        if log_scale and sizes.min() > 0:
            bins = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), n_bins)
            ax_hist.set_xscale("log")
        else:
            bins = n_bins

        # Use density=True so histogram integrates to 1
        ax_hist.hist(sizes, bins=bins, alpha=0.7, color="steelblue", edgecolor="black", density=True)

        # Add KDE if reasonable number of points
        if len(sizes) > 5:
            try:
                kde = gaussian_kde(sizes)
                if log_scale and sizes.min() > 0:
                    x_range = np.logspace(
                            np.log10(sizes.min()), np.log10(sizes.max()), 200
                    )
                else:
                    x_range = np.linspace(sizes.min(), sizes.max(), 200)
                kde_values = kde(x_range)
                # KDE is already normalized to integrate to 1 (matching density=True histogram)
                ax_hist.plot(x_range, kde_values, "r-", linewidth=2, label="KDE")
                ax_hist.set_ylabel("Density", color="black")
            except (ValueError, np.linalg.LinAlgError):
                # KDE fitting failed (singular matrix, etc.), skip overlay
                pass

        # Add percentile markers
        for p, val in zip(percentiles[1::2], percentile_values[1::2]):  # Every other
            ax_hist.axvline(val, color="green", linestyle="--", alpha=0.5)
            ax_hist.text(
                    val,
                    ax_hist.get_ylim()[1] * 0.9,
                    f"P{p}",
                    rotation=90,
                    va="top",
                    fontsize=8,
            )

        ax_hist.set_xlabel("Object Size (pixels)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Size Distribution (Histogram + KDE)")
        ax_hist.grid(True, alpha=0.3)

        # Panel B: Cumulative distribution
        ax_cdf = fig.add_subplot(gs[0, 1])
        sorted_sizes = np.sort(sizes)
        cumsum_count = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        cumsum_area = np.cumsum(sorted_sizes) / np.sum(sorted_sizes)

        if log_scale and sizes.min() > 0:
            ax_cdf.set_xscale("log")

        ax_cdf.plot(sorted_sizes, cumsum_count, "b-", linewidth=2,
                    label="Objects retained")
        ax_cdf.plot(sorted_sizes, cumsum_area, "r-", linewidth=2, label="Area retained")
        ax_cdf.set_xlabel("Size Threshold (pixels)")
        ax_cdf.set_ylabel("Cumulative Fraction Retained")
        ax_cdf.set_title("Cumulative Distribution")
        ax_cdf.legend()
        ax_cdf.grid(True, alpha=0.3)

        # Panel C: Threshold sensitivity
        ax_sens = fig.add_subplot(gs[0, 2])
        threshold_range = sorted_sizes[
            ::max(1, len(sorted_sizes) // 50)]  # Sample points
        n_retained = [np.sum(sizes >= t) for t in threshold_range]

        if log_scale and sizes.min() > 0:
            ax_sens.set_xscale("log")

        ax_sens.plot(threshold_range, n_retained, "o-", color="purple", linewidth=2)
        ax_sens.set_xlabel("Size Threshold (pixels)")
        ax_sens.set_ylabel("Objects Retained")
        ax_sens.set_title("Threshold Sensitivity")
        ax_sens.grid(True, alpha=0.3)

        # Panels D-F: Filtered previews
        gray = self._root_image.gray[:]

        # Create label->size lookup for efficient filtering
        label_to_size = dict(zip(props["label"], sizes))

        for idx, threshold in enumerate(thresholds[:3]):  # Max 3 previews
            ax_prev = fig.add_subplot(gs[1, idx])

            # Use boolean masks instead of copying entire objmap
            labels_to_remove = [lbl for lbl, size in label_to_size.items() if
                                size < threshold]
            mask_to_remove = np.isin(objmap, labels_to_remove)

            # Direct boolean indexing - no objmap copy
            mask_accepted = (objmap > 0) & ~mask_to_remove
            mask_rejected = (objmap > 0) & mask_to_remove

            # Create overlay
            ax_prev.imshow(gray, cmap="gray")

            # Color accepted objects (green)
            if mask_accepted.any():
                overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
                overlay[mask_accepted] = [0, 1, 0, 0.4]
                ax_prev.imshow(overlay)
                del overlay  # Explicitly free overlay memory

            # Show rejected objects dimly (red)
            if mask_rejected.any():
                overlay_rejected = np.zeros((*gray.shape, 4), dtype=np.float32)
                overlay_rejected[mask_rejected] = [1, 0, 0, 0.2]
                ax_prev.imshow(overlay_rejected)
                del overlay_rejected  # Explicitly free overlay memory

            n_accepted = len(sizes) - len(labels_to_remove)
            ax_prev.set_title(
                    f"Threshold: {int(threshold)} px\n"
                    f"Retained: {n_accepted}/{len(sizes)} objects"
            )
            ax_prev.axis("off")

            # Explicitly clean up temporary arrays
            del mask_accepted, mask_rejected, mask_to_remove

        plt.suptitle("Object Size Distribution Analysis", fontsize=14,
                     fontweight="bold")

        axes = fig.axes

        return fig, np.array(axes)

    def size_viewer(
            self,
            figsize: Tuple[int, int] = (15, 10),
    ) -> Any:
        """Visualize object size distribution with interactive threshold selection.

        Displays comprehensive size distribution statistics with preview panels showing
        the effects of different size thresholds on colony detection, with an interactive
        slider for real-time threshold adjustment.

        This method requires ipywidgets and is intended for use in Jupyter notebooks
        or IPython environments with interactive widget support.

        Args:
            figsize: Figure size as (width, height) in inches. Default: (15, 10).

        Returns:
            ipywidgets.VBox: Container with interactive slider and live preview output.
            The slider controls size threshold from min to max object size, and the
            preview updates in real-time showing accepted (green) and rejected (red)
            objects for each threshold value.

        Raises:
            ValueError: If no labeled objects detected or ipywidgets not available.
            ImportError: If ipywidgets is not installed.

        Note:
            This method is designed for interactive exploration during pipeline
            development. For batch processing, use size_distribution() instead.
        """
        # Validate parameters
        self._validate_figsize(figsize)

        # Check for ipywidgets availability
        if not HAS_WIDGETS:
            raise ValueError(
                    "Interactive size distribution requires ipywidgets. "
                    "Install with: pip install ipywidgets"
            )

        # Check for labeled objects
        objmap = self._root_image.objmap[:]
        if self._root_image.num_objects == 0:
            raise ValueError(
                    "No labeled objects detected. Apply an ObjectDetector first."
            )

        # Extract object sizes using regionprops
        props = regionprops_table(objmap, properties=["label", "area"])
        sizes = np.asarray(props["area"])
        labels = np.asarray(props["label"])

        if len(sizes) == 0:
            raise ValueError("No objects found in object map.")

        # Placeholder for interactive previews
        gray = self._root_image.gray[:]

        # Create label->size lookup for efficient filtering
        label_to_size = dict(zip(props["label"], sizes))

        # Create interactive widgets
        # Create slider
        slider = widgets.IntSlider(
                value=int(np.median(sizes)),
                min=int(sizes.min()),
                max=int(sizes.max()),
                step=1,
                description="Threshold:",
                continuous_update=False,
        )

        # Create output for live preview
        output = widgets.Output()

        # Track current figure in closure to prevent memory leak
        fig_preview_ref = None

        def update_preview(change):
            nonlocal fig_preview_ref

            with output:
                output.clear_output(wait=True)
                threshold = change["new"]

                # CRITICAL: Close previous figure to free memory
                if fig_preview_ref is not None:
                    plt.close(fig_preview_ref)

                # Create filtered mask efficiently using lookup
                filtered_objmap = objmap.copy()
                labels_to_remove = labels[sizes < threshold]
                filtered_objmap[np.isin(filtered_objmap, labels_to_remove)] = 0

                # Display preview
                fig_preview, ax_preview = plt.subplots(figsize=figsize)
                fig_preview_ref = fig_preview  # Store reference for next update

                ax_preview.imshow(gray, cmap="gray")

                mask_accepted = filtered_objmap > 0
                if mask_accepted.any():
                    overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
                    overlay[mask_accepted] = [0, 1, 0, 0.4]
                    ax_preview.imshow(overlay)

                mask_rejected = (objmap > 0) & ~mask_accepted
                if mask_rejected.any():
                    overlay_rejected = np.zeros((*gray.shape, 4), dtype=np.float32)
                    overlay_rejected[mask_rejected] = [1, 0, 0, 0.2]
                    ax_preview.imshow(overlay_rejected)

                n_accepted = len(sizes) - len(labels_to_remove)
                ax_preview.set_title(
                        f"Interactive Preview: {threshold} px threshold\n"
                        f"Retained: {n_accepted}/{len(sizes)} objects"
                )
                ax_preview.axis("off")
                plt.tight_layout()
                plt.show()

        slider.observe(update_preview, names="value")

        widget_container = widgets.VBox([slider, output])
        display(widget_container)

        # Trigger initial update
        update_preview({"new": slider.value})

        return widget_container



