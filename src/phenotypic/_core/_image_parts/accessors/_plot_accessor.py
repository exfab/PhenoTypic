from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple, Optional, Dict, Any, Union

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import distance_transform_edt, label as ndi_label
from scipy.stats import gaussian_kde
from skimage.morphology import (
    disk,
    square,
    diamond,
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing,
)
from skimage.measure import regionprops_table
import warnings

# Optional interactive widgets
try:
    import ipywidgets as widgets
    from IPython.display import display

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


class PlotAccessor:
    """Provides quality-of-life plots for developing image processing pipelines.

    This accessor offers sophisticated visualization methods to help understand how
    morphological operations, size filtering, and spatial patterns affect colony
    detection in arrayed microbial cultures on solid agar media. These plots are
    designed for pipeline development and parameter tuning rather than publication.

    All methods support flexible data requirements, automatically detecting whether
    labeled objects (objmap) or binary masks (objmask) are available, and adapting
    their analysis accordingly.

    Examples:
        .. dropdown:: Access plot methods through an Image instance

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import OtsuDetector

                # Load and detect colonies
                image = Image.imread('plate.jpg')
                detector = OtsuDetector()
                detected = detector.apply(image)

                # Access plot methods
                fig, axes = detected.plot.morph_progression()
                fig, ax = detected.plot.structural_response_curve()
                fig, ax = detected.plot.size_distribution()
    """

    def __init__(self, root_image: Image) -> None:
        """Initialize PlotAccessor with a reference to the parent Image.

        Args:
            root_image: The parent Image instance containing detection results
                and image data.
        """
        self._root_image = root_image

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

        Shows how morphological operations (opening, closing, erosion, dilation)
        affect object boundaries as structuring element size increases. Each panel
        displays the original image with color-coded boundaries overlaid for a
        specific kernel size, enabling identification of critical transition points
        where colonies merge, separate, or disappear.

        This visualization is essential for selecting optimal morphological parameters
        in colony detection pipelines. By observing when small colonies vanish
        (erosion too aggressive), when nearby colonies merge (dilation too large),
        or when gaps fill in unexpectedly (closing radius too big), you can
        empirically determine the appropriate kernel sizes for your specific imaging
        conditions.

        Args:
            operation: Morphological operation to apply. Options:
                - 'opening': Erosion followed by dilation; removes small objects
                  and thin connections while preserving overall colony shape.
                - 'closing': Dilation followed by erosion; fills small holes and
                  bridges narrow gaps between colony fragments.
                - 'erosion': Shrinks colony boundaries; removes thin protrusions
                  and uncertain edge pixels.
                - 'dilation': Expands colony boundaries; bridges nearby fragments
                  and recovers faint boundary pixels.
            kernel_sizes: List of kernel radii (pixels) to test. If None,
                automatically generates 7-9 sizes spanning the range [1, 15]
                pixels to cover typical colony spacing on standard plates.
            shape: Structuring element shape:
                - 'disk': Circular (isotropic); best for preserving round colonies.
                - 'square': Square with 8-connectivity; emphasizes grid-aligned edges.
                - 'diamond': Diamond with 4-connectivity; de-emphasizes diagonals.
            use_binary: If True, operates on binary mask (objmask). If False
                (default), converts labeled objects (objmap) to binary for analysis.
                Labeled objects provide richer per-colony information.
            figsize: Figure size as (width, height) in inches. If None, automatically
                sized based on number of kernel sizes (approximately 4×4 per panel).
            cmap: Colormap for boundary overlays. Each kernel size gets a distinct
                color. Default 'tab10' provides 10 perceptually distinct colors.

        Returns:
            Tuple containing:
                - fig: matplotlib Figure object
                - axes: numpy array of Axes objects in grid layout

        Raises:
            ValueError: If no objects detected (both objmap and objmask are empty).

        Interpretation:
            **Identifying optimal kernel sizes:**

            - **Small colonies vanish**: If colonies disappear at small kernel sizes
              (especially for erosion/opening), the operation is too aggressive.
              Reduce kernel size or use gentler preprocessing.

            - **Colonies merge unexpectedly**: If distinct adjacent colonies merge
              at moderate kernel sizes (dilation/closing), the kernel is too large.
              Reduce size or improve initial detection to better separate colonies.

            - **Stable transitions**: Look for kernel sizes where boundary changes
              are minimal between consecutive sizes—these represent stable parameter
              regions less sensitive to small tuning adjustments.

            - **Edge artifacts**: If colony boundaries near image edges behave
              differently than interior colonies, consider cropping or border removal
              in your pipeline.

            **Color interpretation:**
            Each color represents boundaries after applying the operation with that
            specific kernel size. Overlapping colors indicate regions where boundaries
            changed between kernel sizes—high overlap suggests high sensitivity to
            that parameter range.

        Examples:
            .. dropdown:: Visualize opening operation to identify small object removal

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.detect import OtsuDetector

                    # Detect colonies with some noise
                    image = Image.imread('noisy_plate.jpg')
                    detector = OtsuDetector()
                    detected = detector.apply(image)

                    # See how opening removes small artifacts
                    fig, axes = detected.plot.morph_progression(
                        operation='opening',
                        kernel_sizes=[1, 3, 5, 7, 9],
                        shape='disk'
                    )
                    plt.suptitle('Opening: Removing Small Objects')
                    plt.show()

                    # Identify kernel size where real colonies remain but noise is gone

            .. dropdown:: Test closing operation to bridge colony fragments

                .. code-block:: python

                    # Detect colonies that may be fragmented
                    detected = OtsuDetector().apply(image)

                    # Test different closing kernel sizes to bridge gaps
                    fig, axes = detected.plot.morph_progression(
                        operation='closing',
                        kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        shape='disk'
                    )
                    plt.suptitle('Closing: Bridging Colony Fragments')

                    # Find kernel size that reconnects fragments without merging
                    # distinct colonies
        """
        # Check data availability
        if use_binary or self._root_image.num_objects == 0:
            mask = self._root_image.objmask[:].astype(bool)
            if not mask.any():
                raise ValueError(
                        "No objects detected. Apply an ObjectDetector first."
                )
        else:
            objmap = self._root_image.objmap[:]
            if objmap.max() == 0:
                raise ValueError(
                        "No labeled objects. Apply an ObjectDetector first."
                )
            mask = (objmap > 0).astype(bool)

        # Auto-generate kernel sizes if not provided
        if kernel_sizes is None:
            # Generate 7-9 sizes from 1 to 15 pixels
            kernel_sizes = list(range(1, 16, 2))  # [1, 3, 5, 7, 9, 11, 13, 15]

        # Calculate grid layout
        n_kernels = len(kernel_sizes)
        n_cols = min(3, n_kernels)
        n_rows = int(np.ceil(n_kernels/n_cols))

        # Set figure size
        if figsize is None:
            figsize = (4*n_cols, 4*n_rows)

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Get grayscale image for background
        gray = self._root_image.gray[:]

        # Get colormap
        colors = plt.cm.get_cmap(cmap)

        # Process each kernel size
        for idx, k_size in enumerate(kernel_sizes):
            # Create structuring element
            if shape == "disk":
                selem = disk(k_size)
            elif shape == "square":
                selem = square(2*k_size + 1)
            elif shape == "diamond":
                selem = diamond(k_size)

            # Apply morphological operation
            if operation == "opening":
                result = binary_opening(mask, footprint=selem)
            elif operation == "closing":
                result = binary_closing(mask, footprint=selem)
            elif operation == "erosion":
                result = binary_erosion(mask, footprint=selem)
            elif operation == "dilation":
                result = binary_dilation(mask, footprint=selem)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Extract boundary (result - erosion of result)
            if result.any():
                boundary = result & ~binary_erosion(result, footprint=disk(1))
            else:
                boundary = np.zeros_like(result)

            # Create overlay
            ax = axes[idx]
            ax.imshow(gray, cmap="gray")

            # Overlay boundary with color
            if boundary.any():
                overlay = np.zeros((*gray.shape, 4))
                color = colors(idx/n_kernels)
                overlay[boundary] = color
                ax.imshow(overlay, alpha=0.7)

            ax.set_title(f"Kernel size: {k_size}")
            ax.axis("off")

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
            kernel_range: Union[Tuple[int, int], list[int]] = (1, 20),
            metric: Literal["count", "total_area", "mean_size"] = "count",
            shape: Literal["disk", "square", "diamond"] = "disk",
            use_binary: bool = False,
            figsize: Tuple[int, int] = (10, 6),
            show_derivative: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot how object metrics respond to morphological kernel size changes.

        Quantifies the sensitivity of detection results to morphological operation
        parameters by measuring how object count, total area, or mean size changes
        across a range of kernel sizes. The first derivative (rate of change) helps
        identify stable parameter regions and critical transition points where small
        parameter adjustments cause large changes in detection outcomes.

        This plot is critical for data-driven parameter selection. Optimal kernel
        sizes typically lie in "stable zones" where the metric plateaus (low
        derivative magnitude), indicating the parameter is robust to small tuning
        variations. Conversely, steep regions (high derivative) indicate sensitive
        zones where parameter choice significantly impacts results.

        Args:
            operation: Morphological operation to test ('opening', 'closing',
                'erosion', 'dilation'). See morph_progression() for detailed
                descriptions.
            kernel_range: Either a tuple (min, max) to test integer kernel sizes
                in that range (inclusive), or an explicit list of kernel sizes to test.
                Default (1, 20) covers typical colony spacing on standard plates.
            metric: Metric to track across kernel sizes:
                - 'count': Number of distinct objects (connected components).
                  Decreases with erosion/opening, may decrease with dilation if
                  objects merge.
                - 'total_area': Sum of all object pixels. Decreases with
                  erosion/opening, increases with dilation/closing.
                - 'mean_size': Average object size in pixels. Can increase or
                  decrease depending on whether small objects are preferentially
                  removed.
            shape: Structuring element shape ('disk', 'square', 'diamond').
            use_binary: If False (default), works with labeled objects (objmap).
                If True, uses binary mask (objmask).
            figsize: Figure size as (width, height) in inches.
            show_derivative: If True (default), adds a twin y-axis showing the
                derivative (rate of change) to highlight sensitive regions.

        Returns:
            Tuple containing:
                - fig: matplotlib Figure object
                - ax: Primary Axes object (derivative on twin axis if enabled)

        Raises:
            ValueError: If no objects detected or kernel_range is invalid.

        Interpretation:
            **Identifying optimal kernel sizes:**

            - **Plateaus (low derivative)**: Stable regions where small parameter
              changes have minimal impact. Preferred for robust pipelines.

            - **Steep slopes (high derivative)**: Sensitive regions where small
              changes cause large metric shifts. Avoid unless biologically justified.

            - **Inflection points (derivative zero-crossings)**: Transitions between
              removal phases (e.g., dust removed vs. small colonies removed).

            - **Count vs. area trade-offs**: For opening/erosion, count drops faster
              than area if small objects are removed first. For closing/dilation,
              area increases faster than count drops if fragments merge.

            **For colony phenotyping:**

            - **Opening for noise removal**: Select kernel size where count stabilizes
              after initial drop (dust removed, real colonies remain).

            - **Closing for fragment merging**: Select kernel size where count drops
              to expected colony number without excessive area inflation.

            - **Edge detection**: Stable count regions indicate robust detection less
              sensitive to illumination variations across plates.

        Examples:
            .. dropdown:: Find optimal opening kernel for noise removal

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.detect import OtsuDetector

                    # Detect colonies with noise artifacts
                    image = Image.imread('noisy_plate.jpg')
                    detected = OtsuDetector().apply(image)

                    # Plot object count vs. kernel size
                    fig, ax = detected.plot.structural_response_curve(
                        operation='opening',
                        kernel_range=(1, 15),
                        metric='count',
                        show_derivative=True
                    )
                    plt.title('Opening: Noise Removal Parameter Selection')

                    # Look for plateau after initial count drop—this kernel size
                    # removes noise while preserving real colonies

            .. dropdown:: Compare metrics to understand operation effects

                .. code-block:: python

                    # Plot multiple metrics to understand trade-offs
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    for idx, metric in enumerate(['count', 'total_area', 'mean_size']):
                        detected.plot.structural_response_curve(
                            operation='closing',
                            metric=metric,
                            show_derivative=False
                        )
                        plt.sca(axes[idx])
                        plt.title(f'Closing effect on {metric}')

                    plt.tight_layout()
                    # Reveals how closing trades count reduction for area increase
        """
        # Check data availability
        if use_binary or self._root_image.num_objects == 0:
            mask_ref = self._root_image.objmask[:].astype(bool)
            if not mask_ref.any():
                raise ValueError(
                        "No objects detected. Apply an ObjectDetector first."
                )
        else:
            objmap = self._root_image.objmap[:]
            if objmap.max() == 0:
                raise ValueError(
                        "No labeled objects. Apply an ObjectDetector first."
                )
            mask_ref = (objmap > 0).astype(bool)

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
                selem = square(2*k_size + 1)
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
                value = np.sum(result)/num if num > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            metric_values.append(value)

        metric_values = np.array(metric_values)

        # Compute derivative (finite difference)
        if len(kernel_sizes) > 2:
            derivative = np.gradient(metric_values, kernel_sizes)
        else:
            derivative = np.diff(metric_values)/np.diff(kernel_sizes)
            derivative = np.concatenate([[derivative[0]], derivative])

        # Create plot
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot primary metric
        color = "tab:blue"
        ax1.set_xlabel("Kernel Size (pixels)")
        ax1.set_ylabel(f"{metric.replace('_', ' ').title()}", color=color)
        ax1.plot(kernel_sizes, metric_values, "o-", color=color, linewidth=2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Mark stable regions (low derivative magnitude)
        if show_derivative and len(kernel_sizes) > 2:
            ax2 = ax1.twinx()
            color = "tab:orange"
            ax2.set_ylabel("Rate of Change (derivative)", color=color)
            ax2.plot(kernel_sizes, derivative, "s--", color=color, alpha=0.7)
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.axhline(0, color="gray", linestyle=":", linewidth=1)

            # Shade stable zones (low absolute derivative)
            stable_threshold = np.std(derivative)*0.5
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
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize spatial sensitivity to morphological parameter changes.

        Creates a heatmap showing how much object boundaries shift spatially across
        different kernel sizes, revealing which image regions are most sensitive to
        morphological parameter choices. High-displacement regions indicate areas
        where small parameter changes cause large boundary movements, suggesting
        either problematic detection or genuine biological heterogeneity.

        This spatial analysis is essential for understanding whether detection
        sensitivity is uniform across the plate or if certain regions (edges, corners,
        illumination gradients) require special handling. It also helps identify
        whether boundary instability stems from imaging artifacts or biological
        variation in colony morphology.

        Args:
            operation: Morphological operation to test ('opening', 'closing',
                'erosion', 'dilation').
            kernel_sizes: List of kernel sizes to compare. If None, uses
                [1, 3, 5, 7, 9] to span typical small-to-moderate adjustments.
            reference_size: Kernel size to use as baseline for displacement
                calculation. If None, uses the median kernel size from the list.
            shape: Structuring element shape ('disk', 'square', 'diamond').
            use_binary: If False (default), uses objmap. If True, uses objmask.
            figsize: Figure size as (width, height) in inches.
            cmap: Colormap for heatmap. Sequential colormaps (plasma, viridis,
                inferno) emphasize magnitude; diverging colormaps (RdBu) show
                positive/negative displacements if using signed distances.

        Returns:
            Tuple containing:
                - fig: matplotlib Figure with 2 subplots (heatmap and statistics)
                - ax: Array of Axes [heatmap_ax, stats_ax]

        Raises:
            ValueError: If no objects detected or insufficient kernel sizes provided.

        Interpretation:
            **High-displacement regions:**

            - **Image boundaries**: Objects touching edges often show high
              displacement due to edge effects. Consider cropping or border removal.

            - **Illumination gradients**: If displacement follows lighting patterns
              (e.g., higher at image periphery), your detection threshold may be
              illumination-dependent. Apply background correction.

            - **Small/faint colonies**: Colonies near detection threshold show high
              boundary instability. Improve preprocessing or accept that these
              represent uncertain detections.

            - **Touching colonies**: Boundaries between nearly-touching colonies
              are highly sensitive to morphological operations. May need watershed
              or other separation strategies.

            **Low-displacement regions:**

            - **Well-defined colonies**: Large, high-contrast colonies with clear
              boundaries show minimal displacement—robust detections.

            - **Interior regions**: Pixels far from boundaries naturally show low
              displacement. Focus interpretation on boundary proximity zones.

            **Quantitative metrics (statistics panel):**

            - **Mean displacement**: Overall sensitivity. Higher values indicate
              parameter-sensitive detection across the plate.

            - **95th percentile**: Worst-case sensitivity. Regions above this value
              represent highly uncertain detections.

            - **Spatial autocorrelation**: If displacement is spatially clustered
              (not random), systematic issues exist (e.g., illumination gradient).

        Examples:
            .. dropdown:: Identify illumination-dependent detection boundaries

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.detect import OtsuDetector

                    # Detect colonies on plate with vignetting
                    image = Image.imread('vignetted_plate.jpg')
                    detected = OtsuDetector().apply(image)

                    # Visualize boundary sensitivity
                    fig, axes = detected.plot.boundary_displacement(
                        operation='erosion',
                        kernel_sizes=[1, 2, 3, 4, 5],
                        cmap='plasma'
                    )
                    plt.suptitle('Boundary Stability: Erosion Sensitivity')

                    # High displacement at periphery suggests illumination-dependent
                    # thresholding—apply background correction before detection

            .. dropdown:: Assess closing parameter robustness for fragment merging

                .. code-block:: python

                    # Test closing stability for merging fragments
                    fig, axes = detected.plot.boundary_displacement(
                        operation='closing',
                        kernel_sizes=[2, 4, 6, 8, 10],
                        reference_size=6,
                        shape='disk'
                    )

                    # High displacement between fragments indicates sensitive merging
                    # parameter—select kernel size carefully or improve detection
        """
        # Check data availability
        if use_binary or self._root_image.num_objects == 0:
            mask_ref = self._root_image.objmask[:].astype(bool)
            if not mask_ref.any():
                raise ValueError(
                        "No objects detected. Apply an ObjectDetector first."
                )
        else:
            objmap = self._root_image.objmap[:]
            if objmap.max() == 0:
                raise ValueError(
                        "No labeled objects. Apply an ObjectDetector first."
                )
            mask_ref = (objmap > 0).astype(bool)

        # Auto-generate kernel sizes if not provided
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7, 9]

        if len(kernel_sizes) < 2:
            raise ValueError("Need at least 2 kernel sizes for displacement analysis.")

        # Select reference size
        if reference_size is None:
            reference_size = kernel_sizes[len(kernel_sizes)//2]

        # Get grayscale for visualization
        gray = self._root_image.gray[:]

        # Compute distance transforms for all kernel sizes
        distance_maps = {}

        for k_size in kernel_sizes:
            # Create structuring element
            if shape == "disk":
                selem = disk(k_size)
            elif shape == "square":
                selem = square(2*k_size + 1)
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

            # Compute signed distance transform (positive inside, negative outside)
            if result.any():
                dist_inside = distance_transform_edt(result)
                dist_outside = distance_transform_edt(~result)
                dist_signed = dist_inside - dist_outside
            else:
                dist_signed = -distance_transform_edt(~result)

            distance_maps[k_size] = dist_signed

        # Get reference distance map
        dist_ref = distance_maps[reference_size]

        # Compute displacement magnitude from reference
        displacement = np.zeros_like(dist_ref)
        for k_size, dist_map in distance_maps.items():
            if k_size != reference_size:
                displacement += np.abs(dist_map - dist_ref)

        # Normalize by number of comparisons
        displacement = displacement/(len(kernel_sizes) - 1)

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
            f"{'='*25}\n\n"
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

        return fig, axes

    def size_distribution(
            self,
            interactive: bool = False,
            thresholds: Optional[list[int]] = None,
            figsize: Tuple[int, int] = (15, 10),
            log_scale: bool = True,
    ) -> Union[Tuple[plt.Figure, np.ndarray], Tuple[plt.Figure, np.ndarray, Any]]:
        """Visualize object size distribution with filtering preview panels.

        Displays comprehensive size distribution statistics with preview panels showing
        the effects of different size thresholds on colony detection. This visualization
        is essential for selecting minimum/maximum size thresholds to remove artifacts
        (dust, debris, scratches) while retaining biologically relevant colonies.

        The multi-panel layout enables data-driven threshold selection by showing both
        statistical summaries (histogram, CDF) and spatial previews of filtering outcomes.
        Interactive mode (if ipywidgets available) allows real-time threshold adjustment
        with live preview updates.

        Args:
            interactive: If True and ipywidgets is installed, creates an interactive
                slider for threshold selection with live preview. If False or ipywidgets
                unavailable, generates static plots at predetermined thresholds.
            thresholds: List of size thresholds (pixels) to preview in static mode.
                If None, automatically selects 5th, 50th, and 95th percentile sizes.
                Ignored in interactive mode.
            figsize: Figure size as (width, height) in inches.
            log_scale: If True (default), uses log scale for x-axis in histogram
                and threshold sensitivity plots. Recommended when size distribution
                spans multiple orders of magnitude (common in noisy plates).

        Returns:
            If interactive=False or ipywidgets unavailable:
                Tuple of (fig, axes) where axes is 2D array of subplots
            If interactive=True and ipywidgets available:
                Tuple of (fig, axes, widget_container) where widget_container
                is the ipywidgets HBox/VBox with interactive controls

        Raises:
            ValueError: If no labeled objects detected.

        Interpretation:
            **Panel A (Histogram with KDE):**
            - **Bimodal distributions**: Two peaks suggest distinct populations
              (e.g., real colonies vs. debris). Set threshold between peaks.
            - **Long tail to small sizes**: Indicates noise/artifacts. Threshold
              should exclude this tail while retaining main distribution.
            - **Log-normal shape**: Common for biological colonies; use log scale
              for better visualization.

            **Panel B (Cumulative Distribution):**
            - **Objects retained curve**: Shows fraction of objects kept vs. threshold.
              Steep regions indicate many objects near that size.
            - **Area retained curve**: Shows fraction of total area kept. If area
              curve drops slowly while object curve drops fast, you're removing
              small artifacts efficiently.
            - **Divergence point**: Where the curves separate indicates transition
              from artifact-dominated to colony-dominated sizes.

            **Panel C (Threshold Sensitivity):**
            - **Plateau regions**: Ranges where threshold choice doesn't significantly
              affect object count—robust parameter selections.
            - **Steep drops**: Sensitive ranges where small threshold changes remove
              many objects. Avoid unless biologically justified.

            **Panels D-F (Filtered Previews):**
            - **5th percentile**: Very permissive; includes nearly all detected objects.
              If noise visible here, detection needs improvement before filtering.
            - **50th percentile (median)**: Moderate threshold. Should remove clear
              artifacts while retaining all plausible colonies.
            - **95th percentile**: Very strict; retains only largest objects. Useful
              for identifying if large artifacts (plate edges, condensation) exist.

            **Biological context for agar plates:**
            - **Typical colony size range**: 50-5000 pixels at standard imaging
              resolution. Smaller objects are usually dust, agar texture, or
              condensation droplets.
            - **Expected distribution**: Relatively narrow if growth conditions uniform;
              wide if plate has growth gradients or multiple species.
            - **Fragmented colonies**: If expected colonies appear as multiple small
              objects, threshold analysis reveals this—consider morphological closing
              before detection.

        Examples:
            .. dropdown:: Select size threshold to remove dust and debris

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.detect import OtsuDetector

                    # Detect colonies on a noisy plate
                    image = Image.imread('dusty_plate.jpg')
                    detected = OtsuDetector().apply(image)

                    # Visualize size distribution
                    fig, axes = detected.plot.size_distribution(
                        interactive=False,
                        thresholds=None,  # Auto-select percentiles
                        log_scale=True
                    )

                    # Examine histogram: look for gap between dust peak (small)
                    # and colony peak (larger). Set threshold in the gap.

            .. dropdown:: Interactive threshold selection with live preview

                .. code-block:: python

                    # Requires ipywidgets (install with: pip install ipywidgets)
                    fig, axes, widgets = detected.plot.size_distribution(
                        interactive=True
                    )

                    # Use slider to adjust threshold and see immediate effect
                    # on preview images. Find value that removes artifacts
                    # while preserving all real colonies.

                    # Note: Interactive mode works best in Jupyter notebooks
        """
        # Check for labeled objects
        objmap = self._root_image.objmap[:]
        if objmap.max() == 0:
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
        bin_width = 2*iqr/(len(sizes) ** (1/3))
        n_bins = int(np.ceil((sizes.max() - sizes.min())/bin_width))
        n_bins = max(10, min(n_bins, 100))  # Clamp between 10 and 100

        # Auto-select thresholds if not provided
        if thresholds is None:
            thresholds = [
                percentile_values[1],  # 5th percentile
                percentile_values[4],  # 50th percentile (median)
                percentile_values[7],  # 95th percentile
            ]

        # Check for interactive mode
        use_interactive = interactive and HAS_WIDGETS

        if not use_interactive and interactive:
            warnings.warn(
                    "ipywidgets not available. Falling back to static visualization. "
                    "Install with: pip install ipywidgets"
            )

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

        ax_hist.hist(sizes, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")

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
                # Scale KDE to match histogram height
                kde_scaled = kde_values*len(sizes)*(sizes.max() - sizes.min())/n_bins
                ax2 = ax_hist.twinx()
                ax2.plot(x_range, kde_scaled, "r-", linewidth=2, label="KDE")
                ax2.set_ylabel("Density (KDE)", color="r")
                ax2.tick_params(axis="y", labelcolor="r")
            except:
                pass  # Skip KDE if it fails

        # Add percentile markers
        for p, val in zip(percentiles[1::2], percentile_values[1::2]):  # Every other
            ax_hist.axvline(val, color="green", linestyle="--", alpha=0.5)
            ax_hist.text(
                    val,
                    ax_hist.get_ylim()[1]*0.9,
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
        cumsum_count = np.arange(1, len(sorted_sizes) + 1)/len(sorted_sizes)
        cumsum_area = np.cumsum(sorted_sizes)/np.sum(sorted_sizes)

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
        threshold_range = sorted_sizes[::max(1, len(sorted_sizes)//50)]  # Sample points
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

        for idx, threshold in enumerate(thresholds[:3]):  # Max 3 previews
            ax_prev = fig.add_subplot(gs[1, idx])

            # Create filtered mask
            filtered_objmap = objmap.copy()
            for label in np.unique(objmap):
                if label == 0:
                    continue
                if np.sum(objmap == label) < threshold:
                    filtered_objmap[objmap == label] = 0

            # Create overlay
            ax_prev.imshow(gray, cmap="gray")

            # Color accepted objects
            mask_accepted = filtered_objmap > 0
            if mask_accepted.any():
                overlay = np.zeros((*gray.shape, 4))
                overlay[mask_accepted] = [0, 1, 0, 0.4]  # Green, semi-transparent
                ax_prev.imshow(overlay)

            # Show rejected objects dimly
            mask_rejected = (objmap > 0) & ~mask_accepted
            if mask_rejected.any():
                overlay_rejected = np.zeros((*gray.shape, 4))
                overlay_rejected[mask_rejected] = [1, 0, 0,
                                                   0.2]  # Red, very transparent
                ax_prev.imshow(overlay_rejected)

            n_accepted = len(np.unique(filtered_objmap)) - 1  # Exclude background
            ax_prev.set_title(
                    f"Threshold: {int(threshold)} px\n"
                    f"Retained: {n_accepted}/{len(sizes)} objects"
            )
            ax_prev.axis("off")

        plt.suptitle("Object Size Distribution Analysis", fontsize=14,
                     fontweight="bold")

        # Create interactive widgets if requested and available
        widget_container = None
        if use_interactive:
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

            def update_preview(change):
                with output:
                    output.clear_output(wait=True)
                    threshold = change["new"]

                    # Create filtered mask
                    filtered_objmap = objmap.copy()
                    for label in np.unique(objmap):
                        if label == 0:
                            continue
                        if np.sum(objmap == label) < threshold:
                            filtered_objmap[objmap == label] = 0

                    # Display preview
                    fig_preview, ax_preview = plt.subplots(figsize=(8, 6))
                    ax_preview.imshow(gray, cmap="gray")

                    mask_accepted = filtered_objmap > 0
                    if mask_accepted.any():
                        overlay = np.zeros((*gray.shape, 4))
                        overlay[mask_accepted] = [0, 1, 0, 0.4]
                        ax_preview.imshow(overlay)

                    mask_rejected = (objmap > 0) & ~mask_accepted
                    if mask_rejected.any():
                        overlay_rejected = np.zeros((*gray.shape, 4))
                        overlay_rejected[mask_rejected] = [1, 0, 0, 0.2]
                        ax_preview.imshow(overlay_rejected)

                    n_accepted = len(np.unique(filtered_objmap)) - 1
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

        axes = fig.axes

        if widget_container is not None:
            return fig, axes, widget_container
        else:
            return fig, axes

    def spatial_size_map(
            self,
            mode: Literal["median", "mean", "percentile", "absolute"] = "median",
            value: Optional[float] = None,
            robust: bool = True,
            cmap: str = "RdBu_r",
            figsize: Tuple[int, int] = (14, 6),
            alpha: float = 0.6,
    ) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
        """Visualize spatial distribution of object sizes with diverging colormap.

        Creates a pseudo-color map where each object is colored by its size relative
        to a configurable center value, revealing spatial patterns that may indicate
        illumination gradients, growth gradients, edge effects, or systematic imaging
        artifacts. The diverging colormap (blue=smaller, white=center, red=larger)
        makes deviations from expected size immediately visible.

        This visualization is critical for quality control in arrayed colony phenotyping.
        Spatial size patterns can indicate:
        - Position effects (edge colonies grow differently due to evaporation/temperature)
        - Illumination gradients (detection threshold varies with lighting)
        - Contamination or spreading (unusually large objects in specific regions)
        - Systematic pipetting errors (size gradient across plate rows/columns)

        Args:
            mode: Method for calculating the diverging center (white point):
                - 'median': Robust center; 50% of objects colored warm, 50% cool.
                - 'mean': Arithmetic mean; sensitive to outliers.
                - 'percentile': Specific percentile (requires value parameter).
                - 'absolute': User-specified size in pixels (requires value parameter).
            value: Required for 'percentile' mode (0-100) or 'absolute' mode (size in
                pixels). Ignored for 'median' and 'mean' modes.
            robust: If True, uses trimmed statistics (5% trim on each end) to reduce
                outlier influence. Applies to 'mean' mode only.
            cmap: Diverging colormap name. Default 'RdBu_r' (blue=small, red=large).
                Other options: 'coolwarm', 'bwr', 'seismic'.
            figsize: Figure size as (width, height) in inches.
            alpha: Overlay transparency (0=invisible, 1=opaque). Default 0.6 shows
                both original image and size coloring.

        Returns:
            Tuple containing:
                - fig: matplotlib Figure object
                - axes: Array of 2 Axes [map_ax, colorbar_ax]
                - metadata: Dictionary with keys:
                    - 'center': Computed center value (pixels)
                    - 'center_mode': Mode used ('median', 'mean', etc.)
                    - 'n_below': Number of objects below center
                    - 'n_above': Number of objects above center
                    - 'fraction_below': Fraction of objects below center
                    - 'vmin', 'vmax': Colormap normalization bounds
                    - 'mean_size': Mean object size
                    - 'median_size': Median object size
                    - 'std_size': Standard deviation of sizes

        Raises:
            ValueError: If no labeled objects detected, or if mode='percentile' or
                'absolute' but value is not provided.

        Interpretation:
            **Color patterns:**

            - **Uniform color**: Homogeneous colony sizes across plate—ideal for
              phenotyping. Indicates consistent growth conditions and detection.

            - **Radial gradient**: Center different from edges suggests:
              - Temperature gradient (edges cooler due to evaporation)
              - Illumination vignetting (edges darker, threshold effect)
              - Agar drying at periphery affecting growth

            - **Row/column pattern**: Systematic gradient along one axis suggests:
              - Pipetting volume gradient (dilution errors)
              - Plate tilt during imaging (illumination gradient)
              - Incubator shelf effect (temperature variation)

            - **Random patches**: Isolated regions of unusual size indicate:
              - Contamination (large objects) or poor growth (small objects)
              - Agar defects or air bubbles affecting local growth
              - Detection artifacts (uneven background in specific regions)

            - **Edge effects**: All edge colonies different from interior suggests:
              - True biological effect (edge wells grow differently)
              - Imaging artifact (optical distortion, vignetting)
              - Need for border removal in analysis pipeline

            **Choosing the center (mode parameter):**

            - **'median' (default)**: Most robust choice. Insensitive to outliers,
              guarantees 50% of colonies on each side of color scale. Use when size
              distribution is skewed or contains outliers.

            - **'mean'**: Use when size distribution is symmetric and you want the
              arithmetic average. More sensitive to outliers than median.

            - **'percentile'**: Emphasize specific deviations. Examples:
              - value=75: Highlights 25% smallest objects (blue)—good for finding
                undergrown or fragmented colonies.
              - value=25: Highlights 25% largest objects (red)—good for finding
                contamination or merged colonies.

            - **'absolute'**: Use when you have a biological expectation. Examples:
              - Expected colony size from pilot experiments: Compare current plate
                to historical data.
              - Target size from assay design: Identify wells that didn't reach
                expected growth.
              - Positive control size: Color relative to known good growth.

            **Metadata usage:**
            The returned metadata dictionary enables programmatic analysis:

            >>> fig, axes, meta = image.plot.spatial_size_map()
            >>> if meta['fraction_below'] > 0.7:
            >>>     print("Warning: Most colonies undersized—check growth conditions")
            >>> if meta['std_size'] > meta['mean_size'] * 0.5:
            >>>     print("High size variability—check for contamination or gradient")

        Examples:
            .. dropdown:: Detect illumination gradient affecting detection

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.detect import OtsuDetector

                    # Detect colonies on plate with vignetting
                    image = Image.imread('vignetted_plate.jpg')
                    detected = OtsuDetector().apply(image)

                    # Visualize size distribution
                    fig, axes, meta = detected.plot.spatial_size_map(
                        mode='median',
                        cmap='RdBu_r'
                    )

                    # Blue edges, red center suggests detection threshold too low
                    # at periphery—apply background correction before detection

            .. dropdown:: Compare to expected colony size from pilot experiment

                .. code-block:: python

                    # Historical data: expected colony size = 250 pixels
                    expected_size = 250

                    fig, axes, meta = detected.plot.spatial_size_map(
                        mode='absolute',
                        value=expected_size,
                        cmap='RdBu_r'
                    )

                    # Blue regions: undergrown (smaller than expected)
                    # Red regions: overgrown (larger than expected)
                    # Reveals spatial growth variation relative to expectation

            .. dropdown:: Identify smallest 10% of colonies for quality control

                .. code-block:: python

                    # Highlight unusually small colonies
                    fig, axes, meta = detected.plot.spatial_size_map(
                        mode='percentile',
                        value=90,  # 90th percentile as center
                        cmap='RdBu_r'
                    )

                    # Most colonies will be blue (smaller than 90th percentile)
                    # Red colonies are in top 10% by size
                    # Helps identify if small colonies cluster spatially (artifact)
                    # or distribute randomly (biological variation)
        """
        # Check for labeled objects
        objmap = self._root_image.objmap[:]
        if objmap.max() == 0:
            raise ValueError(
                    "No labeled objects detected. Apply an ObjectDetector first."
            )

        # Extract object properties
        props = regionprops_table(
                objmap, properties=["label", "area", "centroid"]
        )
        sizes = props["area"]
        centroids_r = props["centroid-0"]
        centroids_c = props["centroid-1"]
        labels = props["label"]

        if len(sizes) == 0:
            raise ValueError("No objects found in object map.")

        # Calculate center value based on mode
        if mode == "median":
            center = np.median(sizes)
        elif mode == "mean":
            if robust:
                # Trimmed mean (5% trim on each end)
                sorted_sizes = np.sort(sizes)
                trim_count = int(len(sizes)*0.05)
                if trim_count > 0:
                    trimmed = sorted_sizes[trim_count:-trim_count]
                else:
                    trimmed = sorted_sizes
                center = np.mean(trimmed)
            else:
                center = np.mean(sizes)
        elif mode == "percentile":
            if value is None:
                raise ValueError("mode='percentile' requires value parameter (0-100)")
            center = np.percentile(sizes, value)
        elif mode == "absolute":
            if value is None:
                raise ValueError(
                        "mode='absolute' requires value parameter (size in pixels)")
            center = value
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Calculate statistics
        n_below = np.sum(sizes < center)
        n_above = np.sum(sizes >= center)
        fraction_below = n_below/len(sizes)

        # Normalize sizes relative to center
        # Use symmetric scale: max deviation from center
        vmin = sizes.min()
        vmax = sizes.max()

        # Create pseudo-color image
        size_map = np.zeros_like(objmap, dtype=float)
        for label, size in zip(labels, sizes):
            size_map[objmap == label] = size

        # Mask background
        size_map_masked = np.ma.masked_where(objmap == 0, size_map)

        # Get grayscale for background
        gray = self._root_image.gray[:]

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 20, figure=fig, wspace=0.5)
        ax_map = fig.add_subplot(gs[0, :18])
        ax_cbar = fig.add_subplot(gs[0, 19])

        # Plot grayscale background
        ax_map.imshow(gray, cmap="gray", alpha=1 - alpha)

        # Overlay size map with diverging colormap
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax_map.imshow(size_map_masked, cmap=cmap, alpha=alpha, norm=norm)

        ax_map.set_title(
                f"Spatial Size Map (Center: {center:.1f} px, Mode: {mode})",
                fontsize=12,
                fontweight="bold",
        )
        ax_map.axis("off")

        # Add colorbar with annotations
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label("Object Size (pixels)", fontsize=10)

        # Mark center on colorbar
        ax_cbar.axhline(center, color="white", linewidth=2, linestyle="--")
        ax_cbar.text(
                1.05,
                center,
                f"Center\n{center:.1f}",
                transform=ax_cbar.transData,
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Add statistics text
        stats_text = (
            f"Objects below center: {n_below} ({fraction_below*100:.1f}%)\n"
            f"Objects above center: {n_above} ({(1 - fraction_below)*100:.1f}%)\n"
            f"Mean: {np.mean(sizes):.1f} px\n"
            f"Median: {np.median(sizes):.1f} px\n"
            f"Std: {np.std(sizes):.1f} px"
        )

        ax_map.text(
                0.02,
                0.98,
                stats_text,
                transform=ax_map.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Prepare metadata dictionary
        metadata = {
            "center"        : center,
            "center_mode"   : mode,
            "n_below"       : n_below,
            "n_above"       : n_above,
            "fraction_below": fraction_below,
            "vmin"          : vmin,
            "vmax"          : vmax,
            "mean_size"     : np.mean(sizes),
            "median_size"   : np.median(sizes),
            "std_size"      : np.std(sizes),
        }

        plt.tight_layout()

        return fig, np.array([ax_map, ax_cbar]), metadata

    def size_scatter(
            self,
            color_by: Literal[
                "intensity_std", "solidity", "eccentricity"] = "intensity_std",
            figsize: Tuple[int, int] = (12, 8),
            show_regression: bool = True,
            show_marginals: bool = True,
            alpha: float = 0.6,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Correlate object size with mean intensity to distinguish colonies from artifacts.

        Creates a scatter plot of object size vs. mean intensity, revealing relationships
        that distinguish true colonies from imaging artifacts, debris, or contamination.
        True colonies typically exhibit consistent size-intensity relationships (e.g.,
        larger colonies are proportionally brighter due to greater biomass), while
        artifacts show anomalous patterns (e.g., small but very bright dust particles).

        This visualization enables data-driven filtering rules based on feature
        correlations rather than arbitrary thresholds. By fitting a regression line
        through the main population, outliers become apparent as objects far from
        the expected relationship—these can be flagged for removal or manual inspection.

        Args:
            color_by: Secondary feature to color-code points:
                - 'intensity_std': Standard deviation of intensity within object.
                  High values indicate heterogeneous objects (merged colonies, artifacts).
                - 'solidity': Ratio of object area to convex hull area. Low values
                  indicate irregular shapes (merged colonies, debris).
                - 'eccentricity': Ellipse eccentricity (0=circle, 1=line). High values
                  indicate elongated objects (artifacts, spreading colonies).
            figsize: Figure size as (width, height) in inches.
            show_regression: If True (default), fits and displays robust linear
                regression in log-log space with 95% confidence band. Helps identify
                outliers and expected size-intensity relationship.
            show_marginals: If True (default), adds marginal histograms on top and
                right showing size and intensity distributions separately.
            alpha: Point transparency (0=invisible, 1=opaque). Default 0.6 reveals
                point density in overlapping regions.

        Returns:
            Tuple containing:
                - fig: matplotlib Figure object
                - ax: Primary scatter plot Axes (main plot if marginals disabled)

        Raises:
            ValueError: If no labeled objects detected or insufficient objects
                for regression (<3 objects).

        Interpretation:
            **Size-intensity relationship (regression line):**

            - **Slope ≈ 1 (log-log)**: Integrated density constant—larger colonies
              are proportionally brighter. Expected for uniform colonies where size
              scales with biomass.

            - **Slope ≈ 0**: Intensity independent of size—colonies have similar
              brightness regardless of size. Suggests detection threshold dominates
              (all colonies detected at same intensity).

            - **Slope < 0**: Smaller objects brighter than larger—indicates artifacts.
              Small bright spots are usually dust, condensation, or optical defects
              rather than colonies.

            - **Slope > 1**: Larger objects disproportionately brighter—may indicate
              three-dimensional growth (colonies growing vertically as well as
              horizontally) or merged colonies.

            **Outlier patterns:**

            - **Small, very bright** (high intensity, low size): Dust, lens artifacts,
              condensation droplets. These should be filtered by setting minimum size
              AND maximum intensity thresholds.

            - **Large, very dim** (low intensity, high size): Over-segmentation
              (background regions incorrectly labeled as objects), shadows, or plate
              edges. Consider raising detection threshold or applying border removal.

            - **High intensity_std (color coding)**: Heterogeneous objects—merged
              colonies, colonies with internal structure, or artifacts. May need
              morphological separation (watershed) or should be flagged in analysis.

            - **Low solidity (color coding)**: Irregular shapes—touching colonies,
              spreading phenotypes, or artifacts with holes. Morphological filtering
              or shape-based object refinement may help.

            - **High eccentricity (color coding)**: Elongated objects—merged colonies
              in rows/columns (arrayed format artifact), spreading/filamentous
              phenotypes, or scratches on plate. Spatial pattern analysis can
              distinguish biological vs. technical causes.

            **Deriving filtering rules:**

            1. **Identify main population**: Points clustered near regression line
               represent typical colonies. Note their size and intensity ranges.

            2. **Define outlier regions**: Use Mahalanobis distance or distance from
               regression line to define "too far" threshold (typically 2-3 standard
               deviations).

            3. **Set multi-feature filters**: Combine size, intensity, and secondary
               features (solidity, eccentricity) to create robust filtering rules:

               >>> # Example filtering rules from scatter plot analysis
               >>> size_min = 50  # Remove small artifacts
               >>> size_max = 5000  # Remove large merged objects
               >>> intensity_min = 0.1  # Remove dim background regions
               >>> intensity_max = 0.9  # Remove saturated bright spots
               >>> solidity_min = 0.8  # Remove irregular artifacts

            4. **Validate spatially**: After filtering, check spatial distribution
               using spatial_size_map() to ensure removed objects aren't clustered
               (which might indicate biological effect, not artifact).

        Examples:
            .. dropdown:: Identify and remove dust artifacts by size-intensity pattern

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.detect import OtsuDetector

                    # Detect colonies on dusty plate
                    image = Image.imread('dusty_plate.jpg')
                    detected = OtsuDetector().apply(image)

                    # Analyze size-intensity correlation
                    fig, ax = detected.plot.size_scatter(
                        color_by='intensity_std',
                        show_regression=True
                    )

                    # Look for small, bright outliers above regression line
                    # These are typically dust with high intensity but small size
                    # Set filtering rule: remove objects with size < 100 AND
                    # intensity > regression prediction + 2*std

            .. dropdown:: Detect merged colonies using solidity as color feature

                .. code-block:: python

                    # Color by solidity to identify irregular objects
                    fig, ax = detected.plot.size_scatter(
                        color_by='solidity',
                        show_regression=True,
                        show_marginals=True
                    )

                    # Low solidity (blue points) with large size suggests merged
                    # colonies (irregular shape due to merging) or colonies with
                    # internal voids

                    # Filtering rule: Flag large objects (size > 1000) with low
                    # solidity (< 0.8) for manual inspection or watershed separation

            .. dropdown:: Compare intensity relationships across multiple plates

                .. code-block:: python

                    # Analyze multiple plates to check consistency
                    plates = ['plate1.jpg', 'plate2.jpg', 'plate3.jpg']
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                    for idx, plate_path in enumerate(plates):
                        image = Image.imread(plate_path)
                        detected = OtsuDetector().apply(image)

                        fig_temp, ax_temp = detected.plot.size_scatter(
                            show_regression=True,
                            show_marginals=False
                        )

                        # Copy to main figure
                        # (In practice, would extract data and plot manually)

                    # Consistent regression slopes across plates indicate robust
                    # detection; varying slopes suggest plate-specific issues
                    # (illumination differences, growth variations)
        """
        # Check for labeled objects
        objmap = self._root_image.objmap[:]
        if objmap.max() == 0:
            raise ValueError(
                    "No labeled objects detected. Apply an ObjectDetector first."
            )

        gray = self._root_image.gray[:]

        # Extract features using regionprops
        properties = [
            "label",
            "area",
            "intensity_mean",
            "intensity_std",
            "solidity",
            "eccentricity",
        ]
        props = regionprops_table(objmap, intensity_image=gray, properties=properties)

        sizes = props["area"]
        intensities = props["intensity_mean"]

        if len(sizes) < 3:
            raise ValueError(
                    "Need at least 3 objects for scatter plot analysis. "
                    f"Found {len(sizes)} objects."
            )

        # Get color feature
        if color_by == "intensity_std":
            color_values = props["intensity_std"]
            color_label = "Intensity Std"
        elif color_by == "solidity":
            color_values = props["solidity"]
            color_label = "Solidity"
        elif color_by == "eccentricity":
            color_values = props["eccentricity"]
            color_label = "Eccentricity"
        else:
            raise ValueError(f"Unknown color_by: {color_by}")

        # Create figure
        if show_marginals:
            # Use GridSpec for marginal plots
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(
                    4,
                    4,
                    figure=fig,
                    hspace=0.05,
                    wspace=0.05,
                    left=0.1,
                    right=0.95,
                    top=0.95,
                    bottom=0.1,
            )
            ax_main = fig.add_subplot(gs[1:, :-1])
            ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

            # Hide tick labels for marginals
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)

        else:
            fig, ax_main = plt.subplots(figsize=figsize)

        # Main scatter plot
        scatter = ax_main.scatter(
                sizes,
                intensities,
                c=color_values,
                cmap="viridis",
                alpha=alpha,
                s=50,
                edgecolors="black",
                linewidths=0.5,
        )

        ax_main.set_xlabel("Object Size (pixels)", fontsize=12)
        ax_main.set_ylabel("Mean Intensity", fontsize=12)
        ax_main.set_xscale("log")
        ax_main.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_main)
        cbar.set_label(color_label, fontsize=10)

        # Fit regression line in log-log space
        if show_regression and len(sizes) >= 3:
            # Log transform for linear fit
            log_sizes = np.log10(sizes)
            log_intensities = np.log10(intensities + 1e-10)  # Avoid log(0)

            # Linear fit in log space
            coeffs = np.polyfit(log_sizes, log_intensities, 1)
            poly = np.poly1d(coeffs)

            # Generate prediction line
            size_range = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
            log_size_range = np.log10(size_range)
            log_intensity_pred = poly(log_size_range)
            intensity_pred = 10 ** log_intensity_pred

            # Plot regression line
            ax_main.plot(
                    size_range,
                    intensity_pred,
                    "r-",
                    linewidth=2,
                    label=f"Log-log fit: slope={coeffs[0]:.2f}",
            )

            # Add confidence band (±2 std of residuals)
            residuals = log_intensities - poly(log_sizes)
            std_residuals = np.std(residuals)

            intensity_upper = 10 ** (log_intensity_pred + 2*std_residuals)
            intensity_lower = 10 ** (log_intensity_pred - 2*std_residuals)

            ax_main.fill_between(
                    size_range,
                    intensity_lower,
                    intensity_upper,
                    alpha=0.2,
                    color="red",
                    label="95% confidence",
            )

            ax_main.legend(loc="upper left", fontsize=10)

            # Interpretation text
            if coeffs[0] < -0.1:
                interpretation = "Negative slope: Small objects brighter (artifacts?)"
            elif coeffs[0] < 0.1:
                interpretation = "Flat slope: Intensity independent of size"
            elif coeffs[0] < 1.1:
                interpretation = "Slope ≈ 1: Proportional relationship (uniform colonies)"
            else:
                interpretation = "Steep slope: Large objects disproportionately bright"

            ax_main.text(
                    0.02,
                    0.98,
                    interpretation,
                    transform=ax_main.transAxes,
                    verticalalignment="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Marginal distributions
        if show_marginals:
            # Top histogram (size distribution)
            ax_top.hist(
                    sizes,
                    bins=50,
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="black",
            )
            ax_top.set_ylabel("Count", fontsize=10)
            ax_top.set_xscale("log")

            # Right histogram (intensity distribution)
            ax_right.hist(
                    intensities,
                    bins=50,
                    orientation="horizontal",
                    color="coral",
                    alpha=0.7,
                    edgecolor="black",
            )
            ax_right.set_xlabel("Count", fontsize=10)

        plt.suptitle(
                "Size-Intensity Correlation Analysis", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        return fig, ax_main


__all__ = "PlotAccessor",
