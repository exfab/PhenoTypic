from __future__ import annotations

from typing import Literal, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from skimage.measure import regionprops_table

from phenotypic.sdk_.register import register_plotter

from ._base_plotter import BasePlotter


@register_plotter
class SpatialPlotter(BasePlotter):
    """Provides spatial analysis visualization methods for colony phenotyping.

    This class offers methods to visualize spatial patterns in colony detection
    and size distributions, helping identify systematic biases, gradients, or
    artifacts in arrayed microbial cultures.
    """

    call_name = "spatial_size_map"

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
        artifacts.

        Args:
            mode: Method for calculating the diverging center (white point).
            value: Required for 'percentile' mode or 'absolute' mode.
            robust: If True, uses trimmed statistics for outlier resistance.
            cmap: Diverging colormap name.
            figsize: Figure size as (width, height) in inches.
            alpha: Overlay transparency (0=invisible, 1=opaque).

        Returns:
            Tuple containing (fig, axes, metadata) with spatial map and statistics.

        Raises:
            ValueError: If no labeled objects detected or invalid parameters.
        """
        # Validate parameters
        self._validate_figsize(figsize)
        self._validate_cmap(cmap)
        self._validate_alpha(alpha)

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
                trim_count = int(len(sizes) * 0.05)
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
        fraction_below = n_below / len(sizes)

        # Normalize sizes relative to center
        # Use symmetric scale: max deviation from center
        vmin = sizes.min()
        vmax = sizes.max()

        # Create pseudo-color image from object sizes
        # Map each label to its actual size from regionprops
        label_to_size = {lbl: sz for lbl, sz in zip(labels, sizes)}
        size_map = np.zeros_like(objmap, dtype=np.float64)
        for label, size in label_to_size.items():
            size_map[objmap == label] = size

        # Mask background (uses memory view, not copy)
        size_map_masked = np.ma.masked_where(objmap == 0, size_map)

        # Clean up large temporary structures
        del label_to_size, labels  # No longer needed

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
            f"Objects below center: {n_below} ({fraction_below * 100:.1f}%)\n"
            f"Objects above center: {n_above} ({(1 - fraction_below) * 100:.1f}%)\n"
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

        fig.set_layout_engine("constrained")

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

        Args:
            color_by: Secondary feature to color-code points.
            figsize: Figure size as (width, height) in inches.
            show_regression: If True, fits and displays robust linear regression.
            show_marginals: If True, adds marginal histograms.
            alpha: Point transparency (0=invisible, 1=opaque).

        Returns:
            Tuple containing (fig, ax) where ax is the primary scatter plot axes.

        Raises:
            ValueError: If no labeled objects detected or insufficient objects.
        """
        # Validate parameters
        self._validate_figsize(figsize)
        self._validate_alpha(alpha)

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
            # Log transform for linear fit - use copies to avoid modifying originals
            log_sizes = np.log10(sizes, dtype=np.float64)
            log_intensities = np.log10(intensities + 1e-10, dtype=np.float64)  # Avoid log(0)

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

            intensity_upper = 10 ** (log_intensity_pred + 2 * std_residuals)
            intensity_lower = 10 ** (log_intensity_pred - 2 * std_residuals)

            ax_main.fill_between(
                    size_range,
                    intensity_lower,
                    intensity_upper,
                    alpha=0.2,
                    color="red",
                    label="95% confidence",
            )

            ax_main.legend(loc="upper left", fontsize=10)

            # Clean up intermediate arrays
            del log_sizes, log_intensities, residuals

            # Interpretation text - fix undefined variable issue
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
        fig.set_layout_engine("constrained")

        return fig, ax_main



