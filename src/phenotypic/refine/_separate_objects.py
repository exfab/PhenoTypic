"""Grid-based watershed segmentation for separating touching colonies.

Refines detected colonies by separating touching/merged objects using watershed segmentation
seeded at grid intersection points inferred from colony layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

import gc
import numpy as np

from phenotypic.abc_ import ObjectRefiner


class SeparateObjects(ObjectRefiner):
    """Separate touching or merged colonies using distance-transform watershed segmentation.

    Finds peaks in the Euclidean distance transform of the object mask and
    uses them as seed markers for watershed region growing, producing a
    distinct labeled region for each colony. For GridImages, one peak per
    grid cell is used; for plain Images, peaks are spaced by ``min_distance``.

    For a comparison of colony separation strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Pinned library plates (96-well, 384-well) where thresholding
          merges adjacent colonies into a single detection.
        - Dense yeast or bacterial arrays where colony boundaries are
          difficult to resolve by thresholding alone.
        - Variable colony sizes where the distance transform naturally
          adapts seed strength to each colony's diameter.
        - Non-grid images where a global minimum-spacing constraint is
          sufficient to individualize colonies.

    Consider Also:
        - :class:`MaskOpening` for gentle morphological separation of
          lightly touching colonies without watershed.
        - :class:`MaskErosion` when uniform inward shrinking is enough to
          break weak contact between adjacent colony masks.
        - :class:`GridAlignmentRefiner` when off-grid placement artefacts
          are the primary concern rather than merged regions.

    Args:
        min_distance: Minimum pixel distance between watershed seed peaks
            for plain Images. Ignored for GridImages, which use one peak
            per grid cell. Higher values reduce over-segmentation; lower
            values detect more seeds and produce finer splits. Typical
            range: 5--50. A reasonable starting point is roughly the
            expected minimum colony radius, so that each colony seeds at
            most one peak. Default: 10.

    Returns:
        Image: Input image with ``objmap`` updated so that previously
        merged colonies are separated into distinct labeled regions.

    Raises:
        ValueError: If no peaks are detected in the score map, or if
            watershed produces an empty result.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for
        separation and merging workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for a comparison of
        colony separation approaches.
    """

    min_distance: int = 10

    @staticmethod
    def _make_elevation_map(score_map: np.ndarray) -> np.ndarray:
        """Compute elevation map for watershed segmentation.

        Args:
            score_map: Combined intensity/distance score map from peak detection.

        Returns:
            Elevation map where high-score regions become valleys (local minima).

        Note:
            Inverts score_map so markers at high-score colony centers are placed
            in valleys, allowing watershed to flood into adjacent regions.
        """
        # Invert score_map: high scores → low elevation (valleys)
        # Markers at peaks (high score) become markers at valleys (low elevation)
        # Watershed floods from valleys upward into surrounding regions
        inverted = 1.0 - score_map
        return inverted.astype(np.float32)

    @staticmethod
    def _compute_peak_scores(
            objmask: np.ndarray,
            distance: np.ndarray,
    ) -> np.ndarray:
        """Compute distance-based score for peak detection.

        Uses distance transform to identify colony centers (far from edges).
        Peaks are placed at positions maximizing distance from object boundaries.

        Args:
            objmask: Binary mask of detected objects.
            distance: Distance transform (distance from edges).

        Returns:
            Normalized score map where higher values indicate better peak positions.
        """
        # Normalize distance to [0, 1]
        score = distance.astype(float)
        if score.max() > 0:
            score /= score.max()

        # Constrain to objmask region
        score[~objmask] = 0

        return score

    def _find_peaks_gridimage(
            self,
            score_map: np.ndarray,
            objmask: np.ndarray,
            row_edges: np.ndarray,
            col_edges: np.ndarray,
    ) -> np.ndarray:
        """Find one peak per grid cell using grid boundaries.

        Iterates through each grid cell and finds the position with maximum score.
        Only creates peaks in cells that contain detected objects.

        Args:
            score_map: Combined intensity/distance score map.
            objmask: Binary mask of detected objects.
            row_edges: Grid row boundaries (nrows+1 values).
            col_edges: Grid column boundaries (ncols+1 values).

        Returns:
            Array of shape (N, 2) with (row, col) peak positions in image coordinates.
        """
        peaks = []

        # Iterate through each grid cell
        for r in range(len(row_edges) - 1):
            r0, r1 = row_edges[r], row_edges[r + 1]
            for c in range(len(col_edges) - 1):
                c0, c1 = col_edges[c], col_edges[c + 1]

                # Extract cell region
                cell_score = score_map[r0:r1, c0:c1]
                cell_mask = objmask[r0:r1, c0:c1]

                # Skip empty cells
                if not cell_mask.any():
                    continue

                # Find peak in cell (constrained to objmask)
                cell_score_masked = cell_score.copy()
                cell_score_masked[~cell_mask] = -np.inf

                # Get position of maximum score
                peak_r_local, peak_c_local = np.unravel_index(
                        cell_score_masked.argmax(), cell_score_masked.shape
                )

                # Convert to image coordinates
                peak_r = r0 + peak_r_local
                peak_c = c0 + peak_c_local

                peaks.append([peak_r, peak_c])

        return np.array(peaks, dtype=int)

    def _find_peaks_regular(
            self,
            score_map: np.ndarray,
            objmask: np.ndarray,
            min_distance: int,
    ) -> np.ndarray:
        """Find peaks globally using minimum distance constraints.

        Uses peak_local_max to find local maxima with specified minimum spacing.
        Only finds peaks within detected object regions.

        Args:
            score_map: Combined intensity/distance score map.
            objmask: Binary mask of detected objects.
            min_distance: Minimum pixel distance between peaks.

        Returns:
            Array of shape (N, 2) with (row, col) peak positions.
        """
        from skimage.feature import peak_local_max

        # Find peaks with minimum distance constraint
        peaks = peak_local_max(
                score_map,
                min_distance=min_distance,
                labels=objmask.astype(int),  # Constrain to objmask
                exclude_border=False,
        )

        return peaks

    @staticmethod
    def _create_markers_from_peaks(
            objmask: np.ndarray,
            peaks: np.ndarray,
    ) -> np.ndarray:
        """Create watershed markers from peak positions.

        Places unique integer labels at each peak position to serve as
        watershed seed markers.

        Args:
            objmask: Binary mask of detected objects.
            peaks: Array of shape (N, 2) with (row, col) peak coordinates.

        Returns:
            Integer marker array with unique labels at peak positions.

        Note:
            Peaks are assumed to be within image bounds and objmask regions.
            No validation or filtering is performed.
        """
        markers = np.zeros_like(objmask, dtype=np.int32)

        # Place markers at peak positions
        for idx, (peak_r, peak_c) in enumerate(peaks, start=1):
            markers[peak_r, peak_c] = idx

        return markers

    def _operate(self, image: Image | GridImage) -> Image:
        """Separate touching colonies using intensity and distance-based peak detection.

        This method finds peaks based on intensity and distance transform, then uses
        watershed segmentation to separate touching colonies. For GridImages, peaks
        are constrained to one per grid cell. For regular Images, peaks are spaced
        by minimum distance.

        Returns:
            Image: Modified image with separated objmap and updated objmask.

        Raises:
            ValueError: If no peaks detected or watershed fails.
        """
        from scipy.ndimage import distance_transform_edt
        from skimage import segmentation
        import logging

        logger = logging.getLogger(__name__)

        objmask = image.objmask[:]
        logger.info(
                f"SeparateObjects: objmask shape={objmask.shape}, True pixels={objmask.sum()}")

        # Compute distance transform for peak detection
        distance = distance_transform_edt(objmask)
        logger.info(
                f"SeparateObjects: distance range=[{distance.min():.2f}, {distance.max():.2f}]")

        score_map = self._compute_peak_scores(objmask, distance)
        logger.info(
                f"SeparateObjects: score_map range=[{score_map.min():.2f}, {score_map.max():.2f}], non-zero={np.count_nonzero(score_map)}")

        # Peak detection: GridImage vs regular Image
        try:
            from phenotypic import GridImage

            is_gridimage = isinstance(image, GridImage)
        except ImportError:
            is_gridimage = False

        logger.info(f"SeparateObjects: is_gridimage={is_gridimage}")

        if is_gridimage:
            # Use grid boundaries to constrain peaks (one per cell)
            row_edges = np.round(image.grid.get_row_edges()).astype(
                    int)  # type: ignore[attr-defined]
            col_edges = np.round(image.grid.get_col_edges()).astype(
                    int)  # type: ignore[attr-defined]
            logger.info(
                    f"SeparateObjects: grid={len(row_edges) - 1}x{len(col_edges) - 1} cells")
            peaks = self._find_peaks_gridimage(score_map, objmask, row_edges, col_edges)
        else:
            # Use global peak detection with minimum distance
            peaks = self._find_peaks_regular(score_map, objmask, self.min_distance)

        logger.info(f"SeparateObjects: detected {len(peaks)} peaks")
        if len(peaks) > 0:
            logger.info(f"SeparateObjects: peak positions sample: {peaks[:5]}")

        # Create markers from peaks
        markers = self._create_markers_from_peaks(objmask, peaks)
        logger.info(
                f"SeparateObjects: markers max={markers.max()}, non-zero pixels={np.count_nonzero(markers)}")

        # Validate that at least one marker was created
        if markers.max() == 0:
            raise ValueError(
                    "No valid peaks detected - no local maxima found in score map. "
                    "Check that image has detected objects (objmask) and try adjusting "
                    "use_intensity/use_distance parameters or min_distance for regular Images."
            )

        logger.info(f"SeparateObjects: using {markers.max()} markers for watershed")

        # Compute elevation map for watershed segmentation
        # Use same metric as peak detection so markers are at local minima
        elevation = self._make_elevation_map(score_map)
        logger.info(
                f"SeparateObjects: elevation range=[{elevation.min():.2f}, {elevation.max():.2f}]")

        # Watershed segmentation
        objmap = segmentation.watershed(
                elevation,
                markers=markers,
                mask=objmask,  # Limit to detected regions only
        )

        n_regions = objmap.max()
        n_labeled_pixels = np.sum(objmap > 0)
        logger.info(
                f"SeparateObjects: watershed result - max_label={n_regions}, labeled_pixels={n_labeled_pixels}")

        # Validate result
        if objmap.max() == 0:
            n_markers = markers.max()
            raise ValueError(
                    f"Watershed produced empty result. Created {n_markers} markers but "
                    f"got 0 regions. Labeled {n_labeled_pixels} pixels. "
                    f"objmask has {objmask.sum()} True pixels."
            )

        # Convert to proper dtype
        if objmap.dtype != image._OBJMAP_DTYPE:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        # Update image with separated map
        # NOTE: Do NOT call relabel() - it would merge touching regions back together,
        # undoing the watershed separation. Watershed already produces properly labeled regions.
        image.objmap[:] = objmap

        gc.collect()

        return image
