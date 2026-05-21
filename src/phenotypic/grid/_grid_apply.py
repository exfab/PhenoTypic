from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import GridCorrector
from phenotypic.tools_.typing_ import OperationField


class GridApply(GridCorrector):
    """Apply a PhenoTypic operation to each grid section of a plate image.

    ``GridApply`` slices a :class:`GridImage` into its individual grid
    sections (wells), applies the wrapped operation to each section in
    isolation, and writes the transformed sections back into the parent
    image. It is the building block prefab pipelines use to run a
    per-well detection or enhancement sub-pipeline.

    Args:
        image_op: The operation (or nested :class:`~phenotypic.ImagePipeline`)
            applied to every grid section. Serialized as a class-tagged
            payload so the concrete operation type round-trips through
            JSON.
        reset_enh_matrix: Whether to reset the ``detect_mat`` attribute of
            each section before applying ``image_op``. Defaults to True.

    Returns:
        GridImage: The input image with every grid section transformed by
        ``image_op``.
    """

    image_op: OperationField
    reset_enh_matrix: bool = True

    def _operate(self, image: GridImage) -> GridImage:
        """Apply ``image_op`` to every grid section of ``image``.

        Args:
            image: The grid image whose sections are transformed.

        Returns:
            GridImage: The image with each section transformed in place.

        Raises:
            RuntimeError: If ``image_op`` raises while processing a
                section; the offending ``(row, col)`` index is reported.
        """
        row_edges = image.grid.get_row_edges()
        col_edges = image.grid.get_col_edges()
        for row_i in range(len(row_edges) - 1):
            for col_i in range(len(col_edges) - 1):
                subimage = image[
                    row_edges[row_i] : row_edges[row_i + 1],
                    col_edges[col_i] : col_edges[col_i + 1],
                ]
                try:
                    self.image_op.apply(subimage, inplace=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Error applying operation to section {row_i, col_i}: {e}"
                    )

                image[
                    row_edges[row_i] : row_edges[row_i + 1],
                    col_edges[col_i] : col_edges[col_i + 1],
                ] = subimage

        return image


GridApply.apply.__doc__ = GridApply._operate.__doc__
