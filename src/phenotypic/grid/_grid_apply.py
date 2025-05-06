from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

from phenotypic.abstract import ImageOperation
from phenotypic.pipeline import ImagePipeline


class GridApply:
    """Accepts a PhenoTypic operation as a parameter and applies it to the individual grid sectionss of an image.

    Parameters:
        image_op (ImageOperation): A PhenoTypic operation to be applied to each grid section.
        reset_enh_matrix (bool): Whether to reset the enh_matrix attribute of the image before applying the operation.
    """

    def __init__(self, image_op: ImageOperation | ImagePipeline,
                 reset_enh_matrix: bool = True):
        self.operation = image_op
        self.reset_enh_matrix = reset_enh_matrix

    def apply(self, image: GridImage):
        row_edges = image.grid.get_row_edges()
        col_edges = image.grid.get_col_edges()
        for row_i in range(len(row_edges) - 1):
            for col_i in range(len(col_edges) - 1):
                subimage = image[
                           row_edges[row_i]:row_edges[row_i + 1],
                           col_edges[col_i]:col_edges[col_i + 1]
                           ]
                self.operation.apply(subimage, inplace=True)

                image[
                row_edges[row_i]:row_edges[row_i + 1],
                col_edges[col_i]:col_edges[col_i + 1]
                ] = subimage

        return image
