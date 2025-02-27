from phenoscope.grid import GriddedImage


class GridApply:
    """Accepts a phenoscope operation as a parameter and applies it to the individual grid's of an image."""

    def __init__(self, phenoscope_operation):
        self.operation = phenoscope_operation

    def apply(self, image: GriddedImage):
        row_edges = image.grid.get_row_edges()
        col_edges = image.grid.get_col_edges()
        for row_i in range(len(row_edges) - 1):
            for col_i in range(len(col_edges) - 1):
                subimage = image[
                           row_edges[row_i]:row_edges[row_i + 1],
                           col_edges[col_i]:col_edges[col_i + 1]
                           ]
                self.operation._operate(subimage, inplace=True)

                image.det_matrix[
                row_edges[row_i]:row_edges[row_i + 1],
                col_edges[col_i]:col_edges[col_i + 1]
                ] = subimage.det_matrix[:]

                image.obj_map[
                row_edges[row_i]:row_edges[row_i + 1],
                col_edges[col_i]:col_edges[col_i + 1]
                ] = subimage.obj_map[:]

        return image
