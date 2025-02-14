import numpy as np

from phenoscope.grid import GriddedImage
from phenoscope.grid.interface import GridMapModifier
from phenoscope.grid.features import GridLinRegStatsExtractor


class MinResidualErrorModifier(GridMapModifier):
    """
    This map modifier removes objects from sctions where there are multiple based on their distance from the linreg predicted location.
    This modifier is relatively slow, but shows good results in removing the correct obj when paired with small object removers and other filters.
    """

    # TODO: Add a setting to retain a certain number of objects in the event of removal

    def _operate(self, image: GriddedImage) -> GriddedImage:
        # Get the section objects in order of most amount. More objects in a section means
        # more potential spread that can affect linreg results.
        max_iter = (image.n_rows * image.n_cols) * 4

        # Initialize extractor here to save obj construction time
        linreg_stat_extractor = GridLinRegStatsExtractor()

        # Get initial section obj count
        section_obj_counts = image.get_section_count(ascending=False)

        n_iters = 0
        # Check that there exist sections with more than one object
        while n_iters < max_iter and (section_obj_counts > 1).any():
            # Get the current object map. This is inside the loop to ensure latest version each iteration
            obj_map = image.object_map

            # Get the section idx with the most objects
            section_with_most_obj = section_obj_counts.idxmax()

            # Set the target_section for linreg_stat_extractor
            linreg_stat_extractor.section_num = section_with_most_obj

            # Get the section info
            section_info = linreg_stat_extractor.extract(image)

            # Isolate the object id with the smallest residual error
            min_err_obj_id = section_info.loc[:, linreg_stat_extractor.LABEL_RESIDUAL_ERR].idxmin()

            # Isolate which objects within the section should be dropped
            objects_to_drop = section_info.index.drop(min_err_obj_id).to_numpy()

            # Set the objects with the labels to the background value
            obj_map[np.isin(obj_map, objects_to_drop)] = 0

            image.object_map = obj_map

            # Reset section obj count and add counter
            section_obj_counts = image.get_section_count(ascending=False)
            n_iters += 1

        return image
