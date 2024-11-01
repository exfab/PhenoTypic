import numpy as np

from phenoscope.grid import GriddedImage
from phenoscope.grid.interface import GridMapModifier
from phenoscope.grid.feature_extraction import GridLinRegStatsExtractor


class MinResidualErrorModifier(GridMapModifier):
    """
    This map modifier removes objects from sctions where there are multiple based on their distance from the linreg predicted location.
    This modifier is relatively slow, but shows good results in removing the correct obj when paired with small object removers and other filters.
    """
    #TODO: Add a setting to retain a certain number of objects in the event of removal

    def _operate(self, image: GriddedImage) -> GriddedImage:
        # Get the section objects in order of most amount. More objects in a section means
        # more potential spread that can affect linreg results.
        section_obj_counts = image.get_section_count(ascending=False)

        # Since single object sections can't reduce further, we isolate multi object sections to reduce runtime
        multi_obj_section_nums = section_obj_counts[section_obj_counts > 1].index.to_list()

        # Initialize extractor here to save obj construction time
        linreg_stat_extractor = GridLinRegStatsExtractor()

        # Iterate through multiple object sections
        for section_n in multi_obj_section_nums:
            # Get the current object map. Put inside loop so that each iteration, we get the newest updated map
            obj_map = image.object_map

            # Set the stat_extractor section to curr section
            linreg_stat_extractor.section_num = section_n

            # Get curr section info
            section_info = linreg_stat_extractor.extract(image)

            # Isolate the object id with the smallest residual error
            min_err_obj_id = section_info.loc[:, linreg_stat_extractor.LABEL_RESIDUAL_ERR].idxmin()

            # Isolate which objects within the section should be dropped
            obj_to_drop = section_info.index.drop(min_err_obj_id).to_numpy()

            # Remove obj from obj map copy by setting it to the background number (0)
            obj_map[np.isin(obj_map, obj_to_drop)] = 0

            # Set image.obj_map to the new edited obj_map.
            image.object_map = obj_map

        return image
