"""
This class adds image and measurements tracking using statuses.

The status will be attributes stored within each image's group similar to measurements.

Location:
    /phenotypic/image_sets/images/<image_name>/status

"""
import h5py
from ._image_set_core import ImageSetCore
class ImageSetStatus(ImageSetCore):
    def reset_status(self):
        with self._main_hdf.writer as handle:
            images_group = self._main_hdf(handle, self._)
