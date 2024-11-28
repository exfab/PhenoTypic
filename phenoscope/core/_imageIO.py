from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from pathlib import Path
import numpy as np

from ._imageShow import ImageShow


LABEL_COLOR_ARRAY = 'color_array'
LABEL_ARRAY = 'array'
LABEL_ENHANCED_ARRAY = 'enhanced_array'
LABEL_OBJECT_MASK = 'object_mask'
LABEL_OBJECT_MAP = 'object_map'
LABEL_CUSTOM_FILE_EXTENSION_PREFIX = 'ps'

class ImageIO(ImageShow):
    def imread(self, filepath: str):
        input_img = imread(Path(filepath))
        if input_img.ndim == 3:
            self.color_array = input_img
        elif input_img.ndim == 2:
            self.array = input_img

    # TODO: Add option to save color image
    def imsave(self, filepath:Path):
        fpath = Path(filepath)
        # if self.color_array is not None:
        #     imsave(fname=fpath, arr=self.color_array, check_contrast=False)
        # else:
        imsave(fname=fpath, arr=img_as_ubyte(self.array), check_contrast=False)

    # TODO: Implement way to save metadata
    def savez(self, savepath:Path, save_metadata:bool=False):
        """
        Provides a way to save the current image object and the progress in the current pipeline. This method preserves the array data
        compared to saving it as an image filetype such as jpg or png.
        :param savepath: (pathlike) The filepath where to save the current image object as a phenoscope-created npz file.
        :return:
        """
        if savepath is None: raise ValueError(f'savepath not specified.')

        savepath = Path(savepath)
        savepath = savepath.parent / f'{savepath.stem}.{LABEL_CUSTOM_FILE_EXTENSION_PREFIX}.npz'
        save_dict = {}

        if self.color_array is not None: save_dict[LABEL_COLOR_ARRAY] = self.color_array

        if self.array is not None: save_dict[LABEL_ARRAY] = self.array

        if self.enhanced_array is not None: save_dict[LABEL_ENHANCED_ARRAY] = self.enhanced_array

        if self.object_mask is not None: save_dict[LABEL_OBJECT_MASK] = self.object_mask

        if self.object_map is not None: save_dict[LABEL_OBJECT_MAP] = self.object_map

        if save_dict:
            np.savez(savepath, **save_dict)
        else:
            raise AttributeError(f'Image has no data to save.')

    def loadz(self, filepath:Path):
        """
        Imports the data from a phenoscope-created numpy array.
        :param filepath: (Pathlike) points to where the
        :return:
        """
        if filepath is None: raise ValueError(f'filepath not specified.')

        fpath = Path(filepath)
        if f'.{LABEL_CUSTOM_FILE_EXTENSION_PREFIX}.npz' not in fpath.name:
            raise ValueError(f'File is does not contain .{LABEL_CUSTOM_FILE_EXTENSION_PREFIX}.npz at the end, and cannot be read as phenoscope-created npz file.')


        data = np.load(filepath)
        keys = data.keys()
        if LABEL_COLOR_ARRAY in keys: self.color_array = data[LABEL_COLOR_ARRAY]
        if LABEL_ARRAY in keys: self.array = data[LABEL_ARRAY]
        if LABEL_OBJECT_MASK in keys: self.object_mask = data[LABEL_OBJECT_MASK]
        if LABEL_OBJECT_MAP in keys: self.object_map = data[LABEL_OBJECT_MAP]

        # Numpy loads have to be closed at the end
        data.close()