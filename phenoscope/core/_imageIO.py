from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from pathlib import Path
import numpy as np
import os
import pandas as pd

from ._imageMetadata import ImageMetadata


LABEL_ARRAY = 'color_array'
LABEL_MATRIX = 'array'
LABEL_ENHANCED_MATRIX = 'enhanced_array'
LABEL_OBJECT_MASK = 'object_mask'
LABEL_OBJECT_MAP = 'object_map'
LABEL_CUSTOM_FILE_EXTENSION_PREFIX = '.psnpz'
LABEL_METADATA_MATRIX = 'metadata_matrix'

class ImageIO(ImageMetadata):
    def imread(self, filepath: str):
        input_img = imread(Path(filepath))
        if input_img.ndim == 3:
            self.array = input_img
        elif input_img.ndim == 2:
            self.matrix = input_img

    # TODO: Add option to save color image
    def imsave(self, filepath:Path):
        fpath = Path(filepath)
        # if self.color_array is not None:
        #     imsave(fname=fpath, arr=self.color_array, check_contrast=False)
        # else:
        imsave(fname=fpath, arr=img_as_ubyte(self.matrix), check_contrast=False)

    # TODO: Implement way to save metadata
    def savez(self, savepath:Path):
        """
        Provides a way to save the current image object and the progress in the current pipeline. This method preserves the array data
        compared to saving it as an image filetype such as jpg or png.
        :param savepath: (pathlike) The filepath where to save the current image object as a phenoscope-created npz file.
        :return:
        """
        if savepath is None: raise ValueError(f'savepath not specified.')

        savepath = Path(savepath)

        # Checks if the file already exists to prevent overwriting (deprecated)
        # if savepath.exists():
        #     num_matching_files = pd.Series(os.listdir(savepath.parent)).str.contains(savepath.stem).sum()
        #     savepath = savepath.parent / (f'{savepath.stem}({num_matching_files})' + savepath.suffix)

        temp_savepath = savepath.parent/(savepath.stem + '.npz')
        if str(savepath).endswith(f'{LABEL_CUSTOM_FILE_EXTENSION_PREFIX}') is False:
            savepath = savepath.parent /( f'{savepath.stem}' + LABEL_CUSTOM_FILE_EXTENSION_PREFIX)

        save_dict = {}

        if self.array is not None: save_dict[LABEL_ARRAY] = self.array

        if self.matrix is not None: save_dict[LABEL_MATRIX] = self.matrix

        if self.enhanced_matrix is not None: save_dict[LABEL_ENHANCED_MATRIX] = self.enhanced_matrix

        if self.object_mask is not None: save_dict[LABEL_OBJECT_MASK] = self.object_mask

        if self.object_map is not None: save_dict[LABEL_OBJECT_MAP] = self.object_map

        if self._metadata:
            save_dict[LABEL_METADATA_MATRIX] = np.array(tuple(zip(
                self._metadata.keys(),
                self._metadata.values(),
                self._metadata_dtype.values()
            )))

        if save_dict:
            np.savez(temp_savepath, **save_dict)
            os.rename(temp_savepath, str(savepath))
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
        if fpath.suffix == '.npz':
            raise Warning(f'File is an npz file and not a psnpz file. The Image data may not load properly.')

        elif fpath.suffix!=LABEL_CUSTOM_FILE_EXTENSION_PREFIX:
            raise ValueError(f'File is not a {LABEL_CUSTOM_FILE_EXTENSION_PREFIX} file, and cannot be interpreted.')


        data = np.load(fpath)
        keys = data.keys()
        if LABEL_ARRAY in keys: self.array = data[LABEL_ARRAY]
        if LABEL_MATRIX in keys: self.matrix = data[LABEL_MATRIX]
        if LABEL_ENHANCED_MATRIX in keys: self.enhanced_matrix = data[LABEL_ENHANCED_MATRIX]
        if LABEL_OBJECT_MASK in keys: self.object_mask = data[LABEL_OBJECT_MASK]
        if LABEL_OBJECT_MAP in keys: self.object_map = data[LABEL_OBJECT_MAP]
        if LABEL_METADATA_MATRIX in keys: self._load_metadata(data[LABEL_METADATA_MATRIX])

        # Numpy load has to be closed at the end
        data.close()
        return self

    def _load_metadata(self, metadata_matrix):
        self._metadata = dict(zip(metadata_matrix[:, 0], metadata_matrix[:, 1]))
        self._metadata_dtype = dict(zip(metadata_matrix[:, 0], metadata_matrix[:, 2]))
        self.validate_metadata_dtype()
