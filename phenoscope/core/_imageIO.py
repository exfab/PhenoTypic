from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from pathlib import Path
import numpy as np
import os
import pandas as pd

from ._imageMeasurements import ImageMeasurements


LABEL_ARRAY = 'array'
LABEL_MATRIX = 'matrix'
LABEL_ENHANCED_MATRIX = 'enhanced_array'
LABEL_OBJECT_MASK = 'object_mask'
LABEL_OBJECT_MAP = 'object_map'
LABEL_METADATA_MATRIX = 'metadata_matrix'

LABEL_CUSTOM_FILE_EXTENSION_PREFIX = '.psnpz'
METADATA_RECARRAY = 'metadata_recarray'
MEASUREMENT_PREPEND = 'psnpz_measurement_'

class ImageIO(ImageMeasurements):
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

        if len(self.metadata)>0:
            save_dict[METADATA_RECARRAY] = self.metadata.to_recarray()

        if len(self.measurements)>0:
            measurement_dict = self.measurements.to_dict()
            renamed_measurements = {f'{MEASUREMENT_PREPEND}{key}': value for key, value in measurement_dict.items()}
            save_dict = {**save_dict, **renamed_measurements}

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

        if METADATA_RECARRAY in keys:
            metadata_recarray:np.recarray = data[METADATA_RECARRAY]
            metadata_keys = metadata_recarray.dtype.names
            metadata_values = metadata_recarray[0]
            for metadata_idx, metadata_key in enumerate(metadata_keys):
                self.metadata[metadata_key] = metadata_values[metadata_idx]

        # Searches for unique prepend in keyword names
        measurement_keys = [key for key in keys
                            if MEASUREMENT_PREPEND in key]
        if len(measurement_keys)>0:
            for measurement_key in measurement_keys:

                # Need to reconstruct dataframe from array
                table = pd.DataFrame(data=data[measurement_key])
                table.set_index(table.columns[0], inplace=True) # Index column should be the first column based on to_recarrays_dict() protocol.
                self.measurements[measurement_key.replace(f'{MEASUREMENT_PREPEND}',"")] = table # Removes prepend from measurement names

        # Numpy load has to be closed at the end
        data.close()
        return self

    def legacy_loadz(self, filepath:Path):
        """
           Imports the data from a phenoscope-created numpy array.
           :param filepath: (Pathlike) points to where the
           :return:
           """
        if filepath is None: raise ValueError(f'filepath not specified.')

        fpath = Path(filepath)
        if fpath.suffix == '.npz':
            raise Warning(f'File is an npz file and not a psnpz file. The Image data may not load properly.')

        elif fpath.suffix != LABEL_CUSTOM_FILE_EXTENSION_PREFIX:
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

    def _recarray_to_metadata(self, metadata_recarray:np.recarray):
        keys = metadata_recarray.dtype.names
        metadata_values = metadata_recarray[0]
        for key in keys:
            self.metadata[key] = metadata_values[key]


    def _load_metadata(self, metadata_matrix):
        for row in range(metadata_matrix.shape[0]):
            self.metadata[metadata_matrix[row, 0]] = [metadata_matrix[row, 1]]
