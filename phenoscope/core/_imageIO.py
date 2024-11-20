from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from pathlib import Path

from ._imageShow import ImageShow


class ImageIO(ImageShow):
    def imread(self, filepath: str):
        input_img = imread(Path(filepath))
        if input_img.ndim == 3:
            self.color_array = input_img
        elif input_img.ndim == 2:
            self.array = input_img

    # TODO: Add option to save color image
    def imsave(self, filepath):
        fpath = Path(filepath)
        # if self.color_array is not None:
        #     imsave(fname=fpath, arr=self.color_array, check_contrast=False)
        # else:
        imsave(fname=fpath, arr=img_as_ubyte(self.array), check_contrast=False)
