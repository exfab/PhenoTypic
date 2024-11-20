from skimage.io import imread, imsave
from pathlib import Path

from ._imageShow import ImageShow


class ImageIO(ImageShow):
    def imread(self, filepath: str):
        input_img = imread(Path(filepath))
        if input_img.ndim == 3:
            self.color_array = input_img
        elif input_img.ndim == 2:
            self.array = input_img

    def imsave(self, filepath):
        fpath = Path(filepath)
        if self.color_array is not None:
          imsave(fname=fpath, arr=self.color_array)
        else:
            imsave(fname=fpath, arr=self.array)