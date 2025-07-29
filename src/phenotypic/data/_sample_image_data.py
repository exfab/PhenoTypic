import os
from pathlib import Path
from typing import List, Literal, Union

__current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from skimage.io import imread
import numpy as np

from phenotypic import Image, GridImage


# TODO: Update filepaths for this file

def _image_loader(filepath, mode: Literal['array', 'Image', 'GridImage']) -> Union[np.array, Image, GridImage]:
    match mode:
        case 'array':
            return imread(filepath)
        case 'Image':
            return Image.imread(filepath)
        case 'GridImage':
            return GridImage.imread(filepath)
        case _:
            return imread(filepath)


def load_plate_12hr() -> np.array:
    """Returns a plate image of a K. Marxianus colony 96 array plate at 12 hrs"""
    return imread(__current_file_dir / 'StandardDay1.jpg')


def load_plate_72hr() -> np.array:
    """Return a image of a k. marxianus colony 96 array plate at 72 hrs"""
    return imread(__current_file_dir / 'StandardDay6.jpg')


def load_plate_series() -> List[np.array]:
    """Return a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(__current_file_dir / 'PlateSeries')
    fnames.sort()
    for fname in fnames:
        series.append(imread(__current_file_dir / 'PlateSeries' / fname))
    return series


def load_early_colony() -> np.array:
    """Returns a colony image array of K. Marxianus at 12 hrs"""
    return imread(__current_file_dir / 'early_colony.png')


def load_faint_early_colony():
    """Returns a faint colony image array of K. Marxianus at 12 hrs"""
    return imread(__current_file_dir / 'early_colony_faint.png')


def load_colony():
    """Returns a colony image array of K. Marxianus at 72 hrs"""
    return imread(__current_file_dir / 'later_colony.png')


def load_smear_plate_12hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(__current_file_dir / 'difficult/1_1S_16.jpg')


def load_smear_plate_24hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(__current_file_dir / 'difficult/2_2Y_6.jpg')


def load_lactose_series(mode: Literal['array', 'Image', 'GridImage'] = 'array') -> List[Union[np.array, Image, GridImage]]:
    """Return a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(__current_file_dir / 'lactose')
    fnames.sort()
    for fname in fnames:
        filepath = __current_file_dir / 'lactose' / fname
        series.append(_image_loader(filepath, mode))
    return series
