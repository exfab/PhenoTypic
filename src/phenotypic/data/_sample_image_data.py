import os
from pathlib import Path

__current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from skimage.io import imread


# TODO: Update filepaths for this file

def load_plate_12hr():
    """Returns a plate image of a K. Marxianus colony 96 array plate at 12 hrs"""
    return imread(__current_file_dir / 'StandardDay1.jpg')


def load_plate_72hr():
    """Return a image of a k. marxianus colony 96 array plate at 72 hrs"""
    return imread(__current_file_dir / 'StandardDay6.jpg')


def load_plate_series():
    """Return a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(__current_file_dir / 'PlateSeries')
    fnames.sort()
    for fname in fnames:
        series.append(imread(__current_file_dir / 'PlateSeries' / fname))
    return series


def load_colony_12_hr():
    return imread(__current_file_dir / 'early_colony.png')


def load_faint_colony_12hr():
    return imread(__current_file_dir / 'early_colony_faint.png')


def load_colony_72hr():
    """Returns a colony image array of K. Marxianus"""
    return imread(__current_file_dir / 'later_colony.png')


def load_smear_plate_12hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(__current_file_dir / 'difficult/1_1S_16.jpg')


def load_smear_plate_24hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(__current_file_dir / 'difficult/2_2Y_6.jpg')
