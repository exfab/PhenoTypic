import os
from pathlib import Path

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from skimage.io import imread


def load_plate_12hr():
    """Returns a plate image of a K. Marxianus colony 96 array plate at 12 hrs"""
    return imread(current_file_dir / 'StandardDay1.jpg')


def load_plate_72hr():
    """Return's an image of a k. marxianus colony 96 array plate at 72 hrs"""
    return imread(current_file_dir / 'StandardDay6.jpg')


def load_plate_series():
    """Return's a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(current_file_dir / 'PlateSeries')
    fnames.sort()
    for fname in fnames:
        series.append(imread(current_file_dir / 'PlateSeries' / fname))
    return series


def load_colony_12_hr():
    return imread(current_file_dir / 'StdDay1-Results/well_imgs/StdDay1_well_3.png')


def load_faint_colony_12hr():
    return imread(current_file_dir / 'StdDay1-Results/well_imgs/StdDay1_well_15.png')


def load_colony_72hr():
    """Returns a colony image array of K. Marxianus"""
    return imread(current_file_dir / 'StdDay6-Results/well_imgs/StdDay6_well005.png')

def load_smear_plate_12hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(current_file_dir/'difficult/1_1S_16.jpg')

def load_smear_plate_24hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(current_file_dir/'difficult/2_2Y_6.jpg')
