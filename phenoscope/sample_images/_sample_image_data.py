import os
from pathlib import Path

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from .. import imread


def PlateDay1():
    return imread(current_file_dir / 'StandardDay1.jpg')


def PlateDay6():
    return imread(current_file_dir / 'StandardDay6.jpg')


def PlateSeries():
    series = []
    fnames = os.listdir(current_file_dir / 'PlateSeries')
    fnames.sort()
    for fname in fnames:
        series.append(imread(current_file_dir / 'PlateSeries' / fname))
    return series


def EarlyColony():
    return imread(current_file_dir / 'StdDay1-Results/well_imgs/StdDay1_well_3.png')


def FaintColony():
    return imread(current_file_dir / 'StdDay1-Results/well_imgs/StdDay1_well_15.png')


def Colony():
    return imread(current_file_dir / 'StdDay6-Results/well_imgs/StdDay6_well005.png')

def SmearPlateDay1():
    return imread(current_file_dir/'difficult/1_1S_16.jpg')

def SmearPlateDay2():
    return imread(current_file_dir/'difficult/2_2Y_6.jpg')
