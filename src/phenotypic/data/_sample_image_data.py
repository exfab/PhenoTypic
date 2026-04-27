from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Union

import pandas as pd

__current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from skimage.io import imread

from typing import Iterable, TYPE_CHECKING
import numpy as np

from ._remote_registry import fetch_snp

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage


def _image_loader(
        filepath,
        mode: Literal["array", "Image", "GridImage", "filepath"],
        **kwargs
) -> Union[np.ndarray, Image, GridImage]:
    from phenotypic import Image, GridImage

    match mode:
        case "array":
            return imread(filepath)
        case "Image":
            return Image.imread(filepath, **kwargs)
        case "GridImage":
            return GridImage.imread(filepath, **kwargs)
        case "filepath":
            return filepath
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def load_plate_12hr(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    """Returns a plate image of a K. Marxianus colony 96 array plate at 12 hrs"""
    return _image_loader(__current_file_dir / "StandardDay1.jpg", mode)


def load_plate_72hr(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    """Return a image of a k. marxianus colony 96 array plate at 72 hrs"""
    return _image_loader(__current_file_dir / "StandardDay6.jpg", mode)


def load_plate_series(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> List[Union[np.ndarray, Image, GridImage]]:
    """Return a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(__current_file_dir / "PlateSeries")
    fnames.sort()
    for fname in fnames:
        filepath = __current_file_dir / "PlateSeries" / fname
        series.append(_image_loader(filepath, mode))
    return series


def load_early_colony(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    """Returns a colony image array of K. Marxianus at 12 hrs"""
    return _image_loader(__current_file_dir / "early_colony.png", mode)


def load_faint_early_colony(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    """Returns a faint colony image array of K. Marxianus at 12 hrs"""
    return _image_loader(__current_file_dir / "early_colony_faint.png", mode)


def load_colony(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    """Returns a colony image array of K. Marxianus at 72 hrs"""
    return _image_loader(__current_file_dir / "later_colony.png", mode)


def load_smear_plate_12hr(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return _image_loader(__current_file_dir / "difficult/1_1S_16.jpg", mode)


def load_smear_plate_24hr(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return _image_loader(__current_file_dir / "difficult/2_2Y_6.jpg", mode)


def load_lactose_series(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> List[Union[np.ndarray, Image, GridImage]]:
    """Return a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(__current_file_dir / "lactose")
    fnames.sort()
    for fname in fnames:
        filepath = __current_file_dir / "lactose" / fname
        series.append(_image_loader(filepath, mode))
    return series


def yield_sample_dataset(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Iterable[Union[np.ndarray, Image, GridImage]]:
    """Return a series of plate images across 6 time samples"""
    fnames = [
        x
        for x in os.listdir(__current_file_dir / "PhenoTypicSampleSubset")
        if x.endswith(".jpg")
    ]
    fnames.sort()
    for fname in fnames:
        filepath = __current_file_dir / "PhenoTypicSampleSubset" / fname
        yield _image_loader(filepath, mode)


def load_meas() -> pd.DataFrame:
    """
    Loads sample measurements for 3 strains using each of the measurement modules

    Returns:
        pd.DataFrame: A DataFrame containing the loaded measurement data.
    """
    return pd.read_csv(__current_file_dir / "meas/all_meas.csv", index_col=0)


def load_quickstart_meas() -> pd.DataFrame:
    return pd.read_csv(
            __current_file_dir / "meas/GettingStartedMeas.csv", index_col=0
    )


def load_area_meas() -> pd.DataFrame:
    """
    Loads sample measurements for 3 strains using area measurements

    Returns:
        pd.DataFrame: A DataFrame containing the sample area measurement data.
    """
    return pd.read_csv(__current_file_dir / "meas/area_meas.csv", index_col=0)


def load_imager_plate(
        mode: Literal["array", "Image", "GridImage"] = "array",
) -> Union[np.ndarray, Image, GridImage]:
    return _image_loader(__current_file_dir / "RHODOTORULA_RAW.cr3", mode=mode)


def load_yeast_plate(
        mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load an image Neurospora filamentous fungi 96-well plate image cropped to the center.

    Returns a pre-cropped plate image of Neurospora filamentous fungi arranged
    in a 96-well grid. Irregular hyphal morphology with spreading growth --
    suitable for filamentous fungi tutorials and how-to guides.

    Args:
        mode: Return format. Default: ``'GridImage'``.

    Returns:
        The plate image in the requested format.
    """
    return _image_loader(
            __current_file_dir / "snp-imager-samples/"
                                 "RhodotorulaYeastCenterCrop.png",
            mode=mode,
            nrows=2,
            ncols=4
    )


def load_yeast_plate_large(
        mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a cropped Rhodotorula yeast 96-well plate image.

    Returns a pre-cropped plate image of Rhodotorula yeast colonies arranged
    in a standard 96-well grid (8 rows x 12 columns). Round colony morphology
    with clean background -- suitable as the default sample for most tutorials
    and how-to guides.

    Args:
        mode: Return format. Default: ``'GridImage'``.

    Returns:
        The plate image in the requested format.
    """
    return _image_loader(
            fetch_snp("RhodotorulaYeastCropped.png"), mode
    )


def load_fungi_plate(
        mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load an image Neurospora filamentous fungi 96-well plate image cropped to the center.

    Returns a pre-cropped plate image of Neurospora filamentous fungi arranged
    in a 96-well grid. Irregular hyphal morphology with spreading growth --
    suitable for filamentous fungi tutorials and how-to guides.

    Args:
        mode: Return format. Default: ``'GridImage'``.

    Returns:
        The plate image in the requested format.
    """
    return _image_loader(
            __current_file_dir / "snp-imager-samples/"
                                 "NeurosporaFilamentousFungiCenterCrop.png",
            mode=mode,
            nrows=2,
            ncols=4
    )


def load_fungi_plate_large(
        mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a cropped Neurospora filamentous fungi 96-well plate image.

    Returns a pre-cropped plate image of Neurospora filamentous fungi arranged
    in a 96-well grid. Irregular hyphal morphology with spreading growth --
    suitable for filamentous fungi tutorials and how-to guides.

    Args:
        mode: Return format. Default: ``'GridImage'``.

    Returns:
        The plate image in the requested format.
    """
    return _image_loader(
            fetch_snp("NeurosporaFilamentousFungiCropped.png"),
            mode,
            nrows=6,
            ncols=10
    )


def load_yeast_plate_full(
        mode: Literal["array", "Image", "GridImage"] = "Image",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a full (uncropped) Rhodotorula yeast plate image with color checker.

    Returns the full plate image including scanner margins and a color checker
    target. Use for tutorials and how-to guides that demonstrate cropping
    (``ImageCropper``), padding (``ImagePadder``), vignetting correction
    (``VignetteCorrector``), or color correction (``ColorCorrector``).

    Args:
        mode: Return format. Default: ``'Image'``.

    Returns:
        The full plate image in the requested format.
    """
    return _image_loader(
            fetch_snp("RhodotorulaYeastFullPlate.png"), mode
    )


def load_fungi_plate_full(
        mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a full (uncropped) Neurospora filamentous fungi plate image with color checker.

    Returns the full plate image including scanner margins and a color checker
    target. Use for fungi-specific correction workflow tutorials.

    Args:
        mode: Return format. Default: ``'Image'``.

    Returns:
        The full plate image in the requested format.
    """
    return _image_loader(
            fetch_snp("NeurosporaFilamentousFungiFullPlate.png"),
            mode,
            nrows=6, ncols=10
    )
