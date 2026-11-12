from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Union

import pandas as pd

__current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from skimage.io import imread

from typing import TYPE_CHECKING
import numpy as np

from ._remote_registry import fetch_snp

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage


def _image_loader(
    filepath,
    mode: Literal["array", "Image", "GridImage", "filepath"],
    **kwargs,
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


def load_meas() -> pd.DataFrame:
    """
    Loads sample measurements for 3 strains using each of the measurement modules

    This example data predates 0.20.0: ``Size_ConvexArea`` and ``Shape_Solidity``
    are blank because their values were computed with the retired hull-perimeter
    bug, and the ``Shape_*BoundaryDist`` values used the whole-plate distance
    transform rather than each colony's own.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded measurement data.
    """
    return pd.read_csv(__current_file_dir / "meas/all_meas.csv", index_col=0)


def load_area_meas() -> pd.DataFrame:
    """
    Loads sample measurements for 3 strains using area measurements

    This example data predates 0.20.0: ``Size_ConvexArea`` and ``Shape_Solidity``
    are blank because their values were computed with the retired hull-perimeter
    bug, and the ``Shape_*BoundaryDist`` values used the whole-plate distance
    transform rather than each colony's own.

    Returns:
        pd.DataFrame: A DataFrame containing the sample area measurement data.
    """
    return pd.read_csv(__current_file_dir / "meas/area_meas.csv", index_col=0)


def load_yeast_plate(
    mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a yeast plate image cropped to the center.

    Returns a pre-cropped plate image of yeast colonies on a
    2 x 4 grid (``nrows=2``, ``ncols=4``) -- suitable for yeast tutorials and
    how-to guides.

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
        ncols=4,
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
    return _image_loader(fetch_snp("RhodotorulaYeastCropped.png"), mode)


def load_fungi_plate(
    mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a Neurospora filamentous fungi plate image cropped to the center.

    Returns a pre-cropped plate image of Neurospora filamentous fungi on a
    2 x 4 grid (``nrows=2``, ``ncols=4``). Irregular hyphal morphology with
    spreading growth -- suitable for filamentous fungi tutorials and how-to
    guides.

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
        ncols=4,
    )


def load_fungi_plate_large(
    mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a cropped Neurospora filamentous fungi plate image.

    Returns a pre-cropped plate image of Neurospora filamentous fungi on a
    6 x 10 grid (``nrows=6``, ``ncols=10``). Irregular hyphal morphology with
    spreading growth -- suitable for filamentous fungi tutorials and how-to
    guides.

    Args:
        mode: Return format. Default: ``'GridImage'``.

    Returns:
        The plate image in the requested format.
    """
    return _image_loader(
        fetch_snp("NeurosporaFilamentousFungiCropped.png"),
        mode,
        nrows=6,
        ncols=10,
    )


def load_yeast_plate_full(
    mode: Literal["array", "Image", "GridImage"] = "Image",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a full (uncropped) Rhodotorula yeast plate image with color checker.

    Returns the full plate image including scanner margins and a color checker
    target. Use for tutorials and how-to guides that demonstrate cropping
    (``CropImage``), padding (``PadImage``), or color correction
    (``ColorCorrector``).

    Args:
        mode: Return format. Default: ``'Image'``.

    Returns:
        The full plate image in the requested format.
    """
    return _image_loader(fetch_snp("RhodotorulaYeastFullPlate.png"), mode)


def load_fungi_plate_full(
    mode: Literal["array", "Image", "GridImage"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage]:
    """Load a full (uncropped) Neurospora filamentous fungi plate image with color checker.

    Returns the full plate image including scanner margins and a color checker
    target. Use for fungi-specific correction workflow tutorials.

    Args:
        mode: Return format. Default: ``'GridImage'``.

    Returns:
        The full plate image in the requested format.
    """
    return _image_loader(
        fetch_snp("NeurosporaFilamentousFungiFullPlate.png"),
        mode,
        nrows=6,
        ncols=10,
    )


def load_bacteria_plate_full(
    mode: Literal["array", "Image", "GridImage", "filepath"] = "GridImage",
) -> Union[np.ndarray, Image, GridImage, str]:
    """Load a full bacterial colony plate image, already color-corrected and denoised.

    Returns a whole 8 x 12 pinned plate of mature bacterial colonies, imaged
    with the plate holder, the plate wall and color-checker strips at the
    left and right edges. The colonies vary in size, pigment (cream, yellow,
    orange) and surface (smooth, wrinkled, rough). Some grid positions are
    empty by design, so a full detection finds 54 colonies, not 96.

    The image is a 16-bit RGB TIFF (4012 x 6016 pixels, about 100 MB). It is a
    TIFF rather than a PNG because 16-bit RGB PNGs are read back as 8-bit. It
    has already been color-corrected and denoised, so it is ready input for
    :class:`~phenotypic.prefab.BacterSamPipeline`, whose default crop fits
    this frame. Use it for bacterial detection tutorials and for tests that
    need a real plate rather than a synthetic one.

    The file is downloaded from the ``data-v1`` GitHub Release on first use
    and cached; see ``PHENOTYPIC_DATA_DIR`` to change the cache location.

    Args:
        mode: Return format. ``'GridImage'`` (the default) returns a
            :class:`GridImage` with an 8 x 12 grid. ``'Image'`` returns a
            plain :class:`Image`, ``'array'`` the ``uint16`` pixel array of
            shape ``(4012, 6016, 3)``, and ``'filepath'`` the cached file's
            path.

    Returns:
        The full plate image in the requested format.
    """
    path = fetch_snp("BacteriaFullPlate.tif")
    if mode == "GridImage":
        return _image_loader(path, mode, nrows=8, ncols=12)
    return _image_loader(path, mode)
