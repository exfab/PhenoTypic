from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Union

import pandas as pd

__current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from skimage.io import imread

import math
from typing import Iterable, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING: from phenotypic import Image, GridImage


# TODO: Update filepaths for this file

def _image_loader(filepath, mode: Literal['array', 'Image', 'GridImage']) -> Union[np.array, Image, GridImage]:
    from phenotypic import Image, GridImage

    match mode:
        case 'array':
            return imread(filepath)
        case 'Image':
            return Image.imread(filepath)
        case 'GridImage':
            return GridImage.imread(filepath)
        case _:
            return imread(filepath)


def make_synthetic_colony(
        h: int = 256,
        w: int = 256,
        bit_depth: int = 8,
        colony_rgb: Tuple[float, float, float] = (0.96, 0.88, 0.82),
        agar_rgb: Tuple[float, float, float] = (0.55, 0.56, 0.54),
        seed: int = 1,
) -> np.ndarray:
    """Generate a single bright fungal colony on solid-media agar. Returns an RGB NumPy array.

    Args:
        h: Image height (pixels).
        w: Image width (pixels).
        bit_depth: 8 or 16.
        colony_rgb: Linear RGB in [0,1] for colony tint. Will be forced lighter than agar.
        agar_rgb: Linear RGB in [0,1] for agar background.
        seed: RNG seed.

    Returns:
        np.ndarray: HxWx3 RGB, dtype uint8 or uint16.

    Notes:
        - Colony is lighter than background via screen-like blend.
        - No Petri dish. Scene is a cropped colony with padding on agar.
    """
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    rng = np.random.default_rng(seed)

    def _perlin_like(h: int, w: int, scales: Iterable[int]) -> np.ndarray:
        acc = np.zeros((h, w), dtype=np.float32);
        total = 0.0
        for s in scales:
            gh, gw = max(1, h//s), max(1, w//s)
            g = rng.random((gh + 1, gw + 1)).astype(np.float32)
            y = np.linspace(0, gh, h, endpoint=False);
            x = np.linspace(0, gw, w, endpoint=False)
            y0 = np.floor(y).astype(int);
            x0 = np.floor(x).astype(int)
            y1 = np.clip(y0 + 1, 0, gh);
            x1 = np.clip(x0 + 1, 0, gw)
            wy = y - y0;
            wx = x - x0
            a = g[y0[:, None], x0[None, :]];
            b = g[y0[:, None], x1[None, :]]
            c = g[y1[:, None], x0[None, :]];
            d = g[y1[:, None], x1[None, :]]
            acc += ((a*(1 - wx) + b*wx)*(1 - wy)[:, None] +
                    (c*(1 - wx) + d*wx)*wy[:, None])
            total += 1.0
        acc = acc/max(total, 1e-6)
        return (acc - acc.min())/(np.ptp(acc) + 1e-6)

    def _colony_mask(h: int, w: int, cy: float, cx: float, base_r: float) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w]
        theta = np.arctan2(yy - cy, xx - cx)
        ntheta = 512
        ang = np.linspace(-math.pi, math.pi, ntheta, endpoint=False)
        radial_noise = 0.08*rng.standard_normal(ntheta).astype(np.float32)
        r_lookup = base_r*(1.0 + np.interp(theta, ang, radial_noise, period=2*math.pi))
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        edge_soft = max(base_r*0.05, 1.0)
        t = (r_lookup - d)/edge_soft
        mask = np.clip(0.5*(np.tanh(t) + 1.0), 0.0, 1.0)
        tex = _perlin_like(h, w, scales=(32, 16, 8))
        return np.clip(mask*(0.85 + 0.15*tex), 0.0, 1.0)

    # Agar background with mild texture
    agar = np.array(agar_rgb, dtype=np.float32)
    bg_tex = 0.025*(_perlin_like(h, w, scales=(64, 32)) - 0.5)
    bg = np.clip(agar[None, None, :] + bg_tex[..., None], 0.0, 1.0)

    # Colony placement
    cy, cx = h*0.5, w*0.5
    r = min(h, w)*0.35
    m = _colony_mask(h, w, cy, cx, r)[..., None]

    # Colony color, forced light
    col = np.array(colony_rgb, dtype=np.float32)
    col = np.clip(col, 0.86, 0.99)

    # Screen-like blending inside colony mask to guarantee lighter-than-agar
    colony_region = 1.0 - (1.0 - bg)*(1.0 - col[None, None, :])
    img = bg*(1.0 - m) + colony_region*m

    # Quantize
    img = np.clip(img, 0.0, 1.0)
    if bit_depth == 8:
        return (img*255.0 + 0.5).astype(np.uint8)
    else:
        return (img*65535.0 + 0.5).astype(np.uint16)


def load_synthetic_colony(mode: Literal['array', 'Image'] = 'array') -> Union[np.ndarray, Image]:
    """
    Loads synthetic colony data from a pre-saved file and returns it in the specified mode.

    This function provides two modes for handling the synthetic colony data: 'array' and 'Image'.
    Depending on the mode specified, it either returns the array directly or converts it into an
    Image object. When 'Image' mode is selected, the object mask is also applied to the Image object.

    Args:
        mode (Literal['array', 'Image']): Specifies the format in which the synthetic colony
            data should be returned. Use 'array' to return the raw data as an array or 'Image'
            to return an Image object with the corresponding objmask.

    Returns:
        Union[np.ndarray, Image]: The synthetic colony data, either as a numpy array or an
        Image object, depending on the specified mode.

    Raises:
        ValueError: If the mode is neither 'array' nor 'Image'.
    """
    from phenotypic import Image

    data = np.load(__current_file_dir/'synthetic_colony.npz')
    match mode:
        case 'array':
            return data['array']
        case 'Image':
            image = Image(data['array'])
            image.objmask[:] = data['objmask']
            return image
        case _:
            raise ValueError('Invalid mode')


def make_synthetic_plate(
        nrows: int = 8,
        ncols: int = 12,
        plate_h: int = 2048,
        plate_w: int = 3072,
        bit_depth: int = 8,
        colony_rgb: Tuple[float, float, float] = (0.96, 0.88, 0.82),
        agar_rgb: Tuple[float, float, float] = (0.55, 0.56, 0.54),
        seed: int = 1,
        spacing_factor: float = 0.85,
        colony_size_variation: float = 0.15,
) -> np.ndarray:
    """Generate a synthetic array plate with multiple colonies arranged in a grid.

    Args:
        nrows: Number of rows in the plate array (e.g., 8 for 96-well plate).
        ncols: Number of columns in the plate array (e.g., 12 for 96-well plate).
        plate_h: Total plate image height (pixels).
        plate_w: Total plate image width (pixels).
        bit_depth: 8 or 16.
        colony_rgb: Linear RGB in [0,1] for colony tint. Will be forced lighter than agar.
        agar_rgb: Linear RGB in [0,1] for agar background.
        seed: RNG seed for reproducibility.
        spacing_factor: Factor controlling spacing between colonies (0-1). Lower = more spacing.
        colony_size_variation: Random variation in colony sizes (0-1). 0 = uniform size.

    Returns:
        np.ndarray: plate_h x plate_w x 3 RGB array, dtype uint8 or uint16.

    Example:
        # Create a standard 96-well plate (8x12)
        plate = make_synthetic_plate(rows=8, cols=12, plate_h=2048, plate_w=3072)
        
        # Create a 384-well plate (16x24)
        plate = make_synthetic_plate(rows=16, cols=24, plate_h=2048, plate_w=3072)
    """
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    rng = np.random.default_rng(seed)

    def _perlin_like(h: int, w: int, scales: Iterable[int]) -> np.ndarray:
        acc = np.zeros((h, w), dtype=np.float32)
        total = 0.0
        for s in scales:
            gh, gw = max(1, h//s), max(1, w//s)
            g = rng.random((gh + 1, gw + 1)).astype(np.float32)
            y = np.linspace(0, gh, h, endpoint=False)
            x = np.linspace(0, gw, w, endpoint=False)
            y0 = np.floor(y).astype(int)
            x0 = np.floor(x).astype(int)
            y1 = np.clip(y0 + 1, 0, gh)
            x1 = np.clip(x0 + 1, 0, gw)
            wy = y - y0
            wx = x - x0
            a = g[y0[:, None], x0[None, :]]
            b = g[y0[:, None], x1[None, :]]
            c = g[y1[:, None], x0[None, :]]
            d = g[y1[:, None], x1[None, :]]
            acc += ((a*(1 - wx) + b*wx)*(1 - wy)[:, None] +
                    (c*(1 - wx) + d*wx)*wy[:, None])
            total += 1.0
        acc = acc/max(total, 1e-6)
        return (acc - acc.min())/(np.ptp(acc) + 1e-6)

    def _colony_mask(h: int, w: int, cy: float, cx: float, base_r: float) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w]
        theta = np.arctan2(yy - cy, xx - cx)
        ntheta = 512
        ang = np.linspace(-math.pi, math.pi, ntheta, endpoint=False)
        radial_noise = 0.08*rng.standard_normal(ntheta).astype(np.float32)
        r_lookup = base_r*(1.0 + np.interp(theta, ang, radial_noise, period=2*math.pi))
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        edge_soft = max(base_r*0.05, 1.0)
        t = (r_lookup - d)/edge_soft
        mask = np.clip(0.5*(np.tanh(t) + 1.0), 0.0, 1.0)
        tex = _perlin_like(h, w, scales=(32, 16, 8))
        return np.clip(mask*(0.85 + 0.15*tex), 0.0, 1.0)

    # Create agar background with texture
    agar = np.array(agar_rgb, dtype=np.float32)
    bg_tex = 0.025*(_perlin_like(plate_h, plate_w, scales=(128, 64, 32)) - 0.5)
    bg = np.clip(agar[None, None, :] + bg_tex[..., None], 0.0, 1.0)

    # Calculate grid spacing
    margin_y = plate_h/(nrows + 1)
    margin_x = plate_w/(ncols + 1)
    spacing_y = plate_h/(nrows + 1)
    spacing_x = plate_w/(ncols + 1)

    # Base colony radius
    base_r = min(spacing_y, spacing_x)*spacing_factor*0.5

    # Colony color, forced light
    col = np.array(colony_rgb, dtype=np.float32)
    col = np.clip(col, 0.86, 0.99)

    # Create mask for all colonies
    img = bg.copy()

    for row in range(nrows):
        for col_idx in range(ncols):
            # Calculate center position
            cy = margin_y + row*spacing_y
            cx = margin_x + col_idx*spacing_x

            # Add small random offset
            cy += rng.uniform(-spacing_y*0.05, spacing_y*0.05)
            cx += rng.uniform(-spacing_x*0.05, spacing_x*0.05)

            # Vary colony size
            r = base_r*(1.0 + rng.uniform(-colony_size_variation, colony_size_variation))

            # Generate colony mask
            m = _colony_mask(plate_h, plate_w, cy, cx, r)[..., None]

            # Apply colony with screen-like blending
            colony_region = 1.0 - (1.0 - img)*(1.0 - col[None, None, :])
            img = img*(1.0 - m) + colony_region*m

    # Quantize
    img = np.clip(img, 0.0, 1.0)
    if bit_depth == 8:
        return (img*255.0 + 0.5).astype(np.uint8)
    else:
        return (img*65535.0 + 0.5).astype(np.uint16)


# Example:
# img = make_synthetic_colony(h=384, w=384, bit_depth=16)
# plate = make_synthetic_plate(nrows=8, ncols=12, plate_h=2048, plate_w=3072)

def load_plate_12hr() -> np.array:
    """Returns a plate image of a K. Marxianus colony 96 array plate at 12 hrs"""
    return imread(__current_file_dir/'StandardDay1.jpg')


def load_plate_72hr() -> np.array:
    """Return a image of a k. marxianus colony 96 array plate at 72 hrs"""
    return imread(__current_file_dir/'StandardDay6.jpg')


def load_plate_series() -> List[np.array]:
    """Return a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(__current_file_dir/'PlateSeries')
    fnames.sort()
    for fname in fnames:
        series.append(imread(__current_file_dir/'PlateSeries'/fname))
    return series


def load_early_colony() -> np.array:
    """Returns a colony image array of K. Marxianus at 12 hrs"""
    return imread(__current_file_dir/'early_colony.png')


def load_faint_early_colony():
    """Returns a faint colony image array of K. Marxianus at 12 hrs"""
    return imread(__current_file_dir/'early_colony_faint.png')


def load_colony():
    """Returns a colony image array of K. Marxianus at 72 hrs"""
    return imread(__current_file_dir/'later_colony.png')


def load_smear_plate_12hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(__current_file_dir/'difficult/1_1S_16.jpg')


def load_smear_plate_24hr():
    """Returns a plate image array of K. Marxianus that contains noise such as smears"""
    return imread(__current_file_dir/'difficult/2_2Y_6.jpg')


def load_lactose_series(mode: Literal['array', 'Image', 'GridImage'] = 'array') -> List[
    Union[np.array, Image, GridImage]]:
    """Return a series of plate images across 6 time samples"""
    series = []
    fnames = os.listdir(__current_file_dir/'lactose')
    fnames.sort()
    for fname in fnames:
        filepath = __current_file_dir/'lactose'/fname
        series.append(_image_loader(filepath, mode))
    return series


def load_meas() -> pd.DataFrame:
    return pd.read_csv(__current_file_dir/'meas/meas.csv', index_col=0)
