from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Literal, Tuple, Union

import numpy as np

import phenotypic
from phenotypic import Image
from phenotypic.data._sample_image_data import __current_file_dir


# --- Global Helper Functions ---

def _perlin_noise(
        h: int, w: int, scales: Iterable[int], rng: np.random.Generator
) -> np.ndarray:
    """
    Generates normalized perlin-like noise in [0, 1].
    Refactored from _perlin_like to be a standalone helper.
    """
    acc = np.zeros((h, w), dtype=np.float32)
    total = 0.0
    for s in scales:
        gh, gw = max(1, h // s), max(1, w // s)
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
        acc += (a * (1 - wx) + b * wx) * (1 - wy)[:, None] + (
                c * (1 - wx) + d * wx
        ) * wy[:, None]
        total += 1.0
    acc = acc / max(total, 1e-6)
    return (acc - acc.min()) / (np.ptp(acc) + 1e-6)


def _radial_colony_mask(
        h: int, w: int, cy: float, cx: float, base_r: float, rng: np.random.Generator
) -> np.ndarray:
    """Generates the soft, textured mask for the solid colony core."""
    yy, xx = np.mgrid[0:h, 0:w]

    # Polar coordinates
    dy, dx = yy - cy, xx - cx
    d = np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx)

    # Radial noise (irregular radius)
    ntheta = 512
    ang = np.linspace(-math.pi, math.pi, ntheta, endpoint=False)
    radial_noise = 0.08 * rng.standard_normal(ntheta).astype(np.float32)
    r_lookup = base_r * (
            1.0 + np.interp(theta, ang, radial_noise, period=2 * math.pi)
    )

    # Soft edge thresholding
    edge_soft = max(base_r * 0.05, 1.0)
    t = (r_lookup - d) / edge_soft
    mask = np.clip(0.5 * (np.tanh(t) + 1.0), 0.0, 1.0)

    # Internal texture
    tex = _perlin_noise(h, w, scales=(32, 16, 8), rng=rng)
    return np.clip(mask * (0.85 + 0.15 * tex), 0.0, 1.0)


def _filament_mask(
        h: int,
        w: int,
        cy: float,
        cx: float,
        base_r: float,
        density: float,
        reach_factor: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates a mask for filamentous hyphae with anti-aliasing and solid connections.
    """
    # 1. Define Bounding Box
    max_reach = base_r * reach_factor
    pad = 10
    y_min = max(0, int(cy - max_reach - pad))
    y_max = min(h, int(cy + max_reach + pad))
    x_min = max(0, int(cx - max_reach - pad))
    x_max = min(w, int(cx + max_reach + pad))

    sh, sw = y_max - y_min, x_max - x_min
    if sh <= 0 or sw <= 0:
        return np.zeros((h, w), dtype=np.float32)

    # 2. Local Grid
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    dy, dx = yy - cy, xx - cx
    dist = np.sqrt(dy ** 2 + dx ** 2)
    dist = np.maximum(dist, 1e-5)  # Avoid div by zero
    theta = np.arctan2(dy, dx)

    # 3. Domain Warping (Large scales = gentle curves, no jagged static)
    noise = _perlin_noise(sh, sw, scales=(int(sh / 1.5), int(sh / 3)), rng=rng)

    warp_strength = np.clip((dist - base_r * 0.5) / base_r, 0.0, 1.0)
    warped_theta = theta + (noise - 0.5) * 0.5 * warp_strength

    # 4. Anti-Aliased Density
    # Limit max frequency by circumference to prevent dots/aliasing
    max_freq = (2 * np.pi * dist) / 3.0
    safe_density = np.minimum(density, max_freq)

    # 5. Continuous Waveform (Connected Branches)
    signal = np.cos(warped_theta * safe_density)

    # Threshold to create strands
    strand_width = 0.8
    strands = np.clip((signal - strand_width) * 5.0, 0.0, 1.0)

    # 6. Radial Masking (Hard Fade)
    d_norm = (dist - base_r * 0.5) / (max_reach - base_r * 0.5)
    # Power of 2 cleans up fringe noise
    radial_fade = np.clip(1.0 - d_norm, 0.0, 1.0) ** 2.0

    # Soft connection to core
    core_blend = np.clip((dist - base_r * 0.6) / (base_r * 0.2), 0.0, 1.0)

    mask_slice = strands * radial_fade * core_blend

    # Texture
    mask_slice *= (0.8 + 0.2 * noise)

    full_mask = np.zeros((h, w), dtype=np.float32)
    full_mask[y_min:y_max, x_min:x_max] = mask_slice

    return full_mask


def _create_agar_background(
        h: int, w: int, agar_rgb: Tuple[float, float, float], rng: np.random.Generator
) -> np.ndarray:
    """Creates the textured agar background."""
    agar = np.array(agar_rgb, dtype=np.float32)
    # Scaled noise for texture
    bg_noise = _perlin_noise(h, w, scales=(128, 64, 32), rng=rng)
    bg_tex = 0.025 * (bg_noise - 0.5)
    return np.clip(agar[None, None, :] + bg_tex[..., None], 0.0, 1.0)


def _screen_blend(
        bg_img: np.ndarray, mask: np.ndarray, colony_rgb: np.ndarray
) -> np.ndarray:
    """Helper for screen blending colonies (ensures they are lighter than bg)."""
    col = np.clip(colony_rgb, 0.86, 0.99)
    if col.ndim == 1:
        col = col[None, None, :]

    colony_region = 1.0 - (1.0 - bg_img) * (1.0 - col)
    return bg_img * (1.0 - mask) + colony_region * mask


def _quantize(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """Converts float [0,1] image to uint8/uint16."""
    img = np.clip(img, 0.0, 1.0)
    if bit_depth == 8:
        return (img * 255.0 + 0.5).astype(np.uint8)
    elif bit_depth == 16:
        return (img * 65535.0 + 0.5).astype(np.uint16)
    else:
        raise ValueError("bit_depth must be 8 or 16")


# --- Main Generators ---

def make_synthetic_colony(
        h: int = 256,
        w: int = 256,
        bit_depth: int = 8,
        colony_rgb: Tuple[float, float, float] = (0.96, 0.88, 0.82),
        agar_rgb: Tuple[float, float, float] = (0.55, 0.56, 0.54),
        seed: int = 1,
) -> np.ndarray:
    """Generate a single bright fungal colony on solid-media agar."""
    rng = np.random.default_rng(seed)

    # 1. Background
    img = _create_agar_background(h, w, agar_rgb, rng)

    # 2. Geometry
    cy, cx = h * 0.5, w * 0.5
    r = min(h, w) * 0.35

    # 3. Mask
    m = _radial_colony_mask(h, w, cy, cx, r, rng)[..., None]

    # 4. Blend
    col_rgb = np.array(colony_rgb, dtype=np.float32)
    img = _screen_blend(img, m, col_rgb)

    return _quantize(img, bit_depth)


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
    """Generate a synthetic array plate with multiple circular colonies."""
    rng = np.random.default_rng(seed)

    # 1. Background
    img = _create_agar_background(plate_h, plate_w, agar_rgb, rng)

    # 2. Grid Setup
    margin_y = plate_h / (nrows + 1)
    margin_x = plate_w / (ncols + 1)
    spacing_y = plate_h / (nrows + 1)
    spacing_x = plate_w / (ncols + 1)
    base_r = min(spacing_y, spacing_x) * spacing_factor * 0.5
    col_rgb = np.array(colony_rgb, dtype=np.float32)

    # 3. Iterate
    for row in range(nrows):
        for col_idx in range(ncols):
            cy = margin_y + row * spacing_y + rng.uniform(-spacing_y * 0.05,
                                                          spacing_y * 0.05)
            cx = margin_x + col_idx * spacing_x + rng.uniform(-spacing_x * 0.05,
                                                              spacing_x * 0.05)
            r = base_r * (1.0 + rng.uniform(-colony_size_variation,
                                            colony_size_variation))

            m = _radial_colony_mask(plate_h, plate_w, cy, cx, r, rng)[..., None]
            img = _screen_blend(img, m, col_rgb)

    return _quantize(img, bit_depth)


def make_synthetic_filamentous_plate(
        nrows: int = 8,
        ncols: int = 12,
        plate_h: int = 2048,
        plate_w: int = 3072,
        bit_depth: int = 8,
        colony_rgb: Tuple[float, float, float] = (0.96, 0.90, 0.85),
        agar_rgb: Tuple[float, float, float] = (0.55, 0.56, 0.54),
        seed: int = 1,
        spacing_factor: float = 0.85,
        colony_size_variation: float = 0.15,
        filament_density: float = 100.0,
        filament_reach: float = 2.2,
) -> np.ndarray:
    """
    Generate a synthetic plate with filamentous (hairy/branching) colonies.
    """
    rng = np.random.default_rng(seed)

    # 1. Background
    img = _create_agar_background(plate_h, plate_w, agar_rgb, rng)

    # 2. Grid Setup
    margin_y = plate_h / (nrows + 1)
    margin_x = plate_w / (ncols + 1)
    spacing_y = plate_h / (nrows + 1)
    spacing_x = plate_w / (ncols + 1)
    base_r_global = min(spacing_y, spacing_x) * spacing_factor * 0.5
    col_rgb = np.array(colony_rgb, dtype=np.float32)

    # 3. Iterate
    for row in range(nrows):
        for col_idx in range(ncols):
            cy = margin_y + row * spacing_y + rng.uniform(-spacing_y * 0.05,
                                                          spacing_y * 0.05)
            cx = margin_x + col_idx * spacing_x + rng.uniform(-spacing_x * 0.05,
                                                              spacing_x * 0.05)
            r = base_r_global * (1.0 + rng.uniform(-colony_size_variation,
                                                   colony_size_variation))

            # Core
            core_mask = _radial_colony_mask(plate_h, plate_w, cy, cx, r * 0.6, rng)

            # Filaments
            local_density = filament_density * rng.uniform(0.9, 1.1)
            fil_mask = _filament_mask(
                    plate_h, plate_w, cy, cx, r * 0.6,
                    density=local_density,
                    reach_factor=filament_reach,
                    rng=rng
            )

            # Combine
            combined_mask = np.maximum(core_mask, fil_mask)
            img = _screen_blend(img, combined_mask[..., None], col_rgb)

    return _quantize(img, bit_depth)


# --- Loaders ---

def load_synthetic_colony(
        mode: Literal["array", "Image"] = "array",
) -> Union[np.ndarray, Image]:
    """Loads synthetic colony data from a pre-saved file."""
    from phenotypic import Image

    data = np.load(
            Path(os.path.relpath(__current_file_dir / "synthetic_colony.npz",
                                 Path.cwd()))
    )
    match mode:
        case "array":
            return data["array"]
        case "Image":
            image = Image(data["array"])
            image.objmask[:] = data["objmask"]
            return image
        case _:
            raise ValueError("Invalid mode")


def load_synth_yeast_plate():
    """Returns a phenotypic.GridImage of a synthetic plate with the colonies detected"""
    import phenotypic
    from skimage.io import imread

    dirpath = __current_file_dir / "synthetic_test_plate"

    image = phenotypic.GridImage.imread(
            filepath=dirpath / "yeast_plate_rgb.png"
    )
    image.objmap[:] = imread(dirpath / "yeast_plate_objmap.png")
    image.name = "Synthetic96PlateWithObjects"
    return image


def load_synth_filamentous_plate():
    import phenotypic
    from skimage.io import imread

    dirpath = __current_file_dir / "synthetic_test_plate"

    image = phenotypic.GridImage.imread(
            filepath=dirpath / "filamentous_plate_rgb.png"
    )
    image.objmap[:] = imread(dirpath / "filamentous_plate_objmap.png")
    image.name = "Synthetic96PlateWithObjects"
    return image
