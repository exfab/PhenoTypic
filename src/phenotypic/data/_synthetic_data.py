from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Literal, Tuple, Union

import numpy as np

import phenotypic
from phenotypic import Image
from phenotypic.data._sample_image_data import __current_file_dir


# --- Helper Functions ---

def _perlin_noise(
        h: int, w: int, scales: Iterable[int], rng: np.random.Generator
) -> np.ndarray:
    """Generates normalized perlin-like noise in [0, 1]."""
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
    """Generates a soft, textured radial mask for a colony."""
    yy, xx = np.mgrid[0:h, 0:w]
    # Calculate angles for radial noise
    theta = np.arctan2(yy - cy, xx - cx)
    ntheta = 512
    ang = np.linspace(-math.pi, math.pi, ntheta, endpoint=False)

    # Add radial noise to the radius
    radial_noise = 0.08 * rng.standard_normal(ntheta).astype(np.float32)
    r_lookup = base_r * (
            1.0 + np.interp(theta, ang, radial_noise, period=2 * math.pi)
    )

    # Distance field and soft edge
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edge_soft = max(base_r * 0.05, 1.0)
    t = (r_lookup - d) / edge_soft
    mask = np.clip(0.5 * (np.tanh(t) + 1.0), 0.0, 1.0)

    # Apply internal texture
    tex = _perlin_noise(h, w, scales=(32, 16, 8), rng=rng)
    return np.clip(mask * (0.85 + 0.15 * tex), 0.0, 1.0)


def _screen_blend(
        bg_img: np.ndarray, mask: np.ndarray, colony_rgb: np.ndarray
) -> np.ndarray:
    """Applies a screen-like blend to ensure colony is lighter than agar."""
    # Ensure colony color is broadcastable
    col = np.clip(colony_rgb, 0.86, 0.99)
    if col.ndim == 1:
        col = col[None, None, :]

    # Screen blend: 1 - (1 - A) * (1 - B)
    colony_region = 1.0 - (1.0 - bg_img) * (1.0 - col)

    # Linear interpolation based on mask
    return bg_img * (1.0 - mask) + colony_region * mask


def _quantize(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """Clips and converts float image to target bit depth."""
    img = np.clip(img, 0.0, 1.0)
    if bit_depth == 8:
        return (img * 255.0 + 0.5).astype(np.uint8)
    elif bit_depth == 16:
        return (img * 65535.0 + 0.5).astype(np.uint16)
    else:
        raise ValueError("bit_depth must be 8 or 16")


def _create_agar_background(
        h: int, w: int, agar_rgb: Tuple[float, float, float], rng: np.random.Generator
) -> np.ndarray:
    """Creates the base agar background with mild texture."""
    agar = np.array(agar_rgb, dtype=np.float32)
    # Different scales for plate vs single colony?
    # Using the larger scales (from make_synthetic_plate) covers both reasonably well
    bg_tex = 0.025 * (_perlin_noise(h, w, scales=(128, 64, 32), rng=rng) - 0.5)
    return np.clip(agar[None, None, :] + bg_tex[..., None], 0.0, 1.0)


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
    Generates a mask for filamentous hyphae extending from a center point.
    Uses domain-warped polar coordinates to create wavy, branching structures.
    """
    # Optimization: Work within a bounding box to save processing time
    # Filaments extend further than the base colony
    max_reach = base_r * reach_factor
    y_min, y_max = max(0, int(cy - max_reach)), min(h, int(cy + max_reach))
    x_min, x_max = max(0, int(cx - max_reach)), min(w, int(cx + max_reach))

    sh, sw = y_max - y_min, x_max - x_min
    if sh <= 0 or sw <= 0:
        return np.zeros((h, w), dtype=np.float32)

    # Local grid generation
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    dy, dx = yy - cy, xx - cx
    dist = np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx)

    # 1. Domain Warping: Distort the angle 'theta' with noise
    # This makes the filaments wiggle instead of radiating straight out.
    # We use a small local noise patch for speed.
    noise = _perlin_noise(sh, sw, scales=(int(sh / 4), int(sh / 8)), rng=rng)

    # Distortion increases with distance (straight at root, chaotic at tips)
    warp_strength = np.clip((dist - base_r * 0.5) / base_r, 0.0, 2.0)
    warped_theta = theta + (noise - 0.5) * 1.5 * warp_strength

    # 2. Filament Generation: High-frequency interference pattern
    # Combining two sine waves creates a pattern that looks like branching/interference
    primary = np.sin(density * warped_theta)
    secondary = np.sin(density * 2.5 * warped_theta + rng.uniform(0, 6))

    # Combine and threshold to get thin strands
    raw_signal = (primary + 0.6 * secondary) / 1.6
    strands = np.clip((raw_signal - 0.5) * 3.0, 0.0, 1.0)

    # 3. Radial Masking (Fade out)
    # Mask out the center (solid colony handles this) and fade the tips
    d_norm = (dist - base_r) / (max_reach - base_r)
    radial_fade = np.clip(1.0 - d_norm, 0.0, 1.0)

    # Soften the connection to the core
    core_blend = np.clip((dist - base_r * 0.8) / (base_r * 0.2), 0.0, 1.0)

    # Combine
    mask_slice = strands * radial_fade * core_blend

    # Add random noise texture to the filaments themselves so they aren't flat blocks
    mask_slice *= (0.7 + 0.3 * noise)

    # Place slice back into full accumulator
    full_mask = np.zeros((h, w), dtype=np.float32)
    full_mask[y_min:y_max, x_min:x_max] = mask_slice

    return full_mask


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

    # 3. Mask Generation
    m = _radial_colony_mask(h, w, cy, cx, r, rng)[..., None]

    # 4. Blending
    col_rgb = np.array(colony_rgb, dtype=np.float32)
    img = _screen_blend(img, m, col_rgb)

    # 5. Output
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
    """Generate a synthetic array plate with multiple colonies arranged in a grid."""
    rng = np.random.default_rng(seed)

    # 1. Background
    img = _create_agar_background(plate_h, plate_w, agar_rgb, rng)

    # 2. Grid Geometry
    margin_y = plate_h / (nrows + 1)
    margin_x = plate_w / (ncols + 1)
    spacing_y = plate_h / (nrows + 1)
    spacing_x = plate_w / (ncols + 1)
    base_r = min(spacing_y, spacing_x) * spacing_factor * 0.5

    col_rgb = np.array(colony_rgb, dtype=np.float32)

    # 3. Iterate and Blend Colonies
    for row in range(nrows):
        for col_idx in range(ncols):
            # Center position with jitter
            cy = margin_y + row * spacing_y + rng.uniform(-spacing_y * 0.05,
                                                          spacing_y * 0.05)
            cx = margin_x + col_idx * spacing_x + rng.uniform(-spacing_x * 0.05,
                                                              spacing_x * 0.05)

            # Radius with variation
            r = base_r * (1.0 + rng.uniform(-colony_size_variation,
                                            colony_size_variation))

            # Generate Mask
            m = _radial_colony_mask(plate_h, plate_w, cy, cx, r, rng)[..., None]

            # Blend into existing image
            img = _screen_blend(img, m, col_rgb)

    # 4. Output
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
        filament_density: float = 120.0,
        filament_reach: float = 2.5,
) -> np.ndarray:
    """
    Generate a synthetic plate with filamentous (hairy/branching) fungal colonies.

    Args:
        nrows: Number of rows.
        ncols: Number of columns.
        plate_h: Image height.
        plate_w: Image width.
        bit_depth: 8 or 16.
        colony_rgb: Tint of the fungal mass.
        agar_rgb: Background color.
        seed: Random seed.
        spacing_factor: 0-1 control of colony grid spacing.
        colony_size_variation: 0-1 variation in base colony size.
        filament_density: Controls the number/tightness of branches.
                          ~50 is loose (rhizoid), ~150 is dense (fuzzy mould).
        filament_reach: How far filaments extend relative to the colony core radius.

    Returns:
        np.ndarray: The generated image array.
    """
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

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

    # 3. Generate Colonies
    for row in range(nrows):
        for col_idx in range(ncols):
            # Coordinates
            cy = margin_y + row * spacing_y + rng.uniform(-spacing_y * 0.05,
                                                          spacing_y * 0.05)
            cx = margin_x + col_idx * spacing_x + rng.uniform(-spacing_x * 0.05,
                                                              spacing_x * 0.05)

            # Size variation
            r = base_r_global * (1.0 + rng.uniform(-colony_size_variation,
                                                   colony_size_variation))

            # A. Solid Core Mask (The dense center of the fungus)
            # We make the core slightly smaller (0.7x) so filaments appear to dominate
            core_mask = _radial_colony_mask(plate_h, plate_w, cy, cx, r * 0.7, rng)

            # B. Filament Mask (The hairy edges)
            # Vary density slightly per colony for realism
            local_density = filament_density * rng.uniform(0.9, 1.1)
            fil_mask = _filament_mask(
                    plate_h, plate_w, cy, cx, r * 0.7,
                    density=local_density,
                    reach_factor=filament_reach,
                    rng=rng
            )

            # Combine masks: Union of Core and Filaments
            combined_mask = np.maximum(core_mask, fil_mask)

            # Apply to image
            img = _screen_blend(img, combined_mask[..., None], col_rgb)

    return _quantize(img, bit_depth)


# --- Loaders ---

def load_synthetic_colony(
        mode: Literal["array", "Image"] = "array",
) -> Union[np.ndarray, Image]:
    """
    Loads synthetic colony data from a pre-saved file.
    """
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


def load_synth_plate():
    """Returns a phenotypic.GridImage of a synthetic plate with the colonies detected"""
    import phenotypic
    from skimage.io import imread

    dirpath = __current_file_dir / "synthetic_test_plate"

    image = phenotypic.GridImage.imread(
            filepath=dirpath / "circular_detect_plate_rgb.tif"
    )
    image.objmap[:] = imread(dirpath / "circular_detect_plate_objmap.png")
    image.name = "Synthetic96PlateWithObjects"
    return image
