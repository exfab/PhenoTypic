"""Peter Kovesi's synthetic test images, for testing phase-congruency operators.

Ported from ``src/syntheticimages.jl`` of ImagePhaseCongruency.jl. The originals
describe themselves as images that "cause considerable grief for gradient based
operators", which is exactly why they are the right controls for a congruency
measure: an independent, adversarial ground truth authored by the algorithm's own
author rather than by us.

  Copyright (c) 2015-2017 Peter Kovesi -- peterkovesi.com

  MIT License:

  Permission is hereby granted, free of charge, to any person obtaining a copy of
  this software and associated documentation files (the "Software"), to deal in the
  Software without restriction, including without limitation the rights to use, copy,
  modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
  and to permit persons to whom the Software is furnished to do so, subject to the
  following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
  OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Faithfulness notes, in the order they bite:

  * Only ODD harmonics are summed (``1:2:(2*nscales-1)``). Even harmonics would break
    the half-wave symmetry that makes the feature type well defined.
  * ``ampexponent = -1`` sums ``1/k`` -> a square wave (step features). ``-2`` with
    ``offset = pi/2`` sums ``cos(k x)/k^2`` -> a triangle wave (line features).
  * Julia's ``[f(x, y) for x = l:u, y = l:u]`` puts ``x`` on the FIRST axis, so these
    use ``indexing="ij"`` and ``theta = arctan2(Y, X)`` with ``X`` the row coordinate.
    Getting this backwards transposes every image.
  * ``circsine``'s ``trim`` option is not ported: in the original it multiplies by
    ``(r < c) + (r >= c)``, which is identically 1.
  * ``noiseonf`` needs the radial frequency grid with ``DC = 0``, which is
    ``construct_filter_grids(...)[3]`` (``freq``), *not* ``[0]`` (``radius``, whose DC
    is fudged to 1). Reading index 0 would put the DC bin's radius at 65 instead of 1.
"""

from __future__ import annotations

import numpy as np

from phenotypic.enhance._monogenic_kernels import construct_filter_grids


def centred_axis(sze: int) -> np.ndarray:
    """Kovesi's ``l:u``: ``-sze/2 : sze/2-1`` when even, ``-(sze-1)/2 : (sze-1)/2`` when odd."""
    if sze % 2 == 0:
        return np.arange(-sze // 2, sze // 2, dtype=float)
    return np.arange(-(sze - 1) // 2, (sze - 1) // 2 + 1, dtype=float)


def step2line(sze: int = 512, *, nscales: int = 50, ampexponent: float = -1.0,
              ncycles: float = 1.5, phasecycles: float = 0.25) -> np.ndarray:
    """A phase-congruent image whose FEATURE TYPE sweeps step -> line down the rows.

    Every row is the same odd-harmonic series with a growing phase offset. The
    congruency points sit at fixed columns ``x = m*pi`` for every row: there each odd
    harmonic ``sin(k x + phi)`` has phase ``m*pi + phi``, so they align whatever ``phi``
    is. Congruency is therefore constant down the image while the feature morphs from a
    step into a line. Gradient magnitude is not.
    """
    x = np.arange(sze) / (sze - 1) * ncycles * 2 * np.pi
    offsets = phasecycles * 2 * np.pi * np.arange(sze) / sze
    img = np.zeros((sze, sze))
    for scale in range(1, 2 * nscales, 2):  # ODD harmonics only
        img += float(scale) ** ampexponent * np.sin(scale * x[None, :] + offsets[:, None])
    return img


def circsine(sze: int = 512, *, wavelength: float = 40.0, nscales: int = 50,
             ampexponent: float = -1.0, offset: float = 0.0, p: int = 2) -> np.ndarray:
    """Concentric circular waveform. Isophotes are exact circles: curvature ``1/r``, always."""
    if p % 2:
        raise ValueError("p should be an even number")
    ax = centred_axis(sze)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    r = (X ** p + Y ** p) ** (1.0 / p)
    img = np.zeros_like(r)
    for scale in range(1, 2 * nscales, 2):
        img += float(scale) ** ampexponent * np.sin(scale * r * 2 * np.pi / wavelength + offset)
    return img


def starsine(sze: int = 512, *, ncycles: float = 10.0, nscales: int = 50,
             ampexponent: float = -1.0, offset: float = 0.0) -> np.ndarray:
    """An angular waveform: radial rays at every orientation at once."""
    ax = centred_axis(sze)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    theta = np.arctan2(Y, X)  # Julia: atan(y, x), y on the SECOND axis
    img = np.zeros_like(theta)
    for scale in range(1, 2 * nscales, 2):
        img += float(scale) ** ampexponent * np.sin(scale * ncycles * theta + offset)
    return img


def noiseonf(sze: int, p: float, *, seed: int = 0) -> np.ndarray:
    """``1/f^p`` noise: random phase, amplitude spectrum replaced by ``1/radius^p``.

    The phase spectrum is pure noise, so there is no congruency anywhere -- the negative
    control for the Rayleigh threshold ``T``. ``p = 1.5`` is roughly the amplitude
    falloff of natural images.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.normal(size=(sze, sze)))
    magnitude = np.abs(spectrum)
    magnitude[magnitude == 0.0] = 1.0
    radius = construct_filter_grids(sze, sze)[3] * sze + 1.0
    return np.real(np.fft.ifft2((spectrum / magnitude) / radius ** p))


def unit_variance(img: np.ndarray) -> np.ndarray:
    """Zero mean, unit standard deviation. ``epsilon`` is absolute, so scale matters."""
    return (img - img.mean()) / img.std()
