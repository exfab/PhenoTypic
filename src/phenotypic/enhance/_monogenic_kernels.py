"""Shared frequency-domain kernels for the phase-congruency operations.

Pure functions with no :class:`~phenotypic._core._image.Image` dependency, so they
are unit-testable without fixtures. Used by :class:`FocusEdgePhase` (Kovesi's
``phasecong3``) and :class:`FocusEdgeMonogenicPhase` (Kovesi's ``phasecongmono``).

References:
    Peter Kovesi, ``ImagePhaseCongruency.jl`` (Julia) and ``MatlabFns/PhaseCongruency``
    (MATLAB). The MIT-licensed ``phasepack`` is a third, independent transcription.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

#: Division guard for ``phasecongmono``. All three references agree on ``1e-4``:
#: Julia ``phasecongruency.jl`` line 441, MATLAB ``phasecongmono.m`` line 153,
#: ``phasepack/phasecongmono.py`` line 129.
#:
#: **Do not unify this with** :class:`FocusEdgePhase`'s ``1e-5``. That value belongs
#: to *Julia's* ``phasecong3`` (``phasecongruency.jl`` line 1272); MATLAB's
#: ``phasecong3.m`` line 144 uses ``1e-4``, and ``FocusEdgePhase`` is a port of the
#: Julia.
EPSILON_MONOGENIC = 1e-4


def construct_filter_grids(
        rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct frequency domain grids for filter construction.

    Grids are quadrant-shifted so the DC component is at ``[0, 0]``. Every axis is
    divided by ``N``, matching **both** of Kovesi's implementations
    (``frequencyfilt.jl`` l.73, ``filtergrid.m`` l.49). ``k/N`` is the true DFT bin
    frequency.

    ``phasepack`` differs at odd sizes -- ``linspace(-0.5, 0.5, N, endpoint=True)`` in
    its ``filtergrid``, and ``/(N - 1)`` in its ``lowpassfilter`` -- which is a
    ``phasepack`` bug, not a Kovesi divergence. All three agree at even sizes, which is
    why the golden fixture is 64x64.

    Args:
        rows: Number of rows in image.
        cols: Number of columns in image.

    Returns:
        Tuple of ``(radius, sintheta, costheta, freq, fx, fy)`` where:

        - ``radius``: radial frequency with ``DC = 1`` so ``log(radius)`` is safe
        - ``sintheta``: ``fx / freq``, the angular filter's sine grid
        - ``costheta``: ``fy / freq``, the angular filter's cosine grid
        - ``freq``: radial frequency with ``DC = 0``
        - ``fx``, ``fy``: the raw signed frequency grids, for :func:`riesz_multiplier`

    Note:
        ``radius`` is bit-equal to the ``freq_safe`` used to build ``sintheta``/``costheta``
        -- both are ``freq`` with ``[0, 0]`` set to 1. So ``sintheta == fx / radius``
        exactly, and :func:`riesz_multiplier` can take ``(fx, fy, radius)`` and divide once.
    """
    if cols % 2 == 1:  # odd
        fx_range = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / cols
    else:  # even
        fx_range = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2 == 1:  # odd
        fy_range = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / rows
    else:  # even
        fy_range = np.arange(-rows / 2, rows / 2) / rows

    fx_range = ifftshift(fx_range)
    fy_range = ifftshift(fy_range)

    fx, fy = np.meshgrid(fx_range, fy_range)

    freq = np.sqrt(fx ** 2 + fy ** 2)

    radius = freq.copy()
    radius[0, 0] = 1.0

    freq_safe = freq.copy()
    freq_safe[0, 0] = 1.0
    sintheta = fx / freq_safe
    costheta = fy / freq_safe

    sintheta[0, 0] = 0.0
    costheta[0, 0] = 0.0

    return radius, sintheta, costheta, freq, fx, fy


def lowpass_filter(
        radius: np.ndarray, cutoff: float = 0.45, order: int = 15
) -> np.ndarray:
    """Butterworth lowpass, Kovesi's ``lowpassfilter(size, 0.45, 15)``.

    Args:
        radius: Radial frequency grid.
        cutoff: Normalized cutoff frequency.
        order: Butterworth order; the exponent is ``2 * order``.

    Returns:
        Lowpass transfer function, same shape as ``radius``.
    """
    return 1.0 / (1.0 + (radius / cutoff) ** (2 * order))


def log_gabor_scale(
        radius: np.ndarray, lowpass: np.ndarray, wavelength: float, sigma_onf: float
) -> np.ndarray:
    """One log-Gabor radial bandpass, lowpassed, with the DC bin zeroed.

    Args:
        radius: Radial frequency grid with ``DC = 1``.
        lowpass: Precomputed :func:`lowpass_filter` output.
        wavelength: Centre wavelength in pixels; the centre frequency is its reciprocal.
        sigma_onf: Ratio of the filter's Gaussian sigma to its centre frequency.

    Returns:
        The transfer function, same shape as ``radius``.
    """
    f0 = 1.0 / wavelength

    with np.errstate(divide="ignore", invalid="ignore"):
        log_rad_over_f0 = np.log(radius / f0)

    log_gabor = np.exp(-(log_rad_over_f0 ** 2) / (2 * np.log(sigma_onf) ** 2))
    log_gabor[0, 0] = 0.0

    return log_gabor * lowpass


def log_gabor_radial(
        radius: np.ndarray,
        n_scale: int,
        min_wavelength: float,
        mult: float,
        sigma_onf: float,
) -> List[np.ndarray]:
    """Construct log-Gabor filters for each scale.

    Log-Gabor filters have Gaussian transfer functions on a logarithmic frequency
    scale, providing a constant shape ratio across scales.

    Args:
        radius: Radial frequency grid with ``DC = 1``.
        n_scale: Number of scales.
        min_wavelength: Wavelength of the finest scale, in pixels.
        mult: Wavelength multiplier between successive scales.
        sigma_onf: Ratio of the filter's Gaussian sigma to its centre frequency.

    Returns:
        List of ``n_scale`` filter arrays.
    """
    lowpass = lowpass_filter(radius)
    return [
        log_gabor_scale(radius, lowpass, min_wavelength * (mult ** s), sigma_onf)
        for s in range(n_scale)
    ]


def riesz_multiplier(
        fx: np.ndarray, fy: np.ndarray, radius: np.ndarray
) -> np.ndarray:
    """Kovesi's ``packedmonogenicfilters``: ``H = (i*fx - fy)/radius``.

    Packs both odd (Riesz) channels into one complex array, so a single ``ifft2``
    yields ``h1`` in the real part and ``h2`` in the imaginary part.

    **Divide each component by ``radius``, because that is what the references compute.**
    All three source texts *look* alike -- ``phasecongmono.m`` l.183 ``H = (1i*u1 - u2)./radius``,
    ``frequencyfilt.jl`` l.238 ``H = (im.*fx .- fy)./f``, ``phasepack/phasecongmono.py``
    l.156 ``H = (1j * u1 - u2) / radius`` -- but the three languages disagree *below* the
    source text, so transcribing the glyphs is not transcribing the arithmetic.

    - MATLAB's ``./`` and Julia's ``/(z::Complex, x::Real)`` (``base/complex.jl`` l.348,
      ``Complex(real(z)/x, imag(z)/x)``) perform a **true division per component**.
    - numpy's ``complex128 / float64`` promotes the denominator and runs ``nc_quot``. With a
      zero imaginary denominator that branch reduces to ``scl = 1/r`` followed by a
      **multiply**, which is not the same rounding. Verified bit-exactly: ``(1j*a - b)/r``
      equals ``(a*(1/r))*1j - (b*(1/r))`` on 200k samples, and differs from componentwise
      division on 42.8% of them by up to 1.41 ulp.

    So the naive numpy port is bit-faithful to ``phasepack`` -- a third-party transcription
    with no test suite and a known odd-grid bug -- and *not* to Kovesi. We ship Kovesi's
    arithmetic. Confirmed by running l.238 in Julia and comparing raw IEEE-754 bit patterns:
    the componentwise form is bit-identical, the numpy form is not.

    Cost: the golden fixture was generated by ``phasepack``, so agreement loosens from
    ``3.52e-14`` to ``5.32e-14`` -- still 7.27 orders inside ``rtol = 1e-6``. Accuracy is
    not what is being traded; provenance is. Recorded as drift ``M8``.

    ``radius`` carries the ``[0, 0] = 1`` fudge, so the DC bin comes out ``0`` on its own
    (both references also force it: ``frequencyfilt.jl`` l.240 ``H[1,1] = 0``).

    **Axis convention.** Swapping ``fx`` and ``fy`` rotates every orientation by 90
    degrees while leaving ``pc`` unchanged to ``1.5e-17``. The sign on ``fy`` encodes a
    y-up convention; flipping it mirrors every orientation about the x-axis, which
    axis-aligned test edges cannot see. Both bugs are caught by ``starsine`` and, since
    the fixture now stores ``orientation``, by the golden fixture.

    Args:
        fx: Signed horizontal frequency grid from :func:`construct_filter_grids`.
        fy: Signed vertical frequency grid from :func:`construct_filter_grids`.
        radius: Radial frequency with ``DC = 1``.

    Returns:
        Complex transfer function with a zero DC bin.
    """
    # Componentwise, NOT `(1j * fx - fy) / radius`. numpy would turn that into a
    # reciprocal-multiply and drift up to 1.41 ulp from Kovesi. See above.
    return (fx / radius) * 1j - (fy / radius)


def periodic_fft2(img: np.ndarray) -> np.ndarray:
    """Moisan's periodic component of the FFT -- Kovesi's ``perfft2``.

    ``fft2`` treats the image as tiled, so the intensity jump between opposite
    borders leaks a cross-shaped artifact into every frequency band. The
    periodic/smooth decomposition splits ``img = p + s`` with ``s`` carrying that
    jump, and this returns ``F(p)``.

    Kovesi's MATLAB ``phasecongmono.m`` line 156 uses it; his Julia line 446 does
    not (``IMG = fft(img)   # Use fft rather than perfft2``). The shipped operation
    follows the Julia. This exists so the golden fixture, generated from
    ``phasepack`` (a MATLAB transcription), remains reproducible.

    Args:
        img: Real 2-D array.

    Returns:
        Complex spectrum of the periodic component.
    """
    rows, cols = img.shape
    smooth = np.zeros_like(img, dtype=np.float64)
    smooth[0, :] = img[0, :] - img[-1, :]
    smooth[-1, :] = -smooth[0, :]
    smooth[:, 0] += img[:, 0] - img[:, -1]
    smooth[:, -1] -= img[:, 0] - img[:, -1]

    cx, cy = np.meshgrid(
            2 * np.pi * np.arange(cols) / cols,
            2 * np.pi * np.arange(rows) / rows,
    )
    denominator = 2.0 * (2.0 - np.cos(cx) - np.cos(cy))
    denominator[0, 0] = 1.0  # avoid /0

    smooth_fft = fft2(smooth) / denominator
    smooth_fft[0, 0] = 0.0  # the mean belongs to the periodic component

    return fft2(img) - smooth_fft


def rayleigh_mode(amplitude: np.ndarray, n_bins: int = 50) -> float:
    """Estimate the Rayleigh distribution parameter from amplitude data.

    For filter responses to Gaussian noise, amplitudes follow a Rayleigh distribution
    whose mode equals sigma.

    **Bins are anchored at ``data.min()`` and zeros are retained.** This is a fork: the
    references do not agree, and we ship the Julia branch, as M4 does for ``perfft2``.

    - Julia, ``phasecongruency.jl`` l.648-652: ``edges, counts = build_histogram(X, nbins)``
      -- Images.jl, whose ``partition_interval(nbins, minval, maxval)`` is
      ``range(minval, step=(maxval-minval)/nbins, length=nbins)`` with
      ``minval = minimum_finite(img)``. **Min-anchored.**
    - ``phasepack``, ``tools.py`` l.86: ``n, edges = np.histogram(data, nbins)``.
      **Min-anchored**, and it generated our golden fixture.
    - MATLAB, ``phasecongmono.m`` l.466-468: ``edges = 0:mx/nbins:mx``. **Zero-anchored.**
      The lone outlier.

    Settled by executing Kovesi's Julia ``rayleighmode`` against real ``Images.jl`` on the
    exact amplitude array ``_phasecong3`` feeds it. Julia and ``phasepack`` return
    ``0.0009652525656640632``; MATLAB returns ``0.0009652419842992787``. We return Julia's.

    Note the two anchors **coincide whenever the data contain an exact zero**, because
    Julia's ``minimum_finite`` sees it. They diverge only on strictly positive data --
    which is the case for every amplitude array in practice.

    An earlier version of this port dropped zeros before histogramming. *That* was the real
    undeclared deviation: neither Julia nor ``phasepack`` drops them, and dropping them
    moves the anchor off ``0`` exactly when the reference would have put it there. Zeros are
    now retained. On the shipped plates the amplitude arrays contain no exact zeros, so this
    correction is bit-identical there. ``drift-register.md`` M6.

    Args:
        amplitude: Array of amplitude values.
        n_bins: Number of histogram bins. Kovesi's default is 50.

    Returns:
        Estimated Rayleigh sigma, or ``0.0`` if the maximum is non-positive.
    """
    data = amplitude.flatten()
    if data.size == 0 or float(data.max()) <= 0.0:
        return 0.0

    # np.histogram(data, n_bins) is bit-for-bit what phasepack computes and what Julia's
    # build_histogram computes. Do NOT hand it explicit zero-anchored edges: that is
    # MATLAB's recipe, and it is the branch we do not ship.
    hist, edges = np.histogram(data, n_bins)

    mode_idx = int(np.argmax(hist))
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2)


def spread_weight(
        sum_amplitude: np.ndarray,
        max_amplitude: np.ndarray,
        n_scale: int,
        cutoff: float,
        g: float,
        epsilon: float,
) -> np.ndarray:
    """Kovesi's sigmoidal frequency-spread weighting ``W``.

    Penalizes narrow frequency distributions. ``width`` is 0 when a single scale
    responds and 1 when all scales respond equally.

    Args:
        sum_amplitude: Sum of per-scale amplitudes.
        max_amplitude: Elementwise max of the per-scale amplitudes.
        n_scale: Number of scales; must be at least 2 (the divisor is ``n_scale - 1``).
        cutoff: Fractional width below which the weight is penalized.
        g: Sharpness of the sigmoid.
        epsilon: Division guard. ``1e-5`` for ``phasecong3``, ``1e-4`` for
            ``phasecongmono`` -- the callers differ, so it is a parameter.

    Returns:
        Weight array in ``(0, 1)``.
    """
    width = (sum_amplitude / (max_amplitude + epsilon) - 1) / (n_scale - 1)
    return 1.0 / (1.0 + np.exp((cutoff - width) * g))


@dataclass(frozen=True)
class MonogenicResult:
    """Output of :func:`monogenic_phase_congruency`.

    Attributes:
        pc: Phase congruency in ``[0, 1]``. High where the log-Gabor components are
            maximally in phase, independent of their amplitude.
        orientation: Feature orientation in radians, ``(-pi/2, pi/2]``. ``0`` is a
            vertical edge (intensity varying across columns), ``pi/2`` a horizontal one.
            Measured with y increasing upward -- that is what the sign on ``sum_h2``
            encodes.
        feature_type: Local weighted mean phase angle in radians, ``[-pi/2, pi/2]``.
            ``0`` is a step edge, ``+pi/2`` a bright line, ``-pi/2`` a dark line.
        threshold: The Rayleigh noise threshold ``T`` actually applied.
        n_clamped: How many pixels needed the ``acos`` argument clipped into
            ``[-1, 1]``. Must be ``0``; a non-zero value means roundoff escaped the
            ``epsilon`` guard.
    """

    pc: np.ndarray
    orientation: np.ndarray
    feature_type: np.ndarray
    threshold: float
    n_clamped: int


def monogenic_phase_congruency(
        img: np.ndarray,
        *,
        n_scale: int = 4,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_onf: float = 0.55,
        k: float = 3.0,
        cutoff: float = 0.5,
        g: float = 10.0,
        deviation_gain: float = 1.5,
        noise_method: float = -1.0,
        periodic: bool = False,
) -> MonogenicResult:
    """Kovesi's ``phasecongmono``: phase congruency from the monogenic signal.

    An isotropic log-Gabor bandpass supplies the even channel; the Riesz transform
    supplies the two odd channels. There is no orientation sweep -- orientation falls
    out of the odd pair::

        PC = W * max(1 - deviation_gain*acos(E/(sumAn + eps)), 0) * max(E - T, 0)/(E + eps)

    This is **not** ``phasecong3``'s formula. The noise threshold is applied as a
    multiplicative fraction rather than subtracted from the numerator (Kovesi: subtracting
    it early "would interfere with the phase deviation computation"), and the phase
    deviation term is ``acos(E/sumAn)`` scaled by ``deviation_gain``.

    Args:
        img: Real 2-D array. Not required to lie in any particular range.
        n_scale: Number of log-Gabor scales. Must be at least 2.
        min_wavelength: Wavelength of the finest scale, in pixels.
        mult: Wavelength multiplier between successive scales.
        sigma_onf: Ratio of each filter's Gaussian sigma to its centre frequency.
        k: Number of noise standard deviations above the mean at which ``T`` is set.
            ``phasecongmono``'s default is ``3.0``, not ``phasecong3``'s ``2.0``.
        cutoff: Fractional frequency-spread below which ``W`` penalizes the response.
        g: Sharpness of ``W``'s sigmoid.
        deviation_gain: Scales the phase-deviation term. Kovesi: "sensible values are
            from 1 to about 2."
        noise_method: ``-1`` estimates the Rayleigh parameter from the median of the
            finest scale's amplitude; ``-2`` from its histogram mode; any value ``>= 0``
            is used verbatim as ``T`` (so ``0.0`` disables thresholding).
        periodic: Bandpass the image's periodic component (Moisan's decomposition,
            Kovesi's ``perfft2``) rather than the raw FFT. Kovesi's MATLAB does this;
            his Julia explicitly does not, and we follow the Julia. **Leave this
            ``False``** except when reproducing the golden fixture, which was generated
            from ``phasepack`` (a MATLAB transcription).

    Returns:
        A :class:`MonogenicResult`.

    Raises:
        ValueError: If ``noise_method`` is negative but is neither ``-1`` nor ``-2``.

    References:
        Kovesi, P. "Image features from phase congruency." *Videre* 1(3), 1--26 (1999).
    """
    img = np.asarray(img, dtype=np.float64)
    rows, cols = img.shape
    epsilon = EPSILON_MONOGENIC

    # The docstring above promises "must be at least 2"; nothing enforced it, and both
    # illegal values fail *silently*. Measured on a 64x64 step edge: n_scale=1 divides by
    # `n_scale - 1 == 0` inside spread_weight and returns an all-zero `pc` with only a
    # RuntimeWarning; n_scale=0 skips the scale loop entirely, leaving max_amplitude all
    # zero, and returns an all-zero `pc` with **no warning at all**. Callers reaching this
    # function directly -- FocusEdgeColorPhase will -- would get a plausible array of zeros.
    # The operations guard themselves with Field(ge=2); the kernel must too. Drift M9.
    #
    # No reference validates this: Kovesi's phasecongmono divides by (nscale-1) unguarded.
    if n_scale < 2:
        raise ValueError(
                f"n_scale must be at least 2; got {n_scale!r}. spread_weight divides by "
                f"(n_scale - 1), so n_scale=1 returns an all-zero pc with a RuntimeWarning "
                f"and n_scale=0 returns an all-zero pc silently."
        )

    # Two more silent-NaN holes, same family as the n_scale guard. Drift M10.
    #   sigma_onf == 1.0  =>  log(1.0) == 0  =>  log_gabor_scale divides by 2*0**2 == 0
    #   mult      == 1.0  =>  every scale collapses onto min_wavelength, and the geometric
    #                         noise sum divides by (1 - 1/mult) == 0
    # Both return an all-NaN `pc`, which is strictly worse than an all-zero one: it survives
    # `detect_mat in [0,1]` checks by never comparing true. Neither reference validates
    # these; Kovesi divides unguarded.
    if sigma_onf >= 1.0:
        raise ValueError(
                f"sigma_onf must be < 1.0; got {sigma_onf!r}. log(sigma_onf) is the "
                f"log-Gabor's Gaussian width, so sigma_onf=1.0 divides by zero and returns "
                f"an all-NaN pc."
        )
    if mult <= 1.0:
        raise ValueError(
                f"mult must be > 1.0; got {mult!r}. The geometric noise sum divides by "
                f"(1 - 1/mult), so mult=1.0 returns an all-NaN threshold and pc."
        )

    if noise_method < 0 and not (
            abs(noise_method + 1.0) < epsilon or abs(noise_method + 2.0) < epsilon
    ):
        raise ValueError(
                f"noise_method must be -1 (median), -2 (Rayleigh mode), or >= 0 "
                f"(a literal threshold); got {noise_method!r}. A value like -1.5 would "
                f"silently leave tau = 0 and reduce T to epsilon."
        )

    radius, _, _, _, fx, fy = construct_filter_grids(rows, cols)
    riesz = riesz_multiplier(fx, fy, radius)
    lowpass = lowpass_filter(radius)

    spectrum = periodic_fft2(img) if periodic else fft2(img)

    sum_amplitude = np.zeros((rows, cols), dtype=np.float64)
    max_amplitude = np.zeros((rows, cols), dtype=np.float64)
    sum_even = np.zeros((rows, cols), dtype=np.float64)
    sum_h1 = np.zeros((rows, cols), dtype=np.float64)
    sum_h2 = np.zeros((rows, cols), dtype=np.float64)
    tau: float = 0.0

    for s in range(n_scale):
        log_gabor = log_gabor_scale(
                radius, lowpass, min_wavelength * (mult ** s), sigma_onf
        )
        band = spectrum * log_gabor

        even = np.real(ifft2(band))
        odd = ifft2(band * riesz)
        h1, h2 = odd.real, odd.imag
        amplitude = np.sqrt(even * even + h1 * h1 + h2 * h2)

        sum_amplitude += amplitude
        sum_even += even
        sum_h1 += h1
        sum_h2 += h2

        if s == 0:
            # sum_amplitude == amplitude here; Kovesi reads the accumulator.
            # The dispatch compares against `epsilon`, as phasecongruency.jl:512
            # does. Out-of-range values already raised above.
            if abs(noise_method + 1.0) < epsilon:
                tau = float(np.median(sum_amplitude)) / np.sqrt(np.log(4.0))
            elif abs(noise_method + 2.0) < epsilon:
                tau = rayleigh_mode(sum_amplitude)
            max_amplitude = amplitude.copy()
        else:
            max_amplitude = np.maximum(max_amplitude, amplitude)

    weight = spread_weight(sum_amplitude, max_amplitude, n_scale, cutoff, g, epsilon)

    if noise_method >= 0:
        threshold = float(noise_method)
    else:
        # Filter bandwidths are scaled inversely, so the noise sums geometrically.
        total_tau = tau * (1.0 - (1.0 / mult) ** n_scale) / (1.0 - 1.0 / mult)
        noise_mean = total_tau * np.sqrt(np.pi / 2.0)
        noise_sigma = total_tau * np.sqrt((4.0 - np.pi) / 2.0)
        # The epsilon floor is phasepack's, not Kovesi's. Inactive unless img is constant.
        threshold = float(max(noise_mean + k * noise_sigma, epsilon))

    energy = np.sqrt(sum_even ** 2 + sum_h1 ** 2 + sum_h2 ** 2)

    ratio = energy / (sum_amplitude + epsilon)
    n_clamped = int(np.count_nonzero((ratio > 1.0) | (ratio < -1.0)))
    phase_deviation = np.maximum(
            1.0 - deviation_gain * np.arccos(np.clip(ratio, -1.0, 1.0)), 0.0
    )
    pc = weight * phase_deviation * np.maximum(energy - threshold, 0.0) / (energy + epsilon)

    # Kovesi writes atan(-sumh2/sumh1). arctan2 is equal mod pi and never divides by
    # zero; fold it back into (-pi/2, pi/2] so the [0,1] map is a straight affine one.
    orientation = np.arctan2(-sum_h2, sum_h1)
    orientation = np.where(orientation > np.pi / 2, orientation - np.pi, orientation)
    orientation = np.where(orientation <= -np.pi / 2, orientation + np.pi, orientation)

    # sqrt(h1^2 + h2^2), NOT np.hypot. `hypot` appears in no reference: all three write the
    # plain form -- phasecongmono.m:297 `sqrt(sumh1.^2+sumh2.^2)`, phasecongruency.jl:583
    # `sqrt(sumh1^2 + sumh2^2)`, phasepack:278 `np.sqrt(sumh1*sumh1 + sumh2*sumh2)`. hypot is
    # the overflow-safe algorithm and rounds differently: measured, the two disagree on 4.5%
    # of elements and `feature_type` on 2.6%. Same species as M8 -- a numpy convenience
    # substituted beneath identical-looking source text. The fixture's rtol=1e-6 on `ft` has
    # 6.7 orders of slack and cannot see it.
    feature_type = np.arctan2(sum_even, np.sqrt(sum_h1 ** 2 + sum_h2 ** 2))

    return MonogenicResult(pc, orientation, feature_type, threshold, n_clamped)
