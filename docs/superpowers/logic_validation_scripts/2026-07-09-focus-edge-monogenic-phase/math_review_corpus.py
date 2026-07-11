"""Refresh and verify the domain-stripped review corpus (cluster E).

The corpus is a standalone copy of the phase-congruency implementation, its tests, its
spec and its plan, with every trace of the application domain removed, so a reviewer can
judge the mathematics without learning what the software is for. It also carries the
reference implementations, the papers and the golden fixture.

**The corpus is durable and additive.** Do not rebuild it. When a source file in the
repository changes, run::

    python math_review_corpus.py refresh --sandbox <root> --repo <root>

which re-derives only the files that come from the repo, applies the rename table below,
and leaves ``refs/``, ``refimpl/``, ``papers/``, ``tests/fixtures/``, ``kernels/_data.py``
and ``kernels/_standalone.py`` untouched -- those are hand-authored or third-party and are
the whole reason the corpus is worth keeping.

Then::

    python math_review_corpus.py verify --sandbox <root> --repo <root>

runs the seven gates and exits non-zero if any fails.

Gate 7 is the one that matters. ``refresh`` cannot invent prose: if a repo file gains a new
sentence containing a banned word, gate 1 fails and prints the offending lines for a human
to rewrite. That is deliberate -- a strip pass that silently guesses is worse than one that
stops.

stdlib only. Never imports the host package. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys
from collections import Counter

# --------------------------------------------------------------------------------------
# The ban-list. `refs/` is NOT excluded from gate 1: it is provably clean (0 hits across
# all 11 files), so including it catches contamination of the reference copies. Only
# `papers/` is excluded, and only because it is third-party text we may not alter -- its
# two hits are the word "Biolog", from the journal *Biological Cybernetics* in the
# reference lists of felsberg2004 and shi2019.
# --------------------------------------------------------------------------------------
BAN_WORDS = [
    "colony", "colonies", "agar", "plate", "yeast", "fungi", "fungal", "hypha", "hyphal",
    "mycel", "septa", "microbe", "microbio", "phenotyp", "petri", "culture", "biolog",
    "organism",
]
BAN_RE = re.compile("|".join(BAN_WORDS), re.IGNORECASE)

EXCLUDE_DIRS = {"papers", "__pycache__", ".pytest_cache"}

# --------------------------------------------------------------------------------------
# Rename table, longest-first. Applied in order, so `FocusEdgeMonogenicPhase` must precede
# `FocusEdgePhase`, and `plates` must precede `plate`.
# --------------------------------------------------------------------------------------
# Rename table, applied in order as REGEXES.
#
# Naive `str.replace` is wrong here and silently corrupts code: replacing "plate" with
# "sample image" inside the identifier `test_clamp_never_fires_on_a_shipped_plate` produces
# `..._shipped_sample image` -- a space inside a function name. Gate 7's `ast.parse` caught
# it; nothing else would have. So identifier forms are handled first, with underscores, and
# the prose forms are anchored on word boundaries.
RENAMES: list[tuple[str, str]] = [
    # 1. exact symbols
    (r"phenotypic\._core\._image\.Image", "Image"),
    (r"\bFocusEdgeColorPhase\b", "ColorPhaseCongruencyEnhancer"),
    (r"\bFocusEdgeMonogenicPhase\b", "MonogenicPhaseCongruencyEnhancer"),
    (r"\bFocusEdgePhase\b", "OrientedPhaseCongruencyEnhancer"),
    (r"\bFocusEdge\b", "EdgeResponseEnhancer"),   # the marker ABC; \b spares FocusEdgeFrangi etc.
    (r"\bload_synth_yeast_plate\b", "load_sample_image_a"),
    (r"\bload_yeast_plate\b", "load_sample_image_b"),
    (r"\bload_fungi_plate\b", "load_sample_image_c"),
    # 2. snake_case identifier fragments -- the lookbehind requires a word character, so
    #    prose ("the plate") never matches here, only identifiers ("_plate").
    (r"_plate_channels\b", "_sample_channels"),
    (r"(?<=[A-Za-z0-9])_plates\b", "_sample_images"),
    (r"(?<=[A-Za-z0-9])_plate\b", "_sample_image"),
    (r"(?<=[A-Za-z0-9])_colonies\b", "_objects"),
    (r"(?<=[A-Za-z0-9])_colony\b", "_object"),
    # 3. prose, on word boundaries
    (r"\bshipped plates\b", "shipped sample images"),
    (r"\breal plates\b", "natural images"),
    (r"\breal plate\b", "natural image"),
    (r"\bcolony boundaries\b", "step edges"),
    (r"\bcolony boundary\b", "step edge"),
    (r"\bhyphal ridges\b", "line features"),
    (r"\bhyphal ridge\b", "line feature"),
    (r"\bplates\b", "sample images"),
    (r"\bplate\b", "sample image"),
    (r"\bcolonies\b", "objects"),
    (r"\bcolony\b", "object"),
]

# Per-file import rewiring. The stripped tree is a flat package: `kernels/` and `tests/`.
# These run AFTER `RENAMES`, so the right-hand sides already carry the new class names.
IMPORT_REWRITES: dict[str, list[tuple[str, str]]] = {
    "kernels/_monogenic_kernels.py": [],
    "kernels/_focus_edge_phase.py": [
        ("from ..abc_ import EdgeResponseEnhancer", "from ._standalone import EdgeResponseEnhancer"),
        ("from ..sdk_.typing_ import TuneSpec", "from .typing_ import TuneSpec"),
        ("    from phenotypic._core._image import Image", "    from ._standalone import Image"),
        ("from phenotypic._core._image import Image", "from ._standalone import Image"),
    ],
    "kernels/_focus_edge_monogenic_phase.py": [
        ("from ..abc_ import EdgeResponseEnhancer", "from ._standalone import EdgeResponseEnhancer"),
        ("from ..sdk_.typing_ import MonogenicOutput, TuneSpec", "from .typing_ import MonogenicOutput, TuneSpec"),
        ("    from phenotypic._core._image import Image", "    from ._standalone import Image"),
    ],
    "tests/test_monogenic_kernels.py": [
        ("from phenotypic.data import load_sample_image_c, load_sample_image_a, load_sample_image_b",
         "from kernels._data import load_sample_image_a, load_sample_image_b, load_sample_image_c"),
        ("from phenotypic.enhance._monogenic_kernels import", "from kernels._monogenic_kernels import"),
        ('_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures"',
         '_FIXTURE = Path(__file__).resolve().parent / "fixtures"'),
    ],
    "tests/test_focus_edge_monogenic_phase.py": [
        ("from phenotypic import Image, ImagePipeline", "from kernels._standalone import Image, ImagePipeline"),
        ("from phenotypic.data import load_sample_image_a", "from kernels._data import load_sample_image_a"),
        ("from phenotypic.enhance import MonogenicPhaseCongruencyEnhancer, OrientedPhaseCongruencyEnhancer",
         "from kernels import MonogenicPhaseCongruencyEnhancer, OrientedPhaseCongruencyEnhancer"),
        ("from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency",
         "from kernels._monogenic_kernels import monogenic_phase_congruency"),
    ],
    "tests/test_phase_congruency.py": [
        ("from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency",
         "from kernels._monogenic_kernels import monogenic_phase_congruency"),
        ('Path(__file__).resolve().parents[2] / "fixtures"',
         'Path(__file__).resolve().parent / "fixtures"'),
        ("from phenotypic import Image", "from kernels._standalone import Image"),
        ("from phenotypic.data import load_sample_image_a", "from kernels._data import load_sample_image_a"),
        ("from phenotypic.enhance import OrientedPhaseCongruencyEnhancer",
         "from kernels import OrientedPhaseCongruencyEnhancer\nfrom kernels.typing_ import TuneSpec"),
        ("import phenotypic.enhance._focus_edge_phase as fep", "import kernels._focus_edge_phase as fep"),
        ("from phenotypic.enhance._monogenic_kernels import EPSILON_MONOGENIC",
         "from kernels._monogenic_kernels import EPSILON_MONOGENIC"),
        ("from phenotypic.sdk_.typing_ import TuneSpec", "from kernels.typing_ import TuneSpec"),
    ],
    "tests/_kovesi_synthetic.py": [
        ("from phenotypic.enhance._monogenic_kernels import construct_filter_grids",
         "from kernels._monogenic_kernels import construct_filter_grids"),
    ],
    "verify_claims.py": [
        ("**not** import ``phenotypic``", "**not** import the host package"),
    ],
    "kernels/_focus_edge_color_phase.py": [
        ("from ..abc_ import EdgeResponseEnhancer", "from ._standalone import EdgeResponseEnhancer"),
        # Relative host imports: invisible to gates 1-3 (no banned word), fatal at import time.
        ("from ..sdk_.typing_ import (", "from .typing_ import ("),
        ("    from phenotypic._core._image import Image", "    from ._standalone import Image"),
    ],
    "tests/test_color_phase_kernels.py": [
        ("from phenotypic.enhance._monogenic_kernels import congruency_from_accumulators",
         "from kernels._monogenic_kernels import congruency_from_accumulators"),
    ],
}

# Prose the rename table cannot derive: sentences whose *framing*, not merely whose nouns,
# names the application domain. Encoded once, so a refresh reproduces the same judgement
# instead of asking a human to re-make it. `refresh` refuses to overwrite a live file while
# any banned word survives, so a new domain sentence in the repo stops the pipeline rather
# than leaking. Keys are the repo text, verbatim.
PROSE_PATCHES: dict[str, list[tuple[str, str]]] = {
    'tests/test_phase_congruency.py': [
        ('"""Integration tests with phenotypic data and Image class."""',
         '"""Integration tests with synthetic data and the Image class."""'),
    ],
    'kernels/_focus_edge_phase.py': [
        ('    """Enhance colony edges in ``detect_mat`` using contrast-invariant phase congruency.\n\n    Detects features where log-Gabor Fourier components are maximally in\n    phase, producing an edge response that depends on phase agreement rather\n    than amplitude. The result is invariant to local illumination level and\n    scanner vignetting, making faint or translucent colony boundaries visible\n    even where intensity-gradient methods fail. For algorithm details see\n    :doc:`/explanation/what_enhancement_does`.\n\n    Best For:\n        - Colony boundaries that vary in opacity or contrast across the plate\n          due to pigmentation differences, agar depth variation, or colony age.\n        - Plates with scanner vignetting or uneven illumination where\n          gradient-based filters produce inconsistent edge strength.\n        - Faint, translucent colonies on bright agar where the amplitude\n          signal is weak but phase coherence is preserved.\n        - Filamentous fungi plates where edges span a wide range of\n          orientations and ``n_orient=8`` captures all hyphal angles.\n\n    Consider Also:\n        - :class:`FocusEdgeFrangi` for elongated hyphae when vesselness\n          selectivity for ridge shape is more important than illumination\n          invariance.\n        - :class:`FocusEdgeHessian` for multi-scale ridge and edge detection\n          with explicit blob-sensitivity control.\n        - :class:`SharpenEdgeGauss` for edge sharpening that preserves the\n          original intensity profile on uniformly illuminated plates.\n\n    Args:\n        n_scale: Number of log-Gabor octave scales. More scales integrate\n            phase evidence over a wider spatial frequency range, giving a\n            smoother but spatially broader response. Typical range: 3--6.\n            Default: 4.\n        n_orient: Number of oriented filter lobes. 6 gives 30-degree angular\n            spacing (suitable for circular yeast colony edges); 8 gives\n            22.5-degree spacing and better sensitivity to hyphae at arbitrary\n            angles. Typical range: 4--8. Default: 6.\n        min_wavelength: Center wavelength (pixels) of the finest log-Gabor\n            scale. Should be matched to the narrowest expected colony edge\n            width; must be >= 2 (Nyquist limit enforced by validator). Smaller\n            values detect finer high-frequency features; larger values focus on\n            broader edges. Default: 3.0.\n        mult: Ratio between successive scale wavelengths. Together with\n            ``sigma_onf`` it determines inter-scale spectral coverage. ``2.1``\n            is the upstream default. For even coverage of the spectrum Kovesi\n            recommends pairing ``mult`` and ``sigma_onf`` together; e.g.\n            ``sigma_onf=0.55`` / ``mult=3`` gives roughly 2-octave bandwidth\n            and ``sigma_onf=0.75`` / ``mult=1.6`` gives roughly 1-octave\n            bandwidth. Re-tune both whenever either changes. Must be > 1.\n            Default: 2.1.\n        sigma_onf: Log-Gabor bandwidth ratio (standard deviation of the\n            Gaussian transfer function divided by the filter center frequency).\n            Smaller values give wider bandwidth (more octaves per scale,\n            broadband, suited for plates with a wide range of colony sizes);\n            larger values give narrower, more frequency-selective bandwidth.\n            For even spectral coverage pair with ``mult`` per the upstream\n            table (``0.55`` with ``mult=3``, ``0.75`` with ``mult=1.6``).\n            Valid range: 0.1--1.0. Default: 0.55.\n        k: Noise threshold multiplier in units of the estimated Rayleigh\n            noise standard deviation. Higher values (5--20) suppress more\n            noise at the cost of missing faint colony edges; lower values\n            (1--3) maximise edge recall on clean images. Value 0 disables\n            noise thresholding entirely. Default: 2.0.\n        cutoff: Frequency spread penalty threshold. Phase congruency values\n            are penalised via a sigmoid when the multi-scale amplitude spread\n            falls below this fraction, discouraging single-scale responses.\n            Valid range: (0, 1) exclusive. Default: 0.5.\n        g: Sigmoid sharpness controlling the transition from penalised to\n            unpenalised frequency spread. Higher values create a near-binary\n            gate; lower values create a gradual blend. Must be > 0.\n            Default: 10.0.\n        noise_method: Noise threshold estimation strategy. ``-1.0`` (default)\n            estimates from the median of the smallest-scale filter amplitude\n            (robust, recommended for heterogeneous plate populations). ``-2.0``\n            uses the Rayleigh histogram mode (more robust on images with\n            strong background gradients). Any value >= 0 bypasses estimation\n            and uses that value as a fixed threshold, enabling fully\n            deterministic pipelines. Default: -1.0.\n        output: Which phase congruency quantity to store in ``detect_mat``.\n            ``\'pc_sum\'`` (default) is the mean phase congruency across all\n            orientations, normalised to [0, 1]; best general-purpose edge map\n            for downstream thresholding. ``\'M\'`` is the maximum eigenvalue of\n            the phase congruency covariance tensor (edge strength along\n            continuous curves). ``\'m\'`` is the minimum eigenvalue (corner and\n            junction strength). Accepted values: ``\'pc_sum\'``, ``\'M\'``,\n            ``\'m\'``. Default: ``\'pc_sum\'``.\n\n    Returns:\n        Image: Input image with ``detect_mat`` replaced by the phase\n        congruency map, clipped to [0, 1]. ``rgb`` and ``gray`` are unchanged.\n\n    Raises:\n        ValueError: If ``n_scale`` < 2, ``n_orient`` < 1,\n            ``min_wavelength`` < 2, ``mult`` <= 1, ``sigma_onf`` outside\n            [0.1, 1.0], ``k`` < 0, ``cutoff`` outside (0, 1), or ``g`` <= 0.\n\n    References:\n        [1] P. Morrone and R. A. Owens, "Feature detection from local\n        energy," *Pattern Recognit. Lett.*, vol. 6, no. 5, pp. 303--313,\n        Dec. 1987.\n\n        [2] M. C. Morrone and D. C. Burr, "Feature detection in human\n        vision: A phase-dependent energy model," *Proc. R. Soc. London,\n        Ser. B*, vol. 235, no. 1280, pp. 221--245, Dec. 1988.\n\n        [3] P. Kovesi, "Phase congruency: A low-level image invariant,"\n        *Psychol. Res.*, vol. 64, no. 2, pp. 136--148, Aug. 2000.\n\n        [4] D. J. Field, "Relations between the statistics of natural images\n        and the response properties of cortical cells," *J. Opt. Soc. Am.\n        A*, vol. 4, no. 12, pp. 2379--2394, Dec. 1987.\n\n    See Also:\n        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a\n        visual walkthrough of contrast-invariant enhancement on plate images.\n        :doc:`/explanation/what_enhancement_does` for background on\n        phase congruency and the Local Energy Model.\n    """\n',
         '    """Enhance step edges in ``detect_mat`` using contrast-invariant phase congruency.\n\n    Detects features where log-Gabor Fourier components are maximally in\n    phase, producing an edge response that depends on phase agreement rather\n    than amplitude. The result is invariant to local illumination level and\n    slowly varying sensor gain, making low-contrast step edges visible\n    even where intensity-gradient methods fail.\n\n    Best For:\n        - Step edges whose contrast varies across the field of view, so that a\n          single gradient threshold cannot hold everywhere.\n        - Images with vignetting or uneven illumination where gradient-based\n          filters produce inconsistent edge strength.\n        - Faint, low-amplitude boundaries where the amplitude signal is weak\n          but phase coherence is preserved.\n        - Images whose edges span a wide range of orientations, where\n          ``n_orient=8`` captures all angles.\n\n    Consider Also:\n        - A Frangi vesselness filter for elongated line features when\n          selectivity for ridge shape is more important than illumination\n          invariance.\n        - A Hessian ridge filter for multi-scale ridge and edge detection\n          with explicit blob-sensitivity control.\n        - An unsharp-mask sharpener for edge sharpening that preserves the\n          original intensity profile on uniformly illuminated images.\n\n    Args:\n        n_scale: Number of log-Gabor octave scales. More scales integrate\n            phase evidence over a wider spatial frequency range, giving a\n            smoother but spatially broader response. Typical range: 3--6.\n            Default: 4.\n        n_orient: Number of oriented filter lobes. 6 gives 30-degree angular\n            spacing (suitable for isotropic, curved step edges); 8 gives\n            22.5-degree spacing and better sensitivity to line features at\n            arbitrary angles. Typical range: 4--8. Default: 6.\n        min_wavelength: Center wavelength (pixels) of the finest log-Gabor\n            scale. Should be matched to the narrowest expected edge\n            width; must be >= 2 (Nyquist limit enforced by validator). Smaller\n            values detect finer high-frequency features; larger values focus on\n            broader edges. Default: 3.0.\n        mult: Ratio between successive scale wavelengths. Together with\n            ``sigma_onf`` it determines inter-scale spectral coverage. ``2.1``\n            is the upstream default. For even coverage of the spectrum Kovesi\n            recommends pairing ``mult`` and ``sigma_onf`` together; e.g.\n            ``sigma_onf=0.55`` / ``mult=3`` gives roughly 2-octave bandwidth\n            and ``sigma_onf=0.75`` / ``mult=1.6`` gives roughly 1-octave\n            bandwidth. Re-tune both whenever either changes. Must be > 1.\n            Default: 2.1.\n        sigma_onf: Log-Gabor bandwidth ratio (standard deviation of the\n            Gaussian transfer function divided by the filter center frequency).\n            Smaller values give wider bandwidth (more octaves per scale,\n            broadband, suited for images with a wide range of feature sizes);\n            larger values give narrower, more frequency-selective bandwidth.\n            For even spectral coverage pair with ``mult`` per the upstream\n            table (``0.55`` with ``mult=3``, ``0.75`` with ``mult=1.6``).\n            Valid range: 0.1--1.0. Default: 0.55.\n        k: Noise threshold multiplier in units of the estimated Rayleigh\n            noise standard deviation. Higher values (5--20) suppress more\n            noise at the cost of missing faint edges; lower values\n            (1--3) maximise edge recall on clean images. Value 0 disables\n            noise thresholding entirely. Default: 2.0.\n        cutoff: Frequency spread penalty threshold. Phase congruency values\n            are penalised via a sigmoid when the multi-scale amplitude spread\n            falls below this fraction, discouraging single-scale responses.\n            Valid range: (0, 1) exclusive. Default: 0.5.\n        g: Sigmoid sharpness controlling the transition from penalised to\n            unpenalised frequency spread. Higher values create a near-binary\n            gate; lower values create a gradual blend. Must be > 0.\n            Default: 10.0.\n        noise_method: Noise threshold estimation strategy. ``-1.0`` (default)\n            estimates from the median of the smallest-scale filter amplitude\n            (robust, recommended for heterogeneous feature populations).\n            ``-2.0`` uses the Rayleigh histogram mode (more robust on images\n            with strong background gradients). Any value >= 0 bypasses\n            estimation and uses that value as a fixed threshold, enabling fully\n            deterministic pipelines. Default: -1.0.\n        output: Which phase congruency quantity to store in ``detect_mat``.\n            ``\'pc_sum\'`` (default) is the mean phase congruency across all\n            orientations, normalised to [0, 1]; best general-purpose edge map\n            for downstream thresholding. ``\'M\'`` is the maximum eigenvalue of\n            the phase congruency covariance tensor (edge strength along\n            continuous curves). ``\'m\'`` is the minimum eigenvalue (corner and\n            junction strength). Accepted values: ``\'pc_sum\'``, ``\'M\'``,\n            ``\'m\'``. Default: ``\'pc_sum\'``.\n\n    Returns:\n        Image: Input image with ``detect_mat`` replaced by the phase\n        congruency map, clipped to [0, 1]. ``rgb`` and ``gray`` are unchanged.\n\n    Raises:\n        ValueError: If ``n_scale`` < 2, ``n_orient`` < 1,\n            ``min_wavelength`` < 2, ``mult`` <= 1, ``sigma_onf`` outside\n            [0.1, 1.0], ``k`` < 0, ``cutoff`` outside (0, 1), or ``g`` <= 0.\n\n    References:\n        [1] P. Morrone and R. A. Owens, "Feature detection from local\n        energy," *Pattern Recognit. Lett.*, vol. 6, no. 5, pp. 303--313,\n        Dec. 1987.\n\n        [2] M. C. Morrone and D. C. Burr, "Feature detection in human\n        vision: A phase-dependent energy model," *Proc. R. Soc. London,\n        Ser. B*, vol. 235, no. 1280, pp. 221--245, Dec. 1988.\n\n        [3] P. Kovesi, "Phase congruency: A low-level image invariant,"\n        *Psychol. Res.*, vol. 64, no. 2, pp. 136--148, Aug. 2000.\n\n        [4] D. J. Field, "Relations between the statistics of natural images\n        and the response properties of cortical cells," *J. Opt. Soc. Am.\n        A*, vol. 4, no. 12, pp. 2379--2394, Dec. 1987.\n\n    See Also:\n        :class:`MonogenicPhaseCongruencyEnhancer` for the isotropic Riesz-based\n        variant, which drops the orientation sweep.\n    """\n'),
    ],
    'kernels/_focus_edge_monogenic_phase.py': [
        ('    """Enhance colony edges in ``detect_mat`` using monogenic phase congruency.\n\n    Detects features where the log-Gabor Fourier components are maximally in phase,\n    producing an edge response that depends on phase agreement rather than amplitude.\n    The result is invariant to local illumination level and scanner vignetting, so\n    faint or translucent colony boundaries stay visible where intensity-gradient\n    methods fail.\n\n    Unlike :class:`FocusEdgePhase`, which sweeps a bank of oriented filters, this uses\n    the **Riesz transform** to obtain the two odd (quadrature) channels isotropically.\n    Orientation falls out of that pair instead of being searched for, so there is no\n    ``n_orient`` parameter and the filter bank is ``n_orient`` times smaller.\n\n    Best For:\n        - Colony boundaries that vary in opacity or contrast across the plate\n        - Filamentous edges where an oriented bank\'s angular quantization blurs the\n          response between two adjacent orientations\n        - Plates where you want a cheaper, isotropic alternative to\n          :class:`FocusEdgePhase`\n\n    Args:\n        n_scale: Number of log-Gabor scales. Must be at least 2 -- the frequency-spread\n            weight divides by ``n_scale - 1``. More scales widen the frequency coverage\n            at linear cost.\n        min_wavelength: Wavelength of the finest scale, in pixels. Raise it to ignore\n            fine texture such as agar speckle.\n        mult: Wavelength multiplier between successive scales. ``2.1`` with\n            ``sigma_onf=0.55`` gives roughly two-octave filter bandwidths.\n        sigma_onf: Ratio of each filter\'s Gaussian sigma to its centre frequency.\n            Smaller means narrower bandwidth, more scales needed for coverage.\n        k: Number of noise standard deviations above the mean at which the noise\n            threshold sits. **``phasecongmono``\'s default is 3.0**, not\n            :class:`FocusEdgePhase`\'s 2.0. Raise it on noisy scans.\n        deviation_gain: Scales the phase-deviation term, sharpening edge localization.\n            Kovesi: "sensible values are from 1 to about 2." Above ~2 the response\n            becomes very sparse.\n        cutoff: Fractional frequency-spread below which the response is penalized, so\n            that a feature excited at a single scale scores lower than a broadband one.\n        g: Sharpness of the frequency-spread sigmoid.\n        noise_method: ``-1`` estimates the Rayleigh noise parameter from the median of\n            the finest scale\'s amplitude; ``-2`` uses its histogram mode. Any value\n            ``>= 0`` is used verbatim as the threshold, so ``0.0`` disables it.\n        output: Which map to write to ``detect_mat``. ``"pc"`` is the congruency in\n            ``[0, 1]``. ``"orientation"`` and ``"feature_type"`` are angles in\n            ``[-pi/2, pi/2]``, mapped to ``[0, 1]`` by ``(theta + pi/2)/pi``, since\n            ``detect_mat`` must lie in the unit interval; invert the map to recover\n            radians. For ``"orientation"``, ``0.5`` is a vertical edge and ``1.0`` a\n            horizontal one. For ``"feature_type"``, ``0.5`` is a step edge, ``1.0`` a\n            bright line and ``0.0`` a dark line.\n\n            **The two angle maps are diagnostic, not detectable.** An angle is defined\n            everywhere, including where there is no feature, so the output is a noise field\n            wherever ``pc`` is small. On ``load_synth_yeast_plate`` 89.6% of pixels have\n            ``pc < 0.02``; over those, ``"orientation"`` spans the full ``[0, 1]`` with\n            ``std = 0.307`` and only 3.3% lie near the ``0.5`` that means "vertical edge".\n            Kovesi consumes his ``or`` masked by ``pc`` (his comment: *"Quantize to 0 - 180\n            degrees (for NONMAXSUP)"*). Feed ``"pc"`` to a detector; read the angles for\n            inspection, or mask them yourself.\n\n            ``"orientation"``\'s true image is ``(0, 1]``, not ``[0, 1]``: the fold is\n            half-open, so ``-pi/2`` is unattainable. ``"feature_type"`` attains both ends.\n\n    Returns:\n        Image: Input image with ``detect_mat`` replaced by the selected monogenic map,\n        clipped to ``[0, 1]``. ``rgb`` and ``gray`` are unchanged.\n\n    Raises:\n        ValidationError: If ``n_scale`` < 2, ``min_wavelength`` < 2, ``mult`` <= 1,\n            ``sigma_onf`` outside ``[0.1, 1.0]``, ``k`` < 0, ``deviation_gain`` <= 0,\n            ``cutoff`` outside ``(0, 1)``, ``g`` <= 0, or ``output`` is not one of\n            ``"pc"``, ``"orientation"``, ``"feature_type"``.\n\n    Examples:\n        Enhance colony boundaries on a synthetic yeast plate. Phase congruency responds\n        at colony rims regardless of how opaque each colony is:\n\n        >>> from phenotypic.data import load_synth_yeast_plate\n        >>> from phenotypic.enhance import FocusEdgeMonogenicPhase\n        >>> image = load_synth_yeast_plate()\n        >>> enhanced = FocusEdgeMonogenicPhase().apply(image)\n        >>> bool(enhanced.detect_mat[:].max() > 0.5)\n        True\n\n        Ask instead whether each feature is a step (a colony rim) or a line (a hypha or\n        a scratch). ``0.5`` is a step edge:\n\n        >>> feature_type = FocusEdgeMonogenicPhase(output="feature_type")\n        >>> classified = feature_type.apply(load_synth_yeast_plate())\n        >>> bool(0.0 <= classified.detect_mat[:].min() <= classified.detect_mat[:].max() <= 1.0)\n        True\n\n    Note:\n        This is a port of Kovesi\'s ``phasecongmono``. The field notebook attributes\n        monogenic phase congruency to Wang Lijuan et al., CCDC 2014; that paper was not\n        consulted and this operation does not claim to reproduce its formulation.\n\n    See Also:\n        :class:`FocusEdgePhase` for the oriented log-Gabor bank, which additionally\n        yields corner strength via the moment tensor.\n    """\n',
         '    """Enhance step edges in ``detect_mat`` using monogenic phase congruency.\n\n    Detects features where the log-Gabor Fourier components are maximally in phase,\n    producing an edge response that depends on phase agreement rather than amplitude.\n    The result is invariant to local illumination level and slowly varying sensor gain,\n    so faint, low-contrast step edges stay visible where intensity-gradient\n    methods fail.\n\n    Unlike :class:`OrientedPhaseCongruencyEnhancer`, which sweeps a bank of oriented\n    filters, this uses the **Riesz transform** to obtain the two odd (quadrature)\n    channels isotropically. Orientation falls out of that pair instead of being searched\n    for, so there is no ``n_orient`` parameter and the filter bank is ``n_orient`` times\n    smaller.\n\n    Best For:\n        - Step edges whose contrast varies across the field of view\n        - Line features where an oriented bank\'s angular quantization blurs the\n          response between two adjacent orientations\n        - Images where you want a cheaper, isotropic alternative to\n          :class:`OrientedPhaseCongruencyEnhancer`\n\n    Args:\n        n_scale: Number of log-Gabor scales. Must be at least 2 -- the frequency-spread\n            weight divides by ``n_scale - 1``. More scales widen the frequency coverage\n            at linear cost.\n        min_wavelength: Wavelength of the finest scale, in pixels. Raise it to ignore\n            fine texture such as sensor speckle.\n        mult: Wavelength multiplier between successive scales. ``2.1`` with\n            ``sigma_onf=0.55`` gives roughly two-octave filter bandwidths.\n        sigma_onf: Ratio of each filter\'s Gaussian sigma to its centre frequency.\n            Smaller means narrower bandwidth, more scales needed for coverage.\n        k: Number of noise standard deviations above the mean at which the noise\n            threshold sits. **``phasecongmono``\'s default is 3.0**, not\n            :class:`OrientedPhaseCongruencyEnhancer`\'s 2.0. Raise it on noisy inputs.\n        deviation_gain: Scales the phase-deviation term, sharpening edge localization.\n            Kovesi: "sensible values are from 1 to about 2." Above ~2 the response\n            becomes very sparse.\n        cutoff: Fractional frequency-spread below which the response is penalized, so\n            that a feature excited at a single scale scores lower than a broadband one.\n        g: Sharpness of the frequency-spread sigmoid.\n        noise_method: ``-1`` estimates the Rayleigh noise parameter from the median of\n            the finest scale\'s amplitude; ``-2`` uses its histogram mode. Any value\n            ``>= 0`` is used verbatim as the threshold, so ``0.0`` disables it.\n        output: Which map to write to ``detect_mat``. ``"pc"`` is the congruency in\n            ``[0, 1]``. ``"orientation"`` and ``"feature_type"`` are angles in\n            ``[-pi/2, pi/2]``, mapped to ``[0, 1]`` by ``(theta + pi/2)/pi``, since\n            ``detect_mat`` must lie in the unit interval; invert the map to recover\n            radians. For ``"orientation"``, ``0.5`` is a vertical edge and ``1.0`` a\n            horizontal one. For ``"feature_type"``, ``0.5`` is a step edge, ``1.0`` a\n            bright line and ``0.0`` a dark line.\n\n            **The two angle maps are diagnostic, not detectable.** An angle is defined\n            everywhere, including where there is no feature, so the output is a noise field\n            wherever ``pc`` is small. On ``load_sample_image_a`` 89.7% of pixels have\n            ``pc < 0.02``; over those, ``"orientation"`` spans the full ``[0, 1]`` with\n            ``std = 0.292``; only 4.3% lie within ``0.02`` of the ``0.5`` that means\n            "vertical edge", and 10.5% within ``0.05`` -- the latter is what a *uniform*\n            angle would give, which is the point.\n            Kovesi consumes his ``or`` masked by ``pc`` (his comment: *"Quantize to 0 - 180\n            degrees (for NONMAXSUP)"*). Feed ``"pc"`` to a detector; read the angles for\n            inspection, or mask them yourself.\n\n            ``"orientation"``\'s true image is ``(0, 1]``, not ``[0, 1]``: the fold is\n            half-open, so ``-pi/2`` is unattainable. ``"feature_type"`` attains both ends.\n\n    Returns:\n        Image: Input image with ``detect_mat`` replaced by the selected monogenic map,\n        clipped to ``[0, 1]``. ``rgb`` and ``gray`` are unchanged.\n\n    Raises:\n        ValidationError: If ``n_scale`` < 2, ``min_wavelength`` < 2, ``mult`` <= 1,\n            ``sigma_onf`` outside ``[0.1, 1.0]``, ``k`` < 0, ``deviation_gain`` <= 0,\n            ``cutoff`` outside ``(0, 1)``, ``g`` <= 0, or ``output`` is not one of\n            ``"pc"``, ``"orientation"``, ``"feature_type"``.\n\n    Examples:\n        Enhance step edges on a synthetic sample image. Phase congruency responds\n        at disc rims regardless of how much contrast each rim carries:\n\n        >>> from kernels._data import load_sample_image_a\n        >>> from kernels._focus_edge_monogenic_phase import MonogenicPhaseCongruencyEnhancer\n        >>> image = load_sample_image_a()\n        >>> enhanced = MonogenicPhaseCongruencyEnhancer().apply(image)\n        >>> bool(enhanced.detect_mat[:].max() > 0.5)\n        True\n\n        Ask instead whether each feature is a step (a disc rim) or a line (a thin ridge\n        or a scratch). ``0.5`` is a step edge:\n\n        >>> feature_type = MonogenicPhaseCongruencyEnhancer(output="feature_type")\n        >>> classified = feature_type.apply(load_sample_image_a())\n        >>> bool(0.0 <= classified.detect_mat[:].min() <= classified.detect_mat[:].max() <= 1.0)\n        True\n\n    Note:\n        This is a port of Kovesi\'s ``phasecongmono``. The field notebook attributes\n        monogenic phase congruency to Wang Lijuan et al., CCDC 2014; that paper was not\n        consulted and this operation does not claim to reproduce its formulation.\n\n    See Also:\n        :class:`OrientedPhaseCongruencyEnhancer` for the oriented log-Gabor bank, which\n        additionally yields corner strength via the moment tensor.\n    """\n'),
    ],

    # ---- colour phase congruency (added 2026-07-10) ----
    'fusion_algebra.py': [
        ('Never imports `phenotypic`: the point is to check',
         'Never imports the host package: the point is to check'),
    ],
    'tests/test_color_phase_pfom.py': [
        ('import phenotypic\nfrom phenotypic.enhance import (\n    FocusEdgeColorPhase,\n    FocusEdgeMonogenicPhase,\n    FocusEdgePhase,\n)\n',
         'from kernels import (\n    FocusEdgeColorPhase,\n    FocusEdgeMonogenicPhase,\n    FocusEdgePhase,\n)\nfrom kernels._standalone import Image\n'),
        ('tuple[phenotypic.Image, np.ndarray]',
         'tuple[Image, np.ndarray]'),
        ('return phenotypic.Image((rgb * 255).round().astype(np.uint8)), ideal',
         'return Image((rgb * 255).round().astype(np.uint8)), ideal'),
        ('FocusEdgePhase().apply(phenotypic.Image(rgb)).detect_mat[:].astype(float)',
         'FocusEdgePhase().apply(Image(rgb)).detect_mat[:].astype(float)'),
        ('.apply(phenotypic.Image(rgb))',
         '.apply(Image(rgb))'),
    ],
    'tests/test_color_phase_kernels.py': [
        ('    @staticmethod\n    def _plate_channels():\n        from phenotypic.data import load_synth_yeast_plate\n        from phenotypic.enhance import FocusEdgeColorPhase\n\n        image = load_synth_yeast_plate()\n        return [\n            monogenic_channel_response(channel)\n            for channel in FocusEdgeColorPhase()._extract_channels(image)\n        ]\n',
         '    @staticmethod\n    def _plate_channels():\n        from kernels._data import load_sample_image_d\n        from kernels._focus_edge_color_phase import FocusEdgeColorPhase\n\n        image = load_sample_image_d()\n        return [\n            monogenic_channel_response(channel)\n            for channel in FocusEdgeColorPhase()._extract_channels(image)\n        ]\n'),
        ('from phenotypic.enhance._color_phase_kernels import (',
         'from kernels._color_phase_kernels import ('),
        ('from phenotypic.enhance._monogenic_kernels import (',
         'from kernels._monogenic_kernels import ('),
        ('import phenotypic.enhance._color_phase_kernels as cpk',
         'import kernels._color_phase_kernels as cpk'),
        ('from phenotypic.enhance._color_phase_kernels import _weighted_scalars',
         'from kernels._color_phase_kernels import _weighted_scalars'),
        ('from phenotypic.enhance._color_phase_kernels import _fused_vector',
         'from kernels._color_phase_kernels import _fused_vector'),
        ('``load_synth_yeast_plate`` at ``w = (1, 2, 3)``',
         'the chromatic sample image at ``w = (1, 2, 3)``'),
        ('Measured on\n    ``load_synth_yeast_plate``',
         'Measured on\n    the chromatic sample image'),
    ],
    'kernels/_focus_edge_color_phase.py': [
        ('no phase agreement -- pigment speckle, agar grain, Bayer demosaic noise -- **veto** an',
         'no phase agreement -- sensor grain, surface texture, demosaic noise -- **veto** an'),
        ('    Best For:\n        - **Filamentous plates.** This is where the measured benefit lives. Under lateral\n          chromatic aberration on ``load_synth_filamentous_plate``, ``fusion="joint"``\n          localizes boundaries to ``1.008`` px against ``FocusEdgeMonogenicPhase``\'s\n          ``1.158`` px, and its error is flat in the aberration.\n        - Plates where agar texture produces spurious luminance edges that carry no matching\n          chromatic structure, so an incoherent chroma channel can veto them.\n\n    Consider Also:\n        - :class:`FocusEdgeMonogenicPhase` on **round-colony plates**. Measured: on\n          ``load_synth_yeast_plate`` under the same aberration it beats *every* fusion mode\n          (``1.143`` px, against ``joint`` ``1.375``, ``coherent`` ``1.700``, ``l2``\n          ``1.776``). **On round colonies, colour buys nothing under chromatic aberration.**\n        - :class:`FocusEdgeMonogenicPhase` when the plate is near-achromatic, which this\n          operation rejects outright.\n        - :class:`FocusEdgePhase` when you also want corner strength from the moment tensor.\n',
         '    Best For:\n        - **Images whose structure is filamentous.** This is where the measured benefit\n          lives. Under lateral chromatic aberration on a filamentous test image,\n          ``fusion="joint"`` localizes boundaries to ``1.008`` px against\n          :class:`FocusEdgeMonogenicPhase`\'s ``1.158`` px, and its error is flat in the\n          aberration.\n        - Images where surface texture produces spurious luminance edges that carry no\n          matching chromatic structure, so an incoherent chroma channel can veto them.\n\n    Consider Also:\n        - :class:`FocusEdgeMonogenicPhase` on images of **compact, convex objects**.\n          Measured: under the same aberration it beats *every* fusion mode (``1.143`` px,\n          against ``joint`` ``1.375``, ``coherent`` ``1.700``, ``l2`` ``1.776``). **On\n          compact objects, colour buys nothing under chromatic aberration.**\n        - :class:`FocusEdgeMonogenicPhase` when the image is near-achromatic, which this\n          operation rejects outright.\n        - :class:`FocusEdgePhase` when you also want corner strength from the moment tensor.\n'),
        ('        **Colour is not free, and on round colonies it is not even useful.** The\n        chromatic-aberration experiment behind this operation\n        (``docs/superpowers/plans/2026-07-09-focus-edge-color-phase/experiments/``) measured\n        boundary localization under a radial R/B misregistration. On the *filamentous* plate\n        ``fusion="joint"`` wins. On the *yeast* plate, plain\n        :class:`FocusEdgeMonogenicPhase` on luminance beats all three fusion modes at every\n        aberration level. Lateral CA **creates** chromatic edges, and ``joint`` asserts them\n        coherently -- so its detected edge follows the displaced chroma rather than merging\n        it, and its error grows five times faster than luminance-only\'s. Reach for this\n        operation when the structure you want is filamentous, not merely because the plate\n        is coloured.\n',
         '        **Colour is not free, and on compact objects it is not even useful.** The\n        chromatic-aberration experiment behind this operation measured boundary localization\n        under a radial R/B misregistration. On the *filamentous* test image\n        ``fusion="joint"`` wins. On the *compact-object* test image, plain\n        :class:`FocusEdgeMonogenicPhase` on luminance beats all three fusion modes at every\n        aberration level. Lateral CA **creates** chromatic edges, and ``joint`` asserts them\n        coherently -- so its detected edge follows the displaced chroma rather than merging\n        it, and its error grows five times faster than luminance-only\'s. Reach for this\n        operation when the structure you want is filamentous, not merely because the image\n        is coloured.\n'),
        ('    Examples:\n        Fuse three channels of a synthetic yeast plate. The output is a congruency map in\n        ``[0, 1]``, like every other :class:`FocusEdge`:\n\n        >>> from phenotypic.data import load_synth_yeast_plate\n        >>> from phenotypic.enhance import FocusEdgeColorPhase\n        >>> enhanced = FocusEdgeColorPhase().apply(load_synth_yeast_plate())\n        >>> bool(0.0 <= enhanced.detect_mat[:].min() <= enhanced.detect_mat[:].max() <= 1.0)\n        True\n\n        Switch chroma off and recover the luminance-only monogenic port exactly:\n\n        >>> luminance_only = FocusEdgeColorPhase(chroma_weight_1=0.0, chroma_weight_2=0.0)\n        >>> bool(luminance_only.apply(load_synth_yeast_plate()).detect_mat[:].max() > 0.5)\n        True\n',
         '    Examples:\n        Fuse three channels of a synthetic chromatic image. The output is a congruency map\n        in ``[0, 1]``, like every other :class:`FocusEdge`:\n\n        >>> from kernels._data import load_sample_image_d\n        >>> from kernels import FocusEdgeColorPhase\n        >>> enhanced = FocusEdgeColorPhase().apply(load_sample_image_d())\n        >>> bool(0.0 <= enhanced.detect_mat[:].min() <= enhanced.detect_mat[:].max() <= 1.0)\n        True\n\n        Switch chroma off and recover the luminance-only monogenic port exactly:\n\n        >>> luminance_only = FocusEdgeColorPhase(chroma_weight_1=0.0, chroma_weight_2=0.0)\n        >>> bool(luminance_only.apply(load_sample_image_d()).detect_mat[:].max() > 0.5)\n        True\n'),
    ],
}
# repo path -> sandbox path. Only these are re-derived by `refresh`.
DERIVED: dict[str, str] = {
    "src/phenotypic/enhance/_monogenic_kernels.py": "kernels/_monogenic_kernels.py",
    "src/phenotypic/enhance/_focus_edge_phase.py": "kernels/_focus_edge_phase.py",
    "src/phenotypic/enhance/_focus_edge_monogenic_phase.py": "kernels/_focus_edge_monogenic_phase.py",
    "tests/unit/enhance/_kovesi_synthetic.py": "tests/_kovesi_synthetic.py",
    "tests/unit/enhance/test_monogenic_kernels.py": "tests/test_monogenic_kernels.py",
    "tests/unit/enhance/test_focus_edge_monogenic_phase.py": "tests/test_focus_edge_monogenic_phase.py",
    "tests/unit/enhance/test_phase_congruency.py": "tests/test_phase_congruency.py",
    "docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py": "verify_claims.py",
    "src/phenotypic/enhance/_color_phase_kernels.py": "kernels/_color_phase_kernels.py",
    "src/phenotypic/enhance/_focus_edge_color_phase.py": "kernels/_focus_edge_color_phase.py",
    "tests/unit/enhance/test_color_phase_kernels.py": "tests/test_color_phase_kernels.py",
    "tests/unit/enhance/test_color_phase_pfom.py": "tests/test_color_phase_pfom.py",
    "docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py": "fusion_algebra.py",
}

#: Repo files deliberately **kept out** of the corpus, and why. Recorded so that the
#: reviewer is told what it cannot see rather than left to infer that it does not exist.
EXCLUDED: dict[str, str] = {
    "tests/unit/enhance/test_focus_edge_color_phase.py": (
        "107 tests, of which 25 load one of the host application's fixed sample images and "
        "assert constants measured against those exact pixels (a 115.7x seam ratio, a 1.0989 "
        "un-clipped maximum, a 3e-3 rotation residual). Those pixels cannot enter this corpus "
        "without disclosing the application domain, and the constants are meaningless against "
        "the synthetic images that stand in for them. The FILE IS ABSENT; the operation it "
        "tests is present as `kernels/_focus_edge_color_phase.py`. Do not infer from its "
        "absence that the operation is untested upstream."
    ),
}

# Hand-authored or third-party. `refresh` must never touch these.
PRESERVED = [
    "kernels/_standalone.py", "kernels/_data.py", "kernels/typing_.py", "kernels/__init__.py",
    "tests/__init__.py", "conftest.py", "tests/fixtures/phasecongmono_golden.npz",
    "refs", "refimpl", "papers", "spec", "plan",
]

MANIFEST = [
    "kernels/_monogenic_kernels.py", "kernels/_focus_edge_phase.py",
    "kernels/_focus_edge_monogenic_phase.py", "kernels/typing_.py",
    "tests/test_monogenic_kernels.py", "tests/_kovesi_synthetic.py",
    "tests/test_focus_edge_monogenic_phase.py", "tests/test_phase_congruency.py",
    "tests/fixtures/phasecongmono_golden.npz",
    "tests/fixtures/phasecong3_characterization.npz", "verify_claims.py",
    "plan/plan.md", "plan/reviews", "spec/README.md", "spec/references.md",
    "spec/drift-register.md", "spec/monogenic-phase-congruency.md",
    "spec/color-phase-congruency.md", "spec/conformal-lift.md",
    "kernels/_color_phase_kernels.py", "kernels/_focus_edge_color_phase.py",
    "tests/test_color_phase_kernels.py", "tests/test_color_phase_pfom.py",
    "fusion_algebra.py",
    "refs", "refimpl", "papers",
]

# The only file whose structure must be *identical* to the repo original. It has no host
# imports, so nothing legitimate can change. The rest legitimately diverge at imports and
# data loaders; for those we compare numeric-constant and operator multisets instead.
MUST_BE_AST_IDENTICAL = [
    ("src/phenotypic/enhance/_monogenic_kernels.py", "kernels/_monogenic_kernels.py"),
    ("docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py", "verify_claims.py"),
    ("src/phenotypic/enhance/_color_phase_kernels.py", "kernels/_color_phase_kernels.py"),
    ("docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py", "fusion_algebra.py"),
]

# `parents[2]` in the repo test becomes `parent` in the flat sandbox: a directory depth,
# not logic. Recorded so gate 7 does not flag it forever.
# `parents[2]` in a repo test becomes `parent` in the flat sandbox: a directory depth,
# not logic. Recorded so gate 7 does not flag it forever.
KNOWN_CONSTANT_DELTAS = {
    "tests/test_monogenic_kernels.py": {"2": -1},
    "tests/test_phase_congruency.py": {"2": -1},
}


def _iter_files(root: pathlib.Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if set(rel.parts) & EXCLUDE_DIRS:
            continue
        if p.name.endswith(".candidate"):  # a blocked refresh's work-in-progress
            continue
        yield p


def _numeric_profile(path: pathlib.Path) -> tuple[Counter, Counter] | None:
    """Numeric constants and binary operators. ``None`` if the file is missing or unparseable.

    Returning ``None`` rather than raising keeps a broken sandbox file reportable as a gate
    failure instead of a traceback -- a gate that dies does not tell you which gate died.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    consts = Counter(
        repr(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float, complex))
        and not isinstance(n.value, bool)
    )
    ops = Counter(type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BinOp))
    return consts, ops


def _strip(text: str, sand_rel: str) -> str:
    """Rename identifiers, rewire imports, then apply the recorded prose judgements."""
    for old, new in PROSE_PATCHES.get(sand_rel, []):
        text = text.replace(old, new)
    for pattern, repl in RENAMES:
        text = re.sub(pattern, repl, text)
    for old, new in IMPORT_REWRITES.get(sand_rel, []):
        text = text.replace(old, new)
    return text


def refresh(sandbox: pathlib.Path, repo: pathlib.Path, only: list[str] | None) -> int:
    """Re-derive the repo-sourced files. **Never overwrites a live file with dirty text.**

    A candidate that still contains a banned word means the repo grew a sentence the tables
    do not cover. Overwriting would leak the domain into a corpus whose entire purpose is
    not to have one, and the leak would be invisible until the reviewer read it. So the
    candidate is kept beside the target for a human to finish, the live file is left exactly
    as it was, and the exit code is non-zero.
    """
    targets = {k: v for k, v in DERIVED.items() if not only or v in only or k in only}
    if not targets:
        print(f"no derived file matches {only!r}", file=sys.stderr)
        return 2

    promoted, blocked = [], []
    for repo_rel, sand_rel in targets.items():
        src = repo / repo_rel
        if not src.is_file():
            print(f"MISSING repo source: {repo_rel}", file=sys.stderr)
            return 1

        text = _strip(src.read_text(encoding="utf-8"), sand_rel)
        residual = [
            (i, line.strip()[:88])
            for i, line in enumerate(text.splitlines(), 1)
            if BAN_RE.search(line)
        ]

        dst = sandbox / sand_rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if residual:
            cand = dst.with_suffix(dst.suffix + ".candidate")
            cand.write_text(text, encoding="utf-8")
            blocked.append((sand_rel, residual, cand))
            continue

        dst.write_text(text, encoding="utf-8")
        promoted.append(sand_rel)

    for f in promoted:
        print(f"  refreshed {f}")

    if blocked:
        print("\nBLOCKED -- these still name the domain. The live files were NOT touched.")
        for sand_rel, residual, cand in blocked:
            print(f"\n  {sand_rel}  ({len(residual)} lines)  candidate: {cand.name}")
            for i, line in residual[:12]:
                print(f"    {i:>4}: {line}")
        print("\nStrip those lines in the candidate, then either move it over the target or")
        print("record the rewrite in PROSE_PATCHES so the next refresh reproduces it.")
        print("Never let refresh guess at prose.")
        return 1

    print("\nrefresh clean. Run `verify`.")
    return 0


def verify(sandbox: pathlib.Path, repo: pathlib.Path, python: list[str]) -> int:
    failures: list[str] = []

    # ---- gates 1-3: the corpus reveals nothing about the domain ------------------------
    for gate, pattern, label in (
        (1, BAN_RE, "ban-list"),
        (2, re.compile(r"import phenotypic"), "'import phenotypic'"),
        (3, re.compile(r"phenotypic", re.IGNORECASE), "'phenotypic'"),
    ):
        hits = []
        for p in _iter_files(sandbox):
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if pattern.search(line):
                        hits.append(f"{p.relative_to(sandbox)}:{i}: {line.strip()[:90]}")
            except (UnicodeDecodeError, OSError):
                continue
        status = "PASS" if not hits else "FAIL"
        print(f"GATE {gate}  {label:24s} hits={len(hits):3d}  {status}")
        for h in hits[:10]:
            print(f"          {h}")
        if hits:
            failures.append(f"gate {gate}")

    # ---- gate 4: manifest completeness -------------------------------------------------
    missing = [m for m in MANIFEST if not (sandbox / m).exists()]
    print(f"GATE 4  manifest                 missing={len(missing)}  {'PASS' if not missing else 'FAIL'}")
    for m in missing:
        print(f"          MISSING {m}")
    if missing:
        failures.append("gate 4")

    # ---- gate 5: the stripped checks run standalone -------------------------------------
    proc = subprocess.run(python + ["verify_claims.py"], cwd=sandbox, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    last = out.strip().splitlines()[-1] if out.strip() else "<no output>"
    ok5 = (
        proc.returncode == 0
        and "21/21 checks passed" in out
        and "max|dpc|" in out
        and "SKIPPED" not in out
        and "FIXTURE MISSING" not in out
    )
    print(f"GATE 5  verify_claims standalone exit={proc.returncode}  '{last[:40]}'  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        failures.append("gate 5")

    # ---- gate 6: stripped kernels reproduce the golden fixture --------------------------
    snippet = (
        "import sys,numpy as np;sys.path.insert(0,'.')\n"
        "from kernels._monogenic_kernels import monogenic_phase_congruency as f\n"
        "from tests._kovesi_synthetic import step2line,starsine,circsine,noiseonf,unit_variance as u\n"
        "g=np.load('tests/fixtures/phasecongmono_golden.npz');n=64\n"
        "s=np.zeros((n,n));s[:,n//2:]=1.0\n"
        "c={'step':s,'step2line':u(step2line(n)),'starsine':u(starsine(n,ncycles=8)),"
        "'circsine':u(circsine(n,wavelength=16.0)),'noiseonf':u(noiseonf(n,1.5,seed=1))}\n"
        "w=0.0;ok=True\n"
        "for k,i in c.items():\n"
        "    r=f(i,periodic=True)\n"
        "    w=max(w,float(np.abs(g[k+'__pc']-r.pc).max()))\n"
        "    ok&=bool(np.allclose(g[k+'__pc'],r.pc,rtol=1e-6,atol=1e-9))\n"
        "print(f'{w:.4e} {ok}')\n"
    )
    proc = subprocess.run(python + ["-c", snippet], cwd=sandbox, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    ok6 = proc.returncode == 0 and tail.endswith("True")
    print(f"GATE 6  golden fixture           {tail or proc.stderr.strip()[:50]}  {'PASS' if ok6 else 'FAIL'}")
    if not ok6:
        failures.append("gate 6")

    # ---- gate 6b: the colour op reduces to the monogenic port at zero chroma -----------
    # The corpus's own version of §7 test 1, run against the *stripped* kernels: with both
    # chroma weights at 0 the fused vector collapses to the luminance channel, so every
    # fusion mode must return `monogenic_phase_congruency` on that channel, bit-for-bit.
    snippet_c = (
        "import sys,numpy as np;sys.path.insert(0,'.')\n"
        "from kernels import ColorPhaseCongruencyEnhancer as C\n"
        "from kernels._data import load_sample_image_d\n"
        "from kernels._monogenic_kernels import monogenic_phase_congruency as f\n"
        "from kernels._standalone import _rgb_to_lab\n"
        "im=load_sample_image_d()\n"
        "lum=np.asarray(_rgb_to_lab(im.rgb[:])[...,0],dtype=np.float64)\n"
        "ref=f(lum);ok=True\n"
        "for mode in ('joint','coherent','l2'):\n"
        "    r=C(fusion=mode,chroma_weight_1=0.0,chroma_weight_2=0.0)._color_phase_congruency(im)\n"
        "    ok&=bool(np.array_equal(r.pc,ref.pc) and np.array_equal(r.orientation,ref.orientation))\n"
        "print(ok)\n"
    )
    proc = subprocess.run(python + ["-c", snippet_c], cwd=sandbox, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    ok6b = proc.returncode == 0 and tail == "True"
    print(f"GATE 6b zero-chroma == port      {tail or proc.stderr.strip()[:60]}  {'PASS' if ok6b else 'FAIL'}")
    if not ok6b:
        failures.append("gate 6b")

    # ---- gate 8: the stripped numeric oracle runs standalone ----------------------------
    proc = subprocess.run(python + ["fusion_algebra.py"], cwd=sandbox, capture_output=True, text=True)
    out8 = proc.stdout + proc.stderr
    last8 = out8.strip().splitlines()[-1] if out8.strip() else "<no output>"
    ok8 = proc.returncode == 0 and "4/4 checks passed" in out8
    print(f"GATE 8  fusion_algebra standalone exit={proc.returncode}  '{last8[:40]}'  {'PASS' if ok8 else 'FAIL'}")
    if not ok8:
        failures.append("gate 8")

    # ---- gate 7: the transform touched no logic -----------------------------------------
    tool = pathlib.Path(__file__).with_name("ast_structural_equivalence.py")
    ok7 = True
    for repo_rel, sand_rel in MUST_BE_AST_IDENTICAL:
        proc = subprocess.run(
            [sys.executable, str(tool), str(repo / repo_rel), str(sandbox / sand_rel)],
            capture_output=True, text=True,
        )
        identical = proc.returncode == 0
        ok7 &= identical
        print(f"GATE 7  {sand_rel:28s} {'IDENTICAL' if identical else 'DIVERGES'}")

    for repo_rel, sand_rel in DERIVED.items():
        if (repo_rel, sand_rel) in MUST_BE_AST_IDENTICAL:
            continue
        a = _numeric_profile(repo / repo_rel)
        b = _numeric_profile(sandbox / sand_rel)
        if a is None or b is None:
            ok7 = False
            which = repo_rel if a is None else sand_rel
            print(f"GATE 7  {sand_rel:28s} UNPARSEABLE OR MISSING ({which})")
            continue
        co, oo = a
        cs, os_ = b
        allowed = KNOWN_CONSTANT_DELTAS.get(sand_rel, {})
        dc = {k: (co[k], cs[k]) for k in set(co) | set(cs) if cs[k] - co[k] != allowed.get(k, 0)}
        do = {k: (oo[k], os_[k]) for k in set(oo) | set(os_) if oo[k] != os_[k]}
        if dc or do:
            ok7 = False
            print(f"GATE 7  {sand_rel:28s} CONSTANT/OPERATOR DRIFT")
            if dc:
                print(f"          constants (repo,sandbox): {dc}")
            if do:
                print(f"          operators (repo,sandbox): {do}")
        else:
            print(f"GATE 7  {sand_rel:28s} constants+operators match")
    if not ok7:
        failures.append("gate 7")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all 7 gates pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh and verify the domain-stripped review corpus.")
    ap.add_argument("mode", choices=["verify", "refresh"])
    ap.add_argument("--sandbox", required=True, type=pathlib.Path)
    ap.add_argument("--repo", required=True, type=pathlib.Path)
    ap.add_argument("--only", nargs="*", help="refresh: limit to these sandbox paths")
    ap.add_argument(
        "--python", default=None,
        help="verify: interpreter for the sandbox, e.g. \"uv run --project <repo> python\"",
    )
    a = ap.parse_args()

    if not a.sandbox.is_dir():
        print(f"sandbox not found: {a.sandbox}", file=sys.stderr)
        return 2

    if a.mode == "refresh":
        return refresh(a.sandbox, a.repo, a.only)

    python = a.python.split() if a.python else ["uv", "run", "--project", str(a.repo), "python"]
    return verify(a.sandbox, a.repo, python)


if __name__ == "__main__":
    raise SystemExit(main())
