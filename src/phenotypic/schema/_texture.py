"""Second-order texture features derived from the gray-level co-occurrence matrix."""

import re

from ._measurement_info import Entry
from ._tiers import DiscriminativeFeature

# Current spelling: ``{cat}_{scale:02d}px-deg{angle:03d}-{label}`` and
# ``{cat}_{scale:02d}px-avg-{label}``. Only the canonical emitted spelling is
# accepted: ``{scale:02d}`` is a *minimum* width, so the scale is ``0[1-9]``
# (1-9) or ``[1-9]\d+`` (10 and up, including 3+ digit scales), never ``5``,
# ``005`` or ``00``; the angle is one of the four GLCM directions.
#
# Both patterns are applied with ``fullmatch``, never ``match`` + ``$``: ``$``
# also matches before a trailing newline, so ``"...-Contrast\n"`` would be
# claimed as texture while every static header is compared exactly.
_TEXTURE_HEADER_RE = re.compile(
    r"(?P<cat>[A-Za-z0-9]+)_(?:0[1-9]|[1-9]\d+)px-"
    r"(?:deg(?:000|045|090|135)|avg)-(?P<label>[A-Za-z0-9]+)"
)

# Legacy spelling ``{cat}_{label}-deg###-scale##`` / ``{cat}_{label}-avg-scale##``,
# written by every run before the rename. Still recognized so stored tables keep
# their texture ownership; if it stopped matching, those columns would fall
# through the "unknown external header" classifiers and be treated as metadata.
_LEGACY_TEXTURE_HEADER_RE = re.compile(
    r"(?P<cat>[A-Za-z0-9]+)_(?P<label>[^-]+)-(?:deg\d{3}|avg)-scale\d{2,}"
)


class TEXTURE(DiscriminativeFeature):
    """Second-order texture features derived from the gray-level co-occurrence matrix (GLCM).

    All features assume normalized GLCMs computed at one pixel offset per measurer and averaged
    across directions unless otherwise noted. Values depend on quantization, window size,
    and scale; interpret ranges comparatively within the same imaging setup.

    Textures are calculated along the 0, 45, 90, and 135 degree axes across the surface
    of an object. This is denoted by the axis placeholder. The texture is also computed
    along different scales in the image, and this scale can be converted to real
    distance measurements using the px-per-mm conversion. If your image's are from ExFAB
    BioFoundry we provide this for you, otherwise it can be found using ImageJ. As an
    example, a scale of 10 with 40 px-per-mm means that the measurement is the texture
    measured across every 0.25 mm on the surface of an object.

    Texture_<scale>px-deg<axis>-<feature_name>

    where ``<scale>`` is zero-padded to at least two digits (``05px``, ``10px``,
    ``100px``) and ``<axis>`` to three (``deg000``, ``deg045``, ``deg090``,
    ``deg135``). We also average the texture across all degrees to provide:

    Texture_<scale>px-avg-<feature_name>

    For example, ``Texture_05px-deg045-Contrast`` and ``Texture_05px-avg-Contrast``.
    Columns written before this spelling (``Texture_Contrast-deg000-scale05``,
    ``Texture_Contrast-avg-scale05``) are still recognized.

    """

    @classmethod
    def category(cls) -> str:
        return "Texture"

    ANGULAR_SECOND_MOMENT = Entry(
            "AngularSecondMoment",
            """Angular second moment (energy / uniformity). Measures the degree of local homogeneity
            (Σ p(i,j)²). High values → uniform texture (e.g., smooth, yeast-like colonies with consistent
            mycelial density). Low values → heterogeneous surfaces (e.g., sectored, wrinkled, or mixed
            sporulation zones). Reflects colony surface regularity rather than brightness.""",
    )

    CONTRAST = Entry(
            "Contrast",
            """Contrast (local intensity variation; Σ (i–j)² p(i,j)). High values indicate strong gray-level
            differences (e.g., sharply defined rings, radial sectors, raised or folded regions). Low values
            indicate gradual tonal changes or uniformly pigmented colonies. Quantifies visual roughness
            and zonation amplitude.""",
    )

    CORRELATION = Entry(
            "Correlation",
            """Linear gray-level correlation between neighboring pixels. Positive, high values suggest
            structured spatial dependence (e.g., oriented radial hyphae or concentric patterns); near-zero
            values indicate uncorrelated, disordered growth (e.g., diffuse cottony mycelium). Sensitive to
            illumination gradients and directional GLCM computation.""",
    )

    VARIANCE = Entry(
            "HaralickVariance",
            """GLCM variance (Σ (i–μ)² p(i,j)). Captures spread of co-occurring gray-level pairs, distinct
            from raw intensity variance. High values → complex, multi-zone textures with variable
            hyphal/spore densities. Low values → consistent gray-level relationships and simpler colony
            surfaces.""",
    )

    INVERSE_DIFFERENCE_MOMENT = Entry(
            "InverseDifferenceMoment",
            """Homogeneity (Σ p(i,j) / (1 + (i–j)²)). High values → smooth, locally uniform textures
            (e.g., glabrous colonies, uniform aerial mycelium). Low values → abrupt gray-level changes
            (e.g., granular sporulation, wrinkled surfaces). Typically inversely correlated with Contrast.""",
    )

    SUM_AVERAGE = Entry(
            "SumAverage",
            """Mean of gray-level sums (Σ k·p_{x+y}(k)). Reflects the average intensity combination of
            neighboring pixels. In fungal colonies, can loosely parallel mean colony brightness when
            illumination and exposure are controlled, but remains a second-order rather than first-order
            intensity metric.""",
    )

    SUM_VARIANCE = Entry(
            "SumVariance",
            """Variance of gray-level sum distribution. High values → heterogeneous brightness zones
            (e.g., alternating dense/sparse or pigmented/non-pigmented regions). Low values → uniform
            tone across the colony. Often correlated with Contrast; use comparatively within one setup.""",
    )

    SUM_ENTROPY = Entry(
            "SumEntropy",
            """Entropy of the gray-level sum distribution. High values → diverse brightness combinations
            and irregular zonation. Low values → repetitive or periodic brightness patterns (e.g., evenly
            spaced rings). Indicates spatial unpredictability of summed intensities.""",
    )

    ENTROPY = Entry(
            "Entropy",
            """Global GLCM entropy (–Σ p(i,j)·log p(i,j)). Measures total texture disorder and information
            content. High values → complex, irregular colony surfaces (powdery, fuzzy, or sectored growth).
            Low values → simple, smooth, predictable patterns (glabrous or uniform colonies). Sensitive to
            gray-level quantization and image dynamic range.""",
    )

    DIFFERENCE_VARIANCE = Entry(
            "DiffVariance",
            """Variance of gray-level difference distribution. High values → mixture of smooth and textured
            regions (e.g., smooth margins with wrinkled centers). Low values → consistent edge content.
            Highlights heterogeneity in edge magnitude across the colony.""",
    )

    DIFFERENCE_ENTROPY = Entry(
            "DiffEntropy",
            """Entropy of gray-level difference distribution. High values → irregular, unpredictable
            intensity transitions (e.g., random sporulation or uneven mycelial networks). Low values →
            regular periodic transitions (e.g., concentric zonation). Reflects randomness of local contrast
            rather than its magnitude.""",
    )

    IMC1 = Entry(
            "InfoCorrelation1",
            """Information measure of correlation 1. Compares joint vs marginal entropies to quantify
            mutual dependence between gray levels. Positive values → structured, predictable textures
            (e.g., organized radial growth); near-zero → independence between adjacent regions.
            Direction of sign varies with implementation.""",
    )

    IMC2 = Entry(
            "InfoCorrelation2",
            """Information measure of correlation 2 (√[1 – exp(–2 (H_xy2–H_xy))]). Always ≥ 0.
            Values approaching 1 → strong spatial dependence and organized architecture (e.g., symmetric
            rings, radial structure). Values near 0 → random, independent patterns. Captures nonlinear
            organization missed by linear correlation.""",
    )

    @classmethod
    def header_scheme(cls) -> str:
        return "texture"

    @classmethod
    def member_for_header(cls, column: str):
        """Return the member that owns an emitted texture column, or ``None``.

        Recognizes the current ``{cat}_{scale:02d}px-deg{angle:03d}-{label}`` /
        ``{cat}_{scale:02d}px-avg-{label}`` spelling first, then the legacy
        ``{cat}_{label}-deg###-scale##`` / ``{cat}_{label}-avg-scale##`` spelling
        that stored tables still carry.
        """
        for pattern in (_TEXTURE_HEADER_RE, _LEGACY_TEXTURE_HEADER_RE):
            match = pattern.fullmatch(column)
            if match is not None:
                break
        else:
            return None
        if match.group("cat") != cls.category():
            return None
        label = match.group("label")
        for member in cls:
            if member.label == label:
                return member
        return None

    @classmethod
    def get_headers(cls, scale: int, matrix_name=None) -> list[str]:
        """Return the 65 texture column names for one GLCM ``scale``.

        Ordering contract: the first 52 names are feature-outer x angle-inner
        (every feature in ``get_labels()`` order, each at 0, 45, 90, 135
        degrees), followed by the 13 direction averages in the same feature
        order. ``MeasureTexture`` fills values by position (``[:-13]`` /
        ``[-13:]``, a feature-major ravel, and 4-wide averaging strides), so
        reordering these names silently mislabels the values.

        Args:
            scale: GLCM pixel offset; emitted zero-padded to at least two
                digits with a ``px`` suffix (``05px``).
            matrix_name: Unused; kept for call-site compatibility.

        Returns:
            list[str]: e.g. ``Texture_05px-deg000-AngularSecondMoment`` first and
            ``Texture_05px-avg-InfoCorrelation2`` last.
        """
        angles = [0, 45, 90, 135]
        prefix = f"{cls.category()}_{scale:02d}px"
        labels: list[str] = []
        for member in cls.get_labels():
            for angle in angles:
                labels.append(f"{prefix}-deg{angle:03d}-{member}")

        for member in cls.get_labels():
            labels.append(f"{prefix}-avg-{member}")
        return labels
