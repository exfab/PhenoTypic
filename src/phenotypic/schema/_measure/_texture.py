"""Second-order texture features derived from the gray-level co-occurrence matrix."""

import re

from .._base._change_notes import SINGLE_CHANNEL_PRODUCED_NOTE, append_change_note
from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import DiscriminativeFeature

# ``scale`` is emitted with ``{scale:02d}`` (a *minimum* width), so scales >= 100
# render more than two digits — match ``\d{2,}`` so large GLCM offsets stay
# recognizable rather than silently degrading to unrecognized columns.
_TEXTURE_HEADER_RE = re.compile(
    r"^(?P<cat>[A-Za-z0-9]+)_(?P<label>[^-]+)-(?:deg\d{3}|avg)-scale\d{2,}$"
)


class TEXTURE(DiscriminativeFeature):
    """Second-order texture features derived from the gray-level co-occurrence matrix (GLCM).

    All features assume normalized GLCMs computed at one or more pixel offsets and averaged
    across directions unless otherwise noted. Values depend on quantization, window size,
    and scale; interpret ranges comparatively within the same imaging setup.

    Textures are calculated along the 0, 45, 90, and 135 degree axes across the surface
    of an object. This is denoted by the axis placeholder. The texture is also computed
    along different scales in the image, and this scale can be converted to real
    distance measurements using the px-per-mm conversion. If your image's are from ExFAB
    BioFoundry we provide this for you, otherwise it can be found using ImageJ. As an
    example, a scale of 10 with 40 px-per-mm means that the measurement is the texture
    measured across every 0.25 mm on the surface of an object.

    Texture_<feature_name>-deg<axis>-scale<scale>

    We also average the texture across all degrees to provide:

    Texture_<feature_name>-avg-scale<scale>

    Each scale value passed to ``MeasureTexture(scale=...)`` writes its own full set of
    these columns, so measuring at ``scale=[5, 10]`` yields both the ``scale05`` and the
    ``scale10`` columns side by side.

    """

    @classmethod
    def metric_family(cls) -> str:
        return "Texture"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.UNITLESS

    @classmethod
    def change_note(cls) -> str:
        return SINGLE_CHANNEL_PRODUCED_NOTE

    ANGULAR_SECOND_MOMENT = Entry(
            "AngularSecondMoment",
            """How uniform the colony's texture is: highest when the same few pairs of gray
            levels recur between pixels a fixed distance apart (also called energy or
            uniformity). Measures the degree of local homogeneity
            (Σ p(i,j)²). High values → uniform texture (e.g., smooth, yeast-like colonies with consistent
            mycelial density). Low values → heterogeneous surfaces (e.g., sectored, wrinkled, or mixed
            sporulation zones). Reflects colony surface regularity rather than brightness.""",
    )

    CONTRAST = Entry(
            "Contrast",
            """How different the gray levels of pixels a fixed distance apart are, with each
            difference weighted by its square so that large jumps count most. It is
            Σ (i–j)² p(i,j) over the gray-level co-occurrence matrix. High values indicate strong gray-level
            differences (e.g., sharply defined rings, radial sectors, raised or folded regions). Low values
            indicate gradual tonal changes or uniformly pigmented colonies. Quantifies visual roughness
            and zonation amplitude.""",
    )

    CORRELATION = Entry(
            "Correlation",
            """How closely the gray levels of pixels a fixed distance apart rise and fall
            together, as a linear correlation from −1 to 1. It reads 1 for a colony with a
            single gray level. Positive, high values suggest
            structured spatial dependence (e.g., oriented radial hyphae or concentric patterns); near-zero
            values indicate uncorrelated, disordered growth (e.g., diffuse cottony mycelium). Sensitive to
            illumination gradients and directional GLCM computation.""",
    )

    VARIANCE = Entry(
            "HaralickVariance",
            """Spread of the gray levels that occur in pixel pairs a fixed distance apart,
            around their mean gray level μ. It is Σ (i–μ)² p(i,j) over the gray-level
            co-occurrence matrix. Captures spread of co-occurring gray-level pairs, distinct
            from raw intensity variance. High values → complex, multi-zone textures with variable
            hyphal/spore densities. Low values → consistent gray-level relationships and simpler colony
            surfaces.""",
    )

    INVERSE_DIFFERENCE_MOMENT = Entry(
            "InverseDifferenceMoment",
            """How similar the gray levels of pixels a fixed distance apart are, from 1 when
            every pair matches down toward 0 as their differences grow (also called
            homogeneity). It is Σ p(i,j) / (1 + (i–j)²). High values → smooth, locally uniform textures
            (e.g., glabrous colonies, uniform aerial mycelium). Low values → abrupt gray-level changes
            (e.g., granular sporulation, wrinkled surfaces). Typically inversely correlated with Contrast.""",
    )

    SUM_AVERAGE = Entry(
            "SumAverage",
            """Average of the summed gray levels of pixel pairs a fixed distance apart. It is
            Σ k·p_{x+y}(k), where p_{x+y}(k) is the probability that the two gray levels of
            a pair add up to k. Reflects the average intensity combination of
            neighboring pixels. In fungal colonies, can loosely parallel mean colony brightness when
            illumination and exposure are controlled, but remains a second-order rather than first-order
            intensity metric.""",
    )

    SUM_VARIANCE = Entry(
            "SumVariance",
            """Spread of the summed gray levels of pixel pairs a fixed distance apart, as the
            variance of the sum. It is Σ k²·p_{x+y}(k) minus SumAverage squared. High values → heterogeneous brightness zones
            (e.g., alternating dense/sparse or pigmented/non-pigmented regions). Low values → uniform
            tone across the colony. Often correlated with Contrast; use comparatively within one setup.""",
    )

    SUM_ENTROPY = Entry(
            "SumEntropy",
            """How unpredictable the summed gray level of a pixel pair a fixed distance apart
            is, as the base-2 Shannon entropy of the distribution of the sums. High values → diverse brightness combinations
            and irregular zonation. Low values → repetitive or periodic brightness patterns (e.g., evenly
            spaced rings). Indicates spatial unpredictability of summed intensities.""",
    )

    ENTROPY = Entry(
            "Entropy",
            """How unpredictable the pair of gray levels is for pixels a fixed distance apart,
            as the base-2 Shannon entropy of the gray-level co-occurrence matrix. It is
            –Σ p(i,j)·log₂ p(i,j). Measures total texture disorder and information
            content. High values → complex, irregular colony surfaces (powdery, fuzzy, or sectored growth).
            Low values → simple, smooth, predictable patterns (glabrous or uniform colonies). Sensitive to
            gray-level quantization and image dynamic range.""",
    )

    DIFFERENCE_VARIANCE = Entry(
            "DiffVariance",
            """Variance of the probabilities in the distribution of gray-level differences
            between pixels a fixed distance apart, not the variance of the differences
            themselves. It is the variance, over k = 0 to the colony's highest quantized gray
            level, of p_{x−y}(k), the probability that the gray levels of a pair differ by k
            (the mahotas default). High values mean the differences concentrate on a few values,
            as on a uniform surface where most neighbouring pixels match; low values mean they
            spread over many values, as on a surface rough throughout. Releases before 0.21.0
            described high values as a mixture of smooth and textured regions; for the values
            PhenoTypic writes, that reading runs backwards.""",
    )

    DIFFERENCE_ENTROPY = Entry(
            "DiffEntropy",
            """How unpredictable the gray-level difference between pixels a fixed distance
            apart is, as the base-2 Shannon entropy of the distribution of the differences
            |i – j|. High values → irregular, unpredictable
            intensity transitions (e.g., random sporulation or uneven mycelial networks). Low values →
            regular periodic transitions (e.g., concentric zonation). Reflects randomness of local contrast
            rather than its magnitude.""",
    )

    IMC1 = Entry(
            "InfoCorrelation1",
            """How much the gray level of one pixel tells you about the gray level of a pixel
            a fixed distance away, as a value of 0 or below that is more negative the
            stronger the dependence. It is (HXY – HXY1) /
            max(HX, HY), where HXY is Entropy, HXY1 = –Σ p(i,j)·log₂(p_x(i)·p_y(j)), and
            HX and HY are the entropies of the row and column sums p_x and p_y. Compares joint vs marginal entropies to quantify
            mutual dependence between gray levels. More negative values → structured, predictable
            textures (e.g., organized radial growth); near-zero → independence between adjacent
            regions. The value is 0 or below by definition (Haralick, 1973), as mahotas computes
            it and PhenoTypic has always written it. Releases before 0.21.0 described this as
            "Positive values → structured, predictable textures"; for the values those releases
            wrote, that meant more negative values.""",
    )

    IMC2 = Entry(
            "InfoCorrelation2",
            """How strongly the gray levels of pixels a fixed distance apart depend on each
            other, from 0 for independent gray levels toward 1 for strong dependence. It is
            √[1 – exp(–2 (H_xy2–H_xy))], where H_xy is Entropy and H_xy2 is the entropy
            of p_x(i)·p_y(j). Always ≥ 0.
            Values approaching 1 → strong spatial dependence and organized architecture (e.g., symmetric
            rings, radial structure). Values near 0 → random, independent patterns. Captures nonlinear
            organization missed by linear correlation.""",
    )

    @classmethod
    def header_scheme(cls) -> str:
        return "texture"

    @classmethod
    def member_for_header(cls, column: str):
        """Recognize TEXTURE's ``{cat}_{label}-deg###-scale##`` / ``-avg-scale##``."""
        match = _TEXTURE_HEADER_RE.match(column)
        if match is None or match.group("cat") != cls.metric_family():
            return None
        label = match.group("label")
        for member in cls:
            if member.label == label:
                return member
        return None

    @classmethod
    def header(cls, member: "TEXTURE", direction: str, scale: "int | str") -> str:
        """Return one texture header, ``{family}_{label}-{direction}-scale{scale}``.

        The single formatter behind :meth:`get_headers` and the documented
        pattern, so the two cannot disagree.

        Args:
            member: The texture feature.
            direction: ``deg000``, ``deg045``, ``deg090``, ``deg135`` or ``avg``,
                or a placeholder such as ``<direction>``.
            scale: The GLCM pixel offset, zero-padded to two digits, or a
                placeholder string such as ``<x>``, used verbatim.

        Returns:
            The header, e.g. ``Texture_Contrast-deg045-scale05``.
        """
        scale_text = f"{scale:02d}" if isinstance(scale, int) else scale
        return f"{cls.metric_family()}_{member.label}-{direction}-scale{scale_text}"

    @classmethod
    def get_headers(cls, scale: int, matrix_name=None) -> list[str]:
        """Return full texture labels with angles in order 0, 45, 90, 135 for each feature and the
        average across degrees of each feature at the end."""
        directions = [f"deg{angle:03d}" for angle in (0, 45, 90, 135)]
        labels = [
            cls.header(member, direction, scale)
            for member in cls
            for direction in directions
        ]
        labels.extend(cls.header(member, "avg", scale) for member in cls)
        return labels


TEXTURE.__doc__ = append_change_note(TEXTURE.__doc__, TEXTURE.change_note())
