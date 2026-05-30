"""Camera capture metadata captured at color-checker profile fit time.

A root-polynomial colour correction matrix is only valid for images shot on the
**same camera body and lens** as the colour-checker calibration photo (and,
ideally, under comparable exposure settings).  :class:`CaptureMetadata` records
the calibration photo's EXIF — camera make/model, lens model, ISO, exposure
time, F-number, and focal length — so :class:`ColorCorrector` can later compare
it against every image it corrects and warn when the optics differ.

EXIF is read from an :class:`~phenotypic._core._image.Image`'s *imported*
metadata, which the public ``image.metadata`` accessor does not surface (its
:class:`~collections.ChainMap` covers only private/protected/public).  EXIF tag
names also vary by reader (``exifread`` ``"EXIF FNumber"``, ``exiftool``
``"EXIF:FNumber"``, PIL ``"TIFF:FNumber"``), so :meth:`CaptureMetadata.from_image`
normalises keys before matching.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from phenotypic._core._image import Image

# ---------------------------------------------------------------------------
# Field <-> EXIF key mapping
# ---------------------------------------------------------------------------
#: Camera-identity fields whose mismatch invalidates the colour match and so
#: warrants a user-facing warning.
_CRITICAL_FIELDS: tuple[str, ...] = ("camera_make", "camera_model", "lens_model")

#: Exposure-setting fields whose differences are logged at info level only.
_INFORMATIONAL_FIELDS: tuple[str, ...] = (
    "iso",
    "exposure_time",
    "f_number",
    "focal_length",
)

#: Accepted normalised EXIF key names per field, in priority order.  Only the
#: canonical seconds/F-number/millimetre tags are used for the numeric exposure
#: fields — APEX variants (``ApertureValue``, ``ShutterSpeedValue``) are
#: deliberately excluded so a profile and a target image are always compared in
#: the same units.
_CANDIDATES: dict[str, tuple[str, ...]] = {
    "camera_make"  : ("make",),
    "camera_model" : ("model", "cameramodelname", "uniquecameramodel"),
    "lens_model"   : ("lensmodel", "lens", "lensid", "lenstype", "lensinfo"),
    "iso"          : (
        "iso",
        "isospeedratings",
        "isospeed",
        "photographicsensitivity",
        "recommendedexposureindex",
    ),
    "exposure_time": ("exposuretime",),
    "f_number"     : ("fnumber",),
    "focal_length" : ("focallength",),
}

#: Matches the first numeric token (int, float, or scientific) in a string.
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_key(key: object) -> str:
    """Collapse an EXIF tag name to a prefix-free, lowercase, alnum-only token.

    Strips any reader prefix (the text before the last ``':'`` or space) and
    removes non-alphanumeric characters so ``"EXIF FNumber"``, ``"EXIF:FNumber"``,
    and ``"TIFF:FNumber"`` all collapse to ``"fnumber"``.

    Args:
        key: A metadata key (typically ``str``).

    Returns:
        The normalised key, or ``""`` for non-string input.
    """
    if not isinstance(key, str):
        return ""
    tail = re.split(r"[:\s]", key.strip())[-1]
    return re.sub(r"[^0-9a-z]", "", tail.lower())


def _to_float(value: object) -> float | None:
    """Coerce an EXIF value to ``float`` seconds/F-number/millimetres.

    Handles plain numbers, fraction strings (``"1/60"`` -> ``0.0167``),
    ``"f/5.6"`` -> ``5.6``, and unit-suffixed strings (``"55.0 mm"`` -> ``55.0``);
    UTF-8 ``bytes`` are decoded first.  Booleans and unparseable values yield
    ``None``.

    Args:
        value: Raw metadata value.

    Returns:
        The parsed float, or ``None`` when no number could be recovered.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if "/" in s:
            num_str, _, den_str = s.partition("/")
            num_m = _NUM_RE.search(num_str)
            den_m = _NUM_RE.search(den_str)
            if num_m and den_m:
                den = float(den_m.group())
                if den != 0.0:
                    return float(num_m.group()) / den
        match = _NUM_RE.search(s)
        if match:
            return float(match.group())
    return None


def _to_int(value: object) -> int | None:
    """Coerce an EXIF value to ``int`` (e.g. ISO).

    Args:
        value: Raw metadata value.

    Returns:
        The rounded integer, or ``None`` when no number could be recovered.
    """
    as_float = _to_float(value)
    return int(round(as_float)) if as_float is not None else None


def _clean_str(value: object) -> str | None:
    """Coerce an EXIF value to a trimmed ``str``, or ``None`` when empty.

    UTF-8 ``bytes`` are decoded before stripping.

    Args:
        value: Raw metadata value.

    Returns:
        The stripped string, or ``None`` for empty/``None`` input.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text or None


class CaptureMetadata(BaseModel):
    """Camera EXIF recorded from a colour-checker calibration photo.

    Populated by :meth:`from_image` at :meth:`ColorCheckerProfile.fit` time and
    compared against corrected images by :class:`ColorCorrector` via
    :meth:`compare`.  Every field is optional: tags absent from the source
    image (or images read without EXIF) remain ``None`` and are skipped during
    comparison.

    Attributes:
        camera_make: Camera manufacturer (EXIF ``Make``).
        camera_model: Camera body model (EXIF ``Model``).
        lens_model: Lens model (EXIF ``LensModel`` / ``LensID``).
        iso: ISO sensitivity (EXIF ``ISO`` / ``ISOSpeedRatings``).
        exposure_time: Exposure time in seconds (EXIF ``ExposureTime``).
        f_number: Aperture F-number (EXIF ``FNumber``).
        focal_length: Focal length in millimetres (EXIF ``FocalLength``).

    Examples:
        Differences are split into camera-identity (critical) and
        exposure-setting (informational) buckets:

        >>> a = CaptureMetadata(
        ...     camera_model="Canon EOS R100", lens_model="RF 50mm", iso=800
        ... )
        >>> b = CaptureMetadata(
        ...     camera_model="Canon EOS R5", lens_model="RF 50mm", iso=400
        ... )
        >>> critical, informational = a.compare(b)
        >>> critical
        ["camera_model (profile='Canon EOS R100', image='Canon EOS R5')"]
        >>> informational
        ['iso (profile=800, image=400)']
    """

    model_config = ConfigDict(extra="forbid")

    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    iso: int | None = None
    exposure_time: float | None = None
    f_number: float | None = None
    focal_length: float | None = None

    @classmethod
    def from_image(cls, image: Image) -> CaptureMetadata:
        """Extract capture metadata from an image's EXIF.

        Reads the image's *imported* metadata (where ``imread`` deposits EXIF)
        with precedence, then falls back to the public ``image.metadata`` view
        for any manually-set keys.  Keys are normalised via
        :func:`_normalize_key` so the canonical tag is found regardless of the
        reader's prefix/spelling.

        Args:
            image: Source image, ideally read via
                :meth:`Image.imread` so EXIF is present.

        Returns:
            A populated :class:`CaptureMetadata`; fields with no matching EXIF
            tag are left ``None``.
        """
        imported = dict(getattr(getattr(image, "_metadata", None), "imported", {}) or {})
        try:
            combined = dict(image.metadata.items())
        except Exception:
            combined = {}

        # Imported (EXIF) takes precedence over manually-set public keys.
        normalized: dict[str, object] = {}
        for source in (imported, combined):
            for raw_key, value in source.items():
                norm = _normalize_key(raw_key)
                if norm and norm not in normalized:
                    normalized[norm] = value

        raw: dict[str, object | None] = {}
        for field, candidates in _CANDIDATES.items():
            raw[field] = next(
                (normalized[c] for c in candidates if c in normalized), None
            )

        return cls(
            camera_make=_clean_str(raw["camera_make"]),
            camera_model=_clean_str(raw["camera_model"]),
            lens_model=_clean_str(raw["lens_model"]),
            iso=_to_int(raw["iso"]),
            exposure_time=_to_float(raw["exposure_time"]),
            f_number=_to_float(raw["f_number"]),
            focal_length=_to_float(raw["focal_length"]),
        )

    def compare(self, other: CaptureMetadata) -> tuple[list[str], list[str]]:
        """Diff this metadata against another, bucketed by severity.

        A field is only compared when it is non-``None`` on **both** sides, so
        an EXIF-less target image never generates spurious differences.

        Args:
            other: Metadata to compare against (typically extracted from the
                image being corrected).

        Returns:
            A ``(critical, informational)`` tuple of human-readable diff
            strings.  *critical* covers camera/lens identity; *informational*
            covers ISO/exposure/aperture/focal length.
        """
        critical: list[str] = []
        informational: list[str] = []
        for field in _CRITICAL_FIELDS + _INFORMATIONAL_FIELDS:
            mine = getattr(self, field)
            theirs = getattr(other, field)
            if mine is None or theirs is None:
                continue
            if self._values_match(mine, theirs):
                continue
            message = f"{field} (profile={mine!r}, image={theirs!r})"
            if field in _CRITICAL_FIELDS:
                critical.append(message)
            else:
                informational.append(message)
        return critical, informational

    @staticmethod
    def _values_match(a: object, b: object) -> bool:
        """Return whether two field values are equivalent.

        Numerics compare with a small relative tolerance; strings compare
        case-insensitively after trimming; everything else compares by ``==``.
        """
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-9)
        if isinstance(a, str) and isinstance(b, str):
            return a.strip().casefold() == b.strip().casefold()
        return a == b
