"""Base class for creating standardized measurement information enumerations.

This module provides infrastructure for defining measurement types with consistent naming
conventions, descriptive metadata, and automatic documentation generation.
"""

import re
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from ._rembi import REMBI_MODULE

_DERIVATION_TYPES: Final = frozenset({"parameterization", "normalization", "diagnostic"})
# Not used by _classify; consumed by the Task 10 coverage-gate test.
_VALID_KINDS: Final = frozenset({"identity", "quality", "primary", "derived"})

# NOTE: keep the tier rows here in sync with :data:`_BADGE_SPECS` below — both
# encode the same (tier, kind) → Tier-1/2/3 taxonomy (this one as plain text for
# the frozen ``use_label`` contract, the other as rendered badges). Adding a tier
# means editing both.
_USE_LABELS: Final[dict[tuple[int | None, str], str]] = {
    (1, "primary"): "Direct phenotype (Tier 1)",
    (2, "primary"): "Descriptive trait (Tier 2)",
    (3, "primary"): "Discriminative feature (Tier 3)",
    # Derived Tier-1/Tier-2 columns (parameterization members) carry the same
    # trust contract as their primary counterparts, so they reuse the exact
    # same badge strings — no "(derived)" suffix.
    (1, "derived"): "Direct phenotype (Tier 1)",
    (2, "derived"): "Descriptive trait (Tier 2)",
}

#: MyST ``(label)=`` anchors in ``explanation/measurement_classification_system``.
#: Tier badges deep-link to the trust-contract table; kind-only badges (Quality /
#: Identity / normalization-Derived) link to the page top.
_ANCHOR_TIERS: Final = "measurement-tiers"
_ANCHOR_PAGE: Final = "measurement-classification"

#: Visual-badge spec per ``(tier, kind)``: ``(text, sphinx-design color, ref anchor)``.
#: Keyed identically to :data:`_USE_LABELS` (tier first); keep the tier rows in the
#: two dicts in sync. The colours encode the
#: single-value-trust gradient — green (report alone) → blue (interpret
#: directionally) → amber (use only in aggregate) — with greys for the
#: non-outcome Identity/Quality columns. Rendered as ``:bdg-ref-{color}:`` xref
#: badges (sphinx-design) so each pill links to the explanation page.
_BADGE_SPECS: Final[dict[tuple[int | None, str], tuple[str, str, str]]] = {
    (1, "primary"): ("Tier 1 · Direct phenotype", "success", _ANCHOR_TIERS),
    (2, "primary"): ("Tier 2 · Descriptive trait", "primary", _ANCHOR_TIERS),
    (3, "primary"): ("Tier 3 · Discriminative feature", "warning", _ANCHOR_TIERS),
    # Parameterization-derived members inherit their input's trust contract, so
    # they reuse the Tier-1/Tier-2 pills verbatim.
    (1, "derived"): ("Tier 1 · Direct phenotype", "success", _ANCHOR_TIERS),
    (2, "derived"): ("Tier 2 · Descriptive trait", "primary", _ANCHOR_TIERS),
    # Non-tiered kinds: de-emphasised greys for design/quality columns, cyan for
    # normalization-derived outputs (tier inherited at runtime, not declared here).
    (None, "quality"): ("Quality", "secondary", _ANCHOR_PAGE),
    (None, "identity"): ("Identity / design", "muted", _ANCHOR_PAGE),
    (None, "derived"): ("Derived", "info", _ANCHOR_PAGE),
}


@dataclass(frozen=True, slots=True)
class Entry:
    """Declarative value for a :class:`MeasurementInfo` member.

    ``label`` and ``desc`` are positional (matching the old ``(label, desc)``
    tuple ergonomics). The new fields are keyword-only, so ``bio_desc``/``image``
    can never be positionally swapped with ``desc``.

    Args:
        label: Short, unprefixed measurement name (e.g. ``"Area"``).
        desc: Technical/algorithm description of what is computed.
        bio_desc: Biological relevance / use-case. Human-authored only.
        image: Path, relative to ``_assets/measurements/``, of an illustrative
            figure (e.g. ``"shape/area.png"``); ``None`` for no figure.
    """

    label: str
    desc: str = ""
    _: KW_ONLY
    bio_desc: str = ""
    image: str | None = None
    tier: int | None = None
    derivation_type: str | None = None
    derives_from: str | None = None
    rembi_module: "REMBI_MODULE | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Entry.label must be a non-empty string")
        for name in ("desc", "bio_desc"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Entry.{name} must be a string")
        if self.image is not None and not isinstance(self.image, str):
            raise TypeError("Entry.image must be a string path or None")
        if self.tier is not None and self.tier not in (1, 2, 3):
            raise ValueError("Entry.tier must be 1, 2, 3, or None")
        if self.derivation_type is not None and self.derivation_type not in _DERIVATION_TYPES:
            raise ValueError(
                f"Entry.derivation_type must be one of {sorted(_DERIVATION_TYPES)} or None"
            )
        if self.derives_from is not None and not isinstance(self.derives_from, str):
            raise TypeError("Entry.derives_from must be a string token or None")
        if self.rembi_module is not None:
            from ._rembi import REMBI_MODULE
            if not isinstance(self.rembi_module, REMBI_MODULE):
                raise TypeError("Entry.rembi_module must be a REMBI_MODULE or None")


#: Source-root-absolute URL prefix for measurement asset images. Root-absolute so
#: the same table resolves whether embedded in autodoc pages or measurements_ref
#: pages. The docs build copies _assets/measurements/** into <srcdir>/_static/.
_ASSET_URL_PREFIX: Final = "/_static/measurements"

#: RST roles whose rendered output is meaningful content (LaTeX math, sub/super-
#: script) rather than a cross-reference. These are left intact in table cells;
#: only cross-reference roles (``:class:``/``:meth:``/``:mod:``/…) are flattened,
#: because those resolve poorly inside a list-table cell. Without this, formula
#: descriptions (e.g. ``Shape_Circularity``) would render as literal text.
_SAFE_RST_ROLES: Final = frozenset({"math", "sub", "sup"})

_RST_ROLE_RE: Final = re.compile(r":([a-zA-Z0-9_.:]+):`([^`]+)`")


def _rst_cell_text(text: str) -> str:
    """Escape RST that parses poorly in a list-table cell.

    Cross-reference roles are flattened to inline literals; content roles in
    :data:`_SAFE_RST_ROLES` (``:math:`` and friends) are preserved so formulas
    still render. Literal pipe characters are escaped.
    """

    def _flatten(match: "re.Match[str]") -> str:
        role, content = match.group(1), match.group(2)
        return match.group(0) if role in _SAFE_RST_ROLES else f"``{content}``"

    return _RST_ROLE_RE.sub(_flatten, text).replace("|", r"\|")


def _render_info_table(
    rows: list[tuple[str, str, str, str | None, str]],
    *,
    title: str,
    name_header: str = "Name",
    desc_header: str = "Description",
) -> str:
    """Render a list-table; Type/Biology/Image columns appear only when populated.

    Args:
        rows: ``(name_cell, desc, bio_desc, image_relpath_or_None, type_badge)``
            per member. ``type_badge`` is raw RST (a sphinx-design ``:bdg-ref:``
            role) inserted unescaped, so the pill renders as a link.
        title: Bold table caption (rendered ``Category: **{title}**``).
        name_header: Header for the first (name) column.
        desc_header: Header for the description column.
    """
    has_bio = any(row[2] for row in rows)
    has_img = any(row[3] for row in rows)
    has_use = any(row[4] for row in rows)

    lines = [
        f".. list-table:: Category: **{title}**",
        "   :header-rows: 1",
        "",
        f"   * - {name_header}",
        f"     - {desc_header}",
    ]
    if has_use:
        lines.append("     - Type")
    if has_bio:
        lines.append("     - Biology")
    if has_img:
        lines.append("     - Image")

    for name, desc, bio, img, use in rows:
        lines.append(f"   * - ``{name}``")
        lines.append(f"     - {_rst_cell_text(desc)}")
        if has_use:
            lines.append(f"     - {use}")
        if has_bio:
            lines.append(f"     - {_rst_cell_text(bio)}")
        if has_img:
            if img:
                lines.append(f"     - .. image:: {_ASSET_URL_PREFIX}/{img}")
                lines.append("          :width: 110px")
            else:
                lines.append("     -")
    return "\n".join(lines)


def _classify(member: "MeasurementInfo") -> tuple[str, int | None]:
    """Resolve (kind, tier) for a member; raise if unclassified.

    Precedence (highest to lowest):
    1. ``derivation_type == "diagnostic"`` → ``("quality", None)``
    2. ``derivation_type == "normalization"`` → ``("derived", None)``
    3. ``derivation_type == "parameterization"`` → ``("derived", tier_override)``
       (raises if ``tier_override`` is ``None``)
    4. Explicit ``Entry(tier=...)`` override combined with the class ``kind()``
    5. Class-level ``kind()`` / ``tier()`` (from a classification base class)
    """
    if member.derivation_type == "diagnostic":
        return ("quality", None)
    if member.derivation_type == "normalization":
        return ("derived", None)  # tier inherited from the runtime target
    cls = type(member)
    if member.derivation_type == "parameterization":
        # Explicit tier annotation overrides the default (None) for parameterization members
        # so that kinetics/magnitude params can be flagged Tier 1 while shape params are Tier 2.
        # Unlike normalization (whose tier legitimately defers to the runtime target),
        # parameterization members must always declare a tier.
        if member.tier_override is None:
            raise ValueError(
                f"{member!r}: parameterization member needs an explicit Entry(tier=...)"
            )
        return ("derived", member.tier_override)
    if member.tier_override is not None:
        return (cls.kind() or "primary", member.tier_override)
    kind, tier = cls.kind(), cls.tier()
    if kind is None:
        raise ValueError(f"{member!r}: no kind assigned (re-parent to a classification base)")
    if kind == "primary" and tier is None:
        raise ValueError(
            f"{member!r}: primary member needs a tier (a tier base class or Entry(tier=...))"
        )
    if kind == "derived" and tier is None:
        raise ValueError(
            f"{member!r}: derived member needs a derivation_type "
            "(diagnostic/normalization/parameterization)"
        )
    return (kind, tier)


class MeasurementInfo(str, Enum):
    """Base class for creating standardized measurement information enumerations.

    This class provides a structured way to define measurement types with consistent naming
    conventions, descriptive metadata, and automatic documentation generation. By inheriting
    from both str and Enum, MeasurementInfo enables measurement definitions to behave as
    enumeration members while maintaining string representation. The class automatically
    prefixes measurement labels with a category name, ensuring consistent naming across code
    and outputs.

    **Key Purposes:**

    - Standardize measurement naming conventions (category_label format) to reduce errors
    - Centralize measurement definitions with labels and descriptions in one place
    - Automatically generate RST documentation tables from measurement definitions
    - Provide easy access to headers, labels, and category information for analysis workflows
    - Enable type-safe column names in measurement DataFrames

    **Usage with MeasureFeatures Subclasses:**

    MeasurementInfo enums are used internally by measurement operations ([MeasureSize](src/phenotypic/measure/_measure_size.py),
    [MeasureShape](src/phenotypic/measure/_measure_shape.py), [MeasureColor](src/phenotypic/measure/_measure_color.py), etc.)
    to define column names in output DataFrames. Each measurement class defines its own enum
    and uses the enum values as DataFrame column headers.

    Attributes:
        label (str): The short label for the measurement (without category prefix). Set
            automatically by __new__ from the ``Entry.label`` field.
        desc (str): The technical description of what the measurement represents. Set
            automatically by __new__ from the ``Entry.desc`` field. Defaults to empty
            string if not provided.
        bio_desc (str): The human-authored biological relevance. Set automatically by
            __new__ from the ``Entry.bio_desc`` field. Defaults to empty string.
        image (str | None): Path (relative to ``_assets/measurements/``) of an
            illustrative figure, or ``None``. Set automatically from ``Entry.image``.
        pair (tuple[str, str]): A tuple of (label, description) for convenient access to
            both pieces of information together.
        CATEGORY (property): The category name returned by the category() classmethod.
            Provides instance-level access to the measurement category.

    Examples:
        Define a custom measurement enumeration:

        >>> from phenotypic.schema import Entry, MeasurementInfo
        >>> class SHAPE(MeasurementInfo):
        ...     @classmethod
        ...     def category(cls):
        ...         return 'Shape'
        ...
        ...     AREA = Entry('Area', 'Total number of pixels in the detected object')
        ...     PERIMETER = Entry('Perimeter', 'Total length of object boundary in pixels')

        Access measurement information and generate headers:

        >>> SHAPE.AREA
        <SHAPE.AREA: 'Shape_Area'>
        >>> str(SHAPE.AREA)
        'Shape_Area'
        >>> SHAPE.AREA.label
        'Area'
        >>> SHAPE.AREA.desc
        'Total number of pixels in the detected object'
        >>> SHAPE.AREA.CATEGORY
        'Shape'
        >>> SHAPE.get_labels()
        ['Area', 'Perimeter']
        >>> SHAPE.get_headers()
        ['Shape_Area', 'Shape_Perimeter']

        Use in DataFrame column naming (as MeasureFeatures do internally):

        >>> import pandas as pd
        >>> measurements = pd.DataFrame({
        ...     str(SHAPE.AREA): [1024, 956, 1101],
        ...     str(SHAPE.PERIMETER): [128, 120, 135]
        ... })
        >>> measurements.columns.tolist()
        ['Shape_Area', 'Shape_Perimeter']
        >>> measurements[str(SHAPE.AREA)]
        0    1024
        1     956
        2    1101
        Name: Shape_Area, dtype: int64

        Generate and append RST documentation:

        >>> table = SHAPE.rst_table()
        >>> class MeasureShape:
        ...     '''Measures object morphology.'''
        ...     pass
        >>> MeasureShape.__doc__ = SHAPE.append_rst_to_doc(MeasureShape)
    """

    # Per-member attributes assigned in ``__new__``. Declared here (as bare
    # annotations, never assigned at class scope) so they are visible to type
    # checkers without becoming enum members — CPython's Enum ignores
    # annotations that have no value.
    label: str
    desc: str
    bio_desc: str
    image: str | None
    pair: tuple[str, str]
    tier_override: int | None
    derivation_type: str | None
    derives_from: str | None
    rembi_module_override: "REMBI_MODULE | None"

    @classmethod
    def category(cls) -> str:
        """Return the category name for this measurement enumeration.

        Subclasses must implement this method to provide a category name that will be used
        to prefix all measurement labels. This ensures consistent naming conventions across
        the codebase.

        Returns:
            str: The category name (e.g., 'Shape', 'Color', 'Texture'). This string is
                prepended to each measurement label with an underscore separator to form
                the full header name (e.g., 'Shape_Area').

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @classmethod
    def kind(cls) -> str | None:
        """Coarse classification kind, or None until assigned by a base class."""
        return None

    @classmethod
    def tier(cls) -> int | None:
        """Trust tier (1/2/3) for primary measurements, or None."""
        return None

    @classmethod
    def rembi_module(cls) -> "REMBI_MODULE | None":
        """REMBI module for this enum, or None until a subclass declares it."""
        return None

    @property
    def CATEGORY(self) -> str:
        """Get the category name for this measurement instance.

        Provides instance-level access to the category name defined by the category()
        classmethod. This is useful when you have a measurement instance and need to know
        which category it belongs to without explicitly referencing the class.

        Returns:
            str: The category name from the enum class's category() method.
        """
        return type(self).category()

    def __new__(cls, entry: "Entry"):
        """Create a member from an :class:`Entry`.

        The enum value is the category-prefixed header (e.g. ``Shape_Area``);
        ``label``/``desc``/``bio_desc``/``image`` are stored as instance
        attributes. Anything other than an ``Entry`` raises ``TypeError`` at
        class-creation time.
        """
        if not isinstance(entry, Entry):
            raise TypeError(
                f"{cls.__name__} members must be declared as Entry(...); "
                f"got {entry!r}. Raw tuples/strings are not accepted."
            )
        full = f"{cls.category()}_{entry.label}"
        obj = str.__new__(cls, full)
        obj._value_ = full
        obj.label = entry.label
        obj.desc = entry.desc
        obj.bio_desc = entry.bio_desc
        obj.image = entry.image
        obj.pair = (entry.label, entry.desc)
        obj.tier_override = entry.tier
        obj.derivation_type = entry.derivation_type
        obj.derives_from = entry.derives_from
        obj.rembi_module_override = entry.rembi_module
        return obj

    def __str__(self) -> str:
        """Return the string representation of this measurement as the prefixed name.

        Returns the full enumeration value, which is the category-prefixed label
        (e.g., 'Shape_Area'). This is used when the measurement is converted to a string
        or used in string formatting.

        Returns:
            str: The full prefixed name of the measurement (e.g., '{category}_{label}').
        """
        return self._value_

    @property
    def resolved_kind(self) -> str:
        """The coarse kind for this member (identity/quality/primary/derived)."""
        return _classify(self)[0]

    @property
    def resolved_tier(self) -> int | None:
        """The trust tier (1/2/3) for this member, or None for non-primary."""
        return _classify(self)[1]

    @property
    def resolved_rembi_module(self) -> "REMBI_MODULE":
        """Total REMBI-module resolver: override > enum declaration > fallback."""
        from ._rembi import REMBI_MODULE
        if self.rembi_module_override is not None:
            return self.rembi_module_override
        mod = type(self).rembi_module()
        if mod is not None:
            return mod
        return REMBI_MODULE.ANALYZED_DATA

    @property
    def use_label(self) -> str:
        """Short human-readable 'how to apply' label, empty for non-primary."""
        try:
            kind, tier = _classify(self)
        except ValueError:
            return ""
        return _USE_LABELS.get((tier, kind), "")

    @property
    def use_badge(self) -> str:
        """RST sphinx-design xref badge for this member's classification.

        A coloured ``:bdg-ref-{color}:`` pill linking to the measurement-
        classification explanation page (tier badges deep-link to the trust-
        contract table). Unlike :attr:`use_label`, this covers *every* kind —
        Identity, Quality and normalization-Derived included — so the rendered
        reference table classifies every column. Returns ``""`` for members that
        do not classify (e.g. enums that subclass ``MeasurementInfo`` directly).
        """
        try:
            kind, tier = _classify(self)
        except ValueError:
            return ""
        spec = _BADGE_SPECS.get((tier, kind))
        if spec is None:
            return ""
        text, color, anchor = spec
        return f":bdg-ref-{color}:`{text} <{anchor}>`"

    @classmethod
    def get_labels(cls) -> list[str]:
        """Get all measurement labels without category prefix.

        Returns a list of the short labels (without category prefix) for all measurements
        defined in this enumeration. These come from each member's ``Entry.label`` field.
        Useful for creating human-readable lists or column names when the category context
        is already established.

        Returns:
            list[str]: List of measurement labels in enumeration order (e.g.,
                ['Area', 'Perimeter']). Does not include the category prefix; to get
                prefixed names, use get_headers().
        """
        return [m.label for m in cls]

    @classmethod
    def get_headers(cls) -> list[str]:
        """Get all measurement headers with category prefix.

        Returns a list of the full enumeration values (with category prefix) for all
        measurements defined in this enumeration. These strings are suitable for use as
        DataFrame column names, dictionary keys, or in any context where the full
        categorized name is needed.

        Returns:
            list[str]: List of prefixed measurement names in enumeration order
                (e.g., ['Shape_Area', 'Shape_Perimeter']). Each header includes the
                category prefix separated by an underscore.
        """
        return [m.value for m in cls]

    @classmethod
    def rst_table(
        cls,
        *,
        title: str | None = None,
        header: tuple[str, str] = ("Name", "Description"),
        use_headers: bool = False,
    ) -> str:
        """Render an RST list-table of this enum's members.

        Adds a Biology column when any member sets ``bio_desc`` and an Image
        column when any sets ``image`` (each suppressed otherwise).

        Args:
            title: Table caption; defaults to the category name.
            header: ``(name_column_header, description_column_header)``.
            use_headers: Name cell shows the prefixed value (``Shape_Area``)
                instead of the bare label (``Area``).
        """
        title = title or cls.category()
        name_header, desc_header = header
        rows = [
            (
                m.value if use_headers else m.label,
                m.desc,
                m.bio_desc,
                m.image,
                m.use_badge,
            )
            for m in cls
        ]
        return _render_info_table(
            rows, title=title, name_header=name_header, desc_header=desc_header
        )

    @classmethod
    def append_rst_to_doc(cls, module: str | object) -> str:
        """Append the measurement documentation table to a module or class docstring.

        Generates the RST documentation table for this measurement enumeration and appends
        it to the provided module's or class's existing docstring. This is useful for
        automatically documenting which measurements a class produces or uses.

        If the input is a string, it is treated as the docstring itself. If it is an object
        (class, function, module), its __doc__ attribute is used.

        Args:
            module (str | object): Either a docstring string or an object (class, function,
                module) whose __doc__ attribute contains the docstring. The existing
                docstring is preserved, and the measurement table is appended with a blank
                line separator.

        Returns:
            str: The original docstring (or string) with the RST measurement table appended,
                separated by two blank lines. The returned string is ready to be assigned back
                to the target's __doc__ attribute.
        """
        if isinstance(module, str):
            return module + "\n\n" + cls.rst_table()
        else:
            doc = module.__doc__ or ""
            return doc + "\n\n" + cls.rst_table()
