"""Base class for creating standardized measurement information enumerations.

This module provides infrastructure for defining measurement types with consistent naming
conventions, descriptive metadata, and automatic documentation generation.
"""

import inspect
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final, cast

from ._categories import CATEGORIES

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

#: MyST ``(label)=`` anchors in ``docs/source/measurements_ref/tier_system.md``.
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

#: Category badges render as *outline* pills (``:bdg-ref-{color}-line:``) so
#: they read as a separate axis from the solid Type pills. A sphinx-design
#: semantic color (asserted in test_classification). Each links to the
#: category's section on the generated Categories page. Not ``dark``:
#: pydata-sphinx-theme maps it to one fixed near-black with no dark-mode
#: variant, so an outline ``dark`` pill vanishes on the dark theme. ``info``
#: is mode-aware, and the outline style keeps it distinct from the solid
#: Derived pill that shares the colour.
_CATEGORY_BADGE_COLOR: Final = "info"


def _normalize_categories(value: object) -> frozenset[CATEGORIES]:
    """Coerce ``Entry(categories=...)`` to a frozenset of CATEGORIES members.

    A bare member is checked **before** iteration: CATEGORIES is a ``str``
    enum, so iterating a member would yield its characters. Membership is by
    ``isinstance``, never equality, so a raw string equal to a member's value
    (``"StartingMetrics"``) is rejected.

    Reading is looser than writing: because of the ``str`` mixin,
    ``"StartingMetrics" in entry.categories`` is ``True``. Only this write
    path is type-exact.
    """
    if isinstance(value, CATEGORIES):
        return frozenset({value})
    if isinstance(value, str):
        raise TypeError(
            f"Entry.categories takes CATEGORIES members, not strings; got {value!r}. "
            "Use CATEGORIES.<NAME>."
        )
    # A mapping iterates as its keys, which would silently drop the values the
    # author evidently meant to attach; there is nowhere to store them.
    if isinstance(value, Mapping):
        raise TypeError(
            f"Entry.categories takes CATEGORIES members, not a mapping; got {value!r}"
        )
    if not isinstance(value, Iterable):
        raise TypeError(
            f"Entry.categories must be a CATEGORIES member or an iterable of them; got {value!r}"
        )
    items = tuple(value)
    bad = [item for item in items if not isinstance(item, CATEGORIES)]
    if bad:
        raise TypeError(f"Entry.categories accepts only CATEGORIES members; got {bad!r}")
    return frozenset(items)


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
        categories: Curated categories (CATEGORIES members) this measurement
            belongs to. A bare member or an iterable of members (a mapping is
            refused); normalized to a frozenset.
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
    categories: "CATEGORIES | Iterable[CATEGORIES]" = frozenset()

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
        object.__setattr__(self, "categories", _normalize_categories(self.categories))


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


#: Column where a list-table cell's text starts (``"     - "``); continuation
#: lines of a multi-line cell must sit exactly here to stay one paragraph.
_CELL_INDENT: Final = " " * 7

#: CSS class on every schema table, so the docs can let long column names wrap.
_TABLE_CLASS: Final = "phenotypic-measurement-table"


def _cell_block(text: str) -> str:
    """Lay out a multi-line cell as one paragraph at the cell's column.

    Descriptions are triple-quoted with source indentation. Left as is, the
    continuation lines sit deeper than the first line and docutils reads the
    cell as a definition list (first line bold, the rest indented), so dedent
    them and re-indent to the cell column.
    """
    return inspect.cleandoc(text).replace("\n", "\n" + _CELL_INDENT)


#: One list-table row; both ``_render_info_table`` callers must build it via ``_info_row``.
_InfoRow = tuple[str, str, str, str | None, str, str]


def _info_row(name_cell: str, member: "MeasurementInfo") -> _InfoRow:
    """Build one :func:`_render_info_table` row for *member*, named *name_cell*."""
    return (
        name_cell,
        member.desc,
        member.bio_desc,
        member.image,
        member.use_badge,
        member.category_badges,
    )


def _render_info_table(
    rows: list[_InfoRow],
    *,
    title: str,
    name_header: str = "Name",
    desc_header: str = "Description",
) -> str:
    """Render a list-table; Type/Categories/Biology/Image columns appear only when populated.

    Args:
        rows: ``(name_cell, desc, bio_desc, image_relpath_or_None, type_badge,
            category_badges)`` per member. Both badge cells are raw RST
            (sphinx-design ``:bdg-ref:`` roles) inserted unescaped, so the
            pills render as links.
        title: Bold table caption (rendered ``Metric family: **{title}**``).
        name_header: Header for the first (name) column.
        desc_header: Header for the description column.
    """
    has_bio = any(row[2] for row in rows)
    has_img = any(row[3] for row in rows)
    has_use = any(row[4] for row in rows)
    has_cat = any(row[5] for row in rows)

    lines = [
        f".. list-table:: Metric family: **{title}**",
        "   :header-rows: 1",
        f"   :class: {_TABLE_CLASS}",
        "",
        f"   * - {name_header}",
        f"     - {desc_header}",
    ]
    if has_use:
        lines.append("     - Type")
    if has_cat:
        lines.append("     - Categories")
    if has_bio:
        lines.append("     - Biology")
    if has_img:
        lines.append("     - Image")

    for name, desc, bio, img, use, cats in rows:
        lines.append(f"   * - ``{name}``")
        lines.append(f"     - {_cell_block(_rst_cell_text(desc))}")
        if has_use:
            lines.append(f"     - {use}")
        if has_cat:
            lines.append(f"     - {cats}")
        if has_bio:
            lines.append(f"     - {_cell_block(_rst_cell_text(bio))}")
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


#: Names removed by the ``category()`` → ``metric_family()`` hard rename.
_LEGACY_FAMILY_NAMES: Final = ("category", "CATEGORY")


def _refuse_legacy_category(cls: type) -> None:
    """Refuse a subclass still overriding the removed ``category`` API.

    Any attribute with a legacy name is refused, not only methods: a member
    named ``CATEGORY`` would shadow the old property, so it is reserved too.
    """
    for klass in cls.__mro__:
        for name in _LEGACY_FAMILY_NAMES:
            if name not in vars(klass):
                continue
            value = vars(klass)[name]
            if isinstance(value, (classmethod, staticmethod, property)) or callable(value):
                advice = (
                    "the category() classmethod is now metric_family() and "
                    "the CATEGORY property is now METRIC_FAMILY. Rename the override."
                )
            else:
                advice = (
                    f"{name!r} is a reserved name since the category() -> "
                    "metric_family() rename. Rename the member."
                )
            raise TypeError(f"{klass.__name__} defines {name!r}, which was removed: {advice}")


class MeasurementInfo(str, Enum):
    """Base class for creating standardized measurement information enumerations.

    This class provides a structured way to define measurement types with consistent naming
    conventions, descriptive metadata, and automatic documentation generation. By inheriting
    from both str and Enum, MeasurementInfo enables measurement definitions to behave as
    enumeration members while maintaining string representation. The class automatically
    prefixes measurement labels with a metric family name, ensuring consistent naming across
    code and outputs.

    **Key Purposes:**

    - Standardize measurement naming conventions (family_label format) to reduce errors
    - Centralize measurement definitions with labels and descriptions in one place
    - Automatically generate RST documentation tables from measurement definitions
    - Provide easy access to headers, labels, and metric family information for analysis
      workflows
    - Enable type-safe column names in measurement DataFrames

    **Usage with MeasureFeatures Subclasses:**

    MeasurementInfo enums are used internally by measurement operations ([MeasureSize](src/phenotypic/measure/_measure_size.py),
    [MeasureShape](src/phenotypic/measure/_measure_shape.py), [MeasureColor](src/phenotypic/measure/_measure_color.py), etc.)
    to define column names in output DataFrames. Each measurement class defines its own enum
    and uses the enum values as DataFrame column headers.

    Attributes:
        label (str): The short label for the measurement (without metric family prefix). Set
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
        METRIC_FAMILY (property): The metric family name returned by the metric_family()
            classmethod. Provides instance-level access to the measurement's metric family.

    Examples:
        Define a custom measurement enumeration:

        >>> from phenotypic.schema import Entry, MeasurementInfo
        >>> class SHAPE(MeasurementInfo):
        ...     @classmethod
        ...     def metric_family(cls):
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
        >>> SHAPE.AREA.METRIC_FAMILY
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
    categories: frozenset[CATEGORIES]

    @classmethod
    def metric_family(cls) -> str:
        """Return the metric family for this measurement enumeration.

        Subclasses must implement this method to provide a metric family name that will be
        used to prefix all measurement labels. This ensures consistent naming conventions
        across the codebase.

        Returns:
            str: The family name (e.g., 'Size', 'Color', 'Texture'). This string is
                prepended to each measurement label with an underscore separator to form
                the full header name (e.g., 'Size_Area').

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_legacy_category(cls)

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

    @classmethod
    def change_note(cls) -> str:
        """Return an RST change-note block rendered above this enum's table.

        Rendered by :meth:`append_rst_to_doc` (measurer class docs and the
        enum's own docstring) and by the Measurements reference page. Empty
        by default; an enum overrides it when a release changes its public
        columns. Keep the text in ``phenotypic.schema._base._change_notes``.

        Returns:
            str: RST, typically a ``.. versionchanged::`` directive, or ``""``.
        """
        return ""

    @classmethod
    def header_scheme(cls) -> str:
        """Naming scheme for this enum's DataFrame output headers.

        ``"static"`` (default) → exact ``{family}_{label}``;
        ``"metric_qualified"`` → ``{family}_{metric}_{label}`` (growth
        models + model metrics, where ``{metric}`` is a runtime value);
        ``"texture"`` → TEXTURE's ``-deg/-scale`` suffix scheme.
        """
        return "static"

    @classmethod
    def member_for_header(cls, column: str) -> "MeasurementInfo | None":
        """Return the member *column* is an output header for, or ``None``."""
        if cls.header_scheme() == "metric_qualified":
            parsed = parse_qualified_header(cls, column)
            return parsed[1] if parsed is not None else None
        for member in cls:
            if member.value == column:
                return member
        return None

    @classmethod
    def owns_header(cls, column: str) -> bool:
        """Whether *column* is one of this enum's output headers."""
        return cls.member_for_header(column) is not None

    @property
    def METRIC_FAMILY(self) -> str:
        """Get the metric family name for this measurement instance.

        Provides instance-level access to the metric family defined by the metric_family()
        classmethod. This is useful when you have a measurement instance and need to know
        which metric family it belongs to without explicitly referencing the class.

        Returns:
            str: The metric family name from the enum class's metric_family() method.
        """
        return type(self).metric_family()

    def __new__(cls, entry: "Entry"):
        """Create a member from an :class:`Entry`.

        The enum value is the family-prefixed header (e.g. ``Size_Area``);
        ``label``/``desc``/``bio_desc``/``image`` are stored as instance
        attributes. Anything other than an ``Entry`` raises ``TypeError`` at
        class-creation time.
        """
        _refuse_legacy_category(cls)
        if not isinstance(entry, Entry):
            raise TypeError(
                f"{cls.__name__} members must be declared as Entry(...); "
                f"got {entry!r}. Raw tuples/strings are not accepted."
            )
        full = f"{cls.metric_family()}_{entry.label}"
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
        obj.categories = cast("frozenset[CATEGORIES]", entry.categories)
        return obj

    def __str__(self) -> str:
        """Return the string representation of this measurement as the prefixed name.

        Returns the full enumeration value, which is the family-prefixed label
        (e.g., 'Size_Area'). This is used when the measurement is converted to a string
        or used in string formatting.

        Returns:
            str: The full prefixed name of the measurement (e.g., '{family}_{label}').
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

    @property
    def category_badges(self) -> str:
        """RST outline badges, one per category, linking to the Categories page.

        Empty when the member has no category.
        """
        return " ".join(
            f":bdg-ref-{_CATEGORY_BADGE_COLOR}-line:`{c.display_name} <{c.anchor}>`"
            for c in CATEGORIES.in_order(self.categories)
        )

    @classmethod
    def get_labels(cls) -> list[str]:
        """Get all measurement labels without metric family prefix.

        Returns a list of the short labels (without metric family prefix) for all
        measurements defined in this enumeration. These come from each member's
        ``Entry.label`` field. Useful for creating human-readable lists or column names
        when the metric family context is already established.

        Returns:
            list[str]: List of measurement labels in enumeration order (e.g.,
                ['Area', 'Perimeter']). Does not include the metric family prefix; to get
                prefixed names, use get_headers().
        """
        return [m.label for m in cls]

    @classmethod
    def get_headers(cls) -> list[str]:
        """Get all measurement headers with metric family prefix.

        Returns a list of the full enumeration values (with metric family prefix) for all
        measurements defined in this enumeration. These strings are suitable for use as
        DataFrame column names, dictionary keys, or in any context where the full
        prefixed name is needed.

        Returns:
            list[str]: List of prefixed measurement names in enumeration order
                (e.g., ['Size_Area', 'Size_Perimeter']). Each header includes the
                metric family prefix separated by an underscore.
        """
        return [m.value for m in cls]

    @classmethod
    def rst_table(
        cls,
        *,
        title: str | None = None,
        header: tuple[str, str] = ("Name", "Description"),
        use_headers: bool = False,
        members: "Iterable[MeasurementInfo] | None" = None,
        header_for: "Callable[[MeasurementInfo], str] | None" = None,
    ) -> str:
        """Render an RST list-table of this enum's members.

        Adds a Categories column when any member carries a category, a
        Biology column when any sets ``bio_desc`` and an Image column when
        any sets ``image`` (each suppressed otherwise).

        Args:
            title: Table caption; defaults to the metric family name.
            header: ``(name_column_header, description_column_header)``.
            use_headers: Name cell shows the prefixed value (``Size_Area``)
                instead of the bare label (``Area``).
            members: The members to list, in the order given; defaults to
                every member. The Measurements reference passes only the
                members that appear in output tables.
            header_for: Name-cell text for a member, overriding
                ``use_headers``. The Measurements reference passes the
                producer's ``output_header`` so the table shows the header
                actually written (``QC_ICC_Metric``, not ``QC_Metric``).
        """
        title = title or cls.metric_family()
        name_header, desc_header = header
        if header_for is None:
            header_for = (lambda m: m.value) if use_headers else (lambda m: m.label)
        listed = list(cls) if members is None else list(members)
        foreign = [m for m in listed if not isinstance(m, cls)]
        if foreign:
            raise ValueError(f"{cls.__name__}.rst_table got foreign members: {foreign!r}")
        rows = [_info_row(header_for(m), m) for m in listed]
        return _render_info_table(
            rows, title=title, name_header=name_header, desc_header=desc_header
        )

    @classmethod
    def append_rst_to_doc(cls, module: str | object) -> str:
        """Append the measurement documentation table to a module or class docstring.

        Generates the RST documentation table for this measurement enumeration and appends
        it to the provided module's or class's existing docstring. This is useful for
        automatically documenting which measurements a class produces or uses.
        When the enum defines a :meth:`change_note`, it is placed between the
        docstring and the table.

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
        doc = module if isinstance(module, str) else (module.__doc__ or "")
        note = cls.change_note()
        parts = [doc, note, cls.rst_table()] if note else [doc, cls.rst_table()]
        return "\n\n".join(parts)


def qualified_header(member: "MeasurementInfo", token: str) -> str:
    """Runtime output header ``{Family}_{token}_{label}``.

    Embeds the fitted-metric *token* so a reader can tell which measurement a
    growth-model parameter was trained on. Inverse of
    :func:`parse_qualified_header`.
    """
    return f"{member.METRIC_FAMILY}_{token}_{member.label}"


def parse_qualified_header(
    info_cls: "type[MeasurementInfo]", column: str
) -> "tuple[str, MeasurementInfo] | None":
    """Inverse of :func:`qualified_header` for one enum.

    Returns ``(metric_token, member)`` when *column* is a metric-qualified
    header of *info_cls* (``{family}_{metric}_{label}`` with a non-empty metric),
    else ``None``. Anchors on the longest matching member-label suffix, so an
    underscore inside the metric token stays unambiguous.
    """
    prefix = info_cls.metric_family() + "_"
    if not column.startswith(prefix):
        return None
    best: "tuple[str, MeasurementInfo] | None" = None
    for member in info_cls:
        suffix = "_" + member.label
        if column.endswith(suffix):
            metric = column[len(prefix): len(column) - len(suffix)]
            if metric and (best is None or len(member.label) > len(best[1].label)):
                best = (metric, member)
    return best
