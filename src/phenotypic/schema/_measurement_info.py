"""Base class for creating standardized measurement information enumerations.

This module provides infrastructure for defining measurement types with consistent naming
conventions, descriptive metadata, and automatic documentation generation.
"""

from dataclasses import KW_ONLY, dataclass
from enum import Enum
from textwrap import dedent
from typing import Final


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

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Entry.label must be a non-empty string")
        for name in ("desc", "bio_desc"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Entry.{name} must be a string")
        if self.image is not None and not isinstance(self.image, str):
            raise TypeError("Entry.image must be a string path or None")


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
            automatically by __new__ from the first element of the enum value tuple.
        desc (str): The description of what the measurement represents. Set automatically
            by __new__ from the second element of the enum value tuple. Defaults to empty
            string if not provided.
        pair (tuple[str, str]): A tuple of (label, description) for convenient access to
            both pieces of information together.
        CATEGORY (property): The category name returned by the category() classmethod.
            Provides instance-level access to the measurement category.

    Examples:
        Define a custom measurement enumeration:

        >>> from phenotypic.schema import MeasurementInfo
        >>> class SHAPE(MeasurementInfo):
        ...     @classmethod
        ...     def category(cls):
        ...         return 'Shape'
        ...
        ...     AREA = ('Area', 'Total number of pixels in the detected object')
        ...     PERIMETER = ('Perimeter', 'Total length of object boundary in pixels')

        Access measurement information and generate headers:

        >>> SHAPE.AREA
        <Shape_Area: 'Shape_Area'>
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

    def __new__(cls, *args):
        """Create a member from an ``Entry`` (preferred) or, transiently, a
        legacy ``(label, desc)`` tuple / bare ``label`` string.

        The legacy branch is a migration scaffold removed once all members use
        ``Entry``. The enum value is the category-prefixed header (e.g.
        ``Shape_Area``); ``label``/``desc``/``bio_desc``/``image`` are stored as
        instance attributes.
        """
        if len(args) == 1 and isinstance(args[0], Entry):
            entry = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            entry = Entry(args[0])
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            entry = Entry(args[0], args[1])
        else:
            raise TypeError(
                f"{cls.__name__} members must be declared as Entry(...); "
                f"got {args!r}."
            )
        full = f"{cls.category()}_{entry.label}"
        obj = str.__new__(cls, full)
        obj._value_ = full
        obj.label = entry.label
        obj.desc = entry.desc
        obj.bio_desc = entry.bio_desc
        obj.image = entry.image
        obj.pair = (entry.label, entry.desc)
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

    @classmethod
    def get_labels(cls) -> list[str]:
        """Get all measurement labels without category prefix.

        Returns a list of the short labels (without category prefix) for all measurements
        defined in this enumeration. These are the first element of each enum value tuple.
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
    ) -> str:
        """Generate an RST (reStructuredText) table documenting the measurements.

        Creates a formatted list-table in reStructuredText format that documents all
        measurements in this enumeration, including their labels and descriptions. This
        is useful for generating API documentation, parameter guides, or measurement
        reference tables that will be rendered in Sphinx documentation.

        Args:
            title (str, optional): The title for the RST table. Defaults to the class name
                (e.g., 'SHAPE'). The title is formatted as "Category: **{title}**" in the
                output.
            header (tuple[str, str], optional): A tuple of (left_column, right_column)
                header names. Defaults to ("Name", "Description"). The left column typically
                contains measurement labels and the right column contains their descriptions.

        Returns:
            str: A formatted reStructuredText list-table string. The output includes:
                - RST list-table directive with the title
                - Column headers (Name and Description by default)
                - One row per measurement with label and description
                The returned string is ready to be embedded in Sphinx documentation files
                or appended to docstrings.
        """
        title = title or cls.category()
        left, right = header
        lines = [
            f".. list-table:: Category: **{title}**",
            "   :header-rows: 1",
            "",
            f"   * - {left}",
            f"     - {right}",
        ]
        for m in cls:
            lines += [
                f"   * - ``{m.label}``",
                f"     - {m.desc}",
            ]
        return dedent("\n".join(lines))

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
