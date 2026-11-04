"""The closed list of measurement units, with the data needed to convert them.

Every non-metadata measurement column carries exactly one :class:`UNITS`
member. Output tables stay in pixels; a unit records how a column scales with
image resolution, so a reader with a pixels-per-mm value can convert it::

    df[SIZE.AREA] / px_per_mm ** SIZE.AREA.unit.length_power

``length_power`` is the power of length in the unit: 1 for ``px``, 2 for
``px²``, 0 for anything that does not change with image scale, and ``None``
for a value that is not a quantity at all (a label, a string, a status).

Some columns have no unit of their own: a fitted growth rate is in the unit of
the column it was fitted on, per unit of time. Those members are tagged with an
**input-dependent** unit (``OF_INPUT``, ``OF_INPUT_SQUARED``,
``OF_INPUT_PER_TIME``), and the producer's ``output_unit(member, on)`` resolves
the tag against the fitted column with :meth:`UNITS.resolve`. Reading
``length_power`` on an unresolved input-dependent unit raises ``TypeError``,
so ``px_per_mm ** None`` can never happen silently. Their ``symbol`` is a
readable placeholder (``input unit/time``), never ``""``, so nothing renders a
fitted rate as unitless.

To add a unit, add one :class:`UnitEntry` line to :class:`UNITS`.

This module imports only the standard library, keeping ``phenotypic.schema``
import-light.
"""

from __future__ import annotations

import re
from dataclasses import KW_ONLY, dataclass
from enum import Enum, unique

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
#: A symbol containing any of these is compound, so a power wraps it in
#: parentheses ("(1/time)²"); a bare symbol takes the power directly ("px²").
_COMPOUND_RE = re.compile(r"[\s/·()⁰¹²³⁴⁵⁶⁷⁸⁹]")
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _is_int(value: object) -> bool:
    """Whether *value* is an int and not a bool (``True ** 2`` is a trap)."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    """A concrete unit: what a column's values are actually in.

    Returned by :meth:`UNITS.resolve` and by a producer's
    ``output_unit(member, on)``. Unlike a :class:`UNITS` member it is never
    input-dependent, so ``length_power`` is always safe to read.

    Args:
        symbol: Display symbol, e.g. ``"px/time"``; ``""`` for a unitless value.
        length_power: Power of length in the unit (``px_per_mm ** length_power``
            converts it), or ``None`` when the value is not a quantity.
    """

    symbol: str
    length_power: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str):
            raise TypeError(f"ResolvedUnit.symbol must be a string, got {self.symbol!r}")
        if self.length_power is not None and not _is_int(self.length_power):
            raise TypeError(
                f"ResolvedUnit.length_power must be an int or None, got {self.length_power!r}"
            )


@dataclass(frozen=True, slots=True)
class UnitEntry:
    """Declarative value for a :class:`UNITS` member.

    A static unit has ``input_power == 0``. An input-dependent unit has
    ``input_power >= 1`` and ``length_power=None``: its length power comes from
    the input column when it is resolved. The powers are keyword-only, so two
    adjacent ints can never be swapped positionally.

    Args:
        key: Unique lower-case key, the member's enum value (e.g. ``"px2"``).
            Two members with equal values would alias, so it must be unique.
        symbol: Display symbol (e.g. ``"px²"``); ``""`` for a unitless value or
            a non-quantity. An input-dependent unit carries a readable
            placeholder (``"input unit/time"``), never ``""``, so a renderer
            cannot show a fitted rate as unitless; resolve it for the real one.
        desc: Technical description of what the unit measures.
        length_power: Power of length in the unit, or ``None`` for a value that
            is not a quantity or a unit that depends on its input.
        input_power: Power of the input column's unit; 0 for a static unit.
        per_time: Whether the unit is per unit of the fitted time column.
    """

    key: str
    symbol: str
    desc: str
    _: KW_ONLY
    length_power: int | None = 0
    input_power: int = 0
    per_time: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise TypeError(f"UnitEntry.key must be a string, got {self.key!r}")
        if not _KEY_RE.fullmatch(self.key):
            raise ValueError(f"UnitEntry.key must be a lower-case token, got {self.key!r}")
        if not isinstance(self.symbol, str):
            raise TypeError(f"UnitEntry.symbol must be a string, got {self.symbol!r}")
        if not isinstance(self.desc, str):
            raise TypeError(f"UnitEntry.desc must be a string, got {self.desc!r}")
        if not self.desc.strip():
            raise ValueError("UnitEntry.desc must be a non-empty string")
        if self.length_power is not None and not _is_int(self.length_power):
            raise TypeError(
                f"UnitEntry.length_power must be an int or None, got {self.length_power!r}"
            )
        if not _is_int(self.input_power):
            raise TypeError(f"UnitEntry.input_power must be an int, got {self.input_power!r}")
        if self.input_power < 0:
            raise ValueError(
                f"UnitEntry.input_power must be non-negative, got {self.input_power!r}"
            )
        if not isinstance(self.per_time, bool):
            raise TypeError(f"UnitEntry.per_time must be a bool, got {self.per_time!r}")
        if self.input_power and self.length_power is not None:
            raise ValueError(
                f"UnitEntry {self.key!r} depends on its input, so its length power "
                "comes from the input; declare it with length_power=None"
            )


def _has_top_level_solidus(symbol: str) -> bool:
    """Whether *symbol* has a ``/`` outside parentheses (``rad/px``, not ``(rad/px)²``)."""
    depth = 0
    for char in symbol:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "/" and depth == 0:
            return True
    return False


def _resolved_symbol(input_symbol: str, power: int, per_time: bool) -> str:
    """Symbol of *input_symbol* raised to *power*, then per time when asked.

    A unitless input stays ``""`` under any power, and per time reads
    ``"1/time"`` rather than ``"/time"``. A symbol that already has a solidus
    is wrapped before ``/time`` is appended, so ``rad/px`` per time reads
    ``(rad/px)/time``: two solidi without parentheses are ambiguous.
    """
    symbol = input_symbol
    if symbol and power != 1:
        base = f"({symbol})" if _COMPOUND_RE.search(symbol) else symbol
        symbol = base + str(power).translate(_SUPERSCRIPT)
    if per_time:
        if not symbol:
            symbol = "1/time"
        elif _has_top_level_solidus(symbol):
            symbol = f"({symbol})/time"
        else:
            symbol = f"{symbol}/time"
    return symbol


@unique
class UNITS(str, Enum):
    """The closed list of measurement units.

    Each member's value is its lower-case key. Members expose ``symbol``,
    ``desc``, ``length_power``, ``input_power``, ``per_time``,
    ``is_input_dependent`` and :meth:`resolve`. ``length_power`` raises
    ``TypeError`` on an input-dependent member; resolve it first.

    Examples:
        >>> from phenotypic.schema import UNITS, ResolvedUnit
        >>> UNITS.PX2.length_power
        2
        >>> UNITS.OF_INPUT_PER_TIME.resolve(ResolvedUnit("px", 1))
        ResolvedUnit(symbol='px/time', length_power=1)
    """

    symbol: str
    desc: str
    input_power: int
    per_time: bool
    _length_power: int | None

    def __new__(cls, entry: UnitEntry) -> UNITS:
        if not isinstance(entry, UnitEntry):
            raise TypeError(
                f"UNITS members must be declared as UnitEntry(...); got {entry!r}"
            )
        obj = str.__new__(cls, entry.key)
        obj._value_ = entry.key
        obj.symbol = entry.symbol
        obj.desc = entry.desc
        obj._length_power = entry.length_power
        obj.input_power = entry.input_power
        obj.per_time = entry.per_time
        return obj

    def __str__(self) -> str:
        return self._value_

    @property
    def is_input_dependent(self) -> bool:
        """Whether the unit is the input column's unit to some power."""
        return self.input_power >= 1

    @property
    def length_power(self) -> int | None:
        """Power of length in the unit, or ``None`` for a non-quantity.

        Raises:
            TypeError: The unit depends on the input column; resolve it with
                the producer's ``output_unit(member, on)`` first.
        """
        if self.is_input_dependent:
            raise TypeError(
                f"UNITS.{self.name} depends on the input column; resolve it with "
                "the producer's output_unit(member, on)"
            )
        return self._length_power

    def resolve(self, input_unit: ResolvedUnit | None) -> ResolvedUnit:
        """The concrete unit, given the unit of the column the producer ran on.

        A static unit ignores *input_unit* and resolves to itself. An
        input-dependent unit raises the input to ``input_power`` and divides by
        time when ``per_time`` is set: ``OF_INPUT_PER_TIME`` on ``px²`` is
        ``px²/time`` with length power 2.

        Args:
            input_unit: Unit of the input column, or ``None`` when unknown.

        Raises:
            ValueError: The unit depends on its input, and *input_unit* is
                ``None`` or is not a quantity (``length_power is None``).
        """
        if not self.is_input_dependent:
            return ResolvedUnit(self.symbol, self._length_power)
        if input_unit is None or input_unit.length_power is None:
            raise ValueError(
                f"UNITS.{self.name} is in the unit of its input column, and that "
                f"unit is unknown (got {input_unit!r})"
            )
        return ResolvedUnit(
            _resolved_symbol(input_unit.symbol, self.input_power, self.per_time),
            input_unit.length_power * self.input_power,
        )

    PX = UnitEntry("px", "px", "Length in pixels.", length_power=1)
    PX2 = UnitEntry("px2", "px²", "Area in square pixels, a pixel count.", length_power=2)
    INTENSITY = UnitEntry(
        "intensity",
        "intensity",
        "Grayscale pixel intensity, on the scale of the image's gray layer.",
    )
    INTENSITY_PX2 = UnitEntry(
        "intensity_px2",
        "intensity·px²",
        "Grayscale intensity summed over an object's pixels, so it grows with "
        "the object's area.",
        length_power=2,
    )
    RADIAN = UnitEntry("radian", "rad", "Angle in radians.")
    DEGREE = UnitEntry("degree", "°", "Angle in degrees.")
    CIELAB = UnitEntry(
        "cielab",
        "CIELAB",
        "A CIELAB colour coordinate (L*, a* or b*), or a colour difference (ΔE) "
        "between two colours, in CIELAB units.",
    )
    CIELAB2 = UnitEntry(
        "cielab2", "CIELAB²", "A squared CIELAB coordinate, such as a variance of L*, a* and b*."
    )
    TURN = UnitEntry("turn", "turn", "Angle as a fraction of a full turn, in [0, 1).")
    RADIAN_PER_PX = UnitEntry(
        "radian_per_px", "rad/px", "Angle change per pixel of distance.", length_power=-1
    )
    RADIAN_PER_PX2 = UnitEntry(
        "radian_per_px2",
        "rad/px²",
        "Angle change per pixel per pixel, the gradient of an angle-per-pixel rate.",
        length_power=-2,
    )
    RADIAN_PX = UnitEntry(
        "radian_px",
        "rad·px",
        "Angle multiplied by a pixel distance, such as the area under an angle "
        "profile along a radius.",
        length_power=1,
    )
    DEGREE_PER_PX = UnitEntry(
        "degree_per_px", "°/px", "Angle change in degrees per pixel of distance.", length_power=-1
    )
    DEGREE_PER_PX2 = UnitEntry(
        "degree_per_px2",
        "°/px²",
        "Change of a degrees-per-pixel rate per pixel of distance.",
        length_power=-2,
    )
    UNITLESS = UnitEntry(
        "unitless", "", "Dimensionless: a ratio, fraction, count, index or score."
    )
    TIME = UnitEntry(
        "time", "time", "The unit of the time column a model was fitted against, such as hours."
    )
    PER_TIME = UnitEntry(
        "per_time",
        "1/time",
        "Per unit of the time column a model was fitted against, such as per hour.",
        per_time=True,
    )
    FRAME = UnitEntry(
        "frame", "frame", "A position or count in an image series, in frames (images)."
    )
    NOT_A_QUANTITY = UnitEntry(
        "not_a_quantity",
        "",
        "Not a quantity: a label, identifier, flag, string or status. Nothing to convert.",
        length_power=None,
    )
    OF_INPUT = UnitEntry(
        "of_input",
        "input unit",
        "The unit of the column the producer ran on.",
        length_power=None,
        input_power=1,
    )
    OF_INPUT_SQUARED = UnitEntry(
        "of_input_squared",
        "(input unit)²",
        "The square of the unit of the column the producer ran on, as for a "
        "mean squared error.",
        length_power=None,
        input_power=2,
    )
    OF_INPUT_PER_TIME = UnitEntry(
        "of_input_per_time",
        "input unit/time",
        "The unit of the column the producer ran on, per unit of time, as for a "
        "fitted growth rate.",
        length_power=None,
        input_power=1,
        per_time=True,
    )
