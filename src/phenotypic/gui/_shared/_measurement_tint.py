"""Turn one measurement value into the text and the colour a card wears.

A colony card that shows a number is only half of the feature. The other
half is that a grid of cards reads as a *map*: the eye should find the dead
quadrant, the edge gradient, the one contaminated row before it reads a
single digit. That is what the tint is for, and it is why the scale is built
over **the values in view** rather than the column's global range -- the grid
is already filtered, and a scale anchored to a range the user cannot see
renders every visible card the same shade.

Three decisions are pinned here rather than left to each call site.

**One sequential ramp, and it is the brand's.**
:data:`~phenotypic.sdk_.viz.figures.SEQUENTIAL_COLORSCALE` -- near-transparent
navy through sky to full navy (DESIGN.md "06 -- Heatmap Colorscale" / "12 --
Continuous Colorbar") -- is the single continuous ramp already used by the
heatmap tab, so a value reads the same way on a card as it does on a plate
map. A diverging ramp suits deviation from a control and is a later addition;
nothing here assumes sequential beyond this constant.

**One number format.** Integral values render without a decimal point,
everything else to four significant figures, and anything outside
``[1e-4, 1e6)`` in scientific notation. A single rule keeps a column of
numbers aligned enough to scan; per-column formatting would read better for
any one column and worse across the grid.

**Ink is computed, not chosen.** The ramp spans from a near-white tint to
full navy, so a fixed text colour is unreadable at one end. The label's ink
follows the tint's relative luminance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from phenotypic.sdk_.viz.figures import SEQUENTIAL_COLORSCALE

__all__ = [
    "MeasurementScale",
    "TileMeasurement",
    "format_measurement_value",
    "sequential_tint",
]

#: Opaque backdrop the ramp's translucent low end is composited over. The
#: ramp's 0.0 stop is ``rgba(0,54,96,0.08)``; a card needs an opaque colour
#: (it also drives the ink contrast decision), and the card sits on the
#: surface token.
_COMPOSITE_BACKDROP: tuple[int, int, int] = (255, 255, 255)

#: Ink for a light tint and for a dark one. Navy is the body ink; white is
#: the only readable choice against the ramp's full-navy end.
_INK_DARK: str = "#003660"
_INK_LIGHT: str = "#FFFFFF"

#: Relative-luminance threshold at which the label flips from navy to white.
#: 0.45 rather than a nominal 0.5 because the ramp's midpoint (``#56B4E9``,
#: luminance ~0.42) still carries navy legibly and the flip should happen
#: after it, not on it.
_INK_FLIP_LUMINANCE: float = 0.45


def _parse_stop(color: str) -> tuple[float, float, float, float]:
    """Parse one ramp stop into ``(r, g, b, alpha)`` with 0-255 channels.

    Accepts the two spellings :data:`SEQUENTIAL_COLORSCALE` uses: ``#rrggbb``
    and ``rgba(r,g,b,a)``.

    Args:
        color: A ramp stop's CSS colour string.

    Returns:
        Red, green and blue on 0-255 plus alpha on 0-1.

    Raises:
        ValueError: If the string is neither spelling.
    """
    text = color.strip()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) != 6:
            raise ValueError(f"Unsupported hex colour: {color!r}")
        return (
            float(int(digits[0:2], 16)),
            float(int(digits[2:4], 16)),
            float(int(digits[4:6], 16)),
            1.0,
        )
    if text.startswith("rgba(") and text.endswith(")"):
        parts = [part.strip() for part in text[5:-1].split(",")]
        if len(parts) != 4:
            raise ValueError(f"Unsupported rgba colour: {color!r}")
        red, green, blue, alpha = (float(part) for part in parts)
        return (red, green, blue, alpha)
    raise ValueError(f"Unsupported colour: {color!r}")


def _composite_over_backdrop(
    channels: tuple[float, float, float, float],
) -> tuple[int, int, int]:
    """Flatten a possibly-translucent stop onto :data:`_COMPOSITE_BACKDROP`."""
    red, green, blue, alpha = channels
    return tuple(  # type: ignore[return-value]
        int(round(alpha * channel + (1.0 - alpha) * backdrop))
        for channel, backdrop in zip(
            (red, green, blue), _COMPOSITE_BACKDROP, strict=True
        )
    )


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Return WCAG relative luminance for an opaque sRGB triple."""
    linear = []
    for channel in rgb:
        value = channel / 255.0
        linear.append(
            value / 12.92
            if value <= 0.04045
            else ((value + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def sequential_tint(fraction: float) -> str:
    """Sample the brand sequential ramp at ``fraction``.

    Args:
        fraction: Position on the ramp. Clamped into ``[0, 1]``, so a
            degenerate scale (every value equal) can pass ``0.0`` without
            a guard at the call site.

    Returns:
        An opaque ``#rrggbb`` colour -- opaque because the ramp's low end is
        translucent navy and a card needs a colour it can also compute an
        ink contrast against.

    Examples:
        >>> sequential_tint(0.0)
        '#ebeff2'
        >>> sequential_tint(1.0)
        '#003660'
    """
    position = min(1.0, max(0.0, float(fraction)))
    stops = SEQUENTIAL_COLORSCALE
    for index in range(len(stops) - 1):
        low_pos, low_color = stops[index]
        high_pos, high_color = stops[index + 1]
        if position > high_pos and index + 2 < len(stops):
            continue
        span = high_pos - low_pos
        local = 0.0 if span <= 0 else (position - low_pos) / span
        local = min(1.0, max(0.0, local))
        low = _composite_over_backdrop(_parse_stop(low_color))
        high = _composite_over_backdrop(_parse_stop(high_color))
        blended = tuple(
            int(round(low[channel] + local * (high[channel] - low[channel])))
            for channel in range(3)
        )
        return "#{:02x}{:02x}{:02x}".format(*blended)
    raise ValueError("Sequential colorscale must carry at least two stops")


def _ink_for(tint: str) -> str:
    """Return the readable label colour for a card wearing ``tint``."""
    rgb = _composite_over_backdrop(_parse_stop(tint))
    if _relative_luminance(rgb) >= _INK_FLIP_LUMINANCE:
        return _INK_DARK
    return _INK_LIGHT


def format_measurement_value(value: float) -> str:
    """Format one measurement for a card label.

    The one rule, applied to every column: an integral value renders without
    a decimal point, anything else to four significant figures, and a
    magnitude outside ``[1e-4, 1e6)`` in scientific notation. ``Shape_Area``
    and ``ColorLab_DeltaE2000MedianFromMedoid`` would each read better under
    their own rule, but a grid of cards is scanned across columns as often as
    down one, and a format that changes per column defeats that.

    Args:
        value: The measured value.

    Returns:
        The label text. ``"n/a"`` for a NaN or infinite value -- which is a
        measured outcome (an empty mask divides by zero), not an error.

    Examples:
        >>> format_measurement_value(86770.0)
        '86770'
        >>> format_measurement_value(0.535051)
        '0.5351'
        >>> format_measurement_value(770.0)
        '770'
        >>> format_measurement_value(1.2345e-9)
        '1.235e-09'
    """
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return "n/a"
    if number == 0.0:
        return "0"
    if number.is_integer() and abs(number) < 1e6:
        return str(int(number))
    magnitude = abs(number)
    if magnitude >= 1e6 or magnitude < 1e-4:
        return f"{number:.4g}"
    decimals = max(0, 3 - int(math.floor(math.log10(magnitude))))
    return f"{number:.{decimals}f}"


@dataclass(frozen=True)
class MeasurementScale:
    """A column's name and the value range the visible cards span.

    Built over the values actually on the grid, never the column's range
    across the whole run: the grid is filtered, and rescaling to what is
    shown is the difference between a visible gradient and a uniformly
    coloured grid.

    Attributes:
        column: Column name, as shown in the legend.
        minimum: Smallest value among the cards in view.
        maximum: Largest value among the cards in view.
    """

    column: str
    minimum: float
    maximum: float

    @classmethod
    def over(cls, column: str, values: list[float]) -> MeasurementScale | None:
        """Build the scale spanning ``values``, or ``None`` if there are none.

        Args:
            column: Column name.
            values: Every finite value on the grid.

        Returns:
            The scale, or ``None`` when no card carries a value -- in which
            case there is nothing to tint and nothing to legend.
        """
        finite = [
            float(value)
            for value in values
            if not (math.isnan(value) or math.isinf(value))
        ]
        if not finite:
            return None
        return cls(column=column, minimum=min(finite), maximum=max(finite))

    def fraction_of(self, value: float) -> float:
        """Position ``value`` on ``[0, 1]`` within this scale.

        A degenerate scale (every visible value identical) maps to ``0.0``:
        one flat shade is the honest rendering of "no variation here", and
        it avoids a divide-by-zero at every call site.
        """
        span = self.maximum - self.minimum
        if span <= 0:
            return 0.0
        return (float(value) - self.minimum) / span

    def measurement_for(self, value: float) -> TileMeasurement:
        """Render one value as the label and colours its card wears."""
        tint = sequential_tint(self.fraction_of(value))
        return TileMeasurement(
            text=format_measurement_value(value),
            tint=tint,
            ink=_ink_for(tint),
        )


@dataclass(frozen=True)
class TileMeasurement:
    """One card's measurement display: the text and the colours it wears.

    Passed to
    :func:`~phenotypic.gui._shared.tiles.build_tile_cell` as a single
    optional argument so the card gains a measurement without gaining any
    new identity plumbing -- it already receives the ``label`` the value was
    joined on.

    Attributes:
        text: The formatted value, rendered in the card's mono face.
        tint: Opaque ``#rrggbb`` fill for the card's value ribbon and frame
            ring.
        ink: Readable label colour against ``tint``.
    """

    text: str
    tint: str
    ink: str
