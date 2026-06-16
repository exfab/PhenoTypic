"""Measurement info for perceptual color composition using 11-color model."""

from ._measurement_info import Entry, MeasurementInfo


class ColorComposition(MeasurementInfo):
    """Classify colony pixels into 11 perceptual color categories and measure composition.

    Applies a priority-based color model (neutrals → special colors → hues)
    to classify each colony pixel into one of 11 categories: Black, White,
    Gray, Pink, Brown, Red, Orange, Yellow, Green, Cyan, Blue, Purple.
    Returns per-colony percentage breakdowns as DataFrame columns.
    """

    @classmethod
    def category(cls):
        return "ColorComposition"

    # Define the 11 color categories with descriptions
    BLACK_PCT = Entry("BlackPct", "Percentage of pixels classified as black (Value < 20)")
    WHITE_PCT = Entry(
        "WhitePct",
        "Percentage of pixels classified as white (Saturation < 15, Value > 85)",
    )
    GRAY_PCT = Entry(
        "GrayPct",
        "Percentage of pixels classified as gray (Saturation < 15, Value 20-85)",
    )
    PINK_PCT = Entry(
        "PinkPct",
        "Percentage of pixels classified as pink (Red/Magenta hue, Saturation 20-60, Value > 80)",
    )
    BROWN_PCT = Entry(
        "BrownPct",
        "Percentage of pixels classified as brown (Red/Orange hue, Value 20-60)",
    )
    RED_PCT = Entry(
        "RedPct",
        "Percentage of pixels classified as red (Hue 0-15° or 345-360°)",
    )
    ORANGE_PCT = Entry("OrangePct", "Percentage of pixels classified as orange (Hue 15-45°)")
    YELLOW_PCT = Entry("YellowPct", "Percentage of pixels classified as yellow (Hue 45-75°)")
    GREEN_PCT = Entry("GreenPct", "Percentage of pixels classified as green (Hue 75-150°)")
    CYAN_PCT = Entry("CyanPct", "Percentage of pixels classified as cyan (Hue 150-180°)")
    BLUE_PCT = Entry("BluePct", "Percentage of pixels classified as blue (Hue 180-250°)")
    PURPLE_PCT = Entry(
        "PurplePct",
        "Percentage of pixels classified as purple/magenta (Hue 250-345°)",
    )

    @classmethod
    def all_headers(cls):
        """Return all color composition measurement headers."""
        return [
            str(cls.BLACK_PCT),
            str(cls.WHITE_PCT),
            str(cls.GRAY_PCT),
            str(cls.PINK_PCT),
            str(cls.BROWN_PCT),
            str(cls.RED_PCT),
            str(cls.ORANGE_PCT),
            str(cls.YELLOW_PCT),
            str(cls.GREEN_PCT),
            str(cls.CYAN_PCT),
            str(cls.BLUE_PCT),
            str(cls.PURPLE_PCT),
        ]
