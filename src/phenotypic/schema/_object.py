"""The label and description of the per-object identifier column."""

from ._measurement_info import Entry, MeasurementInfo


class OBJECT(MeasurementInfo):
    """Per-object identifier shared by every measurement table.

    ``Object_Label`` is written as the first column of every per-object
    measurement DataFrame and serves as the per-image primary key that the
    pipeline merges measurers on. Each detected colony in a plate image is
    assigned a unique integer label, so size, shape, intensity, and color
    measurements for the same colony line up row-for-row across operators when
    joined on this column.
    """

    @classmethod
    def category(cls):
        return "Object"

    LABEL = Entry(
        "Label",
        "Integer label uniquely identifying each detected object (colony) within "
        "its source image. Acts as the per-image primary key shared by every "
        "measurement table, letting all per-colony measurements be joined together "
        "and traced back to a single colony on the agar plate.",
    )
