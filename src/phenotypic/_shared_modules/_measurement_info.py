from enum import Enum
from textwrap import dedent


class MeasurementInfo(Enum):
    """
    The labels and descriptions of the measurement information. This class helps with module consistency and documentation automation.

    Note:
        - Overwrite the CATEGORY label and add other measurement info here

    """

    @property
    def CATEGORY(self) -> str:
        """Overwrite this in inherited classes; should return a string with the category name"""
        raise NotImplementedError

    def __init__(self, label, desc=None):
        self.label, self.desc = label, desc

    def __str__(self):
        return f'{self.CATEGORY}_{self.label}'

    def __iter__(self):
        return (
            member for member in super().__iter__()
            if not hasattr(member, 'name') or member is not member.name != 'CATEGORY'
        )

    @classmethod
    def iter_labels(cls):
        """Yield all measurement info members except CATEGORY."""
        return (member.label for member in cls if member is not cls.CATEGORY)

    @classmethod
    def get_labels(cls):
        return [member.label for member in cls if member is not cls.CATEGORY]

    @classmethod
    def get_headers(cls):
        """Return full measurement info labels for use in pandas dataframe columns."""
        return [f'{x}' for x in cls.iter_labels() if cls.name.endswith('_') is False]

    @classmethod
    def rst_table(
            cls,
            *,
            title: str | None = None,
            header: tuple[str, str] = ("Label", "Description"),
    ) -> str:
        """
        Generates an RST table in the "list-table" format with the specified title and
        header. Includes rows based on the class's iterable members that provide labels
        and descriptions.

        Args:
            title: Optional title for the table. If none is provided, the name of the
                class is used as the default title.
            header: A tuple containing the header labels for the table. Defaults to
                ("Label", "Description").

        Returns:
            str: A string containing the formatted RST table.
        """
        title = title or cls.__name__
        left, right = header

        lines: list[str] = [
            f".. list-table:: {title}",
            "   :header-rows: 1",
            "",
            f"   * - {left}",
            f"     - {right}",
        ]

        for member in cls:
            lines.extend(
                [
                    f"   * - ``{member.label}``",
                    f"     - {member.desc}",
                ],
            )
        return dedent("\n".join(lines))

    @classmethod
    def append_rst_to_doc(cls, module) -> str:
        """
        returns a string with the RST table appended to the module docstring.
        """
        if isinstance(module, str):
            return module + "\n\n" + cls.rst_table()
        else:
            return module.__doc__ + "\n\n" + cls.rst_table()