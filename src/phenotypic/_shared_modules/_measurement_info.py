from enum import Enum
from textwrap import dedent


class MeasurementInfo(str, Enum):
    # Subclasses must implement this
    @classmethod
    def category(cls) -> str:
        raise NotImplementedError

    # Public, instance-level property (what you wanted to keep)
    @property
    def CATEGORY(self) -> str:
        return type(self).category()

    def __new__(cls, label: str, desc: str | None = None):
        cat = cls.category()  # use classmethod here
        full = f"{cat}_{label}"
        obj = str.__new__(cls, full)
        obj._value_ = full
        obj.label = label
        obj.desc = desc or label
        obj.pair = (label, obj.desc)
        return obj

    @classmethod
    def get_labels(cls):
        return [m.label for m in cls]

    @classmethod
    def get_headers(cls):
        return [m.value for m in cls]

    @classmethod
    def rst_table(
            cls,
            *,
            title: str | None = None,
            header: tuple[str, str] = ("Name", "Description"),
    ) -> str:
        title = title or cls.__name__
        left, right = header
        lines = [
            f".. list-table:: {title}",
            "   :header-rows: 1",
            "",
            f"   * - {left}",
            f"     - {right}",
        ]
        for m in cls:
            lines += [f"   * - ``{m.label}``", f"     - {m.desc}"]
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
