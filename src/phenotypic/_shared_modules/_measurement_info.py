"""Base class for creating measurement information enumerations.

This class provides a standardized way to define measurement types with categories,
labels, and descriptions. It combines Enum functionality with string behavior,
automatically prefixing labels with category names.

Examples
--------
Create a custom measurement enumeration:

>>> from phenotypic._shared_modules._measurement_info import MeasurementInfo
>>>
>>> class COLOR(MeasurementInfo):
...     @classmethod
...     def category(cls):
...         return 'Color'
...
...     RED = ('Red', 'Red channel intensity')
...     GREEN = ('Green', 'Green channel intensity')
...     BLUE = ('Blue', 'Blue channel intensity')

Access measurement information:

>>> COLOR.RED
<Color_Red: 'Color_Red'>
>>> str(COLOR.RED)
'Color_Red'
>>> COLOR.RED.label
'Red'
>>> COLOR.RED.desc
'Red channel intensity'
>>> COLOR.RED.CATEGORY
'Color'

Get all labels and headers:

>>> COLOR.get_labels()
['Red', 'Green', 'Blue']
>>> COLOR.get_headers()
['Color_Red', 'Color_Green', 'Color_Blue']

Generate RST documentation table:

>>> print(COLOR.rst_table())
.. list-table:: COLOR
   :header-nrows: 1
   <BLANKLINE>
   * - Name
     - Description
   * - ``Red``
     - Red channel intensity
   * - ``Green``
     - Green channel intensity
   * - ``Blue``
     - Blue channel intensity

Append documentation to a class docstring:

>>> class ColorProcessor:
...     '''Processes color measurements.'''
...     pass
>>>
>>> ColorProcessor.__doc__ = COLOR.append_rst_to_doc(ColorProcessor)
>>> print(ColorProcessor.__doc__)
Processes color measurements.
<BLANKLINE>
.. list-table:: COLOR
   :header-nrows: 1
   <BLANKLINE>
   * - Name
     - Description
   * - ``Red``
     - Red channel intensity
   * - ``Green``
     - Green channel intensity
   * - ``Blue``
     - Blue channel intensity
"""

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
        obj.desc = desc if desc else ""
        obj.pair = (label, obj.desc)
        return obj

    def __str__(self):
        return self._value_

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
            f".. list-table:: Category: **{title}**"
            f"",
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
    def append_rst_to_doc(cls, module) -> str:
        """
        returns a string with the RST table appended to the module docstring.
        """
        if isinstance(module, str):
            return module + "\n\n" + cls.rst_table()
        else:
            return module.__doc__ + "\n\n" + cls.rst_table()
