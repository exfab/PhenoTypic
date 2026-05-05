"""Marker mixin for operations whose centres are picked interactively."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image


_T = TypeVar("_T", bound="PointPickerMixin")


class PointPickerMixin:
    """Shared plumbing for operations parameterised by user-picked ``(y, x)`` points.

    Operations that take a list of point coordinates as a primary parameter
    (currently :class:`~phenotypic.detect.ManualPointDetector` and
    :class:`~phenotypic.refine.ManualSelector`) inherit from this mixin to
    pick up three things at once:

    * **Coordinate coercion.** ``__setattr__`` converts list/tuple input
      assigned to the centres parameter into a NumPy array, so downstream
      ``_operate`` code can treat it uniformly. ``None`` passes through
      unchanged so that "no points yet" remains a valid state.
    * **A blocking napari picker.** :meth:`napari` opens a desktop
      :class:`~phenotypic.tools_.napari_.PointPickerWidget` and writes the
      confirmed picks back to the operation. Closing the viewer without
      confirming is a no-op (existing centres are preserved).
    * **A GUI introspection hook.** The ``_point_picker_param_name`` class
      attribute tells the Dash builder which ``__init__`` parameter to
      replace with an interactive picker button instead of a free-form text
      input.

    Subclasses that store coordinates under a name other than ``"centers"``
    can override :attr:`_point_picker_param_name`; both ``__setattr__`` and
    :meth:`napari` use that attribute.

    Attributes:
        _point_picker_param_name: Name of the ``__init__`` parameter holding
            the picked ``(y, x)`` coordinates. Defaults to ``"centers"``.
    """

    _point_picker_param_name: ClassVar[str] = "centers"

    def __setattr__(self, name: str, value: object) -> None:
        if name == self._point_picker_param_name and value is not None:
            value = np.asarray(value)
        super().__setattr__(name, value)

    def napari(self: _T, image: Image) -> _T:
        """Interactively pick coordinates using a napari viewer.

        Opens a blocking napari viewer displaying the plate image layers.
        Click points on the image, then click **Confirm** in the dock
        widget. The picked coordinates are stored in the attribute named by
        :attr:`_point_picker_param_name`. If the viewer is closed without
        confirming any points, existing coordinates are preserved.

        Args:
            image: The Image to display for coordinate selection.

        Returns:
            The mixin-bearing instance, for method chaining.

        Raises:
            ImportError: If napari is not installed.
        """
        from phenotypic.tools_.napari_ import PointPickerWidget

        points = PointPickerWidget(max_points=None).run(image)
        if len(points) > 0:
            setattr(self, self._point_picker_param_name, points)
        return self
