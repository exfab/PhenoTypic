"""Marker mixin for operations whose centres are picked interactively."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

if TYPE_CHECKING:
    from phenotypic._core._image import Image

_T = TypeVar("_T", bound="PointPickerMixin")


class PointPickerMixin:
    """Shared plumbing for operations parameterised by user-picked ``(y, x)`` points.

    Operations that take a list of point coordinates as a primary parameter
    (currently :class:`~phenotypic.detect.ManualPointDetector` and
    :class:`~phenotypic.refine.ManualRefine`) inherit from this mixin to
    pick up two things at once:

    * **A blocking napari picker.** :meth:`napari` opens a desktop
      :class:`~phenotypic.tools_.napari_.PointPickerWidget` and writes the
      confirmed picks back to the operation. Closing the viewer without
      confirming is a no-op (existing centres are preserved).
    * **A GUI introspection hook.** The ``_point_picker_param_name`` class
      attribute tells the Dash builder which field to replace with an
      interactive picker button instead of a free-form text input.

    Subclasses that store coordinates under a name other than ``"centers"``
    can override :attr:`_point_picker_param_name`; :meth:`napari` uses that
    attribute.

    Coordinate coercion (list/tuple/``np.ndarray`` → a canonical form)
    moves to a ``field_validator`` on the coordinate field of each
    consuming operation as part of the pydantic migration — this mixin no
    longer overrides ``__setattr__`` (which would clash with pydantic's).

    Attributes:
        _point_picker_param_name: Name of the field holding the picked
            ``(y, x)`` coordinates. Defaults to ``"centers"``.
    """

    _point_picker_param_name: ClassVar[str] = "centers"

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
