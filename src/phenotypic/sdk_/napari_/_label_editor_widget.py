"""Blocking napari-based labels editor with save-back-to-image."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import napari


class LabelEditorWidget:
    """Open a napari viewer to edit an image's object labels and save them back.

    The viewer shows ``rgb`` (when present), ``gray``, and ``detect_mat`` as
    image layers plus one editable labels layer seeded from the called accessor
    (``objmap`` or ``objmask``). A dock panel provides "Save to Image" and
    "Discard & Close" buttons. ``run`` blocks until the viewer is closed.
    """

    def run(
        self,
        image,
        accessor_name: str,
        *,
        viewer: napari.Viewer | None = None,
    ) -> np.ndarray | None:
        """Open the editor and block until closed.

        Args:
            image: A PhenoTypic ``Image`` whose layers are displayed and whose
                ``objmap``/``objmask`` is edited.
            accessor_name: ``"objmap"`` or ``"objmask"`` — selects which accessor
                the editable layer is seeded from and saved back to.
            viewer: Optional existing napari viewer to reuse. When ``None`` a new
                ``napari.Viewer(title="Label Editor")`` is created.

        Returns:
            The array written back to the image on save, or ``None`` if the user
            discarded or closed the viewer without saving.

        Raises:
            ImportError: If napari is not installed.
        """
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
            _HAS_NAPARI,
        )

        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[napari]"
            )
        import napari

        active_viewer = (
            viewer if viewer is not None else napari.Viewer(title="Label Editor")
        )

        if not image.rgb.isempty():
            image.rgb.napari(viewer=active_viewer, layer_name="rgb")
        image.gray.napari(viewer=active_viewer, layer_name="gray")
        image.detect_mat.napari(viewer=active_viewer, layer_name="detect_mat")

        if accessor_name == "objmask":
            seed = image.objmask[:].astype(np.uint8)
        else:
            seed = np.asarray(image.objmap[:])
        labels_layer = active_viewer.add_labels(seed, name=f"{accessor_name}_edit")
        if accessor_name == "objmask":
            labels_layer.selected_label = 1
        labels_layer.mode = "paint"
        active_viewer.layers.selection.active = labels_layer

        panel = _make_label_editor_panel(
            active_viewer, labels_layer, image, accessor_name
        )
        active_viewer.window.add_dock_widget(panel, name="Label Editor", area="right")

        # Only drive a blocking event loop when we own the viewer. When the
        # caller supplies one (advanced reuse), they manage their own loop and
        # the panel's Save button writes back whenever they click it.
        if viewer is None:
            napari.run()

        return panel.saved_labels


class _LabelEditorPanelLogic:
    """Qt-independent Save/Discard logic for the label editor dock widget.

    Kept separate from the ``QWidget`` subclass (built lazily in
    :func:`_make_label_editor_panel`) so the write-back behaviour is importable
    and unit-testable without a Qt event loop. The concrete dock widget mixes
    this into ``QWidget`` and supplies ``_viewer``, ``_labels_layer``,
    ``_image``, ``_accessor_name``, and ``saved_labels``.
    """

    _viewer: Any
    _labels_layer: Any
    _image: Any
    _accessor_name: str
    saved_labels: np.ndarray | None

    def _save(self) -> None:
        """Write the edited labels back through the accessor, then close.

        ``objmap`` saves the integer array verbatim (label IDs preserved);
        ``objmask`` binarizes the layer (``> 0``) and saves it as a mask, which
        relabels the object map.
        """
        data = self._labels_layer.data
        if self._accessor_name == "objmask":
            self._image.objmask[:] = data > 0
            self.saved_labels = self._image.objmask[:]
        else:
            self._image.objmap[:] = np.asarray(data)
            self.saved_labels = self._image.objmap[:]
        self._viewer.close()

    def _discard(self) -> None:
        """Close the viewer without writing any edits back."""
        self._viewer.close()


def _make_label_editor_panel(viewer, labels_layer, image, accessor_name: str):
    """Build the napari dock widget with Save/Discard controls.

    The ``QWidget`` subclass is defined here (lazily, on first call) rather than
    at module import so ``qtpy`` stays an optional dependency. Defining it with
    ``QWidget`` as a real base at class-creation time also avoids mutating
    ``__bases__`` after the fact, which CPython forbids when the base's
    deallocator differs (PyQt6's sip-wrapped ``QWidget`` vs ``object``).

    Args:
        viewer: The napari viewer instance.
        labels_layer: The editable napari Labels layer.
        image: The PhenoTypic ``Image`` to write edits back to.
        accessor_name: ``"objmap"`` or ``"objmask"``.

    Returns:
        A ``QWidget`` dock panel with Save/Discard buttons wired to the
        :class:`_LabelEditorPanelLogic` callbacks.
    """
    from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

    class _LabelEditorPanel(QWidget, _LabelEditorPanelLogic):
        def __init__(self) -> None:
            QWidget.__init__(self)

            self._viewer = viewer
            self._labels_layer = labels_layer
            self._image = image
            self._accessor_name = accessor_name
            self.saved_labels = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            self._save_btn = QPushButton("Save to Image")
            self._discard_btn = QPushButton("Discard & Close")
            layout.addWidget(self._save_btn)
            layout.addWidget(self._discard_btn)

            self._save_btn.clicked.connect(self._save)
            self._discard_btn.clicked.connect(self._discard)

    return _LabelEditorPanel()
