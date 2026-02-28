from __future__ import annotations

from typing import TYPE_CHECKING, Union, NamedTuple, Optional

if TYPE_CHECKING:
    import napari
    from phenotypic import Image, GridImage

from phenotypic.abc_ import BaseOperation
from ._image_pipeline_core import ImagePipelineCore


def _napari_layers_for(operation: BaseOperation) -> list[tuple[str, bool]] | None:
    """Return napari layer specifications for the given operation type.

    Each tuple is ``(accessor_name, is_labels)`` indicating which image
    accessor to visualise and whether it should be rendered as a labels
    layer.

    This intentionally differs from :func:`_layers_modified_by` in
    ``_image_pipeline_core``: that function tracks *all* layers an
    operation modifies (for delta HDF5 storage), while this function
    selects only the layers worth *visualising* after each step.  For
    example, ``ImageCorrector`` modifies all four layers, but we only
    add ``rgb`` and ``gray`` to napari because those capture the visual
    change.  Availability checks (empty RGB, zero objects) are handled
    at the call site, not here.

    Args:
        operation: An operation instance from the pipeline.

    Returns:
        List of ``(accessor_name, is_labels)`` tuples describing which
        layers to add after this operation, or ``None`` for read-only
        operations that should be skipped.
    """
    from phenotypic.abc_ import (
        ImageCorrector,
        ImageEnhancer,
        MeasureFeatures,
        ObjectDetector,
        ObjectRefiner,
        GridFinder,
    )

    if isinstance(operation, MeasureFeatures):
        return None
    if isinstance(operation, GridFinder):
        return None
    if isinstance(operation, ImageEnhancer):
        return [("detect_mat", False)]
    if isinstance(operation, (ObjectDetector, ObjectRefiner)):
        return [("objmap", True)]
    if isinstance(operation, ImageCorrector):
        return [("rgb", False), ("gray", False)]
    return [("rgb", False), ("gray", False), ("detect_mat", False), ("objmap", True)]


class NapariPipelineResult(NamedTuple):
    """Result of ``apply_napari``.

    Attributes:
        image: The final processed image.
        viewer: The napari viewer with all pipeline layers.
    """

    image: Union[GridImage, Image]
    viewer: napari.Viewer


class NapariPipelineViewer(ImagePipelineCore):
    """Pipeline mixin that adds napari visualisation via :meth:`apply_napari`."""

    def apply_napari(
        self,
        image: Image,
        inplace: bool = False,
        reset: bool | None = None,
        viewer: napari.Viewer | None = None,
    ) -> NapariPipelineResult:
        """Apply the pipeline and progressively add layers to a napari viewer.

        Creates (or reuses) a napari viewer and adds the original image layers
        as a baseline, then adds the modified layer after each operation
        completes. Layer names follow the pattern
        ``{step:02d}_{OperationName}_{accessor}``.

        Args:
            image: The input image to process.
            inplace: If ``True`` the image is modified in place; otherwise a
                copy is made first. Defaults to ``False``.
            reset: Whether to reset the image before applying operations.
                ``None`` (default) uses the pipeline-level setting.
            viewer: An existing napari viewer to add layers to. If ``None``
                (default), a new viewer is created.

        Returns:
            NapariPipelineResult: Named tuple with the final image and
            the napari viewer reference.

        Raises:
            ImportError: If napari is not installed.
        """
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
            _HAS_NAPARI,
        )

        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[gui]"
            )

        import napari as _napari

        if viewer is None:
            viewer = _napari.Viewer()
            from phenotypic.gui._smart_grid import install_smart_grid
            install_smart_grid(viewer)

        effective_reset = reset if reset is not None else self._reset
        img = image if inplace else image.copy()
        if effective_reset:
            img.reset()

        # Add baseline layers: original state before any operations
        if not img.rgb.isempty():
            img.rgb.napari(viewer=viewer, layer_name="00_original_rgb")
        img.gray.napari(viewer=viewer, layer_name="00_original_gray")
        img.detect_mat.napari(viewer=viewer, layer_name="00_original_detect_mat")

        def _add_napari_layers(i, key, current_img, operation):
            layers = _napari_layers_for(operation)
            if layers is None:
                return  # Skip read-only operations
            for accessor_name, is_labels in layers:
                if accessor_name == "rgb" and current_img.rgb.isempty():
                    continue
                if accessor_name == "objmap" and current_img.num_objects == 0:
                    continue
                full_name = f"{i + 1:02d}_{key}_{accessor_name}"
                accessor = getattr(current_img, accessor_name)
                accessor.napari(viewer=viewer, layer_name=full_name)

        self._run_operations(img, on_op_complete=_add_napari_layers)
        return NapariPipelineResult(image=img, viewer=viewer)
