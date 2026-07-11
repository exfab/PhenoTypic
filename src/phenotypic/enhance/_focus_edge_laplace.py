from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import laplace
from typing import Annotated, Optional

from ..abc_ import FocusEdge
from phenotypic.sdk_ import NormalizedOutputMixin
from phenotypic.sdk_.typing_ import NdArrayField, NormOut, TuneSpec


class FocusEdgeLaplace(NormalizedOutputMixin, FocusEdge):
    """Enhance colony edges in ``detect_mat`` with a discrete Laplacian operator.

    Applies a discrete second-derivative Laplacian that responds strongly to
    rapid intensity changes, highlighting colony margins and ring-like features
    such as swarming fronts. The output is an edge-response map suitable as a
    preprocessing step for contour detection, watershed seeding, or separating
    touching colonies.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Emphasizing colony edges before edge-based or contour-based
          segmentation.
        - Detecting ring patterns around swarming colonies for motility
          phenotyping.
        - Generating boundary seeds for watershed segmentation when colonies
          are touching.

    Consider Also:
        - :class:`FocusEdgeHessian` for multi-scale ridge and edge detection
          with additional control over scale and background suppression.
        - :class:`SharpenEdgeGauss` for edge enhancement that retains the
          original intensity profile rather than producing a pure edge map.
        - :class:`FocusEdgePhase` for contrast-invariant edge detection under
          uneven illumination.

    Args:
        kernel_size: Size of the Laplacian convolution kernel in pixels.
            Smaller values (3) capture fine colony edges but amplify noise;
            larger values (5--7) smooth noise and emphasize broader colony
            boundaries. Default: 3.
        mask: Boolean or 0/1 array restricting processing to a region of
            interest (e.g., the circular plate area). ``None`` processes the
            full image. Default: ``None``.
        norm: Output range policy applied to the raw Laplacian response.
            ``"rescale"`` (default) linearly remaps the full signed response
            onto [0, 1], preserving the bipolar edge structure. ``"clip"``
            saturates the negative lobe to 0, discarding roughly half the edge
            response; ``None`` passes the raw signed values through untouched.
            Default: ``"rescale"``.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Laplacian edge
        response, range-guarded into [0, 1] per ``norm``. ``rgb`` and ``gray``
        are unchanged.

    Note:
        Changed in 0.18.0: the raw Laplacian is signed by construction (on the
        synthetic yeast plate it spans roughly [-1.52, +1.48] with ~114k
        negative pixels), which violated the ``detect_mat`` [0, 1] contract.
        Output is now range-guarded, defaulting to ``norm="rescale"`` so the
        full bipolar edge response is preserved inside [0, 1] rather than
        having its negative half clipped away. This is a breaking change:
        downstream thresholds tuned against the old raw response must be
        re-tuned. Pass ``norm=None`` to recover the pre-0.18.0 signed output.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a visual
        walkthrough of edge enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for how edge-response maps
        fit into the pipeline model.
    """

    kernel_size: Annotated[Optional[int], TuneSpec(3, 7, step=2)] = 3
    mask: NdArrayField | None = None
    norm: NormOut = "rescale"

    def _operate(self, image: Image) -> Image:
        response = laplace(
            image=image.detect_mat[:],
            ksize=self.kernel_size,
            mask=self.mask,
        )
        image.detect_mat[:] = self._apply_norm(response)
        return image
