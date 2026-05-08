"""Mixin adding optional Generalized Anscombe Transform variance stabilization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class _GATSupportMixin:
    """Optional Generalized Anscombe Transform (GAT) wrapping for noise-driven ops.

    Subclasses opt into Poisson-Gaussian variance stabilization by setting
    ``use_gat=True`` at construction. When active, calls to :meth:`_gat_apply`
    wrap the inner denoising step in a forward-GAT -> denoise (with the
    subclass's noise parameters retargeted to stabilized-domain values,
    typically 1.0) -> inverse-GAT pipeline. When inactive, ``_gat_apply``
    is a thin pass-through.

    Subclass contract:
        ``_GAT_NOISE_PARAMS`` (ClassVar[dict[str, float]]):
            Maps init-param names to their stabilized-domain values
            (typically 1.0). The mixin temporarily overrides these on
            ``self`` for the duration of the inner denoise call. Examples:
            ``{"sigma_psd": 1.0}`` for BM3D; ``{"h": 1.0, "sigma": 1.0}``
            for non-local means.
        ``_GAT_DEFER_ATTRS`` (ClassVar[tuple[str, ...]]):
            Boolean attributes that must be ``False`` inside the GAT region.
            Use for output-clip flags or skimage's ``rescale_sigma`` -- any
            knob that would corrupt the stabilized round-trip if left at its
            default. Restored after the inner call returns.

    GAT init parameters (added to every subclass automatically via cooperative
    ``super().__init__``):
        ``use_gat`` (bool): Enable GAT wrapping. Default ``False``.
        ``gat_gain`` (float): Camera gain in electrons per ADU. Default 1.0.
        ``gat_mu`` (float): Read-noise mean (baseline offset). Default 0.0.
        ``gat_read_sigma`` (float): Read-noise standard deviation.
            ``0.0`` assumes pure Poisson noise. Default 0.0.
        ``gat_scale_factor`` (float | None): Multiplier converting normalized
            [0, 1] data to counts. ``None`` auto-detects from
            ``image.metadata.bit_depth`` (8-bit -> 255, 16-bit -> 65535).

    Reference:
        M. Mäkitalo and A. Foi, "Optimal inversion of the generalized Anscombe
        transformation for Poisson-Gaussian noise," IEEE Trans. Image Process.,
        vol. 22, no. 1, pp. 91-103, Jan. 2013.
    """

    _GAT_NOISE_PARAMS: ClassVar[dict[str, float]] = {}
    _GAT_DEFER_ATTRS: ClassVar[tuple[str, ...]] = ()

    def __init__(
            self,
            *args,
            use_gat: bool = False,
            gat_gain: float = 1.0,
            gat_mu: float = 0.0,
            gat_read_sigma: float = 0.0,
            gat_scale_factor: float | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if gat_gain <= 0:
            raise ValueError(f"gat_gain must be > 0, got {gat_gain}")
        if gat_read_sigma < 0:
            raise ValueError(
                f"gat_read_sigma must be >= 0, got {gat_read_sigma}"
            )
        if gat_scale_factor is not None and gat_scale_factor <= 0:
            raise ValueError(
                f"gat_scale_factor must be > 0, got {gat_scale_factor}"
            )
        self.use_gat = bool(use_gat)
        self.gat_gain = float(gat_gain)
        self.gat_mu = float(gat_mu)
        self.gat_read_sigma = float(gat_read_sigma)
        self.gat_scale_factor = (
            float(gat_scale_factor) if gat_scale_factor is not None else None
        )

    def _gat_apply(
            self,
            image: Image,
            target_attr: str,
            fn: Callable[[Image], None],
    ) -> None:
        """Wrap ``fn(image)`` in forward GAT -> inner -> inverse GAT.

        When ``self.use_gat`` is False, calls ``fn(image)`` directly with no
        overhead and no behavior change. When True, forward-stabilizes
        ``image[target_attr]`` in place, retargets the noise/defer attrs
        declared by the subclass, calls ``fn(image)`` (which should mutate
        ``image[target_attr]`` -- the inner denoiser body), then applies the
        unbiased inverse GAT and clips the result to [0, 1].

        Args:
            image: Image being processed.
            target_attr: Name of the image accessor to wrap (typically
                ``"detect_mat"``; ``"gray"`` for grayscale-domain correctors).
            fn: Callable that mutates ``image[target_attr]`` in place. Usually
                ``self._denoise_<channel>`` defined on the subclass.
        """
        if not self.use_gat:
            fn(image)
            return

        # Deferred import: avoids a circular load between
        # phenotypic.enhance.__init__ and this mixin.
        from phenotypic.enhance._anscombe import (
            gat_forward,
            gat_inverse,
            resolve_scale_factor,
        )

        scale = resolve_scale_factor(image, self.gat_scale_factor)

        # Forward GAT (counts -> stabilized). Write directly to
        # ``image._data.<attr>`` so the gray accessor's [0, 1] range guard
        # does not reject the stabilized values; subclass ``fn`` reads via
        # ``image.<attr>[:]`` which returns the underlying _data unchanged.
        data = getattr(image._data, target_attr)
        setattr(
            image._data,
            target_attr,
            gat_forward(
                data * scale,
                self.gat_mu,
                self.gat_read_sigma,
                self.gat_gain,
            ),
        )

        snapshot = {
            k: getattr(self, k)
            for k in (*self._GAT_NOISE_PARAMS, *self._GAT_DEFER_ATTRS)
        }
        try:
            for k, v in self._GAT_NOISE_PARAMS.items():
                setattr(self, k, v)
            for k in self._GAT_DEFER_ATTRS:
                setattr(self, k, False)
            fn(image)
        finally:
            for k, v in snapshot.items():
                setattr(self, k, v)

        # Inverse GAT (stabilized -> counts -> [0, 1]). Final write goes
        # through _data again -- the result is in [0, 1] after the clip,
        # so it would also pass the gray accessor's guard.
        recovered = gat_inverse(
            getattr(image._data, target_attr),
            self.gat_mu,
            self.gat_read_sigma,
            self.gat_gain,
        )
        setattr(
            image._data,
            target_attr,
            (recovered / scale).clip(0.0, 1.0),
        )
